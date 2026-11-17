"""Dispatch simulator — replicates `CortexOrchestrator._process_core`
regex ordering without instantiating the orchestrator (which would
touch Palace, Ollama, ChromaDB, the audit log, etc.).

The eval framework uses this to verify routing decisions in
milliseconds rather than seconds. It is a **deliberate duplication**
of the dispatch sequence; when `_process_core` reorders or adds a
new regex matcher, this file must be updated. The framework
self-tests in `tests/test_evals_framework.py` catch drift early.

Why not call the real orchestrator?
  - Construction cost: ~3-5s per fresh instance (Palace init,
    Ollama handshake, registry load).
  - Side effects: writes to audit log, history, Palace.
  - Test isolation: every eval run would mutate the dev machine's
    state. Pure regex simulation has zero side effects.
  - Speed: 1000+ cases in <100ms vs minutes.

What we deliberately SKIP from the real flow (acceptable for V1):
  - Pending action / pending query state (no state in eval)
  - Gate / lease evaluation — the DYNAMIC part only (search triggers,
    whose mode depends on lease state). Static state-CHANGE phrasings
    (revoke / grant) ARE simulated since M6-A1: their real mode is
    always SYSTEM_UPDATE.
  - Reflex layer (covered by separate detector eval)
  - Skill registry (Palace-backed, runtime)
  - LLM call paths (engine.chat / chat_with_tools)
  - `_needs_realtime` LLM classifier

What we DO replicate exactly:
  - strip_vocatives normalisation
  - Studier verbs, list-emails, silence
  - Spotify family (8 verbs) — order matters here
  - macOS family (13 verbs)
  - Reminder / Calendar / Planner regex front-doors
  - MAC_QUIT_APP_RE then MAC_OPEN_APP_RE (quit before open: claude-B's
    bloco D ordering decision in sessão 12)
  - TOOL_HINT_RE check that decides between chat_with_tools and
    plain chat (reported as would_use_tools)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.core.gatekeeper import GRANT_PATTERNS, LEASE_PATTERNS, REVOKE_PATTERNS

from src.core.dispatch.normalisation import (
    normalise_stt_aliases,
    strip_courtesy,
    strip_vocatives,
)
# Every macOS dispatch family (volume/brightness/dark/lock/screenshot/app-list
# in s58; sleep + the post-planner close/quit/switch/open verbs in s59) now
# lives in the registry — none are matched directly here anymore.
from src.core.dispatch import registry as dispatch_registry
from src.core.dispatch.planner_trigger import PLANNER_TRIGGER_RE
from src.core.dispatch.flow_control import (
    APPROVAL_RE,
    DENIAL_RE,
)
from src.core.dispatch.heuristics import TOOL_HINT_RE, multi_intent_match
# studier_cmds + the contact/email triggers were migrated to the registry (s59)
# and are no longer matched directly here.


@dataclass
class DispatchOutcome:
    """The result of routing `text` through the simulated dispatch.

    `mode` mirrors what the orchestrator would have returned as the
    second element of its `(reply, mode)` tuple. `matcher` names the
    regex (or pseudo-matcher like 'fallthrough') that won. `capture`
    is the relevant captured group, when the matching regex has one.
    `would_use_tools` says whether TOOL_HINT_RE would have routed the
    LLM through chat_with_tools.
    """
    mode: str
    matcher: str
    capture: Optional[str] = None
    would_use_tools: bool = False


# ---------------------------------------------------------------------
# Ordered tables — keep in lockstep with orchestrator._process_core
# ---------------------------------------------------------------------

# (regex, matcher_name, mode, capture_group_index_or_None)
#
# capture_index: None means the matcher doesn't expose a group. An int
# means group(int) is reported in DispatchOutcome.capture.
# The studier (REVIEW/APPROVE/REJECT/SHOW/LIST/STUDY) + contacts (SAVE/LOOKUP) +
# email (SEND_STRUCTURED/SEND_BY_NAME/LIST_EMAILS) families — the last, inter-
# leaved inline block — were migrated to the dispatch registry (s59). Their
# routing (including the LIST_EMAILS-before-SHOW_STUDY precedence) is now
# reported by registry.simulate, in the orchestrator's real execution order.

# Silence + Mac-sleep were migrated to the dispatch registry (s59) — their
# routing (mode SILENCE / MAC_SLEEP) is reported by registry.simulate, not a
# local table here.

# The Spotify family (pause/resume/next/prev/now/like/volume/transfer/playlist/
# radio/play) was migrated to the dispatch registry (s58) — its routing is now
# reported by registry.simulate, not a local table here.

# The macOS volume/brightness/dark/lock/screenshot/app-list block was migrated
# to the dispatch registry (s58) — its routing is now reported by
# registry.simulate, not a local table here.

# The post-planner macOS app-control verbs (close/quit/switch/open) were
# migrated to the registry's POST_PLANNER_ROUTES (s59) — their routing is
# reported by `registry.simulate(routing, POST_PLANNER_ROUTES)`, after the
# planner, not by a local table here.


def _try_table(table, text: str) -> Optional[DispatchOutcome]:
    """Walk an ordered list of (regex, name, mode, group_id). First
    match wins. `group_id` may be int (numeric group), str (named
    group), or None (no capture). Returns the DispatchOutcome on
    match, None on miss."""
    for rx, name, mode, group_id in table:
        m = rx.match(text)
        if not m:
            continue
        captured: Optional[str] = None
        if group_id is not None:
            # Catch both IndexError (bad numeric index) and the re
            # module's named-group error if the group doesn't exist.
            # Either points to a desync between this table and the
            # source regex — surface as no-capture rather than crash.
            try:
                raw = m.group(group_id)
                if raw is not None:
                    captured = raw.strip()
            except Exception:
                pass
        return DispatchOutcome(
            mode=mode,
            matcher=name,
            capture=captured,
        )
    return None


def simulate_dispatch(text: str) -> DispatchOutcome:
    """Run a single input through the simulated dispatch.

    Returns the FIRST matcher to fire, in the same order as
    `_process_core`. If nothing matches, returns mode='OFFLINE',
    matcher='fallthrough', with `would_use_tools` set per
    TOOL_HINT_RE — that's the signal the real orchestrator uses to
    decide between chat_with_tools (LLM with tools) and pure chat.

    Flow control replies (sim/não) are NOT routed here — those depend
    on pending state that the simulator doesn't model. They're tested
    via REGEX cases directly against APPROVAL_RE / DENIAL_RE.
    """
    # 1. Normalisation — same as orchestrator entry.
    routing = strip_vocatives((text or "").strip())
    if not routing:
        return DispatchOutcome(mode="INPUT_ERROR", matcher="empty_input")
    # 1.5 Courtesy wrapper strip (s75, Onda 8 A) — `route_text` feeds the
    # registry/planner segments exactly like _process_core's dispatch
    # phase; gate/approval/tool-hint checks below keep `routing`,
    # mirroring the orchestrator's use of the original text outside the
    # dispatch phase. (strip_courtesy guards approval/denial internally.)
    route_text = normalise_stt_aliases(strip_courtesy(routing))

    # 2. Registry-backed routes (single source of truth — see dispatch/
    # registry.py). EVERY regex-first dispatch family now lives in the registry
    # (migrated s58/s59: studier, contacts, email, silence, mac_sleep, Spotify,
    # macOS, reminder, calendar, notes). The walk reports the front-door routing
    # match in the orchestrator's real execution order, without inspecting the
    # body or executing (eval-level fidelity — the real orchestrator's body
    # parsers can still decline and fall through to the LLM path). The
    # LIST_EMAILS-before-SHOW_STUDY precedence is preserved inside the registry.
    sim = dispatch_registry.simulate(route_text)
    if sim is not None:
        return DispatchOutcome(mode=sim.mode, matcher=sim.matcher, capture=sim.capture)
    if PLANNER_TRIGGER_RE.match(route_text):
        return DispatchOutcome(
            mode="PLANNER", matcher="PLANNER_TRIGGER_RE"
        )

    # 6.5 Auto-planner for natural multi-intent (sessão 15) — sits
    # AFTER the explicit "planeja:" trigger so a phrase with both the
    # explicit prefix AND a multi-intent body still reports the
    # explicit matcher. The simulator only reports the routing
    # decision; the real orchestrator's _handle_planner_request may
    # still fail to decompose and fall through to chat_with_tools.
    if multi_intent_match(route_text):
        return DispatchOutcome(
            mode="PLANNER", matcher="multi_intent_autoroute"
        )

    # 7. Second registry segment — the broad macOS app-control verbs
    # (close/quit/switch/open), which run AFTER the planner (s59). Same
    # single-source registry, a different ordered list (POST_PLANNER_ROUTES).
    post = dispatch_registry.simulate(route_text, dispatch_registry.POST_PLANNER_ROUTES)
    if post is not None:
        return DispatchOutcome(mode=post.mode, matcher=post.matcher, capture=post.capture)

    # 7.5 Gate STATE-CHANGE phrasings (lease revoke / grant). The gate's
    # dynamic part stays skipped (search triggers depend on lease state),
    # but revoke and grant are static regexes whose real-orchestrator mode
    # is always SYSTEM_UPDATE regardless of state (LEASE_REVOKED /
    # ALREADY_OFFLINE / LEASE_GRANTED all map to it). Simulating them lets
    # the eval set pin the M6-A1 collision: "desliga a internet" must land
    # HERE, not in MAC_QUIT_APP_RE above.
    lowered = routing.lower()
    if any(re.search(p, lowered) for p in REVOKE_PATTERNS):
        return DispatchOutcome(mode="SYSTEM_UPDATE", matcher="gate_revoke")
    if any(re.search(p, lowered) for p in LEASE_PATTERNS) or any(
        re.search(p, lowered) for p in GRANT_PATTERNS
    ):
        return DispatchOutcome(mode="SYSTEM_UPDATE", matcher="gate_lease_grant")

    # 8. Approval / denial (when no pending state, they should fall
    # through to LLM in the real orchestrator; the simulator reports
    # the matcher so eval cases can assert "this WOULD have approved
    # if there was a pending action").
    if APPROVAL_RE.match(routing):
        return DispatchOutcome(mode="APPROVAL", matcher="APPROVAL_RE")
    if DENIAL_RE.match(routing):
        return DispatchOutcome(mode="DENIAL", matcher="DENIAL_RE")

    # 9. Fallthrough — would head to reflex → skill registry → LLM.
    # We report whether TOOL_HINT_RE would have nudged this into
    # chat_with_tools (LLM with tools available) vs pure chat.
    would_use_tools = bool(TOOL_HINT_RE.search(routing))
    return DispatchOutcome(
        mode="OFFLINE",
        matcher="fallthrough",
        would_use_tools=would_use_tools,
    )
