"""
Skill Fast Path Registry — Tier 3.2.

Dynamic, Palace-persisted registry that bypasses the heavy LLM pipeline
for repetitive user inputs. A skill is a trigger regex paired with a
deterministic handler. When the user's text matches an enabled skill's
regex, the orchestrator executes the handler directly and returns —
zero LLM, sub-100ms.

Sits between the Reflex Layer (universal hardcoded greetings/time/date)
and the LLM dispatch. The existing regex command paths (Spotify,
reminder, calendar) stay where they are: those are stateful action
dispatchers, not the same shape as a "skill response". A skill is
specifically: a stored pattern → deterministic reply.

Design constraints:
  - Match must stay <1ms. We compile every enabled skill's regex once
    at __init__ time and on each register/remove, then linearly scan.
    Linear is fine to thousands; if we ever need more we move to
    Aho-Corasick or a trie.
  - Handler execution is deterministic. No LLM round-trips inside a
    skill — that would defeat the latency win. For LLM-augmented
    behaviours, the user should go through the normal pipeline.
  - Skills are persisted in the Palace `skills` table (Schema v3),
    NOT in code, so they can be added/removed at runtime without a
    deploy. Future voice command "registra skill ..." will wire to
    register().

Handler types:
  - "static_response": payload = {"text": "..."}. Returns the literal
    text. Most useful for frequently-asked personal facts where the
    answer never changes ("qual meu CPF" → stored value).
  - "tool_sequence" (M8-A1, s66): payload = {"steps": [{"tool": ...,
    "args": {...}}, ...]}. A macro — one trigger fires N tool steps in
    order. This is the missing middle between the skill_promoter
    (which detects repeated multi-step workflows) and this registry
    (which until s66 could only return text). THE GATE CONTRACT:
    every step passes through tool_risk at execution time — S1/S2
    steps run via an injected executor, an S3 step is DIVERTED to an
    injected stager (the same human sim/não gate as a direct S3 tool
    call). Without both callbacks the handler refuses to run
    (fail-closed → falls through to the LLM). Otherwise a macro would
    be an S3 bypass: "registra skill" once, fire send_email forever
    without confirmation. Registration enforces: at most ONE S3 step,
    and it must be the LAST one (steps after a staged action would
    silently never run).

Future handler types (explicitly rejected to keep each rev shippable):
  - "prompt_template": payload = {"system": ..., "user_template": ...}
    — overlap with engine.chat, defer until we know we need it.

The registry tracks usage statistics (invocation_count, last_used_at)
on every hit. Future revisions can use these to demote stale skills
or surface candidates for promotion to hardcoded dispatch.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.tool_risk import S3, risk_level

log = logging.getLogger(__name__)

# Executor for one S1/S2 macro step: (tool, args) -> user-facing result
# string. The orchestrator injects a wrapper around its per-tool dispatch.
StepRunner = Callable[[str, Dict[str, Any]], Any]
# Stager for an S3 macro step: (tool, args) -> the human confirmation
# prompt. The orchestrator injects the SAME staging used by direct S3
# tool calls (_pending_action + sim/não), so a macro can never approve
# itself.
StepStager = Callable[[str, Dict[str, Any]], Any]


# ---------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    """In-memory representation of a row from the `skills` table.

    `_compiled` is the pre-built regex pattern object; kept on the
    dataclass so match() doesn't have to recompile on every input.
    """
    id: int
    name: str
    trigger_regex: str
    handler_type: str
    handler_payload: Dict[str, Any]
    enabled: bool
    invocation_count: int
    _compiled: re.Pattern


# Names allowed for handler_type — keep this tight so a malformed row
# can't trigger arbitrary behaviour.
_KNOWN_HANDLERS = frozenset({"static_response", "tool_sequence"})

# Macro length cap. The promoter mines 2..4-step workflows
# (skill_promoter MIN/MAX_SEQ_LEN); 5 leaves one step of headroom for
# hand-written macros without letting a row turn into a programme.
MAX_SEQUENCE_STEPS = 5


def _parse_steps(payload: Dict[str, Any]):
    """Validate and normalise a tool_sequence payload.

    Returns a list of (tool, args) tuples on success, or an error
    MESSAGE STRING on bad shape. Shared by register-time validation
    (which raises on the message) and execute-time defence (which logs
    and falls through — a row inserted via direct SQL must not crash
    the hot path)."""
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return "tool_sequence handler requires payload.steps (non-empty list)"
    if len(steps) > MAX_SEQUENCE_STEPS:
        return (
            f"tool_sequence allows at most {MAX_SEQUENCE_STEPS} steps "
            f"(got {len(steps)})"
        )
    parsed: List[Tuple[str, Dict[str, Any]]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"tool_sequence step {i} must be an object"
        tool = step.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            return f"tool_sequence step {i} requires tool (non-empty string)"
        args = step.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return f"tool_sequence step {i}: args must be an object"
        parsed.append((tool.strip(), args))
    return parsed


class SkillRegistryError(Exception):
    """Raised for caller-facing problems (duplicate name, invalid regex,
    unknown handler type). Internal SQL/JSON errors surface as the
    underlying exception type — caller handles those separately."""


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------


class SkillRegistry:
    """Palace-backed dynamic dispatch table for repetitive inputs.

    Construction loads all enabled skills into memory with compiled
    regexes. Subsequent register/remove calls refresh the in-memory
    cache so match() never hits SQLite on the hot path.
    """

    # Capping the regex compilation work means an attacker (or a
    # confused user) can't paste a million-character pattern that
    # blocks the orchestrator on registration.
    MAX_REGEX_LEN = 500
    MAX_NAME_LEN = 80
    MAX_PAYLOAD_LEN = 4000

    def __init__(self, palace: Any):
        self._palace = palace
        self._skills: List[Skill] = []
        self._reload()

    # ----- private helpers -----

    def _reload(self) -> None:
        """Re-read enabled skills from Palace and rebuild the in-memory
        list. Compile-failed rows are logged and skipped — a single
        broken regex must NOT make the whole registry unusable."""
        rows = self._palace.execute_read(
            "SELECT * FROM skills WHERE enabled = 1 ORDER BY id ASC"
        )
        compiled: List[Skill] = []
        for row in rows:
            try:
                pat = re.compile(row["trigger_regex"], re.IGNORECASE)
            except re.error as e:
                log.warning(
                    "SkillRegistry: skill %r has invalid regex %r (%s) — skipping",
                    row["name"], row["trigger_regex"], e,
                )
                continue
            try:
                payload = json.loads(row["handler_payload"] or "{}")
                if not isinstance(payload, dict):
                    payload = {}
            except json.JSONDecodeError:
                log.warning(
                    "SkillRegistry: skill %r has malformed payload — using {}",
                    row["name"],
                )
                payload = {}
            compiled.append(Skill(
                id=row["id"],
                name=row["name"],
                trigger_regex=row["trigger_regex"],
                handler_type=row["handler_type"],
                handler_payload=payload,
                enabled=bool(row["enabled"]),
                invocation_count=row["invocation_count"],
                _compiled=pat,
            ))
        self._skills = compiled

    def _validate(
        self,
        name: str,
        trigger_regex: str,
        handler_type: str,
        handler_payload: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, str]:
        """Centralised input validation for register(). Raises
        SkillRegistryError on any deviation. Returns (name, regex,
        handler_type, payload_json) — the form that lands in SQL."""
        if not isinstance(name, str) or not name.strip():
            raise SkillRegistryError("name is required")
        name = name.strip()
        if len(name) > self.MAX_NAME_LEN:
            raise SkillRegistryError(f"name longer than {self.MAX_NAME_LEN} chars")

        if not isinstance(trigger_regex, str) or not trigger_regex.strip():
            raise SkillRegistryError("trigger_regex is required")
        if len(trigger_regex) > self.MAX_REGEX_LEN:
            raise SkillRegistryError(f"trigger_regex longer than {self.MAX_REGEX_LEN} chars")
        try:
            re.compile(trigger_regex)
        except re.error as e:
            raise SkillRegistryError(f"invalid regex: {e}") from e

        if handler_type not in _KNOWN_HANDLERS:
            raise SkillRegistryError(
                f"unknown handler_type {handler_type!r} (known: "
                f"{sorted(_KNOWN_HANDLERS)})"
            )

        payload_json = json.dumps(handler_payload or {}, ensure_ascii=False)
        if len(payload_json) > self.MAX_PAYLOAD_LEN:
            raise SkillRegistryError(
                f"handler_payload serialises to {len(payload_json)} bytes, "
                f"limit is {self.MAX_PAYLOAD_LEN}"
            )

        # Per-handler payload shape checks.
        if handler_type == "static_response":
            payload = handler_payload or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SkillRegistryError(
                    "static_response handler requires payload.text (non-empty string)"
                )

        if handler_type == "tool_sequence":
            steps = _parse_steps(handler_payload or {})
            if isinstance(steps, str):  # shape error message
                raise SkillRegistryError(steps)
            # The gate contract, enforced at the door: an S3 step stages
            # and ends the turn waiting for sim/não, so anything after it
            # would silently never run — and more than one would need a
            # chained-approval flow that doesn't exist. Reject both shapes
            # now instead of surprising at execution time.
            s3_at = [i for i, (tool, _) in enumerate(steps)
                     if risk_level(tool) == S3]
            if len(s3_at) > 1:
                raise SkillRegistryError(
                    "tool_sequence allows at most ONE S3 step per macro "
                    f"(got {len(s3_at)}: steps {s3_at})"
                )
            if s3_at and s3_at[0] != len(steps) - 1:
                raise SkillRegistryError(
                    f"the S3 step (step {s3_at[0]}, "
                    f"{steps[s3_at[0]][0]!r}) must be the LAST step — it "
                    "stages for human confirmation, so later steps would "
                    "never run"
                )

        return name, trigger_regex, handler_type, payload_json

    # ----- public API -----

    def register(
        self,
        name: str,
        trigger_regex: str,
        handler_type: str = "static_response",
        handler_payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a new skill and refresh the in-memory cache. Returns
        the new row id. Raises SkillRegistryError on duplicate name or
        any validation failure. Use update() to modify an existing
        skill (not yet implemented in v1 — caller can remove + register)."""
        name, trigger_regex, handler_type, payload_json = self._validate(
            name, trigger_regex, handler_type, handler_payload,
        )

        now = datetime.now(timezone.utc).isoformat()
        try:
            _, lastrowid = self._palace.execute_write(
                """
                INSERT INTO skills
                    (name, trigger_regex, handler_type, handler_payload,
                     enabled, invocation_count, created_at)
                VALUES (?, ?, ?, ?, 1, 0, ?)
                """,
                (name, trigger_regex, handler_type, payload_json, now),
            )
        except Exception as e:
            # UNIQUE violation on name lands here as sqlite3.IntegrityError.
            # Re-raise as our caller-facing exception so the API surface
            # stays consistent.
            raise SkillRegistryError(f"could not register skill {name!r}: {e}") from e

        self._reload()
        return lastrowid

    def remove(self, name: str) -> bool:
        """Delete a skill by name. Returns True if a row was deleted,
        False if no such skill existed. Cache is refreshed on every
        call."""
        rowcount, _ = self._palace.execute_write(
            "DELETE FROM skills WHERE name = ?", (name,),
        )
        self._reload()
        return rowcount > 0

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Toggle a skill on/off without deleting its persistent state
        (preserves invocation_count + last_used_at). Returns True if a
        row was updated."""
        rowcount, _ = self._palace.execute_write(
            "UPDATE skills SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        self._reload()
        return rowcount > 0

    def list_all(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Return raw rows for inspection / admin UI. Not used on the
        hot path."""
        sql = "SELECT * FROM skills"
        params: tuple = ()
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY id ASC"
        rows = self._palace.execute_read(sql, params)
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["handler_payload"] = json.loads(d.get("handler_payload") or "{}")
            except json.JSONDecodeError:
                d["handler_payload"] = {}
            out.append(d)
        return out

    def match(self, text: str) -> Optional[Skill]:
        """Return the first enabled skill whose compiled regex matches
        the input. None when nothing matches. Pure read — does NOT
        bump invocation_count. Call execute() to register the hit."""
        if not isinstance(text, str) or not text.strip():
            return None
        for skill in self._skills:
            if skill._compiled.search(text):
                return skill
        return None

    def execute(
        self,
        skill: Skill,
        text: str,
        run_step: Optional[StepRunner] = None,
        stage_step: Optional[StepStager] = None,
    ) -> Optional[str]:
        """Run the skill's handler and return the user-facing reply.
        Bumps invocation_count and last_used_at as a side effect.
        Returns None on handler failure so the orchestrator can fall
        through to the LLM rather than serving a stale or wrong reply.

        `run_step`/`stage_step` are the tool_sequence gate plumbing
        (see module docstring); static_response ignores them. Unknown
        handler_type returns None and logs a warning — defensive
        against a row that survived validation (e.g. via direct SQL)."""
        bump = True
        try:
            if skill.handler_type == "static_response":
                reply = skill.handler_payload.get("text")
                if not isinstance(reply, str) or not reply.strip():
                    log.warning(
                        "SkillRegistry.execute: skill %r has no text payload",
                        skill.name,
                    )
                    return None
            elif skill.handler_type == "tool_sequence":
                result = self._execute_sequence(skill, run_step, stage_step)
                if result is None:
                    return None
                reply, bump = result
            else:
                log.warning(
                    "SkillRegistry.execute: skill %r has unknown handler_type %r",
                    skill.name, skill.handler_type,
                )
                return None
        except Exception:
            log.exception(
                "SkillRegistry.execute: handler for %r raised", skill.name,
            )
            return None

        # Update stats only after we know we're about to return a real
        # reply. A bumped count on a failed execution would skew the
        # promotion / demotion signal. (`bump` is False when a macro
        # stopped mid-sequence on a step failure — the partial report is
        # a real reply, but not a success worth counting.)
        return self._bump_and_return(skill, reply) if bump else reply

    def _execute_sequence(
        self,
        skill: Skill,
        run_step: Optional[StepRunner],
        stage_step: Optional[StepStager],
    ) -> Optional[Tuple[str, bool]]:
        """Run a tool_sequence macro under the per-step gate contract.

        Returns (reply, bump) — bump False when the macro stopped on a
        step failure — or None when the macro cannot run AT ALL (bad
        shape, missing callbacks). The None/raise distinction matters:
        None means "nothing executed, safe to fall through to the LLM";
        once any step has run we NEVER return None, because the LLM
        fallback could re-execute side effects. All refusal checks
        therefore happen BEFORE step one.
        """
        steps = _parse_steps(skill.handler_payload)
        if isinstance(steps, str):
            # Row bypassed register() (direct SQL / hand edit) — refuse.
            log.warning(
                "SkillRegistry: tool_sequence %r refused: %s", skill.name, steps,
            )
            return None
        if run_step is None:
            # The caller didn't wire the gated executor — e.g. a bare
            # registry used outside the orchestrator. Fail closed.
            log.warning(
                "SkillRegistry: tool_sequence %r refused: no run_step "
                "executor injected (macros only run through the gated "
                "orchestrator path)", skill.name,
            )
            return None
        levels = [risk_level(tool) for tool, _ in steps]
        if S3 in levels and stage_step is None:
            # An S3 step with nowhere to stage = the macro WOULD bypass
            # the human gate. Refuse before anything runs.
            log.warning(
                "SkillRegistry: tool_sequence %r refused: contains an S3 "
                "step (%s) and no stage_step gate was injected",
                skill.name, steps[levels.index(S3)][0],
            )
            return None

        parts: List[str] = []
        for i, ((tool, args), level) in enumerate(zip(steps, levels)):
            if level == S3:
                # Diverted to the SAME human gate as a direct S3 call.
                # register() guarantees this is the last step; the
                # runtime re-check covers a tool whose risk was promoted
                # to S3 after the macro was registered. A raising stager
                # is reported, not propagated — earlier steps already
                # ran, so returning None (→ LLM fallback) could
                # re-execute their side effects.
                try:
                    out = stage_step(tool, dict(args))
                except Exception as e:
                    log.exception(
                        "SkillRegistry: macro %r step %d (%s) stager raised",
                        skill.name, i, tool,
                    )
                    parts.append(
                        f"A macro '{skill.name}' parou no passo {i + 1} "
                        f"({tool}): {e}"
                    )
                    return ("\n".join(p for p in parts if p), False)
                parts.append(out if isinstance(out, str) else str(out))
                skipped = len(steps) - i - 1
                if skipped:
                    parts.append(
                        f"(macro '{skill.name}': {skipped} passo(s) após a "
                        "confirmação não foram executados)"
                    )
                break
            try:
                out = run_step(tool, dict(args))
            except Exception as e:
                log.exception(
                    "SkillRegistry: macro %r step %d (%s) raised",
                    skill.name, i, tool,
                )
                parts.append(
                    f"A macro '{skill.name}' parou no passo {i + 1} "
                    f"({tool}): {e}"
                )
                return ("\n".join(p for p in parts if p), False)
            out = out if isinstance(out, str) else str(out)
            if out.startswith("[SYSTEM ERROR"):
                # Tool-level failure convention. Stop here — later steps
                # may depend on this one having worked.
                parts.append(
                    f"A macro '{skill.name}' parou no passo {i + 1} "
                    f"({tool}): {out}"
                )
                return ("\n".join(p for p in parts if p), False)
            if out.strip():
                parts.append(out)

        reply = "\n".join(p for p in parts if p).strip()
        return (reply or "Feito.", True)

    def _bump_and_return(self, skill: Skill, reply: str) -> str:
        """Record the successful invocation (count + last_used_at) and
        hand the reply back. Stats are a soft signal — a failed bump
        never fails the user-visible reply."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._palace.execute_write(
                """
                UPDATE skills
                   SET invocation_count = invocation_count + 1,
                       last_used_at = ?
                 WHERE id = ?
                """,
                (now, skill.id),
            )
        except Exception:
            # Stats are a soft signal; don't fail the user-visible
            # reply because the bump didn't land.
            log.exception("SkillRegistry.execute: stats bump failed")

        return reply
