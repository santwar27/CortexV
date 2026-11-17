"""Model profile — the single flip for the brain swap (C3: M1-B3 + M2-B1 +
M11 §6, the most-cited item of the Fable 5 review).

The 8B needed a tower of guards the review calls "8B-bound": artefact
strippers, a hallucination detector, a refusal retry, inline language
directives, an empty-specs anti-fabrication gate. They are load-bearing
TODAY and (hypothesis) dead weight on a stronger model. Before this module
the inventory of those guards existed only in review prose — the day of the
swap would be archaeology. Now it is a table, and the swap is a flip:

    export CORTEX_MODEL_PROFILE=gemma4

Every wired guard call site asks `guard_active("<name>")`; the table says
in which profiles the guard is OFF. The `off_in` sets encode the review's
STARTING HYPOTHESIS for swap day — each one is falsifiable per guard: if
`make eval-llm-rebaseline` on the new model shows artefacts coming back,
remove the profile from that guard's `off_in` and the guard is back,
one line, no archaeology.

Default profile is "8b" (all guards ON), so this module changes nothing
until the env var is set. Unknown values fall back to "8b" — the safe
direction is guards-on.

Inventory-only entries (`wired=False`) are guards whose flip already has a
mechanism of its own (env var / explicit decision) or whose off-behaviour
must be decided with swap-day data; they live here so the inventory is
complete, and the anti-drift test keeps their `site` pointers honest.
"""

import os

PROFILES = ("8b", "gemma4", "frontier")
_DEFAULT = "8b"

_warned_unknown = False


def model_profile() -> str:
    """Resolve the active profile from CORTEX_MODEL_PROFILE (default 8b).

    Read per call (cheap) so tests can flip it with monkeypatch and a
    long-lived process picks up a corrected value on restart semantics
    identical to every other CORTEX_* env knob.
    """
    raw = (os.getenv("CORTEX_MODEL_PROFILE") or _DEFAULT).strip().lower()
    if raw not in PROFILES:
        global _warned_unknown
        if not _warned_unknown:
            print(
                f"[model_profile] unknown CORTEX_MODEL_PROFILE={raw!r} — "
                f"falling back to '8b' (all guards ON). Valid: {PROFILES}"
            )
            _warned_unknown = True
        return _DEFAULT
    return raw


# name → spec. `site` is "relative/path.py::symbol"; the anti-drift test in
# tests/test_model_profile.py asserts the symbol still exists in that file,
# so a refactor that moves or deletes a guard must update this table.
GUARDS = {
    # ── wired: the call site consults guard_active() ────────────────────
    "json_artifact_sanitiser": {
        "off_in": frozenset({"gemma4", "frontier"}),
        "wired": True,
        "site": "src/core/engine.py::_sanitise_stream_line",
        "note": (
            "M2-B1: the artefact regex trio + JSON unwrap + leak guard, on "
            "both the full reply and the C9 stream filter. 8B reaches for "
            "{'kw':...} envelopes under pressure; a stronger model should "
            "not. OFF means replies pass through verbatim."
        ),
    },
    "hallucination_detector": {
        "off_in": frozenset({"gemma4", "frontier"}),
        "wired": True,
        "site": "src/core/engine.py::chat_with_tools",
        "note": (
            "M2-B1: detect_hallucinated_action on the no-tool-call reply "
            "('Pronto, criei o lembrete' with no tool run). Conservative "
            "patterns, but the failure mode it polices is 8B-specific. "
            "NOT detect_fake_tool_call: a call written as text is a fact "
            "(nothing ran), checked under every profile."
        ),
    },
    "knowledge_refusal_retry": {
        "off_in": frozenset({"gemma4", "frontier"}),
        "wired": True,
        "site": "src/core/engine.py::_retry_past_refusal",
        "note": (
            "M2-B1: the s38 [KNOWLEDGE]-refusal retry (one extra LLM call, "
            "history stripped). The induced-refusal pathology was measured "
            "on the 8B; OFF saves the double call on a model that obeys "
            "the [KNOWLEDGE] directive."
        ),
    },
    "inline_language_directive": {
        "off_in": frozenset({"gemma4", "frontier"}),
        "wired": True,
        "site": "src/core/engine.py::_language_directive",
        "note": (
            "M2-B1: '[Responda em português brasileiro]' prepended to every "
            "turn because the 8B ignores the system-prompt mirror-language "
            "rule. OFF trusts the system prompt (and stops polluting the "
            "Ollama prefix cache with a per-turn marker)."
        ),
    },
    "tool_specs_hint_gate": {
        "off_in": frozenset({"gemma4", "frontier"}),
        "wired": True,
        "site": "src/core/orchestrator.py::_tool_specs_for",
        "note": (
            "M1-B3: empty specs list when only music/macOS hinted, so the "
            "8B can't fabricate create_reminder from 'amo música' (s38). "
            "OFF hands the full TOOL_SPECS to every tool-hinted turn and "
            "trusts the model not to fabricate."
        ),
    },
    "episodic_no_record": {
        "off_in": frozenset({"frontier"}),
        "wired": True,
        "site": "src/core/engine.py::_episodic_no_record",
        "note": (
            "s84: a question about Pedro's own past whose subject appears "
            "nowhere in the context gets a fixed 'no record — want to tell "
            "me?' instead of an LLM call. The 8B invented 4 in 25 (a dog's "
            "name, a Christmas trip). Kept ON for gemma4 — its rate was never "
            "measured; flip it off only after the same sampling run says so."
        ),
    },
    # ── inventory-only: flip documented, site pinned, no new wiring ─────
    "chat_temperature_anchor": {
        "off_in": frozenset(),
        "wired": False,
        "site": "src/core/engine.py::_CHAT_TEMPERATURE",
        "note": (
            "M2-B1: the s38 anti-refusal 0.3 default. Already has its own "
            "flip — CORTEX_CHAT_TEMPERATURE env / constructor arg (M2-A6); "
            "swap day sets the env instead of consulting this table."
        ),
    },
    "needs_realtime_probe": {
        "off_in": frozenset(),
        "wired": False,
        "site": "src/core/orchestrator.py::_needs_realtime",
        "note": (
            "M1-B3: one LLM call (~1-2s) to classify 'needs internet'. "
            "Whether a stronger model keeps it, replaces it, or absorbs it "
            "is a ROUTING design decision — decide with swap-day latency "
            "data, not by profile flip."
        ),
    },
    "lease_phrase_regexes": {
        "off_in": frozenset(),
        "wired": False,
        "site": "src/core/gatekeeper.py::GRANT_PATTERNS",
        "note": (
            "M6-B1 lists the grant/revoke phrasings as 8B-era cost, but "
            "they are the deterministic voice path into gate state — "
            "SOBERANIA r4-adjacent. Removing them means an LLM mediates "
            "lease changes; that needs an explicit Pedro decision, never "
            "a profile flip."
        ),
    },
}


def guard_active(name: str) -> bool:
    """True when the named guard should run under the active profile.

    Raises KeyError on an unknown name — a typo at a call site must fail
    the suite loudly, not silently default a guard on or off.
    """
    return model_profile() not in GUARDS[name]["off_in"]
