"""
Regex dispatch patterns for the orchestrator.

Each submodule owns the regex constants for one skill / verb family
or one decision family:

Skill-specific triggers:
  - spotify           — play / pause / resume / next / prev / volume / like / now
  - reminders         — "me lembre de X amanhã às 4h" trigger + body splitter
  - calendar          — "adicione no calendário X" / "agenda evento X" trigger
  - planner_trigger   — "planeja: …" multi-step trigger
  - studier_cmds      — Nightly Studier queue + review/approval verbs

Generic command surface:
  - triggers          — read / write / creative / search-strip / list-emails
  - flow_control      — approval / denial / silence / batch-ref (replies to prompts)
  - heuristics        — tool-hint / offline-hint / realtime-marker (routing hints)

The orchestrator imports from this package and re-binds the regexes
as class attributes on `CortexOrchestrator` so existing test access
patterns like `CortexOrchestrator._SPOTIFY_PAUSE_RE` keep working
without modification. New code should prefer importing from the
submodules directly:

    from src.core.dispatch.flow_control import APPROVAL_RE

Phases done so far:
  - phase 1 (a4d28ea): skill-specific triggers — spotify/reminders/
    calendar/planner_trigger.
  - phase 2 (a99cd0c): generic commands + studier verbs (one grab-bag
    `commands.py` module — superseded below).
  - phase 3A (9866920): pending-action machinery extracted to
    `src/core/orchestrator_pending.py` (PendingActionMixin).
  - phase 3B (3da8e78): full staging lifecycle moves to
    `src/core/staging_mixin.py` (StagingMixin) — adds
    _execute_tool / _execute_tool_unstaged / _execute_staged_plan.
  - phase 2.1 (this): split `commands.py` into three coherent modules
    by function — flow_control / triggers / heuristics — so each
    file holds patterns that belong together by purpose, not by
    "what was left over". `commands.py` removed.
"""
from src.core.dispatch import (
    calendar,
    flow_control,
    heuristics,
    planner_trigger,
    reminders,
    spotify,
    studier_cmds,
    triggers,
)

__all__ = [
    "calendar",
    "flow_control",
    "heuristics",
    "planner_trigger",
    "reminders",
    "spotify",
    "studier_cmds",
    "triggers",
]
