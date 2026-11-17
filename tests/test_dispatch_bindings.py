"""
Identity guards for the orchestrator dispatch split.

The orchestrator re-binds regex constants from `src/core/dispatch/*`
and methods from `src/core/staging_mixin.py` so legacy access
patterns like `CortexOrchestrator._READ_RE` and
`CortexOrchestrator._execute_pending_action` keep working without
touching test code.

Both surfaces (module-level import + class-attribute reach-in) must
point at the SAME object — otherwise patches/edits made in one place
would silently drift from the other. Each test below asserts `is`
identity, which is what catches drift (an accidental re-compile or a
copy-paste of the body back into the orchestrator).

Add a line here whenever a new dispatch symbol gets a class-attr
re-binding. The cost of forgetting is exactly the bug this file
exists to catch.
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
from src.core.orchestrator import CortexOrchestrator
from src.core import staging_mixin


class TestSpotifyBindings:
    def test_play(self):
        assert CortexOrchestrator._SPOTIFY_PLAY_RE is spotify.SPOTIFY_PLAY_RE

    def test_device_hint(self):
        assert (
            CortexOrchestrator._SPOTIFY_DEVICE_HINT_RE
            is spotify.SPOTIFY_DEVICE_HINT_RE
        )

    def test_pause(self):
        assert CortexOrchestrator._SPOTIFY_PAUSE_RE is spotify.SPOTIFY_PAUSE_RE

    def test_resume(self):
        assert CortexOrchestrator._SPOTIFY_RESUME_RE is spotify.SPOTIFY_RESUME_RE

    def test_next(self):
        assert CortexOrchestrator._SPOTIFY_NEXT_RE is spotify.SPOTIFY_NEXT_RE

    def test_prev(self):
        assert CortexOrchestrator._SPOTIFY_PREV_RE is spotify.SPOTIFY_PREV_RE

    def test_now(self):
        assert CortexOrchestrator._SPOTIFY_NOW_RE is spotify.SPOTIFY_NOW_RE

    def test_volume(self):
        assert CortexOrchestrator._SPOTIFY_VOLUME_RE is spotify.SPOTIFY_VOLUME_RE

    def test_like(self):
        assert CortexOrchestrator._SPOTIFY_LIKE_RE is spotify.SPOTIFY_LIKE_RE


class TestRemindersBindings:
    def test_trigger(self):
        assert (
            CortexOrchestrator._REMINDER_TRIGGER_RE
            is reminders.REMINDER_TRIGGER_RE
        )

    def test_body(self):
        assert CortexOrchestrator._REMINDER_BODY_RE is reminders.REMINDER_BODY_RE


class TestCalendarBindings:
    def test_trigger(self):
        assert (
            CortexOrchestrator._CALENDAR_TRIGGER_RE
            is calendar.CALENDAR_TRIGGER_RE
        )

    def test_trigger_content(self):
        assert (
            CortexOrchestrator._CALENDAR_TRIGGER_CONTENT_RE
            is calendar.CALENDAR_TRIGGER_CONTENT_RE
        )

    def test_vague(self):
        assert (
            CortexOrchestrator._VAGUE_CALENDAR_RE
            is calendar.VAGUE_CALENDAR_RE
        )


class TestPlannerTriggerBinding:
    def test_trigger(self):
        assert (
            CortexOrchestrator._PLANNER_TRIGGER_RE
            is planner_trigger.PLANNER_TRIGGER_RE
        )


class TestTriggersBindings:
    """Command triggers — verbs that dispatch to a handler."""

    def test_read(self):
        assert CortexOrchestrator._READ_RE is triggers.READ_RE

    def test_write(self):
        assert CortexOrchestrator._WRITE_RE is triggers.WRITE_RE

    def test_creative(self):
        assert CortexOrchestrator._CREATIVE_RE is triggers.CREATIVE_RE

    def test_search_strip(self):
        assert CortexOrchestrator._SEARCH_STRIP_RE is triggers.SEARCH_STRIP_RE

    def test_list_emails(self):
        assert CortexOrchestrator._LIST_EMAILS_RE is triggers.LIST_EMAILS_RE


class TestFlowControlBindings:
    """Conversation flow control — replies to prompts."""

    def test_approval(self):
        assert CortexOrchestrator._APPROVAL_RE is flow_control.APPROVAL_RE

    def test_denial(self):
        assert CortexOrchestrator._DENIAL_RE is flow_control.DENIAL_RE

    def test_silence(self):
        assert CortexOrchestrator._SILENCE_RE is flow_control.SILENCE_RE

    def test_all_ref(self):
        assert CortexOrchestrator._ALL_REF_RE is flow_control.ALL_REF_RE


class TestHeuristicsBindings:
    """Routing-decision heuristics — informational, not dispatching."""

    def test_tool_hint(self):
        assert CortexOrchestrator._TOOL_HINT_RE is heuristics.TOOL_HINT_RE

    def test_offline_hint(self):
        assert (
            CortexOrchestrator._OFFLINE_HINT_RE is heuristics.OFFLINE_HINT_RE
        )

    def test_realtime_marker(self):
        assert (
            CortexOrchestrator._REALTIME_MARKER_RE
            is heuristics.REALTIME_MARKER_RE
        )

    def test_math_only(self):
        assert CortexOrchestrator._MATH_ONLY_RE is heuristics.MATH_ONLY_RE


class TestStudierBindings:
    """Nightly Studier verbs (phase 2 of the split)."""

    def test_study(self):
        assert CortexOrchestrator._STUDY_RE is studier_cmds.STUDY_RE

    def test_list_studies(self):
        assert (
            CortexOrchestrator._LIST_STUDIES_RE is studier_cmds.LIST_STUDIES_RE
        )

    def test_review(self):
        assert CortexOrchestrator._REVIEW_RE is studier_cmds.REVIEW_RE

    def test_show_study(self):
        assert CortexOrchestrator._SHOW_STUDY_RE is studier_cmds.SHOW_STUDY_RE

    def test_approve_study(self):
        assert (
            CortexOrchestrator._APPROVE_STUDY_RE
            is studier_cmds.APPROVE_STUDY_RE
        )

    def test_reject_study(self):
        assert (
            CortexOrchestrator._REJECT_STUDY_RE
            is studier_cmds.REJECT_STUDY_RE
        )


class TestStagingMixinBindings:
    """Phase 3A + 3B: StagingMixin must be in the MRO and all eight
    staging methods (5 from 3A + 3 from 3B) must resolve to the
    mixin's implementation (not a copy left behind in the
    orchestrator)."""

    def test_mixin_in_mro(self):
        assert (
            staging_mixin.StagingMixin
            in CortexOrchestrator.__mro__
        )

    # --- Phase 3A: pending action lifecycle ---

    def test_load(self):
        assert (
            CortexOrchestrator._load_pending_action
            is staging_mixin.StagingMixin._load_pending_action
        )

    def test_persist(self):
        assert (
            CortexOrchestrator._persist_pending_action
            is staging_mixin.StagingMixin._persist_pending_action
        )

    def test_human_confirmation_prompt(self):
        assert (
            CortexOrchestrator._human_confirmation_prompt
            is staging_mixin.StagingMixin._human_confirmation_prompt
        )

    def test_execute(self):
        assert (
            CortexOrchestrator._execute_pending_action
            is staging_mixin.StagingMixin._execute_pending_action
        )

    def test_cancel(self):
        assert (
            CortexOrchestrator._cancel_pending_action
            is staging_mixin.StagingMixin._cancel_pending_action
        )

    # --- Phase 3B: tool dispatch + plan executor ---

    def test_execute_tool(self):
        assert (
            CortexOrchestrator._execute_tool
            is staging_mixin.StagingMixin._execute_tool
        )

    def test_execute_tool_unstaged(self):
        assert (
            CortexOrchestrator._execute_tool_unstaged
            is staging_mixin.StagingMixin._execute_tool_unstaged
        )

    def test_execute_staged_plan(self):
        assert (
            CortexOrchestrator._execute_staged_plan
            is staging_mixin.StagingMixin._execute_staged_plan
        )
