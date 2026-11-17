"""
Background-safe gate routing in StagingMixin._execute_tool, plus the
_run_action refactor parity.

The contract: a destructive (S3) tool dispatched while the orchestrator is
in BACKGROUND mode must land in the gate batch queue and must NOT touch the
inline `_pending_action` slot (which belongs to whoever is chatting
interactively). The interactive path must be byte-for-byte unchanged.

Builds the orchestrator via __new__ (no __init__) — the same trick the
existing refactor-parity tests use — so we don't pay the engine/Palace
startup and can set just the attributes these methods read.
"""
import pytest

from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.gate_queue import GateQueue
from src.core.orchestrator import CortexOrchestrator


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o.background = False
    o.background_origin = "background"
    # Point the module-level queue that _execute_tool enqueues into at an
    # isolated temp file (no real ~/.cortex write).
    q = GateQueue(str(tmp_path / "gate_queue.jsonl"))
    monkeypatch.setattr("src.core.staging_mixin.gate_queue", q)
    return o, q


class TestBackgroundEnqueues:
    def test_s3_in_background_enqueues_not_stages(self, orch):
        o, q = orch
        o.background = True
        reply = o._execute_tool("send_email", {"to": "a@b", "subject": "s", "body": "x"})
        # Returns the skip-followup marker (no second LLM round) + a notice.
        assert reply.startswith(SKIP_FOLLOWUP_MARKER)
        assert "enfileirada" in reply
        # Inline slot untouched...
        assert o._pending_action is None
        # ...action is in the queue.
        pend = q.pending()
        assert len(pend) == 1
        assert pend[0]["type"] == "send_email"
        assert pend[0]["args"] == {"to": "a@b", "subject": "s", "body": "x"}

    def test_background_origin_label_is_carried(self, orch):
        o, q = orch
        o.background = True
        o.background_origin = "scheduled:weekly"
        o._execute_tool("create_reminder", {"title": "x"})
        assert q.pending()[0]["origin"] == "scheduled:weekly"

    def test_multiple_background_actions_all_queue(self, orch):
        o, q = orch
        o.background = True
        o._execute_tool("send_email", {"to": "a@b"})
        o._execute_tool("create_event", {"summary": "Sync"})
        types = sorted(e["type"] for e in q.pending())
        assert types == ["create_event", "send_email"]


class TestInteractiveUnchanged:
    def test_s3_interactive_still_stages_inline(self, orch):
        o, q = orch
        o.background = False
        # s85: a live send needs a recipient the user actually wrote.
        o.remember_emails_in("manda pra a@b.com")
        reply = o._execute_tool("send_email", {"to": "a@b.com", "subject": "s", "body": "x"})
        # Inline behaviour preserved: stages in _pending_action, returns the
        # confirmation brief, and queues NOTHING.
        assert reply.startswith(SKIP_FOLLOWUP_MARKER)
        assert "CONFIRMATION REQUIRED" in reply
        staged = dict(o._pending_action)
        staged.pop("staged_at", None)  # TTL stamp (M1-A9), time-dependent
        staged.pop("id", None)  # the action's identity (s88), random
        assert staged == {
            "type": "send_email",
            "args": {"to": "a@b.com", "subject": "s", "body": "x"},
            "origin": None,
        }
        assert q.pending() == []

    def test_background_default_is_false_via_getattr(self, tmp_path, monkeypatch):
        # An orchestrator that never set .background (e.g. older code path)
        # must behave interactively, not silently swallow actions.
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        o._PENDING_ACTION_FILE = str(tmp_path / "pa.json")
        q = GateQueue(str(tmp_path / "gq.jsonl"))
        monkeypatch.setattr("src.core.staging_mixin.gate_queue", q)
        o.remember_emails_in("manda pra a@b.com")   # s85: a written recipient
        o._execute_tool("send_email", {"to": "a@b.com"})
        assert o._pending_action is not None
        assert q.pending() == []


class TestRunActionRefactorParity:
    """_execute_pending_action now clears the slot then delegates to
    _run_action. Verify the slot is still cleared and the dispatch still
    routes (using the unknown-tool branch, which needs no integrations)."""

    def test_unknown_action_clears_slot_and_errors(self, orch):
        o, _ = orch
        o._pending_action = {"type": "bogus_tool", "args": {}}
        reply, mode = o._execute_pending_action()
        assert mode == "AGENT_ACTION_ERROR"
        assert "desconhecida" in reply
        assert o._pending_action is None  # slot cleared

    def test_run_action_does_not_touch_slot(self, orch):
        o, _ = orch
        o._pending_action = {"type": "sentinel", "args": {}}
        # _run_action is pure dispatch — must not clear the slot itself.
        reply, mode = o._run_action("bogus_tool", {})
        assert mode == "AGENT_ACTION_ERROR"
        assert o._pending_action == {"type": "sentinel", "args": {}}


class TestPolicyWiring:
    """The gate policy layer (Cap 13) pre-decides background actions.
    Default REVIEW keeps the old behaviour; AUTO_REJECT drops without
    running; AUTO_APPROVE runs via _run_action. Every outcome is recorded
    in the queue for audit."""

    def test_background_mac_sleep_auto_rejected(self, orch):
        o, q = orch
        o.background = True
        reply = o._execute_tool("mac_sleep", {})
        assert "recusada por política" in reply
        # Not left pending (a human never has to see it)...
        assert q.pending() == []
        # ...but recorded in the queue as rejected, for the audit trail.
        assert len(q._read_all()) >= 1
        rejected = [r for r in q._read_all() if r.get("event") == "rejected"]
        assert len(rejected) == 1
        # inline slot never touched
        assert o._pending_action is None

    def test_background_mac_quit_app_auto_rejected(self, orch):
        o, q = orch
        o.background = True
        o._execute_tool("mac_quit_app", {"app_name": "Safari"})
        assert q.pending() == []
        assert any(r.get("event") == "rejected" for r in q._read_all())

    def test_background_send_email_still_reviews(self, orch):
        # Default policy = REVIEW: unchanged from pre-policy behaviour.
        o, q = orch
        o.background = True
        o._execute_tool("send_email", {"to": "a@b"})
        assert len(q.pending()) == 1
        assert q.pending()[0]["type"] == "send_email"

    def test_auto_approve_runs_via_run_action(self, orch, monkeypatch):
        o, q = orch
        o.background = True
        # Force an AUTO_APPROVE decision (no such rule ships) to exercise
        # the wiring; stub _run_action so no real integration is hit.
        from src.core import gate_policy
        monkeypatch.setattr(
            "src.core.staging_mixin.gate_policy.evaluate",
            lambda *a, **k: gate_policy.PolicyDecision(gate_policy.AUTO_APPROVE, "test"),
        )
        ran = {}
        monkeypatch.setattr(
            type(o), "_run_action",
            lambda self, name, args, effect_id=None: (ran.setdefault("call", (name, args)),
                                      ("Lembrete criado.", "AGENT_ACTION_DONE"))[1],
            raising=False,
        )
        reply = o._execute_tool("create_reminder", {"title": "x"})
        assert "auto-aprovada" in reply
        assert ran["call"] == ("create_reminder", {"title": "x"})
        # recorded as executed, not left pending
        assert q.pending() == []
        assert any(r.get("event") == "executed" for r in q._read_all())
