"""
The inline gate must record every "sim"/"não" decision to the audit log so
the policy detector has evidence to work from. These tests
assert the `gate`/`approved` and `gate`/`rejected` events fire with the
action type — without triggering the real integration side effects.
"""
import pytest

import src.core.audit as audit_mod
from src.core.orchestrator import CortexOrchestrator


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CortexOrchestrator, "_PENDING_ACTION_FILE",
        str(tmp_path / "pending_action.json"),
    )
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    # Don't actually send emails / create events — we only test the audit
    # emission around the decision, not the execution.
    o._run_action = lambda name, args, effect_id=None: ("ok", "AGENT_ACTION_DONE")
    return o


@pytest.fixture
def spy(monkeypatch):
    events = []

    def _log(trace_id, stage, event, payload=None, latency_ms=None):
        events.append((stage, event, payload or {}))

    monkeypatch.setattr(audit_mod.audit, "log", _log)
    return events


def test_approval_logs_gate_approved(orch, spy):
    orch._pending_action = {"type": "create_reminder", "args": {"title": "x"}}
    orch._execute_pending_action()
    gate = [(e, p) for (s, e, p) in spy if s == "gate"]
    assert ("approved", {"type": "create_reminder", "channel": "inline"}) in gate


def test_denial_logs_gate_rejected(orch, spy):
    orch._pending_action = {"type": "send_email", "args": {"to": "a@b.c"}}
    orch._cancel_pending_action()
    gate = [(e, p) for (s, e, p) in spy if s == "gate"]
    assert ("rejected", {"type": "send_email", "channel": "inline"}) in gate


def test_approval_does_not_log_rejected(orch, spy):
    orch._pending_action = {"type": "create_event", "args": {}}
    orch._execute_pending_action()
    events = [e for (s, e, p) in spy if s == "gate"]
    assert "approved" in events
    assert "rejected" not in events


def test_plan_approval_does_not_emit_single_action_event(orch, spy, monkeypatch):
    # execute_plan is a multi-step beast, not a single S3 tool — it must not
    # pollute the per-type approval stats the detector reads.
    monkeypatch.setattr(orch, "_execute_staged_plan",
                        lambda action=None: ("done", "AGENT_ACTION_DONE"))
    orch._pending_action = {"type": "execute_plan", "args": {"passos": []}}
    orch._execute_pending_action()
    gate = [e for (s, e, p) in spy if s == "gate"]
    assert gate == []
