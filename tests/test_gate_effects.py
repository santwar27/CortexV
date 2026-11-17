"""Queue approval and effect identity must survive both execution paths."""
import fcntl
from unittest.mock import Mock

import pytest

from src import gate_review
from src.core import gate_policy, staging_mixin
from src.core.audit import audit
from src.core.gate_queue import GateQueue
from src.integrations import gmail_client
from tests.test_effect_runner import context
from tests.test_pending_identity import EMAIL


@pytest.fixture
def gate(context, tmp_path, monkeypatch):
    o, store = context
    q = GateQueue(str(tmp_path / "gate.jsonl"))
    monkeypatch.setattr(staging_mixin, "gate_queue", q)
    monkeypatch.setattr(gate_policy, "evaluate", lambda *a: gate_policy.PolicyDecision(
        gate_policy.AUTO_APPROVE, "test"))
    return o, store, q


def test_enqueue_failure(gate, monkeypatch):
    o, _, q = gate
    monkeypatch.setattr(q, "_append", lambda r: False)
    log, run = Mock(), Mock()
    monkeypatch.setattr(audit, "log", log)
    monkeypatch.setattr(o, "_run_action", run)
    assert q.enqueue("send_email", {}) is None
    assert o._gate_background_action("send_email", {}) == (
        "Não consegui enfileirar a ação 'send_email' para revisão — não fiz nada.")
    run.assert_not_called()
    assert not any(c.args[2] == "queued" for c in log.call_args_list)


@pytest.mark.parametrize("failure", ["open", "flock", "append"])
def test_approval_failure(gate, monkeypatch, failure):
    o, _, q = gate
    gid = q.enqueue("send_email", {})
    before = q._read_all()
    if failure == "open":
        monkeypatch.setattr(staging_mixin.os, "open", Mock(side_effect=OSError("lock")))
    elif failure == "flock":
        monkeypatch.setattr(fcntl, "flock", Mock(side_effect=OSError("lock")))
    else:
        append = q._append
        monkeypatch.setattr(q, "_append", lambda r: False if r["event"] == "approved" else append(r))
    log, run = Mock(), Mock(return_value=("ok", "AGENT_ACTION_DONE"))
    monkeypatch.setattr(audit, "log", log)
    monkeypatch.setattr(o, "_run_action", run)
    assert not q.resolve(gid, "approved")
    assert q._read_all() == before
    reply = o._gate_background_action("send_email", {})
    run.assert_not_called()
    assert "revisão" in reply and q.pending()[-1]["id"] in reply
    if failure != "append":
        log.assert_any_call("–", "gate", "decision_lock_failed")


def test_rejection_failure_stays_pending(gate, monkeypatch):
    o, _, q = gate
    monkeypatch.setattr(gate_policy, "evaluate", lambda *a: gate_policy.PolicyDecision(
        gate_policy.AUTO_REJECT, "test"))
    append = q._append
    monkeypatch.setattr(q, "_append", lambda r: False if r["event"] == "rejected" else append(r))
    run = Mock()
    monkeypatch.setattr(o, "_run_action", run)

    reply = o._gate_background_action("send_email", {})

    pending = q.pending()
    assert len(pending) == 1
    gid = pending[0]["id"]
    assert q.status(gid) == "queued"
    assert reply == f"Ação 'send_email' enfileirada para sua revisão (id {gid})."
    run.assert_not_called()


def test_default_runner_forwards_id(gate, monkeypatch):
    o, _, q = gate
    gid = q.enqueue("send_email", {})
    run = Mock(return_value=("ok", "AGENT_ACTION_DONE"))
    default_runner = gate_review._default_runner
    monkeypatch.setattr(gate_review, "_default_runner", run)
    assert gate_review.approve(gid, queue=q) == "[executado] ok"
    run.assert_called_once_with("send_email", {}, effect_id=gid)
    monkeypatch.setattr("src.core.orchestrator.CortexOrchestrator", lambda: o)
    monkeypatch.setattr(o, "_run_action", run)
    # Exercise the actual default adapter too, not just approve's seam.
    default_runner("send_email", {}, effect_id=gid)
    assert run.call_count == 2
    assert run.call_args.kwargs == {"effect_id": gid}


@pytest.mark.parametrize("path", ["background", "cli"])
@pytest.mark.parametrize("result,state,tag", [
    ("done", "executed", "executado"), ("error", "failed", "FALHOU"),
    ("timeout", "unknown", "NÃO SEI"), ("await", "failed", "FALHOU"),
    ("running", "unknown", "NÃO SEI"), ("raise", "failed", "FALHOU")])
def test_outcomes_and_identity(gate, monkeypatch, path, result, state, tag):
    o, store, q = gate
    call = Mock(side_effect=TimeoutError("lost")) if result == "timeout" else Mock(
        return_value={"id": "sent", "to": EMAIL["args"]["to"]})
    monkeypatch.setattr(gmail_client, "send_email", call)
    if result in {"error", "await", "raise"}:
        run = Mock(side_effect=RuntimeError("broken")) if result == "raise" else Mock(
            return_value=("result", "AGENT_AWAITING_CONFIRMATION" if result == "await"
                          else "AGENT_ACTION_ERROR"))
        monkeypatch.setattr(o, "_run_action", run)
    if result == "running":
        transition = store.transition
        monkeypatch.setattr(store, "transition", lambda eid, de, para, detalhe=None:
                            False if para == "feito" else transition(eid, de, para, detalhe))
    if path == "background":
        reply = o._gate_background_action("send_email", EMAIL["args"])
        gid = q._read_all()[0]["id"]
    else:
        gid = q.enqueue("send_email", EMAIL["args"])
        monkeypatch.setattr(gate_review, "_default_runner", o._run_action)
        reply = gate_review.approve(gid, queue=q)
    assert q.status(gid) == state
    assert f"[{tag}]" in reply
    if result in {"done", "timeout", "running"}:
        assert store.get(gid) is not None
        assert o._run_action("send_email", EMAIL["args"], effect_id=gid)[1] == "AGENT_ACTION_ERROR"
        call.assert_called_once()
    if result == "raise":
        assert "broken" in q._read_all()[-1]["detail"]


def test_injected_runner_keeps_two_arguments(gate):
    _, _, q = gate
    gid = q.enqueue("send_email", {})
    def runner(kind, args):
        raise RuntimeError("injected failure")
    assert gate_review.approve(gid, queue=q, runner=runner) == "[FALHOU] injected failure"
    assert q.status(gid) == "failed"
