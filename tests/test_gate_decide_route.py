"""One explicit UI decision shares the terminal's lock, dispatch and effect id."""
import fcntl
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from starlette.requests import Request

from src import gate_review
from src.core.audit import audit
from src.core.gate_queue import GateQueue
from src.core.staging_mixin import Delivery
from src.integrations import gmail_client
from src.ui import routes
from tests.test_effect_reconcile import no_integrations
from tests.test_effect_runner import context
from tests.test_pending_identity import EMAIL
from tests.test_ui_routes import client, ui_app

AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def gate(ui_app, context, tmp_path, monkeypatch):
    o, store = context
    q = GateQueue(str(tmp_path / "gate.jsonl"))
    monkeypatch.setattr("src.core.gate_queue.gate_queue", q)
    ui_app.state.orchestrator = o
    gid = q.enqueue("send_email", EMAIL["args"])
    def send(**kwargs):
        effect = store.get(gid)
        assert effect.state == "executando" and effect.args == q.get(gid)["args"]
        assert effect.origin is None
        return {"id": "sent", "to": kwargs["to"]}
    call, log = Mock(side_effect=send), Mock()
    monkeypatch.setattr(gmail_client, "send_email", call)
    monkeypatch.setattr(audit, "log", log)
    blocked = Mock(side_effect=AssertionError("must use the live orchestrator"))
    monkeypatch.setattr(gate_review, "_default_runner", blocked)
    monkeypatch.setattr("src.core.orchestrator.CortexOrchestrator.__init__", blocked)
    yield o, store, q, gid, call, log
    blocked.assert_not_called()


def post(client, gid, decision="approve", **extra):
    return client.post("/ui/gate/decide", headers=AUTH,
                       json={"id": gid, "decision": decision, **extra})


def test_auth_and_validation(gate, client):
    _, _, q, gid, call, log = gate
    before = Path(q.path).read_bytes()
    assert client.post("/ui/gate/decide", json={"id": gid, "decision": "approve"}).status_code == 401
    for body in ({"id": gid, "decision": "all"}, {"id": 123, "decision": "approve"},
                 {"id": "x" * 65, "decision": "approve"}, {"id": "", "decision": "approve"},
                 {"id": gid, "decision": "approve", "lang": "fr"}):
        assert client.post("/ui/gate/decide", headers=AUTH, json=body).status_code == 422
    assert Path(q.path).read_bytes() == before
    call.assert_not_called()
    log.assert_not_called()


@pytest.mark.parametrize("outcome", ["executed", "unknown", "failed"])
def test_approve_once_and_view(gate, client, monkeypatch, outcome):
    o, store, q, gid, call, log = gate
    if outcome == "unknown":
        call.side_effect = TimeoutError("lost")
    elif outcome == "failed":
        monkeypatch.setattr(o, "_run_action", lambda kind, args, effect_id:
                            ("token: fictional-secret", "AGENT_ACTION_ERROR"))
    other = q.enqueue("send_email", {"to": "other@example.com"})
    log.reset_mock()
    response = post(client, gid)
    assert response.status_code == 200
    data = response.json()
    assert data["decided"] is True and data["result"] == data["status"] == outcome
    assert q.status(gid) == outcome and q.status(other) == "queued"
    if outcome == "failed":
        assert data["reply"] == "token: [REDACTED]"
        call.assert_not_called()
    else:
        assert store.get(gid).state == ("feito" if outcome == "executed" else "desconhecido")
        assert data["reply"].startswith("Email enviado" if outcome == "executed" else "Não sei")
        call.assert_called_once_with(**q.get(gid)["args"], message_id=gmail_client.message_id_for(gid))
    again = post(client, gid)
    assert again.status_code == 409
    assert again.json() == {"decided": False, "result": "already_decided", "status": outcome}
    assert call.call_count == (0 if outcome == "failed" else 1)
    view = client.get("/ui/gate", headers=AUTH).json()
    assert [e["id"] for e in view["pending"]] == [other]
    assert next(e for e in view["recent"] if e["id"] == gid)["status"] == outcome
    decisions = [c for c in log.call_args_list if c.args[2] == "ui_decision"]
    assert decisions == [
        (("–", "gate", "ui_decision"), {"payload": {"id": gid, "decision": "approve", "result": result}})
        for result in (outcome, "already_decided")]


def test_reject_and_unknown(gate, client):
    _, _, q, gid, call, log = gate
    response = post(client, gid, "reject")
    assert response.status_code == 200
    assert response.json()["result"] == response.json()["status"] == "rejected"
    assert response.json()["decided"] is True
    assert post(client, gid).status_code == 409
    before = Path(q.path).read_bytes()
    response = post(client, "missing")
    assert response.status_code == 404
    assert response.json() == {"decided": False, "result": "not_found", "status": None}
    assert Path(q.path).read_bytes() == before
    log.assert_any_call("–", "gate", "ui_decision", payload={
        "id": "missing", "decision": "approve", "result": "not_found"})
    call.assert_not_called()


@pytest.mark.parametrize("failure", ["lock", "write", "orchestrator"])
def test_unavailable_stays_pending(gate, client, ui_app, monkeypatch, failure):
    _, _, q, gid, call, log = gate
    before = Path(q.path).read_bytes()
    if failure == "lock":
        monkeypatch.setattr(fcntl, "flock", Mock(side_effect=OSError("lock")))
    elif failure == "write":
        monkeypatch.setattr(q, "_append", lambda record: False)
    else:
        del ui_app.state.orchestrator
    response = post(client, gid)
    assert response.status_code == 503
    if failure != "orchestrator":
        assert response.json() == {"decided": False, "result": "not_recorded", "status": "queued"}
        assert gate_review.approve(gid, queue=q) == f"id '{gid}' já foi decidido (status: queued)."
        assert gate_review.reject(gid, queue=q) == f"id '{gid}' já foi decidido (status: queued)."
    assert Path(q.path).read_bytes() == before and q.pending()[0]["id"] == gid
    decisions = [c for c in log.call_args_list if c.args[2] == "ui_decision"]
    assert len(decisions) == 1
    assert decisions[0].kwargs["payload"] == {"id": gid, "decision": "approve", "result": "not_recorded"}
    call.assert_not_called()
    if failure == "orchestrator":
        assert post(client, gid, "reject").json()["result"] == "rejected"


def test_terminal_during_ui_send(gate, client):
    _, _, q, gid, call, _ = gate
    send = call.side_effect
    def race(**kwargs):
        separate_queue = GateQueue(q.path)
        assert gate_review.approve(gid, queue=separate_queue) == (
            f"id '{gid}' já foi decidido (status: approved).")
        return send(**kwargs)
    call.side_effect = race
    assert post(client, gid).json()["result"] == "executed"
    call.assert_called_once()


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_stale_thread_state(gate, ui_app, lang):
    o, store, _, gid, call, _ = gate
    delivery = Delivery()
    delivery.cancelled.set()
    o.set_turn_delivery(delivery)
    o.set_turn_origin("ui", "other")
    o.set_turn_pending_id("other-pending")
    o._reply_lang = "en"
    request = Request({"type": "http", "app": ui_app, "method": "POST", "path": "/ui/gate/decide",
                       "headers": [(b"x-cortex-key", b"test-key-xyz")]})
    payload = routes.GateDecisionPayload(id=gid, decision="approve", **({"lang": lang} if lang == "en" else {}))
    response = routes.ui_gate_decide(payload, request)
    data = json.loads(response.body)
    assert response.status_code == 200 and data["result"] == "executed"
    assert data["reply"].startswith("Email enviado" if lang == "pt" else "Email sent")
    assert store.get(gid).origin is None
    assert o._turn_delivery() is o._turn_origin() is o._turn_pending_id() is None
    assert o._reply_lang_state() == {} and o._reply_lang == o._reply_lang_fallback == "en"
    call.assert_called_once()


def test_terminal_strings(gate, monkeypatch):
    o, _, q, gid, call, _ = gate
    o.set_turn_origin(None)
    runner = lambda kind, args: o._run_action(kind, args, effect_id=gid)
    assert gate_review.approve(gid, queue=q, runner=runner) == (
        "[executado] Email enviado para joao@exemplo.com (id sent).")
    assert gate_review.approve(gid, queue=q, runner=runner) == f"id '{gid}' já foi decidido (status: executed)."
    gid = q.enqueue("send_email", EMAIL["args"])
    call.side_effect = TimeoutError("lost")
    assert gate_review.approve(gid, queue=q, runner=runner) == (
        "[NÃO SEI] Não sei se deu certo (lost). Confere antes de pedir de novo — não vou repetir sozinho.")
    gid = q.enqueue("send_email", {})
    assert gate_review.reject(gid, queue=q) == f"id '{gid}' rejeitado — não vou executar."
    assert gate_review.approve("missing", queue=q) == "id 'missing' não encontrado na fila."
    assert gate_review.reject("missing", queue=q) == "id 'missing' não encontrado na fila."
