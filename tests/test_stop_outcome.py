"""Stop keeps the real outcome even when the SSE consumer has gone away."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.requests import Request

from src.core import effect_store, planner
from src.integrations import gmail_client
from src.ui import routes
from tests.test_dispatch_parity import _build_real_orch
from tests.test_ui_routes import client, ui_app, _parse_sse


AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def rig(tmp_path, monkeypatch, ui_app, client):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(routes, "_SSE_JOIN_TIMEOUT_S", .01)
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.engine_analytical.max_history = 10
    monkeypatch.setattr(o, "_recipient_known", lambda *_: True)
    ui_app.state.orchestrator = o
    cid = client.post("/ui/conversations", headers=AUTH, json={}).json()["id"]
    append = Mock(wraps=routes.chat_store.append_turn)
    monkeypatch.setattr(routes.chat_store, "append_turn", append)
    saved = threading.Event()
    def persist(**kw):
        turn = append._mock_wraps(**kw)
        saved.set()
        return turn
    append.side_effect = persist

    def post(text, lang="pt"):
        response = client.post("/ui/chat/stream", headers=AUTH, json={
            "text": text, "language": lang, "conversation_id": cid})
        assert response.status_code == 200
        return _parse_sse(response.text)

    def cancel():
        turn_id, = routes._active_turns
        assert client.post("/ui/chat/cancel", headers=AUTH,
                           json={"id": turn_id}).json() == {"cancelled": True}
        return routes._active_turns[turn_id]

    def last():
        assert saved.wait(5), "stopped turn was never persisted"
        return client.get(f"/ui/conversations/{cid}", headers=AUTH).json()["turns"][-1]

    def stream(lang="pt", text="sim"):
        # TestClient buffers SSE. Drive the same route's generator directly
        # for a genuine disconnect, retaining TestClient for cancel/history.
        monkeypatch.setattr(routes, "StreamingResponse", lambda events, **kw: events)
        request = Request({"type": "http", "app": ui_app,
                           "headers": [(b"x-cortex-key", b"test-key-xyz")]})
        events = routes.ui_chat_stream(routes.ChatPayload(
            text=text, language=lang, conversation_id=cid), request)
        assert _parse_sse(next(events))[0]["type"] == "turn"
        next(events)  # baseline status, before run() starts
        return events

    return SimpleNamespace(o=o, post=post, cancel=cancel, last=last,
                           stream=stream, saved=saved, append=append)


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("disconnect", [False, True])
def test_plan_stop_keeps_summary_and_completed_effect(rig, monkeypatch, lang, disconnect):
    plan = planner.Plan("enviar", [planner.Step(f"p{i}", "send_email", {
        "to": "ana@example.com", "subject": "Teste", "body": "texto"})
        for i in range(1, 4)])
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: plan)
    summary = Mock(return_value="all done")
    monkeypatch.setattr(planner, "summarise", summary)
    rig.post("planeja: enviar três emails", lang)
    pid = rig.o._pending_action["id"]
    rig.append.reset_mock()
    rig.saved.clear()
    release = threading.Event()
    cancelled = threading.Event()

    def send(**kw):
        if disconnect:
            rig.o._token_emitter()("Em andamento.")
        rig.cancel()
        cancelled.set()
        if disconnect:
            rig.o._emit_status("thinking")
            assert release.wait(5)
        return {"id": "sent", "to": kw["to"]}
    call = Mock(side_effect=send)
    monkeypatch.setattr(gmail_client, "send_email", call)
    if disconnect:
        events = rig.stream(lang)
        try:
            next(events)  # integration status: effect is still in flight
            assert cancelled.wait(5)
            events.close()
        finally:
            release.set()
    else:
        frames = rig.post("sim", lang)
    turn = rig.last()
    expected = ("Parei: fiz 1 de 3 passos; os outros não rodaram." if lang == "pt"
                else "Stopped: 1 of 3 steps done; the rest did not run.")
    assert turn["reply"] == ("Em andamento.\n\n" if disconnect else "") + expected
    assert turn["status_mode"] == "AGENT_ACTION_CANCELLED"
    assert rig.append.call_count == call.call_count == 1
    assert effect_store.store.get(f"{pid}:p1").state == "feito"
    assert effect_store.store.get(f"{pid}:p2") is None
    assert effect_store.store.get(f"{pid}:p3") is None
    summary.assert_not_called()
    if not disconnect:
        assert frames[-1] == {"type": "final", "turn": turn}


@pytest.mark.parametrize("lang,expected", [("pt", "— parado —"), ("en", "— stopped —")])
@pytest.mark.parametrize("disconnect", [False, True])
def test_stop_before_worker_starts(rig, lang, expected, disconnect):
    events = rig.stream(lang)
    rig.cancel()
    if disconnect:
        events.close()
    else:
        frames = _parse_sse("".join(events))
    assert rig.last()["reply"] == expected
    assert rig.last()["status_mode"] == "AGENT_ACTION_CANCELLED"
    assert rig.append.call_count == 1
    if not disconnect:
        assert frames[-1]["turn"] == rig.last()
    rig.o.engine_analytical.chat_stream.assert_not_called()


def test_stop_in_single_s3_call_keeps_its_reply(rig, monkeypatch):
    rig.post("mandar um email para: ana@example.com - assunto: Teste -- Corpo: texto")
    pid = rig.o._pending_action["id"]
    rig.append.reset_mock()
    rig.saved.clear()

    def send(**kw):
        rig.cancel()
        return {"id": "sent", "to": kw["to"]}
    monkeypatch.setattr(gmail_client, "send_email", send)
    frames = rig.post("sim")
    turn = rig.last()
    assert "ana@example.com" in turn["reply"] and "enviado" in turn["reply"]
    assert turn["status_mode"] == "AGENT_ACTION_DONE"
    assert effect_store.store.get(pid).state == "feito"
    assert rig.append.call_count == 1 and frames[-1]["turn"] == turn


@pytest.mark.parametrize("stop", [False, True])
def test_completion_racing_stop_is_persisted_once(rig, monkeypatch, stop):
    # Cancel inside append_turn: the worker and generator can both observe it.
    persist = rig.append.side_effect

    def append(**kw):
        if stop:
            rig.cancel()
        return persist(**kw)
    rig.append.side_effect = append
    rig.o.engine_analytical.chat_stream.return_value = iter(["um ", "dois"])
    frames = rig.post("me explica RAG em detalhes")
    turn = rig.last()
    assert turn["reply"] == "um dois" and turn["status_mode"] == "OFFLINE"
    assert rig.append.call_count == 1
    assert frames[-1] == {"type": "final", "turn": turn}


def test_plain_chat_stop_does_not_drain_llm(rig):
    produced = []

    def tokens(*a, **kw):
        for token in ["um ", "dois", " três", " quatro"]:
            produced.append(token)
            yield token
            if len(produced) == 2:
                rig.cancel()
    rig.o.engine_analytical.chat_stream.side_effect = tokens
    frames = rig.post("me explica RAG em detalhes")
    assert rig.last()["reply"] == "um dois"
    assert rig.last()["status_mode"] == "AGENT_ACTION_CANCELLED"
    assert produced == ["um ", "dois", " três"]  # next chunk detects Stop
    assert rig.append.call_count == 1 and frames[-1]["turn"] == rig.last()


def test_disconnect_after_worker_completion_keeps_one_turn(rig):
    rig.o.engine_analytical.chat_stream.return_value = iter(["concluído"])
    events = rig.stream(text="me explica RAG em detalhes")
    next(events)  # token, before the sentinel/final is read
    delivery, = routes._active_turns.values()
    assert delivery.finished.wait(5)
    events.close()
    assert rig.last()["reply"] == "concluído"
    assert rig.append.call_count == 1
