"""Stop prevents the next effect; an in-flight call may finish."""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from starlette.requests import Request

from src.core import planner
from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.staging_mixin import Delivery, StagingMixin
from tests.test_dispatch_parity import _build_real_orch
from tests.test_ui_routes import client, ui_app, _parse_sse


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o._reply_lang = "pt"
    o.background = False
    return o


def _plan():
    return planner.Plan("test", [planner.Step(f"p{i}", "read_document",
                        {"filename": f"{i}.txt"}) for i in range(1, 4)])


def test_planner_stops_between_steps():
    ran = Mock(return_value="ok")
    result = planner.execute(_plan(), ran, should_stop=lambda: ran.call_count > 0)
    assert ran.call_count == 1 and not result.success and result.cancelled
    assert result.results[0].success
    assert all(r.skipped and r.error == "cancelled" for r in result.results[1:])


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("state", ["absent", "live", "cancelled"])
def test_staged_plan(orch, monkeypatch, lang, state):
    delivery = Delivery()
    orch.set_turn_delivery(delivery if state != "absent" else None)
    orch._pending_action = {"args": {"lang": lang, "passos": [
        {"id": s.id, "tool": s.tool, "args": s.args} for s in _plan().passos]}}
    def read(*args):
        if state == "cancelled":
            delivery.cancelled.set()
        return "ok"
    orch.reader.read_document.side_effect = read
    summary = Mock(return_value="summary")
    monkeypatch.setattr(planner, "summarise", summary)
    reply, mode = orch._execute_staged_plan()
    if state == "cancelled":
        assert reply == ("Parei: fiz 1 de 3 passos; os outros não rodaram." if lang == "pt"
                         else "Stopped: 1 of 3 steps done; the rest did not run.")
        assert mode == "AGENT_ACTION_CANCELLED"
        summary.assert_not_called()
        assert orch.reader.read_document.call_count == 1
    else:
        assert (reply, mode) == ("summary", "AGENT_ACTION_DONE")
        assert orch.reader.read_document.call_count == 3
        summary.assert_called_once()


@pytest.mark.parametrize("state", ["absent", "live", "cancelled"])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_tool_and_persistence_guards(orch, monkeypatch, state, lang):
    orch._reply_lang = lang
    if state != "absent":
        delivery = Delivery()
        orch.set_turn_delivery(delivery)
        if state == "cancelled":
            delivery.cancelled.set()
    monkeypatch.setattr(orch, "_recipient_known", lambda *_: True)
    out = orch._execute_tool("send_email", {"to": "fake@example.com"})
    assert out.startswith(SKIP_FOLLOWUP_MARKER)
    tool = orch._execute_tool_unstaged("read_document", {"filename": "fake.txt"})
    disk = Path(orch._PENDING_ACTION_FILE)
    if state == "cancelled":
        assert out == SKIP_FOLLOWUP_MARKER + ("Parei — não fiz mais nada." if lang == "pt"
                                             else "Stopped — nothing else was done.")
        assert planner._output_looks_failed(tool) and not disk.exists()
        orch.reader.read_document.assert_not_called()
    else:
        assert disk.exists() and tool == "conteúdo"
        orch.reader.read_document.assert_called_once()
    before = disk.read_text() if disk.exists() else None
    orch._pending_action = {"type": "send_email", "args": {"subject": "new"}}
    orch._persist_pending_action()
    if state == "cancelled":
        assert not disk.exists() and orch._pending_action is None
        disk.write_text(json.dumps({"type": "old", "args": {}, "id": "old"}))
        orch._pending_action = {"type": "new", "args": {}}
        orch._persist_pending_action()
        assert orch._pending_action == json.loads(disk.read_text())
        assert orch._pending_action["type"] == "old"
    else:
        assert disk.read_text() != before


def test_cancel_helper_fail_soft():
    class Broken(StagingMixin):
        def _turn_delivery(self):
            raise RuntimeError("broken context")
    assert not Broken()._turn_cancelled()
    assert not StagingMixin()._turn_cancelled()


def test_deferred_worker_sees_external_cancellation(orch, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    delivery = Delivery()
    observed = []
    monkeypatch.setattr(orch, "_is_pure_chat", lambda *_: False)
    def process(*args, **kwargs):
        observed.append((threading.get_ident(), orch._turn_delivery()))
        entered.set()
        assert release.wait(5)
        return orch._execute_tool_unstaged("read_document", {"filename": "x"}), "TEST"
    monkeypatch.setattr(orch, "process", process)
    results = []
    def consume():
        orch.set_turn_delivery(delivery)
        results.extend(orch.process_stream("pesquisa coisas", language="pt"))
    consumer = threading.Thread(target=consume)
    consumer.start()
    try:
        assert entered.wait(5)
        delivery.cancelled.set()
    finally:
        release.set()
        consumer.join(5)
    assert not consumer.is_alive()
    assert observed[0][0] != consumer.ident and observed[0][1] is delivery
    orch.reader.read_document.assert_not_called()
    assert planner._output_looks_failed(results[-1][0])


@pytest.mark.parametrize("finish", [False, True])
def test_ui_lifecycle(client, ui_app, monkeypatch, finish):
    from src.ui import routes
    deliveries = []
    monkeypatch.setattr(ui_app.state.orchestrator, "set_turn_delivery",
                        lambda d: deliveries.append(d) if d else None, raising=False)
    monkeypatch.setattr(routes, "StreamingResponse", lambda events, **kw: events)
    request = Request({"type": "http", "app": ui_app,
                       "headers": [(b"x-cortex-key", b"test-key-xyz")]})
    events = routes.ui_chat_stream(routes.ChatPayload(text="oi"), request)
    first = _parse_sse(next(events))[0]
    assert first["type"] == "turn" and first["id"]
    headers = {"X-Cortex-Key": "test-key-xyz"}
    assert client.post("/ui/chat/cancel", json={"id": first["id"]}).status_code == 401
    assert client.post("/ui/chat/cancel", headers=headers,
                       json={"id": "unknown"}).json() == {"cancelled": False}
    if finish:
        assert _parse_sse("".join(events))[-1]["type"] == "final"
        assert deliveries[0].taken.is_set() and not deliveries[0].cancelled.is_set()
    else:
        delivery = routes._active_turns[first["id"]]
        assert client.post("/ui/chat/cancel", headers=headers,
                           json={"id": first["id"]}).json() == {"cancelled": True}
        assert delivery.cancelled.is_set()
        # Disconnect even before the worker starts must remove the registration.
        events.close()
    assert client.post("/ui/chat/cancel", headers=headers,
                       json={"id": first["id"]}).json() == {"cancelled": False}


@pytest.mark.parametrize("voice", [False, True])
@pytest.mark.parametrize("finish", [False, True])
def test_stream_disconnect(orch, ui_app, monkeypatch, voice, finish):
    from src import api
    from src.ui import routes
    deliveries = []
    install = orch.set_turn_delivery
    def set_delivery(d):
        if d:
            deliveries.append(d)
        install(d)
    monkeypatch.setattr(orch, "set_turn_delivery", set_delivery)
    monkeypatch.setattr(orch, "process_stream", lambda *a, **kw: iter([
        ("first", False, None, "pt"), ("last", True, "OFFLINE", "pt")]))
    request = Request({"type": "http", "app": ui_app,
                       "headers": [(b"x-cortex-key", b"test-key-xyz")]})
    module = api if voice else routes
    monkeypatch.setattr(module, "StreamingResponse", lambda events, **kw: events)
    if voice:
        monkeypatch.setattr(api, "orchestrator", orch)
        events = asyncio.run(api.process_query_stream.__wrapped__(
            request, api.CortexRequest(user_input="oi"), _api_key="test-key-xyz"))
    else:
        ui_app.state.orchestrator = orch
        events = routes.ui_chat_stream(routes.ChatPayload(text="oi"), request)
    if finish:
        list(events)
    else:
        while not deliveries:
            next(events)
        events.close()
    assert deliveries[0].taken.is_set() == finish
    assert deliveries[0].cancelled.is_set() == (not finish)


def test_ui_worker_stops_consuming_after_cancel(orch, ui_app, client, monkeypatch):
    from src.ui import routes
    entered, release, finished = (threading.Event() for _ in range(3))
    after = Mock()
    def stream(*args, **kwargs):
        try:
            entered.set()
            assert release.wait(5)
            yield "in-flight result", False, None, "pt"
            after()
            yield "extra", True, "OFFLINE", "pt"
        finally:
            finished.set()
    ui_app.state.orchestrator = orch
    monkeypatch.setattr(orch, "process_stream", stream)
    result = []
    consumer = threading.Thread(target=lambda: result.append(client.post(
        "/ui/chat/stream", headers={"X-Cortex-Key": "test-key-xyz"}, json={"text": "oi"})))
    consumer.start()
    try:
        assert entered.wait(5)
        turn_id, = routes._active_turns
        assert client.post("/ui/chat/cancel", headers={"X-Cortex-Key": "test-key-xyz"},
                           json={"id": turn_id}).json() == {"cancelled": True}
    finally:
        release.set()
        consumer.join(5)
    assert finished.is_set() and not consumer.is_alive()
    after.assert_not_called()
