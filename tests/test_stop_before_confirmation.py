"""A turn whose stream was abandoned leaves no action anyone can approve.

Pedro asks for an email and presses Stop before the confirmation shows (the
UI), or the voice listener drops the stream: the server's turn goes on and
stages the email. With option A (64d6d2e) a "sim" still confirms the action
that is pending — so a "sim" in another conversation sent an email whose
preview never reached any screen (s88, Codex lote 7, reproduced).

The client going away is reproduced the way Starlette sees it: the route's
generator stops being pulled. Both routes failed before the fix (423f751,
xfail); the action now carries the turn's Delivery and is withdrawn when the
client never takes the final frame.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

import src.api as api_mod
from src.ui import routes
from tests.test_dispatch_parity import _build_real_orch
from tests.test_ui_routes import client, ui_app, _reset_global_buffer  # noqa: F401

H = {"X-Cortex-Key": "test-key-xyz"}
EMAIL_ASK = "mandar um email para: joao@exemplo.com - assunto: Atraso -- Corpo: Vou atrasar."


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = _build_real_orch(tmp_path)
    o.background = False
    o._offer = None
    o.ran = []
    monkeypatch.setattr(o, "_run_action", lambda name, args, effect_id=None: (
        o.ran.append(name), ("ok", "AGENT_ACTION_DONE"))[1])
    return o


@pytest.fixture(autouse=True)
def short_grace(monkeypatch):
    """How long a staged action waits for its preview to be taken."""
    from src.core import staging_mixin
    monkeypatch.setattr(staging_mixin, "UNDELIVERED_GRACE_S", 0.3, raising=False)


def _capture_stream(monkeypatch, module):
    """The route's generator, as Starlette would pull it."""
    captured = {}

    def fake_streaming_response(gen, **_kw):
        from starlette.responses import Response
        captured["gen"] = gen
        return Response("")
    monkeypatch.setattr(module, "StreamingResponse", fake_streaming_response)
    return captured


def _wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)


def _yes_elsewhere(o):
    *_, (reply, _f, mode, _l) = o.process_stream("sim", None, conversation_history=[
        {"role": "assistant", "content": "Quer que eu escreva as respostas no PDF?"}])
    return mode


class TestTheUi:

    def test_stop_before_the_confirmation(self, client, ui_app, orch, monkeypatch):
        ui_app.state.orchestrator = orch
        captured = _capture_stream(monkeypatch, routes)
        client.post("/ui/chat/stream", headers=H, json={"text": EMAIL_ASK})
        gen = captured["gen"]
        # "thinking" goes out before the turn starts; the next pull starts it,
        # and Stop lands while it runs: no further pull — whatever that pull
        # produced never reaches the browser.
        frames = threading.Thread(target=lambda: (next(gen), next(gen)))
        frames.start()
        _wait_until(lambda: orch._pending_action is not None)
        time.sleep(0.6)                            # past the grace
        _yes_elsewhere(orch)
        assert orch.ran == []


class TestTheVoiceStream:

    def test_dropped_before_the_confirmation(self, orch, monkeypatch):
        monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
        monkeypatch.setattr(api_mod, "orchestrator", orch)
        monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
        captured = _capture_stream(monkeypatch, api_mod)
        TestClient(api_mod.app).post("/ask_stream", headers=H,
                                     json={"user_input": EMAIL_ASK})
        frames = threading.Thread(target=lambda: next(captured["gen"]))
        frames.start()                             # the first frame — then gone
        _wait_until(lambda: orch._pending_action is not None)
        time.sleep(0.6)
        _yes_elsewhere(orch)
        assert orch.ran == []


class TestNothingChangesWhenThePreviewIsTaken:

    def test_the_ui_preview_read_to_the_end_stays_approvable(self, client, ui_app,
                                                              orch, monkeypatch):
        ui_app.state.orchestrator = orch
        client.post("/ui/chat/stream", headers=H, json={"text": EMAIL_ASK})
        time.sleep(0.6)
        _yes_elsewhere(orch)
        assert orch.ran == ["send_email"]

    def test_the_voice_preview_read_to_the_end_stays_approvable(self, orch, monkeypatch):
        monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
        monkeypatch.setattr(api_mod, "orchestrator", orch)
        monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
        TestClient(api_mod.app).post("/ask_stream", headers=H,
                                     json={"user_input": EMAIL_ASK})
        time.sleep(0.6)
        _yes_elsewhere(orch)
        assert orch.ran == ["send_email"]
