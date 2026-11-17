"""A blocked turn must leave the shared server loop free (s89, 1E)."""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.core.staging_mixin import StagingMixin
from src.ui.conversation import buffer


_AUTH = {"X-Cortex-Key": "teste-fake"}
_SECRET = "secret at /private/hidden.env TOKEN=never-publish"


class _Orchestrator:
    set_turn_origin = StagingMixin.set_turn_origin
    _turn_origin = StagingMixin._turn_origin

    def __init__(self):
        self._origin_local = threading.local()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.seen = []
        self.fail = False

    def process(self, text, *args, **kwargs):
        self.seen.append((threading.get_ident(), self._turn_origin()))
        if self.fail:
            raise RuntimeError(_SECRET)
        if text == "block":
            self.entered.set()
            assert self.release.wait(4), "turn release timed out"
        return "ok", "OK"

    def process_stream(self, *args, **kwargs):
        raise RuntimeError(_SECRET)
        yield  # pragma: no cover — generator marker

    def set_status_emitter(self, emitter):
        pass

    def set_token_emitter(self, emitter):
        pass


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CORTEX_API_KEY", "teste-fake")
    monkeypatch.setattr(api, "SECRET_API_KEY", "teste-fake")
    monkeypatch.setattr(api, "_ALLOWLIST", [])
    api._failed_auth.clear()
    api.limiter.reset()
    orch = _Orchestrator()
    monkeypatch.setattr(api, "orchestrator", orch)
    monkeypatch.setattr(api.app.state, "orchestrator", orch)
    buffer.clear()
    # The context keeps both requests on the same server event loop.
    with TestClient(api.app, client=("127.0.0.1", 50000)) as client:
        yield client, orch
    buffer.clear()
    api.limiter.reset()
    api._failed_auth.clear()


@pytest.mark.parametrize("endpoint", ["/ask", "/ask_voice"])
def test_blocked_turn_does_not_block_other_request(server, endpoint):
    client, orch = server
    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(client.post, endpoint, headers=_AUTH,
                               json={"user_input": "block"})
        try:
            assert orch.entered.wait(1)
            second = callers.submit(client.post, "/ask", headers=_AUTH,
                                    json={"user_input": "quick"})
            assert second.result(timeout=1).status_code == 200
            assert not first.done()
        finally:
            orch.release.set()
        assert first.result(timeout=2).status_code == 200


@pytest.mark.parametrize("endpoint,origin", [("/ask", "terminal"),
                                            ("/ask_voice", "voice")])
@pytest.mark.parametrize("fail", [False, True])
def test_origin_is_set_and_cleared_on_turn_worker(server, endpoint, origin, fail):
    client, orch = server
    orch.fail = fail

    async def single_worker():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1

    async def probe():
        return await anyio.to_thread.run_sync(
            lambda: (threading.get_ident(), orch._turn_origin()))

    client.portal.call(single_worker)
    loop_id = client.portal.call(threading.get_ident)
    response = client.post(endpoint, headers=_AUTH, json={"user_input": "oi"})
    assert response.status_code == (500 if fail else 200)
    worker_id, seen_origin = orch.seen[0]
    assert worker_id != loop_id
    assert seen_origin == {"channel": origin, "conversation": None}
    # Probe the very same pool thread after the handler's finally ran.
    assert client.portal.call(probe) == (worker_id, None)


def _assert_ref_logged(ref, caplog, logger):
    assert re.fullmatch(r"[0-9a-f]{8}", ref)
    records = [r for r in caplog.records if r.name == logger and ref in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert _SECRET in caplog.text


@pytest.mark.parametrize("endpoint", ["/ask", "/ask_voice"])
def test_http_error_ref_matches_traceback(server, endpoint, caplog):
    client, orch = server
    orch.fail = True
    with caplog.at_level(logging.ERROR, logger="cortex.api"):
        response = client.post(endpoint, headers=_AUTH, json={"user_input": "oi"})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert _SECRET not in response.text + str(response.headers)
    _assert_ref_logged(response.headers.get("X-Cortex-Ref", ""), caplog, "cortex.api")


@pytest.mark.parametrize("endpoint,logger", [("/ask_stream", "cortex.api"),
                                            ("/ui/chat/stream", "cortex.ui")])
def test_stream_error_ref_matches_traceback(server, endpoint, logger, caplog):
    client, _ = server
    payload = {"text" if endpoint.startswith("/ui/") else "user_input": "oi",
               "language": "pt"}
    with caplog.at_level(logging.ERROR, logger=logger):
        response = client.post(endpoint, headers=_AUTH, json=payload)
    assert response.status_code == 200
    assert _SECRET not in response.text
    frames = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    error = next(f for f in frames if "error" in f or f.get("type") == "error")
    _assert_ref_logged(error.get("ref", ""), caplog, logger)
    if endpoint == "/ask_stream":
        assert error == {"error": "internal error", "reply": api._STREAM_FAILURE_REPLY["pt"],
                         "language": "pt", "ref": error["ref"]}
        assert frames[-1] == {"done": True, "status_mode": "INTERNAL_ERROR", "language": "pt"}
    else:
        assert error == {"type": "error", "detail": "internal error", "ref": error["ref"]}
        assert frames[-1] == error


@pytest.mark.parametrize("endpoint", ["/ask", "/ask_voice"])
def test_sync_handlers_keep_auth_and_rate_limit(server, endpoint, monkeypatch):
    client, orch = server
    response = client.post(endpoint, headers={"X-Cortex-Key": "wrong"},
                           json={"user_input": "oi"})
    assert response.status_code == 403
    assert orch.seen == []
    # Exhausted storage must still be enforced by the real slowapi wrapper.
    monkeypatch.setattr(api.limiter._limiter, "hit", lambda *a, **k: False)
    response = client.post(endpoint, headers=_AUTH, json={"user_input": "oi"})
    assert response.status_code == 429
    assert orch.seen == []
