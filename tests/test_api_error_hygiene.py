"""
M10-A8 — a 500 from /ask or /ask_voice must not leak str(e) to the client.

Before s69, both handlers did `except Exception as e: raise
HTTPException(500, detail=str(e))`: the exception text (which can carry
internal paths or secrets) went out on the wire, and the traceback was
lost (no log call at all — unlike the old /ui/chat, which always logged). Now the
client gets a generic detail and the traceback lands in the `cortex.api`
log.

These tests go through TestClient over the real app (auth included) with
the module-global orchestrator swapped for one that raises with a
deliberately sensitive message.
"""
import logging

import pytest
from fastapi.testclient import TestClient

import src.api as api_mod


_SENSITIVE = "boom at /Users/pedro/.env SECRET=abc123"


class _BoomOrchestrator:
    def process(self, *args, **kwargs):
        raise RuntimeError(_SENSITIVE)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "orchestrator", _BoomOrchestrator())
    # The IP allowlist is parsed from the real .env at import; TestClient's
    # pseudo-host ("testclient") is not an IP, so it would be denied before
    # the route runs. Empty list = allow all (the back-compat default).
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    return TestClient(api_mod.app, raise_server_exceptions=False)


_AUTH = {"X-Cortex-Key": "test-key-xyz"}


class TestAskErrorHygiene:

    def test_ask_500_detail_is_generic(self, client, caplog):
        with caplog.at_level(logging.ERROR, logger="cortex.api"):
            r = client.post("/ask", headers=_AUTH,
                            json={"user_input": "oi"})
        assert r.status_code == 500
        assert r.json()["detail"] == "internal error"
        assert _SENSITIVE not in r.text
        # The traceback is preserved server-side instead of being lost.
        assert _SENSITIVE in caplog.text

    def test_ask_voice_500_detail_is_generic(self, client, caplog):
        with caplog.at_level(logging.ERROR, logger="cortex.api"):
            r = client.post("/ask_voice", headers=_AUTH,
                            json={"user_input": "oi"})
        assert r.status_code == 500
        assert _SENSITIVE not in r.text
        assert _SENSITIVE in caplog.text
