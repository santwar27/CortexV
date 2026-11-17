"""Voice echo — /ask_stream and /ask_voice mirror the spoken turn into the
/ui conversation buffer so the dashboard drawer shows, in writing, what
Cortex answered out loud (Pedro's ask, s76).

The typed path (/ui/chat/stream) already buffers its own turns; these two
voice entrypoints didn't, so a spoken reply left no readable trace. The
mirror is tagged source="voice" (the drawer renders it with a 🎙️ marker)
and is best-effort — a buffer failure must never break the spoken turn.

Goes through TestClient over the real app (auth + the module-global
orchestrator swapped for a stub), same harness as test_api_error_hygiene.
"""
import json

import pytest
from fastapi.testclient import TestClient

import src.api as api_mod
from src.ui.conversation import buffer as conversation_buffer


class _StubOrchestrator:
    """Minimal orchestrator: a fixed reply for /ask_voice (process) and a
    two-frame stream for /ask_stream (process_stream)."""

    def process(self, user_input, image_b64=None, language=None, conversation_history=None):
        return "Resposta falada.", "OK"

    def process_stream(self, user_input, image_b64=None, language=None, conversation_history=None):
        # Non-streamable shape: a single final chunk carrying the whole reply.
        yield ("Tudo certo por aqui.", True, "OFFLINE", "pt")


class _BoomStream:
    def process(self, *a, **k):
        return "x", "OK"

    def process_stream(self, *a, **k):
        raise RuntimeError("engine died")
        yield  # pragma: no cover - generator marker


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "orchestrator", _StubOrchestrator())
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    return TestClient(api_mod.app, raise_server_exceptions=False)


_AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture(autouse=True)
def clean_buffer():
    conversation_buffer.clear()
    yield
    conversation_buffer.clear()


def _last():
    turns = conversation_buffer.snapshot()
    return turns[-1] if turns else None


class TestAskVoiceEcho:
    def test_ask_voice_records_voice_turn(self, client):
        r = client.post("/ask_voice", headers=_AUTH,
                        json={"user_input": "que horas são"})
        assert r.status_code == 200
        turn = _last()
        assert turn is not None
        assert turn["user"] == "que horas são"
        assert turn["reply"] == "Resposta falada."
        assert turn["source"] == "voice"
        assert turn["status_mode"] == "OK"

    def test_ask_voice_stores_raw_reply_not_speech_stripped(self, client):
        # The buffer keeps the full reply for reading; the BODY is the
        # speech-stripped version, but the stored turn is the raw reply.
        client.post("/ask_voice", headers=_AUTH, json={"user_input": "oi"})
        assert _last()["reply"] == "Resposta falada."


class TestAskStreamEcho:
    def test_ask_stream_records_voice_turn(self, client):
        r = client.post("/ask_stream", headers=_AUTH,
                        json={"user_input": "resume isso"})
        assert r.status_code == 200
        # Drain the stream so the generator's finally-block runs.
        _ = r.text
        turn = _last()
        assert turn is not None
        assert turn["user"] == "resume isso"
        assert turn["reply"] == "Tudo certo por aqui."
        assert turn["source"] == "voice"

    def test_ask_stream_accumulates_token_chunks(self, client, monkeypatch):
        def streamed(user_input, image_b64=None, language=None, conversation_history=None):
            yield ("Oi ", False, None, "pt")
            yield ("Pedro", False, None, "pt")
            yield ("", True, "OFFLINE", "pt")
        monkeypatch.setattr(api_mod.orchestrator, "process_stream", streamed)
        r = client.post("/ask_stream", headers=_AUTH,
                        json={"user_input": "fala"})
        _ = r.text
        assert _last()["reply"] == "Oi Pedro"

    def test_ask_stream_error_records_clean_sentence(self, monkeypatch):
        monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
        monkeypatch.setattr(api_mod, "orchestrator", _BoomStream())
        monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
        c = TestClient(api_mod.app, raise_server_exceptions=False)
        r = c.post("/ask_stream", headers=_AUTH, json={"user_input": "oi"})
        _ = r.text
        turn = _last()
        # The recorded reply is the clean localised failure sentence, with
        # the failed turn marked INTERNAL_ERROR — not a blank or partial.
        assert turn is not None
        assert turn["status_mode"] == "INTERNAL_ERROR"
        # The clean localised failure sentence (lang resolved server-side),
        # not a blank or a partial token stream.
        assert turn["reply"] in api_mod._STREAM_FAILURE_REPLY.values()


class TestEchoIsBestEffort:
    def test_buffer_failure_does_not_break_ask_voice(self, client, monkeypatch):
        # A buffer hiccup must never surface as a 500 on the spoken turn.
        monkeypatch.setattr(conversation_buffer, "append",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("full")))
        r = client.post("/ask_voice", headers=_AUTH, json={"user_input": "oi"})
        assert r.status_code == 200

    def test_empty_user_is_not_recorded(self, client, monkeypatch):
        # Image-only voice turn with no transcript → nothing to show.
        monkeypatch.setattr(api_mod.orchestrator, "process",
                            lambda *a, **k: ("alguma resposta", "OK"))
        client.post("/ask_voice", headers=_AUTH,
                    json={"user_input": "", "image_b64": "QUJD"})
        assert _last() is None


# --- spoken turns reach the DURABLE chat too (s82) --------------------------
# The buffer above is an in-memory deque that dies with the process. When the
# chat page moved to the durable store (fatia 2), the voice mirror was left
# pointing at the old drawer, so spoken replies stopped showing up in the chat.
# The user noticed and couldn't tell whether he'd asked for that — he hadn't: it
# was drift. chat.db held daily=3 and ui=11 turns, and voice=0.

class TestVoiceTurnsPersist:
    def test_spoken_turn_lands_in_the_chat_store(self, client, monkeypatch):
        saved = {}

        class FakeStore:
            def conversation_titled(self, title):     # get-or-create (s88)
                saved["title"] = title
                return "conv-1"

            def append_turn(self, **kw):
                saved["turn"] = kw
                return {"id": "t1"}

        import src.ui.chat_store as cs
        monkeypatch.setattr(cs, "store", FakeStore())
        client.post("/ask_voice", headers=_AUTH, json={"user_input": "quem foi einstein"})
        assert saved["turn"]["user"] == "quem foi einstein"
        assert saved["turn"]["reply"] == "Resposta falada."
        assert saved["turn"]["source"] == "voice"
        assert saved["title"].startswith("🎙️ Voz")

    def test_the_days_turns_share_one_conversation(self, client, monkeypatch):
        """One conversation per DAY. A sidebar row per "toca uma música"
        would be unusable."""
        created = []
        appended = []

        class FakeStore:
            def conversation_titled(self, title):     # get-or-create (s88)
                if title not in created:
                    created.append(title)
                return "conv-1"

            def append_turn(self, **kw):
                appended.append(kw["conversation_id"])
                return {"id": "t"}

        from src.api import _voice_conversation_title as _title
        import src.ui.chat_store as cs
        monkeypatch.setattr(cs, "store", FakeStore())
        client.post("/ask_voice", headers=_AUTH, json={"user_input": "um"})
        client.post("/ask_voice", headers=_AUTH, json={"user_input": "dois"})
        assert len(created) == 1                 # created once
        assert appended == ["conv-1", "conv-1"]  # both turns in it

    def test_a_store_failure_never_breaks_the_spoken_reply(self, client, monkeypatch):
        """The words were already spoken; storage is the bonus."""
        class Boom:
            def conversation_titled(self, title):
                raise RuntimeError("db locked")

        import src.ui.chat_store as cs
        monkeypatch.setattr(cs, "store", Boom())
        r = client.post("/ask_voice", headers=_AUTH, json={"user_input": "oi"})
        assert r.status_code == 200              # reply still served
        assert _last()["source"] == "voice"      # drawer still updated

    def test_empty_turns_are_not_persisted(self, client, monkeypatch):
        calls = []

        class FakeStore:
            def list_conversations(self, limit=50):
                calls.append("list")
                return []

            def create_conversation(self, title=None):
                calls.append("create")
                return {"id": "c"}

            def append_turn(self, **kw):
                calls.append("append")

        import src.ui.chat_store as cs
        monkeypatch.setattr(cs, "store", FakeStore())
        from src.api import _record_voice_turn
        _record_voice_turn("", "reply", "OK")
        _record_voice_turn("user", "", "OK")
        assert calls == []


# --- and the suite must never write into the user's REAL chat ------------------
# s82 cont.: the tests above drive /ask_voice and /ask_stream through the real
# app, and _record_voice_turn now persists to the durable store. Only the
# TestVoiceTurnsPersist cases below bring a FakeStore; the rest hit the
# singleton. Without the conftest isolation that meant six synthetic turns
# ("que horas são" -> "Resposta falada.") landing in ~/.cortex/chat.db under a
# "🎙️ Voz · <day>" conversation the user never spoke. These pin the guard.

class TestTheRealChatIsNeverTouched:
    def test_store_is_redirected_away_from_the_production_db(self):
        from src.ui.chat_store import ChatStore, store as chat_store
        assert chat_store.db_path != ChatStore.DEFAULT_PATH

    def test_an_unpatched_voice_turn_lands_in_the_isolated_store(self, client):
        # No FakeStore here on purpose: this is the shape that polluted the
        # real DB. It must still persist — isolation, not silent disabling.
        from src.ui.chat_store import store as chat_store
        client.post("/ask_voice", headers=_AUTH,
                    json={"user_input": "isto e um teste"})
        convos = chat_store.list_conversations(limit=50)
        assert any(c["title"].startswith("\U0001f399") for c in convos)


def test_ask_stream_error_frame_carries_no_exception_text(monkeypatch):
    """str(e) rode in the SSE "error" field — a path, an internal detail
    (s88, Codex lote 7). A fixed code now; the spoken reply is unchanged."""
    class _Leaky:
        def process_stream(self, *a, **k):
            raise RuntimeError("falhou lendo /Users/pedro/Documents/segredo.pdf")
            yield  # pragma: no cover
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "orchestrator", _Leaky())
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    r = TestClient(api_mod.app, raise_server_exceptions=False).post(
        "/ask_stream", headers=_AUTH, json={"user_input": "oi"})
    assert "segredo.pdf" not in r.text
    frame = json.loads([l for l in r.text.splitlines() if '"error"' in l][0][5:])
    assert frame["error"] == "internal error"
    assert frame["reply"] in api_mod._STREAM_FAILURE_REPLY.values()
