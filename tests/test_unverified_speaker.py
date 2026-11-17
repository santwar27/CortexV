"""A room-microphone turn never becomes a fact about Pedro (s88, Codex lote 12
— invariant 3).

The listener answers anyone who says the wake word and speaker-ID is still
shadow only, so "meu hobby é escalada" said by a visitor reached the Palace
as user/hobby=escalada (reproduced through /ask_stream with an isolated
Palace). The turn is still remembered as conversation, marked
speaker=unverified; neither the passive regex nor the nightly "said by the
user" check takes it as Pedro's words. Typed turns are unchanged.
"""
import threading

from fastapi.testclient import TestClient

import src.api as api_mod
from src.core.memory import (
    CortexMemory, speaker_unverified, unverified_speaker,
    unverified_speaker_stream,
)
from src.memory.palace import Palace
from src.memory.preferences import recent_user_words
from tests.test_dispatch_parity import _build_real_orch


def _memory():
    palace = Palace(db_path=":memory:")
    return CortexMemory(palace=palace), palace


def _speakers(palace):
    return [r[0] for r in palace._conn.execute(
        "SELECT json_extract(metadata, '$.speaker') FROM documentos "
        "WHERE tipo='conversation' ORDER BY id")]


class TestTheMemory:

    def test_a_room_turn_is_remembered_but_not_a_fact(self):
        mem, palace = _memory()
        with unverified_speaker():
            mem.store_interaction("meu hobby é escalada", "Que legal!")
        assert palace.get_fact("user", "hobby") is None
        assert _speakers(palace) == ["unverified"]

    def test_a_typed_turn_still_becomes_a_fact(self):
        mem, palace = _memory()
        mem.store_interaction("meu hobby é escalada", "Que legal!")
        assert palace.get_fact("user", "hobby")["objeto"] == "escalada"
        assert _speakers(palace) == [None]

    def test_the_nightly_check_does_not_take_room_words_as_pedros(self):
        mem, palace = _memory()
        with unverified_speaker():
            mem.store_interaction("meu hobby é escalada", "Que legal!")
        mem.store_interaction("eu moro em Lisboa", "Anotado.")
        own = recent_user_words(palace, hours=24)
        assert "lisboa" in own
        assert "escalada" not in own


class TestTheStreamWrapper:

    def test_every_pull_is_marked_on_the_thread_that_pulls_it(self):
        def events():
            for _ in range(3):
                yield speaker_unverified()

        gen = unverified_speaker_stream(events())
        seen = [next(gen)]
        # Starlette pulls a sync generator from pool threads, one per pull.
        t = threading.Thread(target=lambda: seen.append(next(gen)))
        t.start()
        t.join()
        seen.append(next(gen))
        assert seen == [True, True, True]
        assert speaker_unverified() is False      # restored between pulls

    def test_closing_the_wrapper_closes_the_stream_inside_the_mark(self):
        closed = []

        def events():
            try:
                yield 1
                yield 2
            finally:
                closed.append(speaker_unverified())

        gen = unverified_speaker_stream(events())
        next(gen)
        gen.close()
        assert closed == [True]


class TestTheOrchestrator:

    def test_the_pure_chat_stream_is_marked(self, tmp_path):
        o = _build_real_orch(tmp_path)
        seen = []

        def chat_stream(*_a, **_k):
            seen.append(speaker_unverified())
            yield "Que legal!"
        o.engine_analytical.chat_stream.side_effect = chat_stream
        list(unverified_speaker_stream(o.process_stream("meu hobby é escalada")))
        assert seen == [True]

    def test_the_deferred_worker_carries_the_mark(self, tmp_path, monkeypatch):
        o = _build_real_orch(tmp_path)
        seen = []

        def process_core(*_a, **_k):
            seen.append((threading.current_thread().name, speaker_unverified()))
            return "resposta", "OK"
        monkeypatch.setattr(o, "_process_core", process_core)
        text = "meu hobby é escalada. qual arquivo eu leio sobre isso no meu computador?"
        assert not o._is_pure_chat(text)
        list(unverified_speaker_stream(o.process_stream(text)))
        assert seen == [("cortex-stream-turn", True)]


class _RecordingOrchestrator:
    def __init__(self):
        self.seen = []

    def process(self, user_input, image_b64=None, language=None, conversation_history=None):
        self.seen.append(speaker_unverified())
        return "ok", "OK"

    def process_stream(self, user_input, image_b64=None, language=None, conversation_history=None):
        self.seen.append(speaker_unverified())
        yield ("Que legal!", True, "OFFLINE", "pt")


def _client(monkeypatch, orch):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "orchestrator", orch)
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    return TestClient(api_mod.app, raise_server_exceptions=False)


class TestTheApi:

    def test_the_listener_endpoint_marks_its_turn(self, monkeypatch):
        orch = _RecordingOrchestrator()
        r = _client(monkeypatch, orch).post(
            "/ask_stream", headers={"X-Cortex-Key": "test-key-xyz"},
            json={"user_input": "meu hobby é escalada"})
        assert r.status_code == 200
        assert orch.seen == [True]

    def test_a_typed_ask_is_not_marked(self, monkeypatch):
        orch = _RecordingOrchestrator()
        r = _client(monkeypatch, orch).post(
            "/ask", headers={"X-Cortex-Key": "test-key-xyz"},
            json={"user_input": "meu hobby é escalada"})
        assert r.status_code == 200
        assert orch.seen == [False]
