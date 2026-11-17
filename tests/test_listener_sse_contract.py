"""
listener ⇄ /ask_stream — the C8 (s62) side of the SSE contract.

Four behaviours, all client-side mirrors of the server changes:

1. The `{"language": ...}` prelude is authoritative for the TTS voice —
   the hand-maintained PT word list survives only as a fallback for a
   server that predates the contract (M5-A4/M10-A3).
2. An `error` frame is SPOKEN (server's localised `reply`, or a fixed
   local phrase) — a bare {"error"} used to mean total silence, which by
   voice is indistinguishable from a missed wake word (M5-A3/M10-A5).
3. An API failure (connection refused / timeout / raise_for_status) is
   spoken via local TTS only — never via the API that just failed.
4. A stale /tmp/cortex_silence flag is cleared on wake, so a shush
   pressed with nothing playing can't mute the next turn (M5-A2).

Everything external is faked: no audio hardware, no HTTP, no model.
"""
import os

import pytest
import requests

from src.voice import listener
from src.voice.tts_config import TTSConfig


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Resp:
    """requests Response stand-in that yields canned SSE lines."""
    def __init__(self, lines, status_error=False):
        self._lines = lines
        self._status_error = status_error
        self.closed = False

    def raise_for_status(self):
        if self._status_error:
            raise requests.HTTPError("500 Server Error")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class _DoneProc:
    """Popen stand-in for afplay/say that's already finished."""
    def __init__(self, *a, **k):
        pass

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass


@pytest.fixture
def piper_env(monkeypatch, tmp_path):
    """Pin the TTS engine to piper with distinguishable per-lang models and
    capture every synth call as (model, text). Popen is inert. The silence
    flag is isolated so a real /tmp/cortex_silence on the host can't mute
    the fake speech and flake these tests."""
    monkeypatch.setattr(listener, "SILENCE_FLAG",
                        str(tmp_path / "cortex_silence"))
    synths = []
    monkeypatch.setattr(
        listener, "TTS_CONFIG",
        TTSConfig("piper", "pt-model.onnx", "en-model.onnx",
                  "Luciana", "Daniel", "test"),
    )
    def fake_synth(model, text, wav_path):
        synths.append((model, text))
        return True
    monkeypatch.setattr(listener, "_synth_piper", fake_synth)
    monkeypatch.setattr(listener.subprocess, "Popen", _DoneProc)
    return synths


def _patch_post(monkeypatch, resp):
    monkeypatch.setattr(
        listener.requests, "post",
        lambda *a, **k: resp,
    )


# ---------------------------------------------------------------------------
# 1. language prelude drives the voice
# ---------------------------------------------------------------------------

class TestServerLanguageAuthority:
    def test_prelude_language_beats_text_heuristics(self, monkeypatch, piper_env):
        # "ok, done." reads as EN to every client heuristic — but the server
        # says the turn is PT (it generated the words; it knows). PT voice.
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"chunk": "ok, done."}',
            b'data: {"done": true, "status_mode": "OFFLINE", "language": "pt"}',
        ]))
        status, full = listener.stream_and_speak("qualquer coisa")
        assert status == "OFFLINE"
        assert piper_env, "the reply must have been synthesised"
        model, text = piper_env[0]
        assert model == "pt-model.onnx"

    def test_no_prelude_falls_back_to_old_heuristics(self, monkeypatch, piper_env):
        # Pre-C8 server: no language frame anywhere. The accent/word-list
        # heuristics must still work exactly as before.
        _patch_post(monkeypatch, _Resp([
            b'data: {"chunk": "ok, done."}',
            b'data: {"done": true, "status_mode": "OFFLINE"}',
        ]))
        listener.stream_and_speak("whatever")
        model, text = piper_env[0]
        assert model == "en-model.onnx"

    def test_choose_voice_lang_server_priority(self):
        # Unit-level: server lang wins over heuristics and stt_lang;
        # invalid server values are ignored (defensive).
        assert listener.choose_voice_lang("ok, done", "en", server_lang="pt") == "pt"
        assert listener.choose_voice_lang("ok, feito", "en", server_lang=None) == "pt"
        assert listener.choose_voice_lang("ok, feito", "en", server_lang="xx") == "pt"
        # s76 amendment to C8: an "en" announcement no longer beats PT
        # evidence in the WORDS — the streaming chat path announces the
        # input language before generation and the 8B can disobey the
        # directive (live: alba read "Não consegui abrir o arquivo...").
        # This exact case was pinned the other way pre-s76, on the theory
        # that the server always knows its words; the chat path broke it.
        assert listener.choose_voice_lang("São 20:01.", "pt", server_lang="en") == "pt"
        # The reverse stays absolute: a "pt" announcement is never
        # overridden (no accents is not proof of EN).
        assert listener.choose_voice_lang("Spotify: paused", "pt", server_lang="pt") == "pt"


def test_corrected_done_reply_updates_final_text_after_live_speech(monkeypatch, piper_env):
    def lines():
        yield b'data: {"language": "en"}'
        yield b'data: {"chunk": "I cannot create files. "}'
        assert piper_env, "speech must begin before the final event"
        yield b'data: {"done": true, "status_mode": "OFFLINE", "language": "en", "reply": "I can save files."}'

    _patch_post(monkeypatch, _Resp(lines()))
    status, full = listener.stream_and_speak("can you create files?")
    assert status == "OFFLINE"
    assert full == "I can save files."
    # The final metadata cannot retract already played speech, and must not
    # append a second whole answer to it.
    assert [text for _, text in piper_env] == ["I cannot create files."]


# ---------------------------------------------------------------------------
# 2. error frames are spoken, not swallowed
# ---------------------------------------------------------------------------

class TestErrorFrameSpoken:
    def test_server_error_reply_is_spoken(self, monkeypatch, piper_env):
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"error": "engine_stream_failure", '
            b'"reply": "Deu um erro aqui no servidor. Tenta de novo.", '
            b'"language": "pt"}',
        ]))
        status, full = listener.stream_and_speak("pergunta")
        assert status == "INTERNAL_ERROR"
        assert piper_env, "the error must be HEARD, not printed only"
        model, text = piper_env[0]
        assert "Deu um erro" in text
        assert model == "pt-model.onnx"

    def test_bare_error_frame_speaks_canned_phrase(self, monkeypatch, piper_env):
        # Pre-C8 server: {"error": ...} with no speakable reply. The fixed
        # local phrase fills in, in the best-known language (stt here).
        _patch_post(monkeypatch, _Resp([
            b'data: {"error": "boom"}',
        ]))
        status, _ = listener.stream_and_speak("question", stt_lang="en")
        assert status == "INTERNAL_ERROR"
        model, text = piper_env[0]
        assert text == listener._SERVER_ERROR_PHRASES["en"]
        assert model == "en-model.onnx"

    def test_mid_stream_error_after_chunks_is_spoken(self, monkeypatch, piper_env):
        # Some text flowed, then the engine died: the partial sentence may
        # have been spoken; the error must ALSO be spoken.
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"chunk": "Os agentes "}',
            b'data: {"error": "engine_stream_failure", '
            b'"reply": "Tive um problema pra gerar a resposta. Tenta de novo.", '
            b'"language": "pt"}',
        ]))
        status, full = listener.stream_and_speak("me explica agentes")
        assert status == "INTERNAL_ERROR"
        spoken = " ".join(t for _, t in piper_env)
        assert "Tive um problema" in spoken


# ---------------------------------------------------------------------------
# 3. API failure → offline-safe spoken error
# ---------------------------------------------------------------------------

class TestOfflineErrorSpeech:
    def test_speak_offline_error_uses_local_tts_only(self, monkeypatch, piper_env):
        # The API just failed — speaking the failure must not touch it.
        def _no_http(*a, **k):
            raise AssertionError("offline error speech must not call the API")
        monkeypatch.setattr(listener.requests, "post", _no_http)
        listener.speak_offline_error("en")
        model, text = piper_env[0]
        assert text == listener._API_ERROR_PHRASES["en"]
        assert model == "en-model.onnx"

    def test_speak_offline_error_defaults_to_pt(self, monkeypatch, piper_env):
        # No usable language hint → PT (the daemon's household language).
        listener.speak_offline_error(None)
        listener.speak_offline_error("de")
        assert [m for m, _ in piper_env] == ["pt-model.onnx", "pt-model.onnx"]

    def test_speak_offline_error_never_raises(self, monkeypatch):
        # An error about an error must never crash the wake loop.
        monkeypatch.setattr(
            listener, "TTS_CONFIG",
            TTSConfig("piper", "pt.onnx", "en.onnx", "Luciana", "Daniel", "t"),
        )
        monkeypatch.setattr(listener, "_synth_piper",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
        listener.speak_offline_error("pt")  # must not raise

    def test_connection_error_is_spoken(self, monkeypatch):
        # handle_utterance: requests.post dies → the user HEARS about it.
        spoken = []
        monkeypatch.setattr(listener, "speak_offline_error",
                            lambda lang: spoken.append(lang))
        def _refuse(*a, **k):
            raise requests.ConnectionError("connection refused")
        monkeypatch.setattr(listener.requests, "post", _refuse)
        listener.handle_utterance("que horas são", None, None, stt_lang="pt")
        assert spoken == ["pt"]

    def test_http_500_is_spoken(self, monkeypatch):
        # raise_for_status (the M4-A1 class: flaky network → 500 on "pausa")
        # lands in the same except — and must be just as audible.
        spoken = []
        monkeypatch.setattr(listener, "speak_offline_error",
                            lambda lang: spoken.append(lang))
        _patch_post(monkeypatch, _Resp([], status_error=True))
        listener.handle_utterance("pausa", None, None, stt_lang="pt")
        assert spoken == ["pt"]


# ---------------------------------------------------------------------------
# 4. stale silence flag cleared on wake
# ---------------------------------------------------------------------------

class TestStaleSilenceFlag:
    def test_clear_removes_orphan_flag(self, monkeypatch, tmp_path):
        flag = tmp_path / "cortex_silence"
        flag.touch()
        monkeypatch.setattr(listener, "SILENCE_FLAG", str(flag))
        listener._clear_stale_silence_flag()
        assert not flag.exists()

    def test_clear_is_noop_without_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(listener, "SILENCE_FLAG",
                            str(tmp_path / "cortex_silence"))
        listener._clear_stale_silence_flag()  # must not raise

    def test_orphan_flag_no_longer_mutes_after_clear(self, monkeypatch,
                                                     tmp_path, piper_env):
        # End-to-end of the M5-A2 scenario: shush pressed with nothing
        # playing leaves an orphan flag; the wake-time clear must let the
        # next turn speak normally instead of aborting on the first frame.
        flag = tmp_path / "cortex_silence"
        flag.touch()
        monkeypatch.setattr(listener, "SILENCE_FLAG", str(flag))
        listener._clear_stale_silence_flag()  # what main() now does on wake
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"chunk": "S\xc3\xa3o 20:01."}',
            b'data: {"done": true, "status_mode": "REFLEX", "language": "pt"}',
        ]))
        status, full = listener.stream_and_speak("que horas são")
        assert status == "REFLEX"
        assert piper_env, "the turn must be spoken — the orphan flag is gone"


# ---------------------------------------------------------------------------
# 5. the shush must not be outrun by the next sentence
# ---------------------------------------------------------------------------

class TestShushDuringPlayback:
    """The hotkey is pressed WHILE a sentence is playing — the only moment it
    is ever pressed — and _speak is blocked in _wait_say() at that point.

    The silence flag was checked only at the TOP of _speak, before the synth
    and before that blocking wait. So the next sentence had already passed
    the guard: the kill cut the audio playing, _wait_say returned, and the
    already-prepared sentence started anyway. Pedro reported it as "the
    shortcut doesn't work immediately" (s82, 09/09). The barge path had its
    re-check after the wait; the shush path never got one.

    Streaming made it visible rather than causing it: a web-search reply used
    to arrive as ONE chunk, so there was rarely a next sentence queued.
    """

    def test_a_shush_while_speaking_stops_the_next_sentence(
            self, monkeypatch, tmp_path):
        flag = tmp_path / "cortex_silence"
        monkeypatch.setattr(listener, "SILENCE_FLAG", str(flag))
        monkeypatch.setattr(
            listener, "TTS_CONFIG",
            TTSConfig("piper", "pt-model.onnx", "en-model.onnx",
                      "Luciana", "Daniel", "test"),
        )
        monkeypatch.setattr(listener, "_synth_piper",
                            lambda model, text, wav: True)

        played = []

        class _ShushOnWait(_DoneProc):
            """Playing this sentence is when the user hits the hotkey."""
            def __init__(self, args):
                super().__init__(args)
                played.append(args[-1])

            def wait(self, timeout=None):
                flag.touch()   # the hotkey, mid-playback
                return 0

        monkeypatch.setattr(listener.subprocess, "Popen", _ShushOnWait)
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"chunk": "Primeira frase. Segunda frase. Terceira."}',
            b'data: {"done": true, "status_mode": "ONLINE_RAG", "language": "pt"}',
        ]))
        listener.stream_and_speak("quem ganhou o jogo")
        # The first sentence plays; the shush lands during it; nothing after
        # it may reach the speakers.
        assert len(played) == 1, f"spoke past the shush: {played}"

    def test_without_a_shush_every_sentence_still_speaks(
            self, monkeypatch, piper_env):
        # The guard must not cost a normal multi-sentence reply.
        _patch_post(monkeypatch, _Resp([
            b'data: {"language": "pt"}',
            b'data: {"chunk": "Primeira frase. Segunda frase."}',
            b'data: {"done": true, "status_mode": "ONLINE_RAG", "language": "pt"}',
        ]))
        listener.stream_and_speak("quem ganhou o jogo")
        assert len(piper_env) == 2
