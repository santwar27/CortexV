"""
listener.stream_and_speak — barge-in during the THINKING phase.

Before the LLM emits its first token, the main loop is blocked reading
the SSE stream and nobody reads the mic — so the user couldn't interrupt
until speech started. A background "thinking watcher" reads the mic +
runs the wake model during that wait and aborts on "Hey Cortex".

Everything external is faked: no audio hardware, no HTTP, no model.
"""
import time

import numpy as np
import pytest

from src.voice import listener


@pytest.fixture(autouse=True)
def _isolated_silence_flag(monkeypatch, tmp_path):
    """stream_and_speak polls the real /tmp/cortex_silence between chunks.
    A flag dropped mid-suite by another test — or by the LIVE system
    (cortex-shush.sh / _handle_silence) while the suite runs on this same
    machine — turns a normal reply into SPEECH_INTERRUPTED (flaked on the
    2026-06-12 pre-push). Point the listener at a private path instead."""
    monkeypatch.setattr(listener, "SILENCE_FLAG",
                        str(tmp_path / "cortex_silence"))


class _FakeStream:
    """InputStream stand-in: read() returns a fixed mono frame."""
    def __init__(self):
        self.reads = 0

    def read(self, n):
        self.reads += 1
        return np.zeros((n, 1), dtype=np.int16), None

    # context-manager + lifecycle no-ops in case they're touched
    def start(self): pass
    def stop(self): pass
    def close(self): pass


class _WakeHot:
    """Wake model that always reports a score above any barge threshold."""
    def __init__(self):
        self.reset_calls = 0

    def predict(self, frame):
        return {listener.WAKE_CONFIG.lookup_key: 0.99}

    def reset(self):
        self.reset_calls += 1


class _WakeCold:
    """Wake model that never fires."""
    def predict(self, frame):
        return {listener.WAKE_CONFIG.lookup_key: 0.0}

    def reset(self):
        pass


class _SlowResp:
    """requests Response stand-in that 'thinks' (blocks) before yielding,
    and whose close() makes the in-flight iteration stop — mirroring how
    requests raises when the underlying connection is closed mid-read."""
    def __init__(self, lines, think_s=0.5):
        self._lines = lines
        self._think_s = think_s
        self.closed = False

    def raise_for_status(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.closed = True

    def iter_lines(self):
        # Simulate the model "thinking": block, checking for close().
        waited = 0.0
        while waited < self._think_s:
            if self.closed:
                raise RuntimeError("connection closed mid-read")
            time.sleep(0.02)
            waited += 0.02
        for ln in self._lines:
            if self.closed:
                raise RuntimeError("connection closed mid-read")
            yield ln

    def close(self):
        self.closed = True


def _patch_post(monkeypatch, resp, captured=None):
    def fake_post(url, json=None, headers=None, stream=None, timeout=None):
        if captured is not None:
            captured["body"] = json
        return resp
    monkeypatch.setattr(listener.requests, "post", fake_post)


def test_barge_during_thinking_aborts_before_speech(monkeypatch):
    # Wake fires immediately; the response "thinks" for 0.5s before any
    # token. The watcher must trip barge and close the response so we never
    # speak. Guard _speak so a regression that speaks would be caught.
    spoke = []
    monkeypatch.setattr(listener, "_synth_piper",
                        lambda *a, **k: spoke.append(a) or True)
    monkeypatch.setattr(listener.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not play audio after thinking-barge")))
    resp = _SlowResp(lines=[b'data: {"chunk": "ola"}',
                            b'data: {"done": true, "status_mode": "OFFLINE"}'],
                     think_s=1.0)
    _patch_post(monkeypatch, resp)

    status, full = listener.stream_and_speak(
        "pergunta", stream=_FakeStream(), wake=_WakeHot(),
    )
    assert status == "BARGE_IN"
    assert resp.closed, "watcher must close the response to unblock the read"


def test_no_barge_when_wake_cold_speaks_normally(monkeypatch):
    # Cold wake → no barge → the reply is spoken normally and completes.
    played = []
    monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
    monkeypatch.setattr(listener.subprocess, "Popen",
                        lambda *a, **k: _DoneProc(played))
    resp = _SlowResp(lines=[b'data: {"chunk": "Ola Pedro."}',
                            b'data: {"done": true, "status_mode": "OFFLINE"}'],
                     think_s=0.1)
    _patch_post(monkeypatch, resp)

    status, full = listener.stream_and_speak(
        "oi", stream=_FakeStream(), wake=_WakeCold(),
    )
    assert status == "OFFLINE"
    assert "Ola Pedro." in full
    assert played, "a normal reply should have played audio"


def test_stt_language_forwarded_to_server(monkeypatch):
    # The STT-detected language must ride in the request body so the server
    # replies in the same language (fixes PT question → EN reply when the
    # server's text heuristic mislabels accent-less PT). Regression guard.
    monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
    monkeypatch.setattr(listener.subprocess, "Popen", lambda *a, **k: _DoneProc([]))
    captured = {}
    resp = _SlowResp(lines=[b'data: {"chunk": "ola"}',
                            b'data: {"done": true, "status_mode": "OFFLINE"}'],
                     think_s=0.0)
    _patch_post(monkeypatch, resp, captured)

    listener.stream_and_speak(
        "toca musica", stream=_FakeStream(), wake=_WakeCold(), stt_lang="pt",
    )
    assert captured["body"].get("language") == "pt"

    # Unknown/garbage STT lang must NOT be forwarded (server falls back to
    # its own heuristic rather than getting a bad hint).
    captured.clear()
    resp2 = _SlowResp(lines=[b'data: {"done": true, "status_mode": "OFFLINE"}'],
                      think_s=0.0)
    _patch_post(monkeypatch, resp2, captured)
    listener.stream_and_speak(
        "hello", stream=_FakeStream(), wake=_WakeCold(), stt_lang="de",
    )
    assert "language" not in captured["body"]


class _DoneProc:
    """Popen stand-in for afplay that's already finished."""
    def __init__(self, sink):
        sink.append(True)

    def poll(self):
        return 0  # already exited

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass


class _PacedStream(_FakeStream):
    """A mic that delivers frames at a pace, so the watcher's frame clock
    (and the filler it times) moves with real time."""
    def read(self, n):
        time.sleep(0.01)
        return super().read(n)


class _Resp500(_SlowResp):
    def __init__(self):
        super().__init__(lines=[], think_s=0.0)

    def raise_for_status(self):
        import requests
        raise requests.HTTPError("500 Server Error")


class TestTheWatcherEndsWithTheRequest:
    """A refused connection or an HTTP 500 raised before the try/finally that
    stops the thinking watcher, and the watcher stayed alive: reading the mic
    against the wake loop, sharing the wake model, and saying "Só um momento"
    after the error phrase (s88, Codex lote 12 — invariant 7). The existing
    error tests passed stream=None, so no watcher ever started in them."""

    @pytest.fixture
    def filler(self, monkeypatch):
        said = []
        # 25 frames of the paced mic below: ~250 ms, after the error.
        monkeypatch.setattr(listener, "THINKING_FILLER_S", 2.0)
        monkeypatch.setattr(listener, "_speak_local_phrase",
                            lambda phrases, lang: said.append(lang))
        return said

    def _after_the_error(self, post, monkeypatch, filler):
        import requests
        import threading
        monkeypatch.setattr(listener.requests, "post", post)
        stream = _PacedStream()
        before = threading.active_count()
        with pytest.raises(requests.RequestException):
            listener.stream_and_speak("pergunta", stream=stream,
                                      wake=_WakeCold(), stt_lang="pt")
        reads = stream.reads
        time.sleep(0.5)                     # past the filler's ~250 ms
        assert stream.reads == reads, "the watcher still reads the mic"
        assert threading.active_count() == before
        assert filler == []

    def test_a_refused_connection(self, monkeypatch, filler):
        import requests

        def refuse(*a, **k):
            time.sleep(0.05)                # the watcher is reading by now
            raise requests.ConnectionError("connection refused")
        self._after_the_error(refuse, monkeypatch, filler)

    def test_an_http_500(self, monkeypatch, filler):
        resp = _Resp500()

        def post(*a, **k):
            time.sleep(0.05)
            return resp
        self._after_the_error(post, monkeypatch, filler)
        assert resp.closed


class _PlayingProc:
    """A sentence that plays for `polls` polls unless terminated."""
    def __init__(self, polls=200):
        self.left = polls
        self.terminated = False

    def poll(self):
        if self.terminated or self.left <= 0:
            return 0
        self.left -= 1
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True


def test_the_mute_cuts_the_last_sentence(monkeypatch):
    """After the last SSE line only the final wait runs, and it did not look
    at the silence flag: "cala" during the last sentence let it play to the
    end (s88, Codex lote 12)."""
    procs = []

    def popen(*_a, **_k):
        proc = _PlayingProc()
        procs.append(proc)
        open(listener.SILENCE_FLAG, "w").close()    # Pedro says "cala" now
        return proc
    monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
    monkeypatch.setattr(listener.subprocess, "Popen", popen)
    resp = _SlowResp(lines=[b'data: {"chunk": "Ola Pedro."}',
                            b'data: {"done": true, "status_mode": "OFFLINE"}'],
                     think_s=0.0)
    _patch_post(monkeypatch, resp)

    status, _full = listener.stream_and_speak(
        "oi", stream=_PacedStream(), wake=_WakeCold(),   # paced: no filler
    )
    assert len(procs) == 1 and procs[0].terminated
    assert status == "SPEECH_INTERRUPTED"


class TestTheMuteHoldsForLocalPhrases:
    """`_speak_local_phrase` and the thinking filler started audio without
    looking at the silence flag: muted before the 5 s, "Só um momento" still
    played; muted before an HTTP error, the error phrase still played (s88,
    Codex lote 12)."""

    def test_a_local_phrase_is_not_started(self, monkeypatch):
        started = []
        monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
        monkeypatch.setattr(listener.subprocess, "Popen",
                            lambda *a, **k: started.append(a) or _DoneProc([]))
        open(listener.SILENCE_FLAG, "w").close()
        listener.speak_offline_error("pt")
        assert started == []

    def test_unmuted_it_still_speaks(self, monkeypatch):
        started = []
        monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
        monkeypatch.setattr(listener.subprocess, "Popen",
                            lambda *a, **k: started.append(a) or _DoneProc([]))
        listener.speak_offline_error("pt")
        assert len(started) == 1

    def test_no_filler_once_muted(self, monkeypatch):
        said = []
        monkeypatch.setattr(listener, "THINKING_FILLER_S", 0.2)
        monkeypatch.setattr(listener, "_speak_local_phrase",
                            lambda phrases, lang: said.append(lang))
        open(listener.SILENCE_FLAG, "w").close()
        resp = _SlowResp(lines=[b'data: {"done": true, "status_mode": "OFFLINE"}'],
                         think_s=0.6)               # past the 0.2 s filler
        _patch_post(monkeypatch, resp)
        listener.stream_and_speak("pergunta", stream=_PacedStream(),
                                  wake=_WakeCold(), stt_lang="pt")
        assert said == []



class _HungProc:
    """afplay/say that never exits until stopped."""
    def __init__(self, *a, **k):
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None

    def wait(self, timeout=None):
        if self.stopped:
            return 0
        import subprocess
        if timeout is None:
            raise AssertionError("an unbounded wait on playback hangs forever")
        time.sleep(min(timeout, 0.05))
        raise subprocess.TimeoutExpired("afplay", timeout)

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.stopped = True


class TestAHungPlaybackIsStopped:
    """Every wait on afplay/say was unbounded: one that never exits held the
    speech queue for good (s88, Codex lote 12 — invariant 7)."""

    def test_a_local_phrase(self, monkeypatch):
        procs = []
        monkeypatch.setattr(listener, "PLAYBACK_MAX_S", 0.05)
        monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
        monkeypatch.setattr(listener.subprocess, "Popen",
                            lambda *a, **k: procs.append(_HungProc()) or procs[-1])
        listener.speak_offline_error("pt")
        assert procs and procs[0].stopped

    def test_a_sentence_while_listening_for_barge_in(self, monkeypatch):
        procs = []
        monkeypatch.setattr(listener, "PLAYBACK_MAX_S", 0.2)
        monkeypatch.setattr(listener, "_synth_piper", lambda *a, **k: True)
        monkeypatch.setattr(listener.subprocess, "Popen",
                            lambda *a, **k: procs.append(_HungProc()) or procs[-1])
        resp = _SlowResp(lines=[b'data: {"chunk": "Ola Pedro."}',
                                b'data: {"done": true, "status_mode": "OFFLINE"}'],
                         think_s=0.0)
        _patch_post(monkeypatch, resp)
        status, _ = listener.stream_and_speak("oi", stream=_PacedStream(),
                                              wake=_WakeCold())
        assert status == "OFFLINE"
        assert procs and procs[0].stopped

    def test_the_chime_past_its_cap(self, monkeypatch):
        proc = _HungProc()
        monkeypatch.setattr(listener, "play_wake_chime", lambda: proc)
        listener.acknowledge_wake(_FakeStream())
        assert proc.stopped



class TestTheDaemonLogIsMasked:
    """stdout goes to logs/voice.out.log, and every transcript and reply was
    printed whole: a dictated password sat there in clear (s88, Codex lote 12)."""

    def test_no_transcript_is_printed_raw(self):
        from pathlib import Path
        src = Path(listener.__file__).read_text()
        assert "{text!r}" not in src
        assert src.count("{redact_text(text)!r}") == 3

    def test_the_reply_line_is_masked(self, monkeypatch, capsys):
        monkeypatch.setattr(listener, "stream_and_speak",
                            lambda *a, **k: ("OFFLINE", "Anotado: sua senha é exemplo123."))
        listener.handle_utterance("minha senha é exemplo123", None, None, stt_lang="pt")
        out = capsys.readouterr().out
        assert "stream done" in out
        assert "exemplo123" not in out


class TestAPlainSimAnswersAConfirmation:
    """Only AWAITING_APPROVAL (the lease) opened the window without the wake
    word. Every S3 confirmation ("dorme o Mac", email, agenda) is
    AGENT_AWAITING_CONFIRMATION, so the "sim" after "responda sim" was never
    heard (s88, Codex lote 12). Pedro's call: just "sim", no wake word."""

    class _Mic:
        read_available = 0

    def _turn(self, monkeypatch, first_status):
        sent = []

        def stream_and_speak(text, **_k):
            sent.append(text)
            if len(sent) == 1:
                return first_status, "Quer que eu coloque o Mac pra dormir? Responda sim."
            return "AGENT_ACTION_DONE", "Feito."
        monkeypatch.setattr(listener, "stream_and_speak", stream_and_speak)
        monkeypatch.setattr(listener, "listen_followup",
                            lambda stream: (np.zeros(1600, dtype=np.int16), False))
        monkeypatch.setattr(listener, "transcribe", lambda pcm, w: ("sim", "pt", 0.99))
        listener.handle_utterance("dorme o mac", self._Mic(), None, stt_lang="pt")
        return sent

    @pytest.mark.parametrize("status", [
        "AGENT_AWAITING_CONFIRMATION", "AWAITING_APPROVAL", "REFINE_STAGED"])
    def test_the_sim_reaches_the_server(self, monkeypatch, status):
        assert self._turn(monkeypatch, status) == ["dorme o mac", "sim"]

    def test_an_ordinary_reply_opens_no_window(self, monkeypatch):
        assert self._turn(monkeypatch, "OFFLINE") == ["dorme o mac"]
