"""
Wake acknowledgement chime (s76).

A short chime fires the instant the wake word is detected, so the user knows
they were heard before speaking. No audio hardware here — afplay is faked.
Same posture as test_listener_tts.

The chime is played NON-blocking and the mic is read+discarded for its
duration (s76 cont.45c): blocking on afplay starved the InputStream and
desynced CoreAudio on some device states ("deaf after the chime"); shadowing
the chime with continuous reads prevents the starvation instead of recovering
from it.
"""
import numpy as np

import src.voice.listener as l


class _FakeProc:
    """afplay stand-in. Non-blocking: poll() returns None while 'running'
    (alive_polls times) then 0 (exited)."""

    def __init__(self, args=None, alive_polls=0):
        self.args = args
        self.alive_polls = alive_polls
        self.polls = 0

    def poll(self):
        if self.polls < self.alive_polls:
            self.polls += 1
            return None
        return 0


def _spy_popen(calls, alive_polls=0):
    def popen(args, *a, **k):
        calls.append(args)
        return _FakeProc(args, alive_polls=alive_polls)
    return popen


# --- play_wake_chime: non-blocking launch, returns the handle ----------------

def test_chime_plays_afplay_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(l, "WAKE_ACK", "chime")
    monkeypatch.setattr(l, "WAKE_CHIME_PATH", "/snd/Blow.aiff")
    monkeypatch.setattr(l.subprocess, "Popen", _spy_popen(calls))
    proc = l.play_wake_chime()
    assert calls == [["afplay", "/snd/Blow.aiff"]]
    assert proc is not None        # returns the handle so the caller can drain


def test_chime_skipped_when_off(monkeypatch):
    calls = []
    monkeypatch.setattr(l, "WAKE_ACK", "off")
    monkeypatch.setattr(l.subprocess, "Popen", _spy_popen(calls))
    assert l.play_wake_chime() is None    # off → nothing to wait on
    assert calls == []                    # afplay never invoked


def test_chime_is_failsoft(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("afplay")
    monkeypatch.setattr(l, "WAKE_ACK", "chime")
    monkeypatch.setattr(l.subprocess, "Popen", boom)
    assert l.play_wake_chime() is None    # dead chime → None, never raises


def test_defaults_chime_on_with_blow(monkeypatch):
    # The shipped defaults: ack on, the user's chosen Blow sound.
    monkeypatch.delenv("CORTEX_WAKE_ACK", raising=False)
    monkeypatch.delenv("CORTEX_WAKE_CHIME", raising=False)
    ack = (l.os.getenv("CORTEX_WAKE_ACK") or "chime").strip().lower()
    chime = l.os.getenv("CORTEX_WAKE_CHIME") or "/System/Library/Sounds/Blow.aiff"
    assert ack == "chime"
    assert chime.endswith("Blow.aiff")


# --- acknowledge_wake: read THROUGH the chime so the stream is never starved -

class _CountingStream:
    """Counts reads and returns a real (frame, overflowed) pair — since s82
    acknowledge_wake KEEPS the frames it reads (the pre-roll), so a stub that
    returns None no longer models the stream."""

    def __init__(self):
        self.reads = 0

    def read(self, n):
        self.reads += 1
        return np.zeros((n, 1), dtype=np.int16), False


def test_acknowledge_wake_reads_through_the_chime_and_keeps_the_audio(monkeypatch):
    """Chime ON: play non-blocking and keep READING the stream while afplay
    runs (so PortAudio never starves → no CoreAudio desync).

    s82 changed what happens to those frames. They used to be discarded and
    the tail drained; with a 1.4s chime that threw away the first second and a
    half of anyone speaking straight through the ack. They are now RETURNED as
    the pre-roll and prepended to the recording — and the drain is gone,
    because the backlog it discarded is exactly where the opening words sit.
    """
    proc = _FakeProc(alive_polls=3)
    monkeypatch.setattr(l, "play_wake_chime", lambda: proc)
    drained = []
    monkeypatch.setattr(l, "drain", lambda s: drained.append(True))
    s = _CountingStream()
    pre = l.acknowledge_wake(s)
    assert s.reads == 3          # one read per "still playing" poll → no gap
    assert drained == []         # NOT drained any more — that was the bug
    assert pre.size == 3 * l.FRAME_SIZE   # every frame read comes back


def test_acknowledge_wake_noop_when_off(monkeypatch):
    # Chime OFF (or afplay launch failed) → play_wake_chime returns None, so
    # the stream is left untouched: byte-for-byte the pre-chime path.
    monkeypatch.setattr(l, "play_wake_chime", lambda: None)
    drained = []
    monkeypatch.setattr(l, "drain", lambda s: drained.append(True))
    s = _CountingStream()
    l.acknowledge_wake(s)
    assert s.reads == 0
    assert drained == []


def test_acknowledge_wake_caps_reads_if_chime_hangs(monkeypatch):
    # A wedged afplay (poll() never exits) must not deafen the daemon: the
    # read loop is capped at ~2s of frames, then we move on.
    never_exits = _FakeProc(alive_polls=10_000)
    monkeypatch.setattr(l, "play_wake_chime", lambda: never_exits)
    monkeypatch.setattr(l, "drain", lambda s: None)
    s = _CountingStream()
    pre = l.acknowledge_wake(s)
    cap = int(2.0 * l.SAMPLE_RATE / l.FRAME_SIZE)
    assert s.reads == cap        # stopped at the safety cap, not forever
    assert pre.size == cap * l.FRAME_SIZE   # and the pre-roll is bounded too


def test_acknowledge_wake_is_failsoft_on_read_error(monkeypatch):
    # A read that raises mid-chime must not kill the turn.
    proc = _FakeProc(alive_polls=5)
    monkeypatch.setattr(l, "play_wake_chime", lambda: proc)
    monkeypatch.setattr(l, "drain", lambda s: None)

    class _Boom:
        def read(self, n):
            raise RuntimeError("coreaudio")

    l.acknowledge_wake(_Boom())   # must not raise
