"""
Capture-stream watchdog — auto-recover the mic after sleep (s76 cont.45).

A Clamshell sleep/wake re-indexes the MacBook mic and leaves the boot-time
InputStream STALE: read() returns silence forever (no error), the wake model
only ever sees zeros, and the daemon goes deaf until a MANUAL restart. The
watchdog detects that (sustained near-zero RMS) and reopens the mic BY NAME on
its own, so a sleep no longer needs a hand-restart.

No audio hardware — sounddevice is faked. Same posture as test_listener_wake_ack.
"""
import src.voice.listener as l


# --- StreamWatchdog: dead-stream detection -----------------------------------

def test_watchdog_fires_after_sustained_silence():
    wd = l.StreamWatchdog(dead_rms=1.0, dead_frames=5)
    # Four silent frames: not dead yet.
    assert [wd.observe(0.0) for _ in range(4)] == [False, False, False, False]
    # The fifth crosses the threshold → dead, and it stays dead until reset
    # (the caller resets right after reopening, so it fires once per spell).
    assert wd.observe(0.0) is True
    assert wd.observe(0.0) is True


def test_watchdog_loud_frame_resets_the_run():
    # A live mic's noise floor / any speech resets the silence run, so a
    # healthy-but-quiet stream never false-fires the reopen.
    wd = l.StreamWatchdog(dead_rms=1.0, dead_frames=3)
    assert wd.observe(0.0) is False
    assert wd.observe(0.0) is False
    assert wd.observe(50.0) is False     # noise floor / speech — resets the run
    # Run restarts from zero; two more silent frames still aren't enough.
    assert wd.observe(0.0) is False
    assert wd.observe(0.0) is False
    assert wd.observe(0.0) is True       # now 3 consecutive → dead


def test_watchdog_threshold_is_strict_below_boot_selftest():
    # The dead threshold sits BELOW the boot self-test's MUTED line (5.0): a
    # frame at the ambient noise floor (e.g. 3.0) reads as ALIVE, never accrues.
    wd = l.StreamWatchdog(dead_rms=1.0, dead_frames=2)
    assert wd.observe(3.0) is False
    assert wd.observe(3.0) is False
    assert wd.silent_run == 0


def test_watchdog_reset_clears_the_run():
    wd = l.StreamWatchdog(dead_rms=1.0, dead_frames=2)
    wd.observe(0.0)
    wd.reset()
    assert wd.silent_run == 0
    assert wd.observe(0.0) is False      # back to needing 2 again


def test_watchdog_defaults_pull_from_module(monkeypatch):
    monkeypatch.setattr(l, "STREAM_DEAD_RMS", 2.5)
    monkeypatch.setattr(l, "STREAM_DEAD_FRAMES", 42)
    wd = l.StreamWatchdog()
    assert wd.dead_rms == 2.5
    assert wd.dead_frames == 42


# --- reopen_mic_stream: tear down + reopen BY NAME, fail-soft ----------------

class _FakeOldStream:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


def test_reopen_closes_old_and_opens_fresh_by_name(monkeypatch):
    old = _FakeOldStream()
    fresh = object()
    opened = []
    monkeypatch.setattr(l, "find_macbook_mic", lambda: 0)   # re-RESOLVED index
    monkeypatch.setattr(l, "open_mic_stream",
                        lambda idx: opened.append(idx) or fresh)
    monkeypatch.setattr(l.sd, "query_devices", lambda i: {"name": "MacBook Pro Microphone"})
    out = l.reopen_mic_stream(old)
    assert out is fresh                       # caller gets the new stream
    assert old.calls == ["stop", "close"]     # stale stream torn down
    assert opened == [0]                       # reopened on the re-resolved index


def test_reopen_failsoft_returns_none_on_open_error(monkeypatch):
    old = _FakeOldStream()
    monkeypatch.setattr(l, "find_macbook_mic", lambda: 0)

    def boom(idx):
        raise RuntimeError("PaErrorCode -9996")
    monkeypatch.setattr(l, "open_mic_stream", boom)
    out = l.reopen_mic_stream(old)
    assert out is None                        # the caller must retry, not read old
    assert old.calls == ["stop", "close"]


def test_reopen_tears_down_old_even_if_stop_raises(monkeypatch):
    # A stop/close that raises on the dead stream must not abort the reopen.
    class _Boom:
        def stop(self):
            raise RuntimeError("coreaudio")

        def close(self):
            raise RuntimeError("coreaudio")

    fresh = object()
    monkeypatch.setattr(l, "find_macbook_mic", lambda: 1)
    monkeypatch.setattr(l, "open_mic_stream", lambda idx: fresh)
    monkeypatch.setattr(l.sd, "query_devices", lambda i: {"name": "MacBook"})
    out = l.reopen_mic_stream(_Boom())        # must not raise
    assert out is fresh


def test_open_mic_stream_starts_the_stream(monkeypatch):
    started = []

    class _FakeStream:
        def start(self):
            started.append(True)

    captured = {}

    def fake_inputstream(**kw):
        captured.update(kw)
        return _FakeStream()
    monkeypatch.setattr(l.sd, "InputStream", fake_inputstream)
    s = l.open_mic_stream(3)
    assert isinstance(s, _FakeStream)
    assert started == [True]                  # started → ready to read()
    assert captured["device"] == 3
    assert captured["samplerate"] == l.SAMPLE_RATE
    assert captured["blocksize"] == l.FRAME_SIZE



# --- recover_mic_stream: a failed reopen is retried, never read closed -------

def test_a_failed_reopen_is_retried_not_read(monkeypatch):
    """The first reopen fails after closing the old stream; the loop used to
    keep that closed stream and the next read ended the daemon (s88, Codex
    lote 12). Recovery now returns only a stream that opened."""
    old = _FakeOldStream()
    fresh = object()
    attempts = []

    def reopen(stream):
        attempts.append(stream)
        if stream is not None:
            stream.stop()
            stream.close()
        return fresh if len(attempts) >= 3 else None
    monkeypatch.setattr(l, "reopen_mic_stream", reopen)
    slept = []
    out = l.recover_mic_stream(old, sleep=slept.append)
    assert out is fresh
    assert attempts == [old, None, None]      # the closed one is never reused
    assert slept == [1.0, 2.0]                # backoff between attempts


def test_a_missing_device_after_wake_is_one_failed_attempt(monkeypatch):
    """find_macbook_mic() exits when the device is not listed; right after a
    wake that is a transient, not the end of the daemon."""
    calls = []

    def reopen(stream):
        calls.append(stream)
        if len(calls) == 1:
            raise SystemExit("ERROR: no input device matching 'MacBook' found.")
        return "fresh"
    monkeypatch.setattr(l, "reopen_mic_stream", reopen)
    assert l.recover_mic_stream(_FakeOldStream(), sleep=lambda s: None) == "fresh"
