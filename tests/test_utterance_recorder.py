"""Temporary utterance recording for STT A/B tests (s84) —
src/voice/utterance_recorder.py and its hook in listener.transcribe()."""
import json
import wave

import numpy as np
import pytest

from src.voice import utterance_recorder as rec


@pytest.fixture
def flag(tmp_path, monkeypatch):
    f = tmp_path / "record_on"
    monkeypatch.setattr(rec, "FLAG", str(f))
    return f


def _audio(seconds=0.5, rate=16000):
    return (np.sin(np.linspace(0, 440, int(seconds * rate))) * 8000).astype(np.int16)


def test_nothing_is_written_while_off(flag, tmp_path):
    out = tmp_path / "utt"
    assert rec.save(_audio(), 16000, "play daisies", "en", 0.8, directory=out) is None
    assert not out.exists()


def test_on_writes_a_readable_wav_and_what_was_heard(flag, tmp_path):
    flag.touch()
    out = tmp_path / "utt"
    audio = _audio(0.5)
    path = rec.save(audio, 16000, "Play you comb by Justin Bieber", "en", 0.87, directory=out)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == len(audio)
    (line,) = (out / rec.LOG_NAME).read_text().splitlines()
    entry = json.loads(line)
    assert entry["file"] == path.name
    assert entry["heard"] == "Play you comb by Justin Bieber"
    assert entry["seconds"] == 0.5


def test_only_the_newest_are_kept(flag, tmp_path, monkeypatch):
    flag.touch()
    monkeypatch.setattr(rec, "MAX_KEEP", 3)
    out = tmp_path / "utt"
    # Distinct, ordered names without waiting on the clock (save() reads
    # the clock twice per call, so the fake one never runs out).
    import datetime as _dt
    import itertools
    ticks = itertools.count()
    monkeypatch.setattr(rec, "datetime", type("D", (), {
        "now": staticmethod(lambda tz=None: _dt.datetime(2026, 10, 4, 10, 0, 0)
                            + _dt.timedelta(seconds=next(ticks))),
    }))
    for i in range(5):
        rec.save(_audio(0.1), 16000, f"utt {i}", "en", 0.9, directory=out)
    wavs = sorted(p.name for p in out.glob("utt-*.wav"))
    assert len(wavs) == 3
    heard = [json.loads(l)["heard"] for l in (out / rec.LOG_NAME).read_text().splitlines()]
    assert heard == ["utt 2", "utt 3", "utt 4"]


def test_a_failure_never_breaks_the_turn(flag, tmp_path):
    flag.touch()
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")
    assert rec.save(_audio(), 16000, "x", "en", 0.9, directory=blocked) is None


def test_transcribe_hands_every_capture_to_the_recorder(monkeypatch):
    # One chokepoint: command, follow-up and barge-in all call transcribe().
    from src.voice import listener
    seen = []
    monkeypatch.setattr(listener.utterance_recorder, "save",
                        lambda a, r, text, lang, prob: seen.append((r, text, lang)) or None)

    class _Seg:
        text = "Play daisies by Justin Bieber"

    class _Info:
        language, language_probability = "en", 0.85

    class _Whisper:
        def transcribe(self, audio, **kw):
            return [_Seg()], _Info()

    text, lang, prob = listener.transcribe(_audio(), _Whisper())
    assert text == "Play daisies by Justin Bieber"
    assert seen == [(listener.SAMPLE_RATE, "Play daisies by Justin Bieber", "en")]


def test_transcribe_uses_beam_5_by_default(monkeypatch):
    # s84: beam 5 got 10/15 spoken names right on the user's recordings, beam 1
    # got 8, with no damage to the control sentences.
    from src.voice import listener
    monkeypatch.setattr(listener.utterance_recorder, "save", lambda *a, **k: None)
    seen = {}

    class _Info:
        language, language_probability = "pt", 0.9

    class _Whisper:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return [], _Info()

    listener.transcribe(_audio(), _Whisper())
    assert seen["beam_size"] == listener.STT_BEAM_SIZE == 5


class TestOnlyPedrosFlagRecords:
    """/tmp is shared by every account on the Mac: a flag another one left
    under this name switched the recording on (s88, Codex lote 12)."""

    def test_a_flag_owned_by_another_account(self, flag, tmp_path, monkeypatch):
        import os
        flag.touch()
        real = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real + 1)
        out = tmp_path / "utt"
        assert rec.save(_audio(), 16000, "x", "en", 0.9, directory=out) is None
        assert not out.exists()

    def test_a_link_in_the_flags_name(self, flag, tmp_path):
        target = tmp_path / "elsewhere"
        target.touch()
        flag.symlink_to(target)
        out = tmp_path / "utt"
        assert rec.save(_audio(), 16000, "x", "en", 0.9, directory=out) is None

    def test_pedros_own_flag_records(self, flag, tmp_path):
        flag.touch()
        out = tmp_path / "utt"
        assert rec.save(_audio(), 16000, "x", "en", 0.9, directory=out) is not None



def test_a_dictated_secret_is_masked_in_the_log(flag, tmp_path):
    """s88, Codex lote 12: the log kept "minha senha é exemplo123" whole,
    while the audit masks the same transcript."""
    flag.touch()
    out = tmp_path / "utt"
    rec.save(_audio(0.1), 16000, "minha senha é exemplo123", "pt", 0.9, directory=out)
    log = (out / rec.LOG_NAME).read_text()
    assert "exemplo123" not in log
    assert "minha senha é [REDACTED]" in json.loads(log.splitlines()[0])["heard"]



def test_the_take_after_the_clocks_go_back_is_kept(flag, tmp_path, monkeypatch):
    """25/10/2026: three takes near 01:50 BST, then one at 01:05 GMT — an
    hour later, but its wall-clock name sorts first. Sorted by name, the
    new take was the one deleted (s88, Codex lote 12 — invariant 5)."""
    import datetime as _dt
    import os
    import time as _time
    flag.touch()
    monkeypatch.setattr(rec, "MAX_KEEP", 3)
    out = tmp_path / "utt"
    out.mkdir()
    now = _time.time()
    for i, minute in enumerate((50, 51, 52)):        # BST, the hour before
        p = out / f"utt-20261025-01{minute}00-000.wav"
        p.write_bytes(b"x")
        t = now - 3600 + i * 60
        os.utime(p, (t, t))

    class D:                                          # 01:05 GMT, the repeat
        @staticmethod
        def now(tz=None):
            return _dt.datetime(2026, 10, 25, 1, 5, tzinfo=tz)
    monkeypatch.setattr(rec, "datetime", D)
    path = rec.save(_audio(0.1), 16000, "a nova", "pt", 0.9, directory=out)
    assert path is not None and path.exists()
    names = {p.name for p in out.glob("utt-*.wav")}
    assert "utt-20261025-015000-000.wav" not in names      # the oldest went
    assert len(names) == 3


def test_names_are_utc(flag, tmp_path, monkeypatch):
    import datetime as _dt
    flag.touch()
    seen = []

    class D:
        @staticmethod
        def now(tz=None):
            seen.append(tz)
            return _dt.datetime(2026, 10, 25, 1, 5, tzinfo=tz)
    monkeypatch.setattr(rec, "datetime", D)
    path = rec.save(_audio(0.1), 16000, "x", "pt", 0.9, directory=tmp_path / "utt")
    assert seen[0] is _dt.timezone.utc
    assert path.name.startswith("utt-20261025-010500-") and path.name.endswith("Z.wav")
