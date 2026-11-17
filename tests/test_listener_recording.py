"""
record_after_wake / listen_followup — the M5-A6/A7 recording contract.

The old recorder had a fixed 8s cap that truncated long dictation
SILENTLY (an email body cut mid-sentence reached staging) and, on a
false wake, shipped ~8s of near-silence to Whisper. The new contract:

  - speech extends the recording up to REC_HARD_MAX_S;
  - trailing silence ends it naturally (truncated=False);
  - exhausting the hard ceiling mid-speech returns truncated=True
    (callers warn and DROP the command);
  - no speech at all within REC_NO_SPEECH_S returns an EMPTY array
    early — Whisper never runs on silence.

All exercised against a fake stream — no audio hardware, no Whisper.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import src.voice.listener as l


LOUD = np.full((l.FRAME_SIZE, 1), 3000, dtype=np.int16)    # rms 3000 > 500
QUIET = np.zeros((l.FRAME_SIZE, 1), dtype=np.int16)
# Mid-level mic: speech RMS that STRADDLES the 500 threshold — some frames
# just above, some just below (the real-world case the clean LOUD/QUIET
# fixtures never modelled, which is why the s75 strict-consecutive arming
# passed tests but went deaf live on 2026-06-15).
MID = np.full((l.FRAME_SIZE, 1), 700, dtype=np.int16)      # rms 700 > 500
DIP = np.full((l.FRAME_SIZE, 1), 300, dtype=np.int16)      # rms 300 < 500


class FakeStream:
    """Serves a scripted sequence of frames; silence after it runs out."""

    def __init__(self, frames, overflow_at=()):
        self.frames = list(frames)
        self.reads = 0
        self.overflow_at = set(overflow_at)

    def read(self, n):
        i = self.reads
        self.reads += 1
        frame = self.frames[i] if i < len(self.frames) else QUIET
        return frame, (i in self.overflow_at)


def _frames(seconds: float) -> int:
    return int(seconds * l.SAMPLE_RATE / l.FRAME_SIZE)


@pytest.fixture(autouse=True)
def fast_windows(monkeypatch):
    """Small windows so tests run on dozens of frames, not thousands."""
    monkeypatch.setattr(l, "REC_HARD_MAX_S", 4.0)
    monkeypatch.setattr(l, "REC_NO_SPEECH_S", 1.0)


@pytest.fixture(autouse=True)
def energy_gate(monkeypatch):
    """Drive the recorder with the RMS fallback, not Silero.

    These tests are about the recorder's STATE MACHINE — when it arms, when
    trailing silence closes the window, when the hard cap fires — and they
    feed constant-amplitude tones. Silero is right to say a 3000-amplitude
    square tone is not speech, but that verdict would make every timing test
    here meaningless. So the gate runs in fallback mode and LOUD/QUIET keep
    meaning "speech"/"silence"; the detector itself is covered by
    TestVadGate against real audio."""
    monkeypatch.setattr(l, "_vad_gate", None)
    gate = l.VadGate.__new__(l.VadGate)
    gate.ok = False                    # → is_speech falls back to RMS
    monkeypatch.setattr(l, "_vad_gate", gate)


class TestRecordAfterWake:
    def test_wake_without_speech_returns_empty_early(self):
        # M5-A7: nobody spoke — empty audio, and the loop gave up at the
        # no-speech window instead of recording the whole hard cap.
        s = FakeStream([])  # pure silence
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size == 0
        assert truncated is False
        assert s.reads <= _frames(1.0) + 1

    def test_natural_end_is_not_truncated(self):
        speech = [LOUD] * _frames(0.9)
        s = FakeStream(speech)  # silence follows automatically
        pcm, truncated = l.record_after_wake(s)
        assert truncated is False
        assert pcm.size >= _frames(0.9) * l.FRAME_SIZE

    def test_speech_extends_past_the_old_8s_cap(self, monkeypatch):
        # The headline of M5-A6: continuous speech longer than the old
        # fixed cap is captured WHOLE (here: cap analogue 1s, speech 2s,
        # hard ceiling 4s).
        monkeypatch.setattr(l, "REC_HARD_MAX_S", 4.0)
        speech = [LOUD] * _frames(2.0)
        s = FakeStream(speech)
        pcm, truncated = l.record_after_wake(s)
        assert truncated is False
        assert pcm.size >= _frames(2.0) * l.FRAME_SIZE  # nothing cut

    def test_nonstop_speech_hits_hard_cap_and_flags_truncation(self):
        speech = [LOUD] * _frames(10.0)  # talks past the 4s ceiling
        s = FakeStream(speech)
        pcm, truncated = l.record_after_wake(s)
        assert truncated is True
        assert pcm.size == _frames(4.0) * l.FRAME_SIZE  # capped, not lost

    def test_pure_silence_never_flags_truncation(self, monkeypatch):
        # Misconfig guard: no-speech window >= hard cap must not label
        # silence as a truncation.
        monkeypatch.setattr(l, "REC_NO_SPEECH_S", 99.0)
        s = FakeStream([])
        pcm, truncated = l.record_after_wake(s)
        assert truncated is False

    # ---- s75 (Onda 8 B): sustained-speech arming — smoke finding 2 ----
    # The live bug: 'listened 1.76s' → empty transcription. A one-frame
    # transient (breath / wake-word tail) armed the recorder at ~0.26s
    # and REC_SILENCE_S closed it at ~1.76s, before the question started.

    def test_one_frame_transient_does_not_arm(self):
        # OLD behaviour: returned ~1.76s of near-silence to Whisper.
        # NEW: the transient never arms; the give-up clock returns empty.
        s = FakeStream([LOUD])  # 80ms blip, then silence
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size == 0
        assert truncated is False

    def test_transient_then_pause_then_question_is_captured(self):
        # The s74 Copa scenario: breath, think-pause, then the real
        # question — the question must be recorded whole.
        blip = [LOUD]
        pause = [QUIET] * _frames(0.4)
        question = [LOUD] * _frames(0.9)
        s = FakeStream(blip + pause + question)
        pcm, truncated = l.record_after_wake(s)
        assert truncated is False
        # Everything from the blip on is kept (chunks start at frame 0).
        assert pcm.size >= (len(blip) + len(pause) + len(question)) * l.FRAME_SIZE

    def test_short_word_still_arms(self):
        # "sim" / "para" ≈ 0.3-0.4s sustained — must still count as speech
        # (REC_ARM_SPEECH_S default 0.24s = 3 frames).
        s = FakeStream([LOUD] * 4)
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size > 0
        assert truncated is False

    def test_straddling_mic_arms(self):
        # THE 2026-06-15 live regression: speech whose RMS alternates just
        # above/below the threshold. Strict-consecutive arming reset the run
        # every other frame and NEVER reached arm_frames → empty (deaf). The
        # dip-tolerant run must accumulate across single dips and record.
        s = FakeStream([MID, DIP] * 10)
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size > 0          # armed + captured (was 0 before the fix)
        assert truncated is False

    def test_sustained_quiet_still_resets(self, monkeypatch):
        # The dip tolerance must NOT swallow a real transient: one MID frame
        # then sustained quiet must still give up empty (never arms).
        monkeypatch.setattr(l, "_ARM_QUIET_RESET", 2)
        s = FakeStream([MID])  # one mid frame, then silence
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size == 0
        assert truncated is False

    def test_overflow_is_counted_not_fatal(self, capsys):
        speech = [LOUD] * _frames(0.9)
        s = FakeStream(speech, overflow_at={1, 2})
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size > 0
        # M5-A10c: dropped frames leave a trace for "wake não pegou" debug.
        assert "overflow" in capsys.readouterr().out


class TestListenFollowup:
    def test_no_speech_within_window_returns_none(self):
        s = FakeStream([])
        pcm, truncated = l.listen_followup(s, max_wait_s=0.05)
        assert pcm is None
        assert truncated is False

    def test_followup_natural_end(self):
        speech = [LOUD] * _frames(0.9)
        s = FakeStream(speech)
        pcm, truncated = l.listen_followup(s, max_wait_s=5.0)
        assert pcm is not None and pcm.size > 0
        assert truncated is False

    def test_followup_nonstop_speech_flags_truncation(self):
        speech = [LOUD] * _frames(10.0)
        s = FakeStream(speech)
        pcm, truncated = l.listen_followup(s, max_wait_s=5.0)
        assert pcm is not None
        assert truncated is True

    def test_followup_transient_does_not_arm(self):
        # s75: one hot frame inside the follow-up window used to start a
        # recording that trailing silence shipped to Whisper as noise.
        s = FakeStream([LOUD])  # blip, then silence until the deadline
        pcm, truncated = l.listen_followup(s, max_wait_s=0.05)
        assert pcm is None
        assert truncated is False

    def test_followup_straddling_mic_arms(self):
        # Same 2026-06-15 fix in the follow-up path: alternating just-above/
        # just-below speech must arm instead of being dropped frame by frame.
        s = FakeStream([MID, DIP] * 10)
        pcm, truncated = l.listen_followup(s, max_wait_s=5.0)
        assert pcm is not None and pcm.size > 0


class TestConfigAndPhrases:
    def test_env_float_malformed_falls_back(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TEST_FLOAT_X", "30s")
        assert l._env_float("CORTEX_TEST_FLOAT_X", 7.5) == 7.5
        monkeypatch.setenv("CORTEX_TEST_FLOAT_X", "12")
        assert l._env_float("CORTEX_TEST_FLOAT_X", 7.5) == 12.0
        monkeypatch.delenv("CORTEX_TEST_FLOAT_X", raising=False)
        assert l._env_float("CORTEX_TEST_FLOAT_X", 7.5) == 7.5

    def test_truncation_phrases_exist_for_both_langs(self):
        assert set(l._TRUNCATED_PHRASES) == {"pt", "en"}
        for phrase in l._TRUNCATED_PHRASES.values():
            assert phrase.strip()

    def test_tts_wavs_are_private_tempfiles(self):
        # M5-A10d: per-process unguessable 0600 files, not a fixed
        # world-writable-dir path. Two of them since M5-A8 (double-buffer).
        import os
        assert len(l._TTS_WAV_PATHS) == 2
        for path in l._TTS_WAV_PATHS:
            assert "cortex_tts_" in os.path.basename(path)
            assert (os.stat(path).st_mode & 0o777) == 0o600


# --- mic calibration (s82) --------------------------------------------------
# The threshold was a hardcoded 500. Measuring the user's own recordings, the
# MEDIAN frame of his speech is ~225 — 64% of real speech read as silence.
# One mis-tuned constant produced BOTH complaints: cut off mid-sentence
# (armed on a peak, own quiet syllables closed the window) and "the mic takes
# forever to close" (never armed, so the window ran to the 30s ceiling).

# A quiet speaker: every frame is real speech, but below the old 500 line.
SOFT = np.full((l.FRAME_SIZE, 1), 225, dtype=np.int16)
ROOM = np.full((l.FRAME_SIZE, 1), 100, dtype=np.int16)     # noise floor


class TestQuietSpeakerRegression:
    def test_soft_speech_is_deaf_at_the_old_threshold(self, monkeypatch):
        """Documents the bug: at 500, a 225-RMS voice never arms, so the
        recorder gives up as if nobody had spoken."""
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        s = FakeStream([SOFT] * _frames(3.0))
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size == 0          # heard nothing — this was the live bug

    def test_soft_speech_is_captured_at_a_calibrated_threshold(self, monkeypatch):
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 200)
        speech = _frames(1.5)
        s = FakeStream([SOFT] * speech + [ROOM] * _frames(2.0))
        pcm, truncated = l.record_after_wake(s)
        assert pcm.size > 0
        assert truncated is False     # closed naturally on real silence
        # captured the speech plus the trailing-silence window, not a fragment
        assert pcm.size >= speech * l.FRAME_SIZE

    def test_room_noise_alone_does_not_arm_a_calibrated_recorder(self, monkeypatch):
        # The other half: the threshold must stay clear of the room, or the
        # recorder arms on nothing and runs to the hard ceiling.
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 200)
        s = FakeStream([ROOM] * _frames(4.0))
        pcm, _ = l.record_after_wake(s)
        assert pcm.size == 0


class TestThresholdFromFloor:
    def test_places_the_threshold_above_the_measured_floor(self):
        assert l.threshold_from_floor(100) == pytest.approx(250)
        assert l.threshold_from_floor(80) == pytest.approx(200)

    def test_clamped_both_ways(self):
        # A near-dead stream must not produce a threshold of ~0 (everything
        # becomes speech); a slammed door must not leave the mic deaf.
        assert l.threshold_from_floor(0.0) == l.CALIBRATION_MIN
        assert l.threshold_from_floor(99999) == l.CALIBRATION_MAX


class TestMeasureNoiseFloor:
    def test_uses_the_median_so_one_bang_does_not_skew_it(self):
        frames = [ROOM] * 20 + [np.full((l.FRAME_SIZE, 1), 9000, dtype=np.int16)]
        floor = l.measure_noise_floor(FakeStream(frames), seconds=1.68)
        assert floor == pytest.approx(100, abs=2)   # the bang is ignored


class TestCalibrateThreshold:
    def test_sets_the_threshold_from_the_room(self, monkeypatch):
        monkeypatch.setattr(l, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD_EXPLICIT", False)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        got = l.calibrate_threshold(FakeStream([ROOM] * 40))
        assert got == pytest.approx(250)
        assert l.REC_RMS_THRESHOLD == pytest.approx(250)

    def test_an_explicit_env_value_always_wins(self, monkeypatch):
        """If Pedro ran the calibration script and wrote the number down, a
        boot-time guess must never quietly replace it."""
        monkeypatch.setattr(l, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD_EXPLICIT", True)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 180)
        assert l.calibrate_threshold(FakeStream([ROOM] * 40)) is None
        assert l.REC_RMS_THRESHOLD == 180

    def test_disabled_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(l, "CALIBRATION_ENABLED", False)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        assert l.calibrate_threshold(FakeStream([ROOM] * 40)) is None
        assert l.REC_RMS_THRESHOLD == 500

    def test_a_dead_stream_does_not_calibrate_to_zero(self, monkeypatch):
        # A stale CoreAudio capture reads ~0 forever (StreamWatchdog's case).
        # Calibrating off that would make every frame "speech".
        monkeypatch.setattr(l, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD_EXPLICIT", False)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        dead = np.zeros((l.FRAME_SIZE, 1), dtype=np.int16)
        assert l.calibrate_threshold(FakeStream([dead] * 40)) is None
        assert l.REC_RMS_THRESHOLD == 500

    def test_never_raises_into_startup(self, monkeypatch):
        class Boom:
            def read(self, n):
                raise RuntimeError("mic exploded")
        monkeypatch.setattr(l, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD_EXPLICIT", False)
        assert l.calibrate_threshold(Boom()) is None


class TestCalibrationNeverMakesItWorse:
    """Boot calibration exists to RESCUE a mic quieter than the default
    assumes. Letting it raise the threshold above the shipped default would
    let one second of silence silently make things worse — which is exactly
    what nearly happened on the first live boot: the room measured ~300, and
    an unclamped factor of 2.5 would have set 750 against the 500 it replaced.
    """

    def test_a_noisy_room_never_exceeds_the_shipped_default(self):
        assert l.threshold_from_floor(301) <= 500
        assert l.threshold_from_floor(10_000) <= 500

    def test_a_quiet_room_still_lowers_it(self):
        assert l.threshold_from_floor(100) == pytest.approx(250)
        assert l.threshold_from_floor(60) == pytest.approx(150)


# --- the speech detector itself (s82 cont.) ---------------------------------
# Why the energy gate was replaced: measured in the user's room, the noise floor
# is 765 RMS and his speech 1079 — a 1.41x ratio where EVERY candidate
# threshold sits below the floor. No number separates those. Silero separates
# by content: noise LOUDER than his voice still scores near zero.

def _tone(amp, seconds=1.0):
    return np.full((int(l.SAMPLE_RATE * seconds), 1), amp, dtype=np.int16)[:, 0]


class TestVadGate:
    def test_falls_back_to_rms_when_the_model_is_unavailable(self, monkeypatch):
        # A voice daemon that won't start is worse than one that mis-hears.
        gate = l.VadGate.__new__(l.VadGate)
        gate.ok = False
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        assert gate.is_speech(_tone(3000)) is True
        assert gate.is_speech(_tone(100)) is False

    def test_a_broken_model_degrades_instead_of_raising(self, monkeypatch):
        class Boom:
            def __call__(self, *a, **k):
                raise RuntimeError("onnx down")
        gate = l.VadGate()
        if not gate.ok:
            pytest.skip("silero unavailable in this environment")
        gate._model = Boom()
        monkeypatch.setattr(l, "REC_RMS_THRESHOLD", 500)
        assert gate.is_speech(_tone(3000)) is True     # fell back, didn't raise

    def test_loud_noise_is_not_speech(self):
        """The property the RMS gate could never have: rejecting sound that is
        LOUDER than the speech it must not be confused with."""
        gate = l.VadGate()
        if not gate.ok:
            pytest.skip("silero unavailable in this environment")
        rng = np.random.default_rng(0)
        # white noise at ~2035 RMS — well above the user's 1079 speech level
        noise = (rng.normal(0, 2035 / 32768, l.SAMPLE_RATE)
                 * 32768).astype(np.int16)
        for i in range(0, len(noise) - l.FRAME_SIZE, l.FRAME_SIZE):
            gate.is_speech(noise[i:i + l.FRAME_SIZE])
        assert gate.is_speech(noise[-l.FRAME_SIZE:]) is False

    def test_silence_is_not_speech(self):
        gate = l.VadGate()
        if not gate.ok:
            pytest.skip("silero unavailable in this environment")
        quiet = np.zeros(l.FRAME_SIZE, dtype=np.int16)
        for _ in range(20):
            gate.is_speech(quiet)
        assert gate.is_speech(quiet) is False

    def test_real_speech_is_speech(self):
        """Pedro's own enrolment clips — the only real speech available
        offline. Skipped rather than faked when they aren't there."""
        import glob
        import wave
        files = sorted(glob.glob(
            os.path.expanduser("~/.cortex/enroll_clips/*.wav")))
        if not files:
            pytest.skip("no enrolment clips on this machine")
        gate = l.VadGate()
        if not gate.ok:
            pytest.skip("silero unavailable in this environment")
        w = wave.open(files[0])
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        seen = False
        for i in range(0, len(pcm) - l.FRAME_SIZE, l.FRAME_SIZE):
            if gate.is_speech(pcm[i:i + l.FRAME_SIZE]):
                seen = True
                break
        assert seen, "Silero found no speech in a clip of Pedro speaking"

    def test_reset_drops_the_rolling_context(self):
        gate = l.VadGate()
        if not gate.ok:
            pytest.skip("silero unavailable in this environment")
        gate.is_speech(_tone(3000))
        assert gate._buf.size > 0
        gate.reset()
        assert gate._buf.size == 0


def test_every_recording_path_shares_one_speech_test():
    """Three copies of a speech test would drift; this project has paid for
    that before (feedback-cross-cutting-all-entry-points)."""
    src = (Path(l.__file__)).read_text()
    assert src.count("if is_speech_frame(mono):") == 3
    assert "if rms(mono) > REC_RMS_THRESHOLD:" not in src


# --- wake pre-roll (s82) ----------------------------------------------------
# The chime is 1.4s long (Blow.aiff) and its frames used to be DISCARDED, so
# anyone speaking straight through lost the opening of the command. the user's
# audit: "para hoje e qual o clima de amanhã" — "o que eu tenho" gone.
# Keeping the audio is safe: the chime echo transcribes identically (measured
# at 30% and 60% mix) and Silero scores the chime 0.07-0.13, far under the
# 0.5 that would arm the recorder.

class TestWakePreroll:
    def test_preroll_audio_reaches_the_recording(self):
        speech = _frames(1.5)
        pre = np.concatenate([LOUD[:, 0]] * 6)          # ~0.5s over the chime
        s = FakeStream([LOUD] * speech + [QUIET] * _frames(2.0))
        pcm, _ = l.record_after_wake(s, preroll=pre)
        assert pcm.size >= pre.size + speech * l.FRAME_SIZE - l.FRAME_SIZE
        # the pre-roll is at the FRONT — the words came first in time
        assert np.array_equal(pcm[:pre.size], pre)

    def test_no_preroll_behaves_exactly_as_before(self):
        s = FakeStream([LOUD] * _frames(1.0) + [QUIET] * _frames(2.0))
        a, _ = l.record_after_wake(s)
        s2 = FakeStream([LOUD] * _frames(1.0) + [QUIET] * _frames(2.0))
        b, _ = l.record_after_wake(s2, preroll=np.empty(0, dtype=np.int16))
        assert a.size == b.size

    def test_a_partial_frame_of_preroll_is_ignored_not_crashed(self):
        s = FakeStream([LOUD] * _frames(1.0) + [QUIET] * _frames(2.0))
        pcm, _ = l.record_after_wake(s, preroll=np.zeros(7, dtype=np.int16))
        assert pcm.size > 0        # too short to use, and harmless

    def test_chime_off_returns_an_empty_preroll(self, monkeypatch):
        monkeypatch.setattr(l, "WAKE_ACK", "off")
        out = l.acknowledge_wake(FakeStream([QUIET] * 10))
        assert out.size == 0



class TestACommandSaidOverTheChime:
    """The pre-roll's verdicts were ignored: a short command said entirely
    over the chime, followed by silence, left the recorder unarmed and the
    turn ended as "no speech" (s88, Codex lote 12). With Silero (the live
    gate) they now arm it; the energy fallback keeps ignoring them, since it
    would arm on the chime's own echo."""

    @staticmethod
    def _silero_like(monkeypatch):
        # A gate that reports Silero mode, with the RMS verdict standing in
        # for the model (these fixtures are tones, see energy_gate).
        l._vad_gate.ok = True
        monkeypatch.setattr(l, "is_speech_frame",
                            lambda mono: l.rms(mono) > l.REC_RMS_THRESHOLD)

    def test_the_command_is_kept(self, monkeypatch):
        self._silero_like(monkeypatch)
        pre = np.concatenate([LOUD[:, 0]] * 8 + [QUIET[:, 0]] * 8)
        s = FakeStream([QUIET] * _frames(3.0))
        pcm, truncated = l.record_after_wake(s, preroll=pre)
        assert pcm.size > 0 and not truncated
        assert np.array_equal(pcm[:pre.size], pre)

    def test_a_quiet_chime_still_means_no_speech(self, monkeypatch):
        self._silero_like(monkeypatch)
        pre = np.concatenate([QUIET[:, 0]] * 16)
        pcm, _ = l.record_after_wake(FakeStream([QUIET] * _frames(3.0)), preroll=pre)
        assert pcm.size == 0

    def test_the_energy_fallback_does_not_arm_on_the_chime(self):
        pre = np.concatenate([LOUD[:, 0]] * 8 + [QUIET[:, 0]] * 8)
        pcm, _ = l.record_after_wake(FakeStream([QUIET] * _frames(3.0)), preroll=pre)
        assert pcm.size == 0
