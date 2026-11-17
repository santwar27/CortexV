"""
Etapa 9C — Voice listener (foreground).

Full pipeline: wake word -> record until silence -> Whisper STT ->
POST /ask -> macOS `say` TTS -> resume listening.

Mic: hardcoded to MacBook built-in (substring match), regardless of
the system default. Speaker: follows the system default device (so
TTS plays through AirPods if connected).

When the API replies with a reply that waits for "sim"/"não" (see
FOLLOWUP_STATUSES), a 10s follow-up window opens so the user can answer
without needing to say the wake word again.

Run:
    ./venv/bin/python -m src.voice.listener
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel

from src.core.redaction import redact_text
from src.core.text_utils import strip_markdown_for_speech
from src.voice import utterance_recorder
from src.voice.speaker_id import SpeakerVerifier
from src.voice.telemetry import VoiceTelemetry
from src.voice.tts_config import resolve_tts_config
from src.voice.laugh_try import prepare_tts, synth_piper
from src.voice.wake_config import resolve_wake_config, resolve_wake_threshold

load_dotenv()

API_URL = os.getenv("CORTEX_API_URL", "http://localhost:8000")
API_KEY = os.getenv("CORTEX_API_KEY")
API_TIMEOUT_S = 90

# Wake model resolved at import time via CORTEX_WAKE_MODEL_PATH env var.
# Unset → 'hey_jarvis' built-in (Etapa 9 placeholder). Set to a .onnx
# path → custom model (e.g. 'src/voice/models/hey_cortex.onnx' after the
# Colab training session). See src/voice/wake_config.py for the contract.
WAKE_CONFIG = resolve_wake_config()
WAKE_THRESHOLD = resolve_wake_threshold()
WAKE_COOLDOWN_S = 2.0

# Voice barge-in: saying the wake word DURING a reply cuts it and records
# the next command. On by default; set CORTEX_BARGE_IN=0 (or false/no/off)
# to fall back to plain blocking playback — the rollback if speaker echo
# ever false-triggers the wake model and cuts replies on its own.
BARGE_IN_ENABLED = os.getenv("CORTEX_BARGE_IN", "1").strip().lower() not in (
    "0", "false", "no", "off", "",
)
# Barge-in uses its OWN threshold, separate from (and by default higher
# than) the normal wake threshold. During playback the mic also picks up
# the speaker echo, so we want to be more conservative about what counts
# as "the user said the wake word again" to avoid false cut-offs. Tune
# via CORTEX_BARGE_THRESHOLD; falls back to 0.5 on a bad/empty value.
BARGE_THRESHOLD = resolve_wake_threshold(os.getenv("CORTEX_BARGE_THRESHOLD") or "0.5")

SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80 ms at 16 kHz — openWakeWord chunk size


def _env_float(name: str, default: float) -> float:
    """Float env with fallback — a malformed value must not kill the
    daemon at import (same lesson as CORTEX_LEASE_HOURS, M1-A12b)."""
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        print(f"  [config: {name} malformed → {default}]")
        return default


# Recording window (M5-A6/A7). The old fixed 8s cap truncated long
# dictation SILENTLY (an email body cut mid-sentence reaches staging and
# can be approved half-written) and, on a false wake, shipped 8s of
# near-silence to Whisper. Now: speech keeps the recorder alive up to a
# HARD ceiling; trailing silence ends it; and a wake with no speech at
# all gives up early, skipping Whisper entirely.
REC_HARD_MAX_S = _env_float("CORTEX_REC_HARD_MAX_S", 30.0)
REC_NO_SPEECH_S = _env_float("CORTEX_REC_NO_SPEECH_S", 5.0)
REC_SILENCE_S = _env_float("CORTEX_REC_SILENCE_S", 1.5)
REC_MIN_S = _env_float("CORTEX_REC_MIN_S", 0.8)
# Speech/silence threshold in RMS. s82 — this was HARDCODED at 500, which is
# far above what the user's mic actually delivers: measuring his own recordings,
# the MEDIAN frame of his speech is ~225 and only the peaks (p90 ~1100) clear
# 500, so 64% of real speech reads as "silence". That single mis-tuning
# produced BOTH voice complaints at once:
#   - armed on a loud syllable, then his own quieter syllables counted as
#     trailing silence and closed the window mid-sentence → Whisper received
#     only the tail ("no Spotify.", "MZ.", "É isso." — all from ~2s captures
#     in the live audit);
#   - or never armed at all while background speech kept it alive, so the
#     window ran to the 30s hard cap → "demora pra fechar o microfone".
# A fixed constant cannot know a microphone or a room, so the default is now
# CALIBRATED from the measured noise floor at boot (see calibrate_threshold);
# this value is only the fallback when calibration is off or fails. An
# explicit CORTEX_REC_RMS_THRESHOLD always wins over both.
REC_RMS_THRESHOLD = _env_float("CORTEX_REC_RMS_THRESHOLD", 500)
REC_RMS_THRESHOLD_EXPLICIT = os.getenv("CORTEX_REC_RMS_THRESHOLD") is not None
# Arming window (s75, Onda 8 B — smoke finding 2). A single 80ms frame
# above RMS (a breath, the tail of the wake word) used to arm the
# recorder; REC_SILENCE_S then closed the window at ~1.76s while the
# user was still drawing breath for the actual question — Whisper got
# near-silence and transcribed nothing. Real speech must SUSTAIN above
# the RMS threshold for this long (consecutive frames) before the
# trailing-silence close is armed. 0.24s = 3 frames: short words
# ("sim", "para") still arm; a one-frame transient never does.
# REC_NO_SPEECH_S also went 3s → 5s: the give-up clock keeps running
# through unarmed transients, and a think-pause after the wake word is
# normal before a long question.
REC_ARM_SPEECH_S = _env_float("CORTEX_REC_ARM_SPEECH_S", 0.24)
# How many CONSECUTIVE quiet frames reset the arming run (s75 follow-up, live
# 2026-06-15). The original s75 arming reset the run on ANY sub-threshold
# frame — which never armed on a mid-level mic whose speech RMS straddles
# REC_RMS_THRESHOLD (frames alternate just-above/just-below, the run resets
# every other frame, arm_frames is never reached → voice went DEAF after the
# wake word). Tolerating a single dip (default 2) lets such speech accumulate
# to arm_frames while still rejecting a one-frame transient (breath / wake
# tail). Raise to 3 via env if a very quiet mic still fails to arm.
_ARM_QUIET_RESET = max(1, int(_env_float("CORTEX_REC_ARM_QUIET", 2)))

# Capture-stream watchdog — auto-recover after sleep (s76 cont.45). The
# InputStream is opened ONCE at boot; a Clamshell sleep/wake re-indexes the
# MacBook mic ([1]→[0]) and leaves the old stream STALE — read() returns
# silence forever (no error), so the wake model only ever sees zeros and the
# daemon goes deaf until a MANUAL restart (this is the s76 cont.44 root cause).
# A live mic always reads a nonzero noise floor; a desynced CoreAudio capture
# returns ~0 indefinitely. So: sustained near-zero RMS (< STREAM_DEAD_RMS) for
# STREAM_DEAD_S seconds ⇒ reopen the mic BY NAME (find_macbook_mic survives the
# re-index; a stop/start on the old object keeps the dead index). The threshold
# sits well below the boot self-test's 5.0 so a quiet room never false-fires,
# and reopening a healthy stream is harmless (~100 ms) — so biasing toward
# reopening is safe. Kill-switch CORTEX_STREAM_WATCHDOG=0.
STREAM_WATCHDOG_ENABLED = os.getenv(
    "CORTEX_STREAM_WATCHDOG", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
STREAM_DEAD_RMS = _env_float("CORTEX_STREAM_DEAD_RMS", 1.0)
STREAM_DEAD_S = _env_float("CORTEX_STREAM_DEAD_S", 60.0)
STREAM_DEAD_FRAMES = max(1, int(STREAM_DEAD_S * SAMPLE_RATE / FRAME_SIZE))

FOLLOWUP_WAIT_S = 10.0
# Replies that wait for a "sim"/"não" — the three statuses that wait for a
# confirmation. Only the lease approval opened the
# window: "dorme o Mac" answered "responda sim", and a plain "sim" was never
# heard, because every S3 confirmation (email, agenda, lembrete, fechar app,
# dormir) is AGENT_AWAITING_CONFIRMATION (s88, Codex lote 12). the user's call
# (18/10): he answers with just "sim", no wake word.
FOLLOWUP_STATUSES = ("AWAITING_APPROVAL", "AGENT_AWAITING_CONFIRMATION",
                     "REFINE_STAGED")

WHISPER_MODEL_SIZE = "small"
# Beam search width for STT (s84). Measured on 14 of the user's recorded
# utterances (data/voice/utterances): beam 1 got 8 of 15 spoken names right
# ("Dazis", "U-Com by JSTB", "a Jonga"), beam 5 got 10, with no damage to the
# 5 control sentences and +0.14 s mean (1.69 → 1.83 s). Hotwords were tested
# too and rejected as an always-on setting: they fixed names but turned
# "qual o clima em Londres hoje" into "Cole Klima, London Resort".
# CORTEX_STT_BEAM=1 restores the old behaviour.
STT_BEAM_SIZE = int(os.getenv("CORTEX_STT_BEAM", "5"))

MIC_NAME_SUBSTR = "MacBook"

# TTS engine: Piper (neural, local) by default when the voice models are
# present, else the macOS `say` builtin. Resolved once; see tts_config.py.
TTS_CONFIG = resolve_tts_config()

# Double-buffered temp WAVs for Piper output (M5-A8): sentence N+1 is
# synthesised into one buffer while afplay is still playing the other, so
# a sentence boundary no longer pays the synth gap (~0.05-0.2s warm).
# Playback stays strictly serial (`_wait_say` blocks before the next
# afplay starts), so alternating TWO paths is enough to guarantee the
# file being written is never the file being played. Per-process
# tempfiles, unguessable + 0600 (M5-A10d).


def _mk_tts_wav() -> str:
    fd, path = tempfile.mkstemp(prefix="cortex_tts_", suffix=".wav")
    os.close(fd)  # piper/wave reopen by path; keep only the path
    return path


_TTS_WAV_PATHS = (_mk_tts_wav(), _mk_tts_wav())
_tts_wav_idx = 0


def _next_tts_wav() -> str:
    """Flip to the buffer the previous synth did NOT use, and return it."""
    global _tts_wav_idx
    _tts_wav_idx = 1 - _tts_wav_idx
    return _TTS_WAV_PATHS[_tts_wav_idx]


# Loaded PiperVoice objects, keyed by model path. Lazy + cached so each
# voice model's ~0.7s load is paid once (on first use of that language),
# not per sentence; warm synth is then ~0.05-0.2s/sentence.
_PIPER_VOICES: dict = {}


def _get_piper_voice(model_path: str):
    """Return the cached PiperVoice for `model_path`, loading on first
    use. Raises on load failure — callers decide the fallback."""
    voice = _PIPER_VOICES.get(model_path)
    if voice is None:
        from piper import PiperVoice

        voice = PiperVoice.load(model_path)
        _PIPER_VOICES[model_path] = voice
    return voice


def preload_piper_voices(langs: tuple = ("pt",)) -> None:
    """Warm the Piper voice cache at daemon boot (M5-A8): the ~0.7s model
    load moves from the FIRST spoken turn (audible silence) to startup
    (nobody waiting). PT only by default — this daemon runs in a PT-BR
    household; EN still lazy-loads on first use. Fail-soft: a missing
    model just means the first turn pays the load like before."""
    if TTS_CONFIG.engine != "piper":
        return
    for lang in langs:
        try:
            _get_piper_voice(TTS_CONFIG.piper_model_for(lang))
            print(f"  [tts: piper voice <{lang}> pre-loaded]")
        except Exception as e:  # noqa: BLE001 — warm-up must never block boot
            print(f"  [tts: piper pre-load <{lang}> failed ({e}) — lazy load]")


def _synth_piper(model_path: str, text: str, out_path: str) -> bool:
    """Synthesise `text` to a WAV at `out_path` with Piper. Returns True on
    success, False (so the caller falls back to `say`) on any failure —
    a bad synth must never kill the reply."""
    return synth_piper(model_path, text, out_path, voice_loader=_get_piper_voice)

# When stdout isn't a TTY (i.e. launchd redirects to a log file) we skip
# the rolling score bar so the log doesn't bloat with thousands of carriage
# returns per minute. Wake-detect / transcription lines still print.
DAEMON_MODE = not sys.stdout.isatty()


def find_macbook_mic() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and MIC_NAME_SUBSTR in d["name"]:
            return i
    sys.exit(f"ERROR: no input device matching '{MIC_NAME_SUBSTR}' found.")


def open_mic_stream(mic_index: int):
    """Open and START a fresh capture stream on `mic_index`. Factored out of
    main() so the watchdog can reopen the mic after a sleep/wake leaves the
    old stream stale (s76 cont.45)."""
    stream = sd.InputStream(
        device=mic_index,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=FRAME_SIZE,
    )
    stream.start()
    return stream


def reopen_mic_stream(old_stream):
    """Tear down a stale capture stream and open a fresh one, RE-RESOLVING the
    MacBook mic by name (find_macbook_mic) so a sleep/wake device re-index
    ([1]→[0]) is picked up — a stop/start on the old object would keep the dead
    index. Fail-soft: on any failure returns None. The old stream is closed by
    then, so the caller must not read it again — recover_mic_stream retries."""
    try:
        if old_stream is not None:
            old_stream.stop()
            old_stream.close()
    except Exception:
        pass
    try:
        idx = find_macbook_mic()
        stream = open_mic_stream(idx)
        name = sd.query_devices(idx)["name"]
        print(f"  [mic stream reopened → [{idx}] {name}]", flush=True)
        return stream
    except Exception as e:
        print(f"  [mic stream reopen FAILED: {e}]", flush=True)
        return None


MIC_REOPEN_MAX_BACKOFF_S = 30.0


def recover_mic_stream(old_stream, sleep=time.sleep):
    """Reopen the mic until it opens. A failed reopen used to hand the main
    loop back the stream it had just closed: the next read raised and ended
    the daemon, and nothing restarts it — the listener is not a launchd job
    (s88, Codex lote 12 — invariant 7). Retries with a backoff instead; the
    daemon stays deaf, not dead, until the device is back.

    find_macbook_mic() exits the process when no matching device is listed
    (a boot-time check); here the device may be briefly gone after a wake,
    so that exit counts as one failed attempt."""
    delay = 1.0
    try:
        new = reopen_mic_stream(old_stream)
    except SystemExit as e:
        print(f"  [mic stream reopen FAILED: {e}]", flush=True)
        new = None
    while new is None:
        sleep(delay)
        delay = min(delay * 2, MIC_REOPEN_MAX_BACKOFF_S)
        try:
            new = reopen_mic_stream(None)
        except SystemExit as e:
            print(f"  [mic stream reopen FAILED: {e}]", flush=True)
    return new


def speaker_shadow_check(
    pcm: np.ndarray,
    telem: Optional[VoiceTelemetry],
    verifier: Optional[SpeakerVerifier],
    context: str,
) -> None:
    """Onda 12 Fase A — SHADOW ONLY. Scores the same int16 buffer Whisper
    is about to transcribe against the enrolled voiceprint and logs it
    (audit, stage=voice, event=speaker_shadow). The return is None and no
    caller branches on anything here: enforcement is Fase B, and it only
    lands after this data sizes the real false-negative rate.

    Fail-soft like every guard in the hot path: any error degrades to an
    "unavailable" observation, never to a broken voice turn."""
    if telem is None or verifier is None:
        return
    try:
        samples = pcm.astype(np.float32) / 32768.0
        t0 = time.time()
        score = verifier.score(samples)
        score_ms = (time.time() - t0) * 1000.0
        telem.speaker_shadow(
            context=context,
            score=score,
            duration_s=len(pcm) / SAMPLE_RATE,
            score_ms=score_ms,
        )
        if score is not None:
            print(f"  [speaker shadow] {context}: score={score:.3f} "
                  f"({score_ms:.0f}ms)")
        else:
            print(f"  [speaker shadow] {context}: unavailable")
    except Exception as e:
        print(f"  [speaker shadow] {context} failed: {e}")


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


class StreamWatchdog:
    """Flags a DEAD capture stream so the loop can reopen the mic instead of
    needing a manual restart (s76 cont.45). The post-sleep stale-stream symptom
    is sustained silence: a live mic reads a nonzero noise floor, a desynced
    CoreAudio capture returns ~0 forever. observe() counts consecutive
    near-zero frames and fires True when the run first reaches dead_frames; any
    louder frame resets the run, so a real wake (loud) never trips it. Stateful
    and tiny — the loop just feeds it each frame's RMS."""

    def __init__(self, dead_rms: Optional[float] = None,
                 dead_frames: Optional[int] = None):
        self.dead_rms = STREAM_DEAD_RMS if dead_rms is None else dead_rms
        self.dead_frames = STREAM_DEAD_FRAMES if dead_frames is None else dead_frames
        self.silent_run = 0

    def observe(self, frame_rms: float) -> bool:
        """Feed one frame's RMS. Returns True once sustained silence reaches
        dead_frames — the caller then reopens the mic and calls reset() (which
        also gives a natural ~dead_frames back-off before the next attempt)."""
        if frame_rms < self.dead_rms:
            self.silent_run += 1
        else:
            self.silent_run = 0
        return self.silent_run >= self.dead_frames

    def reset(self) -> None:
        self.silent_run = 0


# --- speech detection: Silero VAD (s82 cont.) --------------------------------
# The recorder used to ask "is this frame LOUD?". That question has no good
# answer in the user's room: measured floor 765 RMS against speech at 1079 — a
# signal only 1.41x above the noise, where every candidate threshold sits
# BELOW the floor. The calibration script correctly refused to invent a
# number; the mistake was mine, in thinking the problem was picking a better
# one. An energy gate cannot separate speech from noise at that ratio, by
# definition.
#
# So ask a better question: "is this SPEECH?". Silero VAD ships inside
# faster-whisper (assets/silero_vad_v6.onnx) — already installed, no new
# dependency — and separates by CONTENT, not level. Measured on this machine:
#   the user's speech ............ p_max 0.99
#   white noise at his floor .. p_max 0.13
#   white noise at his SPEECH level ... p_max 0.07   (louder, still rejected)
#   120 Hz fan hum ............ p_max 0.11
# Cost: 4.8 ms over a 2 s window, against an 80 ms frame budget.
#
# The RMS threshold stays as a FALLBACK for when the model can't load — a
# voice daemon that won't start is worse than one that mis-hears.
VAD_ENABLED = os.getenv(
    "CORTEX_VAD", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
VAD_THRESHOLD = _env_float("CORTEX_VAD_THRESHOLD", 0.5)
# ~1 s of context, aligned to Silero's 512-sample window (31 * 512 = 15872).
VAD_WINDOW_SAMPLES = int(_env_float("CORTEX_VAD_WINDOW_SAMPLES", 15872))


class VadGate:
    """Rolling speech detector: feed frames, ask `is_speech`.

    Silero scores 512-sample windows, so we keep the last ~1 s aligned to that
    and take the MAX probability across it: speech anywhere in the recent
    window counts as "still talking", which is what the trailing-silence timer
    needs (a pause between words must not read as the end of the sentence).

    Falls back to the RMS threshold when the model is unavailable — `ok` says
    which mode is live so the caller can log it honestly."""

    def __init__(self, threshold: float = None, window_samples: int = None):
        self.threshold = VAD_THRESHOLD if threshold is None else threshold
        self.window = VAD_WINDOW_SAMPLES if window_samples is None else window_samples
        self._buf = np.zeros(0, dtype=np.float32)
        self._model = None
        self.ok = False
        if VAD_ENABLED:
            try:
                from faster_whisper.vad import get_vad_model
                self._model = get_vad_model()
                self.ok = True
            except Exception:  # noqa: BLE001 — degrade to RMS, never crash
                self._model = None
                self.ok = False

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        """True when the recent window contains speech. On any failure falls
        back to the energy test, so behaviour degrades rather than breaks."""
        if not self.ok:
            return rms(frame_int16) > REC_RMS_THRESHOLD
        try:
            f = frame_int16.astype(np.float32) / 32768.0
            self._buf = np.concatenate([self._buf, f])[-self.window:]
            n = (len(self._buf) // 512) * 512
            if n < 512:
                return False
            probs = np.asarray(self._model(self._buf[-n:])).ravel()
            return bool(probs.max() > self.threshold)
        except Exception:  # noqa: BLE001
            return rms(frame_int16) > REC_RMS_THRESHOLD


# --- mic calibration (s82) --------------------------------------------------
# A fixed RMS threshold cannot know a microphone, a preamp gain, or a room.
# The user's mic delivers a median speech frame of ~225 against a hardcoded 500,
# and the result was a recorder that both cut him off mid-sentence and ran to
# the 30s ceiling. So: measure the room's NOISE FLOOR at boot and place the
# threshold above it, instead of shipping a constant that happens to suit
# whatever machine it was written on.
#
# The floor is all a boot-time measurement can honestly get (asking the user to
# speak on every start would be worse than the bug). For the precise personal
# number — floor AND speech measured through the real capture path — there is
# `scripts/voice_calibrate.py`, whose output goes in .env and wins over this.
CALIBRATION_ENABLED = os.getenv(
    "CORTEX_MIC_CALIBRATE", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
CALIBRATION_S = _env_float("CORTEX_MIC_CALIBRATE_S", 1.2)
# Multiplier over the measured floor. 2.5 keeps a quiet room's hiss out while
# staying well under a speaking voice; the clamp stops a pathological reading
# (a slammed door during calibration, or a dead stream reading ~0) from
# producing a threshold that is deaf or permanently open.
CALIBRATION_FACTOR = _env_float("CORTEX_MIC_CALIBRATE_FACTOR", 2.5)
CALIBRATION_MIN = _env_float("CORTEX_MIC_CALIBRATE_MIN", 120.0)
# The ceiling is the SHIPPED DEFAULT on purpose: boot calibration may only
# LOWER the threshold, never raise it. Found the hard way (s82, first live
# boot): the self-test measured avg_rms 301 in the user's room, and at factor 2.5
# an unclamped calibration would have set 750 — worse than the 500 it was
# replacing, silently, while claiming to be an improvement. Auto-calibration
# exists to rescue a mic quieter than the default assumes; a room that truly
# needs a HIGHER threshold needs the measured number from
# scripts/voice_calibrate.py, not a guess from one second of silence.
CALIBRATION_MAX = _env_float("CORTEX_MIC_CALIBRATE_MAX", 500.0)


def measure_noise_floor(stream, seconds: float = None) -> float:
    """MEDIAN frame RMS over a short window — the room with nobody talking.

    Median, not mean: one cough or door slam during calibration would drag a
    mean up and leave the mic deaf for the whole session."""
    seconds = CALIBRATION_S if seconds is None else seconds
    frames = max(1, int(seconds * SAMPLE_RATE / FRAME_SIZE))
    levels = []
    for _ in range(frames):
        audio, _overflowed = stream.read(FRAME_SIZE)
        levels.append(rms(audio[:, 0]))
    return float(np.median(levels)) if levels else 0.0


def threshold_from_floor(floor: float) -> float:
    """Place the speech threshold above a measured noise floor, clamped."""
    return float(min(CALIBRATION_MAX, max(CALIBRATION_MIN,
                                          floor * CALIBRATION_FACTOR)))


def calibrate_threshold(stream) -> Optional[float]:
    """Set REC_RMS_THRESHOLD from the measured room. Returns the new value, or
    None when calibration is off / overridden / unusable.

    An explicit CORTEX_REC_RMS_THRESHOLD always wins: if Pedro ran the
    calibration script and wrote a number down, a boot-time guess must not
    quietly replace it. Never raises — a failed calibration keeps the default,
    because a voice daemon that won't start is worse than one that mis-hears."""
    global REC_RMS_THRESHOLD
    if not CALIBRATION_ENABLED or REC_RMS_THRESHOLD_EXPLICIT:
        return None
    try:
        floor = measure_noise_floor(stream)
        # A dead/stale capture reads ~0 (see StreamWatchdog); calibrating off
        # that would set the floor threshold and make everything "speech".
        if floor < 1.0:
            print("  [mic: silent stream during calibration — keeping default]",
                  flush=True)
            return None
        REC_RMS_THRESHOLD = threshold_from_floor(floor)
        # flush=True is load-bearing, not tidiness: the daemon's stdout is
        # redirected to logs/voice.out.log, so an unflushed line sits in the
        # buffer for hours. On the first live boot this exact line was
        # invisible while the calibration HAD run — the one message that says
        # what the mic is set to, missing precisely when it is needed.
        print(f"  [mic: noise floor {floor:.0f} → speech threshold "
              f"{REC_RMS_THRESHOLD:.0f}]", flush=True)
        # Also to the audit, which survives log rotation and is greppable
        # months later next to the turns it explains.
        try:
            from src.core.audit import audit
            audit.log("–", "voice", "mic_calibrated",
                      payload={"floor": round(floor, 1),
                               "threshold": round(REC_RMS_THRESHOLD, 1)})
        except Exception:  # noqa: BLE001 — telemetry never blocks startup
            pass
        return REC_RMS_THRESHOLD
    except Exception as e:  # noqa: BLE001 — never block startup on this
        print(f"  [mic: calibration failed ({e}) — keeping default]", flush=True)
        return None


_vad_gate: "Optional[VadGate]" = None


def vad_gate() -> VadGate:
    """The shared speech detector. Lazy so the ONNX session is built once, on
    the first recording, instead of at import time."""
    global _vad_gate
    if _vad_gate is None:
        _vad_gate = VadGate()
        print(f"  [vad: {'silero' if _vad_gate.ok else 'rms fallback'}]",
              flush=True)
    return _vad_gate


def is_speech_frame(mono: np.ndarray) -> bool:
    """Speech test used by EVERY recording path (record_after_wake,
    listen_followup, barge-in capture). One function on purpose: three copies
    of a speech test drift, and this project has paid for that more than once
    (see feedback-cross-cutting-all-entry-points)."""
    return vad_gate().is_speech(mono)


def record_after_wake(stream, preroll: "Optional[np.ndarray]" = None
                      ) -> tuple[np.ndarray, bool]:
    """Record one utterance after the wake word. Returns (audio, truncated).

    M5-A6/A7 semantics:
      - speech (RMS above threshold) keeps the recorder alive up to
        REC_HARD_MAX_S; trailing silence of REC_SILENCE_S ends it — long
        dictation (an email body) is no longer cut at a fixed 8s;
      - `truncated=True` means the HARD ceiling fired while the user was
        still talking — the caller must warn instead of silently shipping
        half a command to staging;
      - no speech at all within REC_NO_SPEECH_S → returns an EMPTY array
        immediately (false wake / user gave up), so ~8s of near-silence
        never reaches Whisper (latency + STT-hallucination risk);
      - (s75) arming requires REC_ARM_SPEECH_S of SUSTAINED speech — a
        one-frame transient (breath, wake-word tail) no longer arms the
        trailing-silence close, which used to end the window at ~1.76s
        before the user's actual question started."""
    chunks = []
    vad_gate().reset()
    started = False
    speech_run = 0
    quiet_run = 0
    silence_frames = 0
    overflows = 0
    hard_max_frames = int(REC_HARD_MAX_S * SAMPLE_RATE / FRAME_SIZE)
    silence_target = int(REC_SILENCE_S * SAMPLE_RATE / FRAME_SIZE)
    min_frames = int(REC_MIN_S * SAMPLE_RATE / FRAME_SIZE)
    no_speech_frames = max(1, int(REC_NO_SPEECH_S * SAMPLE_RATE / FRAME_SIZE))
    arm_frames = max(1, int(REC_ARM_SPEECH_S * SAMPLE_RATE / FRAME_SIZE))
    # Audio captured while the wake chime played (s82). It goes into the
    # recording so nothing spoken over the chime is lost, and it is fed
    # through the detector so its rolling window already holds that speech —
    # which is what lets the first live frame arm immediately instead of
    # waiting for the speaker to produce another loud syllable.
    # Its verdicts arm the recorder too (s88, Codex lote 12): a short command
    # said entirely over the 1.4 s chime ("pausa") and then silence left the
    # recorder unarmed once the rolling window went quiet, and the turn ended
    # as "no speech". Only with Silero, which scores the chime itself
    # 0.07-0.13 (measured, s82); the energy fallback would arm on its echo.
    arm_on_preroll = vad_gate().ok
    if preroll is not None and preroll.size >= FRAME_SIZE:
        usable = (preroll.size // FRAME_SIZE) * FRAME_SIZE
        for i in range(0, usable, FRAME_SIZE):
            frame = preroll[i:i + FRAME_SIZE]
            chunks.append(frame)
            spoken = is_speech_frame(frame)
            if not arm_on_preroll:
                continue
            if spoken:
                speech_run += 1
                quiet_run = 0
                if speech_run >= arm_frames:
                    started = True
            else:
                quiet_run += 1
                if quiet_run >= _ARM_QUIET_RESET:
                    speech_run = 0

    truncated = True  # loop exhausting the hard cap mid-speech = cut
    for i in range(hard_max_frames):
        audio, overflowed = stream.read(FRAME_SIZE)
        if overflowed:
            overflows += 1  # M5-A10c: dropped frames explain "wake não pegou"
        mono = audio[:, 0]
        chunks.append(mono)
        if is_speech_frame(mono):
            speech_run += 1
            quiet_run = 0
            if speech_run >= arm_frames:
                started = True
            silence_frames = 0
        elif started:
            silence_frames += 1
            if silence_frames >= silence_target and i >= min_frames:
                truncated = False  # natural end: user stopped talking
                break
        else:
            # Quiet frame before arming. A SINGLE dip (a sub-threshold frame
            # between syllables on a mid-level mic) does not reset the run —
            # only SUSTAINED quiet (>= _ARM_QUIET_RESET frames) does. Rejects
            # a one-frame transient (breath/wake tail) yet still arms when the
            # speaker's RMS straddles the threshold (live 2026-06-15 fix).
            quiet_run += 1
            if quiet_run >= _ARM_QUIET_RESET:
                speech_run = 0
            if i + 1 >= no_speech_frames:
                # Wake fired but nobody spoke — don't run Whisper on silence.
                if overflows:
                    print(f"  [mic: {overflows} overflow(s) during recording]")
                return np.empty(0, dtype=np.int16), False

    if overflows:
        print(f"  [mic: {overflows} overflow(s) during recording]")
    # `started` guard: a misconfigured no-speech window larger than the
    # hard cap must not label pure silence as a truncation.
    return np.concatenate(chunks), truncated and started


def listen_followup(
    stream, max_wait_s: float = FOLLOWUP_WAIT_S,
) -> tuple[Optional[np.ndarray], bool]:
    """No wake required — record next utterance if it arrives within the
    window. Returns (audio_or_None, truncated) — same M5-A6 contract as
    record_after_wake: speech extends up to REC_HARD_MAX_S, and
    `truncated` means the hard ceiling cut the user off mid-speech.
    (s75) Same sustained-speech arming as record_after_wake: a one-frame
    transient inside the follow-up window no longer starts a recording
    that the trailing-silence close ships to Whisper as near-silence."""
    chunks = []
    silence_target = int(REC_SILENCE_S * SAMPLE_RATE / FRAME_SIZE)
    min_frames = int(REC_MIN_S * SAMPLE_RATE / FRAME_SIZE)
    hard_max_frames = int(REC_HARD_MAX_S * SAMPLE_RATE / FRAME_SIZE)
    arm_frames = max(1, int(REC_ARM_SPEECH_S * SAMPLE_RATE / FRAME_SIZE))

    deadline = time.time() + max_wait_s
    speech_started = False
    speech_run = 0
    quiet_run = 0
    while time.time() < deadline:
        audio, _ = stream.read(FRAME_SIZE)
        mono = audio[:, 0]
        if is_speech_frame(mono):
            chunks.append(mono)
            speech_run += 1
            quiet_run = 0
            if speech_run >= arm_frames:
                speech_started = True
                break
        else:
            # A single dip doesn't reset the run (same straddle-the-threshold
            # fix as record_after_wake, live 2026-06-15); only SUSTAINED quiet
            # drops the transient. The dip frame is kept so the lead-in chunks
            # stay contiguous through the gap.
            quiet_run += 1
            if quiet_run >= _ARM_QUIET_RESET:
                chunks = []
                speech_run = 0
            elif chunks:
                chunks.append(mono)

    if not speech_started:
        return None, False

    silence_frames = 0
    truncated = True  # exhausting the hard cap mid-speech = cut
    while len(chunks) < hard_max_frames:
        audio, _ = stream.read(FRAME_SIZE)
        mono = audio[:, 0]
        chunks.append(mono)
        if is_speech_frame(mono):
            silence_frames = 0
        else:
            silence_frames += 1
            if silence_frames >= silence_target and len(chunks) >= min_frames:
                truncated = False  # natural end
                break
    return np.concatenate(chunks), truncated


# Cortex is a PT/EN assistant. Whisper auto-detect occasionally picks a
# wrong language on short PT audio with Brazilian proper nouns — seen live
# transcribing "toca matuê no spotify" as German ("...und Spotify") or
# Polish ("matuje niespoczywanie"). When the guess is neither pt nor en,
# we re-transcribe forcing PT (the dominant language) so the text isn't
# foreign gibberish — which also keeps the regex dispatch (toca/toque…)
# from missing and dumping the turn onto the slow LLM path.
_STT_ACCEPTED_LANGS = {"pt", "en"}
_STT_FALLBACK_LANG = "pt"


def transcribe(audio_int16: np.ndarray, whisper: WhisperModel) -> tuple[str, str, float]:
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    def _run(language: Optional[str]) -> tuple[str, str, float]:
        segments, info = whisper.transcribe(
            audio_f32,
            beam_size=STT_BEAM_SIZE,
            vad_filter=True,
            language=language,
        )
        text = " ".join(s.text for s in segments).strip()
        return text, info.language, info.language_probability

    text, lang, prob = _run(None)  # auto-detect first
    if lang not in _STT_ACCEPTED_LANGS:
        print(f"  [stt: lang '{lang}' not pt/en → re-running as '{_STT_FALLBACK_LANG}']")
        text, lang, prob = _run(_STT_FALLBACK_LANG)
    # Opt-in local recording for STT A/B tests (s84, scripts/cortex-record.sh).
    # Here, not at the call sites: every capture — command, follow-up,
    # barge-in — goes through this function.
    if utterance_recorder.save(audio_int16, SAMPLE_RATE, text, lang, prob):
        print("  [● gravado para teste de transcrição]")
    return text, lang, prob


_SENTENCE_BOUNDARY_RE = re.compile(r'([.!?]+["\']?\s+|\n\n)')

# Soft breaks for early-speaking when the LLM is still building a long
# first sentence. Lets `say` start within ~2s even when the opening
# sentence is 200+ chars instead of waiting for the period.
_SOFT_BREAK_RE = re.compile(r'[,;:—–]\s+')
_MIN_SOFT_SPEAK_CHARS = 60  # ~2s of TTS at a moderate pace

# Flag file dropped by orchestrator._handle_silence() (cx cala / silencio /
# stop talking / etc). When seen mid-stream, listener kills its current
# `say`, drains the rest of the SSE response, and returns early.
SILENCE_FLAG = "/tmp/cortex_silence"


def _silence_requested() -> bool:
    try:
        if os.path.exists(SILENCE_FLAG):
            os.remove(SILENCE_FLAG)
            return True
    except OSError:
        pass
    return False


def detect_reply_language(text: str) -> str:
    if re.search(r"[ãõçáéíóúâêôà]", text, re.IGNORECASE):
        return "pt"
    return "en"


# Common PT words that carry no accent — so an accent-less PT reply
# ("ok, feito", "claro, pronto") is still voiced in PT. Kept tight (function
# words / confirmations) so it doesn't fire on English text.
#
# s60: extended with the accent-less PT vocabulary of the DETERMINISTIC command
# replies (now that commands localise, a PT reply like "Pausado." / "Anotado." /
# "Abrindo o Spotify." must be voiced PT, not fall to the EN default). Every word
# here is PT-SPECIFIC — none appears in the English command replies (those use
# Paused/Opening/Noted/…), so EN replies stay EN. Words identical across both
# languages (volume, mac, spotify, email, id) are deliberately NOT listed.
_PT_VOICE_HINT_RE = re.compile(
    r"\b(?:voce|nao|sim|com|que|uma|para|pra|claro|feito|pronto|agora|"
    r"aqui|tudo|bem|obrigad\w*|tocando|certo|isso|"
    # command-reply vocabulary (s60) — all PT-distinct
    r"abrindo|abri|fechando|fechei|fechado|fechada|pausado|continuando|"
    r"voltando|anotado|curtida|passando|travada|tela|silenciado|desmutado|"
    r"brilho|foco|sistema|contato|salvo|salva|atualizado|lembrete|evento|"
    r"estudo|estudos|estudar|criado|criada|removido|removida|aprovado|"
    r"aprovados|rejeitado|descartado|descartados|guardei|desfiz|dormir|"
    r"nota|notas|fila|enviado|cancelado|"
    # s63: lease replies ("Internet desligada." / "Internet liberada até…").
    # The server announces the language since C8 and this list is meant to
    # be frozen — but the live incident of 2026-06-10 was exactly a stale
    # pre-C8 listener voicing "Internet desligada." in EN via this
    # heuristic, so the fallback keeps the lease vocabulary too.
    r"desligada|desligado|liberada|liberado)\b",
    re.IGNORECASE,
)


def choose_voice_lang(
    reply_text: str,
    stt_lang: "Optional[str]",
    server_lang: "Optional[str]" = None,
) -> str:
    """Pick the TTS voice language for a reply.

    `server_lang` is the language the SSE stream announced (s62, C8): the
    orchestrator fixes `_reply_lang` at the start of the turn and every
    deterministic reply is written in it, so when the server says the
    language we BELIEVE it — it knows the words it generated better than
    any client-side guess. Exception (s76): on the streaming CHAT path the
    announcement predates generation and the 8B can disobey the language
    directive, so positive PT evidence in the text overrides an "en"
    announcement (never the reverse). The heuristics below survive only as
    a fallback for a server that predates the `language` frame.

    Fallback: the voice must match the WORDS being spoken — so we detect
    from the GENERATED reply (accents OR common PT words), NOT from what
    the user said. The 8B doesn't reliably mirror the user's language and
    Whisper mislabels accent-less PT as EN; trusting `stt_lang` for the
    voice produced an English voice reading Portuguese words (live bug,
    s53). `stt_lang` is used only as a last resort when the reply is empty
    (no words to detect from) — it never overrides a clear detection, so
    an English reply is never voiced in PT."""
    if server_lang in _STT_ACCEPTED_LANGS:
        # …unless the words contradict it (s76 smoke): on the pure-chat
        # streaming path the server announces the language of the INPUT,
        # fixed before the LLM generates a single token — and the 8B
        # disobeys the language directive when the memory context is
        # PT-heavy, so an "en" turn can stream PT text into the alba
        # voice. PT evidence in the text (accents / PT-specific words)
        # is near-proof — EN text never carries ã/ç and the hint list is
        # PT-distinct by construction — so it outvotes the announcement.
        # One direction only: absence of PT evidence is NOT proof of EN,
        # so a "pt" announcement is never overridden. Known trade-off: a
        # legit EN reply quoting an accented PT name ("Playing Matuê…")
        # flips to the PT voice — one short sentence with the wrong
        # accent, far cheaper than whole PT paragraphs in the EN voice.
        t = (reply_text or "").strip()
        if (server_lang == "en" and t
                and (detect_reply_language(t) == "pt"
                     or _PT_VOICE_HINT_RE.search(t))):
            return "pt"
        return server_lang
    t = (reply_text or "").strip()
    if not t:
        return stt_lang if stt_lang in _STT_ACCEPTED_LANGS else "en"
    if detect_reply_language(t) == "pt" or _PT_VOICE_HINT_RE.search(t):
        return "pt"
    return "en"


# Fixed error phrases spoken WITHOUT the server's help (M5-A3). They must
# never depend on the API that just failed — synthesised locally via the
# same piper/say path as normal replies. PT default: this daemon runs in
# The user's PT-BR household; a known stt_lang/server_lang refines it.
_API_ERROR_PHRASES = {
    "pt": "Não consegui falar com o servidor do Cortex.",
    "en": "I couldn't reach the Cortex server.",
}
_SERVER_ERROR_PHRASES = {
    "pt": "Deu um erro aqui no servidor. Tenta de novo.",
    "en": "Something went wrong on the server. Please try again.",
}
# M5-A6: spoken when the HARD recording ceiling cut the user off
# mid-speech. The truncated command is DISCARDED on purpose — a half
# email body reaching staging is approvable by a distracted "sim".
_TRUNCATED_PHRASES = {
    "pt": "Acho que te cortei no meio — repete mais curto, por favor?",
    "en": "I think I cut you off — could you repeat that more briefly?",
}


# Spoken "still working" filler (s82). By voice, silence is indistinguishable
# from a crash. the user approved turning the internet on and then heard NOTHING
# for 26 seconds while the search ran — the audit shows the turn was healthy
# the whole time (AUTO_SEARCH, replied at +33s), but there was no way for him
# to know that. The ONLINE_RAG path is p95 ~47s on the 8B, so this is the
# normal case, not an incident.
#
# Deliberately local-only and content-free: it says "still here", never
# guesses WHAT is taking long (the voice path has no status feed — the
# orchestrator's _emit_status is installed only by the UI). Claiming
# "searching the internet" when the delay was something else would be worse
# than saying nothing.
THINKING_FILLER_S = _env_float("CORTEX_THINKING_FILLER_S", 5.0)

# The longest one playback (a sentence, a local phrase) may take. Every wait
# on afplay/say was unbounded: one that never exits held the speech queue for
# good — and, as the filler, the watcher's join kept the reply from starting
# (s88, Codex lote 12 — invariant 7). Past it the process is stopped.
PLAYBACK_MAX_S = _env_float("CORTEX_PLAYBACK_MAX_S", 120.0)


def _wait_playback(proc, max_s: Optional[float] = None) -> None:
    """Wait for a playback process for at most PLAYBACK_MAX_S, then stop it
    (terminate, kill if that is ignored)."""
    limit = PLAYBACK_MAX_S if max_s is None else max_s
    try:
        proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        print(f"  [tts: playback still running after {limit:.0f}s — stopped]",
              flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
_THINKING_FILLER = {
    "pt": "Só um momento.",
    "en": "One moment.",
}


def _speak_local_phrase(phrases: dict, lang_hint: Optional[str]) -> None:
    """Speak a canned phrase using ONLY local TTS (piper/say) — never the
    API. Errors are swallowed: a failed warning must not crash the wake
    loop. PT default: this daemon runs in Pedro's PT-BR household."""
    lang = lang_hint if lang_hint in _STT_ACCEPTED_LANGS else "pt"
    phrase = phrases[lang]
    # A "cala" raised during this turn holds for these phrases too (s88,
    # Codex lote 12): the wake clears any older flag, so one that is up now
    # was asked for in this turn. Non-destructive — the read loop consumes it.
    if os.path.exists(SILENCE_FLAG):
        return
    try:
        proc_args: Optional[list[str]] = None
        if TTS_CONFIG.engine == "piper":
            wav_path = _next_tts_wav()
            if _synth_piper(TTS_CONFIG.piper_model_for(lang), phrase, wav_path):
                proc_args = ["afplay", wav_path]
        if proc_args is None:
            proc_args = ["say", "-v", TTS_CONFIG.say_voice_for(lang), phrase]
        _wait_playback(subprocess.Popen(proc_args))
    except Exception:
        pass


def speak_offline_error(lang_hint: Optional[str]) -> None:
    """Speak the canned 'couldn't reach the server' phrase using ONLY
    local TTS. By voice, an unspoken error is indistinguishable from a
    missed wake word — the user talks to the air and gets silence, the
    worst voice failure mode (M5-A3)."""
    _speak_local_phrase(_API_ERROR_PHRASES, lang_hint)


def speak_truncation_warning(lang_hint: Optional[str]) -> None:
    """Tell the user the recorder's hard ceiling cut them off (M5-A6)."""
    _speak_local_phrase(_TRUNCATED_PHRASES, lang_hint)


# Wake acknowledgement (s76): a short chime the INSTANT the wake word fires,
# so the user KNOWS they were heard before they start speaking — the missing
# "I'm listening" cue (the Siri ding). It plays BEFORE recording starts (serial,
# blocking) so it can't leak into the mic, which is why a SHORT sound matters:
# the chime's length is dead time before the recorder arms. Default "Blow"
# (the user picked it by ear). CORTEX_WAKE_ACK=chime|off toggles it; CORTEX_WAKE_CHIME
# overrides the file. Only on the primary wake (idle → "hey cortex"), not on
# barge-in, where cutting the reply is itself the feedback.
WAKE_ACK = (os.getenv("CORTEX_WAKE_ACK") or "chime").strip().lower()
WAKE_CHIME_PATH = (os.getenv("CORTEX_WAKE_CHIME")
                   or "/System/Library/Sounds/Blow.aiff")


def play_wake_chime():
    """START the wake-ack chime WITHOUT blocking, and return the afplay Popen
    so the caller can keep reading the mic while it plays. Returns None when the
    chime is off or afplay fails to launch (nothing to wait on). Fail-soft — a
    missing file / no afplay must never crash the wake loop."""
    if WAKE_ACK != "chime":
        return None
    try:
        return subprocess.Popen(["afplay", WAKE_CHIME_PATH])
    except Exception:
        return None


def acknowledge_wake(stream) -> "np.ndarray":
    """Chime that the wake word landed, RETURNING the audio captured while it
    played (the pre-roll) instead of discarding it.

    s82 — this used to throw those frames away, and the chime is 1.4 s long
    (Blow.aiff). Anyone who speaks straight through — "Hey Cortex, o que eu
    tenho para hoje…" — loses the first second and a half of the command.
    Pedro's live audit shows exactly that: "para hoje e qual o clima de
    amanhã", with "o que eu tenho" gone. The comment above the chime constant
    already called its length "dead time"; nobody had measured it against how
    a person actually talks.

    Keeping the audio is safe on both counts that motivated discarding it, and
    both were measured rather than assumed:
      - transcription: the chime echo mixed at 30% and 60% in front of a real
        clip transcribes IDENTICALLY to the clean clip;
      - false arming: Silero scores the chime 0.07-0.13, far below the 0.5
        that would arm the recorder.
    So the echo is harmless where it lands, and the words are not.

    The chime must NOT block the read loop. The original chime did
    `afplay … .wait()` (blocking, so it couldn't leak into the mic), which left
    ~1s with nobody reading the InputStream — that gap overflows PortAudio and
    desyncs the CoreAudio capture on some device states (post-sleep, mic
    re-indexed): the next reads come back silent and `record_after_wake` gives
    up. That was the intermittent "deaf right after the chime" (s76 cont.44; the
    stop/start added there only marginally recovered it — still flaky live,
    cont.45c). So instead of recovering AFTER the damage, we PREVENT it: play
    the chime non-blocking and keep reading+DISCARDING frames for its duration.
    The stream is consumed continuously (no overflow, no desync) and the chime
    echo is dropped, so `record_after_wake` starts on clean LIVE audio.

    No-op when the chime is off (or afplay failed to launch): nothing starved
    the stream, so the path stays byte-for-byte what it was before the chime.
    Fully fail-soft — a read hiccup here must never kill the turn."""
    proc = play_wake_chime()
    if proc is None:
        return np.empty(0, dtype=np.int16)
    # Each stream.read() blocks ~80 ms (one FRAME_SIZE), so this paces at the
    # audio clock and just shadows the chime — it does not busy-spin. The cap
    # is a safety valve so a hung afplay can never deafen the daemon.
    max_frames = int(2.0 * SAMPLE_RATE / FRAME_SIZE)
    kept = []
    try:
        n = 0
        while proc.poll() is None and n < max_frames:
            audio, _ = stream.read(FRAME_SIZE)
            kept.append(audio[:, 0])
            n += 1
        if proc.poll() is None:
            proc.terminate()  # past the cap: a hung afplay is not left behind
        # NB: no drain() here any more. Draining discarded the buffered
        # backlog — which is precisely where the opening words sit.
    except Exception:
        pass
    return np.concatenate(kept) if kept else np.empty(0, dtype=np.int16)


def _clear_stale_silence_flag() -> None:
    """A wake-initiated turn proves any earlier shush no longer applies:
    clear a leftover silence flag so it can't mute this turn. The flag is
    raised unconditionally by cortex-shush.sh even when nothing is playing,
    and nothing else cleans it — an orphan flag would abort the next turn
    on its first SSE check, hours later, with no audible feedback (M5-A2)."""
    try:
        os.remove(SILENCE_FLAG)
    except OSError:
        pass


def stream_and_speak(
    user_input: str,
    telem: Optional[VoiceTelemetry] = None,
    stream=None,
    wake=None,
    stt_lang: Optional[str] = None,
) -> tuple[str, str]:
    """
    POST to /ask_stream, parse SSE, and fire `say` per sentence as chunks
    arrive. Returns (status_mode, full_reply_text).

    Voice is chosen once on the first completed sentence (or after the
    full reply if nothing has a sentence boundary). Subsequent `say`
    invocations serialise via Popen.wait() so they don't overlap.

    `telem` (optional) marks `thinking_done()` the instant the first audio
    plays — the boundary between "pensando" and "respondendo".

    Voice barge-in: when `stream` + `wake` are passed, the mic is read
    while a sentence plays and the wake model runs on it. Saying the wake
    word ("Hey Cortex") again mid-reply kills the speech and returns
    status "BARGE_IN", so the caller can immediately capture the new
    command. The wake word is robust against the playback echo because
    Cortex never speaks its own wake word, so the reply audio can't
    self-trigger. Without stream/wake (e.g. tests) it degrades to a plain
    blocking wait — no barge-in, same as before.
    """
    headers = {"X-Cortex-Key": API_KEY, "Content-Type": "application/json"}
    # Pass the STT-detected language to the server. Whisper already knows
    # what language the user spoke; without this the server re-guesses from
    # the text via an accent heuristic, which misses accent-less PT ("toca
    # musica", "que dia e hoje") and replies in English to a PT question
    # (live bug). The server canonicalises/validates it (_normalise_language).
    body = {"user_input": user_input}
    if stt_lang in _STT_ACCEPTED_LANGS:
        body["language"] = stt_lang

    lang: Optional[str] = None
    # Language announced by the server's SSE prelude (s62, C8). When
    # present it is authoritative for the TTS voice — the orchestrator
    # fixed `_reply_lang` for the turn and wrote every deterministic reply
    # in it. The old client-side heuristics stay as fallback only.
    server_lang: Optional[str] = None
    buf = ""
    full_text = ""
    status_mode = "OFFLINE"
    say_proc: Optional[subprocess.Popen] = None
    spoke_once = False
    barged = False

    # Clear the wake model's activation buffer before we start listening
    # during playback. Without this, the residual activation from the
    # initial "Hey Cortex" (that started this turn) is still in the buffer
    # and re-fires the instant _wait_say resumes predict() — a false
    # barge-in that cuts the reply ~immediately. Reset = start listening
    # clean; a genuine new "Hey Cortex" re-accumulates from scratch.
    if wake is not None:
        wake.reset()

    # "Thinking" barge-in: while we wait for the LLM's first token, the
    # main loop is blocked on r.iter_lines() and nobody reads the mic — so
    # the user couldn't interrupt until speech started. This background
    # watcher reads the mic + runs the wake model DURING that wait, setting
    # `barged` if it hears "Hey Cortex". It stops the instant the first
    # sentence plays (then _wait_say takes over the mic — they never read
    # the stream concurrently). Only armed when barge-in is wired.
    _think_watch_stop = threading.Event()
    # Holds the live streaming response so the watcher can abort the HTTP
    # read the instant a barge is heard (the main loop is otherwise blocked
    # in iter_lines() waiting for the first token — closing the response
    # makes that read raise, so we break out immediately instead of waiting
    # for the server to finally send something).
    _resp_box: dict = {"r": None}

    def _thinking_watcher() -> None:
        nonlocal barged
        # Frames tick at the audio clock (~80 ms each), so counting them is a
        # clock that costs nothing and needs no extra thread.
        frames_seen = 0
        filler_frames = int(THINKING_FILLER_S * SAMPLE_RATE / FRAME_SIZE)
        filler_said = False
        while not _think_watch_stop.is_set():
            try:
                audio, _ = stream.read(FRAME_SIZE)
            except Exception:
                return  # mic read failed — let the main path proceed
            frames_seen += 1
            if (not filler_said and THINKING_FILLER_S > 0
                    and frames_seen >= filler_frames):
                filler_said = True
                if os.path.exists(SILENCE_FLAG):
                    continue  # muted while thinking: no "Só um momento"
                # Logged so "did it actually say something?" is answerable
                # from the audit instead of from memory.
                try:
                    from src.core.audit import audit
                    audit.log("–", "voice", "thinking_filler",
                              payload={"after_s": THINKING_FILLER_S})
                except Exception:  # noqa: BLE001
                    pass
                # Spoken from THIS thread on purpose: the watcher owns the mic
                # until the first real sentence stops it, so there is no other
                # thread to collide with over the audio device. Blocking for
                # the ~1 s of the filler costs barge-in during that second —
                # a fair trade for not leaving the user in silence.
                _speak_local_phrase(_THINKING_FILLER, stt_lang)
                continue
            mono = audio[:, 0]
            score = float(wake.predict(mono).get(WAKE_CONFIG.lookup_key, 0.0))
            if score >= BARGE_THRESHOLD:
                barged = True
                print(f"  [barge-in while thinking: wake score={score:.2f}]")
                r = _resp_box.get("r")
                if r is not None:
                    try:
                        r.close()  # unblock the main iter_lines() read
                    except Exception:
                        pass
                return

    _think_thread: Optional[threading.Thread] = None
    if stream is not None and wake is not None:
        _think_thread = threading.Thread(target=_thinking_watcher, daemon=True)
        _think_thread.start()

    def _stop_thinking_watcher() -> None:
        """Stop the thinking-phase mic watcher and reclaim the stream before
        any _speak/_wait_say reads it. Idempotent."""
        nonlocal _think_thread
        if _think_thread is not None:
            _think_watch_stop.set()
            _think_thread.join()
            _think_thread = None
            # The watcher accumulated activation in the wake model; reset so
            # the playback-phase barge-in (_wait_say) starts clean.
            if wake is not None:
                wake.reset()

    def _wait_say() -> None:
        """Block until the current `say` finishes, OR cut it short if the
        user barges in with the wake word. Sets `barged` + terminates the
        speech on a barge. Falls back to a plain wait when barge-in isn't
        wired (no stream/wake)."""
        nonlocal barged
        if say_proc is None:
            return
        if stream is None or wake is None:
            _wait_playback(say_proc)
            return
        deadline = time.monotonic() + PLAYBACK_MAX_S
        while say_proc.poll() is None:
            if time.monotonic() > deadline:
                print(f"  [tts: playback still running after "
                      f"{PLAYBACK_MAX_S:.0f}s — stopped]", flush=True)
                say_proc.terminate()
                return
            # The mute, here too (s88, Codex lote 12): the read loop checks
            # the flag only between SSE lines, and after the LAST line this
            # wait is all that runs — the final sentence played to its end.
            # Non-destructive; the caller consumes the flag.
            if os.path.exists(SILENCE_FLAG):
                say_proc.terminate()
                return
            try:
                audio, _ = stream.read(FRAME_SIZE)
            except Exception:
                _wait_playback(say_proc)  # mic read failed — don't hang the reply
                return
            mono = audio[:, 0]
            score = float(wake.predict(mono).get(WAKE_CONFIG.lookup_key, 0.0))
            if score >= BARGE_THRESHOLD:
                barged = True
                say_proc.terminate()
                print(f"  [barge-in: wake score={score:.2f}]")
                return

    def _speak(sentence: str) -> None:
        nonlocal say_proc, spoke_once, lang
        # First spoken sentence ends the "thinking" phase: stop the
        # thinking-watcher so it stops reading the mic before _wait_say
        # (below) starts reading it — they must never touch the stream
        # concurrently. No-op after the first call.
        _stop_thinking_watcher()
        if barged:
            return  # thinking-watcher heard the wake word — don't speak
        # Honour a stop raised mid-reply (cortex-shush / cx cala): don't
        # start a NEW sentence while the silence flag is up. The main loop
        # consumes the flag and breaks; this guard (non-destructive check,
        # so the loop still sees the flag) closes the window between
        # `killall say` cutting the current sentence and the loop's next
        # check — without it the next buffered sentence would still fire.
        if os.path.exists(SILENCE_FLAG):
            return
        speech_lang = lang or choose_voice_lang(full_text or sentence, stt_lang, server_lang)
        cleaned = strip_markdown_for_speech(sentence, language=speech_lang).strip()
        if not cleaned:
            return
        # Language is decided once, on the first spoken sentence. The voice
        # must match the WORDS being read, so we detect from the reply text
        # (not stt_lang): the 8B doesn't reliably mirror the user's language
        # and Whisper mislabels accent-less PT as EN, which used to give an
        # EN voice reading PT words (s53 bug). See choose_voice_lang.
        if lang is None:
            lang = speech_lang
            print(f"  tts <{lang}> via {TTS_CONFIG.engine}")
            # Record all three so "falei em inglês e respondeu em português"
            # is diagnosable from the audit next time (s82): stt vs what the
            # server announced vs what the TTS actually read.
            if telem is not None:
                telem.languages(stt=stt_lang, server=server_lang, spoke=lang)
        # Double-buffer (M5-A8): synthesise THIS sentence into the spare
        # WAV BEFORE waiting on the previous afplay, so synth overlaps
        # playback and the sentence boundary loses its audible gap. The
        # barge-in / silence-flag / serialisation machinery is unchanged —
        # only the WAV path alternates (playback stays serial via
        # _wait_say, so two buffers suffice). A Piper synth failure
        # degrades to `say` for this sentence only. On a barge during
        # _wait_say the prepared WAV is simply never played.
        proc_args = prepare_tts(
            cleaned, lang, TTS_CONFIG,
            _next_tts_wav() if TTS_CONFIG.engine == "piper" else "",
            synth=_synth_piper,
        )
        if say_proc is not None:
            _wait_say()
        if barged:
            return  # don't start a new sentence after a barge-in
        # The shush needs the SAME re-check the barge gets. _wait_say above
        # blocks for as long as the previous sentence takes to play — which
        # is exactly when the user reaches for the hotkey. Checking the flag
        # only at the top of _speak meant the already-synthesised next
        # sentence started anyway: the kill cut the audio, and one more
        # sentence spoke over the "stop". Non-destructive, like the top
        # guard, so the read loop still consumes the flag and reports
        # SPEECH_INTERRUPTED.
        if os.path.exists(SILENCE_FLAG):
            return
        say_proc = subprocess.Popen(proc_args)
        if not spoke_once:
            spoke_once = True
            if telem is not None:
                telem.thinking_done()  # first audio → end of "thinking"

    interrupted = False
    # The watcher above is already reading the mic. A refused connection or
    # an HTTP 500 raises before the try/finally below that stops it, and it
    # used to stay alive: reading the stream against the wake loop, sharing
    # the wake model, and saying "Só um momento" after the error phrase
    # (s88, Codex lote 12 — invariant 7). Stop it before the error leaves.
    r = None
    try:
        r = requests.post(
            f"{API_URL}/ask_stream",
            json=body,
            headers=headers,
            stream=True,
            timeout=API_TIMEOUT_S,
        )
        r.raise_for_status()
    except BaseException:
        _stop_thinking_watcher()
        if r is not None:
            r.close()
        raise
    with r:
        # M5-A10a (reviewed, no code change needed): a barge landing in the
        # window before this assignment can't close `r` via the watcher —
        # but the read loop below checks `barged` BEFORE every next(), so
        # that barge still breaks immediately, without waiting for a token.
        _resp_box["r"] = r  # let the thinking-watcher abort this read on barge
        try:
            line_iter = r.iter_lines()
            while True:
                if barged:
                    break  # thinking-watcher heard the wake word
                try:
                    raw = next(line_iter)
                except StopIteration:
                    break
                except Exception:
                    # The watcher closed the response on a barge — that's
                    # the expected unblock path; otherwise re-raise.
                    if barged:
                        break
                    raise
                if barged:
                    break  # wake word heard mid-reply (set inside _speak)
                if _silence_requested():
                    interrupted = True
                    status_mode = "SPEECH_INTERRUPTED"
                    if say_proc is not None and say_proc.poll() is None:
                        say_proc.terminate()
                    print("  [interrupted by cx cala]")
                    break

                if not raw or not raw.startswith(b"data: "):
                    continue
                try:
                    payload = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue

                # Language frame (s62, C8): the prelude arrives before the
                # first chunk; done/error frames repeat it. Capture from
                # any frame — the first sentence hasn't picked a voice yet.
                _srv = payload.get("language")
                if _srv in _STT_ACCEPTED_LANGS:
                    server_lang = _srv

                if payload.get("done"):
                    status_mode = payload.get("status_mode") or "OFFLINE"
                    if buf.strip():
                        _speak(buf)
                        buf = ""
                    # Final text may correct a complete live reply. Keep the
                    # authoritative transcript; speech already played stays.
                    if isinstance(payload.get("reply"), str):
                        full_text = payload["reply"]
                    break

                if "error" in payload:
                    # An error must be HEARD, not become silence (M5-A3):
                    # by voice, a silent failure is indistinguishable from
                    # a missed wake word. Prefer the server's localised
                    # speakable `reply`; fall back to the fixed local
                    # phrase for servers predating the C8 contract.
                    print(f"  stream error: {payload['error']}")
                    status_mode = "INTERNAL_ERROR"
                    spoken = payload.get("reply") or _SERVER_ERROR_PHRASES[
                        server_lang
                        or (stt_lang if stt_lang in _STT_ACCEPTED_LANGS else "pt")
                    ]
                    _speak(spoken)
                    break

                chunk = payload.get("chunk", "")
                if not chunk:
                    continue
                buf += chunk
                full_text += chunk

                while True:
                    if barged:
                        break  # stop splitting/speaking once the user barged in
                    # Hard sentence boundary takes priority — gives the most
                    # natural prosody when available.
                    m = _SENTENCE_BOUNDARY_RE.search(buf)
                    if m:
                        end = m.end()
                        piece = buf[:end]
                        buf = buf[end:]
                        _speak(piece)
                        continue
                    # No sentence boundary yet. If the buffer has grown past
                    # the soft threshold, cut at the LAST comma/colon/dash so
                    # the user starts hearing audio without waiting for the
                    # full sentence to complete.
                    if len(buf) >= _MIN_SOFT_SPEAK_CHARS:
                        matches = list(_SOFT_BREAK_RE.finditer(buf))
                        if matches:
                            cut = matches[-1].end()
                            piece = buf[:cut]
                            buf = buf[cut:]
                            _speak(piece)
                            continue
                    break
        finally:
            # Always tear down the thinking-watcher, on every exit path
            # (normal end, barge, silence, error, exception) — so it never
            # leaks a thread holding the mic into the next turn.
            _stop_thinking_watcher()

    # The final sentence is also barge-able (user can cut the tail off).
    if not interrupted and not barged and say_proc is not None:
        _wait_say()
        if _silence_requested():
            status_mode = "SPEECH_INTERRUPTED"

    if barged:
        status_mode = "BARGE_IN"

    return status_mode, full_text


def drain(stream) -> None:
    n = stream.read_available
    if n and n > 0:
        stream.read(n)


def handle_utterance(
    text: str,
    stream,
    whisper: WhisperModel,
    wake=None,
    telem: Optional[VoiceTelemetry] = None,
    stt_lang: Optional[str] = None,
    verifier: Optional[SpeakerVerifier] = None,
) -> None:
    """Stream from /ask_stream, speak sentence-by-sentence, loop for follow-ups.

    `wake` enables voice barge-in: saying the wake word during playback
    cuts the reply and immediately records the next command (no need to
    re-say the wake word).

    `stt_lang` is the language Whisper detected for the user's speech; it's
    forwarded to the server so the reply is in the same language (and picks
    the right TTS voice). Re-transcriptions below (barge / follow-up) refresh
    it from the new utterance."""
    while True:
        t0 = time.time()
        try:
            status, full_text = stream_and_speak(
                text, telem=telem, stream=stream, wake=wake, stt_lang=stt_lang
            )
        except requests.RequestException as e:
            # Covers connection refused/timeout AND raise_for_status (the
            # M4-A1 class: flaky network → HTTP 500 on "pausa"). Say so out
            # loud — local TTS only, the API just proved unreachable (M5-A3).
            print(f"  API error: {e}\n")
            speak_offline_error(stt_lang)
            return
        elapsed = time.time() - t0
        if telem is not None:
            telem.replied(status_mode=status)  # latency = time spent speaking
        # flush=True: stdout is redirected to logs/voice.out.log, so without it
        # this line sits in the buffer for hours — and it is the only record of
        # WHAT Cortex actually said. Asked to review two spoken answers (s82),
        # the text turned out to be unreadable everywhere: not in the audit (by
        # design — that is observability, not content) and stuck in this buffer.
        # Masked (s88, Codex lote 12): this log took every transcript and
        # reply whole, while the audit masks the same text (telemetry.py).
        print(f"  [stream done in {elapsed:.1f}s, status={status}] "
              f"{redact_text(full_text)[:120]!r}", flush=True)

        if status == "BARGE_IN":
            # Wake word heard mid-reply: speech is already cut. Capture the
            # new command straight away — the user is mid-sentence, so no
            # second wake word needed. Reuses record → transcribe → loop.
            print("  barge-in → ouvindo novo comando...")
            if wake is not None:
                wake.reset()
            drain(stream)
            pcm, truncated = record_after_wake(stream)
            if pcm.size == 0:
                print("  barge-in sem fala — voltando ao wake.\n")
                return
            dur = len(pcm) / SAMPLE_RATE
            print(f"  recorded {dur:.1f}s, transcribing...")  # M5-A10b: dur era calculado e não mostrado
            speaker_shadow_check(pcm, telem, verifier, "barge_in")
            text, lang, lang_prob = transcribe(pcm, whisper)
            if telem is not None:
                telem.transcribed_extra("barge_in", text, lang, lang_prob)
            print(f"  <{lang} p={lang_prob:.2f}>  {redact_text(text)!r}")
            if truncated:
                # M5-A6: command cut at the hard ceiling — warn and drop,
                # never ship half a command to staging.
                print("  truncated at hard cap — asking to repeat.\n")
                speak_truncation_warning(lang)
                return
            if len(text.strip()) < 2:
                print("  barge-in sem comando — voltando ao wake.\n")
                return
            stt_lang = lang  # new command may be in a different language
            continue  # re-enter with the new command

        if status not in FOLLOWUP_STATUSES:
            print("  done.\n")
            return

        print(f"  awaiting follow-up ({FOLLOWUP_WAIT_S:.0f}s, no wake needed)...")
        drain(stream)
        pcm, truncated = listen_followup(stream)
        if pcm is None:
            print("  no follow-up heard.\n")
            return
        dur = len(pcm) / SAMPLE_RATE
        print(f"  recorded {dur:.1f}s follow-up, transcribing...")
        # The follow-up is the security-relevant capture (this is where a
        # spoken "sim"/"pode mandar" approves a staged S3) AND the shortest
        # one — shadow must measure whether sub-second approvals even
        # produce a usable score before Fase B can be designed.
        speaker_shadow_check(pcm, telem, verifier, "followup")
        text, lang, lang_prob = transcribe(pcm, whisper)
        if telem is not None:
            telem.transcribed_extra("followup", text, lang, lang_prob)
        print(f"  <{lang} p={lang_prob:.2f}>  {redact_text(text)!r}")
        if truncated:
            print("  follow-up truncated at hard cap — asking to repeat.\n")
            speak_truncation_warning(lang)
            return
        if len(text.strip()) < 2:
            print("  empty follow-up.\n")
            return
        stt_lang = lang  # follow-up may be in a different language
        # loop: re-call /ask_stream with this follow-up


def main() -> None:
    if not API_KEY:
        sys.exit("ERROR: CORTEX_API_KEY not set in .env")

    print("Loading wake-word model...")
    # Force ONNX runtime explicitly. openwakeword 0.6.0 on Python 3.12 hard-
    # fails on missing tflite-runtime instead of falling back automatically
    # (regression vs Python 3.9 where it was just a warning). We have
    # onnxruntime + the .onnx variant of each model already on disk.
    wake = WakeModel(wakeword_models=[WAKE_CONFIG.model_spec], inference_framework="onnx")

    print(f"Loading Whisper ({WHISPER_MODEL_SIZE}, int8 CPU)...")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    # Pipeline telemetry → audit log (stage="voice"). Fail-soft: never
    # crashes the loop. Surfaces on the /ui "Voz" panel.
    telem = VoiceTelemetry()

    # Speaker-ID shadow (Onda 12 Fase A). Observation only — scores every
    # captured utterance against the enrolled voiceprint and logs it; no
    # behaviour depends on the result. Warm the ONNX model at boot (one
    # throwaway embed) so the ~0.3s load never lands on the first turn;
    # missing model/voiceprint degrade to "unavailable" events.
    verifier = SpeakerVerifier()
    if verifier.is_available():
        verifier.embed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        print(f"Speaker : shadow ON (voiceprint ok, "
              f"threshold ref={verifier.threshold})")
    else:
        print("Speaker : shadow UNAVAILABLE (model/voiceprint missing — "
              "see scripts/enroll_speaker.py)")

    mic_index = find_macbook_mic()
    mic_name = sd.query_devices(mic_index)["name"]
    print(f"Mic     : [{mic_index}] {mic_name}")
    print(f"API     : {API_URL}")
    print(
        f"Wake    : '{WAKE_CONFIG.lookup_key}' ({WAKE_CONFIG.source})  "
        f"threshold={WAKE_THRESHOLD}"
    )
    if WAKE_ACK == "chime":
        print(f"Ack     : chime → {WAKE_CHIME_PATH}")
    else:
        print("Ack     : off (CORTEX_WAKE_ACK)")
    if TTS_CONFIG.engine == "piper":
        print(f"TTS     : piper ({TTS_CONFIG.source})")
        print(f"  PT    : {TTS_CONFIG.piper_model_pt}")
        print(f"  EN    : {TTS_CONFIG.piper_model_en}")
        # Warm the PT voice now (M5-A8): boot pays the ~0.7s model load
        # instead of the first spoken turn.
        preload_piper_voices()
    else:
        print(f"TTS     : say ({TTS_CONFIG.source})")
        print(f"  PT    : '{TTS_CONFIG.say_voice_pt}'")
        print(f"  EN    : '{TTS_CONFIG.say_voice_en}'")
    print(f"Barge-in: {'on' if BARGE_IN_ENABLED else 'off'} "
          f"(threshold={BARGE_THRESHOLD}; say '{WAKE_CONFIG.phrase_hint}' mid-reply)")
    if STREAM_WATCHDOG_ENABLED:
        print(f"Watchdog: reopen mic after {STREAM_DEAD_S:.0f}s of silence "
              f"(CORTEX_STREAM_WATCHDOG)")
    else:
        print("Watchdog: off (CORTEX_STREAM_WATCHDOG)")
    print(f"Say '{WAKE_CONFIG.phrase_hint}' followed by your question. Ctrl+C to exit.\n")

    last_wake = 0.0
    tick = 0

    stream = open_mic_stream(mic_index)
    watchdog = StreamWatchdog()
    try:
        # Startup self-test: confirm audio is actually flowing. If macOS is
        # silently denying the mic (common with launchd-spawned processes
        # under TCC), the stream opens but returns all zeros.
        print("Audio self-test: sampling 2s of mic input...", flush=True)
        test_rms_max = 0.0
        test_rms_sum = 0.0
        test_frames = 25  # ~2 seconds at 80 ms/frame
        for _ in range(test_frames):
            audio, _ = stream.read(FRAME_SIZE)
            r = rms(audio[:, 0])
            test_rms_max = max(test_rms_max, r)
            test_rms_sum += r
        test_rms_avg = test_rms_sum / test_frames
        print(
            f"Audio self-test: max_rms={test_rms_max:.1f}, "
            f"avg_rms={test_rms_avg:.1f}",
            flush=True,
        )
        if test_rms_max < 5.0:
            print(
                "WARNING: mic stream appears MUTED (max RMS < 5). "
                "macOS likely blocking the launchd-spawned process from "
                "the microphone (separate TCC context from Terminal). "
                "Grant access via System Settings -> Privacy & Security "
                "-> Microphone, then reload the daemon.",
                flush=True,
            )

        # Calibrate the speech threshold against THIS mic in THIS room (s82).
        # Runs right after the self-test, while the room is still quiet and
        # nobody is talking to the daemon yet — the ideal moment to measure a
        # noise floor. No-op when disabled or when .env pins the threshold.
        calibrate_threshold(stream)

        while True:
            audio, _ = stream.read(FRAME_SIZE)
            mono = audio[:, 0]
            # Auto-recover a stale stream (s76 cont.45): sustained dead silence
            # means a sleep/wake left the capture desynced — reopen the mic by
            # name so the daemon recovers on its own instead of going deaf until
            # a manual restart. A real wake is loud, so it resets the run first.
            if STREAM_WATCHDOG_ENABLED and watchdog.observe(rms(mono)):
                print(" " * 60, end="\r")
                print(f"  [no live audio for {STREAM_DEAD_S:.0f}s — capture "
                      f"stream looks dead; reopening mic...]", flush=True)
                stream = recover_mic_stream(stream)
                wake.reset()
                watchdog.reset()   # back off ~STREAM_DEAD_S before retrying
                continue
            score = float(wake.predict(mono).get(WAKE_CONFIG.lookup_key, 0.0))
            tick += 1

            now = time.time()
            if score >= WAKE_THRESHOLD and (now - last_wake) >= WAKE_COOLDOWN_S:
                last_wake = now
                telem.wake(score)
                _clear_stale_silence_flag()  # M5-A2: new turn, old shush is moot
                print(" " * 60, end="\r")
                print(f"WAKE  score={score:.2f}  listening...")
                # Chime "I heard you", then hand a clean stream to the recorder
                # (the blocking chime can otherwise leave it reading silence).
                preroll = acknowledge_wake(stream)

                pcm, truncated = record_after_wake(stream, preroll=preroll)
                if pcm.size == 0:
                    # M5-A7: wake fired but nobody spoke — Whisper never
                    # runs on silence; back to the wake loop quietly.
                    print("  sem fala após o wake — voltando.\n")
                    telem.session_done()
                    drain(stream)
                    wake.reset()
                    continue
                dur = len(pcm) / SAMPLE_RATE
                telem.listened(dur)
                print(f"  recorded {dur:.1f}s, transcribing...")
                speaker_shadow_check(pcm, telem, verifier, "command")

                text, lang, lang_prob = transcribe(pcm, whisper)
                telem.transcribed(text, lang, lang_prob)
                print(f"  <{lang} p={lang_prob:.2f}>  {redact_text(text)!r}")

                if truncated:
                    # M5-A6: hard ceiling cut the dictation — warn and drop
                    # (half an email body must never reach staging).
                    print("  truncated at hard cap — asking to repeat.\n")
                    speak_truncation_warning(lang)
                elif len(text.strip()) >= 2:
                    handle_utterance(
                        text, stream, whisper,
                        wake=wake if BARGE_IN_ENABLED else None,
                        telem=telem,
                        stt_lang=lang,
                        verifier=verifier,
                    )
                else:
                    print("  empty transcription, skipping.\n")

                telem.session_done()
                drain(stream)
                wake.reset()
            elif tick % 12 == 0 and not DAEMON_MODE:
                bar = "#" * int(score * 20)
                print(f"  score={score:.2f}  {bar:<20}", end="\r", flush=True)
    finally:
        # Was an `with sd.InputStream(...)`; manual lifecycle (so the watchdog
        # can swap `stream`) means we own teardown. Fail-soft.
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
