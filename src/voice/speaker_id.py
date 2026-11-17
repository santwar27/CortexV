"""
Speaker identity — local voice verification (Onda 12, shadow-first).

Closes the one identity gap in the security model: the API has a key,
the /ui has a cookie, but the VOICE surface — the one that can approve
a staged S3 action by saying "pode mandar" — has no idea WHO is
speaking. This module turns the audio buffer that already goes to
Whisper into a `speaker score`: cosine similarity between a small
speaker-embedding model's output and Pedro's enrolled voiceprint.

Sovereignty: reinforces red line 2 (audio never leaves the machine —
the model is a local ONNX, ~40ms per utterance on the M1) and red
line 4 (the eventual gate rule is `score >= threshold`, a narrow
deterministic predicate, never an LLM verdict).

Rollout is the house pattern (Arbiter precedent): Fase 0 = enrollment
+ threshold lab with real voice; Fase A = SHADOW (score logged on
every voice turn, nothing enforced); Fase B = enforcement on S3
approvals / lease grants only, after shadow data sizes the false
negative rate. This module is the instrument; it enforces nothing by
itself.

Privacy of the voiceprint: it lives OUTSIDE the repo
(~/.cortex/voiceprint.npz, chmod 600 — written by
scripts/enroll_speaker.py). An embedding is not reversible to audio,
but it is still personal data: it never ships in git.

Honest limits (documented for the day enforcement lands): a replay
attack (a good recording of Pedro) passes — this protects against
casual misuse by whoever is in the room, not a determined adversary.
A hoarse/sick voice may score low — which is why enforcement keeps a
non-voice fallback (/ui) and the threshold is env-tunable.

Everything here is FAIL-SOFT: any failure returns None rather than
raising, because the call site is the voice hot path — a broken
speaker model must degrade to "score unavailable" (shadow logs it),
never to a dead listener.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

# Model + voiceprint live under ~/.cortex (outside the repo): the model
# because it's a 28MB binary, the voiceprint because it's personal data.
DEFAULT_MODEL_PATH = "~/.cortex/models/speaker_campplus_voxceleb.onnx"
DEFAULT_VOICEPRINT_PATH = "~/.cortex/voiceprint.npz"

# Pinned download for the model (3D-Speaker CAM++ trained on VoxCeleb,
# served from the sherpa-onnx release page). The lab script verifies the
# checksum before trusting a fresh download.
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"  # (sic — typo is upstream's tag name)
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)
MODEL_SHA256 = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"

# PLACEHOLDER until the Fase 0 lab measures real-voice separation —
# shadow mode never enforces, so this number gates nothing today.
_DEFAULT_THRESHOLD = 0.45

# Embeddings on sub-second snippets are garbage; below this the score is
# meaningless and we'd rather report "unavailable" than a random number.
MIN_AUDIO_SECONDS = 0.5

SAMPLE_RATE = 16000


def resolve_speaker_threshold(env_value: Optional[str] = None) -> float:
    """CORTEX_SPEAKER_THRESHOLD with the same degrade-to-default contract
    as resolve_wake_threshold: empty / non-numeric / out-of-range values
    fall back rather than silently disabling (0.0 accepts everyone) or
    bricking (>1 rejects everyone — cosine of L2-normalised vectors)."""
    raw = env_value if env_value is not None else os.environ.get(
        "CORTEX_SPEAKER_THRESHOLD", "")
    raw = (raw or "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_THRESHOLD
    if value <= 0.0 or value >= 1.0:
        return _DEFAULT_THRESHOLD
    return value


def resolve_model_path(env_value: Optional[str] = None) -> str:
    raw = env_value if env_value is not None else os.environ.get(
        "CORTEX_SPEAKER_MODEL_PATH", "")
    return os.path.expanduser((raw or "").strip() or DEFAULT_MODEL_PATH)


def resolve_voiceprint_path(env_value: Optional[str] = None) -> str:
    raw = env_value if env_value is not None else os.environ.get(
        "CORTEX_SPEAKER_VOICEPRINT", "")
    return os.path.expanduser((raw or "").strip() or DEFAULT_VOICEPRINT_PATH)


def normalise(vec: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalise so cosine similarity is a plain dot product. None for
    zero/NaN vectors (a degenerate embedding must not score ~1.0 against
    everything via numerical accident)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return v / norm


def compute_centroid(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Voiceprint = normalised mean of the (normalised) clip embeddings.
    Used by the enrollment script and the lab's leave-one-out pass."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    rows = [normalise(r) for r in arr]
    if any(r is None for r in rows):
        return None
    return normalise(np.mean(rows, axis=0))


def _default_extractor_factory(model_path: str):
    """Build the sherpa-onnx extractor. Imported lazily so the test suite
    (and any non-voice process that imports this module) never pays for —
    or depends on — the ONNX runtime."""
    import sherpa_onnx  # heavyweight: only in the voice daemon

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model_path, num_threads=2,
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


class SpeakerVerifier:
    """Scores an utterance against the enrolled voiceprint.

    `extractor_factory` is injectable for tests (a fake returning fixed
    vectors); production uses the lazy sherpa-onnx factory. The extractor
    is built on first use, not at construction — the listener creates the
    verifier at boot and must not block startup on a 0.3s model load
    (let alone crash on a missing model file)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        voiceprint_path: Optional[str] = None,
        extractor_factory: Callable = _default_extractor_factory,
    ):
        self.model_path = model_path or resolve_model_path()
        self.voiceprint_path = voiceprint_path or resolve_voiceprint_path()
        self.threshold = resolve_speaker_threshold()
        self._extractor_factory = extractor_factory
        self._extractor = None
        self._extractor_failed = False
        self._centroid: Optional[np.ndarray] = None
        self._centroid_loaded = False

    # ── availability ────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when both halves exist on disk: the model and an enrolled
        voiceprint. Shadow wiring logs `unavailable` when False instead
        of skipping silently — absence must be visible in the audit."""
        return os.path.isfile(self.model_path) and self._load_centroid() is not None

    # ── internals ───────────────────────────────────────────────────

    def _load_centroid(self) -> Optional[np.ndarray]:
        if not self._centroid_loaded:
            self._centroid_loaded = True
            try:
                with np.load(self.voiceprint_path) as data:
                    self._centroid = normalise(data["centroid"])
            except Exception:
                self._centroid = None
        return self._centroid

    def _get_extractor(self):
        if self._extractor is None and not self._extractor_failed:
            try:
                self._extractor = self._extractor_factory(self.model_path)
            except Exception as e:
                # One loud line, then permanent fail-soft: retrying a
                # missing/corrupt model on every utterance would add a
                # failed load to each voice turn.
                print(f"[speaker_id] extractor load failed: {e}")
                self._extractor_failed = True
        return self._extractor

    # ── scoring ─────────────────────────────────────────────────────

    def embed(self, samples: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """L2-normalised embedding of a float32 [-1, 1] mono buffer —
        the same shape the listener already hands to Whisper. None on
        any failure (too short, model missing, runtime error)."""
        try:
            arr = np.asarray(samples, dtype=np.float32).reshape(-1)
            if arr.shape[0] < int(MIN_AUDIO_SECONDS * sample_rate):
                return None
            ex = self._get_extractor()
            if ex is None:
                return None
            stream = ex.create_stream()
            stream.accept_waveform(sample_rate, arr)
            stream.input_finished()
            return normalise(np.array(ex.compute(stream), dtype=np.float32))
        except Exception as e:
            print(f"[speaker_id] embed failed: {e}")
            return None

    def score(self, samples: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> Optional[float]:
        """Cosine similarity vs the voiceprint, or None when unavailable."""
        centroid = self._load_centroid()
        if centroid is None:
            return None
        emb = self.embed(samples, sample_rate)
        if emb is None:
            return None
        return float(emb @ centroid)

    def verify(self, samples: np.ndarray,
               sample_rate: int = SAMPLE_RATE
               ) -> tuple[Optional[bool], Optional[float]]:
        """(verified, score). verified is None — not False — when the
        score couldn't be computed: enforcement (Fase B) must be able to
        tell "not Pedro" from "couldn't check", because their fallbacks
        differ (refuse vs degrade-to-/ui-confirmation)."""
        s = self.score(samples, sample_rate)
        if s is None:
            return (None, None)
        return (s >= self.threshold, s)
