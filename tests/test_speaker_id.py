"""Onda 12 Bloco 0: speaker_id — resolvers, centroid math, fail-soft.

No ONNX model, no sherpa-onnx import, no mic: the extractor is an
injected fake. What's pinned here is the CONTRACT the listener (Fase A)
and the gate policy (Fase B) will rely on:
  - score/verify return None — never raise, never False — when the
    check can't run (voice hot path must degrade, and enforcement must
    distinguish "not Pedro" from "couldn't check");
  - degenerate inputs (short audio, zero vectors, missing files) are
    unavailable, not accidental matches.
"""

import numpy as np
import pytest

from src.voice.speaker_id import (
    MIN_AUDIO_SECONDS,
    SAMPLE_RATE,
    SpeakerVerifier,
    compute_centroid,
    normalise,
    resolve_model_path,
    resolve_speaker_threshold,
    resolve_voiceprint_path,
)


# ── fakes ────────────────────────────────────────────────────────────

class _FakeStream:
    def __init__(self):
        self.samples = None

    def accept_waveform(self, sr, samples):
        self.samples = np.asarray(samples)

    def input_finished(self):
        pass


class _FakeExtractor:
    """Returns a fixed vector, or one derived from the audio's first
    sample so tests can steer the embedding per call."""

    def __init__(self, vec=None):
        self.vec = vec

    def create_stream(self):
        self._stream = _FakeStream()
        return self._stream

    def compute(self, stream):
        if self.vec is not None:
            return list(self.vec)
        head = float(stream.samples[0])
        return [head, 1.0 - head, 0.25]


def _audio(seconds=2.0, head=0.5):
    buf = np.full(int(seconds * SAMPLE_RATE), 0.1, dtype=np.float32)
    buf[0] = head
    return buf


def _write_voiceprint(path, centroid, embeddings=None):
    if embeddings is None:
        embeddings = np.stack([centroid, centroid])
    np.savez(path, centroid=np.asarray(centroid, dtype=np.float32),
             embeddings=np.asarray(embeddings, dtype=np.float32))


def _verifier(tmp_path, vec, centroid):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    vp = tmp_path / "voiceprint.npz"
    _write_voiceprint(vp, centroid)
    return SpeakerVerifier(
        model_path=str(model), voiceprint_path=str(vp),
        extractor_factory=lambda p: _FakeExtractor(vec=vec),
    )


# ── resolvers (same degrade-to-default contract as wake_config) ─────

def test_threshold_default(monkeypatch):
    monkeypatch.delenv("CORTEX_SPEAKER_THRESHOLD", raising=False)
    assert resolve_speaker_threshold() == 0.45


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "0.0", "1.0", "1.5", "-0.2"])
def test_threshold_bad_values_degrade(raw):
    assert resolve_speaker_threshold(raw) == 0.45


def test_threshold_valid_value():
    assert resolve_speaker_threshold("0.62") == 0.62


def test_path_resolvers_expand_and_default(monkeypatch):
    monkeypatch.delenv("CORTEX_SPEAKER_MODEL_PATH", raising=False)
    monkeypatch.delenv("CORTEX_SPEAKER_VOICEPRINT", raising=False)
    assert resolve_model_path().endswith("speaker_campplus_voxceleb.onnx")
    assert "~" not in resolve_model_path()
    assert resolve_voiceprint_path().endswith("voiceprint.npz")
    assert resolve_model_path("/x/m.onnx") == "/x/m.onnx"


# ── vector math ──────────────────────────────────────────────────────

def test_normalise_unit_norm_and_degenerate():
    v = normalise(np.array([3.0, 4.0]))
    assert pytest.approx(float(np.linalg.norm(v))) == 1.0
    assert normalise(np.zeros(4)) is None
    assert normalise(np.array([np.nan, 1.0])) is None


def test_compute_centroid_is_normalised_mean():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = compute_centroid(embs)
    assert pytest.approx(float(c[0]), abs=1e-6) == pytest.approx(float(c[1]), abs=1e-6)
    assert pytest.approx(float(np.linalg.norm(c))) == 1.0
    assert compute_centroid(np.zeros((0, 2))) is None
    assert compute_centroid(np.array([[0.0, 0.0]])) is None  # zero row


# ── scoring ──────────────────────────────────────────────────────────

def test_score_matches_cosine(tmp_path):
    v = _verifier(tmp_path, vec=[1.0, 0.0, 0.0], centroid=[1.0, 0.0, 0.0])
    assert pytest.approx(v.score(_audio()), abs=1e-6) == 1.0


def test_verify_threshold_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_SPEAKER_THRESHOLD", "0.5")
    matched = _verifier(tmp_path, vec=[1.0, 0.0], centroid=[1.0, 0.0])
    ok, score = matched.verify(_audio())
    assert ok is True and score == pytest.approx(1.0)
    orthogonal = _verifier(tmp_path, vec=[0.0, 1.0], centroid=[1.0, 0.0])
    ok, score = orthogonal.verify(_audio())
    assert ok is False and score == pytest.approx(0.0, abs=1e-6)


# ── fail-soft: None, never raise, never accidental match ────────────

def test_short_audio_is_unavailable(tmp_path):
    v = _verifier(tmp_path, vec=[1.0, 0.0], centroid=[1.0, 0.0])
    short = _audio(seconds=MIN_AUDIO_SECONDS / 2)
    assert v.embed(short) is None
    assert v.verify(short) == (None, None)


def test_missing_voiceprint_is_unavailable(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    v = SpeakerVerifier(
        model_path=str(model),
        voiceprint_path=str(tmp_path / "nope.npz"),
        extractor_factory=lambda p: _FakeExtractor(vec=[1.0, 0.0]),
    )
    assert not v.is_available()
    assert v.score(_audio()) is None
    assert v.verify(_audio()) == (None, None)


def test_extractor_failure_is_unavailable_and_not_retried(tmp_path):
    calls = []

    def exploding_factory(path):
        calls.append(path)
        raise RuntimeError("corrupt model")

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    vp = tmp_path / "voiceprint.npz"
    _write_voiceprint(vp, [1.0, 0.0])
    v = SpeakerVerifier(model_path=str(model), voiceprint_path=str(vp),
                        extractor_factory=exploding_factory)
    assert v.score(_audio()) is None
    assert v.score(_audio()) is None
    assert len(calls) == 1  # permanent fail-soft: no per-utterance retry


def test_degenerate_model_vector_is_unavailable(tmp_path):
    # A zero embedding must be None, not a spurious ~1.0 cosine.
    v = _verifier(tmp_path, vec=[0.0, 0.0], centroid=[1.0, 0.0])
    assert v.embed(_audio()) is None
    assert v.verify(_audio()) == (None, None)


def test_is_available_requires_both_halves(tmp_path):
    vp = tmp_path / "voiceprint.npz"
    _write_voiceprint(vp, [1.0, 0.0])
    missing_model = SpeakerVerifier(
        model_path=str(tmp_path / "nope.onnx"), voiceprint_path=str(vp),
        extractor_factory=lambda p: _FakeExtractor(),
    )
    assert not missing_model.is_available()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    both = SpeakerVerifier(
        model_path=str(model), voiceprint_path=str(vp),
        extractor_factory=lambda p: _FakeExtractor(),
    )
    assert both.is_available()
