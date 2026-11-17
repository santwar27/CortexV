"""
Speaker-ID threshold lab (Onda 12, Fase 0).

Measures how well the enrolled voiceprint separates Pedro from
not-Pedro, and suggests a threshold WITH DATA instead of a guess.

    ./venv/bin/python scripts/speaker_id_lab.py

Positives: leave-one-out over the enrollment clips — each clip scored
against the centroid of the OTHERS (scoring a clip against a centroid
that contains it would inflate the numbers).

Negatives: macOS `say` voices synthesizing the same phrases (including
the exact S3-approval wording — the attack we care about), embedded and
scored against the full voiceprint. CAVEAT, printed in the report too:
TTS voices are synthetic and the model was trained on real human speech
(VoxCeleb), so synthetic negatives are indicative, not definitive — the
real negative distribution comes from shadow mode (Fase A) and, if
available, real human clips dropped into ~/.cortex/lab_negatives/*.wav
(16kHz mono PCM16; any other human voice works, e.g. family — with
their knowledge).

Nothing leaves the machine.
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice.speaker_id import (  # noqa: E402
    MODEL_SHA256,
    SAMPLE_RATE,
    SpeakerVerifier,
    compute_centroid,
    normalise,
    resolve_model_path,
    resolve_voiceprint_path,
)

SAY_VOICES = ["Luciana", "Daniel", "Samantha", "Fred", "Kathy", "Ralph"]
PHRASES = [
    "Pode mandar o e-mail, por favor.",
    "Sim, pode crer.",
    "Conecta a internet por quatro horas.",
    "Hey Cortex, abre o Spotify.",
]
NEGATIVES_DIR = os.path.expanduser("~/.cortex/lab_negatives")


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            sys.exit(f"ERROR: {path} must be 16kHz mono PCM16.")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _say_negatives(verifier: SpeakerVerifier) -> dict:
    scores = {}
    centroid = normalise(np.load(resolve_voiceprint_path())["centroid"])
    with tempfile.TemporaryDirectory() as tmp:
        for voice in SAY_VOICES:
            per_voice = []
            for i, phrase in enumerate(PHRASES):
                aiff = os.path.join(tmp, f"{voice}_{i}.aiff")
                wav = os.path.join(tmp, f"{voice}_{i}.wav")
                try:
                    subprocess.run(["say", "-v", voice, "-o", aiff, phrase],
                                   check=True, capture_output=True)
                    subprocess.run(
                        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                         "-c", "1", aiff, wav],
                        check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    continue  # voice not installed — skip, report the rest
                emb = verifier.embed(_read_wav(wav))
                if emb is not None:
                    per_voice.append(float(emb @ centroid))
            if per_voice:
                scores[f"say:{voice}"] = per_voice
    return scores


def _human_negatives(verifier: SpeakerVerifier) -> dict:
    scores = {}
    centroid = normalise(np.load(resolve_voiceprint_path())["centroid"])
    for path in sorted(glob.glob(os.path.join(NEGATIVES_DIR, "*.wav"))):
        emb = verifier.embed(_read_wav(path))
        if emb is not None:
            scores[f"human:{os.path.basename(path)}"] = [float(emb @ centroid)]
    return scores


def main() -> None:
    model_path = resolve_model_path()
    if not os.path.isfile(model_path):
        sys.exit(f"ERROR: model missing at {model_path}.")
    digest = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    if digest != MODEL_SHA256:
        sys.exit(f"ERROR: model checksum mismatch ({digest[:12]}…) — "
                 "re-download; do not trust an unverified binary.")

    vp_path = resolve_voiceprint_path()
    if not os.path.isfile(vp_path):
        sys.exit("ERROR: no voiceprint — run scripts/enroll_speaker.py first.")
    with np.load(vp_path) as data:
        clip_embs = np.asarray(data["embeddings"], dtype=np.float32)
    if clip_embs.shape[0] < 3:
        sys.exit("ERROR: need ≥3 enrollment clips for leave-one-out.")

    verifier = SpeakerVerifier(model_path=model_path)

    # Positives: leave-one-out.
    positives = []
    for i in range(clip_embs.shape[0]):
        others = np.delete(clip_embs, i, axis=0)
        loo_centroid = compute_centroid(others)
        clip = normalise(clip_embs[i])
        positives.append(float(clip @ loo_centroid))

    negatives = {}
    negatives.update(_say_negatives(verifier))
    negatives.update(_human_negatives(verifier))
    neg_flat = [s for v in negatives.values() for s in v]

    print("\n── Speaker-ID lab ──────────────────────────────")
    print(f"POSITIVOS (leave-one-out, {len(positives)} clips):")
    print(f"  min {min(positives):.3f} · média {np.mean(positives):.3f} "
          f"· max {max(positives):.3f}")
    print(f"NEGATIVOS ({len(neg_flat)} amostras):")
    for name, vals in sorted(negatives.items()):
        print(f"  {name}: " + " ".join(f"{v:.3f}" for v in vals))
    if not neg_flat:
        sys.exit("ERROR: no negatives produced.")
    print(f"  máx geral: {max(neg_flat):.3f}")

    gap = min(positives) - max(neg_flat)
    print(f"\nGAP (pior positivo − pior negativo): {gap:.3f}")
    if gap <= 0:
        print("⚠️  SEM separação limpa — NÃO ligar enforcement; investigar "
              "(clips fracos? re-enrollar; modelo? testar outro).")
        return
    mid = max(neg_flat) + gap / 2
    fp_safe = min(positives) - gap * 0.25
    print(f"Threshold candidato (meio do gap):        {mid:.3f}")
    print(f"Threshold candidato (FP-safe, ¾ do gap):  {fp_safe:.3f}")
    print("\nLembretes: negativos de TTS são indicativos, não definitivos —")
    print("a distribuição real vem do shadow (Fase A). Pra negativo humano")
    print(f"real: jogar .wav 16kHz mono em {NEGATIVES_DIR}/ e re-rodar.")
    print("Setar o escolhido em CORTEX_SPEAKER_THRESHOLD (env do listener).")


if __name__ == "__main__":
    main()
