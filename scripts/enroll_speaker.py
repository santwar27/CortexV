"""
Speaker enrollment — records Pedro's voiceprint (Onda 12, Fase 0).

Run this IN YOUR INTERACTIVE TERMINAL (the mic grant lives there, same
constraint as the voice listener):

    ./venv/bin/python scripts/enroll_speaker.py

It records N short clips (default 8, ~4s each — you read the prompted
phrase, in your normal voice, normal distance from the Mac), embeds each
with the local speaker model, and writes:

  ~/.cortex/voiceprint.npz        (centroid + per-clip embeddings, chmod 600)
  ~/.cortex/enroll_clips/*.wav    (raw clips, chmod 700 dir — kept so a
                                   future/better model can re-enroll
                                   without re-recording; delete them any
                                   time, only the .npz is needed live)

Nothing leaves the machine. Re-running overwrites the previous
voiceprint. After enrolling, run scripts/speaker_id_lab.py to measure
separation and pick the threshold with data.
"""
from __future__ import annotations

import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice.speaker_id import (  # noqa: E402
    SAMPLE_RATE,
    SpeakerVerifier,
    compute_centroid,
    resolve_model_path,
    resolve_voiceprint_path,
)

CLIP_SECONDS = 4.0
NUM_CLIPS = 8

# Mix of PT and EN, biased toward what is actually said to the Cortex —
# the embeddings should sample the voice you USE with it, including the
# courtesy wrappers and the wake phrase itself.
PHRASES = [
    "Hey Cortex, qual é o próximo jogo do Brasil na Copa do Mundo?",
    "Abre o Spotify e toca uma música boa pra trabalhar.",
    "Pode mandar o e-mail, por favor.",
    "Conecta a internet por quatro horas.",
    "Hey Cortex, can you open Spotify please?",
    "Que horas são agora? E como vai ficar o tempo amanhã?",
    "Cria um lembrete pra ligar pro dentista na segunda de manhã.",
    "Sim, pode crer. Para de tocar, obrigado.",
]


def _pick_input_device() -> int:
    import sounddevice as sd

    substr = os.environ.get("CORTEX_MIC_SUBSTR", "MacBook")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and substr in d["name"]:
            return i
    sys.exit(f"ERROR: no input device matching '{substr}' found "
             f"(set CORTEX_MIC_SUBSTR to override).")


def _record_clip(device: int, seconds: float) -> np.ndarray:
    import sounddevice as sd

    frames = int(seconds * SAMPLE_RATE)
    buf = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                 dtype="float32", device=device)
    sd.wait()
    return buf.reshape(-1)


def _save_wav(path: str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> None:
    model_path = resolve_model_path()
    if not os.path.isfile(model_path):
        sys.exit(
            f"ERROR: speaker model not found at {model_path}.\n"
            "Download it first (see MODEL_URL in src/voice/speaker_id.py)."
        )

    verifier = SpeakerVerifier(model_path=model_path)
    device = _pick_input_device()

    clips_dir = os.path.expanduser("~/.cortex/enroll_clips")
    os.makedirs(clips_dir, exist_ok=True)
    os.chmod(clips_dir, 0o700)

    print(f"\nEnrollment: {NUM_CLIPS} clips de {CLIP_SECONDS:.0f}s. "
          "Fala no teu tom normal, distância normal do Mac.\n")

    embeddings, kept = [], 0
    for i, phrase in enumerate(PHRASES[:NUM_CLIPS], start=1):
        print(f"[{i}/{NUM_CLIPS}] Lê em voz alta: “{phrase}”")
        input("    ENTER pra começar a gravar... ")
        print("    🎙️  gravando...")
        samples = _record_clip(device, CLIP_SECONDS)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        emb = verifier.embed(samples)
        if emb is None or rms < 0.005:
            print(f"    ⚠️  clip fraco (rms={rms:.4f}) — descartado, repete:")
            emb = None
            while emb is None:
                input("    ENTER pra regravar... ")
                print("    🎙️  gravando...")
                samples = _record_clip(device, CLIP_SECONDS)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                emb = verifier.embed(samples)
                if rms < 0.005:
                    emb = None
        _save_wav(os.path.join(clips_dir, f"clip_{i:02d}.wav"), samples)
        embeddings.append(emb)
        kept += 1
        print(f"    ✓ ok (rms={rms:.4f})")
        time.sleep(0.3)

    centroid = compute_centroid(np.stack(embeddings))
    if centroid is None:
        sys.exit("ERROR: centroid computation failed — re-run enrollment.")

    out = resolve_voiceprint_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, centroid=centroid, embeddings=np.stack(embeddings))
    os.chmod(out, 0o600)

    sims = [float(e @ centroid) for e in embeddings]
    print(f"\n✓ Voiceprint salvo em {out} ({kept} clips).")
    print(f"  Similaridade clip→centroid: min {min(sims):.3f} · "
          f"média {np.mean(sims):.3f} · max {max(sims):.3f}")
    print("  Agora roda: ./venv/bin/python scripts/speaker_id_lab.py")


if __name__ == "__main__":
    main()
