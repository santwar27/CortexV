#!/usr/bin/env python3
"""Measure THIS microphone in THIS room and recommend the speech threshold.

Why this exists (s82): the recorder's speech/silence threshold was a constant
(500) written on some machine, somewhere. On Pedro's mic the median frame of
real speech is ~225 — so 64% of his speech read as silence, and the recorder
both cut him off mid-sentence and ran to its 30-second ceiling. A number that
depends on a microphone, a preamp and a room cannot be a constant.

The daemon calibrates its own floor at boot, but a floor is only half the
picture. This script measures the other half — YOUR VOICE, through the real
capture path — and prints the number to put in .env.

Run it from a Terminal (macOS gives the mic to Terminal; a launchd job lives
in a different permission context):

    venv/bin/python scripts/voice_calibrate.py

Read-only: it records into memory to compute levels and never writes audio to
disk. Nothing leaves the machine (SOBERANIA regra 2).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voice.listener import (  # noqa: E402
    FRAME_SIZE,
    SAMPLE_RATE,
    find_macbook_mic,
    rms,
)

# NO WAKE WORD in these phrases (s82 cont.). The first version opened with
# "Hey Cortex, que horas são agora?" — which woke the running daemon, so its
# chime and spoken reply played into the microphone WHILE this script was
# measuring. the user's floor came out at 765 RMS, and a good part of that was
# Cortex talking to itself. A measuring tool must not trigger the thing it is
# measuring.
PHRASES = [
    "Que horas são agora, e como vai ficar o tempo amanhã?",
    "Põe uma música para tocar, por favor.",
    "Hoje eu comi 55 gramas de aveia e uma banana.",
]
QUIET_S = 3.0


def _read_levels(stream, seconds: float) -> np.ndarray:
    frames = max(1, int(seconds * SAMPLE_RATE / FRAME_SIZE))
    out = []
    for _ in range(frames):
        audio, _overflowed = stream.read(FRAME_SIZE)
        out.append(rms(audio[:, 0]))
    return np.array(out)


def _countdown(msg: str, seconds: int = 3) -> None:
    for n in range(seconds, 0, -1):
        print(f"\r{msg} em {n}...", end="", flush=True)
        time.sleep(1)
    print(f"\r{msg} AGORA          ", flush=True)


def _listener_running() -> bool:
    """True when the voice daemon is up. Its wake chime and spoken replies go
    into the same microphone this script is measuring, so a reading taken with
    it running is measuring Cortex, not the room."""
    try:
        out = subprocess.run(["pgrep", "-f", "src.voice.listener"],
                             capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001 — a failed check must not block the tool
        return False


def main() -> int:
    import sounddevice as sd

    if _listener_running():
        print("⚠️  O daemon de voz está RODANDO.")
        print("   O chime e a resposta falada dele entram neste mesmo")
        print("   microfone e sobem o ruído medido — a calibração sai errada.")
        print()
        print("   Pare antes de medir:  pkill -f 'src.voice.listener'")
        print("   Depois de calibrar:   bash scripts/cortex-voice-start.sh")
        print()
        try:
            if input("Medir mesmo assim? [s/N] ").strip().lower() not in ("s", "sim", "y"):
                return 1
        except EOFError:
            return 1

    device = find_macbook_mic()
    print("=" * 62)
    print("Calibração do microfone do Cortex")
    print("=" * 62)
    print(f"Dispositivo: {device}\n")

    stream = sd.InputStream(
        device=device, channels=1, samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE, dtype="int16",
    )
    stream.start()
    try:
        print(f"1/2 — SILÊNCIO. Fique quieto por {QUIET_S:.0f}s "
              "(deixe o quarto como fica normalmente).")
        _countdown("Medindo o silêncio")
        floor_levels = _read_levels(stream, QUIET_S)
        floor = float(np.median(floor_levels))
        print(f"     piso de ruído: mediana {floor:.0f}, "
              f"pico {floor_levels.max():.0f}\n")

        print("2/2 — FALA. Leia as frases no seu tom NORMAL, "
              "na distância de sempre.")
        speech = []
        for phrase in PHRASES:
            print(f'     "{phrase}"')
            _countdown("     Fale")
            speech.append(_read_levels(stream, 4.0))
        speech_levels = np.concatenate(speech)
    finally:
        stream.stop()
        stream.close()

    # Frames while speaking include the natural gaps between words, so the
    # median of the whole window sits well below the voice itself. What matters
    # is how much of the speech clears a candidate threshold.
    print()
    print("=" * 62)
    print(f"Piso de ruído (mediana):  {floor:6.0f}")
    print(f"Sua fala — mediana:       {np.median(speech_levels):6.0f}")
    print(f"Sua fala — p75 / p90:     {np.percentile(speech_levels, 75):6.0f}"
          f" / {np.percentile(speech_levels, 90):6.0f}")
    print()
    print("Candidatos (quanto da sua fala fica ACIMA, e a folga sobre o ruído):")
    best = None
    for cand in (120, 150, 200, 250, 300, 400, 500, 700):
        above = float((speech_levels > cand).mean() * 100)
        margin = cand / floor if floor > 0 else float("inf")
        flag = ""
        # Want a healthy share of speech above the line AND clear separation
        # from the room. Below ~1.6x the floor the room starts arming the
        # recorder; below ~45% of frames the recorder struggles to arm and
        # closes mid-sentence.
        if above >= 45 and margin >= 1.6:
            flag = "  <- viável"
            if best is None:
                best = cand
        print(f"   {cand:>4}: {above:5.1f}% da fala acima | "
              f"{margin:4.1f}x o ruído{flag}")

    ratio = (np.median(speech_levels) / floor) if floor > 0 else float("inf")
    print()
    print(f"Fala / ruído: {ratio:.2f}x")
    print()
    # Since s82 the recorder decides with Silero VAD (content), not with this
    # threshold (energy). The number below only matters as the FALLBACK for
    # when the VAD model can't load — which is why a bad ratio here is now a
    # diagnosis, not a blocker.
    if best is None:
        print("Nenhum limiar de energia separa sua fala do ruído deste ambiente")
        print(f"(a fala está só {ratio:.2f}x acima do ruído — abaixo de ~1.6x")
        print("nenhum número funciona).")
        print()
        print("ISSO NÃO TE BLOQUEIA: desde a s82 o gravador decide por VAD")
        print("(Silero, que separa por CONTEÚDO), e o limiar abaixo só vale")
        print("como fallback se o modelo não carregar.")
        print()
        print("Ainda assim vale reduzir o ruído — ajuda o VAD e o Whisper:")
        print("  · veja o que está soando (ventilador, janela, TV, cooler)")
        print("  · fale mais perto do microfone")
        print("  · um headset/AirPods melhora muito a relação sinal-ruído")
        return 1

    print(f"RECOMENDADO (fallback): {best}")
    print()
    print("Ponha no .env (tem prioridade sobre a calibração automática):")
    print(f"   CORTEX_REC_RMS_THRESHOLD={best}")
    print()
    print("Lembrando: desde a s82 o caminho normal é o VAD; este número só")
    print("entra em jogo se o modelo do VAD não carregar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
