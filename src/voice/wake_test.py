"""
Wake-word standalone smoke.

Listens on the MacBook built-in mic and prints a banner whenever the
configured wake word fires. Used to validate the openWakeWord stack
in isolation — before tomorrow's `hey_cortex.onnx` drop, and right
after it lands to confirm detection works.

Wake model resolved via `CORTEX_WAKE_MODEL_PATH` (see
`src/voice/wake_config.py`). Unset → 'hey_jarvis' built-in.

Run:
    # built-in fallback (current placeholder)
    ./venv/bin/python -m src.voice.wake_test

    # custom model (after Colab training drops the .onnx)
    CORTEX_WAKE_MODEL_PATH=src/voice/models/hey_cortex.onnx \\
        ./venv/bin/python -m src.voice.wake_test
"""
import sys
import time

import numpy as np  # noqa: F401  — kept for parity with listener.py audio path
import sounddevice as sd
from openwakeword.model import Model

from src.voice.wake_config import resolve_wake_config, resolve_wake_threshold

WAKE_CONFIG = resolve_wake_config()
THRESHOLD = resolve_wake_threshold()
COOLDOWN_S = 2.0
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80 ms at 16 kHz — openWakeWord's expected chunk
MIC_NAME_SUBSTR = "MacBook"


def find_macbook_mic() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and MIC_NAME_SUBSTR in d["name"]:
            return i
    sys.exit(
        f"ERROR: no input device matching '{MIC_NAME_SUBSTR}' found.\n"
        f"Available devices:\n{sd.query_devices()}"
    )


def main() -> None:
    print(
        f"Loading openWakeWord model "
        f"({WAKE_CONFIG.lookup_key}, source={WAKE_CONFIG.source})..."
    )
    # Force ONNX runtime explicitly — same reason as listener.py: openwakeword
    # 0.6.0 on Python 3.12 hard-fails on missing tflite-runtime instead of
    # falling back automatically.
    model = Model(wakeword_models=[WAKE_CONFIG.model_spec], inference_framework="onnx")

    mic_index = find_macbook_mic()
    mic_name = sd.query_devices(mic_index)["name"]
    print(f"Using mic: [{mic_index}] {mic_name}")
    print(f"Threshold={THRESHOLD}, cooldown={COOLDOWN_S}s")
    print(f"Say '{WAKE_CONFIG.phrase_hint}' to wake. Ctrl+C to exit.\n")

    last_wake = 0.0
    tick = 0

    with sd.InputStream(
        device=mic_index,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=FRAME_SIZE,
    ) as stream:
        while True:
            audio, _ = stream.read(FRAME_SIZE)
            mono = audio[:, 0]
            scores = model.predict(mono)
            score = float(scores.get(WAKE_CONFIG.lookup_key, 0.0))
            tick += 1

            now = time.time()
            if score >= THRESHOLD and (now - last_wake) >= COOLDOWN_S:
                last_wake = now
                print(" " * 60, end="\r")
                print(f"WAKE DETECTED  score={score:.3f}")
            elif tick % 12 == 0:
                bar = "#" * int(score * 20)
                print(f"  score={score:.3f}  {bar:<20}", end="\r", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
