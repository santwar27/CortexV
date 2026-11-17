#!/usr/bin/env python3
"""Quick mic RMS meter — diagnose why the voice recorder won't arm.

Replicates the listener's EXACT audio path (16 kHz, 80 ms int16 frames,
RMS = sqrt(mean(x^2)) on channel 0) and the arming threshold, so the numbers
are directly comparable to `record_after_wake`'s REC_RMS_THRESHOLD. When the
wake word fires but nothing is transcribed, the usual cause is speech that
never sustains above that bar — this shows whether that's happening.

Run it in the SAME terminal context as the voice daemon (so it inherits the
mic TCC grant), speak a command at your normal volume, and read the result:

  ./venv/bin/python scripts/mic_rms_check.py [seconds]
"""
import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SIZE = 1280            # 80 ms — same chunk as the listener
MIC_NAME_SUBSTR = "MacBook"  # same device selection as the listener
THRESHOLD = 500             # the listener's REC_RMS_THRESHOLD


def pick_mic() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and MIC_NAME_SUBSTR in d["name"]:
            return i
    sys.exit(f"no input device matching '{MIC_NAME_SUBSTR}'")


def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    idx = pick_mic()
    print(f"Mic: [{idx}] {sd.query_devices(idx)['name']}")
    print(f"Arm threshold (REC_RMS_THRESHOLD): {THRESHOLD}")
    print(f"Speak a command NOW, normal volume, for {secs:.0f}s...")
    rmss = []
    with sd.InputStream(device=idx, channels=1, samplerate=SAMPLE_RATE,
                        dtype="int16", blocksize=FRAME_SIZE) as stream:
        for _ in range(int(secs * SAMPLE_RATE / FRAME_SIZE)):
            audio, _ = stream.read(FRAME_SIZE)
            rmss.append(float(np.sqrt(np.mean(audio[:, 0].astype(np.float64) ** 2))))
    arr = np.array(rmss)
    above = int((arr > THRESHOLD).sum())
    print(f"\nframes: {len(arr)}  |  max={arr.max():.0f}  avg={arr.mean():.0f}")
    print(f"frames above {THRESHOLD}: {above}/{len(arr)} "
          f"({100 * above / max(1, len(arr)):.0f}%)")
    if above >= 3:
        print("\n→ Speech DOES cross the bar. Arming should work — look elsewhere.")
    else:
        print("\n→ Speech NEVER sustains above the bar — that's why the recorder")
        print("  won't arm. Fix: lower the threshold or raise effective mic gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
