#!/usr/bin/env python3
"""Compare STT settings on the SAME recorded audio (s84).

Reads the utterances saved by scripts/cortex-record.sh (data/voice/
utterances/, gitignored) and a gold file next to them, truth.json:

    {"utt-20261006-121728-644.wav": ["daisies", "justin bieber"], ...}

listing the names actually spoken in each recording ([] for a control
sentence with no name). Each setting transcribes every recording the way
the listener does — same model, VAD, auto language + pt fallback — and is
scored on: names right, names INSERTED where none was said (the hotwords
failure mode), and seconds per utterance.

Read-only: nothing is sent anywhere, nothing is written.

    venv/bin/python scripts/stt_ab.py                      # beam 1 vs beam 5
    venv/bin/python scripts/stt_ab.py --hotwords "Djonga, Matuê" --beam 5
    venv/bin/python scripts/stt_ab.py --show               # every transcript
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "voice" / "utterances"


def fold(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", (s or "").lower())
                if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(model, audio, beam: int, hotwords: str) -> str:
    def _run(lang):
        segs, info = model.transcribe(audio, beam_size=beam, vad_filter=True,
                                      language=lang, hotwords=hotwords or None)
        return " ".join(s.text for s in segs).strip(), info.language
    text, lang = _run(None)
    if lang not in ("pt", "en"):
        text, lang = _run("pt")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, action="append",
                    help="beam size(s) to compare (default: 1 and 5)")
    ap.add_argument("--hotwords", default="", help="comma-separated hotwords")
    ap.add_argument("--model", default="small")
    ap.add_argument("--show", action="store_true", help="print every transcript")
    args = ap.parse_args()

    truth_path = DIR / "truth.json"
    if not truth_path.exists():
        print(f"no gold file: {truth_path} (see this script's docstring)")
        return 2
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    files = [f for f in truth if (DIR / f).exists()]
    if not files:
        print("none of the recordings in truth.json are on disk")
        return 2
    every_name = sorted({n for names in truth.values() for n in names})

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    audio = {f: load_wav(DIR / f) for f in files}

    for beam in (args.beam or [1, 5]):
        right = total = inserted = 0
        times, outs = [], []
        for f in files:
            t0 = time.time()
            text = transcribe(model, audio[f], beam, args.hotwords)
            times.append(time.time() - t0)
            folded = fold(text)
            said = [fold(n) for n in truth[f]]
            right += sum(n in folded for n in said)
            total += len(said)
            inserted += sum(1 for n in map(fold, every_name)
                            if n in folded and n not in said)
            outs.append((f, text))
        label = f"model={args.model} beam={beam}" + (" +hotwords" if args.hotwords else "")
        print(f"{label}: names right {right}/{total} · inserted {inserted} · "
              f"{np.mean(times):.2f}s mean, {max(times):.2f}s max ({len(files)} utterances)")
        if args.show:
            for f, text in outs:
                print(f"    {f}: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
