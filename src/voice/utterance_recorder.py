"""Temporary local recording of voice utterances, for STT A/B tests (s84).

Why: Whisper small hears the same word differently between tries — "Daisies"
came out as "you know", "you come", "Yukon", "you comb", "you call" — and the
fix candidates (faster-whisper `hotwords`, a bigger model) can only be judged
on the SAME audio. Cortex never kept audio, so there was nothing to compare.

OFF unless a flag file exists, checked on every utterance, so it flips with no
listener restart (same mechanism as cortex-eyes.sh / cortex-shush.sh). The
flag lives in /tmp: a reboot turns recording off by itself.

    ./scripts/cortex-record.sh on | off | status

What it keeps: one 16-bit mono WAV per utterance plus a line in
utterances.jsonl with what Whisper heard, in data/voice/utterances/
(gitignored). Only the newest MAX_KEEP are kept; older ones are deleted —
only files this module wrote, in its own folder. Fail-soft: recording must
never break a voice turn.
"""
from __future__ import annotations

import json
import os
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.redaction import redact_text

FLAG = os.getenv("CORTEX_RECORD_FLAG", "/tmp/cortex_record_on")
DIR = Path(os.getenv(
    "CORTEX_RECORD_DIR",
    str(Path(__file__).resolve().parents[2] / "data" / "voice" / "utterances"),
))
LOG_NAME = "utterances.jsonl"
MAX_KEEP = int(os.getenv("CORTEX_RECORD_MAX", "50"))


def enabled() -> bool:
    """On only for Pedro's own flag file (s88, Codex lote 12): /tmp is shared
    by every account on the Mac, and a file another one left under this name
    — or a link there — switched the recording of his voice on."""
    from src.core.flag_file import own_flag
    return own_flag(FLAG)


def save(audio_int16: np.ndarray, sample_rate: int, text: str,
         lang: str, prob: float, directory: Optional[Path] = None) -> Optional[Path]:
    """Write the utterance and what was heard, if recording is on. Returns
    the WAV path, or None when off or on any failure."""
    if not enabled():
        return None
    d = Path(directory) if directory is not None else DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        # UTC (s88, Codex lote 12 — invariant 5): a local stamp repeats in
        # the hour after the clocks go back, so two takes could share a name.
        stamp = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                 + f"-{int(time.time() * 1000) % 1000:03d}Z")
        path = d / f"utt-{stamp}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(np.asarray(audio_int16, dtype=np.int16).tobytes())
        with open(d / LOG_NAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "file": path.name,
                "ts": datetime.now(timezone.utc).isoformat(),
                # Masked like the audit (s88, Codex lote 12): "minha senha é
                # exemplo123" went into this log whole. The WAV is the take
                # The user opted in to keep; the text is a copy of it.
                "heard": redact_text(text),
                "lang": lang,
                "lang_prob": round(float(prob), 3),
                "seconds": round(len(audio_int16) / float(sample_rate), 2),
            }, ensure_ascii=False) + "\n")
        _rotate(d)
        return path
    except Exception as e:  # noqa: BLE001 — never break the voice turn
        print(f"  [record: falhou ao gravar a fala: {e}]")
        return None


def _rotate(d: Path) -> None:
    """Keep the newest MAX_KEEP recordings; drop older WAVs and their log
    lines. Touches only utt-*.wav and the log in this folder.

    Newest by when each file was written (s88, Codex lote 12 — invariant 5),
    not by its name: names were local wall-clock stamps, and after the clocks
    go back (25/10, 02:00 BST → 01:00 GMT) a take at 01:05 sorted before the
    ones from 01:50 an hour earlier and was the first deleted. Names written
    before the change stay local, which is why the order cannot come from
    them even now."""
    wavs = sorted(d.glob("utt-*.wav"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    extra = wavs[:-MAX_KEEP] if len(wavs) > MAX_KEEP else []
    if not extra:
        return
    gone = {p.name for p in extra}
    for p in extra:
        p.unlink(missing_ok=True)
    log = d / LOG_NAME
    if log.exists():
        keep = [ln for ln in log.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln).get("file") not in gone]
        log.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
