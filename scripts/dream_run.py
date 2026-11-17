#!/usr/bin/env python3
"""
Entry point for the Dream Loop (Fase 2 — DORMANT).

Generates one nightly "dream" (free-association over random Palace fragments)
and stores it as pending for review. The whole feature is gated by
CORTEX_DREAM_LOOP (default off): if the flag is off this exits immediately, so
even a loaded LaunchAgent is a no-op until the Mac Studio flips the flag and a
strong creative model makes the dream worth reading.

Invoked by:
  - launchd/com.cortex.dream.plist (nightly, opt-in — only when you load it)
  - manually: CORTEX_DREAM_LOOP=1 ./venv/bin/python scripts/dream_run.py

Review the result by voice/text ("me conta o último sonho" / "guarda" /
"descarta") or read data/dreams/dreams.jsonl.
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dream_loop import DreamLoop, _dream_enabled  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("dream_run")

    if not _dream_enabled():
        log.info("Dream Loop is OFF (CORTEX_DREAM_LOOP) — dormant, nothing to do.")
        return 0

    # Build the real Palace (same db + embedder as CortexMemory). Imported
    # lazily so the dormant/disabled path doesn't pull heavy deps.
    from src.memory.palace import Palace
    from src.memory.embeddings import OllamaEmbedder

    log.info("Dream Loop — starting nightly run")
    palace = Palace(db_path="data/palace.db", embed_fn=OllamaEmbedder())
    loop = DreamLoop(palace)  # generate_fn=None → lazy Ollama (CORTEX_MODEL_CREATIVE)
    dream = loop.dream()
    if dream is None:
        log.info("No dream produced (empty Palace or empty generation).")
        return 0
    log.info("Dream saved: id=%s (pending review)", dream.id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
