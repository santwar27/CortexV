#!/usr/bin/env python3
"""
Entry point for the CORTEX Nightly Studier.

Drains the study queue, researches every pending topic, saves a
Markdown briefing per topic to data/knowledge/pending/.

Invoked by:
  - the launchd LaunchAgent at the scheduled hour (Fase C)
  - manually: `./venv/bin/python scripts/nightly_run.py`

Logging: INFO-level to stdout (captured by the LaunchAgent into
logs/nightly.out.log when running unattended).
"""

import logging
import sys
from pathlib import Path

# Allow running both as a script (`python scripts/nightly_run.py`) and
# as a module (`python -m scripts.nightly_run`) from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The nightly LaunchAgent only sets PATH — without this, feature flags in
# .env (CORTEX_CURIOSITY, CORTEX_DEEP_STUDIER, …) never reach the nightly
# process and the dormant scaffolds can't be flipped on from one place.
# Mirrors src/api.py. Gap found 2026-07-08 when enabling curiosity.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.core.studier import NightlyStudier  # noqa: E402
from src.core.log_trim import trim_log  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("nightly_run")

    for name in ("uvicorn.out.log", "uvicorn.err.log"):
        path = LOGS_DIR / name
        try:
            trim_log(path)
        except Exception as exc:
            # Log maintenance must never prevent the scheduled study run.
            log.warning("API log trim failed: path=%s error=%s", path, type(exc).__name__)

    log.info("CORTEX Nightly Studier — starting run")
    studier = NightlyStudier()
    produced = studier.run()

    if not produced:
        log.info("Nothing to produce. Done.")
        return 0

    log.info(f"Done. Produced {len(produced)} briefing(s):")
    for path in produced:
        log.info(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
