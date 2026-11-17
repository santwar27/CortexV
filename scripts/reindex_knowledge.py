#!/usr/bin/env python3
"""
Reindex every approved Nightly Studier briefing into the knowledge base.

Use this:
  - one-off, after upgrading the embedding model
  - after manually editing files under data/knowledge/approved/
  - after restoring approved/ from backup

Idempotent: existing entries are upserted by deterministic IDs derived
from filename, so running twice doesn't duplicate.

Run with: ./venv/bin/python scripts/reindex_knowledge.py
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Same gap as nightly_run.py (fixed s78): without .env the embedder falls
# back to its hardcoded default model — after the bge-m3 swap that would
# write 768-dim vectors into a 1024-dim palace, making every reindexed
# chunk silently invisible to semantic search (dim-mismatch skip).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.core.knowledge_base import KnowledgeBase  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("reindex")

    kb = KnowledgeBase()
    before = kb.count()
    log.info(f"KnowledgeBase currently has {before} document(s) indexed.")

    count = kb.reindex_all()
    after = kb.count()

    log.info(f"Reindexed {count} file(s). Collection size: {before} → {after}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
