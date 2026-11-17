#!/usr/bin/env python3
"""
Re-embed every palace document with the CURRENT embedder.

Needed whenever the embedding model changes (first use: nomic-embed-text →
bge-m3, s78 2026-07-08 — see scripts/benchmark_embeddings.py for the A/B
that justified the swap): vectors from different models live in different
spaces AND different dims, so the old BLOBs are useless the moment the
query side moves to the new model. The palace search skips mixed-dim docs
rather than crash, so an un-reembedded doc is simply invisible to the
semantic channel — this script makes them visible again.

Safety:
  - REFUSES to run without a fresh backup flag (--backup-done) unless
    --dry-run. Take the backup first:
        cp data/palace.db data/palace.db.bak-pre-embed-swap
  - Idempotent: re-running just overwrites vectors with identical ones.
  - A doc whose embed call fails keeps its OLD vector (better a skipped
    doc than a lost one); failures are listed at the end.

Usage:
  ./venv/bin/python scripts/reembed_palace.py --dry-run
  ./venv/bin/python scripts/reembed_palace.py --backup-done
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import numpy as np  # noqa: E402

from src.memory.embeddings import OllamaEmbedder  # noqa: E402

# Mirrors the benchmark cap: the retrieval signal lives in the opening of
# each doc, and a consistent cap keeps embed latency bounded.
DOC_CHAR_CAP = 4000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/palace.db")
    ap.add_argument("--dry-run", action="store_true",
                    help="count and report, write nothing")
    ap.add_argument("--backup-done", action="store_true",
                    help="assert you took a DB backup before writing")
    args = ap.parse_args()

    if not args.dry_run and not args.backup_done:
        print("RECUSADO: faça o backup primeiro e passe --backup-done:\n"
              "  cp data/palace.db data/palace.db.bak-pre-embed-swap")
        return 1

    embedder = OllamaEmbedder()
    print(f"Embedder: {embedder.model}")

    db = sqlite3.connect(args.db)
    rows = db.execute(
        "SELECT id, texto FROM documentos "
        "WHERE texto IS NOT NULL AND length(trim(texto)) > 0"
    ).fetchall()
    print(f"{len(rows)} docs a re-embedar em {args.db}"
          + (" (DRY-RUN — nada será escrito)" if args.dry_run else ""))
    if args.dry_run:
        return 0

    t0 = time.perf_counter()
    ok, failed = 0, []
    for doc_id, texto in rows:
        try:
            vec = embedder.embed(texto[:DOC_CHAR_CAP])
        except (RuntimeError, ValueError) as e:
            failed.append((doc_id, str(e)[:80]))
            continue  # keep the old vector — invisible beats lost
        db.execute(
            "UPDATE documentos SET embedding = ? WHERE id = ?",
            (np.asarray(vec, dtype=np.float32).tobytes(), doc_id),
        )
        ok += 1
        if ok % 50 == 0:
            db.commit()
            print(f"  {ok}/{len(rows)}…")
    db.commit()
    db.close()

    print(f"✓ {ok} docs re-embedados em {time.perf_counter() - t0:.0f}s")
    if failed:
        print(f"✗ {len(failed)} falharam (vetor ANTIGO mantido — ficam fora "
              f"da busca semântica até novo re-embed):")
        for doc_id, err in failed:
            print(f"  doc {doc_id}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
