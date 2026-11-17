#!/usr/bin/env python3
"""Does Palace retrieval miss today? — step 1 of the reranker decision (s83).

Why this exists. docs/design/reranker_decision.md decided a cross-encoder is
viable, but not whether it is NEEDED. A reranker can only reorder the
candidates the hybrid search already found, so the useful question is not
"is the top-3 wrong?" but "is the top-3 wrong AND was the right turn sitting
just below it?". This script measures that on Pedro's real sentences.

It EXECUTES NOTHING: the Palace is opened read-only and `mark_used` is never
called, so the frequency channel is not fed by the replay. It only embeds
text.

What it reproduces. The conversation path of
`CortexMemory.retrieve_relevant_context`: `palace.search(k=3,
tipo="conversation")`, then the relevance floor and the echo dedup. The
candidate POOL (what a reranker would get to reorder) comes from
`Palace.candidate_pool`, the list `search()` takes its head from; the first
version of this script rebuilt the fusion by hand, and a self-check that the
pool's head equals `search()` stays as a guard against that ever returning.

Two outputs:
  - stats on stdout: pool sizes (how often >= 20, the reranker's gate),
    how often the floor leaves nothing;
  - a review file with a fixed-seed sample, top-3 plus the rest of the pool
    for each query, for labelling by hand. It holds Pedro's conversations,
    so it goes to data/reviews/ (gitignored), never to the repo.

Usage:
    venv/bin/python scripts/palace_retrieval_probe.py
    venv/bin/python scripts/palace_retrieval_probe.py --sample 40 --seed 83
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# router_replay loads the .env on import (bge-m3, not nomic) — the same
# reason that file gives applies here.
from router_replay import AUDIT, harvest  # noqa: E402

from src.core.memory import CortexMemory  # noqa: E402
from src.memory.embeddings import OllamaEmbedder  # noqa: E402
from src.memory.palace import Palace, _deserialize_embedding  # noqa: E402

TIPO = CortexMemory.DOC_TIPO
K = 3                                   # turns injected, as production calls it
FETCH_K = CortexMemory.convo_fetch_k(K)  # candidates it asks the Palace for
RERANK_GATE = 20


def reflex_hits(path: Path) -> Set[str]:
    """Sentences the reflex layer answered — they never reach the Palace."""
    hits: Set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("stage") == "reflex" and row.get("event") == "hit":
                text = (row.get("payload") or {}).get("input")
                if isinstance(text, str):
                    hits.add(text.strip())
    return hits


def pool(palace: Palace, query: str) -> List[Dict]:
    """The full RRF-ordered candidate pool behind search(k=3) — read from
    Palace.candidate_pool, the same list search() takes its head from."""
    cands = palace.candidate_pool(query, k=FETCH_K, tipo=TIPO)
    if not cands:
        return []
    ids = [c["id"] for c in cands]
    ph = ",".join("?" * len(ids))
    rows = palace._conn.execute(
        f"SELECT id, texto, created_at FROM documentos WHERE id IN ({ph})", ids
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [{
        "id": c["id"], "score": c["score"], "sim": c["sem_sim"],
        "channels": "".join(ch if ch in c["channels"] else "-" for ch in "SFQ"),
        "created_at": by_id[c["id"]]["created_at"][:10],
        "user": CortexMemory._user_side(by_id[c["id"]]["texto"]),
    } for c in cands if c["id"] in by_id]


def production_view(query: str, cands: List[Dict]) -> List[Dict]:
    """Floor + echo dedup + the short-turn and question cuts over the pool, in pool order,
    like production does: of the first FETCH_K candidates, the first K that
    survive the cuts are injected. Keeps every candidate, marking which ones
    production injects (`injected`) and why the rest would be cut (`cut`)."""
    floor = CortexMemory._CONVO_SIM_FLOOR
    seen = {CortexMemory._echo_key(query)}
    out = []
    taken = 0
    for pos, c in enumerate(cands):
        c = dict(c, pos=pos + 1)
        key = CortexMemory._echo_key(c["user"])
        if c["sim"] is not None and c["sim"] < floor:
            c["cut"] = "floor"
        elif not key or key in seen:
            c["cut"] = "echo"
        elif len(key) < CortexMemory._MIN_USER_CHARS:
            c["cut"] = "short"
        elif "?" in c["user"] and CortexMemory._skip_questions():
            c["cut"] = "quest"
        else:
            c["cut"] = None
            seen.add(key)
        c["injected"] = pos < FETCH_K and c["cut"] is None and taken < K
        taken += c["injected"]
        out.append(c)
    return out


def true_cosine(palace: Palace, query_vec, doc_id: int):
    """Raw cosine of one stored turn against the query, whatever channel
    surfaced it — so "injected below the floor" means the same thing before
    and after a change to how the channels score. None if no usable vector."""
    row = palace._conn.execute(
        "SELECT embedding FROM documentos WHERE id = ?", (doc_id,)).fetchone()
    if row is None or row["embedding"] is None:
        return None
    v = _deserialize_embedding(row["embedding"])
    if v.shape != query_vec.shape or float(np.linalg.norm(v)) < 1e-9:
        return None
    return float(np.dot(query_vec, v / np.linalg.norm(v)))


def recall_report(palace: Palace, path: Path) -> None:
    """Memory questions whose answer is a sentence Pedro said before: does
    that sentence reach the prompt? The set lives in data/reviews/ (it holds
    his facts); each item is {"q", "expect": [doc ids], "source"}."""
    items = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    print(f"\nrecall set ({path.name}, {len(items)} questions):")
    for it in items:
        v = production_view(it["q"], pool(palace, it["q"]))
        injected = {c["id"] for c in v if c["injected"]}
        found = [c["pos"] for c in v if c["id"] in it["expect"]]
        ok = bool(injected & set(it["expect"]))
        hits += ok
        where = "INJECTED" if ok else (f"in pool at #{min(found)}" if found else "absent")
        print(f"  {'✓' if ok else '✗'} [{it['source'][:5]}] {it['q'][:60]!r} — {where}")
    print(f"recall: {hits}/{len(items)} reach the prompt")


def fmt(c: Dict) -> str:
    sim = "  —  " if c["sim"] is None else f"{c['sim']:.3f}"
    tag = "INJ  " if c["injected"] else (c["cut"] or "    ").ljust(5)
    text = " ".join(c["user"].replace("User:", "", 1).split())[:220]
    return f"  #{c['pos']:<2} {tag} sim={sim} [{c['channels']}] {c['created_at']}  {text}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=83)
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "reviews"
                                         / "palace_probe_sample.md"))
    ap.add_argument("--recall", default=str(PROJECT_ROOT / "data" / "reviews"
                                            / "palace_recall_set.json"),
                    help="memory-question set; skipped if the file is missing")
    args = ap.parse_args()

    reflex = reflex_hits(AUDIT)
    queries = [q for q in harvest(AUDIT) if q not in reflex]
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} distinct sentences reach the Palace "
          f"({len(reflex)} reflex hits excluded)")

    palace = Palace(db_path=str(PROJECT_ROOT / "data" / "palace.db"),
                    embed_fn=OllamaEmbedder(), read_only=True)

    floor = CortexMemory._CONVO_SIM_FLOOR
    sizes, injected_n, drift = [], Counter(), 0
    below_floor, unscored, below_queries = 0, 0, 0
    views: Dict[str, List[Dict]] = {}
    for n, q in enumerate(queries, 1):
        cands = pool(palace, q)
        head = [d["id"] for d in palace.search(q, k=FETCH_K, tipo=TIPO)]
        if head != [c["id"] for c in cands[:FETCH_K]]:
            drift += 1
        v = production_view(q, cands)
        views[q] = v
        sizes.append(len(cands))
        injected_n[sum(c["injected"] for c in v)] += 1
        inj = [c for c in v if c["injected"]]
        if inj:
            qv = np.asarray(palace.embed_fn(q), dtype=np.float32)
            qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
            low = [c for c in inj
                   if (cos := true_cosine(palace, qv, c["id"])) is not None
                   and cos < floor]
            below_floor += len(low)
            below_queries += bool(low)
            unscored += sum(c["sim"] is None for c in inj)
        if n % 50 == 0:
            print(f"  … {n}/{len(queries)}", file=sys.stderr)

    print(f"\nself-check: pool head != search() in {drift}/{len(queries)} queries"
          + ("" if drift == 0 else "  ⚠️ rebuild drifted from production"))
    print(f"pool size: min {min(sizes)} · median {statistics.median(sizes)} · "
          f"max {max(sizes)} · >= {RERANK_GATE}: "
          f"{sum(s >= RERANK_GATE for s in sizes)}/{len(sizes)}")
    print("convo turns injected per query (after floor + echo + short + quest): "
          + ", ".join(f"{k}→{injected_n[k]}" for k in sorted(injected_n)))
    print(f"injected below the floor ({floor}) by true cosine: {below_floor} "
          f"in {below_queries} queries · injected with no semantic score: {unscored}")
    if Path(args.recall).exists():
        recall_report(palace, Path(args.recall))

    rng = random.Random(args.seed)
    sample = rng.sample(queries, min(args.sample, len(queries)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"# Palace retrieval probe — sample of {len(sample)} "
                 f"(seed {args.seed}, {datetime.now():%Y-%m-%d %H:%M})\n\n")
        fh.write("Legend: INJ = enters the prompt today; floor/echo/short/quest = cut by "
                 "production. Channels: S semantic, F FTS, Q frequency.\n"
                 "Label each: OK · BELOW (right turn in pool, not injected) · "
                 "ABSENT (right turn not in pool) · N/A (no past turn could help).\n\n")
        for i, q in enumerate(sample, 1):
            fh.write(f"## {i}. {q}\n\nlabel: \n\n```\n")
            for c in views[q]:
                fh.write(fmt(c) + "\n")
            fh.write("```\n\n")
    print(f"\nreview sample → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
