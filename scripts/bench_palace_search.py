"""
Palace search benchmark — synthetic 50k docs (M3-A5, review Fable 5).

Answers ONE question before the Mac Studio: where is the wall? The
semantic channel deserializes and cosines EVERY embedding per search
(full scan in Python), and the store is append-only with no retention —
so search latency degrades silently as the Palace grows. This script
measures it at controlled sizes instead of waiting for the degradation
to be felt in voice-loop latency.

DELIBERATELY OUT OF CI (make check): it builds a ~600MB temp DB and
takes minutes. Run by hand:

    venv/bin/python scripts/bench_palace_search.py
    venv/bin/python scripts/bench_palace_search.py --sizes 1000,10000 --searches 20

Everything is deterministic (seeded numpy RNG, fixed vocabulary), so two
runs on the same machine are comparable. The embed dim matches
production (768, nomic-embed-text); texts get a small vocabulary so FTS5
has realistic term overlap.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.palace import Palace  # noqa: E402

DIM = 768  # production dim (nomic-embed-text) — 3KB BLOB per doc

_WORDS = (
    "spotify email calendario lembrete reuniao projeto cortex memoria "
    "musica artista evento amanha estudo nota compra viagem treino "
    "whisper piper listener studio servidor backup python sqlite"
).split()


def _rng_embed(rng: np.ndarray):
    def fn(_text: str) -> np.ndarray:
        v = rng.standard_normal(DIM).astype(np.float32)
        return v / np.linalg.norm(v)
    return fn


def _synthetic_text(rng, n_words: int = 12) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def bench(sizes: list[int], searches: int) -> None:
    rng = np.random.default_rng(42)
    embed = _rng_embed(rng)

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "bench.db")
        palace = Palace(db_path=db, embed_fn=embed)
        built = 0
        print(f"dim={DIM} · {searches} searches por tamanho · DB em tmpfs do SO")
        for size in sizes:
            t0 = time.perf_counter()
            for _ in range(size - built):
                palace.add_document_with_embedding("conversation", _synthetic_text(rng))
            built = size
            build_s = time.perf_counter() - t0

            queries = [_synthetic_text(rng, 5) for _ in range(searches)]
            # warm-up (page cache, sqlite stats)
            palace.search(queries[0], k=3)
            lat = []
            for q in queries:
                t0 = time.perf_counter()
                palace.search(q, k=3)
                lat.append((time.perf_counter() - t0) * 1000)
            lat.sort()
            p50 = lat[len(lat) // 2]
            p95 = lat[int(len(lat) * 0.95) - 1]
            db_mb = os.path.getsize(db) / 1048576
            print(
                f"docs={size:>6} · build +{build_s:5.1f}s · DB {db_mb:6.1f}MB · "
                f"search p50 {p50:7.1f}ms · p95 {p95:7.1f}ms"
            )
        palace.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Palace.search at synthetic sizes.")
    ap.add_argument("--sizes", default="1000,5000,10000,25000,50000",
                    help="comma-separated doc counts (cumulative build)")
    ap.add_argument("--searches", type=int, default=30, help="searches per size")
    args = ap.parse_args(argv)
    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
    bench(sizes, args.searches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
