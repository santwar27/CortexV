#!/usr/bin/env python3
"""
A/B benchmark for embedding models on the REAL palace data (read-only).

Motivation (s78, 2026-07-08): retrieval quality is the cheapest M1-now
lever to make Cortex smarter — the generation model is pinned (8B until
Mac Studio) but the embedder is tiny and swappable. Lesson from the Qwen
episode applies: measure anti-regression BEFORE swapping, never after.

What it measures, per model:
  1. Knowledge routing — query = briefing topic (from documentos.metadata),
     relevant = that briefing's chunks. recall@3 + MRR.
  2. Conversation paraphrase — hand-written PT/EN queries targeting real
     stored turns (matched by marker substring, robust to id changes).
     recall@3 + MRR.
  3. Band separation — cosine of (query, its target) vs (query, random
     non-targets). The _CONVO_SIM_FLOOR in memory.py (0.55) is calibrated
     to nomic's compressed band (unrelated ~0.59, follow-ups ~0.71);
     a new embedder RESHAPES this scale, so the floor must be recalibrated
     from these numbers as part of any swap.
  4. Latency — mean seconds per embed call (the query-side cost paid on
     every single turn).

Usage:
  ./venv/bin/python scripts/benchmark_embeddings.py \
      --models nomic-embed-text,mxbai-embed-large,bge-m3

Read-only: never writes to the palace. Embeddings live in RAM only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.embeddings import OllamaEmbedder  # noqa: E402

# Docs are embedded with a consistent cap so model comparison is fair and
# the run stays fast; retrieval signal lives overwhelmingly in the opening
# of each stored doc (topic header / User: turn).
DOC_CHAR_CAP = 4000

# Hand-written paraphrase queries against REAL palace content (inspected
# 2026-07-08). Each case: (query, marker) — every doc whose text contains
# `marker` counts as relevant. Point-in-time by design: if the palace
# changes and a marker matches nothing, the case is skipped with a note.
PARAPHRASE_CASES = [
    ("que tipo de música me anima antes do treino?", "funk para me animar"),
    ("quais estilos de música brasileira eu gosto?", "MPB"),
    ("when do I have one-to-one meetings with my boss?", "one-to-one meetings"),
    ("até que idade eu morei no Brasil?", "morei no Brasil ate meus 17"),
    ("qual é o número do meu CPF?", "CPF é"),
    ("me explica o que é aprendizado de máquina", "Machine Learning (ML)"),
    ("como diminuir o brilho da tela do Mac?", "brilho do"),
    ("como funciona o mecanismo de atenção nos transformers?",
     "# Topic: attention mechanism"),
    ("qual assistente de código local é melhor, Qwen ou DeepSeek?",
     "Qwen 2.5 Coder vs DeepSeek"),
    ("como fazer login OAuth seguro num app desktop?", "PKCE"),
    ("qual a previsão do tempo em Londres?", "weather in London"),
    ("ligar a internet por comando de voz", "Liga a internet"),
]

K = 3          # recall@K
N_UNRELATED = 25  # random non-targets per query for band stats
RNG_SEED = 42


def load_docs(db_path: str):
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT id, tipo, texto, metadata FROM documentos "
        "WHERE texto IS NOT NULL AND length(trim(texto)) > 0"
    ).fetchall()
    db.close()
    return [
        {"id": r[0], "tipo": r[1], "texto": r[2], "metadata": r[3]}
        for r in rows
    ]


def knowledge_cases(docs):
    """Auto task: query = briefing topic, relevant = its chunk ids.
    Topics that are filenames (perfil_*.md) make poor natural queries
    and are skipped."""
    by_topic = {}
    for d in docs:
        if d["tipo"] != "knowledge" or not d["metadata"]:
            continue
        try:
            topic = json.loads(d["metadata"]).get("topic") or ""
        except json.JSONDecodeError:
            continue
        if not topic or topic.endswith(".md"):
            continue
        by_topic.setdefault(topic, set()).add(d["id"])
    return [(topic, ids) for topic, ids in sorted(by_topic.items())]


def paraphrase_case_ids(docs):
    cases = []
    for query, marker in PARAPHRASE_CASES:
        ids = {d["id"] for d in docs if marker in d["texto"]}
        if ids:
            cases.append((query, ids))
        else:
            print(f"  (case pulado — marker sem match: {marker!r})")
    return cases


def cosine_matrix(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return mn @ qn


def evaluate_model(model: str, docs) -> dict:
    embedder = OllamaEmbedder(model=model)
    print(f"\n>>> {model}: embedando {len(docs)} docs…")

    t0 = time.perf_counter()
    latencies = []
    vectors = []
    kept_docs = []
    for d in docs:
        t = time.perf_counter()
        try:
            v = embedder.embed(d["texto"][:DOC_CHAR_CAP])
        except (RuntimeError, ValueError) as e:
            print(f"  doc {d['id']} falhou: {e}")
            continue
        latencies.append(time.perf_counter() - t)
        vectors.append(v)
        kept_docs.append(d)
    matrix = np.vstack(vectors)
    dims = matrix.shape[1]
    print(f"  {len(kept_docs)} docs embedados em "
          f"{time.perf_counter() - t0:.0f}s (dims={dims})")

    id_to_row = {d["id"]: i for i, d in enumerate(kept_docs)}
    rng = np.random.default_rng(RNG_SEED)

    tasks = {
        "knowledge": knowledge_cases(kept_docs),
        "paraphrase": paraphrase_case_ids(kept_docs),
    }
    results = {"model": model, "dims": dims,
               "embed_latency_mean_s": statistics.mean(latencies)}
    related_sims, unrelated_sims = [], []

    for name, cases in tasks.items():
        hits_at_k, rranks = [], []
        for query, relevant_ids in cases:
            try:
                qv = embedder.embed(query)
            except (RuntimeError, ValueError):
                continue
            sims = cosine_matrix(qv, matrix)
            order = np.argsort(-sims)
            ranked_ids = [kept_docs[i]["id"] for i in order]
            first = next((r + 1 for r, did in enumerate(ranked_ids)
                          if did in relevant_ids), None)
            hits_at_k.append(1.0 if first is not None and first <= K else 0.0)
            rranks.append(1.0 / first if first is not None else 0.0)

            # Band stats: this query's sim to its BEST target vs to
            # random non-targets.
            target_rows = [id_to_row[i] for i in relevant_ids
                           if i in id_to_row]
            related_sims.append(max(float(sims[r]) for r in target_rows))
            non_targets = [i for i in range(len(kept_docs))
                           if kept_docs[i]["id"] not in relevant_ids]
            sample = rng.choice(non_targets,
                                size=min(N_UNRELATED, len(non_targets)),
                                replace=False)
            unrelated_sims.extend(float(sims[i]) for i in sample)

        results[name] = {
            "cases": len(hits_at_k),
            f"recall@{K}": statistics.mean(hits_at_k) if hits_at_k else 0.0,
            "mrr": statistics.mean(rranks) if rranks else 0.0,
        }

    results["band"] = {
        "related_mean": statistics.mean(related_sims),
        "related_min": min(related_sims),
        "unrelated_mean": statistics.mean(unrelated_sims),
        "unrelated_p90": float(np.percentile(unrelated_sims, 90)),
        "gap_min_related_vs_p90_unrelated":
            min(related_sims) - float(np.percentile(unrelated_sims, 90)),
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="nomic-embed-text,mxbai-embed-large,bge-m3")
    ap.add_argument("--db", default="data/palace.db")
    args = ap.parse_args()

    docs = load_docs(args.db)
    print(f"Palace: {len(docs)} docs carregados de {args.db} (READ-ONLY).")

    all_results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        all_results.append(evaluate_model(model, docs))

    print("\n" + "=" * 64)
    print("RESULTADO (recall@3 / MRR — maior é melhor)")
    print("=" * 64)
    for r in all_results:
        b = r["band"]
        print(f"\n{r['model']}  (dims={r['dims']}, "
              f"{r['embed_latency_mean_s']*1000:.0f}ms/embed)")
        for task in ("knowledge", "paraphrase"):
            t = r[task]
            print(f"  {task:<11} recall@3={t['recall@3']:.2f}  "
                  f"mrr={t['mrr']:.2f}  ({t['cases']} casos)")
        print(f"  banda: related mean={b['related_mean']:.3f} "
              f"min={b['related_min']:.3f} | unrelated "
              f"mean={b['unrelated_mean']:.3f} p90={b['unrelated_p90']:.3f}")
        print(f"  gap (min related − p90 unrelated): "
              f"{b['gap_min_related_vs_p90_unrelated']:+.3f}"
              f"  ← informa o _CONVO_SIM_FLOOR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
