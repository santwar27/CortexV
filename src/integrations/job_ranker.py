"""
Rank stored jobs against the master resume (Autonomous Job Hunter slice 2, s76).

Cosine similarity between the resume embedding and each job's (title +
description) embedding, written to jobs.score (0..1). Everything is LOCAL and
deterministic given the embedder: the resume is the personal half and never
leaves the machine (SOBERANIA r1) — embeddings come from the local Ollama
model (nomic-embed-text), the same one Palace uses. No cloud, no LLM judgement.

The embedder is injectable (`embed: text -> np.ndarray`), so tests pass a
deterministic stub and stay offline; production defaults to OllamaEmbedder.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from src.integrations.jobs import JobLog

log = logging.getLogger(__name__)

Embed = Callable[[str], "np.ndarray"]


def _default_embedder() -> Embed:
    from src.memory.embeddings import OllamaEmbedder
    return OllamaEmbedder()


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_jobs(joblog: JobLog, *, embed: Optional[Embed] = None,
              limit: int = 1000, only_unscored: bool = False) -> int:
    """Score up to `limit` stored jobs against the resume. Returns how many
    were scored. 0 if there's no resume or nothing to rank. Fail-soft per job:
    an embedding error skips that job, never aborts the run.

    `only_unscored=True` skips jobs that already carry a score — so the daily
    refresh embeds only the freshly-fetched ones, not the whole ledger."""
    resume = joblog.get_resume()
    if not resume or not (resume["text"] or "").strip():
        log.warning("job_ranker: no resume stored — nothing to rank against")
        return 0

    embed = embed or _default_embedder()
    try:
        r_vec = embed(resume["text"])
    except Exception:
        log.exception("job_ranker: failed to embed resume")
        return 0

    scored = 0
    for row in joblog.list(limit=limit, order="recent"):
        if only_unscored and row["score"] is not None:
            continue
        text = f"{row['title']} {row['description'] or ''}".strip()
        if not text:
            continue
        try:
            j_vec = embed(text)
        except Exception:
            log.warning("job_ranker: skip job %s (embed failed)", row["id"])
            continue
        score = round((_cosine(r_vec, j_vec) + 1.0) / 2.0, 4)  # map [-1,1]→[0,1]
        if joblog.set_score(row["id"], score):
            scored += 1
    return scored
