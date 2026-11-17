"""Embedding similarity for LLM behaviour evals (fase 2, sessão B).

Substring matching (sessão A) answers "did the reply contain the right
words". It can't answer "did the reply *drift* from the wording we
recorded last time" — a model swap or prompt-version bump can keep the
substrings yet change the answer's shape entirely. That's what cosine
similarity against a stored baseline catches.

The embedder is the same local `nomic-embed-text` the Palace uses (via
`OllamaEmbedder`). It's INJECTABLE: production passes nothing and gets
the real Ollama-backed embedder; the framework self-tests pass a stub
lambda so they stay offline and fast (inside `make check`).

Threshold rationale: Metal/MPS isn't bit-perfect even with a fixed
seed (reduction-order non-determinism in float16), so two runs of the
same prompt drift a few tokens. 0.92 cosine tolerates that token-level
wobble while still flagging a genuinely different answer. It is NOT
== 1.0 by design — see the fase 2 decisions in HANDOFF.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

# Cosine below this against the baseline reply = "drifted from baseline".
# Informational only — drift never fails the run (drift policy).
SIMILARITY_THRESHOLD = 0.92

# A text -> vector callable. The real one hits Ollama; tests inject a stub.
Embedder = Callable[[str], "np.ndarray"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clamped to [-1, 1].

    Returns 0.0 if either vector is all-zeros (undefined direction) —
    we never raise here, so a degenerate embedding shows as maximal
    drift rather than crashing the report.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    # Float error can nudge an identical pair past 1.0; clamp.
    return max(-1.0, min(1.0, sim))


def _default_embedder() -> Embedder:
    """Lazily build the production Ollama-backed embedder. Imported
    inside the function so importing this module never requires numpy's
    HTTP stack or a running Ollama (keeps self-tests that inject a stub
    fully offline)."""
    from src.memory.embeddings import OllamaEmbedder

    return OllamaEmbedder().embed


def text_similarity(
    current: str,
    baseline: str,
    embedder: Optional[Embedder] = None,
) -> float:
    """Cosine similarity between two texts' embeddings.

    Empty/blank text on either side embeds to nothing meaningful, so we
    short-circuit: identical (both blank) → 1.0, one blank → 0.0. This
    avoids handing the embedder an empty string (OllamaEmbedder raises
    on that) and keeps the drift report sensible.
    """
    cur = (current or "").strip()
    base = (baseline or "").strip()
    if not cur and not base:
        return 1.0
    if not cur or not base:
        return 0.0
    emb = embedder or _default_embedder()
    return cosine_similarity(emb(cur), emb(base))


def baseline_drifts(similarity: float) -> bool:
    """True when a similarity score falls below the drift threshold."""
    return similarity < SIMILARITY_THRESHOLD


def score_against_baseline(
    current: Dict[str, str],
    baseline: Dict[str, str],
    embedder: Optional[Embedder] = None,
) -> Dict[str, dict]:
    """Compare current replies to a baseline, case by case.

    Returns `{case_name: {"similarity": float, "drift": bool}}` for every
    case present in BOTH mappings. Cases only in `current` (newly added,
    no baseline yet) or only in `baseline` (removed) are skipped — the
    caller reports those separately as "no baseline" rather than as drift.

    The embedder is resolved once and reused across cases so a real
    Ollama-backed run pays connection setup a single time.
    """
    emb = embedder or _default_embedder()
    out: Dict[str, dict] = {}
    for name, cur_reply in current.items():
        if name not in baseline:
            continue
        sim = text_similarity(cur_reply, baseline[name], embedder=emb)
        out[name] = {"similarity": sim, "drift": baseline_drifts(sim)}
    return out
