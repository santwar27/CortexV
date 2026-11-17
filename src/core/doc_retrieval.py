"""Pick the part of a document that answers the question (s83).

Clipping a long document to its first 6.000 characters is honest but dumb:
the answer is wherever it is. This splits the text, embeds the pieces with
the model the Palace already uses (bge-m3), and keeps the ones closest to
the question — reassembled in READING ORDER, because a document read out of
order is a different document.

Three guarantees, in the order they matter:

1. **It never lies about what it read.** The caller gets back HOW the text
   was chosen — whole, retrieved or clipped — and says so to the user.
2. **It never raises into a turn.** Ollama's embedding endpoint answered 500
   twice this week under load; any failure falls back to the plain clip,
   which is exactly today's behaviour.
3. **It does not re-embed a document per question.** Pedro asks four things
   about the same PDF; the chunks are cached by content hash.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Characters per chunk. Small enough that several fit the budget (so an
# answer spread over two sections still arrives whole), big enough to carry
# a self-contained passage — a numbered exam question, a clause, a section.
CHUNK_CHARS = 1200
# Carried from the previous chunk, so a sentence cut in half is still whole
# in one of them.
CHUNK_OVERLAP = 150
# Cached documents (content hash → chunks + vectors). A handful is plenty:
# this is one person reading one file at a time.
_CACHE_MAX = 8
_cache: dict = {}


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split on paragraph boundaries, packing them up to `size`.

    A paragraph longer than `size` (a wall of OCR text with no blank lines)
    is cut on sentence ends where possible, and only then on raw length."""
    chunks: List[str] = []
    current = ""
    for paragraph in _split_paragraphs(text):
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            buffer = ""
            for sentence in sentences:
                while len(sentence) > size:          # no sentence breaks at all
                    chunks.append(sentence[:size])
                    sentence = sentence[size - overlap:]
                if len(buffer) + len(sentence) + 1 > size:
                    chunks.append(buffer.strip())
                    buffer = buffer[-overlap:] if overlap else ""
                buffer = f"{buffer} {sentence}".strip()
            if buffer:
                current = buffer
            continue
        if len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = (current[-overlap:] + "\n\n" + paragraph) if overlap else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return [c for c in (chunk.strip() for chunk in chunks) if c]


def _default_embed(text: str):
    from src.memory.embeddings import OllamaEmbedder
    return OllamaEmbedder().embed(text)


def _cosine(a, b) -> float:
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _chunk_vectors(text: str, embed: Callable) -> Tuple[List[str], List]:
    """Chunks and their vectors, cached by document content."""
    key = hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()
    hit = _cache.get(key)
    if hit is not None:
        return hit
    chunks = chunk_text(text)
    vectors = [embed(chunk) for chunk in chunks]
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (chunks, vectors)
    return chunks, vectors


def select_relevant(text: str, question: Optional[str], *, budget: int,
                    embed: Optional[Callable] = None) -> Tuple[str, str]:
    """→ (context, how), how being "whole" | "retrieved" | "clipped".

    "clipped" is the honest failure mode and covers every reason retrieval
    could not run: no question to search with, embeddings unavailable, a
    document that does not chunk."""
    body = text or ""
    if len(body) <= budget:
        return (body, "whole")
    query = (question or "").strip()
    if not query or os.getenv("CORTEX_DOC_RETRIEVAL", "1").strip().lower() in (
            "0", "false", "off"):
        return (body[:budget], "clipped")
    try:
        embed = embed or _default_embed
        chunks, vectors = _chunk_vectors(body, embed)
        if len(chunks) < 2:
            return (body[:budget], "clipped")
        query_vector = embed(query)
        scored = sorted(
            ((_cosine(query_vector, vector), index)
             for index, vector in enumerate(vectors)),
            reverse=True,
        )
        chosen: List[int] = []
        used = 0
        for _score, index in scored:
            cost = len(chunks[index]) + 2
            if used + cost > budget:
                continue
            chosen.append(index)
            used += cost
        if not chosen:
            return (body[:budget], "clipped")
        # Reading order, with a visible gap where text was left out, so the
        # model does not read two distant passages as one continuous one.
        chosen.sort()
        pieces: List[str] = []
        for position, index in enumerate(chosen):
            if position and index != chosen[position - 1] + 1:
                pieces.append("[…]")
            pieces.append(chunks[index])
        return ("\n\n".join(pieces), "retrieved")
    except Exception:                     # noqa: BLE001 — never raise into a turn
        logger.exception("doc_retrieval: falling back to a plain clip")
        return (body[:budget], "clipped")
