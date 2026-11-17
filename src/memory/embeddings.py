"""
Local embedding provider for the Palace memory layer.

Wraps Ollama's nomic-embed-text endpoint and returns numpy float32 vectors
(matches the format Palace serialises into the embedding BLOB column).

The Palace itself never calls Ollama directly — it receives a callable via
DI. This module is the production implementation; tests pass a deterministic
lambda instead so they stay pure-logic.
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK or host.endswith(".localhost") or host.startswith("127.")


class OllamaEmbedder:
    """Thin HTTP client around POST /api/embeddings."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_URL = "http://localhost:11434/api/embeddings"
    DEFAULT_TIMEOUT_S = 30

    def __init__(
        self,
        model: Optional[str] = None,
        url: Optional[str] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ):
        self.model = model or os.getenv("CORTEX_EMBED_MODEL", self.DEFAULT_MODEL)
        self.url = url or os.getenv("CORTEX_EMBED_URL", self.DEFAULT_URL)
        self.timeout_s = timeout_s

    def embed(self, text: str) -> np.ndarray:
        """
        Return the embedding for `text` as a float32 numpy array.

        Raises RuntimeError on HTTP/JSON failure so the caller can decide
        whether to skip the document, retry, or fall back. We never return
        a zero vector silently — that would poison cosine similarity.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OllamaEmbedder.embed: text must be a non-empty string")
        # What is embedded is the Palace's memory and the user's résumé: it does
        # not leave the machine (SOBERANIA 1). CORTEX_EMBED_URL pointed
        # anywhere took the whole résumé there (s88, Codex lote 11, reproduced
        # with the transport stubbed). Another machine is a decision, made by
        # setting CORTEX_ALLOW_REMOTE_EMBED=1 — not a URL typed by mistake.
        if not _is_local(self.url) and os.getenv("CORTEX_ALLOW_REMOTE_EMBED") != "1":
            raise RuntimeError(
                f"OllamaEmbedder: refusing a non-local embed URL "
                f"({urlparse(self.url).hostname}); set CORTEX_ALLOW_REMOTE_EMBED=1 "
                f"to allow it on purpose")

        try:
            resp = requests.post(
                self.url,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"OllamaEmbedder: HTTP failure: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"OllamaEmbedder: invalid JSON: {e}") from e

        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(
                f"OllamaEmbedder: missing 'embedding' in response from {self.url}"
            )
        return np.asarray(vec, dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self.embed(text)
