"""
Deep-research dispatch (Fase 2 — DORMANT wiring).

Routes "pesquise a fundo / profundamente X" to the DeepStudier scaffold. The
whole path is gated by CORTEX_DEEP_STUDIER (default off): when the flag is OFF
the handler returns None and the orchestrator FALLS THROUGH to normal routing
(a regular web search) — so this is inert until the Mac Studio flips the flag
and a strong model makes the multi-step synthesis worthwhile.

Narrow + anchored: it only fires on an explicit "deep/a fundo" qualifier, so a
plain "pesquise X" still goes to the normal search path.
"""
from __future__ import annotations

import re

DEEP_RESEARCH_RE = re.compile(
    r"^(?:"
    # PT — "pesquise/investigue/estude (a fundo|profundamente|em profundidade) (sobre) X"
    r"(?:pesquis(?:e|a|ar)|investig(?:ue|ar|a)|estud(?:e|a|ar))\s+"
    r"(?:a\s+fundo|profundamente|em\s+profundidade|profund[ao])\s+"
    r"(?:sobre\s+|acerca\s+de\s+|o\s+|a\s+|de\s+|d[oa]\s+)?"
    r"|pesquisa\s+profunda\s+(?:sobre\s+|de\s+)?"
    r"|"
    # EN — "deep research (on/about) X"
    r"deep\s+research\s+(?:on\s+|about\s+)?"
    r")(?P<topic>.+?)[\s.!?]*$",
    re.IGNORECASE | re.DOTALL,
)
