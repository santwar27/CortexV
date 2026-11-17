"""
Cognitive Arbiter — design-reference ch. 8 (idea 4.1). The single ROI-highest idea
from the book per Pedro's notes; this is the deterministic skeleton built
ahead of the hardware (the "build-now-wire-later" pattern already used by
gate_queue — infra ahead of the producer).

What it is: a router that looks at an incoming query and decides WHICH path
should answer it, BEFORE any heavy model runs. It is the natural evolution
of the two zero-LLM layers that already exist (`reflex` + `skill_fast_path`):
those answer trivial queries; the arbiter classifies *everything* so the
right engine (and eventually the right MODEL) is picked per query.

Two layers, by design:

  • Layer 1 — LIVE NOW. A deterministic cascade of regex/heuristics that
    classifies a query into one Route with a confidence score. Zero LLM,
    pure, fully testable, and cheap enough to run on every turn. This is
    what ships today (in shadow mode — it observes and logs, it does not
    yet change routing; see orchestrator CORTEX_ARBITER_SHADOW).

  • Layer 2 — DORMANT until Mac Studio + Gemma 4. When the cascade is
    UNSURE (confidence < LOW_CONFIDENCE_THRESHOLD) a tiny LLM breaks the
    tie; and each Route maps to a distinct model tier (light / heavy /
    long-context). Both sit behind CORTEX_ARBITER_CASCADE (default off):
    while off, `classify()` always returns the deterministic cascade
    decision and never calls a model. Mac Studio day = flip the flag +
    wire `_llm_tiebreak` to the mini model + point ROUTE_MODEL_HINT at
    the real engines.

Routes (the reference cascade):
  REFLEX   — trivial/universal; already handled by reflex_layer.
  LOOKUP   — needs fresh/external info (price, news, weather, a URL).
  MEMORY   — personal recall ("lembra…", "qual é o meu…", "o que eu disse").
  SIMPLES  — short general-knowledge Q a light model can answer.
  COMPLEXO — reasoning / multi-step / code ("explique", "compare", "por quê").
  LONGO    — a large pasted blob that needs a long-context model.

Design rules (mirrors reflex.py):
  - classify() is PURE and side-effect free → the caller logs the audit
    event, so this stays trivially testable.
  - Markers use re.search (intent can sit mid-sentence); REFLEX delegates
    to the already-anchored reflex_layer so the two never disagree.
  - When nothing matches we fall back to SIMPLES with LOW confidence — that
    low score is exactly the signal that will wake Layer 2 once enabled.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.core.reflex import ReflexLayer, reflex_layer


class Route(str, Enum):
    """Where a query should be answered. str-valued so it logs as plain text."""
    REFLEX = "REFLEX"
    LOOKUP = "LOOKUP"
    MEMORY = "MEMORY"
    SIMPLES = "SIMPLES"
    COMPLEXO = "COMPLEXO"
    LONGO = "LONGO"


@dataclass(frozen=True)
class ArbiterDecision:
    """One routing decision. `layer` records how it was reached:
    'cascade' (deterministic), 'llm' (Layer-2 tie-break), or 'fallback'
    (nothing matched → SIMPLES at low confidence)."""
    route: Route
    confidence: float
    reason: str
    layer: str = "cascade"


# Below this score the cascade is "unsure" → Layer 2 (when enabled) gets a
# vote. 0.8 keeps the confident keyword routes (LOOKUP/MEMORY/COMPLEXO at
# 0.8+) deterministic and only escalates the bare SIMPLES fallback.
LOW_CONFIDENCE_THRESHOLD = 0.8

# A query is LONGO when the INPUT itself is big enough to want a long-context
# model — not merely a hard question. Tuned high so a short "explique X"
# stays COMPLEXO; only a genuine paste/transcript trips it.
LONG_INPUT_WORDS = 120
LONG_INPUT_CHARS = 700

# Each route's eventual model tier. DORMANT: today everything runs on the
# single analytical engine; on Mac Studio this maps to real engines
# (light = small fast model, heavy = Gemma 4, long = long-context model).
ROUTE_MODEL_HINT: dict[Route, Optional[str]] = {
    Route.REFLEX: None,        # answered before any model
    Route.LOOKUP: "analytical",
    Route.MEMORY: "analytical",
    Route.SIMPLES: "light",
    Route.COMPLEXO: "heavy",
    Route.LONGO: "long",
}


# ---------------------------------------------------------------------------
# Marker tables. Each entry: (compiled_regex, route, confidence, reason).
# Checked in order — first match wins. Kept separate per route so the
# distribution and the rationale are legible in tests and audit logs.
# ---------------------------------------------------------------------------

# LOOKUP — the query needs information that is fresh, external, or online.
_LOOKUP_MARKERS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"https?://|www\.", re.IGNORECASE), 0.95, "url"),
    (re.compile(r"\b(?:pre[çc]o|cota[çc][ãa]o|quanto\s+custa|quanto\s+est[áa])\b",
                re.IGNORECASE), 0.9, "price"),
    (re.compile(r"\b(?:not[íi]cia|not[íi]cias|manchete|aconteceu\s+hoje|"
                r"[úu]ltimas?\s+not[íi]cias)\b", re.IGNORECASE), 0.9, "news"),
    (re.compile(r"\b(?:clima|tempo|previs[ãa]o|temperatura)\b.*\b(?:em|no|na|hoje|amanh[ãa])\b",
                re.IGNORECASE), 0.88, "weather"),
    (re.compile(r"\b(?:quem\s+(?:ganhou|venceu)|placar|resultado\s+(?:do|da)\s+jogo|"
                r"que\s+horas?\s+(?:é|comeca|come[çc]a)\s+o\s+jogo)\b",
                re.IGNORECASE), 0.85, "live_event"),
    (re.compile(r"\b(?:pesquis[ae]|procur[ae]\s+(?:na\s+)?(?:internet|web|google)|"
                r"busca\s+(?:na\s+)?(?:internet|web))\b", re.IGNORECASE), 0.85, "web_search"),
    (re.compile(r"\b(?:search|look\s+up|google)\b", re.IGNORECASE), 0.82, "web_search_en"),
]

# MEMORY — personal recall about the user or the conversation.
_MEMORY_MARKERS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(?:lembr[ae]|relembr[ae]|recorda)\b", re.IGNORECASE), 0.88, "recall_verb"),
    (re.compile(r"\bqual\s+(?:é\s+)?(?:o\s+|a\s+|os\s+|as\s+)?meu[s]?\b", re.IGNORECASE),
     0.88, "what_is_my"),
    (re.compile(r"\bqual\s+(?:é\s+)?(?:o\s+|a\s+)?minha\b", re.IGNORECASE), 0.88, "what_is_my_f"),
    (re.compile(r"\bo\s+que\s+(?:eu\s+)?(?:te\s+disse|falei|comentei|anotei|pedi)\b",
                re.IGNORECASE), 0.85, "what_i_said"),
    (re.compile(r"\b(?:meu\s+nome|onde\s+(?:eu\s+)?moro|minha\s+(?:casa|fam[íi]lia))\b",
                re.IGNORECASE), 0.85, "personal_fact"),
    (re.compile(r"\b(?:remember|recall|what\s+did\s+i\s+(?:say|tell))\b",
                re.IGNORECASE), 0.82, "recall_en"),
]

# COMPLEXO — reasoning, multi-step, comparison, or code generation.
_COMPLEXO_MARKERS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(?:explica|explique|explicar)\b", re.IGNORECASE), 0.85, "explain"),
    (re.compile(r"\b(?:compar[ae]|compare|diferen[çc]a\s+entre)\b", re.IGNORECASE),
     0.85, "compare"),
    (re.compile(r"\b(?:analis[ae]|avalie|avalia)\b", re.IGNORECASE), 0.83, "analyze"),
    (re.compile(r"\b(?:por\s*qu[êe]|porqu[êe]|por\s+que)\b", re.IGNORECASE), 0.82, "why"),
    (re.compile(r"\bcomo\s+funciona\b", re.IGNORECASE), 0.82, "how_works"),
    (re.compile(r"\b(?:escrev[ae]|cri[ae]|implement[ae]|gera|gere)\b.*"
                r"\b(?:c[óo]digo|fun[çc][ãa]o|script|programa|classe)\b",
                re.IGNORECASE), 0.85, "code"),
    (re.compile(r"\b(?:explain|compare|analyze|why\s+(?:does|is|do))\b",
                re.IGNORECASE), 0.8, "reason_en"),
]


def _cascade_enabled() -> bool:
    """True when Layer 2 (LLM tie-break + per-route model routing) is ON.
    DORMANT default: off until Mac Studio. Any of 1/true/yes/on enables it."""
    return os.environ.get("CORTEX_ARBITER_CASCADE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _word_count(text: str) -> int:
    return len(text.split())


class CognitiveArbiter:
    """Layer-1 deterministic router with a dormant Layer-2 hook."""

    def __init__(self, reflex: Optional[ReflexLayer] = None):
        self._reflex = reflex if reflex is not None else reflex_layer

    # -- public API --------------------------------------------------------

    def classify(self, text: str) -> ArbiterDecision:
        """Route a query. PURE: never logs, never calls a model while Layer 2
        is disabled. Returns the deterministic cascade decision; when the
        cascade is unsure AND Layer 2 is enabled, the tie-break may override."""
        text = (text or "").strip()
        if not text:
            return ArbiterDecision(Route.SIMPLES, 0.0, "empty", layer="fallback")

        decision = self._cascade(text)

        # Layer 2 only wakes for low-confidence calls, and only when enabled.
        if decision.confidence < LOW_CONFIDENCE_THRESHOLD and _cascade_enabled():
            tb = self._llm_tiebreak(text, decision)
            if tb is not None:
                return tb
        return decision

    def model_hint(self, route: Route) -> Optional[str]:
        """The model tier a route should run on. DORMANT mapping — used by
        the orchestrator only once Layer 2 is enabled."""
        return ROUTE_MODEL_HINT.get(route)

    # -- Layer 1: deterministic cascade ------------------------------------

    def _cascade(self, text: str) -> ArbiterDecision:
        # 1. REFLEX — delegate to the already-anchored reflex layer so the
        #    two can never disagree about what is "trivial".
        if self._reflex.is_hit(text):
            return ArbiterDecision(Route.REFLEX, 1.0, "reflex_hit")

        # 2. LONGO — a genuinely large input wants a long-context model,
        #    regardless of topic. Checked before keyword routes so a big
        #    paste that happens to contain "explique" still routes by size.
        if _word_count(text) >= LONG_INPUT_WORDS or len(text) >= LONG_INPUT_CHARS:
            return ArbiterDecision(Route.LONGO, 0.9, "long_input")

        # 3-5. Keyword routes — strongest evidence wins ACROSS all tables, with
        #    table order (LOOKUP > MEMORY > COMPLEXO) only as the tiebreak on
        #    EQUAL confidence. (Earlier this returned the first matching table,
        #    so a low-confidence LOOKUP marker could beat a high-confidence
        #    MEMORY recall — e.g. "o que eu te disse" losing to a stray keyword.)
        best: Optional[tuple[float, Route, str]] = None
        for table, route in (
            (_LOOKUP_MARKERS, Route.LOOKUP),
            (_MEMORY_MARKERS, Route.MEMORY),
            (_COMPLEXO_MARKERS, Route.COMPLEXO),
        ):
            for pattern, conf, reason in table:
                # Strictly-greater keeps the first-seen (table+within-table
                # order) on ties, so the result stays deterministic.
                if pattern.search(text) and (best is None or conf > best[0]):
                    best = (conf, route, reason)
        if best is not None:
            return ArbiterDecision(best[1], best[0], best[2])

        # 6. Fallback — a short query with no markers. SIMPLES at LOW
        #    confidence: that low score is the wake-signal for Layer 2.
        return ArbiterDecision(Route.SIMPLES, 0.5, "no_marker", layer="fallback")

    # -- Layer 2: DORMANT (Mac Studio + Gemma 4) ---------------------------

    def _llm_tiebreak(
        self, text: str, cascade: ArbiterDecision  # noqa: ARG002 — dormant
    ) -> Optional[ArbiterDecision]:
        """Mini-LLM tie-break for low-confidence calls. DORMANT: returns None
        so `classify` keeps the deterministic decision. On Mac Studio this
        prompts a small model to pick a Route and returns an ArbiterDecision
        with layer='llm'. Guarded by _cascade_enabled() at the call site, so
        it is never reached while the flag is off."""
        return None


# Module-level singleton — import directly, like reflex_layer.
arbiter = CognitiveArbiter()
