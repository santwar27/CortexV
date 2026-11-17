"""
SemanticRouter — intent understanding by MEANING, not by regex form.

Design: docs/design/m1_interpretation_plan.md (item 1). The dispatch registry
matches the FORM of a phrase; anything the regex author didn't anticipate falls
to the weak 8B, which hallucinates ("me de a lista de estudo" → "não tenho
acesso ao perfil", s78). This layer closes that gap on the M1, locally: it
embeds a small set of canonical example phrases per dispatch intent (bge-m3, the
embedder already installed s78) and, given a query, finds the nearest intent by
cosine similarity. Nearest above a confidence floor AND with enough margin over
the runner-up wins; otherwise it declines (returns None) and the caller falls
through to the LLM exactly as today.

Safety by construction:
  - This routes ONLY to dispatch modes that already exist — it never invents an
    action. Worst case it picks a wrong KNOWN intent and that handler's own
    guards catch it. It generates nothing.
  - It is a SAFETY NET below the regex: the orchestrator runs the deterministic
    registry first (unchanged, zero regression) and only consults this on a
    regex MISS.
  - Shadow-first (item 1b): the orchestrator logs what this WOULD route without
    acting, so thresholds are calibrated on real audit misses before it goes
    live (same discipline as the Arbiter shadow and the embedding A/B).

Two layers, same split as the other scaffolds:
  • Layer 1 — LIVE, deterministic and fully testable with an injected embed_fn:
    the catalogue, the one-time embedding of examples, the nearest-intent search,
    and the score/margin thresholds.
  • Layer 2 — the QUALITY of the match is model-bound only in that a stronger
    embedder (or the Mac Studio model doing true intent parsing) would raise the
    ceiling. The mechanism here is complete and runs today on bge-m3.

The whole layer sits behind CORTEX_SEMANTIC_ROUTER (default off).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

EmbedFn = Callable[[str], "np.ndarray"]


def _semantic_router_enabled() -> bool:
    """True when the semantic router may actually ROUTE (item 1c). DORMANT
    default: off — flip only after the shadow data calibrates the thresholds."""
    return os.environ.get("CORTEX_SEMANTIC_ROUTER", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _semantic_router_shadow_enabled() -> bool:
    """True when the router runs in SHADOW (item 1b): observe + log what it
    WOULD route, without routing. Separate from the live flag (mirrors the
    Arbiter's shadow/cascade split) so data can be collected safely first."""
    return os.environ.get("CORTEX_SEMANTIC_ROUTER_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass(frozen=True)
class IntentExamples:
    """Canonical example phrasings for one dispatch mode. `mode` matches the
    registry's routing-label mode exactly (e.g. 'STUDIER_LIST') so a match can
    be handed straight to the existing handler."""
    mode: str
    phrases: Tuple[str, ...]


@dataclass(frozen=True)
class RouterMatch:
    """One routing decision. `score` is the cosine to the nearest example;
    `margin` is score minus the second-best mode's score (the ambiguity
    signal); `example` is the nearest phrase, for audit/debug."""
    mode: str
    score: float
    margin: float
    example: str


@dataclass(frozen=True)
class RouterDecision:
    """The FULL outcome of one attempt — accepted OR declined, with the losing
    candidate intact.

    Why this exists: `route()` returns None on a decline and throws the
    candidate away, which makes the shadow log blind exactly where calibration
    happens. A query that missed the floor by 0.01 and one that missed by 0.30
    are completely different facts, and only the first tells us where the
    threshold belongs (item 1c). Keeping the near-misses turns every declined
    turn into a data point instead of 1 in 11.

    `reason` is None when accepted, else why it lost:
      'below_score'  — top candidate under min_score (not confident enough)
      'below_margin' — confident but ambiguous (runner-up too close)
      'no_candidate' — nothing to compare (empty query/catalogue)
      'embed_failed' / 'dim_mismatch' — the embedder let us down
    On the last three there is no candidate, so mode/example are None."""
    mode: Optional[str]
    score: float
    margin: float
    example: Optional[str]
    accepted: bool
    reason: Optional[str]

    def as_match(self) -> Optional["RouterMatch"]:
        """The RouterMatch this decision represents, or None if declined —
        so `route()` stays a thin wrapper and the two can never disagree."""
        if not self.accepted or self.mode is None:
            return None
        return RouterMatch(
            mode=self.mode, score=self.score,
            margin=self.margin, example=self.example or "",
        )


# Confidence floor and ambiguity margin, env-tunable. The floor is the one
# calibrated for routing (item 1c, s83: 614 real sentences, first false
# positive at 0.719). The default was still 0.62, the pre-calibration guess:
# CORTEX_SEMANTIC_ROUTER=1 alone — the Mac Studio runbook's step — routed at
# it, where "pode parar" (0.657) pauses Spotify (s88, Codex lote 8).
_CALIBRATED_MIN_SCORE = "0.74"
_DEFAULT_MIN_SCORE = float(os.getenv("CORTEX_SEMANTIC_MIN_SCORE",
                                     _CALIBRATED_MIN_SCORE))
_DEFAULT_MIN_MARGIN = float(os.getenv("CORTEX_SEMANTIC_MIN_MARGIN", "0.04"))


# Starter catalogue — the intents where a missed phrasing hurts most, with
# diverse PT+EN phrasings. NOT exhaustive: the registry has ~50 modes; this
# covers the high-traffic ones and grows as shadow data shows real misses.
# Each phrase is a thing a user might actually say, not a regex.
DEFAULT_CATALOG: Tuple[IntentExamples, ...] = (
    IntentExamples("STUDIER_LIST", (
        "me mostra minha lista de estudo",
        "quais estudos estão na fila",
        "o que tem pra estudar hoje",
        "lista os estudos pendentes",
        "show my study list",
        "what's in the study queue",
    )),
    IntentExamples("STUDIER_REVIEW", (
        "quero revisar os estudos prontos",
        "me mostra os briefings pra aprovar",
        "revisar estudos",
        "review the finished studies",
    )),
    IntentExamples("JOBS_TOP", (
        "tem vaga nova pra mim",
        "quais são minhas melhores vagas",
        "me mostra as vagas de emprego",
        "any new jobs for me",
        "show my top job matches",
    )),
    IntentExamples("JOBS_RERANK", (
        "re-rankeia minhas vagas",
        "recalcula o ranking das vagas",
        "atualiza a ordem das vagas",
        "re-rank my jobs",
    )),
    IntentExamples("WEATHER", (
        "como está o tempo hoje",
        "vai chover amanhã",
        "qual a previsão do tempo",
        "what's the weather like",
    )),
    IntentExamples("NOTE_LIST", (
        "me mostra minhas notas",
        "lista as anotações",
        "list my notes",
    )),
    IntentExamples("NOTE_CREATE", (
        "anota uma coisa pra mim",
        "cria uma nota nova",
        "take a note",
        "write this down",
    )),
    IntentExamples("SPOTIFY_PLAY", (
        "toca uma música",
        "põe um som pra tocar",
        "play some music",
    )),
    IntentExamples("SPOTIFY_PAUSE", (
        "pausa a música",
        "para o som",
        "pause the music",
    )),
    IntentExamples("MAC_SUMMARISE", (
        "resume isso pra mim",
        "faz um resumo do que está na tela",
        "summarise this",
    )),
    IntentExamples("MAC_WHAT_IS", (
        "o que é isso na tela",
        "que negócio é esse aqui",
        "what is this on screen",
    )),
    IntentExamples("FITNESS_MACROS", (
        "quantos macros eu comi hoje",
        "meus macros do dia",
        "how many calories today",
    )),
    # LOGGING a meal is a different intent from READING the day's macros, and
    # the catalogue lacked it entirely (s82): a real "hoje eu comi 55 gramas
    # de aveia…" landed nearest to FITNESS_MACROS at 0.617 — so even a lower
    # floor would have routed it to the wrong handler (read instead of write).
    # Phrasings deliberately carry the leading time/subject words the regex
    # used to reject, since that is exactly the gap this layer exists to cover.
    IntentExamples("FITNESS_MEAL", (
        "hoje eu comi aveia com banana",
        "comi 200 gramas de frango no almoço",
        "registra que eu comi um ovo",
        "acabei de comer um sanduíche",
        "i ate two eggs this morning",
        "log my breakfast",
    )),
)


def _unit(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _no_candidate(reason: str) -> "RouterDecision":
    """A decline with nothing to report — the query never reached scoring."""
    return RouterDecision(
        mode=None, score=0.0, margin=0.0, example=None,
        accepted=False, reason=reason,
    )


class SemanticRouter:
    """Routes a query to the nearest dispatch intent by embedding similarity."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        catalog: Sequence[IntentExamples] = DEFAULT_CATALOG,
        min_score: Optional[float] = None,
        min_margin: Optional[float] = None,
    ):
        self._embed = embed_fn
        self.min_score = _DEFAULT_MIN_SCORE if min_score is None else min_score
        self.min_margin = _DEFAULT_MIN_MARGIN if min_margin is None else min_margin
        # Embed every example ONCE at construction, grouped by mode. Unit-
        # normalised so routing is a cheap dot product per query.
        self._modes: List[str] = []
        self._vecs: List["np.ndarray"] = []   # unit vectors
        self._phrases: List[str] = []
        for entry in catalog:
            for phrase in entry.phrases:
                try:
                    vec = _unit(np.asarray(self._embed(phrase), dtype=np.float32))
                except Exception:
                    log.exception("SemanticRouter: failed to embed %r", phrase)
                    continue
                self._modes.append(entry.mode)
                self._vecs.append(vec)
                self._phrases.append(phrase)
        self._matrix = np.vstack(self._vecs) if self._vecs else np.empty((0, 0))

    def route(self, query: str) -> Optional[RouterMatch]:
        """Return the best intent match, or None if no intent is confident and
        unambiguous enough (→ caller falls through to the LLM). Never raises.

        Thin wrapper over `explain()` so the live path and the shadow log can
        never disagree about what would have been routed."""
        return self.explain(query).as_match()

    def explain(self, query: str) -> RouterDecision:
        """Same computation as `route()`, but ALWAYS reports the best candidate
        and why it won or lost — the shadow log's calibration data (see
        RouterDecision). Never raises; a broken embedder is a decline."""
        query = (query or "").strip()
        if not query or self._matrix.size == 0:
            return _no_candidate("no_candidate")
        try:
            q = _unit(np.asarray(self._embed(query), dtype=np.float32))
        except Exception:
            log.exception("SemanticRouter: query embed failed (non-fatal)")
            return _no_candidate("embed_failed")
        if q.shape[0] != self._matrix.shape[1]:
            # embedder dim changed under us — decline, don't crash
            return _no_candidate("dim_mismatch")

        sims = self._matrix @ q  # cosine, all vectors are unit
        # Best cosine PER MODE (nearest-example), then rank modes.
        best_per_mode: Dict[str, Tuple[float, str]] = {}
        for i, mode in enumerate(self._modes):
            s = float(sims[i])
            if mode not in best_per_mode or s > best_per_mode[mode][0]:
                best_per_mode[mode] = (s, self._phrases[i])
        ranked = sorted(best_per_mode.items(), key=lambda kv: -kv[1][0])

        top_mode, (top_score, top_phrase) = ranked[0]
        second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = top_score - second_score

        # Confidence first, then ambiguity: when both fail, 'below_score' is
        # the primary fact (raising the floor subsumes it).
        reason: Optional[str] = None
        if top_score < self.min_score:
            reason = "below_score"
        elif margin < self.min_margin:
            reason = "below_margin"

        return RouterDecision(
            mode=top_mode, score=top_score, margin=margin, example=top_phrase,
            accepted=reason is None, reason=reason,
        )
