"""
Temporal retrieval helpers — pure logic, no SQLite touch.

Three pieces feed Palace.search() when temporal-aware retrieval is on:

  1. detect_temporal_intent(query) — pt-BR/EN regex sweep over the query
     to spot windows ("ontem", "semana passada") and decay-off cues
     ("sempre", "geralmente", "prefiro"). Returns a TemporalIntent that
     Palace consumes when assembling its SQL + ranking.

  2. compute_age_days(created_at_iso, now) — float days between the
     document's ISO timestamp and a reference `now`. Robust to stored
     timestamps with or without explicit UTC offset; assumes UTC when
     no tz is present. Returns 0.0 for unparseable inputs so a corrupt
     row never wrecks the whole search.

  3. _TIPO_LAMBDAS — per-tipo decay constants used as `exp(-λ × age)`.
     Values from design-reference ch. 7: conversation (episodic) decays fast,
     knowledge (project context) decays slow, profile-like content
     barely decays at all. Unknown tipos fall back to the conservative
     default.

Reference: design-reference ch. 7 (`memory/retrieval.py`).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------
# decay constants (λ in `exp(-λ × age_days)`)
# ---------------------------------------------------------------------
#
# design-reference ch. 7 half-lives, ported to Cortex tipo names:
#   conversation  — episodic chat turns; half-life ~7 days  → λ ≈ 0.099
#   knowledge     — approved briefings; half-life ~14 days  → λ ≈ 0.0495
#   profile       — preferences and identity; near-atemporal
#   _default      — conservative slow decay for unknown tipos
#
# Half-life-to-lambda conversion: λ = ln(2)/H. We round to the reference's
# canonical values for clarity rather than carrying long decimals.

_TIPO_LAMBDAS = {
    "conversation": 0.10,
    "knowledge":    0.05,
    "profile":      0.005,
    "_default":     0.05,
}


def lambda_for(tipo: Optional[str]) -> float:
    """Return the decay constant for a document tipo, with safe fallback."""
    if tipo is None:
        return _TIPO_LAMBDAS["_default"]
    return _TIPO_LAMBDAS.get(tipo, _TIPO_LAMBDAS["_default"])


# ---------------------------------------------------------------------
# temporal intent detection
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalIntent:
    """Structured outcome of inspecting a query for temporal cues.

    decay_off      — true when the query is about long-standing preferences
                     ("sempre", "geralmente"). The exp(-λ × age) modulator
                     is skipped so older but matching docs stay in play.
    window_days    — when set, hard-filter candidates to docs whose
                     created_at falls within `now - window_days` to `now`.
                     None means no window filter.
    query_stripped — query string with the temporal cue removed, so the
                     downstream channels (semantic + FTS5) don't try to
                     match the cue word against document content. Equal
                     to None when no cue fired, in which case callers
                     should fall back to the original query.
    label          — human-readable tag for logging/debugging.
    """
    decay_off: bool = False
    window_days: Optional[int] = None
    query_stripped: Optional[str] = None
    label: str = "default"

    def effective_query(self, original: str) -> str:
        """Convenience: return query_stripped when set, else the original."""
        if self.query_stripped is not None and self.query_stripped.strip():
            return self.query_stripped
        return original


# Word-boundary regex (Unicode safe). Each entry maps a pattern to either
# a window in days or the decay_off sentinel. Order matters: longer/more
# specific patterns come first so "semana passada" wins over "passada".
#
# Windows are conservative — `ontem` legitimately covers ~2 calendar days
# depending on TZ + when the user asks; we include today as a safety buffer.

_WINDOW_PATTERNS = (
    # === pt-BR ===
    (re.compile(r"\b(semana\s+passada|[úu]ltima\s+semana)\b", re.IGNORECASE), 8,  "last_week"),
    (re.compile(r"\b(m[êe]s\s+passado|[úu]ltimo\s+m[êe]s)\b",  re.IGNORECASE), 31, "last_month"),
    (re.compile(r"\b(ontem)\b",                                 re.IGNORECASE), 2,  "yesterday"),
    (re.compile(r"\b(hoje|agora\s+de\s+manh[ãa]|esta\s+manh[ãa]|esta\s+tarde|esta\s+noite)\b",
                                                                 re.IGNORECASE), 1,  "today"),
    # === EN (lightweight — the user mostly types pt-BR, but the engine sees both) ===
    (re.compile(r"\b(last\s+week)\b",                            re.IGNORECASE), 8,  "last_week"),
    (re.compile(r"\b(last\s+month)\b",                           re.IGNORECASE), 31, "last_month"),
    (re.compile(r"\b(yesterday)\b",                              re.IGNORECASE), 2,  "yesterday"),
    (re.compile(r"\b(today|this\s+morning|this\s+afternoon|tonight)\b",
                                                                 re.IGNORECASE), 1,  "today"),
)

# "Sempre / geralmente / prefiro" and English cousins disable decay so
# profile-like documents (which may be old but still relevant) stay
# competitive. We intentionally do NOT add a window — the user wants
# the long-running fact, wherever it lives in history.

_DECAY_OFF_RE = re.compile(
    r"\b("
    r"sempre|geralmente|prefiro|costumo|minha\s+prefer[êe]ncia|sou\s+(o|um|uma)|"
    r"always|usually|i\s+prefer|i\s+usually|my\s+preference"
    r")\b",
    re.IGNORECASE,
)


_WS_COLLAPSE_RE = re.compile(r"\s+")


def _strip_cue(query: str, pattern: re.Pattern) -> str:
    """Remove the matched cue from `query` and collapse whitespace."""
    stripped = pattern.sub(" ", query)
    return _WS_COLLAPSE_RE.sub(" ", stripped).strip()


def detect_temporal_intent(query: str) -> TemporalIntent:
    """Inspect `query` for temporal cues. Default = no-op intent (full
    history, normal decay). Window cues take precedence over decay_off:
    if a user says "lembra ontem o que eu sempre disse", we honor the
    "ontem" window because it's the more specific constraint.

    When a cue fires, the cue text is removed from `query_stripped` so
    downstream channels (semantic + FTS5) can run against just the
    content terms — otherwise BM25 would AND the cue word against every
    document, killing the query."""
    if not isinstance(query, str) or not query.strip():
        return TemporalIntent()

    for pat, days, label in _WINDOW_PATTERNS:
        if pat.search(query):
            return TemporalIntent(
                decay_off=False,
                window_days=days,
                query_stripped=_strip_cue(query, pat),
                label=label,
            )

    if _DECAY_OFF_RE.search(query):
        return TemporalIntent(
            decay_off=True,
            window_days=None,
            query_stripped=_strip_cue(query, _DECAY_OFF_RE),
            label="decay_off",
        )

    return TemporalIntent()


# ---------------------------------------------------------------------
# age math
# ---------------------------------------------------------------------


def compute_age_days(created_at_iso: Optional[str],
                     now: Optional[datetime] = None) -> float:
    """Return non-negative age in days for an ISO-8601 timestamp.

    - `created_at_iso` is whatever Palace wrote: `datetime.now(timezone.utc).isoformat()`.
    - `now` defaults to current UTC.
    - Missing TZ in the stored string is treated as UTC (the source of
      truth in Palace).
    - Returns 0.0 when the input is unparseable, so a single corrupt row
      cannot blow up the whole search call.
    - Returns 0.0 for timestamps in the future (clock skew between
      writers should not amplify a doc's score by giving it a negative
      age that flips exp(-λ × age) into a boost).
    """
    if not created_at_iso:
        return 0.0
    try:
        ts = datetime.fromisoformat(created_at_iso)
    except (TypeError, ValueError):
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ref = now if now is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = ref - ts
    days = delta.total_seconds() / 86400.0
    return max(0.0, days)


def decay_factor(tipo: Optional[str],
                 created_at_iso: Optional[str],
                 now: Optional[datetime] = None) -> float:
    """Convenience: `exp(-λ_tipo × age_days)` capped to [0, 1]."""
    lam = lambda_for(tipo)
    age = compute_age_days(created_at_iso, now)
    return math.exp(-lam * age)


# How much of a document's semantic score its age can take away (s83). The
# half-lives above were applied as a straight multiplier, and with the median
# conversation turn 118 days old that multiplier was ~1e-6: the semantic
# channel only ever saw the last two weeks, and "where do I work?" could not
# find the turn where the user said it. Age is now a tiebreak, not a verdict —
# a very old document loses at most this share, and between two equally
# relevant ones the newer still wins. Calibrated on the palace probe
# (scripts/palace_retrieval_probe.py), not guessed.
RECENCY_WEIGHT = float(os.getenv("CORTEX_RECENCY_WEIGHT", "0.2"))


def recency_weight(tipo: Optional[str],
                   created_at_iso: Optional[str],
                   now: Optional[datetime] = None) -> float:
    """Multiplier for a semantic score: `(1 - w) + w × decay_factor`, in
    [1 - w, 1]. w = 1 restores the old pure-decay ranking."""
    w = min(max(RECENCY_WEIGHT, 0.0), 1.0)
    return (1.0 - w) + w * decay_factor(tipo, created_at_iso, now)
