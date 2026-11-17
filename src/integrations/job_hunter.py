"""
Job Hunter daily refresh (slice 5 v2, s76).

Ties the three halves together for the unattended morning pull: FETCH fresh
apprenticeships from the gov.uk board (ats), KEEP those matching ANY keyword in
the title AND an allowed apprenticeship level, UPSERT into the local ledger
(dedup by URL preserves an existing score/status — only genuinely new postings
arrive as status="new"), then RANK only the new ones against the stored résumé.

`src/daily_driver.py` calls this fail-soft before assembling the briefing, so the
"new top matches" it reads are genuinely fresh. It's a PUBLIC read (job postings
come in, nothing personal goes out — the résumé stays local in the ranker,
SOBERANIA r1) and AUDITED like the other proactive background reads (M7-A4).
The HTTP transport and the embedder are injectable, so tests run fully offline.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Sequence

from src.integrations import ats
from src.integrations.job_ranker import rank_jobs
from src.integrations.jobs import JobLog, get_log

log = logging.getLogger(__name__)

# The user's standing search (s76 cont.43): SWE-adjacent apprenticeships near home,
# at the levels he can enter. Keywords match the TITLE (ANY = union); level is
# the FAA course.level. Distance is MILES (the API has no time-based radius).
# These are the fallback DEFAULTS; each is tunable via env (CORTEX_JOBS_*) so the
# standing search can be retargeted without editing code (s76 cont.44).
DEFAULT_KEYWORDS = ("software", "developer", "data", "ai", "it",
                    "cyber", "network", "business")
DEFAULT_LEVELS = (3, 4)
DEFAULT_LOCATION = "reading"
DEFAULT_DISTANCE_MILES = 30

_ZERO = {"fetched": 0, "matched": 0, "new": 0, "ranked": 0}


# --- env-tunable defaults (read at call time; explicit kwargs still win) -------
# Precedence per field: explicit kwarg > env (CORTEX_JOBS_*) > module DEFAULT.
# Empty/invalid env falls back to the DEFAULT — a malformed override never blanks
# the search (which would match nothing / hit the wrong geo).

def _env_keywords() -> Sequence[str]:
    """CORTEX_JOBS_KEYWORDS: comma-separated TITLE keywords (ANY matches)."""
    parts = tuple(k.strip() for k in os.getenv("CORTEX_JOBS_KEYWORDS", "").split(",")
                  if k.strip())
    return parts or DEFAULT_KEYWORDS


def _env_levels() -> Sequence[int]:
    """CORTEX_JOBS_LEVELS: comma-separated apprenticeship levels, e.g. "3,4"."""
    lvls = tuple(int(t) for t in os.getenv("CORTEX_JOBS_LEVELS", "").split(",")
                 if t.strip().isdigit())
    return lvls or DEFAULT_LEVELS


def _env_location() -> str:
    """CORTEX_JOBS_LOCATION: home city for the FAA geo search."""
    return os.getenv("CORTEX_JOBS_LOCATION", "").strip() or DEFAULT_LOCATION


def _env_distance() -> int:
    """CORTEX_JOBS_DISTANCE_MI: search radius in miles (positive int)."""
    raw = os.getenv("CORTEX_JOBS_DISTANCE_MI", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_DISTANCE_MILES


def refresh_jobs(*, keywords: Optional[Sequence[str]] = None,
                 levels: Optional[Sequence[int]] = None,
                 location: Optional[str] = None,
                 distance_miles: Optional[int] = None,
                 joblog: Optional[JobLog] = None,
                 embed: Optional[Callable] = None,
                 fetch_json: Optional[Callable] = None) -> dict:
    """Fetch → filter (keyword in title ∧ level) → upsert → rank the new ones.

    Search params default to env (CORTEX_JOBS_*) → module DEFAULTs; an explicit
    kwarg overrides both (tests pin them this way). Returns
    {"fetched","matched","new","ranked"}. Fully fail-soft: any failure returns
    zeros so the caller (the morning briefing) just falls back to whatever the
    ledger already holds."""
    keywords = keywords if keywords is not None else _env_keywords()
    levels = levels if levels is not None else _env_levels()
    location = location if location is not None else _env_location()
    distance_miles = distance_miles if distance_miles is not None else _env_distance()
    if location.lower() not in ats.FAA_GEO_CITIES:
        log.warning("job_hunter: location %r has no known geo centre — the search "
                    "will be NATIONAL (no distance filter). Known cities: %s",
                    location, ", ".join(sorted(ats.FAA_GEO_CITIES)))
    kws = [k.lower() for k in keywords if k]
    lvls = set(levels)
    jl = joblog or get_log()

    # Proactive external read → audit breadcrumb (this runs unattended under
    # launchd; the log makes its gov.uk traffic visible). Read-only: audit, not gate.
    try:
        from src.core.audit import audit
        audit.log("–", "background_read", "jobs_refresh", payload={
            "source": "findapprenticeship", "location": location,
            "distance_mi": distance_miles, "keywords": kws, "levels": sorted(lvls),
        })
    except Exception:  # noqa: BLE001 — audit must never break the refresh
        pass

    # A failed fetch carries "error" — the counts alone said the same for "the
    # site changed its reply", "the network was down" and "no vacancy today"
    # (s88, Codex lote 11).
    try:
        postings = ats.fetch("findapprenticeship", location,
                             distance_miles=distance_miles, fetch_json=fetch_json)
    except ats.UnexpectedResponse:
        log.exception("job_hunter: the site's reply changed shape")
        return {**_ZERO, "error": "format"}
    except Exception:  # noqa: BLE001 — network/HTTP/JSON
        log.exception("job_hunter: fetch failed")
        return {**_ZERO, "error": "fetch"}

    matched = [
        p for p in postings
        if p.level in lvls and any(ats.keyword_in_text(p.title, k) for k in kws)
    ]
    if not matched:
        return {**_ZERO, "fetched": len(postings)}

    try:
        new, _ = jl.upsert(matched)
        ranked = rank_jobs(jl, embed=embed, only_unscored=True)
    except Exception:  # noqa: BLE001 — store/embed failure leaves the ledger as-is
        log.exception("job_hunter: upsert/rank failed")
        return {**_ZERO, "fetched": len(postings), "matched": len(matched)}

    return {"fetched": len(postings), "matched": len(matched),
            "new": new, "ranked": ranked}
