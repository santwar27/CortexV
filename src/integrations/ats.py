"""
ATS job-board adapters for the Autonomous Job Hunter (slice 1, s76).

Fetch PUBLIC job postings from Applicant Tracking Systems that expose stable
JSON boards — Greenhouse, Lever and Ashby (token-based), plus gov.uk Find an
Apprenticeship (geo-based). We deliberately target ATS endpoints instead of
scraping Indeed/career pages: the JSON is documented,
auth-free and far more stable than fighting bot-detection (see
memory/ideas_backlog.md "Autonomous Job Hunter").

Privacy/network (SOBERANIA r1): job postings are PUBLIC — they come in, nothing
personal goes out. This is a plain read (like web_search), not an S3 action.
The HTTP transport is INJECTABLE (`fetch_json`) so tests run fully offline; the
resume half (slice 2) is what stays local forever.
"""
from __future__ import annotations

import html
import logging
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urlencode

from src.core import turn_metrics

log = logging.getLogger(__name__)

_HTTP_TIMEOUT_S = 12
_USER_AGENT = "Cortex-JobHunter/1.0 (personal job search)"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class JobPosting:
    source: str          # ats name, e.g. "greenhouse"
    board: str           # board token, e.g. "stripe"
    company: str         # display name (falls back to the board token)
    ext_id: str          # the ATS's own job id (for reference/dedup hints)
    title: str
    location: str
    url: str             # canonical posting URL — the dedup key in JobLog
    description: str     # plain text (HTML stripped)
    posted_at: Optional[str]
    distance: Optional[float] = None   # miles from the search centre (geo sources)
    level: Optional[int] = None        # apprenticeship level (FAA course.level: 2/3/4/6/7)
    closing_at: Optional[str] = None   # application deadline (ISO; FAA closingDate)


# Type of the injectable transport: url -> parsed JSON dict.
FetchJson = Callable[[str], dict]


def _http_get_json(url: str) -> dict:
    """Production transport. Imported lazily so tests never need `requests`."""
    import requests

    turn_metrics.mark_online("jobs")  # Mark before a possibly partial request.
    resp = requests.get(url, timeout=_HTTP_TIMEOUT_S,
                        headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def strip_html(raw: str) -> str:
    """ATS description HTML (often entity-escaped) -> readable plain text."""
    s = html.unescape(raw or "")     # &lt;p&gt; -> <p>
    s = _TAG_RE.sub(" ", s)          # drop tags
    s = html.unescape(s)             # decode any entities that were inside text
    return _WS_RE.sub(" ", s).strip()


def _first(d: dict, *keys: str, default: str = "") -> str:
    """First non-empty value among `keys` (tolerant field mapping for ATS feeds
    whose exact key names vary)."""
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default


def _epoch_ms_to_date(v: object) -> Optional[str]:
    """Lever's `createdAt` is epoch-millis → "YYYY-MM-DD". Non-numeric/zero → None."""
    if isinstance(v, (int, float)) and v > 0:
        try:
            return datetime.fromtimestamp(v / 1000).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return None
    return None


# --- application deadlines (FAA closingDate) ---------------------------------
# The single most actionable field for an apprenticeship hunt: miss the closing
# date and the application is gone. Shared by every surface (voice, `make jobs`,
# the morning briefing) so the phrasing stays in one place.

def days_until_close(closing_at: Optional[str]) -> Optional[int]:
    """CALENDAR days from today until the closing date, in local time. 0 = closes
    today, 1 = tomorrow, NEGATIVE = already past, None = absent/unparseable.

    Counts whole calendar days (not 24h periods) so a deadline at 23:59 today
    reads as "today", not "tomorrow" — and so it never under-counts by one the
    way a raw timedelta would (4.9 days → 4). Fail-soft: a weird date never
    raises into a surface.

    Uses the date AS THE SOURCE WROTE IT, without converting the timezone
    (s83). The FAA states a closing DAY and stamps it "23:59:59Z"; converting
    that to British Summer Time lands on 00:59 the NEXT day, so every summer
    deadline read one day long — Cortex said "fecha em 4 dias" for a vacancy
    whose own GOV.UK page said "Closes in 3 days (Friday 2 October)". A
    deadline is a date on a calendar, not an instant."""
    if not closing_at:
        return None
    try:
        # Accept the FAA's trailing "Z" on every Python version.
        dt = datetime.fromisoformat(str(closing_at).strip().replace("Z", "+00:00"))
        return (dt.date() - datetime.now().astimezone().date()).days
    except Exception:
        return None


def format_deadline(closing_at: Optional[str], lang: str = "pt") -> str:
    """Human 'closes in N days' phrase for voice/text. '' when there's no usable
    date (so a surface can just append it). PT default; 'en' for English."""
    d = days_until_close(closing_at)
    if d is None:
        return ""
    if lang == "en":
        if d < 0:
            return "closed"
        return {0: "closes today", 1: "closes tomorrow"}.get(d, f"closes in {d} days")
    if d < 0:
        return "encerrada"
    return {0: "fecha hoje", 1: "fecha amanhã"}.get(d, f"fecha em {d} dias")


def is_deadline_urgent(closing_at: Optional[str], within_days: int = 3) -> bool:
    """True when the deadline is open but ≤ within_days away — drives an urgency
    flag in the terminal listing."""
    d = days_until_close(closing_at)
    return d is not None and 0 <= d <= within_days


def is_closed(closing_at: Optional[str]) -> bool:
    """True only when we KNOW the vacancy has already closed.

    The single definition of "closed" for every surface that recommends a job
    (s82) — the conversation handler and the morning briefing both filter on
    this, so they can never drift apart about what counts as dead.

    Deliberately asymmetric: an ABSENT or unparseable date is NOT closed. Many
    boards omit a deadline entirely, and hiding those would silently shrink the
    list to nothing. We only drop what we can prove is past; anything uncertain
    still gets shown, deadline phrase and all."""
    d = days_until_close(closing_at)
    return d is not None and d < 0


class UnexpectedResponse(ValueError):
    """The site answered, but not in the shape the adapter reads. Not the
    same as "no vacancies": a Greenhouse reply moved under {"data": …} read as
    an empty board, and the command said "Nenhuma vaga" (s88, Codex lote 11 —
    invariant 7)."""


def _list_at(data, *keys: str) -> list:
    """The posting list under the first of `keys` present — an empty list is a
    valid empty board — or UnexpectedResponse when none is there."""
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]
    raise UnexpectedResponse(f"expected a list under {' or '.join(keys)!r}")


def fetch_greenhouse(board_token: str, *,
                     fetch_json: Optional[FetchJson] = None,
                     **_opts) -> List[JobPosting]:
    """Public Greenhouse board → JobPostings.

    Endpoint: boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
    (content=true includes the HTML job description)."""
    board_token = (board_token or "").strip()
    if not board_token:
        return []
    url = (f"https://boards-api.greenhouse.io/v1/boards/"
           f"{board_token}/jobs?content=true")
    data = (fetch_json or _http_get_json)(url)
    out: List[JobPosting] = []
    for j in _list_at(data, "jobs"):
        loc = (j.get("location") or {}).get("name", "") or ""
        out.append(JobPosting(
            source="greenhouse",
            board=board_token,
            company=(j.get("company_name") or board_token),
            ext_id=str(j.get("id", "")),
            title=(j.get("title") or "").strip(),
            location=loc.strip(),
            url=(j.get("absolute_url") or "").strip(),
            description=strip_html(j.get("content", "")),
            posted_at=(j.get("updated_at") or j.get("first_published")),
        ))
    return out


def fetch_lever(board_token: str, *,
                fetch_json: Optional[FetchJson] = None,
                **_opts) -> List[JobPosting]:
    """Public Lever board → JobPostings.

    Endpoint: api.lever.co/v0/postings/{token}?mode=json — a flat JSON ARRAY
    (no envelope). The feed carries no company display name, so `company` falls
    back to the board token; `categories.location` holds the office, `createdAt`
    is epoch-millis, and `descriptionPlain` is already plain text."""
    board_token = (board_token or "").strip()
    if not board_token:
        return []
    url = f"https://api.lever.co/v0/postings/{board_token}?mode=json"
    data = (fetch_json or _http_get_json)(url)
    out: List[JobPosting] = []
    if not isinstance(data, list):
        raise UnexpectedResponse("expected a list of postings")
    for j in data:
        cats = j.get("categories") or {}
        out.append(JobPosting(
            source="lever",
            board=board_token,
            company=board_token,
            ext_id=str(j.get("id", "")),
            title=(j.get("text") or "").strip(),
            location=(cats.get("location") or "").strip(),
            url=_first(j, "hostedUrl", "applyUrl").strip(),
            description=(j.get("descriptionPlain")
                        or strip_html(_first(j, "description"))),
            posted_at=_epoch_ms_to_date(j.get("createdAt")),
        ))
    return out


def fetch_ashby(board_token: str, *,
                fetch_json: Optional[FetchJson] = None,
                **_opts) -> List[JobPosting]:
    """Public Ashby board → JobPostings.

    Endpoint: api.ashbyhq.com/posting-api/job-board/{token} → {"jobs": [...]}.
    `descriptionPlain` is already plain text, `location` is a flat string, and
    `publishedAt` is ISO. No company name in the feed → board token. Titles can
    carry a stray leading space, so we strip."""
    board_token = (board_token or "").strip()
    if not board_token:
        return []
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_token}"
    data = (fetch_json or _http_get_json)(url)
    out: List[JobPosting] = []
    for j in _list_at(data, "jobs"):
        out.append(JobPosting(
            source="ashby",
            board=board_token,
            company=board_token,
            ext_id=str(j.get("id", "")),
            title=(j.get("title") or "").strip(),
            location=(j.get("location") or "").strip(),
            url=_first(j, "jobUrl", "applyUrl").strip(),
            description=(j.get("descriptionPlain")
                        or strip_html(_first(j, "descriptionHtml"))),
            posted_at=(j.get("publishedAt") or None),
        ))
    return out


def fetch_recruitee(board_token: str, *,
                    fetch_json: Optional[FetchJson] = None,
                    **_opts) -> List[JobPosting]:
    """Public Recruitee board → JobPostings.

    Endpoint: {token}.recruitee.com/api/offers/ → {"offers": [...]}. `careers_url`
    is the public posting (the dedup key); `description` is HTML; `created_at` is
    ISO; location is a flat `location` string, else city + country_code. No
    company display name in the feed → board token. Recruitee hosts mostly
    FULL-TIME roles (not UK apprenticeships) — added for forward-looking SWE-role
    tracking. PROVISIONAL: the mapping is tolerant (several candidate keys);
    calibrate against a live board via `fetch_jobs.py recruitee <token> --raw`."""
    board_token = (board_token or "").strip()
    if not board_token:
        return []
    url = f"https://{board_token}.recruitee.com/api/offers/"
    data = (fetch_json or _http_get_json)(url)
    out: List[JobPosting] = []
    for j in _list_at(data, "offers"):
        loc = (j.get("location") or "").strip()
        if not loc:
            loc = ", ".join(p for p in (j.get("city"), j.get("country_code")) if p)
        out.append(JobPosting(
            source="recruitee",
            board=board_token,
            company=board_token,
            ext_id=str(j.get("id", "")),
            title=(j.get("title") or "").strip(),
            location=loc,
            url=_first(j, "careers_url", "careers_apply_url", "url").strip(),
            description=strip_html(_first(j, "description", "requirements")),
            posted_at=(_first(j, "created_at", "published_at") or None),
        ))
    return out


# --- gov.uk Find an Apprenticeship (official "Display Vacancy Advert" API) ---
#
# THE source for UK apprenticeships (Greenhouse tech boards don't carry them).
# Endpoint + headers are confirmed; the RESPONSE schema and the exact filter
# param names live behind the developer-portal sign-in, so the field mapping
# below is TOLERANT (tries several candidate keys) and meant to be calibrated
# against a real payload via `fetch_jobs.py findapprenticeship --raw` once the user
# has a key (CORTEX_FAA_KEY) — same dance as the Health Auto Export sink.

_FAA_URL = "https://api.apprenticeships.education.gov.uk/vacancies/vacancy"
_FAA_KEY_ENV = "CORTEX_FAA_KEY"
# Known city centres so "--location reading" drives the server-side geo filter
# (lat/lon) and the API's per-vacancy `distance` becomes "miles from there".
_FAA_GEO = {
    "london": (51.5074, -0.1278),
    "manchester": (53.4808, -2.2426),
    "birmingham": (52.4862, -1.8904),
    "reading": (51.4543, -0.9781),
}
# Cities whose geo centre is known — anything else means a NATIONAL search (no
# distance filter). Public so callers can validate a configured location.
FAA_GEO_CITIES = frozenset(_FAA_GEO)


def _faa_transport(api_key: str) -> FetchJson:
    def _get(url: str) -> dict:
        import requests
        turn_metrics.mark_online("jobs")  # FAA has its own HTTP transport.
        resp = requests.get(url, timeout=_HTTP_TIMEOUT_S, headers={
            "User-Agent": _USER_AGENT,
            "Ocp-Apim-Subscription-Key": api_key,
            "X-Version": "2",
            "Accept": "application/json",
        })
        resp.raise_for_status()
        return resp.json()
    return _get


def _map_faa(v: dict) -> JobPosting:
    """One FAA vacancy → JobPosting. Field names verified against a live v2
    payload (s76 cont.36): addresses[] (plural), vacancyUrl, vacancyReference,
    employerName, postedDate; description is already plain text."""
    addrs = v.get("addresses") or []
    addr = addrs[0] if addrs else {}
    loc = ", ".join(p for p in (addr.get("addressLine3"),
                                addr.get("addressLine4"),
                                addr.get("postcode")) if p)
    if not loc and v.get("isNationalVacancy"):
        loc = "National / Remote"
    ref = _first(v, "vacancyReference", "reference", "id")
    url = _first(v, "vacancyUrl", "url")
    if not url and ref:
        url = ("https://www.findapprenticeship.service.gov.uk/"
               f"apprenticeship/reference/{ref}")
    dist = v.get("distance")
    lvl = (v.get("course") or {}).get("level")
    return JobPosting(
        source="findapprenticeship",
        board="uk",
        company=_first(v, "employerName", "providerName", default="?"),
        ext_id=str(ref),
        title=_first(v, "title"),
        location=loc,
        url=url,
        description=strip_html(_first(v, "description")),
        posted_at=(_first(v, "postedDate", "startDate") or None),
        distance=(float(dist) if isinstance(dist, (int, float)) else None),
        level=(int(lvl) if isinstance(lvl, (int, float)) else None),
        closing_at=(_first(v, "closingDate") or None),
    )


def fetch_findapprenticeship(location: str = "", *,
                             fetch_json: Optional[FetchJson] = None,
                             api_key: Optional[str] = None,
                             distance_miles: int = 20,
                             page_size: int = 100,
                             max_pages: int = 10,
                             raw_sink: Optional[Callable[[dict], None]] = None,
                             **_opts) -> List[JobPosting]:
    """gov.uk apprenticeship vacancies → JobPostings.

    Needs a subscription key (arg or env CORTEX_FAA_KEY); without one it
    fail-softs to []. `location` (e.g. "london") drives the SERVER-side geo
    filter (lat/lon/distanceInMiles) when the city is known — the API has no
    free-text search, so keyword narrowing is done client-side by the caller.
    Paginates until a short page or `max_pages`. `raw_sink` (used by --raw) gets
    the first raw JSON response."""
    api_key = (api_key or os.getenv(_FAA_KEY_ENV) or "").strip()
    if not api_key and fetch_json is None:
        log.warning("findapprenticeship: no API key — set %s in .env", _FAA_KEY_ENV)
        return []

    geo = _FAA_GEO.get((location or "").strip().lower())
    base: Dict[str, object] = {"pageSize": page_size}
    if geo:
        base["lat"], base["lon"] = geo[0], geo[1]
        base["distanceInMiles"] = distance_miles
        base["sort"] = "DistanceAsc"     # nearest first, server-side

    get = fetch_json or _faa_transport(api_key)
    out: List[JobPosting] = []
    for page in range(1, max_pages + 1):
        params = dict(base, pageNumber=page)
        data = get(f"{_FAA_URL}?{urlencode(params)}")
        if page == 1 and raw_sink is not None:
            raw_sink(data)
        items = _list_at(data, "vacancies", "results")
        out.extend(_map_faa(v) for v in items)
        if len(items) < page_size:
            break
    return out


def keyword_in_text(text: str, keyword: str) -> bool:
    """Whole-WORD, case-insensitive keyword match. Short keywords like "AI" or
    "IT" match as TOKENS ("AI Engineer", "IT Support Apprentice") and never as
    substrings ("trAIning", "rep**ai**r", "qua**lit**y", "commun**it**y") — the
    substring approach made "AI"/"IT" unusable. Punctuation/spaces are word
    boundaries, so "Data & AI" and "data-driven" both hit "data"/"AI"."""
    kw = (keyword or "").strip()
    if not kw:
        return False
    return re.search(r"\b" + re.escape(kw) + r"\b", text or "", re.IGNORECASE) is not None


def matches(p: JobPosting, *, keywords: Optional[List[str]] = None,
            location: Optional[str] = None, title_only: bool = False) -> bool:
    """True if the posting passes the filter.

    `keywords` — ALL must appear, as whole WORDS (case-insensitive; see
    keyword_in_text). By default the haystack is title + description;
    `title_only=True` matches the TITLE only, which cuts noise (e.g. an admin
    role that merely mentions "software" in its body). `location` — substring of
    the posting's location ("london" matches "London, UK"). Empty filter → True."""
    if keywords:
        hay = p.title if title_only else f"{p.title} {p.description}"
        if not all(keyword_in_text(hay, k) for k in keywords if k):
            return False
    if location:
        if location.lower() not in (p.location or "").lower():
            return False
    return True


def filter_postings(postings: List[JobPosting], *,
                    keywords: Optional[List[str]] = None,
                    location: Optional[str] = None,
                    title_only: bool = False) -> List[JobPosting]:
    return [p for p in postings
            if matches(p, keywords=keywords, location=location,
                       title_only=title_only)]


# Registry. greenhouse/lever/ashby take a board TOKEN as `query`;
# findapprenticeship takes a geo (city) instead.
ADAPTERS: Dict[str, Callable[..., List[JobPosting]]] = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
    "recruitee": fetch_recruitee,
    "findapprenticeship": fetch_findapprenticeship,
}

# Sources whose `query` is a board token (so the CLI can demand one upfront).
TOKEN_SOURCES = frozenset({"greenhouse", "lever", "ashby", "recruitee"})


def fetch(source: str, query: str = "", *,
          fetch_json: Optional[FetchJson] = None, **opts) -> List[JobPosting]:
    """Dispatch to the adapter for `source`. `query` is the board token for
    Greenhouse / the location for Find an Apprenticeship; `opts` pass through
    (api_key, search_term, distance_miles, raw_sink, ...). Unknown source → []
    (fail-soft)."""
    adapter = ADAPTERS.get((source or "").strip().lower())
    if adapter is None:
        log.warning("ats: unknown source %r (have: %s)",
                    source, ", ".join(sorted(ADAPTERS)))
        return []
    return adapter(query, fetch_json=fetch_json, **opts)
