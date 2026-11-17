"""ATS adapter tests (Autonomous Job Hunter slice 1, s76).

Fully offline — the HTTP transport is injected, so no network is touched.
"""
from src.integrations import ats
from src.integrations.ats import (JobPosting, fetch_ashby,
                                  fetch_findapprenticeship, fetch_greenhouse,
                                  fetch_lever, filter_postings, matches,
                                  strip_html)


def _p(title="Software Engineer Apprentice", location="London, UK",
       description="Join our team"):
    return JobPosting(source="greenhouse", board="acme", company="Acme",
                      ext_id="1", title=title, location=location,
                      url="u", description=description, posted_at=None)


# Shape mirrors a real Greenhouse board (?content=true): content is
# HTML-entity-escaped HTML, location is a nested {"name": ...}.
_FAKE_GH = {
    "jobs": [
        {
            "id": 4567,
            "title": "Senior Backend Engineer",
            "updated_at": "2026-06-10T12:00:00Z",
            "location": {"name": "Remote — EU"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/4567",
            "content": "&lt;p&gt;Build &amp; scale Python services.&lt;/p&gt;",
        },
        {
            "id": 8910,
            "title": "Data Scientist",
            "updated_at": "2026-06-11T09:30:00Z",
            "location": {"name": "Lisbon"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/8910",
            "content": "&lt;ul&gt;&lt;li&gt;ML&lt;/li&gt;&lt;/ul&gt;",
        },
    ]
}


def test_strip_html_decodes_and_strips():
    out = strip_html("&lt;p&gt;Build &amp; scale&lt;/p&gt;")
    assert out == "Build & scale"
    assert "<" not in out and "&lt;" not in out


def test_strip_html_empty():
    assert strip_html("") == "" and strip_html(None) == ""


def test_fetch_greenhouse_parses_fields():
    jobs = fetch_greenhouse("acme", fetch_json=lambda url: _FAKE_GH)
    assert len(jobs) == 2
    j = jobs[0]
    assert isinstance(j, JobPosting)
    assert j.source == "greenhouse" and j.board == "acme"
    assert j.title == "Senior Backend Engineer"
    assert j.location == "Remote — EU"
    assert j.url == "https://boards.greenhouse.io/acme/jobs/4567"
    assert j.ext_id == "4567"
    assert j.description == "Build & scale Python services."   # HTML stripped
    assert j.posted_at == "2026-06-10T12:00:00Z"


def test_fetch_greenhouse_builds_content_url():
    seen = {}

    def spy(url):
        seen["url"] = url
        return {"jobs": []}

    fetch_greenhouse("stripe", fetch_json=spy)
    assert "boards-api.greenhouse.io/v1/boards/stripe/jobs" in seen["url"]
    assert "content=true" in seen["url"]


def test_fetch_greenhouse_empty_and_blank_token():
    assert fetch_greenhouse("acme", fetch_json=lambda url: {"jobs": []}) == []
    assert fetch_greenhouse("", fetch_json=lambda url: _FAKE_GH) == []   # no call


def test_fetch_dispatch_known_and_unknown():
    jobs = ats.fetch("greenhouse", "acme", fetch_json=lambda url: _FAKE_GH)
    assert len(jobs) == 2
    assert ats.fetch("indeed", "acme", fetch_json=lambda url: _FAKE_GH) == []   # unknown


# --- Lever (flat JSON array; schema verified live, s76) -------------------
# createdAt is epoch-millis; descriptionPlain is plain, falls back to stripped
# HTML; company has no display name → board token.
_FAKE_LEVER = [
    {
        "id": "88499546-e9f7-4403-87a5-240050bd7c5b",
        "text": "Accounts Payable Analyst",
        "categories": {"location": "New York, NY", "team": "Accounting",
                       "commitment": "Permanent"},
        "hostedUrl": "https://jobs.lever.co/spotify/88499546",
        "applyUrl": "https://jobs.lever.co/spotify/88499546/apply",
        "createdAt": 1778529611285,
        "descriptionPlain": "You'll join the Finance team.",
        "description": "<p>You'll join the Finance team.</p>",
    },
    {
        "id": "99",
        "text": "Backend Engineer",
        "categories": {"location": "Remote"},
        "hostedUrl": "https://jobs.lever.co/spotify/99",
        "createdAt": 0,                          # 0 → no date
        "descriptionPlain": "",                  # empty → fall back to HTML
        "description": "&lt;p&gt;Build &amp; scale.&lt;/p&gt;",
    },
]


def test_fetch_lever_parses_fields():
    jobs = fetch_lever("spotify", fetch_json=lambda url: _FAKE_LEVER)
    assert len(jobs) == 2
    j = jobs[0]
    assert isinstance(j, JobPosting)
    assert j.source == "lever" and j.board == "spotify" and j.company == "spotify"
    assert j.title == "Accounts Payable Analyst"
    assert j.location == "New York, NY"          # from categories.location
    assert j.url == "https://jobs.lever.co/spotify/88499546"   # hostedUrl wins
    assert j.ext_id == "88499546-e9f7-4403-87a5-240050bd7c5b"
    assert j.description == "You'll join the Finance team."
    assert (j.posted_at or "").startswith("2026-")   # epoch-ms → YYYY-MM-DD


def test_fetch_lever_fallbacks():
    j = fetch_lever("spotify", fetch_json=lambda url: _FAKE_LEVER)[1]
    assert j.description == "Build & scale."      # descriptionPlain empty → HTML
    assert j.posted_at is None                    # createdAt 0 → None
    assert j.location == "Remote"


def test_fetch_lever_builds_url():
    seen = {}
    fetch_lever("spotify", fetch_json=lambda url: seen.update(url=url) or [])
    assert "api.lever.co/v0/postings/spotify" in seen["url"]
    assert "mode=json" in seen["url"]


def test_fetch_lever_empty_and_blank_token():
    assert fetch_lever("spotify", fetch_json=lambda url: []) == []
    assert fetch_lever("", fetch_json=lambda url: _FAKE_LEVER) == []      # no call


# --- Ashby ({"jobs": [...]}; schema verified live, s76) -------------------
# descriptionPlain is plain (falls back to HTML); titles may carry a leading
# space; location is a flat string; company → board token.
_FAKE_ASHBY = {
    "apiVersion": "1",
    "jobs": [
        {
            "id": "34413f8d",
            "title": " Security Engineer, Cloud",          # leading space
            "location": "New York, NY (HQ)",
            "jobUrl": "https://jobs.ashbyhq.com/ramp/34413f8d",
            "applyUrl": "https://jobs.ashbyhq.com/ramp/34413f8d/application",
            "descriptionPlain": "ABOUT RAMP. Build finance infra.",
            "descriptionHtml": "<p>ABOUT RAMP.</p>",
            "publishedAt": "2026-04-07T17:12:35.753+00:00",
            "isRemote": True, "isListed": True,
        },
        {
            "id": "77",
            "title": "Data Engineer",
            "location": "Remote",
            "jobUrl": "https://jobs.ashbyhq.com/ramp/77",
            "descriptionPlain": "",                        # empty → fall back
            "descriptionHtml": "&lt;b&gt;SQL &amp; Python&lt;/b&gt;",
            "publishedAt": "2026-04-01T00:00:00Z",
        },
    ],
}


def test_fetch_ashby_parses_fields():
    jobs = fetch_ashby("ramp", fetch_json=lambda url: _FAKE_ASHBY)
    assert len(jobs) == 2
    j = jobs[0]
    assert j.source == "ashby" and j.board == "ramp" and j.company == "ramp"
    assert j.title == "Security Engineer, Cloud"   # leading space stripped
    assert j.location == "New York, NY (HQ)"
    assert j.url == "https://jobs.ashbyhq.com/ramp/34413f8d"   # jobUrl wins
    assert j.ext_id == "34413f8d"
    assert j.description == "ABOUT RAMP. Build finance infra."
    assert j.posted_at == "2026-04-07T17:12:35.753+00:00"


def test_fetch_ashby_description_fallback():
    j = fetch_ashby("ramp", fetch_json=lambda url: _FAKE_ASHBY)[1]
    assert j.description == "SQL & Python"          # descriptionPlain empty → HTML


def test_fetch_ashby_builds_url():
    seen = {}
    fetch_ashby("ramp", fetch_json=lambda url: seen.update(url=url) or {"jobs": []})
    assert "api.ashbyhq.com/posting-api/job-board/ramp" in seen["url"]


def test_fetch_ashby_empty_and_blank_token():
    assert fetch_ashby("ramp", fetch_json=lambda url: {"jobs": []}) == []
    assert fetch_ashby("", fetch_json=lambda url: _FAKE_ASHBY) == []      # no call


def test_fetch_dispatch_lever_and_ashby():
    assert len(ats.fetch("lever", "spotify", fetch_json=lambda u: _FAKE_LEVER)) == 2
    aj = ats.fetch("ashby", "ramp", fetch_json=lambda u: _FAKE_ASHBY)
    assert len(aj) == 2 and aj[0].source == "ashby"


# --- Recruitee ({"offers": [...]}; PROVISIONAL mapping, full-time roles) ------
# careers_url is the public posting; location is flat or city+country_code;
# description is HTML; created_at is ISO. No company name → board token.
_FAKE_RECRUITEE = {
    "offers": [
        {
            "id": 1234567,
            "title": "Senior Backend Engineer",
            "slug": "senior-backend-engineer",
            "careers_url": "https://acme.recruitee.com/o/senior-backend-engineer",
            "careers_apply_url": "https://acme.recruitee.com/o/senior-backend-engineer/c/new",
            "location": "London, United Kingdom",
            "city": "London", "country_code": "GB",
            "created_at": "2026-06-01T12:00:00.000+02:00",
            "description": "&lt;p&gt;Build &amp; scale services.&lt;/p&gt;",
        },
        {
            "id": 99,
            "title": "Data Engineer",
            "careers_url": "https://acme.recruitee.com/o/data-engineer",
            "location": "",                      # empty → fall back to city+country
            "city": "Lisbon", "country_code": "PT",
            "created_at": "2026-05-20T09:00:00Z",
            "description": "&lt;b&gt;SQL&lt;/b&gt;",
        },
    ],
}


def test_fetch_recruitee_parses_fields():
    jobs = ats.fetch_recruitee("acme", fetch_json=lambda u: _FAKE_RECRUITEE)
    assert len(jobs) == 2
    j = jobs[0]
    assert j.source == "recruitee" and j.board == "acme" and j.company == "acme"
    assert j.title == "Senior Backend Engineer"
    assert j.location == "London, United Kingdom"
    assert j.url == "https://acme.recruitee.com/o/senior-backend-engineer"
    assert j.ext_id == "1234567"
    assert j.description == "Build & scale services."       # HTML stripped
    assert j.posted_at == "2026-06-01T12:00:00.000+02:00"


def test_fetch_recruitee_location_fallback():
    j = ats.fetch_recruitee("acme", fetch_json=lambda u: _FAKE_RECRUITEE)[1]
    assert j.location == "Lisbon, PT"                       # city + country_code


def test_fetch_recruitee_builds_url_and_blank_token():
    seen = {}
    ats.fetch_recruitee("acme", fetch_json=lambda u: seen.update(url=u) or {"offers": []})
    assert seen["url"] == "https://acme.recruitee.com/api/offers/"
    assert ats.fetch_recruitee("", fetch_json=lambda u: _FAKE_RECRUITEE) == []   # no call


def test_fetch_dispatch_recruitee():
    jobs = ats.fetch("recruitee", "acme", fetch_json=lambda u: _FAKE_RECRUITEE)
    assert len(jobs) == 2 and jobs[0].source == "recruitee"


def test_token_sources_registry():
    assert ats.TOKEN_SOURCES == frozenset(
        {"greenhouse", "lever", "ashby", "recruitee"})
    assert "findapprenticeship" not in ats.TOKEN_SOURCES   # geo, not a token


# --- filtering (apprenticeships in London) --------------------------------

def test_matches_keyword_in_title_and_description():
    assert matches(_p(title="SWE Apprentice"), keywords=["apprentice"])
    assert matches(_p(title="Engineer", description="hiring an apprentice now"),
                   keywords=["apprentice"])
    assert not matches(_p(title="Senior Engineer", description="full time"),
                       keywords=["apprentice"])
    # whole-WORD: "apprentice" must NOT match inside "apprenticeship"
    assert not matches(_p(title="Apprenticeship Scheme", description="grad"),
                       keywords=["apprentice"])


def test_matches_keyword_case_insensitive_and_all_required():
    j = _p(title="Software Engineer Apprentice")
    assert matches(j, keywords=["APPRENTICE", "software"])   # both present
    assert not matches(j, keywords=["apprentice", "data"])   # 'data' missing


def test_matches_location_substring():
    assert matches(_p(location="London, UK"), location="london")
    assert not matches(_p(location="Berlin, Germany"), location="london")


def test_matches_empty_filter_passes():
    assert matches(_p(), keywords=None, location=None)
    assert matches(_p(), keywords=[], location="")


def test_filter_postings_combines_keyword_and_location():
    jobs = [
        _p(title="SWE Apprentice", location="London, UK"),
        _p(title="SWE Apprentice", location="Berlin"),
        _p(title="Senior SWE", location="London, UK"),
    ]
    out = filter_postings(jobs, keywords=["apprentice"], location="london")
    assert len(out) == 1 and out[0].location == "London, UK"


def test_matches_title_only_cuts_body_noise():
    admin = _p(title="Business Administrator Apprentice",
               description="you will use our software daily")
    swe = _p(title="Software Developer Apprentice", description="join us")
    # default (title+body) matches both; title_only matches only the SWE one
    assert matches(admin, keywords=["software"])
    assert not matches(admin, keywords=["software"], title_only=True)
    assert matches(swe, keywords=["software"], title_only=True)


def test_filter_postings_title_only():
    jobs = [_p(title="Software Developer", description="x"),
            _p(title="Admin", description="uses software")]
    assert len(filter_postings(jobs, keywords=["software"], title_only=True)) == 1


# --- gov.uk Find an Apprenticeship adapter (schema provisional, calibrated
#     live; the mapping is tolerant so these lock the logic, not field names) --
# Shape verified against a live v2 payload (s76 cont.36): addresses[] plural,
# vacancyReference/vacancyUrl/employerName/postedDate, plain-text description.
_FAA_PAGE = {
    "vacancies": [
        {
            "vacancyReference": "1000012345",
            "title": "Software Development Apprentice",
            "employerName": "Acme Ltd",
            "providerName": "Acme Training",
            "addresses": [{"addressLine1": "1 High St", "addressLine3": "London",
                           "addressLine4": "Greater London", "postcode": "EC1A 1BB",
                           "latitude": 51.5, "longitude": -0.1}],
            "vacancyUrl": ("https://www.findapprenticeship.service.gov.uk/"
                           "apprenticeship/reference/1000012345"),
            "postedDate": "2026-06-01T09:00:00Z",
            "closingDate": "2026-09-01T23:59:59Z",
            "description": "Learn & build software here.",
        }
    ],
    "total": 1, "totalFiltered": 1, "totalPages": 1,
}


def test_fetch_faa_maps_fields():
    jobs = fetch_findapprenticeship("london", api_key="k",
                                    fetch_json=lambda u: _FAA_PAGE)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "findapprenticeship" and j.board == "uk"
    assert j.title == "Software Development Apprentice"
    assert j.company == "Acme Ltd"
    assert "London" in j.location and "EC1A 1BB" in j.location
    assert j.ext_id == "1000012345"
    assert j.url.endswith("1000012345")
    assert j.description == "Learn & build software here."
    assert j.closing_at == "2026-09-01T23:59:59Z"   # deadline captured


def test_fetch_faa_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("CORTEX_FAA_KEY", raising=False)
    assert fetch_findapprenticeship("london") == []      # no key, no transport


def test_fetch_faa_uses_env_key(monkeypatch):
    monkeypatch.setenv("CORTEX_FAA_KEY", "envkey")
    jobs = fetch_findapprenticeship("london", fetch_json=lambda u: _FAA_PAGE)
    assert len(jobs) == 1


def test_fetch_faa_passes_geo_params():
    seen = []

    def spy(url):
        seen.append(url)
        return {"vacancies": []}

    fetch_findapprenticeship("reading", api_key="k", fetch_json=spy)
    url = seen[0]
    assert "lat=51.4543" in url and "lon=-0.9781" in url
    assert "distanceInMiles=20" in url and "pageNumber=1" in url
    assert "sort=DistanceAsc" in url               # nearest-first server-side


def test_fetch_faa_captures_distance():
    page = {"vacancies": [{"vacancyReference": "1", "title": "X",
                           "distance": 2.5, "addresses": [{"addressLine3": "Reading"}]}]}
    j = fetch_findapprenticeship("reading", api_key="k", fetch_json=lambda u: page)[0]
    assert j.distance == 2.5


def test_fetch_faa_captures_course_level():
    # course.level is the apprenticeship level (the slice-5 v2 filter signal).
    page = {"vacancies": [
        {"vacancyReference": "1", "title": "SW Dev L3", "course": {"level": 3},
         "addresses": [{"addressLine3": "Reading"}]},
        {"vacancyReference": "2", "title": "No course block"},   # → level None
    ]}
    jobs = fetch_findapprenticeship("reading", api_key="k", fetch_json=lambda u: page)
    assert jobs[0].level == 3
    assert jobs[1].level is None


def test_fetch_faa_unknown_city_no_geo():
    seen = []
    fetch_findapprenticeship("atlantis", api_key="k",
                             fetch_json=lambda u: (seen.append(u), {"vacancies": []})[1])
    assert "lat=" not in seen[0]            # unknown city → nationwide, no geo


def test_fetch_faa_paginates_until_short_page():
    import urllib.parse as up

    def transport(url):
        page = int(up.parse_qs(up.urlparse(url).query)["pageNumber"][0])
        if page == 1:
            return {"vacancies": [{"vacancyReference": "a1"},
                                  {"vacancyReference": "a2"}]}
        return {"vacancies": [{"vacancyReference": "b1"}]}   # short → stop

    jobs = fetch_findapprenticeship("london", api_key="k", page_size=2,
                                    fetch_json=transport)
    assert len(jobs) == 3


def test_fetch_faa_raw_sink_gets_first_page():
    box = {}
    fetch_findapprenticeship("london", api_key="k",
                             fetch_json=lambda u: _FAA_PAGE,
                             raw_sink=lambda d: box.update(d=d))
    assert box["d"] is _FAA_PAGE


def test_fetch_dispatch_findapprenticeship():
    jobs = ats.fetch("findapprenticeship", "london", api_key="k",
                     fetch_json=lambda u: _FAA_PAGE)
    assert len(jobs) == 1 and jobs[0].source == "findapprenticeship"


# --- application deadlines (closing date) ------------------------------------
# The most actionable field for the apprenticeship hunt. Dates are built RELATIVE
# to today so the assertions don't rot as the calendar moves.
from datetime import datetime, timezone, timedelta  # noqa: E402


def _iso_in(days: int) -> str:
    # Base on the LOCAL calendar date, because days_until_close counts in LOCAL
    # time. Basing this on the UTC date flaked at the UTC-midnight boundary
    # (s80: machine UTC still 07-24 23:xx while local BST already 07-25 00:xx,
    # so _iso_in(0) built a 07-24 date that read as "yesterday" locally → -1).
    # Noon keeps it on the same local calendar day for any real-world tz offset.
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()}T12:00:00Z"


def test_days_until_close_counts_calendar_days():
    assert ats.days_until_close(_iso_in(0)) == 0      # today
    assert ats.days_until_close(_iso_in(1)) == 1      # tomorrow
    assert ats.days_until_close(_iso_in(5)) == 5
    assert ats.days_until_close(_iso_in(-2)) == -2    # already past


def test_days_until_close_uses_the_date_the_source_printed():
    """The FAA stamps a closing DAY as "23:59:59Z". Converting that to British
    Summer Time lands on 00:59 the next day, which read one day long all
    summer: Cortex said "fecha em 4 dias" for a vacancy whose GOV.UK page said
    "Closes in 3 days (Friday 2 October)" (s83)."""
    from datetime import timedelta as _td
    for days in (0, 1, 3, 25, -1):
        d = datetime.now().astimezone().date() + _td(days=days)
        assert ats.days_until_close(f"{d.isoformat()}T23:59:59Z") == days


def test_days_until_close_bad_input_is_none():
    assert ats.days_until_close(None) is None
    assert ats.days_until_close("") is None
    assert ats.days_until_close("not a date") is None


def test_format_deadline_pt_and_en():
    assert ats.format_deadline(None) == ""             # nothing to append
    assert ats.format_deadline(_iso_in(0)) == "fecha hoje"
    assert ats.format_deadline(_iso_in(1)) == "fecha amanhã"
    assert ats.format_deadline(_iso_in(5)) == "fecha em 5 dias"
    assert ats.format_deadline(_iso_in(-1)) == "encerrada"
    assert ats.format_deadline(_iso_in(0), "en") == "closes today"
    assert ats.format_deadline(_iso_in(1), "en") == "closes tomorrow"
    assert ats.format_deadline(_iso_in(5), "en") == "closes in 5 days"
    assert ats.format_deadline(_iso_in(-1), "en") == "closed"


def test_is_deadline_urgent_window():
    assert ats.is_deadline_urgent(_iso_in(0)) is True
    assert ats.is_deadline_urgent(_iso_in(3)) is True
    assert ats.is_deadline_urgent(_iso_in(4)) is False   # outside the 3-day window
    assert ats.is_deadline_urgent(_iso_in(-1)) is False   # already closed
    assert ats.is_deadline_urgent(None) is False


# --- whole-word keyword matching (so "AI"/"IT" are usable, not substring noise)
def test_keyword_in_text_matches_short_tokens_not_substrings():
    kw = ats.keyword_in_text
    # "AI" / "IT" match as WORDS...
    assert kw("AI Engineer Apprentice", "ai")
    assert kw("Data & AI Apprentice", "ai")
    assert kw("IT Support Apprentice", "it")
    # ...but NOT as substrings (this was the whole point of the change)
    assert not kw("Childcare Training Apprentice", "ai")   # trAIning
    assert not kw("Repair Apprentice", "ai")               # repAIr
    assert not kw("Quality Apprentice", "it")              # qualITy
    assert not kw("Childcare Apprentice", "it")
    # longer keywords behave intuitively
    assert kw("Software Developer Apprentice", "software")
    assert kw("Cyber Security Apprentice", "cyber")
    assert not kw("Gardener Apprentice", "software")
    # whole-word strictness: "data" ≠ "database"
    assert kw("Data Analyst Apprentice", "data")
    assert not kw("Database Administrator Apprentice", "data")
    assert not kw("anything", "")                          # empty keyword → no match


# --- is_closed: the single definition of "dead vacancy" (s82) ----------------
# Both surfaces that RECOMMEND a job (conversation handler, morning briefing)
# filter on this, so its edge cases are load-bearing.

def test_is_closed_only_when_provably_past():
    assert ats.is_closed(_iso_in(-1)) is True
    assert ats.is_closed(_iso_in(-30)) is True
    assert ats.is_closed(_iso_in(0)) is False     # closes TODAY — still open
    assert ats.is_closed(_iso_in(1)) is False


def test_is_closed_is_false_when_the_date_is_unknown():
    # Deliberately asymmetric: many boards omit the deadline. Treating unknown
    # as closed would silently empty the list of perfectly good vacancies.
    assert ats.is_closed(None) is False
    assert ats.is_closed("") is False
    assert ats.is_closed("not a date") is False


def test_is_closed_agrees_with_format_deadline():
    # Guard against drift: whenever the phrase says "encerrada"/"closed",
    # is_closed must agree, and vice versa.
    for days in (-10, -1, 0, 1, 30):
        iso = _iso_in(days)
        says_closed = ats.format_deadline(iso) == "encerrada"
        assert ats.is_closed(iso) is says_closed
