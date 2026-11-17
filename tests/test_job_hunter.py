"""
Job Hunter daily refresh (slice 5 v2, s76): fetch → filter → upsert → rank.

Fully offline: the FAA transport (fetch_json) and the embedder are injected,
the ledger is in-memory.
"""
import numpy as np

from src.integrations import job_hunter as jh
from src.integrations.jobs import JobLog


# A live-shaped FAA page (course.level is the filter signal; titles drive the
# keyword union). Mix in a keyword-miss and a level-miss to prove the filter.
_FAA = {
    "vacancies": [
        {"vacancyReference": "1", "title": "Software Developer Apprentice",
         "course": {"level": 3}, "addresses": [{"addressLine3": "Reading"}],
         "description": "build software", "distance": 2.0},
        {"vacancyReference": "2", "title": "Business Analyst Apprentice",
         "course": {"level": 4}, "addresses": [{"addressLine3": "London"}],
         "description": "analyse the business", "distance": 15.0},
        {"vacancyReference": "3", "title": "Nursery Practitioner Apprentice",
         "course": {"level": 3}, "addresses": [{"addressLine3": "Reading"}],
         "description": "childcare", "distance": 1.0},               # keyword miss
        {"vacancyReference": "4", "title": "Software Engineer Degree Apprentice",
         "course": {"level": 6}, "addresses": [{"addressLine3": "London"}],
         "description": "build", "distance": 20.0},                  # level miss
    ]
}


def _fixed_embed(_text):
    return np.array([1.0, 0.0])      # any non-zero vector → a definite score


def _ledger_with_resume():
    jl = JobLog(db_path=":memory:")
    jl.set_resume("software engineer, python, apprenticeship")
    return jl


def test_refresh_filters_by_keyword_and_level():
    jl = _ledger_with_resume()
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed,
                          fetch_json=lambda u: _FAA)
    # #1 (software/dev, L3) and #2 (business, L4) survive; #3 (no kw), #4 (L6) cut.
    assert out["fetched"] == 4
    assert out["matched"] == 2
    assert out["new"] == 2
    assert out["ranked"] == 2
    titles = sorted(r["title"] for r in jl.list(limit=10))
    assert titles == ["Business Analyst Apprentice", "Software Developer Apprentice"]


def test_refresh_dedups_and_only_ranks_new(monkeypatch):
    jl = _ledger_with_resume()
    jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA)
    # Second run, same board: nothing new, and already-scored jobs aren't re-ranked.
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA)
    assert out["new"] == 0
    assert out["ranked"] == 0          # only_unscored → the existing scores stand


def test_refresh_no_match_is_zeros_but_reports_fetched():
    jl = _ledger_with_resume()
    page = {"vacancies": [
        {"vacancyReference": "9", "title": "Chef Apprentice",
         "course": {"level": 2}, "addresses": [{"addressLine3": "Reading"}]},
    ]}
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: page)
    assert out == {"fetched": 1, "matched": 0, "new": 0, "ranked": 0}


def test_refresh_fail_soft_on_fetch_error():
    jl = _ledger_with_resume()
    def boom(_u):
        raise RuntimeError("FAA down")
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=boom)
    # still fail-soft, and now it says it failed (s88, Codex lote 11): the
    # bare zeros read the same as "no vacancy today"
    assert out == {"fetched": 0, "matched": 0, "new": 0, "ranked": 0, "error": "fetch"}
    assert jl.count() == 0             # nothing stored on failure


def test_refresh_custom_keywords_and_levels():
    jl = _ledger_with_resume()
    # Only "business" at level 4 → just #2.
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA,
                          keywords=("business",), levels=(4,))
    assert out["matched"] == 1
    assert jl.list(limit=5)[0]["title"] == "Business Analyst Apprentice"


# --- env-tunable defaults (s76 cont.44) ---------------------------------------

def test_env_keywords_and_levels_drive_the_filter(monkeypatch):
    monkeypatch.setenv("CORTEX_JOBS_KEYWORDS", "business")
    monkeypatch.setenv("CORTEX_JOBS_LEVELS", "4")
    jl = _ledger_with_resume()
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA)
    assert out["matched"] == 1          # only #2 (business, L4)
    assert jl.list(limit=5)[0]["title"] == "Business Analyst Apprentice"


def test_explicit_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("CORTEX_JOBS_KEYWORDS", "business")
    monkeypatch.setenv("CORTEX_JOBS_LEVELS", "4")
    jl = _ledger_with_resume()
    # Explicit kwargs win over env → the broad search matches #1 and #2 again.
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA,
                          keywords=("software", "business"), levels=(3, 4))
    assert out["matched"] == 2


def test_env_location_and_distance_resolve():
    # Helpers: a known city + a positive radius come through.
    assert jh._env_distance() == jh.DEFAULT_DISTANCE_MILES   # unset → default (30)


def test_env_location_and_distance_from_env(monkeypatch):
    monkeypatch.setenv("CORTEX_JOBS_LOCATION", "Manchester")
    monkeypatch.setenv("CORTEX_JOBS_DISTANCE_MI", "10")
    assert jh._env_location() == "Manchester"
    assert jh._env_distance() == 10


def test_env_malformed_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("CORTEX_JOBS_KEYWORDS", "  ,  ")
    monkeypatch.setenv("CORTEX_JOBS_LEVELS", "abc")
    monkeypatch.setenv("CORTEX_JOBS_LOCATION", "   ")
    monkeypatch.setenv("CORTEX_JOBS_DISTANCE_MI", "-5")
    assert jh._env_keywords() == jh.DEFAULT_KEYWORDS
    assert jh._env_levels() == jh.DEFAULT_LEVELS
    assert jh._env_location() == jh.DEFAULT_LOCATION
    assert jh._env_distance() == jh.DEFAULT_DISTANCE_MILES


def test_unknown_location_warns_national(monkeypatch, caplog):
    # A city with no known geo centre → a loud warning (the search goes national).
    monkeypatch.setenv("CORTEX_JOBS_LOCATION", "atlantis")
    jl = _ledger_with_resume()
    with caplog.at_level("WARNING"):
        jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: _FAA)
    assert any("NATIONAL" in r.message for r in caplog.records)


def test_refresh_keyword_match_is_whole_word_not_substring():
    # The s76 cont.46 fix: short keywords like "AI"/"IT" must match as WORDS,
    # never as substrings — otherwise "trAIning"/"qualITy" flood the ledger.
    jl = _ledger_with_resume()
    page = {"vacancies": [
        {"vacancyReference": "10", "title": "AI Engineer Apprentice",
         "course": {"level": 4}, "addresses": [{"addressLine3": "Reading"}],
         "description": "ml", "distance": 1.0},                       # "AI" word → keep
        {"vacancyReference": "11", "title": "IT Support Apprentice",
         "course": {"level": 3}, "addresses": [{"addressLine3": "Reading"}],
         "description": "it", "distance": 1.0},                       # "IT" word → keep
        {"vacancyReference": "12", "title": "Childcare Training Apprentice",
         "course": {"level": 3}, "addresses": [{"addressLine3": "Reading"}],
         "description": "kids", "distance": 1.0},      # 'ai' in trAIning → must DROP
        {"vacancyReference": "13", "title": "Quality Inspector Apprentice",
         "course": {"level": 3}, "addresses": [{"addressLine3": "Reading"}],
         "description": "qa", "distance": 1.0},        # 'it' in qualITy → must DROP
    ]}
    out = jh.refresh_jobs(joblog=jl, embed=_fixed_embed, fetch_json=lambda u: page,
                          keywords=("ai", "it"), levels=(3, 4))
    assert out["matched"] == 2
    titles = sorted(r["title"] for r in jl.list(limit=10))
    assert titles == ["AI Engineer Apprentice", "IT Support Apprentice"]
