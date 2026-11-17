"""Job ranker tests (Job Hunter slice 2, s76). Offline — deterministic embed."""
import numpy as np

from src.integrations.ats import JobPosting
from src.integrations.job_ranker import rank_jobs
from src.integrations.jobs import JobLog


_VOCAB = ["python", "java", "react", "data"]


def _embed(text: str) -> np.ndarray:
    """Deterministic stub: a vector of keyword counts, so a job sharing the
    resume's words scores higher. Never zero (avoids degenerate cosine)."""
    t = (text or "").lower()
    v = np.array([float(t.count(w)) for w in _VOCAB], dtype=np.float32)
    if float(v.sum()) == 0.0:
        v = np.full(len(_VOCAB), 0.01, dtype=np.float32)
    return v


def _job(url, title, desc=""):
    return JobPosting(source="findapprenticeship", board="uk", company="X",
                      ext_id="1", title=title, location="London", url=url,
                      description=desc, posted_at=None)


def _db_with(resume, jobs):
    db = JobLog(db_path=":memory:")
    db.set_resume(resume)
    db.upsert(jobs)
    return db


def test_rank_orders_by_similarity():
    db = _db_with("python python developer with data",
                  [_job("u1", "Python Data Apprentice"),
                   _job("u2", "Java React Frontend")])
    n = rank_jobs(db, embed=_embed)
    assert n == 2
    ranked = db.list(order="score")
    assert ranked[0]["url"] == "u1"          # python/data match on top
    assert ranked[0]["score"] > ranked[1]["score"]


def test_rank_writes_scores_in_unit_range():
    db = _db_with("python", [_job("u1", "Python Apprentice")])
    rank_jobs(db, embed=_embed)
    s = db.list()[0]["score"]
    assert s is not None and 0.0 <= s <= 1.0


def test_rank_no_resume_returns_zero():
    db = JobLog(db_path=":memory:")
    db.upsert([_job("u1", "Python Apprentice")])
    assert rank_jobs(db, embed=_embed) == 0


def test_rank_uses_title_and_description():
    db = _db_with("react",
                  [_job("u1", "Engineer", desc="react react react"),
                   _job("u2", "Engineer", desc="pure java backend")])
    rank_jobs(db, embed=_embed)
    ranked = db.list(order="score")
    assert ranked[0]["url"] == "u1"          # matched via description


def test_rank_fail_soft_on_embed_error():
    calls = {"n": 0}

    def flaky(text):
        calls["n"] += 1
        if "boom" in text.lower():
            raise RuntimeError("embed down")
        return _embed(text)

    db = _db_with("python",
                  [_job("u1", "Python Apprentice"), _job("u2", "boom role")])
    n = rank_jobs(db, embed=flaky)
    assert n == 1                            # the good one scored, bad skipped


def test_rank_resume_embed_failure_returns_zero():
    db = _db_with("python", [_job("u1", "Python Apprentice")])

    def boom(text):
        raise RuntimeError("ollama down")

    assert rank_jobs(db, embed=boom) == 0


def test_rank_only_unscored_skips_already_scored():
    db = _db_with("python", [_job("u1", "Python Apprentice"),
                             _job("u2", "Data Apprentice")])
    rank_jobs(db, embed=_embed)                       # both scored
    u1 = next(r for r in db.list() if r["url"] == "u1")
    frozen = u1["score"]
    # New job arrives; only_unscored ranks just it, leaving the old scores.
    db.upsert([_job("u3", "React Apprentice")])
    calls = {"jobs": []}

    def spy(text):
        calls["jobs"].append(text)
        return _embed(text)

    n = rank_jobs(db, embed=spy, only_unscored=True)
    assert n == 1                                     # only the new one
    # the resume + exactly one job got embedded (not the 2 already scored)
    assert sum("apprentice" in t.lower() for t in calls["jobs"]) == 1
    u1_after = next(r for r in db.list() if r["url"] == "u1")
    assert u1_after["score"] == frozen                # untouched
