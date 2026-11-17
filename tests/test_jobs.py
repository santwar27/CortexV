"""JobLog store tests (Autonomous Job Hunter slice 1, s76). In-memory DB."""
from src.integrations.ats import JobPosting
from src.integrations.jobs import JobLog


def _job(url, title="Engineer", **kw):
    base = dict(source="greenhouse", board="acme", company="Acme",
                ext_id="1", title=title, location="Remote",
                description="desc", posted_at="2026-06-10")
    base.update(kw)
    return JobPosting(url=url, **base)


def _db():
    return JobLog(db_path=":memory:")


def test_upsert_inserts_and_counts():
    db = _db()
    new, total = db.upsert([_job("u1"), _job("u2")])
    assert (new, total) == (2, 2)
    assert db.count() == 2


def test_upsert_dedup_by_url_refreshes_volatile():
    db = _db()
    db.upsert([_job("u1", title="Old Title")])
    new, total = db.upsert([_job("u1", title="New Title")])
    assert (new, total) == (0, 1)        # not new, but processed
    assert db.count() == 1               # still one row
    assert db.list()[0]["title"] == "New Title"   # refreshed


def test_upsert_preserves_score_and_status():
    db = _db()
    db.upsert([_job("u1")])
    row = db.list()[0]
    db.set_status(row["id"], "applied")
    # simulate a ranking having written a score
    conn = db._connect()
    conn.execute("UPDATE jobs SET score = 0.9 WHERE id = ?", (row["id"],))
    conn.commit()
    # re-fetch the same job → score/status must survive
    db.upsert([_job("u1", title="Refreshed")])
    row2 = db.list()[0]
    assert row2["title"] == "Refreshed"
    assert row2["status"] == "applied"
    assert row2["score"] == 0.9


def test_upsert_skips_postings_without_url():
    db = _db()
    new, total = db.upsert([_job(""), _job("u1")])
    assert (new, total) == (1, 1)
    assert db.count() == 1


def test_upsert_empty_is_noop():
    assert _db().upsert([]) == (0, 0)


def test_list_filters_by_status_and_source():
    db = _db()
    db.upsert([_job("u1"), _job("u2"), _job("u3", source="lever")])
    db.set_status(db.list(source="lever")[0]["id"], "dismissed")
    assert len(db.list(source="greenhouse")) == 2
    assert len(db.list(status="dismissed")) == 1
    assert db.count(status="new") == 2


def test_set_status_rejects_unknown():
    db = _db()
    db.upsert([_job("u1")])
    assert db.set_status(db.list()[0]["id"], "bogus") is False


def test_clear_by_source_and_all():
    db = _db()
    db.upsert([_job("u1"), _job("u2", source="lever")])
    assert db.clear(source="lever") == 1
    assert db.count() == 1
    assert db.clear() == 1
    assert db.count() == 0


def test_upsert_stores_distance():
    db = _db()
    db.upsert([_job("u1", distance=3.5)])
    assert db.list()[0]["distance"] == 3.5


def test_list_orders_nearest_first_nulls_last():
    db = _db()
    db.upsert([_job("u1", distance=10.0), _job("u2", distance=2.0),
               _job("u3", distance=None)])
    assert [r["distance"] for r in db.list()] == [2.0, 10.0, None]


def test_list_order_by_score():
    db = _db()
    db.upsert([_job("u1"), _job("u2")])
    conn = db._connect()
    conn.execute("UPDATE jobs SET score = 0.9 WHERE url = 'u2'")
    conn.execute("UPDATE jobs SET score = 0.3 WHERE url = 'u1'")
    conn.commit()
    assert [r["url"] for r in db.list(order="score")] == ["u2", "u1"]


def test_migrate_adds_missing_columns():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, "
                 "url TEXT UNIQUE, title TEXT)")
    JobLog._migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "distance" in cols and "score" in cols


def test_resume_set_get_and_replace():
    db = _db()
    assert db.get_resume() is None
    assert db.set_resume("my cv text", name="cv.pdf")
    row = db.get_resume()
    assert row["text"] == "my cv text" and row["name"] == "cv.pdf"
    db.set_resume("updated cv", name="cv2.md")    # single master → replaced
    assert db.get_resume()["text"] == "updated cv"


def test_set_resume_rejects_empty():
    db = _db()
    assert db.set_resume("   ") is False
    assert db.get_resume() is None


def test_set_score_persists():
    db = _db()
    db.upsert([_job("u1")])
    jid = db.list()[0]["id"]
    assert db.set_score(jid, 0.77)
    assert db.list()[0]["score"] == 0.77


def test_upsert_persists_and_refreshes_closing_at():
    db = _db()
    db.upsert([_job("u1", closing_at="2026-09-01T23:59:59Z")])
    assert db.list()[0]["closing_at"] == "2026-09-01T23:59:59Z"   # round-trips
    # closing_at is volatile — a re-fetch refreshes it (deadline can change).
    db.upsert([_job("u1", closing_at="2026-10-15T23:59:59Z")])
    assert db.list()[0]["closing_at"] == "2026-10-15T23:59:59Z"


def test_closing_at_defaults_null_when_absent():
    db = _db()
    db.upsert([_job("u1")])                 # JobPosting.closing_at defaults None
    assert db.list()[0]["closing_at"] is None


# --- list_open: never recommend a vacancy that already closed (s82) ---------
from datetime import datetime, timedelta  # noqa: E402


def _iso_in(days: int) -> str:
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()}T12:00:00Z"


def test_list_open_drops_closed_keeps_open():
    db = _db()
    db.upsert([_job("u1", title="Open", closing_at=_iso_in(5)),
               _job("u2", title="Closed", closing_at=_iso_in(-1))])
    titles = [r["title"] for r in db.list_open(limit=10, order="recent")]
    assert titles == ["Open"]


def test_list_open_keeps_rows_with_no_deadline():
    # Unknown deadline is NOT closed — boards often omit it.
    db = _db()
    db.upsert([_job("u1", title="NoDate", closing_at=None)])
    assert [r["title"] for r in db.list_open(limit=10)] == ["NoDate"]


def test_list_open_still_returns_a_full_limit():
    # The bug this guards: filtering AFTER a SQL LIMIT would return 1 open row
    # instead of 3, because the closed ones ate the slots.
    db = _db()
    jobs = []
    for i in range(5):                       # 5 closed, best-scored first
        jobs.append(_job(f"c{i}", title=f"Closed {i}", closing_at=_iso_in(-1)))
    for i in range(3):
        jobs.append(_job(f"o{i}", title=f"Open {i}", closing_at=_iso_in(9)))
    db.upsert(jobs)
    for row in db.list(limit=100):
        # score the closed ones HIGHEST so they'd win an unfiltered top-3
        db.set_score(row["id"], 0.99 if row["title"].startswith("Closed") else 0.5)
    rows = db.list_open(limit=3, order="score")
    assert len(rows) == 3
    assert all(r["title"].startswith("Open") for r in rows)


def test_list_open_respects_status_and_order():
    db = _db()
    db.upsert([_job("u1", title="A", closing_at=_iso_in(5)),
               _job("u2", title="B", closing_at=_iso_in(5)),
               _job("u3", title="Dead", closing_at=_iso_in(-2))])
    for row in db.list(limit=100):
        db.set_score(row["id"], {"A": 0.7, "B": 0.9, "Dead": 1.0}[row["title"]])
        if row["title"] == "A":
            db.set_status(row["id"], "seen")
    rows = db.list_open(limit=10, status="new", order="score")
    assert [r["title"] for r in rows] == ["B"]      # A filtered by status,
                                                    # Dead by deadline


def test_list_open_empty_ledger_is_empty_not_an_error():
    assert _db().list_open(limit=3) == []


# --- list_open: never recommend a vacancy the board dropped (s83) -----------
# findapprenticeship removes a vacancy from the feed the day it closes, and
# many carry no deadline at all, so the deadline test alone recommended a
# June vacancy in late September. `fetched_at` is the "last seen" the ledger
# was already keeping.

def _day(days: int) -> str:
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()} 07:00:03"


def _set_seen(db, url: str, when: str) -> None:
    conn = db._connect()
    conn.execute("UPDATE jobs SET fetched_at = ? WHERE url = ?", (when, url))
    conn.commit()


def test_list_open_drops_vacancies_gone_from_the_feed():
    db = _db()
    db.upsert([_job("live", title="Live"), _job("ghost", title="Ghost")])
    _set_seen(db, "live", _day(0))
    _set_seen(db, "ghost", _day(-10))       # missed every run since
    _set_seen(db, "live", _day(0))
    # a second fetch day must exist for a cutoff to be knowable
    db.upsert([_job("filler", title="Filler")])
    _set_seen(db, "filler", _day(-1))
    titles = [r["title"] for r in db.list_open(limit=10, order="recent")]
    assert "Ghost" not in titles
    assert "Live" in titles


def test_list_open_forgives_one_missed_fetch():
    """The 07:00 cron can fail or come back partial — one miss is not death."""
    db = _db()
    db.upsert([_job("today", title="Today"), _job("yesterday", title="Yesterday")])
    _set_seen(db, "today", _day(0))
    _set_seen(db, "yesterday", _day(-1))    # seen in the previous run only
    titles = [r["title"] for r in db.list_open(limit=10, order="recent")]
    assert set(titles) == {"Today", "Yesterday"}


def test_feed_cutoff_needs_two_fetch_days():
    """With one fetch ever, a vanished vacancy is indistinguishable from a
    freshly fetched one — so nothing is dropped."""
    db = _db()
    db.upsert([_job("u1", title="Only")])
    assert db.feed_cutoffs() == {}
    assert [r["title"] for r in db.list_open(limit=10)] == ["Only"]


def test_feed_cutoff_is_per_source():
    db = _db()
    db.upsert([_job("a1", title="Aged", source="findapprenticeship"),
               _job("a2", title="Fresh", source="findapprenticeship"),
               _job("a3", title="Filler", source="findapprenticeship"),
               _job("b1", title="OtherBoard", source="greenhouse")])
    _set_seen(db, "a1", _day(-10))
    _set_seen(db, "a2", _day(0))
    _set_seen(db, "a3", _day(-1))           # three fetch days for this source,
                                            # so "Aged" missed two runs, not one
    _set_seen(db, "b1", _day(-10))          # its source has ONE fetch day
    titles = [r["title"] for r in db.list_open(limit=10, order="recent")]
    assert "Aged" not in titles             # dropped by its own source's cutoff
    assert "OtherBoard" in titles           # untouched: no cutoff for greenhouse


def test_a_vacancy_that_comes_back_to_the_feed_is_recommended_again():
    """Nothing is deleted and the rule is computed on read, so a wrong call
    heals itself as soon as the board lists the vacancy again."""
    db = _db()
    db.upsert([_job("u1", title="Back"), _job("u2", title="Other")])
    _set_seen(db, "u1", _day(-10))
    _set_seen(db, "u2", _day(0))
    db.upsert([_job("filler")])
    _set_seen(db, "filler", _day(-1))
    assert "Back" not in [r["title"] for r in db.list_open(limit=10)]
    db.upsert([_job("u1", title="Back")])   # listed again today
    assert "Back" in [r["title"] for r in db.list_open(limit=10)]


# --- the fetch days are recorded, not inferred (s88, Codex lote 11) ----------

def _fetched_on(db, day: str, *jobs) -> None:
    """One successful fetch on `day` that saw exactly `jobs`."""
    db.upsert(list(jobs))
    conn = db._connect()
    for j in jobs:
        conn.execute("UPDATE jobs SET fetched_at = ? WHERE url = ?", (day, j.url))
    conn.execute("UPDATE fetch_runs SET day = ? WHERE day = ? AND board = ?",
                 (day[:10], datetime.now().strftime("%Y-%m-%d"), jobs[0].board))
    conn.commit()


def test_a_vacancy_gone_since_june_is_not_recommended_forever():
    """One live vacancy refreshed daily and one gone since June: the last-seen
    days were just "today" and "June", so the cutoff was June and the June one
    stayed in the top 3 indefinitely."""
    db = _db()
    gone, live = _job("gone", title="Gone"), _job("live", title="Live")
    _fetched_on(db, "2026-06-10 07:00:00", gone, live)
    for d in range(-4, 1):
        _fetched_on(db, _day(d), live)
    assert [r["title"] for r in db.list_open(limit=10)] == ["Live"]


def test_refreshing_one_board_does_not_hide_another():
    db = _db()
    a = _job("a", title="BoardA", board="acme")
    b = _job("b", title="BoardB", board="other")
    _fetched_on(db, _day(-20), b)                 # board B fetched once, long ago
    for d in range(-3, 1):
        _fetched_on(db, _day(d), a)               # only board A refreshed since
    assert {r["title"] for r in db.list_open(limit=10)} == {"BoardA", "BoardB"}


# --- a reply in another shape is not an empty board (s88, Codex lote 11) -----

import pytest  # noqa: E402

from src.integrations import ats as _ats  # noqa: E402


@pytest.mark.parametrize("source,reply", [
    ("greenhouse", {"data": {"jobs": [{"id": 1, "title": "x"}]}}),
    ("ashby", {"postings": []}),
    ("recruitee", {"jobs": []}),
    ("lever", {"postings": []}),
    ("findapprenticeship", {"data": {"vacancies": [{"title": "x"}]}}),
])
def test_a_changed_reply_is_an_error_not_no_vacancies(source, reply):
    with pytest.raises(_ats.UnexpectedResponse):
        _ats.fetch(source, "reading" if source == "findapprenticeship" else "acme",
                   fetch_json=lambda url: reply)


@pytest.mark.parametrize("source,reply", [
    ("greenhouse", {"jobs": []}), ("findapprenticeship", {"vacancies": []}),
    ("lever", []),
])
def test_a_valid_empty_reply_is_still_no_vacancies(source, reply):
    assert _ats.fetch(source, "reading" if source == "findapprenticeship" else "acme",
                      fetch_json=lambda url: reply) == []


def test_refresh_says_why_it_found_nothing():
    from src.integrations.job_hunter import refresh_jobs
    changed = refresh_jobs(joblog=JobLog(db_path=":memory:"),
                           fetch_json=lambda url: {"data": {}}, embed=lambda t: [1.0])
    empty = refresh_jobs(joblog=JobLog(db_path=":memory:"),
                         fetch_json=lambda url: {"vacancies": []}, embed=lambda t: [1.0])
    assert changed.get("error") == "format" and "error" not in empty
