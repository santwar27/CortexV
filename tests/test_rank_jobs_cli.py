"""
`make jobs` / `rank_jobs.py --list` read-only view (s76 cont.43).

Covers `_print_top`: the fast, no-Ollama listing of already-ranked matches.
"""
from scripts.rank_jobs import _print_top
from src.integrations.ats import JobPosting
from src.integrations.jobs import JobLog


def _posting(url, title, **kw):
    base = dict(source="findapprenticeship", board="uk", company="X", ext_id="1",
                title=title, location="Reading", description="d", posted_at=None)
    base.update(kw)
    return JobPosting(url=url, **base)


def test_print_top_lists_ranked_with_urls(capsys):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("https://x/1", "Software Apprentice")])
    db.set_score(db.list()[0]["id"], 0.83)
    rc = _print_top(db, 10, header="já rankeadas")
    out = capsys.readouterr().out
    assert rc == 0
    assert "83%" in out and "Software Apprentice" in out
    assert "https://x/1" in out                 # the apply link is shown
    assert "já rankeadas" in out


def test_print_top_empty_ledger_hints(capsys):
    rc = _print_top(JobLog(db_path=":memory:"), 10, header="já rankeadas")
    assert rc == 1
    assert "fetch_jobs" in capsys.readouterr().out


def test_print_top_unranked_hints(capsys):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("u", "T")])             # stored but never scored
    rc = _print_top(db, 10, header="já rankeadas")
    assert rc == 1
    assert "rank_jobs" in capsys.readouterr().out


from datetime import datetime, timezone, timedelta  # noqa: E402


def _iso_in(days):
    # LOCAL calendar date — days_until_close counts in local time, so a UTC
    # base flaked at the UTC-midnight boundary (s80). See tests/test_ats.py.
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()}T12:00:00Z"


def test_print_top_shows_deadline(capsys):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("https://x/1", "SW Apprentice", closing_at=_iso_in(40))])
    db.set_score(db.list()[0]["id"], 0.83)
    _print_top(db, 10, header="já rankeadas")
    out = capsys.readouterr().out
    assert "fecha em 40 dias" in out
    assert "⚠️" not in out                       # 40 days away → not urgent


def test_print_top_flags_urgent_deadline(capsys):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("https://x/1", "SW Apprentice", closing_at=_iso_in(2))])
    db.set_score(db.list()[0]["id"], 0.83)
    _print_top(db, 10, header="já rankeadas")
    out = capsys.readouterr().out
    assert "fecha em 2 dias" in out and "⚠️" in out   # ≤3 days → urgency flag


def test_print_top_no_deadline_when_absent(capsys):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("https://x/1", "SW Apprentice")])   # no closing_at
    db.set_score(db.list()[0]["id"], 0.83)
    _print_top(db, 10, header="já rankeadas")
    out = capsys.readouterr().out
    assert "fecha" not in out                    # nothing appended
