"""A failed write leaves no transaction open (s88, Codex lote 11 — invariant 6).

fitness.db and jobs.db are written by the API, the terminal and the launchd
jobs. A statement failing halfway through a batch was caught fail-soft with
no rollback: the store's long-lived connection kept the transaction open and
the next process got "database is locked" (reproduced). A second connection
with a short timeout stands for the other process.
"""
import sqlite3

from src.integrations import jobs as jobs_mod
from src.integrations.fitness import FitnessLog, LoggedItem


def _posting(url, **kw):
    fields = {k: None for k in jobs_mod.JobPosting.__dataclass_fields__}
    fields.update(source="faa", board="faa", title="t", url=url)
    fields.update(kw)
    return jobs_mod.JobPosting(**fields)


def _other_process_can_write(db, sql):
    other = sqlite3.connect(db, timeout=0.3)
    try:
        other.execute(sql)
        other.commit()
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        other.close()


def test_a_job_batch_failing_halfway_is_undone_and_unlocked(tmp_path):
    db = str(tmp_path / "jobs.db")
    log = jobs_mod.JobLog(db_path=db)
    bad = _posting("https://a/2", closing_at={"date": "2026-11-01"})  # can't bind
    assert log.upsert([_posting("https://a/1"), bad]) == (0, 0)
    assert not log._connect().in_transaction
    assert _other_process_can_write(
        db, "INSERT INTO jobs (source, board, title, url, fetched_at) "
            "VALUES ('x', 'x', 't', 'https://b/1', 'now')")
    urls = [r[0] for r in log._connect().execute("SELECT url FROM jobs")]
    assert "https://a/1" not in urls          # the batch is all or nothing


def test_a_meal_failing_halfway_is_undone_and_unlocked(tmp_path):
    db = str(tmp_path / "fit.db")
    log = FitnessLog(db_path=db)
    ok = LoggedItem(food="frango", grams=200, kcal=330, protein=62, carb=0, fat=7.2)
    bad = LoggedItem(food="x", grams=1, kcal={"no": 1}, protein=0, carb=0, fat=0)
    assert log.log_meal([ok, bad]) is False
    assert not log._connect().in_transaction
    assert _other_process_can_write(db, "INSERT INTO foods (name, kcal, protein, carb, fat) "
                                        "VALUES ('z', 1, 1, 1, 1)")
    assert log.today_totals()["n"] == 0
