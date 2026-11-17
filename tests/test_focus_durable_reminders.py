"""Reminder intent survives a lost queue; only a disk claim may create."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from src.core import focus, focus_mixin
from src.core.audit import audit
from src.integrations import reminders_client as rem
from tests.test_focus_reminder_outcomes import (
    TASK, ProcessDied, Unconfirmed, clock, events, inline, orch, start, turn,
)
from tests.test_focus_reminders import END, NUDGE, FakeReminders, T0


@pytest.fixture(autouse=True)
def fake(monkeypatch, tmp_path):
    """All three borders are fake in EVERY test; no worker is submitted."""
    monkeypatch.setenv("HOME", str(tmp_path))
    client = FakeReminders()
    for name in ("ensure_list", "create_reminder", "complete_reminder"):
        monkeypatch.setattr(rem, name, getattr(client, name))
    return client


@pytest.fixture
def other():
    store = focus.FocusStore(db_path=focus.store.db_path)
    yield store
    if store._conn is not None:
        store._conn.close()


def run_create(session):
    focus_mixin._create_reminders_job(
        session.id, (TASK, "end", session.ends_at.isoformat()),
        (TASK, "nudge", focus.later(session.ends_at, focus.NUDGE_AFTER).isoformat()))


def states(session):
    return session.reminder_state, session.nudge_state


def completes(fake):
    return [c[1] for c in fake.calls if c[0] == "complete"]


def test_death_before_worker_leaves_two_pending(orch, other, fake):
    session = start(orch)
    assert fake.calls == []
    work = other.unfinished_reminders()
    assert [(w.session_id, w.which, w.state) for w in work] == [
        (session.id, "end", "pending"), (session.id, "nudge", "pending")]


def test_duplicate_job_needs_disk_claim(orch, other, fake, monkeypatch):
    session = start(orch)
    run_create(session)
    run_create(session)
    monkeypatch.setattr(focus, "store", other)
    run_create(session)
    assert len([c for c in fake.calls if c[0] == "create"]) == 2
    assert states(other.get(session.id)) == ("done", "done")
    assert other.unfinished_reminders() == []


def test_two_connections_race_for_one_claim(other):
    session = focus.store.start(TASK, "step", 25, T0)
    assert other.available()
    barrier = Barrier(2)

    def claim(store):
        barrier.wait(timeout=5)
        return store.claim_reminder(session.id, "end")

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(claim, store) for store in (focus.store, other)]
        assert sorted(f.result(timeout=10) for f in futures) == [False, True]
    assert states(other.get(session.id)) == ("running", "pending")
    assert [w.state for w in other.unfinished_reminders()] == ["running", "pending"]


@pytest.mark.parametrize("stream", [False, True])
def test_stop_before_worker_uses_durable_path(orch, other, fake, stream):
    session = start(orch)
    assert turn(orch, "parei: next", stream)[1] == "FOCUS_CLOSED"
    saved = other.get(session.id)
    assert saved.completion_wanted
    assert states(saved) == ("skipped", "skipped")
    run_create(session)
    assert fake.calls == []
    assert other.unfinished_reminders() == []


@pytest.mark.parametrize("during", [1, 2], ids=["end", "nudge"])
def test_stop_from_another_process_during_create(
        orch, other, fake, monkeypatch, during):
    session = start(orch)
    create = fake.create_reminder

    def stop_inside_create(*args, **kwargs):
        count = len([c for c in fake.calls if c[0] == "create"]) + 1
        if count == during:
            assert other.stop("next", T0).id == session.id
            # The stop worker sees the running create and must leave it alone.
            focus_mixin._complete_reminders_job(session.id)
        return create(*args, **kwargs)

    monkeypatch.setattr(rem, "create_reminder", stop_inside_create)
    run_create(session)
    saved = other.get(session.id)
    assert saved.reminder == END
    assert completes(fake) == ([END] if during == 1 else [END, NUDGE])
    assert states(saved) == ("done", "skipped" if during == 1 else "done")
    assert saved.reminder_completion == "done"
    assert saved.nudge_completion == (None if during == 1 else "done")
    assert other.unfinished_reminders() == []


def test_start_closes_and_records_intent_atomically(other):
    old = focus.store.start(TASK, "step", 25, T0)
    new, closed = focus.store.start_closing("new", "step", 25, T0)
    assert closed == [old.id]
    saved = other.get(old.id)
    assert not saved.is_open and saved.completion_wanted
    assert states(saved) == ("skipped", "skipped")
    assert states(other.get(new.id)) == ("pending", "pending")
    assert {w.session_id for w in other.unfinished_reminders()} == {new.id}


def test_failed_start_rolls_back_close_and_intent(other):
    old = focus.store.start(TASK, "step", 25, T0)
    with focus.store._lock, focus.store._conn as conn:
        conn.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON sessions "
                     "BEGIN SELECT RAISE(ABORT, 'private task violet'); END")
    assert focus.store.start_closing("new", "step", 25, T0) == (None, [])
    saved = other.get(old.id)
    assert saved.is_open and not saved.completion_wanted
    assert states(saved) == ("pending", "pending")


@pytest.mark.parametrize("outcome", ["done", "not_found", "failed"])
def test_completion_result_is_durable(orch, other, fake, inline, outcome):
    session = start(orch)
    fake.fail_on = {"missing"} if outcome == "not_found" else {"complete"} if outcome == "failed" else set()
    turn(orch, "parei: next")
    saved = other.get(session.id)
    assert (saved.reminder_completion, saved.nudge_completion) == (outcome, outcome)
    if outcome == "done":
        focus_mixin._complete_reminders_job(session.id)
        assert completes(fake) == [END, NUDGE]
        assert other.unfinished_reminders() == []
    else:
        assert [w.completion for w in other.unfinished_reminders()] == [outcome, outcome]
        fake.fail_on = set()
        focus_mixin._complete_reminders_job(session.id)
        assert other.unfinished_reminders() == []
        saved = other.get(session.id)
        assert (saved.reminder_completion, saved.nudge_completion) == ("done", "done")


@pytest.mark.parametrize("outcome", ["uncertain", "no_id", "failed", "list_failed"])
def test_unconfirmed_or_failed_end_never_retried(
        orch, other, fake, monkeypatch, outcome):
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        if outcome == "no_id":
            return {}
        raise (rem.RemindersUnavailable if outcome == "failed" else Unconfirmed)(TASK)

    monkeypatch.setattr(rem, "ensure_list" if outcome == "list_failed" else "create_reminder", fail)
    session = start(orch)
    run_create(session)
    run_create(session)
    saved = other.get(session.id)
    expected = "failed" if outcome in ("failed", "list_failed") else "unknown"
    assert states(saved) == (expected, "skipped")
    assert saved.reminder == expected
    other.stop("next", T0)
    focus_mixin._complete_reminders_job(session.id)
    assert attempts == [1] and completes(fake) == []
    assert other.unfinished_reminders() == []


OLD_SCHEMA = """
CREATE TABLE sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT NOT NULL,
    first_step  TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ends_at     TEXT NOT NULL,
    closed_at   TEXT,
    next_step   TEXT,
    reminder        TEXT,
    nudge_reminder  TEXT
);
"""


def test_migration_preserves_legacy_rows_and_completion(tmp_path, fake, monkeypatch):
    path = str(tmp_path / "legacy.db")
    rows = [
        (1, TASK, "a", T0.isoformat(), T0.isoformat(), None, None, END, NUDGE),
        (2, "closed", "b", T0.isoformat(), T0.isoformat(), T0.isoformat(), "next", focus.REMINDER_FAILED, focus.REMINDER_UNKNOWN),
        (3, "nulls", "c", T0.isoformat(), T0.isoformat(), T0.isoformat(), None, None, None),
    ]
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    for _ in range(2):
        store = focus.FocusStore(db_path=path)
        try:
            assert store.available()
            actual = store._conn.execute(
                "SELECT id, task, first_step, started_at, ends_at, closed_at, "
                "next_step, reminder, nudge_reminder FROM sessions ORDER BY id").fetchall()
            assert [tuple(r) for r in actual] == rows
            assert store.unfinished_reminders() == []
        finally:
            if store._conn is not None:
                store._conn.close()
    store = focus.FocusStore(db_path=path)
    monkeypatch.setattr(focus, "store", store)
    try:
        assert store.stop("next", T0).id == 1
        focus_mixin._complete_reminders_job(1)
        focus_mixin._complete_reminders_job(1)
        assert completes(fake) == [END, NUDGE]
        assert store.unfinished_reminders() == []
        assert states(store.get(1)) == (None, None)
    finally:
        store._conn.close()


@pytest.mark.parametrize("write", ["claim", "end", "nudge", "complete"])
def test_failed_worker_write_is_private_and_fail_soft(
        orch, other, fake, caplog, write):
    session = start(orch)
    if write == "complete":
        run_create(session)
        other.stop("next", T0)
    condition = {
        "claim": "NEW.reminder_state = 'running'",
        "end": "NEW.reminder_state = 'done'",
        "nudge": "NEW.nudge_state = 'done'",
        "complete": "NEW.reminder_completion = 'done'",
    }[write]
    with focus.store._lock, focus.store._conn as conn:
        conn.execute(f"CREATE TRIGGER reject_write BEFORE UPDATE ON sessions WHEN {condition} "
                     "BEGIN SELECT RAISE(ABORT, 'private task violet'); END")
    if write == "complete":
        focus_mixin._complete_reminders_job(session.id)
    else:
        run_create(session)
    assert events("reminder_not_saved")
    assert TASK not in Path(audit.path).read_text()
    assert TASK not in caplog.text
    assert not focus.store._conn.in_transaction
    if write == "claim":
        assert fake.calls == []
    elif write == "end":
        assert states(other.get(session.id)) == ("running", "skipped")
        assert len([c for c in fake.calls if c[0] == "create"]) == 1
    elif write == "nudge":
        saved = other.get(session.id)
        assert states(saved) == ("done", "running")
        assert (saved.reminder, saved.nudge_reminder) == (END, None)
    else:
        saved = other.get(session.id)
        assert (saved.reminder_completion, saved.nudge_completion) == (None, "done")


def test_other_process_stops_before_any_worker(orch, other, fake, monkeypatch):
    session = start(orch)
    assert other.stop("next", T0).completion_wanted
    monkeypatch.setattr(focus, "store", other)
    run_create(session)
    focus_mixin._complete_reminders_job(session.id)
    assert fake.calls == []
    assert states(other.get(session.id)) == ("skipped", "skipped")
    assert other.unfinished_reminders() == []


def test_stop_between_end_reread_and_nudge_claim(orch, other, fake, monkeypatch):
    session = start(orch)
    claim = focus.store.claim_reminder

    def stop_before_claim(sid, which):
        if which == "nudge":
            other.stop("next", T0)
            focus_mixin._complete_reminders_job(sid)
        return claim(sid, which)

    monkeypatch.setattr(focus.store, "claim_reminder", stop_before_claim)
    run_create(session)
    assert completes(fake) == [END]
    assert len([c for c in fake.calls if c[0] == "create"]) == 1
    assert states(other.get(session.id)) == ("done", "skipped")


@pytest.mark.parametrize("during", [1, 2], ids=["end", "nudge"])
def test_running_create_survives_death_without_retry(
        orch, other, fake, monkeypatch, during):
    session = start(orch)
    create = fake.create_reminder

    def die(*args, **kwargs):
        count = len([c for c in fake.calls if c[0] == "create"]) + 1
        if count == during:
            raise ProcessDied()
        return create(*args, **kwargs)

    monkeypatch.setattr(rem, "create_reminder", die)
    with pytest.raises(ProcessDied):
        run_create(session)
    monkeypatch.setattr(rem, "create_reminder", create)
    monkeypatch.setattr(focus, "store", other)
    run_create(session)
    other.stop("next", T0)
    focus_mixin._complete_reminders_job(session.id)
    assert len([c for c in fake.calls if c[0] == "create"]) == during - 1
    assert completes(fake) == ([] if during == 1 else [END])
    assert [(w.which, w.state) for w in other.unfinished_reminders()] == [
        ("end" if during == 1 else "nudge", "running")]


def test_competing_job_observes_claims_before_reminders(
        orch, other, fake, monkeypatch):
    session = start(orch)
    ensure, create = fake.ensure_list, fake.create_reminder

    def ensure_after_claim(*args):
        assert other.get(session.id).reminder_state == "running"
        return ensure(*args)

    def create_after_claim(*args, **kwargs):
        which = "nudge" if any(c[0] == "create" for c in fake.calls) else "end"
        saved = other.get(session.id)
        assert getattr(saved, "reminder_state" if which == "end" else "nudge_state") == "running"
        if which == "nudge":
            assert saved.reminder == END and saved.reminder_state == "done"
        with monkeypatch.context() as patch:
            patch.setattr(focus, "store", other)
            run_create(session)
        return create(*args, **kwargs)

    monkeypatch.setattr(rem, "ensure_list", ensure_after_claim)
    monkeypatch.setattr(rem, "create_reminder", create_after_claim)
    run_create(session)
    assert len([c for c in fake.calls if c[0] == "create"]) == 2


def test_late_failed_completion_cannot_overwrite_done(orch, other, fake, inline):
    session = start(orch)
    turn(orch, "parei: next")
    assert other.record_completion(session.id, "end", END, "failed")
    assert other.get(session.id).reminder_completion == "done"
    focus_mixin._complete_reminders_job(session.id)
    assert completes(fake) == [END, NUDGE]


def test_migration_failure_rolls_back_and_reply_is_fail_soft(
        orch, tmp_path, monkeypatch, caplog):
    path = str(tmp_path / "migration-failure.db")
    real_connect = sqlite3.connect
    with real_connect(path) as conn:
        conn.execute(OLD_SCHEMA)
        conn.execute("INSERT INTO sessions (task, first_step, started_at, ends_at) "
                     "VALUES (?, 'step', ?, ?)", (TASK, T0.isoformat(), T0.isoformat()))

    class FailingMigration(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("ALTER TABLE sessions ADD COLUMN nudge_state"):
                raise sqlite3.OperationalError(TASK)
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=FailingMigration)

    store = focus.FocusStore(db_path=path)
    monkeypatch.setattr(focus, "store", store)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        assert turn(orch, f"foco: {TASK}")[1] == "FOCUS_ASK_STEP"
        assert turn(orch, "open the form")[1] == "FOCUS_ERROR"
        assert store.current() is None and not store.available()
        assert store._conn is None
    with real_connect(path) as conn:
        assert len(conn.execute("PRAGMA table_info(sessions)").fetchall()) == 9
        assert conn.execute("SELECT task FROM sessions").fetchall() == [(TASK,)]
    assert TASK not in caplog.text
    assert TASK not in Path(audit.path).read_text()
    try:
        assert store.current().task == TASK  # migration can succeed on a later open
    finally:
        store._conn.close()


def test_two_first_openers_migrate_safely(tmp_path):
    path = str(tmp_path / "concurrent-migration.db")
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(OLD_SCHEMA)
    stores = [focus.FocusStore(db_path=path) for _ in range(2)]
    barrier = Barrier(2)

    def open_store(store):
        barrier.wait(timeout=5)
        return store.available()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(open_store, store) for store in stores]
            assert all(f.result(timeout=10) for f in futures)
        assert stores[0].start(TASK, "step", 25, T0)
        assert states(stores[1].current()) == ("pending", "pending")
    finally:
        for store in stores:
            if store._conn is not None:
                store._conn.close()
