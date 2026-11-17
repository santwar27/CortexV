"""Unconfirmed creates stay unknown; every returned id is saved at once."""
import json
from datetime import timedelta
from pathlib import Path

import pytest

import src.integrations.reminders_client as rem
from src.core import focus, focus_mixin
from src.core.audit import audit
from tests.test_focus_reminders import END, NUDGE, FakeReminders, T0


TASK = "private task violet"
UNKNOWN_LINES = (
    "(Não sei se o lembrete no celular foi criado — o Lembretes não confirmou.)",
    "(I don't know whether the phone reminder was created — Reminders didn't confirm.)",
)
FAILED_LINES = (
    "(O lembrete no celular não foi criado — o Lembretes não respondeu.)",
    "(The phone reminder wasn't created — Reminders didn't respond.)",
)


class Unconfirmed(rem.RemindersUnavailable):
    uncertain = True


class ProcessDied(BaseException):
    pass


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    """Every test patches all three borders; no real Reminders.app calls."""
    f = FakeReminders()
    for name in ("ensure_list", "create_reminder", "complete_reminder"):
        monkeypatch.setattr(rem, name, getattr(f, name))
    return f


@pytest.fixture
def inline(monkeypatch):
    """Run the queued Reminders jobs right away, in order."""
    monkeypatch.setattr(focus_mixin, "_in_background", lambda job, *args: job(*args))


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = T0

        def __call__(self):
            return self.now
    c = Clock()
    monkeypatch.setattr(focus_mixin, "local_now", c)
    return c


@pytest.fixture
def orch(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("HOME", str(tmp_path))
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    for engine in (o.engine_analytical, o.engine_creative):
        for method in ("chat", "chat_with_tools", "chat_stream"):
            getattr(engine, method).side_effect = AssertionError("LLM called")
    return o


def turn(orch, text, stream=False):
    if not stream:
        return orch._process_core(text)
    parts, mode = [], None
    for chunk, final, m, _lang in orch.process_stream(text):
        parts.append(chunk)
        if final:
            mode = m
    return "".join(parts), mode


def start(orch):
    assert turn(orch, f"foco: {TASK}")[1] == "FOCUS_ASK_STEP"
    assert turn(orch, "open the form")[1] == "FOCUS_STARTED"
    return focus.store.current()


def events(event):
    return [e for e in map(json.loads, Path(audit.path).read_text().splitlines())
            if e.get("stage") == "focus" and e["event"] == event]


@pytest.mark.parametrize("outcome", ["timeout", "no_id"])
@pytest.mark.parametrize("en", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_unknown_end_is_honest_running_and_time_up(
        orch, fake, inline, clock, monkeypatch, outcome, en, stream):
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        if outcome == "timeout":
            raise Unconfirmed(TASK)
        return {"id": None}
    monkeypatch.setattr(rem, "create_reminder", create)
    session = start(orch)
    assert session.reminder == focus.REMINDER_UNKNOWN
    assert session.nudge_reminder is None
    assert len(calls) == 1  # Neither nudge nor blind retry.
    question = "where was I?" if en else "onde parei?"
    for elapsed in (0, 30):
        clock.now = T0 + timedelta(minutes=elapsed)
        reply, mode = turn(orch, question, stream)
        assert mode == "FOCUS_WHERE"
        assert reply.endswith(UNKNOWN_LINES[en])
        assert all(line not in reply for line in FAILED_LINES)
    turn(orch, "parei: next", stream)
    reply, _ = turn(orch, question, stream)
    assert all(line not in reply for line in UNKNOWN_LINES)
    assert len(calls) == 1
    assert not any(c[0] == "complete" for c in fake.calls)
    error = "Unconfirmed" if outcome == "timeout" else "RemindersUnavailable"
    assert [e["payload"] for e in events("reminder_failed")] == [
        {"step": "create", "error": error, "uncertain": True}]


@pytest.mark.parametrize("en", [False, True])
def test_known_failure_keeps_the_old_line(orch, inline, monkeypatch, en, caplog):
    def create(*args, **kwargs):
        raise rem.RemindersUnavailable(TASK)
    monkeypatch.setattr(rem, "create_reminder", create)
    assert start(orch).reminder == focus.REMINDER_FAILED
    reply, _ = turn(orch, "where was I?" if en else "onde parei?")
    assert reply.endswith(FAILED_LINES[en])
    assert [e["payload"] for e in events("reminder_failed")] == [
        {"step": "create", "error": "RemindersUnavailable"}]
    assert TASK not in Path(audit.path).read_text()
    assert TASK not in caplog.text


def test_uncertain_list_setup_is_a_known_create_failure(orch, fake, inline, monkeypatch):
    def ensure_list(name):
        raise Unconfirmed(TASK)
    monkeypatch.setattr(rem, "ensure_list", ensure_list)
    assert start(orch).reminder == focus.REMINDER_FAILED
    assert fake.calls == []
    assert events("reminder_failed")[0]["payload"] == {
        "step": "create", "error": "Unconfirmed", "uncertain": True}


def test_end_id_is_on_disk_before_the_nudge(orch, fake, inline, monkeypatch):
    create = fake.create_reminder
    seen = []

    def with_read(*args, **kwargs):
        if any(c[0] == "create" for c in fake.calls):
            seen.append(focus.store.current().reminder)
            other = focus.FocusStore(db_path=focus.store.db_path)
            try:
                seen.append(other.current().reminder)
            finally:
                other._conn.close()
        return create(*args, **kwargs)
    monkeypatch.setattr(rem, "create_reminder", with_read)
    session = start(orch)
    assert seen == [END, END]
    assert (session.reminder, session.nudge_reminder) == (END, NUDGE)


def test_process_death_during_nudge_keeps_end_id_for_another_process(
        orch, fake, inline, monkeypatch):
    session = focus.store.start(TASK, "open the form", 25, T0)
    create = fake.create_reminder

    def dies_on_nudge(*args, **kwargs):
        if any(c[0] == "create" for c in fake.calls):
            raise ProcessDied()
        return create(*args, **kwargs)
    monkeypatch.setattr(rem, "create_reminder", dies_on_nudge)
    end = (TASK, "body", session.ends_at.isoformat())
    nudge = (TASK, "body", (session.ends_at + focus.NUDGE_AFTER).isoformat())
    with pytest.raises(ProcessDied):
        focus_mixin._create_reminders_job(session.id, end, nudge)
    other = focus.FocusStore(db_path=focus.store.db_path)
    try:
        saved = other.current()
        assert (saved.reminder, saved.nudge_reminder) == (END, None)
        monkeypatch.setattr(focus, "store", other)
        turn(orch, "parei: next")
        assert [c[1] for c in fake.calls if c[0] == "complete"] == [END]
    finally:
        other._conn.close()


@pytest.mark.parametrize("outcome", ["timeout", "no_id", "rejected"])
def test_nudge_outcomes_keep_end_id_and_complete_only_it(
        orch, fake, inline, monkeypatch, outcome):
    create = fake.create_reminder

    def fail_nudge(*args, **kwargs):
        if any(c[0] == "create" for c in fake.calls):
            if outcome == "no_id":
                return None
            error = rem.RemindersUnavailable if outcome == "rejected" else Unconfirmed
            raise error(TASK)
        return create(*args, **kwargs)
    monkeypatch.setattr(rem, "create_reminder", fail_nudge)
    session = start(orch)
    expected = focus.REMINDER_FAILED if outcome == "rejected" else focus.REMINDER_UNKNOWN
    assert (session.reminder, session.nudge_reminder) == (END, expected)
    payload = {"step": "create_nudge", "error":
               "Unconfirmed" if outcome == "timeout" else "RemindersUnavailable"}
    if outcome != "rejected":
        payload["uncertain"] = True
    assert events("reminder_failed")[0]["payload"] == payload
    turn(orch, "parei: next")
    assert [c[1] for c in fake.calls if c[0] == "complete"] == [END]


@pytest.mark.parametrize("stop", [True, False], ids=["stop", "new_focus"])
def test_completion_skips_both_sentinels(orch, fake, inline, clock, stop):
    session = start(orch)
    focus.store.set_reminders(session.id, focus.REMINDER_UNKNOWN, focus.REMINDER_FAILED)
    if stop:
        turn(orch, "parei: next")
    else:
        clock.now = T0 + timedelta(minutes=30)
        start(orch)
    assert not any(c[0] == "complete" for c in fake.calls)


@pytest.mark.parametrize("failed_write", [1, 2], ids=["end", "nudge"])
def test_unsaved_id_is_audited_once_without_task_text(
        orch, fake, inline, monkeypatch, caplog, failed_write):
    real_write = focus.store.finish_reminder
    writes = []

    def write(*args):
        writes.append(args)
        return False if len(writes) == failed_write else real_write(*args)
    monkeypatch.setattr(focus.store, "finish_reminder", write)
    start(orch)  # The worker must not raise on a failed write.
    step = "create" if failed_write == 1 else "create_nudge"
    assert [e["payload"] for e in events("reminder_not_saved")] == [{"step": step}]
    assert TASK not in Path(audit.path).read_text()
    assert TASK not in caplog.text
    creates = [c for c in fake.calls if c[0] == "create"]
    assert len(creates) == failed_write  # Nudge needs a saved end id.
    if failed_write == 2:
        assert focus.store.current().reminder == END
