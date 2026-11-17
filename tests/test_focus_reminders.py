"""The end-of-focus reminder, off the reply path (s86 bloco 2b).

Measured (bloco 0, real Reminders.app): create ≈ 2 s, complete ≈ 1 s,
delete 28–37 s, and two 30 s timeouts in one day. The reply never waits:
the reminder is created and completed on a background worker, completed
rather than deleted, and a failure is told at the next "onde parei?".
The alarm in the "Cortex Foco" list reaches the iPhone through iCloud.
"""
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import src.integrations.reminders_client as rem
from src.core import focus
from src.core import focus_mixin
from src.core.audit import audit

LONDON = ZoneInfo("Europe/London")
T0 = datetime(2026, 10, 16, 15, 15, tzinfo=LONDON)


class FakeReminders:
    def __init__(self):
        self.calls = []
        self.fail_on = set()
        self.delay = 0.0

    def ensure_list(self, name):
        self.calls.append(("ensure_list", name))

    def create_reminder(self, title, due_iso=None, body=None, list_name=None):
        time.sleep(self.delay)
        if "create" in self.fail_on:
            raise rem.RemindersUnavailable("Reminders.app didn't respond in 30s")
        self.calls.append(("create", title, due_iso, list_name))
        n = sum(1 for c in self.calls if c[0] == "create")
        return {"title": title, "due_iso": due_iso, "list": list_name,
                "id": f"x-apple-reminder://AAAA-{n}"}

    def complete_reminder(self, ref, list_name):
        if "complete" in self.fail_on:
            raise rem.RemindersUnavailable("rejected")
        self.calls.append(("complete", ref, list_name))
        return 0 if "missing" in self.fail_on else 1


@pytest.fixture
def fake(monkeypatch):
    f = FakeReminders()
    for name in ("ensure_list", "create_reminder", "complete_reminder"):
        monkeypatch.setattr(rem, name, getattr(f, name))
    return f


@pytest.fixture
def real_worker(monkeypatch, fake):
    """The real background worker — drained on teardown, BEFORE the fake
    client is put back, even when the test failed half-way. A latency
    assertion failing under load skipped the drain, the teardown restored
    the real client, and the job's second create (the nudge) went to it: a
    real reminder in Pedro's list (s88, Codex lote 5, reproduced against a
    sentinel). Fixtures tear down in reverse: this one before `fake`."""
    monkeypatch.setattr(focus_mixin, "_in_background",
                        lambda job, *args: focus_mixin._REMINDERS_WORKER.submit(job, *args))
    yield
    focus_mixin._REMINDERS_WORKER.submit(lambda: None).result(timeout=30)


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
    return o


def start(orch, task="candidatura", step="abrir o formulário"):
    orch._process_core(f"foco: {task}")
    return orch._process_core(step)


END, NUDGE = "x-apple-reminder://AAAA-1", "x-apple-reminder://AAAA-2"


class TestTheReminders:
    """The end of the focus, and the nudge 10 min later (Pedro, 16/10: a
    second reminder — Mac AND iPhone, at the exact minute)."""

    def test_a_start_creates_both_in_their_own_list(self, orch, fake, inline):
        _reply, mode = start(orch)
        assert mode == "FOCUS_STARTED"
        assert fake.calls == [
            ("ensure_list", "Cortex Foco"),
            ("create", "Fim do foco: candidatura", "2026-10-16T15:40:00+01:00", "Cortex Foco"),
            ("create", "Ainda em candidatura?", "2026-10-16T15:50:00+01:00", "Cortex Foco"),
        ]
        s = focus.store.current()
        assert (s.reminder, s.nudge_reminder) == (END, NUDGE)

    def test_a_stop_completes_both_never_deletes(self, orch, fake, inline):
        start(orch)
        orch._process_core("parei: seção 2")
        assert fake.calls[-2:] == [("complete", END, "Cortex Foco"),
                                   ("complete", NUDGE, "Cortex Foco")]

    def test_a_bare_stop_completes_them_too(self, orch, fake, inline):
        start(orch)
        orch._process_core("parei")
        assert [c[1] for c in fake.calls if c[0] == "complete"] == [END, NUDGE]

    def test_a_new_focus_completes_the_old_ones_first(self, orch, fake, inline, clock):
        start(orch)
        clock.now = T0 + timedelta(minutes=30)            # time up: no question
        start(orch, task="estudo", step="ler o texto A")
        kinds = [c[0] for c in fake.calls]
        assert kinds == ["ensure_list", "create", "create",
                         "complete", "complete",
                         "ensure_list", "create", "create"]
        assert [c[1] for c in fake.calls if c[0] == "complete"] == [END, NUDGE]

    def test_a_secret_in_the_task_never_reaches_icloud(self, orch, fake, inline):
        """The title carries the task, syncs to iCloud and shows on the
        locked iPhone — and nobody confirms it (s88, Codex lote 4)."""
        start(orch, task="revisar acesso; senha do banco é Abc123")
        titles = [c[1] for c in fake.calls if c[0] == "create"]
        assert len(titles) == 2
        assert not any("Abc123" in t for t in titles)
        assert all("senha do banco é [REDACTED]" in t for t in titles)

    def test_a_focus_another_process_opened_gets_its_alarms_completed(
            self, orch, fake, inline, monkeypatch):
        """The terminal opens B and saves its two ids between the UI's read
        and its write: start() closed B, but the UI had read "nothing open"
        and completed nothing — B's alarms rang (s88, Codex lote 4,
        reproduced). B lands right before the UI's write here; the ids of
        what it closed come from that write itself."""
        terminal = focus.FocusStore(db_path=focus.store.db_path)   # same file
        real_write = focus.store.start_closing
        landed = {}

        def b_lands_first(*args):
            b = terminal.start("tarefa B", "passo B", 25, T0)
            terminal.set_reminders(b.id, "x-apple-reminder://B-1", "x-apple-reminder://B-2")
            landed["b"] = b.id
            return real_write(*args)
        monkeypatch.setattr(focus.store, "start_closing", b_lands_first)
        _reply, mode = start(orch)                                 # the UI opens A
        assert mode == "FOCUS_STARTED"
        assert terminal.get(landed["b"]).closed_at is not None
        assert [c[1] for c in fake.calls if c[0] == "complete"] == [
            "x-apple-reminder://B-1", "x-apple-reminder://B-2"]

    def test_start_says_which_it_closed(self, orch):
        a, closed = focus.store.start_closing("A", "a", 25, T0)
        assert closed == []
        b, closed = focus.store.start_closing("B", "b", 25, T0)
        assert closed == [a.id]

    def test_the_english_title(self, orch, fake, inline):
        orch._process_core("focus: cover letter")
        orch._process_core("open the template")
        assert fake.calls[1][1] == "End of focus: cover letter"
        assert fake.calls[2][1] == "Still on cover letter?"


class TestTheReplyNeverWaits:

    def test_a_slow_reminders_app(self, orch, fake, real_worker):
        """The real worker: Reminders takes 2 s; the reply comes at once."""
        fake.delay = 2.0
        orch._process_core("foco: candidatura")
        t0 = time.perf_counter()
        _reply, mode = orch._process_core("abrir o formulário")
        assert mode == "FOCUS_STARTED" and time.perf_counter() - t0 < 0.5
        focus_mixin._REMINDERS_WORKER.submit(lambda: None).result(timeout=10)   # drain
        assert focus.store.current().reminder == END


    def test_every_real_worker_test_drains_through_the_fixture(self):
        """No test may submit to the real worker itself: only real_worker
        drains it before the fake client goes."""
        import inspect
        import tests.test_focus_reminders as me
        source = inspect.getsource(me)
        needle = "_REMINDERS_WORKER" + ".submit(job"              # not this line
        start = source.index("def real_worker(")
        end = source.index("@pytest.fixture", start)
        assert source.count(needle) == 1
        assert start < source.index(needle) < end                  # the fixture's


class TestWhenRemindersFails:

    def test_a_failed_create_is_told_at_onde_parei(self, orch, fake, inline):
        fake.fail_on = {"create"}
        _reply, mode = start(orch)
        assert mode == "FOCUS_STARTED"                      # the focus still runs
        assert focus.store.current().reminder == focus.REMINDER_FAILED
        reply, _ = orch._process_core("onde parei?")
        assert "lembrete no celular não foi criado" in reply

    def test_a_failed_create_is_told_after_the_time_is_up_too(self, orch, fake, inline, clock):
        """Asked only after the 25 min — exactly when the phone should have
        rung and didn't. The warning lived in the "still running" branch
        alone (s88, Codex lote 4)."""
        fake.fail_on = {"create"}
        start(orch)
        clock.now = T0 + timedelta(minutes=40)
        reply, mode = orch._process_core("onde parei?")
        assert mode == "FOCUS_WHERE" and "o tempo acabou" in reply
        assert "lembrete no celular não foi criado" in reply

    def test_a_failed_nudge_keeps_the_end_reminder(self, orch, fake, monkeypatch, inline):
        real = fake.create_reminder

        def only_the_first(title, *a, **kw):
            if title.startswith("Ainda em"):
                raise rem.RemindersUnavailable("timeout")
            return real(title, *a, **kw)
        monkeypatch.setattr(rem, "create_reminder", only_the_first)
        start(orch)
        s = focus.store.current()
        assert (s.reminder, s.nudge_reminder) == (END, focus.REMINDER_FAILED)
        orch._process_core("parei: x")
        assert [c[1] for c in fake.calls if c[0] == "complete"] == [END]

    def test_nothing_to_complete_after_a_failed_create(self, orch, fake, inline):
        fake.fail_on = {"create"}
        start(orch)
        orch._process_core("parei: x")
        assert not [c for c in fake.calls if c[0] == "complete"]

    def test_a_failed_complete_never_reaches_the_reply(self, orch, fake, inline):
        start(orch)
        fake.fail_on = {"complete"}
        reply, mode = orch._process_core("parei: seção 2")
        assert mode == "FOCUS_CLOSED"

    def test_a_reminder_moved_off_the_list_is_not_counted_done(self, orch, fake, inline):
        """Moved to another list on the iPhone, it isn't found where Cortex
        looks: the old one-liner said nothing and the job counted it done —
        the alarm still rings (s88, Codex lote 4, seen in the real app)."""
        start(orch)
        fake.fail_on = {"missing"}
        orch._process_core("parei: x")
        events = [json.loads(line) for line in open(audit.path)]
        mine = [e for e in events if e.get("stage") == "focus"]
        assert not [e for e in mine if e["event"] == "reminder_completed"]
        assert [e["payload"] for e in mine if e["event"] == "reminder_failed"] == [
            {"step": "complete", "error": "LookupError"},
            {"step": "complete_nudge", "error": "LookupError"}]

    def test_an_id_not_saved_yet_is_skipped(self, orch, fake):
        """Started in another process seconds ago: no id yet — no complete."""
        start(orch)                                  # conftest drops the jobs
        focus_mixin._complete_reminders_job(focus.store.current().id)
        assert fake.calls == []

    def test_the_audit_keeps_no_task_text(self, orch, fake, inline):
        fake.fail_on = {"create"}
        start(orch, task="candidatura secreta")
        events = [json.loads(line) for line in open(audit.path)]
        mine = [e for e in events if e.get("stage") == "focus"]
        failed = next(e for e in mine if e["event"] == "reminder_failed")
        assert failed["payload"] == {"step": "create", "error": "RemindersUnavailable"}
        assert not any("secreta" in json.dumps(e) for e in mine)


class TestTheClient:
    """The AppleScript the client sends — no Reminders.app runs here."""

    def _capture(self, monkeypatch, out=""):
        sent = []
        monkeypatch.setattr(rem, "_run", lambda script: sent.append(script) or out)
        return sent

    def test_create_returns_the_id(self, monkeypatch):
        self._capture(monkeypatch, out="reminder id x-apple-reminder://8CEAD0DC-0598-4C8C\n")
        made = rem.create_reminder("Fim do foco: x", "2026-10-16T15:40:00+01:00",
                                   list_name="Cortex Foco")
        assert made["id"] == "x-apple-reminder://8CEAD0DC-0598-4C8C"

    def test_create_without_an_id_in_the_reply(self, monkeypatch):
        self._capture(monkeypatch, out="")
        assert rem.create_reminder("x")["id"] is None

    def test_ensure_list_creates_only_if_missing(self, monkeypatch):
        sent = self._capture(monkeypatch)
        rem.ensure_list("Cortex Foco")
        assert 'if not (exists list "Cortex Foco")' in sent[0]
        assert "container of default list" in sent[0]          # iCloud, not "On My Mac"

    def test_complete_by_id_inside_its_list(self, monkeypatch):
        """A bare `reminder id` searched every account: 130 s, then a 30 s
        timeout against the real app. Inside the list: 1 s."""
        sent = self._capture(monkeypatch)
        rem.complete_reminder("x-apple-reminder://8CEAD0DC-0598", "Cortex Foco")
        which = '(reminders whose id is "x-apple-reminder://8CEAD0DC-0598")'
        assert 'tell list "Cortex Foco"' in sent[0]
        assert f"if (count of {which}) is 0 then return \"0\"" in sent[0]
        assert f"set completed of {which} to true" in sent[0]
        # never `set matches to (reminders whose …)`: 24 s in the real app
        assert "set matches" not in sent[0]

    @pytest.mark.parametrize("out, found", [("1\n", 1), ("0\n", 0), ("", 0)])
    def test_complete_says_whether_it_found_it(self, monkeypatch, out, found):
        self._capture(monkeypatch, out=out)
        assert rem.complete_reminder("x-apple-reminder://8CEAD0DC-0598", "Cortex Foco") == found

    @pytest.mark.parametrize("bad", ["", "abc", 'x-apple-reminder://A") to true\ndo shell script "rm',
                                     "x-apple-reminder://ZZZ"])
    def test_complete_refuses_anything_but_an_id(self, monkeypatch, bad):
        sent = self._capture(monkeypatch)
        with pytest.raises(rem.RemindersUnavailable):
            rem.complete_reminder(bad, "Cortex Foco")
        assert sent == []
