"""Focus sessions: the words and the record (s86 bloco 1).

Measured before (real 8B, both entry points): 1 useful reply in 30.
"foco: candidatura…" became a job search, the first step "abrir o
formulário" tried to open an app called "formulário", "parei: …" got "não
posso ajudar com isso", and "onde parei?" came back as "Curitiba, Minas
Gerais." The words are matched here, by rule; the record lives in SQLite.
"""
import sqlite3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.core import focus
from src.core.focus import FocusStore

LONDON = ZoneInfo("Europe/London")
T0 = datetime(2026, 10, 16, 15, 15, tzinfo=LONDON)


@pytest.fixture
def store(tmp_path):
    return FocusStore(db_path=str(tmp_path / "focus.db"))


class TestStartWords:

    @pytest.mark.parametrize("text, expected", [
        ("foco: candidatura da vaga", ("candidatura da vaga", 25)),
        ("Foco: estudar para a prova", ("estudar para a prova", 25)),
        ("foco 40: estudar", ("estudar", 40)),
        ("foco 40 min: estudar", ("estudar", 40)),
        ("foco 15minutos: email da escola", ("email da escola", 15)),
        ("cortex, foco: imposto", ("imposto", 25)),
        ("focus: CV", ("CV", 25)),
    ])
    def test_a_start(self, text, expected):
        assert focus.parse_start(text) == expected

    def test_minutes_are_clamped_not_refused(self):
        assert focus.parse_start("foco 2: x") == ("x", 5)
        assert focus.parse_start("foco 500: x") == ("x", 180)

    @pytest.mark.parametrize("text", [
        "meu foco é o college",
        "foco",
        "foco:",
        "foco:   ",
        "qual o foco da vaga?",
    ])
    def test_not_a_start(self, text):
        assert focus.parse_start(text) is None

    def test_the_task_is_one_line_and_bounded(self):
        task, _ = focus.parse_start("foco: candidatura\n  da vaga  " + "x" * 400)
        assert "\n" not in task and len(task) <= 200


class TestStopWords:

    @pytest.mark.parametrize("text, expected", [
        ("parei: preencher a seção 2", "preencher a seção 2"),
        ("Parei: revisar o texto.", "revisar o texto"),   # the final dot goes
        ("cortex, parei: mandar o email", "mandar o email"),
        ("parei", ""),
        ("Parei.", ""),
        ("parei:", ""),
    ])
    def test_a_stop(self, text, expected):
        assert focus.parse_stop(text) == expected

    @pytest.mark.parametrize("text", [
        "parei de fumar",
        "eu parei no meio",
        "por que você parou?",
    ])
    def test_not_a_stop(self, text):
        assert focus.parse_stop(text) is None


class TestWhereWasIWords:

    @pytest.mark.parametrize("text", [
        "onde parei?",
        "Onde eu parei?",
        "onde é que eu parei",
        "o que eu estava fazendo?",
        "o que eu tava fazendo",
        "cortex, onde parei?",
        # The user's (16/10)
        "onde que eu parei?",
        "onde a gente parou?",
        "onde paramos?",
        "em que eu parei?",
        "onde e que eu parei",          # no accent
        "where was I?",
        "What was I doing?",
    ])
    def test_asks(self, text):
        assert focus.asks_where_was_i(text)

    @pytest.mark.parametrize("text", [
        "onde parei o carro?",
        "o que eu estava fazendo ontem às 10?",
        "onde fica a parada?",
        "em que ano eu parei de estudar?",
        "onde paramos o carro?",
    ])
    def test_a_normal_question(self, text):
        assert not focus.asks_where_was_i(text)


class TestTheRecord:

    def test_start_opens_one_session(self, store):
        s = store.start("candidatura", "abrir o formulário", 25, T0)
        assert s.is_open and s.ends_at == T0 + timedelta(minutes=25)
        assert store.current() == store.last()
        assert store.current().first_step == "abrir o formulário"

    def test_a_new_start_closes_the_open_one(self, store):
        old = store.start("candidatura", "abrir", 25, T0)
        new = store.start("estudo", "ler o texto A", 25, T0 + timedelta(minutes=5))
        assert store.current().id == new.id
        assert store.last().id == new.id
        rows = [store._one("SELECT * FROM sessions WHERE id = ?", (old.id,))]
        assert not rows[0].is_open and rows[0].next_step is None

    def test_stop_keeps_the_next_step(self, store):
        store.start("candidatura", "abrir", 25, T0)
        done = store.stop("preencher a seção 2", T0 + timedelta(minutes=20))
        assert not done.is_open and done.next_step == "preencher a seção 2"
        assert store.current() is None
        assert store.last().next_step == "preencher a seção 2"

    def test_a_bare_stop_saves_no_next_step(self, store):
        store.start("candidatura", "abrir", 25, T0)
        assert store.stop("", T0).next_step is None

    def test_stop_with_nothing_open_invents_nothing(self, store):
        assert store.stop("x", T0) is None
        assert store.last() is None

    def test_the_reminder_ref_is_kept(self, store):
        s = store.start("candidatura", "abrir", 25, T0)
        assert store.set_reminders(s.id, "x-apple-reminder://ABC", "x-apple-reminder://DEF")
        assert store.current().reminder == "x-apple-reminder://ABC"
        assert store.current().nudge_reminder == "x-apple-reminder://DEF"

    def test_survives_a_new_process(self, tmp_path):
        path = str(tmp_path / "focus.db")
        FocusStore(db_path=path).start("candidatura", "abrir", 25, T0)
        again = FocusStore(db_path=path).current()
        assert again.task == "candidatura" and again.ends_at == T0 + timedelta(minutes=25)

    def test_a_broken_path_degrades_to_no_session(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        s = FocusStore(db_path=str(blocker / "focus.db"))   # parent is a file
        assert s.start("a", "b", 25, T0) is None
        assert s.current() is None and s.last() is None


class TestTheClockChange:

    def test_25_minutes_across_the_clocks_going_back(self, store):
        """01:50 BST on 25/10/2026; at 02:00 the clocks go back to 01:00.
        25 minutes later is 01:15 GMT — the wall-clock sum said 02:15."""
        start = datetime(2026, 10, 25, 1, 50, tzinfo=LONDON)
        s = store.start("estudo", "ler", 25, start)
        assert s.ends_at.isoformat() == "2026-10-25T01:15:00+00:00"
        assert store.current().ends_at.isoformat() == "2026-10-25T01:15:00+00:00"

    def test_time_is_up_in_the_repeated_hour(self, store):
        start = datetime(2026, 10, 25, 1, 50, tzinfo=LONDON)
        s = store.start("estudo", "ler", 25, start)
        # 01:10 GMT (fold=1) comes AFTER 01:50 BST, but before the end.
        assert not s.time_is_up(datetime(2026, 10, 25, 1, 10, fold=1, tzinfo=LONDON))
        assert s.time_is_up(datetime(2026, 10, 25, 1, 15, fold=1, tzinfo=LONDON))


class TestTheNudgeTime:
    """The nudge is a second reminder 10 min past the end (Pedro, 16/10)."""

    def test_ten_minutes_after_the_end(self, store):
        s = store.start("candidatura", "abrir", 25, T0)
        assert focus.later(s.ends_at, focus.NUDGE_AFTER) == T0 + timedelta(minutes=35)

    def test_across_the_clocks_going_back(self):
        """Ends 01:55 BST on 25/10; 10 min later is 01:05 GMT, not 02:05."""
        ends = datetime(2026, 10, 25, 1, 55, tzinfo=LONDON)
        assert focus.later(ends, focus.NUDGE_AFTER).isoformat() == "2026-10-25T01:05:00+00:00"


class TestANewStartWhileOneIsOpen:
    """Pedro's rule (16/10): ask only while the old session is still inside
    its time — a slip can't end a session that was going well, and once the
    time is up, starting again costs no extra word."""

    def test_mid_session_asks_first(self, store):
        old = store.start("candidatura", "abrir", 25, T0)
        assert focus.on_start_while_open(old, T0 + timedelta(minutes=10)) == focus.ASK_FIRST

    def test_after_the_time_is_up_starts_new(self, store):
        old = store.start("candidatura", "abrir", 25, T0)
        assert focus.on_start_while_open(old, old.ends_at) == focus.START_NEW
        assert focus.on_start_while_open(old, old.ends_at + timedelta(hours=2)) == focus.START_NEW


# --- review fixes (s86) -------------------------------------------------

class TestNoRegexFreeze:
    """"parei: a" + 1,000 spaces + "b" took 1.3 s and 8,000 spaces ran past
    two minutes: two \\s* around a lazy .*? let the engine try every way to
    split the spaces. These run on every message in the reply path."""

    @pytest.mark.parametrize("text", [
        "parei: a" + " " * 2000 + "b",
        "parei" + " " * 2000 + "x",
        "foco: a" + " " * 2000 + "b c",
        "foco 40" + " " * 2000 + "x",
        "cortex" + " " * 2000 + "x",
        "onde parei" + " " * 2000 + "x",
    ])
    def test_long_whitespace_is_instant(self, text):
        t0 = time.perf_counter()
        focus.parse_start(text)
        focus.parse_stop(text)
        focus.asks_where_was_i(text)
        assert time.perf_counter() - t0 < 0.1

    def test_the_spaces_inside_a_step_are_collapsed(self):
        assert focus.parse_stop("parei:   preencher    a seção 2  !") == "preencher a seção 2"
        assert focus.parse_start("foco 40 :   ler   o texto  ") == ("ler o texto", 40)


class TestAFailedWriteLeavesNothingOpen:
    """The UI, the terminal and the nudge driver are three PROCESSES on one
    file. A start() whose INSERT failed after its UPDATE left the transaction
    open: the other processes got "database is locked", and this one saw no
    open session while they still saw the old one."""

    def test_a_failed_start_is_rolled_back(self, tmp_path):
        path = str(tmp_path / "focus.db")
        s = FocusStore(db_path=path)
        s.start("antigo", "a", 25, T0)
        s._conn.execute(
            "CREATE TRIGGER boom BEFORE INSERT ON sessions WHEN NEW.task = 'novo' "
            "BEGIN SELECT RAISE(ABORT, 'io'); END")
        s._conn.commit()

        assert s.start("novo", "b", 25, T0) is None
        assert not s._conn.in_transaction
        assert s.current().task == "antigo"          # the UPDATE was undone too
        other = sqlite3.connect(path, timeout=0.5)   # another process
        other.execute("UPDATE sessions SET next_step = 'x' WHERE id = 1")
        other.commit()
        other.close()


class TestStopIsOneStep:
    """stop() read the open session, then wrote to it. A "foco:" from the UI
    landing in between had the next step written on the session it had just
    closed, and the new session stayed open with no one told. The read before
    the write is where the other process gets in: it is simulated here, and
    with the fix stop() no longer has that read at all."""

    def test_a_start_in_between_cannot_misplace_the_next_step(self, tmp_path):
        path = str(tmp_path / "focus.db")
        terminal, ui = FocusStore(db_path=path), FocusStore(db_path=path)
        terminal.start("candidatura", "abrir", 25, T0)
        real_current = terminal.current

        def read_then_ui_starts():
            seen = real_current()
            ui.start("estudo", "ler", 25, T0 + timedelta(minutes=1))
            return seen
        terminal.current = read_then_ui_starts

        done = terminal.stop("seção 2", T0 + timedelta(minutes=2))
        assert done.next_step == "seção 2"
        # Whatever got the next step was the session open AT the write: no
        # session is left open behind the "parei".
        assert ui.current() is None


class TestAWriteThatChangedNothing:
    """A write that changed no row is not a success."""

    def test_no_such_session(self, store):
        assert store.set_reminders(999, "x-apple-reminder://X", None) is False
        assert store.set_next_step(999, "x") is False


class TestABrokenRowIsNoSession:
    """The module promises a storage hiccup never reaches the reply. A row
    with a broken date was converted outside the try."""

    def test_a_broken_date(self, tmp_path):
        path = str(tmp_path / "focus.db")
        s = FocusStore(db_path=path)
        s.start("candidatura", "abrir", 25, T0)
        s._conn.execute("UPDATE sessions SET ends_at = 'quebrado'")
        s._conn.commit()
        assert s.current() is None and s.last() is None


class TestTheCommandSaysTheLanguage:
    """"foco: imposto" has no accent and read as English to the detector."""

    @pytest.mark.parametrize("text, lang", [
        ("foco: imposto", "pt"), ("foco 40: cv", "pt"), ("parei", "pt"),
        ("onde parei?", "pt"), ("focus: tax return", "en"),
        ("Where was I?", "en"), ("cortex, what was I doing?", "en"),
    ])
    def test_lang(self, text, lang):
        assert focus.command_lang(text) == lang
