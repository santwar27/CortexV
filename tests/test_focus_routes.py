"""Focus sessions answered by fixed rules, in both entry points (s86 bloco 2a).

Measured before (real 8B, process and process_stream): 1 useful reply in 30.
"foco: candidatura…" became a job search; the first step "abrir o
formulário" tried to open an app called "formulário" (4 of 4); "parei: …"
got "não posso ajudar com isso"; "onde parei?" answered "Curitiba, Minas
Gerais." (6 of 6). Every reply here is a fixed text, and the 8B is never
called in the cycle.
"""
import json
import string
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.core import focus
from src.core import focus_mixin
from src.core.audit import audit
from src.core.focus_mixin import FOCUS_TEXTS

LONDON = ZoneInfo("Europe/London")
T0 = datetime(2026, 10, 16, 15, 15, tzinfo=LONDON)       # a Friday


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def clock(monkeypatch):
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


def say(orch, text):
    return orch._process_core(text)


def no_llm(orch):
    orch.engine_analytical.chat.assert_not_called()
    orch.engine_analytical.chat_with_tools.assert_not_called()


class TestTheCycle:

    def test_start_stop_come_back(self, orch, clock):
        reply, mode = say(orch, "foco: candidatura da vaga")
        assert mode == "FOCUS_ASK_STEP" and "candidatura da vaga" in reply

        reply, mode = say(orch, "abrir o formulário")
        assert mode == "FOCUS_STARTED"
        assert "15:40" in reply and "abrir o formulário" in reply
        orch.macos.open_app.assert_not_called()          # the bloco 0 bug

        clock.advance(minutes=20)
        reply, mode = say(orch, "parei: preencher a seção 2")
        assert mode == "FOCUS_CLOSED" and "preencher a seção 2" in reply

        clock.advance(hours=2)
        reply, mode = say(orch, "onde parei?")
        assert mode == "FOCUS_WHERE"
        assert "candidatura da vaga" in reply and "preencher a seção 2" in reply
        assert "hoje às 15:35" in reply

        reply, mode = say(orch, "sim")
        assert mode == "FOCUS_STARTED" and "preencher a seção 2" in reply
        assert focus.store.current().first_step == "preencher a seção 2"
        no_llm(orch)

    def test_a_command_without_accents_is_answered_in_portuguese(self, orch):
        """"foco: imposto" read as English to the detector (no accent)."""
        reply, _ = say(orch, "foco: imposto")
        assert reply == FOCUS_TEXTS["ask_step"][0].format(task="imposto")
        reply, _ = say(orch, "ok")                # "ok" reads as English too
        assert FOCUS_TEXTS["default_step"][0] in reply

    def test_other_durations(self, orch):
        say(orch, "foco 40: estudar")
        reply, _ = say(orch, "ler o texto A")
        assert "15:55" in reply

    @pytest.mark.parametrize("vague", ["não sei", "sei lá", "ok", "beleza"])
    def test_a_vague_answer_starts_with_the_default_step(self, orch, vague):
        say(orch, "foco: imposto")
        reply, mode = say(orch, vague)
        assert mode == "FOCUS_STARTED"
        assert FOCUS_TEXTS["default_step"][0] in reply

    def test_no_means_no_focus(self, orch):
        say(orch, "foco: imposto")
        _reply, mode = say(orch, "não")
        assert mode == "FOCUS_CANCELLED" and focus.store.current() is None

    def test_a_question_is_not_a_first_step(self, orch):
        say(orch, "foco: imposto")
        _reply, mode = say(orch, "que horas são?")
        assert not mode.startswith("FOCUS_")
        assert focus.store.current() is None

    def test_the_question_lasts_one_turn(self, orch):
        say(orch, "foco: imposto")
        say(orch, "que horas são?")
        _reply, mode = say(orch, "abrir o site")
        assert not mode.startswith("FOCUS_")


class TestANewFocusWhileOneIsOpen:
    """Pedro's rule: ask only while the old one is still inside its time."""

    def _open(self, orch):
        say(orch, "foco: candidatura")
        say(orch, "abrir o formulário")

    def test_mid_session_asks_and_no_keeps_it(self, orch, clock):
        self._open(orch)
        clock.advance(minutes=10)
        reply, mode = say(orch, "foco: estudar")
        assert mode == "FOCUS_ASK_SWITCH"
        assert "candidatura" in reply and "estudar" in reply and "15:40" in reply
        reply, mode = say(orch, "não")
        assert mode == "FOCUS_KEPT" and focus.store.current().task == "candidatura"

    def test_mid_session_yes_switches(self, orch, clock):
        self._open(orch)
        clock.advance(minutes=10)
        say(orch, "foco: estudar")
        _reply, mode = say(orch, "sim")
        assert mode == "FOCUS_ASK_STEP"
        assert focus.store.current().task == "candidatura"   # not before the step
        _reply, mode = say(orch, "ler o texto A")
        assert mode == "FOCUS_STARTED" and focus.store.current().task == "estudar"

    def test_after_the_time_is_up_no_question(self, orch, clock):
        self._open(orch)
        clock.advance(minutes=30)
        _reply, mode = say(orch, "foco: estudar")
        assert mode == "FOCUS_ASK_STEP"


class TestStopping:

    def test_a_bare_stop_asks_for_the_next_step(self, orch):
        say(orch, "foco: candidatura")
        say(orch, "abrir o formulário")
        reply, mode = say(orch, "parei")
        assert mode == "FOCUS_ASK_NEXT" and "candidatura" in reply
        reply, mode = say(orch, "revisar o texto")
        assert mode == "FOCUS_NEXT_SAVED" and "revisar o texto" in reply
        assert focus.store.last().next_step == "revisar o texto"

    def test_stop_with_nothing_open_says_so(self, orch):
        reply, mode = say(orch, "parei: x")
        assert mode == "FOCUS_NONE"
        no_llm(orch)

    @pytest.mark.parametrize("command, expected", [
        ("parei", "FOCUS_ASK_NEXT"),
        ("parei:", "FOCUS_ASK_NEXT"),
        ("onde parei?", "FOCUS_WHERE"),
    ])
    def test_a_pending_web_question_does_not_swallow_it(self, orch, command, expected):
        """Short enough for the "garbled STT? ask sim ou não again" rule — it
        answered "Não entendi — sim ou não?" and the focus stayed open (s88,
        Codex lote 1)."""
        say(orch, "foco: candidatura")
        say(orch, "abrir o formulário")
        orch._pending_query = "qual o placar do jogo do Brasil"
        _reply, mode = say(orch, command)
        assert mode == expected
        assert orch._pending_query is None


class TestComingBack:

    def test_nothing_saved_yet(self, orch):
        reply, mode = say(orch, "onde parei?")
        assert mode == "FOCUS_WHERE" and reply == FOCUS_TEXTS["where_none"][0]
        no_llm(orch)

    def test_while_running(self, orch, clock):
        say(orch, "foco: candidatura")
        say(orch, "abrir o formulário")
        clock.advance(minutes=5)
        reply, _ = say(orch, "o que eu estava fazendo?")
        assert "candidatura" in reply and "15:40" in reply and "abrir o formulário" in reply

    def test_time_up_without_a_stop_offers_more(self, orch, clock):
        say(orch, "foco: candidatura")
        say(orch, "abrir o formulário")
        clock.advance(minutes=40)
        reply, mode = say(orch, "onde parei?")
        assert mode == "FOCUS_WHERE" and "hoje às 15:40" in reply
        _reply, mode = say(orch, "sim")
        assert mode == "FOCUS_ASK_STEP"          # no next step known: ask one

    @pytest.mark.parametrize("later, said", [
        (timedelta(days=1), "ontem às 15:15"),
        (timedelta(days=3), "na sexta às 15:15"),      # closed on Friday 16/10
        (timedelta(days=12), "em 16/10"),
    ])
    def test_when_it_was(self, orch, clock, later, said):
        say(orch, "foco: candidatura")
        say(orch, "abrir")
        say(orch, "parei: seção 2")
        clock.advance(**{"seconds": later.total_seconds()})
        reply, _ = say(orch, "onde parei?")
        assert said in reply

    def test_in_english(self, orch):
        say(orch, "focus: write the cover letter")
        say(orch, "open the template")
        say(orch, "parei: second paragraph")
        reply, _ = say(orch, "where was I?")
        assert reply.startswith("You were on write the cover letter")


class TestBothEntryPoints:
    """process_stream's shortcuts (reflex, streamed chat) must step aside —
    the same predicate _process_core's route uses."""

    def _stream(self, orch, text):
        parts, mode = [], None
        for chunk, final, m, _lang in orch.process_stream(text):
            parts.append(chunk)
            if final:
                mode = m
        return "".join(parts), mode

    def test_the_stream_runs_the_cycle(self, orch):
        _r, mode = self._stream(orch, "foco: candidatura")
        assert mode == "FOCUS_ASK_STEP"
        _r, mode = self._stream(orch, "abrir o formulário")
        assert mode == "FOCUS_STARTED"
        orch.macos.open_app.assert_not_called()
        _r, mode = self._stream(orch, "onde parei?")
        assert mode == "FOCUS_WHERE"
        orch.engine_analytical.chat_stream.assert_not_called()

    def test_pure_chat_vetoes(self, orch):
        assert orch._is_pure_chat("foco: candidatura") is False
        assert orch._is_pure_chat("parei: seção 2") is False
        assert orch._is_pure_chat("onde parei?") is False
        orch._offer = {"kind": "focus_step", "task": "x", "minutes": 25}
        assert orch._is_pure_chat("revisar o texto") is False   # a first step

    def test_a_reflex_phrase_is_never_a_first_step(self, orch):
        """"que horas são" typed without the "?" must still get the time."""
        self._stream(orch, "foco: candidatura")
        _r, mode = self._stream(orch, "que horas são")
        assert mode == "REFLEX" and focus.store.current() is None

    def test_without_a_focus_the_reflex_still_answers(self, orch):
        _r, mode = self._stream(orch, "obrigado")
        assert mode == "REFLEX"


class TestTheTexts:
    """Pedro writes these. A text may drop a field, never invent one: an
    unknown {name} would print as-is in the reply."""

    PROVIDED = {
        "ask_step": {"task"}, "started": {"task", "until", "first_step"},
        "default_step": set(), "cancelled": set(),
        "ask_switch": {"task", "old_task", "until"}, "kept": {"old_task", "until"},
        "closed": {"next_step"}, "ask_next": {"task"}, "next_saved": {"next_step"},
        "next_skipped": set(), "nothing_open": set(),
        "where_running": {"task", "until", "first_step"},
        "where_time_up": {"task", "minutes", "when"},
        "where_next": {"task", "when", "next_step", "minutes"},
        "where_no_next": {"task", "when", "minutes"},
        "where_none": set(), "error": set(),
        "reminder_title": {"task"}, "reminder_body": set(), "reminder_failed": set(),
        "nudge_title": {"task"}, "nudge_body": set(),
        "reminder_unknown": set(),
    }
    # What a text must keep for the reply to make sense.
    REQUIRED = {
        "ask_step": {"task"}, "started": {"until", "first_step"},
        "ask_switch": {"old_task", "task"}, "closed": {"next_step"},
        "where_running": {"task"}, "where_time_up": {"task"},
        "where_next": {"task", "next_step"}, "where_no_next": {"task"},
        "reminder_title": {"task"},          # Pedro, 16/10: the task in the title
        "nudge_title": {"task"},
    }

    def test_every_route_text_exists(self):
        assert set(FOCUS_TEXTS) == set(self.PROVIDED)

    @pytest.mark.parametrize("key", sorted(FOCUS_TEXTS))
    def test_fields(self, key):
        for lang, text in zip(("pt", "en"), FOCUS_TEXTS[key]):
            used = {f for _, f, _, _ in string.Formatter().parse(text) if f}
            assert used <= self.PROVIDED[key], f"{key} ({lang}) names {used - self.PROVIDED[key]}"
            assert self.REQUIRED.get(key, set()) <= used, (
                f"{key} ({lang}) must keep {self.REQUIRED[key] - used}")


class TestTheAuditKeepsNoTaskText:

    def test_counts_only(self, orch, clock):
        say(orch, "foco: candidatura secreta")
        say(orch, "abrir o formulário secreto")
        clock.advance(minutes=10)
        say(orch, "parei: passo secreto")
        say(orch, "onde parei?")
        events = [json.loads(line) for line in open(audit.path)]
        mine = [e for e in events if e.get("stage") == "focus"]
        assert {e["event"] for e in mine} >= {"ask_step", "start", "close", "where"}
        assert not any("secret" in json.dumps(e) for e in mine)
        close = next(e for e in mine if e["event"] == "close")
        assert close["payload"] == {"next_step": True, "early": True, "minutes": 10}
