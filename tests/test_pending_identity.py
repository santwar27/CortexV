"""A "sim" confirms the action whose preview was shown — never another one.

The pending action was a single slot: "sim" ran whatever was in it at that
moment (C1 + Codex lote 2, reproduced s88). Three ways it ran the wrong one:

  1. Two conversations: conversation A's slow "planeja:" finished after B's
     reminder preview and wrote its plan over it; B's "sim" ran A's plan.
  2. One tab: Stop aborts the browser side only — the server's turn goes on,
     stages its plan after the reminder Pedro asked for next, and his "sim"
     runs a plan whose preview nobody received.
  3. Server and terminal: each process keeps its own copy. The terminal's
     "sim" sent again an email the UI had already sent — or had CANCELLED.

The model and the integrations are stubbed: what each test records is what
WOULD run. All five failed on the code before the fix (commit 909caa0, as
xfail): now the action has an id and the disk is the truth.
"""
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core import planner as _planner
from tests.test_dispatch_parity import _build_real_orch

LONDON = ZoneInfo("Europe/London")
PLAN_ASK = "planeja: manda um email pro João dizendo que atraso"
REMINDER_ASK = "me lembra de pagar a conta amanhã às 10"
EMAIL = {"type": "send_email",
         "args": {"to": "joao@exemplo.com", "subject": "Atraso", "body": "Vou atrasar."}}


def _orch(tmp_path, monkeypatch, ran, who="server"):
    o = _build_real_orch(tmp_path)
    o.background = False
    o._offer = None
    monkeypatch.setattr(o, "_run_action", lambda name, args, effect_id=None: (
        ran.append((who, name)), ("ok", "AGENT_ACTION_DONE"))[1])
    monkeypatch.setattr(o, "_execute_staged_plan", lambda action=None: (
        ran.append((who, "PLAN")), ("ok", "AGENT_PLAN_DONE"))[1])
    return o


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr("src.core.dispatch.registry.local_now",
                        lambda: datetime(2026, 10, 17, 15, 0, tzinfo=LONDON))


@pytest.fixture
def slow_plan(monkeypatch):
    """decompose() waits for the test to let it finish — the 8B thinking."""
    entered, release = threading.Event(), threading.Event()

    def decompose(objective, **_kw):
        entered.set()
        assert release.wait(10)
        return _planner.Plan(objetivo=objective, passos=[_planner.Step(
            id="p1", tool="send_email", depende_de=[], descricao="avisar",
            args={"to": "joao@exemplo.com", "subject": "Atraso", "body": "Vou atrasar."})])
    monkeypatch.setattr(_planner, "decompose", decompose)
    return entered, release


def _turn(o, text, history=None):
    *_, (reply, _final, mode, _lang) = o.process_stream(
        text, None, conversation_history=history or [])
    return reply, mode


class TestTwoConversations:

    def test_b_s_yes_runs_b_s_reminder_not_a_s_plan(self, tmp_path, monkeypatch,
                                                     clock, slow_plan):
        entered, release = slow_plan
        ran = []
        o = _orch(tmp_path, monkeypatch, ran)
        a = {}
        worker = threading.Thread(target=lambda: a.setdefault("r", _turn(o, PLAN_ASK)))
        worker.start()
        assert entered.wait(10)                         # A: decompose() running
        preview, mode = _turn(o, REMINDER_ASK)          # B: its own preview
        assert mode == "AGENT_AWAITING_CONFIRMATION" and "pagar a conta" in preview
        release.set()
        worker.join(10)
        assert a["r"][1] == "AGENT_STAGE_CONFLICT"      # A is told, not staged
        _turn(o, "sim", [{"role": "assistant", "content": preview}])
        assert ran == [("server", "create_reminder")]


class TestOneTabAfterStop:

    def test_a_plan_nobody_saw_is_never_run(self, tmp_path, monkeypatch, clock, slow_plan):
        entered, release = slow_plan
        ran = []
        o = _orch(tmp_path, monkeypatch, ran)
        # The user asks for a plan and presses Stop: the browser drops the stream,
        # the server's turn goes on to the end — nobody reads its reply.
        worker = threading.Thread(target=lambda: _turn(o, PLAN_ASK))
        worker.start()
        assert entered.wait(10)
        history = [{"role": "user", "content": PLAN_ASK}]
        preview, _ = _turn(o, REMINDER_ASK, history)
        release.set()
        worker.join(10)
        assert o._pending_action["type"] == "create_reminder"
        _turn(o, "sim", history + [{"role": "assistant", "content": preview}])
        assert ran == [("server", "create_reminder")]


class TestServerAndTerminal:

    def _pair(self, tmp_path, monkeypatch, ran):
        server = _orch(tmp_path, monkeypatch, ran, "server")
        server._pending_action = dict(EMAIL)
        server._persist_pending_action()                 # the UI staged it
        terminal = _orch(tmp_path, monkeypatch, ran, "terminal")
        terminal._pending_action = terminal._load_pending_action()   # opens now
        return server, terminal

    def test_sent_by_the_ui_is_not_sent_again(self, tmp_path, monkeypatch):
        ran = []
        server, terminal = self._pair(tmp_path, monkeypatch, ran)
        server.process("sim")
        terminal.process("sim")
        assert ran == [("server", "send_email")]

    def test_cancelled_in_the_ui_is_not_sent_from_the_terminal(self, tmp_path, monkeypatch):
        ran = []
        server, terminal = self._pair(tmp_path, monkeypatch, ran)
        server.process("não")
        terminal.process("sim")
        assert ran == []

    def test_two_yeses_at_once_send_once(self, tmp_path, monkeypatch):
        """The terminal's turn has already read the action when the UI's "sim"
        consumes it: only one of the two may run it."""
        ran = []
        server, terminal = self._pair(tmp_path, monkeypatch, ran)
        real = terminal._execute_pending_action

        def ui_gets_there_first():
            server.process("sim")
            return real()
        monkeypatch.setattr(terminal, "_execute_pending_action", ui_gets_there_first)
        terminal.process("sim")
        assert ran == [("server", "send_email")]
