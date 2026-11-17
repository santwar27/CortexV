"""'marca dentista amanhã às 10' books tomorrow at 10 — without the 8B (s85, 4.2).

Measured: no route took verb-first booking without a calendar keyword, so
"marca dentista amanhã às 10" got "não consegui agendar" 6 in 6; and with
the tools offered ("marcar …") the 8B staged the event on 23/05, 20/05,
19/05 at -03:00 for a "tomorrow" of 16/10, 6 in 6. The deterministic
calendar handler already parses "<title> <time>"; CALENDAR_VERB_RE only has
to hand it the body. The noun "marca" must stay out.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.core.dispatch.calendar import CALENDAR_VERB_RE
from src.core.orchestrator import CortexOrchestrator


@pytest.fixture
def orch(tmp_path):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    return o


def _tomorrow_at(hour):
    now = datetime.now(ZoneInfo("Europe/London"))
    return (now + timedelta(days=1)).date(), hour


class TestBooking:

    @pytest.mark.parametrize("text,hour", [
        ("marca dentista amanhã às 10", 10),
        ("marca o dentista amanhã às 10", 10),
        ("marque consulta no dentista amanhã às 15h", 15),
        ("marcar dentista amanhã às 10", 10),
        ("pode marcar o dentista amanhã às 9?", 9),
    ])
    def test_tomorrow_at_the_hour_said(self, orch, text, hour):
        reply, mode = orch._process_core(text)
        assert mode == "AGENT_AWAITING_CONFIRMATION", (text, reply)
        args = orch._pending_action["args"]
        start = datetime.fromisoformat(args["start_iso"]).astimezone(
            ZoneInfo("Europe/London"))
        assert (start.date(), start.hour) == _tomorrow_at(hour)
        assert "dentista" in args["summary"].lower()
        orch.engine_analytical.chat_with_tools.assert_not_called()


class TestTheNounStaysOut:

    @pytest.mark.parametrize("text", [
        "qual a marca do seu carro?",
        "marca boa é a Nike",
        "marca o ponto",                 # a verb, but no time: not a booking
        "essa marca é cara",
    ])
    def test_not_a_booking(self, orch, text):
        _reply, mode = orch._process_core(text)
        assert mode != "AGENT_AWAITING_CONFIRMATION", text

    def test_the_body_starts_at_the_title(self):
        m = CALENDAR_VERB_RE.match("marca dentista amanhã às 10")
        assert m and m.group("body").strip().lower().startswith("dentista")


class TestBothEntryPoints:

    def test_the_stream_does_not_chat_over_it(self, orch):
        assert orch._is_pure_chat("marca dentista amanhã às 10") is False

    def test_bound_and_vetoed(self):
        assert "_CALENDAR_VERB_RE" in CortexOrchestrator._PURE_CHAT_BLOCKING_RES
