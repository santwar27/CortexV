"""Calendar questions use the local clock, never a model (Task 128)."""
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.core.reflex import reflex_layer
from tests.test_dispatch_parity import _build_real_orch


DATE_PT = "Hoje é terça-feira, 3 de novembro de 2026."
DATE_EN = "Today is Tuesday, 3 November 2026."
YEARS_PT = [
    "em que ano estamos", "estamos em que ano", "que ano é hoje",
    "que ano é esse", "qual o ano atual", "qual é o ano",
    "em que ano a gente está",
]
YEARS_EN = ["what year is it", "what year are we in", "which year is it"]
DATES_PT = [
    "que dia é hoje", "qual a data de hoje", "qual data de hoje",
    "qual a data", "que data é hoje", "data de hoje", "hoje é que dia",
    "hoje é qual dia", "que dia da semana", "qual o dia da semana",
    "que dia da semana é hoje", "hoje é que dia da semana",
    "qual é a data de hoje", "me diz que dia é hoje", "qual o dia de hoje",
]
DATES_EN = [
    "what's the date today", "what is today's date", "what's today's date",
    "what day is it today", "what day of the week is it", "what's the date",
    "what is the date", "what is today", "what's today", "what day is it",
    "what day is today",
]
COMBINED_PT = [
    f"{first}{join}{second}"
    for year in YEARS_PT
    for first, second in [(year, "qual o dia de hoje"), ("que dia é hoje", year)]
    for join in [" e ", ", "]
]
COMBINED_EN = [
    f"{first}{join}{second}"
    for year in YEARS_EN
    for first, second in [(year, "what's the date"), ("what's the date", year)]
    for join in [" and ", ", "]
]
TOLERANCES = [
    ("  EM QUE ANO ESTAMOS?!  ", "Estamos em 2026."),
    ("me diz em que ano estamos agora!", "Estamos em 2026."),
    ("me fala qual e o ano hoje.", "Estamos em 2026."),
    ("me fala qual o ano atual agora?", "Estamos em 2026."),
    ("qual o ano atual hoje", "Estamos em 2026."),
    ("me informa em que ano a gente esta?", "Estamos em 2026."),
    ("me fala ai que ano e esse atual", "Estamos em 2026."),
    ("me informa que dia e hoje agora?", DATE_PT),
    ("me fala que dia da semana e hoje atual!", DATE_PT),
    ("me diz hoje e que dia da semana?", DATE_PT),
    ("qual e a data de hoje hoje.", DATE_PT),
    ("me diz em que ano estamos agora e me fala qual o dia de hoje?", DATE_PT),
    ("me informa que dia e hoje, me diz em que ano a gente esta atual!", DATE_PT),
    ("WHAT YEAR IS IT?!", "It's 2026."),
    ("what year are we in today.", "It's 2026."),
    ("what's the date now?", DATE_EN),
    ("what day of the week is it today!", DATE_EN),
    ("what year is it today and what's the date now?", DATE_EN),
]
CALENDAR_CASES = (
    [(text, "Estamos em 2026.") for text in YEARS_PT]
    + [(text, "It's 2026.") for text in YEARS_EN]
    + [(text, DATE_PT) for text in DATES_PT + COMBINED_PT]
    + [(text, DATE_EN) for text in DATES_EN + COMBINED_EN]
    + TOLERANCES
)


@pytest.fixture
def clock(monkeypatch):
    clock = Mock(return_value=datetime(2026, 11, 3, 12, 30))
    monkeypatch.setattr("src.core.reflex._now_local", clock)
    return clock


@pytest.fixture(params=["reflex", "process", "process_stream"])
def respond(request, tmp_path):
    if request.param == "reflex":
        return reflex_layer.try_respond

    orch = _build_real_orch(tmp_path)
    orch.background = False
    orch._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    forbidden = []
    for engine in (orch.engine_analytical, orch.engine_creative):
        for name in ("chat", "chat_stream", "chat_with_tools", "generate"):
            method = getattr(engine, name)
            method.side_effect = lambda *a, **kw: pytest.fail("calendar called engine")
            forbidden.append(method)
    for name in ("_needs_realtime", "_handle_planner_request"):
        method = Mock(side_effect=lambda *a, **kw: pytest.fail("calendar classified/planned"))
        setattr(orch, name, method)
        forbidden.append(method)

    def call(text):
        if request.param == "process":
            reply, mode = orch.process(text)
            assert mode == "REFLEX"
        else:
            frames = list(orch.process_stream(text))
            assert len(frames) == 1
            reply, final, mode, lang = frames[0]
            assert final is True
            assert mode == "REFLEX"
            assert lang == ("pt" if reply.startswith(("Hoje", "Estamos", "São")) else "en")
        for method in forbidden:
            method.assert_not_called()
        return reply

    return call


@pytest.mark.parametrize("text, expected", CALENDAR_CASES)
def test_calendar_from_one_local_clock_reading(respond, clock, text, expected):
    assert respond(text) == expected
    clock.assert_called_once_with()


@pytest.mark.parametrize("text, before, after", [
    ("em que ano estamos", "Estamos em 2026.", "Estamos em 2027."),
    ("what year is it", "It's 2026.", "It's 2027."),
    ("em que ano estamos e qual o dia de hoje",
     "Hoje é quinta-feira, 31 de dezembro de 2026.",
     "Hoje é sexta-feira, 1 de janeiro de 2027."),
    ("what's the date and what year is it",
     "Today is Thursday, 31 December 2026.", "Today is Friday, 1 January 2027."),
    ("hora e data", "São 23:59. Hoje é quinta-feira, 31 de dezembro de 2026.",
     "São 00:00. Hoje é sexta-feira, 1 de janeiro de 2027."),
])
def test_year_rollover_uses_one_fresh_snapshot_per_reply(respond, clock, text, before, after):
    clock.side_effect = [datetime(2026, 12, 31, 23, 59, 59), datetime(2027, 1, 1)]
    assert respond(text) == before
    clock.assert_called_once_with()
    assert respond(text) == after
    assert clock.call_count == 2


@pytest.mark.parametrize("text", [
    "que dia é a prova?", "que dia é o jogo do Brasil?", "qual o dia da reunião?",
    "marca pra que dia?", "lembra daquele dia?",
    "em que ano foi a independência do Brasil?", "em que ano eu me formei?",
    "o que eu tenho hoje?", "minha agenda de hoje", "que dia é amanhã?",
    "what day is the meeting?", "what year did the war end?", "what's on today?",
    "em que ano estamos e que dia é a prova?",
    "que dia é hoje e qual o dia da reunião?",
    "what year is it and what day is the meeting?",
    "what's the date tomorrow?", "que ano é esse filme?",
    "me diz que dia é amanhã?", "que dia e amanha?",
])
def test_calendar_misses_do_not_read_clock(clock, text):
    assert reflex_layer.try_respond(text) is None
    clock.assert_not_called()
