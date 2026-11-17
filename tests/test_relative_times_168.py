"""Relative times are facts computed by code, on both scheduling paths (168)."""
from datetime import datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from src.core import clock
from src.core.clock import add_elapsed, ensure_event_end, parse_relative_times
from src.core.tool_risk import _PREVIEW_BUILDERS, build_preview

LONDON = ZoneInfo("Europe/London")
NOW = datetime(2026, 11, 16, 20, 18, tzinfo=LONDON)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(clock, "local_zone", lambda: LONDON)
    monkeypatch.setattr("src.core.dispatch.registry.local_now", lambda: NOW)
    monkeypatch.setattr("src.core.staging_mixin.local_now", lambda: NOW)


@pytest.fixture
def orch(tmp_path):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    return o


@pytest.mark.parametrize("expression,minutes", [
    ("daqui a 2 horas", 120), ("DAQUI A 30 MINUTOS", 30),
    ("em 1 hora", 60), ("dentro de 45 min", 45), ("em 2 h", 120),
    ("daqui a meia hora", 30), ("em uma hora", 60), ("em duas horas", 120),
    ("em três minutos", 3), ("em quatro horas", 240), ("em cinco horas", 300),
    ("em seis min", 6), ("em sete horas", 420), ("em oito horas", 480),
    ("em nove horas", 540), ("em dez horas", 600),
    ("daqui a uma hora e meia", 90), ("em 2h30", 150),
    ("dentro de 1 hora e 15 minutos", 75), ("in 2 hours", 120),
    ("in 20 minutes", 20), ("in an hour", 60), ("in half an hour", 30),
    ("in an hour and a half", 90), ("in one hour and 15 minutes", 75),
    ("em 168 horas", 10080),
])
def test_offsets_and_exact_spans(expression, minutes):
    text = f"prefix {expression} suffix"
    times = parse_relative_times(text)
    assert len(times.offsets) == 1
    offset, = times.offsets
    assert offset.delta == timedelta(minutes=minutes)
    assert offset.span == (7, 7 + len(expression))
    assert text[slice(*offset.span)] == expression
    assert not times.durations


@pytest.mark.parametrize("text,duration", [
    ("às 2 horas", None), ("2 horas da tarde", None), ("amanhã às 1h", None),
    ("2h30", None), ("uma hora e meia", None), ("in the morning", None),
    ("por meia hora", 30), ("por 2 horas", 120), ("for 30 minutes", 30),
    ("de 1 hora", 60), ("durante 40 min", 40), ("for an hour and a half", 90),
    ("em 0 minutos", None), ("in 0 hours", None), ("em 169 horas", None),
    ("in 10081 minutes", None), ("em -2 horas", None), ("em 2h99", None),
    ("por 0 horas", None), ("for 169 hours", None),
])
def test_non_offsets(text, duration):
    times = parse_relative_times(text)
    assert not times.offsets
    assert [item.delta for item in times.durations] == (
        [] if duration is None else [timedelta(minutes=duration)])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text,tool,title,offset,duration", [
    ("agenda uma reunião daqui a 2 horas com o time", "create_event",
     "uma reunião com o time", 120, 60),
    ("marca uma call com o Bruno daqui a 30 minutos", "create_event",
     "call com o Bruno", 30, 60),
    ("agenda academia em 1 hora", "create_event", "academia", 60, 60),
    ("agenda um café daqui a 3 horas por meia hora", "create_event", "café", 180, 30),
    ("me lembra daqui a 45 minutos de ligar pro João", "create_reminder",
     "ligar pro João", 45, None),
    ("me lembra em 2 horas de tirar a roupa da máquina", "create_reminder",
     "tirar a roupa da máquina", 120, None),
    ("schedule a meeting in 2 hours with the team", "create_event",
     "a meeting with the team", 120, 60),
    ("remind me in 20 minutes to drink water", "create_reminder", "drink water", 20, None),
    ("me lembra em 30 minutos tomar remédio", "create_reminder", "tomar remédio", 30, None),
    ("me lembra daqui a uma hora e meia de sair", "create_reminder", "sair", 90, None),
    ("agenda café daqui a uma hora de 1 hora", "create_event", "café", 60, 60),
    ("schedule coffee in half an hour for 30 minutes", "create_event", "coffee", 30, 30),
    ("agenda café em 2h30 por 1h30", "create_event", "café", 150, 90),
])
def test_real_code_routes(orch, text, tool, title, offset, duration, stream):
    if stream:
        chunks = list(orch.process_stream(text))
        reply = "".join(chunk[0] for chunk in chunks)
        assert chunks[-1][2] == "AGENT_AWAITING_CONFIRMATION"
    else:
        reply, mode = orch._process_core(text)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
    pending = orch._pending_action
    assert pending["type"] == tool
    args = pending["args"]
    assert args["summary" if tool == "create_event" else "title"] == title
    start = NOW + timedelta(minutes=offset)
    assert args["start_iso" if tool == "create_event" else "due_iso"] == start.isoformat()
    if duration:
        assert args["end_iso"] == (start + timedelta(minutes=duration)).isoformat()
    assert "assumi" not in reply and "assumed" not in reply
    orch.engine_analytical.chat_with_tools.assert_not_called()
    orch.engine_analytical.chat.assert_not_called()


def test_front_body_without_trigger(orch):
    orch._handle_reminder_regex("em 30 minutos tomar remédio", NOW)
    assert orch._pending_action["args"] == {
        "title": "tomar remédio", "due_iso": "2026-11-16T20:48:00+00:00"}


@pytest.mark.parametrize("body,expected", [
    ("academia às 2 horas", "2026-11-16T14:00:00+00:00"),
    ("academia 2 da tarde", "2026-11-16T14:00:00+00:00"),
    ("academia amanhã às 1h", "2026-11-17T13:00:00+00:00"),
])
def test_clock_times_keep_existing_behaviour(orch, body, expected):
    orch._handle_calendar_regex(body, NOW)
    assert orch._pending_action["args"]["start_iso"] == expected


def test_duration_is_never_a_start(orch):
    assert orch._handle_calendar_regex("academia por 2 horas", NOW) is None
    orch._handle_reminder_regex("academia por 2 horas", NOW)
    assert orch._pending_action["args"] == {"title": "academia por 2 horas", "due_iso": None}
    orch._pending_action = None
    orch._handle_calendar_regex("academia amanhã às 10 por meia hora", NOW)
    assert orch._pending_action["args"] == {
        "summary": "academia", "start_iso": "2026-11-17T10:00:00+00:00",
        "end_iso": "2026-11-17T10:30:00+00:00"}


@pytest.mark.parametrize("body,start", [
    ("correr por 30 minutos amanhã às 7", "2026-11-17T07:00:00+00:00"),
    ("correr amanhã às 7 por 30 minutos", "2026-11-17T07:00:00+00:00"),
    ("correr amanhã por 30 minutos às 7", "2026-11-17T07:00:00+00:00"),
    ("correr daqui a 2 horas por 30 minutos", "2026-11-16T22:18:00+00:00"),
    ("run in 2 hours for 30 minutes", "2026-11-16T22:18:00+00:00"),
])
def test_only_events_consume_title_duration(orch, body, start):
    orch._handle_reminder_regex(body, NOW)
    expected_title = "run for 30 minutes" if body.startswith("run") else "correr por 30 minutos"
    assert orch._pending_action["args"] == {"title": expected_title, "due_iso": start}
    orch._handle_calendar_regex(body, NOW)
    assert orch._pending_action["args"] == {
        "summary": "run" if body.startswith("run") else "correr",
        "start_iso": start,
        "end_iso": (datetime.fromisoformat(start) + timedelta(minutes=30)).isoformat(),
    }


MIXED_TIME_REQUESTS = [
    "agenda reunião amanhã às 10 e me lembra daqui a 2 horas de ligar pro joão",
    "me lembra daqui a 30 minutos de ver a reunião de amanhã às 15h",
]
OTHER_TIMES = [
    "às 10", "10h", "10:30", "at 3", "3pm", "3 am", "hoje", "amanhã",
    "depois de amanhã", "today", "tomorrow", "tonight", "segunda-feira",
    "terça", "quarta feira", "quinta", "sexta", "sábado", "domingo",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "17/11", "dia 20", "2026-11-20",
]


@pytest.mark.parametrize("text", MIXED_TIME_REQUESTS + [
    f"café em 2h30 por 1h30 {other}" for other in OTHER_TIMES
])
def test_competing_times_defer_code_and_preserve_model_values(orch, monkeypatch, text):
    audit = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", audit)
    assert len(parse_relative_times(text).offsets) == 1
    for handler in (orch._handle_calendar_regex, orch._handle_reminder_regex):
        assert handler(text, NOW) is None
        assert orch._pending_action is None
    orch.remember_emails_in(text)
    for tool, args in (
        ("create_event", {"summary": "reunião", "start_iso": "2026-11-17T10:00:00+00:00",
                          "end_iso": "2026-11-17T11:00:00+00:00"}),
        ("create_reminder", {"title": "ligar pro joão", "due_iso": "2026-11-16T22:18:00+00:00"}),
    ):
        orch._execute_tool(tool, dict(args))
        assert orch._pending_action["args"] == args
    assert not any(call.args[2] == "schedule_times_normalised" for call in audit.call_args_list)


@pytest.mark.parametrize("text", MIXED_TIME_REQUESTS)
@pytest.mark.parametrize("stream", [False, True])
def test_mixed_time_requests_reach_model_with_both_tool_times_untouched(orch, text, stream):
    actions = [
        ("create_event", {"summary": "reunião", "start_iso": "2026-11-17T10:00:00+00:00",
                          "end_iso": "2026-11-17T11:00:00+00:00"}),
        ("create_reminder", {"title": "ligar pro joão", "due_iso": "2026-11-16T22:18:00+00:00"}),
    ]

    def tool_calls(*args, **kwargs):
        for tool, values in actions:
            brief = orch._execute_tool(tool, dict(values))
            assert orch._pending_action["args"] == values
        return brief, "create_reminder"

    orch.engine_analytical.chat_with_tools.side_effect = tool_calls
    if stream:
        list(orch.process_stream(text))
    else:
        orch._process_core(text)
    orch.engine_analytical.chat_with_tools.assert_called_once()


def test_competing_time_still_repairs_invalid_event_end(orch):
    orch.remember_emails_in(MIXED_TIME_REQUESTS[0])
    orch._execute_tool("create_event", {
        "summary": "reunião", "start_iso": "2026-11-17T10:00:00+00:00", "end_iso": "bad"})
    assert orch._pending_action["args"] == {
        "summary": "reunião", "start_iso": "2026-11-17T10:00:00+00:00",
        "end_iso": "2026-11-17T11:00:00+00:00"}


@pytest.mark.parametrize("now,expression,start,end", [
    (datetime(2026, 11, 16, 23, 30, tzinfo=LONDON), "em 2 horas",
     "2026-11-17T01:30:00+00:00", "2026-11-17T02:30:00+00:00"),
    (datetime(2026, 10, 24, 23, 30, tzinfo=LONDON), "em 4 horas",
     "2026-10-25T02:30:00+00:00", "2026-10-25T03:30:00+00:00"),
    (datetime(2026, 10, 25, 0, 30, tzinfo=LONDON), "em 1 hora",
     "2026-10-25T01:30:00+01:00", "2026-10-25T01:30:00+00:00"),
])
def test_midnight_and_dst_on_both_paths(orch, monkeypatch, now, expression, start, end):
    orch._handle_calendar_regex(f"café {expression}", now)
    assert orch._pending_action["args"]["start_iso"] == start
    assert orch._pending_action["args"]["end_iso"] == end
    monkeypatch.setattr("src.core.staging_mixin.local_now", lambda: now)
    orch.remember_emails_in(f"agenda café {expression}")
    orch._execute_tool("create_event", {"summary": "café", "start_iso": "wrong", "end_iso": "wrong"})
    assert orch._pending_action["args"]["start_iso"] == start
    assert orch._pending_action["args"]["end_iso"] == end
    orch._execute_tool("create_reminder", {"title": "café", "due_iso": "wrong"})
    assert orch._pending_action["args"]["due_iso"] == start


@pytest.mark.parametrize("text,old_start,old_end,end", [
    ("agenda café daqui a 2 horas", "2026-11-16T22:18:00+00:00",
     "2026-11-16T22:18:00+00:00", "2026-11-16T23:18:00+00:00"),
    ("agenda café daqui a 3 horas por meia hora", "2026-11-16T22:48:00+00:00",
     "2026-11-16T23:18:00+00:00", "2026-11-16T23:48:00+00:00"),
    ("schedule coffee in 2 hours for 30 minutes", "2026-11-16T20:18:00+00:00",
     "2026-11-16T22:18:00+00:00", "2026-11-16T22:48:00+00:00"),
    ("agenda café em 2h30 por 1h30", "2026-11-16T22:18:00+00:00",
     "2026-11-16T23:18:00+00:00", "2026-11-17T00:18:00+00:00"),
])
def test_model_overwrite_and_one_private_audit(orch, monkeypatch, text, old_start, old_end, end):
    audit = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", audit)
    orch.remember_emails_in(text)
    args = {"summary": "café", "start_iso": old_start, "end_iso": old_end, "location": "office"}
    brief = orch._execute_tool("create_event", args)
    result = orch._pending_action["args"]
    offset = parse_relative_times(text).offsets[0].delta
    assert result["start_iso"] == add_elapsed(NOW, offset).isoformat()
    assert result["end_iso"] == end
    assert result["location"] == "office"
    assert args["start_iso"] == old_start  # the tool's original dict is not mutated
    assert end in brief
    events = [call for call in audit.call_args_list if call.args[2] == "schedule_times_normalised"]
    assert len(events) == 1
    fields = sorted(k for k in ("start_iso", "end_iso") if result[k] != args[k])
    assert events[0].kwargs["payload"] == {"tool": "create_event", "fields": fields}


def test_model_reminder(orch, monkeypatch):
    audit = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", audit)
    orch.remember_emails_in("remind me in 20 minutes to drink water")
    orch._execute_tool("create_reminder", {"title": "drink water", "due_iso": None})
    assert orch._pending_action["args"]["due_iso"] == "2026-11-16T20:38:00+00:00"
    events = [call for call in audit.call_args_list if call.args[2] == "schedule_times_normalised"]
    assert len(events) == 1
    assert events[0].kwargs["payload"] == {"tool": "create_reminder", "fields": ["due_iso"]}


@pytest.mark.parametrize("text", ["agenda café amanhã às 10", "in 2 hours or in 3 hours"])
@pytest.mark.parametrize("tool", ["create_event", "create_reminder"])
def test_no_single_offset_leaves_model_values(orch, monkeypatch, text, tool):
    audit = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", audit)
    orch.remember_emails_in(text)
    args = ({"summary": "café", "start_iso": "2026-11-17T10:00:00+00:00",
             "end_iso": "2026-11-17T11:00:00+00:00"} if tool == "create_event"
            else {"title": "café", "due_iso": "2026-11-17T10:00:00+00:00"})
    orch._execute_tool(tool, args)
    assert orch._pending_action["args"] == args
    assert not any(call.args[2] == "schedule_times_normalised" for call in audit.call_args_list)


@pytest.mark.parametrize("end", [None, "2026-11-16T22:18:00+00:00",
                                      "2026-11-16T21:18:00+00:00", "2026-11-16T24:00:00", "bad"])
def test_end_guard_before_staging_and_preview(orch, end):
    args = {"summary": "café", "start_iso": "2026-11-16T22:18:00+00:00"}
    if end is not None:
        args["end_iso"] = end
    expected = "2026-11-16T23:18:00+00:00"
    assert ensure_event_end(args)["end_iso"] == expected
    assert expected in build_preview("create_event", args)["llm"]
    brief = orch._execute_tool("create_event", args)
    assert orch._pending_action["args"]["end_iso"] == expected
    assert expected in brief


def test_invalid_start_hour_never_previews_or_executes(orch):
    args = {"summary": "café", "start_iso": "2026-11-16T24:00:00", "end_iso": None}
    assert "24:00" not in build_preview("create_event", args)["human"]
    assert "SYSTEM ERROR" in orch._execute_tool("create_event", args)
    assert orch._pending_action is None
    assert orch._run_action("create_event", args)[1] == "AGENT_ACTION_ERROR"
    assert "SYSTEM ERROR" in orch._execute_tool_unstaged_call("create_event", args)


PT_LABELS = {
    "send_email": "o envio do email", "create_event": "a criação do evento",
    "create_reminder": "a criação do lembrete", "mac_quit_app": "o fechamento do X",
    "ps5_standby": "o repouso do PS5", "mac_sleep": "o sleep do Mac",
    "read_file_at": "a leitura de x.py", "propose_code_change": "a proposta de mudança em x.py",
    "apply_code_change": "a mudança em x.py", "pdf_merge": "a junção dos PDFs",
    "pdf_split": "a separação das páginas", "pdf_images": "a extração das imagens",
    "pdf_png": "a gravação das imagens", "run_code_tests": "os testes de X",
    "code_root_add": "a autorização de /X/x.py", "read_files": "a leitura de x.py",
    "read_code_project": "a leitura de X", "write_answers_pdf": "a cópia com as respostas",
    "unknown": "a ação",
}


@pytest.mark.parametrize("tool,label", PT_LABELS.items())
def test_every_portuguese_cancel_label(orch, tool, label):
    assert set(PT_LABELS) == set(_PREVIEW_BUILDERS) | {"unknown"}
    args = {"app_name": "X", "path": "/X/x.py", "root": "/X", "paths": ["/X/x.py"],
            "profile": {"cwd": "/X"}, "input": {"path": "/X/x.pdf"}}
    assert build_preview(tool, args)["cancel_label"] == label
    orch._pending_action = {"type": tool, "args": args}
    reply, mode = orch._cancel_pending_action()
    assert reply == f"Cancelado. Não vou prosseguir com {label}."
    assert mode == "AGENT_ACTION_CANCELLED"
    orch._reply_lang = "en"
    english_label = build_preview(tool, args, lang="en")["cancel_label"]
    orch._pending_action = {"type": tool, "args": args}
    assert orch._cancel_pending_action()[0] == f"Cancelled. I won't go ahead with {english_label}."


@pytest.mark.parametrize("stream", [False, True])
def test_model_staging_uses_this_turn_text(orch, stream):
    # This wording reaches the tool path rather than a regex scheduling route.
    text = "preciso de um evento in 2 hours for 30 minutes"
    orch.remember_emails_in("in 6 hours")  # a previous turn cannot win

    def tool_call(*args, **kwargs):
        brief = orch._execute_tool("create_event", {
            "summary": "meeting", "start_iso": NOW.isoformat(), "end_iso": NOW.isoformat()})
        return brief, "create_event"

    orch.engine_analytical.chat_with_tools.side_effect = tool_call
    if stream:
        list(orch.process_stream(text))
    else:
        orch._process_core(text)
    orch.engine_analytical.chat_with_tools.assert_called_once()
    assert orch._pending_action["args"]["start_iso"] == "2026-11-16T22:18:00+00:00"
    assert orch._pending_action["args"]["end_iso"] == "2026-11-16T22:48:00+00:00"


def test_quoted_attachment_offset_cannot_supply_time(orch):
    from src.core.orchestrator import CortexOrchestrator
    wrapped = ("<<<UNTRUSTED_CONTENT>>>in 2 hours<<<END_UNTRUSTED_CONTENT>>>"
               "\nPERGUNTA: describe this")
    orch.remember_emails_in(CortexOrchestrator._pedros_words(wrapped))
    args = {"title": "meeting", "due_iso": "2026-11-17T10:00:00+00:00"}
    orch._execute_tool("create_reminder", args)
    assert orch._pending_action["args"] == args


def test_end_guard_uses_elapsed_hour_in_repeated_dst_hour():
    args = {"start_iso": "2026-10-25T01:30:00+01:00", "end_iso": None}
    assert ensure_event_end(args)["end_iso"] == "2026-10-25T01:30:00+00:00"


@pytest.mark.parametrize("end", [123, [], {}])
def test_unparseable_end_type_is_repaired(end):
    assert ensure_event_end({"start_iso": NOW.isoformat(), "end_iso": end})["end_iso"] == (
        "2026-11-16T21:18:00+00:00")


@pytest.mark.parametrize("text", ["me lembra em casa", "me lembra daquele dia",
                                  "me lembra de quando eu era criança"])
def test_reminder_trigger_does_not_expand_to_non_offsets(text):
    from src.core.dispatch.reminders import REMINDER_TRIGGER_RE
    assert REMINDER_TRIGGER_RE.match(text) is None


@pytest.mark.parametrize("body,minutes", [("em 2h30 de sair", 150),
                                         ("dentro de dez minutos de sair", 10)])
def test_new_reminder_prefix_accepts_compounds_and_words(orch, body, minutes):
    orch._process_core("me lembra " + body)
    assert orch._pending_action["args"] == {
        "title": "sair", "due_iso": (NOW + timedelta(minutes=minutes)).isoformat()}
