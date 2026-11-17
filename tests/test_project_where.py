"""Project resumption is literal, durable and shared with the focus cycle."""
import json
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from src.core import focus, focus_mixin, projects, projects_mixin
from src.core.audit import audit
from src.core.memory import speaker_unverified, unverified_speaker
from tests.test_dispatch_parity import _build_real_orch
from tests.test_project_routes import store, turn
from tests.test_focus_routes import Clock


@pytest.fixture
def setup(tmp_path, monkeypatch, store):
    clock = Clock()
    clock.now = datetime(2026, 10, 19, 15, 15, tzinfo=ZoneInfo("Europe/London"))
    monkeypatch.setattr(focus_mixin, "local_now", clock)
    monkeypatch.setattr(projects_mixin, "local_now", clock, raising=False)
    monkeypatch.setattr(projects, "_now", lambda: clock().astimezone(timezone.utc).isoformat())
    fs = focus.FocusStore(str(tmp_path / "focus.db"))
    monkeypatch.setattr(focus, "store", fs)

    def make():
        o = _build_real_orch(tmp_path)
        o.background = False
        o._LAST_ANNOUNCE_PATH = str(tmp_path / "announce.txt")
        o.studies_review.list_pending = Mock(return_value=[{"topic": "Estudo pendente"}])
        for engine in (o.engine_analytical, o.engine_creative):
            for method in ("chat", "chat_with_tools", "chat_stream"):
                getattr(engine, method).side_effect = AssertionError("LLM called")
        o.set_turn_origin("ui", "A")
        return o
    yield make, clock
    if fs._conn:
        fs._conn.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("en", [False, True])
def test_ui_notes_restart_voice(setup, store, monkeypatch, stream, en):
    make, clock = setup
    o = make()
    turn(o, "projeto: Currículo")
    turn(o, "objetivo: conseguir a vaga")
    today = clock.now
    clock.now = today - timedelta(days=10)
    turn(o, "feito: antigo")
    clock.now = today - timedelta(days=3)
    turn(o, "decidi: usar uma página")
    clock.now = today - timedelta(days=1)
    turn(o, "feito: rascunho")
    clock.now = today
    turn(o, "falta: revisar a seção 2")
    turn(o, "próximo: mandar para a vaga X")
    store._conn.close()
    store._conn = None
    reopened = projects.ProjectStore(store.db_path)
    monkeypatch.setattr(projects, "store", reopened)
    o = make()  # No in-memory offer or conversation survives the restart.
    seen = []
    original = o._handle_projects

    def handle(own):
        seen.append((threading.get_ident(), speaker_unverified()))
        return original(own)
    monkeypatch.setattr(o, "_handle_projects", handle)
    o.set_turn_origin("voice")
    o._pending_query = "placar do jogo"
    question = "where was I?" if en else "onde a gente parou?"
    expected = (
        'Project “Currículo” — goal: conseguir a vaga.\n'
        'Last done: rascunho (yesterday at 15:15).\n'
        'To do: revisar a seção 2.\n'
        'Last decision: usar uma página (on Friday at 15:15).\n'
        'Next step: mandar para a vaga X (today at 15:15).\n'
        'Shall we do 25 min on it? (yes)' if en else
        'Projeto “Currículo” — objetivo: conseguir a vaga.\n'
        'Último feito: rascunho (ontem às 15:15).\n'
        'Falta: revisar a seção 2.\n'
        'Última decisão: usar uma página (na sexta às 15:15).\n'
        'Próximo passo: mandar para a vaga X (hoje às 15:15).\n'
        'Bora 25 min nisso? (sim)')
    try:
        with unverified_speaker():  # Same room-speaker mark as /ask_stream.
            assert turn(o, question, stream, "voice", None) == (expected, "PROJECT_WHERE")
        assert seen == [(seen[0][0], True)]
        assert (seen[0][0] != threading.get_ident()) == stream
        assert o._pending_query is None
        assert o._offer == {"kind": "focus_more", "task": "Currículo",
                            "first_step": "mandar para a vaga X", "lang": "en" if en else "pt"}
        assert turn(o, "yes" if en else "sim", stream, "voice", None)[1] == "FOCUS_STARTED"
        session = focus.store.current()
        assert (session.task, session.first_step, session.ran_minutes(session.ends_at)) == (
            "Currículo", "mandar para a vaga X", 25)
    finally:
        if reopened._conn:
            reopened._conn.close()


@pytest.mark.parametrize("en", [False, True])
def test_linked_empty_and_no_next(setup, store, en):
    make, _ = setup
    o = make()
    house, cv = store.create_or_get("Casa"), store.create_or_get("Currículo")
    store.link_conversation("A", house.id)
    store.set_active(cv.id)
    question = "where was I?" if en else "onde paramos?"
    expected = ('Project “Casa”: nothing noted yet. You can note with “done:”, “todo:” or “next:”.'
                if en else 'Projeto “Casa”: nada anotado ainda. Dá pra anotar com “feito:”, '
                '“falta:” ou “próximo:”.')
    assert turn(o, question, True) == (expected, "PROJECT_WHERE")
    assert o._offer is None
    turn(o, "feito: parede")
    assert turn(o, question)[0] == ('Project “Casa”.\nLast done: parede (today at 15:15).'
        if en else 'Projeto “Casa”.\nÚltimo feito: parede (hoje às 15:15).')
    assert o._offer is None
    store.set_objective(house.id, "pintar")
    assert "pintar" in turn(o, question)[0] and o._offer is None
    events = [json.loads(line) for line in open(audit.path)]
    mine = [e for e in events if e.get("stage") == "project" and e["event"] == "where"]
    assert [e["payload"] for e in mine] == [
        {"state": "empty"}, {"state": "no_next_step"}, {"state": "no_next_step"}]
    assert "Casa" not in json.dumps(mine) and "parede" not in json.dumps(mine)


@pytest.mark.parametrize("en", [False, True])
def test_voice_dates_todo_cap_and_open_focus(setup, store, en):
    make, clock = setup
    o = make()
    p = store.create_or_get("Casa")
    store.set_active(p.id)
    today = clock.now
    clock.now -= timedelta(days=10)
    store.add_entry(p.id, "feito", "antigo", "voice")
    clock.now = today
    for n in range(55):  # Count must not inherit entries()'s default limit of 50.
        store.add_entry(p.id, "falta", str(n))
    closed = store.add_entry(p.id, "falta", "fechado")
    store.close_entry(closed.id)
    turn(o, "próximo: abrir", channel="voice", conversation=None)
    question = "where was I?" if en else "onde parei?"
    session = focus.store.start("outra tarefa", "primeiro", 25, today)
    reply, mode = turn(o, question, True)
    assert mode == "PROJECT_WHERE"
    assert ('Last done: antigo (on 09/10, by voice).' if en else
            'Último feito: antigo (em 09/10, pela voz).') in reply
    assert ('To do: 54; 53; 52 (+52).' if en else 'Falta: 54; 53; 52 (+52).') in reply
    assert ('Next step: abrir (today at 15:15, by voice).' if en else
            'Próximo passo: abrir (hoje às 15:15, pela voz).') in reply
    line = 'Focus open: outra tarefa until 15:40.' if en else 'Foco aberto: outra tarefa até 15:40.'
    assert line in reply and "fechado" not in reply and session.is_open
    clock.now += timedelta(minutes=25)
    assert line not in turn(o, question)[0]
    focus.store.stop("depois", clock.now)
    assert line not in turn(o, question)[0]
    events = [json.loads(line) for line in open(audit.path)]
    assert [e["payload"] for e in events if e.get("stage") == "project" and
            e["event"] == "where"] == [{"state": "next_step"}] * 3


@pytest.mark.parametrize("state", ["none", "running", "expired", "next", "closed", "reminder_failed"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("en", [False, True])
def test_no_project_is_identical_to_focus(setup, state, stream, en):
    make, clock = setup
    o, reference = make(), make()
    if state != "none":
        session = focus.store.start("foco antigo", "abrir", 25, clock.now)
        if state == "expired":
            clock.now += timedelta(minutes=30)
        elif state in ("next", "closed"):
            focus.store.stop("revisar" if state == "next" else "", clock.now)
        elif state == "reminder_failed":
            focus.store.set_reminders(session.id, focus.REMINDER_FAILED, None)
    reference._reply_lang = "en" if en else "pt"
    expected = reference._focus_where(clock.now)
    assert turn(o, "where was I?" if en else "onde parei?", stream) == expected
    assert o._offer == getattr(reference, "_offer", None)


@pytest.mark.parametrize("failure", ["available", "resolve", "entries", "sql"])
def test_broken_project_falls_back(setup, store, monkeypatch, failure):
    make, clock = setup
    o = make()
    p = store.create_or_get("Casa")
    store.set_active(p.id)
    focus.store.start("antigo", "abrir", 25, clock.now)
    o._reply_lang = "pt"
    expected = o._focus_where(clock.now)
    if failure == "sql":
        with store._conn:
            store._conn.execute("DROP TABLE entries")
    else:
        method = {"available": "available", "resolve": "progress_project",
                  "entries": "where_entries"}[failure]
        monkeypatch.setattr(store, method, Mock(side_effect=OSError("disk")), raising=False)
    assert turn(o, "onde a gente parou?", True) == expected
