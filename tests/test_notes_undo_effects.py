"""Notes record intent; undo retries only after known failure or cancellation."""
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core import action_ledger, effect_store
from src.core.staging_mixin import Delivery
from src.integrations import applescript
from src.integrations.notes_client import NotesUnavailable
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch
from tests.test_effect_reconcile import boot, crash


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = effect_store.store  # conftest gives this a real per-test database
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.set_turn_origin("ui", "conversation")
    o.notes = SimpleNamespace(create_note=Mock(), append_to_note=Mock(return_value={"title": "Work"}))
    return o, store


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("case", ["done", "timeout", "unexpected", "known", "miss", "unavailable", "stop"])
def test_notes(context, tmp_path, monkeypatch, stream, lang, append, case):
    if case == "miss" and not append:
        pytest.skip("only append can miss")
    o, store = context
    action = "append_to_note" if append else "create_note"
    args = ({"title": "Work", "text": "review"} if append else
            {"title": "Work", "body": "review", "folder": "Office"})
    command = (("acrescenta na nota Work: review" if lang == "pt" else
                "append to note Work: review") if append else
               ("anota na pasta Office: Work\nreview" if lang == "pt" else
                "create a note in folder Office: Work\nreview"))
    client = getattr(o.notes, action)
    seen = []

    def call(**kwargs):
        seen.append(kwargs)
        [effect] = store.in_states(["executando"])
        assert effect.action == action and effect.args == args
        assert len(effect.id) == 32
        assert effect.origin == {"channel": "ui", "conversation": "conversation"}
        if case == "timeout":
            applescript.run("fake", app_label="Notes", unavailable_exc=NotesUnavailable)
        if case == "unexpected":
            raise RuntimeError("lost")
        if case in {"known", "miss"}:
            raise NotesUnavailable("No note found matching Work" if case == "miss" else
                                   "rejected the request")
        return {"title": "Work"}

    client.side_effect = call
    monkeypatch.setattr(applescript.subprocess, "run",
                        Mock(side_effect=subprocess.TimeoutExpired("fake", 30)))
    if case == "unavailable":
        blocker = tmp_path / "blocked"
        blocker.write_text("x")
        store.db_path = str(blocker / "effects.db")
    elif case == "stop":
        delivery = Delivery()
        o.set_turn_delivery(delivery)
        create = store.create

        def stop(*args, **kwargs):
            effect = create(*args, **kwargs)
            delivery.cancelled.set()
            return effect
        monkeypatch.setattr(store, "create", stop)
    reply, mode = _turn(o, command, stream, lang)
    if case in {"unavailable", "stop"}:
        client.assert_not_called()
        assert mode == ("AGENT_ACTION_CANCELLED" if case == "stop" else "NOTE_ERROR")
        assert ("Parei antes" if lang == "pt" else "Stopped before") in reply if case == "stop" else (
            "Não consegui registrar" if lang == "pt" else "I couldn't record") in reply
    else:
        client.assert_called_once()
        assert seen == [args]
        assert mode == ("NOTE_APPEND" if append else "NOTE_CREATE") if case in {"done", "miss"} else mode == "NOTE_ERROR"
    if case == "unavailable":
        assert action_ledger.ledger.load_entries() == []
        return
    state = {"done": "feito", "timeout": "desconhecido", "unexpected": "desconhecido",
             "known": "falhou", "miss": "falhou", "stop": "cancelado"}[case]
    [effect] = store.in_states([state])
    assert [t.to_state for t in store.history(effect.id)] == (
        ["preparando", "cancelado"] if case == "stop" else ["preparando", "executando", state])
    entries = action_ledger.ledger.load_entries()
    assert [e["action"] for e in entries] == ([action] if case == "done" else [])
    if case in {"timeout", "unexpected"}:
        assert reply == (f"Não sei se deu certo ({effect.detail}). Confere antes de pedir de novo — não vou repetir sozinho."
                         if lang == "pt" else f"I don't know whether it went through ({effect.detail}). Please check before asking again — I won't retry on my own.")
    elif case == "known":
        assert reply == (f"[SYSTEM ERROR] Não consegui {'editar' if append else 'criar'} a nota: rejected the request"
                         if lang == "pt" else f"[SYSTEM ERROR] Couldn't {'edit' if append else 'create'} the note: rejected the request")
    elif case == "miss":
        assert reply == ('Não achei nenhuma nota chamada "Work".' if lang == "pt" else 'I found no note called "Work".')
    elif case == "done":
        assert entries[0]["args"] == {"title": "Work"}
        assert reply == (('Adicionado à nota "Work".' if lang == "pt" else 'Added to note "Work".') if append else
                         ('Anotado: "Work" (pasta Office).' if lang == "pt" else 'Noted: "Work" (folder Office).'))


def test_empty_notes_do_not_record(context):
    o, store = context
    assert o._handle_note_create(" ") is None
    assert o._handle_note_append("", "text") is None
    assert o._handle_note_append("Work", " ") is None
    assert store.in_states(["preparando", "executando", "feito", "falhou"]) == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["sequential", "concurrent", "crash", "unavailable", "stop"])
def test_undo_once(context, tmp_path, monkeypatch, stream, lang, case):
    o, store = context
    ledger = action_ledger.ledger
    ledger.record("create_note", {"title": "Work"}, source="dispatch")
    [target] = ledger.load_entries()
    eid = f"undo:{target['ts']}:{target['action']}"
    command = "desfaz" if lang == "pt" else "undo"

    def reverse(spec):
        effect = store.get(eid)
        assert effect.state == "executando" and effect.args == spec == target["reverse"]
        return "removed", "AGENT_ACTION_DONE"
    o._run_reverse = Mock(side_effect=reverse)
    if case in {"sequential", "concurrent", "crash"}:
        # Lost ledger writes leave the same target visible to the next undo.
        monkeypatch.setattr(ledger, "record", Mock(side_effect=RuntimeError("crash")) if case == "crash" else Mock())
    if case == "concurrent":
        barrier = threading.Barrier(2)
        create = store.create

        def together(*args, **kwargs):
            barrier.wait(timeout=10)
            return create(*args, **kwargs)
        monkeypatch.setattr(store, "create", together)
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda _: _turn(o, command, stream, lang), range(2)))
    elif case == "unavailable":
        blocker = tmp_path / "blocked"
        blocker.write_text("x")
        store.db_path = str(blocker / "effects.db")
        replies = [_turn(o, command, stream, lang)]
    elif case == "stop":
        delivery = Delivery()
        o.set_turn_delivery(delivery)
        create = store.create

        def stop(*args, **kwargs):
            effect = create(*args, **kwargs)
            delivery.cancelled.set()
            return effect
        monkeypatch.setattr(store, "create", stop)
        replies = [_turn(o, command, stream, lang)]
    else:
        replies = [_turn(o, command, stream, lang), _turn(o, command, stream, lang)]
    if case in {"unavailable", "stop"}:
        o._run_reverse.assert_not_called()
        assert replies[0][1] == ("AGENT_ACTION_CANCELLED" if case == "stop" else "AGENT_ACTION_ERROR")
        return
    o._run_reverse.assert_called_once_with(target["reverse"])
    if case == "crash":
        duplicates = [("Não sei se desfiz isso — confere antes de pedir de novo; não vou repetir sozinho."
                       if lang == "pt" else "I don't know whether I undid that — please check before asking again; I won't retry on my own.",
                       "UNDO_NONE")]
    else:
        duplicates = [("Isso já foi desfeito." if lang == "pt" else "That was already undone.", "UNDO_NONE")]
        if case == "concurrent":
            duplicates.append(("Isso está sendo desfeito agora." if lang == "pt" else
                               "That is being undone right now.", "UNDO_NONE"))
    assert any(reply in duplicates for reply in replies)
    assert store.get(eid).state == ("desconhecido" if case == "crash" else "feito")
    assert any(mode == ("AGENT_ACTION_ERROR" if case == "crash" else "UNDO_DONE") for _, mode in replies)


def _seed_undo_attempt(store, eid, reverse, state):
    assert store.create("undo", reverse, effect_id=eid) is not None
    if state == "preparando":
        return
    if state in {"esperando", "cancelado"}:
        assert store.transition(eid, "preparando", state)
    else:
        assert store.transition(eid, "preparando", "executando")
        if state != "executando":
            assert store.transition(eid, "executando", state)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("state", ["falhou", "cancelado"])
@pytest.mark.parametrize("failures", [1, 2])
def test_undo_retries_known_failures(context, monkeypatch, stream, lang, state, failures):
    from src.integrations import calendar_client

    o, store = context
    ledger = action_ledger.ledger
    ledger.record("create_event", {"summary": "Work"}, result_id="event-1")
    [target] = ledger.load_entries()
    base = f"undo:{target['ts']}:{target['action']}"
    delete = Mock(side_effect=([calendar_client.CalendarUnavailable("unreachable")] * failures + [None])
                  if state == "falhou" else None)
    monkeypatch.setattr(calendar_client, "delete_event", delete)
    delivery = Delivery()
    o.set_turn_delivery(delivery)
    create = store.create
    ids = [base] + [f"{base}:{i}" for i in range(2, failures + 2)]

    def stop(*args, **kwargs):
        effect = create(*args, **kwargs)
        if effect.id in ids[:-1]:
            delivery.cancelled.set()
        return effect

    if state == "cancelado":
        monkeypatch.setattr(store, "create", stop)
    for eid in ids[:-1]:
        delivery.cancelled.clear()
        _, mode = _turn(o, "desfaz" if lang == "pt" else "undo", stream, lang)
        assert mode == ("AGENT_ACTION_ERROR" if state == "falhou" else "AGENT_ACTION_CANCELLED")
        assert store.get(eid).state == state
        assert ledger.load_entries() == [target]
    delivery.cancelled.clear()
    _, mode = _turn(o, "desfaz" if lang == "pt" else "undo", stream, lang)
    assert mode == "UNDO_DONE"
    assert store.get(ids[-1]).state == "feito"
    assert [e["action"] for e in ledger.load_entries()] == ["create_event", "undo"]
    assert delete.call_count == (failures + 1 if state == "falhou" else 1)
    assert delete.call_args_list == [(("event-1",), {})] * delete.call_count


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("race", [False, True])
@pytest.mark.parametrize("states,pt,en", [
    (["feito"], "Isso já foi desfeito.", "That was already undone."),
    *[([state], "Isso está sendo desfeito agora.", "That is being undone right now.")
      for state in ("preparando", "esperando", "executando")],
    (["desconhecido"],
     "Não sei se desfiz isso — confere antes de pedir de novo; não vou repetir sozinho.",
     "I don't know whether I undid that — please check before asking again; I won't retry on my own."),
    (["desconhecido", "executando"], "Isso está sendo desfeito agora.", "That is being undone right now."),
    (["desconhecido", "executando", "feito"], "Isso já foi desfeito.", "That was already undone."),
])
def test_undo_attempt_state_replies(context, monkeypatch, stream, lang, race, states, pt, en):
    o, store = context
    ledger = action_ledger.ledger
    ledger.record("create_note", {"title": "Work"})
    [target] = ledger.load_entries()
    base = f"undo:{target['ts']}:{target['action']}"
    _seed_undo_attempt(store, base, target["reverse"], "falhou")
    o._run_reverse = Mock()

    def seed():
        for i, state in enumerate([*states, "cancelado"], 2):
            _seed_undo_attempt(store, f"{base}:{i}", target["reverse"], state)

    if race:
        create = store.create

        def competing_claim(*args, **kwargs):
            # A competing caller claims after the precheck, before run_effect.
            assert kwargs == {} and args[3] == f"{base}:2"
            monkeypatch.setattr(store, "create", create)
            seed()
            return create(*args, **kwargs)

        monkeypatch.setattr(store, "create", competing_claim)
    else:
        seed()
    assert _turn(o, "desfaz" if lang == "pt" else "undo", stream, lang) == (
        pt if lang == "pt" else en, "UNDO_NONE")
    o._run_reverse.assert_not_called()
    assert store.get(f"{base}:{len(states) + 3}") is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_concurrent_undo_retry_runs_once(context, monkeypatch, stream, lang):
    o, store = context
    ledger = action_ledger.ledger
    ledger.record("create_note", {"title": "Work"})
    [target] = ledger.load_entries()
    base = f"undo:{target['ts']}:{target['action']}"
    _seed_undo_attempt(store, base, target["reverse"], "falhou")
    monkeypatch.setattr(ledger, "record", Mock())
    barrier, entered, release = threading.Barrier(2), threading.Event(), threading.Event()
    create = store.create

    def together(*args, **kwargs):
        assert args[3] == f"{base}:2"
        barrier.wait(timeout=10)
        return create(*args, **kwargs)

    def reverse(spec):
        assert spec == target["reverse"]
        entered.set()
        assert release.wait(timeout=10)
        return "removed", "AGENT_ACTION_DONE"

    monkeypatch.setattr(store, "create", together)
    o._run_reverse = Mock(side_effect=reverse)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_turn, o, "desfaz" if lang == "pt" else "undo", stream, lang)
                   for _ in range(2)]
        try:
            assert entered.wait(timeout=10)
            from concurrent.futures import FIRST_COMPLETED, wait
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            assert len(done) == 1
            assert done.pop().result() == (
                "Isso está sendo desfeito agora." if lang == "pt" else "That is being undone right now.",
                "UNDO_NONE")
        finally:
            release.set()
        replies = [f.result(timeout=10) for f in futures]
    o._run_reverse.assert_called_once_with(target["reverse"])
    assert any(mode == "UNDO_DONE" for _, mode in replies)
    assert store.get(f"{base}:2").state == "feito"
    assert store.get(f"{base}:3") is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_restart_labels(tmp_path, monkeypatch, stream, lang):
    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "restart.db"
    crash(db, [("append_to_note", {"title": "Work", "text": "review"}),
               ("undo", {"tool": "delete_note", "args": {"title": "Work"}})])
    store = effect_store.EffectStore(str(db))
    try:
        o = boot(tmp_path, monkeypatch, store)
        reply, _ = _turn(o, "me explica RAG em detalhes", stream, lang)
        assert ("acrescentar a uma nota; desfazer uma ação" if lang == "pt" else
                "add to a note; undo an action") in reply
    finally:
        if store._conn is not None:
            store._conn.close()
