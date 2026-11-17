"""Plan approval records each effect before dispatch, including lost replies."""
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src.core import effect_store, planner
from src.core.planner import summarise
from src.core.staging_mixin import Delivery
from src.integrations import calendar_client, gmail_client, reminders_client
from tests.test_dispatch_parity import _build_real_orch
from tests.test_effect_reconcile import boot
from tests.test_pending_origin import _turn


def plan(first="create_reminder", body="literal"):
    return planner.Plan("lembrar e enviar", [
        planner.Step("p1", first, {"title": "Comprar pão", "to": "ana@example.com"}),
        planner.Step("p2", "send_email", {"to": "ana@example.com", "body": body}, ["p1"])])


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CORTEX_RUNNER_FIELDS", "1")
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    monkeypatch.setattr(o, "_recipient_known", lambda *_: True)
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "tudo enviado")
    calls = {}
    for module, action in [(gmail_client, "send_email"),
                           (calendar_client, "create_event"),
                           (reminders_client, "create_reminder")]:
        calls[action] = Mock(return_value={"id": "sent", "to": "ana@example.com",
                                          "title": "Comprar pão", "summary": "Reunião"})
        monkeypatch.setattr(module, action, calls[action])

    def stage(p=None, lang="pt", stream=False):
        monkeypatch.setattr(planner, "decompose", lambda *a, **kw: p or plan())
        assert _turn(o, "planeja: lembrar e enviar", lang=lang, stream=stream)[1] == (
            "AGENT_AWAITING_CONFIRMATION")
        return o._pending_action["id"]
    return o, store, calls, stage


@pytest.mark.parametrize("stream", [False, True])
def test_identified_effects_and_references(rig, stream):
    o, store, calls, stage = rig
    pid = stage(plan(body="{p1.title}: {p1.text}"), stream=stream)

    def remind(**kw):
        assert store.get(f"{pid}:p1").state == "executando"
        return {"id": "reminder", "title": kw["title"]}
    calls["create_reminder"].side_effect = remind
    reply, mode = _turn(o, "sim", stream=stream)
    assert mode == "AGENT_AWAITING_CONFIRMATION" and "Comprar pão: Lembrete" in reply
    next_id = o._pending_action["id"]
    assert next_id != pid
    assert [s["id"] for s in o._pending_action["args"]["passos"]] == ["p2"]

    def send(**kw):
        effect = store.get(f"{next_id}:p2")
        assert effect.state == "executando" and all(kw[k] == v for k, v in effect.args.items())
        assert kw["body"] == "Comprar pão: Lembrete 'Comprar pão' criado."
        return {"id": "sent", "to": kw["to"]}
    calls["send_email"].side_effect = send
    assert _turn(o, "sim", stream=stream)[1] == "AGENT_ACTION_DONE"
    for eid in [f"{pid}:p1", f"{next_id}:p2"]:
        assert [h.to_state for h in store.history(eid)] == ["preparando", "executando", "feito"]
    calls["create_reminder"].assert_called_once()
    calls["send_email"].assert_called_once()


@pytest.mark.parametrize("stream", [False, True])
def test_two_effects_share_approved_plan_identity(rig, stream):
    o, store, calls, stage = rig
    pid = stage(stream=stream)
    for sid, action in [("p1", "create_reminder"), ("p2", "send_email")]:
        result = calls[action].return_value
        def call(sid=sid, result=result, **kw):
            assert store.get(f"{pid}:{sid}").state == "executando"
            return result
        calls[action].side_effect = call
    assert _turn(o, "sim", stream=stream)[1] == "AGENT_ACTION_DONE"
    for sid in ["p1", "p2"]:
        assert [h.to_state for h in store.history(f"{pid}:{sid}")] == [
            "preparando", "executando", "feito"]
    _turn(o, "sim", stream=stream)
    assert calls["create_reminder"].call_count == calls["send_email"].call_count == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("first", ["send_email", "create_event", "create_reminder"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_failure_stops_and_uncertainty_has_fixed_last_line(rig, stream, lang, first, uncertain):
    o, store, calls, stage = rig
    pid = stage(plan(first), lang, stream)
    exc = {"send_email": gmail_client.GmailUnavailable,
           "create_event": calendar_client.CalendarUnavailable,
           "create_reminder": reminders_client.RemindersUnavailable}[first]("denied")
    if uncertain:
        exc = TimeoutError("lost") if first == "send_email" else exc
        exc.uncertain = True
    calls[first].side_effect = exc
    reply, mode = _turn(o, "sim", lang=lang, stream=stream)
    assert mode == "AGENT_ACTION_ERROR"
    assert store.get(f"{pid}:p1").state == ("desconhecido" if uncertain else "falhou")
    assert store.get(f"{pid}:p2") is None
    assert sum(c.call_count for c in calls.values()) == 1
    if uncertain:
        label = (o._UNDO_ACTION_LABELS_EN if lang == "en" else o._UNDO_ACTION_LABELS)[first]
        line = (f"⚠️ Não sei se o passo p1 ({label}) foi feito — confere antes de pedir de novo; "
                "não vou repetir sozinho." if lang == "pt" else
                f"⚠️ I don't know whether step p1 ({label}) went through — please check before "
                "asking again; I won't retry on my own.")
        assert reply == "tudo enviado\n\n" + line


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["unavailable", "cancelled", "duplicate"])
def test_no_call_without_record_or_after_stop(rig, monkeypatch, tmp_path, stream, case):
    o, store, calls, stage = rig
    pid = stage(stream=stream)
    if case == "unavailable":
        blocker = tmp_path / "blocked"
        blocker.write_text("blocked")
        store.db_path = str(blocker / "effects.db")
    elif case == "duplicate":
        store.create("create_reminder", {}, effect_id=f"{pid}:p1")
    else:
        delivery = Delivery()
        o.set_turn_delivery(delivery)
        create = store.create
        def cancel(*a, **kw):
            effect = create(*a, **kw)
            delivery.cancelled.set()
            return effect
        monkeypatch.setattr(store, "create", cancel)
    assert _turn(o, "sim", stream=stream)[1] in {
        "AGENT_ACTION_ERROR", "AGENT_ACTION_CANCELLED"}
    assert all(c.call_count == 0 for c in calls.values())
    assert store.get(f"{pid}:p2") is None
    if case == "cancelled":
        assert [h.to_state for h in store.history(f"{pid}:p1")] == ["preparando", "cancelado"]


def test_crash_in_second_effect_reconciles_without_retry(rig, tmp_path, monkeypatch):
    o, store, calls, stage = rig
    pid = stage()
    code = """
import json, os, sys
from pathlib import Path
from src.core import effect_store
from src.integrations import gmail_client, reminders_client
from tests.test_dispatch_parity import _build_real_orch
o = _build_real_orch(Path(sys.argv[1]))
o.background, o._offer, o._reply_lang = False, None, 'pt'
o._PENDING_ACTION_FILE = str(Path(sys.argv[1]) / 'pending_action.json')
o._pending_action = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
o._recipient_known = lambda *_: True
effect_store.store = effect_store.EffectStore(sys.argv[2])
reminders_client.create_reminder = lambda **kw: {'id': 'r', 'title': kw['title']}
gmail_client.send_email = lambda **kw: os._exit(23)
o._execute_pending_action()
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), store.db_path],
                           env=dict(os.environ), capture_output=True, text=True, timeout=20)
    assert child.returncode == 23, child.stderr
    assert store.get(f"{pid}:p1").state == "feito"
    assert store.get(f"{pid}:p2").state == "executando"
    fresh = boot(tmp_path, monkeypatch, effect_store.EffectStore(store.db_path))
    assert store.get(f"{pid}:p2").state == "desconhecido"
    reply, _ = _turn(fresh, "me explica RAG em detalhes")
    assert "⚠️" in reply and "ana@example.com" in reply
    assert "⚠️" not in _turn(fresh, "me explica RAG em detalhes")[0]
    _turn(fresh, "sim")
    assert all(c.call_count == 0 for c in calls.values())


@pytest.mark.parametrize("with_step_executor", [False, True])
def test_planner_executor_contract(with_step_executor):
    p = plan(body="{p1}")
    old = Mock(return_value="ok")
    step = Mock(return_value={"text": "feito", "fields": {"title": "pão"}})
    result = planner.execute(p, old, **({"step_executor": step} if with_step_executor else {}))
    assert result.success
    if with_step_executor:
        old.assert_not_called()
        assert step.call_args_list[0].args == (p.passos[0], p.passos[0].args)
        assert step.call_args_list[1].args[1]["body"] == "feito"
    else:
        assert old.call_args_list[1].args == ("send_email", {"to": "ana@example.com", "body": "ok"})


@pytest.mark.parametrize("stop", [False, True])
def test_legacy_id_and_uncertainty_with_summary_fallback(rig, monkeypatch, stop):
    o, store, calls, stage = rig
    stage(plan("send_email"), "en")
    o._pending_action.pop("id")
    monkeypatch.setenv("CORTEX_RUNNER_FIELDS", "0")
    monkeypatch.setattr(planner, "summarise", summarise)
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", Mock(side_effect=RuntimeError("offline")))
    delivery = Delivery()
    o.set_turn_delivery(delivery)

    def lost(**kw):
        if stop:
            delivery.cancelled.set()
        raise TimeoutError("lost")
    calls["send_email"].side_effect = lost
    reply, mode = o._execute_staged_plan()
    assert mode == "AGENT_ACTION_ERROR"
    assert reply.endswith("⚠️ I don't know whether step p1 (send an email) went through — "
                          "please check before asking again; I won't retry on my own.")
    [effect] = store.in_states(["desconhecido"])
    assert effect.id and ":p1" not in effect.id
    calls["send_email"].assert_called_once()
