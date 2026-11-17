"""Approved effects record intent before calling integrations; uncertainty never retries."""
import subprocess
from unittest.mock import Mock

import pytest

from src import gate_review
from src.core import effect_store
from src.core.audit import audit
from src.core.gate_queue import GateQueue
from src.core.staging_mixin import Delivery
from src.integrations import applescript, calendar_client, gmail_client, reminders_client
from src.tools import macos_controller as mac
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch
from tests.test_pending_identity import EMAIL


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store, raising=False)
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.set_turn_origin("ui", "conversation")
    yield o, store
    if store._conn is not None:
        store._conn.close()


def stage(o, action="send_email"):
    args = {"app_name": "Notion"} if action == "mac_quit_app" else dict(EMAIL["args"])
    o._pending_action = {"type": action, "args": args}
    o._persist_pending_action()
    return o._pending_action["id"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["done", "unavailable", "duplicate", "cancelled"])
def test_inline_intent(context, tmp_path, monkeypatch, stream, lang, case):
    o, store = context
    eid = stage(o)
    delivery = Delivery()
    o.set_turn_delivery(delivery)
    create = store.create
    if case == "unavailable":
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("blocked")
        store.db_path = str(blocker / "effects.db")
    elif case == "duplicate":
        create("send_email", {}, effect_id=eid)
    elif case == "cancelled":
        def cancel_after_approval(*args, **kwargs):
            effect = create(*args, **kwargs)
            delivery.cancelled.set()
            return effect
        monkeypatch.setattr(store, "create", cancel_after_approval)

    def send(**kwargs):
        effect = store.get(eid)
        assert effect.state == "executando" and effect.args == EMAIL["args"]
        assert effect.origin == {"channel": "ui", "conversation": "conversation", "turn": delivery.id}
        return {"id": "sent", "to": kwargs["to"]}
    call = Mock(side_effect=send)
    monkeypatch.setattr(gmail_client, "send_email", call)
    log = Mock()
    monkeypatch.setattr(audit, "log", log)
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    if case == "done":
        assert mode == "AGENT_ACTION_DONE"
        assert reply == ("Email enviado para joao@exemplo.com (id sent)." if lang == "pt"
                         else "Email sent to joao@exemplo.com (id sent).")
        assert [t.to_state for t in store.history(eid)] == ["preparando", "executando", "feito"]
        call.assert_called_once()
        _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
        call.assert_called_once()
    else:
        call.assert_not_called()
        if case == "cancelled":
            assert mode == "AGENT_ACTION_CANCELLED"
            assert reply == ("Parei antes de executar — não fiz nada." if lang == "pt" else
                             "Stopped before running it — nothing was done.")
            assert [t.to_state for t in store.history(eid)] == ["preparando", "cancelado"]
        else:
            assert mode == "AGENT_ACTION_ERROR"
            assert reply == ("Não consegui registrar essa ação antes de executar — não fiz nada."
                             if lang == "pt" else
                             "I couldn't record that action before running it — I did nothing.")
        log.assert_any_call("–", "effects", "cancelled" if case == "cancelled" else
                            "not_recorded", payload={"id": eid, "action": "send_email"})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("action", ["send_email", "create_event", "create_reminder",
                                    "mac_quit_app", "mac_sleep"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_outcomes(context, monkeypatch, stream, lang, action, uncertain):
    o, store = context
    eid = stage(o, action)
    if action.startswith("mac_"):
        o.macos = mac.MacOSController()
        run = Mock(side_effect=([subprocess.CompletedProcess([], 0, "id", "")]
                               if action == "mac_quit_app" else []) + [
            subprocess.TimeoutExpired("effect", 5) if uncertain else
            subprocess.CompletedProcess([], 1, "", "denied")])
        monkeypatch.setattr(mac.subprocess, "run", run)
        count = 2 if action == "mac_quit_app" else 1
    else:
        module, exception, pt, en = {
            "send_email": (gmail_client, gmail_client.GmailUnavailable,
                           "Falha ao enviar email: denied", "Failed to send email: denied"),
            "create_event": (calendar_client, calendar_client.CalendarUnavailable,
                             "Falha ao criar evento: denied", "Failed to create event: denied"),
            "create_reminder": (reminders_client, reminders_client.RemindersUnavailable,
                                "Falha ao criar lembrete: denied", "Failed to create reminder: denied"),
        }[action]
        error = (TimeoutError("lost") if action == "send_email" else ConnectionError("lost"))
        if not uncertain:
            error = exception("denied")
        elif action == "create_reminder":
            error = exception("lost")
            error.uncertain = True
        run, count = Mock(side_effect=error), 1
        monkeypatch.setattr(module, action, run)
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    assert mode == "AGENT_ACTION_ERROR" and run.call_count == count
    effect = store.get(eid)
    assert effect.state == ("desconhecido" if uncertain else "falhou")
    if uncertain:
        if action == "mac_quit_app":
            assert reply == (
                "Não sei se Notion fechou (timeout após 5s). Confere antes de pedir de novo — "
                "não vou repetir sozinho." if lang == "pt" else
                "I don't know whether Notion closed (timed out after 5s). Please check before "
                "asking again — I won't retry on my own.")
        elif action == "mac_sleep":
            assert reply == (
                "Não sei se o Mac foi dormir (timeout após 5s). Confere antes de pedir de novo — "
                "não vou repetir sozinho." if lang == "pt" else
                "I don't know whether the Mac went to sleep (timed out after 5s). Please check before "
                "asking again — I won't retry on my own.")
        else:
            assert reply == (f"Não sei se deu certo ({effect.detail}). Confere antes de pedir de novo — "
                             "não vou repetir sozinho." if lang == "pt" else
                             f"I don't know whether it went through ({effect.detail}). Please check before "
                             "asking again — I won't retry on my own.")
        assert not store.transition(eid, "desconhecido", "executando")
    elif action.startswith("mac_"):
        pt = ("não consegui fechar 'Notion': denied" if action == "mac_quit_app" else
              "não consegui colocar o Mac pra dormir: denied")
        en = ("couldn't close 'Notion': denied" if action == "mac_quit_app" else
              "couldn't put the Mac to sleep: denied")
        assert reply == (pt if lang == "pt" else en)
    else:
        assert reply == (pt if lang == "pt" else en)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_timeout_without_message(context, monkeypatch, stream, lang):
    o, store = context
    eid = stage(o)
    call = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(gmail_client, "send_email", call)
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    assert mode == "AGENT_ACTION_ERROR"
    assert reply == (
        "Não sei se deu certo (TimeoutError). Confere antes de pedir de novo — "
        "não vou repetir sozinho." if lang == "pt" else
        "I don't know whether it went through (TimeoutError). Please check before "
        "asking again — I won't retry on my own.")
    effect = store.get(eid)
    assert effect.state == "desconhecido" and effect.detail == "TimeoutError"
    assert store.history(eid)[-1].detail == "TimeoutError"
    call.assert_called_once()


@pytest.mark.parametrize("target", ["executando", "feito", "falhou", "desconhecido"])
def test_transition_failure(context, monkeypatch, target):
    from src.core.effect_runner import run_effect
    _, store = context
    transition = store.transition
    monkeypatch.setattr(store, "transition", lambda eid, de, para, detalhe=None:
                        False if para == target else transition(eid, de, para, detalhe))
    result = ("done", "AGENT_ACTION_DONE") if target != "falhou" else ("failed", "OTHER_MODE")
    call = Mock(side_effect=RuntimeError("lost")) if target == "desconhecido" else Mock(return_value=result)
    log = Mock()
    monkeypatch.setattr(audit, "log", log)
    out = run_effect("send_email", {"body": "private"}, call, effect_id="id", store=store)
    if target == "executando":
        call.assert_not_called()
        assert out[1] == "AGENT_ACTION_ERROR" and store.get("id").state == "preparando"
        event = "not_recorded"
    else:
        call.assert_called_once()
        assert store.get("id").state == "executando"
        if target == "desconhecido":
            assert out[0].startswith("Não sei se deu certo (lost)") and out[1] == "AGENT_ACTION_ERROR"
        else:
            assert out == result
        event = "result_not_recorded"
    log.assert_called_once_with("–", "effects", event, payload={"id": "id", "action": "send_email"})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_gate_and_exclusions(context, monkeypatch, tmp_path, stream, lang):
    o, store = context
    for name in ("read_file_at", "read_files", "write_answers_pdf"):
        monkeypatch.setattr(o, "_run_" + name, lambda args: ("ok", "AGENT_ACTION_DONE"))
        stage(o, name)
        assert _turn(o, "sim" if lang == "pt" else "yes", stream, lang) == ("ok", "AGENT_ACTION_DONE")
    assert o._run_action("unknown", {})[1] == "AGENT_ACTION_ERROR"
    assert store.in_states(["feito", "preparando", "executando", "falhou"]) == []
    call = Mock(return_value={"id": "sent", "to": "fake@example.com"})
    monkeypatch.setattr(gmail_client, "send_email", call)
    queue = GateQueue(str(tmp_path / "gate.jsonl"))
    gid = queue.enqueue("send_email", EMAIL["args"])
    assert gate_review.approve(gid, queue=queue, runner=o._run_action).startswith("[executado]")
    assert len(store.in_states(["feito"])) == 1
    call.assert_called_once()


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_timeout_contract(monkeypatch, lang):
    run = Mock(side_effect=subprocess.TimeoutExpired("effect", 5))
    monkeypatch.setattr(mac.subprocess, "run", run)
    with pytest.raises(reminders_client.RemindersUnavailable) as caught:
        applescript.run("fake", app_label="Reminders", unavailable_exc=reminders_client.RemindersUnavailable)
    assert caught.value.uncertain is True
    ok, msg = mac._run(["fake"], lang=lang)
    assert ok is None
    controller = mac.MacOSController()
    sleep = (f"não sei se o Mac foi dormir ({msg})." if lang == "pt" else
             f"I don't know whether the Mac went to sleep ({msg}).")
    quit_reply = (f"não sei se notion fechou ({msg})." if lang == "pt" else
                  f"I don't know whether notion closed ({msg}).")
    assert controller.sleep_mac_outcome(lang=lang) == (None, sleep)
    assert controller.sleep_mac(lang=lang) == sleep
    assert controller.quit_app_outcome("notion", lang=lang) == (None, quit_reply)
    assert controller.quit_app("notion", lang=lang) == quit_reply
    assert ("timeout" if lang == "pt" else "timed out") in controller.mute(lang=lang)
