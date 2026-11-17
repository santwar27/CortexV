"""Undo must stop at the Mac action, never reach the earlier reminder."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.action_ledger import ledger
from src.tools import macos_controller as mac
from tests.test_dispatch_parity import _build_real_orch


@pytest.fixture
def context(tmp_path, monkeypatch):
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.macos = SimpleNamespace(
        quit_app_outcome=Mock(return_value=(True, "Fechando Safari.")),
        sleep_mac_outcome=Mock(return_value=(True, "Mac indo dormir.")),
        open_app_outcome=Mock(return_value=(True, "Abrindo Safari.")))
    deleted = Mock(return_value=1)
    monkeypatch.setattr("src.integrations.reminders_client.delete_reminder", deleted)
    ledger.record("create_reminder", {"title": "leite"}, result_id="rem1")
    return o, deleted


def _turn(o, text, stream=False):
    # No language override: the command itself must select the undo language.
    if stream:
        *_, (reply, _final, mode, _lang) = o.process_stream(text)
        return reply, mode
    return o.process(text)


def _confirm(o, stream=False, sleep=False):
    ask = "põe o mac pra dormir" if sleep else "encerra o Safari"
    assert _turn(o, ask, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    o.macos.quit_app_outcome.assert_not_called()
    o.macos.sleep_mac_outcome.assert_not_called()
    return _turn(o, "sim", stream)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word,lang,expected", [
    ("desfaz", "pt", "Pronto, desfiz: abri o Safari de novo — o que não estava salvo não volta."),
    ("undo", "en", "Done, undid: reopened Safari — anything unsaved doesn't come back."),
])
def test_quit_then_reopen(context, stream, word, lang, expected):
    o, deleted = context
    assert _confirm(o, stream) == ("Fechando Safari.", "AGENT_ACTION_DONE")
    entry = ledger.load_entries()[-1]
    assert entry["action"] == "mac_quit_app" and entry["source"] == "gate"
    assert entry["args"] == {"app_name": "Safari", "was_running": True}
    assert entry["reversible"] is True
    assert _turn(o, word, stream) == (expected, "UNDO_DONE")
    o.macos.open_app_outcome.assert_called_once_with("Safari", lang=lang)
    deleted.assert_not_called()
    undo = ledger.load_entries()[-1]
    assert undo["action"] == "undo" and undo["undone_ts"] == entry["ts"]


@pytest.mark.parametrize("closed_lang", ["pt", "en"])
def test_already_closed_is_not_reopened(context, closed_lang):
    o, deleted = context
    reply = mac.already_closed_text("Safari", closed_lang)
    o.macos.quit_app_outcome.return_value = (True, reply)
    ask, yes = (("encerra o Safari", "sim") if closed_lang == "pt" else
                ("quit Safari", "yes"))
    assert _turn(o, ask)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert _turn(o, yes) == (reply, "AGENT_ACTION_DONE")
    entry = ledger.load_entries()[-1]
    assert entry["action"] == "mac_quit_app"
    assert entry["args"] == {"app_name": "Safari", "was_running": False}
    assert entry["reversible"] is False
    assert _turn(o, "desfaz") == (
        "A última coisa que fiz foi fechar um app, e isso não dá pra desfazer.",
        "UNDO_IRREVERSIBLE")
    o.macos.open_app_outcome.assert_not_called()
    deleted.assert_not_called()
    assert len(ledger.load_entries()) == 2


@pytest.mark.parametrize("sleep", [False, True])
@pytest.mark.parametrize("ok", [None, False, True])
def test_mac_outcomes_stop_undo_at_the_right_action(context, sleep, ok):
    o, deleted = context
    action = "mac_sleep" if sleep else "mac_quit_app"
    label = "colocar o Mac pra dormir" if sleep else "fechar um app"
    subject = "o Mac foi dormir" if sleep else "Safari fechou"
    reply = (f"não sei se {subject} (timeout após 5s)." if ok is None else
             "não consegui executar: denied" if ok is False else
             "Mac indo dormir." if sleep else "Fechando Safari.")
    getattr(o.macos, "sleep_mac_outcome" if sleep else "quit_app_outcome").return_value = (ok, reply)
    result = _confirm(o, sleep=sleep)
    if ok is None:
        assert result == (reply[0].upper() + reply[1:] +
                          " Confere antes de pedir de novo — não vou repetir sozinho.",
                          "AGENT_ACTION_ERROR")
    else:
        assert result == (reply, "AGENT_ACTION_DONE" if ok else "AGENT_ACTION_ERROR")
    entries = ledger.load_entries()
    if ok is False:
        assert len(entries) == 1 and entries[0]["action"] == "create_reminder"
        assert _turn(o, "desfaz")[1] == "UNDO_DONE"
        deleted.assert_called_once_with("leite", reminder_id="rem1")
    else:
        entry = entries[-1]
        assert entry["action"] == action and entry["source"] == "gate"
        args = {} if sleep else {"app_name": "Safari"}
        args.update({"uncertain": True} if ok is None else
                    {} if sleep else {"was_running": True})
        assert entry["args"] == args
        if sleep or ok is None:
            assert entry["reversible"] is False
            assert _turn(o, "desfaz") == (
                f"A última coisa que fiz foi {label}, e isso não dá pra desfazer.",
                "UNDO_IRREVERSIBLE")
            assert len(ledger.load_entries()) == 2
            deleted.assert_not_called()
    o.macos.open_app_outcome.assert_not_called()


@pytest.mark.parametrize("word,lang", [("desfaz", "pt"), ("undo", "en")])
def test_reopen_failure_is_honest(context, word, lang):
    o, deleted = context
    _confirm(o)
    reply = "não consegui abrir 'Safari': x"
    o.macos.open_app_outcome.return_value = (False, reply)
    expected = (f"Não consegui reabrir o Safari: {reply}" if lang == "pt" else
                f"I couldn't reopen Safari: {reply}")
    assert _turn(o, word) == (expected, "AGENT_ACTION_ERROR")
    o.macos.open_app_outcome.assert_called_once_with("Safari", lang=lang)
    deleted.assert_not_called()
    assert len(ledger.load_entries()) == 2


@pytest.mark.parametrize("word,prefix", [
    ("desfaz", "Pronto, desfiz:"), ("desfaça", "Pronto, desfiz:"),
    ("cancela a última ação", "Pronto, desfiz:"), ("UNDO", "Done, undid:"),
])
def test_undo_phrase_selects_language(context, word, prefix):
    o, deleted = context
    reply, mode = _turn(o, word)
    assert mode == "UNDO_DONE" and reply.startswith(prefix)
    deleted.assert_called_once_with("leite", reminder_id="rem1")


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("ok", [True, False, None])
def test_open_outcome_preserves_legacy_reply(monkeypatch, lang, ok):
    run = Mock(return_value=(ok, "x"))
    monkeypatch.setattr(mac, "_run", run)
    controller = mac.MacOSController()
    expected = (("Abrindo Safari." if lang == "pt" else "Opening Safari.") if ok else
                ("não consegui abrir 'Safari': x" if lang == "pt" else "couldn't open 'Safari': x"))
    assert controller.open_app_outcome(" Safari ", lang=lang) == (ok, expected)
    assert controller.open_app(" Safari ", lang=lang) == expected
    assert run.call_count == 2
    run.assert_called_with(["open", "-a", "Safari"], lang=lang)
