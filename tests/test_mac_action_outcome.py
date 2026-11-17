"""S3 completion follows the controller outcome, never the reply's casing."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import gate_review
from src.core.gate_queue import GateQueue
from src.tools.macos_controller import MacOSController
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("name,steps,ok,pt,en", [
    ("notion", [(True, "id"), (True, "")], True,
     "Fechando notion.", "Closing notion."),
    ("notion", [(True, "id"), (False, "-600")], True,
     "notion já estava fechado.", "notion was already closed."),
    ("notion", [(True, "id"), (False, "isn't running")], True,
     "notion já estava fechado.", "notion was already closed."),
    ("notion", [(True, "id"), (False, "is not running")], True,
     "notion já estava fechado.", "notion was already closed."),
    ("notion", [(False, "-1728")], False,
     "não achei nenhum app chamado 'notion'.", "I couldn't find an app called 'notion'."),
    ("notion", [(True, "id"), (False, "denied")], False,
     "não consegui fechar 'notion': denied", "couldn't close 'notion': denied"),
    ("  ", [], False,
     "preciso do nome do app pra fechar.", "I need the app name to close it."),
])
def test_quit_outcome_and_legacy_text(monkeypatch, lang, name, steps, ok, pt, en):
    script = Mock(side_effect=steps * 2)
    monkeypatch.setattr("src.tools.macos_controller._osascript", script)
    controller = MacOSController()
    expected = pt if lang == "pt" else en
    assert controller.quit_app_outcome(name, lang=lang) == (ok, expected)
    assert controller.quit_app(name, lang=lang) == expected
    assert script.call_count == len(steps) * 2
    assert all(call.kwargs == {"lang": lang} for call in script.call_args_list)


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("ok", [False, True])
def test_sleep_outcome_and_legacy_text(monkeypatch, lang, ok):
    run = Mock(return_value=(ok, "denied"))
    monkeypatch.setattr("src.tools.macos_controller._run", run)
    controller = MacOSController()
    pt = "Mac indo dormir." if ok else "não consegui colocar o Mac pra dormir: denied"
    en = "Mac going to sleep." if ok else "couldn't put the Mac to sleep: denied"
    expected = pt if lang == "pt" else en
    assert controller.sleep_mac_outcome(lang=lang) == (ok, expected)
    assert controller.sleep_mac(lang=lang) == expected
    assert run.call_count == 2
    run.assert_called_with(["/usr/bin/pmset", "sleepnow"], lang=lang)


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o.background = False
    o._offer = None
    return o


CASES = [
    ("mac_quit_app", False, "não consegui fechar 'notion': denied",
     "couldn't close 'notion': denied"),
    ("mac_quit_app", True, "Fechando notion.", "Closing notion."),
    ("mac_quit_app", True, "notion já estava fechado.", "notion was already closed."),
    ("mac_sleep", False, "não consegui colocar o Mac pra dormir: denied",
     "couldn't put the Mac to sleep: denied"),
    ("mac_sleep", True, "Mac indo dormir.", "Mac going to sleep."),
]


def _controller(ok, reply):
    # Legacy methods reproduce the old false-DONE path before the fix.
    return SimpleNamespace(quit_app=Mock(return_value=reply),
                           sleep_mac=Mock(return_value=reply),
                           quit_app_outcome=Mock(return_value=(ok, reply)),
                           sleep_mac_outcome=Mock(return_value=(ok, reply)))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("action,ok,pt,en", CASES)
def test_confirmation_uses_outcome(orch, stream, lang, action, ok, pt, en):
    reply = pt if lang == "pt" else en
    orch.macos = _controller(ok, reply)
    ask = (("encerra o notion" if lang == "pt" else "quit notion")
           if action == "mac_quit_app" else
           ("põe o mac pra dormir" if lang == "pt" else "put the mac to sleep"))
    assert _turn(orch, ask, stream, lang)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action["type"] == action
    orch.macos.quit_app_outcome.assert_not_called()
    orch.macos.sleep_mac_outcome.assert_not_called()
    result = _turn(orch, "sim" if lang == "pt" else "yes", stream, lang)
    assert result == (reply, "AGENT_ACTION_DONE" if ok else "AGENT_ACTION_ERROR")
    if action == "mac_quit_app":
        orch.macos.quit_app_outcome.assert_called_once_with("notion", lang=lang)
    else:
        orch.macos.sleep_mac_outcome.assert_called_once_with(lang=lang)
    orch.macos.quit_app.assert_not_called()
    orch.macos.sleep_mac.assert_not_called()


@pytest.mark.parametrize("action", ["mac_quit_app", "mac_sleep"])
def test_gate_records_controller_failure(orch, tmp_path, action):
    reply = "não consegui executar: denied"
    orch.macos = _controller(False, reply)
    queue = GateQueue(str(tmp_path / "gate.jsonl"))
    gid = queue.enqueue(action, {"app_name": "notion"} if action == "mac_quit_app" else {})
    out = gate_review.approve(gid, queue=queue, runner=orch._run_action)
    assert queue.status(gid) == "failed"
    assert out == f"[FALHOU] {reply}"
