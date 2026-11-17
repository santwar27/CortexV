"""Task 166a: label budget, thinking override, and private failure diagnostics."""
import importlib
import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.core import ollama_client
from src.core import orchestrator as orch_mod


_QUESTION = "e quem fez os gols?"


def _orch():
    orch = orch_mod.CortexOrchestrator.__new__(orch_mod.CortexOrchestrator)
    orch._intent_cache = OrderedDict()
    orch.engine_analytical = SimpleNamespace(model_name="cortex-lite")
    return orch


@pytest.mark.parametrize("think_env", [None, "1"])
def test_classifier_always_disables_thinking_with_enough_budget(monkeypatch, think_env):
    try:
        with monkeypatch.context() as env:
            if think_env is None:
                env.delenv("CORTEX_THINK", raising=False)
            else:
                env.setenv("CORTEX_THINK", think_env)
            importlib.reload(ollama_client)
            # Refresh the imported flag without recreating the orchestrator
            # class referenced by other test modules and isolation fixtures.
            env.setattr(orch_mod, "_THINK", ollama_client.THINK)
            assert orch_mod._THINK is (think_env == "1")
            with patch.object(orch_mod.ollama, "generate", return_value={"response": "YES"}) as gen:
                assert _orch()._needs_realtime(_QUESTION) is True
            gen.assert_called_once()
            kwargs = gen.call_args.kwargs
            assert kwargs["think"] is False
            assert kwargs["options"]["num_predict"] >= 8
            assert kwargs["options"]["temperature"] == 0.0
            assert kwargs["keep_alive"] == ollama_client.KEEP_ALIVE
    finally:
        importlib.reload(ollama_client)


@pytest.mark.parametrize("reply,expected", [
    ("YES", True), (" yes.", True), ("SIM", True), ("NO", False),
])
def test_labels(reply, expected):
    with patch.object(orch_mod.ollama, "generate", return_value={"response": reply}) as gen, \
            patch.object(orch_mod.audit, "log") as audit_log:
        assert _orch()._needs_realtime(_QUESTION) is expected
    gen.assert_called_once()
    audit_log.assert_not_called()


@pytest.mark.parametrize("reply,reason", [
    ("", "empty"), ("maybe", "unparseable"), ("nonsense", "unparseable"),
])
def test_unclear_reply_is_offline_and_audited_once_without_question(reply, reason):
    orch = _orch()
    with patch.object(orch_mod.ollama, "generate", return_value={"response": reply}) as gen, \
            patch.object(orch_mod.audit, "log") as audit_log:
        assert orch._needs_realtime(_QUESTION) is False
        assert orch._needs_realtime("  " + _QUESTION.upper() + "  ") is False
    gen.assert_called_once()
    audit_log.assert_called_once_with(
        "–", "orchestrator", "realtime_classifier_unclear",
        payload={"reason": reason, "reply_length": len(reply)},
    )
    assert _QUESTION not in json.dumps(audit_log.call_args.kwargs["payload"])


def test_exception_keeps_offline_failsafe():
    with patch.object(orch_mod.ollama, "generate", side_effect=RuntimeError("unavailable")) as gen:
        assert _orch()._needs_realtime(_QUESTION) is False
    gen.assert_called_once()


@pytest.mark.parametrize("question,expected", [
    ("quem ganhou o jogo do Flamengo", True),
    ("o que é placar no futebol", False),
    ("2+2", False),
])
def test_fast_paths_skip_classifier(monkeypatch, question, expected):
    monkeypatch.setenv("CORTEX_HUMILITY", "1")
    with patch.object(orch_mod.ollama, "generate") as gen:
        assert _orch()._needs_realtime(question) is expected
    gen.assert_not_called()


def test_successful_label_is_cached():
    orch = _orch()
    with patch.object(orch_mod.ollama, "generate", return_value={"response": "YES"}) as gen:
        assert orch._needs_realtime(_QUESTION) is True
        assert orch._needs_realtime(_QUESTION.upper()) is True
    gen.assert_called_once()
