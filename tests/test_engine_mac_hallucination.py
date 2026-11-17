"""Task 110: fake-model tests; no real model, memory store or Mac controls."""
from unittest.mock import Mock

import pytest

from src.core.hallucination import HALLUCINATION_FALLBACK


LIE = "Diminuí o brilho da tela do seu Mac para 0."
MAC_FALLBACK = {
    "pt": "Não fiz isso — não consegui acionar o controle do Mac. Tenta direto, "
          "por exemplo: 'diminua o brilho', 'volume 30', 'modo escuro', 'abre o Safari'.",
    "en": "I didn't do that — I couldn't reach the Mac control. Try it directly, "
          "e.g. 'lower the brightness', 'volume 30', 'dark mode', 'open Safari'.",
}
TOOLS = [{"type": "function", "function": {"name": "fake_mac_control"}}]


@pytest.fixture
def rig(monkeypatch):
    from src.core.engine import OllamaEngine

    memory = Mock()
    memory.retrieve_relevant_context.return_value = ""
    memory.get_user_profile.return_value = ""
    monkeypatch.setattr("src.core.engine.CortexMemory", lambda: memory)
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "8b")
    engine = OllamaEngine()
    model = Mock(return_value={"message": {"content": LIE, "tool_calls": []}})
    executor = Mock(return_value="Control applied.")
    audit = Mock()
    monkeypatch.setattr("src.core.engine.ollama.chat", model)
    monkeypatch.setattr("src.core.engine.audit.log", audit)
    return engine, model, executor, audit


@pytest.mark.parametrize("language,claim,user_input", [
    ("pt", LIE, "Diminua brilho no mac para 0"),
    ("en", "I lowered the brightness.", "Lower the brightness on my Mac"),
    ("pt", "Diminuiu o brilho da tela para 0%.", "Dominua o brilho da tela"),
    ("pt", "Diminuiu o volume para 0%.", "deixa o volume mais baixinho"),
    ("pt", "Aqui está um print da sua tela agora.", "tira um print da minha tela agora"),
])
def test_mac_no_tool_uses_honest_replacement_everywhere(rig, language, claim, user_input):
    engine, model, executor, audit = rig
    model.return_value = {"message": {"content": claim, "tool_calls": []}}
    result = engine.chat_with_tools(user_input, TOOLS, executor, language)

    expected = MAC_FALLBACK[language]
    assert result == (expected, None)
    executor.assert_not_called()
    model.assert_called_once()
    audit.assert_called_once_with("–", "engine", "hallucination_blocked", payload={
        "original_head": claim[:120], "user_input": user_input[:80], "kind": "mac",
    })
    assert engine.history == [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": expected},
    ]
    engine.memory.store_interaction.assert_called_once_with(user_input, expected)


def test_mac_claim_after_executed_tool_is_untouched(rig):
    engine, model, executor, audit = rig
    model.side_effect = [
        {"message": {"content": "", "tool_calls": [{"function": {
            "name": "fake_mac_control", "arguments": {"brightness": 0},
        }}]}},
        {"message": {"content": LIE}},
    ]
    result = engine.chat_with_tools("Diminua brilho no mac para 0", TOOLS, executor, "pt")
    assert result == (LIE, "fake_mac_control")
    executor.assert_called_once_with("fake_mac_control", {"brightness": 0})
    assert model.call_count == 2
    audit.assert_not_called()
    assert engine.history[-1] == {"role": "assistant", "content": LIE}
    engine.memory.store_interaction.assert_called_once_with("Diminua brilho no mac para 0", LIE)


def test_mac_claim_with_guard_off_is_untouched(rig, monkeypatch):
    engine, model, executor, audit = rig
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "gemma4")
    assert engine.chat_with_tools("Diminua brilho no mac para 0", TOOLS, executor, "pt") == (LIE, None)
    executor.assert_not_called()
    audit.assert_not_called()
    assert engine.history[-1]["content"] == LIE
    engine.memory.store_interaction.assert_called_once_with("Diminua brilho no mac para 0", LIE)


def test_mac_translation_stands_down(rig):
    engine, model, executor, audit = rig
    user_input = "Traduza para português: I lowered the brightness."
    assert engine.chat_with_tools(user_input, TOOLS, executor, "pt") == (LIE, None)
    executor.assert_not_called()
    audit.assert_not_called()
    engine.memory.store_interaction.assert_called_once_with(user_input, LIE)


@pytest.mark.parametrize("language", ["pt", "en"])
def test_existing_family_keeps_original_fallback(rig, language):
    engine, model, executor, audit = rig
    model.return_value = {"message": {"content": "Enviei o email.", "tool_calls": []}}
    user_input = "Manda email pro João"
    expected = HALLUCINATION_FALLBACK[language]
    assert engine.chat_with_tools(user_input, TOOLS, executor, language) == (expected, None)
    executor.assert_not_called()
    audit.assert_called_once_with("–", "engine", "hallucination_blocked", payload={
        "original_head": "Enviei o email.", "user_input": user_input,
    })
    assert engine.history[-1]["content"] == expected
    engine.memory.store_interaction.assert_called_once_with(user_input, expected)
