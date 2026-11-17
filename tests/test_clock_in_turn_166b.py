"""The minute clock belongs only to the current, in-flight user turn."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.core import engine as mod


NOTE = "[SYSTEM NOTE: Current local time: 10:01 (IST).]"
QUESTION = "Explain how a clock works.\nKeep it brief."
REPLY = "A clock measures time."


class LocalDateTime(datetime):
    def astimezone(self, tz=None):
        # Freeze the local timezone too, independent of the test host's TZ.
        return self if tz is None else super().astimezone(tz)


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=LocalDateTime(
        2026, 11, 15, 10, 1, tzinfo=timezone(timedelta(hours=5, minutes=30), "IST")))
    monkeypatch.setattr(mod, "datetime", SimpleNamespace(now=lambda: clock.now))
    return clock


@pytest.fixture
def engine(clock, monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "8b")
    memory = SimpleNamespace(
        get_user_profile=Mock(return_value=""),
        retrieve_relevant_context=Mock(return_value=""),
        store_interaction=Mock(),
    )
    # Run the real constructor, replacing memory before it can open a Palace.
    with patch.object(mod, "CortexMemory", return_value=memory):
        engine = mod.OllamaEngine()
    return engine


@pytest.mark.parametrize("tz_name", ["IST", ""])
def test_system_prompt_is_stable_across_minutes(engine, clock, tz_name):
    clock.now = clock.now.replace(tzinfo=timezone(timedelta(hours=5, minutes=30), tz_name))
    engine.memory.get_user_profile.return_value = "USER PROFILE: exact bytes\n\n"
    engine.push_turn_context("TURN CONTEXT: exact bytes")
    first = engine.system_prompt
    first_text_only = engine.text_only_system_prompt
    clock.now = clock.now.replace(minute=2)
    assert engine.system_prompt == first
    assert engine.text_only_system_prompt == first_text_only
    label = tz_name or "+0530"
    assert first["content"] == (
        "TEMPORAL CONTEXT (use this for any relative date the user "
        "mentions like 'hoje', 'amanhã', 'next Tuesday'):\n"
        "- Current date: 2026-11-15 (Sunday)\n"
        f"- Local timezone: {label} (+0530)\n"
        "When generating ISO 8601 timestamps for tools like create_event, "
        "include the same timezone offset shown above.\n\n"
        "USER PROFILE: exact bytes\n\n"
        "TURN CONTEXT: exact bytes\n\n" + engine._tool_system_body
    )
    assert "10:01" not in first["content"]
    assert "10:02" not in first["content"]
    assert "Current local time" not in first["content"]
    header = first["content"][:-len(engine._tool_system_body)]
    assert first_text_only["content"] == header + engine._system_body + mod._NO_TOOLS_NOTE
    assert f"10:02 ({label})" in engine._enrich_input(QUESTION, "", None)


def run_turn(engine, path, question=QUESTION):
    if path == "stream":
        return "".join(engine.chat_stream(question, response_language="en"))
    if path.startswith("tools"):
        return engine.chat_with_tools(
            question, [], lambda *_: "Clock details.", response_language="en")[0]
    return engine.chat(question, response_language="en")


@pytest.mark.parametrize("context", ["", "Earlier clock discussion."])
@pytest.mark.parametrize("path", ["chat", "stream", "tools_direct", "tools_followup"])
def test_clock_is_sent_but_clean_input_is_stored_and_guarded(
        engine, clock, monkeypatch, context, path):
    engine.memory.retrieve_relevant_context.return_value = context
    old_history = [{"role": "user", "content": "Hello."},
                   {"role": "assistant", "content": "Hello."}]
    engine.history = deepcopy(old_history)
    guard = Mock(wraps=engine._guard_ps5_claim)
    monkeypatch.setattr(engine, "_guard_ps5_claim", guard)
    seen = []

    def fake_chat(**kwargs):
        seen.append(deepcopy(kwargs["messages"]))
        clock.now = clock.now.replace(minute=2)
        if path == "tools_followup" and len(seen) == 1:
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "lookup", "arguments": {}}}]}}
        response = {"message": {"content": REPLY}}
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(mod.ollama, "chat", fake_chat)
    assert run_turn(engine, path) == REPLY
    assert len(seen) == (2 if path == "tools_followup" else 1)
    directive = engine._language_directive("en")
    expected = f"{directive}{QUESTION}\n\n{NOTE}"
    if context:
        expected = f"PAST CONTEXT:\n{context}\n\nCURRENT QUERY: {expected}"
    for messages in seen:
        users = [m["content"] for m in messages if m["role"] == "user"]
        assert users == ["Hello.", expected]
        assert "Current local time" not in messages[0]["content"]
    assert engine.history == old_history + [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": REPLY}]
    assert engine.memory.retrieve_relevant_context.call_args.args == (QUESTION,)
    engine.memory.store_interaction.assert_called_once_with(QUESTION, REPLY)
    assert guard.call_args.args[1] == QUESTION


def test_only_current_turn_gets_a_fresh_clock(engine, clock, monkeypatch):
    seen = []

    def fake_chat(**kwargs):
        seen.append(deepcopy(kwargs["messages"]))
        return {"message": {"content": REPLY}}

    monkeypatch.setattr(mod.ollama, "chat", fake_chat)
    run_turn(engine, "chat")
    clock.now = clock.now.replace(minute=2)
    run_turn(engine, "chat", "Explain the minute hand.")
    assert seen[0][0] == seen[1][0]
    assert seen[1][1] == {"role": "user", "content": QUESTION}
    assert seen[1][-1]["content"].endswith(NOTE.replace("10:01", "10:02"))
    assert sum("[SYSTEM NOTE:" in m["content"] for m in seen[1]) == 1


@pytest.mark.parametrize("num_ctx", [300, 10])
def test_context_trimming_preserves_question_and_clock(engine, num_ctx):
    content = engine._enrich_input(QUESTION, "Memory. " * 1000, "en")
    messages = [{"role": "system", "content": "S" * 70},
                {"role": "user", "content": content}]
    original = deepcopy(messages)
    fitted = mod.fit_messages_to_ctx(messages, num_ctx=num_ctx, gen_reserve=0)
    assert fitted[0] == messages[0]
    assert len(fitted[-1]["content"]) < len(content)
    assert fitted[-1]["content"].startswith("PAST CONTEXT:\n")
    assert fitted[-1]["content"].endswith(
        f"\n\nCURRENT QUERY: {engine._language_directive('en')}{QUESTION}\n\n{NOTE}")
    assert messages == original


@pytest.mark.parametrize("path", ["chat", "stream"])
def test_refusal_retry_keeps_original_turn_clock(engine, clock, monkeypatch, path):
    question = "qual meu passaporte"
    reply = "Seu passaporte brasileiro é AB123456."
    engine.memory.retrieve_relevant_context.return_value = "[KNOWLEDGE] passaporte: AB123456"
    seen = []

    def fake_chat(**kwargs):
        seen.append(deepcopy(kwargs["messages"]))
        clock.now = clock.now.replace(minute=2)
        content = ("Desculpe, mas não posso fornecer informações de identificação pessoal."
                   if len(seen) == 1 else reply)
        response = {"message": {"content": content}}
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(mod.ollama, "chat", fake_chat)
    assert run_turn(engine, path, question) == reply
    assert len(seen) == 2
    assert seen[0][-1] == seen[1][-1]
    assert seen[1][-1]["content"].endswith(NOTE)
    assert engine.history == [{"role": "user", "content": question},
                              {"role": "assistant", "content": reply}]
    engine.memory.store_interaction.assert_called_once_with(question, reply)
