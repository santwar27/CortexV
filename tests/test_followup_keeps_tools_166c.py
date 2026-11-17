"""Keep the schema prefix in round 2 without executing a second tool."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.core import engine as mod


QUESTION = "List my events. USER_TEXT_PRIVATE"
REPLY = "You have two events."
TOOLS = [
    {"type": "function", "function": {
        "name": "list_events", "description": "Read calendar events."}},
    {"type": "function", "function": {
        "name": "read_document", "description": "Read a document."}},
]


def tool_response(name="list_events", arguments=None, content=""):
    return {"message": {"role": "assistant", "content": content, "tool_calls": [
        {"function": {"name": name, "arguments": arguments if arguments is not None else {}}}
    ]}}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "8b")
    memory = SimpleNamespace(
        get_user_profile=Mock(return_value=""),
        retrieve_relevant_context=Mock(return_value=""),
        store_interaction=Mock(),
    )
    # Populate the real engine state, stubbing memory before it opens a Palace.
    with patch.object(mod, "CortexMemory", return_value=memory):
        return mod.OllamaEngine()


def run_turn(engine, executor):
    return engine.chat_with_tools(QUESTION, TOOLS, executor, response_language="en")


def assert_same_followup_settings(first, retry):
    assert retry["messages"] is first["messages"]
    assert retry["options"] is first["options"]
    for key in ("model", "keep_alive", "think"):
        assert retry[key] == first[key]


def test_text_followup_keeps_the_same_tools(engine):
    executor = Mock(return_value="Two calendar events.")
    with patch.object(mod.ollama, "chat", side_effect=[
        tool_response(), {"message": {"content": REPLY}},
    ]) as chat:
        assert run_turn(engine, executor) == (REPLY, "list_events")

    assert chat.call_count == 2
    assert all(call.kwargs["tools"] is TOOLS for call in chat.call_args_list)
    executor.assert_called_once_with("list_events", {})
    engine.memory.store_interaction.assert_called_once_with(QUESTION, REPLY)


@pytest.mark.parametrize("content", ["", "I read SECRET_DOCUMENT_PRIVATE."])
def test_followup_call_retries_without_tools_and_audits_only_names(engine, content):
    executor = Mock(return_value="Two calendar events.")
    followup = tool_response("read_document", {"path": "ARGS_PRIVATE"}, content)
    followup["message"]["tool_calls"].append(
        {"function": {"name": "find_file", "arguments": {"query": "QUERY_PRIVATE"}}})
    with patch.object(mod.ollama, "chat", side_effect=[
        tool_response(), followup, {"message": {"content": REPLY}},
    ]) as chat, patch.object(mod.audit, "log") as audit:
        assert run_turn(engine, executor) == (REPLY, "list_events")

    executor.assert_called_once_with("list_events", {})
    assert chat.call_count == 3
    first, second, fallback = [call.kwargs for call in chat.call_args_list]
    assert first["tools"] is second["tools"] is TOOLS
    assert "tools" not in fallback
    assert_same_followup_settings(second, fallback)
    events = [call for call in audit.call_args_list
              if call.args[2] == "tool_followup_called_tool"]
    assert len(events) == 1
    # Exact payload equality excludes arguments, discarded content and user text.
    assert events[0].kwargs["payload"] == {
        "tool": "list_events", "followup_tools": ["read_document", "find_file"]}
    engine.memory.store_interaction.assert_called_once_with(QUESTION, REPLY)


def test_fallback_failure_uses_existing_failure_reply_without_memory(engine):
    executor = Mock(return_value="Two calendar events.")
    with patch.object(mod.ollama, "chat", side_effect=[
        tool_response(), tool_response("read_document"), RuntimeError("PRIVATE_ERROR"),
    ]) as chat, patch.object(mod.audit, "log") as audit:
        assert run_turn(engine, executor) == (
            "The tool ran, but I failed to put the answer together. Could you ask again?",
            "list_events",
        )

    executor.assert_called_once_with("list_events", {})
    assert chat.call_count == 3
    assert "tools" not in chat.call_args_list[2].kwargs
    engine.memory.store_interaction.assert_not_called()
    assert sum(call.args[2] == "tool_followup_called_tool"
               for call in audit.call_args_list) == 1
    failures = [call.kwargs["payload"] for call in audit.call_args_list
                if call.args[2] == "tool_followup_failed"]
    assert failures == [{"tool": "list_events", "error": "RuntimeError"}]


def message_cost(messages):
    return sum(mod._est_tokens(m.get("content", "") or "")
               + mod._est_tokens(str(m.get("tool_calls") or "")) for m in messages)


def test_schema_reserve_cuts_more_and_preserves_system_and_question():
    messages = [{"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": QUESTION}]
    call = tool_response()["message"]
    result = {"role": "tool", "name": "list_events", "content": "Event details. " * 2000}
    tools_cost = mod._est_tokens(json.dumps(TOOLS, ensure_ascii=False))
    num_ctx, gen_reserve = 900, 200
    default = mod.fit_tool_followup(messages, call, result, num_ctx, gen_reserve)
    reserved = mod.fit_tool_followup(messages, call, result, num_ctx, gen_reserve,
                                     extra_reserve=tools_cost)

    assert len(reserved[-1]["content"]) < len(default[-1]["content"]) < len(result["content"])
    assert reserved[0] is messages[0]
    assert reserved[1] is messages[1]
    assert message_cost(reserved) + tools_cost <= num_ctx - gen_reserve
    assert default == mod.fit_tool_followup(
        messages, call, result, num_ctx, gen_reserve, extra_reserve=0)


def test_engine_reserves_same_schema_cost_in_followup(engine):
    result = "Event details. " * 5000
    with patch.object(mod.ollama, "chat", side_effect=[
        tool_response(), {"message": {"content": REPLY}},
    ]) as chat, patch.object(engine, "_fit_payload", wraps=engine._fit_payload) as first_fit, \
            patch.object(mod, "fit_tool_followup", wraps=mod.fit_tool_followup) as followup_fit:
        assert run_turn(engine, Mock(return_value=result)) == (REPLY, "list_events")

    tools_cost = mod._est_tokens(json.dumps(TOOLS, ensure_ascii=False))
    assert first_fit.call_args.kwargs["extra_reserve"] == tools_cost
    assert followup_fit.call_args.kwargs["extra_reserve"] == tools_cost
    second = chat.call_args_list[1].kwargs
    assert len(second["messages"][-1]["content"]) < len(result)
    assert message_cost(second["messages"]) + tools_cost <= (
        mod._NUM_CTX - engine._turn_gen_reserve(second["options"]))


@pytest.mark.parametrize("calls_again", [False, True])
def test_unparseable_args_never_execute_even_if_followup_calls_again(engine, calls_again):
    executor = Mock()
    responses = [tool_response(arguments="{not valid JSON")]
    if calls_again:
        responses.append(tool_response("read_document"))
    responses.append({"message": {"content": "Please rephrase the request."}})
    with patch.object(mod.ollama, "chat", side_effect=responses) as chat:
        assert run_turn(engine, executor) == ("Please rephrase the request.", None)

    executor.assert_not_called()
    assert chat.call_count == (3 if calls_again else 2)
    assert chat.call_args_list[1].kwargs["tools"] is TOOLS
    assert "NOT executed" in chat.call_args_list[1].kwargs["messages"][-1]["content"]


@pytest.mark.parametrize("literal", [False, True])
def test_completed_tool_results_skip_followup(engine, literal):
    result = (mod.LiteralToolReply(REPLY) if literal else mod.SKIP_FOLLOWUP_MARKER + REPLY)
    executor = Mock(return_value=result)
    with patch.object(mod.ollama, "chat", return_value=tool_response()) as chat:
        assert run_turn(engine, executor) == (REPLY, "list_events")

    executor.assert_called_once_with("list_events", {})
    chat.assert_called_once()
    engine.memory.store_interaction.assert_not_called()
