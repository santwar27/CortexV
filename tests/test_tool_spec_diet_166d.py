"""166d: bounded prompt cost without changing the tool schema or contracts."""
import json
import re

from src.core.tool_specs import TOOL_SPECS


# Frozen before the description diet. Keep the full schema, including array
# items and the event parameter named "description" (which is not prose).
FROZEN_STRUCTURE = [
    {"type": "function", "function": {
        "name": "read_document",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"},
            "query": {"type": "string"},
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "find_file",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "content": {"type": "boolean"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "read_file_at",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "question": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_document",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["filename", "content"]},
    }},
    {"type": "function", "function": {
        "name": "list_emails",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer"},
            "query": {"type": "string"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "send_email",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        }, "required": ["to", "subject", "body"]},
    }},
    {"type": "function", "function": {
        "name": "list_events",
        "parameters": {"type": "object", "properties": {
            "time_min": {"type": "string"},
            "time_max": {"type": "string"},
            "max_results": {"type": "integer"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "create_event",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "start_iso": {"type": "string"},
            "end_iso": {"type": "string"},
            "location": {"type": "string"},
            "description": {"type": "string"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        }, "required": ["summary", "start_iso", "end_iso"]},
    }},
    {"type": "function", "function": {
        "name": "create_reminder",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "due_iso": {"type": "string"},
        }, "required": ["title"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
]


def _without_descriptions(value):
    if isinstance(value, dict):
        return {
            key: _without_descriptions(item)
            for key, item in value.items()
            if not (key == "description" and isinstance(item, str))
        }
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def _descriptions(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "description" and isinstance(item, str):
                yield item
            else:
                yield from _descriptions(item)
    elif isinstance(value, list):
        for item in value:
            yield from _descriptions(item)


def test_total_json_size_budget():
    total = len(json.dumps(TOOL_SPECS, ensure_ascii=False))
    sizes = {
        spec["function"]["name"]: len(json.dumps(spec, ensure_ascii=False))
        for spec in TOOL_SPECS
    }
    assert total <= 5600, f"Total JSON size: {total}; per-tool sizes: {sizes}"


def test_frozen_schema_and_tool_order():
    assert _without_descriptions(TOOL_SPECS) == FROZEN_STRUCTURE


def test_no_model_written_confirmation_reply_instructions():
    for description in _descriptions(TOOL_SPECS):
        assert not re.search(r"\b(?:sim|não)\b", description, re.IGNORECASE), (
            f"Confirmation reply wording belongs in code: {description}"
        )


def test_preserved_contract_phrases():
    functions = {spec["function"]["name"]: spec["function"] for spec in TOOL_SPECS}
    event = functions["create_event"]["parameters"]["properties"]
    assert "EXACT user title" in event["summary"]["description"]
    assert "spelling/language/punctuation" in event["summary"]["description"]
    assert "never translate, pluralise or paraphrase" in event["summary"]["description"]
    for name in ("start_iso", "end_iso"):
        assert "ISO 8601 with offset" in event[name]["description"]
    assert "default duration is 1 hour" in event["end_iso"]["description"]
    reminder = functions["create_reminder"]["parameters"]["properties"]
    assert "EXACT user wording" in reminder["title"]["description"]
    assert "ISO 8601 with offset" in reminder["due_iso"]["description"]
    assert "Stable knowledge: answer directly" in functions["web_search"]["description"]


def test_named_document_reading_vs_file_location():
    functions = {spec["function"]["name"]: spec["function"] for spec in TOOL_SPECS}
    read = functions["read_document"]["description"]
    find = functions["find_file"]["description"]
    assert "Open, read or summarise a named document" in read
    assert "Use find_file when asked where a file is" in read
    assert "Find where a file is" in find
    assert "location is unknown" in find
    assert "Use read_document to open, read or summarise a named document" in find


def test_email_requests_include_polite_forms_and_recipient():
    email = next(spec["function"] for spec in TOOL_SPECS
                 if spec["function"]["name"] == "send_email")
    assert "any request to send, write, reply to or forward an email" in email["description"]
    assert "polite or indirect requests" in email["description"]
    assert "pode mandar" in email["description"]
    assert "manda um email" in email["description"]
    assert "recipient address in to" in email["description"]


def test_relative_times_use_conversation_clock():
    functions = {spec["function"]["name"]: spec["function"] for spec in TOOL_SPECS}
    event = functions["create_event"]["parameters"]["properties"]
    reminder = functions["create_reminder"]["parameters"]["properties"]
    for description in (
        reminder["due_iso"]["description"],
        event["start_iso"]["description"],
        event["end_iso"]["description"],
        functions["list_events"]["description"],
    ):
        assert "current date and local time given in the conversation" in description
        assert "system prompt" not in description
