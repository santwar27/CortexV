"""The displayed preview identifies the disk action a UI decision may settle."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import planner
from tests.test_dispatch_parity import _build_real_orch
from tests.test_ui_routes import client, ui_app, _parse_sse

AUTH = {"X-Cortex-Key": "test-key-xyz"}
# Routing removes the transcript's final sentence mark, preserving the rest.
BODY = "First line\n" + "Whole content, including this line.\n" * 40 + "Last line"
EMAIL = f"send email to joao@example.com; subject: Meeting; body: {BODY}."
STALE = {
    "pt": "Essa prévia não está mais valendo — não fiz nada. Se ainda quiser, me pede de novo.",
    "en": "That preview is no longer valid — I did nothing. If you still want it, ask me again.",
}


@pytest.fixture
def contract(client, ui_app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background, o._offer = False, None
    o.engine_analytical.max_history = 10
    for eng in (o.engine_analytical, o.engine_creative):
        for method in ("chat", "chat_with_tools", "chat_stream"):
            getattr(eng, method).side_effect = lambda *a, **kw: pytest.fail("Unexpected LLM")
    send = Mock(return_value={"id": "fake-email", "to": "joao@example.com"})
    monkeypatch.setattr("src.integrations.gmail_client.send_email", send)
    ui_app.state.orchestrator = o
    ids = [client.post("/ui/conversations", headers=AUTH,
                       json={"title": name}).json()["id"] for name in ("A", "B")]

    def ask(text=EMAIL, cid=None, **extra):
        response = client.post("/ui/chat/stream", headers=AUTH, json={
            "text": text, "conversation_id": cid or ids[0], **extra})
        assert response.status_code == 200
        frames = _parse_sse(response.text)
        assert frames[-1]["type"] == "final", frames
        return frames

    return o, send, ids, ask


def disk(o):
    return json.loads(Path(o._PENDING_ACTION_FILE).read_text())


@pytest.mark.parametrize("lang,query_lang", [
    ("pt", None), ("pt", "pt"), ("pt", ""), ("pt", "invalid"), ("en", "en"),
])
def test_preview_read_and_final(contract, client, lang, query_lang):
    o, send, ids, ask = contract
    assert client.get("/ui/pending", headers=AUTH).json() == {"pending": None}
    frames = ask(language=lang)
    action = disk(o)
    assert "lang" not in action
    expiry = (datetime.fromisoformat(action["staged_at"])
              + timedelta(minutes=o._PENDING_TTL_MIN)).isoformat()
    assert action["by"] == frames[0]["id"]
    assert frames[-1]["pending"] == {"id": action["id"], "expires_at": expiry}
    expected = {
        "id": action["id"], "type": "send_email", "preview": frames[-1]["turn"]["reply"],
        "origin": {"channel": "ui", "conversation": ids[0]},
        "staged_at": action["staged_at"], "expires_at": expiry,
        "expired": False, "here": True,
    }
    o._pending_action = None  # A read must use disk, not this shared RAM slot.
    o._reply_lang = "en" if lang == "pt" else "pt"
    for cid, here in zip(ids, (True, False)):
        params = {"conversation_id": cid}
        if query_lang is not None:
            params["lang"] = query_lang
        response = client.get("/ui/pending", headers=AUTH, params=params)
        assert response.status_code == 200
        assert response.json() == {"pending": {**expected, "here": here}}
        assert BODY in response.json()["pending"]["preview"]
    saved = client.get(f"/ui/conversations/{ids[0]}", headers=AUTH).json()["turns"][-1]
    assert "pending" not in saved and saved == frames[-1]["turn"]
    assert disk(o) == action and o._pending_action is None
    send.assert_not_called()


def test_expired_read_is_read_only(contract, client):
    o, send, ids, ask = contract
    ask()
    action = disk(o)
    action["staged_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    path = Path(o._PENDING_ACTION_FILE)
    path.write_text(json.dumps(action))
    before = path.read_bytes()
    result = client.get("/ui/pending", headers=AUTH,
                        params={"conversation_id": ids[0]}).json()["pending"]
    assert result["expired"] is True and result["here"] is True
    assert datetime.fromisoformat(result["expires_at"]) == (
        datetime.fromisoformat(action["staged_at"]) + timedelta(minutes=o._PENDING_TTL_MIN))
    assert path.read_bytes() == before
    send.assert_not_called()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("lang,yes,no", [("pt", "sim", "não"), ("en", "yes", "no")])
def test_bound_decisions(contract, direct, lang, yes, no):
    o, send, ids, ask = contract
    ask(language=lang)
    old = disk(o)["id"]
    ask(language=lang)  # Same conversation replaces the old preview.
    current = disk(o)

    def decide(word, expected, cid=None):
        if not direct:
            frame = ask(word, cid=cid, pending_id=expected, language=lang)[-1]
            assert "pending" not in frame
            turn = frame["turn"]
            return turn["reply"], turn["status_mode"]
        o.set_turn_origin("ui", cid or ids[0])
        getattr(o, "set_turn_pending_id", lambda _id: None)(expected)
        try:
            return o.process(word, language=lang)
        finally:
            o.set_turn_origin(None)
            getattr(o, "set_turn_pending_id", lambda _id: None)(None)

    for word in (yes, no):
        assert decide(word, old) == (STALE[lang], "AGENT_ACTION_STALE")
        assert disk(o) == current
    assert decide(yes, current["id"], ids[1])[1] == "AGENT_ACTION_ELSEWHERE"
    assert disk(o) == current
    send.assert_not_called()
    assert decide(yes, current["id"])[1] == "AGENT_ACTION_DONE"
    send.assert_called_once()
    assert send.call_args.kwargs["body"] == BODY
    assert decide(yes, current["id"]) == (STALE[lang], "AGENT_ACTION_STALE")
    assert not Path(o._PENDING_ACTION_FILE).exists()
    send.assert_called_once()


@pytest.mark.parametrize("query_lang", [None, "en"])
def test_plan_preview(contract, client, monkeypatch, query_lang):
    o, send, ids, ask = contract
    plan = planner.Plan(objetivo="Send the complete email", passos=[planner.Step(
        id="p1", tool="send_email", args={"body": BODY}, depende_de=[], descricao="Send")])
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: plan)
    frame = ask("planeja: send the email", language="en")[-1]
    o._reply_lang = "pt"
    params = {"conversation_id": ids[0]}
    if query_lang is not None:
        params["lang"] = query_lang
    result = client.get("/ui/pending", headers=AUTH,
                        params=params).json()["pending"]
    assert result["type"] == "execute_plan"
    assert result["preview"] == frame["turn"]["reply"] == planner.render_preview(plan, lang="en")
    send.assert_not_called()


def test_replacement_between_check_and_claim(contract, monkeypatch):
    o, send, _ids, ask = contract
    ask()
    action = disk(o)
    newer = {**action, "id": "replacement"}
    execute = o._execute_pending_action

    def replace_then_claim():
        Path(o._PENDING_ACTION_FILE).write_text(json.dumps(newer))
        return execute()
    monkeypatch.setattr(o, "_execute_pending_action", replace_then_claim)
    turn = ask("sim", pending_id=action["id"], language="pt")[-1]["turn"]
    assert (turn["reply"], turn["status_mode"]) == (STALE["pt"], "AGENT_ACTION_STALE")
    assert disk(o) == newer
    send.assert_not_called()


def test_bound_expiry_and_stale_denial(contract):
    o, send, _ids, ask = contract
    ask()
    action = disk(o)
    action["staged_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    path = Path(o._PENDING_ACTION_FILE)
    path.write_text(json.dumps(action))
    stale = ask("não", pending_id="older-preview", language="pt")[-1]["turn"]
    assert (stale["reply"], stale["status_mode"]) == (STALE["pt"], "AGENT_ACTION_STALE")
    assert disk(o) == action
    assert ask("sim", pending_id=action["id"])[-1]["turn"]["status_mode"] == "STALE_ACTION_DISCARDED"
    assert not path.exists()
    assert ask("sim", pending_id=action["id"])[-1]["turn"]["status_mode"] == "AGENT_ACTION_STALE"
    send.assert_not_called()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("binding", ["current", "stale", "elsewhere", "expired"])
@pytest.mark.parametrize("word", ["sim?", "Sim?", "ok?", "claro?", "sim…"])
def test_doubtful_bound_reply_is_not_a_decision(contract, direct, binding, word):
    o, send, ids, ask = contract
    ask(language="pt")
    action = disk(o)
    if binding == "expired":
        action["staged_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        Path(o._PENDING_ACTION_FILE).write_text(json.dumps(action))
    expected = "old-preview" if binding == "stale" else action["id"]
    cid = ids[1] if binding == "elsewhere" else ids[0]
    # An unrecognised reply falls through to chat on HEAD. Use only fake
    # engines here, including the direct process() entry point.
    for engine in (o.engine_analytical, o.engine_creative):
        for method in ("chat", "chat_stream", "chat_with_tools"):
            getattr(engine, method).side_effect = None
            getattr(engine, method).return_value = "Unclear."
    if direct:
        o.set_turn_origin("ui", cid)
        o.set_turn_pending_id(expected)
        try:
            mode = o.process(word, language="pt")[1]
        finally:
            o.set_turn_origin(None)
            o.set_turn_pending_id(None)
    else:
        mode = ask(word, cid=cid, pending_id=expected, language="pt")[-1]["turn"]["status_mode"]
    assert mode == "OFFLINE"
    send.assert_not_called()
    if binding == "elsewhere":
        assert disk(o) == action
    else:
        assert not Path(o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("word", ["sim.", "Sim!", "não?", "não."])
def test_punctuated_bound_decisions(contract, word):
    o, send, _ids, ask = contract
    ask(language="pt")
    action = disk(o)
    stale = ask(word, pending_id="old-preview", language="pt")[-1]["turn"]
    assert (stale["reply"], stale["status_mode"]) == (STALE["pt"], "AGENT_ACTION_STALE")
    assert disk(o) == action
    mode = ask(word, pending_id=action["id"], language="pt")[-1]["turn"]["status_mode"]
    approved = word in ("sim.", "Sim!")
    assert mode == ("AGENT_ACTION_DONE" if approved else "AGENT_ACTION_CANCELLED")
    assert send.call_count == int(approved)
    assert not Path(o._PENDING_ACTION_FILE).exists()


def test_pending_requires_auth(client):
    assert client.get("/ui/pending").status_code == 401
