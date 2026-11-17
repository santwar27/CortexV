"""Queue buttons decide actions; chat reply instructions stay in chat."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core import tool_risk
from tests.test_gate_view import AUTH, no_integrations, read, view
from tests.test_ui_routes import client, ui_app


ARGS = {
    "to": "ana@example.com", "subject": "Fictional email", "body": "First\nLast",
    "summary": "Fictional event", "start_iso": "2026-10-22T10:00:00+01:00",
    "end_iso": "2026-10-22T11:00:00+01:00", "location": "Example room",
    "attendees": ["ana@example.com"], "title": "Fictional reminder",
    "due_iso": "2026-10-22T09:00:00+01:00", "app_name": "Example App",
    "phrased_as_shutdown": True, "path": "/fictional/paper.pdf",
    "paths": ["/fictional/first.pdf", "/fictional/second.pdf"],
    "paper": "/fictional/paper.pdf", "out": "/fictional/answers.pdf",
    "answers": {"1": "Fictional answer"},
}


def queue_preview(view, client, name, args, lang):
    q, _ = view
    gid = q.enqueue(name, args)
    before = Path(q.path).read_bytes()
    result = read(client, lang=lang)
    [item] = result["recent"]
    assert result["pending"] == [item]
    assert item["id"] == gid and item["args"] == args
    assert Path(q.path).read_bytes() == before
    return item["preview"]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("name", [*tool_risk._PREVIEW_BUILDERS, "execute_plan"])
def test_whole_preview_without_last_instruction(view, client, ui_app, name, lang):
    args = ARGS if name != "execute_plan" else {
        "objetivo": "Fictional two-step plan", "passos": [
            {"id": "p1", "tool": "send_email", "args": ARGS},
            {"id": "p2", "tool": "create_reminder", "args": {"title": "Follow up"},
             "depende_de": ["p1"]},
        ],
    }
    prompt = ui_app.state.orchestrator._human_confirmation_prompt(
        {"type": name, "args": args}, lang=lang)
    preview = queue_preview(view, client, name, args, lang)
    if name == "ps5_standby":
        # Task 130's literal chat prompt has an inline reply instruction.
        expected = ("Put the PS5 in rest mode? An open game may lose unsaved progress."
                    if lang == "en" else
                    "Colocar o PS5 em repouso? Um jogo aberto pode perder o que não foi salvo.")
    else:
        expected = prompt.rsplit("\n", 1)[0]
    assert preview == expected
    assert "Responda 'sim'" not in preview and "Reply 'yes'" not in preview


@pytest.mark.parametrize("lang, expected", [
    ("pt", "Confirmar a ação?"), ("en", "Confirm the action?"),
])
def test_unknown_tool_keeps_question(view, client, lang, expected):
    assert queue_preview(view, client, "fictional_unknown", {}, lang) == expected


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_reply_instruction_inside_email_body_stays(view, client, ui_app, lang):
    body = "First paragraph\nResponda 'sim' para enviar ou 'não' para cancelar.\nLast paragraph"
    args = {"to": "ana@example.com", "body": body}
    prompt = ui_app.state.orchestrator._human_confirmation_prompt(
        {"type": "send_email", "args": args}, lang=lang)
    preview = queue_preview(view, client, "send_email", args, lang)
    assert preview == prompt.rsplit("\n", 1)[0]
    assert body in preview


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_pending_keeps_chat_instruction(view, client, ui_app, tmp_path, monkeypatch, lang):
    o = ui_app.state.orchestrator
    action = {"id": "fictional-staged", "type": "send_email", "args": ARGS,
              "staged_at": datetime.now(timezone.utc).isoformat()}
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(action))
    o._PENDING_ACTION_FILE = str(pending)
    monkeypatch.setattr("src.ui.routes._get_orchestrator", lambda request: o)
    before = pending.read_bytes()
    queue_preview(view, client, "send_email", ARGS, lang)
    response = client.get("/ui/pending", headers=AUTH, params={"lang": lang})
    assert response.status_code == 200
    prompt = o._human_confirmation_prompt(action, lang=lang)
    assert response.json()["pending"]["preview"] == prompt
    assert prompt.endswith("Responda 'sim' para enviar ou 'não' para cancelar." if lang == "pt"
                           else "Reply 'yes' to send or 'no' to cancel.")
    assert pending.read_bytes() == before
