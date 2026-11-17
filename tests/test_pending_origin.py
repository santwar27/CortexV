"""Only the preview's conversation (or voice for UI) may settle an action."""
import builtins
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.audit import audit
from tests.test_pending_identity import _orch, clock, REMINDER_ASK
from tests.test_ui_routes import client, ui_app, _parse_sse


def _origin(o, channel, conversation=None):
    # The no-op lets the pre-fix run expose the behavioural failures too.
    getattr(o, "set_turn_origin", lambda *args: None)(channel, conversation)


def _turn(o, text, channel="ui", conversation="A", stream=False, lang="pt"):
    _origin(o, channel, conversation)
    try:
        if stream:
            *_, (reply, _final, mode, _lang) = o.process_stream(text, language=lang)
            return reply, mode
        return o.process(text, language=lang)
    finally:
        _origin(o, None)


def _disk(o):
    return json.loads(Path(o._PENDING_ACTION_FILE).read_text())


@pytest.fixture
def staged(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("HOME", str(tmp_path))
    ran = []
    o = _orch(tmp_path, monkeypatch, ran)
    assert _turn(o, REMINDER_ASK)[1] == "AGENT_AWAITING_CONFIRMATION"
    return o, ran


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word,lang,message", [
    ("sim", "pt", "Essa confirmação está esperando em outra conversa — responde lá (ou por voz)."),
    ("não", "pt", "Essa confirmação está esperando em outra conversa — responde lá (ou por voz)."),
    ("yes", "en", "That confirmation is waiting in another conversation — answer it there (or by voice)."),
    ("no", "en", "That confirmation is waiting in another conversation — answer it there (or by voice)."),
])
def test_other_conversation_cannot_settle(staged, stream, word, lang, message):
    o, ran = staged
    before = _disk(o)
    reply, mode = _turn(o, word, conversation="B", stream=stream, lang=lang)
    assert (reply, mode) == (message, "AGENT_ACTION_ELSEWHERE")
    assert ran == [] and _disk(o) == before
    events = [json.loads(line) for line in Path(audit.path).read_text().splitlines()]
    wrong = [e for e in events if e.get("event") == "approval_wrong_origin"]
    assert wrong and wrong[-1]["stage"] == "pending"
    assert word not in json.dumps(wrong[-1].get("payload", {}))
    _turn(o, "sim")
    assert ran == [("server", "create_reminder")]


@pytest.mark.parametrize("stream", [False, True])
def test_other_turn_keeps_slot_but_own_turn_discards(staged, stream):
    o, ran = staged
    before = _disk(o)
    assert _turn(o, "que horas são", conversation="B", stream=stream)[1] == "REFLEX"
    assert _disk(o) == before
    _turn(o, "que horas são", stream=stream)
    assert not Path(o._PENDING_ACTION_FILE).exists() and ran == []


def test_other_conversation_cannot_replace_slot(staged):
    o, ran = staged
    before = _disk(o)
    assert _turn(o, REMINDER_ASK, conversation="B")[1] == "AGENT_STAGE_CONFLICT"
    assert _disk(o) == before and ran == []


def test_voice_unrelated_turn_keeps_ui_preview(staged):
    o, _ran = staged
    before = _disk(o)
    _turn(o, "que horas são", channel="voice", stream=True)
    assert _disk(o) == before


@pytest.mark.parametrize("stage_channel,approve_channel", [("ui", None), (None, "ui")])
def test_explicit_none_origin_remains_unrestricted(tmp_path, monkeypatch, clock,
                                                  stage_channel, approve_channel):
    o = _orch(tmp_path, monkeypatch, ran := [])
    _turn(o, REMINDER_ASK, channel=stage_channel)
    assert "origin" in _disk(o)
    _turn(o, "sim", channel=approve_channel)
    assert ran == [("server", "create_reminder")]


@pytest.mark.parametrize("channel", ["terminal", "network"])
def test_other_channel_cannot_approve_ui(staged, channel):
    o, ran = staged
    before = _disk(o)
    assert _turn(o, "sim", channel=channel)[1] == "AGENT_ACTION_ELSEWHERE"
    assert ran == [] and _disk(o) == before


@pytest.mark.parametrize("word", ["sim", "não"])
def test_voice_can_settle_ui(staged, word):
    o, ran = staged
    assert _turn(o, word, channel="voice", stream=True)[1] != "AGENT_ACTION_ELSEWHERE"
    assert ran == ([("server", "create_reminder")] if word == "sim" else [])
    assert not Path(o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("channel", ["voice", "terminal", "network", None])
def test_same_channel_and_unrestricted_turns(tmp_path, monkeypatch, clock, channel):
    o = _orch(tmp_path, monkeypatch, ran := [])
    _turn(o, REMINDER_ASK, channel=channel)
    _turn(o, "sim", channel=channel)
    assert ran == [("server", "create_reminder")]


def test_ui_cannot_approve_voice(tmp_path, monkeypatch, clock):
    o = _orch(tmp_path, monkeypatch, ran := [])
    _turn(o, REMINDER_ASK, channel="voice")
    before = _disk(o)
    assert _turn(o, "sim")[1] == "AGENT_ACTION_ELSEWHERE"
    assert ran == [] and _disk(o) == before


@pytest.mark.parametrize("lang,word,message", [
    ("pt", "sim", "Essa ação é de antes de uma atualização — me pede de novo."),
    ("en", "no", "That action predates an update — please ask again."),
])
@pytest.mark.parametrize("has_id", [False, True])
def test_legacy_disk_action_is_withdrawn(tmp_path, monkeypatch, lang, word, message, has_id):
    o = _orch(tmp_path, monkeypatch, ran := [])
    action = {"type": "send_email", "args": {},
              "staged_at": datetime.now(timezone.utc).isoformat()}
    if has_id:
        action["id"] = "legacy"
    Path(o._PENDING_ACTION_FILE).write_text(json.dumps(action))
    reply, _mode = _turn(o, word, lang=lang)
    assert reply == message and ran == []
    assert not Path(o._PENDING_ACTION_FILE).exists()


def test_stream_worker_stamps_origin(tmp_path, monkeypatch, clock):
    o = _orch(tmp_path, monkeypatch, [])
    assert _turn(o, REMINDER_ASK, stream=True)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert _disk(o)["origin"] == {"channel": "ui", "conversation": "A"}


def test_ui_route_uses_conversation_origin(client, ui_app, tmp_path, monkeypatch, clock):
    o = _orch(tmp_path, monkeypatch, ran := [])
    o.engine_analytical.max_history = 10
    ui_app.state.orchestrator = o
    headers = {"X-Cortex-Key": "test-key-xyz"}
    ids = [client.post("/ui/conversations", headers=headers,
                       json={"title": title}).json()["id"] for title in ("A", "B")]

    def ask(text, cid):
        response = client.post("/ui/chat/stream", headers=headers,
                               json={"text": text, "conversation_id": cid})
        assert response.status_code == 200
        return _parse_sse(response.text)[-1]["turn"]["status_mode"]
    assert ask(REMINDER_ASK, ids[0]) == "AGENT_AWAITING_CONFIRMATION"
    before = _disk(o)
    assert before["origin"] == {"channel": "ui", "conversation": ids[0]}
    assert ask("sim", ids[1]) == "AGENT_ACTION_ELSEWHERE"
    assert ran == [] and _disk(o) == before
    assert ask("sim", ids[0]) == "AGENT_ACTION_DONE"
    assert ran == [("server", "create_reminder")]


def test_two_concurrent_yeses_from_same_origin_run_once(staged, tmp_path, monkeypatch):
    o, ran = staged
    other = _orch(tmp_path, monkeypatch, ran)
    barrier = threading.Barrier(2)
    for instance in (o, other):
        execute = instance._execute_pending_action

        def wait_then_execute(execute=execute):
            barrier.wait(timeout=10)
            return execute()
        monkeypatch.setattr(instance, "_execute_pending_action", wait_then_execute)
    results = []
    threads = [threading.Thread(target=lambda instance=instance: results.append(
        _turn(instance, "sim"))) for instance in (o, other)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert len(results) == 2 and ran == [("server", "create_reminder")]


@pytest.mark.parametrize("stream", [False, True])
def test_failed_write_never_offers_approvable_preview(tmp_path, monkeypatch, clock, stream):
    o = _orch(tmp_path, monkeypatch, ran := [])
    real_open = builtins.open

    def fail_slot(path, mode="r", *args, **kwargs):
        if str(path) in (o._PENDING_ACTION_FILE, o._PENDING_ACTION_FILE + ".tmp") and mode == "w":
            raise OSError("disk full")
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", fail_slot)
    reply, mode = _turn(o, REMINDER_ASK, stream=stream)
    assert mode == "AGENT_STAGE_ERROR" and "não" in reply.lower()
    assert not Path(o._PENDING_ACTION_FILE).exists()
    _turn(o, "sim")
    assert ran == []
