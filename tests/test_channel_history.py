"""Each non-UI entry point reads only its own durable thread for today."""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api as api
import src.main as cli
from src.core.staging_mixin import CODE_TESTS_OUTPUT
from src.ui import chat_store as cs, routes


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = cs.ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cs, "store", store)
    monkeypatch.setattr(routes, "chat_store", store)
    monkeypatch.setattr(api, "SECRET_API_KEY", "test-fake")
    monkeypatch.setattr(api, "_ALLOWLIST", [])
    return store


class RecordingOrch:
    def __init__(self, max_history=20, fail=False):
        self.engine_analytical = self.engine_creative = SimpleNamespace(
            model_name="fake", max_history=max_history)
        self.histories = []
        self.fail = fail

    def process(self, text, image_b64=None, language=None, conversation_history=None):
        self.histories.append(conversation_history)
        if self.fail:
            raise RuntimeError("turn failed")
        return "reply", "OK"

    def process_stream(self, text, image_b64=None, language=None,
                       conversation_history=None):
        self.process(text, image_b64, language, conversation_history)
        yield "reply", True, "OK", "en"


def call(monkeypatch, orch, path="/ask", host="127.0.0.1"):
    monkeypatch.setattr(api, "orchestrator", orch)
    with TestClient(api.app, client=(host, 50000)) as client:
        return client.post(path, headers={"X-Cortex-Key": "test-fake"},
                           json={"user_input": "current"})


def pair(user, reply):
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": reply}]


CHANNELS = [("/ask", "127.0.0.1", cs.TERMINAL_LABEL),
            ("/ask", "::1", cs.TERMINAL_LABEL),
            ("/ask", "localhost", cs.TERMINAL_LABEL),
            ("/ask", "100.64.0.42", cs.NETWORK_LABEL),
            ("/ask_voice", "127.0.0.1", cs.VOICE_LABEL),
            ("/ask_stream", "127.0.0.1", cs.VOICE_LABEL)]


@pytest.mark.parametrize("path,host,label", CHANNELS)
@pytest.mark.parametrize("populated", [False, True])
def test_channel_history(store, monkeypatch, path, host, label, populated):
    if populated:
        for source, name in [("terminal", cs.TERMINAL_LABEL),
                             ("network", cs.NETWORK_LABEL), ("voice", cs.VOICE_LABEL)]:
            cs.append_daily(source, name, name, "saved")
    orch = RecordingOrch()
    assert call(monkeypatch, orch, path, host).status_code == 200
    assert orch.histories == [pair(label, "saved") if populated else []]


def test_read_does_not_create_and_failed_turn_does_not_mirror(store, monkeypatch):
    title = cs.daily_title(cs.TERMINAL_LABEL)
    assert store.find_titled(title) is None
    assert routes._daily_history_msgs(cs.TERMINAL_LABEL, 10) == []
    assert store.list_conversations() == []
    orch = RecordingOrch(fail=True)
    assert call(monkeypatch, orch).status_code == 500
    assert orch.histories == [[]]
    assert store.list_conversations() == []


def test_yesterdays_voice_is_not_used(store, monkeypatch):
    cid = store.conversation_titled(cs.daily_title(
        cs.VOICE_LABEL, datetime.now().astimezone() - timedelta(days=1)))
    store.append_turn(cid, "yesterday", "old reply")
    orch = RecordingOrch()
    assert call(monkeypatch, orch, "/ask_voice").status_code == 200
    assert orch.histories == [[]]


@pytest.mark.parametrize("path,host,label", CHANNELS)
@pytest.mark.parametrize("max_history,turns", [(5, 2), (1, 1), (None, 5),
                                              (MagicMock(), 5)])
def test_limit_and_shared_builder(store, monkeypatch, path, host, label,
                                  max_history, turns):
    cid = store.conversation_titled(cs.daily_title(label))
    for i in range(6):
        store.append_turn(cid, str(i), "saved")
    # Half-turns must never become dangling messages for the 8B.
    store.append_turn(cid, "unfinished", "")
    store.append_turn(cid, "tests", "Failed" + CODE_TESTS_OUTPUT["en"] + "untrusted",
                      status_mode="CODE_TESTS_FAILED")
    orch = RecordingOrch(max_history=max_history)
    if max_history is None:
        del orch.engine_analytical.max_history
    assert call(monkeypatch, orch, path, host).status_code == 200
    expected = []
    for i in range(8 - turns, 6):
        expected += pair(str(i), "saved")
    assert orch.histories == [expected + pair("tests", "Failed")]


def test_restart_reads_from_disk(store, monkeypatch):
    cs.append_daily("voice", cs.VOICE_LABEL, "before restart", "saved")
    store._conn.close()
    reopened = cs.ChatStore(db_path=store.db_path)
    monkeypatch.setattr(cs, "store", reopened)
    monkeypatch.setattr(routes, "chat_store", reopened)
    orch = RecordingOrch()
    assert call(monkeypatch, orch, "/ask_voice").status_code == 200
    assert orch.histories == [pair("before restart", "saved")]


@pytest.mark.parametrize("path,host,label", CHANNELS)
@pytest.mark.parametrize("read", ["find_titled", "get_turns"])
def test_read_errors_are_empty_history(store, monkeypatch, path, host, label, read):
    cs.append_daily("voice", label, "saved", "reply")

    def broken(*args, **kwargs):
        raise RuntimeError("read failed")

    monkeypatch.setattr(store, read, broken)
    orch = RecordingOrch()
    assert call(monkeypatch, orch, path, host).status_code == 200
    assert orch.histories == [[]]


def test_lookup_is_fail_soft(store, monkeypatch):
    monkeypatch.setattr(store, "_connect", lambda: None)
    assert store.find_titled(cs.daily_title(cs.VOICE_LABEL)) is None


def test_terminal_loop_reads_before_each_turn(store, monkeypatch):
    cs.append_daily("terminal", cs.TERMINAL_LABEL, "earlier", "saved")
    cs.append_daily("voice", cs.VOICE_LABEL, "voice", "separate")
    orch = RecordingOrch(max_history=2)
    monkeypatch.setattr(cli, "CortexOrchestrator", lambda: orch)
    typed = iter(["first", "second", "sair"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(typed))
    with pytest.raises(SystemExit):
        cli.run()
    assert orch.histories == [pair("earlier", "saved"), pair("first", "reply")]
