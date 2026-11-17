"""A disk-backed action must be removed before approval or cancellation."""
import os
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.audit import audit
from tests.test_pending_identity import EMAIL, _orch


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ran = []
    o = _orch(tmp_path, monkeypatch, ran)
    o._pending_action = dict(EMAIL)
    o._persist_pending_action()
    return o, ran


def _turn(o, word, stream, lang="pt"):
    if stream:
        *_, (reply, _final, mode, _lang) = o.process_stream(word, language=lang)
        return reply, mode
    return o.process(word, language=lang)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("word,lang", [("sim", "pt"), ("não", "pt"),
                                      ("yes", "en"), ("no", "en")])
def test_remove_failure_preserves_action_for_retry(staged, monkeypatch, stream, word, lang):
    o, ran = staged
    disk = Path(o._PENDING_ACTION_FILE)
    before = disk.read_bytes()
    remove = os.remove

    def fail_pending(path, *args, **kwargs):
        if str(path) == str(disk):
            raise PermissionError("cannot remove pending action")
        return remove(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("src.core.staging_mixin.os.remove", fail_pending)
        log = Mock()
        patch.setattr(audit, "log", log)
        reply, mode = _turn(o, word, stream, lang)
        assert ran == []
        assert (reply, mode) == ((
            "Não consegui tirar essa ação da fila com segurança — não fiz nada. Tenta de novo."
            if lang == "pt" else
            "I couldn't take that action off the queue safely — I did nothing. Please try again."
        ), "AGENT_ACTION_ERROR")
        assert disk.read_bytes() == before and o._pending_action is not None
        log.assert_any_call("–", "pending", "claim_remove_failed", payload={})
    assert _turn(o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    assert ran == [("server", "send_email")] and not disk.exists()


@pytest.mark.parametrize("stream", [False, True])
def test_normal_claim_runs_once(staged, stream):
    o, ran = staged
    assert _turn(o, "sim", stream)[1] == "AGENT_ACTION_DONE"
    _turn(o, "sim", stream)
    assert ran == [("server", "send_email")]
    assert not Path(o._PENDING_ACTION_FILE).exists()


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_yeses_run_once(staged, tmp_path, monkeypatch, stream):
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
        _turn(instance, "sim", stream))) for instance in (o, other)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert sorted(mode for _, mode in results) == ["AGENT_ACTION_DONE", "AGENT_ACTION_STALE"]
    assert ran == [("server", "send_email")]


def test_missing_file_without_disk_record_is_not_failure(staged):
    o, ran = staged
    os.remove(o._PENDING_ACTION_FILE)
    o._pending_action = dict(EMAIL)  # legacy in-memory action, without an id
    assert o._execute_pending_action()[1] == "AGENT_ACTION_DONE"
    assert ran == [("server", "send_email")]
