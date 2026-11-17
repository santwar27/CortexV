"""Delivery receipts decide whether morning briefing jobs were seen."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from src import daily_driver as dd


NOW = datetime(2026, 10, 23, 8, tzinfo=timezone.utc)


@pytest.fixture
def delivery(monkeypatch, tmp_path):
    from src.core.audit import audit
    from src.integrations import job_hunter

    monkeypatch.setenv("HOME", str(tmp_path))
    sources = dd.DailySources(
        events_today=[{"summary": "PRIVATE CALENDAR ITEM", "start": None}],
        unread=[{"from": "PRIVATE EMAIL SENDER <private@example.test>"}],
        jobs=[{"id": 7, "title": "PRIVATE JOB TITLE"},
              {"id": 8, "title": "SECOND PRIVATE JOB TITLE"}],
    )
    chat = Mock()
    chat.create_conversation.return_value = {"id": "fake-conversation"}
    def conversation_titled_on(title, now):
        conversation = chat.create_conversation(title=title)
        return conversation["id"] if conversation is not None else None
    chat.conversation_titled_on = conversation_titled_on
    chat.append_turn.return_value = {"id": "fake-turn"}
    log = Mock()
    log.set_status.return_value = True
    notify = Mock(return_value=True)
    run = Mock(return_value=subprocess.CompletedProcess(["say"], 0))
    clock = Mock(wraps=datetime)
    clock.now.return_value = NOW
    monkeypatch.setattr(dd, "datetime", clock)
    monkeypatch.setattr(dd, "gather", lambda now: sources)
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path / "briefings"))
    monkeypatch.setattr(dd.subprocess, "run", run)
    monkeypatch.setattr("src.core.heartbeat.notify", notify)
    monkeypatch.setattr("src.ui.chat_store.store", chat)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: log)
    monkeypatch.setattr(job_hunter, "refresh_jobs", lambda: dict(job_hunter._ZERO))
    return SimpleNamespace(
        sources=sources, chat=chat, log=log, notify=notify, run=run,
        audit_path=Path(audit.path),
    )


def _assert_seen(delivery, ids):
    assert delivery.log.set_status.call_args_list == [
        call(job_id, "seen") for job_id in ids
    ]


def _assert_audit(delivery, *, notify=None, chat=None, speak=None, jobs_seen=0):
    lines = (delivery.audit_path.read_text().splitlines()
             if delivery.audit_path.exists() else [])
    delivered = [(line, json.loads(line)) for line in lines
                 if json.loads(line).get("stage") == "daily"
                 and json.loads(line).get("event") == "delivered"]
    assert len(delivered) == 1
    line, record = delivered[0]
    assert record["payload"] == {
        "notify": notify, "chat": chat, "speak": speak,
        "jobs": len(delivery.sources.jobs), "jobs_seen": jobs_seen,
    }
    for channel in ("notify", "chat", "speak"):
        assert record["payload"][channel] is locals()[channel]
    assert type(record["payload"]["jobs"]) is int
    assert type(record["payload"]["jobs_seen"]) is int
    for private in (dd.assemble(delivery.sources, NOW), "PRIVATE JOB TITLE",
                    "PRIVATE EMAIL SENDER", "private@example.test",
                    "PRIVATE CALENDAR ITEM"):
        assert private not in line


@pytest.mark.parametrize("failure", ["create", "append"])
def test_failed_chat_with_successful_notification_keeps_jobs_new(delivery, failure):
    if failure == "create":
        delivery.chat.create_conversation.return_value = None
    else:
        delivery.chat.append_turn.return_value = None
    assert dd.main(["--notify", "--to-chat"]) == 0
    _assert_seen(delivery, [])
    _assert_audit(delivery, notify=True, chat=False)


def test_post_to_chat_requires_saved_turn(delivery):
    delivery.chat.append_turn.return_value = None
    assert dd.post_to_chat("PRIVATE BRIEFING TEXT", NOW) is None
    delivery.chat.append_turn.assert_called_once_with(
        conversation_id="fake-conversation", user="o que eu tenho pra hoje?",
        reply="PRIVATE BRIEFING TEXT", status_mode="DAILY", source="daily",
    )


def test_successful_chat_with_failed_notification_marks_jobs_seen(delivery):
    delivery.notify.return_value = False
    assert dd.main(["--notify", "--to-chat"]) == 0
    _assert_seen(delivery, [7, 8])
    assert delivery.chat.append_turn.call_args.kwargs["reply"] == dd.assemble(
        delivery.sources, NOW)
    _assert_audit(delivery, notify=False, chat=True, jobs_seen=2)


def test_notification_alone_does_not_mark_jobs_seen(delivery):
    assert dd.main(["--notify"]) == 0
    _assert_seen(delivery, [])
    delivery.chat.create_conversation.assert_not_called()
    delivery.run.assert_not_called()
    _assert_audit(delivery, notify=True)


@pytest.mark.parametrize("returncode", [1, 0])
def test_speech_exit_code_controls_delivery_and_seen(delivery, returncode):
    delivery.run.return_value = subprocess.CompletedProcess(["say"], returncode)
    assert dd.speak("PRIVATE BRIEFING TEXT") is (returncode == 0)
    delivery.run.reset_mock()
    assert dd.main(["--speak"]) == 0
    _assert_seen(delivery, [7, 8] if returncode == 0 else [])
    delivery.run.assert_called_once()
    _assert_audit(delivery, speak=(returncode == 0),
                  jobs_seen=2 if returncode == 0 else 0)


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("say", 180),
                                 RuntimeError("say unavailable")])
def test_speech_errors_are_failed_deliveries(delivery, error):
    delivery.run.side_effect = error
    assert dd.speak("PRIVATE BRIEFING TEXT") is False
    assert dd.main(["--speak"]) == 0
    _assert_seen(delivery, [])
    _assert_audit(delivery, speak=False)


def test_notification_exception_is_a_failed_delivery(delivery):
    delivery.notify.side_effect = RuntimeError("notification unavailable")
    assert dd.main(["--notify"]) == 0
    _assert_seen(delivery, [])
    _assert_audit(delivery, notify=False)


def test_chat_alone_delivers_text_and_marks_jobs_seen(delivery):
    assert dd.main(["--to-chat"]) == 0
    _assert_seen(delivery, [7, 8])
    _assert_audit(delivery, chat=True, jobs_seen=2)


def test_deliveries_keep_their_order_and_successful_speech_survives_chat_failure(
        delivery):
    calls = Mock()
    calls.attach_mock(delivery.notify, "notify")
    calls.attach_mock(delivery.chat, "chat")
    calls.attach_mock(delivery.run, "say")
    calls.attach_mock(delivery.log, "jobs")
    delivery.chat.append_turn.side_effect = RuntimeError("store unavailable")
    assert dd.main(["--notify", "--to-chat", "--speak"]) == 0
    assert [call[0] for call in calls.mock_calls] == [
        "notify", "chat.create_conversation", "chat.append_turn", "say",
        "jobs.set_status", "jobs.set_status",
    ]
    _assert_seen(delivery, [7, 8])
    _assert_audit(delivery, notify=True, chat=False, speak=True, jobs_seen=2)


@pytest.mark.parametrize("args", [[], ["--stdout"]])
def test_preview_does_not_consume_jobs_or_audit_delivery(delivery, args):
    assert dd.main(args) == 0
    _assert_seen(delivery, [])
    delivery.notify.assert_not_called()
    delivery.chat.create_conversation.assert_not_called()
    delivery.run.assert_not_called()
    assert not delivery.audit_path.exists()


@pytest.mark.parametrize("result", [False, RuntimeError("jobs unavailable")])
def test_audit_counts_only_successful_seen_marks(delivery, result):
    delivery.log.set_status.side_effect = [True, result]
    assert dd.main(["--speak"]) == 0
    _assert_seen(delivery, [7, 8])
    _assert_audit(delivery, speak=True, jobs_seen=1)


def test_audit_failure_does_not_break_delivery(delivery, monkeypatch):
    monkeypatch.setattr("src.core.audit.audit.log",
                        Mock(side_effect=RuntimeError("audit unavailable")))
    assert dd.main(["--speak"]) == 0
    _assert_seen(delivery, [7, 8])
