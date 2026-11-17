"""Undo only reaches dated actions within the last thirty minutes."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import action_ledger as al
from src.core.features_mixin import FeaturesMixin
from src.core.orchestrator import CortexOrchestrator

NOW = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)


@pytest.fixture
def context(tmp_path, monkeypatch):
    import src.integrations.calendar_client as cal
    import src.integrations.reminders_client as rem

    lg = al.ActionLedger(path=str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(al, "ledger", lg)
    # Allow the clock seam to be absent while proving the old implementation fails.
    clock = Mock(return_value=NOW)
    monkeypatch.setattr(FeaturesMixin, "_undo_now", staticmethod(clock), raising=False)
    calls = []
    monkeypatch.setattr(cal, "delete_event", lambda event_id: calls.append(event_id))
    monkeypatch.setattr(rem, "delete_reminder",
                        lambda title, **kw: calls.append({"title": title, **kw}) or 1)
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._reply_lang = "pt"
    return lg, o, calls, clock


def _record(lg, minutes, action="create_event"):
    lg.record(action, {"summary": "Meeting", "title": "Milk"}, result_id="id1",
              extra={"ts": (NOW - timedelta(minutes=minutes)).isoformat()})


def _too_old(lang, age, action="create_event"):
    labels = {"create_event": ("criar um evento", "create an event"),
              "send_email": ("enviar um e-mail", "send an email")}
    pt, en = labels.get(action, (action, action))
    if lang == "pt":
        return (f"Não há nada recente pra desfazer — a última ação ({pt}) foi há {age}. "
                "O desfaz vale até 30 minutos depois.")
    return (f"There's nothing recent to undo — the last action ({en}) was {age} ago. "
            "Undo works up to 30 minutes after.")


@pytest.mark.parametrize("minutes", [10, 30, -5])
def test_recent_boundary_and_future_undo(context, minutes):
    lg, o, calls, clock = context
    _record(lg, minutes)
    assert o._handle_undo()[1] == "UNDO_DONE"
    assert calls == ["id1"]
    entries = lg.load_entries()
    assert entries[-1]["action"] == "undo"
    assert entries[-1]["undone_ts"] == entries[0]["ts"]
    assert clock.call_count == 1


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("minutes,pt,en", [
    (31, "31 min", "31 min"),
    (30 + 1 / 60, "30 min", "30 min"),
    (180, "3 horas", "3 hours"),
    (119, "119 min", "119 min"),
    (120, "2 horas", "2 hours"),
    (24 * 60, "24 horas", "24 hours"),
    (48 * 60, "2 dias", "2 days"),
    (48 * 60 + 59, "2 dias", "2 days"),
])
def test_old_action_reply_and_no_effect(context, lang, minutes, pt, en):
    lg, o, calls, clock = context
    o._reply_lang = lang
    _record(lg, minutes)
    before = Path(lg.path).read_bytes()
    assert o._handle_undo() == (_too_old(lang, pt if lang == "pt" else en), "UNDO_TOO_OLD")
    assert calls == []
    assert Path(lg.path).read_bytes() == before
    assert clock.call_count == 1


def test_second_undo_stops_at_old_event(context):
    lg, o, calls, _ = context
    _record(lg, 120)
    _record(lg, 5, "create_reminder")
    assert o._handle_undo()[1] == "UNDO_DONE"
    before = Path(lg.path).read_bytes()
    assert o._handle_undo() == (_too_old("pt", "2 horas"), "UNDO_TOO_OLD")
    assert len(calls) == 1 and isinstance(calls[0], dict)
    assert Path(lg.path).read_bytes() == before


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("minutes", [120, 5])
def test_window_precedes_irreversible_check(context, lang, minutes):
    lg, o, calls, _ = context
    o._reply_lang = lang
    _record(lg, minutes, "send_email")
    before = Path(lg.path).read_bytes()
    if minutes == 120:
        expected = (_too_old(lang, "2 horas" if lang == "pt" else "2 hours", "send_email"),
                    "UNDO_TOO_OLD")
    else:
        expected = (("A última coisa que fiz foi enviar um e-mail, e isso não dá pra desfazer."
                     if lang == "pt" else
                     "The last thing I did was send an email, and that can't be undone."),
                    "UNDO_IRREVERSIBLE")
    assert o._handle_undo() == expected
    assert calls == []
    assert Path(lg.path).read_bytes() == before


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("ts", [None, "ontem", 123])
def test_undated_action_is_not_undone(context, lang, ts):
    lg, o, calls, _ = context
    o._reply_lang = lang
    _record(lg, 10)
    entry = lg.load_entries()[0]
    entry.pop("ts")
    if ts is not None:
        entry["ts"] = ts
    Path(lg.path).write_text(json.dumps(entry) + "\n", encoding="utf-8")
    before = Path(lg.path).read_bytes()
    expected = ("Não há nada recente pra desfazer." if lang == "pt" else
                "There's nothing recent to undo.")
    assert o._handle_undo() == (expected, "UNDO_NONE")
    assert calls == []
    assert Path(lg.path).read_bytes() == before


@pytest.mark.parametrize("zone", [None, timezone(timedelta(hours=5))])
def test_naive_is_utc_and_offsets_are_respected(context, zone):
    lg, o, calls, _ = context
    ts = NOW - timedelta(minutes=10)
    ts = ts.replace(tzinfo=None) if zone is None else ts.astimezone(zone)
    lg.record("create_event", {}, result_id="id1", extra={"ts": ts.isoformat()})
    assert o._handle_undo()[1] == "UNDO_DONE"
    assert calls == ["id1"]


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_unknown_action_uses_action_name(context, lang):
    lg, o, calls, _ = context
    o._reply_lang = lang
    _record(lg, 120, "unknown_action")
    assert o._handle_undo() == (
        _too_old(lang, "2 horas" if lang == "pt" else "2 hours", "unknown_action"),
        "UNDO_TOO_OLD")
    assert calls == []
    assert len(lg.load_entries()) == 1
