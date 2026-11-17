"""Background review is readable without granting permission or hiding uncertainty."""
import builtins
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import gate_review
from src.core import effect_store
from src.core.gate_queue import GateQueue
from src.core.orchestrator import CortexOrchestrator
from tests.test_effect_reconcile import crash, no_integrations
from tests.test_ui_routes import client, ui_app

AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def view(client, ui_app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    q = GateQueue(str(tmp_path / "gate.jsonl"))
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    assert store.available()
    monkeypatch.setattr("src.core.gate_queue.gate_queue", q)
    monkeypatch.setattr(effect_store, "store", store)
    ui_app.state.orchestrator = CortexOrchestrator.__new__(CortexOrchestrator)
    blocked = Mock(side_effect=AssertionError("read must never execute or build an orchestrator"))
    for target in ("src.gate_review._default_runner", "src.ui.routes._get_orchestrator",
                   "src.core.orchestrator.CortexOrchestrator.__init__",
                   "src.core.orchestrator.CortexOrchestrator._run_action"):
        monkeypatch.setattr(target, blocked)
    yield q, store
    blocked.assert_not_called()


def read(client, **params):
    response = client.get("/ui/gate", headers=AUTH, params=params)
    assert response.status_code == 200
    return response.json()


def snapshot(store):
    with sqlite3.connect(store.db_path) as conn:
        return [conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in ("effects", "transitions")]


def queued(q, gid, index):
    assert q._append({"id": gid, "event": "queued", "type": "send_email",
                      "args": {"to": "ana@example.com"}, "origin": "scheduled:test",
                      "reason": "fictional", "trace_id": "trace-example",
                      "ts": f"2026-10-20T10:{index:02d}:00+01:00"})


def test_every_outcome_and_dead_owner(view, client):
    q, store = view
    crash(store.db_path)
    [effect] = store.in_states(["executando"])
    states = ["queued", "approved", "rejected", "executed", "failed", "unknown"]
    ids = ["waiting", effect.id, "rejected", "executed", "failed", "unknown"]
    for i, (gid, state) in enumerate(zip(ids, states)):
        queued(q, gid, i)
        if state != "queued":
            assert q.resolve(gid, "rejected" if state == "rejected" else "approved")
        if state in ("executed", "failed"):
            assert q.mark_executed(gid, state == "executed", "token: fictional-secret")
        if state == "unknown":
            assert q.mark_unknown(gid, "token: fictional-secret")
    owner_lock = Path(store._owner_path(effect.owner))
    before = Path(q.path).read_bytes(), snapshot(store), owner_lock.stat()
    result = read(client)
    assert read(client) == result
    assert (Path(q.path).read_bytes(), snapshot(store), owner_lock.stat()) == before
    assert result["available"] is True
    assert [e["id"] for e in result["recent"]] == ids[::-1]
    items = {e["id"]: e for e in result["recent"]}
    assert [items[gid]["status"] for gid in ids] == states
    assert items[effect.id]["effect_state"] == "executando"
    assert items[effect.id]["effect_owner_alive"] is False
    assert all(items[gid]["effect_state"] is None for gid in ids if gid != effect.id)
    assert items["unknown"]["effect_owner_alive"] is None
    assert result["pending"] == [items["waiting"]]
    item = items["executed"]
    assert (item["origin"], item["reason"], item["trace_id"]) == (
        "scheduled:test", "fictional", "trace-example")
    assert [e["event"] for e in item["events"]] == ["queued", "approved", "executed"]
    assert item["events"][-1]["detail"] == "token: [REDACTED]"
    for item in result["recent"]:
        for stamp in [item["queued_at"], *(e["at"] for e in item["events"])]:
            assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0
    assert read(client, limit=0)["recent"] == result["recent"][:1]
    assert store.transition(effect.id, "executando", "desconhecido")
    approved = next(e for e in read(client)["recent"] if e["id"] == effect.id)
    assert (approved["status"], approved["effect_state"], approved["effect_owner_alive"]) == (
        "approved", "desconhecido", None)


def test_limits_and_pending_order(view, client):
    q, _ = view
    ids = [q.enqueue("mac_sleep", {}) for _ in range(201)]
    result = read(client, limit=999)
    assert [e["id"] for e in result["recent"]] == ids[:0:-1]
    assert [e["id"] for e in result["pending"]] == ids
    assert len(read(client)["recent"]) == 50


def test_old_pending_survives_recent_limit(view, client, monkeypatch):
    q, _ = view
    queued(q, "old", 0)
    ids = [f"new-{i}" for i in range(60)]
    for i, gid in enumerate(ids):
        queued(q, gid, i)
        assert q.resolve(gid, "approved")
        assert q.mark_executed(gid, True)
    before = Path(q.path).read_bytes()
    history = Mock(wraps=q.history)
    monkeypatch.setattr(q, "history", history)
    result = read(client, limit=5)
    assert [e["id"] for e in result["recent"]] == ids[-5:][::-1]
    assert [e["id"] for e in result["pending"]] == ["old"]
    history.assert_called_once_with(None)
    assert len(q.history(None)) == 61
    assert len(q.history()) == 50
    assert len(q.history(5)) == 5
    assert q.history(0) == []
    assert Path(q.path).read_bytes() == before


def test_whole_safe_preview_and_labels(view, client, ui_app):
    q, _ = view
    body = "\n".join(f"Fictional paragraph {i}: " + "x" * 100 for i in range(30))
    args = {"to": "ana@example.com", "subject": "Example", "body": body}
    gid = q.enqueue("send_email", args)
    o = ui_app.state.orchestrator
    for lang in ("pt", "en", "invalid"):
        item = read(client, lang=lang)["recent"][0]
        resolved = "en" if lang == "en" else "pt"
        assert item["preview"] == o._human_confirmation_prompt(
            {"type": "send_email", "args": args}, lang=resolved).rsplit("\n", 1)[0]
        assert body in item["preview"]
        labels = o._UNDO_ACTION_LABELS_EN if resolved == "en" else o._UNDO_ACTION_LABELS
        assert item["label"] == labels["send_email"]
    # Bypass enqueue to prove the HTTP border independently redacts persisted secrets.
    queued(q, "unsafe", 59)
    records = [json.loads(line) for line in Path(q.path).read_text().splitlines()]
    records[-1]["args"] = {"api_key": "fictional-secret", "body": "password: fictional-body"}
    Path(q.path).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = read(client)
    assert "fictional-secret" not in json.dumps(result)
    assert "fictional-body" not in json.dumps(result)
    unsafe = next(e for e in result["recent"] if e["id"] == "unsafe")
    assert unsafe["args"]["api_key"] == "[REDACTED]"
    assert unsafe["args"]["body"] == "password: [REDACTED]"
    del ui_app.state.orchestrator
    assert all(e["preview"] is None and e["label"] == e["type"]
               for e in read(client)["recent"])
    assert gid in [e["id"] for e in result["recent"]]


def test_read_only_then_terminal_decides_once(view, client):
    q, store = view
    ids = [q.enqueue("mac_sleep", {}) for _ in range(2)]
    before = Path(q.path).read_bytes(), snapshot(store)
    assert read(client) == read(client)
    assert (Path(q.path).read_bytes(), snapshot(store)) == before
    assert not Path(q.path + ".lock").exists()
    runner = Mock(return_value=("ok", "AGENT_ACTION_DONE"))
    gate_review.approve(ids[0], queue=q, runner=runner)
    gate_review.reject(ids[1], queue=q)
    after = Path(q.path).read_bytes()
    for gid in ids:
        assert "já foi decidido" in gate_review.approve(gid, queue=q, runner=runner)
        assert "já foi decidido" in gate_review.reject(gid, queue=q)
    assert Path(q.path).read_bytes() == after
    runner.assert_called_once_with("mac_sleep", {})
    assert [q.status(gid) for gid in ids] == ["executed", "rejected"]


def test_unavailable_missing_malformed_and_auth(view, client):
    q, _ = view
    assert client.get("/ui/gate").status_code == 401
    assert q.history() == []
    assert read(client) == {"available": True, "pending": [], "recent": []}
    Path(q.path).mkdir()
    assert q.history() is None
    assert q.history(None) is None
    assert read(client) == {"available": False, "pending": [], "recent": []}
    Path(q.path).rmdir()
    Path(q.path).write_text('garbage\n{broken\nnull\n[]\n{"id": []}\n')
    queued(q, "valid", 0)
    assert [e["id"] for e in read(client)["recent"]] == ["valid"]
    Path(q.path).write_bytes(b"\xff")
    assert read(client) == {"available": False, "pending": [], "recent": []}


def test_history_one_read_and_status_precedence(view, monkeypatch):
    q, _ = view
    queued(q, "example", 0)
    for event in ("unknown", "executed", "failed", "rejected", "approved"):
        assert q._append({"id": "example", "event": event})
    expected = q.get("example"), q.status("example")
    opened = Mock(wraps=builtins.open)
    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(q, "status", Mock(side_effect=AssertionError("must fold one snapshot")))
    [item] = q.history()
    opened.assert_called_once_with(q.path, "r", encoding="utf-8")
    assert (item["queued"], item["status"]) == expected
    assert [e["event"] for e in item["events"]] == [
        "queued", "unknown", "executed", "failed", "rejected", "approved"]
