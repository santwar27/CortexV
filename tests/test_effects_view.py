"""Screen acknowledgement is durable display state, never permission."""
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import effect_store
from src.core.effect_runner import run_effect
from tests.test_effect_reconcile import ITEMS, boot, child, crash, no_integrations
from tests.test_ui_routes import client, ui_app

AUTH = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def view(client, ui_app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # The reused UI fixture changes cwd; the real child must import this repo.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    return store


def read(client, **params):
    response = client.get("/ui/effects", headers=AUTH, params=params)
    assert response.status_code == 200
    return response.json()


def seen(client, ids):
    response = client.post("/ui/effects/seen", headers=AUTH, json={"ids": ids})
    assert response.status_code == 200
    return response.json()


def snapshot(store):
    with sqlite3.connect(store.db_path) as conn:
        return [conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in ("effects", "transitions")]


def terminal(store, state="desconhecido", args=None, detail=None):
    e = store.create(*ITEMS[0] if args is None else ("send_email", args),
                     origem={"channel": "ui", "conversation": "example"})
    if state == "cancelado":
        assert store.transition(e.id, "preparando", state)
    else:
        assert store.transition(e.id, "preparando", "executando")
        if state != "executando":
            assert store.transition(e.id, "executando", state, detail)
    return e.id


def test_read_only_and_live_dead_owner(view, client, tmp_path, monkeypatch):
    crash(view.db_path)
    boot(tmp_path, monkeypatch, view)
    unknown = view.in_states(["desconhecido"])[0].id
    done = terminal(view, "feito")
    p = child(view.db_path)
    try:
        assert p.stdout.readline().strip() == "ready"
        [live] = view.in_states(["executando"])
        lock = Path(view._owner_path(live.owner))
        initial = read(client)
        assert next(e for e in initial["effects"] if e["id"] == live.id)["owner_alive"] is True
        assert live.id not in [e["id"] for e in initial["notices"]]
        p.kill()
        p.communicate(timeout=10)
        # A separate live child and an unreconciled dead lock coexist on reads.
        p = child(view.db_path)
        assert p.stdout.readline().strip() == "ready"
        before, lock_before = snapshot(view), lock.stat()
        first = read(client)
        assert read(client) == first
        assert snapshot(view) == before and lock.stat() == lock_before
        items = {e["id"]: e for e in first["effects"]}
        assert items[live.id]["owner_alive"] is False
        assert items[live.id]["state"] == view.get(live.id).state == "executando"
        running = [e for e in first["effects"] if e["owner_alive"] is True]
        assert len(running) == 1 and running[0]["id"] != live.id
        assert [e["id"] for e in first["notices"]] == [unknown]
        assert items[done]["owner_alive"] is None
        boot(tmp_path, monkeypatch, view)
        notice = next(e for e in read(client)["notices"] if e["id"] == live.id)
        assert notice["restart"] and notice["chat_notice_pending"]
    finally:
        if p.poll() is None:
            p.kill()
        p.communicate(timeout=10)


@pytest.mark.parametrize("lost", [False, True])
def test_boots_and_lost_chat_delivery(view, client, tmp_path, monkeypatch, lost):
    crash(view.db_path)
    replies = []
    for _ in range(3):
        o = boot(tmp_path, monkeypatch, view)
        if lost:
            replies.append(o._maybe_announce_abandoned("ok", "OFFLINE"))
        [notice] = read(client)["notices"]
        assert notice["seen_at"] is None
        assert notice["chat_notice_pending"] is (not lost)
    before = snapshot(view)
    assert seen(client, [notice["id"]]) == {"seen": [notice["id"]]}
    after = snapshot(view)
    assert after[1] == before[1]  # No history, version or chat-notice mutation.
    assert after[0][0][:-1] == before[0][0][:-1]
    stamp = view.get(notice["id"]).seen_at
    assert stamp.utcoffset().total_seconds() == 0
    assert seen(client, [notice["id"]]) == {"seen": []}
    o = boot(tmp_path, monkeypatch, view)
    replies += [o._maybe_announce_abandoned("ok", "OFFLINE") for _ in range(2)]
    assert read(client)["notices"] == []
    assert view.get(notice["id"]).seen_at == stamp
    assert sum("ana@example.com" in r for r in replies) == 1


def test_concurrent_seen_and_independent_chat_claim(view):
    eid = terminal(view)
    other = effect_store.EffectStore(view.db_path)
    barrier = threading.Barrier(2)

    def mark(store):
        barrier.wait()
        result = store.mark_seen([eid])
        return result, store.get(eid).seen_at

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(mark, (view, other)))
    assert sorted(len(r[0]) for r in results) == [0, 1]
    assert [r[0] for r in results if r[0]] == [[eid]]
    assert results[0][1] == results[1][1] is not None
    eid = terminal(view)
    assert view.transition(eid, "desconhecido", "falhou")
    crash(view.db_path)
    [abandoned] = view.reconcile_abandoned()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(view.claim_notices, [abandoned.id]),
                   pool.submit(other.mark_seen, [abandoned.id])]
        assert [f.result() for f in futures] == [[abandoned.id], [abandoned.id]]


def test_only_unknown_and_runner_redaction(view, client):
    ids = [terminal(view, state) for state in ("executando", "feito", "cancelado")]
    assert view.mark_seen([*ids, "missing"]) == []
    assert all(view.get(eid).seen_at is None for eid in ids)
    args = {"to": "ana@example.com", "api_key": "sk-" + "x" * 32,
            "body": "minha senha é abc123xyz"}
    call = Mock(side_effect=TimeoutError("token: tok987654"))
    run_effect("send_email", args, call, store=view)
    call.assert_called_once()
    response = read(client)
    [notice] = response["notices"]
    assert not notice["restart"] and not notice["chat_notice_pending"]
    rendered = json.dumps(response)
    assert all(secret not in rendered for secret in (args["api_key"], "abc123xyz", "tok987654"))
    assert "[REDACTED]" in rendered and "ana@example.com" in rendered
    assert notice["history"][-1]["detail"] == notice["detail"] == "token: [REDACTED]"
    assert [h["to"] for h in notice["history"]] == ["preparando", "executando", "desconhecido"]
    assert [h["from"] for h in notice["history"]] == [None, "preparando", "executando"]
    for stamp in (notice["created_at"], notice["updated_at"], *(h["at"] for h in notice["history"])):
        assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0
    assert view.get(notice["id"]).args == args
    assert view.history(notice["id"])[-1].detail == "token: tok987654"


def test_seen_is_not_permission(view, client, ui_app, tmp_path, monkeypatch):
    o = boot(tmp_path, monkeypatch, view)
    ui_app.state.orchestrator = o
    o.set_turn_origin("ui", "example")
    o._pending_action = {"type": "send_email", "args": ITEMS[0][1]}
    o._persist_pending_action()
    path = Path(o._PENDING_ACTION_FILE)
    before = path.read_bytes()
    eid = terminal(view)
    audit = Mock()
    monkeypatch.setattr("src.ui.routes.audit.log", audit)
    assert seen(client, [eid]) == {"seen": [eid]}
    assert path.read_bytes() == before
    audit.assert_called_once_with("–", "effects", "seen", payload={"ids": [eid]})


def test_edges_order_labels_and_unavailable(view, client, ui_app, tmp_path, monkeypatch):
    assert client.get("/ui/effects").status_code == 401
    assert client.post("/ui/effects/seen", json={"ids": ["x"]}).status_code == 401
    for ids in ([], ["x"] * 51, ["x" * 65], [1]):
        assert client.post("/ui/effects/seen", headers=AUTH, json={"ids": ids}).status_code == 422
    ids = [terminal(view) for _ in range(201)]
    assert [e["id"] for e in read(client, limit=0)["effects"]] == ids[-1:]
    result = read(client, limit=999)
    assert [e["id"] for e in result["effects"]] == ids[:0:-1]
    assert [e["id"] for e in result["notices"]] == ids[:200]
    o = boot(tmp_path, monkeypatch, view)
    ui_app.state.orchestrator = o
    for lang, labels in (("pt", o._UNDO_ACTION_LABELS), ("en", o._UNDO_ACTION_LABELS_EN),
                         ("invalid", o._UNDO_ACTION_LABELS)):
        assert read(client, limit=1, lang=lang)["effects"][0]["label"] == labels["send_email"]
    unknown = view.create("unmapped_action", {})
    assert read(client, limit=1)["effects"][0]["label"] == unknown.action
    del ui_app.state.orchestrator
    assert read(client, limit=2)["effects"][1]["label"] == "send_email"
    assert view.owner_alive(view._owner_token) is True
    assert view.owner_alive(None) is False and view.owner_alive("missing") is False
    assert not Path(view._owner_path("missing")).exists()
    blocked = tmp_path / "blocked"
    blocked.write_text("x")
    monkeypatch.setattr(effect_store, "store", effect_store.EffectStore(str(blocked / "effects.db")))
    assert read(client) == {"available": False, "effects": [], "notices": []}
    assert seen(client, [ids[0]]) == {"seen": []}
