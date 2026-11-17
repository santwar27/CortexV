"""A terminal reconciliation and repeated boots cannot consume an unseen notice."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.core.effect_store import EffectStore
from tests.test_effect_reconcile import boot, crash


@pytest.mark.parametrize("restart", [False, True])
def test_terminal_reconciliation_survives_boots(tmp_path, monkeypatch, restart):
    db = tmp_path / "effects.db"
    crash(db)
    terminal = EffectStore(str(db))
    [effect] = terminal.reconcile_abandoned()
    assert effect.notice_pending == 1
    assert terminal.history(effect.id)[-1].to_state == "desconhecido"
    first = boot(tmp_path, monkeypatch, EffectStore(str(db)))
    if restart:
        first = boot(tmp_path, monkeypatch, EffectStore(str(db)))
    assert "ana@example.com" in first._maybe_announce_abandoned("ok", "OFFLINE")
    assert first._maybe_announce_abandoned("ok", "OFFLINE") == "ok"
    third = boot(tmp_path, monkeypatch, EffectStore(str(db)))
    assert third._maybe_announce_abandoned("ok", "OFFLINE") == "ok"
    assert terminal.pending_notices() == []


def test_two_orchestrators_claim_once(tmp_path, monkeypatch):
    db = tmp_path / "effects.db"
    crash(db)
    first = boot(tmp_path, monkeypatch, EffectStore(str(db)))
    second = boot(tmp_path, monkeypatch, EffectStore(str(db)))
    assert first._abandoned_effects and second._abandoned_effects
    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(lambda o: o._maybe_announce_abandoned("ok", "OFFLINE"),
                                [first, second]))
    assert sum("ana@example.com" in reply for reply in replies) == 1


def test_migration_and_claim_subset(tmp_path):
    db = tmp_path / "effects.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE effects (id TEXT PRIMARY KEY, action TEXT, args TEXT, "
                     "origin TEXT, state TEXT, version INTEGER, detail TEXT, created_at TEXT, "
                     "updated_at TEXT, owner TEXT)")
        for gid in ["old", "next"]:
            conn.execute("INSERT INTO effects VALUES (?, 'mac_sleep', '{}', NULL, 'executando', "
                         "2, NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', NULL)",
                         (gid,))
    store = EffectStore(str(db))
    assert store.get("old").notice_pending == 0
    assert len(store.reconcile_abandoned()) == 2
    assert [e.id for e in store.pending_notices(limite=1)] == ["old"]
    other = EffectStore(str(db))
    assert other.claim_notices(["next", "missing"]) == ["next"]
    assert store.claim_notices(["old", "next"]) == ["old"]
    assert store.claim_notices([]) == []
    assert store.pending_notices() == []


def test_notices_fail_soft(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("x")
    store = EffectStore(str(blocked / "effects.db"))
    assert store.pending_notices() == []
    assert store.claim_notices(["id"]) == []
