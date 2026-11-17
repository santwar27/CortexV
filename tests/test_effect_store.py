"""Durable intent is storage only; competing processes get one claim."""
import sqlite3
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest

from src.core.effect_store import EffectStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CORTEX_EFFECTS_DB", str(tmp_path / ".cortex/effects.db"))
    result = EffectStore()
    assert result._conn is None
    return result


def test_create_is_literal_unique_and_detached(store):
    args = {"body": "Olá, ação fictícia!\n" * 1000, "nested": ["original"]}
    origin = {"channel": "ui", "conversation": "c1"}
    effect = store.create("send_email", args, origin, effect_id="staged-1")
    assert (effect.id, effect.action, effect.state, effect.version, effect.detail) == (
        "staged-1", "send_email", "preparando", 1, None)
    assert effect.args == args and effect.origin == origin
    args["nested"].append("changed")
    origin["channel"] = "changed"
    assert store.get(effect.id) == effect
    effect.args["nested"].append("local change")
    assert store.get(effect.id).args["nested"] == ["original"]
    original = store.get(effect.id)
    assert store.create("other", {}, effect_id=effect.id) is None
    assert store.get(effect.id) == original
    assert store.create("send_email", {"bad": object()}) is None
    assert store.create("send_email", {}, {"bad": object()}) is None
    generated = store.create("reminder", {})
    assert UUID(generated.id).version == 4 and len(generated.id) == 32
    assert generated.origin is None
    [first] = store.history(effect.id)
    assert (first.effect_id, first.from_state, first.to_state, first.version,
            first.detail, first.created_at) == (effect.id, None, "preparando", 1,
                                               None, effect.created_at)
    for record, field in ((effect, "state"), (first, "to_state")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, "feito")


def test_happy_path_versions_history_and_utc(store):
    effect = store.create("send_email", {})
    states = ["preparando", "esperando", "executando", "feito"]
    for version, (old, new) in enumerate(zip(states, states[1:]), 2):
        assert store.transition(effect.id, old, new, "é" * 600)
        current = store.get(effect.id)
        assert (current.state, current.version, current.detail) == (new, version, "é" * 500)
        assert current.created_at == effect.created_at
    history = store.history(effect.id)
    assert [t.from_state for t in history] == [None] + states[:-1]
    assert [t.to_state for t in history] == states
    assert [t.version for t in history] == [1, 2, 3, 4]
    assert [t.detail for t in history] == [None] + ["é" * 500] * 3
    times = [t.created_at for t in history]
    assert times == sorted(times) and current.updated_at == times[-1]
    assert all(t.utcoffset() == timedelta(0) for t in times + [current.updated_at])


PATHS = {"preparando": [], "esperando": ["esperando"], "executando": ["executando"],
         "desconhecido": ["executando", "desconhecido"], "feito": ["executando", "feito"],
         "falhou": ["falhou"], "cancelado": ["cancelado"]}
ALLOWED = {"preparando": {"esperando", "executando", "cancelado", "falhou"},
           "esperando": {"executando", "cancelado"},
           "executando": {"feito", "falhou", "desconhecido"},
           "desconhecido": {"feito", "falhou"}}


@pytest.mark.parametrize("source", PATHS)
def test_every_state_pair_and_failed_condition(store, source):
    for target in [*PATHS, "invalid"]:
        effect = store.create("reminder", {})
        previous = "preparando"
        for state in PATHS[source]:
            assert store.transition(effect.id, previous, state)
            previous = state
        before, history = store.get(effect.id), store.history(effect.id)
        assert not store.transition(effect.id, "wrong", target)
        assert not store.transition("missing", source, target)
        assert store.get(effect.id) == before and store.history(effect.id) == history
        permitted = target in ALLOWED.get(source, set())
        assert store.transition(effect.id, source, target) is permitted
        if permitted:
            assert store.get(effect.id).version == before.version + 1
            assert store.history(effect.id)[-1].from_state == source
        else:
            assert store.get(effect.id) == before and store.history(effect.id) == history


def test_processes_compete_for_one_conditional_transition(store):
    effect = store.create("send_email", {})
    assert store.transition(effect.id, "preparando", "esperando")
    code = """
import sys
from src.core.effect_store import EffectStore
s = EffectStore(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
print(s.transition(sys.argv[2], 'esperando', 'executando'), flush=True)
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, store.db_path, effect.id],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        assert all(p.stdout.readline().strip() == "ready" for p in processes)
        for p in processes:
            p.stdin.write("go\n")
            p.stdin.flush()
        results = [p.communicate(timeout=20) for p in processes]
        assert all(p.returncode == 0 for p in processes), results
        assert sorted(out.strip() for out, _ in results) == ["False"] * 3 + ["True"]
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
            p.wait()
    assert [t.to_state for t in store.history(effect.id)] == [
        "preparando", "esperando", "executando"]


def test_connections_reopen_filter_limit_and_private_wal(store):
    peer = EffectStore(store.db_path)
    effects = [store.create("reminder", {}) for _ in range(4)]
    for effect in effects[:3]:
        assert peer.transition(effect.id, "preparando", "executando")
    assert store.transition(effects[1].id, "executando", "desconhecido")
    reopened = EffectStore(store.db_path)
    expected = [peer.get(e.id) for e in effects[:3]]
    assert reopened.in_states(["executando", "desconhecido"]) == expected
    assert reopened.in_states(["executando", "desconhecido"], limite=2) == expected[:2]
    assert reopened.in_states([]) == reopened.in_states(["executando"], limite=0) == []
    assert reopened.get(effects[1].id) == store.get(effects[1].id)
    assert reopened.history(effects[1].id) == store.history(effects[1].id)
    assert store.get("missing") is None and store.history("missing") == []
    assert store.available() and peer.available()
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    for suffix in ("", "-wal", "-shm"):
        assert Path(store.db_path + suffix).stat().st_mode & 0o777 == 0o600


def test_fail_soft_and_atomic_history_failure(store, tmp_path, caplog):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    bad = EffectStore(str(blocked / "effects.db"))
    assert bad.create("reminder", {}) is None and bad.get("x") is None
    assert bad.history("x") == bad.in_states(["executando"]) == []
    assert not bad.available() and not bad.transition("x", "preparando", "executando")
    effect = store.create("reminder", {})
    before = store.history(effect.id)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("CREATE TRIGGER fail_history BEFORE INSERT ON transitions "
                     "BEGIN SELECT RAISE(ABORT, 'history unavailable'); END")
    assert not store.transition(effect.id, "preparando", "executando")
    assert store.get(effect.id) == effect and store.history(effect.id) == before
    assert store.create("reminder", {}, effect_id="rolled-back") is None
    assert store.get("rolled-back") is None and store.history("rolled-back") == []
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DROP TRIGGER fail_history")
    assert EffectStore(store.db_path).transition(effect.id, "preparando", "executando")
    assert "history unavailable" in caplog.text and "not-a-directory" in caplog.text
