"""Controlled SQLite write interleavings, without Ollama or live state."""
import logging
import sqlite3
import threading

import pytest

from src.memory.palace import Palace


INSERT_FACT = """
    INSERT INTO triplas (sujeito, predicado, objeto, valid_from, created_at)
    VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
"""


@pytest.fixture
def palace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = Palace(db_path=str(tmp_path / "palace.db"))
    yield p
    p.close()


class _PausedConnection:
    def __init__(self, conn, first_updated, second_attempted, second_updated,
                 first_done):
        self._conn = conn
        self.first_updated = first_updated
        self.second_attempted = second_attempted
        self.second_updated = second_updated
        self.first_done = first_done
        self.overlapped = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *args):
        return self._conn.__exit__(*args)

    def execute(self, sql, params=()):
        already_in_transaction = self._conn.in_transaction
        cur = self._conn.execute(sql, params)
        if sql.lstrip().startswith("UPDATE triplas SET valid_until"):
            if threading.current_thread().name == "writer-A":
                self.first_updated.set()
                assert self.second_attempted.wait(2)
                # The lock (or SQLite on another connection) blocks B here.
                self.second_updated.wait(0.5)
            else:
                self.overlapped = already_in_transaction
                self.second_updated.set()
                assert self.first_done.wait(2)
        return cur


@pytest.mark.parametrize("separate_connections,invalidate", [
    (False, False), (False, True), (True, False),
], ids=["two-adds-one-connection", "invalidate-and-add", "two-connections"])
def test_writes_are_serialised(palace, separate_connections, invalidate):
    other = Palace(db_path=palace.db_path) if separate_connections else palace
    if invalidate:
        palace.add_fact("pedro", "cidade", "Porto")
    events = [threading.Event() for _ in range(4)]
    first_updated, second_attempted, second_updated, first_done = events
    palace._conn = _PausedConnection(palace._conn, *events)
    if separate_connections:
        other._conn = _PausedConnection(other._conn, *events)
    errors, finished = [], []

    def write(first):
        try:
            if not first:
                assert first_updated.wait(2)
                second_attempted.set()
            if first and invalidate:
                palace.invalidate_fact("pedro", "cidade")
            else:
                target = palace if first else other
                obj = "Londres" if first else "Lisboa"
                target.add_fact("pedro", "cidade", obj)
                finished.append(obj)
        except Exception as exc:
            errors.append(exc)
        finally:
            if first:
                first_done.set()

    threads = [threading.Thread(target=write, args=(first,),
                                name=name, daemon=True)
               for first, name in [(True, "writer-A"), (False, "writer-B")]]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=7)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors
        active = palace.list_facts("pedro")
        assert len(active) == 1
        assert palace.get_fact("pedro", "cidade")["objeto"] == finished[-1]
        assert not other._conn.overlapped
        history = palace.list_facts("pedro", include_invalidated=True)
        assert len(history) == 2
        assert sum(f["valid_until"] is not None for f in history) == 1
    finally:
        if separate_connections:
            other.close()


def test_unique_active_index_and_reassertion(palace):
    indexes = {r["name"]: r for r in
               palace._conn.execute("PRAGMA index_list(triplas)")}
    assert indexes["uq_triplas_active"]["unique"] == 1
    assert indexes["uq_triplas_active"]["partial"] == 1
    assert "idx_triplas_active" in indexes
    palace.add_fact("pedro", "cidade", "Londres")
    with pytest.raises(sqlite3.IntegrityError), palace._conn:
        palace._conn.execute(INSERT_FACT, ("pedro", "cidade", "Porto"))
    palace.add_fact("pedro", "cidade", "Lisboa")
    assert palace.get_fact("pedro", "cidade")["objeto"] == "Lisboa"
    assert len(palace.list_facts("pedro")) == 1
    assert len(palace.list_facts("pedro", include_invalidated=True)) == 2
    palace.add_fact("pedro", "trabalho", "Lisboa")
    palace.add_fact("ana", "cidade", "Lisboa")
    assert palace.counts()["triplas_active"] == 3


def test_existing_duplicates_are_preserved(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE triplas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sujeito TEXT NOT NULL, predicado TEXT NOT NULL, objeto TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user_explicit',
            confidence REAL NOT NULL DEFAULT 1.0,
            valid_from TEXT NOT NULL, valid_until TEXT, created_at TEXT NOT NULL
        )
    """)
    conn.executemany(INSERT_FACT, [
        ("pedro", "cidade", obj) for obj in ["Londres", "Porto", "Lisboa"]
    ] + [("ana", "cidade", obj) for obj in ["Recife", "Natal"]])
    conn.execute(INSERT_FACT, ("pedro", "cidade", "Paris"))
    conn.execute("UPDATE triplas SET valid_until = '2026-02-01' WHERE id = 6")
    conn.commit()
    before = conn.execute("SELECT * FROM triplas ORDER BY id").fetchall()
    conn.close()

    with caplog.at_level(logging.WARNING, logger="src.memory.palace"):
        p = Palace(db_path=path)
    try:
        after = p._conn.execute("SELECT * FROM triplas ORDER BY id").fetchall()
        assert [tuple(row) for row in after] == before
        assert not p.execute_read(
            "SELECT 1 FROM sqlite_master WHERE name = 'uq_triplas_active'")
        warnings = [r.getMessage() for r in caplog.records
                    if r.name == "src.memory.palace" and r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "uq_triplas_active" in warnings[0]
        assert "2 duplicate active fact pairs" in warnings[0]
        assert all(value not in warnings[0] for row in before for value in row[1:4])
        p.add_fact("pedro", "cidade", "Braga")
        assert p.get_fact("pedro", "cidade")["objeto"] == "Braga"
        assert len(p.list_facts("pedro")) == 1
        assert len(p.list_facts("pedro", include_invalidated=True)) == 5
        assert len(p.list_facts("ana")) == 2
    finally:
        p.close()
