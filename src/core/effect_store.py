"""Durable effect intentions; recording never grants permission or executes."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from src.core.audit import chmod_600

log = logging.getLogger(__name__)
RESTART_DETAIL = "processo encerrado no meio da execução"
_NEXT = {"preparando": {"esperando", "executando", "cancelado", "falhou"},
         "esperando": {"executando", "cancelado"},
         "executando": {"feito", "falhou", "desconhecido"},
         "desconhecido": {"feito", "falhou"}}


@dataclass(frozen=True)
class Effect:
    id: str
    action: str
    args: dict
    origin: Optional[dict]
    state: str
    version: int
    detail: Optional[str]
    created_at: datetime
    updated_at: datetime
    owner: Optional[str] = None
    notice_pending: int = 0
    seen_at: Optional[datetime] = None


@dataclass(frozen=True)
class Transition:
    effect_id: str
    from_state: Optional[str]
    to_state: str
    version: int
    detail: Optional[str]
    created_at: datetime


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record(cls, row):
    values = dict(row)
    for key in ("created_at", "updated_at", "seen_at"):
        if values.get(key) is not None:
            values[key] = datetime.fromisoformat(values[key])
    if cls is Effect:
        values["args"] = json.loads(values["args"])
        values["origin"] = json.loads(values["origin"]) if values["origin"] is not None else None
    return cls(**values)


class EffectStore:
    """Lazy WAL storage. Transactions pair each conditional claim with history."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/effects.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_EFFECTS_DB", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._owner_token = uuid4().hex
        self._owner_fd: Optional[int] = None

    def _owner_path(self, token):
        # Process state deliberately lives outside the backup's file whitelist.
        return os.path.join(os.path.dirname(self.db_path), "effect_owners", token + ".lock")

    def _ensure_owner(self):
        if self._owner_fd is not None:
            return
        path = self._owner_path(self._owner_token)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            os.close(fd)
            raise
        # Keep the descriptor until process exit; PIDs can be reused.
        self._owner_fd = fd

    def _owner_dead(self, token):
        if token is None:
            return True
        path = self._owner_path(token)
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return True
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass  # Another reconciler may have removed the same dead owner's file.
        return True

    def owner_alive(self, token) -> bool:
        if token == self._owner_token:
            return True
        if token is None:
            return False
        try:
            fd = os.open(self._owner_path(token), os.O_RDONLY)
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError:
            return False
        finally:
            os.close(fd)

    def _secure(self):
        # Payloads include full email bodies, including in SQLite's sidecars.
        if self.db_path != ":memory:":
            for suffix in ("", "-wal", "-shm"):
                chmod_600(self.db_path + suffix)

    def _connect(self):
        if self._conn is not None:
            return self._conn
        conn = None
        try:
            parent = os.path.dirname(self.db_path)
            if parent and self.db_path != ":memory:":
                os.makedirs(parent, exist_ok=True)
            self._secure()
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._secure()  # New WAL files inherit the database's permissions.
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""CREATE TABLE IF NOT EXISTS effects (
                    id TEXT PRIMARY KEY NOT NULL, action TEXT NOT NULL,
                    args TEXT NOT NULL, origin TEXT, state TEXT NOT NULL,
                    version INTEGER NOT NULL, detail TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, owner TEXT)""")
                if "owner" not in {row[1] for row in conn.execute("PRAGMA table_info(effects)")}:
                    conn.execute("ALTER TABLE effects ADD COLUMN owner TEXT")
                if "notice_pending" not in {row[1] for row in conn.execute("PRAGMA table_info(effects)")}:
                    conn.execute("ALTER TABLE effects ADD COLUMN notice_pending INTEGER NOT NULL DEFAULT 0")
                if "seen_at" not in {row[1] for row in conn.execute("PRAGMA table_info(effects)")}:
                    conn.execute("ALTER TABLE effects ADD COLUMN seen_at TEXT")
                conn.execute("""CREATE TABLE IF NOT EXISTS transitions (
                    effect_id TEXT NOT NULL REFERENCES effects(id),
                    from_state TEXT, to_state TEXT NOT NULL, version INTEGER NOT NULL,
                    detail TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (effect_id, version))""")
            self._conn = conn
            return conn
        except Exception:
            if conn is not None:
                conn.close()
            log.exception("effects: could not open the record at %s", self.db_path)
            return None
        finally:
            self._secure()

    def _run(self, action, default=None, write=False):
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return default
                with conn:
                    if write:
                        conn.execute("BEGIN IMMEDIATE")
                    return action(conn)
        except Exception:
            log.exception("effects: record operation failed")
            return default
        finally:
            self._secure()

    def _read(self, sql, args=(), cls=Effect, many=False):
        def read(conn):
            values = [_record(cls, row) for row in conn.execute(sql, args).fetchall()]
            return values if many else (values[0] if values else None)
        return self._run(read, [] if many else None)

    def available(self) -> bool:
        return self._run(lambda c: bool(c.execute("SELECT 1").fetchone()), False)

    def create(self, acao, args, origem=None, effect_id=None) -> Optional[Effect]:
        def create(conn):
            self._ensure_owner()
            payload = json.dumps(args, ensure_ascii=False, allow_nan=False)
            origin = json.dumps(origem, ensure_ascii=False, allow_nan=False) if origem is not None else None
            eid, now = effect_id if effect_id is not None else uuid4().hex, _now()
            conn.execute("INSERT INTO effects (id, action, args, origin, state, version, detail, "
                         "created_at, updated_at, owner) "
                         "VALUES (?, ?, ?, ?, 'preparando', 1, NULL, ?, ?, ?)",
                         (eid, acao, payload, origin, now, now, self._owner_token))
            conn.execute("INSERT INTO transitions VALUES (?, NULL, 'preparando', 1, NULL, ?)",
                         (eid, now))
            return _record(Effect, conn.execute("SELECT * FROM effects WHERE id = ?", (eid,)).fetchone())
        return self._run(create, write=True)

    def reconcile_abandoned(self) -> list[Effect]:
        """Only dead owners can lose their claims; uncertainty never retries."""
        try:
            effects = self._read("SELECT * FROM effects WHERE state IN ('preparando', 'executando') "
                                 "ORDER BY created_at, rowid", many=True)
            unknown = []
            for effect in effects:
                if effect.owner == self._owner_token or not self._owner_dead(effect.owner):
                    continue
                executing = effect.state == "executando"
                target = "desconhecido" if executing else "cancelado"
                detail = (RESTART_DETAIL if executing else
                          "processo encerrado antes de executar")
                if self.transition(effect.id, effect.state, target, detail,
                                   notice_pending=executing) and executing:
                    updated = self.get(effect.id)
                    if updated is not None:
                        unknown.append(updated)
            return unknown
        except Exception:
            log.exception("effects: could not reconcile abandoned intentions")
            return []

    def pending_notices(self, limite=20) -> list[Effect]:
        return self._read("SELECT * FROM effects WHERE state = 'desconhecido' AND notice_pending = 1 "
                          "ORDER BY created_at, rowid LIMIT ?", (max(0, limite),), many=True)

    def recent(self, limite=50) -> list[Effect]:
        return self._read("SELECT * FROM effects ORDER BY created_at DESC, rowid DESC LIMIT ?",
                          (max(0, limite),), many=True)

    def unseen_unknown(self, limite=50) -> list[Effect]:
        return self._read("SELECT * FROM effects WHERE state = 'desconhecido' AND seen_at IS NULL "
                          "ORDER BY created_at, rowid LIMIT ?", (max(0, limite),), many=True)

    def unknown_for_conversation(self, conversation_id, limite=50) -> list[Effect]:
        # Keep old, acknowledged uncertainty linked to either saved reply.
        return self._read(
            "SELECT * FROM effects WHERE state = 'desconhecido' AND ("
            "(json_extract(origin, '$.channel') = 'ui' "
            "AND json_extract(origin, '$.conversation') = ? "
            "AND json_type(origin, '$.conversation') = 'text' "
            "AND json_extract(origin, '$.conversation') != '' "
            "AND json_type(origin, '$.turn') = 'text' AND json_extract(origin, '$.turn') != '') OR "
            "(json_extract(origin, '$.asked.conversation') = ? "
            "AND json_type(origin, '$.asked.conversation') = 'text' "
            "AND json_extract(origin, '$.asked.conversation') != '' "
            "AND json_type(origin, '$.asked.turn') = 'text' "
            "AND json_extract(origin, '$.asked.turn') != '')) "
            "ORDER BY created_at, rowid LIMIT ?",
            (conversation_id, conversation_id, max(0, limite)), many=True)

    def mark_seen(self, ids) -> list[str]:
        ids = tuple(ids)
        if not ids:
            return []
        def mark(conn):
            # Display acknowledgement is independent of state/history and chat claims.
            rows = conn.execute(
                f"UPDATE effects SET seen_at = ? WHERE id IN ({','.join('?' for _ in ids)}) "
                "AND state = 'desconhecido' AND seen_at IS NULL RETURNING id",
                (_now(), *ids)).fetchall()
            return [row["id"] for row in rows]
        return self._run(mark, [], write=True)

    def claim_notices(self, ids) -> list[str]:
        ids = tuple(ids)
        if not ids:
            return []
        def claim(conn):
            # RETURNING identifies only this transaction's claims, across processes.
            rows = conn.execute(
                f"UPDATE effects SET notice_pending = 0 WHERE id IN ({','.join('?' for _ in ids)}) "
                "AND notice_pending = 1 RETURNING id", ids).fetchall()
            return [row["id"] for row in rows]
        return self._run(claim, [], write=True)

    def transition(self, effect_id, de, para, detalhe=None, *, notice_pending=False) -> bool:
        if para not in _NEXT.get(de, ()):
            return False
        def transition(conn):
            now, detail = _now(), detalhe[:500] if detalhe is not None else None
            changed = conn.execute(
                "UPDATE effects SET state = ?, version = version + 1, detail = ?, updated_at = ?, "
                "notice_pending = CASE WHEN ? THEN 1 ELSE notice_pending END "
                "WHERE id = ? AND state = ?", (para, detail, now, notice_pending, effect_id, de)).rowcount
            if changed != 1:
                return False
            # The reservation covers the version read and history, across processes.
            conn.execute("INSERT INTO transitions SELECT id, ?, state, version, detail, updated_at "
                         "FROM effects WHERE id = ?", (de, effect_id))
            return True
        return self._run(transition, False, write=True)

    def get(self, effect_id) -> Optional[Effect]:
        return self._read("SELECT * FROM effects WHERE id = ?", (effect_id,))

    def history(self, effect_id) -> list[Transition]:
        return self._read("SELECT * FROM transitions WHERE effect_id = ? ORDER BY version",
                          (effect_id,), Transition, True)

    def in_states(self, estados, limite=100) -> list[Effect]:
        def read(conn):
            states = tuple(estados)
            sql = (f"SELECT * FROM effects WHERE state IN ({','.join('?' for _ in states)}) "
                   "ORDER BY created_at, rowid LIMIT ?")
            return [_record(Effect, row) for row in conn.execute(sql, (*states, max(0, limite)))]
        return self._run(read, [])


store = EffectStore()
