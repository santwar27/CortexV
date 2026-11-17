"""Persistent chat conversations for the /ui/chat page (fatia 2).

The dashboard chat drawer used an in-memory `ConversationBuffer` that dies
on every uvicorn restart — fine for a telemetry mirror, useless as a real
chat history. This is the durable replacement: a small SQLite ledger of
conversations and their turns, so the chat page gets a Claude-style sidebar
of past conversations that survives restarts.

Design mirrors `src/integrations/contacts.py`: a class with an injectable
path plus a module-level singleton, lazy connection, thread-safe, fail-soft
(a missing/corrupt DB degrades to "no conversations" rather than breaking
the page).

**This is NOT the Cortex memory.** The Palace (`data/palace.db`) remains the
single source of long-term memory — every substantive turn still flows
through the engine's `store_interaction` (embeddings + passive fact
extraction), globally, regardless of which UI conversation it came from. So
switching conversations never loses what Cortex knows; the Palace is always
present. This store is only the UI's organised HISTORY — the visual threads,
not the memory. It lives in ~/.cortex/chat.db (env override CORTEX_CHAT_DB),
outside the project tree, so the backup script picks it up and git never
sees it.

Turn shape matches `ConversationBuffer` (id, ts, asked_ts, user, reply,
status_mode, source, usage, latency_ms) so the frontend renders history from
either source identically.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# First user message becomes the conversation title, trimmed to this many
# chars (Claude-style). Full text still lives in the first turn.
_TITLE_MAX = 48
_DEFAULT_TITLE = "Nova conversa"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


class ChatStore:
    """SQLite-backed store of chat conversations + turns. Thread-safe,
    fail-soft (every method swallows DB errors and returns an empty/false
    result so the chat page never 500s on a storage hiccup)."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/chat.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_CHAT_DB", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # connection (lazy — never fail at import time)
    # ------------------------------------------------------------------
    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        try:
            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    created_ts  TEXT NOT NULL,
                    updated_ts  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
                    id               TEXT NOT NULL,
                    conversation_id  TEXT NOT NULL,
                    user             TEXT NOT NULL,
                    reply            TEXT NOT NULL,
                    status_mode      TEXT NOT NULL,
                    source           TEXT NOT NULL,
                    asked_ts         TEXT,
                    ts               TEXT NOT NULL,
                    usage_json       TEXT,
                    latency_ms       INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_turns_conv
                    ON turns(conversation_id, seq);
                """
            )
            self._conn = conn
            return self._conn
        except Exception:
            log.exception("chat_store: could not open ledger at %s", self.db_path)
            return None

    # ------------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------------
    def create_conversation(self, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create an empty conversation and return it. Title defaults to a
        placeholder that append_turn() replaces with the first user message."""
        conn = self._connect()
        if conn is None:
            return None
        cid = _short_id()
        ts = _now()
        clean_title = (title or "").strip() or _DEFAULT_TITLE
        try:
            with self._lock:
                conn.execute(
                    "INSERT INTO conversations (id, title, created_ts, updated_ts) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, clean_title[:_TITLE_MAX], ts, ts),
                )
                conn.commit()
        except Exception:
            log.exception("chat_store: create_conversation failed")
            return None
        return {"id": cid, "title": clean_title[:_TITLE_MAX],
                "created_ts": ts, "updated_ts": ts, "turn_count": 0}

    def find_titled(self, title: str, *, strict: bool = False) -> Optional[str]:
        """Find without creating a conversation; strict reads expose failures."""
        try:
            conn = self._connect()
            if conn is None:
                if strict:
                    raise sqlite3.OperationalError("Chat store unavailable")
                return None
            title = (title or "").strip()[:_TITLE_MAX] or _DEFAULT_TITLE
            with self._lock:
                row = conn.execute(
                    "SELECT id FROM conversations WHERE title = ? "
                    "ORDER BY created_ts LIMIT 1", (title,)).fetchone()
            return row["id"] if row is not None else None
        except Exception:
            log.exception("chat_store: find_titled failed")
            if strict:
                raise
            return None

    def conversation_titled(self, title: str) -> Optional[str]:
        """The id of the conversation with this exact title, created if there
        is none — one write transaction (BEGIN IMMEDIATE) around both, so a
        second process waits instead of looking at the same empty answer:
        the terminal and /ask's first turns of the day made two "⌨️ Terminal"
        conversations (s88, Codex lote 7, reproduced)."""
        conn = self._connect()
        if conn is None:
            return None
        title = (title or "").strip()[:_TITLE_MAX] or _DEFAULT_TITLE
        try:
            with self._lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        "SELECT id FROM conversations WHERE title = ? "
                        "ORDER BY created_ts LIMIT 1", (title,)).fetchone()
                    if row is not None:
                        cid = row["id"]
                    else:
                        cid, ts = _short_id(), _now()
                        conn.execute(
                            "INSERT INTO conversations (id, title, created_ts, updated_ts) "
                            "VALUES (?, ?, ?, ?)", (cid, title, ts, ts))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return cid
        except Exception:
            log.exception("chat_store: conversation_titled failed")
            return None

    def conversation_titled_on(self, title: str, now: datetime) -> Optional[str]:
        """Get or create this title on now's local date, atomically across
        processes. New records use the supplied instant so retries stay on
        the briefing's day even if the write crosses midnight."""
        try:
            day = now.astimezone().date()
            # Convert each local midnight separately: a DST day is not always
            # 24 hours. Stored timestamps and both bounds are UTC ISO strings.
            start = datetime.combine(day, datetime.min.time()).astimezone(timezone.utc)
            end = datetime.combine(day + timedelta(days=1), datetime.min.time()).astimezone(timezone.utc)
            title = (title or "").strip()[:_TITLE_MAX] or _DEFAULT_TITLE
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return None
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        "SELECT id FROM conversations WHERE title = ? "
                        "AND created_ts >= ? AND created_ts < ? "
                        "ORDER BY created_ts LIMIT 1",
                        (title, start.isoformat(), end.isoformat())).fetchone()
                    if row is not None:
                        cid = row["id"]
                    else:
                        cid = _short_id()
                        ts = now.astimezone(timezone.utc).isoformat()
                        conn.execute(
                            "INSERT INTO conversations (id, title, created_ts, updated_ts) "
                            "VALUES (?, ?, ?, ?)", (cid, title, ts, ts))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return cid
        except Exception:
            log.exception("chat_store: conversation_titled_on failed")
            return None

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """All conversations, most-recently-updated first, each with its turn
        count and a short preview of the last user message (for the sidebar)."""
        conn = self._connect()
        if conn is None:
            return []
        limit = max(1, min(int(limit), 500))
        try:
            with self._lock:
                rows = conn.execute(
                    """
                    SELECT c.id, c.title, c.created_ts, c.updated_ts,
                           COUNT(t.seq) AS turn_count,
                           (SELECT user FROM turns
                              WHERE conversation_id = c.id
                              ORDER BY seq DESC LIMIT 1) AS last_user
                    FROM conversations c
                    LEFT JOIN turns t ON t.conversation_id = c.id
                    GROUP BY c.id
                    ORDER BY c.updated_ts DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except Exception:
            log.exception("chat_store: list_conversations failed")
            return []
        return [
            {
                "id": r["id"], "title": r["title"],
                "created_ts": r["created_ts"], "updated_ts": r["updated_ts"],
                "turn_count": r["turn_count"],
                "preview": (r["last_user"] or "")[:80],
            }
            for r in rows
        ]

    def get_conversation(self, conversation_id: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
        """Read a conversation; strict callers distinguish missing from failed."""
        conn = self._connect()
        if conn is None:
            if strict:
                raise sqlite3.OperationalError("Chat store unavailable")
            return None
        try:
            with self._lock:
                r = conn.execute(
                    "SELECT id, title, created_ts, updated_ts FROM conversations "
                    "WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
        except Exception:
            log.exception("chat_store: get_conversation failed")
            if strict:
                raise
            return None
        if r is None:
            return None
        return {"id": r["id"], "title": r["title"],
                "created_ts": r["created_ts"], "updated_ts": r["updated_ts"]}

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        conn = self._connect()
        if conn is None:
            return False
        clean = (title or "").strip()
        if not clean:
            return False
        try:
            with self._lock:
                cur = conn.execute(
                    "UPDATE conversations SET title = ?, updated_ts = ? WHERE id = ?",
                    (clean[:_TITLE_MAX], _now(), conversation_id),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            log.exception("chat_store: rename_conversation failed")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its turns. Returns True if the
        conversation existed. FK cascade isn't relied on (PRAGMA foreign_keys
        is off by default) — turns are removed explicitly."""
        conn = self._connect()
        if conn is None:
            return False
        try:
            with self._lock:
                conn.execute("DELETE FROM turns WHERE conversation_id = ?",
                             (conversation_id,))
                cur = conn.execute("DELETE FROM conversations WHERE id = ?",
                                   (conversation_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            log.exception("chat_store: delete_conversation failed")
            return False

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------
    def append_turn(
        self,
        conversation_id: str,
        user: str,
        reply: str,
        status_mode: str = "OK",
        source: str = "ui",
        asked_ts: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
        turn_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a turn to a conversation. Bumps updated_ts, and — if this is
        the conversation's first turn and the title is still the placeholder —
        names the conversation after the first user message (Claude-style).
        Returns the stored turn (same shape as ConversationBuffer) or None if
        the conversation doesn't exist / the write fails."""
        conn = self._connect()
        if conn is None:
            return None
        if not isinstance(turn_id, str) or not 1 <= len(turn_id) <= 64:
            turn_id = _short_id()
        ts = _now()
        try:
            with self._lock:
                # The write lock BEFORE the existence check: checked first and
                # written after, a delete from another process landed in
                # between, and the turn was stored onto nothing while the call
                # reported success (s88, Codex lote 7, reproduced).
                conn.execute("BEGIN IMMEDIATE")
                convo = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if convo is None:
                    conn.rollback()
                    return None  # never orphan a turn onto a missing conversation
                n = conn.execute(
                    "SELECT COUNT(*) AS c FROM turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()["c"]
                conn.execute(
                    """
                    INSERT INTO turns (id, conversation_id, user, reply,
                        status_mode, source, asked_ts, ts, usage_json, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id, conversation_id, user, reply, status_mode, source,
                        asked_ts, ts,
                        json.dumps(usage, ensure_ascii=False) if usage is not None else None,
                        latency_ms,
                    ),
                )
                # First turn on a still-unnamed conversation → title from it.
                if n == 0 and convo["title"] == _DEFAULT_TITLE:
                    new_title = " ".join((user or "").split())[:_TITLE_MAX] or _DEFAULT_TITLE
                    conn.execute(
                        "UPDATE conversations SET title = ?, updated_ts = ? WHERE id = ?",
                        (new_title, ts, conversation_id),
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_ts = ? WHERE id = ?",
                        (ts, conversation_id),
                    )
                conn.commit()
        except Exception:
            log.exception("chat_store: append_turn failed")
            try:
                conn.rollback()             # never leave the write lock held
            except Exception:  # noqa: BLE001
                pass
            return None
        return {
            "id": turn_id, "conversation_id": conversation_id,
            "ts": ts, "asked_ts": asked_ts,
            "user": user, "reply": reply,
            "status_mode": status_mode, "source": source,
            "usage": usage, "latency_ms": latency_ms,
        }

    def pop_last_turn(self, conversation_id: str,
                      expected_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Remove and return the conversation's most recent turn (fatia 3:
        regenerate). The turn is deleted so the regenerated answer REPLACES it
        instead of piling up a second copy — and so the short-term history the
        next turn binds (UI 2c) doesn't feed the model the very answer the user
        just rejected.

        Returns the removed turn (so the caller can re-ask its user text), or
        None if the conversation has no turns / doesn't exist / the write
        fails. `updated_ts` is intentionally NOT bumped: the append that
        follows bumps it, and a failed regeneration shouldn't reorder the
        sidebar. The conversation TITLE is left alone — a regenerated first
        turn keeps the same user message, so the title stays accurate.

        Note: this only touches the UI's history ledger. The Palace (long-term
        memory) already recorded the original interaction and is append-only by
        design — regenerating does not retract it.

        `expected_id` is the turn the page shows as its last: when a newer one
        sits on top (a voice turn mirrored into the conversation while the page
        was open), NotTheLastTurn is raised and nothing is deleted — "the
        newest" alone deleted that unseen turn instead (s88, Codex lote 7)."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM turns WHERE conversation_id = ? "
                    "ORDER BY seq DESC LIMIT 1",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    return None
                if expected_id is not None and row["id"] != expected_id:
                    raise NotTheLastTurn(row["id"])
                turn = _row_to_turn(row)
                conn.execute("DELETE FROM turns WHERE seq = ?", (row["seq"],))
                conn.commit()
                return turn
        except NotTheLastTurn:
            raise
        except Exception:
            log.exception("chat_store: pop_last_turn failed")
            return None

    def get_turns(self, conversation_id: str,
                  limit: Optional[int] = None, *, strict: bool = False) -> List[Dict[str, Any]]:
        """Turns of a conversation in chat order (oldest first). With `limit`,
        returns the last `limit` turns (still oldest-first). With `strict`,
        raise on read failures so erasure cannot leave orphaned memories."""
        conn = self._connect()
        if conn is None:
            if strict:
                raise sqlite3.OperationalError("Chat store unavailable")
            return []
        try:
            with self._lock:
                if limit is not None:
                    rows = conn.execute(
                        "SELECT * FROM (SELECT * FROM turns WHERE conversation_id = ? "
                        "ORDER BY seq DESC LIMIT ?) ORDER BY seq ASC",
                        (conversation_id, max(1, int(limit))),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM turns WHERE conversation_id = ? ORDER BY seq ASC",
                        (conversation_id,),
                    ).fetchall()
        except Exception:
            log.exception("chat_store: get_turns failed")
            if strict:
                raise
            return []
        return [_row_to_turn(r) for r in rows]


class NotTheLastTurn(Exception):
    """pop_last_turn(expected_id=…) found a newer turn on top."""

def _row_to_turn(r: sqlite3.Row) -> Dict[str, Any]:
    usage = None
    if r["usage_json"]:
        try:
            usage = json.loads(r["usage_json"])
        except Exception:
            usage = None
    return {
        "id": r["id"], "conversation_id": r["conversation_id"],
        "ts": r["ts"], "asked_ts": r["asked_ts"],
        "user": r["user"], "reply": r["reply"],
        "status_mode": r["status_mode"], "source": r["source"],
        "usage": usage, "latency_ms": r["latency_ms"],
    }


# Module-level singleton — one store for the process lifetime. Tests
# instantiate their own with an injected path (or :memory:) for isolation.
store = ChatStore()


# Turns that happen OUTSIDE the chat page — spoken to the voice listener,
# typed in a terminal — land in one conversation per source per DAY, so the
# chat history holds everything the user said to Cortex. A sidebar row per "toca
# uma música" would be unusable, hence per day.
VOICE_LABEL = "🎙️ Voz"
# s85: the user talked to Cortex in the terminal (`cortex`, `cx`), closed it, and
# the whole conversation was gone — and asked to "bring it here", Cortex
# described an unrelated old one. Terminal turns now land beside the voice ones.
TERMINAL_LABEL = "⌨️ Terminal"
# /ask called over the network (Tailscale: the iPhone, another Mac) — not
# this Mac's terminal, so not labelled as one.
NETWORK_LABEL = "🌐 Rede"


def daily_title(label: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return f"{label} · " + now.astimezone().strftime("%a %d/%m")


def append_daily(source: str, label: str, user: str, reply: str,
                 status_mode: str = "OK", usage: Optional[dict] = None) -> None:
    """Append a turn to `label`'s conversation for today, creating it on the
    day's first turn. Looks `store` up at call time (tests swap it).

    Fail-soft: the reply was already given — spoken, printed — and a storage
    hiccup must never break it. Empty turns are not stored."""
    user = (user or "").strip()
    reply = (reply or "").strip()
    if not user or not reply:
        return
    try:
        convo_id = store.conversation_titled(daily_title(label))
        if convo_id is None:
            return
        store.append_turn(
            conversation_id=convo_id,
            user=user,
            reply=reply,
            status_mode=status_mode or "OK",
            source=source,
            usage=usage,
        )
    except Exception:  # noqa: BLE001 — the reply already went out
        log.exception("%s turn chat_store append failed", source)
