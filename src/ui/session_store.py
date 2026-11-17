"""Revocable browser sessions; only API-keyed token digests reach disk.

Deliberately excluded from scripts/backup.sh's closed list: restoring browser
sessions on another machine is not wanted. Key rotation invalidates them all.
"""
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.core.audit import chmod_600

log = logging.getLogger(__name__)


class SessionStore:
    DEFAULT_PATH = os.path.expanduser("~/.cortex/ui_sessions.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_UI_SESSIONS_DB", self.DEFAULT_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _connect(self):
        if self._conn is not None:
            return self._conn
        conn = None
        try:
            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                chmod_600(self.db_path)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            if self.db_path != ":memory:":
                chmod_600(self.db_path)  # WAL files inherit the database's permissions.
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            with conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY NOT NULL,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL)""")
            self._conn = conn
            return conn
        except Exception:
            if conn is not None:
                conn.close()
            raise

    @staticmethod
    def _digest(token, key):
        return hmac.new(key.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()

    def issue(self, key) -> Optional[str]:
        try:
            token = secrets.token_urlsafe(32)
            digest = self._digest(token, key)
            now = datetime.now(timezone.utc)
            with self._lock:
                conn = self._connect()
                with conn:
                    conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
                    conn.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                                 (digest, now.isoformat(), (now + timedelta(days=30)).isoformat()))
            return token
        except Exception:
            # Omit exception text: auth failures must never disclose credentials.
            log.warning("session_store: issue failed")
            return None

    def valid(self, token, key) -> bool:
        if not isinstance(token, str) or not token or len(token) > 256:
            return False
        try:
            digest = self._digest(token, key)
            with self._lock:
                row = self._connect().execute(
                    "SELECT expires_at FROM sessions WHERE digest = ?", (digest,)).fetchone()
            return row is not None and datetime.fromisoformat(row[0]) > datetime.now(timezone.utc)
        except Exception:
            log.warning("session_store: valid failed")
            return False

    def revoke(self, token, key) -> bool:
        if not isinstance(token, str) or not token or len(token) > 256:
            return False
        try:
            digest = self._digest(token, key)
            with self._lock:
                conn = self._connect()
                with conn:
                    deleted = conn.execute("DELETE FROM sessions WHERE digest = ?", (digest,))
            return deleted.rowcount > 0
        except Exception:
            log.warning("session_store: revoke failed")
            return False


store = SessionStore()
