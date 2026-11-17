"""
Cost tracker — append-only ledger of paid-API usage.

Cortex is local-first (Ollama, Piper, Whisper all run on-device for free),
so the only metered calls today are external search (Tavily) and any future
hosted API (OpenRouter, etc.). This module records one row per billable
call to a small SQLite ledger so `/ui/costs` can answer "what have I spent
this month".

Design mirrors `audit.py`: a class with an injectable path plus a
module-level singleton. Recording must NEVER crash the request pipeline —
a tracker failure degrades silently, exactly like the audit logger.

Metric note (v1): we count CALLS / CREDITS, not dollars. A Tavily basic
search is 1 credit; we store units + unit_kind and leave cost_usd NULL
until a real per-unit price is confirmed for the user's plan. The schema
already carries cost_usd so a future pricing layer is a pure addition.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class CostTracker:
    """SQLite-backed ledger of metered API calls. Thread-safe, fail-soft."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/costs.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_COST_DB", self.DEFAULT_PATH)
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
                CREATE TABLE IF NOT EXISTS cost_events (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        TEXT    NOT NULL,
                    provider  TEXT    NOT NULL,
                    operation TEXT    NOT NULL,
                    units     INTEGER NOT NULL DEFAULT 1,
                    unit_kind TEXT    NOT NULL DEFAULT 'call',
                    cost_usd  REAL,
                    metadata  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_cost_ts ON cost_events(ts);
                CREATE INDEX IF NOT EXISTS idx_cost_provider ON cost_events(provider);
                """
            )
            self._conn = conn
            return self._conn
        except Exception:
            log.exception("cost_tracker: could not open ledger at %s", self.db_path)
            return None

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        operation: str,
        *,
        units: int = 1,
        unit_kind: str = "call",
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one billable-call row. Fail-soft: a tracker error must
        never propagate into the caller's request path."""
        try:
            meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return
                conn.execute(
                    "INSERT INTO cost_events "
                    "(ts, provider, operation, units, unit_kind, cost_usd, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        provider,
                        operation,
                        int(units),
                        unit_kind,
                        cost_usd,
                        meta_json,
                    ),
                )
                conn.commit()
        except Exception:
            log.exception("cost_tracker: record failed (provider=%s op=%s)", provider, operation)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def summary(self, *, since: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate the ledger.

        `since` is an optional ISO timestamp (inclusive lower bound on
        ts). Returns total event count, total known USD (None when every
        matching row has a NULL cost), and a per-(provider, unit_kind)
        breakdown of events + units + cost."""
        empty = {"total_events": 0, "total_cost_usd": None, "since": since, "by_provider": []}
        conn = self._connect()
        if conn is None:
            return empty
        try:
            where, params = ("WHERE ts >= ?", (since,)) if since else ("", ())
            rows = conn.execute(
                f"""
                SELECT provider, unit_kind,
                       COUNT(*)        AS events,
                       SUM(units)      AS units,
                       SUM(cost_usd)   AS cost_usd,
                       COUNT(cost_usd) AS priced_rows
                FROM cost_events
                {where}
                GROUP BY provider, unit_kind
                ORDER BY provider, unit_kind
                """,
                params,
            ).fetchall()
        except Exception:
            log.exception("cost_tracker: summary query failed")
            return empty

        by_provider: List[Dict[str, Any]] = []
        total_events = 0
        total_cost = 0.0
        any_priced = False
        for r in rows:
            total_events += r["events"]
            if r["priced_rows"]:
                any_priced = True
                total_cost += r["cost_usd"] or 0.0
            by_provider.append({
                "provider": r["provider"],
                "unit_kind": r["unit_kind"],
                "events": r["events"],
                "units": r["units"] or 0,
                "cost_usd": r["cost_usd"] if r["priced_rows"] else None,
            })
        return {
            "total_events": total_events,
            "total_cost_usd": round(total_cost, 6) if any_priced else None,
            "since": since,
            "by_provider": by_provider,
        }

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """The most recent ledger rows, newest first."""
        conn = self._connect()
        if conn is None:
            return []
        try:
            limit = max(1, min(int(limit), 500))
            rows = conn.execute(
                "SELECT ts, provider, operation, units, unit_kind, cost_usd, metadata "
                "FROM cost_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception:
            log.exception("cost_tracker: recent query failed")
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except (ValueError, TypeError):
                    pass
            out.append(item)
        return out


# Module-level singleton — import and use directly, like `audit`.
cost_tracker = CostTracker()
