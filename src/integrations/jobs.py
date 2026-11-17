"""
JobLog — local SQLite store for the Autonomous Job Hunter (slice 1, s76).

Keeps the public job postings fetched from ATS boards (see ats.py) in a local
ledger so the slice-2 ranking against Pedro's master resume runs entirely
on-device. Mirrors fitness.py: a class with an injectable path plus a
module-level singleton, errors never propagate into the request path
(fail-soft — a missing/corrupt DB degrades to "couldn't store" rather than
breaking dispatch).

Dedup is by canonical URL: re-fetching a board refreshes the volatile fields
(title/location/description/posted_at) but PRESERVES `score` and `status` so a
job you already ranked or dismissed isn't reset.

Privacy (SOBERANIA r1): the DB lives in ~/.cortex/jobs.db (env override
CORTEX_JOBS_DB), outside the project tree. Job postings are public; the resume
(slice 2) is the personal half and stays local forever.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.core.sqlite_safe import RollbackOnError
from src.integrations.ats import JobPosting, is_closed

log = logging.getLogger(__name__)

_VALID_STATUS = ("new", "seen", "applied", "dismissed")


def _row_value(row, key, default=None):
    """sqlite3.Row has no .get(); tolerate a plain dict too (tests/fakes)."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


class JobLog:
    """SQLite-backed job ledger. Thread-safe, fail-soft."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/jobs.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_JOBS_DB", self.DEFAULT_PATH)
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
            conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                   factory=RollbackOnError)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source      TEXT NOT NULL,
                    board       TEXT NOT NULL,
                    company     TEXT,
                    ext_id      TEXT,
                    title       TEXT NOT NULL,
                    location    TEXT,
                    url         TEXT NOT NULL UNIQUE,
                    description TEXT,
                    posted_at   TEXT,
                    fetched_at  TEXT NOT NULL,
                    score       REAL,
                    distance    REAL,
                    closing_at  TEXT,
                    status      TEXT NOT NULL DEFAULT 'new'
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
                -- One row per successful fetch day of a (source, board): the
                -- evidence that a vacancy missing from it is gone (s88, Codex
                -- lote 11). jobs.fetched_at only says when each was LAST seen.
                CREATE TABLE IF NOT EXISTS fetch_runs (
                    source TEXT NOT NULL,
                    board  TEXT NOT NULL,
                    day    TEXT NOT NULL,
                    PRIMARY KEY (source, board, day)
                );
                CREATE TABLE IF NOT EXISTS resume (
                    id       INTEGER PRIMARY KEY CHECK (id = 1),
                    name     TEXT,
                    text     TEXT NOT NULL,
                    added_at TEXT NOT NULL
                );
                """
            )
            self._migrate(conn)
            self._conn = conn
            return self._conn
        except Exception:
            log.exception("jobs: could not open ledger at %s", self.db_path)
            return None

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a DB was first created (idempotent)."""
        have = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        for col, decl in (("distance", "REAL"), ("score", "REAL"),
                          ("closing_at", "TEXT")):
            if col not in have:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
        conn.commit()

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def upsert(self, postings: List[JobPosting]) -> Tuple[int, int]:
        """Insert/refresh postings. Dedup by URL; refresh volatile fields but
        keep score/status. Returns (new, total) where `new` is how many URLs
        weren't already in the ledger. Postings without a URL are skipped
        (can't dedup). Fail-soft → (0, 0)."""
        rows = [p for p in postings if (p.url or "").strip()]
        if not rows:
            return (0, 0)
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return (0, 0)
                existing = self._existing_urls(conn, [p.url for p in rows])
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.executemany(
                    """
                    INSERT INTO jobs
                      (source, board, company, ext_id, title, location, url,
                       description, posted_at, fetched_at, distance, closing_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                       title       = excluded.title,
                       location    = excluded.location,
                       description = excluded.description,
                       posted_at   = excluded.posted_at,
                       fetched_at  = excluded.fetched_at,
                       distance    = excluded.distance,
                       closing_at  = excluded.closing_at
                    """,
                    [(p.source, p.board, p.company, p.ext_id, p.title,
                      p.location, p.url, p.description, p.posted_at, now,
                      p.distance, p.closing_at)
                     for p in rows],
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO fetch_runs (source, board, day) VALUES (?, ?, ?)",
                    sorted({(p.source, p.board or "", now[:10]) for p in rows}),
                )
                conn.commit()
            new = sum(1 for p in rows if p.url not in existing)
            return (new, len(rows))
        except Exception:
            log.exception("jobs: upsert failed")
            return (0, 0)

    @staticmethod
    def _existing_urls(conn: sqlite3.Connection, urls: List[str]) -> set:
        """URLs already stored, queried in chunks to respect SQLite's variable
        limit (~999)."""
        found: set = set()
        for i in range(0, len(urls), 500):
            chunk = urls[i:i + 500]
            q = "SELECT url FROM jobs WHERE url IN (%s)" % ",".join("?" * len(chunk))
            for row in conn.execute(q, chunk):
                found.add(row["url"])
        return found

    def set_score(self, job_id: int, score: float) -> bool:
        """Write a match score (0..1) for a job. Fail-soft."""
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                conn.execute("UPDATE jobs SET score = ? WHERE id = ?",
                             (float(score), job_id))
                conn.commit()
            return True
        except Exception:
            log.exception("jobs: set_score failed")
            return False

    # ------------------------------------------------------------------
    # resume (the personal half — SOBERANIA r1: stays local, never sent out;
    # ranking embeds it with the LOCAL Ollama model only)
    # ------------------------------------------------------------------

    def set_resume(self, text: str, *, name: Optional[str] = None) -> bool:
        """Store/replace the master resume (single row)."""
        text = (text or "").strip()
        if not text:
            return False
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "INSERT OR REPLACE INTO resume (id, name, text, added_at) "
                    "VALUES (1, ?, ?, ?)", (name, text, now))
                conn.commit()
            return True
        except Exception:
            log.exception("jobs: set_resume failed")
            return False

    def get_resume(self) -> Optional[sqlite3.Row]:
        try:
            conn = self._connect()
            if conn is None:
                return None
            return conn.execute("SELECT * FROM resume WHERE id = 1").fetchone()
        except Exception:
            log.exception("jobs: get_resume failed")
            return None

    def set_status(self, job_id: int, status: str) -> bool:
        """Mark a job seen/applied/dismissed. Fail-soft."""
        if status not in _VALID_STATUS:
            return False
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                conn.execute("UPDATE jobs SET status = ? WHERE id = ?",
                             (status, job_id))
                conn.commit()
            return True
        except Exception:
            log.exception("jobs: set_status failed")
            return False

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def list(self, *, limit: int = 20, status: Optional[str] = None,
             source: Optional[str] = None,
             order: str = "distance") -> List[sqlite3.Row]:
        """Filtered jobs. order="distance" → nearest first (jobs without a
        distance fall to the end, then by recency); order="score" → best match
        first; anything else → most-recently-fetched first."""
        try:
            conn = self._connect()
            if conn is None:
                return []
            where, params = [], []
            if status:
                where.append("status = ?")
                params.append(status)
            if source:
                where.append("source = ?")
                params.append(source)
            clause = ("WHERE " + " AND ".join(where)) if where else ""
            if order == "distance":
                order_sql = "(distance IS NULL), distance ASC, fetched_at DESC"
            elif order == "score":
                order_sql = "(score IS NULL), score DESC, distance ASC"
            else:
                order_sql = "fetched_at DESC, id DESC"
            params.append(int(limit))
            return list(conn.execute(
                f"SELECT * FROM jobs {clause} "
                f"ORDER BY {order_sql} LIMIT ?", params))
        except Exception:
            log.exception("jobs: list failed")
            return []

    # The ledger is a personal one (hundreds of rows), so scanning it in Python
    # to drop closed vacancies is cheap AND always correct. A SQL filter would
    # have to compare a TEXT column against "now" with the same timezone care
    # days_until_close already takes — one parser, one definition, no drift.
    _OPEN_SCAN_CAP = 1000

    # A vacancy may miss ONE fetch day and still count as listed. The 07:00
    # cron can fail or come back partial, and one grace day means two bad runs
    # in a row would be needed to hide a live vacancy. Nothing is deleted and
    # the rule is computed on read, so a wrong call heals itself the moment the
    # feed lists the vacancy again.
    _FEED_GRACE_DAYS = 1

    def feed_cutoffs(self) -> Dict[str, str]:
        """Per source: the fetch DAY a vacancy must have reached to still count
        as listed. Older than this and it is gone from the board (s83).

        findapprenticeship DROPS a vacancy from the feed the day it closes, and
        `fetched_at` is refreshed on every fetch that still sees it — so the
        ledger already records "last seen" and simply wasn't reading it. Checked
        against the live pages: "IT DevOps Apprentice" last seen 17/06, its page
        says closed 18 June; "AI Digital Growth" last seen 17/09, page says
        closed 17 September.

        A source with fewer than two fetch days has no cutoff: we cannot tell a
        vanished vacancy from one we have only ever fetched once.

        The fetch days come from fetch_runs, per (source, board) (s88, Codex
        lote 11). Read off the vacancies' own last-seen days, one live vacancy
        refreshed daily and one gone since June left two distinct days —
        today and June — so the cutoff was June and the June one was
        recommended forever; and a board refreshed set the cutoff for its
        source's other boards. Until a board has the history, the old
        last-seen days still stand in for it.

        Keys are (source, board).
        """
        cutoffs: Dict[Tuple[str, str], str] = {}
        try:
            conn = self._connect()
            if conn is None:
                return cutoffs
            with conn:
                runs = conn.execute(
                    "SELECT source, board, day FROM fetch_runs "
                    "ORDER BY source, board, day DESC").fetchall()
                seen = conn.execute(
                    "SELECT source, COALESCE(board, '') AS board, "
                    "substr(fetched_at, 1, 10) AS day FROM jobs "
                    "GROUP BY source, board, day ORDER BY source, board, day DESC"
                ).fetchall()
        except sqlite3.Error:
            return cutoffs
        fetched: Dict[Tuple[str, str], List[str]] = {}
        for r in runs:
            fetched.setdefault((r["source"], r["board"]), []).append(r["day"])
        legacy: Dict[Tuple[str, str], List[str]] = {}
        for r in seen:
            legacy.setdefault((r["source"], r["board"]), []).append(r["day"])
        for key in set(legacy) | set(fetched):
            # newest-first; index _FEED_GRACE_DAYS is the oldest run a
            # still-listed vacancy is allowed to have been last seen in.
            ds = fetched.get(key, [])
            if len(ds) <= self._FEED_GRACE_DAYS:
                ds = legacy.get(key, [])
            if len(ds) > self._FEED_GRACE_DAYS:
                cutoffs[key] = ds[self._FEED_GRACE_DAYS]
        return cutoffs

    @staticmethod
    def is_gone_from_feed(row, cutoffs: Dict[str, str]) -> bool:
        """True when the board stopped listing this vacancy — see feed_cutoffs.
        Pass the cutoffs in so a listing computes them once, not per row."""
        cutoff = cutoffs.get((_row_value(row, "source"), _row_value(row, "board") or ""))
        if not cutoff:
            return False
        seen = (_row_value(row, "fetched_at") or "")[:10]
        return bool(seen) and seen < cutoff

    def list_open(self, *, limit: int = 20, status: Optional[str] = None,
                  source: Optional[str] = None,
                  order: str = "score") -> List[sqlite3.Row]:
        """Like `list()`, but only vacancies still worth recommending (s82/s83).

        For the surfaces that RECOMMEND a job — the conversation handler and
        the morning briefing — a closed vacancy is worse than no vacancy: it
        pushes a live match off a 3-slot list and sends Pedro to a dead page.
        (The real ledger was 95/182 closed when this landed.) `list()` itself
        is unchanged: the CLI's full listing legitimately shows everything and
        labels it "encerrada".

        Over-fetches and filters so `limit` still yields `limit` OPEN rows
        rather than "up to limit, minus however many were dead"."""
        rows = self.list(limit=self._OPEN_SCAN_CAP, status=status,
                         source=source, order=order)
        # Two ways to be dead, and the second one is the common one: a deadline
        # we can prove is past, or a vacancy the board no longer lists. The
        # deadline test alone let 43 vanished vacancies through — 16 of them
        # with no deadline at all, so they would have been recommended forever
        # (s83: a June vacancy was still in the top 3 in late September).
        cutoffs = self.feed_cutoffs()
        open_rows = [r for r in rows
                     if not is_closed(_row_value(r, "closing_at"))
                     and not self.is_gone_from_feed(r, cutoffs)]
        return open_rows[:max(0, int(limit))]

    def count(self, *, status: Optional[str] = None) -> int:
        try:
            conn = self._connect()
            if conn is None:
                return 0
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            log.exception("jobs: count failed")
            return 0

    def clear(self, *, source: Optional[str] = None) -> int:
        """Delete all jobs (or just one source). Returns rows removed."""
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return 0
                if source:
                    cur = conn.execute("DELETE FROM jobs WHERE source = ?", (source,))
                else:
                    cur = conn.execute("DELETE FROM jobs")
                conn.commit()
                return cur.rowcount or 0
        except Exception:
            log.exception("jobs: clear failed")
            return 0


_log_singleton: Optional[JobLog] = None


def get_log() -> JobLog:
    global _log_singleton
    if _log_singleton is None:
        _log_singleton = JobLog()
    return _log_singleton
