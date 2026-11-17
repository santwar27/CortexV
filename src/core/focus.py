"""Focus sessions — start, stop, pick up where you left off (s86).

Pedro named two snags in his day: putting a task off ("fico enrolando") and
being pulled away half-way ("paro no meio"). A focus session is the smallest
help for both: name the task and a 5-minute first step, get an alarm at the
end, write the next step down when you stop, and ask "onde parei?" later.

The 8B decides none of it. Measured before (s86 bloco 0, real 8B): 1 useful
reply in 30 — "foco: candidatura…" became a job search, "abrir o formulário"
tried to open an app called "formulário", and "onde parei?" came back as
"Curitiba, Paraná." This module is the record and the pure rules; the
orchestrator routes the words here.

Store: ~/.cortex/focus.db (env override CORTEX_FOCUS_DB), outside the project
tree — the task names are Pedro's day, not code. Same shape as chat_store: an
injectable path, a lazy connection, thread-safe, fail-soft (a storage hiccup
degrades to "no session", never to an exception in the reply path). Times are
aware ISO strings in the local zone (clock.local_now()). Every sum and
comparison goes through UTC: Python adds a timedelta to a ZoneInfo time on the
WALL clock, so at 01:50 on 25/10/2026 (clocks go back at 02:00) "+25 min"
landed at 02:15 GMT — 1h25 later — and two times in the same zone compare by
wall clock too, ignoring the repeated hour.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_MINUTES = 25            # Pedro, 16/10
MIN_MINUTES, MAX_MINUTES = 5, 180
# The nudge is a SECOND reminder, 10 min past the end (the user, 16/10): it
# reaches the Mac AND the iPhone at the exact minute — "paro no meio" means
# away from the Mac — where the nudge driver's 15-min tick would land 10-25
# min late, on the Mac only, and lose to an imminent meeting.
NUDGE_AFTER = timedelta(minutes=10)
REMINDERS_LIST = "Cortex Foco"
# Stored in the reminder column when Reminders.app failed to create it, so
# "onde parei?" can say the phone won't ring.
REMINDER_FAILED = "failed"
REMINDER_UNKNOWN = "unknown"    # the create may have run, but no id was confirmed
_TEXT_MAX = 200                 # a task name, not an essay; keeps reminders sane


# ----------------------------------------------------------------------
# the words
# ----------------------------------------------------------------------
_WAKE = r"(?:cortex[\s,]+)?"

# The regexes find the SHAPE only; Python trims the text. A variable part
# between two \s* (": \s*(.*?)\s*[.!]*\s*$") let the engine try every way to
# split a run of spaces: "parei: a" + 1,000 spaces + "b" took 1.3 s, 8,000
# ran past two minutes (s86 review). Each run of spaces now has ONE owner.

# "foco: X", "foco 40: X", "foco 40 min: X", "focus: X". The colon is the
# signal — "meu foco é o college" is a sentence, not a command.
_START_RE = re.compile(
    rf"^\s*{_WAKE}(?:foco|focus)"
    r"(?:\s+(?P<minutes>\d{1,3})(?:\s*(?:min(?:utos?)?|m))?)?"
    r"\s*:(?P<task>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# "parei: preencher a seção 2", or a bare "parei" / "parei." — but not
# "parei de fumar", which is a sentence.
_STOP_RE = re.compile(
    rf"^\s*{_WAKE}parei(?:\s*:(?P<next>.*)|[\s.!]*)$",
    re.IGNORECASE | re.DOTALL,
)

_WHERE_RE = re.compile(
    rf"^\s*{_WAKE}(?:"
    # The user's ways of asking (16/10), accents optional.
    r"onde\s+(?:(?:[ée]\s+)?que\s+)?(?:eu\s+)?parei"
    r"|onde\s+(?:a\s+gente\s+parou|paramos)"
    r"|em\s+que\s+(?:eu\s+)?parei"
    r"|o\s+que\s+eu\s+(?:estava|tava)\s+fazendo"
    r"|where\s+was\s+i"
    r"|what\s+was\s+i\s+doing"
    r")[\s?.!]*$",
    re.IGNORECASE,
)


# "não sei" to "qual o primeiro passo?" is the answer the question exists for:
# the cycle offers a default step instead of a dead end.
_DONT_KNOW_RE = re.compile(
    r"^\s*(?:n[aã]o\s+sei|nem\s+sei|sei\s+l[aá]|i\s+don'?t\s+know|no\s+idea)"
    r"[\s.!]*$",
    re.IGNORECASE,
)


# The command word says the language better than the detector: "foco:
# imposto" has no accent, and read as English (s86 bloco 2a).
_EN_COMMAND_RE = re.compile(
    rf"^\s*{_WAKE}(?:focus|where\s+was|what\s+was)\b", re.IGNORECASE)


def clip_text(text: str) -> str:
    return " ".join((text or "").split())[:_TEXT_MAX]


def parse_start(text: str) -> Optional[Tuple[str, int]]:
    """(task, minutes) for "foco: X" / "foco 40: X"; None otherwise.
    Minutes outside 5–180 are clamped — "foco 2: X" still starts a focus."""
    m = _START_RE.match(text or "")
    if not m:
        return None
    task = clip_text(m.group("task"))
    if not task:
        return None
    minutes = int(m.group("minutes")) if m.group("minutes") else DEFAULT_MINUTES
    return task, max(MIN_MINUTES, min(MAX_MINUTES, minutes))


def parse_stop(text: str) -> Optional[str]:
    """The next step for "parei: X" ("" for a bare "parei"); None when the
    message is not a stop."""
    m = _STOP_RE.match(text or "")
    if not m:
        return None
    return clip_text((m.group("next") or "").strip().rstrip(".!"))


def command_lang(text: str) -> str:
    """ "en" for "focus:", "where was I?", "what was I doing?"; "pt" for the
    rest of the commands ("foco:", "parei", "onde parei?")."""
    return "en" if _EN_COMMAND_RE.match(text or "") else "pt"


def doesnt_know(text: str) -> bool:
    """ "não sei", "sei lá", "I don't know" — the whole message."""
    return bool(_DONT_KNOW_RE.match(text or ""))


def asks_where_was_i(text: str) -> bool:
    """ "onde parei?", "o que eu estava fazendo?", "where was I?" — the whole
    message, so "onde parei o carro?" stays a normal question."""
    return bool(_WHERE_RE.match(text or ""))


# ----------------------------------------------------------------------
# the record
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    id: int
    task: str
    first_step: str
    started_at: datetime
    ends_at: datetime
    closed_at: Optional[datetime] = None
    next_step: Optional[str] = None
    reminder: Optional[str] = None          # the end-of-focus Reminders item
    nudge_reminder: Optional[str] = None    # the one 10 min later
    reminder_state: Optional[str] = None   # NULL means legacy, no durable intent
    nudge_state: Optional[str] = None
    completion_wanted: bool = False
    reminder_completion: Optional[str] = None
    nudge_completion: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def time_is_up(self, now: datetime) -> bool:
        return _utc(now) >= _utc(self.ends_at)

    def ran_minutes(self, now: datetime) -> int:
        """Whole minutes from the start to `now` (or to the close)."""
        end = self.closed_at or now
        return int((_utc(end) - _utc(self.started_at)) / timedelta(minutes=1))


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)


def later(dt: datetime, delta: timedelta) -> datetime:
    """`dt` + `delta` in real time, shown in dt's zone — never on the wall
    clock (see the module docstring: 25/10, +25 min ended 1h25 later)."""
    return (_utc(dt) + delta).astimezone(dt.tzinfo)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _dt(text: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(text) if text else None


def _row_to_session(r: sqlite3.Row) -> Session:
    return Session(
        id=r["id"], task=r["task"], first_step=r["first_step"],
        started_at=_dt(r["started_at"]), ends_at=_dt(r["ends_at"]),
        closed_at=_dt(r["closed_at"]), next_step=r["next_step"],
        reminder=r["reminder"], nudge_reminder=r["nudge_reminder"],
        reminder_state=r["reminder_state"], nudge_state=r["nudge_state"],
        completion_wanted=bool(r["completion_wanted"]),
        reminder_completion=r["reminder_completion"], nudge_completion=r["nudge_completion"],
    )


@dataclass(frozen=True)
class ReminderWork:
    session_id: int
    which: str
    state: str
    ref: Optional[str]
    completion: Optional[str]


_REMINDER_COLUMNS = {
    "end": ("reminder", "reminder_state", "reminder_completion"),
    "nudge": ("nudge_reminder", "nudge_state", "nudge_completion"),
}
_CLOSE_REMINDERS = (
    "completion_wanted = 1, "
    "reminder_state = CASE WHEN reminder_state = 'pending' THEN 'skipped' ELSE reminder_state END, "
    "nudge_state = CASE WHEN nudge_state = 'pending' THEN 'skipped' ELSE nudge_state END"
)


class FocusStore:
    """SQLite record of focus sessions. At most one is open at a time:
    start() closes any open one first. Every method swallows DB errors and
    returns None / [] / False, so a storage hiccup never breaks a reply.

    Two PROCESSES share the file (the UI's uvicorn and the terminal REPL),
    each with a Reminders worker thread; the threading lock covers only one
    process. Every write runs in `with conn:`
    — commit on success, ROLLBACK on error. Without it a start() whose INSERT
    failed kept its UPDATE in an open transaction: the other processes got
    "database is locked" and this one saw a different record (s86 review)."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/focus.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_FOCUS_DB", self.DEFAULT_PATH)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> Optional[sqlite3.Connection]:
        # available() also holds this reentrant lock. First use by two threads
        # must not publish separate connections while the migration is running.
        with self._lock:
            return self._connect_locked()

    def _connect_locked(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        conn = None
        try:
            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            with self._lock, conn:
                conn.execute("PRAGMA journal_mode = WAL")
                # Serialize schema inspection and ALTER across processes; a
                # second opener sees the committed columns, never adds twice.
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        task        TEXT NOT NULL,
                        first_step  TEXT NOT NULL,
                        started_at  TEXT NOT NULL,
                        ends_at     TEXT NOT NULL,
                        closed_at   TEXT,
                        next_step   TEXT,
                        reminder        TEXT,
                        nudge_reminder  TEXT
                    );
                    """
                )
                columns = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
                for name, kind in (
                    ("reminder_state", "TEXT"), ("nudge_state", "TEXT"),
                    ("completion_wanted", "INTEGER NOT NULL DEFAULT 0"),
                    ("reminder_completion", "TEXT"), ("nudge_completion", "TEXT"),
                ):
                    if name not in columns:
                        conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {kind}")
            self._conn = conn
            return self._conn
        except Exception as e:
            if conn is not None:
                conn.close()
            log.warning("focus: could not open the record: %s", type(e).__name__)
            return None

    def available(self) -> bool:
        # Empty reads also mean failure; read models must distinguish them.
        try:
            with self._lock:
                conn = self._connect()
                return conn is not None and bool(conn.execute("SELECT 1").fetchone())
        except Exception:
            log.exception("focus: availability check failed")
            return False

    def _one(self, sql: str, args: tuple = ()) -> Optional[Session]:
        conn = self._connect()
        if conn is None:
            return None
        try:
            with self._lock:
                row = conn.execute(sql, args).fetchone()
            return _row_to_session(row) if row else None
        except Exception as e:
            log.warning("focus: read failed: %s", type(e).__name__)
            return None

    def start(self, task: str, first_step: str, minutes: int,
              now: datetime) -> Optional[Session]:
        """Open a session from now for `minutes`, closing any open one first
        (no next step saved on it — the caller decides, see
        on_start_while_open)."""
        return self.start_closing(task, first_step, minutes, now)[0]

    def start_closing(self, task: str, first_step: str, minutes: int,
                      now: datetime) -> Tuple[Optional[Session], List[int]]:
        """start(), plus the ids of the sessions it closed — told by the same
        UPDATE (RETURNING), not by a current() read before it: a "foco:" from
        another process landed in between, this one closed it, and its alarms
        were never completed (s88, Codex lote 4)."""
        conn = self._connect()
        if conn is None:
            return None, []
        ends = later(now, timedelta(minutes=minutes))
        try:
            with self._lock, conn:
                closed = [r[0] for r in conn.execute(
                    f"UPDATE sessions SET closed_at = ?, {_CLOSE_REMINDERS} WHERE closed_at IS NULL "
                    "RETURNING id", (_iso(now),)).fetchall()]
                cur = conn.execute(
                    "INSERT INTO sessions (task, first_step, started_at, ends_at, reminder_state, nudge_state) "
                    "VALUES (?, ?, ?, ?, 'pending', 'pending')",
                    (clip_text(task), clip_text(first_step), _iso(now), _iso(ends)))
                sid = cur.lastrowid
        except Exception as e:
            log.warning("focus: start failed: %s", type(e).__name__)
            return None, []
        return (Session(id=sid, task=clip_text(task), first_step=clip_text(first_step),
                        started_at=now, ends_at=ends,
                        reminder_state="pending", nudge_state="pending"), closed)

    def get(self, session_id: int) -> Optional[Session]:
        return self._one("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def current(self) -> Optional[Session]:
        """The open session, if any."""
        return self._one(
            "SELECT * FROM sessions WHERE closed_at IS NULL ORDER BY id DESC LIMIT 1")

    def last(self) -> Optional[Session]:
        """The most recent session, open or closed — what "onde parei?" is
        about."""
        return self._one("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")

    def stop(self, next_step: str, now: datetime) -> Optional[Session]:
        """Close the open session with its next step. None when nothing is
        open (the caller says so — it never invents a session).

        One statement picks the open session AND closes it: reading it first
        and writing after let a "foco:" from another process land in between,
        and the next step went to the session it had just closed."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            with self._lock, conn:
                row = conn.execute(
                    f"UPDATE sessions SET closed_at = ?, next_step = ?, {_CLOSE_REMINDERS} "
                    "WHERE id = (SELECT id FROM sessions WHERE closed_at IS NULL "
                    "            ORDER BY id DESC LIMIT 1) "
                    "RETURNING *",
                    (_iso(now), clip_text(next_step) or None)).fetchone()
            return _row_to_session(row) if row else None
        except Exception as e:
            log.warning("focus: stop failed: %s", type(e).__name__)
            return None

    def set_reminders(self, session_id: int, end: Optional[str],
                      nudge: Optional[str]) -> bool:
        """Remember the session's two Reminders items (the end, the nudge 10
        min later), so a stop can mark them done — completing takes ~1 s,
        deleting 28–37 s. REMINDER_FAILED in `end` when the app failed."""
        return self._write(
            "UPDATE sessions SET reminder = ?, nudge_reminder = ? WHERE id = ?",
            (end, nudge, session_id))

    def set_next_step(self, session_id: int, step: str) -> bool:
        """The answer to "qual o próximo passo?" after a bare "parei"."""
        return self._write("UPDATE sessions SET next_step = ? WHERE id = ?",
                           (clip_text(step) or None, session_id))

    def claim_reminder(self, session_id: int, which: str) -> Optional[bool]:
        """True for the sole pending→running winner, False for no work,
        None for a store error (the worker audits only the latter)."""
        ref, state, _ = _REMINDER_COLUMNS[which]
        conn = self._connect()
        if conn is None:
            return None
        prerequisite = (
            "AND reminder_state = 'done' AND reminder IS NOT NULL "
            "AND reminder NOT IN ('failed', 'unknown') " if which == "nudge" else ""
        )
        try:
            with self._lock, conn:
                return conn.execute(
                    f"UPDATE sessions SET {state} = 'running' WHERE id = ? "
                    f"AND {state} = 'pending' AND {ref} IS NULL "
                    "AND closed_at IS NULL AND completion_wanted = 0 " + prerequisite,
                    (session_id,)).rowcount == 1
        except Exception as e:
            log.warning("focus: claim failed: %s", type(e).__name__)
            return None

    def finish_reminder(self, session_id: int, which: str, outcome: str, ref: str) -> bool:
        """Save the id/sentinel and its outcome together; only a claimant finishes."""
        column, state, _ = _REMINDER_COLUMNS[which]
        if outcome not in ("done", "failed", "unknown"):
            return False
        skip = (
            ", nudge_state = CASE WHEN nudge_state = 'pending' THEN 'skipped' ELSE nudge_state END"
            if which == "end" and outcome != "done" else ""
        )
        return self._write(
            f"UPDATE sessions SET {column} = ?, {state} = ?{skip} "
            f"WHERE id = ? AND {state} = 'running'", (ref, outcome, session_id))

    def skip_nudge(self, session_id: int) -> bool:
        return self._write(
            "UPDATE sessions SET nudge_state = CASE WHEN nudge_state = 'pending' "
            "THEN 'skipped' ELSE nudge_state END WHERE id = ?", (session_id,))

    def record_completion(self, session_id: int, which: str, ref: str, result: str) -> bool:
        column, state, completion = _REMINDER_COLUMNS[which]
        if result not in ("done", "not_found", "failed"):
            return False
        # Completing twice is harmless; a concurrent failure must never undo done.
        return self._write(
            f"UPDATE sessions SET {completion} = CASE WHEN {completion} = 'done' "
            f"THEN 'done' ELSE ? END WHERE id = ? AND {column} = ? "
            f"AND {state} IS NOT 'running'", (result, session_id, ref))

    def unfinished_reminders(self) -> List[ReminderWork]:
        """Dormant recovery view. Legacy rows have no intent and are excluded.
        Missing-in-list and failed completions remain unfinished, as do running
        creates (which must NOT be retried). Unknown creates await reconciliation
        by task 93, not a new create."""
        conn = self._connect()
        if conn is None:
            return []
        selects = []
        for which, (ref, state, completion) in _REMINDER_COLUMNS.items():
            selects.append(
                f"SELECT id, '{which}' AS which, {state} AS state, {ref} AS ref, "
                f"{completion} AS completion FROM sessions WHERE {state} IS NOT NULL "
                f"AND ({state} IN ('pending', 'running') OR (completion_wanted = 1 "
                f"AND {ref} IS NOT NULL AND {ref} NOT IN ('failed', 'unknown') "
                f"AND {completion} IS NOT 'done'))")
        try:
            with self._lock:
                rows = conn.execute(" UNION ALL ".join(selects) + " ORDER BY id, which").fetchall()
            return [ReminderWork(r["id"], r["which"], r["state"], r["ref"], r["completion"])
                    for r in rows]
        except Exception as e:
            log.warning("focus: unfinished read failed: %s", type(e).__name__)
            return []

    def _write(self, sql: str, args: tuple) -> bool:
        """True only when exactly one row changed — "no error" is not "done"."""
        conn = self._connect()
        if conn is None:
            return False
        try:
            with self._lock, conn:
                return conn.execute(sql, args).rowcount == 1
        except Exception as e:
            log.warning("focus: write failed: %s", type(e).__name__)
            return False


# ----------------------------------------------------------------------
# the rules
# ----------------------------------------------------------------------
START_NEW = "start_new"     # close the old session, start the new one
ASK_FIRST = "ask_first"     # keep the old one; ask "fecho X e começo Y? (sim)"


def on_start_while_open(old: Session, now: datetime) -> str:
    """A new "foco: …" arrived while `old` is still open. Return START_NEW or
    ASK_FIRST.

    START_NEW never adds friction at the moment of starting — the very moment
    that is hard — but a slip (typing "foco:" twice) silently ends a session
    that was going well. ASK_FIRST protects a session in progress at the cost
    of one more word ("sim"). `old.time_is_up(now)` says whether its time
    already ran out."""
    if old.time_is_up(now):
        return START_NEW
    return ASK_FIRST


store = FocusStore()
