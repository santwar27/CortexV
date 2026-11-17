"""
Contact book — local SQLite ledger of name→email mappings.

Pedro asked for this after the s20 "lara@example.com" bug: typing the
full address every time you want to send mail is friction the assistant
should absorb. "manda email pra lara" should resolve to her stored
address; "qual o email da X?" should answer from disk.

Design mirrors `cost_tracker.py`: a class with an injectable path plus a
module-level singleton. Tracker errors NEVER propagate into the request
path — a missing/corrupt DB degrades to "contact not found" rather than
breaking dispatch.

Privacy note:
- DB lives in ~/.cortex/contacts.db (env override CORTEX_CONTACTS_DB),
  outside the project tree. Backup script picks it up; git never sees it.
- Contacts are NEVER vectorised into Palace memory — they live as a flat
  table, queried by exact + fuzzy name match only.
- The `name` column is the user-facing key (e.g. "lara"). It's case-
  folded on read so "Lara"/"LARA" all resolve the same row.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# Field-label → canonical key, for parse_contact_fields. PT + EN labels.
_FIELD_LABELS = (
    ("name", r"nome|name"),
    ("email", r"e-?mail"),
    ("phone", r"telefone|tel|fone|celular|phone"),
    ("notes", r"notas?|observa[çc][õo]es?|obs|notes?|info"),
)
_LABEL_RE = re.compile(
    r"\b(?P<label>" + "|".join(p for _, p in _FIELD_LABELS) + r")\s*[:=]?\s+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _canon_label(label: str) -> Optional[str]:
    low = label.lower()
    for _key, _pat in _FIELD_LABELS:
        if re.fullmatch(_pat, low, re.IGNORECASE):
            return _key
    return None


def parse_contact_fields(text: str) -> Dict[str, Optional[str]]:
    """Parse a labelled contact description into {name, email, phone, notes}.

    Deterministic — no LLM. Splits on field labels (nome/email/telefone/notas
    + EN equivalents); each value runs until the next label. First occurrence
    of a field wins. An email anywhere is captured as a fallback even without
    a label. With NO labels at all, the whole string is taken as the name
    (so "adiciona contato Lucas" → name=Lucas). Values are trimmed of trailing
    list punctuation."""
    out: Dict[str, Optional[str]] = {"name": None, "email": None, "phone": None, "notes": None}
    text = (text or "").strip()
    if not text:
        return out

    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        out["name"] = text.strip(" ,.;")
    else:
        # Anything before the first label is ignored (it's the trigger tail).
        for i, m in enumerate(matches):
            key = _canon_label(m.group("label"))
            if key is None:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start:end].strip().strip(" ,.;")
            if value and out[key] is None:
                out[key] = value

    if out["email"] is None:
        em = _EMAIL_RE.search(text)
        if em:
            out["email"] = em.group(0)
    if out["email"]:
        out["email"] = out["email"].strip().lower()
        # If the name slot accidentally swallowed the email, drop it from name.
        if out["name"]:
            out["name"] = _EMAIL_RE.sub("", out["name"]).strip(" ,.;") or None
    return out


@dataclass(frozen=True)
class Contact:
    """Row from the contacts table — a single person's address book entry."""
    id: int
    name: str
    email: str
    phone: Optional[str]
    notes: Optional[str]
    created_at: str
    updated_at: str


class ContactBook:
    """SQLite-backed name→email ledger. Thread-safe, fail-soft."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/contacts.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get(
            "CORTEX_CONTACTS_DB", self.DEFAULT_PATH,
        )
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
                CREATE TABLE IF NOT EXISTS contacts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                    email       TEXT    NOT NULL,
                    phone       TEXT,
                    notes       TEXT,
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_contacts_name
                    ON contacts(name COLLATE NOCASE);
                """
            )
            self._conn = conn
            return self._conn
        except Exception:
            log.exception("contacts: could not open ledger at %s", self.db_path)
            return None

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add(self, name: str, email: str = "", *,
            phone: Optional[str] = None,
            notes: Optional[str] = None) -> bool:
        """Insert a new contact. Only `name` is required — email is optional
        (stored as "" when absent) so a contact can be created by voice, where
        email dictation is unreliable, and filled in later via update(). Returns
        True on success, False if the name already exists or the DB is
        unavailable. Use update() for in-place edits."""
        name = (name or "").strip()
        email = (email or "").strip()
        if not name:
            return False
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                # `with conn` rolls back on a failure too: a bare commit()
                # skipped by the IntegrityError left the write lock held, and
                # the other process's writes timed out (s88, Codex lote 2).
                with conn:
                    conn.execute(
                        "INSERT INTO contacts "
                        "(name, email, phone, notes, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (name, email, phone, notes, now, now),
                    )
            return True
        except sqlite3.IntegrityError:
            # UNIQUE violation — caller should call update() instead.
            return False
        except Exception:
            log.exception("contacts: add failed (name=%s)", name)
            return False

    def update(self, name: str, *,
               email: Optional[str] = None,
               phone: Optional[str] = None,
               notes: Optional[str] = None) -> bool:
        """Update mutable fields of an existing contact (matched by name,
        case-insensitive). Returns True if a row was changed."""
        name = (name or "").strip()
        if not name:
            return False
        # Build a dynamic SET clause only for fields the caller passed.
        fields, params = [], []
        if email is not None:
            fields.append("email = ?")
            params.append(email.strip())
        if phone is not None:
            fields.append("phone = ?")
            params.append(phone.strip() or None)
        if notes is not None:
            fields.append("notes = ?")
            params.append(notes.strip() or None)
        if not fields:
            return False
        fields.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(name)
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                with conn:
                    cur = conn.execute(
                        f"UPDATE contacts SET {', '.join(fields)} "
                        f"WHERE name = ? COLLATE NOCASE",
                        params,
                    )
                return cur.rowcount > 0
        except Exception:
            log.exception("contacts: update failed (name=%s)", name)
            return False

    def delete(self, name: str) -> bool:
        """Remove a contact by name. Returns True if a row was deleted."""
        name = (name or "").strip()
        if not name:
            return False
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                with conn:
                    cur = conn.execute(
                        "DELETE FROM contacts WHERE name = ? COLLATE NOCASE",
                        (name,),
                    )
                return cur.rowcount > 0
        except Exception:
            log.exception("contacts: delete failed (name=%s)", name)
            return False

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> Optional[Contact]:
        """Exact name match (case-insensitive). Returns None if absent."""
        name = (name or "").strip()
        if not name:
            return None
        conn = self._connect()
        if conn is None:
            return None
        try:
            # M4-A14g: reads take the same lock as writes — WAL makes the
            # unlocked read benign in practice, but the class docstring
            # promises thread-safety and half a contract isn't one.
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM contacts WHERE name = ? COLLATE NOCASE",
                    (name,),
                ).fetchone()
        except Exception:
            log.exception("contacts: lookup failed (name=%s)", name)
            return None
        return _row_to_contact(row) if row else None

    def lookup_fuzzy(self, query: str, *, max_results: int = 3,
                     cutoff: float = 0.6) -> List[Contact]:
        """Approximate match for forgiving voice/typed input ("larazinha"
        → "lara"). Returns up to `max_results` contacts ranked by name
        similarity. Exact match is always the first result if it exists."""
        query = (query or "").strip()
        if not query:
            return []
        # Exact first — fast path that bypasses difflib.
        exact = self.lookup(query)
        if exact:
            return [exact]
        # Fuzzy: fetch all names, rank, then re-fetch the winners.
        conn = self._connect()
        if conn is None:
            return []
        try:
            with self._lock:  # M4-A14g — see lookup()
                names = [r["name"] for r in conn.execute(
                    "SELECT name FROM contacts"
                ).fetchall()]
        except Exception:
            log.exception("contacts: fuzzy name fetch failed")
            return []
        if not names:
            return []
        q = query.lower()
        ranked: List[str] = []
        # 1. Token match — the query is a whole word in the stored name. Fixes
        # the common case of a first name vs a full name: "Lucas" → "Lucas Lima
        # Santana", "manda email pro João" → "João Silva". Whole-word only, so
        # "ana" does NOT match "Mariana".
        for n in names:
            if q in n.lower().split():
                ranked.append(n)
        # 2. difflib fuzzy for the rest ("larazinha" → "lara").
        remaining = {n.lower(): n for n in names if n not in ranked}
        for w in get_close_matches(q, remaining.keys(), n=max_results, cutoff=cutoff):
            ranked.append(remaining[w])

        out: List[Contact] = []
        for n in ranked[:max_results]:
            c = self.lookup(n)
            if c is not None:
                out.append(c)
        return out

    def list_all(self) -> List[Contact]:
        """All contacts, alphabetical by name. Useful for /ui dashboard
        or a "show me my contacts" command."""
        conn = self._connect()
        if conn is None:
            return []
        try:
            with self._lock:  # M4-A14g — see lookup()
                rows = conn.execute(
                    "SELECT * FROM contacts ORDER BY name COLLATE NOCASE"
                ).fetchall()
        except Exception:
            log.exception("contacts: list_all failed")
            return []
        return [_row_to_contact(r) for r in rows]


def _row_to_contact(row: sqlite3.Row) -> Contact:
    return Contact(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        phone=row["phone"],
        notes=row["notes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# Module-level singleton — import and use directly, like `cost_tracker`.
contact_book = ContactBook()
