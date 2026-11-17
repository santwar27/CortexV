"""Durable projects joining conversations; progress comes only from user words.

Pure command parsers live here; no model decides a project. Transactions cover the
record and its project timestamp, so another process never sees half a write.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

_WAKE = r"(?:cortex[\s,]+)?"
_PICK = re.compile(rf"^\s*{_WAKE}(projeto|project)\s*:(.*)$", re.I | re.S)
_LIST = re.compile(rf"^\s*{_WAKE}(projetos|meus\s+projetos|projects|my\s+projects)\s*$", re.I)
_LEAVE = re.compile(rf"^\s*{_WAKE}(sair\s+do\s+projeto|leave\s+project)\s*$", re.I)
_RENAME = re.compile(
    rf"^\s*{_WAKE}((?:renomeia|renomear|renomeie)\s+o\s+projeto(?:\s+para\b|\s*:)|"
    r"renomear\s+projeto\s*:|"
    r"(?:troca|muda)\s+o\s+nome\s+do\s+projeto(?:\s+para\b|\s*:)|"
    r"rename\s+(?:the\s+)?project(?:\s+to\b|\s*:))(.*)$", re.I | re.S)
_LEAD_IN = re.compile(
    r"^(?:vamos\s+(?:criar|fazer|come[cç]ar|abrir)\s+(?:(?:um|o)\s+)?(?:novo\s+)?projeto|"
    r"quero\s+(?:criar|fazer)\s+(?:um\s+)?projeto|criar?\s+(?:um\s+)?projeto|"
    r"um\s+projeto\s+(?:de|para|pra|sobre|chamado)|"
    r"let['’]s\s+(?:create|start|make)\s+a\s+(?:new\s+)?project|"
    r"a\s+project\s+(?:for|about|called)|novo\s+projeto|new\s+project)\b", re.I)
_FRAME_SEPARATOR = re.compile(
    r"^(?:\s|[,:—–-]|\b(?:de|para|pra|sobre|chamado|called|for|about)\b)+", re.I)
_ARTICLE = re.compile(r"^(?:um|uma|o|a|an|the)\s+(\w+)\b", re.I)
# A small explicit vocabulary avoids removing the article from proper names
# such as The Queens Head. Nothing else in the remaining name is paraphrased.
_COMMON_NOUNS = frozenset((
    "website", "site", "webpage", "app", "application", "aplicacao", "aplicativo",
    "pagina", "blog", "portal", "loja", "shop", "store", "sistema", "system",
    "ferramenta", "tool", "jogo", "game", "dashboard", "pub",
))
_PROGRESS = re.compile(
    rf"^\s*{_WAKE}(feito|done|falta|todo|pr[oó]ximo|next|decidi|decided|objetivo|goal)\s*:(.*)$",
    re.I | re.S)


def parse_command(text: str) -> Optional[tuple[str, str, str]]:
    """Whole-message (action, text, language); empty values are ordinary chat."""
    progress = _PROGRESS.match(text or "")
    if progress:
        word, value = _key(progress[1]), progress[2].strip()
        action = {"done": "feito", "todo": "falta", "next": "proximo",
                  "decidi": "decisao", "decided": "decisao", "goal": "objetivo"}.get(word, word)
        return (action, value, "en" if word in (
            "done", "todo", "next", "decided", "goal") else "pt") if value else None
    pick = _PICK.match(text or "")
    if pick:
        # Keep the original through lookup: old versions stored its first 80
        # characters. Clean and cap only after checking those legacy keys.
        name = " ".join(pick[2].split())
        return ("active", name, "en" if pick[1].lower() == "project" else "pt") if name else None
    rename = _RENAME.match(text or "")
    if rename:
        name = " ".join(rename[2].split())[:80]
        return ("renamed", name, "en" if rename[1].lower().startswith("rename") else "pt") if name else None
    for pattern, action in ((_LIST, "list"), (_LEAVE, "cleared")):
        match = pattern.match(text or "")
        if match:
            en = match[1].lower().startswith(("projects", "my", "leave"))
            return action, "", "en" if en else "pt"
    return None


@dataclass(frozen=True)
class Project:
    id: int
    name: str
    name_key: str
    objective: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Entry:
    id: int
    project_id: int
    kind: str
    text: str
    channel: Optional[str]
    conversation_id: Optional[str]
    source: str
    created_at: datetime
    closed_at: Optional[datetime]


@dataclass(frozen=True)
class Observation:
    id: int
    project_id: int
    kind: str
    tool: str
    reference: str
    channel: Optional[str]
    conversation_id: Optional[str]
    created_at: datetime


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(name: str) -> str:
    return " ".join("".join(c for c in unicodedata.normalize("NFD", name.lower())
                            if not unicodedata.combining(c)).split())


def _clean_name(name: str) -> str:
    """Remove only an anchored frame and a common-noun article; keep case."""
    remaining = " ".join(unicodedata.normalize("NFC", name).split())
    frame = _LEAD_IN.match(remaining)
    changed = frame is not None
    if frame:
        remaining = _FRAME_SEPARATOR.sub("", remaining[frame.end():]).strip()
    article = _ARTICLE.match(remaining)
    if article and _key(article[1]) in _COMMON_NOUNS:
        remaining = remaining[article.start(1):]
        changed = True
    # Preserve literal plain names (including the store's whitespace contract).
    return remaining if remaining and changed else name


def _record(cls, row):
    values = dict(row)
    for name in ("created_at", "updated_at", "closed_at"):
        if values.get(name) is not None:
            values[name] = datetime.fromisoformat(values[name])
    return cls(**values)


class ProjectStore:
    """Lazy WAL record, shared by server and terminal. Errors are fail-soft.

    The lock serialises threads, including first use; `with conn:` commits
    or rolls back each write across processes. Conversation IDs belong to
    chat.db, so only project references have local foreign keys.
    """

    DEFAULT_PATH = os.path.expanduser("~/.cortex/projects.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_PROJECTS_DB", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        if self._conn is not None:
            return self._conn
        conn = None
        try:
            parent = os.path.dirname(self.db_path)
            if parent and self.db_path != ":memory:":
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            with conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA foreign_keys = ON")
                for statement in (
                    """CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                        name_key TEXT NOT NULL UNIQUE, objective TEXT,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS conversation_projects (
                        conversation_id TEXT PRIMARY KEY NOT NULL,
                        project_id INTEGER NOT NULL REFERENCES projects(id))""",
                    """CREATE TABLE IF NOT EXISTS entries (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL REFERENCES projects(id),
                        kind TEXT NOT NULL CHECK(kind IN ('feito','falta','proximo','decisao')),
                        text TEXT NOT NULL CHECK(length(trim(text)) BETWEEN 1 AND 500),
                        channel TEXT CHECK(channel IN ('ui','voice','terminal','network')),
                        conversation_id TEXT,
                        source TEXT NOT NULL CHECK(source = 'user_said'),
                        created_at TEXT NOT NULL, closed_at TEXT)""",
                    """CREATE TABLE IF NOT EXISTS active_project (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        project_id INTEGER NOT NULL REFERENCES projects(id),
                        selected_at TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS observations (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL REFERENCES projects(id),
                        kind TEXT NOT NULL CHECK(kind IN ('read','written','action')),
                        tool TEXT NOT NULL,
                        reference TEXT NOT NULL CHECK(length(trim(reference)) BETWEEN 1 AND 150),
                        channel TEXT CHECK(channel IN ('ui','voice','terminal','network')),
                        conversation_id TEXT, created_at TEXT NOT NULL)""",
                ):
                    conn.execute(statement)
            self._conn = conn
            return conn
        except Exception:
            if conn is not None:
                conn.close()
            log.exception("projects: could not open the record at %s", self.db_path)
            return None

    def _run(self, action, default=None, write=False):
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return default
                with conn:
                    # Reserve before reading the old link/active project: another
                    # process must not move it between our read and write.
                    if write:
                        conn.execute("BEGIN IMMEDIATE")
                    return action(conn)
        except Exception:
            log.exception("projects: record operation failed")
            return default

    def _read(self, sql, args=(), cls=Project, many=False):
        def read(conn):
            rows = conn.execute(sql, args).fetchall()
            values = [_record(cls, r) if cls else r[0] for r in rows]
            return values if many else (values[0] if values else None)
        return self._run(read, [] if many else None)

    def available(self) -> bool:
        # Empty reads also mean failure; commands must not report an empty DB.
        return self._run(lambda c: bool(c.execute("SELECT 1").fetchone()), False)

    def _touch(self, conn, ids):
        for pid in set(ids) - {None}:
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), pid))

    def create_or_get(self, nome: str, objetivo=None) -> Optional[Project]:
        if not _key(nome):
            return None
        def create(conn):
            existing = self._find_name(conn, nome)
            if existing is not None:
                return _record(Project, existing)
            name = _clean_name(nome)[:80]
            now = _now()
            conn.execute("INSERT INTO projects (name, name_key, objective, created_at, updated_at) "
                         "VALUES (?, ?, ?, ?, ?) ON CONFLICT(name_key) DO NOTHING",
                         (name, _key(name), objetivo, now, now))
            return _record(Project, conn.execute(
                "SELECT * FROM projects WHERE name_key = ?", (_key(name),)).fetchone())
        return self._run(create, write=True)

    def set_objective(self, projeto: int, objetivo: Optional[str]) -> bool:
        return self._run(lambda c: c.execute(
            "UPDATE projects SET objective = ?, updated_at = ? WHERE id = ?",
            (objetivo, _now(), projeto)).rowcount == 1, False, write=True)

    def get(self, projeto: int) -> Optional[Project]:
        return self._read("SELECT * FROM projects WHERE id = ?", (projeto,))

    def delete_project(self, project_id: int) -> Optional[list[str]]:
        """Erase the project record atomically; return its former chat links.

        Chat history and Palace documents belong to separate stores. The UI
        removes those first, retaining these links if either operation fails.
        """
        def delete(conn):
            if conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                return None
            conversations = [r[0] for r in conn.execute(
                "SELECT conversation_id FROM conversation_projects "
                "WHERE project_id = ? ORDER BY conversation_id", (project_id,))]
            for table in ("entries", "observations", "conversation_projects", "active_project"):
                conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return conversations
        return self._run(delete, write=True)

    def find(self, nome: str) -> Optional[Project]:
        def find(conn):
            row = self._find_name(conn, nome)
            return _record(Project, row) if row is not None else None
        return self._run(find)

    @staticmethod
    def _find_name(conn, nome):
        # Prefer the literal record if both names already exist. Never migrate
        # or rename old records just because this cleaner was introduced.
        for key in dict.fromkeys((_key(nome), _key(" ".join(nome.split())[:80]),
                                  _key(_clean_name(nome)[:80]))):
            row = conn.execute("SELECT * FROM projects WHERE name_key = ?", (key,)).fetchone()
            if row is not None:
                return row
        return None

    def rename_active(self, nome: str) -> Optional[tuple[str, str, str]]:
        """Resolve active, check uniqueness and rename under one write lock.

        Return (status, old name, new name); None means a failed operation.
        Renames are literal: the conversational cleaner only applies to picks.
        """
        name = " ".join(nome.split())[:80]
        if not name:
            return None
        def rename(conn):
            old = conn.execute("SELECT p.* FROM projects p JOIN active_project a "
                               "ON p.id = a.project_id").fetchone()
            if old is None:
                return "none", "", name
            taken = conn.execute("SELECT id FROM projects WHERE name_key = ? AND id != ?",
                                 (_key(name), old["id"])).fetchone()
            if taken is not None:
                return "taken", old["name"], name
            conn.execute("UPDATE projects SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                         (name, _key(name), _now(), old["id"]))
            return "renamed", old["name"], name
        return self._run(rename, write=True)

    def list(self) -> list[Project]:
        return self._read("SELECT * FROM projects ORDER BY updated_at DESC, id DESC", many=True)

    def link_conversation(self, conversation_id: str, projeto: int) -> bool:
        """Move a conversation; touch both projects when it leaves one."""
        def link(conn):
            old = conn.execute("SELECT project_id FROM conversation_projects WHERE conversation_id = ?",
                               (conversation_id,)).fetchone()
            conn.execute("INSERT INTO conversation_projects VALUES (?, ?) "
                         "ON CONFLICT(conversation_id) DO UPDATE SET project_id = excluded.project_id",
                         (conversation_id, projeto))
            self._touch(conn, [projeto, old[0] if old else None])
            return True
        return self._run(link, False, write=True)

    def project_for_conversation(self, conversation_id: str) -> Optional[Project]:
        return self._read("SELECT p.* FROM projects p JOIN conversation_projects c "
                          "ON p.id = c.project_id WHERE c.conversation_id = ?", (conversation_id,))

    def progress_project(self, conversa=None):
        """Resolve progress's target; False distinguishes a failed read from no project."""
        def resolve(conn):
            row = conn.execute("SELECT p.* FROM projects p JOIN conversation_projects c "
                               "ON p.id = c.project_id WHERE c.conversation_id = ?", (conversa,)).fetchone()
            if row is None:
                row = conn.execute("SELECT p.* FROM projects p JOIN active_project a "
                                   "ON p.id = a.project_id").fetchone()
            return _record(Project, row) if row is not None else None
        return self._run(resolve, False)

    def conversations(self, projeto: int, *, strict: bool = False) -> Optional[list[str]]:
        # Erasure must distinguish a failed read from a project with no history.
        return self._run(lambda c: [r[0] for r in c.execute(
            "SELECT conversation_id FROM conversation_projects "
            "WHERE project_id = ? ORDER BY conversation_id", (projeto,))], None if strict else [])

    def add_entry(self, projeto: int, tipo: str, texto: str, canal=None,
                  conversa=None, fonte="user_said") -> Optional[Entry]:
        """Keep literal words, capped rather than paraphrased; no inferred source."""
        if (tipo not in ("feito", "falta", "proximo", "decisao") or not texto.strip()
                or not texto[:500].strip() or fonte != "user_said"
                or canal not in (None, "ui", "voice", "terminal", "network")):
            return None
        def add(conn):
            row = conn.execute(
                "INSERT INTO entries (project_id, kind, text, channel, conversation_id, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (projeto, tipo, texto[:500], canal, conversa, fonte, _now())).fetchone()
            self._touch(conn, [projeto])
            return _record(Entry, row)
        return self._run(add, write=True)

    def entries(self, projeto: int, tipo=None, abertas_apenas=False, limite=50) -> list[Entry]:
        return self._read("SELECT * FROM entries WHERE project_id = ? "
                          "AND (? IS NULL OR kind = ?) AND (? = 0 OR closed_at IS NULL) "
                          "ORDER BY created_at DESC, id DESC LIMIT ?",
                          (projeto, tipo, tipo, abertas_apenas, max(0, limite)), Entry, True)

    def observe(self, projeto: int, tipo: str, ferramenta: str, referencia: str,
                canal=None, conversa=None) -> Optional[Observation]:
        """Tool success only; separate from user progress and never an approval."""
        reference = " ".join(referencia.split())[:150]
        if (tipo not in ("read", "written", "action") or not reference or not ferramenta.strip()
                or canal not in (None, "ui", "voice", "terminal", "network")):
            return None
        def add(conn):
            row = conn.execute(
                "INSERT INTO observations (project_id, kind, tool, reference, channel, "
                "conversation_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (projeto, tipo, ferramenta, reference, canal, conversa, _now())).fetchone()
            self._touch(conn, [projeto])
            return _record(Observation, row)
        return self._run(add, write=True)

    def observations(self, projeto: int, limite=50) -> list[Observation]:
        return self._read("SELECT * FROM observations WHERE project_id = ? "
                          "ORDER BY created_at DESC, id DESC LIMIT ?",
                          (projeto, -1 if limite is None else max(0, limite)), Observation, True)

    def where_entries(self, projeto: int) -> Optional[list[Entry]]:
        """All progress, newest first; None means failure, never an empty project."""
        return self._run(lambda c: [_record(Entry, r) for r in c.execute(
            "SELECT * FROM entries WHERE project_id = ? ORDER BY created_at DESC, id DESC",
            (projeto,)).fetchall()])

    def note_progress(self, projeto: int, tipo: str, texto: str, canal=None, conversa=None):
        """Add, deduplicate and close progress atomically; return (entry, off_todo)."""
        if (tipo not in ("feito", "falta", "proximo", "decisao") or not texto.strip()
                or not texto[:500].strip() or canal not in (None, "ui", "voice", "terminal", "network")):
            return None
        def note(conn):
            rows = conn.execute("SELECT * FROM entries WHERE project_id = ? "
                                "ORDER BY created_at DESC, id DESC", (projeto,)).fetchall()
            latest = next((r for r in rows if r["kind"] == tipo and r["closed_at"] is None), None)
            duplicate = latest is not None and latest["text"] == texto[:500]
            now = _now()
            off_todo = False
            for row in rows:
                matching = tipo == "feito" and row["kind"] == "falta" and _key(row["text"]) == _key(texto[:500])
                if (matching and duplicate and row["closed_at"] is not None
                        and row["closed_at"] >= latest["created_at"]):
                    off_todo = True  # An exact retry keeps the same reply.
                if row["closed_at"] is None and (matching or (
                        tipo == "proximo" and row["kind"] == "proximo"
                        and (not duplicate or row["id"] != latest["id"]))):
                    conn.execute("UPDATE entries SET closed_at = ? WHERE id = ?", (now, row["id"]))
                    self._touch(conn, [projeto])
                    off_todo = off_todo or matching
            if duplicate:
                return _record(Entry, latest), off_todo
            row = conn.execute(
                "INSERT INTO entries (project_id, kind, text, channel, conversation_id, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'user_said', ?) RETURNING *",
                (projeto, tipo, texto[:500], canal, conversa, now)).fetchone()
            self._touch(conn, [projeto])
            return _record(Entry, row), off_todo
        return self._run(note, write=True)

    def close_entry(self, entry_id: int) -> bool:
        def close(conn):
            row = conn.execute("UPDATE entries SET closed_at = ? WHERE id = ? "
                               "AND closed_at IS NULL RETURNING project_id", (_now(), entry_id)).fetchone()
            self._touch(conn, [row[0]] if row else [])
            return row is not None
        return self._run(close, False, write=True)

    def set_active(self, projeto: int) -> bool:
        def activate(conn):
            old = conn.execute("SELECT project_id FROM active_project").fetchone()
            conn.execute("INSERT INTO active_project VALUES (1, ?, ?) "
                         "ON CONFLICT(singleton) DO UPDATE SET project_id = excluded.project_id, "
                         "selected_at = excluded.selected_at", (projeto, _now()))
            self._touch(conn, [projeto, old[0] if old else None])
            return True
        return self._run(activate, False, write=True)

    def active(self) -> Optional[Project]:
        return self._read("SELECT p.* FROM projects p JOIN active_project a ON p.id = a.project_id")

    def clear_active(self) -> bool:
        def clear(conn):
            row = conn.execute("DELETE FROM active_project RETURNING project_id").fetchone()
            self._touch(conn, [row[0]] if row else [])
            return row is not None
        return self._run(clear, False, write=True)


store = ProjectStore()
