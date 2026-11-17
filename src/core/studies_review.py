"""
Review layer for the Nightly Studier briefings.

Lists, shows, approves, and rejects pending Markdown briefings sitting
in data/knowledge/pending/. Approved files move to data/knowledge/approved/
(indexed into the RAG knowledge base in Fase E). Rejected files are
deleted.

Lookup is by 1-based index (the order returned by list_pending) or by
case-insensitive substring match against the topic. Special keyword
"todos" / "all" applies the action to every pending briefing.
"""

import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from src.core.audit import audit
from src.core.knowledge_base import KnowledgeBase

log = logging.getLogger(__name__)

PENDING_DIR_DEFAULT = "data/knowledge/pending"
APPROVED_DIR_DEFAULT = "data/knowledge/approved"


class IndexFailed(Exception):
    """The briefing could not be stored in the Palace; it was put back in
    pending/, so approving it again retries."""


class StudiesReview:
    """Inspect and triage Nightly Studier briefings."""

    # Regex for extracting metadata from the briefing markdown.
    _TOPIC_RE = re.compile(r"^#\s*Topic:\s*(.+)$", re.MULTILINE)
    _CONFIDENCE_RE = re.compile(
        r"(?:##\s*Confian[çc]a|\*\*Confian[çc]a\*\*)\s*\n+\s*(alta|m[ée]dia|baixa|high|medium|low)",
        re.IGNORECASE,
    )

    # Ordinals (PT-BR + UK English) that map to a 1-based study index.
    # Lets the user say "mostrar o primeiro estudo" / "approve second" etc.
    _ORDINALS = {
        # PT-BR
        "primeiro": 1, "primeira": 1, "1o": 1, "1a": 1, "1º": 1, "1ª": 1,
        "segundo": 2, "segunda": 2, "2o": 2, "2a": 2, "2º": 2, "2ª": 2,
        "terceiro": 3, "terceira": 3, "3o": 3, "3a": 3, "3º": 3, "3ª": 3,
        "quarto": 4, "quarta": 4, "4º": 4, "4ª": 4,
        "quinto": 5, "quinta": 5, "5º": 5, "5ª": 5,
        "sexto": 6, "sexta": 6,
        "sétimo": 7, "sétima": 7, "setimo": 7, "setima": 7,
        "oitavo": 8, "oitava": 8,
        "nono": 9, "nona": 9,
        "décimo": 10, "décima": 10, "decimo": 10, "decima": 10,
        # UK English
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }

    # Words that may stand around a position without being part of a title:
    # "o segundo estudo", "the second one", "estudo número 2".
    _POSITION_FILLER = {"o", "a", "the", "estudo", "study", "one",
                        "número", "numero", "nº"}

    def __init__(
        self,
        pending_dir: str = PENDING_DIR_DEFAULT,
        approved_dir: str = APPROVED_DIR_DEFAULT,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.pending_dir = pending_dir
        self.approved_dir = approved_dir
        os.makedirs(self.pending_dir, exist_ok=True)
        os.makedirs(self.approved_dir, exist_ok=True)
        # Index into the knowledge base on approve. Lazy: if not injected,
        # we instantiate one pointing at the same data dir.
        self.knowledge_base = knowledge_base or KnowledgeBase(
            approved_dir=self.approved_dir
        )

    # --- listing & lookup -------------------------------------------

    def list_pending(self) -> List[Dict]:
        """Return briefings in pending/, oldest first, with extracted metadata."""
        if not os.path.isdir(self.pending_dir):
            return []
        files = sorted(
            (f for f in os.listdir(self.pending_dir) if f.endswith(".md")),
            key=lambda f: os.path.getmtime(os.path.join(self.pending_dir, f)),
        )
        items: List[Dict] = []
        for i, filename in enumerate(files, start=1):
            path = os.path.join(self.pending_dir, filename)
            content = self._read(path)
            items.append({
                "index": i,
                "filename": filename,
                "path": path,
                "topic": self._extract_topic(content) or filename,
                "confidence": self._extract_confidence(content) or "?",
            })
        return items

    # "## Resumo executivo" up to the next heading — the part worth reading
    # aloud when asked what was learned.
    _SUMMARY_RE = re.compile(
        r"##\s*(?:Resumo executivo|Executive summary)\s*\n+(.+?)(?=\n##\s|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    def last_study(self) -> Optional[Dict]:
        """The most recent briefing, pending OR approved, by file time (s83).

        Asked "o que você aprendeu com seu último estudo?", Cortex answered
        about SOLID and FastAPI — old approved knowledge — because there was
        no route and the 8B improvised from memory. The answer has to come
        from the file, and it has to say whether the thing is still waiting
        for review, because an unreviewed briefing is not knowledge yet."""
        newest: Optional[Dict] = None
        for state, folder in (("pending", self.pending_dir),
                              ("approved", self.approved_dir)):
            if not os.path.isdir(folder):
                continue
            for filename in os.listdir(folder):
                if not filename.endswith(".md"):
                    continue
                path = os.path.join(folder, filename)
                try:
                    when = os.path.getmtime(path)
                except OSError:
                    continue
                if newest is None or when > newest["mtime"]:
                    newest = {"state": state, "path": path,
                              "filename": filename, "mtime": when}
        if newest is None:
            return None
        content = self._read(newest["path"])
        summary = self._SUMMARY_RE.search(content)
        newest.update({
            "topic": self._extract_topic(content) or newest["filename"],
            "confidence": self._extract_confidence(content) or "?",
            "summary": (summary.group(1).strip() if summary else "").strip(),
        })
        return newest

    def find(self, ref: str) -> Optional[Dict]:
        """
        Resolve a user-supplied reference to one pending entry. Accepts:
          - a 1-based digit ("1", "2"...)
          - an ordinal in PT-BR or UK English ("primeiro", "second", "1º"...)
          - free-text substring matched against topic or filename
            ("brasil", "tensorflow"...)
        """
        ref = (ref or "").strip()
        if not ref:
            return None

        items = self.list_pending()
        if not items:
            return None

        # 1. Numeric index (1-based)
        if ref.isdigit():
            idx = int(ref)
            return next((it for it in items if it["index"] == idx), None)

        # 2. A position — an ordinal or a number on its own, or with only
        #    articles and "estudo" around it ("o primeiro estudo", "the second
        #    one"). Any ordinal word anywhere used to count: "aprova Segunda
        #    Guerra Mundial" approved study nº 2 (s88, Codex lote 14).
        ref_lower = ref.lower()
        ordinal_idx = self._position(ref_lower)
        if ordinal_idx is not None:
            return next((it for it in items if it["index"] == ordinal_idx), None)

        # 3. Substring match (case-insensitive) against topic or filename
        for it in items:
            if ref_lower in it["topic"].lower() or ref_lower in it["filename"].lower():
                return it
        return None

    def _position(self, ref_lower: str) -> Optional[int]:
        """The 1-based index a reference names by position, or None when it
        is something else (a title)."""
        words = [w for w in ref_lower.split() if w not in self._POSITION_FILLER]
        if len(words) != 1:
            return None
        word = words[0]
        if word.isdigit():
            return int(word)
        return self._ORDINALS.get(word)

    def get_briefing(self, ref: str) -> Optional[Tuple[Dict, str]]:
        """Return (metadata, full_markdown_content) for the referenced briefing."""
        item = self.find(ref)
        if not item:
            return None
        content = self._read(item["path"])
        return (item, content)

    # --- mutating actions -------------------------------------------

    def reconcile_approved(self) -> List[str]:
        """Complete interrupted approvals, leaving failed files for a retry."""
        completed: List[str] = []
        incomplete = 0
        try:
            filenames = sorted(os.listdir(self.approved_dir))
        except Exception:
            log.exception("Studies reconciliation: cannot list approved files (non-fatal)")
            return completed
        for filename in filenames:
            try:
                path = os.path.join(self.approved_dir, filename)
                if not filename.endswith(".md") or not os.path.isfile(path):
                    continue
                if self.knowledge_base.is_fully_indexed(filename):
                    continue
                if (self.knowledge_base.index_file(path)
                        and self.knowledge_base.is_fully_indexed(filename)):
                    completed.append(filename)
                else:
                    incomplete += 1
            except Exception:
                incomplete += 1
                log.exception("Studies reconciliation: file could not be completed (non-fatal)")
        for event, count in (("approval_completed", len(completed)),
                             ("approval_incomplete", incomplete)):
            if count:
                try:
                    audit.log("–", "studies", event, payload={"count": count})
                except Exception:
                    log.exception("Studies reconciliation: audit failed (non-fatal)")
        return completed

    def approve(self, ref: str) -> Optional[str]:
        """
        Move pending → approved AND index the briefing into the knowledge
        base. Returns the new path on success, None when there is no such
        briefing; raises IndexFailed when it could not be indexed.
        """
        self.reconcile_approved()
        item = self.find(ref)
        if not item:
            return None
        return self._approve_path(item["path"])

    def approve_item(self, item: Dict) -> Optional[str]:
        """Approve the briefing `item` names — the file itself, resolved
        once. Re-resolving the reference could land on another study: two
        processes approving "1" approved B and announced A (s88, Codex
        lote 14). None when that file already left pending/."""
        self.reconcile_approved()
        return self._approve_path(item["path"])

    def _approve_path(self, src_path: str) -> Optional[str]:
        """Approved means in the Palace (s88, Codex lote 14 — invariant 7).
        A failed index was only logged: the file left pending/ and Pedro
        heard "agora faz parte da memória permanente" with nothing — or a
        part — in the Palace. On failure the partial rows are dropped and the
        file goes back to pending/, and IndexFailed says why."""
        new_path = self._move_to_approved(src_path)
        if not new_path:
            return None
        error: Optional[Exception] = None
        try:
            indexed = bool(self.knowledge_base.index_file(new_path))
        except Exception as e:  # noqa: BLE001 — reported to the caller below
            indexed, error = False, e
        if indexed:
            return new_path
        log.warning(f"approve(): knowledge indexing failed for {new_path}: {error}")
        self.knowledge_base.remove_file(os.path.basename(new_path))  # fail-soft
        try:
            shutil.move(new_path, src_path)
        except OSError as e:
            log.warning(f"approve(): could not put {new_path} back in pending: {e}")
        raise IndexFailed(str(error) if error else "nothing was indexed")

    def reject(self, ref: str) -> Optional[str]:
        """Delete a pending briefing. Returns the deleted filename or None."""
        item = self.find(ref)
        if not item:
            return None
        return self.reject_item(item)

    def reject_item(self, item: Dict) -> Optional[str]:
        """Delete the briefing `item` names (resolved once, like
        approve_item). None when it already left pending/."""
        try:
            os.remove(item["path"])
        except OSError:
            return None
        return item["filename"]

    def approve_all(self) -> Tuple[List[str], List[str]]:
        """Approve every pending briefing and index each into the KB.
        Returns (approved paths, topics that could not be indexed — those
        stay pending)."""
        self.reconcile_approved()
        moved: List[str] = []
        failed: List[str] = []
        for it in self.list_pending():
            try:
                new_path = self._approve_path(it["path"])
            except IndexFailed:
                failed.append(it["topic"])
                continue
            if new_path:
                moved.append(new_path)
        return moved, failed

    def reject_all(self) -> List[str]:
        """Delete every pending briefing. Returns list of deleted filenames."""
        deleted: List[str] = []
        for it in self.list_pending():
            try:
                os.remove(it["path"])
                deleted.append(it["filename"])
            except OSError:
                continue
        return deleted

    # --- helpers ----------------------------------------------------

    def _move_to_approved(self, src_path: str) -> Optional[str]:
        if not os.path.isfile(src_path):
            return None
        filename = os.path.basename(src_path)
        dst_path = os.path.join(self.approved_dir, filename)
        # If a same-named approved file already exists, suffix with a counter.
        counter = 2
        while os.path.exists(dst_path):
            base, ext = os.path.splitext(filename)
            dst_path = os.path.join(self.approved_dir, f"{base}_v{counter}{ext}")
            counter += 1
        try:
            shutil.move(src_path, dst_path)
        except FileNotFoundError:
            return None  # another process approved or rejected it first
        return dst_path

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    @classmethod
    def _extract_topic(cls, content: str) -> Optional[str]:
        m = cls._TOPIC_RE.search(content)
        return m.group(1).strip() if m else None

    @classmethod
    def _extract_confidence(cls, content: str) -> Optional[str]:
        m = cls._CONFIDENCE_RE.search(content)
        return m.group(1).strip().lower() if m else None
