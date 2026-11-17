"""
Persistent append-only queue of study topics for the Nightly Studier.

Format: one JSON object per line (.jsonl), survives reboot, easy to
inspect manually with `cat data/study_queue.jsonl`. Each entry has a
status field ("pending" | "consumed") so we can atomically mark items
as processed without losing the history.
"""

import fcntl
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional
from uuid import uuid4


class StudyTopicRejected(ValueError):
    """Tagged queue rejection with language-neutral details for callers."""

    def __init__(self, reason: str, topic: str, word: str = "", *,
                 existing: str = "", status: Optional[str] = None):
        self.reason, self.topic, self.word = reason, topic, word
        self.existing, self.status = existing, status
        super().__init__(f"{reason}_TOPIC: {topic}")


class StudyQueue:
    """Append-only JSONL queue of study topics."""

    # Transient topics that become stale within days. RAG knowledge should
    # be stable content; time-sensitive queries belong to the live search
    # path (/ask with Tavily). Users can force with trailing "!!" in the
    # topic string.
    _TRANSIENT_RE = re.compile(
        r"\b(?:"
        r"current affairs|current events|breaking news|latest news|"
        r"noticias? (?:de )?hoje|noticias? atuais|"
        r"weather|clima|previs[ãa]o do tempo|temperatura|"
        r"pre[çc]o|cota[çc][ãa]o|exchange rate|stock price|"
        r"resultado de|score|placar|"
        r"this week|esta semana|hoje|today|"
        r"trending|viral"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, path: str = "data/study_queue.jsonl"):
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            Path(self.path).touch()

    @contextmanager
    def _locked(self):
        """Held, between PROCESSES, by every writer of the queue (s88, Codex
        lote 14): consume_all and cancel read the file, decide, and replace
        it whole, so an add() appended in between — the API answering
        "Anotado" while the nightly worked — vanished with the replace
        (reproduced by pausing consume_all after its read). The replace is
        atomic; the read-decide-write was not. Same pattern as the gate
        queue's _decision_lock. Fail-open on a lock error."""
        try:
            fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # releases the flock

    # Fuzzy-match threshold for duplicate detection. SequenceMatcher
    # ratio of 0.80 catches 'TensorFlow' vs 'tensorflow', 'RAG vs
    # finetuning' vs 'RAG vs fine-tuning', but accepts genuinely
    # different topics ('PyTorch attention' vs 'TensorFlow attention').
    _DUP_RATIO_THRESHOLD = 0.80

    @staticmethod
    def _normalise_for_match(s: str) -> str:
        """Lowercase + strip non-alnum so 'RAG vs fine-tuning' matches
        'RAG vs finetuning' matches 'rag-vs-fine-tuning'."""
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def find_similar(self, topic: str) -> Optional[Dict]:
        """
        Return the most similar existing entry (pending or already
        consumed) that crosses the dup threshold, or None. Used by add()
        for soft-warning duplicate detection.

        CANCELLED entries are skipped (s83): taking a topic off the queue and
        then being told it is a duplicate of itself would make changing your
        mind a one-way door.
        """
        target = self._normalise_for_match(topic)
        if not target:
            return None
        best: Optional[Dict] = None
        best_ratio = 0.0
        for entry in self._read_all():
            if entry.get("status") == "cancelled":
                continue
            existing = self._normalise_for_match(entry.get("topic", ""))
            if not existing:
                continue
            # Cheap substring shortcut for the obvious 'tensorflow'
            # contained-in 'tensorflow basics' case.
            if target in existing or existing in target:
                return entry
            ratio = SequenceMatcher(None, target, existing).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = entry
        return best if best_ratio >= self._DUP_RATIO_THRESHOLD else None

    def add(self, topic: str, extra: Optional[Dict] = None) -> dict:
        """Append a new pending topic. Returns the created entry.

        `extra` merges additional JSON-serializable metadata into the
        entry (e.g. provenance from the CuriosityEngine: origin,
        parent_topic, score). Core fields (topic/status/added_at) always
        win over extra keys.

        Raises ValueError if the topic is:
        - transient (news, weather, prices, scores) — error tag
          TRANSIENT_TOPIC. Append '!!' to force.
        - a near-duplicate of an existing pending or consumed topic —
          error tag DUPLICATE_TOPIC. Append '!!' to force.

        The '!!' force flag bypasses BOTH checks at once.
        """
        topic = (topic or "").strip()
        if not topic:
            raise ValueError("Study topic cannot be empty.")

        # Force flag: trailing '!!' bypasses both transient and dup checks.
        force = topic.endswith("!!")
        if force:
            topic = topic[:-2].strip()
            if not topic:
                raise ValueError("Study topic cannot be empty.")

        transient = self._TRANSIENT_RE.search(topic)
        if not force and transient:
            raise StudyTopicRejected("TRANSIENT", topic, transient.group())

        with self._locked():
            return self._add_locked(topic, force, extra)

    def _add_locked(self, topic: str, force: bool, extra: Optional[Dict]) -> dict:
        if not force:
            similar = self.find_similar(topic)
            if similar:
                raise StudyTopicRejected(
                    "DUPLICATE", topic, existing=similar["topic"], status=similar.get("status"))

        entry = dict(extra) if extra else {}
        entry.update(
            {
                "id": uuid4().hex,
                "topic": topic,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
            }
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def list_pending(self) -> List[Dict]:
        """Return all entries currently marked as pending."""
        return [e for e in self._read_all() if e.get("status") == "pending"]

    def pending_ids(self) -> List[str]:
        """Snapshot tonight's work, giving legacy entries persistent ids."""
        with self._locked():
            entries = self._read_all()
            if any(not e.get("id") for e in entries):
                self._rewrite_atomic(entries)
            return [e["id"] for e in entries if e.get("status") == "pending"]

    def transition(self, entry_id: str, expected: tuple, status: str,
                   **fields) -> Optional[Dict]:
        """Change one entry only if its current state still permits it."""
        with self._locked():
            entries = self._read_all()
            for entry in entries:
                if entry.get("id") == entry_id and entry.get("status") in expected:
                    entry.update(fields)
                    entry["status"] = status
                    self._rewrite_atomic(entries)
                    return entry
        return None

    def recover_interrupted(self) -> List[Dict]:
        """Recover dead-run entries. Caller must hold the separate run lock."""
        with self._locked():
            entries = self._read_all()
            recovered = []
            now = datetime.now(timezone.utc).isoformat()
            for entry in entries:
                if entry.get("status") not in ("studying", "saving"):
                    continue
                briefing = entry.get("briefing")
                if (entry["status"] == "saving" and briefing
                        and Path(briefing).is_file() and Path(briefing).stat().st_size > 0):
                    entry.update(status="consumed", consumed_at=now)
                else:
                    retries = int(entry.get("retries") or 0)
                    entry.update(status="failed" if retries >= 1 else "pending",
                                 retries=retries if retries >= 1 else retries + 1,
                                 interrupted_at=now)
                    entry.pop("briefing", None)
                recovered.append(entry)
            if recovered:
                self._rewrite_atomic(entries)
            return recovered

    def cancel(self, indexes: Optional[List[int]] = None) -> List[Dict]:
        """Mark pending entries as 'cancelled' and return what was cancelled.

        `indexes` are 1-based positions as `list_pending()` (and therefore the
        reply the user just read) numbers them; None cancels every pending
        entry. Added s83: the queue could be added to, listed, shown and
        approved, but never emptied — Pedro changed his mind about three
        topics, tried five phrasings, and got the 8B claiming it had tidied
        the queue without touching it.

        A status, not a deletion: the line stays, so the history is intact and
        `find_similar` no longer blocks re-adding the same topic (it only
        looks at pending and consumed)."""
        with self._locked():
            return self._cancel_locked(indexes)

    def _cancel_locked(self, indexes: Optional[List[int]]) -> List[Dict]:
        entries = self._read_all()
        pending = [e for e in entries if e.get("status") == "pending"]
        if indexes is None:
            targets = pending
        else:
            targets = []
            for i in indexes:
                if 1 <= i <= len(pending):
                    target = pending[i - 1]
                    if target not in targets:
                        targets.append(target)
        if not targets:
            return []
        now = datetime.now(timezone.utc).isoformat()
        for entry in targets:
            entry["status"] = "cancelled"
            entry["cancelled_at"] = now
        self._rewrite_atomic(entries)
        return targets

    def consume_all(self) -> List[Dict]:
        """
        Atomically mark every pending entry as 'consumed' and return the
        snapshot of what was consumed. Used by the Nightly Studier when
        it processes the queue.
        """
        with self._locked():
            return self._consume_locked()

    def _consume_locked(self) -> List[Dict]:
        entries = self._read_all()
        now = datetime.now(timezone.utc).isoformat()
        consumed: List[Dict] = []

        for e in entries:
            if e.get("status") == "pending":
                e["status"] = "consumed"
                e["consumed_at"] = now
                consumed.append(e)

        if consumed:
            self._rewrite_atomic(entries)
        return consumed

    def _read_all(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        results: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines — never fail the whole queue
                    continue
        return results

    def _rewrite_atomic(self, entries: List[Dict]) -> None:
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in entries:
                if not e.get("id"):
                    e["id"] = uuid4().hex
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.path)
