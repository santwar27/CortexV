"""
Gate batch queue — background-safe approval queue for destructive actions.

When Cortex acts in the foreground, a destructive (S3) tool the LLM picks
is staged in the single inline `_pending_action` slot and the human says
"sim"/"não" on the next turn (see `staging_mixin.py`). That slot can't
work for **background** work (a nightly scheduled task, a future Dream
Loop): at 3am there's nobody to answer, and a background `process()` would
also clobber whatever the interactive user has staged.

This is the queue those background S3 actions land in instead. It's an
**append-only JSONL event log** at `~/.cortex/gate_queue.jsonl` (outside
the repo, where the agent has write access — same convention as
`audit.py`/`cost_tracker.py`, design-reference ch. 13). One line per event:

    {"id", "ts", "event", ...}

with `event` in: `queued` (carries type/args/reason/origin/trace_id),
`approved`, `rejected`, `executed`, `failed`, `unknown`. State is derived by
**folding** the events per id — `pending()` is every id that was queued
and not yet decided (no `approved`/`rejected`). Nothing auto-executes:
fail-closed, the human reviews via `python -m src.gate_review`.

Design mirrors `audit.py`: a class with an injectable path plus a
module-level singleton. The queue is the source of truth, so writes are
serialised under a lock; a write failure means the action is simply not
queued — which is fail-closed (nothing runs), the safe direction.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.audit import audit, chmod_600
from src.core.redaction import redact_payload

# Events that close the human-decision step. An id with `queued` but
# neither of these is still awaiting review → pending().
_DECISION_EVENTS = ("approved", "rejected")
# All recognised event names (for validation / docs).
QUEUED, APPROVED, REJECTED, EXECUTED, FAILED, UNKNOWN = (
    "queued", "approved", "rejected", "executed", "failed", "unknown",
)


class GateQueue:
    """Append-only JSONL event log of background S3 actions awaiting
    approval. Thread-safe. State derived by folding events per id."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/gate_queue.jsonl")

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("CORTEX_GATE_QUEUE", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except OSError:
            pass  # degrade — _append guards writes too
        # Owner-only, same M6-A6 rationale as audit.py (S3 args live here).
        # Doubles as the one-shot for files created before this existed.
        chmod_600(self.path)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    def _append(self, record: Dict[str, Any]) -> bool:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                existed = os.path.exists(self.path)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                if not existed:
                    chmod_600(self.path)
                return True
            except OSError:
                return False

    def enqueue(
        self,
        type: str,
        args: Dict[str, Any],
        *,
        reason: Optional[str] = None,
        origin: str = "background",
        trace: Optional[str] = None,
    ) -> Optional[str]:
        """Queue a destructive action for later human review. Returns its id.

        Returns None if recording failed.

        `type` is the tool name (send_email/create_event/…), `args` its
        call args, `origin` who produced it (e.g. "studier", "scheduled:x"),
        `trace` an optional audit trace id to correlate with the audit log.

        Args are REDACTED at this border (M6-A6) and the queue is the
        source of truth for execution — so what the reviewer sees, what
        is persisted and what later runs are the SAME redacted args. A
        background action that needs a raw secret in its args simply
        cannot pass through this queue; that is the fail-closed choice.
        """
        gid = uuid.uuid4().hex[:12]
        if not self._append({
            "id": gid,
            "event": QUEUED,
            "type": type,
            # Border redaction (M6-A6): S3 args land in plaintext JSONL.
            # The human reviewer sees [REDACTED] where a secret was — if
            # the action needs it, that's a reason to reject, not to log it.
            "args": redact_payload(dict(args or {})),
            "reason": reason,
            "origin": origin,
            "trace_id": trace,
        }):
            return None
        try:
            audit.log(trace or "–", "gate", "queued",
                      payload={"id": gid, "type": type, "origin": origin})
        except Exception:
            pass
        return gid

    @contextmanager
    def _decision_lock(self):
        """Held across "still pending?" and the decision's append, between
        PROCESSES: two terminals approving the same id both read it as
        pending and both ran it — the threading lock is per instance (s88,
        Codex lote 10, reproduced). Fail-closed on a lock error: without
        the process lock, no decision may grant execution."""
        fd = None
        try:
            fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            if fd is not None:
                os.close(fd)
            try:
                audit.log("–", "gate", "decision_lock_failed")
            except Exception:
                pass
            yield False
            return
        try:
            yield True
        finally:
            os.close(fd)  # releases the flock

    def resolve(self, id: str, decision: str) -> bool:
        """Record the human's decision (`approved` | `rejected`) for an id.

        No-ops (returns False) if the id is unknown or already decided —
        so a double-approve can't run an action twice, even from two
        processes (_decision_lock)."""
        if decision not in _DECISION_EVENTS:
            raise ValueError(f"decision must be one of {_DECISION_EVENTS}, got {decision!r}")
        with self._decision_lock() as locked:
            if not locked or id not in self._pending_ids():
                return False
            ok = self._append({"id": id, "event": decision})
        try:
            audit.log("–", "gate", decision, payload={"id": id})
        except Exception:
            pass
        return ok

    def mark_executed(self, id: str, ok: bool, detail: str = "") -> bool:
        """Record the outcome of running an approved action."""
        event = EXECUTED if ok else FAILED
        wrote = self._append({"id": id, "event": event, "detail": detail[:200]})
        try:
            audit.log("–", "gate", event, payload={"id": id, "ok": ok})
        except Exception:
            pass
        return wrote

    def mark_unknown(self, id: str, detail: str = "") -> bool:
        """An uncertain effect is final in the queue; never suggest a retry."""
        wrote = self._append({"id": id, "event": UNKNOWN, "detail": detail[:200]})
        try:
            audit.log("–", "gate", UNKNOWN, payload={"id": id})
        except Exception:
            pass
        return wrote

    # ------------------------------------------------------------------
    # read path (fold)
    # ------------------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip malformed — never fail the whole queue
        except OSError:
            return []
        return out

    def _events_by_id(self) -> "Dict[str, List[Dict[str, Any]]]":
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self._read_all():
            gid = rec.get("id")
            if gid:
                grouped.setdefault(gid, []).append(rec)
        return grouped

    def _pending_ids(self) -> set:
        pending = set()
        for gid, events in self._events_by_id().items():
            names = {e.get("event") for e in events}
            if QUEUED in names and not (names & set(_DECISION_EVENTS)):
                pending.add(gid)
        return pending

    def pending(self) -> List[Dict[str, Any]]:
        """Every action still awaiting a human decision, oldest first.
        Each entry is the original `queued` record (id/type/args/reason/
        origin/trace_id/ts)."""
        ids = self._pending_ids()
        out = [
            rec for rec in self._read_all()
            if rec.get("event") == QUEUED and rec.get("id") in ids
        ]
        out.sort(key=lambda r: r.get("ts", ""))
        return out

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Return the `queued` record for an id, or None if unknown."""
        for rec in self._read_all():
            if rec.get("id") == id and rec.get("event") == QUEUED:
                return rec
        return None

    def status(self, id: str) -> Optional[str]:
        """Derived status of an id: queued → approved → executed/failed/unknown,
        or rejected. None if the id is unknown."""
        events = self._events_by_id().get(id)
        if not events:
            return None
        names = {e.get("event") for e in events}
        if UNKNOWN in names:
            return UNKNOWN
        if EXECUTED in names:
            return EXECUTED
        if FAILED in names:
            return FAILED
        if REJECTED in names:
            return REJECTED
        if APPROVED in names:
            return APPROVED
        return QUEUED

    def history(self, limit: Optional[int] = 50) -> Optional[List[Dict[str, Any]]]:
        """Newest queued first; limit=None reads all, return None means unreadable."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (not isinstance(rec, dict) or not isinstance(rec.get("id"), str)
                            or not rec["id"] or not isinstance(rec.get("event"), str)):
                        continue
                    grouped.setdefault(rec["id"], []).append(rec)
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError):
            return None
        out = []
        for events in grouped.values():
            queued = next((e for e in events if e["event"] == QUEUED), None)
            if queued is None:
                continue
            names = {e["event"] for e in events}
            # Fold this snapshot, never re-read status or take the decision lock.
            status = next(s for s in (UNKNOWN, EXECUTED, FAILED, REJECTED, APPROVED, QUEUED)
                          if s in names)
            out.append({"queued": queued, "status": status, "events": events})

        def stamp(item):
            try:
                value = datetime.fromisoformat(item["queued"].get("ts", ""))
                return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
            except (TypeError, ValueError):
                return datetime.min.replace(tzinfo=timezone.utc)

        out = sorted(reversed(out), key=stamp, reverse=True)
        return out if limit is None else out[:max(0, limit)]


def record_outcome(queue, gid, reply, mode) -> str:
    """Translate the runner result once, preserving durable uncertainty."""
    from src.core import effect_store

    effect = effect_store.store.get(gid)
    if effect is not None and effect.state in {"desconhecido", "executando"}:
        queue.mark_unknown(gid, detail=reply)
        return "NÃO SEI"
    ok = mode == "AGENT_ACTION_DONE"
    queue.mark_executed(gid, ok=ok, detail=reply)
    return "executado" if ok else "FALHOU"


# Module-level singleton — import and use directly, like `audit`.
gate_queue = GateQueue()
