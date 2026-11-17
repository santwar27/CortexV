"""
Structured audit log for the CORTEX pipeline.

Appends one JSON record per event to ~/.cortex/audit.jsonl (outside
src/ so the agent's own write path can't tamper with its audit trail).

Usage — single import, module-level singleton:

    from src.core.audit import audit

    trace = audit.new_trace()
    audit.log(trace, "orchestrator", "request", payload={"input": text[:80]})
    t0 = time.perf_counter()
    ...
    audit.log(trace, "orchestrator", "response", latency_ms=(time.perf_counter()-t0)*1000)

the reference principle 3 — "Logs antes de magia":
When something breaks, the JSONL tells you exactly where and how long
each stage took.  grep / jq work fine; no database needed.
"""

import fcntl
import gzip
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.core.redaction import redact_payload

# Rotation threshold (bytes) for the active JSONL. Policy approved s72:
# rotate-and-archive at 10MB, gzip into <dir>/archive/, NEVER delete —
# purging is a separate human decision. gate_queue.jsonl is deliberately
# NOT rotated (it is the source of truth for pending S3 actions).
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024


def _max_bytes_from_env() -> int:
    """CORTEX_AUDIT_MAX_BYTES, malformed → default (audit never crashes)."""
    raw = os.environ.get("CORTEX_AUDIT_MAX_BYTES", "")
    try:
        value = int(raw)
        return value if value > 0 else _DEFAULT_MAX_BYTES
    except (TypeError, ValueError):
        return _DEFAULT_MAX_BYTES


def chmod_600(path: str) -> None:
    """Restrict a sink file to the owner. Fail-soft: missing file or odd
    FS must never break the pipeline the sink serves."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def read_records(path: str, since: Optional[datetime] = None) -> list:
    """Every record of the audit at `path` — its rotated archives included —
    with ts >= `since` when given, oldest first. Fail-soft: a missing or
    unreadable file, or a malformed line, is skipped.

    The archives matter: once the active file rotates, a weekly window that
    straddles the rotation lost Monday-to-Saturday, and the policy detector
    lost a rejection it should have weighed (s88, Codex lote 10). An archive
    is named after the moment it was rotated, so one rotated before `since`
    holds nothing newer and is not opened."""
    folder = os.path.dirname(path) or "."
    stem = os.path.basename(path).rsplit(".", 1)[0]
    stamp_re = re.compile(re.escape(stem) + r"-(\d{8}-\d{6})-[0-9a-f]+\.jsonl(?:\.gz)?$")
    files = []
    try:
        for name in sorted(os.listdir(os.path.join(folder, "archive"))):
            m = stamp_re.match(name)
            if not m:
                continue
            rotated = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            if since is None or rotated >= since:
                files.append(os.path.join(folder, "archive", name))
    except OSError:
        pass
    files.append(path)
    out = []
    for f in files:
        try:
            opener = gzip.open if f.endswith(".gz") else open
            # errors="replace" (s88, Codex lote 13 — invariant 7): a write
            # cut inside a multibyte character ended every reader — latency
            # report, weekly review, policy detector — in UnicodeDecodeError.
            # The damaged line then fails as JSON and is skipped like any.
            with opener(f, "rt", encoding="utf-8", errors="replace") as h:
                for line in h:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if since is not None:
                        try:
                            ts = datetime.fromisoformat(rec.get("ts") or "")
                        except (TypeError, ValueError):
                            continue
                        if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)) < since:
                            continue
                    out.append(rec)
        except (OSError, EOFError):
            continue
    return out


class AuditLogger:
    """Append-only JSONL structured logger, thread-safe."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/audit.jsonl")

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("CORTEX_AUDIT_LOG", self.DEFAULT_PATH)
        self._max_bytes = _max_bytes_from_env()
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except OSError:
            pass  # Audit must never crash the pipeline — silently degrade.
        # The log holds redacted-but-personal data (voice transcripts,
        # tool args) — owner-only, like google_token's _save (M6-A6).
        # Doubles as the one-shot for files created before this existed.
        chmod_600(self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_trace(self) -> str:
        """Return a fresh short trace ID for one logical request."""
        return uuid.uuid4().hex[:8]

    def log(
        self,
        trace_id: str,
        stage: str,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Append one structured record to the JSONL file.

        Args:
            trace_id:   Short hex ID grouping related events. Use new_trace().
            stage:      Pipeline stage name, e.g. 'orchestrator', 'studier',
                        'reflex', 'tool', 'gate'.
            event:      Event label within the stage, e.g. 'request',
                        'response', 'hit', 'miss', 'query_gen', 'search',
                        'synthesis', 'save', 'gate_asked', 'gate_approved'.
            payload:    Optional dict with extra context. Kept shallow —
                        never put raw LLM output (too large).
            latency_ms: Optional elapsed time for this stage in milliseconds.
        """
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "stage": stage,
            "event": event,
        }
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 2)
        if payload:
            # Border redaction (M6-A6): the audit JSONL is plaintext and
            # append-only — a secret that lands here never leaves. Every
            # producer (incl. voice telemetry) funnels through this point.
            record["payload"] = redact_payload(payload)

        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self._file_lock():
            self._rotate_if_needed()
            try:
                existed = os.path.exists(self.path)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                if not existed:
                    chmod_600(self.path)
            except OSError:
                # Audit must never crash the main pipeline.
                pass

    @contextmanager
    def _file_lock(self):
        """Between PROCESSES, around the append and the whole rotation: a
        writer that had opened the file before another process renamed,
        gzipped and removed it wrote into a deleted inode — the record gone
        (s88, Codex lote 10, reproduced). threading.Lock only orders threads
        of one logger. Fail-open: audit must never block the pipeline on a
        lock it cannot make."""
        try:
            fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _rotate_if_needed(self) -> None:
        """Rotate the active JSONL into <dir>/archive/ when it crosses
        the size threshold. Called under self._lock, before the append —
        the record that triggered the check lands in the fresh file.

        Retention policy (approved s72): archive forever, delete never.
        The rename is atomic, so when listener + uvicorn race only one
        rotation wins; the loser just appends to the fresh file. If the
        gzip step fails the uncompressed archive stays — data loss is the
        only unacceptable failure mode here.
        """
        try:
            if os.path.getsize(self.path) < self._max_bytes:
                return
        except OSError:
            return  # missing file etc. — nothing to rotate
        try:
            archive_dir = os.path.join(os.path.dirname(self.path) or ".", "archive")
            os.makedirs(archive_dir, exist_ok=True)
            stem = os.path.basename(self.path).rsplit(".", 1)[0]
            # uuid suffix: two rotations in the same second (or the same
            # path from two processes) must never overwrite an archive.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            rotated = os.path.join(
                archive_dir, f"{stem}-{stamp}-{uuid.uuid4().hex[:6]}.jsonl")
            os.rename(self.path, rotated)
        except OSError:
            return  # lost the race or odd FS — keep appending where we are
        try:
            with open(rotated, "rb") as src, gzip.open(rotated + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            chmod_600(rotated + ".gz")
            os.remove(rotated)
        except OSError:
            pass  # uncompressed archive (0600 already) is fine — never delete

    def span(self, trace_id: str, stage: str, event_start: str, event_end: str):
        """Context manager that logs start + end with elapsed latency.

        Usage:
            with audit.span(trace, "studier", "synthesis_start", "synthesis_end"):
                briefing = llm_call(...)
        """
        return _AuditSpan(self, trace_id, stage, event_start, event_end)

    def read_tail(self, n: int = 50) -> list:
        """Return the last n records as dicts. Useful for debugging."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            records = []
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return records
        except FileNotFoundError:
            return []

    def read_trace(self, trace_id: str) -> list:
        """Return all records matching a trace_id."""
        return [r for r in self.read_tail(1000) if r.get("trace_id") == trace_id]


class _AuditSpan:
    """Internal context manager returned by AuditLogger.span()."""

    def __init__(
        self,
        logger: AuditLogger,
        trace_id: str,
        stage: str,
        event_start: str,
        event_end: str,
    ):
        self._logger = logger
        self._trace_id = trace_id
        self._stage = stage
        self._event_start = event_start
        self._event_end = event_end
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._logger.log(self._trace_id, self._stage, self._event_start)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self._t0) * 1000
        status = "error" if exc_type else self._event_end
        self._logger.log(
            self._trace_id,
            self._stage,
            status,
            latency_ms=elapsed,
            payload={"error": str(exc_val)} if exc_type else None,
        )
        return False  # never suppress exceptions


# Module-level singleton — import and use directly.
audit = AuditLogger()
