"""In-memory conversation buffer for the /ui dashboard.

The dashboard chat drawer needs a short-term history of turns to
display when it opens. We don't want to re-query the Palace for
recent turns (Palace is the long-term memory, not a UI cache) so a
small process-local deque is fine.

Lifecycle:
  - Append on every successful /ui/chat/stream reply.
  - Append on every /ui/conversation/push (voice daemon, future).
  - Read by /ui/conversation for the drawer to backfill on open.
  - Lost on uvicorn restart — intentional. This is a UI nicety, not
    a memory system. Palace remains the source of truth.

Thread safety: FastAPI runs async on a single event loop, but the
orchestrator is sync and can be invoked from worker threads (slowapi
internals, future thread-pool offload). A `threading.Lock` around
mutation costs nothing and protects against the race regardless of
how the call lands.

Source tagging: every turn carries a `source` field ("ui" for typed,
"voice" for the daemon push, "test" for fixtures). The drawer can
style voice turns differently from typed ones once the daemon is
back online.
"""

import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


_DEFAULT_MAXLEN = 50


class ConversationBuffer:
    """Thread-safe ring buffer of conversation turns.

    A turn is a dict: {
        "id":          short uuid (frontend tracks newness),
        "ts":          ISO8601 UTC (when the REPLY landed),
        "asked_ts":    ISO8601 UTC of the request, or None (idea #28:
                       the question bubble shows when it was ASKED, not
                       when the answer arrived — on a 19s turn they
                       differ visibly),
        "user":        str (what the user said/typed),
        "reply":       str (what Cortex responded),
        "status_mode": str (orchestrator status_mode),
        "source":      "ui" | "voice" | "test",
        "usage":       {"prompt_eval_count", "eval_count", "calls"} or
                       None (idea #26: per-turn token totals from the
                       engine; None for turns with no LLM call),
        "latency_ms":  int or None (wall time of orchestrator.process),
    }

    The buffer is bounded — once `maxlen` is reached, the oldest
    turn is evicted on the next append.
    """

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(
        self,
        user: str,
        reply: str,
        status_mode: str = "OK",
        source: str = "ui",
        asked_ts: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add a turn and return it (with id+ts filled in). The three
        optional fields default to None so existing callers (voice push,
        tests) keep producing valid turns with no metrics."""
        turn: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "ts": datetime.now(timezone.utc).isoformat(),
            "asked_ts": asked_ts,
            "user": user,
            "reply": reply,
            "status_mode": status_mode,
            "source": source,
            "usage": usage,
            "latency_ms": latency_ms,
        }
        with self._lock:
            self._buf.append(turn)
        return turn

    def snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a list copy of the last `limit` turns (or all if
        None). Returned list is independent of the internal deque so
        the caller can mutate freely."""
        with self._lock:
            items = list(self._buf)
        if limit is None or limit >= len(items):
            return items
        return items[-limit:]

    def clear(self) -> int:
        """Drop all turns; return how many were removed. Counting under the
        same lock makes the count + wipe atomic (no concurrent append can
        slip between a separate len() and clear())."""
        with self._lock:
            n = len(self._buf)
            self._buf.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# Module-level singleton — the production app uses one buffer for
# the lifetime of the process. Tests instantiate their own when they
# need isolation.
buffer = ConversationBuffer()
