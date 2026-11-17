"""Per-turn token accounting for the /ui chat (idea #26, s69).

The engine already audits Ollama's real token counts on every call
(`_log_token_usage`, M2-A1) — this module turns that same data into a
per-turn total the UI can show under the reply ("2.9k → 13 tok · 19s").
Pedagogical more than financial: a token costs R$0 on local Ollama, but
prompt size is latency and context budget.

Thread-local by design, the same isolation argument as the status
emitter (s48): the /ui stream route runs each turn in its own worker
thread, calls `reset()` there, the engine's `_log_token_usage` adds
into that thread's bucket, and the route `collect()`s after process()
returns. A concurrent voice turn on another thread never touches it —
and on threads that never called `reset()` (voice, CLI, tests) `add()`
is a no-op, so the engine path costs nothing there.

Lives in its own tiny module (not engine.py) so the UI router can
import it without dragging the engine/memory graph into a minimal app.
"""
import threading
from typing import Optional

_local = threading.local()


def reset() -> None:
    """Open an empty bucket for the CURRENT thread's turn."""
    _local.usage = {"prompt_eval_count": 0, "eval_count": 0, "calls": 0,
                    "models": [], "online": []}


def current() -> Optional[dict]:
    """This thread's open bucket, or None — for a turn that hands its work
    to another thread (the stream's deferred worker) to carry it over."""
    return getattr(_local, "usage", None)


def attach(bucket: Optional[dict]) -> None:
    """Make `bucket` (opened on another thread) this thread's too, or clear
    it with None. The stream's worker ran the engine on a thread with no
    bucket, so every token it spent was dropped and the UI showed no usage
    for a web-search turn (s88, Codex lote 13)."""
    _local.usage = bucket


def add(prompt_eval_count, eval_count, model=None) -> None:
    """Accumulate one LLM call into this thread's bucket, if one is
    open. Never raises — telemetry must not break a turn."""
    bucket = getattr(_local, "usage", None)
    if bucket is None:
        return
    try:
        bucket["prompt_eval_count"] += int(prompt_eval_count or 0)
        bucket["eval_count"] += int(eval_count or 0)
        bucket["calls"] += 1
        if isinstance(model, str) and model:
            bucket["models"] = sorted(set(bucket["models"] + [model]))
    except Exception:
        pass


def mark_online(source) -> None:
    """Mark BEFORE an internet attempt: a failed request may have sent data.
    No bucket means no-op; telemetry must never break the turn."""
    try:
        bucket = getattr(_local, "usage", None)
        if bucket is not None and isinstance(source, str) and source:
            bucket["online"] = sorted(set(bucket["online"] + [source]))
    except Exception:
        pass


def collect() -> Optional[dict]:
    """Close and return this thread's bucket. None when no bucket was
    opened OR neither an LLM call nor an internet attempt landed in it.
    Online reflexes still need a truthful privacy badge with zero calls."""
    bucket = getattr(_local, "usage", None)
    _local.usage = None
    if not bucket or (bucket["calls"] == 0 and not bucket["online"]):
        return None
    return bucket
