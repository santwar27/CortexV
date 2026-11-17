"""
System lease — scoped, audited, time-bounded network permission for
BACKGROUND jobs (P-2 decision, Pedro 2026-06-10; M6-A2/C5, review Fable 5).

The human internet lease (gatekeeper.LogicGate) answers "may the
INTERACTIVE assistant touch the web right now?" — granted by a spoken/
typed request and visible in the /ui chip. Background jobs are a
different trust shape: the nightly Studier runs at 02:00 with nobody
awake to grant anything, yet "no governance at all" (the pre-review
state) meant its web traffic was invisible and unbounded.

A SystemLease is the middle ground Pedro chose over "nightly only runs
under a human lease":

  - SCOPED   — a lease names one scope (today: "web_search"); the
               GovernedWebSearcher refuses any other use.
  - BOUNDED  — expires after `minutes` and after `max_calls` searches,
               whichever comes first. A runaway loop can't turn one
               nightly run into unlimited traffic.
  - AUDITED  — acquire / every search / every denial / release all land
               in the audit log (stage "system_lease"), so `make
               latency-report`-style forensics and the /ui can show
               exactly what background traffic happened, query by query.
  - KILLABLE — CORTEX_SYSTEM_LEASE=0 disables acquisition globally; the
               Studier then degrades to "no snippets" instead of
               searching ungoverned.

This file is deliberately separate from gatekeeper.py: the human lease
state (gate_state.json) is never touched, so a background run can never
extend or consume the human's window.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.core.audit import audit


def _enabled() -> bool:
    return os.getenv("CORTEX_SYSTEM_LEASE", "1").strip().lower() not in (
        "0", "false", "no", "",
    )


class SystemLeaseDenied(RuntimeError):
    """Raised by governed tools when the lease refuses a call."""


class SystemLease:
    """One background job's bounded permission window. Build via
    `SystemLease.acquire(...)`; pass to a governed tool wrapper."""

    def __init__(
        self,
        scope: str,
        origin: str,
        expires_at: Optional[datetime],
        max_calls: int,
        trace: str,
        active: bool,
    ):
        self.scope = scope
        self.origin = origin
        self.expires_at = expires_at
        self.max_calls = max_calls
        self.trace = trace
        self.calls_made = 0
        self.denials = 0
        self._active = active
        self._released = False

    # -- lifecycle -----------------------------------------------------

    @classmethod
    def acquire(
        cls,
        scope: str,
        origin: str,
        minutes: int = 90,
        max_calls: int = 60,
    ) -> "SystemLease":
        """Open an audited window. With CORTEX_SYSTEM_LEASE=0 the lease
        comes back INACTIVE (every governed call is refused + audited) —
        the job still runs, just without network."""
        trace = audit.new_trace()
        enabled = _enabled()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(minutes=minutes)
            if enabled else None
        )
        audit.log(trace, "system_lease", "acquire", payload={
            "scope": scope,
            "origin": origin,
            "minutes": minutes,
            "max_calls": max_calls,
            "enabled": enabled,
        })
        return cls(
            scope=scope, origin=origin, expires_at=expires_at,
            max_calls=max_calls, trace=trace, active=enabled,
        )

    def release(self) -> None:
        """Close the window (idempotent). Audits the usage summary."""
        if self._released:
            return
        self._released = True
        self._active = False
        audit.log(self.trace, "system_lease", "release", payload={
            "scope": self.scope,
            "origin": self.origin,
            "calls_made": self.calls_made,
            "denials": self.denials,
        })

    def __enter__(self) -> "SystemLease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- enforcement ---------------------------------------------------

    def is_active(self) -> bool:
        if not self._active or self._released:
            return False
        if self.expires_at and datetime.now(timezone.utc) >= self.expires_at:
            return False
        return self.calls_made < self.max_calls

    def authorize(self, scope: str, detail: str = "") -> None:
        """Gate one call. Raises SystemLeaseDenied (audited) when the
        scope mismatches or the window/cap is exhausted; otherwise
        counts the call and audits it."""
        if scope != self.scope:
            reason = f"scope '{scope}' fora da lease ('{self.scope}')"
        elif self._released or not self._active:
            reason = "lease inativa (released ou kill-switch)"
        elif self.expires_at and datetime.now(timezone.utc) >= self.expires_at:
            reason = "janela expirada"
        elif self.calls_made >= self.max_calls:
            reason = f"cap de {self.max_calls} chamadas atingido"
        else:
            self.calls_made += 1
            audit.log(self.trace, "system_lease", "call", payload={
                "scope": scope,
                "origin": self.origin,
                "n": self.calls_made,
                "detail": detail[:80],
            })
            return
        self.denials += 1
        audit.log(self.trace, "system_lease", "deny", payload={
            "scope": scope,
            "origin": self.origin,
            "reason": reason,
            "detail": detail[:80],
        })
        raise SystemLeaseDenied(reason)


class GovernedWebSearcher:
    """Drop-in wrapper over WebSearcher that routes every `.search()`
    through a SystemLease — the enforcement point the review asked for
    ('lease check no WebSearcher ou injetado', C5b). The inner searcher
    stays a dumb tool; governance lives here, at the boundary."""

    def __init__(self, inner, lease: SystemLease):
        self._inner = inner
        self._lease = lease

    def search(self, query: str, max_results: int = 3) -> str:
        self._lease.authorize("web_search", detail=query)
        return self._inner.search(query, max_results=max_results)
