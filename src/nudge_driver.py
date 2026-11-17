"""
Nudge driver — proactive, timely macOS notifications (innovation #2).

Where the Daily Driver (src/daily_driver.py) speaks ONE summary at 08h, the
Nudge driver runs every ~15min (launchd StartInterval) and surfaces the ONE
thing that matters RIGHT NOW as a gentle macOS notification:

  • ⏰ an imminent calendar event (starts within the next ~20min)
  • 📭 a stale unread email from a KNOWN contact (≥3 days, unanswered)
  • (Block B) a reminder due today/overdue

Deterministic by design (SOBERANIA: a narrow rule decides what interrupts you,
NEVER an LLM verdict). Proactive external reads are audited. A dedup ledger
(data/nudged.json) guarantees each item nudges exactly ONCE, and quiet hours
keep it silent while you sleep.

Run:  python -m src.nudge_driver --stdout    (print what WOULD fire — test)
      python -m src.nudge_driver --notify    (what launchd runs)
      make nudge                             (print only)

`evaluate` is PURE (synthetic sources + state); gather/deliver/main are
fail-soft I/O, mirroring src/daily_driver.py + src/core/digest.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from src.core.digest import _parse_ts  # shared tz-aware ISO parser


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# Config (env-tunable). evaluate() takes these as explicit args so it stays
# pure/testable; these are just the production defaults that main()/gather() use.
IMMINENT_MIN = _env_int("CORTEX_NUDGE_IMMINENT_MIN", 20)
STALE_DAYS = _env_int("CORTEX_NUDGE_STALE_DAYS", 3)
MAX_PER_TICK = _env_int("CORTEX_NUDGE_MAX", 1)
QUIET_START = _env_int("CORTEX_NUDGE_QUIET_START", 22)  # 22h
QUIET_END = _env_int("CORTEX_NUDGE_QUIET_END", 8)       # 08h
_NUDGED_PATH = os.getenv("CORTEX_NUDGED", "data/nudged.json")
_NUDGED_KEEP_DAYS = 2

_ADDR_RE = re.compile(r"<([^>]+)>")


@dataclass
class NudgeSources:
    events: List[Dict[str, Any]] = field(default_factory=list)        # upcoming calendar
    due_reminders: List[Dict[str, Any]] = field(default_factory=list)  # Block B
    stale_emails: List[Dict[str, Any]] = field(default_factory=list)   # unread, known contact


@dataclass
class Nudge:
    id: str        # stable per-item key for dedup
    urgency: int   # higher fires first when capped
    title: str
    body: str
    speak: bool = False


def _sender_email(from_header: str) -> str:
    """Extract the bare address from an RFC 'Name <addr@x>' (or a raw addr)."""
    m = _ADDR_RE.search(from_header or "")
    addr = m.group(1) if m else (from_header or "")
    return addr.strip().lower()


def _sender_name(from_header: str) -> str:
    """Display name (or the address) for the notification body."""
    head = (from_header or "").split("<")[0].strip().strip('"')
    return head or _sender_email(from_header) or "alguém"


def _in_quiet_hours(now: datetime, start_h: int, end_h: int) -> bool:
    """True during the quiet window. Handles the usual wrap past midnight
    (start 22, end 8 → quiet when hour >= 22 OR hour < 8)."""
    h = now.astimezone().hour
    if start_h == end_h:
        return False
    if start_h > end_h:
        return h >= start_h or h < end_h
    return start_h <= h < end_h


def evaluate(
    sources: NudgeSources,
    now: datetime,
    already: Set[str],
    *,
    imminent_min: int = IMMINENT_MIN,
    stale_days: int = STALE_DAYS,
    max_per_tick: int = MAX_PER_TICK,
    quiet_start: int = QUIET_START,
    quiet_end: int = QUIET_END,
) -> List[Nudge]:
    """PURE. Turn sources + state into the nudges to fire NOW. Quiet hours
    suppress everything; already-nudged ids are dropped (one nudge per item);
    candidates rank by urgency (event > reminder > email) and cap at
    max_per_tick. Never raises."""
    if _in_quiet_hours(now, quiet_start, quiet_end):
        return []
    fresh = [n for n in candidates(sources, now, imminent_min=imminent_min,
                                   stale_days=stale_days)
             if n.id not in already]
    fresh.sort(key=lambda n: n.urgency, reverse=True)
    return fresh[:max(0, max_per_tick)]


def candidates(
    sources: NudgeSources,
    now: datetime,
    *,
    imminent_min: int = IMMINENT_MIN,
    stale_days: int = STALE_DAYS,
) -> List[Nudge]:
    """PURE. Everything that warrants a nudge right now, already nudged or
    not, quiet hours or not — what the ledger must keep remembering."""
    cands: List[Nudge] = []

    # ⏰ Imminent calendar events — the flagship: timely and unambiguous.
    for ev in sources.events:
        ts = _parse_ts(ev.get("start"))
        if ts is None:
            continue
        mins = (ts - now).total_seconds() / 60.0
        if 0 <= mins <= imminent_min:
            summary = (ev.get("summary") or "evento").strip()
            when = ts.astimezone().strftime("%H:%M")
            cands.append(Nudge(
                id=f"event:{ev.get('id', '')}:{ev.get('start', '')}",
                urgency=3, title="⏰ Em breve",
                body=f"{summary} às {when} — em {round(mins)} min.",
                speak=True))

    # 📅 Reminders due today or overdue. due_iso is a naive LOCAL wall-clock
    # ISO (reminders_client), so parse it as-is and compare LOCAL dates — NOT
    # via _parse_ts (which would tag it UTC and shift the date near midnight).
    today = now.astimezone().date()
    for r in sources.due_reminders:
        raw = (r.get("due_iso") or "").strip()
        if not raw:
            continue
        try:
            due = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if due.date() <= today:
            title = (r.get("name") or "lembrete").strip()
            # Overdue is said as such: "vence hoje" on Monday's reminder read
            # on Thursday (s88, Codex lote 4).
            when = ("vence hoje" if due.date() == today
                    else f"venceu em {due:%d/%m}")
            cands.append(Nudge(
                id=f"reminder:{title}:{raw}",
                urgency=2, title="📅 Lembrete",
                body=f"{title} — {when}."))

    # 📭 Stale unread email from a known contact (gather already filtered to
    # known contacts; here we apply the age rule).
    for m in sources.stale_emails:
        ts = _parse_ts(m.get("date_iso"))
        if ts is None:
            continue
        days = (now - ts).total_seconds() / 86400.0
        if days >= stale_days:
            subj = (m.get("subject") or "(sem assunto)").strip()
            cands.append(Nudge(
                id=f"email:{m.get('id', '')}",
                urgency=1, title="📭 Email parado",
                body=f"{_sender_name(m.get('from', ''))}: {subj} — "
                     f"há {round(days)} dias sem resposta."))

    return cands


def gather(now: datetime) -> NudgeSources:
    """Fetch each signal, fail-soft (a broken source → empty). Audited."""
    out = NudgeSources()
    try:
        from src.core.audit import audit
        audit.log("–", "background_read", "nudge_gather", payload={
            "sources": ["calendar.list_events", "gmail.list_recent",
                        "reminders.list_reminders"],
        })
    except Exception:  # noqa: BLE001 — audit must never break the nudge
        pass

    try:
        from src.integrations.calendar_client import list_events
        out.events = list_events(max_results=15) or []
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.integrations.reminders_client import list_reminders
        out.due_reminders = list_reminders(limit=15) or []
    except Exception:  # noqa: BLE001
        pass

    # Stale email = UNREAD (proxy for "not dealt with") from a KNOWN contact.
    # Narrow on purpose: the contact book is the deterministic "matters to me"
    # filter, so we never nudge on newsletter/no-reply noise.
    try:
        from src.integrations.contacts import contact_book
        from src.integrations.gmail_client import list_recent
        known = {c.email.strip().lower()
                 for c in (contact_book.list_all() or []) if c.email}
        if known:
            for m in list_recent(count=15, query="is:unread") or []:
                if _sender_email(m.get("from", "")) in known:
                    out.stale_emails.append(m)
    except Exception:  # noqa: BLE001
        pass

    return out


def get_nudged(path: str = _NUDGED_PATH) -> Dict[str, str]:
    """Ledger of id → ISO timestamp of when it was last nudged. Fail-soft."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def add_nudged(ids: List[str], now: datetime, path: str = _NUDGED_PATH,
               keep_days: int = _NUDGED_KEEP_DAYS,
               live: Optional[Set[str]] = None) -> None:
    """Record ids as nudged-now and prune entries older than keep_days so the
    ledger can't grow forever — except the `live` ones, still candidates:
    pruned by age alone, Monday's overdue reminder was nudged again on
    Thursday (s88, Codex lote 4). Fail-soft."""
    data = get_nudged(path)
    cutoff = now - timedelta(days=keep_days)
    keep = live or set()
    data = {k: v for k, v in data.items()
            if k in keep or (_parse_ts(v) or now) >= cutoff}
    for i in ids:
        data[i] = now.isoformat()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


def deliver(nudges: List[Nudge], speak_enabled: bool = False) -> List[str]:
    """Post each nudge as a macOS notification; speak the spoken ones if
    enabled. Fail-soft per nudge. Returns the ids whose banner WAS posted:
    a failed one used to be recorded as nudged all the same, and never came
    back (s88, Codex lote 4)."""
    try:
        from src.core.heartbeat import notify
    except Exception:  # noqa: BLE001
        return []
    delivered: List[str] = []
    for n in nudges:
        try:
            if not notify(n.title, n.body):
                continue
        except Exception:  # noqa: BLE001
            continue
        delivered.append(n.id)
        if n.speak and speak_enabled:
            try:
                from src.daily_driver import speak
                speak(n.body)
            except Exception:  # noqa: BLE001
                pass
    return delivered


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Proactive timely nudges.")
    parser.add_argument("--notify", action="store_true", help="fire macOS notifications (launchd)")
    parser.add_argument("--speak", action="store_true", help="also speak imminent-event nudges")
    parser.add_argument("--stdout", action="store_true", help="print what would fire (default)")
    args = parser.parse_args(argv)

    # Global kill-switch: CORTEX_NUDGE=0 turns the whole engine off.
    if not _env_flag("CORTEX_NUDGE", True):
        if args.stdout or not args.notify:
            print("(nudges desligados — CORTEX_NUDGE=0)")
        return 0

    now = datetime.now(timezone.utc)
    already = set(get_nudged().keys())
    sources = gather(now)
    nudges = evaluate(sources, now, already)

    if args.stdout or not args.notify:
        if nudges:
            for n in nudges:
                print(f"[{n.urgency}] {n.title} — {n.body}")
        else:
            print("(nada a cutucar agora)")
    if args.notify and nudges:
        # Persist ONLY on the real delivery path, so --stdout/test runs never
        # consume an item's one-shot dedup.
        delivered = deliver(
            nudges, speak_enabled=_env_flag("CORTEX_NUDGE_SPEAK", False) or args.speak)
        add_nudged(delivered, now, live={n.id for n in candidates(sources, now)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
