"""
Delta digest — "o que mudou desde a última vez" (Onda 4a).

Pull-based, DETERMINISTIC assembly (template + data, no LLM synthesis) → zero
hallucination. Gathers what changed in Pedro's world since the last time he
asked, advances a "last seen" marker, and renders a terse pt-BR digest.

Sections (v1):
  • 📥 E-mails novos      — gmail, filtered by date since the marker.
  • 🧠 Estudos do Studier — new briefings in data/knowledge/pending (by mtime).
  • 🤖 Ações que tomei    — Action Ledger entries since the marker.
  • 📅 Próximos eventos   — calendar upcoming (forward-looking CONTEXT, not a
                            strict delta: the API surface here can't tell us
                            what was *created/moved*, so this is labelled as
                            "à frente", honestly).

Not included (v1): notes — notes_client.list_notes exposes no modification
date, so "nota tocada" can't be detected without snapshots. Documented, not
faked.

`assemble` is PURE (testable with synthetic sources); `gather`/`get_last_seen`/
`set_last_seen` do fail-soft I/O, mirroring src/core/policy_detector.py.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core.i18n import t

_LAST_SEEN_PATH = os.environ.get(
    "CORTEX_LAST_SEEN", "data/last_seen.json"
)
# First-ever call has no marker → show the trailing window instead of nothing.
_FIRST_RUN_WINDOW = timedelta(hours=24)
# Upcoming-events horizon (forward-looking context).
_EVENTS_HORIZON = timedelta(hours=48)
_STUDIER_DIR = "data/knowledge/pending"
_EMAIL_PAGE = 15


@dataclass
class DeltaSources:
    """Already-filtered, ready-to-render delta. Pure input to `assemble`."""
    emails: List[Dict[str, Any]] = field(default_factory=list)
    studies: List[str] = field(default_factory=list)        # briefing filenames
    actions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Sources that could not be read this time ("emails", "actions",
    # "events"). An empty list and a source that failed used to look the
    # same (s88, Codex lote 13 — invariant 7).
    unavailable: List[str] = field(default_factory=list)
    # Every email in the window, when one page did not hold them all.
    email_total: Optional[int] = None


# The sources whose window the marker closes. A failure in one of them keeps
# the marker where it was: advancing it hid an email that arrived during
# the outage for good — Gmail down at 12:00, the 10:00 email outside the
# window once it was back (reproduced). The agenda is look-ahead context,
# not a delta, so its failure is reported but holds nothing.
_DELTA_SOURCES = ("emails", "actions")


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def get_last_seen(path: Optional[str] = None) -> Optional[datetime]:
    """Read the last-seen marker, fail-soft. None when never set. `path`
    resolves at call time so tests can redirect _LAST_SEEN_PATH."""
    path = path or _LAST_SEEN_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _parse_ts((json.load(f) or {}).get("ts"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def set_last_seen(now: datetime, path: Optional[str] = None) -> None:
    """Persist the last-seen marker, fail-soft (never breaks the reply)."""
    path = path or _LAST_SEEN_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": now.isoformat()}, f)
    except OSError:
        pass


def new_studies_since(since: datetime) -> List[str]:
    """Basenames of Studier briefings (`_STUDIER_DIR/*.md`) modified at/after
    `since`. THE single definition of "new study" (M7-A6): the digest and the
    daily briefing both call this — before, one filtered by the moving
    last_seen marker and the other by a fixed 24h window over the SAME
    folder, and the two could disagree ("o briefing disse 0 estudos, mas o
    'o que mudou' lista 2"). Semantics: file MTIME, not creation date — a
    re-saved file counts as new again (deliberate; mtime is all we have).
    Fail-soft per file; an unreadable dir yields []."""
    out: List[str] = []
    try:
        for p in glob.glob(os.path.join(_STUDIER_DIR, "*.md")):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
            except OSError:
                continue
            if mtime >= since:
                out.append(os.path.basename(p))
    except Exception:  # noqa: BLE001
        pass
    return out


def gather(since: datetime, now: datetime) -> DeltaSources:
    """Fetch + filter each source to what's newer than `since`. Fail-soft:
    any source that errors degrades to empty, never raises. Events are
    upcoming within the horizon (not filtered by `since`)."""
    out = DeltaSources()

    # Proactive external reads are AUDITED (M7-A4, review Fable 5): the
    # digest fires Google API calls without the user in the loop; one
    # breadcrumb per gather makes that traffic visible to forensics and
    # the latency report. Read-only, so audit-not-gate is the contract.
    try:
        from src.core.audit import audit
        audit.log("–", "background_read", "digest_gather", payload={
            "sources": ["gmail.list_recent", "calendar.list_events"],
        })
    except Exception:  # noqa: BLE001 — audit must never break the digest
        pass

    # 📥 New emails. Gmail is asked for the window itself (after:<since>),
    # and counted whole when one page does not hold it: the newest 15 of
    # the inbox were all it saw, so 20 new emails were announced as "15"
    # and the other five never surfaced (s88, Codex lote 13). The date
    # filter stays as a second check.
    try:
        from src.integrations import gmail_client as gmail
        query = f"after:{int(since.timestamp())}"
        page = gmail.list_recent(count=_EMAIL_PAGE, query=query)
        for m in page:
            ts = _parse_ts(m.get("date_iso"))
            if ts is not None and ts >= since:
                out.emails.append(m)
        if len(page) >= _EMAIL_PAGE:
            out.email_total = gmail.count_matching(query)
    except Exception:  # noqa: BLE001 — fail-soft per source
        out.unavailable.append("emails")

    # 🧠 New Studier briefings (shared definition — M7-A6).
    out.studies = new_studies_since(since)

    # 🤖 Actions I took (Action Ledger since the marker).
    try:
        from src.core.action_ledger import ledger
        for e in ledger.load_entries():
            ts = _parse_ts(e.get("ts"))
            if ts is not None and ts >= since:
                out.actions.append(e)
    except Exception:  # noqa: BLE001
        out.unavailable.append("actions")

    # 📅 Upcoming events (forward-looking context, within the horizon).
    try:
        from src.integrations.calendar_client import list_events
        horizon = now + _EVENTS_HORIZON
        for ev in list_events(max_results=10):
            ts = _parse_ts(ev.get("start"))
            # Keep events that start from now up to the horizon. Undated/
            # all-day rows (no parseable start) are kept — better to show.
            if ts is None or (now - timedelta(hours=1)) <= ts <= horizon:
                out.events.append(ev)
    except Exception:  # noqa: BLE001
        out.unavailable.append("events")

    return out


def _fmt_event_start(raw: Any) -> str:
    """Render an event start ISO as a friendly 'dd/mm HH:MM', or '' if it
    can't be parsed (then the caller shows nothing rather than raw ISO)."""
    ts = _parse_ts(raw)
    return ts.astimezone().strftime("%d/%m %H:%M") if ts is not None else ""


def _fmt_since(since: datetime, now: datetime, lang: str = "pt") -> str:
    delta = now - since
    if delta < timedelta(minutes=90):
        return t(lang, "na última hora", "in the last hour")
    if delta < timedelta(hours=36):
        return t(lang, "desde ontem", "since yesterday")
    days = int(delta.total_seconds() // 86400)
    return t(lang, f"nos últimos {days} dias", f"in the last {days} days")


# Human labels for ledger actions in the digest.
_ACTION_LABELS = {
    "send_email": "enviei um e-mail",
    "create_event": "criei um evento",
    "create_reminder": "criei um lembrete",
    "create_note": "criei uma nota",
    "mac_quit_app": "fechei um app",
    "undo": "desfiz uma ação",
}
_ACTION_LABELS_EN = {
    "send_email": "sent an email",
    "create_event": "created an event",
    "create_reminder": "created a reminder",
    "create_note": "created a note",
    "mac_quit_app": "quit an app",
    "undo": "undid an action",
}


def assemble(sources: DeltaSources, since: datetime, now: datetime,
             lang: str = "pt") -> str:
    """Render the delta as a terse bullet digest (no tables) in `lang`. Pure."""
    en = lang == "en"
    blocks: List[str] = []

    if sources.emails:
        n = max(len(sources.emails), sources.email_total or 0)
        head = (f"📥 New emails ({n}):" if en else f"📥 E-mails novos ({n}):")
        lines = [head]
        for m in sources.emails[:5]:
            frm = (m.get("from") or "").split("<")[0].strip() or "—"
            subj = (m.get("subject") or t(lang, "(sem assunto)", "(no subject)")).strip()
            lines.append(f"  • {frm}: {subj}")
        blocks.append("\n".join(lines))

    if sources.actions:
        head = (f"🤖 Actions I took ({len(sources.actions)}):" if en
                else f"🤖 Ações que tomei ({len(sources.actions)}):")
        lines = [head]
        labels = _ACTION_LABELS_EN if en else _ACTION_LABELS
        for e in sources.actions[:5]:
            label = labels.get(e.get("action"), e.get("action", "?"))
            lines.append(f"  • {label}")
        blocks.append("\n".join(lines))

    if sources.studies:
        head = (f"🧠 New Studier briefings ({len(sources.studies)}):" if en
                else f"🧠 Estudos novos do Studier ({len(sources.studies)}):")
        lines = [head]
        for name in sources.studies[:5]:
            lines.append(f"  • {name}")
        blocks.append("\n".join(lines))

    if sources.events:
        lines = ["📅 Upcoming events:" if en else "📅 Próximos eventos:"]
        for ev in sources.events[:5]:
            summary = (ev.get("summary") or t(lang, "(sem título)", "(untitled)")).strip()
            when = _fmt_event_start(ev.get("start"))
            lines.append(f"  • {summary}{(' — ' + when) if when else ''}")
        blocks.append("\n".join(lines))

    header = t(lang, f"O que mudou {_fmt_since(since, now, lang)}:",
               f"What changed {_fmt_since(since, now, lang)}:")
    if sources.unavailable:
        names = ", ".join(_UNAVAILABLE_LABELS[s][1 if en else 0]
                          for s in sources.unavailable if s in _UNAVAILABLE_LABELS)
        blocks.append(t(lang,
                        f"⚠️ Não consegui ler agora: {names}. Pergunto de novo "
                        "na próxima vez, sem perder o que chegou.",
                        f"⚠️ Couldn't read just now: {names}. I'll check again "
                        "next time without losing what arrived."))
        if len(blocks) == 1:
            # Never "tudo tranquilo" over a source that was not read.
            return header + t(lang, "\n  Nada novo no que consegui ler.",
                              "\n  Nothing new in what I could read.") \
                + "\n\n" + blocks[0]
    if not blocks:
        return header + t(lang, "\n  Nada novo — tudo tranquilo por aqui.",
                          "\n  Nothing new — all quiet here.")
    return header + "\n\n" + "\n\n".join(blocks)


_UNAVAILABLE_LABELS = {
    "emails": ("e-mails", "emails"),
    "actions": ("ações que tomei", "actions I took"),
    "events": ("agenda", "calendar"),
}


def _replay_overlap() -> timedelta:
    """How far BEHIND `now` the marker is parked (M7-A7). Resolved at call
    time so tests can tune CORTEX_DIGEST_OVERLAP_MIN."""
    try:
        minutes = float(os.environ.get("CORTEX_DIGEST_OVERLAP_MIN", "10"))
    except ValueError:
        minutes = 10.0
    return timedelta(minutes=max(minutes, 0.0))


def whats_changed(now: Optional[datetime] = None, lang: str = "pt") -> str:
    """Top-level: assemble the delta since the last call and advance the
    marker. The first call (no marker) shows the trailing 24h window.

    Delivery-loss mitigation (M7-A7): the marker is advanced to
    `now - overlap` (CORTEX_DIGEST_OVERLAP_MIN, default 10), not `now`.
    The read-that-writes pattern burns the delta when the reply is lost
    on the way out (SSE drop, TTS cut by "cala a boca") — and real
    delivery confirmation doesn't exist without a client ack. The
    overlap recovers the common case (re-ask shortly after a lost/cut
    digest re-shows the recent items) at the cost of a small repeat
    window on back-to-back asks. The marker never moves BACKWARDS
    (max with the previous value), so repeated asks don't widen it."""
    now = now or datetime.now(timezone.utc)
    last = get_last_seen()
    since = last if last is not None else (now - _FIRST_RUN_WINDOW)
    sources = gather(since, now)
    text = assemble(sources, since, now, lang=lang)
    if any(s in sources.unavailable for s in _DELTA_SOURCES):
        return text   # the window was not read whole: the marker stays
    marker = now - _replay_overlap()
    if last is not None and last > marker:
        marker = last
    set_last_seen(marker)
    return text
