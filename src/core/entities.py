"""
Entity brain v1 — a lightweight people graph (Onda 5, M1-now slice).

Answers "quando falei com X pela última vez?" and "com quem andei falando?"
from DETERMINISTIC, structured sources only — email From headers, calendar
attendees, and the contact book. No free-text NER (the 8B is unreliable at it;
that half waits for the Mac Studio / Gemma 4), so there are no fabricated
entities: every person here came from a real address or attendee field.

`build`/`lookup` are pure (synthetic sources); `gather`/`*_reply` do fail-soft
I/O, mirroring src/core/digest.py.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import Any, Dict, List, Optional, Set

# Leading articles to drop from a captured name ("falei com o João" → "João").
_ARTICLE_RE = re.compile(r"^(?:o|a|os|as|the)\s+", re.IGNORECASE)

# Automated / non-person senders — filtered out so "pessoas recentes" shows
# actual people, not newsletters and notifications. Matched on the address
# local-part (before @). M7-A8: the old list also had
# info/support/team/news/hello/contact — common local-parts of REAL people
# at small domains (a journalist at news@..., a founder at hello@...) that
# were silently dropped. Those are gone; what stays is unequivocally
# machine. A saved contact additionally bypasses this filter entirely (the
# user already said it's a person — see build()). "jobs" only in the
# plural (s88, Codex lote 13): the singular is a first name, and
# "Job <job@empresa.com>" vanished from the graph; the job boards keep
# being caught (jobs-listings@, jobalerts-noreply@, job-alerts@).
_NONPERSON_RE = re.compile(
    r"(?:^|[._-])(?:no[._-]?reply|donotreply|noreply|"
    r"do[._-]?not[._-]?reply|notifications?|alerts?|mailer|bounce|"
    r"newsletter|updates?|marketing|automated|postmaster|daemon|jobs)"
    r"(?:$|[._-])",
    re.IGNORECASE,
)


# Past meetings are interactions; upcoming ones are not (s88, Codex lote 13
# — invariant 7). gather asked the calendar for its default window, now to
# now + 7 days, so the agenda fed only FUTURE events and "quando falei com
# Ana pela última vez?" answered with next week's meeting.
_EVENTS_LOOKBACK = timedelta(days=14)


def _is_person_addr(addr: str) -> bool:
    """True unless the address looks like an automated/bulk sender."""
    if not addr or "@" not in addr:
        return bool(addr)  # a bare display name (calendar) is kept
    local = addr.split("@", 1)[0]
    return not _NONPERSON_RE.search(local)

from src.core.digest import _parse_ts  # shared tz-aware ISO parser
from src.core.i18n import t


@dataclass
class Entity:
    """One person aggregated across sources."""
    name: str
    email: str
    last_seen: Optional[datetime] = None
    count: int = 0
    sources: Set[str] = field(default_factory=set)


def _key(addr: str, name: str) -> str:
    return (addr or "").lower() or (name or "").lower()


def build(
    emails: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    contacts: List[Any],
    now: Optional[datetime] = None,
) -> Dict[str, Entity]:
    """Fold structured sources into a people graph keyed by email (or name when
    no email). Pure. `contacts` are objects with `.name`/`.email`."""
    now = now or datetime.now(timezone.utc)
    email_to_name = {
        (c.email or "").lower(): c.name
        for c in contacts if getattr(c, "email", None)
    }
    by_key: Dict[str, Entity] = {}

    def _upsert(name: str, addr: str, when: Any, source: str) -> None:
        key = _key(addr, name)
        if not key:
            return
        # A saved contact is a person by the user's own word — it outranks
        # the local-part heuristic (M7-A8: "info@..." that IS someone real).
        is_saved_contact = (addr or "").lower() in email_to_name
        if addr and not is_saved_contact and not _is_person_addr(addr):
            return  # skip newsletters / notifications / no-reply senders
        ts = _parse_ts(when)
        disp = name or email_to_name.get((addr or "").lower(), "") or addr
        e = by_key.get(key)
        if e is None:
            by_key[key] = Entity(name=disp, email=addr or "", last_seen=ts,
                                 count=1, sources={source})
            return
        e.count += 1
        e.sources.add(source)
        if ts is not None and (e.last_seen is None or ts > e.last_seen):
            e.last_seen = ts
        # Upgrade a placeholder (email-as-name) to a real display name.
        if ("@" in (e.name or "") or not e.name) and disp and "@" not in disp:
            e.name = disp

    for m in emails:
        name, addr = parseaddr(m.get("from", "") or "")
        _upsert(name, addr, m.get("date_iso"), "e-mail")

    for ev in events:
        start = _parse_ts(ev.get("start"))
        if start is not None and start > now:
            continue  # not happened yet — an appointment, not an interaction
        for addr in (ev.get("attendees") or []):
            nm = email_to_name.get((addr or "").lower(), "")
            _upsert(nm, addr, ev.get("start"), "agenda")

    return by_key


def lookup(query: str, entities: Dict[str, Entity]) -> Optional[Entity]:
    """Best match for `query` (case-insensitive substring on name or email).
    Ties break to the most recently seen. None when nothing matches."""
    q = (query or "").strip().lower()
    if not q:
        return None
    matches = [
        e for e in entities.values()
        if q in (e.name or "").lower() or q in (e.email or "").lower()
    ]
    if not matches:
        return None
    return max(matches, key=lambda e: e.last_seen or datetime.min.replace(tzinfo=timezone.utc))


class _Gathered(tuple):
    """(emails, events, contacts), plus the sources that could not be read.
    Still unpacks as the plain 3-tuple gather always returned."""
    unavailable: List[str] = []


def gather() -> tuple:
    """Fetch the structured sources, fail-soft. Returns (emails, events,
    contacts); `.unavailable` names the ones that failed (s88, Codex lote 13
    — invariant 7): a source that failed and one with nothing in it used to
    look the same, so Gmail down made "não achei interações com a Ana"."""
    emails: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    contacts: List[Any] = []
    unavailable: List[str] = []
    try:
        from src.integrations.gmail_client import list_recent
        emails = list_recent(count=50) or []
    except Exception:  # noqa: BLE001
        unavailable.append("e-mail")
    try:
        from src.integrations.calendar_client import list_events
        now = datetime.now(timezone.utc)
        events = list_events(time_min=(now - _EVENTS_LOOKBACK).isoformat(),
                             time_max=now.isoformat(), max_results=50) or []
    except Exception:  # noqa: BLE001
        unavailable.append("agenda")
    try:
        from src.integrations.contacts import ContactBook
        contacts = ContactBook().list_all() or []
    except Exception:  # noqa: BLE001
        unavailable.append("contatos")
    out = _Gathered((emails, events, contacts))
    out.unavailable = unavailable
    return out


# Graph cache (M7-A5): gather() costs 2 serial network calls (50 emails +
# 25 events) and the people graph changes slowly — same argument as the
# Spotify library cache (M4). TTL resolved at call time so tests can tune it.
_graph_cache: Dict[str, Any] = {}


def invalidate_cache() -> None:
    """Drop the cached graph (tests / explicit refresh)."""
    _graph_cache.clear()


def _cached_entities() -> Dict[str, Entity]:
    """gather()+build() behind a TTL cache. An EMPTY graph is not cached:
    it usually means the sources failed transiently (gather is fail-soft),
    and pinning 'no interactions' for the whole TTL would turn a blip into
    a 5-minute lie. Neither is a graph built with a source missing (s88,
    Codex lote 13): it kept answering without Gmail for five minutes after
    Gmail was back. What was missing is kept in `_graph_cache["missing"]`
    for the replies to say."""
    ttl = float(os.environ.get("CORTEX_ENTITIES_TTL_S", "300"))
    ts = _graph_cache.get("ts")
    if ts is not None and (time.monotonic() - ts) < ttl:
        _graph_cache["missing"] = []
        return _graph_cache["entities"]
    got = gather()
    missing = list(getattr(got, "unavailable", []) or [])
    emails, events, contacts = got
    entities = build(emails, events, contacts)
    _graph_cache["missing"] = missing
    if entities and not missing:
        _graph_cache["ts"] = time.monotonic()
        _graph_cache["entities"] = entities
    return entities


_MISSING_LABELS = {
    "e-mail": ("o e-mail", "email"),
    "agenda": ("a agenda", "the calendar"),
    "contatos": ("os contatos", "contacts"),
}


def _missing_note(lang: str) -> str:
    """" (não consegui ler o e-mail agora)" — or "" when every source was read."""
    missing = _graph_cache.get("missing") or []
    if not missing:
        return ""
    idx = 1 if lang == "en" else 0
    names = (" and " if lang == "en" else " e ").join(
        _MISSING_LABELS.get(m, (m, m))[idx] for m in missing)
    return t(lang, f" (não consegui ler {names} agora)",
             f" (I couldn't read {names} just now)")


def _fmt_date(ts: Optional[datetime], lang: str = "pt") -> str:
    if ts is not None:
        return ts.astimezone().strftime("%d/%m")
    return t(lang, "data desconhecida", "unknown date")


# Internal source tags ("e-mail"/"agenda") → display label per language.
_SOURCE_LABELS = {
    "e-mail": ("e-mail", "email"),
    "agenda": ("agenda", "calendar"),
}


def _sources_label(sources: Set[str], lang: str) -> str:
    idx = 1 if lang == "en" else 0
    labels = sorted(_SOURCE_LABELS.get(s, (s, s))[idx] for s in sources)
    joiner = " and " if lang == "en" else " e "
    return joiner.join(labels) or ("email" if lang == "en" else "e-mail")


def last_contact_reply(query: str, lang: str = "pt") -> str:
    """"Quando falei com X pela última vez?" — deterministic answer."""
    query = _ARTICLE_RE.sub("", (query or "").strip())
    entities = _cached_entities()
    e = lookup(query, entities)
    note = _missing_note(lang)
    if e is None:
        if note:
            return t(lang,
                     f"Não achei interações recentes com '{query}' no que consegui ler"
                     f"{note}.",
                     f"I found no recent interactions with '{query}' in what I could "
                     f"read{note}.")
        return t(lang,
                 f"Não achei interações recentes com '{query}' no e-mail ou na agenda.",
                 f"I found no recent interactions with '{query}' in email or calendar.")
    via = _sources_label(e.sources, lang)
    return t(lang,
             f"Sua última interação com {e.name} foi em {_fmt_date(e.last_seen, lang)} "
             f"(via {via}; {e.count} no total recente){note}.",
             f"Your last interaction with {e.name} was on {_fmt_date(e.last_seen, lang)} "
             f"(via {via}; {e.count} recent in total){note}.")


def recent_people_reply(limit: int = 8, lang: str = "pt") -> str:
    """"Com quem andei falando?" — most recent people, deterministic."""
    entities = _cached_entities()
    note = _missing_note(lang)
    if not entities:
        if note:
            return t(lang, f"Não consegui montar a lista de pessoas{note}.",
                     f"I couldn't build the list of people{note}.")
        return t(lang,
                 "Não tenho interações recentes registradas (e-mail/agenda vazios).",
                 "I have no recent interactions on record (email/calendar empty).")
    ranked = sorted(
        entities.values(),
        key=lambda e: e.last_seen or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    header = t(lang, "Pessoas com quem você interagiu recentemente:",
               "People you've interacted with recently:")
    last_word = "última" if lang != "en" else "last"
    lines = [header]
    for e in ranked[:limit]:
        lines.append(f"  • {e.name} — {last_word} {_fmt_date(e.last_seen, lang)} ({e.count})")
    if note:
        lines.append(t(lang, f"  Lista incompleta{note}.", f"  Incomplete list{note}."))
    return "\n".join(lines)
