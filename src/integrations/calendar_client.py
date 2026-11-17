"""
Google Calendar client wrapper.

Stateless: each call builds a fresh API service from stored credentials.
`list_events` is read-only; `create_event` is gated by the orchestrator's
confirmation flow (the LLM stages the call, the user must approve with
'sim' before this function executes).
"""

import base64
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.core import turn_metrics
from src.core.i18n import t
from src.integrations.google_oauth import load_credentials

log = logging.getLogger(__name__)


# The host's IANA zone name ('Europe/London'); Google Calendar requires the
# IANA form — abbreviations like 'BST' are not accepted. Shared with the
# reminder/calendar routes since s85 (src/core/clock.py).
from src.core.clock import local_iana_timezone as _local_iana_timezone  # noqa: E402


_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _strip_offset(iso: str) -> str:
    """
    Remove a trailing timezone offset/Z from an ISO 8601 string. We strip
    so we can pass a naive-looking dateTime to Google Calendar alongside
    an explicit `timeZone` field — that combination is unambiguous and
    survives whatever the LLM emits (with offset, without, or with 'Z').
    """
    iso = (iso or "").strip()
    return _OFFSET_RE.sub("", iso)


class CalendarUnavailable(Exception):
    """Raised when credentials are missing or the Calendar API errors."""


def _service():
    turn_metrics.mark_online("calendar")  # Includes refresh before any API operation.
    creds = load_credentials()
    if not creds:
        raise CalendarUnavailable(
            "Google credentials not set up. Run scripts/google_oauth_setup.py first."
        )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def list_events(
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict]:
    """
    Return upcoming events from the user's primary calendar.

    time_min/time_max: ISO 8601 strings (e.g. '2026-05-18T00:00:00Z').
    Defaults: time_min = now (UTC), time_max = time_min + 7 days.
    """
    service = _service()

    if not time_min:
        time_min = datetime.now(timezone.utc).isoformat()
    if not time_max:
        try:
            base = datetime.fromisoformat(time_min.replace("Z", "+00:00"))
        except Exception:
            base = datetime.now(timezone.utc)
        time_max = (base + timedelta(days=7)).isoformat()

    try:
        resp = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max(1, min(max_results, 50)),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
    except HttpError as e:
        raise CalendarUnavailable(f"Calendar list failed: {e}") from e

    items = resp.get("items", []) or []
    results: List[Dict] = []
    for ev in items:
        start_block = ev.get("start", {}) or {}
        end_block = ev.get("end", {}) or {}
        results.append({
            "id": ev.get("id"),
            # Data layer stays language-neutral (M4-A9): the old baked-in
            # "(sem título)" leaked PT into EN replies AND defeated the
            # consumers' own localised fallbacks (digest already had
            # t(lang, ...) that never fired because the key arrived
            # pre-filled). Formatters fill the placeholder per language.
            "summary": ev.get("summary", ""),
            "start": start_block.get("dateTime") or start_block.get("date") or "",
            "end": end_block.get("dateTime") or end_block.get("date") or "",
            "location": ev.get("location", ""),
            "attendees": [a.get("email", "") for a in ev.get("attendees", []) or []],
        })
    return results


def event_id_for(effect_id: str) -> str:
    """Choose a Calendar-compatible resource id from the approval's effect id."""
    return base64.b32hexencode(hashlib.sha256(effect_id.encode()).digest()).decode().lower().rstrip("=")


def find_event(event_id: str) -> Optional[Dict]:
    """Read one event by identity; absence is not permission to insert again."""
    try:
        return _service().events().get(calendarId="primary", eventId=event_id).execute()
    except HttpError as e:
        if e.resp.status in (404, 410):
            return None
        raise CalendarUnavailable(f"Calendar get failed: {e}") from e


def _same_event(event: Optional[Dict], body: Dict) -> bool:
    if not event or event.get("summary") != body["summary"]:
        return False
    try:
        for field in ("start", "end"):
            actual = datetime.fromisoformat(_strip_offset(event[field]["dateTime"]))
            expected = datetime.fromisoformat(body[field]["dateTime"])
            if actual.replace(microsecond=0) != expected.replace(microsecond=0):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    location: Optional[str] = None,
    description: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    event_id: Optional[str] = None,
) -> Dict:
    """
    Create an event in the user's primary calendar. Returns the created
    event's id, summary, htmlLink and start time. Should only be called
    after the orchestrator's confirmation flow approves the create.
    """
    if not summary:
        raise CalendarUnavailable("Event needs a non-empty summary.")
    if not start_iso or not end_iso:
        raise CalendarUnavailable("Event needs start_iso and end_iso.")

    service = _service()
    # Strip whatever offset the LLM may or may not have included; pass the
    # IANA timezone separately so Google interprets the wall-clock time
    # exactly as the user intended.
    tz_name = _local_iana_timezone()
    body: Dict = {
        "summary": summary,
        "start": {"dateTime": _strip_offset(start_iso), "timeZone": tz_name},
        "end":   {"dateTime": _strip_offset(end_iso),   "timeZone": tz_name},
    }
    if location:
        body["location"] = location
    if description:
        body["description"] = description
    if attendees:
        body["attendees"] = [{"email": e} for e in attendees if e]
    if event_id is not None:
        body["id"] = event_id

    existing = False
    try:
        created = service.events().insert(
            calendarId="primary",
            body=body,
        ).execute()
    except HttpError as e:
        if event_id is None or e.resp.status != 409:
            raise CalendarUnavailable(f"Calendar create failed: {e}") from e
        try:
            created = find_event(event_id)
        except Exception as check_error:
            error = CalendarUnavailable(f"Calendar event id conflict: lookup failed: {check_error}")
            error.uncertain = True
            raise error from check_error
        if not _same_event(created, body):
            error = CalendarUnavailable("Calendar event id conflict: event differs or is missing.")
            error.uncertain = True
            raise error from e
        existing = True
    except Exception:
        # A lost reply permits one read by the prechosen id, never another insert.
        if event_id is not None:
            try:
                created = find_event(event_id)
                existing = _same_event(created, body)
            except Exception:
                pass
        if not existing:
            raise

    result = {
        "id": created.get("id", ""),
        "summary": created.get("summary", summary),
        "htmlLink": created.get("htmlLink", ""),
        "start": (created.get("start", {}) or {}).get("dateTime", start_iso),
    }
    if existing:
        result["existing"] = True
    return result


def delete_event(event_id: str) -> None:
    """Delete an event from the user's primary calendar by id.

    The exact reverse of create_event — the id returned on create is a stable
    handle, so undo ("Cortex, desfaz isso") is precise here. Raises
    CalendarUnavailable on a missing id or API failure."""
    if not (event_id or "").strip():
        raise CalendarUnavailable("delete_event needs an event id.")
    service = _service()
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
    except HttpError as e:
        raise CalendarUnavailable(f"Calendar delete failed: {e}") from e


def format_events_list(events: List[Dict], lang: str = "pt") -> str:
    """Render an events list as a user-facing string (PT default; `lang`
    follows the turn — M4-A9, same contract as gmail's format_email_list).
    The rows are mostly data; the empty phrase, the untitled placeholder
    and the weekday abbreviation carry language."""
    if not events:
        return t(lang, "Sem eventos no período.", "No events in this period.")
    lines: List[str] = []
    for i, e in enumerate(events, start=1):
        when = _format_when(e.get("start", ""), lang)
        summary = e.get("summary", "") or t(lang, "(sem título)", "(untitled)")
        line = f"{i}. {when} — {summary}"
        if e.get("location"):
            line += f" @ {e['location']}"
        lines.append(line)
    return "\n".join(lines)


# Weekday abbreviations by language (M4-A9). strftime's %a answers in the
# PROCESS locale — locale C under launchd, so PT replies carried "Tue"
# (and setlocale() is process-global + not thread-safe, so the fix is a
# table, not a locale call). Index = datetime.weekday() (0 = Monday).
_WEEKDAYS = {
    "pt": ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"],
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}


def _format_when(iso: str, lang: str = "pt") -> str:
    if not iso:
        return "?"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        weekday = _WEEKDAYS["en" if lang == "en" else "pt"][dt.weekday()]
        return f"{weekday} {dt.strftime('%d/%m %H:%M')}"
    except Exception:
        return iso
