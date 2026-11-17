"""
macOS Reminders.app integration via AppleScript.

Calls `osascript` as a subprocess to create / list reminders in the
user's default list. No OAuth — Reminders is purely local. Lembretes
sync to iPhone/iPad automatically through iCloud.

On the first call, macOS prompts the user to grant Python (the venv
binary) permission to control Reminders. This consent appears in
System Settings → Privacy & Security → Automation. Subsequent calls
run silently.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from src.core.clock import local_wall_time
from src.integrations.applescript import escape as _escape
from src.integrations.applescript import run as _applescript_run

log = logging.getLogger(__name__)

_REMINDER_ID_RE = re.compile(r"x-apple-reminder://[0-9A-Fa-f-]+")


class RemindersUnavailable(Exception):
    """Raised when AppleScript fails or Reminders.app is unreachable."""


def create_reminder(
    title: str,
    due_iso: Optional[str] = None,
    body: Optional[str] = None,
    list_name: Optional[str] = None,
) -> Dict:
    """
    Create a reminder in the user's Reminders.app. Returns a dict with
    the created title and due date for confirmation messages.

    When `list_name` is omitted (the usual case), the reminder is added
    to the user's `default list`.

    The due date is built from a UTC Unix epoch timestamp via Cocoa's
    NSDate — i.e. an absolute moment in time, not a local-time
    construction. This bypasses two iCloud Reminders quirks:
      - `set time of (current date)` produces times that iCloud
        re-interprets if the account's timezone setting disagrees
        with the Mac's locale (seen when a user's iCloud account was
        created in a different country).
      - Naive AppleScript dates have implicit local-timezone
        semantics that don't survive the sync round-trip cleanly.

    `due_iso` is read as a local wall-clock time in the zone of THAT date;
    a stray "Z" / +HH:MM is dropped, a real London one kept (_parse_aware_dt).
    If parsing fails, the reminder is created without a due date.
    """
    if not (title or "").strip():
        raise RemindersUnavailable("Reminder needs a non-empty title.")

    title_esc = _escape(title.strip())
    body_clause = f', body:"{_escape(body)}"' if body else ""
    list_target = (
        f'list "{_escape(list_name)}"' if list_name else "default list"
    )

    dt = _parse_aware_dt(due_iso)
    if dt is None:
        # No due date — bare reminder
        script = (
            'tell application "Reminders"\n'
            f'    tell {list_target}\n'
            f'        make new reminder with properties {{name:"{title_esc}"{body_clause}}}\n'
            '    end tell\n'
            'end tell'
        )
    else:
        epoch_seconds = dt.timestamp()  # UTC absolute, sync-safe
        script = (
            'use framework "Foundation"\n'
            'use scripting additions\n'
            'set absoluteDate to (current application\'s NSDate\'s '
            f'dateWithTimeIntervalSince1970:{epoch_seconds:.0f}) as date\n'
            'tell application "Reminders"\n'
            f'    tell {list_target}\n'
            f'        make new reminder with properties {{name:"{title_esc}", due date:absoluteDate{body_clause}}}\n'
            '    end tell\n'
            'end tell'
        )

    out = _run(script)
    # `make new reminder` answers with its reference ("reminder id
    # x-apple-reminder://…") — the stable handle complete_reminder needs.
    found = _REMINDER_ID_RE.search(out or "")
    return {"title": title.strip(), "due_iso": due_iso,
            "list": list_name or "default",
            "id": found.group(0) if found else None}


def ensure_list(list_name: str) -> None:
    """Create the list in the default list's account (iCloud, so it reaches
    the iPhone) if it doesn't exist yet. `tell list "X"` fails on a missing
    list, so a caller with its own list calls this first."""
    name = _escape((list_name or "").strip())
    if not name:
        raise RemindersUnavailable("ensure_list needs a list name.")
    _run(
        'tell application "Reminders"\n'
        f'    if not (exists list "{name}") then\n'
        '        tell (container of default list) to make new list with '
        f'properties {{name:"{name}"}}\n'
        '    end if\n'
        'end tell'
    )


def complete_reminder(reminder_id: str, list_name: str) -> int:
    """Mark a reminder done by its id — its alarm won't ring. Returns 1, or 0
    when the id isn't in that list (moved or deleted on the iPhone): the
    old one-liner set `completed` on an empty match without a word, and the
    caller counted it done (s88, Codex lote 4 — seen against the real app).
    Looked up INSIDE its list: against the real app (s86), a bare
    `reminder id "…"` took 130 s and then timed out at 30 s — likely
    resolved across every account (iCloud, Hotmail, Outlook) — while
    `tell list "X"` + `whose id is` took 1 s. Completing, not deleting:
    delete took 28–37 s."""
    if not _REMINDER_ID_RE.fullmatch((reminder_id or "").strip()):
        raise RemindersUnavailable(f"not a reminder id: {reminder_id!r}")
    if not (list_name or "").strip():
        raise RemindersUnavailable("complete_reminder needs the reminder's list.")
    # Counted and set through the REFERENCE: `set matches to (reminders
    # whose …)` materialises it and took 24 s against the real app; the
    # count and the set straight on the `whose` take ~1 s each (s88).
    which = f'(reminders whose id is "{reminder_id.strip()}")'
    out = _run(
        'tell application "Reminders"\n'
        f'    tell list "{_escape(list_name.strip())}"\n'
        f'        if (count of {which}) is 0 then return "0"\n'
        f'        set completed of {which} to true\n'
        '        return "1"\n'
        '    end tell\n'
        'end tell'
    )
    try:
        return int((out or "0").strip())
    except ValueError:
        return 0


def delete_reminder(title: str, list_name: Optional[str] = None,
                    reminder_id: Optional[str] = None) -> int:
    """Delete the reminder Cortex created. Returns 1 when deleted, 0 when not
    found, and the match count (>1) when it can't tell which — nothing
    deleted then.

    By `reminder_id` (what create_reminder returns) when there is one —
    completed or not, it is the one Cortex made. Without it, by exact title
    among the INCOMPLETE ones, and only when a single one matches: deleting
    the "last" of several took an older namesake (s88, Codex lote 2)."""
    rid = (reminder_id or "").strip()
    if rid and not _REMINDER_ID_RE.fullmatch(rid):
        raise RemindersUnavailable(f"not a reminder id: {reminder_id!r}")
    if not rid and not (title or "").strip():
        raise RemindersUnavailable("delete_reminder needs a non-empty title.")
    list_target = (
        f'list "{_escape(list_name)}"' if list_name else "default list"
    )
    if rid:
        which = f'(reminders whose id is "{rid}")'
    else:
        which = (f'(reminders whose name is "{_escape(title.strip())}" '
                 'and completed is false)')
    # Through the reference, never `set matches to …` — that materialises
    # the match and took 24 s against the real app (see complete_reminder).
    script = (
        'tell application "Reminders"\n'
        f'    tell {list_target}\n'
        f'        set n to count of {which}\n'
        '        if n is not 1 then return (n as text)\n'
        f'        delete {which}\n'
        '        return "1"\n'
        '    end tell\n'
        'end tell'
    )
    out = _run(script)
    try:
        return int((out or "0").strip())
    except ValueError:
        return 0


# Record/field separators for the list_reminders payload (M4-A11). ASCII
# Unit Separator (0x1F) between fields and Record Separator (0x1E) between
# reminders — control chars a human never types into a reminder name and
# that a due-date string never carries, so they can't collide with the
# data the way the old "||" + linefeed pair did (a name containing "||"
# was silently dropped; a name with a newline split into two bogus rows).
_FS = "\x1f"
_RS = "\x1e"


def list_reminders(list_name: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    List the user's incomplete reminders. When `list_name` is omitted,
    reads from the user's `default list`.

    The cap is enforced INSIDE AppleScript (M4-A11): the loop stops reading
    name/due once `limit` rows are collected, so it's O(limit) Apple events
    instead of materialising every incomplete reminder's properties and
    slicing in Python — the same shape `notes_client.list_notes` uses.
    """
    n = max(1, int(limit))
    list_target = (
        f'list "{_escape(list_name)}"' if list_name else "default list"
    )
    script = (
        'tell application "Reminders"\n'
        f'    tell {list_target}\n'
        '        set fs to (character id 31)\n'
        '        set rs to (character id 30)\n'
        '        set output to ""\n'
        '        set cnt to 0\n'
        '        repeat with r in (reminders whose completed is false)\n'
        f'            if cnt ≥ {n} then exit repeat\n'
        '            set rName to name of r\n'
        '            try\n'
        '                set dd to due date of r\n'
        '                set rDue to dd as string\n'
        # ISO from numeric components → locale-independent (the `as string`
        # form above is locale/region-formatted and unsafe to parse). Zero-pad
        # month/day/hour/minute via text -2 thru -1 of ("0" & n).
        '                set yr to (year of dd) as string\n'
        '                set mo to text -2 thru -1 of ("0" & ((month of dd as integer) as string))\n'
        '                set dy to text -2 thru -1 of ("0" & ((day of dd) as string))\n'
        '                set hh to text -2 thru -1 of ("0" & ((hours of dd) as string))\n'
        '                set mi to text -2 thru -1 of ("0" & ((minutes of dd) as string))\n'
        '                set rIso to yr & "-" & mo & "-" & dy & "T" & hh & ":" & mi & ":00"\n'
        '            on error\n'
        '                set rDue to ""\n'
        '                set rIso to ""\n'
        '            end try\n'
        '            set output to output & rName & fs & rDue & fs & rIso & rs\n'
        '            set cnt to cnt + 1\n'
        '        end repeat\n'
        '        return output\n'
        '    end tell\n'
        'end tell'
    )
    raw = _run(script)
    if not raw.strip():
        return []

    results: List[Dict] = []
    for record in raw.split(_RS):
        if _FS not in record:
            continue
        parts = record.split(_FS, 2)
        name = parts[0].strip()
        due = parts[1].strip() if len(parts) > 1 else ""
        # due_iso: locale-independent ISO (naive, LOCAL wall-clock). "" when
        # the reminder has no due date. Older payloads (2 fields) → "".
        due_iso = parts[2].strip() if len(parts) > 2 else ""
        if name:
            results.append({"name": name, "due": due, "due_iso": due_iso})
    return results[:n]


# --- helpers --------------------------------------------------------

def _run(applescript: str) -> str:
    """Execute an AppleScript and return stdout. Raises RemindersUnavailable
    on failure. Thin wrapper over the shared runner (s73) so call sites and
    the domain exception stay unchanged."""
    return _applescript_run(
        applescript, app_label="Reminders.app",
        unavailable_exc=RemindersUnavailable,
    )


# `_escape` lives in src/integrations/applescript.py now (M4-A3): one
# canonical implementation shared with notes_client and macos_controller.

def _parse_aware_dt(iso: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 string into a TIMEZONE-AWARE datetime in the
    host's local timezone — clock.local_wall_time, the rule the
    confirmation prompt uses too, so the prompt and the reminder agree.

    Why the wall clock wins: the LLM (Llama 3.1, even Q5) sometimes emits
    the wrong offset for a relative time. Observed: user says "amanhã às
    11h", Llama emits "2026-05-19T11:00:00-03:00" (BRT). An offset that is
    one of London's own for that wall time is kept — it tells the repeated
    hour of 25/10 apart (s88, Codex lote 4).
    """
    return local_wall_time(iso)
