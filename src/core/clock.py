"""The local clock — "now" in Pedro's timezone, by NAME (s85).

The reminder and calendar routes resolved "amanhã às 10" against
datetime.now(timezone.utc), so the time was built as 10:00 UTC. Calendar
events survived it (the Google client strips the offset and sends the
wall-clock time with the local zone name); reminders did not — the
Reminders client honours the offset, so in British Summer Time "às 10"
became an 11:00 reminder while the confirmation said 10:00.

A fixed offset (datetime.now().astimezone()) would still be wrong across a
daylight-saving change: on 24/10/2026 "amanhã às 10" is 25/10, which is GMT.
A ZoneInfo zone resolves the offset per date, so this module returns now in
the host's IANA zone (Europe/London), read from /etc/localtime.

local_zone() is that zone as a tzinfo (s88, Codex lote 1). The Reminders
client and the confirmation both drop the ISO offset and keep the wall-clock
time; they used TODAY's fixed offset to put it back, so said on 24/10 the
reminder for 25/10 10:00 rang at 09:00 GMT under a "(BST)" confirmation.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo


def _iana_name() -> Optional[str]:
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        return None
    return target.split("zoneinfo/")[-1] if "zoneinfo/" in target else None


def local_iana_timezone() -> str:
    """The host's IANA zone name ('Europe/London'), or 'UTC' when it cannot
    be read. Google Calendar wants this form — not 'BST'."""
    return _iana_name() or "UTC"


def local_zone() -> tzinfo:
    """The host's zone, resolving the offset per DATE. Today's fixed offset
    only when the name cannot be read — local, at least, never UTC."""
    name = _iana_name()
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001 — an unknown name falls back below
            pass
    return datetime.now().astimezone().tzinfo


def local_now() -> datetime:
    """Now, aware, in the host's zone — with per-date daylight saving."""
    return datetime.now(local_zone())


@dataclass(frozen=True)
class RelativeTime:
    delta: timedelta
    span: tuple[int, int]


@dataclass(frozen=True)
class RelativeTimes:
    offsets: tuple[RelativeTime, ...] = ()
    durations: tuple[RelativeTime, ...] = ()


_NUMBERS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8,
    "nove": 9, "dez": 10, "a": 1, "an": 1, "one": 1, "two": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}
_NUMBER = r"(?:\d+|" + "|".join(_NUMBERS) + r")"
_HOURS = r"(?:horas?|hours?|h)"
_MINUTES = r"(?:minutos?|minutes?|mins?)"
_OFFSET_CUE = r"daqui\s+a|dentro\s+de|em|in"
# Lexical intent gate for reminder dispatch; quantities are still parsed and
# bounded only by parse_relative_times. An ordinary "em casa" is not a time.
RELATIVE_OFFSET_PREFIX = (
    r"(?:" + _OFFSET_CUE + r")\s+(?=\d|(?:" + "|".join(_NUMBERS) + r"|meia|half)\b)"
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:(?P<offset>" + _OFFSET_CUE + r")"
    r"|(?P<duration>por|durante|de|for))\s+"
    r"(?:"
    r"(?P<half>meia\s+hora|half\s+an?\s+hour)"
    r"|(?P<compact_h>\d+)h(?P<compact_m>\d{1,2})"
    r"|(?P<hours>" + _NUMBER + r")\s*" + _HOURS +
    r"(?:\s+(?:e|and)\s+(?:"
    r"(?P<plus_half>meia|a\s+half)"
    r"|(?P<plus_minutes>" + _NUMBER + r")\s*" + _MINUTES + r"))?"
    r"|(?P<minutes>" + _NUMBER + r")\s*" + _MINUTES +
    r")(?!\w)", re.IGNORECASE,
)


def parse_relative_times(text: str) -> RelativeTimes:
    """PT/EN elapsed offsets and durations, with exact cue-inclusive spans.

    A quantity without a cue is never an offset. Durations have their own
    cues and collection; neither kind accepts zero or more than seven days.
    Keeping every match lets callers refuse to resolve multiple offsets.
    """
    offsets, durations = [], []

    def number(value):
        return int(value) if value.isdigit() else _NUMBERS[value.lower()]

    for match in _RELATIVE_TIME_RE.finditer(text):
        if match["half"]:
            minutes = 30
        elif match["compact_h"]:
            if int(match["compact_m"]) >= 60:
                continue
            minutes = int(match["compact_h"]) * 60 + int(match["compact_m"])
        elif match["hours"]:
            minutes = number(match["hours"]) * 60
            if match["plus_half"]:
                minutes += 30
            elif match["plus_minutes"]:
                minutes += number(match["plus_minutes"])
        else:
            minutes = number(match["minutes"])
        if 0 < minutes <= 7 * 24 * 60:
            target = offsets if match["offset"] else durations
            target.append(RelativeTime(timedelta(minutes=minutes), match.span()))
    return RelativeTimes(tuple(offsets), tuple(durations))


_OTHER_TIME_RE = re.compile(
    r"\b(?:"
    r"(?:[àa]s?|at)\s+\d{1,2}\b"
    r"|\d{1,2}(?:h(?:\d{1,2})?|:\d{2}|\s*[ap]\.?m\.?)\b"
    r"|hoje|amanh[ãa]|depois\s+de\s+amanh[ãa]|today|tomorrow|tonight"
    r"|(?:segunda|ter[çc]a|quarta|quinta|sexta)(?:[ -]feira)?|s[áa]bado|domingo"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2}|dia\s+\d{1,2}"
    r")\b", re.IGNORECASE,
)


def has_other_time_expression(text: str, times: RelativeTimes) -> bool:
    """Detect competing times outside the parsed offset/duration spans.

    Mask spans with spaces so quantities such as 'em 2h30' cannot be
    mistaken for clock times and words on either side cannot join up.
    """
    for item in sorted(times.offsets + times.durations,
                       key=lambda item: item.span, reverse=True):
        start, end = item.span
        text = text[:start] + " " * (end - start) + text[end:]
    return bool(_OTHER_TIME_RE.search(text))


def add_elapsed(now: datetime, delta: timedelta) -> datetime:
    """Add elapsed time through UTC, preserving the named zone across DST."""
    if now.tzinfo is None:
        return now + delta
    return (now.astimezone(timezone.utc) + delta).astimezone(now.tzinfo)


def ensure_event_end(args: dict) -> dict:
    """Copy event args, repairing an invalid end before staging/preview/run.

    Read aware ISOs by the same local wall-clock rule as the calendar and
    previews. Leave an invalid/missing start to the caller's validation.
    """
    args = dict(args)
    start_iso = args.get("start_iso")
    end_iso = args.get("end_iso")
    invalid_hour = r"[Tt ](?:2[4-9]|[3-9]\d|\d{3,}:)"
    if isinstance(start_iso, str) and re.search(invalid_hour, start_iso):
        raise ValueError("create_event: invalid start hour")
    start = local_wall_time(start_iso) if isinstance(start_iso, str) else None
    if start is None:
        if isinstance(end_iso, str) and re.search(invalid_hour, end_iso):
            raise ValueError("create_event: invalid start time")
        return args
    end = local_wall_time(end_iso) if isinstance(end_iso, str) else None
    if end is None or end.timestamp() <= start.timestamp():
        args["end_iso"] = add_elapsed(start, timedelta(hours=1)).isoformat()
    return args


_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def local_wall_time(iso: Optional[str]) -> Optional[datetime]:
    """An ISO time read as Pedro's wall clock, aware in the local zone; None
    when it can't be parsed. Shared by the Reminders client and the
    confirmation, so what is shown is what rings.

    The wall-clock numbers win over a stray offset — the 8B writes -03:00
    for "amanhã às 11", or copies today's +01:00 onto a GMT date. But an
    offset that is one of the zone's own for that wall time is kept: it is
    what tells the two 01:15 of 25/10 apart. The focus computes its end in
    UTC and sent 01:15+00:00; read as plain 01:15 it rang 35 min before
    the focus began (s88, Codex lote 4)."""
    s = (iso or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(_OFFSET_RE.sub("", s))
        except ValueError:
            return None
    wall, zone = dt.replace(tzinfo=None), local_zone()
    if dt.utcoffset() is not None:
        for fold in (0, 1):
            candidate = wall.replace(tzinfo=zone, fold=fold)
            if candidate.utcoffset() == dt.utcoffset():
                return candidate
    return wall.replace(tzinfo=zone)
