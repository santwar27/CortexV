"""
Reminder + Calendar regex-staging handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. The dispatch registry routes "me lembre de …" and
"adicione ao calendário: …" here. Both parse a `<title> <time_expr>` body
without an LLM call and STAGE an S3 `_pending_action` (the user's next sim/não
resolves it via the generic pending-action flow). Mixed in so `self._handle_*
_regex(...)` resolves through the MRO; the handlers reach the `_REMINDER_BODY_RE`
/ `_VAGUE_CALENDAR_RE` class attrs and the `_parse_reminder_due` /
`_calendar_end_iso` static helpers (which stay on the orchestrator) + the
staging surface (`_pending_action`, `_persist_pending_action`,
`_human_confirmation_prompt`) via the MRO.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from src.core.clock import (
    add_elapsed, ensure_event_end, has_other_time_expression, parse_relative_times,
)


class SchedulingMixin:
    """Reminder + calendar regex-first staging handlers."""

    def _scheduling_body(self, body: str, now: datetime, *, event: bool = False):
        times = parse_relative_times(body)
        # Let the model interpret competing times instead of choosing which
        # action an offset belongs to. No-offset clock behaviour is unchanged.
        if (len(times.offsets) > 1
                or (times.offsets and has_other_time_expression(body, times))):
            return None
        duration = (times.durations[0].delta
                    if len(times.durations) == 1 else timedelta(hours=1))
        # A reminder has no duration field: these words belong to its title.
        removed = [item.span for item in times.offsets + (times.durations if event else ())]
        if times.offsets:
            iso, assumed = add_elapsed(now, times.offsets[0].delta).isoformat(), None
        else:
            # Mask durations even for reminders so their quantities cannot
            # become clock hours. Keep original positions to preserve titles.
            clock_body = body
            for item in reversed(times.durations):
                start, end = item.span
                clock_body = clock_body[:start] + " " * (end - start) + clock_body[end:]
            match = self._REMINDER_BODY_RE.match(clock_body)
            iso, assumed = None, None
            if match:
                expr = match.group("time_expr").rstrip()
                start = match.start("time_expr")
                # A masked duration may sit between the day and clock time.
                # Remove only visible tokens so its title words survive.
                removed.extend((start + token.start(), start + token.end())
                               for token in re.finditer(r"\S+", expr))
                iso, assumed = self._parse_reminder_due(expr.strip(), now)
        title = body
        for start, end in sorted(removed, reverse=True):
            title = title[:start] + " " + title[end:]
        title = re.sub(r"\s+", " ", title).strip(" ,;.")
        if times.offsets:
            title = re.sub(r"^(?:de|que|to|about)\s+", "", title,
                           flags=re.IGNORECASE)
        return title, iso, assumed, duration

    def _assumed_time_note(self, assumed) -> str:
        """One-line addendum to the confirmation preview when the time
        parser GUESSED the hour (M1-A12c): "às 4h" → 16:00 by the PM
        assumption, bare "amanhã" → 09:00. The preview already shows the
        resolved time; this makes the guess explicit so a 4-da-manhã
        reminder doesn't sail through on a reflexive 'sim'."""
        if not assumed:
            return ""
        return "\n" + self._t(
            f"(sem período explícito — assumi {assumed}; se era outro "
            f"horário, responde 'não' e repete com 'da manhã'/'da noite')",
            f"(no explicit period — I assumed {assumed}; if you meant "
            f"another time, reply 'no' and repeat with am/pm)",
        )

    def _handle_reminder_regex(
        self, body: str, now: datetime
    ) -> Optional[Tuple[str, str]]:
        """Stage a reminder from a regex-parsed body.

        Extracts the title and optional due_iso without an LLM call, stages
        the action in self._pending_action, and returns the human-readable
        confirmation prompt + mode tuple. Returns None if the body is
        unparseable (caller should fall through to the LLM path).
        """
        # A trailing "?" (s84: "lembra da reunião amanhã às 10?") would hide
        # the time from _REMINDER_BODY_RE, which anchors it at the end — the
        # reminder got staged with no due date and the "?" in its title.
        body = body.strip().rstrip("?").strip()
        if not body:
            return None

        parsed = self._scheduling_body(body, now)
        if parsed is None:
            return None
        title, due_iso, assumed, _duration = parsed

        if not title:
            return None

        self._pending_action = {
            "type": "create_reminder",
            "args": {"title": title, "due_iso": due_iso},
        }
        self._persist_pending_action()
        return (
            self._human_confirmation_prompt(self._pending_action)
            + self._assumed_time_note(assumed),
            "AGENT_AWAITING_CONFIRMATION",
        )

    def _handle_calendar_regex(
        self, body: str, now: datetime
    ) -> Optional[Tuple[str, str]]:
        """Stage a calendar event from a regex-parsed body.

        Reuses _REMINDER_BODY_RE + _parse_reminder_due — a reminder body
        and a calendar body have the same shape (`<title> <time_expr>`).
        Unlike reminders, calendar events MUST have a start time, so an
        unparseable body returns None and the caller falls through to the
        LLM path where the model can ask the user for a date.
        """
        # A trailing "?" (s84: "lembra da reunião amanhã às 10?") would hide
        # the time from _REMINDER_BODY_RE, which anchors it at the end — the
        # reminder got staged with no due date and the "?" in its title.
        body = body.strip().rstrip("?").strip()
        if not body:
            return None

        parsed = self._scheduling_body(body, now, event=True)
        if parsed is None:
            return None
        title, start_iso, assumed, duration = parsed

        if not title or not start_iso:
            # No concrete single event. If the body just points at things
            # discussed earlier ("esses jogos", "essas reuniões") with no
            # inline date, the 8B fallback guesses — it was seen lumping three
            # games into one event with a stale time (2026-06-02). Ask for
            # specifics instead of guessing. A non-vague unparseable body
            # (a single event the parser missed) still falls through to the
            # LLM, which can ask for the date itself.
            if self._VAGUE_CALENDAR_RE.search(body):
                return (
                    self._t(
                        "Pra colocar no calendário eu preciso de cada evento com "
                        "data e hora. Me passa um por um — ex.: "
                        "\"Brasil x Marrocos, 13/06 às 23h\".",
                        "To add to the calendar I need each event with a date and "
                        "time. Give me one at a time — e.g.: "
                        "\"Brazil vs Morocco, 06/13 at 11pm\".",
                    ),
                    "CALENDAR_CLARIFY",
                )
            return None

        start = datetime.fromisoformat(start_iso)
        # fromisoformat has a fixed offset; resolve the target date in now's
        # named zone for a duration spanning a clock change.
        if now.tzinfo is not None:
            start = start.astimezone(now.tzinfo)
        end_iso = add_elapsed(start, duration).isoformat()

        self._pending_action = {
            "type": "create_event",
            "args": ensure_event_end({
                "summary": title,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }),
        }
        self._persist_pending_action()
        return (
            self._human_confirmation_prompt(self._pending_action)
            + self._assumed_time_note(assumed),
            "AGENT_AWAITING_CONFIRMATION",
        )
