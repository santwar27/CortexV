"""
Calendar formatting i18n (M4-A9, review Fable 5).

The calendar was PT-only with one EN leak in the middle: `_format_when`
used strftime's %a, which answers in the PROCESS locale (locale C under
launchd) — so a PT reply carried "Tue 14/06". And `list_events` baked
"(sem título)" into the DATA, which both leaked PT into EN replies and
defeated the consumers' own localised fallbacks (digest's t() never
fired). gmail's `format_email_list(lang)` was already the house pattern;
the calendar caught up here.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.integrations.calendar_client import _format_when, format_events_list


# 2026-06-11 is a Thursday; 2026-06-14 is a Sunday.
_THU = "2026-06-11T14:30:00+01:00"
_SUN = "2026-06-14T09:00:00+01:00"


class TestFormatWhen:
    def test_weekday_pt(self):
        assert _format_when(_THU, "pt") == "qui 11/06 14:30"
        assert _format_when(_SUN, "pt") == "dom 14/06 09:00"

    def test_weekday_en(self):
        assert _format_when(_THU, "en") == "Thu 11/06 14:30"
        assert _format_when(_SUN, "en") == "Sun 14/06 09:00"

    def test_default_is_pt(self):
        # The old %a gave "Thu" here in locale C — the regression this
        # file exists to prevent.
        assert _format_when(_THU).startswith("qui")

    def test_garbage_iso_passes_through(self):
        assert _format_when("not-a-date", "en") == "not-a-date"
        assert _format_when("", "pt") == "?"


class TestFormatEventsList:
    def _events(self, summary="Reunião"):
        return [{"start": _THU, "summary": summary, "location": ""}]

    def test_empty_pt_and_en(self):
        assert format_events_list([]) == "Sem eventos no período."
        assert format_events_list([], lang="en") == "No events in this period."

    def test_row_en(self):
        out = format_events_list(self._events("Standup"), lang="en")
        assert out == "1. Thu 11/06 14:30 — Standup"

    def test_untitled_placeholder_localised(self):
        assert "(sem título)" in format_events_list(self._events(""))
        out_en = format_events_list(self._events(""), lang="en")
        assert "(untitled)" in out_en and "sem título" not in out_en

    def test_location_appended(self):
        ev = [{"start": _THU, "summary": "Almoço", "location": "Casa"}]
        assert format_events_list(ev).endswith("— Almoço @ Casa")


class TestListEventsDataIsNeutral:
    def test_missing_summary_stays_empty(self):
        # The data layer must NOT pre-fill a PT placeholder — that's the
        # formatter's job, per language. digest.py's own t() fallback
        # only fires if the key arrives empty.
        fake_service = MagicMock()
        fake_service.events.return_value.list.return_value.execute.return_value = {
            "items": [{"id": "e1", "start": {"dateTime": _THU},
                       "end": {"dateTime": _THU}}],
        }
        with patch("src.integrations.calendar_client._service",
                   return_value=fake_service):
            from src.integrations.calendar_client import list_events
            events = list_events()
        assert events[0]["summary"] == ""


class TestToolPathThreadsLang:
    def test_execute_tool_unstaged_list_events_en(self):
        from src.core.orchestrator import CortexOrchestrator
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._reply_lang = "en"
        with patch("src.integrations.calendar_client._service") as svc:
            svc.return_value.events.return_value.list.return_value \
               .execute.return_value = {"items": []}
            out = orch._execute_tool_unstaged("list_events", {})
        assert out == "No events in this period."
