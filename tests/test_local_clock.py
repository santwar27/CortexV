"""The routes resolve times against the LOCAL zone, by name (s85).

The reminder and calendar routes passed datetime.now(timezone.utc) to the
time parser, so "amanhã às 10" was built as 10:00 UTC. The Reminders client
honours the offset: in British Summer Time the reminder landed at 11:00
while the confirmation said 10:00. A ZoneInfo zone also gets the day the
clocks change right — on 24/10/2026, "amanhã" is GMT.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core import clock
from src.core.orchestrator import CortexOrchestrator

LONDON = ZoneInfo("Europe/London")


class TestParsingInTheLocalZone:

    def test_summer_time_keeps_the_hour_said(self):
        now = datetime(2026, 10, 15, 12, 47, tzinfo=LONDON)
        iso, _ = CortexOrchestrator._parse_reminder_due("amanhã às 10", now)
        assert iso == "2026-10-16T10:00:00+01:00"

    def test_the_day_the_clocks_go_back(self):
        """Said on 24/10 (BST), tomorrow is 25/10 (GMT): the offset follows
        the DATE, not the moment it was said."""
        now = datetime(2026, 10, 24, 12, 0, tzinfo=LONDON)
        iso, _ = CortexOrchestrator._parse_reminder_due("amanhã às 10", now)
        assert iso == "2026-10-25T10:00:00+00:00"


class TestLocalNow:

    def test_reads_the_zone_name_from_the_system(self, monkeypatch):
        monkeypatch.setattr(clock.os, "readlink",
                            lambda _p: "/var/db/timezone/zoneinfo/Europe/London")
        assert clock.local_iana_timezone() == "Europe/London"
        assert clock.local_now().tzinfo == LONDON

    def test_an_unreadable_zone_is_still_local_never_utc_by_accident(self, monkeypatch):
        def boom(_p):
            raise OSError("no symlink")
        monkeypatch.setattr(clock.os, "readlink", boom)
        assert clock.local_iana_timezone() == "UTC"          # what Google gets
        now = clock.local_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == datetime.now().astimezone().utcoffset()

    def test_the_calendar_client_uses_the_same_lookup(self):
        from src.integrations import calendar_client
        assert calendar_client._local_iana_timezone is clock.local_iana_timezone


class TestTheRoutes:

    @pytest.fixture
    def orch(self, tmp_path, monkeypatch):
        from tests.test_dispatch_parity import _build_real_orch
        monkeypatch.setattr("src.core.dispatch.registry.local_now",
                            lambda: datetime(2026, 10, 15, 12, 47, tzinfo=LONDON))
        o = _build_real_orch(tmp_path)
        o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
        o.background = False
        return o

    def test_a_reminder_at_ten_is_at_ten(self, orch):
        _reply, mode = orch._process_core("me lembra de pagar a conta amanhã às 10")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["args"]["due_iso"] == "2026-10-16T10:00:00+01:00"

    def test_an_event_at_ten_is_at_ten(self, orch):
        _reply, mode = orch._process_core("marca dentista amanhã às 10")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["args"]["start_iso"] == "2026-10-16T10:00:00+01:00"


class TestTheClockChangeAfterSaying:
    """Said in BST about a GMT date (s88, Codex lote 1). The Reminders client
    and the confirmation drop the ISO offset and keep the wall clock; they
    put back TODAY's offset, so said on 24/10 the 25/10 10:00 reminder rang
    at 09:00 GMT while the confirmation said "10:00 (BST)"."""

    @pytest.fixture(autouse=True)
    def london(self, monkeypatch):
        monkeypatch.setattr(clock.os, "readlink",
                            lambda _p: "/var/db/timezone/zoneinfo/Europe/London")

    @pytest.mark.parametrize("iso, utc_hour", [
        ("2026-10-25T10:00:00+00:00", 10),     # clocks went back: GMT
        ("2027-03-28T10:00:00+01:00", 9),      # clocks went forward: BST
        # A plan step from the 8B, told to copy TODAY's offset (lote 2): the
        # wall clock still wins, so it is 10:00 GMT, not 09:00.
        ("2026-10-25T10:00:00+01:00", 10),
    ])
    def test_the_reminder_rings_at_the_hour_said(self, iso, utc_hour):
        from datetime import timezone
        from src.integrations import reminders_client as rc
        rings = datetime.fromtimestamp(rc._parse_aware_dt(iso).timestamp(),
                                       timezone.utc)
        assert (rings.hour, rings.minute) == (utc_hour, 0)

    @pytest.mark.parametrize("iso, label", [
        ("2026-10-24T10:00:00+01:00", "sáb 24/10 10:00 (BST)"),
        ("2026-10-25T10:00:00+00:00", "dom 25/10 10:00 (GMT)"),
        ("2027-03-28T10:00:00+01:00", "dom 28/03 10:00 (BST)"),
    ])
    def test_the_confirmation_names_the_zone_of_that_date(self, iso, label):
        assert CortexOrchestrator._format_iso_for_display(iso) == label


class TestTheRepeatedHour:
    """25/10, 01:00–02:00 happens twice. A focus started at the FIRST 01:50
    (BST) ends 25 min later at 01:15+00:00 — computed in UTC, right. The
    Reminders client dropped the offset and read plain 01:15 — the first
    one — so the alarm rang 35 min before the focus began (s88, Codex
    lote 4). An offset that is one of London's own is kept now."""

    @pytest.fixture(autouse=True)
    def london(self, monkeypatch):
        monkeypatch.setattr(clock.os, "readlink",
                            lambda _p: "/var/db/timezone/zoneinfo/Europe/London")

    @staticmethod
    def _rings(iso):
        """The epoch the REAL client builds — only osascript is cut off."""
        import re
        from datetime import timezone
        from src.integrations import reminders_client as rc
        sent = []
        real = rc._run
        rc._run = lambda s: sent.append(s) or "reminder id x-apple-reminder://0"
        try:
            rc.create_reminder("x", iso, list_name="Cortex Foco")
        finally:
            rc._run = real
        epoch = int(re.search(r"dateWithTimeIntervalSince1970:(\d+)", sent[0]).group(1))
        return datetime.fromtimestamp(epoch, timezone.utc)

    def test_the_focus_alarm_rings_after_the_focus_began(self):
        from datetime import timedelta, timezone
        from src.core import focus
        start = datetime(2026, 10, 25, 0, 50, tzinfo=timezone.utc).astimezone(LONDON)
        end = focus.later(start, timedelta(minutes=25))
        assert end.isoformat() == "2026-10-25T01:15:00+00:00"
        # timestamps: == between zones is False for an ambiguous time (PEP 495)
        assert self._rings(end.isoformat()).timestamp() == end.timestamp()

    @pytest.mark.parametrize("iso, utc", [
        ("2026-10-25T01:15:00+01:00", (0, 15)),      # the first 01:15 (BST)
        ("2026-10-25T01:15:00+00:00", (1, 15)),      # the second (GMT)
        ("2026-10-25T01:15:00", (0, 15)),            # no offset: the first
        ("2026-05-19T11:00:00-03:00", (10, 0)),      # the 8B's BRT: wall wins
    ])
    def test_which_instant(self, iso, utc):
        rings = self._rings(iso)
        assert (rings.hour, rings.minute) == utc

    @pytest.mark.parametrize("iso, label", [
        ("2026-10-25T01:15:00+01:00", "dom 25/10 01:15 (BST)"),
        ("2026-10-25T01:15:00+00:00", "dom 25/10 01:15 (GMT)"),
    ])
    def test_the_confirmation_says_which(self, iso, label):
        assert CortexOrchestrator._format_iso_for_display(iso) == label
