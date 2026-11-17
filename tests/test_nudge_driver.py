"""
Nudge driver (innovation #2) — the PURE decision core + state ledger.

evaluate() is exercised with synthetic sources (no calendar/gmail/notify), so
the "what interrupts the user, and when" rules are pinned deterministically:
imminent events fire, quiet hours suppress, each item nudges once (dedup),
urgency ranks + caps. gather/deliver (I/O) are not touched here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import src.nudge_driver as nd
from src.nudge_driver import Nudge, NudgeSources, evaluate

NOW = datetime(2026, 6, 15, 15, 0, tzinfo=timezone.utc)
# Quiet window disabled (start == end → never quiet) unless a test sets it.
OPEN = {"quiet_start": 0, "quiet_end": 0}


def _event(minutes_ahead: float, summary="Tênis com Marcelo", eid="e1"):
    start = (NOW + timedelta(minutes=minutes_ahead)).isoformat()
    return {"id": eid, "summary": summary, "start": start}


def _email(days_old: float, frm="Marcelo <m@x.com>", subject="proposta", mid="m1"):
    return {"id": mid, "from": frm, "subject": subject,
            "date_iso": (NOW - timedelta(days=days_old)).isoformat()}


class TestEvaluateEvents:
    def test_imminent_event_fires_and_is_spoken(self):
        out = evaluate(NudgeSources(events=[_event(10)]), NOW, set(), **OPEN)
        assert len(out) == 1
        n = out[0]
        assert n.urgency == 3 and n.speak is True
        assert "Tênis com Marcelo" in n.body and "em 10 min" in n.body

    def test_event_outside_window_does_not_fire(self):
        far = evaluate(NudgeSources(events=[_event(120)]), NOW, set(), **OPEN)
        past = evaluate(NudgeSources(events=[_event(-5)]), NOW, set(), **OPEN)
        assert far == [] and past == []

    def test_imminent_window_is_tunable(self):
        ev = NudgeSources(events=[_event(40)])
        assert evaluate(ev, NOW, set(), imminent_min=20, **OPEN) == []
        assert len(evaluate(ev, NOW, set(), imminent_min=60, **OPEN)) == 1


class TestEvaluateEmails:
    def test_stale_email_fires(self):
        out = evaluate(NudgeSources(stale_emails=[_email(4)]), NOW, set(), **OPEN)
        assert len(out) == 1 and out[0].urgency == 1
        assert "Marcelo" in out[0].body and "4 dias" in out[0].body

    def test_fresh_email_does_not_fire(self):
        out = evaluate(NudgeSources(stale_emails=[_email(1)]), NOW, set(), **OPEN)
        assert out == []


class TestEvaluateReminders:
    @staticmethod
    def _reminder(due_date, name="Pagar aluguel"):
        return {"name": name, "due": "", "due_iso": f"{due_date.isoformat()}T12:00:00"}

    def test_due_today_fires(self):
        today = NOW.astimezone().date()
        out = evaluate(NudgeSources(due_reminders=[self._reminder(today)]),
                       NOW, set(), **OPEN)
        assert len(out) == 1 and out[0].urgency == 2
        assert "Pagar aluguel" in out[0].body

    def test_overdue_fires(self):
        yest = NOW.astimezone().date() - timedelta(days=1)
        out = evaluate(NudgeSources(due_reminders=[self._reminder(yest)]),
                       NOW, set(), **OPEN)
        assert len(out) == 1

    def test_overdue_is_not_due_today(self):
        """"vence hoje" was said of Monday's reminder on Thursday (s88, Codex
        lote 4)."""
        today = NOW.astimezone().date()
        old = evaluate(NudgeSources(due_reminders=[self._reminder(today - timedelta(days=3))]),
                       NOW, set(), **OPEN)
        assert old[0].body == f"Pagar aluguel — venceu em {today - timedelta(days=3):%d/%m}."
        now_due = evaluate(NudgeSources(due_reminders=[self._reminder(today)]),
                           NOW, set(), **OPEN)
        assert now_due[0].body == "Pagar aluguel — vence hoje."

    def test_future_does_not_fire(self):
        tomorrow = NOW.astimezone().date() + timedelta(days=1)
        out = evaluate(NudgeSources(due_reminders=[self._reminder(tomorrow)]),
                       NOW, set(), **OPEN)
        assert out == []

    def test_no_due_iso_ignored(self):
        out = evaluate(NudgeSources(due_reminders=[{"name": "x", "due_iso": ""}]),
                       NOW, set(), **OPEN)
        assert out == []


class TestRankingAndDedup:
    def test_urgency_ranks_and_caps(self):
        src = NudgeSources(events=[_event(10)], stale_emails=[_email(5)])
        out = evaluate(src, NOW, set(), max_per_tick=1, **OPEN)
        assert len(out) == 1 and out[0].urgency == 3  # event beats email

    def test_full_ranking_order(self):
        today = NOW.astimezone().date()
        src = NudgeSources(
            events=[_event(10)],
            due_reminders=[{"name": "r", "due_iso": f"{today.isoformat()}T12:00:00"}],
            stale_emails=[_email(5)])
        out = evaluate(src, NOW, set(), max_per_tick=3, **OPEN)
        assert [n.urgency for n in out] == [3, 2, 1]  # event > reminder > email

    def test_dedup_drops_already_nudged(self):
        src = NudgeSources(events=[_event(10, eid="e1")])
        first = evaluate(src, NOW, set(), **OPEN)
        already = {first[0].id}
        assert evaluate(src, NOW, already, **OPEN) == []

    def test_cap_zero_returns_nothing(self):
        src = NudgeSources(events=[_event(10)])
        assert evaluate(src, NOW, set(), max_per_tick=0, **OPEN) == []


class TestQuietHours:
    def test_quiet_hours_suppress_everything(self):
        local_h = NOW.astimezone().hour
        # A 2h window centred on now's LOCAL hour → guaranteed quiet.
        out = evaluate(NudgeSources(events=[_event(10)]), NOW, set(),
                       quiet_start=local_h, quiet_end=(local_h + 1) % 24)
        assert out == []

    def test_wrap_past_midnight(self):
        night = datetime(2026, 6, 15, 23, 30, tzinfo=timezone.utc).astimezone()
        morning = night.replace(hour=9)
        assert nd._in_quiet_hours(night, 22, 8) is True
        assert nd._in_quiet_hours(morning, 22, 8) is False


class TestHelpers:
    def test_sender_email_and_name(self):
        assert nd._sender_email("Lara <lara@x.com>") == "lara@x.com"
        assert nd._sender_email("bare@y.com") == "bare@y.com"
        assert nd._sender_name("Lara <lara@x.com>") == "Lara"
        assert nd._sender_name("<just@addr.com>") == "just@addr.com"


class TestNudgedLedger:
    def test_roundtrip_and_prune(self, tmp_path):
        path = str(tmp_path / "nudged.json")
        nd.add_nudged(["a", "b"], NOW, path=path)
        assert set(nd.get_nudged(path)) == {"a", "b"}
        # An entry older than keep_days is pruned on the next write.
        nd.add_nudged(["c"], NOW + timedelta(days=5), path=path, keep_days=2)
        assert set(nd.get_nudged(path)) == {"c"}

    def test_missing_file_is_empty(self, tmp_path):
        assert nd.get_nudged(str(tmp_path / "nope.json")) == {}


class TestMainKillSwitch:
    def test_kill_switch_off_is_quiet_noop(self, monkeypatch, capsys):
        monkeypatch.setenv("CORTEX_NUDGE", "0")
        assert nd.main(["--stdout"]) == 0
        assert "desligados" in capsys.readouterr().out


class TestOnlyWhatWasShownCounts:
    """A banner that failed was recorded as nudged all the same: the next
    tick dropped it as already seen, and it never came back (s88, Codex
    lote 4). Only what notify() posted goes in the ledger now."""

    def _main(self, tmp_path, monkeypatch, posted: bool):
        import src.core.heartbeat as hb
        path = str(tmp_path / "nudged.json")
        monkeypatch.setattr(nd.get_nudged, "__defaults__", (path,))
        monkeypatch.setattr(nd.add_nudged, "__defaults__",
                            (path,) + nd.add_nudged.__defaults__[1:])
        today = datetime.now().date()
        src = NudgeSources(due_reminders=[{"name": "Pagar aluguel",
                                           "due_iso": f"{today}T09:00:00"}])
        monkeypatch.setattr(nd, "gather", lambda now: src)
        # Never quiet: evaluate() binds its defaults at definition, so the
        # module constants alone left this failing from 22h to 08h.
        monkeypatch.setitem(nd.evaluate.__kwdefaults__, "quiet_start", 0)
        monkeypatch.setitem(nd.evaluate.__kwdefaults__, "quiet_end", 0)
        shown = []
        monkeypatch.setattr(hb, "notify",
                            lambda title, body: shown.append(body) or posted)
        nd.main(["--notify"])
        return shown, nd.get_nudged(path)

    def test_a_failed_banner_is_tried_again(self, tmp_path, monkeypatch):
        shown, ledger = self._main(tmp_path, monkeypatch, posted=False)
        assert shown and ledger == {}
        shown, ledger = self._main(tmp_path, monkeypatch, posted=True)
        assert shown == ["Pagar aluguel — vence hoje."] and len(ledger) == 1

    def test_a_shown_banner_is_not_repeated(self, tmp_path, monkeypatch):
        self._main(tmp_path, monkeypatch, posted=True)
        shown, _ = self._main(tmp_path, monkeypatch, posted=True)
        assert shown == []


class TestAStillPendingItemIsNotRenudged:
    """The ledger prunes entries older than 2 days. Monday's overdue reminder,
    still open on Thursday, lost its entry when another one was delivered —
    and was nudged again (s88, Codex lote 4). What is still a candidate is
    kept, whatever its age."""

    def test_monday_s_reminder_on_thursday(self, tmp_path):
        path = str(tmp_path / "nudged.json")
        monday = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
        a = {"name": "Pagar aluguel", "due_iso": "2026-10-12T09:00:00"}
        first = evaluate(NudgeSources(due_reminders=[a]), monday, set(), **OPEN)
        nd.add_nudged([n.id for n in first], monday, path=path)
        thursday = monday + timedelta(days=3)
        b = {"name": "Renovar seguro", "due_iso": "2026-10-15T09:00:00"}
        src = NudgeSources(due_reminders=[a, b])
        fired = evaluate(src, thursday, set(nd.get_nudged(path)), **OPEN)
        assert [n.body.split(" —")[0] for n in fired] == ["Renovar seguro"]
        nd.add_nudged([n.id for n in fired], thursday, path=path,
                      live={n.id for n in nd.candidates(src, thursday)})
        later = thursday + timedelta(minutes=20)
        assert evaluate(src, later, set(nd.get_nudged(path)), **OPEN) == []

    def test_what_left_is_still_pruned(self, tmp_path):
        path = str(tmp_path / "nudged.json")
        nd.add_nudged(["gone"], NOW, path=path)
        nd.add_nudged(["new"], NOW + timedelta(days=5), path=path, live={"new"})
        assert set(nd.get_nudged(path)) == {"new"}
