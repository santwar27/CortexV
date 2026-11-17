"""
Tests for src/core/heartbeat.py — the scheduled-job liveness watchdog (Onda 4b).

`evaluate`/`check` are pure (mtime source injected). `newest_mtime` is exercised
fail-soft on the real filesystem. The osascript notify path is monkeypatched —
no real banners in tests.
"""
from datetime import datetime, timedelta, timezone

from src.core import heartbeat as hb


NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _job(name="studier", max_silence=timedelta(hours=26)):
    return hb.JobSpec(name, ("logs/x.log",), max_silence, "diário")


# --- evaluate (pure) -------------------------------------------------------

def test_evaluate_fresh_is_not_stale():
    s = hb.evaluate(_job(), NOW - timedelta(hours=5), NOW)
    assert s.stale is False and s.age == timedelta(hours=5)


def test_evaluate_old_is_stale():
    s = hb.evaluate(_job(), NOW - timedelta(hours=40), NOW)
    assert s.stale is True


def test_evaluate_never_ran_is_stale():
    s = hb.evaluate(_job(), None, NOW)
    assert s.stale is True and s.last is None and s.age is None


def test_evaluate_boundary_just_within():
    s = hb.evaluate(_job(max_silence=timedelta(hours=26)), NOW - timedelta(hours=25, minutes=59), NOW)
    assert s.stale is False


# --- check (injected mtime source) -----------------------------------------

def test_check_mixes_fresh_and_stale():
    # mtime source keyed on each job's glob tuple: studier fresh, backup stale,
    # weekly fresh (within its 8-day window).
    by_glob = {
        ("logs/nightly.out.log",): NOW - timedelta(hours=9),
        # weekly backup: stale past its 8-day window (21 days — the s88 gap,
        # the agent unloaded since 27/09)
        ("~/Backups/cortex/cortex_*.tar.gz",): NOW - timedelta(days=21),
        ("data/reviews/*.md",): NOW - timedelta(days=3),
    }
    statuses = hb.check(now=NOW, mtime_fn=lambda globs: by_glob.get(globs))
    by_name = {s.name: s for s in statuses}
    assert by_name["studier"].stale is False
    assert by_name["backup"].stale is True
    assert by_name["weekly"].stale is False


def test_check_all_missing_required_stale_optional_dormant():
    # M11-A3: required jobs with no artefact alarm; optional ones (daily/
    # dream — plists shipped but possibly not loaded, P0 #4) are dormant.
    statuses = hb.check(now=NOW, mtime_fn=lambda globs: None)
    by_name = {s.name: s for s in statuses}
    assert {*by_name} == {"studier", "backup", "weekly", "daily", "dream"}
    for required in ("studier", "backup", "weekly"):
        assert by_name[required].stale and not by_name[required].dormant
    for optional in ("daily", "dream"):
        assert by_name[optional].dormant and not by_name[optional].stale


# --- newest_mtime (real FS, fail-soft) -------------------------------------

def test_newest_mtime_missing_is_none():
    assert hb.newest_mtime(("/nonexistent-xyz/*.log",)) is None


def test_newest_mtime_picks_latest(tmp_path):
    import os
    a = tmp_path / "a.md"; a.write_text("a")
    b = tmp_path / "b.md"; b.write_text("b")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    got = hb.newest_mtime((str(tmp_path / "*.md"),))
    assert got == datetime.fromtimestamp(2000, tz=timezone.utc)


# --- render ----------------------------------------------------------------

def test_render_all_ok():
    statuses = [hb.evaluate(_job(name="studier"), NOW - timedelta(hours=2), NOW)]
    out = hb.render(statuses)
    assert "✓" in out and "Todos os jobs" in out
    assert "|" not in out


def test_render_flags_stale():
    statuses = [
        hb.evaluate(_job(name="studier"), NOW - timedelta(hours=2), NOW),
        hb.evaluate(_job(name="backup"), None, NOW),
    ]
    out = hb.render(statuses)
    assert "⚠" in out and "SILENCIOSO" in out
    assert "backup" in out
    assert "1 job(s) silencioso" in out


def test_fmt_age_variants():
    assert hb._fmt_age(None) == "nunca"
    assert hb._fmt_age(timedelta(minutes=30)) == "há 30 min"
    assert hb._fmt_age(timedelta(hours=9)) == "há 9h"
    assert hb._fmt_age(timedelta(days=3)) == "há 3 dias"


# --- notify (osascript monkeypatched) --------------------------------------

def test_notify_builds_osascript(monkeypatch):
    captured = {}

    def _fake(script):
        captured["s"] = script
        return True

    monkeypatch.setattr(hb, "_run_osascript", _fake)
    assert hb.notify("Título", 'com "aspas"') is True
    assert "display notification" in captured["s"]
    assert "with title" in captured["s"]
    assert '\\"aspas\\"' in captured["s"]   # quotes escaped for AppleScript


def test_notify_stale_only_when_stale(monkeypatch):
    calls = []
    monkeypatch.setattr(hb, "notify", lambda t, m: calls.append((t, m)))
    fresh = [hb.evaluate(_job(name="studier"), NOW - timedelta(hours=2), NOW)]
    assert hb.notify_stale(fresh) == 0
    assert calls == []

    stale = fresh + [hb.evaluate(_job(name="backup"), None, NOW)]
    assert hb.notify_stale(stale) == 1
    assert len(calls) == 1 and "backup" in calls[0][1]


def test_notify_is_fail_soft(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("osascript missing")
    monkeypatch.setattr(hb.subprocess, "run", _boom)
    assert hb.notify("t", "m") is False   # swallowed, returns False


# --- main ------------------------------------------------------------------

def test_main_print_only(monkeypatch, capsys):
    monkeypatch.setattr(hb, "check", lambda **k: [
        hb.evaluate(_job(name="studier"), NOW - timedelta(hours=2),
                    datetime.now(timezone.utc))
    ])
    called = {}
    monkeypatch.setattr(hb, "uvicorn_alive", lambda: True)
    monkeypatch.setattr(hb, "notify_stale",
                        lambda s, uvicorn_ok=None: called.setdefault("n", True))
    rc = hb.main([])
    assert rc == 0
    assert "Heartbeat de jobs" in capsys.readouterr().out
    assert "n" not in called   # no --notify → no notification


def test_main_notify_flag(monkeypatch, capsys):
    monkeypatch.setattr(hb, "check", lambda **k: [])
    called = {}
    monkeypatch.setattr(hb, "uvicorn_alive", lambda: True)
    monkeypatch.setattr(hb, "notify_stale",
                        lambda s, uvicorn_ok=None: called.setdefault("n", True))
    assert hb.main(["--notify"]) == 0
    assert called.get("n") is True


# --- M11-A3: uvicorn liveness + dormant arming ------------------------------

def test_optional_job_arms_after_first_run():
    # Once an optional job HAS written its log, silence past the cadence
    # alarms like any required job — dormancy is only for "never ran".
    from datetime import timedelta
    daily = next(j for j in hb.JOBS if j.name == "daily")
    old = NOW - timedelta(hours=30)
    status = hb.evaluate(daily, old, NOW)
    assert status.stale and not status.dormant


_LAUNCHCTL_ALIVE = "123\t0\tcom.cortex.uvicorn\n-\t0\tcom.cortex.backup\n"
_LAUNCHCTL_DEAD = "-\t0\tcom.cortex.uvicorn\n"
_LAUNCHCTL_ABSENT = "-\t0\tcom.cortex.backup\n"


def test_uvicorn_alive_with_pid():
    assert hb.uvicorn_alive(runner=lambda: _LAUNCHCTL_ALIVE) is True


def test_uvicorn_no_pid_is_down():
    # Loaded-without-PID is how a crash-loop looks between respawns.
    assert hb.uvicorn_alive(runner=lambda: _LAUNCHCTL_DEAD) is False


def test_uvicorn_label_absent_is_down():
    assert hb.uvicorn_alive(runner=lambda: _LAUNCHCTL_ABSENT) is False


def test_uvicorn_probe_failure_is_unknown_not_alarm():
    def broken():
        raise OSError("launchctl missing")
    assert hb.uvicorn_alive(runner=broken) is None


def test_render_uvicorn_down_counts_as_problem():
    out = hb.render([], uvicorn_ok=False)
    assert "FORA DO AR" in out
    assert "Verifique o launchd" in out


def test_render_uvicorn_unknown_is_not_alarm():
    out = hb.render([], uvicorn_ok=None)
    assert "não verificado" in out
    assert "Todos os jobs" in out


def test_notify_includes_uvicorn_down(monkeypatch):
    sent = {}
    monkeypatch.setattr(hb, "notify", lambda t, m: sent.setdefault("msg", m))
    n = hb.notify_stale([], uvicorn_ok=False)
    assert n == 1
    assert "uvicorn" in sent["msg"]


def test_render_dormant_is_not_silencioso():
    statuses = hb.check(now=NOW, mtime_fn=lambda globs: None)
    dormant_only = [s for s in statuses if s.dormant]
    out = hb.render(dormant_only, uvicorn_ok=True)
    assert "dormente" in out
    assert "Todos os jobs" in out


def test_a_sunday_backup_is_not_silent_on_monday(tmp_path, monkeypatch):
    import os
    from datetime import datetime, timedelta, timezone
    from src.core import heartbeat as hb
    snap = tmp_path / "cortex_x.tar.gz"
    snap.write_bytes(b"x")
    sunday_3am = datetime(2026, 10, 18, 2, 0, tzinfo=timezone.utc)
    os.utime(snap, (sunday_3am.timestamp(), sunday_3am.timestamp()))
    spec = next(j for j in hb.JOBS if j.name == "backup")
    monday_9am = sunday_3am + timedelta(hours=30)
    assert monday_9am - sunday_3am < spec.max_silence
