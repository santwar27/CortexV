"""
Job heartbeat — liveness watchdog for the scheduled jobs (Onda 4b).

Generalises the "Studier missed a night and nobody noticed" pain (s38) into one
watchdog over ALL the cron-like launchd jobs. For each job it reads the most
recent mtime of the log/artefact that job writes EVERY run, and flags any job
that's been silent past its expected cadence.

Liveness signal = file mtime, deliberately, not an audit event: it captures a
run that fired but produced nothing (e.g. the Studier on an empty-queue night
writes its log but logs no `run_end`). Absence of the artefact = the job didn't
fire = exactly what we want to catch.

Scope: only fixed-cadence jobs (studier nightly, backup nightly, weekly). NOT
the on-demand / continuous ones (uvicorn, voice, arbiter shadow) — for those,
silence means "Pedro didn't use it", not failure; use `make doctor`/`make smoke`.

Run:  venv/bin/python -m src.core.heartbeat            (print only — `make heartbeat`)
      venv/bin/python -m src.core.heartbeat --notify   (also macOS-notify stale jobs)

The --notify path is what the launchd job runs daily; `make heartbeat` is the
manual read. `check`/`evaluate` are pure (injectable mtime source); I/O and the
osascript call are fail-soft.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class JobSpec:
    """A scheduled job + how to tell it's alive. `globs` are file patterns the
    job touches every run; the newest mtime among them is its last heartbeat.

    `optional=True` marks a job whose plist ships in the repo but may not be
    loaded into launchd yet (daily/dream — P0 #4 decision pending): NO
    artefact ever → 'dormente', not an alarm. The moment the job writes its
    first log the cadence watch arms itself automatically."""
    name: str
    globs: Tuple[str, ...]
    max_silence: timedelta
    cadence: str  # human label for the report
    optional: bool = False


# The fixed-cadence jobs. mtime of the log/artefact each writes per run.
# (uvicorn is the always-on service — watched by PID via `uvicorn_alive`,
# not by mtime; see M11-A3.)
JOBS: Tuple[JobSpec, ...] = (
    JobSpec("studier", ("logs/nightly.out.log",), timedelta(hours=26), "diário ~02h"),
    # Weekly since 2026-07-08 (launchd/com.cortex.backup.plist, Weekday 0):
    # a 26h window called Monday's backup silent every week (s88, Codex lote
    # 10). tests/test_heartbeat.py ties this window to the plist's schedule.
    JobSpec("backup", ("~/Backups/cortex/cortex_*.tar.gz",), timedelta(days=8), "semanal dom ~03h"),
    JobSpec("weekly", ("data/reviews/*.md",), timedelta(days=8), "semanal dom ~04h"),
    JobSpec("daily", ("logs/daily.out.log",), timedelta(hours=26), "diário ~08h",
            optional=True),
    JobSpec("dream", ("logs/dream.out.log",), timedelta(hours=26), "diário ~02h30",
            optional=True),
)

# The always-on service whose death is the gravest failure the watchdog can
# catch: launchd's KeepAlive respawns a clean crash, but a crash-LOOP (bad
# .env, port taken) leaves the label with no PID — and the whole system
# offline while every cadence job still looks green.
_UVICORN_LABEL = "com.cortex.uvicorn"


@dataclass
class JobStatus:
    name: str
    cadence: str
    last: Optional[datetime]
    age: Optional[timedelta]
    stale: bool
    dormant: bool = False  # optional job that never ran — plist not loaded


def newest_mtime(globs: Tuple[str, ...]) -> Optional[datetime]:
    """Newest mtime (UTC) across all files matching any glob, or None when
    nothing matches. Fail-soft: an unreadable file is skipped."""
    newest: Optional[datetime] = None
    for pattern in globs:
        for path in glob.glob(os.path.expanduser(pattern)):
            try:
                m = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            except OSError:
                continue
            if newest is None or m > newest:
                newest = m
    return newest


def evaluate(job: JobSpec, last: Optional[datetime], now: datetime) -> JobStatus:
    """Pure: turn a job + its last-heartbeat into a status. No signal at all,
    or older than max_silence, → stale. An `optional` job with no signal at
    all is dormant (not loaded), not stale — but once it HAS run, going
    silent past the cadence alarms like any other."""
    if last is None:
        if job.optional:
            return JobStatus(job.name, job.cadence, None, None, False, dormant=True)
        return JobStatus(job.name, job.cadence, None, None, True)
    age = now - last
    return JobStatus(job.name, job.cadence, last, age, age > job.max_silence)


def check(
    jobs: Tuple[JobSpec, ...] = JOBS,
    *,
    now: Optional[datetime] = None,
    mtime_fn: Callable[[Tuple[str, ...]], Optional[datetime]] = newest_mtime,
) -> List[JobStatus]:
    """Evaluate every job. `mtime_fn` is injectable so the logic is testable
    without touching the filesystem."""
    now = now or datetime.now(timezone.utc)
    return [evaluate(j, mtime_fn(j.globs), now) for j in jobs]


def _fmt_age(age: Optional[timedelta]) -> str:
    if age is None:
        return "nunca"
    secs = age.total_seconds()
    if secs < 3600:
        return f"há {int(secs // 60)} min"
    if secs < 86400:
        return f"há {int(secs // 3600)}h"
    return f"há {int(secs // 86400)} dias"


def uvicorn_alive(runner: Optional[Callable[[], str]] = None) -> Optional[bool]:
    """Liveness of the always-on uvicorn service, by PID in `launchctl list`.

    True = label has a live PID; False = label absent or loaded-without-PID
    (a crash-loop looks exactly like this between respawns); None = launchctl
    itself failed — unknown, deliberately NOT treated as an alarm (a watchdog
    must not cry wolf because its own probe broke). `runner` is the test
    hook: a callable returning raw launchctl stdout."""
    try:
        out = runner() if runner is not None else subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001 — probe failure ≠ service failure
        return None
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == _UVICORN_LABEL:
            return parts[0] != "-"
    return False


def render(statuses: List[JobStatus], uvicorn_ok: Optional[bool] = None) -> str:
    """Render the watchdog as a terse pt-BR bullet report (no tables).
    `uvicorn_ok` (from `uvicorn_alive`) adds the always-on row; None when
    the probe itself failed or the caller skipped it."""
    lines = ["Heartbeat de jobs:"]
    if uvicorn_ok is True:
        lines.append("  ✓ uvicorn (always-on) — vivo")
    elif uvicorn_ok is False:
        lines.append("  ⚠ uvicorn (always-on) — FORA DO AR (sem PID no launchd)")
    elif uvicorn_ok is None:
        lines.append("  · uvicorn (always-on) — não verificado (launchctl indisponível)")
    stale = [s for s in statuses if s.stale]
    for s in statuses:
        if s.dormant:
            lines.append(f"  · {s.name} ({s.cadence}) — dormente (plist não carregado)")
            continue
        mark = "⚠" if s.stale else "✓"
        tail = "SILENCIOSO" if s.stale else "ok"
        lines.append(f"  {mark} {s.name} ({s.cadence}) — {tail}, última {_fmt_age(s.age)}")
    problems = len(stale) + (1 if uvicorn_ok is False else 0)
    if problems:
        names = ", ".join(s.name for s in stale)
        if uvicorn_ok is False:
            names = ("uvicorn, " + names) if names else "uvicorn"
        lines.append(f"\n⚠ {problems} job(s) silencioso(s): {names}. Verifique o launchd.")
    else:
        lines.append("\n✓ Todos os jobs agendados rodaram dentro da janela.")
    return "\n".join(lines)


def _as_applescript_str(s: str) -> str:
    """Quote a Python string as an AppleScript literal (mirrors the escaping
    in reminders_client)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_osascript(script: str) -> bool:
    """Run an osascript one-liner, fail-soft. Returns True on success."""
    try:
        r = subprocess.run(["osascript", "-e", script], timeout=10,
                            capture_output=True, text=True)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 — a watchdog must never crash on notify
        return False


def notify(title: str, message: str) -> bool:
    """Post a native macOS notification banner. Fail-soft."""
    script = (
        f"display notification {_as_applescript_str(message)} "
        f"with title {_as_applescript_str(title)}"
    )
    return _run_osascript(script)


def notify_stale(statuses: List[JobStatus], uvicorn_ok: Optional[bool] = None) -> int:
    """Send ONE combined banner for the stale jobs (uvicorn down counts).
    Returns the problem count (0 → no notification sent)."""
    stale = [s for s in statuses if s.stale]
    parts = [f"{s.name} ({_fmt_age(s.age)})" for s in stale]
    if uvicorn_ok is False:
        parts.insert(0, "uvicorn (FORA DO AR)")
    if not parts:
        return 0
    notify("⚠️ Cortex — jobs silenciosos", ", ".join(parts))
    return len(parts)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Liveness watchdog for the scheduled jobs.")
    parser.add_argument("--notify", action="store_true",
                        help="also send a macOS notification for stale jobs (used by launchd)")
    args = parser.parse_args(argv)

    statuses = check()
    uv = uvicorn_alive()
    print(render(statuses, uvicorn_ok=uv))
    if args.notify:
        notify_stale(statuses, uvicorn_ok=uv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
