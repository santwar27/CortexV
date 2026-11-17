"""
Tests for `scripts/cortex-voice-supervise.sh` — the restart loop that keeps
the voice listener alive (s88).

The listener can't be a launchd job (a launchd-spawned listener hears a
muted mic — TCC, see launchd/com.cortex.voice.plist), so before this a crash
left Cortex deaf until Pedro noticed. The supervisor is sh, so it's tested
as a black box: CORTEX_VOICE_CMD points it at a fake listener that follows
a per-run plan (exit code, signal, or wait to be killed from outside), and
CORTEX_VOICE_NOTIFY_CMD captures the give-up banner instead of osascript.

Also pins the guard patterns in `scripts/cortex-voice-start.sh`: they must
match the real argv of the listener and the supervisor, and NOT a command
that merely names src/voice/listener.py — and `make doctor` (src/doctor.py)
looks for the same two processes.
"""
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

from src import doctor

SCRIPTS =Path(__file__).resolve().parent.parent / "scripts"
SUPERVISOR = SCRIPTS / "cortex-voice-supervise.sh"
START = SCRIPTS / "cortex-voice-start.sh"

FAKE_LISTENER = """#!/bin/sh
echo $$ >> "$FAKE_RUNS"
n=$(wc -l < "$FAKE_RUNS")
action=$(sed -n "${n}p" "$FAKE_PLAN")
case $action in
    exit*) exit ${action#exit } ;;
    kill*) kill -${action#kill } $$ ;;
    wait) exec sleep 30 ;;
esac
exit 0
"""


@pytest.fixture
def harness(tmp_path):
    fake = tmp_path / "fake_listener.sh"
    fake.write_text(FAKE_LISTENER)
    fake.chmod(0o755)
    notify = tmp_path / "notify.sh"
    notify.write_text('#!/bin/sh\necho "$1" >> "$NOTIFY_LOG"\n')
    notify.chmod(0o755)
    runs = tmp_path / "runs"
    plan = tmp_path / "plan"
    notified = tmp_path / "notified"

    def env(steps, **over):
        plan.write_text("\n".join(steps) + "\n")
        e = dict(os.environ)
        e.update({
            "CORTEX_VOICE_CMD": str(fake),
            "CORTEX_VOICE_THROTTLE_S": "0",
            "CORTEX_VOICE_NOTIFY_CMD": str(notify),
            "FAKE_RUNS": str(runs),
            "FAKE_PLAN": str(plan),
            "NOTIFY_LOG": str(notified),
        })
        e.update(over)
        return e

    def run_count():
        return len(runs.read_text().split()) if runs.exists() else 0

    def pids():
        return [int(p) for p in runs.read_text().split()] if runs.exists() else []

    return env, run_count, pids, notified


def supervise(env):
    # "sh", not "/bin/sh": keeps the test process out of the start script's
    # supervisor pattern if the user opens a Terminal mid-suite.
    return subprocess.run(["sh", str(SUPERVISOR)], env=env,
                          capture_output=True, text=True, timeout=30)


class TestPolicy:
    def test_clean_exit_ends_the_supervisor(self, harness):
        env, run_count, _, _ = harness
        r = supervise(env(["exit 0"]))
        assert r.returncode == 0
        assert run_count() == 1

    def test_exception_exit_is_restarted(self, harness):
        env, run_count, _, _ = harness
        r = supervise(env(["exit 1", "exit 0"]))
        assert r.returncode == 0
        assert run_count() == 2
        assert "reiniciando" in r.stderr

    def test_segfault_code_is_restarted(self, harness):
        env, run_count, _, _ = harness
        r = supervise(env(["exit 139", "exit 0"]))
        assert run_count() == 2

    def test_sigkill_is_restarted(self, harness):
        env, run_count, _, _ = harness
        r = supervise(env(["kill KILL", "exit 0"]))
        assert r.returncode == 0
        assert run_count() == 2

    def test_sigterm_is_a_stop_not_a_crash(self, harness):
        """`pkill -f "src.voice.listener"` (the cheatsheet's stop) must stop
        the listener for good, not hand it to the restart loop."""
        env, run_count, _, _ = harness
        r = supervise(env(["kill TERM", "exit 0"]))
        assert r.returncode == 0
        assert run_count() == 1
        assert "parado por sinal" in r.stderr

    def test_crash_loop_gives_up_and_notifies(self, harness):
        env, run_count, _, notified = harness
        r = supervise(env(["exit 1"] * 6, CORTEX_VOICE_MAX_CRASHES="3"))
        assert r.returncode == 1
        assert run_count() == 3
        assert "caiu 3 vezes" in notified.read_text()

    def test_crashes_outside_the_window_never_give_up(self, harness):
        """A crash a day is not a crash loop: each crash past the window
        starts a fresh count."""
        env, run_count, _, notified = harness
        r = supervise(env(["exit 1"] * 4 + ["exit 0"],
                          CORTEX_VOICE_MAX_CRASHES="2",
                          CORTEX_VOICE_WINDOW_S="-1"))
        assert r.returncode == 0
        assert run_count() == 5
        assert not notified.exists()


class TestLiveProcess:
    def test_killed_listener_comes_back_and_a_stop_ends_it(self, harness):
        """The real shape: the listener is running, something kills it from
        outside (a crash), a new one appears; then a SIGTERM (Pedro's stop)
        ends the supervisor with it."""
        env, run_count, pids, _ = harness
        proc = subprocess.Popen(["sh", str(SUPERVISOR)], env=env(["wait", "wait"]),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True)
        try:
            def wait_runs(n):
                deadline = time.time() + 10
                while run_count() < n and time.time() < deadline:
                    time.sleep(0.05)
                assert run_count() == n
                time.sleep(0.2)  # let `exec sleep` replace the fake shell

            wait_runs(1)
            os.kill(pids()[0], signal.SIGKILL)
            wait_runs(2)
            assert pids()[1] != pids()[0]
            os.kill(pids()[1], signal.SIGTERM)
            assert proc.wait(timeout=10) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
            for pid in pids():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _start_pattern(name):
    m = re.search(rf"^{name}='([^']+)'$", START.read_text(), re.M)
    assert m, f"{name} not found in cortex-voice-start.sh"
    return re.compile(m.group(1))


class TestStartGuard:
    def test_listener_pattern_matches_the_real_argv_only(self):
        pat = _start_pattern("LISTENER_PAT")
        live = ("/Library/Frameworks/Python.framework/Versions/3.12/Resources/"
                "Python.app/Contents/MacOS/Python -m src.voice.listener")
        assert pat.search(live)
        assert not pat.search("git add src/voice/listener.py tests/test_x.py")
        assert not pat.search("vim src/voice/listener.py")

    def test_supervisor_pattern_matches_how_start_launches_it(self):
        pat = _start_pattern("SUPERVISOR_PAT")
        launch = re.search(r"^nohup (/bin/sh \S+cortex-voice-supervise\.sh)",
                           START.read_text(), re.M)
        assert launch, "start script no longer launches the supervisor"
        assert pat.search(launch.group(1))
        assert not pat.search(f"sh {SUPERVISOR}")  # the suite's own runs
        assert not pat.search("vim scripts/cortex-voice-supervise.sh")

    def test_supervisor_default_command_is_what_the_guard_looks_for(self):
        default = re.search(r'^CMD=\$\{CORTEX_VOICE_CMD:-"([^"]+)"\}',
                            SUPERVISOR.read_text(), re.M)
        assert default
        assert _start_pattern("LISTENER_PAT").search(default.group(1))

    def test_doctor_looks_for_the_same_processes_as_the_start_script(self):
        assert doctor.LISTENER_PAT == _start_pattern("LISTENER_PAT").pattern
        assert doctor.SUPERVISOR_PAT == _start_pattern("SUPERVISOR_PAT").pattern


class TestDoctor:
    """`make doctor` tells a supervised listener from a bare one (s88)."""

    @pytest.fixture
    def found(self, monkeypatch):
        procs = {}
        monkeypatch.setattr(doctor, "_pgrep", lambda pat: procs.get(pat, []))
        return procs

    def test_supervised_listener_is_ok(self, found):
        found[doctor.LISTENER_PAT] = ["101"]
        found[doctor.SUPERVISOR_PAT] = ["100"]
        r = doctor.check_voice_daemon()
        assert r.status == doctor.OK
        assert "pid 101" in r.detail and "supervised" in r.detail

    def test_bare_listener_warns_and_says_how_to_fix_it(self, found):
        found[doctor.LISTENER_PAT] = ["101"]
        r = doctor.check_voice_daemon()
        assert r.status == doctor.WARN
        assert "no supervisor" in r.detail
        assert "cortex-voice-start.sh" in r.detail

    def test_supervisor_between_restarts_warns(self, found):
        found[doctor.SUPERVISOR_PAT] = ["100"]
        r = doctor.check_voice_daemon()
        assert r.status == doctor.WARN and "restarting" in r.detail

    def test_nothing_running_is_text_only_mode(self, found):
        r = doctor.check_voice_daemon()
        assert r.status == doctor.WARN and "text-only" in r.detail

    def test_probe_failure_is_a_soft_warn(self, monkeypatch):
        def boom(pat):
            raise OSError("no pgrep")
        monkeypatch.setattr(doctor, "_pgrep", boom)
        r = doctor.check_voice_daemon()
        assert r.status == doctor.WARN and "could not check" in r.detail
