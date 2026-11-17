"""Bounded execution in a disposable project copy."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import Mock

import pytest

from src.core import code_commands, code_roots, code_runner, code_sandbox


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CORTEX_CODE_ROOTS_FILE", raising=False)
    folder = tmp_path / "cortex_runner"
    folder.mkdir()
    (folder / "data").write_bytes(b"original\x00\xff")
    assert code_roots.add(str(folder))[0]
    before = set(Path(tempfile.gettempdir()).glob("cortex-code-run-*"))
    threads = set(threading.enumerate())
    processes, popen = [], subprocess.Popen

    def start(*args, **kwargs):
        proc = popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", start)
    yield folder
    assert set(Path(tempfile.gettempdir()).glob("cortex-code-run-*")) == before
    assert not (set(threading.enumerate()) - threads)
    for proc in processes:
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)


@pytest.fixture
def run(project, monkeypatch):
    log = Mock()
    monkeypatch.setattr(code_runner.audit, "log", log)

    def invoke(profile=None, cancelled=None):
        before = {p.name: p.read_bytes() for p in project.iterdir()}
        result = None
        try:
            result = code_runner.run_tests(
                code_commands.tests_profile(str(project)) if profile is None else profile, cancelled)
            return result
        finally:
            assert {p.name: p.read_bytes() for p in project.iterdir()} == before
            log.assert_called_once()
            args, kwargs = log.call_args
            assert args == ("–", "code", "tests_run")
            payload = kwargs["payload"]
            assert set(payload) == {"outcome", "exit_code", "duration_ms", "output_bytes", "truncated"}
            if result is not None:
                assert payload == dict(outcome=result.status, exit_code=result.exit_code,
                    duration_ms=round(result.duration_s * 1000), output_bytes=result.total_bytes,
                    truncated=result.truncated)
            else:
                assert payload["outcome"] == "error"

    return invoke


@pytest.fixture
def script(monkeypatch):
    def install(source, timeout=30):
        def command(profile, workspace):
            assert Path(workspace, "data").read_bytes() == b"original\x00\xff"
            return dict(argv=[sys.executable, "-c", source], cwd=workspace, env=profile["env"],
                        timeout_s=timeout, max_output_bytes=profile["max_output_bytes"])
        monkeypatch.setattr(code_sandbox, "contained_command", command)
    return install


def test_pass_and_sanitize(run, script):
    raw = "ok\x1b[31m\r\n```\x00\x1f\x7f\x80\x9f\t\n````"
    script(f"import sys; sys.stdout.buffer.write({raw.encode()!r} + b'\\xff'); sys.stderr.write('ERR')")
    result = run()
    assert result.status == "passed" and result.exit_code == 0
    assert result.total_bytes == len(raw.encode()) + 4 and not result.truncated
    assert "ok�[31m\n'''�����\t\n''''�" in result.output and "ERR" in result.output
    assert not any(c in result.output for c in ("\x1b", "\r", "```"))
    assert result.detail == ""


@pytest.mark.parametrize("source,code", [("import sys; sys.exit(3)", 3), ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", None)])
def test_failed(run, script, source, code):
    import signal

    script(source)
    result = run()
    assert result.status == "failed"
    assert result.exit_code == (code if code is not None else -signal.SIGKILL)


@pytest.mark.parametrize("stop,fallback", [(False, False), (True, False), (False, True)])
def test_timeout_or_stop(run, script, monkeypatch, stop, fallback):
    script("import time; time.sleep(60)", timeout=1)
    if fallback:
        monkeypatch.setattr(code_runner.os, "killpg", Mock(side_effect=OSError("unavailable")))
    started = time.monotonic()
    result = run(cancelled=(lambda: time.monotonic() - started > .3) if stop else None)
    assert result.status == ("stopped" if stop else "timeout")
    assert result.exit_code < 0 and result.duration_s < 5
    assert time.monotonic() - started < 5
    if fallback:
        code_runner.os.killpg.assert_called_once()


@pytest.mark.parametrize("size", [4000, 2_000_000])
def test_output_tail(run, script, size):
    script(f"import sys; sys.stdout.write('x' * {size} + '\\nEND\\n')")
    result = run()
    assert result.status == "passed" and result.truncated
    assert result.total_bytes == size + 5
    assert len(result.output) == code_runner.SHOWN_CHARS == 3000
    assert result.output.rstrip().endswith("END")


def test_missing_executable(run, monkeypatch):
    monkeypatch.setattr(code_sandbox, "contained_command", lambda p, w:
        dict(argv=[str(Path(w, "missing"))], cwd=w, env=p["env"],
             timeout_s=30, max_output_bytes=65536))
    result = run()
    assert result.status == "error" and result.detail == "FileNotFoundError"


@pytest.mark.parametrize("failure,status", [("sandbox", "unavailable"), ("copy", "too_big"),
                                          ("invalid", "error"), ("command", "error")])
def test_no_start(run, monkeypatch, failure, status):
    popen = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    if failure == "copy":
        monkeypatch.setattr(code_sandbox, "prepare_workspace", Mock(side_effect=ValueError("too big")))
    elif failure == "invalid":
        prepare = Mock(side_effect=AssertionError("must not copy"))
        monkeypatch.setattr(code_sandbox, "prepare_workspace", prepare)
    else:
        error = code_sandbox.SandboxUnavailable if failure == "sandbox" else ValueError
        monkeypatch.setattr(code_sandbox, "contained_command", Mock(side_effect=error("denied")))
    result = run(profile={} if failure == "invalid" else None)
    assert result.status == status and result.exit_code is None
    assert result.total_bytes == 0 and result.output == "" and not result.truncated
    if failure == "invalid":
        assert result.detail == code_commands.validate({})[1]
        prepare.assert_not_called()
    popen.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_exception_cleanup(run, script, error):
    script("import time; time.sleep(60)")
    started = time.monotonic()

    def cancelled():
        if time.monotonic() - started > .3:
            raise error()
        return False

    with pytest.raises(error):
        run(cancelled=cancelled)


@pytest.mark.parametrize("case", ["passing", "failing", "child"])
def test_real_project(project, run, case):
    if not code_sandbox.available():
        pytest.skip("sandbox-exec unavailable here")
    statement = {"passing": "self.assertTrue(True)", "failing": "self.fail('fictitious')",
                 "child": "subprocess.run([sys.executable, '-c', 'pass'])"}[case]
    (project / "test_example.py").write_text("import subprocess, sys, unittest\n"
        "class Example(unittest.TestCase):\n    def test_pass(self): pass\n"
        f"    def test_case(self): {statement}\n")
    result = run()
    assert result.status == ("passed" if case == "passing" else "failed")
    assert result.exit_code == (0 if case == "passing" else 1)
    assert {"passing": "OK", "failing": "FAILED (failures=1)",
            "child": "PermissionError"}[case] in result.output
