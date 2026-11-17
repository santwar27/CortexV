"""One contained test run, called by the "sim" to "roda os testes em <pasta>"."""
from dataclasses import dataclass
import os
import re
import select
import signal
import subprocess
import threading
import time
from typing import Literal

from src.core import code_commands, code_sandbox
from src.core.audit import audit

SHOWN_CHARS = 3000


@dataclass(frozen=True)
class TestRun:
    status: Literal["passed", "failed", "timeout", "stopped", "unavailable", "too_big", "error"]
    exit_code: int | None
    duration_s: float
    output: str
    total_bytes: int
    truncated: bool
    detail: str = ""


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    proc.wait(timeout=2)


def run_tests(profile, cancelled=None) -> TestRun:
    started = time.monotonic()
    workspace = proc = reader = None
    status, detail = "error", ""
    tail, total_bytes, reader_errors = bytearray(), 0, []
    stop_reader = threading.Event()

    def read_output(cap):
        nonlocal total_bytes
        try:
            while not stop_reader.is_set():
                # A bounded select also lets exceptional cleanup stop the reader.
                if not select.select([proc.stdout], [], [], .05)[0]:
                    continue
                chunk = os.read(proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                total_bytes += len(chunk)
                tail.extend(chunk)
                del tail[:-cap]
        except Exception as error:
            reader_errors.append(error)

    def execute():
        nonlocal workspace, proc, reader
        ok, reason = code_commands.validate(profile)
        if not ok:
            return "error", reason
        try:
            workspace = code_sandbox.prepare_workspace(profile["cwd"])
        except ValueError as error:
            return "too_big", str(error)
        try:
            command = code_sandbox.contained_command(profile, workspace)
        except code_sandbox.SandboxUnavailable as error:
            return "unavailable", str(error)
        except ValueError as error:
            return "error", str(error)
        try:
            proc = subprocess.Popen(command["argv"], cwd=command["cwd"], env=command["env"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        except OSError as error:
            return "error", type(error).__name__
        deadline = time.monotonic() + command["timeout_s"]
        reader = threading.Thread(target=read_output, args=(command["max_output_bytes"],),
                                  name="cortex-code-output")
        reader.start()
        while proc.poll() is None:
            if reader_errors:
                raise reader_errors[0]
            if cancelled is not None and cancelled():
                _kill(proc)
                return "stopped", ""
            if time.monotonic() >= deadline:
                _kill(proc)
                return "timeout", ""
            time.sleep(.05)
        return ("passed" if proc.returncode == 0 else "failed"), ""

    try:
        status, detail = execute()
    finally:
        try:
            try:
                if proc is not None and proc.poll() is None:
                    _kill(proc)
            finally:
                if reader is not None and reader.ident is not None:
                    reader.join(timeout=1)
                    if reader.is_alive():
                        stop_reader.set()
                        reader.join(timeout=1)
                        raise RuntimeError("output reader did not finish")
                if reader_errors:
                    raise reader_errors[0]
        except BaseException:
            status = "error"
            raise
        finally:
            try:
                try:
                    if proc is not None and proc.stdout is not None:
                        proc.stdout.close()
                finally:
                    if workspace is not None:
                        code_sandbox.discard_workspace(workspace)
            finally:
                output = tail.decode("utf-8", errors="replace").replace("\r", "")
                output = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "�", output)
                output = re.sub(r"`{3,}", lambda match: "'" * len(match[0]), output)
                result = TestRun(status, proc.returncode if proc is not None else None,
                    time.monotonic() - started, output[-SHOWN_CHARS:], total_bytes,
                    total_bytes > len(tail) or len(output) > SHOWN_CHARS, detail)
                audit.log("–", "code", "tests_run", payload={"outcome": result.status,
                    "exit_code": result.exit_code, "duration_ms": round(result.duration_s * 1000),
                    "output_bytes": result.total_bytes, "truncated": result.truncated})
    return result
