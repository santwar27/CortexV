"""Dormant bounded Poppler jobs. An exited process is not validated PDF content.

Jobs belong to one synchronous caller; do not stage/run/close a job concurrently.
No macOS memory rlimit is effective: callers must cap pages and render dimensions.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Literal
import unicodedata

from src.core import pdf_sandbox
from src.core.audit import audit

MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_INPUTS = 3
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_STDOUT_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 4096
MAX_OUTPUT_FILES = 100
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
TOOL_TIMEOUT_S = 30
JOB_TIMEOUT_S = 60
_POLL_S = .05
_CHUNK = 8192

# Runs in a fresh, isolated interpreter, never in the threaded API after fork.
# Arguments remain data; execv preserves the exact contained argv and fixed env.
# Reset SIGXFSZ because Python ignores it; Poppler must inherit the default action.
_LIMIT_EXEC = """import os, resource, signal, sys
try:
    limit = int(sys.argv[1])
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    os.execv(sys.argv[2], sys.argv[2:])
except Exception:
    os.write(2, b'cortex-pdf-helper-error')
    sys.exit(125)
"""


@dataclass(frozen=True)
class ToolRun:
    status: Literal["exited", "timeout", "stopped", "too_big", "unavailable", "error"]
    exit_code: int | None
    duration_s: float
    stdout: bytes
    stderr_tail: str
    outputs: list[str]
    detail: str = ""


def _discard(path, parent_fd=None):
    """Stream deletion via directory descriptors; never traverse a link.

    The reaped parser may have made its scratch directories read-only. Restore
    owner access without following links, and avoid rmtree's full directory list.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, dir_fd=parent_fd)
    except PermissionError:
        os.chmod(path, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        fd = os.open(path, flags, dir_fd=parent_fd)
    try:
        os.fchmod(fd, 0o700)
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _discard(entry.name, fd)
                else:
                    os.unlink(entry.name, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(path, dir_fd=parent_fd)


class PdfJob:
    """Private scratch lifetime; stage only paths already authorised by the caller."""

    def __init__(self):
        self.root = self.input_dir = self.output_dir = None
        self.deadline = 0.0
        self._active = False
        self._count = self._bytes = 0

    def __enter__(self):
        if self.root is not None:
            raise ValueError("PDF job cannot be reused")
        self.deadline = time.monotonic() + JOB_TIMEOUT_S
        self.root = Path(os.path.realpath(tempfile.mkdtemp(prefix="cortex-pdf-job-")))
        try:
            self.root.chmod(0o700)
            self.input_dir, self.output_dir = self.root / "in", self.root / "out"
            self.input_dir.mkdir(mode=0o700)
            self.output_dir.mkdir(mode=0o700)
            self._active = True
            return self
        except BaseException:
            _discard(self.root)
            raise

    def __exit__(self, *exc):
        self._active = False
        # No parser remains running when run_tool returns.
        _discard(self.root)

    def stage(self, paths) -> list[Path]:
        """Preflight open descriptors, bounded copy, and roll back this batch on error."""
        if not self._active:
            raise ValueError("PDF job is not active")
        paths = list(islice(iter(paths), MAX_INPUTS + 1))
        if self._count + len(paths) > MAX_INPUTS:
            raise ValueError("PDF job allows at most 3 inputs")
        copied = []
        try:
            with ExitStack() as stack:
                sources, declared = [], self._bytes
                for path in paths:
                    # NONBLOCK prevents a substituted FIFO from hanging before fstat.
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    stack.callback(os.close, fd)
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("PDF input must be a regular file")
                    declared += info.st_size
                    if declared > MAX_INPUT_BYTES:
                        raise ValueError("PDF input bytes exceed the job limit")
                    sources.append(fd)
                actual = self._bytes
                for number, fd in enumerate(sources, self._count + 1):
                    target = self.input_dir / f"input-{number}.pdf"
                    dest = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    copied.append(target)
                    with os.fdopen(dest, "wb") as output:
                        while True:
                            if time.monotonic() >= self.deadline:
                                raise ValueError("PDF input staging exceeded the job deadline")
                            chunk = os.read(fd, min(_CHUNK, MAX_INPUT_BYTES - actual + 1))
                            if not chunk:
                                break
                            actual += len(chunk)
                            if actual > MAX_INPUT_BYTES:
                                raise ValueError("PDF input bytes exceed the job limit")
                            output.write(chunk)
                self._count += len(copied)
                self._bytes = actual
                return copied
        except BaseException as error:
            for path in copied:
                path.unlink()
            if isinstance(error, OSError):
                raise ValueError("could not open or copy PDF input") from error
            raise


def _outputs(directory):
    """Bounded no-follow inventory, counting nested entries too to prevent bypasses.

    On overflow the direct-file list is partial (at most MAX_OUTPUT_FILES + 1).
    Directories and links consume the entry budget but are never returned as files.
    """
    names, count, size, exceeded = [], 0, 0, False

    def scan(fd, direct):
        nonlocal count, size, exceeded
        with os.scandir(fd) as entries:
            for entry in entries:
                count += 1
                try:
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):
                        size += info.st_size
                        if direct:
                            names.append(entry.name)
                    elif stat.S_ISDIR(info.st_mode) and count <= MAX_OUTPUT_FILES:
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            scan(child, False)
                        finally:
                            os.close(child)
                except FileNotFoundError:
                    pass  # Renamed entries still consume the bounded scan budget.
                if count > MAX_OUTPUT_FILES or size > MAX_OUTPUT_BYTES:
                    exceeded = True
                if exceeded:
                    return

    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        scan(fd, True)
    finally:
        os.close(fd)
    return sorted(names), exceeded


def _kill_and_wait(proc):
    # Also kill any remaining group members after the leader has exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if proc.poll() is None:
            proc.kill()
    proc.wait()


def run_tool(job, tool, args, *, cancelled=None) -> ToolRun:
    """Expose process facts only; kill/reap on every exit and never log PDF data.

    Ordinary tool/setup failures return error. Ctrl+C propagates after cleanup and
    audit so the surrounding PdfJob context removes its directory as well.
    """
    started = time.monotonic()
    proc = None
    selector = selectors.DefaultSelector()
    stdout, stderr = bytearray(), bytearray()
    outputs = []
    status, detail = "error", ""
    stdout_exceeded = False

    def drain(timeout):
        nonlocal stdout_exceeded
        for key, _ in selector.select(timeout):
            try:
                chunk = os.read(key.fd, _CHUNK)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
            elif key.data == "stdout":
                remaining = MAX_STDOUT_BYTES - len(stdout)
                stdout.extend(chunk[:remaining])
                stdout_exceeded |= len(chunk) > remaining
            else:
                stderr.extend(chunk)
                del stderr[:-MAX_STDERR_BYTES]

    def execute():
        nonlocal proc, outputs
        if not job._active:
            return "error", "PDF job is not active"
        if not isinstance(tool, str) or tool not in pdf_sandbox.TOOLS:
            return "error", "unknown Poppler tool"
        if cancelled is not None and cancelled():
            return "stopped", ""
        if time.monotonic() >= job.deadline:
            return "timeout", "job deadline"
        if not pdf_sandbox.available():
            return "unavailable", "contained Poppler unavailable"
        argv, env = pdf_sandbox.contained_command(
            pdf_sandbox.discover(), tool, args, str(job.input_dir), str(job.output_dir))
        if cancelled is not None and cancelled():
            return "stopped", ""
        if time.monotonic() >= job.deadline:
            return "timeout", "job deadline"
        outputs, exceeded = _outputs(job.output_dir)
        if exceeded:
            return "too_big", "output folder limit; inventory may be partial"
        deadline = min(job.deadline, time.monotonic() + TOOL_TIMEOUT_S)
        proc = subprocess.Popen([sys.executable, "-I", "-c", _LIMIT_EXEC,
                                 str(MAX_FILE_BYTES), *argv], env=env, cwd=job.root,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True,
                                close_fds=True)
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        next_scan = 0.0
        while True:
            if stdout_exceeded:
                return "too_big", "stdout limit"
            if cancelled is not None and cancelled():
                return "stopped", ""
            now = time.monotonic()
            if now >= deadline:
                return "timeout", "tool or job deadline"
            if now >= next_scan:
                outputs, exceeded = _outputs(job.output_dir)
                if exceeded:
                    return "too_big", "output folder limit; inventory may be partial"
                next_scan = now + _POLL_S
            if proc.poll() is not None and not selector.get_map():
                return "exited", ""
            drain(min(_POLL_S, max(0, deadline - time.monotonic())))

    try:
        status, detail = execute()
    except Exception as error:
        status, detail = "error", type(error).__name__
    finally:
        try:
            if proc is not None:
                _kill_and_wait(proc)
                # After SIGKILL/reap only finite pipe buffers remain; no reader
                # thread or inherited open pipe can prolong this drain.
                while selector.get_map() and selector.select(0):
                    drain(0)
                if proc.returncode == -signal.SIGXFSZ or stdout_exceeded:
                    status, detail = "too_big", "file size or stdout limit"
                elif proc.returncode == 125 and stderr == b"cortex-pdf-helper-error":
                    status, detail = "error", "rlimit/exec helper failed"
            if job._active:
                outputs, exceeded = _outputs(job.output_dir)
                if exceeded:
                    status, detail = "too_big", "output folder limit; inventory may be partial"
        except Exception as error:
            status, detail = "error", type(error).__name__
        finally:
            selector.close()
            if proc is not None:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
            tail = "".join("�" if unicodedata.category(c) == "Cc" else c
                           for c in stderr.decode("utf-8", errors="replace"))
            result = ToolRun(status, proc.returncode if proc is not None else None,
                             time.monotonic() - started, bytes(stdout), tail, outputs, detail)
            # Invalid tool values must not become an audit exfiltration channel.
            audit.log("–", "pdf", "tool_run", payload={
                "tool": tool if isinstance(tool, str) and tool in pdf_sandbox.TOOLS else "unknown",
                "status": result.status, "exit_code": result.exit_code,
                "ms": round(result.duration_s * 1000), "outputs": len(result.outputs)})
    return result
