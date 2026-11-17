"""Dormant PDF job lifecycle, real bounded subprocesses, and opt-in Poppler."""
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import Mock

import pytest

from src.core import pdf_runner as runner, pdf_sandbox


@pytest.fixture
def fake(tmp_path, monkeypatch):
    processes, original = [], subprocess.Popen
    threads = set(threading.enumerate())
    roots = set(Path(tempfile.gettempdir()).glob("cortex-pdf-job-*"))
    log = Mock()
    monkeypatch.setattr(runner.audit, "log", log)
    monkeypatch.setattr(pdf_sandbox, "available", lambda: True)
    monkeypatch.setattr(pdf_sandbox, "discover", lambda: object())

    def start(argv, **kwargs):
        assert "preexec_fn" not in kwargs and not kwargs.get("shell")
        assert kwargs["start_new_session"] is True
        assert kwargs["env"] == {"LANG": "C"}
        assert argv[:3] == [sys.executable, "-I", "-c"]
        proc = original(argv, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", start)

    def install(source):
        script = tmp_path / "fake.py"
        script.write_text("import os, sys, time, signal\nfrom pathlib import Path\n"
                          "out = Path(sys.argv[1])\n" + source)

        def command(tools, tool, args, input_dir, output_dir):
            assert tool == "pdfinfo" and tools is not None
            assert Path(input_dir).name == "in" and Path(output_dir).name == "out"
            return [sys.executable, "-I", str(script), str(output_dir), *args], {"LANG": "C"}

        monkeypatch.setattr(pdf_sandbox, "contained_command", command)
        return log

    yield install
    assert set(Path(tempfile.gettempdir()).glob("cortex-pdf-job-*")) == roots
    assert not (set(threading.enumerate()) - threads)
    for proc in processes:
        assert proc.stdout.closed and proc.stderr.closed
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)


def test_job_stage_private_canonical_and_cleanup(tmp_path):
    source = tmp_path / "private title.pdf"
    source.write_bytes(b"%PDF-fictitious")
    with runner.PdfJob() as job:
        root = job.root
        assert root == root.resolve()
        assert root.stat().st_mode & 0o777 == 0o700
        assert root.parent == Path(tempfile.gettempdir()).resolve()
        staged = job.stage([source, source])
        assert staged == [job.input_dir / "input-1.pdf", job.input_dir / "input-2.pdf"]
        assert all(p.read_bytes() == source.read_bytes() for p in staged)
        assert job.stage([source]) == [job.input_dir / "input-3.pdf"]
        with pytest.raises(ValueError, match="3 inputs"):
            job.stage([source])
    assert not root.exists() and source.exists()


@pytest.mark.parametrize("fault", ["symlink", "directory", "fifo", "bytes", "count", "missing"])
def test_stage_refusal_is_atomic(tmp_path, monkeypatch, fault):
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(b"good")
    paths = [good, bad]
    if fault == "symlink":
        bad.symlink_to(good)
    elif fault == "directory":
        bad.mkdir()
    elif fault == "fifo":
        os.mkfifo(bad)
    elif fault == "bytes":
        monkeypatch.setattr(runner, "MAX_INPUT_BYTES", 7)
        bad.write_bytes(b"1234")
    elif fault == "count":
        paths = [good] * 4
    with runner.PdfJob() as job:
        with pytest.raises(ValueError, match="input|regular|open|bytes"):
            job.stage(paths)
        assert list(job.input_dir.iterdir()) == []


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_job_exception_cleanup_does_not_follow_links(tmp_path, exception):
    outside = tmp_path / "keep"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(exception):
        with runner.PdfJob() as job:
            root = job.root
            (job.output_dir / "link").symlink_to(outside, target_is_directory=True)
            raise exception()
    assert not root.exists() and sentinel.read_text() == "keep"


@pytest.mark.parametrize("failure", ["grow", "read"])
def test_stage_copy_failure_rolls_back(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"1234")
    monkeypatch.setattr(runner, "MAX_INPUT_BYTES", 10)
    original, calls = os.read, 0

    def read(fd, count):
        nonlocal calls
        calls += 1
        if calls == 3:  # First copy completed; fail during the second.
            if failure == "grow":
                source.write_bytes(b"x" * 20)
            else:
                raise OSError("read failed")
        return original(fd, count)

    monkeypatch.setattr(runner.os, "read", read)
    with runner.PdfJob() as job:
        with pytest.raises(ValueError):
            job.stage([source, source])
        assert not list(job.input_dir.iterdir())


def test_staging_budget_spans_batches(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"1234")
    monkeypatch.setattr(runner, "MAX_INPUT_BYTES", 7)
    with runner.PdfJob() as job:
        first = job.stage([source])
        with pytest.raises(ValueError, match="bytes"):
            job.stage([source])
        assert list(job.input_dir.iterdir()) == first


def test_partial_job_setup_cleanup(monkeypatch):
    original, roots = Path.mkdir, []

    def mkdir(path, *args, **kwargs):
        if path.name == "out":
            roots.append(path.parent)
            raise OSError("mkdir failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(OSError):
        with runner.PdfJob():
            pytest.fail("must not enter")
    assert roots and all(not root.exists() for root in roots)


def test_exited_is_only_a_fact_and_audit_is_private(fake):
    log = fake("(out / 'z.png').write_bytes(b'broken PNG')\n"
               "(out / 'a.txt').touch()\n(out / 'dir').mkdir()\n"
               "(out / 'dir/nested').write_bytes(b'nested')\n"
               "(out / 'link').symlink_to(out / 'z.png')\n"
               "sys.stdout.buffer.write(b'facts')\n"
               "sys.stderr.buffer.write(b'x' * 6000 + b'END\\x00\\x1b\\r\\n\\t\\x7f')\n")
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "exited" and result.exit_code == 0
    assert result.stdout == b"facts" and result.outputs == ["a.txt", "z.png"]
    assert len(result.stderr_tail) <= 4096 and "END" in result.stderr_tail
    assert not any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in result.stderr_tail)
    with pytest.raises(FrozenInstanceError):
        result.status = "ok"
    log.assert_called_once_with("–", "pdf", "tool_run", payload={
        "tool": "pdfinfo", "status": "exited", "exit_code": 0,
        "ms": round(result.duration_s * 1000), "outputs": 2})


@pytest.mark.parametrize("code", [0, 7])
def test_exit_codes(fake, code):
    fake(f"sys.exit({code})\n")
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "exited" and result.exit_code == code


@pytest.mark.parametrize("stop", [False, True])
def test_timeout_stop_cleanup(fake, monkeypatch, stop):
    fake("time.sleep(60)\n")
    monkeypatch.setattr(runner, "TOOL_TIMEOUT_S", .3)
    started = time.monotonic()
    with runner.PdfJob() as job:
        root = job.root
        result = runner.run_tool(job, "pdfinfo", [], cancelled=(
            lambda: time.monotonic() - started > .15) if stop else None)
    assert result.status == ("stopped" if stop else "timeout")
    assert result.exit_code == -signal.SIGKILL and result.duration_s < 3
    assert not root.exists()


def test_shared_job_deadline(fake, monkeypatch):
    fake("time.sleep(float(sys.argv[2]))\n")
    monkeypatch.setattr(runner, "JOB_TIMEOUT_S", .8)
    with runner.PdfJob() as job:
        assert runner.run_tool(job, "pdfinfo", [".3"]).status == "exited"
        second = runner.run_tool(job, "pdfinfo", ["60"])
        assert second.status == "timeout" and second.duration_s < .7
        third = runner.run_tool(job, "pdfinfo", ["0"])
        assert third.status == "timeout" and third.exit_code is None


@pytest.mark.parametrize("mode", ["stdout", "files", "bytes", "nested", "fsize"])
def test_caps(fake, monkeypatch, mode):
    scripts = {
        "stdout": "sys.stdout.buffer.write(b'x' * (3 * 1024 * 1024))\n",
        "files": "for i in range(101): (out / str(i)).touch()\n",
        "bytes": "(out / 'big').write_bytes(b'x' * 20000)\n",
        "nested": "(out / 'd').mkdir()\n(out / 'd/big').write_bytes(b'x' * 20000)\n",
        "fsize": "signal.signal(signal.SIGXFSZ, signal.SIG_DFL)\n"
                 "with open(out / 'big', 'wb', buffering=0) as f:\n"
                 "    while True: f.write(b'x' * 4096)\n",
    }
    fake(scripts[mode])
    if mode in ("bytes", "nested"):
        monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 10000)
    if mode == "fsize":
        monkeypatch.setattr(runner, "MAX_FILE_BYTES", 8192)
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
        if mode == "fsize":
            assert (job.output_dir / "big").stat().st_size == 8192
            assert result.exit_code == -signal.SIGXFSZ
    assert result.status == "too_big"
    assert len(result.stdout) <= runner.MAX_STDOUT_BYTES
    if mode == "stdout":
        assert len(result.stdout) == 2 * 1024 * 1024
    if mode == "files":
        assert result.outputs == sorted(map(str, range(101)))


def test_concurrent_pipe_drain(fake):
    fake("for _ in range(128):\n"
         "    os.write(2, b'e' * 8192)\n    os.write(1, b'o' * 8192)\n")
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "exited" and len(result.stdout) == 1024 * 1024
    assert result.stderr_tail == "e" * 4096


def test_stdout_exact_cap_is_allowed(fake):
    fake("sys.stdout.buffer.write(b'x' * (2 * 1024 * 1024))\n")
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "exited" and len(result.stdout) == runner.MAX_STDOUT_BYTES


def test_cleanup_unreadable_output_tree(fake):
    fake("(out / 'd').mkdir()\n(out / 'd/f').touch()\n"
         "(out / 'd').chmod(0)\nout.chmod(0)\n")
    with runner.PdfJob() as job:
        root = job.root
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "error" and not root.exists()


def test_disappearing_outputs_still_bound_inventory(tmp_path, monkeypatch):
    entry = Mock()
    entry.stat.side_effect = FileNotFoundError
    scanner = Mock()
    scanner.__enter__ = Mock(return_value=iter([entry] * 1000))
    scanner.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(runner.os, "scandir", Mock(return_value=scanner))
    names, exceeded = runner._outputs(tmp_path)
    assert names == [] and exceeded
    assert entry.stat.call_count == runner.MAX_OUTPUT_FILES + 1


@pytest.mark.parametrize("failure", ["unavailable", "command", "spawn", "cancel"])
def test_fail_soft(fake, monkeypatch, failure):
    log = fake("time.sleep(60)\n")
    if failure == "unavailable":
        monkeypatch.setattr(pdf_sandbox, "available", lambda: False)
    elif failure == "command":
        monkeypatch.setattr(pdf_sandbox, "contained_command", Mock(side_effect=ValueError("private")))
    if failure in ("unavailable", "command", "spawn"):
        start = Mock(side_effect=OSError("private"))
        monkeypatch.setattr(subprocess, "Popen", start)
    started = time.monotonic()

    def cancel():
        if time.monotonic() - started > .15:
            raise RuntimeError("private")
        return False

    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [], cancelled=(
            cancel if failure == "cancel" else None))
    assert result.status == ("unavailable" if failure == "unavailable" else "error")
    assert "private" not in result.detail
    log.assert_called_once()
    if failure in ("unavailable", "command"):
        start.assert_not_called()


def test_ctrl_c_reaps_child_and_audits(fake):
    log = fake("time.sleep(60)\n")
    started = time.monotonic()

    def cancel():
        if time.monotonic() - started > .15:
            raise KeyboardInterrupt()
        return False

    with pytest.raises(KeyboardInterrupt):
        with runner.PdfJob() as job:
            runner.run_tool(job, "pdfinfo", [], cancelled=cancel)
    log.assert_called_once()


def test_helper_failure_is_error(fake, monkeypatch):
    fake("raise AssertionError('must not execute')\n")
    monkeypatch.setattr(runner, "MAX_FILE_BYTES", 1 << 200)
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "pdfinfo", [])
    assert result.status == "error" and result.exit_code == 125


def test_unknown_tool_never_enters_audit(fake, monkeypatch):
    log = fake("pass\n")
    start = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(subprocess, "Popen", start)
    with runner.PdfJob() as job:
        result = runner.run_tool(job, "/private/document.pdf", [])
    assert result.status == "error"
    assert log.call_args.kwargs["payload"]["tool"] == "unknown"
    start.assert_not_called()


def tiny_pdf(page=200):
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page} {page}] "
               "/Resources << >> /Contents 4 0 R >>".encode(),
               b"<< /Length 0 >>\nstream\nendstream"]
    pdf, offsets = b"%PDF-1.4\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(pdf)
    pdf += b"xref\n0 5\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return pdf + f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                    or not Path(pdf_sandbox.SANDBOX_EXEC).exists()
                    or pdf_sandbox.discover() is None,
                    reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("case", ["info", "png", "fsize"])
def test_real_poppler(tmp_path, monkeypatch, case):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    source = tmp_path / "tiny.pdf"
    source.write_bytes(tiny_pdf(2400 if case == "fsize" else 200))
    if case == "fsize":
        monkeypatch.setattr(runner, "MAX_FILE_BYTES", 20 * 1024 * 1024)
    with runner.PdfJob() as job:
        root = job.root
        staged = str(job.stage([source])[0])
        args = [staged] if case == "info" else ["-singlefile", *(
            ["-png", "-scale-to", "512"] if case == "png" else ["-r", "300"]),
            staged, str(job.output_dir / "page")]
        result = runner.run_tool(job, "pdfinfo" if case == "info" else "pdftoppm", args)
        if case == "fsize":
            assert result.status == "too_big", result
            assert result.exit_code == -signal.SIGXFSZ
        else:
            assert result.status == "exited" and result.exit_code == 0, result
            if case == "info":
                assert any(line.split() == [b"Pages:", b"1"] for line in result.stdout.splitlines())
            else:
                assert result.outputs == ["page.png"]
                assert (job.output_dir / "page.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not root.exists()
