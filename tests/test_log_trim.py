"""Log maintenance uses only tmp_path files, including nightly integration."""

import logging
import os
from unittest.mock import Mock

import pytest

from src.core import log_trim
from src.core.log_trim import trim_log


def test_missing_file_is_not_created(tmp_path):
    path = tmp_path / "missing.log"
    assert trim_log(path, limit_bytes=20, tail_bytes=5) == 0
    assert not path.exists()
    assert not path.with_name("missing.log.1").exists()


@pytest.mark.parametrize("size", [0, 19, 20])
def test_at_or_under_limit_is_untouched(tmp_path, size):
    path = tmp_path / "api.log"
    content = b"x" * size
    path.write_bytes(content)
    os.utime(path, ns=(1_000_000_000, 2_000_000_000))
    before = path.stat()

    assert trim_log(path, limit_bytes=20, tail_bytes=5) == 0

    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ino == before.st_ino
    assert not path.with_name("api.log.1").exists()


@pytest.mark.parametrize(
    ("content", "tail_bytes", "expected"),
    [
        (b"older line\npartial line\nlast\n", 15, b"last\n"),
        (b"older line\nlast\n", 5, b"last\n"),
        (b"older line\nlast", 7, b"last"),
        (b"one very long line without a newline", 5, b""),
        (b"older line\nlast\n", 0, b""),
        (b"older line\nlast\n", 100, b"older line\nlast\n"),
    ],
)
def test_trim_preserves_inode_and_bounded_line_aligned_tail(
    tmp_path, caplog, content, tail_bytes, expected
):
    path = tmp_path / "api.log"
    path.write_bytes(content)
    before = path.stat()

    with caplog.at_level(logging.INFO, logger="src.core.log_trim"):
        kept = trim_log(path, limit_bytes=10, tail_bytes=tail_bytes)

    assert kept == len(expected) <= tail_bytes
    assert path.with_name("api.log.1").read_bytes() == expected
    assert path.read_bytes() == b""
    assert path.stat().st_ino == before.st_ino
    [record] = caplog.records
    assert record.getMessage() == (
        f"Log trimmed: path={path} size_before={len(content)} bytes_kept={kept}"
    )


def test_preexisting_append_writer_continues_at_start(tmp_path):
    path = tmp_path / "api.log"
    with path.open("a") as writer:
        writer.write("older line\nlast\n")
        writer.flush()
        inode = os.fstat(writer.fileno()).st_ino

        trim_log(path, limit_bytes=10, tail_bytes=5)
        writer.write("next line\n")
        writer.flush()

        assert path.stat().st_ino == inode
        assert path.read_bytes() == b"next line\n"
        assert path.with_name("api.log.1").read_bytes() == b"last\n"


def test_old_tail_is_overwritten(tmp_path):
    path = tmp_path / "api.log"
    path.write_bytes(b"older line\nlast\n")
    tail = path.with_name("api.log.1")
    tail.write_bytes(b"old snapshot much longer than the new tail\n")

    assert trim_log(path, limit_bytes=10, tail_bytes=5) == 5

    assert tail.read_bytes() == b"last\n"
    assert path.read_bytes() == b""


@pytest.mark.parametrize("symlink_name", ["api.log", "api.log.1"])
@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_is_refused_without_touching_target(
    tmp_path, caplog, symlink_name, dangling
):
    path = tmp_path / "api.log"
    target = tmp_path / "target"
    content = b"older line\nlast\n"
    if not dangling:
        target.write_bytes(content)
        before = target.stat()
    if symlink_name != "api.log":
        path.write_bytes(content)
    link = tmp_path / symlink_name
    link.symlink_to(target)

    assert trim_log(path, limit_bytes=10, tail_bytes=5) == 0

    assert link.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert target.read_bytes() == content
        assert target.stat().st_mtime_ns == before.st_mtime_ns
    if symlink_name == "api.log.1":
        assert path.read_bytes() == content
    else:
        assert not path.with_name("api.log.1").exists()
    assert "refused" in caplog.text


def test_symlink_parent_is_refused(tmp_path, caplog):
    directory = tmp_path / "real"
    directory.mkdir()
    path = directory / "api.log"
    path.write_bytes(b"older line\nlast\n")
    link = tmp_path / "linked"
    link.symlink_to(directory, target_is_directory=True)

    assert trim_log(link / "api.log", limit_bytes=10, tail_bytes=5) == 0

    assert path.read_bytes() == b"older line\nlast\n"
    assert not path.with_name("api.log.1").exists()
    assert "refused" in caplog.text


def test_hard_link_tail_cannot_truncate_source(tmp_path, caplog):
    path = tmp_path / "api.log"
    path.write_bytes(b"older line\nlast\n")
    os.link(path, path.with_name("api.log.1"))

    assert trim_log(path, limit_bytes=10, tail_bytes=5) == 0

    assert path.read_bytes() == b"older line\nlast\n"
    assert "refused" in caplog.text


def test_copy_open_failure_leaves_source_intact(tmp_path, monkeypatch):
    path = tmp_path / "api.log"
    content = b"older line\nlast\n"
    path.write_bytes(content)
    real_open = os.open

    def fail_destination(name, *args, **kwargs):
        if name == "api.log.1":
            raise PermissionError("test copy failure")
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(log_trim.os, "open", fail_destination)
    with pytest.raises(PermissionError):
        trim_log(path, limit_bytes=10, tail_bytes=5)

    assert path.read_bytes() == content


@pytest.mark.parametrize("failure", [None, "uvicorn.out.log", "uvicorn.err.log"])
def test_nightly_trims_exact_api_logs_before_studier_and_survives_failure(
    tmp_path, monkeypatch, caplog, failure
):
    from scripts import nightly_run

    # Never let main() use the real logs directory or construct a real studier.
    monkeypatch.setattr(nightly_run, "LOGS_DIR", tmp_path)
    events = []

    def trim(path):
        events.append(path)
        if path.name == failure:
            raise RuntimeError("test trim failure")

    def build_studier():
        events.append("construct studier")
        return Mock(run=Mock(side_effect=lambda: events.append("run studier") or []))

    monkeypatch.setattr(nightly_run, "trim_log", trim)
    monkeypatch.setattr(nightly_run, "NightlyStudier", Mock(side_effect=build_studier))

    assert nightly_run.main() == 0

    assert events == [
        tmp_path / "uvicorn.out.log",
        tmp_path / "uvicorn.err.log",
        "construct studier",
        "run studier",
    ]
    if failure:
        assert f"API log trim failed: path={tmp_path / failure} error=RuntimeError" in caplog.text
