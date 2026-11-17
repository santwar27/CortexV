"""Dormant publication: real jobs, isolated HOME, real exclusive macOS rename."""
from dataclasses import FrozenInstanceError
from datetime import date
import errno
import os
from pathlib import Path
import stat
import time
import unicodedata
from unittest.mock import Mock

import pytest

from src.core import pdf_publish as publisher, pdf_runner as runner


NAME = "prova - páginas 3-5 - 26-10-2026"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path.resolve() / "home"
    home.mkdir()
    (home / "Documents").mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def root(home):
    root = home.joinpath(*publisher.ROOT_PARTS)
    root.mkdir(parents=True)
    return root


@pytest.fixture
def job():
    with runner.PdfJob() as job:
        (job.output_dir / "a.pdf").write_bytes(b"first\x00content")
        (job.output_dir / "b.png").write_bytes(b"second content")
        yield job


@pytest.fixture
def log(monkeypatch):
    log = Mock()
    monkeypatch.setattr(publisher.audit, "log", log)
    return log


def assert_empty(result, root, status):
    assert result.status == status
    assert result.folder is None and result.files == () and result.bytes == 0
    assert not root.exists() or list(root.iterdir()) == []


def test_folder_name():
    when = date(2026, 10, 26)
    assert publisher.folder_name("prova.pdf", "páginas 3-5", when) == NAME
    assert publisher.folder_name(".. /a:\x00\n\x7fb.PDF", ". /:\tfeito", when) == (
        "a b - feito - 26-10-2026")
    assert publisher.folder_name(".pdf", "\x00 : /", when) == "PDF - resultado - 26-10-2026"
    assert publisher.folder_name("", "", when) == "PDF - resultado - 26-10-2026"
    assert publisher.folder_name("é" * 100 + ".pdf", "ç" * 60, when) == (
        "é" * 80 + " - " + "ç" * 40 + " - 26-10-2026")
    name = publisher.folder_name("cafe\u0301.pdf", "pa\u0301ginas", when)
    assert name == "café - páginas - 26-10-2026"
    assert unicodedata.is_normalized("NFC", name)


def test_next_free_read_only(home):
    assert publisher.next_free(NAME) == NAME
    assert not (home / "Documents/Cortex").exists()


def test_next_free_counts_all_occupied_entries(root):
    assert publisher.next_free(NAME) == NAME
    (root / NAME).write_bytes(b"keep")
    assert publisher.next_free(NAME) == NAME + " (2)"
    (root / (NAME + " (2)")).mkdir()
    assert publisher.next_free(NAME) == NAME + " (3)"
    (root / (NAME + " (3)")).symlink_to(root / "absent")
    assert publisher.next_free(NAME) == NAME + " (4)"
    for number in range(4, 100):
        (root / f"{NAME} ({number})").touch()
    assert publisher.next_free(NAME) is None


def test_publish_complete_files_modes_and_counts_only(home, job, log):
    result = publisher.publish(job, ["b.png", "a.pdf"], NAME)
    root = home.joinpath(*publisher.ROOT_PARTS)
    assert result.status == "published"
    assert Path(result.folder) == root / NAME
    assert result.files == ("b.png", "a.pdf") and result.bytes == 27
    assert sorted(p.name for p in (root / NAME).iterdir()) == ["a.pdf", "b.png"]
    for name in result.files:
        target = root / NAME / name
        assert target.read_bytes() == (job.output_dir / name).read_bytes()
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        assert target.stat().st_ino != (job.output_dir / name).stat().st_ino
    assert [p.name for p in root.iterdir()] == [NAME]
    log.assert_called_once_with("–", "pdf", "publish", payload={
        "status": "published", "files": 2, "bytes": 27})
    with pytest.raises(FrozenInstanceError):
        result.status = "error"


@pytest.mark.parametrize("kind", ["directory", "file", "dangling", "live"])
def test_destination_occupied_at_rename_is_untouched(root, job, monkeypatch, kind):
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"keep")
    original = publisher._rename_exclusive
    destination = root / NAME
    before = None

    def occupy(root_fd, source, name):
        nonlocal before
        assert not destination.exists()
        assert sorted(p.name for p in (root / source).iterdir()) == ["a.pdf", "b.png"]
        if kind == "directory":
            destination.mkdir()
        elif kind == "file":
            destination.write_bytes(b"keep")
        else:
            destination.symlink_to(outside if kind == "live" else outside / "absent")
        before = destination.lstat()
        return original(root_fd, source, name)

    monkeypatch.setattr(publisher, "_rename_exclusive", occupy)
    result = publisher.publish(job, ["a.pdf", "b.png"], NAME)
    assert result.status == "occupied" and result.folder is None and result.files == ()
    assert destination.lstat().st_ino == before.st_ino
    assert [p.name for p in root.iterdir()] == [NAME]
    assert (outside / "sentinel").read_bytes() == b"keep"
    if kind == "file":
        assert destination.read_bytes() == b"keep"
    elif kind == "directory":
        assert list(destination.iterdir()) == []
    else:
        assert destination.is_symlink()


@pytest.mark.parametrize("part", ["Documents", "Cortex", "PDF"])
@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_root_refuses_links_and_non_directories(home, job, part, kind):
    outside = home / "elsewhere"
    outside.mkdir()
    parent = home
    for component in publisher.ROOT_PARTS:
        path = parent / component
        if component == part:
            if path.exists():
                path.rmdir()
            if kind == "symlink":
                path.symlink_to(outside)
            else:
                path.touch()
            break
        path.mkdir(exist_ok=True)
        parent = path
    assert publisher.publish(job, ["a.pdf"], NAME).status == "unavailable"
    assert list(outside.iterdir()) == []
    assert publisher.next_free(NAME) is None


def test_root_permission_denial_is_unavailable(root, job, monkeypatch):
    original = os.open

    def deny(path, flags, *args, **kwargs):
        if path == "Documents":
            raise PermissionError(errno.EACCES, "private path")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(publisher.os, "open", deny)
    assert_empty(publisher.publish(job, ["a.pdf"], NAME), root, "unavailable")


@pytest.mark.parametrize("fault,status", [
    ("slash", "invalid"), ("dotdot", "invalid"), ("missing", "invalid"),
    ("symlink", "invalid"), ("hardlink", "invalid"), ("directory", "invalid"),
    ("fifo", "invalid"), ("duplicates", "invalid"), ("zero", "invalid"),
    ("count", "too_big"), ("bytes", "too_big"), ("inventory_overflow", "too_big"),
])
def test_invalid_files_leave_no_publication(root, job, monkeypatch, fault, status):
    files = ["a.pdf", "bad"]
    target = job.output_dir / "bad"
    if fault == "slash":
        files = ["sub/a.pdf"]
    elif fault == "dotdot":
        files = [".."]
    elif fault == "symlink":
        target.symlink_to(job.output_dir / "a.pdf")
    elif fault == "hardlink":
        os.link(job.output_dir / "a.pdf", target)
    elif fault == "directory":
        target.mkdir()
    elif fault == "fifo":
        os.mkfifo(target)
    elif fault == "duplicates":
        files = ["a.pdf", "a.pdf"]
    elif fault == "zero":
        files = []
    elif fault in ("count", "inventory_overflow"):
        files = [f"{i}.pdf" for i in range(101)]
        for name in files:
            (job.output_dir / name).touch()
        if fault == "inventory_overflow":
            files = ["a.pdf"]
    elif fault == "bytes":
        monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 10)
        files = ["a.pdf"]
    assert_empty(publisher.publish(job, files, NAME), root, status)


@pytest.mark.parametrize("operation", ["write", "fsync"])
@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt])
def test_copy_failure_rolls_back_second_file(root, job, monkeypatch, operation, exception):
    original = getattr(os, operation)
    count = 0

    def fail_second(fd, *args):
        nonlocal count
        if stat.S_ISREG(os.fstat(fd).st_mode):
            count += 1
            if count == 2:
                raise exception("private diagnostic")
        return original(fd, *args)

    monkeypatch.setattr(publisher.os, operation, fail_second)
    result = publisher.publish(job, ["a.pdf", "b.png"], NAME)
    assert_empty(result, root, "error")
    assert "private" not in result.detail


def test_copy_growth_is_bounded(root, job, monkeypatch):
    monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 30)
    original = os.read
    grown = False

    def grow(fd, count):
        nonlocal grown
        if not grown:
            grown = True
            (job.output_dir / "a.pdf").write_bytes(b"x" * 31)
        return original(fd, count)

    monkeypatch.setattr(publisher.os, "read", grow)
    assert_empty(publisher.publish(job, ["a.pdf"], NAME), root, "too_big")


def test_short_writes_copy_all_bytes(root, job, monkeypatch):
    original = os.write
    monkeypatch.setattr(publisher.os, "write", lambda fd, data: original(fd, data[:2]))
    result = publisher.publish(job, ["a.pdf"], NAME)
    assert result.status == "published"
    assert (root / NAME / "a.pdf").read_bytes() == b"first\x00content"


def test_stale_cleanup_only_owned_old_real_staging(root, job):
    old = root / ".cortex-staging-old"
    old.mkdir()
    nested = old / "nested"
    nested.mkdir()
    (nested / "file").write_bytes(b"discard")
    ordinary = root / "user-folder"
    ordinary.mkdir()
    (ordinary / "sentinel").write_bytes(b"keep")
    (nested / "link").symlink_to(ordinary)
    nested.chmod(0o500)
    fresh = root / ".cortex-staging-fresh"
    fresh.mkdir()
    link = root / ".cortex-staging-x"
    link.symlink_to(ordinary)
    file = root / ".cortex-staging-file"
    file.write_bytes(b"keep")
    past = time.time() - 601
    for path in (old, ordinary, link, file):
        os.utime(path, (past, past), follow_symlinks=False)
    result = publisher.publish(job, ["a.pdf"], NAME)
    assert result.status == "published"
    assert not old.exists()
    assert fresh.is_dir() and link.is_symlink() and file.read_bytes() == b"keep"
    assert (ordinary / "sentinel").read_bytes() == b"keep"
    assert {p.name for p in root.iterdir()} == {NAME, fresh.name, link.name, file.name, ordinary.name}


@pytest.mark.parametrize("name", ["", ".hidden", "..", "a/b", "a:b", "a\x00b", "a\nb",
                                  " a", "a ", "a  b", "cafe\u0301", ".cortex-staging-x"])
def test_non_sanitised_name_is_invalid(root, job, name):
    assert_empty(publisher.publish(job, ["a.pdf"], name), root, "invalid")


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_audit_failure_does_not_change_result(root, job, log, exception):
    log.side_effect = exception("private audit failure")
    result = publisher.publish(job, ["a.pdf"], NAME)
    assert result.status == "published" and (root / NAME / "a.pdf").is_file()
    assert log.call_count == 1


def test_failure_is_audited_once_without_paths(root, job, log):
    result = publisher.publish(job, ["private missing name"], NAME)
    assert_empty(result, root, "invalid")
    log.assert_called_once_with("–", "pdf", "publish", payload={
        "status": "invalid", "files": 0, "bytes": 0})


def test_exact_byte_and_file_limits(root, job, monkeypatch):
    monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 27)
    files = ["a.pdf", "b.png"]
    for number in range(98):
        name = f"empty-{number}"
        (job.output_dir / name).touch()
        files.append(name)
    result = publisher.publish(job, files, NAME)
    assert result.status == "published" and len(result.files) == 100 and result.bytes == 27
    assert len(list((root / NAME).iterdir())) == 100


def test_stale_not_owned_by_current_user_is_preserved(root, job, monkeypatch):
    old = root / ".cortex-staging-other-owner"
    old.mkdir()
    os.utime(old, (time.time() - 1000,) * 2)
    monkeypatch.setattr(publisher.os, "getuid", lambda: old.stat().st_uid + 1)
    assert publisher.publish(job, ["a.pdf"], NAME).status == "published"
    assert old.is_dir()


def test_staging_collision_is_never_reused_or_deleted(root, job, monkeypatch):
    existing = root / ".cortex-staging-collision"
    existing.mkdir()
    (existing / "sentinel").write_bytes(b"keep")
    monkeypatch.setattr(publisher.secrets, "token_hex", lambda count: "collision")
    assert publisher.publish(job, ["a.pdf"], NAME).status == "error"
    assert [p.name for p in root.iterdir()] == [existing.name]
    assert (existing / "sentinel").read_bytes() == b"keep"


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_replacement_after_inventory_is_rejected(root, job, monkeypatch, replacement):
    original = runner._outputs

    def replace(directory):
        result = original(directory)
        target = job.output_dir / "a.pdf"
        target.unlink()
        if replacement == "symlink":
            target.symlink_to(job.output_dir / "b.png")
        else:
            os.mkfifo(target)
        return result

    monkeypatch.setattr(runner, "_outputs", replace)
    assert_empty(publisher.publish(job, ["a.pdf"], NAME), root, "invalid")


def test_home_ancestor_symlink_is_not_followed(home, job, monkeypatch):
    alias = home.parent / "home-alias"
    alias.symlink_to(home)
    nested_home = home / "nested"
    nested_home.mkdir()
    (nested_home / "Documents").mkdir()
    monkeypatch.setenv("HOME", str(alias / "nested"))
    assert publisher.publish(job, ["a.pdf"], NAME).status == "unavailable"
    assert list((nested_home / "Documents").iterdir()) == []


def test_post_commit_sync_failure_keeps_truthful_published_result(root, job, monkeypatch, log):
    original = os.fsync
    root_ino = root.stat().st_ino

    def fail_root(fd):
        if os.fstat(fd).st_ino == root_ino:
            raise OSError("private sync error")
        return original(fd)

    monkeypatch.setattr(publisher.os, "fsync", fail_root)
    result = publisher.publish(job, ["a.pdf"], NAME)
    assert result.status == "published" and result.bytes == 13
    assert Path(result.folder) == root / NAME
    assert (root / NAME / "a.pdf").read_bytes() == b"first\x00content"
    assert [p.name for p in root.iterdir()] == [NAME]
    assert "sync" in result.detail and "private" not in result.detail
    log.assert_called_once_with("–", "pdf", "publish", payload={
        "status": "published", "files": 1, "bytes": 13})


def test_strict_umask_still_publishes_0644(root, job):
    previous = os.umask(0o077)
    try:
        result = publisher.publish(job, ["a.pdf"], NAME)
    finally:
        os.umask(previous)
    assert result.status == "published"
    assert stat.S_IMODE((root / NAME / "a.pdf").stat().st_mode) == 0o644


def test_unsupported_exclusive_rename_has_no_fallback(root, job, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.ENOSYS, "unavailable")

    monkeypatch.setattr(publisher, "_rename_exclusive", unsupported)
    monkeypatch.setattr(publisher.os, "rename", Mock(side_effect=AssertionError("unsafe rename")))
    assert_empty(publisher.publish(job, ["a.pdf"], NAME), root, "unavailable")
