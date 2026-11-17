"""Dormant, exclusive publication of a completed PdfJob into the fixed PDF root.

The job has one synchronous owner, as in pdf_runner: no parser or other caller
may mutate/close it during publication. Only renameatx_np(RENAME_EXCL) commits;
there is deliberately no replacing rename fallback. No content is interpreted.
"""
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
from datetime import date
import errno
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Literal
import unicodedata

from src.core import pdf_runner
from src.core.audit import audit

ROOT_PARTS = ("Documents", "Cortex", "PDF")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_STAGING = ".cortex-staging-"
_CHUNK = 64 * 1024
_DETAILS = {
    "published": "PDF results published.",
    "occupied": "Publication folder already occupied.",
    "invalid": "Invalid PDF publication request.",
    "too_big": "PDF publication limit exceeded.",
    "unavailable": "PDF publication unavailable.",
    "error": "PDF publication failed.",
}


@dataclass(frozen=True)
class Published:
    status: Literal["published", "occupied", "invalid", "too_big", "unavailable", "error"]
    folder: str | None
    files: tuple[str, ...]
    bytes: int
    detail: str


def _component(value, limit=None, fallback="PDF"):
    value = unicodedata.normalize("NFC", value)
    value = "".join(" " if c in "/:" or unicodedata.category(c).startswith("C")
                    else c for c in value)
    value = " ".join(value.split()).lstrip(". ")
    return value[:limit].rstrip() or fallback


def folder_name(source_name: str, action: str, when: date) -> str:
    """Make one NFC component, retaining accents and limiting each human part."""
    stem = source_name[:-4] if source_name.lower().endswith(".pdf") else source_name
    return (f"{_component(stem, 80)} - {_component(action, 40, 'resultado')} - "
            f"{when.day:02d}-{when.month:02d}-{when.year:04d}")


def _valid_name(name):
    return isinstance(name, str) and bool(name) and name == _component(name)


def _open_dir(path, stack):
    """Walk an absolute directory path without following even ancestor links."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute directory required")
    fd = os.open(path.anchor, _DIR_FLAGS)
    stack.callback(os.close, fd)
    for part in path.parts[1:]:
        fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
        stack.callback(os.close, fd)
    return fd


def _open_root(stack, *, create):
    home = Path.home()  # Resolve HOME at call time, never at import time.
    fd = _open_dir(home, stack)
    for part in ROOT_PARTS:
        try:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
        except FileNotFoundError:
            if not create or part == "Documents":
                raise
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass  # Re-open the winner with O_NOFOLLOW, including races.
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
        stack.callback(os.close, child)
        fd = child
    return fd, home.joinpath(*ROOT_PARTS)


def next_free(name: str) -> str | None:
    """Read-only preview advice; occupied links count and errors give no advice."""
    if not _valid_name(name):
        return None
    try:
        with ExitStack() as stack:
            try:
                fd, _ = _open_root(stack, create=False)
            except FileNotFoundError:
                return name
            for number in range(1, 100):
                candidate = name if number == 1 else f"{name} ({number})"
                try:
                    os.lstat(candidate, dir_fd=fd)
                except FileNotFoundError:
                    return candidate
    except Exception:
        pass
    return None


def _discard(name, parent_fd, expected=None):
    """Local variant of pdf_runner._discard: fd traversal, no path-based chmod.

    An identity check avoids deleting a substituted top-level entry. Unreadable
    stale directories are left alone rather than chmodding an unpinned path.
    """
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        info = os.fstat(fd)
        if expected is not None and (info.st_dev, info.st_ino) != expected:
            return
        os.fchmod(fd, 0o700)
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _discard(entry.name, fd)
                else:
                    os.unlink(entry.name, dir_fd=fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
            os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(fd)


def _clean_stale(root_fd):
    cutoff = time.time() - 600
    with os.scandir(root_fd) as entries:
        for entry in entries:
            if not entry.name.startswith(_STAGING):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if (stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                        and info.st_mtime < cutoff):
                    _discard(entry.name, root_fd, (info.st_dev, info.st_ino))
            except OSError:
                pass  # A vanished, changed, or inaccessible entry stays untouched.


class _Invalid(Exception):
    pass


class _TooBig(Exception):
    pass


def _sources(job, files, stack):
    if not isinstance(files, list) or not files:
        raise _Invalid
    if len(files) > 100:
        raise _TooBig
    if (any(not isinstance(name, str) or not name or "/" in name or "\0" in name
            or name in (".", "..") for name in files) or len(set(files)) != len(files)):
        raise _Invalid
    try:
        if not job._active:
            raise _Invalid
        out_fd = _open_dir(job.output_dir, stack)
        names, exceeded = pdf_runner._outputs(job.output_dir)
        if exceeded:
            raise _TooBig
        sources, declared = [], 0
        for name in files:
            if name not in names:
                raise _Invalid
            # A substituted FIFO must not block before its fstat rejection.
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=out_fd)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _Invalid
            declared += info.st_size
            if declared > pdf_runner.MAX_OUTPUT_BYTES:
                raise _TooBig
            sources.append((name, fd))
        return sources
    except (OSError, TypeError, ValueError, AttributeError) as error:
        raise _Invalid from error


def _copy(sources, folder_fd):
    total = 0
    for name, source in sources:
        target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o644, dir_fd=folder_fd)
        try:
            os.fchmod(target, 0o644)  # Also honour the contract under a strict umask.
            while True:
                chunk = os.read(source, min(_CHUNK, pdf_runner.MAX_OUTPUT_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > pdf_runner.MAX_OUTPUT_BYTES:
                    raise _TooBig
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(target, remaining)
                    if written <= 0:
                        raise OSError("copy did not advance")
                    remaining = remaining[written:]
            os.fsync(target)
        finally:
            os.close(target)
    os.fsync(folder_fd)
    return total


def _rename_exclusive(root_fd, source, name):
    """Atomic directory move that rejects every occupied destination on macOS."""
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameatx_np
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "exclusive rename unavailable") from error
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(root_fd, os.fsencode(source), root_fd, os.fsencode(name), 0x4) != 0:
        code = ctypes.get_errno()
        raise OSError(code, "exclusive rename failed")


def publish(job, files: list[str], name: str) -> Published:
    """Copy, then commit exactly the previewed name; failures never choose another.

    The successful rename is the commit point. A later fsync/cleanup failure
    cannot roll it back safely; report published with a fixed durability detail.
    Failure counts are zero: they describe published files, not partial copies.
    """
    status, folder, published, size = "error", None, (), 0
    detail = None
    committed = False
    try:
        with ExitStack() as stack:
            if not _valid_name(name):
                raise _Invalid
            try:
                root_fd, root_path = _open_root(stack, create=True)
                _clean_stale(root_fd)
            except (OSError, ValueError):
                status = "unavailable"
            else:
                sources = _sources(job, files, stack)
                staging = _STAGING + secrets.token_hex(16)
                # mkdir collision is an error, never permission to reuse/delete it.
                os.mkdir(staging, 0o700, dir_fd=root_fd)
                identity = None
                try:
                    with ExitStack() as staged:
                        stage_fd = os.open(staging, _DIR_FLAGS, dir_fd=root_fd)
                        staged.callback(os.close, stage_fd)
                        info = os.fstat(stage_fd)
                        identity = (info.st_dev, info.st_ino)
                        os.mkdir(name, 0o755, dir_fd=stage_fd)
                        folder_fd = os.open(name, _DIR_FLAGS, dir_fd=stage_fd)
                        staged.callback(os.close, folder_fd)
                        copied = _copy(sources, folder_fd)
                        try:
                            _rename_exclusive(root_fd, f"{staging}/{name}", name)
                        except OSError as error:
                            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                                status = "occupied"
                            elif error.errno in (errno.ENOSYS, errno.ENOTSUP):
                                status = "unavailable"
                            else:
                                raise
                        else:
                            committed = True
                            status, folder = "published", str(root_path / name)
                            published, size = tuple(n for n, _ in sources), copied
                            os.fsync(root_fd)
                finally:
                    _discard(staging, root_fd, identity)
    except _Invalid:
        status = "invalid"
    except _TooBig:
        status = "too_big"
    except BaseException:
        # This boundary promises a result even on interruption. Cleanup has run;
        # preserve the commit fact so callers cannot mistake success for failure.
        if committed:
            detail = "PDF results published; final sync or cleanup failed."
        else:
            status = "error"
    result = Published(status, folder, published, size, detail or _DETAILS[status])
    try:
        audit.log("–", "pdf", "publish", payload={
            "status": result.status, "files": len(result.files), "bytes": result.bytes})
    except BaseException:
        pass
    return result
