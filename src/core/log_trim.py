"""Bound append-only API logs without replacing their live inodes."""

import errno
import logging
import os
import stat
from contextlib import contextmanager
from pathlib import Path

LIMIT_BYTES = 20 * 1024 * 1024
TAIL_BYTES = 5 * 1024 * 1024

log = logging.getLogger(__name__)


@contextmanager
def _parent_fd(path: Path):
    """Open each parent without following symlinks, including the logs dir."""
    parent = path.absolute().parent
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(parent.anchor, flags)
    try:
        for part in parent.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def trim_log(
    path: Path, *, limit_bytes: int = LIMIT_BYTES, tail_bytes: int = TAIL_BYTES
) -> int:
    """Above the limit, save a line-aligned .1 tail and truncate in place.

    Return bytes kept (zero also means missing, unchanged or refused).
    Log trims/refusals without contents; other I/O errors reach the caller.
    Neither the source, the .1 destination nor their parents may be symlinks.
    A final unterminated line is retained if its start fits in the tail.
    """
    if limit_bytes < 0 or tail_bytes < 0:
        raise ValueError("log trim sizes must be non-negative")
    path = Path(path)
    flags = os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        with _parent_fd(path) as parent:
            fd = os.open(path.name, os.O_RDWR | flags, dir_fd=parent)
            with os.fdopen(fd, "r+b", buffering=0) as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    log.warning("Log trim refused non-regular file: %s", path)
                    return 0
                if before.st_size <= limit_bytes:
                    return 0

                start = max(0, before.st_size - tail_bytes)
                # Include the preceding byte to preserve an already aligned tail.
                source.seek(max(0, start - 1))
                tail = source.read(min(before.st_size, tail_bytes + 1))
                if start:
                    newline = tail.find(b"\n")
                    tail = tail[newline + 1:] if newline >= 0 else b""

                dest = os.open(
                    path.name + ".1", os.O_WRONLY | os.O_CREAT | flags,
                    0o600, dir_fd=parent,
                )
                with os.fdopen(dest, "wb") as snapshot:
                    target = os.fstat(snapshot.fileno())
                    # A hard link must not let .1 truncate the live log (or others).
                    if not stat.S_ISREG(target.st_mode) or target.st_nlink > 1:
                        log.warning("Log trim refused unsafe tail file: %s.1", path)
                        return 0
                    snapshot.truncate(0)
                    snapshot.write(tail)
                    snapshot.flush()
                    # API writes between the copy and truncate may be lost;
                    # truncate immediately after flushing to keep that window short.
                    os.ftruncate(source.fileno(), 0)

                log.info("Log trimmed: path=%s size_before=%d bytes_kept=%d",
                         path, before.st_size, len(tail))
                return len(tail)
    except FileNotFoundError:
        return 0
    except OSError as exc:
        if exc.errno not in (errno.ELOOP, errno.ENOTDIR):
            raise
        log.warning("Log trim refused symlink or non-directory path: %s", path)
        return 0
