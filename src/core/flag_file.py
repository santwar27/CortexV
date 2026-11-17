"""State flag files in /tmp (s88, Codex lote 12).

The silence, screen-off and recording flags live in /tmp, which every user of
the Mac can write. `open(path, "w")` and the shell's `: > path` follow a
symlink planted in a free name and empty whatever it points to — a .zshrc,
say (reproduced on disposable files with all four writers). A flag is raised
without following a link and without truncating anything, and a flag that
grants something is honoured only when it is Pedro's own file.
"""
from __future__ import annotations

import os
import stat


def raise_flag(path: str) -> None:
    """Create the flag at `path` if it is absent. A symlink in its place is
    refused (OSError, ELOOP) — never followed, never truncated."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.close(fd)


def own_flag(path: str) -> bool:
    """The flag is up AND is a regular file owned by this user — not a link,
    not a file another account left under the same name."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()
