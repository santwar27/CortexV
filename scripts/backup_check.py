#!/usr/bin/env python3
"""Drift detector for the ~/.cortex backup manifest.

The problem this solves: Cortex keeps non-reproducible user state OUTSIDE
the project tree (~/.cortex). Every time a feature starts writing there,
`scripts/backup.sh` has to learn about it — and twice now it didn't
(s76 cont.37 missed fitness.db/jobs.db; s82 missed chat.db and the
speaker-ID voiceprint). The failure is SILENT: backups keep succeeding,
and you only discover the hole at restore time, when the chat opens empty.

`tests/test_backup_script.py` pins the files we already know about. It
cannot know about a file a FUTURE feature invents — that is this script's
job: it looks at what is actually on disk and asks "is each of these
either captured or deliberately ignored?".

Run it manually (`make backup-check`) after adding anything that writes to
~/.cortex. Read-only: it never touches the backups or the files.

Exit codes: 0 = everything classified, 1 = something unclassified.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BACKUP_SCRIPT = Path(__file__).resolve().parent / "backup.sh"

# Files we deliberately do NOT back up, each with the reason. "Ignored" is
# a real decision — it must be stated, not silent.
IGNORED = {
    "models": "re-downloadable model files, large",
    ".DS_Store": "macOS junk",
}

# Suffixes that are transient by nature.
IGNORED_SUFFIXES = {
    "-wal": "SQLite write-ahead log — folded in by sqlite3 .backup",
    "-shm": "SQLite shared memory — recreated on open",
    ".lock": "an empty flock target (gate_queue's decision lock) — recreated on use",
}

# Substrings marking a file as a throwaway copy.
IGNORED_MARKERS = (".bak-", ".backup-", ".tmp")


def captured(script_text: str) -> set:
    """The ~/.cortex entries backup.sh really copies: the names in the
    capture loops (`for db in …; do`, `for f in …; do`) and the directories
    it copies with `cp -Rp "$DOT_CORTEX/x"` — read from its ~/.cortex block
    only. Matching a name anywhere in the script's TEXT passed "archive"
    (a comment says "archive everything") and "palace.db" (the Palace under
    data/, another path), neither of which is copied (s88, Codex lote 5)."""
    start = script_text.find('DOT_CORTEX="$HOME/.cortex"')
    end = script_text.find("# ---- Build the tar ----", start)
    if start < 0 or end < 0:
        return set()
    block = script_text[start:end].replace("\\\n", " ")
    names = set()
    for m in re.finditer(r"for\s+\w+\s+in\s+([^;]+);\s*do", block):
        names.update(t for t in m.group(1).split() if not t.startswith(('"', "$")))
    names.update(re.findall(r'cp\s+-Rp?\s+"\$DOT_CORTEX/([^"/$]+)"', block))
    return names


def _classify(name: str, script_text: str) -> str | None:
    """Return None when the entry is accounted for, else why it is not."""
    if name in IGNORED:
        return None
    if any(name.endswith(sfx) for sfx in IGNORED_SUFFIXES):
        return None
    if any(marker in name for marker in IGNORED_MARKERS):
        return None
    if name in captured(script_text):
        return None
    return "not captured by backup.sh and not in the ignore list"


def main(argv: list[str]) -> int:
    dot = Path(argv[1] if len(argv) > 1
               else os.path.expanduser("~/.cortex"))
    if not dot.is_dir():
        print(f"backup-check: {dot} does not exist — nothing to check.")
        return 0

    script_text = BACKUP_SCRIPT.read_text()
    entries = sorted(p.name for p in dot.iterdir())
    problems = [(n, why) for n in entries
                if (why := _classify(n, script_text)) is not None]

    for name in entries:
        state = "UNCLASSIFIED" if any(name == n for n, _ in problems) else "ok"
        print(f"  {state:<13} {name}")

    if problems:
        print(f"\n✗ {len(problems)} file(s) in {dot} are neither backed up "
              f"nor explicitly ignored:")
        for name, why in problems:
            print(f"    - {name}: {why}")
        print("\n  Fix: add it to the capture loop in scripts/backup.sh (and to "
              "REQUIRED_SIDECARS in tests/test_backup_script.py), or to IGNORED "
              "here with the reason.")
        return 1

    print(f"\n✓ every entry in {dot} is captured or deliberately ignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
