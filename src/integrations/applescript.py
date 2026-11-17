"""Shared AppleScript helpers (M4-A3 escape; s73 run — the old decision #5).

`escape`: the correct escape existed twice in the repo (reminders_client,
notes_client) while macos_controller's three call sites only escaped the
double quote. A name ending in a backslash then produced `\\"` — literal
backslash + END of the string literal — and everything after it became
AppleScript code, including `do shell script`. close_window/switch_to_app
are S2 (no confirmation), and args may one day come from the LLM
tool-call path, which is exactly the scenario the S2/S3 risk table
governs. One canonical helper; the order matters: backslash FIRST.

`run`: the identical `_run` lived twice (reminders_client, notes_client) —
same subprocess call, same 30s ceiling, same three failure shapes, differing
only in the app name and the exception type. Each client keeps a thin `_run`
wrapper so its call sites and domain exception stay unchanged.
macos_controller's `_run` is NOT this function and stays where it is: its
contract is (ok, message) tuples that never raise, with TCC-permission
detection and i18n'd error strings — unifying that would force try/except
boilerplate back into every controller call site.
"""
from __future__ import annotations

import subprocess
from typing import Type


def escape(s: str) -> str:
    """Escape backslashes and double quotes for AppleScript string literals."""
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def run(
    script: str,
    *,
    app_label: str,
    unavailable_exc: Type[Exception],
    timeout: int = 30,
) -> str:
    """Execute an AppleScript via osascript and return stdout.

    Raises `unavailable_exc` (the caller's domain exception, e.g.
    NotesUnavailable) on missing osascript, timeout, or non-zero exit.
    30s default: the Foundation framework import (NSDate) adds noticeable
    warm-up on the first call after a uvicorn restart, occasionally pushing
    past 10s — generous ceiling without hanging the user forever.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise unavailable_exc(f"osascript not found: {e}") from e
    except subprocess.TimeoutExpired as e:
        error = unavailable_exc(f"{app_label} didn't respond in {timeout}s")
        error.uncertain = True  # The script may have run before the reply was lost.
        raise error from e
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise unavailable_exc(
            f"{app_label} rejected the request: {stderr or 'unknown error'}"
        )
    return result.stdout or ""
