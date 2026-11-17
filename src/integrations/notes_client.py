"""
macOS Notes.app integration via AppleScript.

Calls `osascript` as a subprocess to create / list / search notes in the
user's default Notes account. No OAuth — Notes is purely local. Notes sync
to iPhone/iPad automatically through iCloud.

Mirrors `reminders_client` (same `_run` / `_escape` helpers, same first-call
macOS Automation consent flow). Creating a note is treated as S2 (recoverable,
not external-facing) — the orchestrator runs it directly, no sim/não gate.

On the first call, macOS prompts the user to grant Python (the venv binary)
permission to control Notes (System Settings → Privacy & Security →
Automation). Subsequent calls run silently.
"""

import html
import logging
import re
from typing import Dict, List, Optional

from src.integrations.applescript import escape as _escape
from src.integrations.applescript import run as _applescript_run

log = logging.getLogger(__name__)

_NOTE_ID_RE = re.compile(r"x-coredata://[0-9A-Fa-f-]+/ICNote/p\d+")


class NotesUnavailable(Exception):
    """Raised when AppleScript fails or Notes.app is unreachable."""


# Folders to skip in list/search so a deleted note never resurfaces. The
# name is localized by macOS — cover the user's likely locales (EN/PT).
_TRASH_FOLDERS = ("Recently Deleted", "Apagados Recentemente")


def _not_trash_clause(var: str = "f") -> str:
    """AppleScript boolean: true when folder `var` is not a trash folder."""
    return " and ".join(
        f'(name of {var}) is not "{_escape(t)}"' for t in _TRASH_FOLDERS
    )


def create_note(title: str, body: str = "", folder: Optional[str] = None) -> Dict:
    """
    Create a note in Notes.app. The note's title is its first line (Notes
    derives the title from the body), so we put `title` first in bold, then
    `body`. Returns a dict for the confirmation message.

    With no `folder`, lands in the default account's default folder — the
    same place a note created by hand with ⌘N goes. With `folder` (v1.1),
    targets that folder, CREATING it when missing — a folder is as
    recoverable as the note itself (S2), and "put it in Trabalho" failing
    because Trabalho doesn't exist yet would be the annoying outcome.
    """
    if not (title or "").strip():
        raise NotesUnavailable("Note needs a non-empty title.")

    # Notes bodies are HTML. Escape the user text, then build a titled body:
    # bold first line (becomes the title) + the rest. Newlines in the body
    # become <br> so multi-line notes keep their shape.
    title_html = html.escape(title.strip())
    body_html = html.escape(body or "").replace("\n", "<br>")
    note_html = f"<div><b>{title_html}</b></div>"
    if body_html:
        note_html += f"<div>{body_html}</div>"

    folder_name = (folder or "").strip()
    if folder_name:
        f_esc = _escape(folder_name)
        script = (
            'tell application "Notes"\n'
            f'    if not (exists folder "{f_esc}") then make new folder '
            f'with properties {{name:"{f_esc}"}}\n'
            f'    return id of (make new note at folder "{f_esc}" '
            f'with properties {{body:"{_escape(note_html)}"}})\n'
            'end tell'
        )
    else:
        script = (
            'tell application "Notes"\n'
            f'    return id of (make new note with properties {{body:"{_escape(note_html)}"}})\n'
            'end tell'
        )
    out = _run(script)
    found = _NOTE_ID_RE.search(out or "")
    return {"title": title.strip(), "has_body": bool(body_html),
            "folder": folder_name or None, "id": found.group(0) if found else None}


def append_to_note(title: str, text: str) -> Dict:
    """
    Append `text` to the end of an existing note (v1.1, find-then-modify).

    Finds by exact name first, falling back to `name contains` — and takes
    the LAST match, mirroring delete_note's collision policy. The body is
    HTML, so the text is escaped and appended as its own <div> block
    (newlines become <br>). Raises NotesUnavailable when no note matches —
    the handler turns that into an honest "não achei" instead of silently
    creating something.

    Returns {"title": <the matched note's real name>, ...} so the
    confirmation can echo what was actually modified.
    """
    if not (title or "").strip():
        raise NotesUnavailable("append_to_note needs a non-empty title.")
    if not (text or "").strip():
        raise NotesUnavailable("append_to_note needs non-empty text.")

    title_esc = _escape(title.strip())
    text_html = html.escape(text.strip()).replace("\n", "<br>")
    addition = f"<div>{text_html}</div>"

    script = (
        'tell application "Notes"\n'
        f'    set matches to (notes whose name is "{title_esc}")\n'
        '    if (count of matches) is 0 then '
        f'set matches to (notes whose name contains "{title_esc}")\n'
        '    if (count of matches) is 0 then return ""\n'
        '    set theNote to last item of matches\n'
        f'    set body of theNote to (body of theNote) & "{_escape(addition)}"\n'
        '    return name of theNote\n'
        'end tell'
    )
    matched = _run(script).strip()
    if not matched:
        raise NotesUnavailable(f'No note found matching "{title.strip()}".')
    return {"title": matched, "appended_chars": len(text.strip())}


def delete_note(title: str, note_id: Optional[str] = None) -> int:
    """Delete by id, or by exact title for legacy ledger lines. Only a single
    match outside trash is deleted; otherwise return the count unchanged.
    Deleting from Recently Deleted would remove the note permanently."""
    if note_id is not None:
        if not isinstance(note_id, str) or not _NOTE_ID_RE.fullmatch(note_id):
            raise NotesUnavailable(f"not a note id: {note_id!r}")
        which = f'id is "{note_id}"'
    else:
        if not (title or "").strip():
            raise NotesUnavailable("delete_note needs a non-empty title.")
        which = f'name is "{_escape(title.strip())}"'
    # Scope both counting and deletion to live folders: trash notes still
    # match application-wide queries by id and by name after deletion.
    script = (
        'tell application "Notes"\n'
        '    set n to 0\n'
        '    repeat with f in folders\n'
        f'        if {_not_trash_clause()} then\n'
        f'            set folderCount to count of (notes of f whose {which})\n'
        '            set n to n + folderCount\n'
        '            if folderCount > 0 then set targetFolder to contents of f\n'
        '        end if\n'
        '    end repeat\n'
        '    if n is not 1 then return (n as text)\n'
        f'    delete (notes of targetFolder whose {which})\n'
        '    return "1"\n'
        'end tell'
    )
    out = _run(script)
    try:
        return int((out or "0").strip())
    except ValueError:
        return 0


def list_notes(limit: int = 10) -> List[Dict]:
    """List up to `limit` note titles, walking folders and SKIPPING the
    Recently Deleted folder (so a deleted note never resurfaces). The cap is
    enforced in AppleScript (stops early — O(limit), not O(all notes)). Order
    is whatever Notes.app yields per folder — NOT sorted by recency."""
    n = max(1, int(limit))
    script = (
        'tell application "Notes"\n'
        '    set output to ""\n'
        '    set cnt to 0\n'
        '    repeat with f in folders\n'
        f'        if {_not_trash_clause()} then\n'
        '            repeat with theNote in (notes of f)\n'
        f'                if cnt < {n} then\n'
        '                    set output to output & (name of theNote) & linefeed\n'
        '                    set cnt to cnt + 1\n'
        '                end if\n'
        '            end repeat\n'
        '        end if\n'
        f'        if cnt >= {n} then exit repeat\n'
        '    end repeat\n'
        '    return output\n'
        'end tell'
    )
    raw = _run(script).strip()
    if not raw:
        return []
    titles = [line.strip() for line in raw.splitlines() if line.strip()]
    return [{"title": t} for t in titles[:n]]


def search_notes(query: str, limit: int = 10) -> List[Dict]:
    """Find notes whose title OR PLAINTEXT contains `query` (case-insensitive),
    skipping the Recently Deleted folder. Searches `plaintext` (not `body`,
    which is raw HTML — 'div'/'br' would match everything and accented chars
    stored as entities wouldn't match). Linear over notes, fine at personal
    scale; capped at `limit` in AppleScript."""
    q = (query or "").strip()
    if not q:
        return []
    q_esc = _escape(q)
    n = max(1, int(limit))
    script = (
        'tell application "Notes"\n'
        '    set output to ""\n'
        '    set cnt to 0\n'
        '    repeat with f in folders\n'
        f'        if {_not_trash_clause()} then\n'
        '            repeat with theNote in '
        f'(notes of f whose name contains "{q_esc}" or plaintext contains "{q_esc}")\n'
        f'                if cnt < {n} then\n'
        '                    set output to output & (name of theNote) & linefeed\n'
        '                    set cnt to cnt + 1\n'
        '                end if\n'
        '            end repeat\n'
        '        end if\n'
        f'        if cnt >= {n} then exit repeat\n'
        '    end repeat\n'
        '    return output\n'
        'end tell'
    )
    raw = _run(script).strip()
    if not raw:
        return []
    titles = [line.strip() for line in raw.splitlines() if line.strip()]
    return [{"title": t} for t in titles[:n]]


# --- helpers --------------------------------------------------------

def _run(applescript: str) -> str:
    """Execute an AppleScript and return stdout. Raises NotesUnavailable on
    failure. Thin wrapper over the shared runner (s73) so call sites and the
    domain exception stay unchanged."""
    return _applescript_run(
        applescript, app_label="Notes.app", unavailable_exc=NotesUnavailable,
    )


# `_escape` and the subprocess plumbing live in src/integrations/applescript.py
# (M4-A3 / s73): one canonical implementation shared with reminders_client.
