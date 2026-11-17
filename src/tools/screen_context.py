"""
Active-window context extraction (inovação #3 — "o Cortex que sabe o que você vê").

Deterministic, AppleScript-based readers of WHAT'S IN FRONT of the user on the
Mac, so voice commands ("responde esse email", "resume isso", "o que é isso?")
can act on the active app/window. Text extraction is the PRIMARY path — the 8B
is strong on text and weak on vision (`vision_processor` is a Studio stub); vision
is only a fallback for the image case (Bloco C).

SOBERANIA: reading the active window is SENSITIVE — it can capture a password or
a private document. It is LOCAL (regra 1 ok), but the extracted content is
TRANSIENT: callers must NEVER persist it to memory/Palace, and this module never
logs the content itself. These functions only READ; they return structured data.

Reuses `macos_controller._osascript`, which already handles the TCC permission
surface (Accessibility / Automation) and never raises. AppleScript is not
unit-testable, so tests mock `_osascript`.
"""
from __future__ import annotations

import base64
import os
import re
import tempfile
from typing import Optional, Tuple

from src.tools.macos_controller import _SCREENCAPTURE, _osascript, _run

# ASCII Unit Separator — a control char that won't appear in a sender, subject
# or email body, so it's a safe field delimiter for the AppleScript output.
_SEP = chr(31)

# Mail: the message the user has selected. Returns "" when nothing is selected
# (so the handler can distinguish "no selection" from a permission error, which
# surfaces as a non-zero osascript exit handled by _osascript). The three fields
# are joined with the Unit Separator so a body full of commas/newlines parses
# cleanly on the Python side.
_MAIL_SELECTION_SCRIPT = (
    'tell application "Mail"\n'
    "    set sel to selection\n"
    "    if sel is {} then return \"\"\n"
    "    set m to item 1 of sel\n"
    "    set d to (ASCII character 31)\n"
    "    return (sender of m) & d & (subject of m) & d & (content of m)\n"
    "end tell"
)


def frontmost_app(lang: str = "pt") -> Optional[str]:
    """Name of the app currently in front, or None on error/empty.

    Drives the "which extractor do I use?" decision in the handlers."""
    ok, out = _osascript(
        'tell application "System Events" to get name of '
        "first process whose frontmost is true",
        lang=lang,
    )
    if not ok:
        return None
    name = (out or "").strip()
    return name or None


# Browsers we can script for the active page's text. Safari uses
# `do JavaScript ... in front document`; the Chromium family uses
# `execute ... javascript`. Both need the user to allow JavaScript from
# Apple Events (Safari: Develop menu; Chrome: View → Developer) — when it's
# off, _osascript surfaces the error string honestly.
_SAFARI_NAMES = {"Safari", "Safari Technology Preview"}
_CHROMIUM_NAMES = {
    "Google Chrome", "Google Chrome Canary", "Brave Browser",
    "Microsoft Edge", "Chromium", "Vivaldi", "Opera", "Arc",
}


def is_browser(app: Optional[str]) -> bool:
    return bool(app) and (app in _SAFARI_NAMES or app in _CHROMIUM_NAMES)


def _safari_script(app: str) -> str:
    return (
        f'tell application "{app}"\n'
        "    set d to (ASCII character 31)\n"
        "    set theDoc to front document\n"
        "    set theURL to URL of theDoc\n"
        "    set theName to name of theDoc\n"
        '    set theText to (do JavaScript "document.body.innerText" in theDoc)\n'
        "    return theURL & d & theName & d & theText\n"
        "end tell"
    )


def _chromium_script(app: str) -> str:
    return (
        f'tell application "{app}"\n'
        "    set d to (ASCII character 31)\n"
        "    set theTab to active tab of front window\n"
        "    set theURL to URL of theTab\n"
        "    set theTitle to title of theTab\n"
        '    set theText to (execute theTab javascript "document.body.innerText")\n'
        "    return theURL & d & theTitle & d & theText\n"
        "end tell"
    )


def browser_page(
    app: str, lang: str = "pt",
) -> Tuple[Optional[dict], Optional[str]]:
    """Active page of a scriptable browser: `({url, title, text}, None)`,
    `(None, error_str)`, or `(None, None)` if it's not a known browser."""
    if app in _SAFARI_NAMES:
        script = _safari_script(app)
    elif app in _CHROMIUM_NAMES:
        script = _chromium_script(app)
    else:
        return None, None
    ok, out = _osascript(script, lang=lang)
    if not ok:
        return None, out
    raw = out or ""
    parts = raw.split(_SEP, 2)
    if len(parts) < 3:
        return None, None
    url, title, text = parts
    return (
        {"url": url.strip(), "title": title.strip(),
         "text": text.strip(), "source": app},
        None,
    )


# Selected text in ANY app via the clipboard: clear the pasteboard, send
# Cmd+C, read it back, then RESTORE the original clipboard. The clear-first
# step lets us tell "nothing was selected" (clipboard stays empty) from a real
# selection. Side effect is bounded to one round-trip and restored; a non-text
# clipboard (image) is lost on restore — documented limitation, acceptable for
# v1. Needs Accessibility (the keystroke) which _osascript already detects.
#
# The mutating steps run inside a `try` so a mid-script failure — most
# importantly the keystroke being blocked when Accessibility isn't granted yet
# (the FIRST-RUN state) — still RESTORES the user's clipboard before re-raising,
# instead of leaving it wiped to "". The re-raise keeps _osascript reporting the
# real permission error to the caller.
_SELECTED_TEXT_SCRIPT = (
    'tell application "System Events"\n'
    # Coercing a NON-TEXT clipboard "as text" raises ("Can't make «class
    # PDF»… into text") — exactly what happens in Preview, where Cmd+C with
    # no selection copies the page as PDF data. Both reads are guarded so a
    # non-text clipboard degrades to "" (→ caller reports "nothing
    # selected") instead of surfacing a scary coercion error. The keystroke
    # itself stays OUTSIDE the inner guards, so an Accessibility denial
    # (-1743) still propagates as a real permission error.
    '    set savedClip to ""\n'
    "    try\n"
    "        set savedClip to (the clipboard) as text\n"
    "    end try\n"
    '    set sel to ""\n'
    "    try\n"
    '        set the clipboard to ""\n'
    '        keystroke "c" using {command down}\n'
    "        delay 0.12\n"
    "        try\n"
    "            set sel to (the clipboard) as text\n"
    "        end try\n"
    "        set the clipboard to savedClip\n"
    "    on error errMsg number errNum\n"
    "        set the clipboard to savedClip\n"
    "        error errMsg number errNum\n"
    "    end try\n"
    "    return sel\n"
    "end tell"
)


def selected_text(lang: str = "pt") -> Tuple[Optional[str], Optional[str]]:
    """Text currently selected in the front app, via a save/restore
    clipboard round-trip. `(text, None)` | `(None, error)` | `(None, None)`
    when nothing is selected."""
    ok, out = _osascript(_SELECTED_TEXT_SCRIPT, lang=lang)
    if not ok:
        return None, out
    text = (out or "").strip()
    return (text, None) if text else (None, None)


# Document-based apps we can read WITHOUT a selection. Two flavours:
#   - TEXT apps hand us the document text straight from AppleScript (works
#     even for an unsaved doc) — TextEdit's `text of front document`.
#   - PATH apps only give us the file path (Preview can't dump a PDF's text
#     over AppleScript); we read the file off disk and parse it locally.
# Other editors (VS Code, etc.) aren't reliably scriptable for their text —
# the selection round-trip already covers them.
_TEXT_DOC_APPS = {"TextEdit"}
_PATH_DOC_APPS = {"Preview"}

# Reading a doc is the FIRST time the server sends Apple Events to Preview/
# TextEdit, so macOS shows the one-time Automation consent dialog. The default
# 5s osascript timeout kills the script before a human can click "Allow" (the
# TextEdit "timeout após 5s" seen in the s76 live smoke). Give the prompt room.
# This only ever waits when the grant is still UNDECIDED — a granted app
# returns in ~100ms and a denied one fails instantly (-1743), so the longer
# ceiling costs nothing in steady state.
_DOC_READ_TIMEOUT_S = 20


def is_doc_app(app: Optional[str]) -> bool:
    return bool(app) and (app in _TEXT_DOC_APPS or app in _PATH_DOC_APPS)


def _textedit_script(app: str) -> str:
    return (
        f'tell application "{app}"\n'
        "    if (count of documents) is 0 then return \"\"\n"
        "    set d to (ASCII character 31)\n"
        "    set theDoc to front document\n"
        "    return (name of theDoc) & d & (text of theDoc)\n"
        "end tell"
    )


def _doc_path_script(app: str) -> str:
    return (
        f'tell application "{app}"\n'
        "    if (count of documents) is 0 then return \"\"\n"
        "    return path of front document\n"
        "end tell"
    )


def _normalise_doc_path(p: str) -> str:
    """AppleScript may hand back a POSIX path ("/Users/…", the modern
    Preview default) or, on older paths, an HFS one ("Macintosh HD:Users:…").
    Normalise to POSIX. The HFS branch drops the leading volume name and
    swaps ':' for '/', which is correct for the boot volume (the common
    case); an exotic external-volume path that doesn't resolve just falls
    through to the selection path, so a wrong guess is never fatal."""
    p = (p or "").strip()
    if not p or p.startswith("/"):
        return p
    if ":" in p:  # HFS — "Volume:Users:pedro:file.pdf" → "/Users/pedro/file.pdf"
        return "/" + "/".join(p.split(":")[1:])
    return p


def front_document(
    app: Optional[str], lang: str = "pt",
) -> Tuple[Optional[dict], Optional[str]]:
    """Text of the document open in the front app (Preview/TextEdit), so
    "resume isso" / "o que é isso?" work on an open doc WITHOUT the user
    having to select anything. `({source, title, text}, None)`,
    `(None, error)`, or `(None, None)` when there's no readable doc / the app
    isn't a known doc app.

    SOBERANIA: same trust model as `browser_page`/`selected_text` — it reads
    what's literally in front of the user, once and transiently. The PATH
    flavour reads the file off disk via `LocalFileReader.read_path` (NO vault
    guard — the file is wherever the user saved it), which is fine here
    because the path comes from the front window, not from an LLM."""
    if app in _TEXT_DOC_APPS:
        ok, out = _osascript(_textedit_script(app), timeout=_DOC_READ_TIMEOUT_S, lang=lang)
        if not ok:
            return None, out
        raw = out or ""
        if not raw.strip():
            return None, None
        name, _sep, text = raw.partition(_SEP)
        text = text.strip()
        if not text:
            return None, None
        return ({"url": "", "title": name.strip() or app,
                 "text": text, "source": app}, None)
    if app in _PATH_DOC_APPS:
        ok, out = _osascript(_doc_path_script(app), timeout=_DOC_READ_TIMEOUT_S, lang=lang)
        if not ok:
            return None, out
        path = _normalise_doc_path(out or "")
        if not path or not os.path.exists(path):
            return None, None
        # Lazy import: pulls in pypdf/docx only when a doc is actually read.
        from src.tools.local_reader import LocalFileReader
        text = LocalFileReader().read_path(path)
        if not text or text.startswith("[SYSTEM ERROR]"):
            # Unsupported type or parse failure — let the caller fall through
            # to the selection path rather than surfacing a parser error.
            return None, None
        return ({"url": "", "title": os.path.basename(path),
                 "text": text.strip(), "source": app}, None)
    return None, None


def capture_screen_b64(lang: str = "pt") -> Tuple[Optional[str], Optional[str]]:
    """Capture the full screen to a TEMP file, return its base64, and DELETE
    the file immediately.

    SOBERANIA: a screen capture is sensitive transient data — unlike
    `macos_controller.screenshot()` (which leaves a PNG in data/screenshots/
    on purpose, for the user's "tira um print"), this NEVER leaves a capture
    of the screen on disk. The bytes exist only long enough to base64 them
    for the local vision model. `(b64, None)` | `(None, error)` |
    `(None, None)` if the capture is empty."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cortex_vision_")
    os.close(fd)
    try:
        ok, err = _run([_SCREENCAPTURE, "-x", path], lang=lang)
        if not ok:
            return None, err
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as e:  # pragma: no cover - capture wrote but unreadable
            return None, str(e)
        if not raw:
            return None, None
        return base64.b64encode(raw).decode("ascii"), None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def active_content(lang: str = "pt") -> Tuple[Optional[dict], Optional[str]]:
    """Best-effort "what's in front to summarise" for "resume isso".

    Order: a scriptable browser's page (highest value, clean) → selected
    text in any app (universal fallback). Returns `({source, title, text},
    None)`, `(None, error)`, or `(None, None)` when there's nothing usable.

    A browser ERROR (e.g. "Allow JavaScript from Apple Events" is off) is
    NOT fatal: it falls through to the selection path, so a user in Safari
    with text highlighted still gets a summary even with JS-from-AppleEvents
    disabled. The browser error is only surfaced if the selection path also
    yields nothing — `browser_err` is the more actionable message to show."""
    app = frontmost_app(lang=lang)
    browser_err = None
    if is_browser(app):
        data, browser_err = browser_page(app, lang=lang)
        if data and data.get("text", "").strip():
            return data, None
        # Browser but no usable text (empty page OR JS blocked) — fall
        # through to the doc/selection paths before giving up.
    # Document open in a scriptable doc app (Preview/TextEdit): read the whole
    # doc with no selection needed. Highest-value non-browser source. A doc
    # error (permission) is non-fatal — fall through to selection like the
    # browser path does.
    doc_err = None
    if is_doc_app(app):
        doc, doc_err = front_document(app, lang=lang)
        if doc and doc.get("text", "").strip():
            return doc, None
    text, err = selected_text(lang=lang)
    if err:
        return None, err
    if text:
        return ({"source": app or "seleção", "title": app or "",
                 "text": text}, None)
    # Nothing from any path — prefer the browser error (it tells the user to
    # enable JS), then a doc error, else "no content".
    return None, browser_err or doc_err


def _parse_sender(raw: str) -> Tuple[str, str]:
    """Split Mail's `sender` ("Pedro Silva <pedro@x.com>") into
    (display_name, email). Falls back gracefully when the format differs:
    a bare address becomes both name and email; a bare name yields an empty
    email so the caller can refuse before staging an unsendable draft."""
    raw = (raw or "").strip()
    m = re.search(r"<([^>]+)>", raw)
    if m:
        email = m.group(1).strip()
        name = raw[: m.start()].strip().strip('"').strip()
        return (name or email), email
    if "@" in raw:
        return raw, raw
    return raw, ""


def mail_selection(
    lang: str = "pt",
) -> Tuple[Optional[dict], Optional[str]]:
    """Read the message selected in Mail.

    Returns `(data, error)`:
      - `(dict, None)` — a parsed selection: keys `sender_name`,
        `sender_email`, `subject`, `content`.
      - `(None, None)` — Mail is reachable but nothing is selected.
      - `(None, error_str)` — a permission/Mail-not-running error (the
        friendly string already localised by `_osascript`).
    """
    ok, out = _osascript(_MAIL_SELECTION_SCRIPT, lang=lang)
    if not ok:
        return None, out
    raw = out or ""
    if not raw.strip():
        return None, None
    parts = raw.split(_SEP, 2)
    if len(parts) < 3:
        # Unexpected shape — treat as no usable selection rather than guess.
        return None, None
    sender_raw, subject, content = parts
    name, email = _parse_sender(sender_raw)
    return (
        {
            "sender_name": name,
            "sender_email": email,
            "subject": subject.strip(),
            "content": content.strip(),
        },
        None,
    )
