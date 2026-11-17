"""A path in the message never reaches the 8B on its own (s85).

Measured on 14/10/2026, in an isolated end-to-end run against the real 8B
(Palace copy, both entry points, three runs each): every message carrying a
path went to plain chat, and 18 of 18 replies pretended to a read that never
happened. A typed `read_file_at(path=".../text.pdf")` got five different
invented texts (one built "the Norman Conquest began on 13 October" out of
the folder name "13-10"); a file that does not exist got its "contents"
described six times out of six; a folder got "I'll read them and answer"
and nothing else, twice with invented file names.

A path is a fact the filesystem can settle in microseconds, so it is settled
here, before any model sees the sentence: a file goes to the same staged read
the tool uses, a folder is listed, a path that is not there is said to be not
there. This module only reads the TEXT and the filesystem; the replies live
in the orchestrator, next to the other read routes.

Which slashes count as a path is the false-positive risk, so the rule is
narrow: the path must start the message or follow whitespace, a quote, a
bracket, a backtick or "=" (path="/Users/…", a pasted tool call); it must
begin with "~/" or have two segments ("/Users/x"); and it must point inside
the user's home, or at /Users or /Volumes. "and/or", "km/h", "13/10", "24/7"
and URLs never qualify, and neither do system paths like /etc or /tmp — a
question about a shell command is not a request to open a file.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from src.tools.file_finder import normalise_user_path

FILE = "file"
FOLDER = "folder"
MISSING = "missing"

_PATH_RE = re.compile(
    r"(?:^|(?<=[\s\"'“”‘’(\[`=]))"
    r"((?:~/|/)(?:\\.|[^\s\"'“”‘’<>`])+)"
)
# What a sentence wraps around a path: "…/13-10." or "(…/a.pdf)".
_TRAILING = ".,;:!?\"'“”‘’)]`"
# How many words an unescaped path with spaces may span ("Mobile Documents/
# college homework/x.pdf"). Generous on purpose: each try is one stat call.
_MAX_SPAN_WORDS = 8
_OTHER_ROOTS = ("/Users/", "/Volumes/")


@dataclass(frozen=True)
class PathMention:
    typed: str      # as written in the message
    path: str       # absolute: the real file/folder, or the expanded text
    kind: str       # FILE | FOLDER | MISSING


def _home() -> str:
    return os.path.realpath(os.path.expanduser("~"))


def _counts(raw: str) -> bool:
    """Shape and location of a candidate, without touching what it names."""
    if raw.startswith("/") and raw.count("/") < 2:
        return False                       # "/or", "/h": one segment
    expanded = os.path.expanduser(raw.replace("\\", ""))
    if expanded.startswith(_OTHER_ROOTS):
        return True
    home = _home()
    real = os.path.realpath(expanded)
    return real == home or real.startswith(home + os.sep)


def _candidates(text: str) -> List[re.Match]:
    out = []
    for match in _PATH_RE.finditer(text or ""):
        raw = match.group(1).rstrip(_TRAILING)
        if raw and _counts(raw):
            out.append(match)
    return out


def mentions_path(text: str) -> bool:
    """Does this message carry a path the route will answer? Text only — no
    filesystem — so it is cheap enough for the streaming pure-chat probe,
    which must say no whenever `find_paths` would say yes."""
    return bool(_candidates(text))


def _resolve_folder(raw: str) -> Optional[str]:
    text = (raw or "").strip().strip('"').strip("'")
    unescaped = re.sub(r"\\(.)", r"\1", text)
    spaced = re.sub(r"\s{2,}", " ", text.replace("\\", " "))
    for candidate in (text, unescaped, spaced):
        expanded = os.path.expanduser(candidate)
        if candidate and os.path.isdir(expanded):
            return expanded.rstrip(os.sep) or os.sep
    return None


def _resolve(raw: str) -> Optional[tuple]:
    folder = _resolve_folder(raw)
    if folder:
        return (folder, FOLDER)
    path = normalise_user_path(raw)
    if path:
        return (path, FILE)
    return None


def _longest_existing(text: str, start: int, raw: str) -> tuple:
    """A path pasted with plain spaces ("…/Mobile Documents/x.pdf") is cut at
    the first space by the regex. Try the following words too, longest
    first, and keep the longest that exists."""
    words = re.split(r"(\s+)", text[start:])
    pieces = words[0::2]
    for n in range(min(len(pieces), _MAX_SPAN_WORDS), 1, -1):
        candidate = "".join(words[: 2 * n - 1]).rstrip(_TRAILING)
        hit = _resolve(candidate)
        if hit:
            return (candidate, *hit)
    return (raw, os.path.expanduser(re.sub(r"\\(.)", r"\1", raw)), MISSING)


def find_paths(text: str) -> List[PathMention]:
    """Every qualifying path in the message, settled against the filesystem."""
    out: List[PathMention] = []
    seen = set()
    for match in _candidates(text):
        full = match.group(1)
        raw = full.rstrip(_TRAILING)
        # "Screenshot (1)" ends in a bracket the sentence did not add.
        hit = _resolve(raw) or (_resolve(full) if full != raw else None)
        # The longer reading first, even when the cut one exists: the folder
        # "Meu" must not win over "Meu arquivo.pdf" beside it (s88, Codex).
        longer = _longest_existing(text, match.start(1), raw)
        if longer[2] != MISSING or not hit:
            typed, path, kind = longer
        else:
            typed, (path, kind) = raw, hit
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(PathMention(typed=typed, path=path, kind=kind))
    return out
