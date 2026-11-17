"""Spotlight-backed file search — "cadê meu currículo?".

The 8B is not the engine here: `mdfind` (Spotlight's CLI) does the actual
search against the OS index — instant, no filesystem walk — and the model only
has to extract what the user is looking for and read the answer back. Same
division of labour the rest of Cortex uses for anything it must be RIGHT about.

Scope and safety (this reaches the user's whole home directory):
  - Searches are confined to $HOME by default (`-onlyin`), so system and other
    users' files never appear.
  - Results are filtered against `_EXCLUDED_FRAGMENTS`: credential stores,
    key material, the Cortex databases themselves, and dependency/VCS noise.
    Filtering happens on the RESULT paths, so a match inside ~/.ssh is dropped
    even if Spotlight indexed it.
  - Returns PATHS and metadata only — never file contents. Reading a hit is a
    separate, deliberate step (`read_path`), which fences what it reads as
    untrusted content.
  - Read-only: classified S1, no confirmation gate (see tool_risk).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 10
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50

# Result paths containing any of these are dropped. Ordered roughly by why:
# secrets first, then Cortex's own state, then noise that would bury real hits.
_EXCLUDED_FRAGMENTS = (
    "/.ssh/", "/.gnupg/", "/.aws/", "/.config/",
    "/.codex/", "/.claude/",           # CLI tools' tokens and sessions
    "/credentials/", "/Keychains/", "/.env",
    "/.cortex/",                       # palace.db, chat.db — Cortex's memory
    "/Library/",                       # caches, keychains, app internals
                                       # …EXCEPT iCloud Drive, below
    "/.Trash/",
    "/node_modules/", "/site-packages/", "/venv/", "/.git/",
    "/__pycache__/",
)


@dataclass
class FileHit:
    path: str
    name: str
    size: Optional[int]
    modified: Optional[str]

    def as_line(self) -> str:
        """One human-readable line — what the LLM reads back to the user."""
        bits = [self.name, f"({self.path})"]
        if self.modified:
            bits.append(f"modificado {self.modified}")
        if self.size is not None:
            bits.append(_human_size(self.size))
        return " · ".join(bits)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{int(size)}GB"


# iCloud Drive lives INSIDE ~/Library, so excluding "/Library/" wholesale hid
# the user's own synced documents along with the app caches it was aimed at —
# a college PDF in ~/Library/Mobile Documents/... was neither findable nor
# readable (s83, found the day after the exclusion list shipped). The
# exception is narrow: this one prefix, not Library at large, and every other
# rule still applies inside it (an .env synced to iCloud stays excluded).
_LIBRARY_EXCEPTIONS = ("/Library/Mobile Documents/com~apple~CloudDocs/",)


# Key material and password stores are excluded by NAME too: the fragments
# only caught them inside their usual folders, so ~/Downloads/id_ed25519,
# client.pem or passwords.csv came back as ordinary hits (s88, Codex lote 3).
# A public key (.pub) is not a secret; ".key" is left out — on a Mac it is
# a Keynote deck far more often than a private key.
_SECRET_NAME_RE = re.compile(
    r"^(?:id_(?:rsa|dsa|ecdsa|ed25519)|\.netrc|\.pgpass|\.npmrc|\.pypirc)$"
    r"|\.(?:pem|p12|pfx|kdbx|keychain-db|ovpn)$"
    r"|^.*\b(?:passwords?|senhas?)\b.*\.csv$",
    re.IGNORECASE,
)


def _is_excluded(path: str) -> bool:
    if _SECRET_NAME_RE.search(os.path.basename(path)):
        return True
    for frag in _EXCLUDED_FRAGMENTS:
        if frag not in path:
            continue
        if frag == "/Library/" and any(ok in path for ok in _LIBRARY_EXCEPTIONS):
            continue                    # iCloud Drive is the user's own stuff
        return True
    return False


def normalise_user_path(raw: str) -> Optional[str]:
    """A path as a HUMAN pasted it → the real file, or None (s83).

    Pedro pasted a shell-escaped path: "Mobile\\ Documents",
    "com\\~apple\\~CloudDocs", "29\\:09\\:26" — and, in the same paste, a
    backslash where a space belonged ("from\\Pearson.pdf"). None of that is
    wrong of him; it is what dragging a file into a terminal produces.

    Tries the literal text, then the un-escaped text, then backslashes read
    as spaces. If the folder resolves but the filename does not, matches by
    squashed name inside THAT folder only — same directory, same characters
    ignoring spaces and punctuation — which recovers a mangled filename
    without ever wandering to another one."""
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        return None
    unescaped = re.sub(r"\\(.)", r"\1", text)
    spaced = re.sub(r"\s{2,}", " ", text.replace("\\", " "))
    for candidate in (text, unescaped, spaced):
        if candidate and os.path.isfile(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    for candidate in (unescaped, spaced):
        folder = os.path.dirname(os.path.expanduser(candidate))
        wanted = _squash(os.path.basename(candidate))
        if not wanted or not os.path.isdir(folder):
            continue
        try:
            for name in os.listdir(folder):
                # A file, never a folder: ".../college/13-10" squash-matched
                # the folder 13-10 itself and came back as "the file" (s85).
                if (_squash(name) == wanted
                        and os.path.isfile(os.path.join(folder, name))):
                    return os.path.join(folder, name)
        except OSError:
            continue
    return None


def _squash(name: str) -> str:
    """A filename with spaces and punctuation removed, for forgiving matches."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


_READABLE_SUFFIXES = (".pdf", ".txt", ".md", ".docx")


def is_readable_path(path: str) -> tuple:
    """May Cortex read this exact file for the user? → (ok, reason).

    The gate for `read_file_at` (s83), deliberately sharing this module's
    `_EXCLUDED_FRAGMENTS` and $HOME scope with the search itself: what cannot
    be FOUND must not become readable by typing the path instead. One list,
    two consumers — the same reason NET_WORDS_RE is shared with the dispatch
    layer.

    Checks the REAL path (symlinks resolved), so a link inside the home folder
    pointing at /etc or another user's files is refused, not followed."""
    raw = (path or "").strip()
    if not raw:
        return (False, "no path given")
    if not os.path.isabs(raw.strip('"').strip("'")):
        return (False, "path must be absolute — use one find_file returned")
    # A pasted path arrives shell-escaped ("Mobile\\ Documents"); resolve that
    # before judging it, or every escaped path reads as "no such file".
    resolved = normalise_user_path(raw)
    real = os.path.realpath(resolved or os.path.expanduser(raw))
    home = os.path.realpath(os.path.expanduser("~"))
    if not (real == home or real.startswith(home + os.sep)):
        return (False, "outside the user's home folder")
    if _is_excluded(real):
        return (False, "in a protected location (credentials, keys, app data)")
    if real.lower().endswith(".py"):
        from src.core import code_roots  # The registry uses folder_gate here.
        if not code_roots.inside(real):
            return (False, "unsupported format — .py only inside a folder authorised for code")
    elif not real.lower().endswith(_READABLE_SUFFIXES):
        return (False, "unsupported format — only .pdf, .txt, .md and .docx")
    if not os.path.isfile(real):
        # iCloud with optimised storage keeps a ".<name>.icloud" stub until the
        # file is downloaded. "No such file" would be true and useless; the
        # user can fix this in one click if we say what it is.
        stub = os.path.join(os.path.dirname(real), f".{os.path.basename(real)}.icloud")
        if os.path.exists(stub):
            return (False, "still in iCloud and not downloaded — open it once "
                           "on the Mac to bring it down")
        return (False, "no such file")
    return (True, real)


def folder_gate(path: str) -> tuple:
    """May Cortex list this folder for the user? → (ok, reason).

    The folder twin of `is_readable_path`, with the same $HOME scope and the
    same protected list (s85): typing "~/.ssh" must not list what typing
    "~/.ssh/id_rsa" would refuse to read. The fragments end in "/", so the
    folder is checked with a trailing separator — "/Users/p/.ssh" alone
    would slip past "/.ssh/"."""
    real = os.path.realpath(os.path.expanduser((path or "").strip()))
    home = os.path.realpath(os.path.expanduser("~"))
    if not (real == home or real.startswith(home + os.sep)):
        return (False, "outside the user's home folder")
    if _is_excluded(real.rstrip(os.sep) + os.sep):
        return (False, "in a protected location (credentials, keys, app data)")
    if not os.path.isdir(real):
        return (False, "no such folder")
    return (True, real)


def readable_files_in(folder: str) -> List[tuple]:
    """The files directly inside `folder` that Cortex could read, as
    (path, size) sorted by name. Hidden files and anything the read gate
    refuses are left out — listing a file is offering to read it. Fail-soft:
    an unreadable folder lists nothing."""
    out = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_file():
                continue
            size = entry.stat().st_size
        except OSError:
            continue
        if is_readable_path(entry.path)[0]:
            out.append((entry.path, size))
    return sorted(out, key=lambda item: os.path.basename(item[0]).casefold())


def human_size(n: int) -> str:
    """Public name for the size formatter the search results already use."""
    return _human_size(n)


class FileFinder:
    """Wraps `mdfind`. Fail-soft: any error returns an empty result list plus
    a short reason, so a search never breaks a turn."""

    def __init__(self, home: Optional[str] = None,
                 timeout: int = _DEFAULT_TIMEOUT_S):
        self.home = home or os.path.expanduser("~")
        self.timeout = timeout

    def search(self, query: str, limit: int = _DEFAULT_LIMIT,
               content: bool = False) -> tuple:
        """Find files whose NAME matches `query` (default) or whose CONTENT
        does (`content=True`). Returns (hits, error) — `error` is None on
        success, else a short human-readable reason."""
        q = (query or "").strip()
        if not q:
            return [], "empty query"
        limit = max(1, min(int(limit or _DEFAULT_LIMIT), _MAX_LIMIT))

        args = ["mdfind", "-onlyin", self.home]
        if not content:
            args += ["-name", q]
        else:
            args.append(q)

        try:
            proc = subprocess.run(args, capture_output=True, text=True,
                                  timeout=self.timeout)
        except FileNotFoundError:
            return [], "mdfind não disponível (Spotlight)"
        except subprocess.TimeoutExpired:
            return [], f"busca excedeu {self.timeout}s"
        except Exception as exc:  # noqa: BLE001 — search must never raise
            return [], f"falha na busca: {type(exc).__name__}"

        if proc.returncode != 0:
            # stderr is a diagnostic, and the error rides in the tool result
            # the follow-up repeats (s88, Codex lote 8).
            log.warning("mdfind failed: %s", (proc.stderr or "").strip()[:200])
            return [], "mdfind falhou"

        hits: List[FileHit] = []
        for line in (proc.stdout or "").splitlines():
            path = line.strip()
            if not path or _is_excluded(path):
                continue
            hits.append(self._describe(path))
            if len(hits) >= limit:
                break
        return hits, None

    @staticmethod
    def _describe(path: str) -> FileHit:
        size = modified = None
        try:
            st = os.stat(path)
            size = st.st_size
            modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
        except OSError:
            pass          # a hit that vanished between index and stat
        return FileHit(path=path, name=os.path.basename(path),
                       size=size, modified=modified)


def format_results(hits: List[FileHit], query: str, error: Optional[str],
                   lang: str = "pt") -> str:
    """Render the search outcome as the tool's string result (what the LLM
    receives). Bounded and factual — no invention, no file contents."""
    if error:
        return (f"[SYSTEM ERROR] find_file: {error}" if lang == "en"
                else f"[SYSTEM ERROR] find_file: {error}")
    if not hits:
        return (f"No file found matching '{query}'." if lang == "en"
                else f"Nenhum arquivo encontrado para '{query}'.")
    head = (f"{len(hits)} file(s) found for '{query}':" if lang == "en"
            else f"{len(hits)} arquivo(s) encontrado(s) para '{query}':")
    return head + "\n" + "\n".join(f"- {h.as_line()}" for h in hits)


finder = FileFinder()
