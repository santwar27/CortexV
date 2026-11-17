import logging
import os
import re

# Whitelist for user-supplied filenames. Catches null bytes, slashes,
# backslashes, control chars, and other oddities that os.path normalisation
# wouldn't reject on its own.
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


log = logging.getLogger(__name__)


class LocalFileWriter:
    """
    Tool to safely write and append to local text files.
    Implements path traversal protection.
    """

    # Engine failure replies must never become persisted "knowledge".
    _ERROR_MARKERS = (
        "Insufficient data due to engine failure",
        "[SYSTEM ERROR]",
        "[OFFLINE]",
    )

    def __init__(self, base_dir: str = "data/documents"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def write_document(self, filename: str, content: str) -> str:
        """APPEND `content` to the named file in the vault (creates it if
        absent). Append, not overwrite, despite the name — repeating the
        same call adds a second copy of the text; nothing is ever lost,
        but nothing is deduplicated either (M4-A14b: documented contract;
        TOOL_SPECS already describes it as append to the LLM). The reply
        says 'appended' so the user hears the real semantics."""
        if any(marker in content for marker in self._ERROR_MARKERS):
            return "[SYSTEM ERROR] Refused to persist content: looks like an engine failure response."

        if not filename or not _FILENAME_RE.match(filename):
            return "[SYSTEM ERROR] Invalid filename. Use only letters, digits, dot, underscore, hyphen."

        # Force .txt extension to prevent accidentally creating executable scripts (.py, .sh)
        if not filename.lower().endswith('.txt'):
            filename += '.txt'

        # realpath (not abspath) resolves symlinks, so a planted symlink
        # inside the vault that points outside is caught by the prefix check.
        base_real = os.path.realpath(self.base_dir)
        candidate = os.path.join(self.base_dir, filename)
        # If the file exists as a symlink, realpath follows it. If it doesn't
        # exist yet, realpath of its dir + the filename keeps the check honest.
        if os.path.lexists(candidate):
            safe_path = os.path.realpath(candidate)
        else:
            safe_path = os.path.join(base_real, filename)
        if not safe_path.startswith(base_real + os.sep) and safe_path != base_real:
            return "[SYSTEM ERROR] Security breach attempt blocked. Path traversal denied."

        try:
            # 'a' (append) mode adds text at the end. Creates the file if it doesn't exist.
            with open(safe_path, 'a', encoding='utf-8') as f:
                f.write(content + "\n\n")
            return f"Information successfully saved (appended) to '{filename}'."
        except Exception as e:
            # The exception goes to the log, never into the result: the follow-up is
            # told to repeat a tool error verbatim, and the 8B read it out 6 times in
            # 6 ("O erro foi: EOF marker not found" — s88, Codex lote 8).
            log.warning("write_document %r failed: %r", filename, e)
            return f"[SYSTEM ERROR] Failed to write document '{filename}'."