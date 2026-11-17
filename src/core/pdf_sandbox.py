"""Dormant fixed Poppler discovery and Seatbelt command boundary; no job runner."""
from dataclasses import dataclass
import os
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping
import unicodedata

TOOLS = ("pdfinfo", "pdftotext", "pdftoppm", "pdfimages", "pdfseparate", "pdfunite")
HOMEBREW_BIN = "/opt/homebrew/bin"
POPPLER_CELLAR = "/opt/homebrew/Cellar/poppler"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
FONTCONFIG_FILE = "/opt/homebrew/etc/fonts/fonts.conf"


@dataclass(frozen=True)
class PopplerTools:
    paths: Mapping[str, str]
    version: str

    def __post_init__(self):
        # A frozen dataclass alone would still expose a mutable caller-owned dict.
        object.__setattr__(self, "paths", MappingProxyType(dict(self.paths)))


def _q(path: str) -> str:
    if not isinstance(path, str) or any(unicodedata.category(c) == "Cc" for c in path):
        raise ValueError("control character or non-string sandbox path")
    return '"' + path.replace('\\', '\\\\').replace('"', '\\"') + '"'


def discover() -> PopplerTools | None:
    """Accept only the six fixed links into a single version's Poppler bin dir."""
    try:
        paths = {}
        versions = set()
        for name in TOOLS:
            path = os.path.realpath(os.path.join(HOMEBREW_BIN, name))
            _q(path)
            parts = os.path.relpath(path, POPPLER_CELLAR).split(os.sep)
            if (len(parts) != 3 or parts[0] in (".", "..")
                    or parts[1:] != ["bin", name]
                    or not os.path.isfile(path) or not os.access(path, os.X_OK)):
                return None
            paths[name] = path
            versions.add(parts[0])
        if len(versions) != 1:
            return None
        return PopplerTools(paths, versions.pop())
    except (OSError, ValueError):
        return None


def _directory(path) -> str:
    try:
        path = os.fspath(path)
    except TypeError as error:
        raise ValueError("directory must be a filesystem path") from error
    _q(path)
    if (not os.path.isabs(path) or os.path.realpath(path) != path
            or not os.path.isdir(path)):
        raise ValueError("directory must be absolute, canonical and existing")
    return path


def profile_text(tools: PopplerTools, input_dir, output_dir) -> str:
    # Refuse forged/stale records as well as paths supplied by a caller. This is
    # validation, not a filesystem lock: Homebrew installation integrity is trusted.
    if not isinstance(tools, PopplerTools) or tools != discover():
        raise ValueError("tools must match fixed Poppler discovery")
    input_dir, output_dir = _directory(input_dir), _directory(output_dir)
    if os.path.commonpath((input_dir, output_dir)) in (input_dir, output_dir):
        raise ValueError("input and output directories must be disjoint")
    executables = " ".join(f"(literal {_q(tools.paths[name])})" for name in TOOLS)
    # Preserve the proven runtime reads. In particular, literal / is necessary
    # for dyld's shared cache: omitting it can SIGABRT before main and emit reports.
    return f'''(version 1)
(deny default)
(allow process-exec {executables})
(allow file-read-metadata)
(allow file-read*
  (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share")
  (subpath "/private/var/db/dyld") (subpath "/Library/Apple") (subpath "/Library/Fonts")
  (subpath "/opt/homebrew/Cellar") (subpath "/opt/homebrew/opt") (subpath "/opt/homebrew/lib")
  (subpath "/opt/homebrew/share") (subpath "/opt/homebrew/etc/fonts")
  (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")
  (subpath {_q(input_dir)}) (subpath {_q(output_dir)}))
(allow file-write* (subpath {_q(output_dir)}) (literal "/dev/null"))
(allow sysctl-read)
(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))
'''


def contained_command(tools: PopplerTools, tool: str, args, input_dir, output_dir) -> tuple[list[str], dict[str, str]]:
    if (not isinstance(tool, str) or tool not in TOOLS
            or not isinstance(tools, PopplerTools) or tool not in tools.paths):
        raise ValueError("unknown Poppler tool")
    if isinstance(args, (str, bytes)):
        raise ValueError("args must be a sequence of strings")
    try:
        args = list(args)
    except TypeError as error:
        raise ValueError("args must be a sequence of strings") from error
    if any(not isinstance(arg, str) or "\0" in arg for arg in args):
        raise ValueError("each argument must be a string without NUL")
    profile = profile_text(tools, input_dir, output_dir)
    output_dir = os.fspath(output_dir)
    env = {"PATH": "/usr/bin:/bin", "HOME": output_dir, "TMPDIR": output_dir,
           "XDG_CACHE_HOME": output_dir, "LANG": "en_US.UTF-8"}
    if os.path.isfile(FONTCONFIG_FILE):
        env["FONTCONFIG_FILE"] = FONTCONFIG_FILE
    return [SANDBOX_EXEC, "-p", profile, tools.paths[tool], *args], env


def available() -> bool:
    """Probe the complete restrictive profile without a PDF; always fail closed."""
    try:
        if not os.path.isfile(SANDBOX_EXEC) or not os.access(SANDBOX_EXEC, os.X_OK):
            return False
        tools = discover()
        if tools is None:
            return False
        with tempfile.TemporaryDirectory(prefix="cortex-pdf-probe-") as raw:
            # macOS /var is an alias; Seatbelt rules must use the real /private path.
            root = os.path.realpath(raw)
            input_dir, output_dir = os.path.join(root, "input"), os.path.join(root, "output")
            os.mkdir(input_dir)
            os.mkdir(output_dir)
            argv, env = contained_command(tools, "pdfinfo", ["-v"], input_dir, output_dir)
            return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
