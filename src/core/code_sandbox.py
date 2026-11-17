"""Dormant command boundary: Seatbelt around a disposable project copy."""
import os
import shutil
import subprocess
import sys
import tempfile

from src.core import code_commands

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
WORKSPACE_PREFIX = "cortex-code-run-"
MAX_FILES = 2000
MAX_BYTES = 50 * 1024 * 1024


class SandboxUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        return (os.path.isfile(SANDBOX_EXEC) and os.access(SANDBOX_EXEC, os.X_OK)
                and subprocess.run([SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
                    stdin=subprocess.DEVNULL, capture_output=True,
                    env={"PATH": "/usr/bin:/bin"}, timeout=5).returncode == 0)
    except Exception:
        return False  # A missing or unusable sandbox never permits execution.


def profile_text(workspace) -> str:
    workspace = os.fspath(workspace)
    if (not os.path.isabs(workspace) or workspace != os.path.realpath(workspace)
            or not os.path.isdir(workspace)):
        raise ValueError("workspace must be an absolute, canonical directory")

    def q(path):
        if any(ord(c) < 32 or ord(c) == 127 for c in path):
            raise ValueError("control character in sandbox path")
        return '"' + path.replace('\\', '\\\\').replace('"', '\\"') + '"'

    return f'''(version 1)
(deny default)
(allow process-exec (literal {q(os.path.realpath(sys.executable))}) (literal {q(os.path.join(sys.base_prefix, "Resources/Python.app/Contents/MacOS/Python"))}))
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read*
  (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share")
  (subpath "/private/var/db/dyld") (subpath "/Library/Apple")
  (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")
  (subpath {q(sys.base_prefix)}) (subpath {q(sys.prefix)}) (subpath {q(workspace)}))
(allow file-write* (subpath {q(workspace)}) (literal "/dev/null"))
'''


def prepare_workspace(folder) -> str:
    dst = tempfile.mkdtemp(prefix=WORKSPACE_PREFIX)
    try:
        count = size = 0

        def walk_error(error):
            raise error  # An unreadable subtree must not bypass the limits.

        for root, dirs, files in os.walk(folder, followlinks=False, onerror=walk_error):
            # Directory symlinks are copied as links, so count their lstat sizes too.
            for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
                count += 1
                size += os.lstat(os.path.join(root, name)).st_size
                if count > MAX_FILES or size > MAX_BYTES:
                    raise ValueError("project exceeds workspace limits")
        shutil.copytree(folder, dst, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".git"), dirs_exist_ok=True)
        return os.path.realpath(dst)
    except Exception:
        shutil.rmtree(dst)
        raise


def _check_workspace(path) -> str:
    path = os.fspath(path)
    if (not os.path.isabs(path) or path != os.path.realpath(path) or os.path.islink(path)
            or not os.path.isdir(path) or not os.path.basename(path).startswith(WORKSPACE_PREFIX)
            or os.path.realpath(os.path.dirname(path)) != os.path.realpath(tempfile.gettempdir())):
        raise ValueError(f"not a disposable {WORKSPACE_PREFIX} workspace")
    return path


def discard_workspace(path):
    shutil.rmtree(_check_workspace(path))


def contained_command(profile, workspace) -> dict:
    ok, reason = code_commands.validate(profile)
    if not ok:
        raise ValueError(reason)
    if not available():
        raise SandboxUnavailable("sandbox-exec unavailable here")
    workspace = _check_workspace(workspace)
    argv = list(profile["argv"])
    for flag in ("-s", "-t"):
        argv[argv.index(flag) + 1] = workspace
    return {"argv": [SANDBOX_EXEC, "-p", profile_text(workspace), *argv], "cwd": workspace,
            "env": dict(profile["env"]), "timeout_s": profile["timeout_s"],
            "max_output_bytes": profile["max_output_bytes"]}
