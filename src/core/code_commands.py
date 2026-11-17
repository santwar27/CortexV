"""One fixed command description; execution waits for a containment proof."""
import os
import shlex
import sys

from src.core import code_roots


def tests_profile(folder: str) -> dict:
    return {"kind": "tests", "argv": [sys.executable, "-I", "-B", "-m", "unittest", "discover",
            "-s", folder, "-t", folder, "-p", "test*.py"], "cwd": folder,
            "env": {"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
            "timeout_s": 30, "network": False, "max_output_bytes": 65536}


def validate(profile: dict) -> tuple[bool, str]:
    if not isinstance(profile, dict) or not isinstance(profile.get("cwd"), str):
        return False, "missing folder"
    cwd = profile["cwd"]
    if not os.path.isabs(cwd) or cwd != os.path.realpath(cwd):
        return False, "folder must be absolute and canonical"
    if not os.path.isdir(cwd) or (cwd not in code_roots.roots() and not code_roots.inside(cwd)):
        return False, "folder is not authorised"
    expected = tests_profile(cwd)
    # Equality alone accepts 0 as False and 30.0 as 30 in a saved JSON record.
    if profile != expected or any(type(profile[k]) is not type(v) for k, v in expected.items()):
        return False, "command differs from fixed profile"
    return True, ""


def render(profile: dict) -> str:
    return shlex.join(profile["argv"])
