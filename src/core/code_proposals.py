"""Private, content-addressed proposal artifacts; never writes source files."""
import hashlib
import json
import os
import re
import tempfile
from typing import Optional


def _directory() -> str:
    return os.path.expanduser(os.environ.get(
        "CORTEX_CODE_PROPOSALS_DIR", "~/.cortex/code_proposals"))


def _encoded(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")


def save(record: dict) -> str:
    data = _encoded(record)
    pid = hashlib.sha256(data).hexdigest()[:16]
    directory = _directory()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".proposal-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, os.path.join(directory, pid + ".json"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return pid


def load(pid: str) -> Optional[dict]:
    if not isinstance(pid, str) or not re.fullmatch(r"[0-9a-f]{16}", pid):
        return None
    try:
        with open(os.path.join(_directory(), pid + ".json"), encoding="utf-8") as f:
            record = json.load(f)
        if isinstance(record, dict) and hashlib.sha256(_encoded(record)).hexdigest()[:16] == pid:
            return record
    except (OSError, ValueError, TypeError):
        pass
    return None


def save_original(pid, data: bytes) -> None:
    record = load(pid)
    if record is None or hashlib.sha256(data).hexdigest() != record.get("original_sha256"):
        raise ValueError("original does not match proposal")
    directory = _directory()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".original-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, os.path.join(directory, pid + ".orig"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_original(pid) -> Optional[bytes]:
    record = load(pid)
    if record is None:
        return None
    try:
        with open(os.path.join(_directory(), pid + ".orig"), "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() == record.get("original_sha256"):
            return data
    except OSError:
        pass
    return None
