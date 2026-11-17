"""Explicit terminal grants for read-only code access; never model authority."""
import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

from src.core.audit import audit
from src.tools.file_finder import folder_gate


def _path() -> str:
    return os.path.expanduser(os.environ.get(
        "CORTEX_CODE_ROOTS_FILE", "~/.cortex/code_roots.json"))


def _load() -> list[dict]:
    try:
        with open(_path(), encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            return []
        for item in records:
            root = item["root"]
            added = datetime.fromisoformat(item["added_at"])
            if (not os.path.isabs(root) or os.path.realpath(root) != root
                    or added.utcoffset() != timedelta(0)):
                return []
        return records
    except (OSError, ValueError, TypeError, KeyError):
        return []  # A partial or damaged registry grants nothing.


def roots() -> list[str]:
    return [item["root"] for item in _load()]


def inside(real_path: str) -> bool:
    real = os.path.realpath(real_path)
    return any(real.startswith(root + os.sep) for root in roots())


def _write(records: list[dict]) -> None:
    path = _path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    # Same-folder replacement keeps readers from seeing a partial grant.
    fd, temporary = tempfile.mkstemp(prefix=".code_roots-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record(event: str, real: str) -> None:
    try:
        audit.log("–", "code_roots", event, payload={"root": real})
    except Exception:
        pass  # Audit availability must not prevent a grant or revocation.


def parse_command(text: str) -> tuple[str, str | None, str] | None:
    """Whole-message fixed phrases; the caller supplies only Pedro's words."""
    text = re.sub(r"^\s*(?:(?:cortex|por favor|please)[\s,]+)*", "", text or "", flags=re.I)
    text = re.sub(r"[,\s]+(?:por favor|please)[?.!]?\s*$", "", text, flags=re.I).strip()
    for action, lang, pattern in (
        ("add", "pt", r"autoriza a pasta (.+?) para código"),
        ("add", "en", r"authori[sz]e the folder (.+?) for code"),
        ("remove", "pt", r"tira a autorização da pasta (.+)"),
        ("remove", "en", r"revoke the folder (.+)"),
        ("list", "pt", r"quais pastas de código"),
        ("list", "en", r"which code folders"),
    ):
        # A revoke ends in the path: its final punctuation may be part of a name.
        match = re.fullmatch(pattern + (r"[?.!]?" if action != "remove" else ""), text, re.I)
        if match:
            path = os.path.expanduser(match[1].strip()) if action != "list" else None
            return (action, path, lang) if path or action == "list" else None
    return None


def check(path: str) -> tuple[bool, str]:
    """Validate without writing; on success return the canonical folder."""
    real = os.path.realpath(os.path.expanduser(path))
    home = os.path.realpath(os.path.expanduser("~"))
    reason = None
    if not os.path.exists(real):
        reason = "não existe"
    elif not os.path.isdir(real):
        reason = "não é uma pasta"
    elif real == home:
        reason = "é a sua pasta pessoal inteira — escolha uma pasta dentro dela"
    elif not real.startswith(home + os.sep):
        reason = "fica fora da sua pasta pessoal"
    elif not folder_gate(real)[0]:
        reason = "é um lugar protegido (credenciais, chaves, dados de apps)"
    if reason:
        return False, f"Não autorizei {path}: {reason}."
    return True, real


def add(path: str) -> tuple[bool, str]:
    ok, real = check(path)
    if not ok:
        return False, real
    records = _load()
    if any(item["root"] == real for item in records):
        return True, f"Essa pasta já estava autorizada: {real}"
    records.append({"root": real, "added_at": datetime.now(timezone.utc).isoformat()})
    _write(records)
    _record("add", real)
    return True, f"Pasta autorizada para o Cortex ler código: {real}"


def remove(path: str) -> tuple[bool, str]:
    real = os.path.realpath(os.path.expanduser(path))
    records = _load()
    remaining = [item for item in records if item["root"] != real]
    if len(remaining) == len(records):
        return False, f"Essa pasta não estava autorizada: {path}"
    _write(remaining)
    _record("remove", real)
    return True, f"Autorização removida: {real}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pastas autorizadas para código")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("add", "remove"):
        commands.add_parser(command).add_argument("path")
    commands.add_parser("list")
    args = parser.parse_args(argv)
    if args.command == "list":
        authorised = roots()
        print("Pastas autorizadas para código:\n" + "\n".join(f"- {r}" for r in authorised)
              if authorised else "Nenhuma pasta autorizada para código.")
        return 0
    try:
        ok, message = (add if args.command == "add" else remove)(args.path)
    except OSError:
        ok, message = False, "Não foi possível guardar as autorizações para código."
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
