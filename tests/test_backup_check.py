"""backup_check approves exactly what backup.sh copies (s88, Codex lote 5).

The checker matched a ~/.cortex name anywhere in backup.sh's TEXT. It
passed "archive" (a comment says "archive everything") and "palace.db"
(the script copies the Palace under data/, another path) — neither is in
the tar. It reads the capture loops of the ~/.cortex block now, and the
proof is end to end: everything it approves comes back out of a real
backup run against a throwaway $HOME.
"""
import importlib.util
import sqlite3
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = ROOT / "scripts" / "backup.sh"

_spec = importlib.util.spec_from_file_location("backup_check", ROOT / "scripts" / "backup_check.py")
backup_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backup_check)


def _captured():
    return backup_check.captured(BACKUP_SCRIPT.read_text())


def test_names_in_comments_or_other_paths_are_not_captured():
    script = ('DOT_CORTEX="$HOME/.cortex"\n'
              "# archive everything — a comment, not a copy\n"
              "for db in chat.db; do\n  :\ndone\n"
              "# ---- Build the tar ----\n")
    assert backup_check.captured(script) == {"chat.db"}
    names = _captured()
    assert "palace.db" not in names         # copied from data/, not ~/.cortex
    assert {"chat.db", "focus.db", "projects.db", "effects.db", "audit.jsonl", "enroll_clips"} <= names
    assert "archive" in names               # the retention's output (lote 6)


def test_the_checker_flags_what_the_backup_skips(tmp_path, capsys):
    dot = tmp_path / ".cortex"
    (dot / "old_exports").mkdir(parents=True)
    (dot / "old_exports" / "old.jsonl").write_text("x")
    (dot / "palace.db").write_bytes(b"")
    assert backup_check.main(["backup_check", str(dot)]) == 1
    out = capsys.readouterr().out
    assert "UNCLASSIFIED  old_exports" in out and "UNCLASSIFIED  palace.db" in out


def test_everything_it_approves_is_in_the_tar(tmp_path):
    home, src, dest = tmp_path / "home", tmp_path / "src", tmp_path / "backups"
    dot = home / ".cortex"
    dot.mkdir(parents=True)
    src.mkdir()
    for name in _captured():
        if name.endswith(".db"):
            con = sqlite3.connect(str(dot / name))
            con.execute("CREATE TABLE t (x)")
            con.commit()
            con.close()
        elif "." in name:
            (dot / name).write_text("sentinel")
        else:                                           # a directory
            (dot / name).mkdir()
            (dot / name / "clip.wav").write_bytes(b"RIFF")
    assert backup_check.main(["backup_check", str(dot)]) == 0
    proc = subprocess.run(
        ["bash", str(BACKUP_SCRIPT)], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home),
             "CORTEX_SRC_DIR": str(src), "CORTEX_BACKUP_DIR": str(dest)})
    assert proc.returncode == 0, proc.stderr
    [snap] = dest.glob("cortex_*.tar.gz")
    with tarfile.open(snap) as tar:
        inside = {n[len("./_home_cortex/"):].split("/")[0] for n in tar.getnames()
                  if n.startswith("./_home_cortex/")} - {""}
        inside = {n for n in inside if not n.startswith("._")}   # macOS xattr sidecars
    assert inside == _captured()


def test_a_lock_file_is_transient():
    """gate_queue.jsonl.lock is an empty flock target (s88, Codex lote 10)."""
    from scripts.backup_check import _classify
    assert _classify("gate_queue.jsonl.lock", "") is None
