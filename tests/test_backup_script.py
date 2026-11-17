"""
End-to-end tests for `scripts/backup.sh`.

The script is bash, so unit-testing it from Python means treating it
as a black box: stage a fake `cortex_project` in a tmp dir, point
the script at it via `CORTEX_SRC_DIR` + `CORTEX_BACKUP_DIR`, run it,
and inspect the tar.gz output. Covers:

  - basic happy path (snapshot created, integrity check passes,
    all whitelisted paths present)
  - missing optional paths don't break the script (e.g.
    study_queue.jsonl absent on first install)
  - palace.db snapshot is live-consistent and re-openable
  - retention keeps the 14 most recent + 8 ISO-week-anchored
  - retention deletes anything outside both sets

These tests do NOT touch the real backup destination
(`~/Backups/cortex/`) — fully sandboxed in tmp.
"""
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from pathlib import Path

import pytest


BACKUP_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "backup.sh"
)


@pytest.fixture
def fake_cortex(tmp_path):
    """Build a minimal cortex_project layout with every whitelisted
    path present. Tests can selectively delete files to exercise
    the "missing optional path" branches."""
    src = tmp_path / "cortex_project"
    src.mkdir()
    (src / ".env").write_text("CORTEX_API_KEY=test\nTAVILY_API_KEY=test\n")

    data = src / "data"
    data.mkdir()
    (data / "gate_state.json").write_text('{"lease_until": null}')
    (data / "last_announce.txt").write_text("2026-05-27")
    (data / "study_queue.jsonl").write_text('{"topic": "test"}\n')

    creds = data / "credentials"
    creds.mkdir()
    (creds / "spotify_token.json").write_text('{"access_token": "fake"}')
    (creds / "google_token.json").write_text('{"token": "fake"}')

    knowledge = data / "knowledge"
    knowledge.mkdir()
    (knowledge / "approved").mkdir()
    (knowledge / "approved" / "note1.md").write_text("# note")
    (knowledge / "pending").mkdir()

    docs = data / "documents"
    docs.mkdir()
    (docs / "Cortex_comandos.txt").write_text("...")

    # Real SQLite Palace with one table so we can verify the .backup
    # snapshot round-trips correctly.
    palace = sqlite3.connect(str(data / "palace.db"))
    palace.execute(
        "CREATE TABLE documentos (id INTEGER PRIMARY KEY, content TEXT)"
    )
    palace.execute(
        "INSERT INTO documentos (content) VALUES ('hello'), ('world')"
    )
    palace.commit()
    palace.close()

    logs = src / "logs"
    logs.mkdir()
    (logs / "auth.log").write_text("2026-05-27 OK ip=127.0.0.1\n")
    (logs / "nightly.out.log").write_text("nightly run ok\n")
    (logs / "nightly.err.log").write_text("")

    return src


@pytest.fixture
def backup_dir(tmp_path):
    return tmp_path / "backups"


def _run_backup(src: Path, dest: Path,
                home: Path = None) -> subprocess.CompletedProcess:
    """Invoke backup.sh against the staged tree. Returns the
    CompletedProcess so callers can assert exit code + stdout/stderr.

    `home` overrides $HOME so the ~/.cortex sidecar block reads a FAKE
    home instead of Pedro's real one — required to assert its contents
    deterministically (and on any other machine). Without one, an EMPTY
    throwaway home: the default used to be the real $HOME, so every
    happy-path tar copied Pedro's audit, contacts, chat and voiceprint
    into pytest's tmp (s88, Codex lote 5 — reproduced with a sentinel)."""
    if home is None:
        home = Path(src).parent / "empty_home"
        home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CORTEX_SRC_DIR": str(src),
        "CORTEX_BACKUP_DIR": str(dest),
        "HOME": str(home),
    }
    return subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )


def _latest_snapshot(backup_dir: Path) -> Path:
    """Resolve the single tar.gz we expect to find."""
    snaps = sorted(backup_dir.glob("cortex_*.tar.gz"))
    assert snaps, f"No snapshot in {backup_dir}: {list(backup_dir.iterdir())}"
    return snaps[-1]


class TestHappyPath:

    def test_the_real_home_is_never_read(self, fake_cortex, backup_dir, monkeypatch, tmp_path):
        """A sentinel stands in for the real $HOME of the run."""
        sentinel_home = tmp_path / "sentinel_home"
        (sentinel_home / ".cortex").mkdir(parents=True)
        (sentinel_home / ".cortex" / "audit.jsonl").write_text("SENTINEL")
        monkeypatch.setenv("HOME", str(sentinel_home))
        assert _run_backup(fake_cortex, backup_dir).returncode == 0
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            assert not [n for n in tar.getnames() if "_home_cortex" in n]

    def test_script_exits_zero(self, fake_cortex, backup_dir):
        result = _run_backup(fake_cortex, backup_dir)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_snapshot_created(self, fake_cortex, backup_dir):
        _run_backup(fake_cortex, backup_dir)
        snap = _latest_snapshot(backup_dir)
        assert snap.exists()
        assert snap.stat().st_size > 0

    def test_snapshot_integrity_check_passes(self, fake_cortex, backup_dir):
        _run_backup(fake_cortex, backup_dir)
        snap = _latest_snapshot(backup_dir)
        # tarfile raises if corrupted; if it opens and lists, integrity is fine.
        with tarfile.open(snap, "r:gz") as t:
            names = t.getnames()
        assert names, "tar reported empty archive"

    def test_all_whitelisted_paths_present(self, fake_cortex, backup_dir):
        _run_backup(fake_cortex, backup_dir)
        with tarfile.open(_latest_snapshot(backup_dir), "r:gz") as t:
            names = set(t.getnames())
        expected = {
            "./.env",
            "./data/palace.db",
            "./data/credentials/spotify_token.json",
            "./data/credentials/google_token.json",
            "./data/gate_state.json",
            "./data/last_announce.txt",
            "./data/study_queue.jsonl",
            "./data/knowledge/approved/note1.md",
            "./data/documents/Cortex_comandos.txt",
            "./logs/auth.log",
            "./logs/nightly.out.log",
            "./logs/nightly.err.log",
        }
        missing = expected - names
        assert not missing, f"missing from snapshot: {missing}"

    def test_palace_snapshot_is_restorable(
        self, fake_cortex, backup_dir, tmp_path
    ):
        """The whole point of sqlite3 .backup: the captured .db has
        to open cleanly and have the original rows."""
        _run_backup(fake_cortex, backup_dir)
        restore = tmp_path / "restore"
        restore.mkdir()
        with tarfile.open(_latest_snapshot(backup_dir), "r:gz") as t:
            t.extractall(restore)
        restored_db = restore / "data" / "palace.db"
        assert restored_db.exists()
        conn = sqlite3.connect(str(restored_db))
        rows = conn.execute("SELECT content FROM documentos ORDER BY id").fetchall()
        conn.close()
        assert rows == [("hello",), ("world",)]

    def test_excluded_paths_not_in_snapshot(self, fake_cortex, backup_dir):
        """venv, __pycache__, voice/, uvicorn logs must not leak in.
        Create some bait files to be sure."""
        (fake_cortex / "venv").mkdir()
        (fake_cortex / "venv" / "marker").write_text("DO NOT BACKUP")
        (fake_cortex / "__pycache__").mkdir()
        (fake_cortex / "__pycache__" / "x.pyc").write_text("")
        (fake_cortex / "data" / "voice").mkdir()
        (fake_cortex / "data" / "voice" / "model.bin").write_text("BIG")
        (fake_cortex / "logs" / "uvicorn.out.log").write_text("runtime")

        _run_backup(fake_cortex, backup_dir)
        with tarfile.open(_latest_snapshot(backup_dir), "r:gz") as t:
            names = "\n".join(t.getnames())
        assert "venv" not in names
        assert "__pycache__" not in names
        assert "data/voice" not in names
        assert "uvicorn.out.log" not in names


class TestMissingOptionalPaths:
    """A fresh install lacks some paths (e.g. study_queue.jsonl
    only appears after the first study). The script must skip
    missing entries silently, not fail."""

    def test_no_study_queue_yet(self, fake_cortex, backup_dir):
        (fake_cortex / "data" / "study_queue.jsonl").unlink()
        result = _run_backup(fake_cortex, backup_dir)
        assert result.returncode == 0
        with tarfile.open(_latest_snapshot(backup_dir), "r:gz") as t:
            names = set(t.getnames())
        # Other paths still made it
        assert "./data/palace.db" in names
        assert "./.env" in names
        # The missing one is just absent (no error)
        assert "./data/study_queue.jsonl" not in names

    def test_no_palace_yet(self, fake_cortex, backup_dir):
        """Edge case: fresh install before Palace was first opened."""
        (fake_cortex / "data" / "palace.db").unlink()
        result = _run_backup(fake_cortex, backup_dir)
        assert result.returncode == 0
        with tarfile.open(_latest_snapshot(backup_dir), "r:gz") as t:
            names = set(t.getnames())
        assert "./data/palace.db" not in names
        # Other paths unaffected
        assert "./.env" in names


class TestRetention:
    """The script keeps:
      (a) the 14 most recent snapshots
      (b) the most recent snapshot from each of the last 8 ISO weeks
    Implementation works on file mtime, so we mock that by touching
    the snapshots after creating them."""

    def _seed(self, backup_dir: Path, count: int, days_apart: int = 1):
        """Drop `count` empty-ish snapshots into backup_dir with
        mtimes spread `days_apart` days apart (most recent = today)."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        day = 86400
        for i in range(count):
            ts = now - i * days_apart * day
            # Use timestamp suffix that matches the real script's naming
            label = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(ts))
            # Disambiguate when several files would land in the same second.
            f = backup_dir / f"cortex_{label}_{i:03d}.tar.gz"
            # Minimal valid tar.gz so the script's integrity check is happy
            # if it ever inspected these (it doesn't — retention is mtime-only).
            with tarfile.open(f, "w:gz") as t:
                pass
            os.utime(f, (ts, ts))

    def test_keeps_all_when_under_14(self, fake_cortex, backup_dir):
        self._seed(backup_dir, count=5, days_apart=1)
        _run_backup(fake_cortex, backup_dir)
        # 5 pre-existing + 1 just created = 6, all kept
        remaining = list(backup_dir.glob("cortex_*.tar.gz"))
        assert len(remaining) == 6

    def test_keeps_14_most_recent_plus_weekly(self, fake_cortex, backup_dir):
        """With 60 snapshots spread one per day (~8.5 weeks):
        retention keeps 14 most recent + ~8 weekly anchors.
        Allow a small range because day-of-week of "today" affects
        whether the 14-daily window already covers some weekly anchors."""
        self._seed(backup_dir, count=60, days_apart=1)
        _run_backup(fake_cortex, backup_dir)
        remaining = list(backup_dir.glob("cortex_*.tar.gz"))
        # 14 daily + up to 8 distinct ISO weeks + 1 new snapshot from
        # the run itself. The 14 daily and the first ~2 weekly anchors
        # overlap, so the keep set is 14 + ~6 unique weeklies + 1 new.
        # Upper bound 23, lower bound 15 (worst case if all weeks
        # already covered by the 14 daily).
        assert 15 <= len(remaining) <= 23, f"got {len(remaining)} remaining"

    def test_deletes_old_unweekly_snapshots(self, fake_cortex, backup_dir):
        """A snapshot from 6 months ago, not the most-recent-in-its-week
        AND outside the 14-day window, must be deleted."""
        self._seed(backup_dir, count=30, days_apart=1)
        # Add an obvious deletion candidate: 200 days old.
        ancient_ts = time.time() - 200 * 86400
        ancient_label = time.strftime(
            "%Y-%m-%d_%H%M%S", time.localtime(ancient_ts)
        )
        ancient = backup_dir / f"cortex_{ancient_label}_ancient.tar.gz"
        with tarfile.open(ancient, "w:gz"):
            pass
        os.utime(ancient, (ancient_ts, ancient_ts))

        _run_backup(fake_cortex, backup_dir)
        assert not ancient.exists(), "ancient snapshot should have been deleted"


class TestSandbox:
    """Sanity: the script must not touch the real ~/Backups/cortex or
    the real cortex_project. Caught by env-var override + tmp paths."""

    def test_real_backup_dir_untouched(self, fake_cortex, backup_dir):
        # If the override didn't work, this would clobber the real dir.
        result = _run_backup(fake_cortex, backup_dir)
        assert result.returncode == 0
        # backup_dir is in pytest's tmp_path — confirm the snapshot
        # actually landed there and not in $HOME/Backups/cortex.
        assert any(backup_dir.glob("cortex_*.tar.gz"))
        # Defensive: ensure we didn't accidentally write to $HOME.
        # Skip if $HOME/Backups/cortex doesn't exist yet (the user hasn't
        # installed the launchd job). If it DOES exist, make sure the
        # snapshot we just created isn't in it.
        real_dir = Path.home() / "Backups" / "cortex"
        if real_dir.exists():
            staged_name = next(backup_dir.glob("cortex_*.tar.gz")).name
            assert not (real_dir / staged_name).exists()


# --- ~/.cortex sidecar state (s82) --------------------------------------
# Cortex writes non-reproducible user state OUTSIDE the project tree. This
# list has drifted twice (s76 cont.37: fitness/jobs; s82: chat.db and the
# voiceprint), and the failure is SILENT — you only find out at restore,
# when the chat opens empty. These tests pin the manifest.

# Every ~/.cortex file the backup MUST capture, with why it is not
# reproducible. Adding a feature that writes to ~/.cortex means adding it
# here and to backup.sh in the same commit.
REQUIRED_SIDECARS = {
    "code_roots.json": "the folders Pedro authorised for code",
    "chat.db":                 "the /ui conversation history — pure user content",
    "focus.db":                "focus sessions and their next steps (s86) — Pedro's day",
    "projects.db":             "projects joining conversations and Pedro's own progress",
    "effects.db":              "durable effect intentions and their transition history",
    "jobs.db":                 "Job Hunter ledger + the stored résumé",
    "fitness.db":              "real Apple Watch runs and meals",
    "contacts.db":             "the contact book",
    "costs.db":                "the cost ledger",
    "audit.jsonl":             "all observability history",
    "voiceprint.npz":          "speaker-ID enrolment — needs re-recording otherwise",
    "network_baseline.json":   "netguard's accumulated 'MACs ever seen'",
    "network_names.json":      "device nicknames",
    "network_blocklist.json":  "networks never to scan — cost an incident to build",
    "model_digests.json":      "pinned model digests for drift detection",
    "gate_queue.jsonl":        "background actions waiting for approval",
}


@pytest.fixture
def fake_home(tmp_path):
    """A stand-in $HOME whose .cortex holds one of every sidecar file."""
    home = tmp_path / "home"
    dot = home / ".cortex"
    dot.mkdir(parents=True)
    for name in REQUIRED_SIDECARS:
        if name.endswith(".db"):
            conn = sqlite3.connect(str(dot / name))
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
            conn.execute("INSERT INTO t (v) VALUES ('kept')")
            conn.commit()
            conn.close()
        else:
            (dot / name).write_bytes(b"sidecar-payload")
    clips = dot / "enroll_clips"
    clips.mkdir()
    (clips / "clip_01.wav").write_bytes(b"RIFF fake wav")
    (dot / "code_proposals").mkdir()
    (dot / "code_proposals/0123456789abcdef.json").write_text('{}')
    return home


class TestSidecarState:
    def test_code_proposals_are_captured(self, fake_cortex, backup_dir, fake_home):
        _run_backup(fake_cortex, backup_dir, home=fake_home)
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            assert "./_home_cortex/code_proposals/0123456789abcdef.json" in tar.getnames()

    def test_every_required_sidecar_is_captured(self, fake_cortex, backup_dir,
                                                fake_home):
        _run_backup(fake_cortex, backup_dir, home=fake_home)
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            names = set(tar.getnames())
        for sidecar, why in REQUIRED_SIDECARS.items():
            assert f"./_home_cortex/{sidecar}" in names, (
                f"{sidecar} missing from the backup — {why}")

    def test_enrolment_clips_are_captured(self, fake_cortex, backup_dir,
                                          fake_home):
        # The voiceprint's raw source: unrecordable without the user sitting
        # down and re-enrolling.
        _run_backup(fake_cortex, backup_dir, home=fake_home)
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            names = set(tar.getnames())
        assert "./_home_cortex/enroll_clips/clip_01.wav" in names

    def test_an_archived_conversation_survives_a_restore(self, fake_cortex,
                                                         backup_dir, tmp_path):
        """The retention pass moves old conversations OUT of the Palace into
        ~/.cortex/archive; the backup took the Palace without them and never
        the archive — a restore lost them (s88, Codex lote 6, reproduced)."""
        import gzip
        from src.memory.consolidator import archive_stale_documents
        from src.memory.palace import Palace
        home = tmp_path / "home"
        palace = Palace(db_path=str(tmp_path / "live_palace.db"))
        palace.add_document("conversation", "User: SENTINELA arquivada\nCortex: ok",
                            created_at="2025-01-01T00:00:00+00:00")
        done = archive_stale_documents(palace, days=180,
                                       archive_dir=str(home / ".cortex" / "archive"))
        palace.close()
        assert done["archived"] == 1
        _run_backup(fake_cortex, backup_dir, home=home)
        restore = tmp_path / "restore"
        restore.mkdir()
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            tar.extractall(restore, filter="data")
        [gz] = (restore / "_home_cortex" / "archive").glob("palace_conversation_*.jsonl.gz")
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            assert "SENTINELA arquivada" in f.read()

    def test_chat_db_restores_with_its_rows(self, fake_cortex, backup_dir,
                                            fake_home, tmp_path):
        # A file that merely EXISTS in the tar isn't enough — the sqlite
        # .backup path has to produce a re-openable DB with the content.
        _run_backup(fake_cortex, backup_dir, home=fake_home)
        restore = tmp_path / "restore"
        restore.mkdir()
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            tar.extractall(restore)
        conn = sqlite3.connect(str(restore / "_home_cortex" / "chat.db"))
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "kept"
        conn.close()

    def test_missing_sidecars_do_not_break_the_backup(self, fake_cortex,
                                                      backup_dir, tmp_path):
        # Fresh install: ~/.cortex may not exist at all.
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        proc = _run_backup(fake_cortex, backup_dir, home=empty_home)
        assert proc.returncode == 0, proc.stderr

    def test_backup_script_names_every_required_sidecar(self):
        """Anti-drift guard, independent of any live machine: the script's
        own text must mention each required file. Cheap, but it is exactly
        the check that would have caught both past regressions."""
        script = BACKUP_SCRIPT.read_text()
        for sidecar, why in REQUIRED_SIDECARS.items():
            assert sidecar in script, (
                f"backup.sh never mentions {sidecar} — {why}")


RESTORE_SCRIPT = BACKUP_SCRIPT.parent / "restore.sh"


class TestTheRestoreGoesBackWhereTheAppReads:
    """The documented restore extracted the tar into the project: ~/.cortex
    came back as cortex_project/_home_cortex/ and the app opened empty
    databases in a fresh ~/.cortex (s88, Codex lote 10, reproduced)."""

    def _restore(self, snap, project, home):
        env = {**os.environ, "CORTEX_SRC_DIR": str(project), "HOME": str(home),
               "CORTEX_RESTORE_SKIP_RUNNING_CHECK": "1"}
        return subprocess.run(["bash", str(RESTORE_SCRIPT), str(snap)],
                              env=env, capture_output=True, text=True)

    def test_projects_survive_a_backup_and_restore(self, fake_cortex,
                                                  backup_dir, fake_home, tmp_path):
        from src.core.projects import ProjectStore
        store = ProjectStore(str(fake_home / ".cortex" / "projects.db"))
        # Replace the generic sidecar fixture with the real schema; keep WAL open.
        (fake_home / ".cortex" / "projects.db").unlink()
        p = store.create_or_get("Horta", "Cultivar temperos fictícios")
        assert store.link_conversation("c1", p.id) and store.set_active(p.id)
        entry = store.add_entry(p.id, "feito", "Semeei", "ui", "c1")
        assert store.close_entry(entry.id)
        assert _run_backup(fake_cortex, backup_dir, home=fake_home).returncode == 0
        home = tmp_path / "restored_home"
        proc = self._restore(_latest_snapshot(backup_dir), tmp_path / "restored", home)
        assert proc.returncode == 0, proc.stderr
        restored = ProjectStore(str(home / ".cortex" / "projects.db"))
        assert restored.active() == store.get(p.id)
        assert restored.project_for_conversation("c1") == restored.active()
        assert restored.entries(p.id) == store.entries(p.id)

    def test_effects_survive_a_backup_and_restore(self, fake_cortex,
                                                 backup_dir, fake_home, tmp_path):
        from src.core.effect_store import EffectStore
        path = fake_home / ".cortex" / "effects.db"
        path.unlink()  # Replace the generic fixture; keep the real WAL open.
        store = EffectStore(str(path))
        effect = store.create("send_email", {"body": "Mensagem fictícia"},
                              {"channel": "ui"}, effect_id="staged-1")
        assert store.transition(effect.id, "preparando", "executando")
        assert store.transition(effect.id, "executando", "desconhecido", "Conferir")
        assert _run_backup(fake_cortex, backup_dir, home=fake_home).returncode == 0
        home = tmp_path / "restored_home"
        proc = self._restore(_latest_snapshot(backup_dir), tmp_path / "restored", home)
        assert proc.returncode == 0, proc.stderr
        restored = EffectStore(str(home / ".cortex" / "effects.db"))
        assert restored.get(effect.id) == store.get(effect.id)
        assert restored.history(effect.id) == store.history(effect.id)

    def test_a_fresh_home_gets_its_rows_back(self, fake_cortex, backup_dir,
                                             fake_home, tmp_path):
        assert _run_backup(fake_cortex, backup_dir, home=fake_home).returncode == 0
        project, home = tmp_path / "new_project", tmp_path / "new_home"
        proc = self._restore(_latest_snapshot(backup_dir), project, home)
        assert proc.returncode == 0, proc.stderr
        for name in REQUIRED_SIDECARS:
            assert (home / ".cortex" / name).exists(), name
        conn = sqlite3.connect(str(home / ".cortex" / "chat.db"))
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "kept"
        conn.close()
        assert (home / ".cortex" / "enroll_clips" / "clip_01.wav").exists()
        assert not (project / "_home_cortex").exists()
        conn = sqlite3.connect(str(project / "data" / "palace.db"))
        assert conn.execute("SELECT COUNT(*) FROM documentos").fetchone()[0] == 2
        conn.close()
        assert (project / ".env").exists()

    def test_a_stale_wal_is_not_replayed_over_the_snapshot(
            self, fake_cortex, backup_dir, fake_home, tmp_path):
        _run_backup(fake_cortex, backup_dir, home=fake_home)
        home = tmp_path / "old_home"
        (home / ".cortex").mkdir(parents=True)
        (home / ".cortex" / "chat.db-wal").write_bytes(b"stale")
        (home / ".cortex" / "chat.db-shm").write_bytes(b"stale")
        (home / ".cortex" / "unrelated.txt").write_text("kept")
        proc = self._restore(_latest_snapshot(backup_dir), tmp_path / "p", home)
        assert proc.returncode == 0, proc.stderr
        assert not (home / ".cortex" / "chat.db-wal").exists()
        assert not (home / ".cortex" / "chat.db-shm").exists()
        assert (home / ".cortex" / "unrelated.txt").read_text() == "kept"

    def test_it_refuses_while_the_api_runs(self, tmp_path):
        """No skip flag: if pgrep finds a uvicorn it must refuse, and a missing
        snapshot is refused either way — never a half restore."""
        env = {**os.environ, "CORTEX_SRC_DIR": str(tmp_path / "p"),
               "HOME": str(tmp_path / "h")}
        proc = subprocess.run(["bash", str(RESTORE_SCRIPT), str(tmp_path / "none.tar.gz")],
                              env=env, capture_output=True, text=True)
        assert proc.returncode != 0 and not (tmp_path / "h" / ".cortex").exists()


class TestTheProjectStateThatWasLeftOut:
    """The ledger "desfaz isso" reads, the deep-study queue, the dreams, the
    skill proposals, the briefings and the reviews were in no list: "backup
    OK", and a restore without them (s88, Codex lote 10)."""

    PATHS = ("data/action_ledger.jsonl", "data/deep_queue.jsonl",
             "data/skill_proposals.jsonl", "data/dreams/dreams.jsonl",
             "data/briefings/2026-10-18.md", "data/reviews/2026-W42.md")

    def test_they_survive_a_restore(self, fake_cortex, backup_dir, tmp_path):
        for rel in self.PATHS:
            p = fake_cortex / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"SENTINEL {rel}\n")
        assert _run_backup(fake_cortex, backup_dir).returncode == 0
        project, home = tmp_path / "restored", tmp_path / "restored_home"
        env = {**os.environ, "CORTEX_SRC_DIR": str(project), "HOME": str(home),
               "CORTEX_RESTORE_SKIP_RUNNING_CHECK": "1"}
        subprocess.run(["bash", str(RESTORE_SCRIPT), str(_latest_snapshot(backup_dir))],
                       env=env, check=True, capture_output=True)
        for rel in self.PATHS:
            assert (project / rel).read_text() == f"SENTINEL {rel}\n", rel

    def test_a_staged_action_does_not_come_back(self, fake_cortex, backup_dir):
        (fake_cortex / "data" / "pending_action.json").write_text("{}")
        _run_backup(fake_cortex, backup_dir)
        with tarfile.open(_latest_snapshot(backup_dir)) as tar:
            assert "./data/pending_action.json" not in tar.getnames()
