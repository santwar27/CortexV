"""
End-to-end tests for `scripts/backup_sync_icloud.sh`.

Same black-box approach as test_backup_script.py: stage fixtures in
tmp_path, point the script at them via env vars, run it, assert on
the output. Covers:

  - Happy path: latest local snapshot is encrypted and lands in the
    iCloud dir as cortex_<TS>.tar.gz.gpg
  - Encryption round-trip: openssl decrypt with the same passphrase
    yields a valid tar that contains expected paths
  - No passphrase configured → exit 1, nothing written
  - No local snapshot to sync → exit 1
  - Wrong passphrase → self-decrypt check fails, nothing left in iCloud
  - Atomic write: no .tmp files left behind on success or failure
  - Multiple local snapshots: only the most recent is synced
  - Retention: mirrors backup.sh policy (14 daily + 8 weekly)
  - Integration with backup.sh: CORTEX_ICLOUD_SYNC=1 triggers sync
    end-to-end; sync failure doesn't fail the parent backup

Passphrase is supplied via CORTEX_BACKUP_PASSPHRASE env var (the
documented testing-only path). The Keychain code path is exercised
by Pedro running setup_backup_passphrase.sh manually — testing it
here would mean polluting the user's real Keychain, which is the
wrong trade-off.
"""
import os
import subprocess
import tarfile
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
BACKUP_SCRIPT = REPO / "scripts" / "backup.sh"
SYNC_SCRIPT = REPO / "scripts" / "backup_sync_icloud.sh"

TEST_PASSPHRASE = "test-passphrase-with-enough-length"


# ---- Reused fixture builder ----

def _build_fake_cortex(tmp_path: Path) -> Path:
    """Minimal cortex_project layout. Mirrors what test_backup_script.py
    builds — just enough for backup.sh to produce a real tar.gz."""
    src = tmp_path / "cortex_project"
    src.mkdir()
    (src / ".env").write_text("CORTEX_API_KEY=test\n")
    data = src / "data"
    data.mkdir()
    (data / "gate_state.json").write_text("{}")
    (data / "last_announce.txt").write_text("2026-05-27")
    # Tiny but valid SQLite so `sqlite3 .backup` doesn't fail
    import sqlite3
    conn = sqlite3.connect(str(data / "palace.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    (data / "credentials").mkdir()
    (data / "credentials" / "fake_token.json").write_text("{}")
    (data / "knowledge").mkdir()
    (data / "documents").mkdir()
    logs = src / "logs"
    logs.mkdir()
    (logs / "auth.log").write_text("ok\n")
    (logs / "nightly.out.log").write_text("ok\n")
    (logs / "nightly.err.log").write_text("")
    return src


# ---- Fixtures ----

@pytest.fixture(autouse=True)
def isolated_backup_home(tmp_path, monkeypatch):
    # backup.sh also snapshots ~/.cortex; subprocesses must never inherit
    # The user's home just because the project and iCloud fixtures are isolated.
    backup_home = tmp_path / "backup_home"
    backup_home.mkdir()
    monkeypatch.setenv("HOME", str(backup_home))


@pytest.fixture
def fake_cortex(tmp_path):
    return _build_fake_cortex(tmp_path)


@pytest.fixture
def local_dir(tmp_path):
    return tmp_path / "local_backups"


@pytest.fixture
def icloud_dir(tmp_path):
    return tmp_path / "icloud"


def _seed_local_snapshot(fake_cortex: Path, local_dir: Path) -> Path:
    """Run backup.sh once and return the newest snapshot in local_dir.
    Safe to call repeatedly — useful for testing 'pick most recent'
    behaviour. Returns the just-created snapshot."""
    snaps_before = set(local_dir.glob("cortex_*.tar.gz")) if local_dir.exists() else set()
    subprocess.run(
        ["bash", str(BACKUP_SCRIPT)],
        env={
            **os.environ,
            "CORTEX_SRC_DIR": str(fake_cortex),
            "CORTEX_BACKUP_DIR": str(local_dir),
        },
        check=True,
        capture_output=True,
    )
    snaps_after = set(local_dir.glob("cortex_*.tar.gz"))
    new_snaps = snaps_after - snaps_before
    assert new_snaps, "backup.sh did not produce a new snapshot"
    # In the rare case backup.sh emitted multiple in one call (shouldn't
    # happen, but be safe), return the newest by mtime.
    return max(new_snaps, key=lambda p: p.stat().st_mtime)


def _run_sync(
    local_dir: Path,
    icloud_dir: Path,
    passphrase: str = TEST_PASSPHRASE,
    extra_env=None,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CORTEX_BACKUP_DIR": str(local_dir),
        "CORTEX_ICLOUD_DIR": str(icloud_dir),
    }
    if passphrase is not None:
        env["CORTEX_BACKUP_PASSPHRASE"] = passphrase
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )


# ---- Happy path ----

class TestHappyPath:

    def test_exit_zero_creates_gpg(self, fake_cortex, local_dir, icloud_dir):
        _seed_local_snapshot(fake_cortex, local_dir)
        result = _run_sync(local_dir, icloud_dir)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        blobs = list(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        assert len(blobs) == 1

    def test_blob_name_matches_source_with_gpg_suffix(
        self, fake_cortex, local_dir, icloud_dir
    ):
        src = _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        blob = next(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        assert blob.name == f"{src.name}.gpg"

    def test_decrypt_roundtrip_yields_original_tar(
        self, fake_cortex, local_dir, icloud_dir, tmp_path
    ):
        """The whole point: encrypt + decrypt must round-trip cleanly."""
        _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        blob = next(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        decrypted = tmp_path / "decrypted.tar.gz"
        subprocess.run(
            [
                "openssl", "enc", "-d", "-aes-256-cbc",
                "-pbkdf2", "-iter", "200000",
                "-in", str(blob),
                "-out", str(decrypted),
                "-pass", f"pass:{TEST_PASSPHRASE}",
            ],
            check=True,
        )
        # The decrypted file is a valid tar.gz that opens cleanly
        with tarfile.open(decrypted, "r:gz") as t:
            names = set(t.getnames())
        assert "./.env" in names
        assert "./data/palace.db" in names

    def test_no_tmp_files_left_behind(self, fake_cortex, local_dir, icloud_dir):
        _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        tmps = list(icloud_dir.glob("*.tmp.*"))
        assert tmps == []

    def test_picks_most_recent_when_multiple_locals(
        self, fake_cortex, local_dir, icloud_dir
    ):
        # Create three snapshots back-to-back, then verify the newest
        # is the one that gets encrypted.
        for _ in range(3):
            _seed_local_snapshot(fake_cortex, local_dir)
            # The backup script applies retention but with only 3
            # snapshots they all survive. Bump mtime to disambiguate.
            time.sleep(1.05)
        # Touch the most-recent file again to be unambiguously newest
        latest = sorted(local_dir.glob("cortex_*.tar.gz"))[-1]
        now = time.time()
        os.utime(latest, (now, now))

        _run_sync(local_dir, icloud_dir)
        blob = next(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        assert blob.name == f"{latest.name}.gpg"


# ---- Failure modes ----

class TestFailures:

    def test_no_passphrase_exits_one(self, fake_cortex, local_dir, icloud_dir):
        _seed_local_snapshot(fake_cortex, local_dir)
        # Wipe both potential passphrase sources for this invocation
        result = subprocess.run(
            ["bash", str(SYNC_SCRIPT)],
            env={
                # Strip env entirely so the Keychain lookup also fails
                # (no USER, no PATH-y stuff) — except PATH which we need.
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", str(local_dir)),
                "CORTEX_BACKUP_DIR": str(local_dir),
                "CORTEX_ICLOUD_DIR": str(icloud_dir),
                # Use a deliberately-nonexistent keychain label so the
                # find-generic-password lookup fails for sure.
                "CORTEX_BACKUP_KEYCHAIN_LABEL": "cortex-test-nonexistent-label-xyz",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "no passphrase" in result.stderr.lower()
        # iCloud dir must be untouched on early failure
        blobs = list(icloud_dir.glob("*"))
        assert blobs == []

    def test_no_local_snapshot_exits_one(self, local_dir, icloud_dir):
        local_dir.mkdir()
        result = _run_sync(local_dir, icloud_dir)
        assert result.returncode != 0
        assert "no snapshot" in result.stderr.lower()

    def test_no_local_dir_at_all_exits_one(self, tmp_path, icloud_dir):
        result = _run_sync(tmp_path / "does-not-exist", icloud_dir)
        assert result.returncode != 0

    def test_decryption_with_wrong_passphrase_fails(
        self, fake_cortex, local_dir, icloud_dir, tmp_path
    ):
        """A user trying to restore with the WRONG passphrase must never
        recover the original backup.

        We deliberately do NOT assert on openssl's exit code alone.
        AES-256-CBC decryption with a wrong key only fails
        deterministically through the PKCS7 padding check on the final
        block, and that check spuriously passes ~1/256 of the time — it
        depends on the random salt, which changes every run. When it
        passes, openssl exits 0 and writes garbage, which made the old
        exit-code assertion flaky (failed once in a pre-push run on
        2026-06-02, passed on re-run and in isolation). So we assert the
        real security property: whatever comes out is not a usable
        tar.gz, so the original backup is never recoverable with the
        wrong key — regardless of how the padding check landed."""
        _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        blob = next(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        out = tmp_path / "wrong-pass-output.tar.gz"
        subprocess.run(
            [
                "openssl", "enc", "-d", "-aes-256-cbc",
                "-pbkdf2", "-iter", "200000",
                "-in", str(blob),
                "-out", str(out),
                "-pass", "pass:wrong-passphrase",
            ],
            capture_output=True,
        )

        def _is_valid_targz(path: Path) -> bool:
            """True only if path is a tar.gz we can actually open and
            list — i.e. a restorable backup. Garbage bytes (or no file
            at all) return False."""
            if not path.exists() or path.stat().st_size == 0:
                return False
            try:
                with tarfile.open(path, "r:gz") as t:
                    t.getnames()
                return True
            except Exception:
                return False

        assert not _is_valid_targz(out), (
            "wrong passphrase yielded a valid tar.gz — backup encryption "
            "is not protecting the data"
        )


# ---- Retention ----

class TestRetention:
    """Mirrors test_backup_script.py::TestRetention. The sync script's
    retention should behave identically — 14 daily + 8 weekly, with
    the 8-week cutoff anchored to now."""

    def _seed_blobs(
        self,
        icloud_dir: Path,
        count: int,
        days_apart: int = 1,
    ):
        """Drop empty-ish .tar.gz.gpg files into icloud_dir with mtimes
        spread back in time."""
        icloud_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for i in range(count):
            ts = now - i * days_apart * 86400
            label = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(ts))
            f = icloud_dir / f"cortex_{label}_{i:03d}.tar.gz.gpg"
            f.write_bytes(b"placeholder")
            os.utime(f, (ts, ts))

    def test_under_14_all_kept(self, fake_cortex, local_dir, icloud_dir):
        self._seed_blobs(icloud_dir, count=5)
        _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        # 5 pre-existing + 1 from sync (encrypted just now) = 6
        remaining = list(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        assert len(remaining) == 6

    def test_first_time_sync_empty_icloud_no_unbound_error(
        self, fake_cortex, local_dir, icloud_dir
    ):
        """Regression: bash 3.2 + `set -u` aborted on `"${ALL[@]}"`
        when the iCloud dir had zero blobs (first sync ever). The
        encrypted blob was created but retention crashed and the
        script reported FAILED. Now the empty-array case must be
        guarded — sync exits zero and the new blob is present."""
        _seed_local_snapshot(fake_cortex, local_dir)
        # Note: icloud_dir does NOT exist yet — first-ever sync.
        result = _run_sync(local_dir, icloud_dir)
        assert result.returncode == 0, (
            f"first-time sync must not crash. stderr: {result.stderr}"
        )
        # Exactly one new blob is present, no retention chaos
        blobs = list(icloud_dir.glob("cortex_*.tar.gz.gpg"))
        assert len(blobs) == 1
        # No stderr mention of unbound (the original bug's signature)
        assert "unbound variable" not in result.stderr

    def test_ancient_outlier_deleted(self, fake_cortex, local_dir, icloud_dir):
        """A 200-day-old blob that wouldn't fit either retention layer
        must be deleted by the sync's retention pass."""
        self._seed_blobs(icloud_dir, count=30, days_apart=1)
        ancient_ts = time.time() - 200 * 86400
        ancient_label = time.strftime(
            "%Y-%m-%d_%H%M%S", time.localtime(ancient_ts)
        )
        ancient = icloud_dir / f"cortex_{ancient_label}_ancient.tar.gz.gpg"
        ancient.write_bytes(b"x")
        os.utime(ancient, (ancient_ts, ancient_ts))

        _seed_local_snapshot(fake_cortex, local_dir)
        _run_sync(local_dir, icloud_dir)
        assert not ancient.exists(), "ancient blob should have been pruned"


# ---- Integration with backup.sh hook ----

class TestBackupIntegration:
    """CORTEX_ICLOUD_SYNC=1 should trigger backup.sh's iCloud hook,
    which invokes backup_sync_icloud.sh. Sync failure must NOT cause
    the parent backup.sh to fail."""

    def _run_backup(
        self, fake_cortex, local_dir, icloud_dir,
        passphrase=TEST_PASSPHRASE, icloud_sync="1", extra=None,
    ):
        env = {
            **os.environ,
            "CORTEX_SRC_DIR": str(fake_cortex),
            "CORTEX_BACKUP_DIR": str(local_dir),
            "CORTEX_ICLOUD_DIR": str(icloud_dir),
            "CORTEX_ICLOUD_SYNC": icloud_sync,
        }
        if passphrase is not None:
            env["CORTEX_BACKUP_PASSPHRASE"] = passphrase
        if extra:
            env.update(extra)
        return subprocess.run(
            ["bash", str(BACKUP_SCRIPT)],
            env=env, capture_output=True, text=True,
        )

    def test_sync_enabled_creates_both(
        self, fake_cortex, local_dir, icloud_dir
    ):
        result = self._run_backup(fake_cortex, local_dir, icloud_dir)
        assert result.returncode == 0
        assert list(local_dir.glob("cortex_*.tar.gz"))
        assert list(icloud_dir.glob("cortex_*.tar.gz.gpg"))

    def test_sync_disabled_creates_local_only(
        self, fake_cortex, local_dir, icloud_dir
    ):
        result = self._run_backup(
            fake_cortex, local_dir, icloud_dir, icloud_sync="0",
        )
        assert result.returncode == 0
        assert list(local_dir.glob("cortex_*.tar.gz"))
        assert not icloud_dir.exists() or not list(
            icloud_dir.glob("cortex_*.tar.gz.gpg")
        )

    def test_sync_failure_does_not_fail_backup(
        self, fake_cortex, local_dir, icloud_dir
    ):
        """If iCloud sync fails (e.g. no passphrase), the local backup
        is still valid and backup.sh exits zero. The user gets a
        warning on stderr but data is safe."""
        result = self._run_backup(
            fake_cortex, local_dir, icloud_dir,
            passphrase=None,
            extra={
                "CORTEX_BACKUP_KEYCHAIN_LABEL":
                    "cortex-test-nonexistent-label-xyz",
            },
        )
        assert result.returncode == 0, (
            f"backup.sh should still exit zero. stderr: {result.stderr}"
        )
        # Local snapshot survived
        assert list(local_dir.glob("cortex_*.tar.gz"))
        # iCloud sync left no garbage
        assert not icloud_dir.exists() or not list(
            icloud_dir.glob("cortex_*.tar.gz.gpg")
        )
        # Failure was reported on stderr
        assert "icloud sync FAILED" in result.stderr
