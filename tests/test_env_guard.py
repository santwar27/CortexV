"""
Tests for the `.env` guard in `scripts/pre-commit` and `scripts/pre-push`.

The guard exists because a leaked `.env` is the cheapest catastrophic
mistake in the book (design-reference ch. 15: the book's author committed one in 2023, the key
was abused within 12 minutes). `.env` is gitignored, but `git add -f` or a
renamed copy slips past that, and `git commit --no-verify` slips past the
commit hook — so the push hook re-checks the tracked tree.

Both hooks are bash, so we test them as black boxes: build a throwaway git
repo in a tmp dir, stage/commit files, run the hook from there, and inspect
the exit code + message. The guard runs *before* the venv check, so on the
"allow" path the hook fails later for a different reason (no venv in the
sandbox) — we assert specifically that the guard did NOT fire by checking
that "BLOCKED" is absent.
"""
import subprocess
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PRE_COMMIT = SCRIPTS / "pre-commit"
PRE_PUSH = SCRIPTS / "pre-push"


@pytest.fixture
def repo(tmp_path):
    """A minimal git repo the hooks can operate on. No hooks installed and
    no venv — so the guard is the only thing that can fire before the venv
    check aborts the run."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    return tmp_path


def _run(hook, repo):
    # input="" so pre-push's `cat` (draining stdin) returns instead of hanging.
    return subprocess.run(
        ["bash", str(hook)],
        cwd=repo,
        input="",
        capture_output=True,
        text=True,
    )


def _stage(repo, name, content="SECRET=x\n"):
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    subprocess.run(["git", "add", "-f", name], cwd=repo, check=True)


def _commit(repo, name, content="SECRET=x\n"):
    _stage(repo, name, content)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=repo, check=True)


# --- pre-commit: staged set ------------------------------------------------

@pytest.mark.parametrize("name", [".env", ".env.local", "config/.env",
                                  ".env.production", "prod.env"])
def test_pre_commit_blocks_env(repo, name):
    _stage(repo, name)
    r = _run(PRE_COMMIT, repo)
    assert r.returncode != 0
    assert "BLOCKED" in r.stdout
    assert name in r.stdout


@pytest.mark.parametrize("name", [".env.example", ".env.local.example",
                                  "config.env.sample", "deploy/.env.dist",
                                  "src/api.py", ".envrc"])
def test_pre_commit_allows_non_secret(repo, name):
    _stage(repo, name, content="PLACEHOLDER=changeme\n")
    r = _run(PRE_COMMIT, repo)
    # Guard must not fire. The hook still fails later (no venv in sandbox),
    # so we assert on the guard's signal specifically, not the exit code.
    assert "BLOCKED" not in r.stdout


# --- pre-push: tracked tree at HEAD ---------------------------------------

def test_pre_push_blocks_committed_env(repo):
    _commit(repo, ".env")
    r = _run(PRE_PUSH, repo)
    assert r.returncode != 0
    assert "BLOCKED" in r.stdout
    assert ".env" in r.stdout


def test_pre_push_allows_clean_tree(repo):
    _commit(repo, "src/api.py", content="print('hi')\n")
    _commit(repo, ".env.example", content="KEY=placeholder\n")
    r = _run(PRE_PUSH, repo)
    assert "BLOCKED" not in r.stdout


# --- pre-push: the verified-tree ledger (s82) -----------------------------
# The hook used to re-run the whole suite (~7 min) on every push. Git opens
# the SSH connection BEFORE running hooks, so GitHub dropped the idle
# connection and the push died with a silent SIGPIPE. The hook now skips the
# re-run when the tree it is pushing already passed the commit gate.

LEDGER = ".git/cortex-verified-trees"


def _run_push(repo, stdin=""):
    return subprocess.run(
        ["bash", str(PRE_PUSH)],
        cwd=repo, input=stdin, capture_output=True, text=True,
    )


def _head_tree(repo):
    return subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def _push_line(repo, sha=None):
    sha = sha or subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
    return f"refs/heads/main {sha} refs/heads/main {'0' * 40}\n"


def _verify(repo, tree):
    (repo / ".git" / "cortex-verified-trees").write_text(tree + "\n")


def test_verified_tree_skips_the_rerun(repo):
    _commit(repo, "src/api.py", content="print('hi')\n")
    _verify(repo, _head_tree(repo))
    r = _run_push(repo, _push_line(repo))
    assert r.returncode == 0
    assert "already verified" in r.stdout
    assert "running pytest" not in r.stdout


def test_unverified_tree_still_runs_the_suite(repo):
    # This is the --no-verify escape hatch closing: a commit that dodged
    # pre-commit leaves no ledger entry, so the full gate still fires.
    _commit(repo, "src/api.py", content="print('hi')\n")
    r = _run_push(repo, _push_line(repo))
    assert "unverified tree" in r.stdout
    # Sandbox has no venv, so it aborts right after announcing the run.
    assert r.returncode != 0


def test_a_different_verified_tree_does_not_count(repo):
    # Guard against a sloppy "ledger is non-empty → skip".
    _commit(repo, "src/api.py", content="print('hi')\n")
    stale = _head_tree(repo)
    _commit(repo, "src/other.py", content="print('new')\n")
    _verify(repo, stale)
    r = _run_push(repo, _push_line(repo))
    assert "unverified tree" in r.stdout


def test_env_guard_fires_even_on_a_verified_tree(repo):
    # The ledger may skip the SUITE; it must never skip the secret guard.
    _commit(repo, ".env")
    _verify(repo, _head_tree(repo))
    r = _run_push(repo, _push_line(repo))
    assert r.returncode != 0
    assert "BLOCKED" in r.stdout


def test_branch_deletion_has_nothing_to_verify(repo):
    # Deleting a remote branch pushes an all-zero local sha. Falls back to
    # HEAD rather than waving anything through.
    _commit(repo, "src/api.py", content="print('hi')\n")
    _verify(repo, _head_tree(repo))
    r = _run_push(repo, f"refs/heads/main {'0' * 40} refs/heads/main {'0' * 40}\n")
    assert r.returncode == 0


def test_empty_stdin_fails_safe_to_head(repo):
    # If we cannot tell what is being pushed, verify HEAD — never skip.
    _commit(repo, "src/api.py", content="print('hi')\n")
    r = _run_push(repo, "")
    assert "unverified tree" in r.stdout    # HEAD is not in the ledger
    _verify(repo, _head_tree(repo))
    r = _run_push(repo, "")
    assert "already verified" in r.stdout
