#!/usr/bin/env bash
# Install Cortex git hooks (symlinks, so updates in scripts/ propagate
# automatically without re-running this).
#
# Installs:
#   pre-commit — pytest + evals before each commit
#   pre-push   — pytest + evals before each push (protects origin/main,
#                the branch the relay scheme pulls from)
#
# Usage:
#   bash scripts/install-hooks.sh
#
# Idempotent: re-run any time to refresh the symlinks. Skips (with a
# warning) any hook path that already exists and ISN'T a symlink — we
# don't want to silently overwrite a custom hook the user put there by hand.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p .git/hooks

# Each hook: source under scripts/ (versioned), symlinked into .git/hooks/.
# bash 3.2 has no associative arrays, so iterate plain hook names.
HOOKS="pre-commit pre-push"
rc=0

for hook in $HOOKS; do
    src="scripts/$hook"
    dst=".git/hooks/$hook"

    if [ ! -f "$src" ]; then
        echo "[install-hooks] ERROR: $src not found"
        rc=1
        continue
    fi

    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "[install-hooks] WARNING: $dst exists and isn't a symlink — skipped."
        echo "                Move or remove it manually, then re-run."
        rc=2
        continue
    fi

    chmod +x "$src"
    ln -sf "../../$src" "$dst"
    echo "[install-hooks] ✓ $dst -> $src"
done

echo
echo "Skip a gate when truly needed:"
echo "  git commit --no-verify   /   git push --no-verify   (use sparingly)"

exit "$rc"
