#!/usr/bin/env bash
# CORTEX restore — put a backup.sh snapshot back where the app reads it.
#
#   bash scripts/restore.sh ~/Backups/cortex/cortex_YYYY-MM-DD_HHMMSS.tar.gz
#
# The snapshot holds two places: the project's state (data/, .env, logs/)
# and, under _home_cortex/, what lives in ~/.cortex (chat, focus, fitness,
# jobs, contacts, costs, audit, archive, voiceprint, netguard files). The
# old instruction — extract into the project — left the second half in
# cortex_project/_home_cortex/ while the app opened EMPTY databases in
# ~/.cortex (s88, Codex lote 10, reproduced).
#
# Stop everything that writes first: a database overwritten under an open
# connection is corrupted, not restored.
#   launchctl unload ~/Library/LaunchAgents/com.cortex.uvicorn.plist
#   (and close any `cortex` terminal)
# Then run this, then load the agent again.
#
# A stale -wal/-shm next to a restored database would be replayed over the
# snapshot on the next open, so they are removed. Files the snapshot does
# not hold are left alone.

set -euo pipefail

SNAP="${1:?usage: restore.sh <snapshot.tar.gz>}"
SRC="${CORTEX_SRC_DIR:-$HOME/cortex_project}"
DOT_CORTEX="$HOME/.cortex"

if [ ! -f "$SNAP" ]; then
    echo "ERROR: no snapshot at $SNAP" >&2
    exit 1
fi
if [ "${CORTEX_RESTORE_SKIP_RUNNING_CHECK:-0}" != "1" ] \
        && pgrep -f "uvicorn src.api" >/dev/null 2>&1; then
    echo "ERROR: the Cortex API is running — stop it first:" >&2
    echo "  launchctl unload ~/Library/LaunchAgents/com.cortex.uvicorn.plist" >&2
    exit 1
fi

TMPDIR="$(mktemp -d -t cortex-restore.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT
tar -xzf "$SNAP" -C "$TMPDIR"

drop_sqlite_helpers() {  # $1 = directory restored into, $2 = staged directory
    local db
    for db in "$2"/*.db; do
        [ -e "$db" ] || continue
        rm -f "$1/$(basename "$db")-wal" "$1/$(basename "$db")-shm"
    done
}

# ~/.cortex first, out of the staged tree, so the project copy below never
# carries an _home_cortex/ into the project.
if [ -d "$TMPDIR/_home_cortex" ]; then
    mkdir -p "$DOT_CORTEX"
    drop_sqlite_helpers "$DOT_CORTEX" "$TMPDIR/_home_cortex"
    cp -Rp "$TMPDIR/_home_cortex/." "$DOT_CORTEX/"
    rm -rf "$TMPDIR/_home_cortex"
fi

mkdir -p "$SRC"
if [ -d "$TMPDIR/data" ]; then
    mkdir -p "$SRC/data"
    drop_sqlite_helpers "$SRC/data" "$TMPDIR/data"
fi
cp -Rp "$TMPDIR/." "$SRC/"

echo "restored $SNAP → $SRC and $DOT_CORTEX"
