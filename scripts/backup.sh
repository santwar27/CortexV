#!/usr/bin/env bash
# CORTEX daily backup — local snapshot of mutable state.
#
# Captures the on-disk state that isn't reproducible from git or
# `pip install -r requirements.txt`: the Palace SQLite DB, OAuth
# credentials, gate/lease state, study queue, local knowledge base,
# user documents, the .env, and the audit/nightly log history.
#
# What we DELIBERATELY skip:
#   - venv/, __pycache__/, *.pyc, .pytest_cache/ — recreable
#   - data/voice/ — gitignored, large, recreable (re-download models)
#   - data/palace.db-shm, data/palace.db-wal — WAL helpers, re-created
#     on next SQLite open. We snapshot the .db itself via `sqlite3
#     .backup` which produces a consistent file without stopping
#     uvicorn.
#   - logs/uvicorn.{out,err}.log — runtime noise, not state.
#   - data/backups/ — avoid backup-of-backup recursion.
#   - .git/ — already on GitHub.
#   - data/pending_action.json, data/last_seen.json, data/nudged.json —
#     in-flight state: a staged action from days ago must not come back
#     to be confirmed, and the digest/nudge markers re-arm themselves.
#
# Restore: scripts/restore.sh <snapshot> — with the API stopped. It puts
# the project half back in the project and _home_cortex/ back in
# ~/.cortex. Extracting the tar into the project, the instruction that
# stood here, left the ~/.cortex half in cortex_project/_home_cortex/
# and the app on empty databases (s88, Codex lote 10).
#
# Retention policy: keep the 14 most recent snapshots regardless,
# plus the most recent snapshot from each of the last 8 ISO weeks.
# Anything older or not week-anchored is deleted.
#
# Scheduled via launchd/com.cortex.backup.plist (daily, 03:00 local).
# Safe to run manually any time. macOS BSD tar compatible — no GNU
# tar extensions used.

set -euo pipefail

# Resolve our own directory NOW, before any `cd` later in the script
# changes CWD. Used at the end to locate the optional iCloud sync
# helper next to us. Works regardless of how we were invoked (relative
# path, absolute path, symlink, launchd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${CORTEX_SRC_DIR:-$HOME/cortex_project}"
DEST="${CORTEX_BACKUP_DIR:-$HOME/Backups/cortex}"
TS="$(date +%Y-%m-%d_%H%M%S)"
SNAPSHOT="$DEST/cortex_$TS.tar.gz"

mkdir -p "$DEST"

# Whitelist of paths (relative to $SRC) that go into the snapshot.
# Explicit list, not exclude-based — anything new in the tree stays
# out until added here on purpose.
INCLUDE_PATHS=(
    ".env"
    "data/credentials"
    "data/gate_state.json"
    "data/last_announce.txt"
    "data/study_queue.jsonl"
    "data/knowledge"
    "data/documents"
    "logs/auth.log"
    "logs/nightly.out.log"
    "logs/nightly.err.log"
    # s88 (Codex lote 10): the action ledger is what "desfaz isso" reads — a
    # restore without it can't undo the event it just made; the deep-study
    # queue, the dreams and the skill proposals wait on the user's decision; the
    # briefings and weekly reviews are his history, regenerable from nothing.
    "data/action_ledger.jsonl"
    "data/deep_queue.jsonl"
    "data/skill_proposals.jsonl"
    "data/dreams"
    "data/briefings"
    "data/reviews"
)

TMPDIR="$(mktemp -d -t cortex-backup.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT
STAGE="$TMPDIR/stage"
mkdir -p "$STAGE"

# ---- Palace SQLite consistent snapshot ----
# `cp data/palace.db` while uvicorn is writing would catch the file
# mid-transaction. `sqlite3 .backup` is the official live-snapshot API:
# it follows the same locking protocol as a normal read transaction
# and produces a fully consistent .db file.
if [ -f "$SRC/data/palace.db" ]; then
    mkdir -p "$STAGE/data"
    sqlite3 "$SRC/data/palace.db" ".backup '$STAGE/data/palace.db'"
fi

# ---- Stage each whitelisted path (only if it exists) ----
# Missing paths are skipped silently. the user's setup may not have every
# optional file yet (e.g. study_queue.jsonl only exists after the
# first study).
for path in "${INCLUDE_PATHS[@]}"; do
    if [ -e "$SRC/$path" ]; then
        parent="$(dirname "$path")"
        if [ "$parent" != "." ]; then
            mkdir -p "$STAGE/$parent"
        fi
        # -R for directories, also handles files transparently.
        # -p preserves modes/times so audit logs keep their original
        # mtime once restored.
        cp -Rp "$SRC/$path" "$STAGE/$path"
    fi
done

# ---- Stage ~/.cortex/ (sidecar state) ----
# Cortex keeps a few SQLite/JSONL files OUTSIDE the project tree so
# uninstall/reinstall doesn't blow them away: audit log, cost ledger,
# contact book. The backup must capture them too — otherwise a restore
# would silently lose all observability + contacts (sessão 21 gap).
#
# ⚠️ This list has drifted TWICE now (s76 cont.37 missed fitness/jobs; s82
# missed chat.db + the voiceprint). Whenever a feature starts writing to
# ~/.cortex, add it HERE in the same commit — `make backup-check` fails on
# any file in ~/.cortex that is neither captured nor explicitly ignored.
DOT_CORTEX="$HOME/.cortex"
if [ -d "$DOT_CORTEX" ]; then
    mkdir -p "$STAGE/_home_cortex"
    # SQLite ledgers — use .backup for live-consistent snapshot, same
    # as palace.db above (sqlite3's official live-snapshot API).
    # fitness.db (real Apple Watch runs + meals) and jobs.db (Job Hunter
    # + resume) are NOT reproducible — must be captured (s76 cont.37 gap).
    # chat.db (the /ui conversation history, s81) is the newest of these and
    # was the s82 gap: it is pure user content, not reproducible from
    # anything, and a restore without it silently opens an EMPTY chat.
    # focus.db (s86) — the focus sessions and their "next step" notes: what
    # "onde parei?" answers from. Pure user content, same as chat.db.
    # projects.db — conversations joined by project and the user's own progress.
    # effects.db — durable effect intentions and their transition history.
    for db in costs.db contacts.db fitness.db jobs.db chat.db focus.db projects.db effects.db; do
        if [ -f "$DOT_CORTEX/$db" ]; then
            sqlite3 "$DOT_CORTEX/$db" ".backup '$STAGE/_home_cortex/$db'"
        fi
    done
    # Append-only / non-DB files — plain cp.
    #   voiceprint.npz — the enrolled speaker-ID print. NOT reproducible
    #     without the user re-recording the enrolment (s82 gap).
    #   network_*.json — the netguard's accumulated baseline, the device
    #     nicknames, and the blocklist (which cost a real incident to build).
    #   model_digests.json — pinned model digests for drift detection.
    #   gate_queue.jsonl — the background actions waiting for approval
    #     ("make gate"); lost, they are lost silently (s88, Codex lote 10).
    #   code_roots.json — the folders the user authorised for code.
    for f in audit.jsonl gate_queue.jsonl voiceprint.npz model_digests.json code_roots.json \
             network_baseline.json network_names.json network_blocklist.json; do
        if [ -f "$DOT_CORTEX/$f" ]; then
            cp -p "$DOT_CORTEX/$f" "$STAGE/_home_cortex/$f"
        fi
    done
    # The raw enrolment clips (~1MB) regenerate the voiceprint if the .npz
    # format ever changes — cheap insurance for something unrecordable.
    if [ -d "$DOT_CORTEX/enroll_clips" ]; then
        cp -Rp "$DOT_CORTEX/enroll_clips" "$STAGE/_home_cortex/enroll_clips"
    fi
    # Saved code proposals are the user's reviewable work state.
    if [ -d "$DOT_CORTEX/code_proposals" ]; then
        cp -Rp "$DOT_CORTEX/code_proposals" "$STAGE/_home_cortex/code_proposals"
    fi
    # archive/ — the retention pass (s72) moves conversations older than
    # 180 days, never used, OUT of the Palace into gzip JSONL here. The
    # Palace in this snapshot no longer holds them: without this copy a
    # restore loses them for good (s88, Codex lote 6).
    if [ -d "$DOT_CORTEX/archive" ]; then
        cp -Rp "$DOT_CORTEX/archive" "$STAGE/_home_cortex/archive"
    fi
fi

# ---- Build the tar ----
# -C $STAGE + "." means "archive everything under STAGE with paths
# relative to STAGE", so the resulting tar extracts cleanly back into
# cortex_project/ regardless of where it was created.
tar -czf "$SNAPSHOT" -C "$STAGE" .

# Verify the archive isn't truncated. tar -tzf reads through the
# whole thing and exits non-zero on corruption.
if ! tar -tzf "$SNAPSHOT" >/dev/null 2>&1; then
    echo "ERROR: snapshot at $SNAPSHOT failed integrity check" >&2
    rm -f "$SNAPSHOT"
    exit 1
fi

SIZE="$(du -h "$SNAPSHOT" | cut -f1)"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) backup OK $SNAPSHOT ($SIZE)"

# ---- Retention ----
# Keep:
#   (a) the 14 most recent snapshots, by mtime
#   (b) the most recent snapshot from each of the last 8 ISO weeks
#       (covers ~2 months of weekly history beyond the 14 daily)
# Anything not in those two sets gets deleted.
#
# Implementation note: macOS ships bash 3.2 which has no associative
# arrays. We use parallel indexed arrays + linear membership tests.
# For ~30 snapshots max this is fast enough.
cd "$DEST"

# Collect all snapshots, newest first.
ALL=()
while IFS= read -r line; do
    ALL+=("$line")
done < <(ls -1t cortex_*.tar.gz 2>/dev/null || true)

# Build KEEP list (indexed array of filenames).
KEEP=()

# Tiny helper: is "$1" already in the KEEP array?
_in_keep() {
    local needle="$1" item
    for item in "${KEEP[@]:-}"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

# (a) Keep the first 14 (most recent).
for ((i = 0; i < ${#ALL[@]} && i < 14; i++)); do
    KEEP+=("${ALL[i]}")
done

# (b) For each of the last 8 ISO weeks (anchored to NOW, not to
# whatever weeks happen to appear in the snapshot list), pick the
# most recent snapshot whose mtime falls inside that week. Snapshots
# older than the 8-week cutoff don't get weekly anchor protection
# even if they're the only one from their week — they're old enough
# that they should fall off retention entirely.
WEEKS_SEEN=()
_week_seen() {
    local needle="$1" item
    for item in "${WEEKS_SEEN[@]:-}"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

# 8 weeks back from now, in epoch seconds.
WEEK_CUTOFF=$(($(date +%s) - 8 * 7 * 86400))

# Guard the empty-ALL case: bash 3.2 + `set -u` treats
# `"${ALL[@]}"` of a declared-but-empty array as unbound and aborts.
# Skip retention when there's nothing to consider. (Latent bug here
# never fired because backup.sh always creates one snapshot before
# this block runs — but the iCloud sync hit it on its first run.)
if [ ${#ALL[@]} -gt 0 ]; then
    for f in "${ALL[@]}"; do
        # File mtime → ISO week. macOS BSD `date` uses -r <timestamp>;
        # GNU `date` uses -d "@<timestamp>". Try BSD first, fall through.
        MTIME="$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f")"
        if [ "$MTIME" -lt "$WEEK_CUTOFF" ]; then
            continue
        fi
        WEEK="$(date -r "$MTIME" +%G-W%V 2>/dev/null || date -d "@$MTIME" +%G-W%V)"
        if ! _week_seen "$WEEK"; then
            WEEKS_SEEN+=("$WEEK")
            if ! _in_keep "$f"; then
                KEEP+=("$f")
            fi
        fi
    done
fi

# Delete everything not in KEEP.
DELETED=0
if [ ${#ALL[@]} -gt 0 ]; then
    for f in "${ALL[@]}"; do
        if ! _in_keep "$f"; then
            rm -f "$f"
            DELETED=$((DELETED + 1))
        fi
    done
fi

if [ "$DELETED" -gt 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) retention: deleted $DELETED old snapshot(s)"
fi

# ---- Optional: encrypted iCloud sync ----
# Off by default. Set CORTEX_ICLOUD_SYNC=1 in the env (or in the
# launchd plist's EnvironmentVariables dict) to enable. The local
# snapshot above is the source of truth — if iCloud sync fails for
# any reason (no passphrase, no network, openssl error), we log it
# and exit zero so the local backup isn't reported as a failure.
if [ "${CORTEX_ICLOUD_SYNC:-0}" = "1" ]; then
    # SCRIPT_DIR was captured at the top of this file, before the
    # retention `cd "$DEST"` changed CWD. So this works even after
    # we wandered off to the backup directory.
    SYNC_SCRIPT="$SCRIPT_DIR/backup_sync_icloud.sh"
    if [ -x "$SYNC_SCRIPT" ]; then
        if ! "$SYNC_SCRIPT"; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) icloud sync FAILED (local backup OK)" >&2
        fi
    else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) icloud sync skipped: $SYNC_SCRIPT not executable" >&2
    fi
fi
