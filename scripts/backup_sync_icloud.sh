#!/usr/bin/env bash
# CORTEX encrypted iCloud sync — second layer on top of backup.sh.
#
# Picks the most recent local snapshot from ~/Backups/cortex/
# (or $CORTEX_BACKUP_DIR), encrypts it with openssl AES-256-CBC +
# PBKDF2 using the passphrase stored in macOS Keychain, and writes
# the ciphertext to ~/Library/Mobile Documents/com~apple~CloudDocs/
# Cortex-Backups/ (the iCloud Drive path). iCloud sync then mirrors
# it to all the user's devices and to Apple's servers — but Apple
# only sees the encrypted blob.
#
# Default: not run. backup.sh invokes this only when
# CORTEX_ICLOUD_SYNC=1 is set. Opt-in so the local backup keeps
# working with zero configuration.
#
# Failures: this script exits non-zero on any problem (passphrase
# missing, openssl error, iCloud dir unwritable). backup.sh handles
# the failure gracefully — it doesn't propagate, so a temporary
# iCloud issue never compromises the local snapshot.
#
# Restore from the encrypted blob:
#   openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 \
#     -in cortex_<TS>.tar.gz.gpg \
#     -out cortex_<TS>.tar.gz \
#     -pass pass:<your passphrase>
#   tar -xzf cortex_<TS>.tar.gz -C $HOME/cortex_project/

set -euo pipefail

LOCAL_DIR="${CORTEX_BACKUP_DIR:-$HOME/Backups/cortex}"
ICLOUD_DIR="${CORTEX_ICLOUD_DIR:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/Cortex-Backups}"
KEYCHAIN_LABEL="${CORTEX_BACKUP_KEYCHAIN_LABEL:-cortex-backup-passphrase}"
# `id -un` is always available (Posix); $USER can be unset in
# stripped-env contexts (some launchd configs, tests). Fall back so
# `set -u` doesn't trip on the missing variable.
KEYCHAIN_ACCOUNT="${USER:-$(id -un)}"
PBKDF2_ITERS="${CORTEX_BACKUP_PBKDF2_ITERS:-200000}"

# ---- Resolve the passphrase ----
# Priority: CORTEX_BACKUP_PASSPHRASE env var (testing only — env vars
# can leak via ps/logs, never use in production), then Keychain.
get_passphrase() {
    if [ -n "${CORTEX_BACKUP_PASSPHRASE:-}" ]; then
        printf '%s' "$CORTEX_BACKUP_PASSPHRASE"
        return 0
    fi
    security find-generic-password \
        -a "$KEYCHAIN_ACCOUNT" \
        -s "$KEYCHAIN_LABEL" \
        -w 2>/dev/null || return 1
}

PASSPHRASE="$(get_passphrase || true)"
if [ -z "$PASSPHRASE" ]; then
    echo "ERROR: no passphrase configured." >&2
    echo "Run scripts/setup_backup_passphrase.sh to set it up, or set" >&2
    echo "CORTEX_BACKUP_PASSPHRASE env var (testing only)." >&2
    exit 1
fi

# ---- Pick the latest local snapshot ----
LATEST="$(ls -1t "$LOCAL_DIR"/cortex_*.tar.gz 2>/dev/null | head -1 || true)"
if [ -z "$LATEST" ]; then
    echo "ERROR: no snapshot in $LOCAL_DIR. Run backup.sh first." >&2
    exit 1
fi

mkdir -p "$ICLOUD_DIR"

BASE="$(basename "$LATEST")"
TARGET="$ICLOUD_DIR/$BASE.gpg"
TMP="$TARGET.tmp.$$"

# ---- Encrypt to .tmp (atomic write via mv on success) ----
# AES-256-CBC + PBKDF2 200k iterations + random salt embedded by openssl.
# -pass pass:<value> reads the passphrase from argv. This script is the
# only place the passphrase touches the process — we got it from
# Keychain (no leak path) or env var (testing only).
if ! openssl enc -aes-256-cbc -pbkdf2 -iter "$PBKDF2_ITERS" -salt \
       -in "$LATEST" \
       -out "$TMP" \
       -pass "pass:$PASSPHRASE" 2>/dev/null; then
    rm -f "$TMP"
    echo "ERROR: openssl encryption failed for $LATEST" >&2
    exit 1
fi

# ---- Self-verify: the blob must decrypt with the same passphrase ----
# Catches the case where Keychain returned a stale/corrupt entry, or
# where openssl wrote a truncated output. Decrypting to /dev/null reads
# the whole file and exits non-zero on MAC mismatch.
if ! openssl enc -d -aes-256-cbc -pbkdf2 -iter "$PBKDF2_ITERS" \
       -in "$TMP" \
       -pass "pass:$PASSPHRASE" \
       -out /dev/null 2>/dev/null; then
    rm -f "$TMP"
    echo "ERROR: encrypted blob failed self-decrypt round-trip" >&2
    exit 1
fi

# Atomic rename — iCloud Drive sees the file appear fully formed,
# never a half-written ciphertext.
mv "$TMP" "$TARGET"

# Defensive: clear the passphrase from this script's env. Doesn't
# really wipe memory but removes it from any subsequent fork's view.
PASSPHRASE=""
unset PASSPHRASE

SIZE="$(du -h "$TARGET" | cut -f1)"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) icloud sync OK $TARGET ($SIZE)"

# ---- Retention (mirror of backup.sh policy) ----
# Keep 14 most recent .tar.gz.gpg + last 8 ISO weeks (8-week cutoff
# anchored to now). Same logic as backup.sh — duplicated rather than
# extracted to avoid a bash library file just for this.
cd "$ICLOUD_DIR"

ALL=()
while IFS= read -r line; do
    ALL+=("$line")
done < <(ls -1t cortex_*.tar.gz.gpg 2>/dev/null || true)

KEEP=()

_in_keep() {
    local needle="$1" item
    for item in "${KEEP[@]:-}"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

# (a) 14 most recent
for ((i = 0; i < ${#ALL[@]} && i < 14; i++)); do
    KEEP+=("${ALL[i]}")
done

# (b) Weekly anchors within the last 8 weeks
WEEKS_SEEN=()
_week_seen() {
    local needle="$1" item
    for item in "${WEEKS_SEEN[@]:-}"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

WEEK_CUTOFF=$(($(date +%s) - 8 * 7 * 86400))

# Guard the empty-ALL case: bash 3.2 + `set -u` treats
# `"${ALL[@]}"` of a declared-but-empty array as unbound and aborts.
# Skip the whole retention block when there's nothing to consider.
if [ ${#ALL[@]} -gt 0 ]; then
    for f in "${ALL[@]}"; do
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
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) icloud retention: deleted $DELETED old blob(s)"
fi
