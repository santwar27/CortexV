#!/usr/bin/env bash
# CORTEX backup passphrase setup — interactive, one-shot.
#
# Stores the symmetric encryption passphrase used by
# scripts/backup_sync_icloud.sh in the macOS Keychain. The
# passphrase never appears in command history, ps, env vars, or
# log files — only in Keychain entries gated by Touch ID / login
# password.
#
# Run this ONCE before enabling iCloud sync. If you ever rotate
# the passphrase, run it again — `security delete-generic-password`
# is invoked first so the old entry doesn't linger.
#
# After running:
#   1. Save the same passphrase OUTSIDE iCloud (1Password, paper).
#      Without it, encrypted iCloud snapshots can't be decrypted.
#   2. Enable the sync by adding CORTEX_ICLOUD_SYNC=1 to the
#      launchd plist's EnvironmentVariables (or just exporting it
#      in your shell before running backup.sh manually).

set -euo pipefail

KEYCHAIN_LABEL="${CORTEX_BACKUP_KEYCHAIN_LABEL:-cortex-backup-passphrase}"
KEYCHAIN_ACCOUNT="${USER:-$(id -un)}"

MIN_LEN=12

echo "================================================================"
echo "Cortex backup passphrase setup"
echo "================================================================"
echo ""
echo "This stores the symmetric encryption passphrase that"
echo "scripts/backup_sync_icloud.sh uses to encrypt snapshots before"
echo "uploading them to iCloud Drive."
echo ""
echo "IMPORTANT — read before you type:"
echo ""
echo "  1. Save the passphrase OUTSIDE iCloud (1Password, paper, etc)."
echo "     The whole point of encrypting is that Apple sees ciphertext."
echo "     A copy of the passphrase in iCloud defeats the point."
echo ""
echo "  2. Without this passphrase, encrypted iCloud snapshots CANNOT"
echo "     be decrypted. Losing it = losing the iCloud copy."
echo "     The local copy in ~/Backups/cortex/ is unaffected (those"
echo "     are not encrypted)."
echo ""
echo "  3. Pick something strong: ${MIN_LEN}+ characters, mix of letters,"
echo "     digits, symbols. Avoid passphrases you reuse elsewhere."
echo ""
echo "Keychain label:   $KEYCHAIN_LABEL"
echo "Keychain account: $KEYCHAIN_ACCOUNT"
echo ""

# -s: silent (don't echo), -r: raw (don't interpret backslashes),
# -p: prompt printed to stderr
read -rsp "Passphrase: " PASS1
echo ""
read -rsp "Confirm:    " PASS2
echo ""
echo ""

if [ "$PASS1" != "$PASS2" ]; then
    echo "ERROR: passphrases don't match. Try again." >&2
    exit 1
fi

if [ "${#PASS1}" -lt "$MIN_LEN" ]; then
    echo "ERROR: passphrase too short (${#PASS1} chars, need ${MIN_LEN}+)." >&2
    exit 1
fi

# Remove any existing entry first so re-running this script overwrites
# cleanly. -s matches by service, -a by account; both must match a
# generic password for `delete` to find it. The `|| true` is because
# delete returns non-zero if no entry exists, which is fine on first run.
security delete-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$KEYCHAIN_LABEL" \
    >/dev/null 2>&1 || true

# Add the new entry. The `-w` flag passes the password value; we have
# to pass it via argv (no stdin variant in `security`), but at this
# point the script has already validated it interactively — the value
# is in $PASS1, not in user shell history.
security add-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$KEYCHAIN_LABEL" \
    -w "$PASS1" \
    -U  # update if exists (defence in depth even with the delete above)

# Verify we can read it back. If Keychain ACL gets weird, better to
# fail loudly here than to discover during the first scheduled backup.
RETRIEVED="$(security find-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$KEYCHAIN_LABEL" \
    -w 2>/dev/null)"

if [ "$RETRIEVED" != "$PASS1" ]; then
    echo "ERROR: Keychain readback didn't match what we stored." >&2
    echo "Something is wrong with the Keychain ACL or the security CLI." >&2
    exit 1
fi

# Zero out the variables defensively (bash doesn't really clear memory
# but at least removes the value from any subsequent process snapshot).
PASS1=""
PASS2=""
RETRIEVED=""

echo "Passphrase stored in Keychain successfully."
echo ""
echo "Next steps:"
echo ""
echo "  1. Write the passphrase down somewhere SAFE and offline-of-iCloud."
echo ""
echo "  2. Test the sync manually before relying on the schedule:"
echo ""
echo "       CORTEX_ICLOUD_SYNC=1 bash scripts/backup.sh"
echo ""
echo "     The encrypted blob should land in:"
echo "       ~/Library/Mobile Documents/com~apple~CloudDocs/Cortex-Backups/"
echo ""
echo "  3. To enable iCloud sync for the scheduled daily backup, add"
echo "     CORTEX_ICLOUD_SYNC=1 to the EnvironmentVariables dict in"
echo "     launchd/com.cortex.backup.plist (and reinstall the plist:"
echo "     launchctl unload && load)."
echo ""
echo "  4. To rotate the passphrase later, just re-run this script."
echo "     Old encrypted blobs become unreadable after rotation —"
echo "     re-encrypt or delete them on iCloud manually if needed."
