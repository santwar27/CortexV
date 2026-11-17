#!/bin/sh
# Cortex Voice — Login Item launcher.
#
# macOS opens Terminal.app to execute .command files. We use that as a
# bridge so the voice listener inherits the user-session TCC (microphone)
# permission that Terminal already has — the same trick that makes the
# manual Plan Q work, but automated at login time.
#
# Flow:
#   1. Pedro logs in to macOS
#   2. Login Items launches this .command file via Terminal
#   3. Terminal runs cortex-voice-start.sh which spawns the listener
#      with nohup+disown (detached from this shell)
#   4. We exit; Terminal closes (if "Close on exit" is set)
#   5. Listener keeps running in background, mic permission active
#
# Manual one-time setup:
#   System Settings → General → Login Items & Extensions
#     → Open at Login → "+" → select this file
#   Terminal → Settings → Profiles → [default] → Shell
#     → "When the shell exits" → "Close if the shell exited cleanly"

$HOME/cortex_project/scripts/cortex-voice-start.sh

# Give the background process a beat to fully detach before this shell ends.
sleep 1
exit 0
