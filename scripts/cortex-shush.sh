#!/bin/sh
# Cortex — stop the assistant talking NOW (manual "mute button").
#
# The voice listener speaks replies via Piper (synth to a WAV, played by
# `afplay /tmp/cortex_tts.wav`) with macOS `say` as the fallback engine.
# Voice barge-in exists (say the wake word mid-reply), but this script is
# the manual instant kill: it stops the current speech from a hotkey.
#
# Two ways to use it:
#   1. Run it from any terminal:  ./scripts/cortex-shush.sh
#   2. Bind it to a global hotkey (recommended) so one keypress silences
#      Cortex from anywhere:
#        Shortcuts.app → new Shortcut → "Run Shell Script" action →
#        point it at this script → assign a keyboard shortcut.
#
# Two-part stop so it's a real "stop", not just "skip this sentence":
#   1. kill the playback — afplay on the Cortex WAV (Piper engine) and/or
#      `say` (fallback engine) — cuts the audio playing RIGHT NOW. The
#      afplay match is pinned to /tmp/cortex_tts.wav so other afplay
#      processes on the machine are untouched.
#   2. raise the silence flag — the listener's stream_and_speak() checks
#      /tmp/cortex_silence and aborts the REST of the reply (stops
#      buffering + speaking further sentences). The listener consumes the
#      flag when it sees it.
#
# Without step 2, a long multi-sentence reply still streaming would just
# move to the next buffered sentence after the kill. With it, the whole
# reply is cut.
SILENCE_FLAG="${CORTEX_SILENCE_FLAG:-/tmp/cortex_silence}"

# Raise the abort flag first so an in-flight stream stops buffering even
# if nothing is mid-sentence yet. Never through a link: `: >` followed one
# planted in /tmp and emptied the file it pointed to (s88, Codex lote 12);
# `touch` never truncates.
[ -L "$SILENCE_FLAG" ] || touch "$SILENCE_FLAG" 2>/dev/null || true

stopped=0
# The listener's temp WAVs (cortex_tts_*.wav under $TMPDIR) — the old
# /tmp/cortex_tts.wav no longer exists (s88, Codex lote 12). Same pattern as
# src/core/system_control_mixin.py CORTEX_AFPLAY_PATTERN.
pkill -f '^afplay .*/cortex_tts_[^/]*\.wav$' 2>/dev/null && stopped=1
killall say 2>/dev/null && stopped=1

if [ "$stopped" -eq 1 ]; then
    echo "shh — stopped."
else
    # Not mid-sentence, but the flag still aborts a reply that's
    # streaming-but-not-yet-speaking, so this isn't a no-op.
    echo "(nothing speaking now — abort flag set for any in-flight reply)"
fi
