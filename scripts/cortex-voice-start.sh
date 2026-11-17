#!/bin/sh
# Cortex Voice — manual launcher (Plan Q).
#
# macOS Sequoia's TCC silently denies microphone access to ad-hoc-signed
# .app bundles, so the LaunchAgent daemon route doesn't work today. This
# script runs the listener under your interactive Terminal context, which
# DOES have a working mic grant (the one you approved during 9C testing).
#
# Run once after each Mac boot. Output is detached + logged. Stop via:
#   pkill -f "src.voice.listener"
#
# Since s88 the listener runs under scripts/cortex-voice-supervise.sh, which
# restarts it after a crash (the KeepAlive launchd can't give us — same TCC
# wall). The stop command above still stops both: the supervisor treats a
# SIGTERM'd listener as "the user stopped it" and exits with it.

cd $HOME/cortex_project || exit 1
mkdir -p logs

# Wake word: prefer the custom hey_cortex model (trained 2026-05-31 on a
# rented GPU via the briankelley pipeline). If the .onnx is present, load
# it; otherwise the resolver falls back to the built-in hey_jarvis. The
# custom model trains conservative (high precision, modest recall), so we
# run it at a lower threshold to recover recall — the ~0 false-positive
# rate leaves headroom. Override by exporting CORTEX_WAKE_THRESHOLD first.
WAKE_MODEL="$HOME/cortex_project/src/voice/models/hey_cortex.onnx"
if [ -f "$WAKE_MODEL" ]; then
    export CORTEX_WAKE_MODEL_PATH="$WAKE_MODEL"
    export CORTEX_WAKE_THRESHOLD="${CORTEX_WAKE_THRESHOLD:-0.35}"
fi

# Voice barge-in: saying "Hey Cortex" DURING a reply cuts it. The custom
# wake model is conservative (recall ~0.4), and during barge-in it has to
# fire OVER the reply playing through the speakers, so the default 0.5 is
# hard to trigger. 0.4 is more sensitive without inviting false barges;
# tune down toward 0.3 if real barges are hard, up if it self-cuts.
export CORTEX_BARGE_THRESHOLD="${CORTEX_BARGE_THRESHOLD:-0.4}"

# Refuse to start a second instance. The patterns match the real argv only:
# a bare "src.voice.listener" also matches any command naming
# src/voice/listener.py (an editor, a `git add`). A supervisor with no
# listener is either waiting to restart it (leave it) or exiting right
# after a stop (`pkill …; sleep 1; start` races it): give it a second.
LISTENER_PAT=' -m src\.voice\.listener$'
SUPERVISOR_PAT='^/bin/sh .*/cortex-voice-supervise\.sh$'
if pgrep -f "$LISTENER_PAT" >/dev/null 2>&1; then
    echo "Cortex Voice listener is already running:"
    pgrep -fl "$LISTENER_PAT"
    exit 0
fi
if pgrep -f "$SUPERVISOR_PAT" >/dev/null 2>&1; then
    sleep 1
    if pgrep -f "$SUPERVISOR_PAT" >/dev/null 2>&1; then
        echo "Cortex Voice supervisor is already running (listener restarting):"
        pgrep -fl "$SUPERVISOR_PAT"
        exit 0
    fi
fi

nohup /bin/sh $HOME/cortex_project/scripts/cortex-voice-supervise.sh \
    > logs/voice.out.log 2> logs/voice.err.log &
disown
sleep 1
echo "Cortex Voice listener started in background (supervisor PID $!)."
echo "Logs: tail -f logs/voice.err.log"
echo "Stop: pkill -f 'src.voice.listener'"
