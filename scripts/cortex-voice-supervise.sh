#!/bin/sh
# Cortex Voice — supervisor (s88): the KeepAlive the listener can't get
# from launchd.
#
# launchd would respawn a crashed listener, but a launchd-spawned listener
# hears a silently MUTED mic (TCC — tried three ways in May 2026, see
# launchd/com.cortex.voice.plist). So the restart loop lives here instead:
# cortex-voice-start.sh runs this script under nohup from the interactive
# Terminal, and every listener it (re)spawns inherits that Terminal's mic
# grant. Before this, an unhandled exception or a PortAudio crash left
# Cortex deaf until the user happened to notice.
#
# Policy (launchd's KeepAlive{SuccessfulExit=false} + ThrottleInterval):
#   exit 0                 → the listener chose to stop: supervisor exits
#   SIGTERM / SIGINT       → `pkill -f "src.voice.listener"` is how the user
#   (exit 143 / 130)         stops it: supervisor exits too, so the stop
#                            command in the cheatsheet keeps working
#   anything else          → crash (exception = 1, SIGABRT = 134,
#                            SIGKILL = 137, SIGSEGV = 139): restart after
#                            THROTTLE_S seconds
#   MAX_CRASHES crashes    → crash loop (bad .env, mic missing): macOS
#   within WINDOW_S          banner + give up; the ~/.zshrc hook retries
#                            on the next Terminal
#
# stdout/stderr are inherited from cortex-voice-start.sh (logs/voice.out.log
# and logs/voice.err.log, truncated once per start), so a restart appends
# after the crash's traceback instead of wiping it. Supervisor lines go to
# stderr, prefixed [supervisor].
#
# Test hooks: CORTEX_VOICE_CMD (the listener command, word-split),
# CORTEX_VOICE_THROTTLE_S, CORTEX_VOICE_MAX_CRASHES, CORTEX_VOICE_WINDOW_S,
# CORTEX_VOICE_NOTIFY_CMD (called with the message instead of osascript).

ROOT=$HOME/cortex_project
CMD=${CORTEX_VOICE_CMD:-"$ROOT/venv/bin/python -m src.voice.listener"}
THROTTLE_S=${CORTEX_VOICE_THROTTLE_S:-10}
MAX_CRASHES=${CORTEX_VOICE_MAX_CRASHES:-5}
WINDOW_S=${CORTEX_VOICE_WINDOW_S:-600}

log() {
    echo "[supervisor $(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

notify() {
    if [ -n "$CORTEX_VOICE_NOTIFY_CMD" ]; then
        "$CORTEX_VOICE_NOTIFY_CMD" "$1"
    else
        osascript -e "display notification \"$1\" with title \"Cortex\"" \
            >/dev/null 2>&1
    fi
}

crashes=0
window_start=0

while :; do
    $CMD
    rc=$?
    case $rc in
        0)
            log "listener saiu normalmente (código 0) — supervisor encerrado."
            exit 0 ;;
        130|143)
            log "listener parado por sinal (código $rc) — supervisor encerrado."
            exit 0 ;;
    esac

    now=$(date +%s)
    if [ $((now - window_start)) -gt "$WINDOW_S" ]; then
        window_start=$now
        crashes=0
    fi
    crashes=$((crashes + 1))

    if [ "$crashes" -ge "$MAX_CRASHES" ]; then
        log "listener caiu $crashes vezes em ${WINDOW_S}s (último código $rc) — desisti."
        notify "O ouvido do Cortex caiu $crashes vezes seguidas e parou. Veja logs/voice.err.log."
        exit 1
    fi
    log "listener caiu (código $rc) — reiniciando em ${THROTTLE_S}s (queda $crashes de $MAX_CRASHES)."
    sleep "$THROTTLE_S"
done
