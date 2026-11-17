#!/bin/sh
# Cortex — stop/start the Cortex from reading your screen (inovação #3).
#
# The screen-awareness commands ("responde esse email", "resume isso", "o que
# é isso?") read the ACTIVE window only at the instant you say them — never
# continuously. This script is the manual on/off button on top of that: while
# it's OFF, those commands refuse to read anything at all.
#
# It works by raising/clearing a flag FILE that the RUNNING Cortex checks live
# (screen_context_mixin._screen_context_enabled) — so the state flips with NO
# uvicorn/listener restart. Same mechanism as the mute button (cortex-shush.sh
# / /tmp/cortex_silence).
#
# Two ways to use it:
#   1. Run it from any terminal:  ./scripts/cortex-eyes.sh        (toggles)
#                                 ./scripts/cortex-eyes.sh off     (force off)
#                                 ./scripts/cortex-eyes.sh on      (force on)
#                                 ./scripts/cortex-eyes.sh status  (just report)
#   2. Bind the toggle to a global hotkey (recommended) so one keypress
#      stops/starts the Cortex seeing your screen from anywhere:
#        Shortcuts.app → new Shortcut → "Run Shell Script" action →
#        point it at this script → assign a keyboard shortcut.
#
# This is OUR button — separate from (and on top of) the macOS TCC grant
# you can also revoke in System Settings → Privacy & Security.
FLAG="${CORTEX_SCREEN_OFF_FLAG:-/tmp/cortex_screen_off}"

action="${1:-toggle}"
case "$action" in
    toggle)
        if [ -e "$FLAG" ]; then action="on"; else action="off"; fi
        ;;
esac

case "$action" in
    off)
        # Never through a link: `: >` followed one planted in /tmp and emptied
        # the file it pointed to (s88, Codex lote 12); `touch` never truncates.
        if [ -L "$FLAG" ]; then
            echo "recusei: $FLAG é um link, não um arquivo — apague-o e tente de novo." >&2
            exit 1
        fi
        touch "$FLAG" 2>/dev/null || true
        echo "🙈 leitura de tela DESLIGADA — os comandos do #3 vão recusar."
        ;;
    on)
        rm -f "$FLAG" 2>/dev/null || true
        echo "👀 leitura de tela LIGADA — os comandos do #3 voltam a funcionar."
        ;;
    status)
        if [ -e "$FLAG" ]; then
            echo "🙈 leitura de tela está DESLIGADA (flag: $FLAG)."
        else
            echo "👀 leitura de tela está LIGADA."
        fi
        ;;
    *)
        echo "uso: $0 [toggle|on|off|status]" >&2
        exit 2
        ;;
esac
