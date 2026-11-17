#!/bin/sh
# Cortex — turn on/off the temporary recording of what you say (s84).
#
# For testing transcription only: while ON, every utterance the voice listener
# transcribes is saved as a WAV in data/voice/utterances/ (gitignored), with a
# line in utterances.jsonl saying what Whisper heard. Only the newest 50 are
# kept. Used to compare fixes (hotwords, a bigger model) on the SAME audio.
#
# It works by raising/clearing a flag FILE the running listener checks on
# every utterance (src/voice/utterance_recorder.py) — no restart. The flag is
# in /tmp, so a reboot turns recording OFF by itself.
#
#   ./scripts/cortex-record.sh          (toggles)
#   ./scripts/cortex-record.sh on       (start recording)
#   ./scripts/cortex-record.sh off      (stop recording)
#   ./scripts/cortex-record.sh status   (state + how many are saved)
#   ./scripts/cortex-record.sh clear    (stop AND delete the recordings)
FLAG="${CORTEX_RECORD_FLAG:-/tmp/cortex_record_on}"
DIR="${CORTEX_RECORD_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/voice/utterances}"

count() { ls "$DIR"/utt-*.wav 2>/dev/null | wc -l | tr -d ' '; }

action="${1:-toggle}"
case "$action" in
    toggle)
        if [ -e "$FLAG" ]; then action="off"; else action="on"; fi
        ;;
esac

case "$action" in
    on)
        # Never through a link: `: >` followed one planted in /tmp and emptied
        # the file it pointed to (s88, Codex lote 12); `touch` never truncates.
        if [ -L "$FLAG" ]; then
            echo "recusei: $FLAG é um link, não um arquivo — apague-o e tente de novo." >&2
            exit 1
        fi
        touch "$FLAG"
        echo "● gravação LIGADA — cada fala vai para $DIR ($(count) salvas até agora)."
        ;;
    off)
        rm -f "$FLAG"
        echo "○ gravação DESLIGADA ($(count) falas salvas em $DIR)."
        ;;
    status)
        if [ -e "$FLAG" ]; then state="LIGADA"; else state="DESLIGADA"; fi
        echo "gravação $state · $(count) falas salvas em $DIR"
        ;;
    clear)
        rm -f "$FLAG"
        rm -f "$DIR"/utt-*.wav "$DIR"/utterances.jsonl 2>/dev/null
        echo "○ gravação DESLIGADA e gravações apagadas."
        ;;
    *)
        echo "uso: $0 [on|off|status|clear]" >&2
        exit 2
        ;;
esac
