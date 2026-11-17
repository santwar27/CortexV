#!/usr/bin/env bash
# CORTEX audit query — latency stats per stage/event.
#
# The whole regex-first dispatch design exists to cut latency, so this
# is the view that tells you whether it's working. For every event that
# recorded a latency_ms, shows n / min / avg / max grouped by
# stage/event, sorted by slowest average first. Watch orchestrator/
# response (end-to-end) and planner/* (multi-step) here.
#
# Usage:
#   scripts/audit_queries/latency.sh              # today (UTC)
#   scripts/audit_queries/latency.sh 2026-05-27   # a specific day
#
# Reads ~/.cortex/audit.jsonl (override: CORTEX_AUDIT_LOG). UTC day.
set -euo pipefail

AUDIT_LOG="${CORTEX_AUDIT_LOG:-$HOME/.cortex/audit.jsonl}"
DAY="${1:-$(date -u +%Y-%m-%d)}"

[ -f "$AUDIT_LOG" ] || { echo "sem audit log em $AUDIT_LOG"; exit 0; }

echo "== latência por stage/event em $DAY (UTC) =="
jq -nR -r --arg day "$DAY" '
  [ inputs | fromjson? | select((.ts | startswith($day)) and .latency_ms != null) ]
  | if length == 0 then "nenhum evento com latência."
    else
      ( group_by(.stage + "/" + .event)
        | map({
            key: (.[0].stage + "/" + .[0].event),
            n:   length,
            min: ([ .[].latency_ms ] | min),
            avg: (([ .[].latency_ms ] | add) / length),
            max: ([ .[].latency_ms ] | max)
          })
        | sort_by(-.avg)
        | .[]
        | "\(.key)  n=\(.n)  min=\(.min | floor)ms  avg=\(.avg | floor)ms  max=\(.max | floor)ms"
      )
    end
' "$AUDIT_LOG"
