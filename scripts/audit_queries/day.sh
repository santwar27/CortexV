#!/usr/bin/env bash
# CORTEX audit query — timeline of one day.
#
# Prints every audit event for a given UTC day in chronological order:
#   HH:MM:SS  <trace>  <stage>/<event>  [latency]  [payload preview]
#
# Usage:
#   scripts/audit_queries/day.sh              # today (UTC)
#   scripts/audit_queries/day.sh 2026-05-27   # a specific day
#
# Day is UTC to match the audit `ts` field and the /ui dashboard's
# count_today (both UTC). Reads ~/.cortex/audit.jsonl, overridable via
# CORTEX_AUDIT_LOG. `fromjson?` skips any half-written line so a crash
# mid-append never breaks the query.
set -euo pipefail

AUDIT_LOG="${CORTEX_AUDIT_LOG:-$HOME/.cortex/audit.jsonl}"
DAY="${1:-$(date -u +%Y-%m-%d)}"

[ -f "$AUDIT_LOG" ] || { echo "sem audit log em $AUDIT_LOG"; exit 0; }

count="$(jq -R -r --arg day "$DAY" \
  'fromjson? | select(.ts | startswith($day)) | .ts' "$AUDIT_LOG" | wc -l | tr -d ' ')"
echo "== $DAY (UTC) — $count eventos =="

jq -R -r --arg day "$DAY" '
  fromjson?
  | select(.ts | startswith($day))
  | "\(.ts[11:19])  \(.trace_id)  \(.stage)/\(.event)"
    + (if .latency_ms then "  \(.latency_ms)ms" else "" end)
    + (if .payload then "  " + (.payload | tostring | .[0:80]) else "" end)
' "$AUDIT_LOG"
