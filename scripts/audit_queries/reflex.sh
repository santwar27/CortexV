#!/usr/bin/env bash
# CORTEX audit query — reflex hit/miss rate.
#
# The reflex layer answers trivial inputs (greetings, time/date, ping)
# with zero LLM cost. A high hit rate means the cheap path is earning
# its keep; a falling rate hints the patterns drifted from how the user
# actually talks. Prints the hit/miss tally + hit-rate for a UTC day.
#
# Usage:
#   scripts/audit_queries/reflex.sh              # today (UTC)
#   scripts/audit_queries/reflex.sh 2026-05-27   # a specific day
#
# Reads ~/.cortex/audit.jsonl (override: CORTEX_AUDIT_LOG). UTC day.
set -euo pipefail

AUDIT_LOG="${CORTEX_AUDIT_LOG:-$HOME/.cortex/audit.jsonl}"
DAY="${1:-$(date -u +%Y-%m-%d)}"

[ -f "$AUDIT_LOG" ] || { echo "sem audit log em $AUDIT_LOG"; exit 0; }

jq -nR -r --arg day "$DAY" '
  [ inputs | fromjson? | select(.stage == "reflex" and (.ts | startswith($day))) ] as $r
  | ($r | map(select(.event == "hit"))  | length) as $h
  | ($r | map(select(.event == "miss")) | length) as $m
  | ($h + $m) as $t
  | "== reflex em \($day) (UTC) ==",
    ("hits: \($h)   miss: \($m)   total: \($t)"
      + (if $t > 0 then "   hit rate: \((100 * $h / $t) | floor)%" else "" end))
' "$AUDIT_LOG"
