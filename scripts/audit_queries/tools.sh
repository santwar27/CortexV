#!/usr/bin/env bash
# CORTEX audit query — tool dispatch tally.
#
# Tools are the only events that mutate state, so this is the "what did
# the agent actually DO" view. Groups stage=="tool" events by tool name
# and shows count, risk level (S1/S2/S3), and the split between direct
# LLM tool-calls (event "dispatch") and steps run inside an approved
# plan (event "dispatch_in_plan"). The risk column surfaces every S3
# (irreversible) action that fired.
#
# Usage:
#   scripts/audit_queries/tools.sh              # today (UTC)
#   scripts/audit_queries/tools.sh 2026-05-27   # a specific day
#
# Reads ~/.cortex/audit.jsonl (override: CORTEX_AUDIT_LOG). UTC day.
set -euo pipefail

AUDIT_LOG="${CORTEX_AUDIT_LOG:-$HOME/.cortex/audit.jsonl}"
DAY="${1:-$(date -u +%Y-%m-%d)}"

[ -f "$AUDIT_LOG" ] || { echo "sem audit log em $AUDIT_LOG"; exit 0; }

echo "== tool dispatches em $DAY (UTC) =="
jq -nR -r --arg day "$DAY" '
  [ inputs | fromjson? | select(.stage == "tool" and (.ts | startswith($day))) ]
  | if length == 0 then "nenhum tool dispatch."
    else
      ( group_by(.payload.name)
        | map({
            tool:    (.[0].payload.name // "?"),
            risk:    (.[0].payload.risk // "?"),
            total:   length,
            direct:  ([ .[] | select(.event == "dispatch") ] | length),
            in_plan: ([ .[] | select(.event == "dispatch_in_plan") ] | length)
          })
        | sort_by(-.total)
        | .[]
        | "\(.total)x  \(.tool)  [\(.risk)]  (direct \(.direct), in-plan \(.in_plan))"
      )
    end
' "$AUDIT_LOG"
