# Cortex — entry-point targets. Each target maps to one command
# you'd otherwise type by hand, with the venv path baked in.
#
# Usage:
#   make test       # pytest suite (~20s)
#   make eval       # deterministic behavioural evals (<1s, offline)
#   make eval-llm   # LLM behaviour evals — REAL Ollama calls (opt-in, slow)
#   make eval-llm-rebaseline  # re-record the LLM baseline for this model
#   make check      # test + eval — same gate as the commit/push hooks
#   make hooks      # install git hooks (pre-commit + pre-push, idempotent)
#   make smoke      # quick local /ui health check (server must be up)
#   make doctor     # full local-stack health check (Ollama, Palace, voice…)
#
# `eval-llm` is deliberately OUT of `check`: it needs Ollama up, takes
# seconds per case, and is drift-tolerant (DRIFT ≠ failure). Override
# the model with CORTEX_MODEL=cortex-gemma make eval-llm. With a baseline
# recorded it also reports cosine drift vs the stored replies.
#
# Targets are .PHONY because we never produce files with these names.

.PHONY: test eval eval-llm eval-llm-rebaseline check hooks smoke doctor gate weekly policy promoter arbiter-report latency-report memory-doctor ledger heartbeat daily nudge jobs contacts handoff-check verify-launchd help

PY := venv/bin/python

help:
	@echo "Cortex make targets:"
	@echo "  test     — pytest suite"
	@echo "  eval     — deterministic behavioural evals (offline)"
	@echo "  eval-llm — LLM behaviour evals (real Ollama, opt-in, slow)"
	@echo "  eval-llm-rebaseline — re-record the LLM baseline (this model)"
	@echo "  check    — test + eval (same gate as commit/push hooks)"
	@echo "  hooks    — install git hooks (pre-commit + pre-push)"
	@echo "  smoke    — curl /ui to verify uvicorn alive"
	@echo "  doctor   — full local-stack health check (Ollama, Palace, voice…)"
	@echo "  gate     — list background actions awaiting approval (gate queue)"
	@echo "  weekly   — write this week's review digest (data/reviews/)"
	@echo "  policy   — suggest gate auto-approve policies from approval history"
	@echo "  promoter — suggest skills from repeated workflows in the audit log"
	@echo "  arbiter-report — route distribution + unsure cases from the shadow classifier"
	@echo "  latency-report — per-route turn latency (p50/p95/max) from the audit log"
	@echo "  memory-doctor — scan the Palace for memory-poisoning patterns (read-only)"
	@echo "  ledger   — show the append-only journal of side-effecting actions"
	@echo "  heartbeat — liveness watchdog for the scheduled jobs (studier/backup/weekly)"
	@echo "  daily    — print today's Daily Driver briefing (the 08h job speaks it)"
	@echo "  nudge    — print what the proactive nudge engine would fire now (15-min job notifies)"
	@echo "  jobs     — list your top-ranked apprenticeship matches (with links; read-only)"
	@echo "  handoff-check — protocol lint: JOURNAL lag + stale suite count (A/B relay)"
	@echo "  verify-launchd — diff repo launchd/ vs loaded ~/Library/LaunchAgents plists"
	@echo "  backup-check — every ~/.cortex file is captured by backup.sh or ignored on purpose"
	@echo "  contacts — list the contact book (add with scripts/add_contact.py)"

test:
	@$(PY) -m pytest -q

eval:
	@$(PY) -m evals

# Real LLM calls via engine.chat(). Needs Ollama running. NOT part of
# `check` — drift-tolerant by design (DRIFT is visible, not fatal).
eval-llm:
	@$(PY) -m evals --llm-only

# Re-record the baseline replies for the active (model, prompt_version).
# Deliberate + opt-in: a baseline says "these answers are correct now".
# Writes evals/baselines/<model>__v<N>.json — commit the diff.
eval-llm-rebaseline:
	@$(PY) -m evals --llm-rebaseline

# `check` runs the same gate the pre-commit hook runs. Use this
# locally when you want to verify before staging instead of waiting
# for the hook to do it at commit time.
check: test eval

hooks:
	@bash scripts/install-hooks.sh

smoke:
	@curl -s -o /dev/null -w "code=%%{http_code}\n" http://127.0.0.1:8000/ui

# Full health check of the local stack — Ollama reachability + pulled
# models, Palace integrity, a live embeddings round-trip, voice model +
# daemon, and the API server. Exit 1 only on a FAIL (a WARN is degraded
# but usable). Handy first move when something feels off.
doctor:
	@$(PY) -m src.doctor

# Review S3 actions queued by background work (nightly task / Dream Loop)
# awaiting your approval. `make gate` lists; approve/reject take an id:
#   venv/bin/python -m src.gate_review approve <id>
gate:
	@$(PY) -m src.gate_review

# Write this week's review digest from the audit log into data/reviews/.
# Runs automatically every Sunday via launchd/com.cortex.weekly.plist;
# this target is the manual / on-demand path. Add --stdout to just print.
weekly:
	@$(PY) -m src.weekly_review

# Suggest gate auto-approve policies from the inline-approval history in the
# audit log. Read-only: it only surfaces
# evidence; you decide whether to add a rule to src/core/gate_policy.py.
policy:
	@$(PY) -m src.core.policy_detector

# Suggest skills from repeated multi-step workflows in the audit log (the reference
# 4.2 skill auto-promoter). Read-only by default: it only surfaces evidence.
# Add --save to file pending proposals; you build the skill by hand for now
# (code generation is dormant until Gemma 4). Use `make promoter`.
promoter:
	@$(PY) -m src.core.skill_promoter

# Read the shadow arbiter's classifications from the audit log:
# route distribution (REFLEX/LOOKUP/MEMORY/SIMPLES/COMPLEXO/LONGO) + the
# low-confidence cases the regex was unsure about. Read-only calibration for
# CORTEX_ARBITER_CASCADE before the Mac Studio; fix markers in arbiter.py.
arbiter-report:
	@$(PY) -m src.core.arbiter_report

# Per-route turn latency (p50/p95/max by response mode) from the audit log.
# Read-only: surfaces the slow paths to investigate (voice latency is a known
# complaint). Does not add hot-path instrumentation — reads what's logged.
latency-report:
	@$(PY) -m src.core.latency_report

# Proactive anti-poisoning sweep of the Palace: stored refusals, engine-error
# echoes, role-confusion turns (s40), contradictory active facts, and exact
# duplicates. Read-only — it only flags; you remove findings by hand
# (palace.delete_document / invalidate_fact). Complements the reactive write
# guards in src/core/memory.py.
memory-doctor:
	@$(PY) -m src.core.memory_doctor

# Show the append-only journal of side-effecting actions (email/event/reminder)
# with risk level + reverse spec. Read-only audit trail; the foundation for a
# future "Cortex, desfaz isso". Stored in data/action_ledger.jsonl.
ledger:
	@$(PY) -m src.core.action_ledger

# Liveness watchdog for the fixed-cadence jobs (studier/backup/weekly): flags
# any that's been silent past its window, by log/artefact mtime. Read-only
# print here; the launchd job (launchd/com.cortex.heartbeat.plist) runs it
# daily with --notify to post a macOS banner. Generalises the studier
# missed-run alarm.
heartbeat:
	@$(PY) -m src.core.heartbeat

# Print today's Daily Driver briefing (agenda / unread / Studier / approvals).
# The launchd job (launchd/com.cortex.daily.plist, 08h) is notification-only
# (banner + data/briefings/<date>.md — no voice, the user's call s65); this
# target just prints it.
daily:
	@$(PY) -m src.daily_driver --stdout

# Print what the proactive nudge engine WOULD fire right now (imminent event /
# stale email / due reminder). The launchd job (com.cortex.nudge.plist, every
# 15 min) runs --notify; this --stdout target is the safe manual smoke (never
# consumes the one-shot dedup ledger).
nudge:
	@$(PY) -m src.nudge_driver --stdout

# List your top-ranked job matches WITH the apply links — read-only (no
# re-ranking, no Ollama), the fast "show me the vagas" view. The ledger is
# populated by the 08h briefing (slice 5 v2) or scripts/fetch_jobs.py.
jobs:
	@$(PY) scripts/rank_jobs.py --top 15 --list

# List the contact book. Add/update a contact (with phone + notes) via:
#   venv/bin/python scripts/add_contact.py --name "..." --email "..." --notes "..."
# The contact book feeds name->email resolution and the entity brain.
contacts:
	@$(PY) scripts/add_contact.py --list

# Protocol lint for the A/B relay: flags when the latest HANDOFF session is
# missing from the JOURNAL (journal lag) and when the suite count quoted in the
# HANDOFF drifted from the real pytest count. Read-only; run at session end.
handoff-check:
	@$(PY) scripts/handoff_check.py

# Launchd drift check: diffs every launchd/com.cortex.*.plist against the
# loaded copy in ~/Library/LaunchAgents (content-wise — key order, format
# and comments ignored). DRIFT = exit 1; "not installed" is informational
# (dream is dormant by decision). Machine-state tool (doctor family) — NOT
# part of `make check`/hooks. Runbook §0 runs it before any migration.
verify-launchd:
	@$(PY) scripts/verify_launchd.py

# Backup drift check: every file in ~/.cortex must be either captured by
# scripts/backup.sh or in that script's explicit ignore list. The manifest
# has silently drifted twice (s76 cont.37, s82) and the symptom only shows
# up at RESTORE time. Machine-state tool (doctor family) — NOT part of
# `make check`. Runbook §0 runs it before any migration.
backup-check:
	@$(PY) scripts/backup_check.py
