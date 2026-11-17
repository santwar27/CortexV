"""CLI entry point for the eval framework.

Usage:
  venv/bin/python -m evals
  venv/bin/python -m evals --category spotify
  venv/bin/python -m evals --kind dispatch
  venv/bin/python -m evals --tag regression
  venv/bin/python -m evals --verbose
  venv/bin/python -m evals --json out.json    (writes to file)
  venv/bin/python -m evals --json -           (writes to stdout)
  venv/bin/python -m evals --llm-only         (only LLM_BEHAVIOR — hits Ollama)
  venv/bin/python -m evals --llm-rebaseline    (record baseline for current model)

By default LLM_BEHAVIOR cases are EXCLUDED (they hit Ollama, slow +
drift-tolerant). `--llm-only` runs just those, opt-in; when a baseline
exists for the active (model, prompt_version) it also scores each reply
by cosine similarity against the recorded one and reports drift.
`--llm-rebaseline` re-records that baseline (explicit, deliberate).

Exit code:
  0 if all cases passed (DRIFT is tolerated, still 0)
  1 if any case failed or errored
  2 if no cases matched the filter (helps catch typo'd category)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evals.core import LLM_KINDS, EvalKind, EvalStatus, load_all_cases
from evals.llm_baseline import load_baseline, replies_from_results, save_baseline
from evals.report import format_terminal
from evals.runner import run
from evals.similarity import score_against_baseline


def _llm_identity(results):
    """Pull (model, prompt_version) off the LLM results. Every
    LLM_BEHAVIOR payload carries both; we read the first that has them.
    Returns (None, None) if no usable result (e.g. all errored before
    reaching the engine)."""
    for r in results:
        model = r.payload.get("model")
        if model is not None:
            return model, r.payload.get("prompt_version")
    return None, None


def _handle_baseline(results, *, rebaseline: bool) -> None:
    """Record or compare the baseline for the active model/prompt.

    Rebaseline writes the current replies to disk. Plain mode loads the
    stored baseline (if any) and annotates each result with cosine
    similarity vs the recorded reply — purely informational, the reporter
    surfaces it and the exit code never changes.
    """
    model, prompt_version = _llm_identity(results)
    if model is None or prompt_version is None:
        sys.stdout.write(
            "  (baseline skipped: no model/prompt_version on results)\n"
        )
        return

    replies = replies_from_results(results)

    if rebaseline:
        path = save_baseline(model, prompt_version, replies)
        sys.stdout.write(f"  baseline written: {path}  ({len(replies)} cases)\n")
        return

    baseline = load_baseline(model, prompt_version)
    if baseline is None:
        sys.stdout.write(
            f"  (no baseline for {model} v{prompt_version} — run "
            "`make eval-llm-rebaseline` to record one)\n"
        )
        return

    scores = score_against_baseline(replies, baseline)
    drifted = 0
    for r in results:
        if "reply" not in r.payload:
            # Not baseline-eligible (e.g. tool-call results carry a tool
            # name, not a reply text — cosine on those is meaningless).
            continue
        score = scores.get(r.case.name)
        if score is None:
            # Reply-bearing case absent from the baseline (newly added) —
            # flag it so the reporter says "no baseline" not "matched".
            r.payload["baseline_status"] = "new"
            continue
        r.payload["baseline_similarity"] = score["similarity"]
        r.payload["baseline_drift"] = score["drift"]
        if score["drift"]:
            drifted += 1
    sys.stdout.write(
        f"  baseline: {model} v{prompt_version} · {len(scores)} compared · "
        f"{drifted} drifted (cosine < 0.92)\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    parser.add_argument(
        "--category", "-c",
        help="filter by case.category (spotify, macos, anti_hallucination, ...)"
    )
    parser.add_argument(
        "--kind", "-k",
        choices=[k.value for k in EvalKind],
        help="filter by case kind"
    )
    parser.add_argument(
        "--tag", "-t",
        help="filter by tag presence in case.tags"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="show every case result, not just failures"
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="emit JSON results to PATH (use '-' for stdout)"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="run ONLY LLM_BEHAVIOR cases (real Ollama calls; opt-in)"
    )
    parser.add_argument(
        "--llm-rebaseline",
        action="store_true",
        help="run LLM cases and (re)record their replies as the baseline "
             "for the active (model, prompt_version). Explicit, opt-in."
    )
    args = parser.parse_args(argv)

    llm_mode = args.llm_only or args.llm_rebaseline
    if llm_mode:
        # Both LLM kinds (chat + tool-call) run in LLM mode. Pre-filter to
        # LLM_KINDS and leave the runner's kind filter unset — a single
        # `kind=` can only select one, which would silently drop tool-call.
        cases = [c for c in load_all_cases(include_llm=True) if c.kind in LLM_KINDS]
        kind = None
        model = (os.environ.get("CORTEX_MODEL")
                 or os.environ.get("CORTEX_MODEL_ANALYTICAL") or "cortex-lite")
        sys.stdout.write(f"Cortex LLM behaviour evals (model={model})\n")
    else:
        cases = load_all_cases()
        kind = EvalKind(args.kind) if args.kind else None
    results = run(cases, category=args.category, kind=kind, tag=args.tag)

    if not results:
        sys.stderr.write(
            f"no cases matched (category={args.category!r}, "
            f"kind={args.kind!r}, tag={args.tag!r}) — check spelling\n"
        )
        return 2

    # Baseline: record (rebaseline) or compare (plain llm-only). Reads
    # the active (model, prompt_version) straight off the results so the
    # engine is never booted twice. Comparison hits Ollama (embeddings),
    # so it only runs in LLM mode.
    if llm_mode:
        _handle_baseline(results, rebaseline=args.llm_rebaseline)

    # Terminal output (always, even with --json — humans benefit from
    # the colour at a glance; --json is purely additive).
    sys.stdout.write(format_terminal(results, verbose=args.verbose))

    if args.json is not None:
        payload = [
            {
                "name": r.case.name,
                "kind": r.case.kind.value,
                "category": r.case.category,
                "tags": r.case.tags,
                "status": r.status.value,
                "summary": r.summary,
                "detail": r.detail,
                "data": r.payload,
            }
            for r in results
        ]
        blob = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json == "-":
            sys.stdout.write(blob + "\n")
        else:
            Path(args.json).write_text(blob + "\n", encoding="utf-8")

    # DRIFT is tolerated — only FAIL/ERROR make the run non-zero.
    any_bad = any(r.status in (EvalStatus.FAIL, EvalStatus.ERROR) for r in results)
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
