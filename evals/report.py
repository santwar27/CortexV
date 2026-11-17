"""Terminal reporter for eval results.

Goal: a human reading the output in 2 seconds knows
  (a) total cases run
  (b) how many passed/failed/errored, by category
  (c) for each failure: case name + summary + detail (one block)

No external deps — ANSI escapes are baked in. Detects a non-TTY
output stream and strips colours so piped runs and CI logs stay
clean.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Iterable, List

from evals.core import EvalResult, EvalStatus


# ANSI escape sequences. Disabled when stdout isn't a TTY (e.g. piped
# to a file, captured by pytest), so logs and CI output stay clean.
_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _green(s: str) -> str:  return _c("32", s)
def _red(s: str) -> str:    return _c("31", s)
def _yellow(s: str) -> str: return _c("33", s)
def _dim(s: str) -> str:    return _c("2",  s)
def _bold(s: str) -> str:   return _c("1",  s)
def _cyan(s: str) -> str:   return _c("36", s)


def _status_tag(status: EvalStatus) -> str:
    if status == EvalStatus.PASS:
        return _green("PASS")
    if status == EvalStatus.FAIL:
        return _red("FAIL")
    if status == EvalStatus.DRIFT:
        return _yellow("DRIFT")
    return _yellow("ERR ")


def format_terminal(results: List[EvalResult], verbose: bool = False) -> str:
    """Render the results as a multi-line string. The caller writes
    it to stdout. Verbose mode prints every case; default mode
    prints only failures/errors plus the per-category summary."""
    if not results:
        return _dim("no results\n")

    lines: list[str] = []
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.case.category].append(r)

    lines.append(_bold("\nCortex evals\n"))
    lines.append(_dim(f"  {len(results)} cases · "))

    counts = Counter(r.status for r in results)
    bits = []
    if counts[EvalStatus.PASS]:
        bits.append(_green(f"{counts[EvalStatus.PASS]} pass"))
    if counts[EvalStatus.FAIL]:
        bits.append(_red(f"{counts[EvalStatus.FAIL]} fail"))
    if counts[EvalStatus.DRIFT]:
        bits.append(_yellow(f"{counts[EvalStatus.DRIFT]} drift"))
    if counts[EvalStatus.ERROR]:
        bits.append(_yellow(f"{counts[EvalStatus.ERROR]} error"))
    lines[-1] += " · ".join(bits) + "\n\n"

    # Per-category summary.
    for cat in sorted(by_cat.keys()):
        cat_results = by_cat[cat]
        cc = Counter(r.status for r in cat_results)
        bits = []
        if cc[EvalStatus.PASS]:
            bits.append(_green(f"{cc[EvalStatus.PASS]}✓"))
        if cc[EvalStatus.FAIL]:
            bits.append(_red(f"{cc[EvalStatus.FAIL]}✗"))
        if cc[EvalStatus.DRIFT]:
            bits.append(_yellow(f"{cc[EvalStatus.DRIFT]}↯"))
        if cc[EvalStatus.ERROR]:
            bits.append(_yellow(f"{cc[EvalStatus.ERROR]}!"))
        lines.append(
            f"  {_cyan(cat):<32s} "
            f"{len(cat_results):>3d} cases  "
            f"{'  '.join(bits)}\n"
        )

    # Baseline drift report (LLM mode only). Renders when any result was
    # scored against a recorded baseline. A case can PASS its substring
    # check yet still have drifted in wording — that only shows here, so
    # this section is independent of the failure blocks below.
    scored = [r for r in results if "baseline_similarity" in r.payload]
    new_vs_baseline = [r for r in results if r.payload.get("baseline_status") == "new"]
    if scored or new_vs_baseline:
        lines.append("\n" + _bold("Baseline similarity (cosine vs recorded reply):\n"))
        for r in sorted(scored, key=lambda x: x.payload["baseline_similarity"]):
            sim = r.payload["baseline_similarity"]
            drifted = r.payload.get("baseline_drift", False)
            tag = _yellow("DRIFT") if drifted else _green(" ok  ")
            sim_txt = f"{sim:.3f}"
            sim_col = _yellow(sim_txt) if drifted else _dim(sim_txt)
            lines.append(f"  {tag}  {sim_col}  {_dim(r.case.name)}\n")
        for r in new_vs_baseline:
            lines.append(
                f"  {_dim(' new ')}  {_dim('  —  ')}  {_dim(r.case.name)} "
                f"{_dim('(no baseline yet)')}\n"
            )

    # Failure / drift / error detail blocks.
    bad = [r for r in results if r.status != EvalStatus.PASS]
    if bad:
        lines.append("\n" + _bold("Failures / drift / errors:\n"))
        for r in bad:
            lines.append(
                f"  {_status_tag(r.status)}  "
                f"{_cyan(r.case.category)} · {r.case.name}\n"
            )
            if r.case.description:
                lines.append(_dim(f"        {r.case.description}\n"))
            if r.summary:
                lines.append(f"        {r.summary}\n")
            if r.detail:
                # Indent each detail line so block reads as one item.
                for dl in r.detail.splitlines():
                    lines.append(_dim(f"        | {dl}\n"))
            lines.append("\n")

    if verbose:
        lines.append("\n" + _bold("All cases:\n"))
        for r in results:
            lines.append(
                f"  {_status_tag(r.status)}  "
                f"{_dim(r.case.category)} · {r.case.name}  "
                f"{_dim(r.summary)}\n"
            )

    # Bottom-line one-liner so a 2-second glance shows overall state.
    # DRIFT alone is NOT a failure — it's expected for drift-tolerant
    # LLM cases and must not turn the run red.
    lines.append("\n")
    if counts[EvalStatus.FAIL] or counts[EvalStatus.ERROR]:
        lines.append(_red(_bold(
            f"FAILED · {counts[EvalStatus.FAIL]} fail, "
            f"{counts[EvalStatus.ERROR]} error\n"
        )))
    elif counts[EvalStatus.DRIFT]:
        lines.append(_green(_bold(
            f"ALL PASS · {counts[EvalStatus.PASS]} cases "
        )) + _yellow(_bold(
            f"({counts[EvalStatus.DRIFT]} drift tolerated)\n"
        )))
    else:
        lines.append(_green(_bold(
            f"ALL GREEN · {counts[EvalStatus.PASS]} cases\n"
        )))
    return "".join(lines)
