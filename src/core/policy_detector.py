"""
Policy detector — design-reference ch. 13 (Tier 3).

Reads the gate decision history from the audit log and surfaces actions the
user approves so consistently that an auto-approve POLICY might be worth
creating — evidence over hunch. It only SUGGESTS; it never writes a policy.
A human reads the suggestion and decides — and ideally narrows it with
conditions, since the reference's lesson is that broad policies are an error
vector ("auto-approve all send_email" is a foot-gun; "auto-approve
create_reminder when title < 100 chars" is reasonable).

Data source: the inline gate (`staging_mixin`) emits
`audit.log(..., "gate", "approved"|"rejected", {"type": <tool>})` on every
"sim"/"não". A suggestion needs at least MIN_OCCURRENCES decisions on the
same action type within WINDOW_DAYS and a 100% approval rate — a single
rejection disqualifies it.

High-harm actions (send_email, mac_sleep, mac_quit_app) are surfaced with a
loud caveat: for those the right move is a narrow, conditioned policy, never
a blanket approve, so the detector recommends AGAINST auto-approving them.

Run:  venv/bin/python -m src.core.policy_detector   (or `make policy`)

`analyze()` is pure (testable with synthetic records); `load_records`/`main`
do fail-soft I/O, mirroring src/weekly_review.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Defaults from design-reference ch. 13: ≥10 decisions, 30-day window, 100% approval.
MIN_OCCURRENCES = 10
WINDOW_DAYS = 30
MIN_APPROVAL_RATE = 1.0

# Irreversible / external-facing actions. We still report their streak (it's
# real evidence), but recommend a narrow conditioned policy instead of a
# blanket auto-approve — the worst case (an email you can't unsend, a Mac
# that sleeps mid-work) is exactly what a human gate is for.
_HIGH_HARM = frozenset({"send_email", "mac_sleep", "mac_quit_app"})

# Path matches AuditLogger.DEFAULT_PATH. Kept local so this module doesn't
# import (and instantiate) the logger just to read it.
_AUDIT_PATH = os.environ.get(
    "CORTEX_AUDIT_LOG", os.path.expanduser("~/.cortex/audit.jsonl")
)


@dataclass
class Suggestion:
    """One action type whose inline-approval streak crosses the threshold."""
    action_type: str
    approvals: int
    rejections: int
    approval_rate: float
    high_harm: bool
    recommendation: str


def _parse_ts(rec: Dict[str, Any]) -> Optional[datetime]:
    raw = rec.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # Treat a naive timestamp as UTC so window math never crashes.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def analyze(
    records: List[Dict[str, Any]],
    *,
    min_occurrences: int = MIN_OCCURRENCES,
    window_days: int = WINDOW_DAYS,
    min_approval_rate: float = MIN_APPROVAL_RATE,
    now: Optional[datetime] = None,
) -> List[Suggestion]:
    """Fold gate decision records into auto-approve suggestions. Pure.

    Counts `gate`/`approved` and `gate`/`rejected` events within the window,
    grouped by `payload.type`. An action qualifies when it has at least
    `min_occurrences` decisions and an approval rate >= `min_approval_rate`.
    Results are sorted by approval count (strongest evidence first)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    counts: Dict[str, Dict[str, int]] = {}
    for rec in records:
        if rec.get("stage") != "gate":
            continue
        event = rec.get("event")
        if event not in ("approved", "rejected"):
            continue
        ts = _parse_ts(rec)
        if ts is None or ts < since:
            continue
        atype = ((rec.get("payload") or {}).get("type") or "").strip()
        if not atype:
            continue
        bucket = counts.setdefault(atype, {"approved": 0, "rejected": 0})
        bucket[event] += 1

    suggestions: List[Suggestion] = []
    for atype, c in counts.items():
        approvals, rejections = c["approved"], c["rejected"]
        total = approvals + rejections
        if total < min_occurrences:
            continue
        rate = approvals / total if total else 0.0
        if rate < min_approval_rate:
            continue
        high_harm = atype in _HIGH_HARM
        if high_harm:
            rec_text = (
                f"você aprovou '{atype}' {approvals}/{total} vezes, mas é uma "
                "ação irreversível/externa — NÃO auto-aprove em bloco; "
                "considere uma policy com condições estreitas (ou mantenha a "
                "revisão manual)."
            )
        else:
            rec_text = (
                f"você aprovou '{atype}' {approvals}/{total} vezes sem recusar "
                "nenhuma — candidato a uma policy auto-approve (de preferência "
                "com uma condição, ex.: limite de tamanho)."
            )
        suggestions.append(Suggestion(
            action_type=atype,
            approvals=approvals,
            rejections=rejections,
            approval_rate=rate,
            high_harm=high_harm,
            recommendation=rec_text,
        ))

    suggestions.sort(key=lambda s: s.approvals, reverse=True)
    return suggestions


def render(suggestions: List[Suggestion], *, window_days: int, min_occurrences: int) -> str:
    """Render suggestions as a terse pt-BR bullet report (no tables)."""
    header = (
        f"Policy detector — aprovações inline dos últimos {window_days} dias "
        f"(mín. {min_occurrences} ocorrências, 100% aprovado):\n"
    )
    if not suggestions:
        return header + "  Nada a sugerir ainda — sem ação com evidência suficiente."
    lines = [header]
    for s in suggestions:
        flag = "⚠ " if s.high_harm else "• "
        lines.append(f"  {flag}{s.recommendation}")
    lines.append(
        "\nNenhuma policy é criada automaticamente — isto é só evidência. "
        "Pra ligar uma regra, edite src/core/gate_policy.py."
    )
    return "\n".join(lines)


def load_records(path: str = _AUDIT_PATH,
                 since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Read the JSONL audit records (since `since`, when given), fail-soft —
    the rotated archives included: a rejection archived by a rotation no
    longer stood against a "100% approved" built on the approvals after it
    (s88, Codex lote 10)."""
    from src.core.audit import read_records
    return read_records(path, since)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest gate auto-approve policies from evidence.")
    parser.add_argument("--days", type=int, default=WINDOW_DAYS, help="lookback window in days")
    parser.add_argument("--min", type=int, default=MIN_OCCURRENCES, help="min decisions to suggest")
    parser.add_argument("--path", default=_AUDIT_PATH, help="audit log path")
    args = parser.parse_args(argv)

    records = load_records(
        args.path, since=datetime.now(timezone.utc) - timedelta(days=args.days))
    suggestions = analyze(
        records, min_occurrences=args.min, window_days=args.days,
    )
    print(render(suggestions, window_days=args.days, min_occurrences=args.min))
    return 0


if __name__ == "__main__":
    sys.exit(main())
