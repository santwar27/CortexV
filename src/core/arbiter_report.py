"""
Arbiter report — reads the shadow classifier's audit trail.

The arbiter (src/core/arbiter.py) runs in SHADOW since s39: on every turn it
classifies the query into a Route + confidence and logs it WITHOUT changing
routing (orchestrator `CORTEX_ARBITER_SHADOW`). This tool is the reader that
turns that pile of `arbiter`/`classify` records into something actionable —
the same "infra ahead of the producer" pattern as `make policy`/`make
promoter`, here closing the loop on what we already turned on.

Two payoffs, both M1-now:
  1. Real route DISTRIBUTION (how many REFLEX / LOOKUP / MEMORY / SIMPLES /
     COMPLEXO / LONGO) — the calibration we need before flipping
     CORTEX_ARBITER_CASCADE on the Mac Studio. Until we see the mix we are
     guessing which routes deserve which model tier.
  2. The LOW-CONFIDENCE cases (confidence < the cascade's wake threshold):
     these are exactly the queries the deterministic regex was UNSURE about.
     Each one is either a marker to add to arbiter.py now, or a query that
     genuinely needs Layer 2 (the dormant LLM tiebreak). Surfacing them lets
     us fix the regex today instead of waiting for the big model.

Run:  venv/bin/python -m src.core.arbiter_report   (or `make arbiter-report`)

`analyze` is pure (testable with synthetic records); `load_records`/`main` do
fail-soft I/O, mirroring src/core/policy_detector.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Mirrors arbiter.LOW_CONFIDENCE_THRESHOLD — a classification below this is
# what would wake Layer 2 once the cascade is enabled, so it's exactly what we
# want to eyeball. Kept local so the report doesn't import the classifier (and
# its reflex/regex machinery) just to read one constant.
LOW_CONFIDENCE_THRESHOLD = 0.8
WINDOW_DAYS = 30

# The six reference routes, in cascade order, so the report always lists them in
# a stable shape even when a route has zero observations.
_ROUTES = ("REFLEX", "LOOKUP", "MEMORY", "SIMPLES", "COMPLEXO", "LONGO")

# Path matches AuditLogger.DEFAULT_PATH (see policy_detector for the rationale).
_AUDIT_PATH = os.environ.get(
    "CORTEX_AUDIT_LOG", os.path.expanduser("~/.cortex/audit.jsonl")
)


@dataclass
class LowConfidenceCase:
    """One classification the cascade was unsure about — a regex-fix candidate."""
    route: str
    confidence: float
    reason: str
    text: str


@dataclass
class ArbiterReport:
    """Folded view of the shadow classifier over the window."""
    total: int = 0
    by_route: Dict[str, int] = field(default_factory=dict)
    by_layer: Dict[str, int] = field(default_factory=dict)
    by_reason: Dict[str, int] = field(default_factory=dict)
    low_confidence: List[LowConfidenceCase] = field(default_factory=list)


def _parse_ts(rec: Dict[str, Any]) -> Optional[datetime]:
    raw = rec.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def analyze(
    records: List[Dict[str, Any]],
    *,
    window_days: int = WINDOW_DAYS,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    now: Optional[datetime] = None,
) -> ArbiterReport:
    """Fold `arbiter`/`classify` records into a distribution + unsure cases.

    Pure. Counts route/layer/reason within the window and collects every
    classification whose confidence is below `low_confidence_threshold`
    (sorted lowest-confidence first — the most ambiguous queries to fix)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    by_route: Counter = Counter()
    by_layer: Counter = Counter()
    by_reason: Counter = Counter()
    low: List[LowConfidenceCase] = []
    total = 0

    for rec in records:
        if rec.get("stage") != "arbiter" or rec.get("event") != "classify":
            continue
        ts = _parse_ts(rec)
        if ts is None or ts < since:
            continue
        payload = rec.get("payload") or {}
        route = str(payload.get("route") or "?")
        layer = str(payload.get("layer") or "?")
        reason = str(payload.get("reason") or "?")
        try:
            conf = float(payload.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.0

        total += 1
        by_route[route] += 1
        by_layer[layer] += 1
        by_reason[reason] += 1
        if conf < low_confidence_threshold:
            low.append(LowConfidenceCase(
                route=route, confidence=conf, reason=reason,
                text=str(payload.get("input") or ""),
            ))

    low.sort(key=lambda c: c.confidence)
    return ArbiterReport(
        total=total,
        by_route=dict(by_route),
        by_layer=dict(by_layer),
        by_reason=dict(by_reason),
        low_confidence=low,
    )


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.0f}%" if total else "0%"


def _sorted_counts(d: Dict[str, int], *, order: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, int]]:
    """Stable ordering: by fixed `order` if given (zeros included), else by
    count descending."""
    if order is not None:
        return [(k, d.get(k, 0)) for k in order]
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)


def render(report: ArbiterReport, *, window_days: int, max_low: int = 30) -> str:
    """Render the report as a terse pt-BR bullet digest (no tables)."""
    header = (
        f"Arbiter shadow — {report.total} classificações nos últimos "
        f"{window_days} dias:"
    )
    if report.total == 0:
        return (
            header + "\n  Nada registrado ainda — confirme que "
            "CORTEX_ARBITER_SHADOW=1 está ligado no orchestrator."
        )

    lines = [header, "", "Distribuição por rota:"]
    for route, n in _sorted_counts(report.by_route, order=_ROUTES):
        lines.append(f"  • {route:<9} {n:>5}  ({_pct(n, report.total)})")

    lines += ["", "Por camada (cascade=regex, fallback=sem marker, llm=Layer 2):"]
    for layer, n in _sorted_counts(report.by_layer):
        lines.append(f"  • {layer:<9} {n:>5}  ({_pct(n, report.total)})")

    n_low = len(report.low_confidence)
    lines += [
        "",
        f"Baixa confiança (< {LOW_CONFIDENCE_THRESHOLD:g}) — {n_low} casos "
        "(candidatos a marker novo na regex, ou Layer 2 quando ligar):",
    ]
    if not report.low_confidence:
        lines.append("  Nenhum — a cascade resolveu tudo com confiança.")
    else:
        for c in report.low_confidence[:max_low]:
            snippet = c.text if len(c.text) <= 60 else c.text[:57] + "…"
            lines.append(
                f"  • [{c.route} {c.confidence:g} {c.reason}] {snippet!r}"
            )
        if n_low > max_low:
            lines.append(f"  … (+{n_low - max_low} mais; use --max-low)")

    lines += [
        "",
        "Read-only. Pra ajustar o roteamento, edite os markers em "
        "src/core/arbiter.py; pra acender a Layer 2, CORTEX_ARBITER_CASCADE.",
    ]
    return "\n".join(lines)


def load_records(path: str = _AUDIT_PATH) -> List[Dict[str, Any]]:
    """Read every JSONL audit record, fail-soft (mirrors policy_detector)."""
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the shadow arbiter's route distribution and unsure cases."
    )
    parser.add_argument("--days", type=int, default=WINDOW_DAYS, help="lookback window in days")
    parser.add_argument("--max-low", type=int, default=30, help="max low-confidence cases to list")
    parser.add_argument("--path", default=_AUDIT_PATH, help="audit log path")
    args = parser.parse_args(argv)

    records = load_records(args.path)
    report = analyze(records, window_days=args.days)
    print(render(report, window_days=args.days, max_low=args.max_low))
    return 0


if __name__ == "__main__":
    sys.exit(main())
