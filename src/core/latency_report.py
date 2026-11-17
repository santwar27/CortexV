"""
Latency report — per-route turn latency from the audit log.

Sibling of arbiter_report: the orchestrator already stamps every turn with
`audit.log(..., "orchestrator", "response", latency_ms=…, payload={"mode": …})`
(see orchestrator.process). The `mode` is the de-facto route the turn took
(OFFLINE, SPOTIFY_PLAY, TOOL_READ_DOCUMENT, AGENT_*, …), so folding latency by
mode answers the standing question "which paths are slow?" — voice latency is
a known complaint (see memory voice_ux_open_items), and a slow mode is the
first place to look.

What it reports per mode: count, p50, p95, max, mean — sorted by p95
descending, so the worst tail latency is at the top. p95 (not mean) is the
headline because tail latency is what a user actually feels.

Scope note: this reads what is ALREADY logged. It deliberately does NOT add
new hot-path instrumentation (e.g. number of LLM rounds per turn) — that's a
runtime change for a later wave; here we stay read-only.

Run:  venv/bin/python -m src.core.latency_report   (or `make latency-report`)

`analyze` is pure (testable with synthetic records); `load_records`/`main` do
fail-soft I/O, mirroring src/core/policy_detector.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

WINDOW_DAYS = 30

# Default source: one record per completed turn, carrying latency + route.
_STAGE = "orchestrator"
_EVENT = "response"
_BUCKET_KEY = "mode"

_AUDIT_PATH = os.environ.get(
    "CORTEX_AUDIT_LOG", os.path.expanduser("~/.cortex/audit.jsonl")
)


@dataclass
class LatencyStats:
    """Latency summary for one route/bucket, in milliseconds."""
    bucket: str
    count: int
    p50: float
    p95: float
    max: float
    mean: float


def _parse_ts(rec: Dict[str, Any]) -> Optional[datetime]:
    raw = rec.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list. pct in [0, 100]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Nearest-rank: rank = ceil(pct/100 * N), clamped to [1, N].
    rank = max(1, min(len(sorted_vals),
                      int(-(-pct * len(sorted_vals) // 100))))
    return sorted_vals[rank - 1]


def analyze(
    records: List[Dict[str, Any]],
    *,
    window_days: int = WINDOW_DAYS,
    stage: str = _STAGE,
    event: str = _EVENT,
    bucket_key: str = _BUCKET_KEY,
    now: Optional[datetime] = None,
) -> List[LatencyStats]:
    """Fold turn-latency records into per-bucket stats. Pure.

    Groups `latency_ms` by `payload[bucket_key]` within the window and computes
    count/p50/p95/max/mean. Sorted by p95 descending (worst tail first)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    buckets: Dict[str, List[float]] = {}
    for rec in records:
        if rec.get("stage") != stage or rec.get("event") != event:
            continue
        lat = rec.get("latency_ms")
        if not isinstance(lat, (int, float)):
            continue
        ts = _parse_ts(rec)
        if ts is None or ts < since:
            continue
        bucket = str((rec.get("payload") or {}).get(bucket_key) or "?")
        buckets.setdefault(bucket, []).append(float(lat))

    out: List[LatencyStats] = []
    for bucket, vals in buckets.items():
        vals.sort()
        out.append(LatencyStats(
            bucket=bucket,
            count=len(vals),
            p50=_percentile(vals, 50),
            p95=_percentile(vals, 95),
            max=vals[-1],
            mean=sum(vals) / len(vals),
        ))

    out.sort(key=lambda s: s.p95, reverse=True)
    return out


def _ms(v: float) -> str:
    """Human-friendly ms/s formatting."""
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def render(stats: List[LatencyStats], *, window_days: int, bucket_key: str = _BUCKET_KEY) -> str:
    """Render per-bucket latency as a terse pt-BR bullet digest (no tables)."""
    header = (
        f"Latência por {bucket_key} — turnos dos últimos {window_days} dias "
        "(ordenado por p95, pior cauda primeiro):"
    )
    if not stats:
        return header + "\n  Nada registrado ainda (sem turnos com latency_ms)."

    total = sum(s.count for s in stats)
    lines = [header, ""]
    for s in stats:
        lines.append(
            f"  • {s.bucket:<28} n={s.count:<4} "
            f"p50={_ms(s.p50)}  p95={_ms(s.p95)}  max={_ms(s.max)}"
        )
    lines += [
        "",
        f"Total: {total} turnos. p95 alto = caminho a investigar "
        "(latência de voz é queixa conhecida — ver voice_ux_open_items).",
    ]
    return "\n".join(lines)


def load_records(path: str = _AUDIT_PATH,
                 since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Every audit record at/after `since`, the rotated archives included —
    the shared reader (audit.read_records), fail-soft. This one opened only
    the active file: after a rotation the 30-day report said "Total: 2
    turnos" over a hundred (s88, Codex lote 13)."""
    from src.core.audit import read_records
    return read_records(path, since)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report per-route turn latency from the audit log."
    )
    parser.add_argument("--days", type=int, default=WINDOW_DAYS, help="lookback window in days")
    parser.add_argument("--stage", default=_STAGE, help="audit stage to read")
    parser.add_argument("--event", default=_EVENT, help="audit event to read")
    parser.add_argument("--bucket", default=_BUCKET_KEY, help="payload key to group by")
    parser.add_argument("--path", default=_AUDIT_PATH, help="audit log path")
    args = parser.parse_args(argv)

    records = load_records(
        args.path, since=datetime.now(timezone.utc) - timedelta(days=args.days))
    stats = analyze(
        records, window_days=args.days, stage=args.stage,
        event=args.event, bucket_key=args.bucket,
    )
    print(render(stats, window_days=args.days, bucket_key=args.bucket))
    return 0


if __name__ == "__main__":
    sys.exit(main())
