"""
Weekly review — a Sunday digest of what Cortex did in the last 7 days.

Run:  venv/bin/python -m src.weekly_review     (or `make weekly`)
      venv/bin/python -m src.weekly_review --days 7 --stdout

Reads the structured audit log (~/.cortex/audit.jsonl — the same append-only
JSONL every pipeline stage writes to) for the trailing window and renders a
plain-Markdown digest into data/reviews/<ISO-year>-W<week>.md: how many
requests, which response modes, reflex hits, tools dispatched, gate-queue
activity, plans run, and health signals (blocked hallucinations, failed
actions). It's the design-reference ch. 14 "weekly review" idea, grounded in the
events the system actually emits rather than a generic template.

Design mirrors `src/doctor.py`: a thin CLI over pure helpers. Aggregation
(`aggregate`) and rendering (`render_markdown`) are pure functions over a
list of records — no I/O — so they unit-test without touching disk or the
Palace. Only `load_records`/`main` do I/O. Reading is fail-soft: a missing
or partly-corrupt log yields an empty/partial digest, never a crash.

No markdown tables (bullets only — they read cleanly and match the house
style). Output lives in data/reviews/ (gitignored, like other generated
artefacts); nothing here is sensitive beyond what's already in the audit log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Same default + env var as src/core/audit.py, so the review reads exactly
# what the pipeline wrote. Kept local to avoid importing the logger (and its
# side effect of creating ~/.cortex) just to read a path.
DEFAULT_AUDIT_PATH = os.path.expanduser("~/.cortex/audit.jsonl")
DEFAULT_WINDOW_DAYS = 7

REVIEWS_DIR = "data/reviews"


@dataclass
class WeeklyStats:
    """Aggregated counters for one window. Pure data — no I/O."""
    total_records: int = 0
    requests: int = 0
    modes: Counter = field(default_factory=Counter)
    reflex_hits: int = 0
    tools: Counter = field(default_factory=Counter)
    gate: Counter = field(default_factory=Counter)
    plans_run: int = 0
    hallucinations_blocked: int = 0
    action_errors: int = 0
    voice_turns: int = 0


def _parse_ts(rec: Dict[str, Any]) -> Optional[datetime]:
    raw = rec.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # Audit writes tz-aware UTC; tolerate a naive value just in case.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_records(path: str, since: datetime) -> List[Dict[str, Any]]:
    """Read JSONL audit records with ts >= `since` — the rotated archives
    too, or a week straddling a rotation lost its first days (s88, Codex
    lote 10). Malformed lines and a missing file are skipped silently — a
    digest must never crash."""
    from src.core.audit import read_records
    return read_records(path, since)


def aggregate(records: List[Dict[str, Any]]) -> WeeklyStats:
    """Fold audit records into WeeklyStats. Pure — no I/O. The event names
    here mirror what the live pipeline emits (orchestrator request/response,
    reflex hit, tool dispatch, gate queued/approved/…, planner execute_start,
    engine hallucination_blocked/invented_action_blocked)."""
    s = WeeklyStats(total_records=len(records))
    for rec in records:
        stage = rec.get("stage")
        event = rec.get("event")
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}

        if stage == "orchestrator":
            if event == "request":
                s.requests += 1
            elif event == "response":
                mode = payload.get("mode")
                if mode:
                    s.modes[mode] += 1
                    if isinstance(mode, str) and mode.endswith("ERROR"):
                        s.action_errors += 1
        elif stage == "reflex" and event == "hit":
            s.reflex_hits += 1
        elif stage == "tool" and event in ("dispatch", "dispatch_in_plan"):
            name = payload.get("name") or "?"
            s.tools[name] += 1
        elif stage == "gate" and event in ("queued", "approved", "rejected", "executed", "failed"):
            s.gate[event] += 1
        elif stage == "planner" and event == "execute_start":
            s.plans_run += 1
        elif stage == "engine" and event in ("hallucination_blocked", "invented_action_blocked"):
            s.hallucinations_blocked += 1
        elif stage == "voice" and event == "done":
            s.voice_turns += 1
    return s


def _fmt_counter(c: Counter, top: int = 8) -> List[str]:
    return [f"  - {name}: {n}" for name, n in c.most_common(top)]


def render_markdown(
    stats: WeeklyStats,
    *,
    since: datetime,
    until: datetime,
    week_label: str,
    pending_studies: Optional[int] = None,
) -> str:
    """Render the digest as plain Markdown (bullets, no tables)."""
    d = since.astimezone().strftime("%d/%m")
    u = until.astimezone().strftime("%d/%m")
    lines: List[str] = [
        f"# Cortex — revisão semanal {week_label}",
        "",
        f"Janela: {d} → {u} ({DEFAULT_WINDOW_DAYS} dias) · "
        f"{stats.total_records} eventos no audit log.",
        "",
        "## Atividade",
        f"- Pedidos atendidos: {stats.requests}",
        f"- Reflexos instantâneos (sem LLM): {stats.reflex_hits}",
        f"- Planos executados: {stats.plans_run}",
        f"- Turnos por voz: {stats.voice_turns}",
    ]

    if stats.modes:
        lines.append("- Modos de resposta:")
        lines += _fmt_counter(stats.modes)

    lines += ["", "## Ferramentas"]
    if stats.tools:
        lines += _fmt_counter(stats.tools)
    else:
        lines.append("  - (nenhuma ferramenta despachada)")

    lines += ["", "## Gate de ações (fila de aprovação)"]
    if stats.gate:
        order = ["queued", "approved", "rejected", "executed", "failed"]
        labels = {
            "queued": "enfileiradas", "approved": "aprovadas",
            "rejected": "rejeitadas", "executed": "executadas",
            "failed": "falharam",
        }
        for ev in order:
            if stats.gate.get(ev):
                lines.append(f"  - {labels[ev]}: {stats.gate[ev]}")
    else:
        lines.append("  - (sem atividade no gate)")

    lines += ["", "## Saúde"]
    lines.append(f"- Alucinações bloqueadas: {stats.hallucinations_blocked}")
    lines.append(f"- Ações com erro: {stats.action_errors}")
    if stats.gate.get("failed"):
        lines.append(f"- Ações de background que falharam: {stats.gate['failed']}")
    if pending_studies is not None:
        lines += ["", "## Estudos", f"- Estudos pendentes de revisão: {pending_studies}"]

    lines.append("")
    return "\n".join(lines)


def _pending_studies_count() -> Optional[int]:
    """Best-effort count of briefings awaiting review. Fail-soft: returns
    None if the Palace/StudiesReview can't be reached (e.g. test env)."""
    try:
        from src.core.studies_review import StudiesReview

        return len(StudiesReview().list_pending())
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    # The job runs under launchd with PATH only: the .env that tells the
    # nightly where the audit is was never read here (s88, Codex lote 10).
    # Same as the nightly; an explicit env var still wins (override=False).
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), ".env"), override=False)
    except Exception:  # noqa: BLE001 — a missing .env is not a failed review
        pass
    parser = argparse.ArgumentParser(
        prog="weekly_review",
        description="Resumo semanal da atividade do Cortex (lê o audit log).",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"janela em dias (default {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--audit", default=os.environ.get("CORTEX_AUDIT_LOG", DEFAULT_AUDIT_PATH),
                        help="caminho do audit.jsonl")
    parser.add_argument("--stdout", action="store_true",
                        help="imprime o relatório em vez de gravar o arquivo")
    args = parser.parse_args(argv)

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=args.days)
    records = load_records(args.audit, since)
    stats = aggregate(records)
    week_label = until.astimezone().strftime("%G-W%V")
    report = render_markdown(
        stats, since=since, until=until, week_label=week_label,
        pending_studies=_pending_studies_count(),
    )

    if args.stdout:
        print(report)
        return 0

    os.makedirs(REVIEWS_DIR, exist_ok=True)
    out_path = os.path.join(REVIEWS_DIR, f"{week_label}.md")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
    except OSError as e:
        print(f"weekly_review: could not write {out_path}: {e}", file=sys.stderr)
        return 1
    print(f"weekly review written → {out_path} ({stats.total_records} events, "
          f"{stats.requests} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
