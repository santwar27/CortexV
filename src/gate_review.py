"""
Gate review CLI — approve or reject background actions awaiting your OK.

Run:  venv/bin/python -m src.gate_review            # list pending
      venv/bin/python -m src.gate_review approve <id>
      venv/bin/python -m src.gate_review reject  <id>

Background work (a nightly scheduled task, a future Dream Loop) can't pop
the inline "sim/não" gate — there's nobody to answer at 3am. When such a
task picks a destructive (S3) tool it lands in the gate batch queue
(`src/core/gate_queue.py`, `~/.cortex/gate_queue.jsonl`). This is the
human review surface for it: list what's waiting, approve to actually run
it (through the SAME per-tool dispatch the inline gate uses,
`orchestrator._run_action`), or reject to drop it. Fail-closed — nothing
runs until you approve it here.

`approve`/`reject`/`render_list` take injectable `queue` + `runner` seams
so the flow is unit-testable without a live orchestrator or real
integrations.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional, Tuple

from src.core.gate_queue import GateQueue, gate_queue, record_outcome

# A runner executes one approved action and returns (reply, mode), where
# mode is "AGENT_ACTION_DONE" on success or "AGENT_ACTION_ERROR" on failure.
Runner = Callable[[str, dict], Tuple[str, str]]


def _default_runner(action_type: str, args: dict, effect_id=None) -> Tuple[str, str]:
    """Execute via the orchestrator's shared per-tool dispatch — the exact
    same code the inline gate runs on 'sim'. Built lazily so listing /
    rejecting never pays the orchestrator's startup cost."""
    from src.core.orchestrator import CortexOrchestrator

    return CortexOrchestrator()._run_action(action_type, args, effect_id=effect_id)


def render_list(queue: Optional[GateQueue] = None) -> str:
    queue = queue or gate_queue
    pend = queue.pending()
    if not pend:
        return "Nenhuma ação aguardando aprovação."
    lines = [f"{len(pend)} ação(ões) aguardando sua aprovação:", ""]
    for e in pend:
        args = e.get("args", {}) or {}
        lines.append(_field(f"  [{e.get('id')}]  ", e.get("type", "?")))
        lines.append(_field("       origin=", e.get("origin", "?"))
                     + f"  queued={e.get('ts', '?')}")
        if args:
            lines.append("       args:")
            for k, v in args.items():
                key = _visible(k).replace("\n", "\\n")
                lines.append(_field(f"         {key}=", v))
        if e.get("reason"):
            lines.append(_field("       reason: ", e["reason"]))
    lines += [
        "",
        "Aprovar:  python -m src.gate_review approve <id>",
        "Rejeitar: python -m src.gate_review reject <id>",
    ]
    return "\n".join(lines)


def _visible(value) -> str:
    # Escape terminal controls only for display; the queue remains execution's source of truth.
    chars = []
    for char in str(value).replace("\r\n", "\n"):
        code = ord(char)
        if ((code < 32 and char not in "\n\t") or 127 <= code <= 159
                or 0x202a <= code <= 0x202e or 0x2066 <= code <= 0x2069):
            char = f"\\x{code:02x}" if code <= 255 else f"\\u{code:04x}"
        chars.append(char)
    return "".join(chars)


def _field(prefix: str, value) -> str:
    return prefix + _visible(value).replace("\n", "\n" + " " * len(prefix))


def decide(
    action_id: str,
    decision: str,
    queue: Optional[GateQueue] = None,
    runner: Optional[Runner] = None,
) -> dict:
    """Record one decision through the queue's lock, then execute at most once."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be approved or rejected")
    queue = queue or gate_queue
    entry = queue.get(action_id)
    if entry is None:
        return {"result": "not_found", "status": None, "reply": None}
    if not queue.resolve(action_id, decision):
        status = queue.status(action_id)
        return {"result": "not_recorded" if status == "queued" else "already_decided",
                "status": status, "reply": None}
    if decision == "rejected":
        return {"result": "rejected", "status": queue.status(action_id),
                "reply": f"id '{action_id}' rejeitado — não vou executar."}
    try:
        action_type, args = entry.get("type", ""), entry.get("args", {}) or {}
        reply, mode = (runner(action_type, args) if runner is not None else
                       _default_runner(action_type, args, effect_id=action_id))
    except Exception as exc:
        reply, mode = str(exc) or type(exc).__name__, "AGENT_ACTION_ERROR"
    tag = record_outcome(queue, action_id, reply, mode)
    return {"result": {"executado": "executed", "FALHOU": "failed", "NÃO SEI": "unknown"}[tag],
            "status": queue.status(action_id), "reply": reply}


def _format_decision(action_id: str, outcome: dict) -> str:
    result = outcome["result"]
    if result == "not_found":
        return f"id '{action_id}' não encontrado na fila."
    if result in ("not_recorded", "already_decided"):
        return f"id '{action_id}' já foi decidido (status: {outcome['status']})."
    if result == "rejected":
        return outcome["reply"]
    tag = {"executed": "executado", "failed": "FALHOU", "unknown": "NÃO SEI"}[result]
    return f"[{tag}] {outcome['reply']}"


def approve(
    action_id: str,
    queue: Optional[GateQueue] = None,
    runner: Optional[Runner] = None,
) -> str:
    """Approve and execute a queued action. Returns a user-facing line."""
    return _format_decision(action_id, decide(action_id, "approved", queue, runner))


def reject(action_id: str, queue: Optional[GateQueue] = None) -> str:
    """Reject (drop) a queued action without running it."""
    return _format_decision(action_id, decide(action_id, "rejected", queue))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gate_review",
        description="Revisar ações de background aguardando aprovação.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="lista as ações pendentes (default)")
    p_app = sub.add_parser("approve", help="aprova e executa uma ação")
    p_app.add_argument("id")
    p_rej = sub.add_parser("reject", help="rejeita (descarta) uma ação")
    p_rej.add_argument("id")

    args = parser.parse_args(argv)

    if args.command == "approve":
        print(approve(args.id))
    elif args.command == "reject":
        print(reject(args.id))
    else:  # list / None
        print(render_list())
    return 0


if __name__ == "__main__":
    sys.exit(main())
