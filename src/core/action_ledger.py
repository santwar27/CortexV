"""
Action Ledger — append-only journal of every side-effecting action.

Whenever Cortex does something with a real-world consequence (sends an email,
creates a calendar event, files a reminder), this records ONE line: when, what
tool, the (truncated) args, the created resource id when we have one, the risk
level (from tool_risk), and — when an action is reversible — a `reverse` SPEC
describing what would undo it.

This module only WRITES the journal and never changes behaviour. Two payoffs:
  1. An audit trail of consequential actions, complementing the gate/lease/
     gatekeeper stack — "what did Cortex actually do, and when?".
  2. "Cortex, desfaz isso" — WIRED and live: src/core/dispatch/undo.py picks
     the last not-yet-undone entry and staging_mixin executes its `reverse`
     spec via the real clients (delete_event / delete_reminder / delete_note).
     This file stays the passive recorder; the undo CONSUMERS live there.

Actions irreversible by nature (`send_email` — can't be unsent; `mac_sleep`)
are recorded with reversible=False and the undo path names that honestly.

Storage: `data/action_ledger.jsonl` (override CORTEX_ACTION_LEDGER), under the
gitignored data/ dir, mirroring gate_queue / DreamStore / skill_proposals.

Run:  venv/bin/python -m src.core.action_ledger   (or `make ledger`)

Mirrors src/core/audit.py (module-level singleton + lock + fail-soft append);
`load_entries`/`render`/`main` mirror src/core/policy_detector.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.redaction import redact_payload

# Per-arg value cap — keeps an email body or long description from bloating the
# journal while leaving enough to identify the action.
_ARG_VALUE_CAP = 200


def _truncate_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy args, capping long string values. Never raises."""
    out: Dict[str, Any] = {}
    try:
        for k, v in (args or {}).items():
            if isinstance(v, str) and len(v) > _ARG_VALUE_CAP:
                out[k] = v[: _ARG_VALUE_CAP - 1] + "…"
            else:
                out[k] = v
    except Exception:  # noqa: BLE001 — ledger must never break a real action
        return {}
    return out


def reverse_spec(
    action: str, args: Dict[str, Any], result_id: Optional[str],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return (reversible, reverse) for an action. The reverse dict is a SPEC
    ({tool, args}) describing the undo operation — executed by the undo path
    (dispatch/undo.py → staging_mixin) when the user asks "desfaz isso".
    Conservative: an action is reversible only when a concrete handle
    (id/title) exists to target the undo. Unknown → not reversible."""
    args = args or {}
    if action == "apply_code_change":
        pid, path = result_id or args.get("id"), args.get("path")
        return (True, {"tool": "restore_code_change", "args": {"id": pid, "path": path}}) if pid and path else (False, None)
    if action == "create_event":
        # Google Calendar event id is a stable handle for a future delete.
        return (bool(result_id), {"tool": "delete_event", "args": {"event_id": result_id}} if result_id else None)
    if action == "create_reminder":
        # The id `make new reminder` answers with (s86) targets the one Cortex
        # made; by title an older namesake could go instead (s88, Codex lote 2).
        # Title only for lines recorded before the id — delete_reminder then
        # refuses when the name is ambiguous.
        title = args.get("title")
        if result_id:
            return (True, {"tool": "delete_reminder",
                           "args": {"reminder_id": result_id, "title": title}})
        return (bool(title), {"tool": "delete_reminder", "args": {"title": title}} if title else None)
    if action == "create_note":
        title = args.get("title")
        if args.get("note_id"):
            return (True, {"tool": "delete_note",
                           "args": {"note_id": args["note_id"], "title": title}})
        return (bool(title), {"tool": "delete_note", "args": {"title": title}} if title else None)
    if action == "mac_quit_app":
        # Legacy and uncertain lines cannot prove Cortex closed a running app.
        if args.get("was_running") is not True or args.get("uncertain"):
            return (False, None)
        app = args.get("app_name")
        return (bool(app), {"tool": "mac_open_app", "args": {"app_name": app}} if app else None)
    # send_email (irreversible), mac_sleep, and anything unknown.
    return (False, None)


class ActionLedger:
    """Append-only writer for the side-effect journal."""

    DEFAULT_PATH = "data/action_ledger.jsonl"

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("CORTEX_ACTION_LEDGER", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        self._local = threading.local()

    @contextmanager
    def observing(self, listener):
        """One optional listener on this thread, restored even on a failed turn."""
        previous = getattr(self._local, "listener", None)
        self._local.listener = listener
        try:
            yield
        finally:
            self._local.listener = previous

    def record(
        self,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        result_id: Optional[str] = None,
        status: str = "done",
        source: str = "inline_gate",
        risk: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one action record. Fail-soft: any I/O error is swallowed so
        the journal can never break the action it is recording.

        `extra` merges extra top-level fields (e.g. an undo record carries
        `undone_ts` pointing at the action it reversed)."""
        try:
            if risk is None:
                from src.core.tool_risk import risk_level
                risk = risk_level(action)
            reversible, reverse = reverse_spec(action, args or {}, result_id)
            # The undo spec carried the args again, unmasked: "senha é X" as a
            # reminder title was redacted in "args" and written whole in
            # "reverse" (s88, Codex lote 10, reproduced). Masked too; an undo
            # that has only the name to go on, and the name was a secret, is
            # given up rather than keeping the secret to find it by.
            if reverse is not None:
                masked = redact_payload(reverse)
                if masked != reverse:
                    has_id = any(k.endswith("_id") and v
                                 for k, v in (masked.get("args") or {}).items())
                    reverse = masked if has_id else None
                    reversible = reversible and has_id
            record: Dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action,
                # Border redaction (M6-A6) BEFORE truncation, so the mask
                # itself can't be clipped. The ledger is pure journal —
                # nothing executes from it, so redacting is free.
                "args": _truncate_args(redact_payload(args or {})),
                "result_id": result_id,
                "risk": risk,
                "status": status,
                "source": source,
                "reversible": reversible,
                "reverse": reverse,
            }
            if extra:
                record.update(extra)
            line = json.dumps(record, ensure_ascii=False)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:  # noqa: BLE001 — never propagate into the action path
            pass
        # The action succeeded even if the journal's own disk write failed.
        try:
            listener = getattr(self._local, "listener", None)
            if listener and status == "done":
                listener(action, redact_payload(args or {}))
        except Exception:
            pass

    def load_entries(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read every JSONL record, fail-soft (mirrors policy_detector)."""
        p = path or self.path
        records: List[Dict[str, Any]] = []
        try:
            with open(p, "r", encoding="utf-8") as f:
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


def undone_ts_set(entries: List[Dict[str, Any]]) -> set:
    """The set of action timestamps that an `undo` record has already
    reversed — so a second 'desfaz' can't double-undo the same action."""
    return {
        e.get("undone_ts")
        for e in entries
        if e.get("action") == "undo" and e.get("undone_ts")
    }


def last_action(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the most recent real action that has NOT been undone (skipping
    `undo` records themselves). None if there's nothing to act on — the
    target for "Cortex, desfaz isso" ('isso' = the last thing I did)."""
    undone = undone_ts_set(entries)
    for e in reversed(entries):
        if e.get("action") == "undo":
            continue
        if e.get("ts") in undone:
            continue
        return e
    return None


# Module-level singleton — single import, mirrors `from src.core.audit import audit`.
ledger = ActionLedger()


def render(entries: List[Dict[str, Any]], *, limit: int = 30) -> str:
    """Render the most recent actions as a terse pt-BR bullet digest (no tables)."""
    header = f"Action Ledger — {len(entries)} ações registradas:"
    if not entries:
        return header + "\n  Nada ainda — nenhuma ação com efeito colateral registrada."

    recent = entries[-limit:][::-1]  # newest first
    lines = [header, ""]
    for e in recent:
        ts = str(e.get("ts", ""))[:19].replace("T", " ")
        action = e.get("action", "?")
        risk = e.get("risk", "?")
        rid = e.get("result_id")
        rid_txt = f" id={rid}" if rid else ""
        rev = "↩︎ reversível" if e.get("reversible") else "✗ irreversível"
        status = e.get("status", "done")
        st_txt = "" if status == "done" else f" [{status}]"
        lines.append(f"  • {ts}  {action} ({risk}){rid_txt}  — {rev}{st_txt}")
    if len(entries) > limit:
        lines.append(f"  … (+{len(entries) - limit} mais antigas; use --limit)")
    lines += [
        "",
        "Read-only. O campo 'reverse' alimenta o \"Cortex, desfaz isso\" "
        "(evento/lembrete; e-mail é irreversível).",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show the side-effecting action ledger.")
    parser.add_argument("--limit", type=int, default=30, help="max recent actions to show")
    parser.add_argument("--path", default=None, help="ledger path (default data/action_ledger.jsonl)")
    args = parser.parse_args(argv)

    entries = ledger.load_entries(args.path)
    print(render(entries, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
