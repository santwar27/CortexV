"""
System status — a deterministic, speakable health summary (Bug B-1, s57).

Why this exists: "status do sistema?" used to fall through to the LLM+retrieval
path, where the 8B wove a weakly-relevant past turn (a football match) into a
fabricated "status" (ideas_backlog Bug B — retrieval contamination). Routing it
to this deterministic handler means the query never touches retrieval: it
reports real, pulled facts (scheduled jobs, pending approvals, memory size).

`assemble` is PURE (template over a signals dict — no I/O, fully testable).
`gather` collects each signal fail-soft: a broken source degrades to a missing
line, never an error reply. Speakable: short pt-BR prose, no emoji / bullets /
tables, so the voice `say` path reads it cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import heartbeat
from src.core.i18n import t

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PALACE_DB = _REPO_ROOT / "data" / "palace.db"


# --- gather (I/O, fail-soft) ----------------------------------------------


def _gather_jobs() -> Optional[Dict[str, Any]]:
    """Scheduled-job watchdog (mtime-based, no network). Returns total count
    and the names of any stale ones, or None if the check itself failed."""
    try:
        statuses = heartbeat.check()
        return {
            "total": len(statuses),
            "stale": [s.name for s in statuses if s.stale],
        }
    except Exception:
        return None


def _gather_gate_pending() -> Optional[int]:
    """Count of actions parked awaiting approval (local JSONL)."""
    try:
        from src.core.gate_queue import gate_queue

        return len(gate_queue.pending())
    except Exception:
        return None


def _gather_palace_docs(db: Path = _PALACE_DB) -> Optional[int]:
    """Document count in the Palace (read-only). None if absent/unreadable.
    Goes through Palace(read_only=True) — M7-A9e: no raw SQL on the palace
    file outside the Palace class."""
    try:
        if not db.exists():
            return None
        from src.memory.palace import Palace

        palace = Palace(db_path=str(db), read_only=True)
        try:
            return palace.counts()["documentos"]
        finally:
            palace.close()
    except Exception:
        return None


def gather() -> Dict[str, Any]:
    return {
        "jobs": _gather_jobs(),
        "gate_pending": _gather_gate_pending(),
        "palace_docs": _gather_palace_docs(),
    }


# --- assemble (pure) -------------------------------------------------------


def assemble(signals: Dict[str, Any], lang: str = "pt") -> str:
    """Render the signals dict into speakable prose in `lang` (PT default).
    Pure."""
    en = lang == "en"
    parts: List[str] = []

    jobs = signals.get("jobs")
    stale = jobs.get("stale") if isinstance(jobs, dict) else None
    if stale:
        names = ", ".join(stale)
        n = len(stale)
        if en:
            noun = "scheduled job" if n == 1 else "scheduled jobs"
            parts.append(f"Heads up: {n} {noun} didn't run in the expected window: {names}.")
        else:
            noun = "trabalho agendado" if n == 1 else "trabalhos agendados"
            verb = "rodou" if n == 1 else "rodaram"
            parts.append(f"Atenção: {n} {noun} não {verb} na janela esperada: {names}.")
    elif isinstance(jobs, dict):
        parts.append(t(lang,
                       "Tudo certo por aqui. Os trabalhos agendados rodaram dentro da janela.",
                       "All good here. The scheduled jobs ran within their window."))
    elif all(signals.get(k) is None for k in ("jobs", "gate_pending", "palace_docs")):
        # Nothing could be checked: saying "tudo certo" here was a health
        # claim with no evidence behind it (s88, Codex lote 13 — invariant 7).
        return t(lang, "Não consegui verificar o estado do sistema agora.",
                 "I couldn't check the system's state just now.")
    else:
        parts.append(t(lang, "Não consegui verificar os trabalhos agendados agora.",
                       "I couldn't check the scheduled jobs just now."))

    docs = signals.get("palace_docs")
    if isinstance(docs, int):
        if en:
            noun = "document" if docs == 1 else "documents"
            parts.append(f"Memory has {docs} {noun}.")
        else:
            noun = "documento" if docs == 1 else "documentos"
            parts.append(f"A memória tem {docs} {noun}.")

    gate = signals.get("gate_pending")
    if isinstance(gate, int) and gate > 0:
        if en:
            noun = "action awaiting" if gate == 1 else "actions awaiting"
            parts.append(f"You have {gate} {noun} approval.")
        else:
            noun = "ação esperando" if gate == 1 else "ações esperando"
            parts.append(f"Você tem {gate} {noun} aprovação.")
    elif isinstance(gate, int):
        parts.append(t(lang, "Nenhuma ação pendente de aprovação.",
                       "No actions pending approval."))

    return " ".join(parts)


def summary(lang: str = "pt") -> str:
    """Deterministic, fail-soft system status — the dispatch entry point."""
    return assemble(gather(), lang=lang)
