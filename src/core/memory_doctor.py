"""
Memory doctor — proactive anti-poisoning audit of the Palace.

The conversation store has two REACTIVE guards (memory.CortexMemory):
`_is_refusal` and the `_ERROR_MARKERS` filter stop NEW poison at write time,
and `_user_side` trims the assistant turn out of retrieval. But they only
protect writes from the moment they shipped — rows stored BEFORE a guard
existed, or poison they don't model, sit in the Palace untouched. Two
self-poisoning bugs surfaced in a single week (refusal echo on "quem
descobriu o Brasil?", s38-39; role-confusion echo on the personal profile,
s40), so a periodic SWEEP that hunts the patterns proactively is high-ROI.

What it flags (read-only — it never deletes):
  • refusals      — stored Cortex turns whose opening trips `_is_refusal`
                    (exactly what the write guard now blocks; these predate it).
  • errors        — stored turns carrying an engine `_ERROR_MARKERS` string.
  • role_confusion— Cortex turns that swap identity ("eu sou o Pedro") or
                    claim the user's profile as its own — the s40 failure mode.
  • contradictions— a (sujeito, predicado) with >1 distinct ACTIVE objeto
                    (add_fact should close priors; >1 active means a leak).
  • duplicates    — identical conversation turns stored 2+ times (retrieval
                    noise: the same text crowds the top-k).

Run:  venv/bin/python -m src.core.memory_doctor   (or `make memory-doctor`)

`scan` is pure (testable with synthetic docs/facts); `load_*`/`main` do
fail-soft I/O, mirroring src/core/policy_detector.py. The remedy is manual
and human-reviewed: `palace.delete_document(id)` / `palace.invalidate_fact(...)`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Single-source the refusal/error definitions from the write guard, so the
# doctor finds EXACTLY what CortexMemory would now refuse to store — no drift
# between "what we block going forward" and "what we hunt retroactively".
from src.core.memory import CortexMemory

# Role-confusion markers (s40). The 8B mis-attributed the user's profile as its
# own identity and claimed to BE the user. Tight, first-person/second-person
# identity-swap phrasings — a normal informative turn won't trip these. Read-
# only + human-reviewed, so a rare false positive is cheap.
_ROLE_CONFUSION_MARKERS: Tuple[re.Pattern, ...] = (
    re.compile(r"\beu sou (?:o |a )?pedro\b", re.IGNORECASE),
    re.compile(r"\bmeu nome é pedro\b", re.IGNORECASE),
    re.compile(r"\bvocê é (?:o |a )?cortex\b", re.IGNORECASE),
    re.compile(r"\bvoce é (?:o |a )?cortex\b", re.IGNORECASE),
    re.compile(r"\bminha configuração de perfil pessoal\b", re.IGNORECASE),
    re.compile(r"\bmeu perfil pessoal\b", re.IGNORECASE),
)

_CONVERSATION_TIPO = CortexMemory.DOC_TIPO  # "conversation"
_DEFAULT_DB = CortexMemory.DEFAULT_PALACE_DB


@dataclass
class Finding:
    """One flagged item. `ref` is what the human acts on (doc id or fact key)."""
    ref: str
    snippet: str


@dataclass
class DoctorReport:
    docs_scanned: int = 0
    facts_scanned: int = 0
    refusals: List[Finding] = field(default_factory=list)
    errors: List[Finding] = field(default_factory=list)
    role_confusion: List[Finding] = field(default_factory=list)
    contradictions: List[Finding] = field(default_factory=list)
    duplicates: List[Finding] = field(default_factory=list)

    def total(self) -> int:
        return (
            len(self.refusals) + len(self.errors) + len(self.role_confusion)
            + len(self.contradictions) + len(self.duplicates)
        )


def _cortex_side(texto: str) -> str:
    """Return the assistant half of a stored 'User: …\\nCortex: …' turn.
    Falls back to the whole text if the marker is absent (defensive)."""
    marker = "\nCortex:"
    idx = texto.find(marker)
    return texto[idx + len(marker):].strip() if idx != -1 else texto


def _snippet(text: str, cap: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def scan(
    documents: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
) -> DoctorReport:
    """Fold conversation docs + active facts into a poison report. Pure.

    `documents` are conversation rows ({id, texto, …}); `facts` are ACTIVE
    triples ({sujeito, predicado, objeto, …}). Nothing is mutated."""
    report = DoctorReport(docs_scanned=len(documents), facts_scanned=len(facts))

    seen_text: Dict[str, int] = defaultdict(int)
    dup_reported: set = set()

    for doc in documents:
        texto = doc.get("texto") or ""
        ref = str(doc.get("id", "?"))
        cortex = _cortex_side(texto)

        # Refusal / error echoes (single-sourced from the write guard).
        if CortexMemory._is_refusal(cortex):
            report.refusals.append(Finding(ref, _snippet(cortex)))
        if any(m in texto for m in CortexMemory._ERROR_MARKERS):
            report.errors.append(Finding(ref, _snippet(texto)))

        # Identity / profile mis-attribution (s40).
        if any(p.search(cortex) for p in _ROLE_CONFUSION_MARKERS):
            report.role_confusion.append(Finding(ref, _snippet(cortex)))

        # Exact-duplicate stored turns — flag from the 2nd occurrence on, once.
        seen_text[texto] += 1
        if seen_text[texto] >= 2 and texto not in dup_reported:
            dup_reported.add(texto)
            report.duplicates.append(Finding(ref, _snippet(texto)))

    # Contradictions: an active (sujeito, predicado) with >1 distinct objeto.
    by_sp: Dict[Tuple[str, str], set] = defaultdict(set)
    for f in facts:
        key = (str(f.get("sujeito")), str(f.get("predicado")))
        by_sp[key].add(str(f.get("objeto")))
    for (suj, pred), objs in sorted(by_sp.items()):
        if len(objs) > 1:
            report.contradictions.append(Finding(
                f"{suj}/{pred}", " ⨯ ".join(sorted(objs))
            ))

    return report


def render(report: DoctorReport, *, limit: int = 20) -> str:
    """Render the report as a terse pt-BR bullet digest (no tables)."""
    header = (
        f"Memory doctor — varri {report.docs_scanned} conversas + "
        f"{report.facts_scanned} fatos ativos no Palace:"
    )
    if report.total() == 0:
        return header + "\n  ✓ Limpo — nenhum padrão-veneno encontrado."

    sections = [
        ("Recusas armazenadas (poison de retrieval — o guard bloquearia hoje)",
         report.refusals, "delete_document"),
        ("Mensagens de erro do engine armazenadas",
         report.errors, "delete_document"),
        ("Confusão de papéis / identidade (modo de falha da s40)",
         report.role_confusion, "delete_document"),
        ("Fatos contraditórios (>1 objeto ativo p/ mesma sujeito+predicado)",
         report.contradictions, "invalidate_fact"),
        ("Conversas duplicadas exatas (ruído no top-k)",
         report.duplicates, "delete_document"),
    ]

    lines = [header]
    for title, findings, remedy in sections:
        if not findings:
            continue
        lines += ["", f"⚠ {title} — {len(findings)}:"]
        for f in findings[:limit]:
            lines.append(f"  • [{f.ref}] {f.snippet!r}")
        if len(findings) > limit:
            lines.append(f"  … (+{len(findings) - limit} mais)")
        lines.append(f"    remédio: palace.{remedy}(…)")

    lines += [
        "",
        "Read-only — nada foi apagado. Revise e remova à mão "
        "(palace.delete_document / invalidate_fact).",
    ]
    return "\n".join(lines)


def read_palace_data(db_path: str = _DEFAULT_DB) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read conversation docs + active facts from the Palace — raising when
    it can't. Opens GENUINELY read-only (M3-A7: SQLite mode=ro, _init_schema
    skipped — a scanner auditing a suspect DB must not mutate the patient,
    and must not CREATE an empty DB at a mistyped path)."""
    from src.memory.palace import Palace
    palace = Palace(db_path=db_path, embed_fn=None, read_only=True)
    try:
        # A high limit so we scan the whole conversation history, not the
        # list_documents default page of 100.
        docs = palace.list_documents(tipo=_CONVERSATION_TIPO, limit=1_000_000)
        facts = palace.list_facts(include_invalidated=False)
    finally:
        close = getattr(palace, "close", None)
        if callable(close):
            close()
    return docs, facts


def load_palace_data(db_path: str = _DEFAULT_DB) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """read_palace_data, fail-soft: a missing or unreadable DB degrades to
    empty lists, never a crash. Not for a verdict — main() uses the raising
    one, so an unreadable DB is never reported clean."""
    try:
        return read_palace_data(db_path)
    except Exception:  # noqa: BLE001 — a read-only audit must never crash
        return [], []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan the Palace for memory-poisoning patterns (read-only)."
    )
    parser.add_argument("--db", default=_DEFAULT_DB, help="palace.db path")
    parser.add_argument("--limit", type=int, default=20, help="max items per category")
    args = parser.parse_args(argv)

    # Couldn't read is not clean: a missing, unreadable or corrupt DB was
    # reported "✓ Limpo" with exit 0 (s88, Codex lote 6, reproduced).
    try:
        docs, facts = read_palace_data(args.db)
    except Exception as e:  # noqa: BLE001
        print(f"✗ Não consegui ler o Palace em {args.db} ({type(e).__name__}) "
              "— nada foi varrido.")
        return 2
    report = scan(docs, facts)
    print(render(report, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
