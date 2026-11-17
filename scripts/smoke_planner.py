"""
Smoke test for Plan-Execute Multi-step (Tier 3.1) against a live Ollama
instance.

Drives the planner end-to-end:
  1. decompose() a realistic pt-BR objective into a Plan
  2. render_preview() the staged plan
  3. execute() with MOCKED tools (we don't actually send emails or
     create calendar events on a real account during a smoke)
  4. summarise() the results back into a user-facing reply

No Palace needed — the planner module is pure logic; the orchestrator
integration is covered by unit tests.

Run from repo root with Ollama up:
  venv/bin/python scripts/smoke_planner.py

What to look for:
  1. Plan structure is sane — id ordering p1, p2, p3, with realistic
     depende_de edges.
  2. Tool selection matches the objective. No invented tool names.
  3. Refs like {p1} (the WHOLE upstream output, no .field accessor)
     appear when one step's output should feed the next. The decompose
     prompt explicitly forbids {p1.subject}/{p1.title}/etc. — tool
     outputs are plain text, not structured records. If a smoke run
     shows the LLM emitting `.field` anyway, the prompt has regressed.
  4. The summary reads as 2-4 concise sentences (NOT a bullet list).
"""
from __future__ import annotations

import os
import sys

# Make src.* importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama  # noqa: E402

from src.core.planner import (  # noqa: E402
    decompose,
    execute,
    render_preview,
    summarise,
)

MODEL = os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite")

ALLOWED_TOOLS = [
    "read_document", "write_document",
    "list_emails", "send_email",
    "list_events", "create_event",
    "create_reminder",
]

# Mix of single-step (should produce 1 step) and multi-step objectives.
OBJECTIVES = [
    "lê o documento ideias.txt",
    "agenda uma reunião com bruno amanhã às 14h e me lembre 30 minutos antes",
    "lê o documento agentes.txt, depois cria um lembrete pra revisá-lo amanhã às 10h",
]


def _section(title: str) -> None:
    print()
    print(f"--- {title} ---")


def _mock_tool_executor(name: str, args: dict) -> str:
    """Returns realistic-looking strings so summary has something to
    work with, but doesn't touch the real OS / network."""
    if name == "read_document":
        return f"[mock] conteúdo do arquivo '{args.get('filename')}': lorem ipsum."
    if name == "write_document":
        return f"[mock] saved to '{args.get('filename')}'."
    if name == "list_emails":
        return "[mock] 3 emails recentes na inbox."
    if name == "send_email":
        return f"[mock] email para {args.get('to')} enviado."
    if name == "list_events":
        return "[mock] próximos eventos: standup 10h, lunch 12h."
    if name == "create_event":
        return f"[mock] evento '{args.get('summary')}' criado."
    if name == "create_reminder":
        return f"[mock] lembrete '{args.get('title')}' criado."
    return f"[SYSTEM ERROR] tool desconhecido: {name}"


def smoke_one(objective: str, idx: int) -> None:
    print(f"\n{'='*70}")
    print(f"OBJETIVO {idx}: {objective}")
    print(f"{'='*70}")

    _section("Decompondo (Ollama — 1ª round-trip)")
    plan = decompose(
        objective,
        llm_chat_fn=ollama.chat,
        model=MODEL,
        allowed_tools=ALLOWED_TOOLS,
    )
    if plan is None:
        print("FAIL: decompose returned None")
        return
    print(f"  plano com {len(plan.passos)} passo(s)")

    _section("Preview (o que o user veria)")
    print(render_preview(plan))

    _section("Executando com tools MOCK")
    result = execute(plan, tool_executor=_mock_tool_executor)
    for r in result.results:
        marker = "✓" if r.success else ("⊘" if r.skipped else "✗")
        out_preview = (str(r.output)[:60] + "…") if r.output and len(str(r.output)) > 60 else r.output
        print(f"  {marker} {r.step_id}: {out_preview if r.success else r.error}")
    print(f"  plano success: {result.success}")

    _section("Summary (Ollama — 2ª round-trip)")
    summary = summarise(result, llm_chat_fn=ollama.chat, model=MODEL)
    print(summary)


def main() -> int:
    print(f"=== Cortex planner smoke ===")
    print(f"Model: {MODEL}")
    print(f"Allowed tools: {', '.join(ALLOWED_TOOLS)}")

    for i, obj in enumerate(OBJECTIVES, 1):
        smoke_one(obj, i)

    print()
    print("=== Calibration questions ===")
    print(
        "1. Single-step objective (#1) gera 1 passo só? Modelo entendeu\n"
        "   que não precisa decompor?\n"
        "2. Multi-step com refs (#2 e #3) tem depende_de correto e\n"
        "   referências {p<id>.<field>} fazendo sentido?\n"
        "3. Tools escolhidos batem com o objetivo? Sem inventar?\n"
        "4. Summary lê como 2-4 frases pt-BR diretas? Sem bullet list?\n"
        "5. Algum slug estranho ou ref quebrada?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
