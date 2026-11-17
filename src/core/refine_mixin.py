"""
Prompt enrichment v1 for `CortexOrchestrator` (idea #30, s74).

Same mixin pattern as weather_mixin. The contract this path pins
(design approved by Pedro in s74):
  - OPT-IN only: runs exclusively behind the explicit REFINE_RE prefix
    ("refina: <pedido>"). No automatic enrichment, ever — the normal
    path keeps zero extra latency and the voice path zero risk.
  - The expanded prompt is ALWAYS staged for approval (the standard
    sim/não machinery, `_pending_action`) — v1 has no "high confidence
    executes directly".
  - "sim" runs the expanded prompt through the FULL normal pipeline
    (`_process_core`): deterministic dispatch first, and an S3 inside
    the expanded prompt stages again — refine is never a gate bypass.
  - Local-only (SOBERANIA regra 1): the meta-prompt carries ONLY the
    user's request text — no Palace, no memory, no profile.
  - LLM failure → honest error, nothing staged. (Unlike the planner,
    we do NOT fall through to free chat: the user explicitly asked for
    a refinement, and silently chatting about "refina: X" instead
    would mask the failure.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import ollama

from src.core.audit import audit
from src.core.ollama_client import KEEP_ALIVE as _KEEP_ALIVE, THINK as _THINK


_MAX_REQUEST_CHARS = 4000

_REFINE_SYSTEM = (
    "You rewrite a short, vague user request into one specific, "
    "actionable prompt.\n"
    "Rules:\n"
    "- Answer in the SAME language as the request.\n"
    "- Make the goal, scope and desired output format explicit.\n"
    "- NEVER invent facts, names, dates or numbers that are not in the "
    "request — leave genuine unknowns as explicit questions inside the "
    "prompt.\n"
    "- Output ONLY the rewritten prompt text. No preamble, no quotes, "
    "no explanation."
)


class RefineMixin:
    """Prompt-enrichment handler. Relies on the host providing
    `self.engine_analytical`, `self._pending_action`,
    `self._persist_pending_action` and `self._t`."""

    def _handle_refine(self, request: str) -> Optional[Tuple[str, str]]:
        request = (request or "").strip()
        if not request:
            return None  # bare "refina:" falls through — nothing to refine

        try:
            response = ollama.chat(
                model=self.engine_analytical.model_name,
                messages=[
                    {"role": "system", "content": _REFINE_SYSTEM},
                    {"role": "user", "content": request[:_MAX_REQUEST_CHARS]},
                ],
                # Temperature 0: the preview the user approves must be
                # reproducible — same lesson as the planner's decompose.
                options={"temperature": 0.0},
                keep_alive=_KEEP_ALIVE, think=_THINK,
            )
            refined = (response["message"]["content"] or "").strip()
        except Exception:
            refined = ""

        if not refined:
            audit.log("–", "refine", "error",
                      payload={"request": request[:60]})
            return (
                self._t(
                    "Não consegui refinar o pedido agora. "
                    "Pode mandar direto, sem o prefixo.",
                    "I couldn't refine the request right now. "
                    "Send it directly, without the prefix.",
                ),
                "REFINE_ERROR",
            )

        self._pending_action = {
            "type": "run_refined_prompt",
            "args": {
                "prompt": refined,
                # The approving "sim" arrives in a new turn whose language
                # heuristic can diverge — ride the request's language along
                # (M1-A8, same lesson as the planner staging).
                "lang": getattr(self, "_reply_lang", "pt") or "pt",
            },
        }
        self._persist_pending_action()
        audit.log("–", "refine", "staged", payload={
            "request": request[:60],
            "refined": refined[:200],
        })
        return (
            self._t(
                f"Prompt refinado:\n\n{refined}\n\nExecuto? (sim/não)",
                f"Refined prompt:\n\n{refined}\n\nRun it? (yes/no)",
            ),
            "REFINE_STAGED",
        )
