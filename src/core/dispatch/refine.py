"""
Prompt-enrichment trigger (idea #30, s74).

v1 is deliberately OPT-IN per turn: enrichment only ever runs when the
user asks for it with an explicit prefix — never automatically. That is
the design Pedro approved: zero extra latency on the normal path, zero
risk on the voice path, and the expanded prompt always comes back
staged (sim/não) before anything executes.

The separator (`:` or `,`) is part of the contract — "refina o texto"
is a normal sentence, "refina: o texto" is a command. Same philosophy
as PLANNER_TRIGGER_RE's explicit verb.
"""
from __future__ import annotations

import re


REFINE_RE = re.compile(
    r"^(?:"
    # pt-BR
    r"refina"
    r"|melhora\s+o\s+prompt"
    r"|enriquece\s+o\s+prompt"
    # EN
    r"|refine"
    r")\s*[:,]\s*(?P<request>.+)$",
    re.IGNORECASE | re.DOTALL,
)
