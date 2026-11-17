"""
System-status dispatch (Bug B-1, s57).

Routes a "how's the system?" query to the deterministic status handler instead
of the LLM+retrieval path, where a weakly-relevant past turn was being woven
into a fabricated status (ideas_backlog Bug B — retrieval contamination).

NARROW by design: it must NOT eat a casual greeting ("oi, como você está?",
"tudo bem?"). So it requires an explicit system noun (sistema / cortex), OR a
health/diagnostic word, OR a "funcionando / rodando / operacional" phrasing —
never bare "está tudo bem" / "como você está", which the reflex/greeting path
owns. Anchored both ends so trailing words ("are you working on X") fall
through to normal routing.
"""
from __future__ import annotations

import re

SYSTEM_STATUS_RE = re.compile(
    r"^(?:"
    # --- PT ---
    # "status do sistema/cortex", "qual o status do sistema", "como está o status…"
    r"(?:qual\s+(?:[éeo]\s+)?(?:o\s+)?|como\s+(?:est[áa]|t[áa]|anda)\s+(?:o\s+)?)?"
    r"status\s+(?:d[oa]\s+)?(?:sistema|cortex)"
    # "como está/tá/anda o sistema/cortex"
    r"|como\s+(?:est[áa]|t[áa]|anda)\s+o\s+(?:sistema|cortex)"
    # "está/tá tudo funcionando/rodando/operacional" (system phrasing, not "bem")
    r"|(?:est[áa]|t[áa])\s+tudo\s+(?:funcionando|rodando|operacional)"
    # "diagnóstico (do sistema)", "saúde do sistema", "checar o sistema"
    r"|diagn[óo]stico(?:\s+d[oa]\s+sistema)?"
    r"|sa[úu]de\s+d[oa]\s+sistema"
    r"|checa(?:r)?\s+o\s+sistema"
    r"|"
    # --- EN ---
    r"system\s+status|status\s+report"
    r"|are\s+you\s+(?:ok|okay|healthy|operational|running)"
    r"|is\s+everything\s+(?:ok|okay|working|running|operational)"
    r"|health[\s-]?check|diagnostics?"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
