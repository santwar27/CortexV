"""
Planner regex trigger.

Extracted from `src/core/orchestrator.py`. The orchestrator re-binds
this as `_PLANNER_TRIGGER_RE` for back-compat.

Conservative trigger: requires an explicit planning verb so casual
mentions of "plano" in normal conversation don't kick off a multi-step
LLM decomposition. The body after the verb becomes the objective
passed to `src.core.planner.decompose`.
"""
from __future__ import annotations

import re


PLANNER_TRIGGER_RE = re.compile(
    r"^(?:"
    # pt-BR
    r"planeja(?:r)?\s*[:,]?\s*"
    r"|planeje\s*[:,]?\s*"
    r"|decomp[õo]e(?:r)?\s*[:,]?\s*"
    r"|monte\s+um\s+plano\s+para\s+"
    r"|cria(?:r)?\s+um\s+plano\s+para\s+"
    # EN
    r"|plan\s*[:,]?\s*"
    r"|make\s+a\s+plan\s+(?:for|to)\s+"
    r"|decompose\s*[:,]?\s*"
    r")(?P<objective>.+)$",
    re.IGNORECASE | re.DOTALL,
)
