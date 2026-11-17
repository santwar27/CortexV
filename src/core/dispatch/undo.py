"""
Undo dispatch — "Cortex, desfaz isso".

Matches an explicit request to reverse the LAST side-effecting action. The
orchestrator's `_handle_undo` reads the Action Ledger, finds the most recent
not-yet-undone action, and runs its recorded `reverse` spec (delete_event /
delete_reminder). Irreversible actions (a sent email) are named honestly
rather than silently skipped.

Kept narrow and anchored so it never eats normal conversation: it's a short
command, not a topic. "cancela" only matches the specific "cancela a última
ação" phrasing — bare "cancela"/"não" stays with the gate-denial path.
"""
from __future__ import annotations

import re

UNDO_RE = re.compile(
    r"^(?:"
    # PT — "desfaz / desfazer / desfaça (isso / essa / aí / a última (ação))"
    r"desfa(?:z(?:er)?|ça)"
    r"(?:\s+(?:isso|essa|a[íi]|a\s+[úu]ltima(?:\s+a[çc][ãa]o)?))?"
    r"|"
    # PT — "cancela / cancelar a última (ação)"
    r"cancela(?:r)?\s+a\s+[úu]ltima(?:\s+a[çc][ãa]o)?"
    r"|"
    # EN — "undo (that / it / the last (action))"
    r"undo(?:\s+(?:that|it|the\s+last(?:\s+action)?))?"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
