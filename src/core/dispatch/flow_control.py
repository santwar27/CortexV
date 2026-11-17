"""
Conversation flow-control regexes — patterns matched against user
replies to an existing prompt, not against new commands.

These four are siblings by function:
  - APPROVAL_RE / DENIAL_RE — "sim" / "não" replies to a staged
    Agent action confirmation prompt. The orchestrator picks one of
    the three branches (approve / deny / fall through to normal
    processing) based on which (if any) matches first.
  - SILENCE_RE — "cala a boca" / "stop" mid-utterance interrupt.
    Drops a flag file the voice listener checks between SSE chunks
    and kills any active `say` subprocess.
  - ALL_REF_RE — "todos" / "all" reply used in batch operations
    (approve-all / reject-all studies).

Patterns kept verbatim from the previous `commands.py` grab-bag.
Edit here, not in the orchestrator.
"""
from __future__ import annotations

import re


# Approval/denial regexes tolerate any common wrapping punctuation
# users might type: quotes, parentheses, backticks. The CLI/iOS
# shortcut sometimes carry these over from tutorial examples and
# bare "'sim'" or "(sim)" used to silently break the match.
# Expanded 2026-05-19: typo tolerance + Brazilian/mineiro variants
# (beleza, bão, fechado, ta bom, isso aí, manda ver, pode crer).
# Expanded 2026-06-11 (M1-A4): the punctuated-ack family — "valeu!",
# "show!", "perfeito.", "blz", "got it" — replies real users give to a
# confirmation prompt. They used to die as "[SYSTEM ERROR] Empty input."
# (strip_vocatives ate them); after that fix they'd fall to the
# anything-else branch, which DROPS the staged action — worse. With an
# action staged the prompt just asked sim/não, so these read as YES.
# Trailing comma allowed too ("ok," / "okay,").
APPROVAL_RE = re.compile(
    r"^[\"'`(\[]?(?:"
    r"sim|ssim|"
    r"yes|yse|yess|ys|"
    r"ok|okay|kk|"
    r"claro|"
    r"autorizo|autoriza|"
    r"permito|permite|"
    r"pode|pode mandar|pode crer|"
    r"vai|"
    r"libera|liberado|"
    r"tudo bem|tudo ok|"
    r"beleza|belez[ao]|blz|"
    r"b[aã]o|bao|"
    r"fechad[oa]|fechou|"
    r"ta bom|t[áa] bom|tabom|"
    r"isso a[ií]|"
    r"manda ver|"
    r"manda|"
    r"valeu|vlws?|"
    r"show|"
    r"perfeito|"
    r"got it|"
    r"confirmo|confirma"
    r")[\"'`)\]]?[.!,]?$",
    re.IGNORECASE,
)

# Expanded 2026-05-19: typo tolerance + colloquial variants
# (esquece, deixa, nada disso, melhor não).
DENIAL_RE = re.compile(
    r"^[\"'`(\[]?(?:"
    r"n[aã]o|naum|naao|"
    r"no|nope|nada|"
    r"cancela|cancel|cancelar|"
    r"negativo|"
    r"nem|"
    r"deixa(?:\s+pra\s+l[aá])?|deixa quieto|"
    r"esquece|esquec[ea]|"
    r"nada disso|"
    r"melhor n[aã]o|n[aã]o quero|"
    r"abort|aborta"
    r")[\"'`)\]]?[.!]?$",
    re.IGNORECASE,
)

# Interrupt the voice listener's current speech mid-utterance.
# Drops a flag file the listener checks between SSE chunks AND kills
# any active `say` subprocess. Use `cx cala` or any of the variants.
SILENCE_RE = re.compile(
    r"^(?:cala(?:r)?|cale(?:\s+a\s+boca)?|"
    r"sil[êe]ncio|silencio|"
    r"fica\s+quieto|quieto|chega|"
    r"para\s+de\s+falar|"
    r"stop(?:\s+(?:talking|speaking))?|shut\s+up)$",
    re.IGNORECASE,
)

# Batch reference — user replied "todos" / "all" to a list prompt
# (approve-all studies, reject-all studies, etc).
ALL_REF_RE = re.compile(
    r"^(?:todos|tudo|all|everything)$",
    re.IGNORECASE,
)
