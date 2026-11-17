"""
Reminder regex dispatch patterns.

Extracted from `src/core/orchestrator.py` so the routing class stays
focused on flow control rather than pattern bodies. The orchestrator
re-binds these constants as class attributes so existing tests that
reach in via `CortexOrchestrator._REMINDER_TRIGGER_RE` keep working.

Pattern philosophy: regex-first dispatch beats the LLM tool-call
fallback for "me lembre da ida ao lago amanhã às 4h" because Llama
3.1 8B Q4 has been seen to hallucinate "you already have a meeting
scheduled" instead of staging the reminder. The regex stages
deterministically; ambiguous bodies fall through to the LLM.
"""
from __future__ import annotations

import re

from src.core.clock import RELATIVE_OFFSET_PREFIX


# "lembra" is also the verb of remembering (s84, class 2 of the adversarial
# sweep). "lembra daquele dia?", "lembra de mim?" and "lembra quando fomos à
# praia?" are reminiscence, and they used to stage a reminder. Two cues keep
# them in chat:
#   _REMINISCE — what follows "lembra" right away: "daquele", "de quando",
#                "quando", "de como", "de mim". This is the cue that survives
#                voice, where the transcript often has no "?".
#   question with no time — a bare "lembra …?" with no day or hour is asking,
#                not booking. "lembra da reunião amanhã às 10?" still books
#                (S3 asks before creating anything).
_REMINISCE = (r"\s+(?:d?aquel[ea]s?\b|de\s+quando\b|quando\b|de\s+como\b|como\b"
              r"|de\s+mim\b)")
_TIME_CUE = r"(?:amanh[ãa]|hoje|\b[àa]s?\s+\d|\b\d{1,2}\s*(?:h|:\d\d)\b)"

# Matches the most common pt-BR + EN phrasings for "remind me to X".
# Group `body` captures everything after the trigger verb.
#
# "me lembra de/que …" (s84) is how a reminder is most often asked in Brazil,
# and it had no route: it went to plain chat, the path whose 8B used to say
# "lembrete criado" without creating one. It requires "de" or "que" — "me
# lembra da minha avó" ("reminds me of my grandma") stays chat — and shares
# the reminiscence cues ("me lembra de quando eu era criança").
REMINDER_TRIGGER_RE = re.compile(
    r"^(?:"
    r"me\s+lembre(?:\s*[–-]\s*me)?\s+(?:de|da|do|que|:)?\s*"
    r"|me\s+lembra(?!" + _REMINISCE + r")\s+"
    r"(?:(?:de|que)\s+|(?=" + RELATIVE_OFFSET_PREFIX + r"))"
    r"|lembre(?:\s*[–-]\s*me)?\s+(?:de|da|do|que)?\s*"
    r"|lembra(?!" + _REMINISCE + r")(?!(?=.*\?\s*$)(?!.*" + _TIME_CUE + r"))"
    r"(?:\s*[–-]\s*me)?\s+(?:de|da|do|que)?\s*"
    r"|cria(?:r)?\s+(?:um\s+)?lembrete\s+(?:de|da|do|para?|pra)?\s*"
    r"|set\s+(?:a\s+)?reminder\s+(?:for|to)?\s*"
    r"|remind\s+me\s+(?:to|about)?\s*"
    r")(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Splits '<title> <time_expr>' from a reminder body. time_expr is the
# temporal clause at the END (amanhã às 4h, hoje às 18h, amanhã, …).
# Works via lazy title capture + anchored time_expr at $. Falls back
# gracefully: if the body has no recognisable temporal clause the whole
# string becomes the title and due_iso is left as None.
#
# Also reused by the calendar dispatch (a calendar body has the same
# shape as a reminder body: title + optional time).
REMINDER_BODY_RE = re.compile(
    r"^(?P<title>.+?)\s+"
    r"(?P<time_expr>"
    # day + optional time  ("amanhã às 4h", "hoje às 18h30", "amanhã")
    r"(?:(?:amanhã|amanha|hoje)\s+)?(?:[àa]s?\s+)?\d{1,2}(?:[h:]\d{0,2})?"
    r"\s*(?:da\s+(?:manh[ãa]|manha|tarde|noite|madrugada))?"
    r"(?:\s+horas?)?"
    # or just a day word alone
    r"|(?:amanhã|amanha|hoje)"
    r")\s*$",
    re.IGNORECASE,
)
