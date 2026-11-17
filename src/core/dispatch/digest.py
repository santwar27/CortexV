"""
"O que mudou desde a última vez" dispatch (Onda 4a).

Matches a request for the proactive delta digest. The orchestrator's
`_handle_whats_changed` assembles a deterministic pull-based summary (new
emails, Studier briefings, actions taken, upcoming events) and advances the
"last seen" marker. Pull-based + template → zero hallucination.

Anchored + narrow so it doesn't eat normal conversation: it's a short
"catch me up" command, not a topic.
"""
from __future__ import annotations

import re

WHATS_CHANGED_RE = re.compile(
    r"^(?:"
    # PT — "o que mudou (desde…)", "o que há/tem de novo", "novidades",
    #       "me atualiza", "o que eu perdi"
    r"o\s+que\s+(?:mudou|h[áa]\s+de\s+novo|tem\s+de\s+novo|aconteceu\s+de\s+novo)"
    r"|(?:quais\s+(?:as\s+)?)?novidades"
    r"|me\s+atualiza(?:r)?|me\s+p[õo]e\s+em\s+dia"
    r"|o\s+que\s+(?:eu\s+)?perdi"
    r"|"
    # EN — "what changed / what's new / what did I miss / catch me up"
    r"what(?:'?s| has)?\s+(?:changed|new|happened)"
    r"|what\s+did\s+i\s+miss"
    r"|catch\s+me\s+up"
    r")(?:\s+desde\s+.+|\s+since\s+.+)?[\s.!?]*$",
    re.IGNORECASE,
)
