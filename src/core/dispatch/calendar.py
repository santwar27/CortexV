"""
Calendar regex dispatch patterns.

Extracted from `src/core/orchestrator.py`. The orchestrator re-binds
this as a class attribute (`_CALENDAR_TRIGGER_RE`) for back-compat
with existing tests.

Why regex-first dispatch instead of LLM tool-call: in production
traces the local 8B tool-call path either hallucinated AppleScript
or refused outright ("não posso criar eventos no calendário do
Google") despite `create_event` being a registered tool. The regex
stages the action deterministically and lets the existing
`_pending_action` confirmation flow take over — same path the
reminder dispatch uses.

Body shape is intentionally identical to the reminder body, so the
calendar handler reuses `dispatch.reminders.REMINDER_BODY_RE` and
the orchestrator's `_parse_reminder_due` to extract <title> +
<start>. end_iso defaults to start + 1h.
"""
from __future__ import annotations

import re


CALENDAR_TRIGGER_RE = re.compile(
    r"^(?:"
    # pt-BR: verb + optional "um evento" + (no|na|ao|em) + (meu)?
    # + (calendar/calendario/agenda) + optional "do google"
    r"(?:adicione|adiciona|adicionar|coloca|coloque|colocar|"
    r"marca|marque|marcar|cadastra(?:r)?)"
    r"\s+(?:um\s+(?:evento|compromisso|encontro)\s+)?"
    r"(?:no|na|ao|em)\s+(?:meu\s+|minha\s+)?"
    r"(?:calend[áa]rio|calendar|agenda)"
    r"(?:\s+do\s+google)?"
    r"\s*[:,]?\s*"
    # pt-BR: "agenda/agende/agendar [um evento] [de|para|pra|:]"
    r"|agend[ae]r?\s+(?:um\s+)?(?:evento|compromisso|encontro)?"
    r"\s*(?:de\s+|para\s+|pra\s+|:)?\s*"
    # pt-BR: "novo evento [:|-]" or "cria(r) evento"
    r"|(?:novo|cria(?:r)?|criar?)\s+evento\s*[:,\-]?\s*"
    # EN: "add [an event] to/on [my] (calendar|schedule)"
    r"|add\s+(?:an?\s+event\s+)?(?:to|on)\s+(?:my\s+)?(?:calendar|schedule)"
    r"\s*[:,]?\s*"
    # EN: "schedule [:|,] <body>". Unlike the pt-BR side we do NOT eat
    # "meeting/event/appointment" — in EN those words ARE the title
    # ("schedule a meeting with Bruno" → title is "meeting with Bruno").
    r"|schedule\s*[:,]?\s*"
    # EN: "create [a/an] [calendar] event [:|-]"
    r"|create\s+(?:an?\s+)?(?:calendar\s+)?event\s*[:,\-]?\s*"
    r")(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)


# Content-FIRST phrasing — the calendar keyword comes AFTER the details:
# "marca reunião com Bruno às 15h no calendário", "coloca esses eventos na
# agenda", "schedule lunch tomorrow on my calendar". CALENDAR_TRIGGER_RE
# above only matches keyword-FIRST ("adicione ao calendário: <body>"), so
# natural content-first requests fell through to the flaky 8B tool-call
# path (discovered 2026-06-02: "adicione esses jogos ao meu calendário"
# lumped 3 games into one event with a stale time). This one captures the
# <body> between the verb and the trailing calendar keyword. The keyword
# tail is the anchor that keeps it from stealing non-calendar commands
# ("adiciona leite na lista", "marca o ponto" don't match). The orchestrator
# tries CALENDAR_TRIGGER_RE first, then this as a fallback.
CALENDAR_TRIGGER_CONTENT_RE = re.compile(
    r"^(?:"
    # pt-BR verbs
    r"(?:marca|marque|marcar|adiciona|adicione|adicionar|"
    r"coloca|coloque|colocar|cadastra(?:r)?|agenda|agende|agendar|"
    r"cria(?:r)?|criar)"
    # EN verbs
    r"|(?:add|schedule|create)"
    r")\s+"
    r"(?P<body>.+?)"
    r"\s+(?:"
    # pt-BR tail: no/na/ao/em [meu] calendário/agenda [do google]
    r"(?:n[oa]|ao|em)\s+(?:meu\s+|minha\s+)?(?:calend[áa]rio|agenda)"
    r"(?:\s+do\s+google)?"
    # EN tail: on/to/in [my] calendar/schedule/agenda
    r"|(?:on|to|in)\s+(?:my\s+)?(?:calendar|schedule|agenda)"
    r")\s*[.!?]*$",
    re.IGNORECASE | re.DOTALL,
)


# Vague references to things discussed earlier — "esses jogos", "essas
# reuniões", "aquele compromisso". When a calendar request points at these
# AND carries no inline date, the local 8B guesses: it was seen lumping
# three games into ONE event with a stale time (2026-06-02). The handler
# uses this to ask for specifics instead of guessing. Demonstratives only
# (esse/essa/este/esta/aquele/aquela ± plural) — "os jogos" is too broad and
# would catch "o jogo de amanhã". Only consulted AFTER a concrete date fails
# to parse, so a demonstrative that DOES carry a date ("esse sábado às 15h")
# never reaches it.
VAGUE_CALENDAR_RE = re.compile(
    r"\b(?:esse|essa|esses|essas|este|esta|estes|estas|"
    r"aquele|aquela|aqueles|aquelas)\b",
    re.IGNORECASE,
)


# Verb-first, with NO calendar keyword: "marca dentista amanhã às 10" (s85,
# fila 4.2). Neither regex above takes it — TRIGGER wants "no calendário",
# CONTENT wants the keyword at the end — so it went to the 8B, which staged
# the event on 23/05, 20/05, 19/05 at -03:00 for a "tomorrow" of 16/10, six
# times in six. The <body> goes to the same deterministic handler, which
# returns None unless the body ends in a time it can parse: that is what
# keeps the noun out ("qual a marca do seu carro?", "marca boa é a Nike").
# ("agenda/agendar dentista amanhã às 10" is already TRIGGER's.)
#
# Only the articles are skipped: "marque consulta no dentista …" keeps
# "consulta no dentista" as the title, which says more on the calendar.
CALENDAR_VERB_RE = re.compile(
    r"^(?:pode\s+)?"               # start of the sentence, optional "pode "
    r"(?:marcar|marque|marca)"     # the verb forms
    r"\s+(?:(?:os|as|o|a|um|uma)\s+)?"   # an article before the title, if any
    r"(?P<body>.+)$",              # the rest: title + when (the handler parses it)
    re.IGNORECASE | re.DOTALL,
)
