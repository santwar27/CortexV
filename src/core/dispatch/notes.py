"""
Apple Notes regex dispatch patterns.

Extracted from `src/core/orchestrator.py` so the routing class stays focused
on flow control. The orchestrator re-binds these as class attributes so
tests reaching in via `CortexOrchestrator._NOTE_CREATE_RE` keep working.

Notes are S2 (recoverable, local-only) → the orchestrator executes the match
DIRECTLY, no sim/não gate. Regex-first dispatch (no TOOL_SPECS entry) keeps
the 8B from fabricating an unrelated tool on note-talk (the s38 lesson).
"""
from __future__ import annotations

import re


# "anota X" / "anote que X" / "cria uma nota: X" / "create a note about X".
# Group `body` is the note content (its first line becomes the title).
NOTE_CREATE_RE = re.compile(
    r"^(?:"
    # PT — "anota / anote / anotar (aí[:,]) (que) X"
    r"anot(?:a|e|ar)(?:\s*:\s*|\s+)(?:a[íi]\s*[:,]?\s*)?(?:que\s+)?"
    r"|"
    # PT — "cria / faz / nova / adiciona (uma) nota [: — - sobre de que] X"
    r"(?:cria(?:r)?|faz(?:er)?|nova|adiciona(?:r)?)\s+(?:uma\s+)?nota"
    r"(?:\s*[:—-]\s*|\s+(?:sobre|de|que|com)\s+|\s+)"
    r"|"
    # EN — "create a note / new note / note down [: about that] X"
    r"(?:create\s+(?:a\s+)?note|new\s+note|note\s+down)"
    r"(?:\s*[:—-]\s*|\s+(?:about|that)\s+|\s+)"
    r")(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# v1.1 (s73) — "anota na pasta Trabalho: X" / "cria uma nota na pasta Ideias —
# X" / "create a note in folder Work: X". The folder/body separator (: — -) is
# REQUIRED so the folder name has an unambiguous right edge. Checked BEFORE
# NOTE_CREATE_RE in the registry: "adiciona nota na pasta X: Y" also matches
# the plain create regex (with "na pasta X: Y" swallowed into the body).
NOTE_CREATE_FOLDER_RE = re.compile(
    r"^(?:"
    r"anot(?:a|e|ar)\s+"
    r"|(?:cria(?:r)?|faz(?:er)?|nova|adiciona(?:r)?)\s+(?:uma\s+)?nota\s+"
    r"|(?:create\s+(?:a\s+)?note|new\s+note)\s+"
    r")"
    r"(?:na\s+pasta|in\s+(?:the\s+)?folder)\s+"
    r"(?P<folder>[^:—-]+?)\s*[:—-]\s*(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# v1.1 (s73) — append, title-first shape: "acrescenta na nota Mercado: pão" /
# "adiciona à nota Ideias — testar X" / "append to note Work: ship it".
# Separator required, same reason as the folder regex.
NOTE_APPEND_RE = re.compile(
    r"^(?:"
    r"(?:adiciona(?:r)?|acrescenta(?:r)?|anexa(?:r)?|coloca(?:r)?|bota(?:r)?)\s+"
    r"(?:isso\s+)?(?:na|[àa])\s+nota\s+"
    r"|(?:append|add)\s+to\s+(?:the\s+)?note\s+"
    r")"
    r"(?P<title>[^:—-]+?)\s*[:—-]\s*(?P<text>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# v1.1 (s73) — append, text-first shape: "adiciona pão na nota Mercado" /
# "append ship it to note Work". No separator needed: "na nota" itself is
# the boundary (lazy text + greedy title keeps the LAST "na nota" split).
NOTE_APPEND_TAIL_RE = re.compile(
    r"^(?:"
    r"(?:adiciona(?:r)?|acrescenta(?:r)?|anexa(?:r)?|coloca(?:r)?|bota(?:r)?)\s+"
    r"(?P<text>.+?)\s+(?:na|[àa])\s+nota\s+"
    r"|(?:append|add)\s+(?P<text_en>.+?)\s+to\s+(?:the\s+)?note\s+"
    r")"
    r"(?P<title>.+?)[\s.!?]*$",
    re.IGNORECASE | re.DOTALL,
)

# "minhas notas" / "lista as notas" / "list my notes".
NOTE_LIST_RE = re.compile(
    r"^(?:"
    r"(?:lista(?:r)?|mostra(?:r)?|ver|quais\s+s[ãa]o)\s+"
    r"(?:as\s+|minhas\s+|todas\s+as\s+)?notas"
    r"|minhas\s+notas"
    r"|(?:list|show)\s+(?:my\s+|all\s+)?notes"
    r")[\s.!?]*$",
    re.IGNORECASE,
)

# "acha a nota sobre X" / "procura nota X" / "search notes for X".
NOTE_SEARCH_RE = re.compile(
    r"^(?:"
    r"(?:acha(?:r)?|procura(?:r)?|busca(?:r)?|encontra(?:r)?|cad[êe])\s+"
    r"(?:a\s+|uma\s+|minha\s+)?nota\s+"
    r"(?:sobre\s+|de\s+|com\s+|que\s+(?:fala\s+(?:de|sobre)\s+)?)?"
    r"|(?:search|find)\s+(?:my\s+)?notes?\s+(?:for|about)\s+"
    r")(?P<query>.+)$",
    re.IGNORECASE | re.DOTALL,
)
