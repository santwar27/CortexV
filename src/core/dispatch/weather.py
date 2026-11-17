"""
Deterministic weather dispatch (idea #27, s73).

Regex-first, no TOOL_SPECS entry, no LLM anywhere in the path: the route
captures the city, the orchestrator checks the lease, Open-Meteo returns
numbers, a template formats them (an eval pins mode=WEATHER so the LLM
can't sneak back in). The arbiter already classified LOOKUP/weather at
0.88 — this regex is that marker plus a city capture.

A CITY IS REQUIRED ("clima em Londres"); bare "como tá o tempo?" falls
through to the old offer-to-search path — guessing the user's city would
be the kind of silent magic this project avoids.
"""
from __future__ import annotations

import re

# "clima em Londres" / "temperatura em São Paulo" / "previsão do tempo em
# Recife hoje" / "como tá o tempo em Lisboa?" / "weather in London" /
# "temperature in Tokyo". Trailing time words (hoje/agora/amanhã…) are
# swallowed so they don't pollute the city capture; punctuation trimmed.
WEATHER_RE = re.compile(
    r"^(?:qual\s+(?:é\s+|e\s+)?(?:a\s+|o\s+)?|como\s+(?:t[áa]|est[áa])\s+(?:a\s+|o\s+)?)?"
    r"(?:clima|tempo|temperatura|previs[ãa]o(?:\s+do\s+tempo)?|weather|temperature)\s+"
    # "de" is deliberately NOT a city preposition: "tempo de cozimento do
    # arroz" would capture a fake city. "clima de Londres" falls through to
    # the old offer-to-search path — no harm, just not the fast lane.
    r"(?:agora\s+|atual\s+|de\s+hoje\s+)?"
    r"(?:em|no|na|in)\s+"
    r"(?P<city>.+?)"
    r"(?:\s+(?:depois\s+de\s+amanh[ãa]|(?:the\s+)?day\s+after\s+tomorrow"
    r"|hoje|agora|amanh[ãa]|today|now|tomorrow))?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

# The day asked for (s88, Codex lote 13 — invariant 7). The route swallows
# "amanhã" so it stays out of the city, and the handler then got the city
# alone: "clima no Rio de Janeiro amanhã" was answered with today's weather.
# Read off the words the user said, in every path that reaches the handler.
_DAY_AFTER_TOMORROW_RE = re.compile(
    r"\bdepois\s+de\s+amanh[ãa]\b|\bday\s+after\s+tomorrow\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"\bamanh[ãa]\b|\btomorrow\b", re.IGNORECASE)


def requested_day(text: str) -> int:
    """0 for today (the default), 1 for tomorrow, 2 for the day after."""
    if _DAY_AFTER_TOMORROW_RE.search(text or ""):
        return 2
    if _TOMORROW_RE.search(text or ""):
        return 1
    return 0
