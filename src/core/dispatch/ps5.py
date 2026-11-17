"""Shared PS5 vocabulary and routing; claims and Mac exclusion reuse the noun."""
import re

PS5_WORDS = r"\b(?:ps\s*5|playstation(?:\s*5)?|play\s+5)\b"
PS5_NOUN_RE = re.compile(PS5_WORDS, re.IGNORECASE)
# Each action has PT/EN phrase forms. Routing and language selection are
# compiled from these same forms; app names never decide the reply language.
_FORMS = {
    "wake": {
        "pt": r"^(?:liga(?:r)?|ligue|acorda(?:r)?|acorde)\s+(?:o\s+)?" + PS5_WORDS
              + r"(?:[\s.!?]*$|\s+e\s+(?:abre|abrir)\b)",
        "en": r"^(?:turn\s+on|wake(?:\s+up)?)\s+(?:the\s+)?" + PS5_WORDS
              + r"(?:[\s.!?]*$|\s+and\s+open\b)",
    },
    "standby": {
        "pt": r"^(?:(?:desliga(?:r)?|desligue)\s+(?:o\s+)?" + PS5_WORDS
              + r"|(?:coloca(?:r)?|coloque|p[õo]e|ponha)\s+(?:o\s+)?" + PS5_WORDS
              + r"\s+em\s+repouso)[\s.!?]*$",
        "en": r"^(?:turn\s+off\s+(?:the\s+)?" + PS5_WORDS
              + r"|put\s+(?:the\s+)?" + PS5_WORDS + r"\s+to\s+sleep)[\s.!?]*$",
    },
    "status": {
        "pt": r"^(?:(?:o\s+)?" + PS5_WORDS + r"\s+(?:est[áa]|t[áa])\s+ligado"
              + r"|o\s+" + PS5_WORDS + r"\s+status|status\s+do\s+" + PS5_WORDS + r")[\s.!?]*$",
        "en": r"^(?:" + PS5_WORDS + r"\s+status|status\s+" + PS5_WORDS
              + r"|is\s+(?:the\s+)?" + PS5_WORDS + r"\s+on)[\s.!?]*$",
    },
    "open": {
        "pt": r"^(?=[\s\S]*" + PS5_WORDS
              + r")(?:abre|abrir|abra|lan[çc]a(?:r)?|inicia(?:r)?)\b[\s\S]*$",
        "en": r"^(?=[\s\S]*" + PS5_WORDS + r")(?:open|launch|start)\b[\s\S]*$",
    },
}


def _action_pattern(action):
    return re.compile("(?:" + "|".join(_FORMS[action].values()) + ")", re.I)


PS5_WAKE_RE = _action_pattern("wake")
PS5_STANDBY_RE = _action_pattern("standby")
PS5_STATUS_RE = _action_pattern("status")
PS5_OPEN_RE = re.compile(r"\b(?:abre|abrir|abra|open|launch|start|inicia(?:r)?)\b", re.I)

# Own console commands before broad Mac/Spotify verbs, while notes and
# knowledge questions that merely mention a console keep their own routes.
PS5_RE = re.compile("|".join(
    "(?P<" + lang + ">" + "|".join(forms[lang] for forms in _FORMS.values()) + ")"
    for lang in ("pt", "en")
), re.I)


def ps5_phrase_language(text: str):
    """Return the language of the routed phrase form, or None off-route."""
    match = PS5_RE.match(text)
    return match.lastgroup if match else None

UNSUPPORTED = {
    "pt": "Ainda não consigo abrir apps ou jogos no PS5 — só ligar, desligar e ver se está ligado.",
    "en": "I can't open apps or games on the PS5 yet — only turn it on, put it in rest mode and check if it's on.",
}
