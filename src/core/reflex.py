"""
Reflex Layer — pre-LLM zero-latency responses.

Handles trivial inputs (greetings, time/date queries, status pings)
without calling Llama. Each match saves ~15s of inference time.

Design reference: Estágio 2 (Reflex) + Cap 8 PADROES_REFLEX.

Usage:
    from src.core.reflex import reflex_layer

    reply = reflex_layer.try_respond(text)
    if reply is not None:
        return reply, "REFLEX"
    # else: continue to LLM path

Design rules:
- Only handle inputs where the answer is DETERMINISTIC and UNIVERSAL.
  "que horas são" → the current time. "oi" → a greeting.
- NEVER handle anything that requires personal context, documents,
  internet, or preference knowledge — those belong downstream.
- Patterns are anchored (full-string match) to avoid false positives.
  A message containing "oi" mid-sentence should not be reflexed.
- Matched patterns log an audit event (stage='reflex', event='hit').
  Misses also log (event='miss') so the distribution is visible.
"""

import re
from datetime import datetime
from typing import Callable, Optional, Union


# Type alias: a reflex response is either a fixed string or a callable
# that receives the matched text and returns a string.
_Response = Union[str, Callable[[str], str]]


def _now_local() -> datetime:
    """Current local time (naive, using local timezone)."""
    return datetime.now()


def _time_reply(text: str, now: Optional[datetime] = None) -> str:  # noqa: ARG001
    now = now or _now_local()
    return f"São {now.strftime('%H:%M')}."


def _date_reply(text: str, now: Optional[datetime] = None) -> str:  # noqa: ARG001
    now = now or _now_local()
    days_pt = ["segunda-feira", "terça-feira", "quarta-feira",
               "quinta-feira", "sexta-feira", "sábado", "domingo"]
    months_pt = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
                 "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    weekday = days_pt[now.weekday()]
    month = months_pt[now.month - 1]
    return f"Hoje é {weekday}, {now.day} de {month} de {now.year}."


def _year_reply(text: str) -> str:  # noqa: ARG001
    now = _now_local()
    return f"Estamos em {now.year}."


def _datetime_reply(text: str) -> str:
    # One reading of the clock for both halves: two readings across midnight
    # said "São 23:59. Hoje é domingo, 25 de outubro" (s88, Codex lote 8).
    now = _now_local()
    return f"{_time_reply(text, now)} {_date_reply(text, now)}"


# English replies — the reply language drives the TTS voice (the listener
# picks the EN voice only when the reply text reads as English). A PT reply
# to an English question made faber answer "what time is it" instead of the
# EN voice. So the EN time/date patterns get their own English replies.
def _time_reply_en(text: str) -> str:  # noqa: ARG001
    now = _now_local()
    return f"It's {now.strftime('%H:%M')}."


def _date_reply_en(text: str) -> str:  # noqa: ARG001
    now = _now_local()
    days_en = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"]
    months_en = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
    weekday = days_en[now.weekday()]
    month = months_en[now.month - 1]
    return f"Today is {weekday}, {now.day} {month} {now.year}."


def _year_reply_en(text: str) -> str:  # noqa: ARG001
    now = _now_local()
    return f"It's {now.year}."


# Calendar clauses are reused in standalone and combined questions. Keep
# the vocabulary narrow: personal events, past years and tomorrow belong
# downstream. A year+date match calls ONLY the date responder (which already
# includes the year), so the whole answer comes from one local clock reading.
_CALENDAR_PREFIX_PT = r"(?:me\s+(?:diz|fala(?:\s+a[íi])?|informa)\s+)?"
_CALENDAR_SUFFIX_PT = r"(?:\s+(?:agora|atual|hoje))?"
_YEAR_PT = (
    _CALENDAR_PREFIX_PT
    + r"(?:em\s+que\s+ano\s+(?:estamos|a\s+gente\s+est[áa])|"
      r"estamos\s+em\s+que\s+ano|que\s+ano\s+[ée]\s+(?:hoje|esse)|"
      r"qual\s+(?:[ée]\s+)?(?:o\s+)?ano(?:\s+atual)?)"
    + _CALENDAR_SUFFIX_PT
)
_DATE_PT = (
    _CALENDAR_PREFIX_PT
    + r"(?:que\s+dia\s+(?:[ée]\s+)?hoje|"
      r"qual\s+(?:[ée]\s+)?(?:a\s+)?data(?:\s+de\s+hoje)?|"
      r"que\s+data\s+(?:[ée]\s+)?hoje|"
      r"hoje\s+[ée]\s+(?:que|qual)\s+dia(?:\s+da\s+semana)?|"
      r"data\s+de\s+hoje|"
      r"que\s+dia\s+da\s+semana(?:\s+(?:[ée]\s+)?hoje)?|"
      r"qual\s+(?:[ée]\s+)?(?:o\s+)?dia\s+"
      r"(?:de\s+hoje|da\s+semana(?:\s+(?:[ée]\s+)?hoje)?))"
    + _CALENDAR_SUFFIX_PT
)
_CALENDAR_JOIN_PT = r"(?:\s+e\s+|\s*,\s*)"
_CALENDAR_SUFFIX_EN = r"(?:\s+(?:now|today))?"
_YEAR_EN = (
    r"(?:what\s+year\s+(?:is\s+it|are\s+we\s+in)|which\s+year\s+is\s+it)"
    + _CALENDAR_SUFFIX_EN
)
_DATE_EN = (
    r"(?:what(?:'s|\s+is)\s+(?:(?:the\s+)?date|today(?:'s\s+date)?)|"
    r"what\s+day\s+(?:of\s+the\s+week\s+)?is\s+(?:it|today))"
    + _CALENDAR_SUFFIX_EN
)
_CALENDAR_JOIN_EN = r"(?:\s+and\s+|\s*,\s*)"


def _calendar_date_pattern(year: str, date: str, join: str) -> re.Pattern:
    """A date alone, or exactly one year and one date in either order."""
    return re.compile(
        rf"^(?:{date}|{year}{join}{date}|{date}{join}{year})[.!?]*$",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Pattern table: (compiled_regex, response_or_callable)
# Checked in order — first match wins.
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern, _Response]] = [
    # Greetings — PT-BR
    # (s87) + "e ai" without the accent, "fala", "ei", "opa", "oie"/"oii" —
    # the greetings strip_vocatives hands over from "<greeting> cortex".
    (re.compile(r"^(?:oi+e?|ol[áa]|ei|hey|e\s*a[íi]|eae|salve|fala|opa|"
                r"bom\s+dia|boa\s+tarde|boa\s+noite)[.!,?]*$",
                re.IGNORECASE),
     "Oi! Em que posso ajudar?"),

    # Goodbye — PT-BR
    (re.compile(r"^(?:tchau|xau|até\s+mais|ate\s+mais|falou|até|flw|vlw|valeu)[.!,?]*$",
                re.IGNORECASE),
     "Até mais!"),

    # Thank you — PT-BR
    (re.compile(r"^(?:obrigad[oa]|obg|brigad[oa]|thanks?|thank\s+you)[.!,?]*$",
                re.IGNORECASE),
     "De nada! Posso ajudar com mais alguma coisa?"),

    # Time query — PT-BR. Broadened from the strict original because
    # voice transcriptions add fillers ("agora") and colloquial forms
    # ("que horas que são", "me diz as horas") that the anchored match
    # missed → they fell through to the slow LLM. Still anchored ($) so
    # longer sentences ("a que horas começa o jogo") don't false-match.
    (re.compile(
        r"^(?:me\s+(?:diz|fala|fala\s+a[íi]|informa)\s+)?"          # opt. "me diz/fala"
        r"(?:"
        r"que\s+horas?\s+(?:que\s+)?(?:são|é|esta|está|tem)|"        # que horas (que) são/é/tem
        r"são\s+que\s+horas?|"                                       # são que horas
        r"qual\s+(?:é\s+)?(?:a\s+)?horas?|"                          # qual (é) (a) hora(s)
        r"(?:a\s+)?hora\s+certa|"                                    # (a) hora certa
        r"(?:as?\s+)?horas?"                                         # (a/as) hora(s) — bare
        r")"
        r"(?:\s+agora)?[.!?]*$",                                     # opt. trailing "agora"
        re.IGNORECASE,
    ), _time_reply),

    # Date/weekday, optionally combined with year — PT-BR.
    (_calendar_date_pattern(_YEAR_PT, _DATE_PT, _CALENDAR_JOIN_PT), _date_reply),
    (re.compile(rf"^{_YEAR_PT}[.!?]*$", re.IGNORECASE), _year_reply),

    # Combined date+time — PT-BR
    (re.compile(
        r"^(?:que\s+(?:horas?\s+e\s+dia|dia\s+e\s+hora)|"
        r"hora\s+e\s+data|"
        r"data\s+e\s+hora)[.!?]*$",
        re.IGNORECASE,
    ), _datetime_reply),

    # English greetings
    (re.compile(r"^(?:hi+|hello+|hey+|good\s+(?:morning|afternoon|evening))[.!,?]*$",
                re.IGNORECASE),
     "Hi! How can I help you?"),

    # English time/date — reply in English so the listener picks the EN voice.
    (re.compile(
        r"^(?:what(?:'s|\s+is)\s+the\s+(?:time|current\s+time)|"
        r"what\s+time\s+is\s+it)[.!?]*$",
        re.IGNORECASE,
    ), _time_reply_en),

    (_calendar_date_pattern(_YEAR_EN, _DATE_EN, _CALENDAR_JOIN_EN), _date_reply_en),
    (re.compile(rf"^{_YEAR_EN}[.!?]*$", re.IGNORECASE), _year_reply_en),

    # Ping / status
    (re.compile(r"^(?:ping|pong|status|alive|online|cortex\?)[.!?]*$",
                re.IGNORECASE),
     "Online e operacional."),
]


class ReflexLayer:
    """Pre-LLM pattern matcher for trivial deterministic queries."""

    def __init__(self):
        self._patterns = _PATTERNS

    def try_respond(self, text: str) -> Optional[str]:
        """Return a deterministic reply if the input matches a reflex pattern.

        Returns None if no pattern matches — caller should continue to LLM.
        The caller is responsible for logging the audit event (hit/miss)
        so this method stays pure and easily testable.
        """
        text = (text or "").strip()
        if not text:
            return None
        for pattern, response in self._patterns:
            if pattern.match(text):
                if callable(response):
                    return response(text)
                return response
        return None

    def is_hit(self, text: str) -> bool:
        """True if try_respond would return a non-None value."""
        return self.try_respond(text) is not None


# Module-level singleton — import directly.
reflex_layer = ReflexLayer()
