"""Pure, sentence-scoped currency policy for market-specific web answers.

This checks currency, not source attribution or numeric truth. Bare numbers
are deliberately outside the policy. No logging, configuration or model I/O.
"""
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from src.tools.web_search import SearchMarket


_WEB_MARKET = ContextVar("web_price_market", default=(None, ""))


@contextmanager
def web_market_scope(market: SearchMarket | None, user_query: str):
    """Only web synthesis opens this scope; reset even on engine errors."""
    token = _WEB_MARKET.set((market, user_query))
    try:
        yield
    finally:
        _WEB_MARKET.reset(token)


def web_market_context():
    return _WEB_MARKET.get()


# Deliberate shortlist of required common currencies, avoiding the wider ISO
# table's ordinary-word/acronym collisions such as ALL, TRY, TOP, CUP or COP.
_ISO_CODES = frozenset((
    "USD EUR GBP INR BRL CAD AUD NZD JPY CNY CHF HKD SGD MXN ARS ZAR SEK NOK DKK PLN"
).split())
_ALIASES = {code.casefold(): code for code in _ISO_CODES}
_ALIASES.update({
    "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD",
    "r$": "BRL", "us$": "USD", "c$": "CAD", "ca$": "CAD",
    "a$": "AUD", "au$": "AUD", "nz$": "NZD", "hk$": "HKD",
    "s$": "SGD", "sg$": "SGD", "cn¥": "CNY", "jp¥": "JPY",
    "₽": "RUB", "₩": "KRW", "₺": "TRY", "₪": "ILS", "₴": "UAH",
    "₫": "VND", "฿": "THB", "₱": "PHP", "৳": "BDT",
})
_CURRENCY_WORDS = {}
for _code, _words in {
    "INR": "rúpia rúpias rupia rupias rupee rupees",
    "EUR": "euro euros",
    "USD": "dólar dólares dolar dolares dollar dollars",
    "BRL": "real reais reals",
    "GBP": "libra libras pound pounds",
    "JPY": "yen yens iene ienes",
}.items():
    _CURRENCY_WORDS.update(dict.fromkeys(_words.split(), _code))
_ALIASES.update(_CURRENCY_WORDS)


def _marker_pattern(alias: str) -> str:
    # Case sensitivity belongs to ISO codes only, including in user queries.
    pattern = (re.escape(alias.upper()) if alias.upper() in _ISO_CODES else
               "(?i:" + re.escape(alias) + ")")
    return (r"(?<![^\W\d_])" + pattern + r"(?![^\W\d_])"
            if alias.isalpha() else pattern)


# Longest markers first: R$/US$/C$ are consumed whole, never as bare $.
_MARKER = re.compile("|".join(
    _marker_pattern(alias)
    for alias in sorted(_ALIASES, key=len, reverse=True)
))
_GAP = r"[ \t\u00a0\u202f*_~`]*"  # horizontal space / Markdown emphasis
_NUMBER = r"\d+(?:[.,]\d+)*"
_BEFORE = re.compile("(" + _NUMBER + ")" + _GAP + r"$")
_AFTER = re.compile(_GAP + "(" + _NUMBER + ")")
# Match the whole amount: Western groups cannot match an Indian suffix.
# A decimal fraction is part of the amount, but bare numbers are still ignored.
_INDIAN_NUMBER = re.compile(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?")
_ABBREVIATIONS = frozenset((
    "dr dra mr mrs ms prof sr sra srta jr st vs etc approx aprox inc ltd co corp "
    "pág pag art av eng jan feb fev mar apr abr jun jul aug ago sep set "
    "oct out nov dec dez e.g i.e"
).split())
_WORD_END = re.compile(r"([^\W\d_]+(?:\.[^\W\d_]+)*)\.$")
_CLOSERS = "\"'’”»)]}*_~`"


def local_price_gap(market: SearchMarket, lang: str) -> str:
    """The 148c market table owns names and Portuguese prepositions."""
    return (f"Não achei o preço {market.location_pt}." if lang == "pt" else
            f"I couldn't find the price in {market.name_en}.")


def _currency_mentions(text: str, market: SearchMarket, *, money_only: bool):
    """Yield (currency, relabelled) pairs; query exemptions use markers only."""
    for marker in _MARKER.finditer(text):
        alias = marker[0].casefold()
        # Words only follow amounts: Euro 2024 and Real 3 x 1 are not prices.
        # A user's explicit currency mention still needs no adjacent amount.
        amount = None
        if money_only:
            amount = _BEFORE.search(text[:marker.start()])
            if amount is None and alias not in _CURRENCY_WORDS:
                amount = _AFTER.match(text, marker.end())
            if amount is None:
                continue
        currency = _ALIASES[alias]
        if marker[0] == "$" and market.code == "CA":
            currency = "CAD"
        if amount is not None and _INDIAN_NUMBER.fullmatch(amount[1]):
            # Even an explicit INR exemption cannot bless an invented £ label.
            yield "INR", currency != "INR"
        else:
            yield currency, False


def _currencies(text: str, market: SearchMarket, *, money_only: bool) -> set[str]:
    return {currency for currency, _ in _currency_mentions(text, market, money_only=money_only)}


def _sentences(text: str, final: bool = False) -> tuple[list[str], str]:
    """Lossless spans; one-character lookahead disambiguates a trailing dot.

    Decimal/grouping dots, initials, acronyms and common EN/PT abbreviations
    stay inside a sentence. Unknown dotted forms without following whitespace
    are conservatively held together. Newlines always close a span.
    """
    spans, start, i = [], 0, 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            spans.append(text[start:i + 1])
            start = i = i + 1
            continue
        if char not in ".!?":
            i += 1
            continue
        if i + 1 == len(text) and not final:
            break
        if char == ".":
            if i and text[i - 1].isdigit() and text[i + 1:i + 2].isdigit():
                i += 1
                continue
            word = _WORD_END.search(text[:i + 1])
            if word and (word[1].casefold() in _ABBREVIATIONS
                         or len(word[1]) == 1
                         or all(len(p) == 1 for p in word[1].split("."))):
                i += 1
                continue
        end = i + 1
        while end < len(text) and text[end] in ".!?" + _CLOSERS:
            end += 1
        if end == len(text) and not final:
            break
        if end == len(text) or text[end].isspace():
            spans.append(text[start:end])
            start = i = end
        else:
            i += 1
    if final and start < len(text):
        spans.append(text[start:])
        start = len(text)
    return spans, text[start:]


@dataclass(frozen=True)
class MarketPriceResult:
    reply: str
    currencies: tuple[str, ...]
    count: int  # removed sentences


class MarketPriceGuard:
    """Turn-local holdback. feed/finish return only checked, emit-ready text."""

    def __init__(self, market: SearchMarket | None, user_query: str, lang: str):
        self.market = market
        self.allowed = ({market.currency} | _currencies(user_query, market, money_only=False)
                        if market is not None else set())
        self.gap = local_price_gap(market, lang) if market is not None else ""
        self.pending = ""
        self.reply = ""
        self.currencies: set[str] = set()
        self.count = 0

    def _check(self, spans: list[str]) -> str:
        accepted = []
        for sentence in spans:
            foreign = {
                currency for currency, relabelled in
                _currency_mentions(sentence, self.market, money_only=True)
                if relabelled or currency not in self.allowed
            }
            if foreign:
                self.currencies.update(foreign)
                self.count += 1
            else:
                if self.count and not self.reply and not accepted:
                    sentence = sentence.lstrip()
                accepted.append(sentence)
        safe = "".join(accepted)
        self.reply += safe
        return safe

    def feed(self, text: str) -> str:
        if self.market is None:
            self.reply += text
            return text
        spans, self.pending = _sentences(self.pending + text)
        return self._check(spans)

    def finish(self) -> str:
        if self.market is None:
            return ""
        spans, self.pending = _sentences(self.pending, final=True)
        tail = self._check(spans)
        # Include the final checked span before deciding whether a price is missing.
        if (self.count and self.gap not in self.reply
                and not (_currencies(self.reply, self.market, money_only=True) & self.allowed)):
            gap = (" " if self.reply and not self.reply[-1].isspace() else "") + self.gap
            self.reply += gap
            tail += gap
        return tail


def check_market_prices(
    reply: str, market: SearchMarket | None, user_query: str, lang: str,
) -> MarketPriceResult:
    guard = MarketPriceGuard(market, user_query, lang)
    guard.feed(reply)
    guard.finish()
    return MarketPriceResult(guard.reply, tuple(sorted(guard.currencies)), guard.count)
