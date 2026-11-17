"""Currency policy is deterministic; these are invented prices, not facts."""
from unittest.mock import MagicMock

import pytest

from src.tools.web_search import WebSearcher


@pytest.fixture
def market(monkeypatch):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "GB")
    return WebSearcher(cost_tracker=MagicMock()).get_market()


def check(reply, market, query="quanto custa o Pixel 10 Pro?", lang="pt"):
    from src.core.market_price_guard import check_market_prices
    return check_market_prices(reply, market, query, lang)


GAP = "Não achei o preço no Reino Unido."


@pytest.mark.parametrize("reply", [
    "O Pixel 10 Pro custa 1.099.999 rúpias na Índia, mas não encontrei o preço "
    "em libras no Reino Unido.",
    "O preço do Pixel 10 Pro em libras é de £1,09,999.",
])
def test_live_indian_price_escapes_are_removed(market, reply):
    result = check(reply, market, "quanto é o Pixel 10 Pro em libras?")
    assert (result.reply, result.currencies, result.count) == (GAP, ("INR",), 1)


@pytest.mark.parametrize("currency,words", [
    ("INR", "rúpia rúpias rupia rupias rupee rupees"),
    ("USD", "dólar dólares dolar dolares dollar dollars"),
    ("JPY", "iene ienes yen yens"),
])
def test_currency_word_spellings_require_preceding_amount(market, currency, words):
    for word in words.split():
        for spelling in (word, word.upper()):
            result = check(f"Custa 999 {spelling}.", market)
            assert (result.reply, result.currencies, result.count) == (GAP, (currency,), 1)
            reply = f"{spelling} 2026 chega amanhã."
            assert check(reply, market).reply == reply
            # Explicit query mentions still need no number.
            reply = f"Custa 999 {spelling}."
            assert check(reply, market, f"quanto custa em {spelling}?").reply == reply


@pytest.mark.parametrize("amount", ["1,09,999", "12,34,567", "1,79,900", "1,23,45,678",
                                      "1,09,999.50"])
@pytest.mark.parametrize("template", [
    "£{amount}", "GBP {amount}", "{amount} GBP", "{amount} libras",
    "**£** {amount}", "€{amount}", "{amount} dólares", "₹{amount}",
])
def test_indian_grouping_with_currency_marker_counts_as_inr(market, amount, template):
    result = check(f"Custa {template.format(amount=amount)}.", market)
    assert (result.reply, result.currencies, result.count) == (GAP, ("INR",), 1)


@pytest.mark.parametrize("amount", ["1,099", "109,999", "1,099,999", "1,099.50",
                                      "109,999.50", "1,099,999.50"])
def test_western_grouping_remains_local_money(market, amount):
    reply = f"Custa £{amount}."
    result = check(reply, market)
    assert (result.reply, result.currencies, result.count) == (reply, (), 0)


@pytest.mark.parametrize("amount", ["1,09,999", "1,09,999.50"])
def test_bare_indian_grouping_is_not_money(market, amount):
    reply = f"O número é {amount}."
    result = check(reply, market)
    assert (result.reply, result.currencies, result.count) == (reply, (), 0)


@pytest.mark.parametrize("query", ["quanto é ₹1,09,999 em libras?",
                                  "quanto é em rúpias?", "quanto é em rupees?"])
def test_rupee_query_allows_rupees_but_rejects_relabelled_amount(market, query):
    reply = "Custa ₹1,09,999."
    assert check(reply, market, query).reply == reply
    result = check("Custa £1,09,999.", market, query)
    assert (result.reply, result.currencies, result.count) == (GAP, ("INR",), 1)


def test_indian_grouping_in_inr_market(monkeypatch):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "IN")
    market = WebSearcher(cost_tracker=MagicMock()).get_market()
    reply = "Custa ₹1,09,999.50."
    assert check(reply, market).reply == reply


def test_stream_holds_relabelled_indian_amount(market):
    from src.core.market_price_guard import MarketPriceGuard

    guard = MarketPriceGuard(market, "quanto é o Pixel 10 Pro em libras?", "pt")
    emitted = ""
    for chunk in ["O preço do Pixel 10 Pro em libras é de £1,", "09,999."]:
        emitted += guard.feed(chunk)
        assert emitted == ""
    emitted += guard.finish()
    assert emitted == GAP
    assert "£1,09,999" not in emitted


@pytest.mark.parametrize("reply", [
    "A Espanha venceu a Euro 2024.",
    "A World Cup 2026 começa em junho.",
    "Os top 10 celulares de 2026 são estes.",
    "A COP 30 foi em Belém.",
    "All 3 models ship today.",
    "Real 3 x 1 Barcelona, ontem.",
    "Try 2 times and wait.",
    "Libra 2 chega amanhã.",
    "eur 899", "Eur 899", "899 eur", "899 eUr",
    "USDollar 999", "999 EUROSport",
    "NPR 999", "MAD999",
    *[f"{word} 30" for word in (
        "TRY", "ALL", "TOP", "CUP", "COP", "MAD", "PEN", "SOS", "BOB", "GEL"
    )],
    *[f"{word} 2026" for word in (
        "euros", "reais", "dólares", "rupias", "libras", "pounds",
        "dollars", "rupees", "yen", "ienes",
    )],
])
def test_non_money_answers_pass_unchanged(market, reply):
    result = check(reply, market, "qual a novidade?")
    assert (result.reply, result.currencies, result.count) == (reply, (), 0)


@pytest.mark.parametrize("code", (
    "USD EUR GBP INR BRL CAD AUD NZD JPY CNY CHF HKD SGD MXN ARS ZAR SEK NOK DKK PLN"
).split())
@pytest.mark.parametrize("template", ["{code} 899", "899 {code}"])
def test_common_iso_codes_are_uppercase_only(market, code, template):
    money = template.format(code=code)
    result = check(money, market, "qual a novidade?")
    assert (result.reply, result.currencies, result.count) == (
        (money, (), 0) if code == "GBP" else (GAP, (code,), 1)
    )
    for spelling in (code.lower(), code.title()):
        reply = template.format(code=spelling)
        result = check(reply, market, "qual a novidade?")
        assert (result.reply, result.currencies, result.count) == (reply, (), 0)


@pytest.mark.parametrize("reply,expected,currencies,count", [
    ("O Pixel 10 Pro custa ₹1,09,999 no Reino Unido.", GAP, ("INR",), 1),
    ("O Pixel 10 Pro custa ₹1,09,999 (256GB) na Índia e £999 (128GB) no Reino Unido.",
     GAP, ("INR",), 1),
    ("O Pixel 10 Pro custa £999 no Reino Unido.",
     "O Pixel 10 Pro custa £999 no Reino Unido.", (), 0),
    ("  Tem 256GB.\nA bateria dura 24 horas!  ",
     "  Tem 256GB.\nA bateria dura 24 horas!  ", (), 0),
    ("Custa INR 999. Custa EUR 899! Tem duas cores.",
     "Tem duas cores. " + GAP, ("EUR", "INR"), 2),
    ("Custa ₹999. " + GAP, GAP, ("INR",), 1),
    (GAP + " Custa ₹999", GAP, ("INR",), 1),
    ("Custa ₹999\nCusta £999", "Custa £999", ("INR",), 1),
    ("O preço é 109999.", "O preço é 109999.", (), 0),
])
def test_sentence_outcome(market, reply, expected, currencies, count):
    result = check(reply, market)
    assert result.reply == expected
    assert result.currencies == currencies
    assert result.count == count  # removed sentences, not number of markers


@pytest.mark.parametrize("money,currency", [
    ("US$ 999", "USD"), ("999 dólares", "USD"), ("EUR 899", "EUR"),
    ("R$ 5.999", "BRL"), ("999 reais", "BRL"), ("1 real", "BRL"),
    ("999 rupees", "INR"), ("1 rupee", "INR"), ("999 rupias", "INR"),
    ("1 rupia", "INR"), ("999 euros", "EUR"), ("1 euro", "EUR"),
    ("999 dollars", "USD"), ("1 dollar", "USD"), ("1 dólar", "USD"),
    ("C$999", "CAD"), ("CAD 999", "CAD"), ("¥999", "JPY"),
    ("JPY 999", "JPY"), ("€899", "EUR"), ("$999", "USD"),
    ("USD\u00a0999", "USD"), ("999 BRL", "BRL"), ("CHF 999", "CHF"),
    ("**₹** 999", "INR"),
    ("999EUR", "EUR"), ("899 EUR", "EUR"), ("899 €", "EUR"),
    ("50 euros", "EUR"), ("50 EUROS", "EUR"), ("5 reais", "BRL"),
])
def test_foreign_money_markers(market, money, currency):
    result = check(f"Custa {money}.", market)
    assert result.reply == GAP
    assert result.currencies == (currency,)


@pytest.mark.parametrize("money", ["£999", "GBP 999", "999 libras", "1 libra",
                                    "999 pounds", "1 pound"])
def test_local_money_markers(market, money):
    reply = f"Custa {money}."
    assert check(reply, market).reply == reply


@pytest.mark.parametrize("reply,currency", [
    ("O ingresso custa 50 euros.", "EUR"),
    ("custa ₹1,09,999", "INR"),
    ("costs 999 dollars", "USD"),
])
def test_foreign_prices_in_ordinary_answers_are_removed(market, reply, currency):
    result = check(reply, market, "qual a novidade?")
    assert (result.reply, result.currencies, result.count) == (GAP, (currency,), 1)


@pytest.mark.parametrize("code,currency", [("US", "USD"), ("CA", "CAD")])
def test_bare_dollar_uses_us_or_canadian_market(monkeypatch, code, currency):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", code)
    market = WebSearcher(cost_tracker=MagicMock()).get_market()
    result = check("Costs $999.", market)
    assert result.reply == "Costs $999."
    assert result.count == 0
    # R$ must never inherit the market's bare-$ exemption.
    assert check("Costs R$999.", market).currencies == ("BRL",)


@pytest.mark.parametrize("query", [
    "quanto é ₹1,09,999 em libras?", "quanto é INR em libras?",
    "what are rupees in pounds?", "converter rupias para libras",
])
def test_users_own_currency_is_allowed(market, query):
    reply = "O Pixel custa ₹1,09,999."
    assert check(reply, market, query).reply == reply
    assert check("Custa EUR 999.", market, query).currencies == ("EUR",)


@pytest.mark.parametrize("reply,query,expected,currencies", [
    ("O iPhone 17 tem 256 GB e custa US$ 1.199. No Reino Unido sai por £1,199.",
     "quanto custa o iPhone 17?", "No Reino Unido sai por £1,199.", ("USD",)),
    ("Custa £1,199. Custa US$ 1.199.",
     "quanto custa?", "Custa £1,199.", ("USD",)),
    ("custa ₹1,09,999. É um bom celular.",
     "quanto custa?", "É um bom celular. " + GAP, ("INR",)),
    ("Custa US$ 1.199. Custa ₹1,09,999.",
     "quanto custa em rupees?", "Custa ₹1,09,999.", ("USD",)),
    ("custa ₹1,09,999. É um bom celular.",
     "quanto custa em rupias?", "custa ₹1,09,999. É um bom celular.", ()),
    ("Custa ₹1,09,999. A moeda é GBP.",
     "quanto custa?", "A moeda é GBP. " + GAP, ("INR",)),
    ("Custa ₹1,09,999 e £999.", "quanto custa?", GAP, ("INR",)),
])
def test_gap_depends_on_money_remaining_in_accepted_text(
    market, reply, query, expected, currencies,
):
    from src.core.market_price_guard import MarketPriceGuard

    result = check(reply, market, query)
    assert (result.reply, result.currencies, result.count) == (
        expected, currencies, int(bool(currencies)),
    )
    for offset in range(len(reply) + 1):
        guard = MarketPriceGuard(market, query, "pt")
        emitted = guard.feed(reply[:offset]) + guard.feed(reply[offset:])
        assert GAP not in emitted  # the gap decision belongs to finish()
        assert emitted + guard.finish() == expected
        assert guard.reply == expected


def test_gap_waits_for_final_local_price(market):
    from src.core.market_price_guard import MarketPriceGuard

    guard = MarketPriceGuard(market, "quanto custa?", "pt")
    assert guard.feed("Custa US$ 1.199. É um bom celular. ") == "É um bom celular."
    assert guard.feed("No Reino Unido sai por £1,199.") == ""
    assert guard.finish() == " No Reino Unido sai por £1,199."
    assert guard.reply == "É um bom celular. No Reino Unido sai por £1,199."


@pytest.mark.parametrize("marker", ["US$", "USD", "dólares", "dollars", "$"])
def test_query_markers_need_no_number(market, marker):
    assert check("Costs US$999.", market, f"price in {marker}?").count == 0


def test_lowercase_iso_code_in_query_does_not_exempt_currency(market):
    assert check("EUR 899", market, "price in eur?").currencies == ("EUR",)


def test_compound_dollar_markers_never_become_bare_dollars(monkeypatch):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "BR")
    market = WebSearcher(cost_tracker=MagicMock()).get_market()
    assert check("PreçoR$999.", market).reply == "PreçoR$999."
    assert check("US$999.", market).currencies == ("USD",)
    assert check("C$999.", market).currencies == ("CAD",)


def test_english_twin_and_no_market(market):
    reply = "The Pixel costs ₹1,09,999 in the United Kingdom."
    assert check(reply, market, "how much is the Pixel?", "en").reply == (
        "I couldn't find the price in the United Kingdom.")
    assert check(reply, None).reply == reply


@pytest.mark.parametrize("reply", [
    "A versão é 1.199. Tem 24 horas.",
    "Dr. Silva diz que custa ₹999. Custa £999.",
    "E.g. the launch costs INR 999. It has two colours.",
    "U.S. sources list USD 999. UK sources list GBP 899.",
    "Custa 999 rupias indianas e 899 euros.\nTem duas cores.",
    "A Espanha venceu a Euro 2024. O ingresso custa 50 euros. Custa £999.",
    "A COP 30 foi em Belém. All 3 models ship today. Try 2 times and wait.",
])
def test_stream_is_chunk_invariant(market, reply):
    from src.core.market_price_guard import MarketPriceGuard
    expected = check(reply, market)
    # Every possible single split, including just after a decimal dot.
    for offset in range(len(reply) + 1):
        guard = MarketPriceGuard(market, "price?", "pt")
        emitted = guard.feed(reply[:offset]) + guard.feed(reply[offset:]) + guard.finish()
        assert emitted == expected.reply
        assert guard.count == expected.count
        assert tuple(sorted(guard.currencies)) == expected.currencies


def test_abbreviations_cannot_release_half_a_foreign_sentence(market):
    result = check("Dr. Silva diz que custa ₹999. Custa £999.", market)
    assert result.reply == "Custa £999."
    assert check("E.g. it costs INR 999.", market).reply == GAP
    assert check("Google Inc. charges INR 999.", market).reply == GAP


def test_one_sentence_can_have_multiple_foreign_currencies(market):
    result = check("INR 999 e EUR 899. USD 999.", market)
    assert (result.currencies, result.count) == (("EUR", "INR", "USD"), 2)
