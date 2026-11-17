"""Optional market preferences preserve the unconfigured provider contract."""
import logging
import re
from dataclasses import FrozenInstanceError
from importlib.metadata import metadata
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.web_search import WebSearcher


MARKETS = [
    ("GB", "united kingdom", "uk-en", "Reino Unido", "the United Kingdom", "GBP", "£"),
    ("US", "united states", "us-en", "Estados Unidos", "the United States", "USD", "$"),
    ("BR", "brazil", "br-pt", "Brasil", "Brazil", "BRL", "R$"),
    ("PT", "portugal", "pt-pt", "Portugal", "Portugal", "EUR", "€"),
    ("IE", "ireland", "ie-en", "Irlanda", "Ireland", "EUR", "€"),
    ("CA", "canada", "ca-en", "Canadá", "Canada", "CAD", "$"),
]
QUERY = "compare two phones"


@pytest.fixture(autouse=True)
def clean_search_env(monkeypatch):
    monkeypatch.delenv("CORTEX_SEARCH_COUNTRY", raising=False)
    monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


@pytest.fixture
def providers():
    response = MagicMock()
    response.json.return_value = {
        "results": [{"title": "T", "content": "snippet", "url": "https://e/1"}],
    }
    with patch("src.tools.web_search.requests.post", return_value=response) as post, \
         patch("src.tools.web_search.DDGS") as ddgs_cls:
        ddgs = ddgs_cls.return_value.__enter__.return_value
        ddgs.text.return_value = [{"title": "T", "body": "snippet", "href": "https://e/1"}]
        yield post, ddgs_cls, ddgs


def _assert_provider_calls(searcher, providers, tavily_country=None, ddgs_region=None):
    post, ddgs_cls, ddgs = providers
    assert searcher._search_tavily(QUERY, 2) == "Source 1 (T):\nsnippet\n[https://e/1]"
    payload = {
        "api_key": "tvly-test", "query": QUERY, "search_depth": "basic",
        "max_results": 2, "include_answer": False,
    }
    if tavily_country is not None:
        payload.update(topic="general", country=tavily_country)
    post.assert_called_once_with(
        WebSearcher.TAVILY_URL, json=payload, timeout=WebSearcher.TAVILY_TIMEOUT_S,
    )
    assert searcher._search_ddg(QUERY, 2) == "Result 1 (T): snippet\n[https://e/1]\n"
    ddgs_cls.assert_called_once_with()
    kwargs = {"max_results": 2}
    if ddgs_region is not None:
        kwargs["region"] = ddgs_region
    ddgs.text.assert_called_once_with(QUERY, **kwargs)


def test_unset_keeps_exact_148a_payload_and_ddgs_call(providers, caplog):
    with caplog.at_level(logging.WARNING, logger="src.tools.web_search"):
        searcher = WebSearcher(cost_tracker=MagicMock())
        _assert_provider_calls(searcher, providers)
    assert not caplog.records


@pytest.mark.parametrize("country", ["GB", "gb", " Gb "])
def test_gb_preferences_and_market(monkeypatch, providers, country):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", country)
    searcher = WebSearcher(cost_tracker=MagicMock())
    _assert_provider_calls(searcher, providers, "united kingdom", "uk-en")
    market = searcher.get_market()
    assert (market.code, market.name_pt, market.name_en, market.currency, market.symbol) == (
        "GB", "Reino Unido", "the United Kingdom", "GBP", "£",
    )
    searcher.cost_tracker.record.assert_called_once_with(
        "tavily", "search", units=1, unit_kind="credit",
        metadata={"query": QUERY[:80], "max_results": 2},
    )


@pytest.mark.parametrize("code,tavily,region,name_pt,name_en,currency,symbol", MARKETS)
def test_supported_markets(monkeypatch, providers, code, tavily, region,
                           name_pt, name_en, currency, symbol):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", code)
    searcher = WebSearcher(cost_tracker=MagicMock())
    _assert_provider_calls(searcher, providers, tavily, region)
    market = searcher.get_market()
    assert (market.code, market.name_pt, market.name_en, market.currency, market.symbol) == (
        code, name_pt, name_en, currency, symbol,
    )
    # ddgs 9.8.0 has no runtime region enum; its own METADATA, lines 124–183,
    # documents the supported values. Read that installed list, not a copy.
    regions = set(re.findall(r"^    (\S+) for ", metadata("ddgs")["Description"], re.M))
    assert regions
    assert region in regions


def test_empty_is_silently_unset(monkeypatch, providers, caplog):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "")
    with caplog.at_level(logging.WARNING, logger="src.tools.web_search"):
        searcher = WebSearcher(cost_tracker=MagicMock())
        _assert_provider_calls(searcher, providers)
    assert searcher.get_market() is None
    assert not caplog.records


@pytest.mark.parametrize("country", ["XX", " ", "united kingdom", "GB\nsecret-value"])
def test_invalid_is_unset_and_warns_once_without_echo(monkeypatch, providers, caplog, country):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", country)
    with caplog.at_level(logging.WARNING, logger="src.tools.web_search"):
        searcher = WebSearcher(cost_tracker=MagicMock())
        _assert_provider_calls(searcher, providers)
        assert searcher.get_market() is None
        assert searcher.get_market() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "CORTEX_SEARCH_COUNTRY" in caplog.records[0].message
    if country.strip():
        assert country not in caplog.text


def test_accessor_is_none_when_unset():
    assert WebSearcher(cost_tracker=MagicMock()).get_market() is None


def test_market_is_fixed_at_construction_and_read_only(monkeypatch, providers):
    unconfigured = WebSearcher(cost_tracker=MagicMock())
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "GB")
    configured = WebSearcher(cost_tracker=MagicMock())
    market = configured.get_market()
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "US")
    _assert_provider_calls(configured, providers, "united kingdom", "uk-en")
    assert unconfigured.get_market() is None
    assert configured.get_market() is market
    with pytest.raises(FrozenInstanceError):
        market.code = "US"


def test_tavily_failure_preserves_market_in_ddgs_fallback(monkeypatch, providers):
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "GB")
    searcher = WebSearcher(cost_tracker=MagicMock())
    post, _, ddgs = providers
    post.side_effect = requests.RequestException("offline test failure")
    assert searcher.search(QUERY, max_results=2) == "Result 1 (T): snippet\n[https://e/1]\n"
    assert post.call_args.kwargs["json"]["country"] == "united kingdom"
    ddgs.text.assert_called_once_with(QUERY, max_results=2, region="uk-en")
    searcher.cost_tracker.record.assert_not_called()


def test_ddg_provider_uses_market_without_tavily(monkeypatch, providers):
    monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", "ddg")
    monkeypatch.setenv("CORTEX_SEARCH_COUNTRY", "GB")
    post, _, ddgs = providers
    searcher = WebSearcher(cost_tracker=MagicMock())
    searcher.search(QUERY, max_results=2)
    post.assert_not_called()
    ddgs.text.assert_called_once_with(QUERY, max_results=2, region="uk-en")
