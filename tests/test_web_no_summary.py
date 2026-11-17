"""Tavily context contains source snippets, never its generated answer."""
from unittest.mock import MagicMock, patch

import pytest

from src.tools.web_search import WebSearcher


@pytest.fixture
def searcher(monkeypatch):
    monkeypatch.delenv("CORTEX_SEARCH_COUNTRY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", "tavily")
    return WebSearcher(cost_tracker=MagicMock())


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    return response


def test_explicitly_disables_provider_answer(searcher):
    with patch("src.tools.web_search.requests.post", return_value=_response({
        "results": [{"title": "T", "content": "snippet", "url": "https://e/1"}],
    })) as post:
        searcher.search("query", max_results=2)

    post.assert_called_once_with(
        WebSearcher.TAVILY_URL,
        json={
            "api_key": "tvly-test", "query": "query", "search_depth": "basic",
            "max_results": 2, "include_answer": False,
        },
        timeout=WebSearcher.TAVILY_TIMEOUT_S,
    )


@pytest.mark.parametrize("extra", [{}, {"answer": "Unsupported provider claim"}])
def test_results_are_sources_only_and_record_one_credit(searcher, extra):
    payload = {**extra, "results": [
        {"title": "T", "content": "  source evidence  ", "url": "https://e/1"},
        {},
    ]}
    with patch("src.tools.web_search.requests.post", return_value=_response(payload)), \
         patch.object(searcher, "_search_ddg") as fallback:
        context = searcher.search("query")

    assert context == (
        "Source 1 (T):\nsource evidence\n[https://e/1]\n\n"
        "Source 2 (No Title):\n\n[]"
    )
    assert "QUICK ANSWER" not in context
    assert "Unsupported provider claim" not in context
    fallback.assert_not_called()
    searcher.cost_tracker.record.assert_called_once_with(
        "tavily", "search", units=1, unit_kind="credit",
        metadata={"query": "query", "max_results": 3},
    )


@pytest.mark.parametrize("payload", [
    {"answer": "Unsupported provider claim"},
    {"answer": "Unsupported provider claim", "results": []},
    {},
    {"results": []},
])
def test_no_results_falls_back_without_recording_tavily_credit(searcher, payload):
    with patch("src.tools.web_search.requests.post", return_value=_response(payload)), \
         patch.object(searcher, "_search_ddg", return_value="DDGS sources") as fallback:
        context = searcher.search("query", max_results=2)

    assert context == "DDGS sources"
    fallback.assert_called_once_with("query", 2)
    searcher.cost_tracker.record.assert_not_called()
