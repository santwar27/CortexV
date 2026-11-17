"""
WebSearcher hardening (M4-A4, review Fable 5).

The DDG fallback ran with `verify=False` — TLS off on the exact path that
feeds web content into the LLM context, so any on-path attacker (public
Wi-Fi, DNS poisoning) could inject forged "results" as trusted sources.
These tests pin verify ON and the no-longer-silent Tavily fallback.
"""
import logging
from unittest.mock import MagicMock, patch

from src.tools.web_search import WebSearcher


def _searcher(monkeypatch, provider="ddg", tavily_key=None):
    monkeypatch.delenv("CORTEX_SEARCH_COUNTRY", raising=False)
    if tavily_key:
        monkeypatch.setenv("TAVILY_API_KEY", tavily_key)
    else:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", provider)
    return WebSearcher(cost_tracker=MagicMock())


class TestDDGVerifiesTLS:
    def test_ddgs_is_not_built_with_verify_false(self, monkeypatch):
        searcher = _searcher(monkeypatch, provider="ddg")
        with patch("src.tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.__enter__.return_value.text.return_value = []
            searcher.search("qualquer coisa")
        ddgs_cls.assert_called_once_with()  # no kwargs — verify stays True

    def test_verify_false_never_returns_to_the_source(self):
        import inspect
        import src.tools.web_search as ws
        assert "verify=False" not in inspect.getsource(ws), (
            "M4-A4 regression: the DDG fallback must never disable TLS "
            "verification — forged results would enter the LLM context."
        )


class TestTavilyFallbackIsLoud:
    def test_tavily_failure_logs_warning_before_ddg(self, monkeypatch, caplog):
        searcher = _searcher(monkeypatch, provider="tavily", tavily_key="tk")
        import requests as _requests
        with patch("src.tools.web_search.requests.post",
                   side_effect=_requests.RequestException("down")), \
             patch("src.tools.web_search.DDGS") as ddgs_cls, \
             caplog.at_level(logging.WARNING, logger="src.tools.web_search"):
            ddgs_cls.return_value.__enter__.return_value.text.return_value = []
            searcher.search("noticias de hoje")
        assert any("falling back to DDG" in r.message for r in caplog.records), (
            "Tavily degradation must be visible in production logs"
        )


class TestSourceContentIsCapped:
    """Long per-source blobs bloat the LLM context — slow synthesis + noise the
    model mis-picks from (s80: a 27s Flamengo turn on a fixture-list dump)."""

    def test_tavily_content_truncated_to_cap(self, monkeypatch):
        searcher = _searcher(monkeypatch, provider="tavily", tavily_key="tk")
        long_content = "x" * 5000
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "answer": "",
            "results": [{"title": "T", "content": long_content,
                         "url": "https://e/1"}],
        }
        with patch("src.tools.web_search.requests.post", return_value=resp):
            out = searcher.search("resultado do último jogo do flamengo")
        # the source body is capped; the whole context stays bounded
        assert "x" * WebSearcher.MAX_SOURCE_CHARS in out
        assert "x" * (WebSearcher.MAX_SOURCE_CHARS + 1) not in out

    def test_ddg_body_truncated_to_cap(self, monkeypatch):
        searcher = _searcher(monkeypatch, provider="ddg")
        long_body = "y" * 5000
        with patch("src.tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.__enter__.return_value.text.return_value = [
                {"title": "T", "body": long_body, "href": "https://e/1"},
            ]
            out = searcher.search("resultado do último jogo do flamengo")
        assert "y" * WebSearcher.MAX_SOURCE_CHARS in out
        assert "y" * (WebSearcher.MAX_SOURCE_CHARS + 1) not in out
