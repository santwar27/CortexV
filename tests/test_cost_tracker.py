"""
CostTracker ledger + WebSearcher cost wiring.

No network, no real ~/.cortex: every tracker is built with an explicit
db_path (":memory:" or a tmp file). The WebSearcher tests stub requests
/ DDGS so only the Tavily-success path can record a billable call.
"""
from unittest.mock import MagicMock

import pytest

from src.core.cost_tracker import CostTracker


@pytest.fixture
def tracker():
    return CostTracker(db_path=":memory:")


class TestRecordAndSummary:
    def test_empty_summary(self, tracker):
        s = tracker.summary()
        assert s["total_events"] == 0
        assert s["total_cost_usd"] is None
        assert s["by_provider"] == []

    def test_record_then_summary_groups_by_provider_and_unit(self, tracker):
        tracker.record("tavily", "search", units=1, unit_kind="credit")
        tracker.record("tavily", "search", units=1, unit_kind="credit")
        tracker.record("openrouter", "chat", units=1200, unit_kind="token")
        s = tracker.summary()
        assert s["total_events"] == 3
        by = {(r["provider"], r["unit_kind"]): r for r in s["by_provider"]}
        assert by[("tavily", "credit")]["events"] == 2
        assert by[("tavily", "credit")]["units"] == 2
        assert by[("openrouter", "token")]["units"] == 1200

    def test_cost_usd_none_until_priced(self, tracker):
        tracker.record("tavily", "search")  # cost_usd defaults to None
        s = tracker.summary()
        assert s["total_cost_usd"] is None
        assert s["by_provider"][0]["cost_usd"] is None

    def test_cost_usd_sums_when_priced(self, tracker):
        tracker.record("openrouter", "chat", cost_usd=0.002)
        tracker.record("openrouter", "chat", cost_usd=0.003)
        s = tracker.summary()
        assert s["total_cost_usd"] == pytest.approx(0.005)
        assert s["by_provider"][0]["cost_usd"] == pytest.approx(0.005)

    def test_since_filter(self, tracker):
        tracker.record("tavily", "search")
        # A far-future lower bound excludes everything recorded "now".
        s = tracker.summary(since="2999-01-01T00:00:00+00:00")
        assert s["total_events"] == 0
        assert s["since"].startswith("2999")


class TestRecent:
    def test_recent_newest_first_with_metadata(self, tracker):
        tracker.record("tavily", "search", metadata={"query": "first"})
        tracker.record("tavily", "search", metadata={"query": "second"})
        rows = tracker.recent()
        assert rows[0]["metadata"]["query"] == "second"
        assert rows[1]["metadata"]["query"] == "first"

    def test_recent_limit_clamped(self, tracker):
        for i in range(5):
            tracker.record("tavily", "search")
        assert len(tracker.recent(limit=2)) == 2
        # Out-of-range limits are clamped to [1, 500], not rejected.
        assert len(tracker.recent(limit=0)) == 1
        assert len(tracker.recent(limit=99999)) == 5


class TestFailSoft:
    def test_unopenable_db_degrades_silently(self, tmp_path):
        # Pointing the ledger at a directory makes sqlite connect fail.
        broken = CostTracker(db_path=str(tmp_path))
        broken.record("tavily", "search")  # must not raise
        assert broken.summary()["total_events"] == 0
        assert broken.recent() == []

    def test_file_backed_roundtrip(self, tmp_path):
        path = str(tmp_path / "costs.db")
        t1 = CostTracker(db_path=path)
        t1.record("tavily", "search", units=1, unit_kind="credit")
        # A fresh tracker on the same file sees the persisted row.
        t2 = CostTracker(db_path=path)
        assert t2.summary()["total_events"] == 1


# =====================================================================
# WebSearcher wiring — only the Tavily-success path is billable
# =====================================================================

def _tavily_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


class TestWebSearcherCostWiring:
    def test_tavily_success_records_one_credit(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", "tavily")
        monkeypatch.setattr(
            "src.tools.web_search.requests.post",
            lambda *a, **k: _tavily_response({
                "answer": "42",
                "results": [{"title": "T", "content": "c", "url": "u"}],
            }),
        )
        from src.tools.web_search import WebSearcher
        tracker = CostTracker(db_path=":memory:")
        searcher = WebSearcher(cost_tracker=tracker)
        out = searcher.search("meaning of life")
        assert out == "Source 1 (T):\nc\n[u]"
        assert "42" not in out
        s = tracker.summary()
        assert s["total_events"] == 1
        assert s["by_provider"][0]["provider"] == "tavily"
        assert s["by_provider"][0]["unit_kind"] == "credit"

    def test_ddg_fallback_records_nothing(self, monkeypatch):
        # No Tavily key → straight to DDG, which is free and must not log.
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("CORTEX_SEARCH_PROVIDER", "tavily")
        fake_ddgs = MagicMock()
        fake_ddgs.__enter__ = MagicMock(return_value=MagicMock(
            text=MagicMock(return_value=[{"title": "x", "body": "y", "href": "z"}])
        ))
        fake_ddgs.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr("src.tools.web_search.DDGS", lambda *a, **k: fake_ddgs)
        from src.tools.web_search import WebSearcher
        tracker = CostTracker(db_path=":memory:")
        searcher = WebSearcher(cost_tracker=tracker)
        searcher.search("free query")
        assert tracker.summary()["total_events"] == 0
