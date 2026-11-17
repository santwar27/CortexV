"""
Per-turn token accounting (idea #26, s69).

The engine audits Ollama's token counts per call (M2-A1); turn_metrics
folds the calls of ONE turn into a thread-local bucket so the /ui chat
can show "2.9k → 13 tok · 19s" under the reply. Contract under test:

  - add() without reset() is a no-op — the voice/CLI threads never open
    a bucket, so the engine hook costs nothing there;
  - collect() with zero calls returns None (a reflex turn has no token
    story — the UI should show nothing, not "0 tok");
  - buckets are thread-local: a concurrent turn on another thread never
    leaks into this one's numbers;
  - the engine's _log_token_usage feeds the bucket with the same counts
    it audits.
"""
import threading
from unittest.mock import patch

import pytest

from src.core import turn_metrics


class TestBucketLifecycle:

    def test_add_without_reset_is_noop(self):
        turn_metrics._local.usage = None
        turn_metrics.add(100, 10)  # must not raise nor create a bucket
        assert turn_metrics.collect() is None

    def test_reset_then_collect_sums_calls(self):
        turn_metrics.reset()
        turn_metrics.add(2900, 13)
        turn_metrics.add(100, 7)
        usage = turn_metrics.collect()
        assert usage == {"prompt_eval_count": 3000, "eval_count": 20,
                         "calls": 2, "models": [], "online": []}

    def test_collect_is_one_shot(self):
        turn_metrics.reset()
        turn_metrics.add(10, 1)
        assert turn_metrics.collect() is not None
        assert turn_metrics.collect() is None

    def test_zero_calls_collects_none(self):
        # Bucket opened but no LLM call landed (reflex/cached turn).
        turn_metrics.reset()
        assert turn_metrics.collect() is None

    def test_none_counts_tolerated(self):
        # Ollama may omit one of the counters; int(None or 0) == 0.
        turn_metrics.reset()
        turn_metrics.add(None, 5)
        usage = turn_metrics.collect()
        assert usage == {"prompt_eval_count": 0, "eval_count": 5,
                         "calls": 1, "models": [], "online": []}

    def test_buckets_are_thread_local(self):
        turn_metrics.reset()
        turn_metrics.add(100, 10)
        other = {}

        def worker():
            # No reset() on this thread: add is a no-op, collect is None.
            turn_metrics.add(999, 999)
            other["usage"] = turn_metrics.collect()

        th = threading.Thread(target=worker)
        th.start()
        th.join()
        assert other["usage"] is None
        assert turn_metrics.collect() == {
            "prompt_eval_count": 100, "eval_count": 10, "calls": 1,
            "models": [], "online": []}


class TestEngineFeedsBucket:

    @pytest.fixture
    def engine(self):
        from src.core.engine import OllamaEngine
        e = OllamaEngine.__new__(OllamaEngine)
        try:
            OllamaEngine.__init__(e)
        except Exception:
            pass

        class _StubMem:
            def retrieve_relevant_context(self, q, n_results=3,
                                          include_knowledge=True):
                return ""

            def store_interaction(self, u, r):
                pass
        e.memory = _StubMem()
        e.history = []
        return e

    def test_chat_accumulates_into_open_bucket(self, engine, monkeypatch):
        monkeypatch.setattr(
            "src.core.engine.ollama.chat",
            lambda **kw: {"message": {"content": "resposta"},
                          "prompt_eval_count": 2900, "eval_count": 13},
        )
        turn_metrics.reset()
        with patch("src.core.engine.audit"):
            engine.chat("oi")
        usage = turn_metrics.collect()
        assert usage["prompt_eval_count"] == 2900
        assert usage["eval_count"] == 13
        assert usage["calls"] == 1

    def test_chat_without_bucket_changes_nothing(self, engine, monkeypatch):
        # The voice/CLI path: no reset() — engine must work exactly as
        # before, and no bucket may appear as a side effect.
        monkeypatch.setattr(
            "src.core.engine.ollama.chat",
            lambda **kw: {"message": {"content": "resposta"},
                          "prompt_eval_count": 2900, "eval_count": 13},
        )
        turn_metrics._local.usage = None
        with patch("src.core.engine.audit"):
            reply = engine.chat("oi")
        assert reply  # turn unaffected
        assert turn_metrics.collect() is None
