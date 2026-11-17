"""
Tests for src/core/deep_studier.py — DeepStudier multi-step research loop.

Strategy: inject a FakeStudier (so the per-step query/search primitives are
deterministic, no real LLM/Tavily) and patch deep_studier.ollama.generate to
return canned plan / step / deep-synthesis text routed by prompt content.

Checks:
- plan parsing (numbered/bulleted lines cleaned, deduped, capped)
- the SEARCH BUDGET is enforced (caps both per-step and global total)
- citations are extracted from snippets and deduped across steps
- plan failure falls back to the literal topic
- report structure + save format
- the feature flag is dormant by default
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.core.deep_studier as ds_mod
from src.core.deep_studier import (
    DeepStudier,
    DeepStudyReport,
    ResearchStep,
    _deep_enabled,
)


class FakeStudier:
    """Stand-in for NightlyStudier exposing only the primitives DeepStudier
    reuses. Records gather calls so budget enforcement is observable."""

    def __init__(self, n_queries: int = 4):
        self.engine = SimpleNamespace(model_name="fake-model")
        self._n_queries = n_queries
        self.gather_calls: list[list[str]] = []

    def is_ml_topic(self, topic: str) -> bool:
        return False

    def generate_queries(self, subq: str, *, study_time=None):
        # More than STEP_MAX_QUERIES so trimming is exercised.
        return [f"{subq} q{i}" for i in range(self._n_queries)]

    def gather_snippets(self, queries, ml_context: bool = False) -> str:
        self.gather_calls.append(list(queries))
        # Each query yields a block with one URL; reuse one URL across two
        # queries so cross-step dedup is exercised.
        blocks = []
        for i, q in enumerate(queries):
            url = "https://example.com/shared" if i == 0 else f"https://example.com/{abs(hash(q)) % 1000}"
            blocks.append(f"Source ({q}):\nsome content about {q}\n[{url}]")
        return "\n\n".join(blocks)


def _fake_generate(*, plan_lines=None):
    """Build an ollama.generate replacement that routes by prompt content."""
    plan_lines = plan_lines or ["What is X?", "How does X work?", "X trade-offs"]
    plan_text = "\n".join(plan_lines)

    def _gen(model, prompt, options=None, **kwargs):  # noqa: ARG001
        # **kwargs tolerates keep_alive (M8-A6), like the real ollama.generate.
        if "Sub-questions:" in prompt:
            return {"response": plan_text}
        if prompt.strip().endswith("Answer:"):
            return {"response": "Concise step finding."}
        # deep synthesis
        return {"response": "## Resumo executivo\nDeep briefing.\n"}

    return _gen


@pytest.fixture
def deep(tmp_path):
    return DeepStudier(studier=FakeStudier(), output_dir=str(tmp_path))


class TestPlanning:
    def test_plan_parsed_and_cleaned(self, deep):
        lines = ["1. What is RAG?", "- How does RAG index?", '"RAG vs fine-tuning"']
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=lines)
            subs = deep._plan_subquestions("RAG")
        assert subs == ["What is RAG?", "How does RAG index?", "RAG vs fine-tuning"]

    def test_plan_capped_at_max(self, deep):
        many = [f"sub question number {i}" for i in range(10)]
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=many)
            subs = deep._plan_subquestions("topic")
        assert len(subs) == DeepStudier.MAX_SUBQUESTIONS

    def test_plan_dedup(self, deep):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(
                plan_lines=["same question here", "same question here", "other question"])
            subs = deep._plan_subquestions("t")
        assert subs == ["same question here", "other question"]

    def test_plan_failure_falls_back_to_topic(self, deep):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = RuntimeError("ollama down")
            assert deep._plan_subquestions("my topic") == ["my topic"]

    def test_empty_plan_response_falls_back(self, deep):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=["", "  ", "a"])  # all too short
            assert deep._plan_subquestions("topic xyz") == ["topic xyz"]


class TestBudget:
    def test_per_step_query_cap(self, deep):
        # FakeStudier offers 4 queries/step; STEP_MAX_QUERIES caps it.
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=["q one here", "q two here"])
            report = deep.research("topic")
        for call in deep.studier.gather_calls:
            assert len(call) <= DeepStudier.STEP_MAX_QUERIES

    def test_global_budget_caps_total_searches(self, deep):
        deep.MAX_SEARCHES_TOTAL = 7
        deep.STEP_MAX_QUERIES = 3
        plan = [f"sub question {i} here" for i in range(5)]
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=plan)
            report = deep.research("topic")
        total = sum(len(c) for c in deep.studier.gather_calls)
        assert total == 7, f"expected exactly the budget, got {total}"
        # 3 + 3 + 1 -> 3 steps, then budget exhausted.
        assert len(report.steps) == 3

    def test_budget_never_exceeded_even_with_many_subqs(self, deep):
        plan = [f"sub question {i} here" for i in range(5)]
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=plan)
            deep.research("topic")
        total = sum(len(c) for c in deep.studier.gather_calls)
        assert total <= DeepStudier.MAX_SEARCHES_TOTAL


class TestCitations:
    def test_extract_sources_dedups(self, deep):
        text = ("a\n[https://x.com/1]\n\nb\n[https://x.com/1]\n\nc\n[https://x.com/2]")
        assert deep._extract_sources(text) == ["https://x.com/1", "https://x.com/2"]

    def test_extract_sources_empty(self, deep):
        assert deep._extract_sources("no urls here") == []

    def test_report_aggregates_and_dedups_across_steps(self, deep):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=["q one here", "q two here"])
            report = deep.research("topic")
        # The shared URL must appear exactly once in the aggregated list.
        assert report.sources.count("https://example.com/shared") == 1

    def test_dedup_helper(self, deep):
        assert deep._dedup(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


class TestInputValidation:
    def test_empty_topic_raises(self, deep):
        with pytest.raises(ValueError):
            deep.research("   ")


class TestReportAndSave:
    def test_research_returns_report(self, deep):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=["q one here", "q two here"])
            report = deep.research("my topic")
        assert isinstance(report, DeepStudyReport)
        assert report.topic == "my topic"
        assert len(report.steps) == 2
        assert all(isinstance(s, ResearchStep) for s in report.steps)
        assert report.briefing.startswith("## Resumo")

    def test_synthesise_deep_handles_no_steps(self, deep):
        assert "Erro" in deep._synthesise_deep("t", [])

    def test_run_and_save_writes_file(self, deep, tmp_path):
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate(plan_lines=["q one here", "q two here"])
            path = deep.run_and_save("My Topic!")
        assert os.path.exists(path)
        assert os.path.basename(path).startswith("deep_my_topic")
        content = open(path, encoding="utf-8").read()
        assert "# Deep study: My Topic!" in content
        assert "Sub-questions researched" in content


class TestDormantFlag:
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_DEEP_STUDIER", None)
            assert _deep_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
    def test_enabled_values(self, val):
        with patch.dict(os.environ, {"CORTEX_DEEP_STUDIER": val}):
            assert _deep_enabled() is True

    def test_research_works_regardless_of_flag(self, deep):
        # The module logic is not gated by the flag — only the (future)
        # orchestrator dispatch is. research() must work with the flag off.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_DEEP_STUDIER", None)
            with patch.object(ds_mod, "ollama") as m:
                m.generate.side_effect = _fake_generate(plan_lines=["q one here"])
                report = deep.research("topic")
        assert isinstance(report, DeepStudyReport)


# ---------------------------------------------------------------------------
# M8-A3 — research() under its own SystemLease
# ---------------------------------------------------------------------------


def _lease_events(event=None):
    import json
    from src.core.audit import audit
    out = []
    try:
        with open(audit.path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("stage") != "system_lease":
                    continue
                if event and rec.get("event") != event:
                    continue
                out.append(rec)
    except FileNotFoundError:
        pass
    return out


class GovernableFakeStudier(FakeStudier):
    """FakeStudier that actually routes searches through web_searcher —
    so the GovernedWebSearcher swap is observable. Mirrors the real
    gather_snippets contract: a denied search is skipped, not raised."""

    def __init__(self, n_queries: int = 2):
        super().__init__(n_queries)
        self.search_calls: list[str] = []

        class _Inner:
            def __init__(inner_self, outer):
                inner_self._outer = outer

            def search(inner_self, query, max_results: int = 3):
                inner_self._outer.search_calls.append(query)
                return f"content about {query}\n[https://example.com/x]"

        self.web_searcher = _Inner(self)

    def gather_snippets(self, queries, ml_context: bool = False) -> str:
        self.gather_calls.append(list(queries))
        blocks = []
        for q in queries:
            try:
                blocks.append(self.web_searcher.search(q))
            except Exception:
                continue  # denied/failed query → thinner snippets
        return "\n\n".join(blocks)


class TestResearchLease:
    def _run(self, tmp_path):
        studier = GovernableFakeStudier()
        deep = DeepStudier(studier=studier, output_dir=str(tmp_path))
        raw = studier.web_searcher
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate()
            report = deep.research("LoRA")
        return studier, raw, report

    def test_searches_run_through_a_lease_and_are_audited(self, tmp_path):
        studier, raw, report = self._run(tmp_path)
        assert studier.search_calls, "searches must have happened"
        acquires = _lease_events("acquire")
        assert any(e["payload"]["origin"] == "deep-studier" for e in acquires)
        calls = [e for e in _lease_events("call")
                 if e["payload"]["origin"] == "deep-studier"]
        assert len(calls) == len(studier.search_calls)
        assert _lease_events("release")

    def test_lease_max_calls_is_pinned_to_search_budget(self, tmp_path):
        """The lease cap and MAX_SEARCHES_TOTAL must never drift apart —
        the lease is the governance mirror of the budget."""
        self._run(tmp_path)
        acq = [e for e in _lease_events("acquire")
               if e["payload"]["origin"] == "deep-studier"][0]
        assert acq["payload"]["max_calls"] == DeepStudier.MAX_SEARCHES_TOTAL

    def test_raw_searcher_restored_after_research(self, tmp_path):
        studier, raw, _ = self._run(tmp_path)
        assert studier.web_searcher is raw

    def test_kill_switch_degrades_to_no_searches(self, tmp_path, monkeypatch):
        """CORTEX_SYSTEM_LEASE=0: research completes (no crash), zero
        searches reach the inner tool, denials are audited."""
        monkeypatch.setenv("CORTEX_SYSTEM_LEASE", "0")
        studier, raw, report = self._run(tmp_path)
        assert studier.search_calls == []
        assert report.briefing  # still produced (degraded)
        denies = [e for e in _lease_events("deny")
                  if e["payload"]["origin"] == "deep-studier"]
        assert denies

    def test_studier_without_searcher_attr_still_works(self, tmp_path):
        """A studier exposing only gather_snippets (no web_searcher) must
        not crash research() — there's nothing to govern at this layer."""
        deep = DeepStudier(studier=FakeStudier(), output_dir=str(tmp_path))
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate()
            report = deep.research("LoRA")
        assert report.briefing


# ---------------------------------------------------------------------------
# M8-A4 — deep studies live in their OWN directory
# ---------------------------------------------------------------------------


class TestDeepOutputDir:
    def test_default_output_dir_is_not_pending(self):
        """digest/daily/studies_review glob data/knowledge/pending/*.md
        blindly — a deep study saved there would be counted as a regular
        Studier briefing and enter the review queue with no consumer
        distinguishing the deep_ prefix. Signature check (no side effects:
        constructing with the default would makedirs the real path)."""
        import inspect
        default = inspect.signature(DeepStudier.__init__).parameters[
            "output_dir"].default
        assert default == "data/knowledge/deep"
        from src.core import digest
        assert default != digest._STUDIER_DIR

    def test_save_writes_deep_prefixed_file_to_output_dir(self, tmp_path):
        deep = DeepStudier(studier=FakeStudier(), output_dir=str(tmp_path))
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate()
            path = deep.run_and_save("LoRA")
        assert os.path.dirname(path) == str(tmp_path)
        assert os.path.basename(path).startswith("deep_")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# M8-A2 — the deep queue + nightly consumption
# ---------------------------------------------------------------------------


class TestDeepQueueNightly:
    def _nightly(self, tmp_path, monkeypatch, regular_topics=()):
        from unittest.mock import MagicMock

        from src.core.studier import NightlyStudier
        from src.core.study_queue import StudyQueue

        queue = StudyQueue(path=str(tmp_path / "queue.jsonl"))
        for t in regular_topics:
            queue.add(t)
        searcher = MagicMock()
        searcher.search.return_value = (
            "Source 1 (Fotossíntese em plantas):\nA fotossíntese em plantas converte luz em energia química.\n[https://example.org/a]"
        )
        engine = MagicMock()
        engine.chat.return_value = "1. consulta um\n2. consulta dois"
        engine.model_name = "fake-model"
        s = NightlyStudier(
            engine=engine, queue=queue, web_searcher=searcher,
            output_dir=str(tmp_path / "pending"),
        )
        monkeypatch.setattr(s, "_backup_palace", lambda: None)
        monkeypatch.setattr(s, "_consolidate_memory", lambda: None)
        monkeypatch.setattr(s, "_consolidate_preferences", lambda: None)
        return s, searcher

    def test_deep_only_night_still_runs(self, tmp_path, monkeypatch):
        """Regular queue empty + deep queue waiting must NOT short-circuit
        (the early-return covered only the regular queue before s66)."""
        monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
        monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep.jsonl"))
        ds_mod.deep_queue().add("quantização de modelos")
        s, _ = self._nightly(tmp_path, monkeypatch)
        with patch.object(ds_mod, "ollama") as m:
            m.generate.side_effect = _fake_generate()
            produced = s.run()
        assert produced, "the deep topic must produce a saved briefing"
        assert ds_mod.deep_queue().list_pending() == []

    def test_flag_off_leaves_deep_queue_untouched(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORTEX_DEEP_STUDIER", raising=False)
        monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep.jsonl"))
        ds_mod.deep_queue().add("quantização de modelos")
        s, _ = self._nightly(tmp_path, monkeypatch,
                             regular_topics=["fotossíntese em plantas"])
        s.run()
        # Deep topic still pending — dormant means dormant.
        assert len(ds_mod.deep_queue().list_pending()) == 1

    def test_deep_failure_does_not_kill_the_night(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_DEEP_STUDIER", "1")
        monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep.jsonl"))
        ds_mod.deep_queue().add("quantização de modelos")
        s, _ = self._nightly(tmp_path, monkeypatch,
                             regular_topics=["fotossíntese em plantas"])

        class _Boom:
            def __init__(self, *a, **k):
                pass

            def run_and_save(self, topic, trace=None):
                raise RuntimeError("deep exploded")

        monkeypatch.setattr("src.core.studier.DeepStudier", _Boom,
                            raising=False)
        monkeypatch.setattr(ds_mod, "DeepStudier", _Boom)
        # The regular topic's synthesis, simulated: this used to reach the
        # real ollama.generate with "fake-model", fail, and count the saved
        # "## Erro" text as output (s88, Codex lote 14).
        import src.core.studier as st_mod
        monkeypatch.setattr(st_mod.ollama, "generate", lambda **kw: {
            "response": "## Resumo executivo\nFotossíntese.\n\n## Confiança\nbaixa"})
        produced = s.run()  # must not raise
        assert produced, "the regular topic must still produce output"
