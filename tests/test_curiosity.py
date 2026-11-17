"""
Tests for src/core/curiosity.py — the REFLECT edge that closes the study
loop (docs/design/autonomous_self_improvement.md §A.2/A.3 step 6).

Strategy: a tmp StudyQueue + a tmp charter file + an injected generate_fn,
so the whole pipeline is deterministic and never touches Ollama or the
real data/ files.

Checks:
- charter parsing (bullets, headings, blockquotes, missing file)
- candidate parsing (bullets/numbering stripped, dedup, bounds, cap)
- relevance scoring (overlap, accent-insensitive, zero-anchor rejected)
- reflect(): happy path enqueues with provenance, cap respected, dupes and
  transients rejected by the queue's own guards, no charter → refuses,
  generation failure → fail-soft empty
- the feature flag is off by default
- NightlyStudier._maybe_reflect wiring: no-op when off, fail-soft when on
"""
import os
from unittest.mock import patch

import pytest

from src.core.curiosity import (
    CuriosityEngine,
    TopicProposal,
    _curiosity_enabled,
    _tokens,
    load_charter,
)
from src.core.study_queue import StudyQueue


CHARTER = """# Study Charter — test

> comment line, not a goal

- Python, FastAPI, SQLite, pytest
- Local LLMs: Ollama, quantization, RAG, embeddings
- Voice pipelines: Whisper, TTS, wake word
"""


@pytest.fixture
def charter_path(tmp_path):
    p = tmp_path / "study_charter.md"
    p.write_text(CHARTER, encoding="utf-8")
    return str(p)


@pytest.fixture
def queue(tmp_path):
    return StudyQueue(path=str(tmp_path / "study_queue.jsonl"))


def make_engine(queue, charter_path, replies, max_proposals=2):
    """Engine with a canned generate_fn returning `replies` verbatim."""
    return CuriosityEngine(
        queue=queue,
        charter_path=charter_path,
        max_proposals=max_proposals,
        generate_fn=lambda prompt: replies,
    )


class TestCharter:
    def test_parses_goal_lines_skipping_structure(self, charter_path):
        goals = load_charter(charter_path)
        assert len(goals) == 3
        assert goals[0].startswith("Python")
        assert all(not g.startswith(("-", "#", ">")) for g in goals)

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_charter(str(tmp_path / "nope.md")) == []

    def test_env_override_path(self, tmp_path, monkeypatch):
        p = tmp_path / "alt.md"
        p.write_text("- goal one\n", encoding="utf-8")
        monkeypatch.setenv("CORTEX_STUDY_CHARTER", str(p))
        assert load_charter() == ["goal one"]

    def test_repo_charter_exists_and_has_goals(self):
        # The engine refuses to run without a charter — the repo ships one.
        assert len(load_charter("data/study_charter.md")) >= 3


class TestCandidateParsing:
    def test_strips_bullets_numbering_quotes(self):
        raw = '- RAG chunking strategies\n2) "Ollama keep_alive tuning".\n* Whisper VAD internals'
        out = CuriosityEngine._parse_candidates(raw)
        assert out == [
            "RAG chunking strategies",
            "Ollama keep_alive tuning",
            "Whisper VAD internals",
        ]

    def test_drops_empty_short_and_oversized(self):
        raw = "\n".join(["", "ab", "x" * 200, "valid topic here"])
        assert CuriosityEngine._parse_candidates(raw) == ["valid topic here"]

    def test_dedupes_normalised_variants(self):
        raw = "RAG chunking strategies\nrag-chunking strategies!\n"
        assert len(CuriosityEngine._parse_candidates(raw)) == 1

    def test_caps_at_max_candidates(self):
        raw = "\n".join(f"distinct topic alpha{i} here" for i in range(10))
        assert len(CuriosityEngine._parse_candidates(raw)) == \
            CuriosityEngine.MAX_CANDIDATES

    def test_empty_generation_yields_nothing(self):
        assert CuriosityEngine._parse_candidates("") == []

    def test_drops_preamble_line_ending_in_colon(self):
        # LLM leaked a preamble as a candidate (live 2026-07-10).
        raw = ("Aqui estão minhas sugestões de estudos subsequentes:\n"
               "Uso seguro do monkeypatch para evitar side effects")
        out = CuriosityEngine._parse_candidates(raw)
        assert out == ["Uso seguro do monkeypatch para evitar side effects"]


class TestRelevance:
    def test_overlap_ratio(self):
        charter = _tokens("Python FastAPI SQLite pytest")
        # 2 of 3 non-stopword tokens anchor on the charter
        rel = CuriosityEngine._relevance("FastAPI dependency injection",
                                         charter)
        assert 0 < rel < 1

    def test_zero_overlap_is_zero(self):
        charter = _tokens("Python FastAPI")
        assert CuriosityEngine._relevance("medieval gardening", charter) == 0.0

    def test_accent_insensitive(self):
        charter = _tokens("quantização de modelos")
        assert CuriosityEngine._relevance("quantizacao avancada",
                                          charter) > 0

    def test_stopwords_never_anchor(self):
        charter = _tokens("o estudo de Python para a web")
        assert CuriosityEngine._relevance("de para o a", charter) == 0.0


class TestReflect:
    def test_happy_path_enqueues_with_provenance(self, queue, charter_path):
        eng = make_engine(queue, charter_path,
                          "RAG chunking strategies\nWhisper VAD tuning")
        entries = eng.reflect("RAG basics", "briefing text about RAG")
        assert len(entries) == 2
        pending = queue.list_pending()
        assert len(pending) == 2
        for e in pending:
            assert e["origin"] == "curiosity"
            assert e["parent_topic"] == "RAG basics"
            assert 0 < e["score"] <= 1
            assert e["status"] == "pending"

    def test_cap_keeps_best_scored(self, queue, charter_path):
        # 3 relevant candidates, cap 1 → only the highest-overlap survives.
        eng = make_engine(
            queue, charter_path,
            "FastAPI Python pytest SQLite\n"       # relevance 1.0
            "Whisper tuning for gardens\n"          # partial
            "embeddings in ancient pottery\n",      # partial
            max_proposals=1,
        )
        entries = eng.reflect("Python testing", "briefing")
        assert len(entries) == 1
        assert entries[0]["topic"] == "FastAPI Python pytest SQLite"

    def test_irrelevant_candidates_dropped(self, queue, charter_path):
        eng = make_engine(queue, charter_path,
                          "medieval gardening techniques")
        assert eng.reflect("RAG basics", "briefing") == []
        assert queue.list_pending() == []

    def test_parent_topic_anchor_rescues_candidate(self, queue, charter_path):
        # "monkeypatch" is NOT in the charter, so on charter-alone this
        # would be dropped as irrelevant (the live 2026-07-10 bug). But the
        # PARENT topic contains "monkeypatch", so the follow-up anchors on
        # it and survives — the "borda do que já sabe".
        eng = make_engine(
            queue, charter_path,
            "monkeypatch teardown ordering gotchas",
        )
        entries = eng.reflect(
            "pytest fixtures: parametrize e monkeypatch", "briefing",
        )
        assert len(entries) == 1
        assert "monkeypatch" in entries[0]["topic"]

    def test_duplicate_in_queue_skipped(self, queue, charter_path):
        queue.add("RAG chunking strategies")
        eng = make_engine(queue, charter_path, "RAG chunking strategies")
        assert eng.reflect("RAG basics", "briefing") == []
        assert len(queue.list_pending()) == 1  # only the original

    def test_transient_candidate_rejected_by_queue_guard(
        self, queue, charter_path
    ):
        # Relevant tokens + a transient phrase: survives relevance but the
        # queue's own TRANSIENT_TOPIC guard must veto it (never force-added).
        eng = make_engine(queue, charter_path,
                          "latest news about Python FastAPI")
        assert eng.reflect("Python", "briefing") == []
        assert queue.list_pending() == []

    def test_no_charter_refuses_and_skips_generation(self, queue, tmp_path):
        called = {"n": 0}

        def gen(prompt):
            called["n"] += 1
            return "Python topic"

        eng = CuriosityEngine(queue=queue,
                              charter_path=str(tmp_path / "missing.md"),
                              max_proposals=2, generate_fn=gen)
        assert eng.reflect("Python", "briefing") == []
        assert called["n"] == 0  # safeguard runs BEFORE any model call
        assert queue.list_pending() == []

    def test_generation_failure_is_fail_soft(self, queue, charter_path):
        def boom(prompt):
            raise RuntimeError("ollama down")

        eng = CuriosityEngine(queue=queue, charter_path=charter_path,
                              max_proposals=2, generate_fn=boom)
        assert eng.reflect("Python", "briefing") == []

    def test_briefing_capped_in_prompt(self, queue, charter_path):
        captured = {}

        def gen(prompt):
            captured["prompt"] = prompt
            return ""

        eng = CuriosityEngine(queue=queue, charter_path=charter_path,
                              generate_fn=gen)
        eng.reflect("Python", "x" * 10_000)
        assert len(captured["prompt"]) < 5_000
        assert "Topic studied: Python" in captured["prompt"]

    def test_proposals_wait_for_next_round(self, queue, charter_path):
        # run() snapshots via consume_all BEFORE studying; anything the
        # reflection enqueues afterwards must stay pending for next night.
        queue.add("Python basics")
        consumed = queue.consume_all()
        assert len(consumed) == 1
        eng = make_engine(queue, charter_path, "FastAPI routing internals")
        eng.reflect("Python basics", "briefing")
        pending = queue.list_pending()
        assert [e["topic"] for e in pending] == ["FastAPI routing internals"]


class TestQueueCap:
    def test_full_queue_blocks_reflection_before_model_call(
        self, queue, charter_path
    ):
        for t in ("SQLite WAL internals", "Whisper audio pipelines",
                  "Kubernetes networking basics"):
            queue.add(t)
        called = {"n": 0}

        def gen(prompt):
            called["n"] += 1
            return "FastAPI Python pytest"

        eng = CuriosityEngine(queue=queue, charter_path=charter_path,
                              queue_cap=3, generate_fn=gen)
        assert eng.reflect("Python", "briefing") == []
        assert called["n"] == 0  # no model call wasted on a full queue
        assert len(queue.list_pending()) == 3  # nothing added

    def test_below_cap_proceeds(self, queue, charter_path):
        queue.add("seeded distinct topic")
        eng = CuriosityEngine(queue=queue, charter_path=charter_path,
                              queue_cap=5,
                              generate_fn=lambda p: "FastAPI Python pytest")
        assert len(eng.reflect("Python", "briefing")) == 1

    def test_env_default(self, queue, charter_path, monkeypatch):
        monkeypatch.setenv("CORTEX_CURIOSITY_QUEUE_CAP", "7")
        eng = CuriosityEngine(queue=queue, charter_path=charter_path)
        assert eng.queue_cap == 7


class TestScoreMath:
    def test_score_is_relevance_times_novelty(self):
        assert TopicProposal("t", 0.5, 1.0).score == 0.5
        assert TopicProposal("t", 0.5, 0.0).score == 0.0


class TestFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_CURIOSITY", raising=False)
        assert _curiosity_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
    def test_on_values(self, monkeypatch, val):
        monkeypatch.setenv("CORTEX_CURIOSITY", val)
        assert _curiosity_enabled() is True


class TestStudierWiring:
    """NightlyStudier._maybe_reflect: dormant by default, fail-soft when on."""

    def _studier(self, tmp_path):
        from src.core.studier import NightlyStudier

        return NightlyStudier(
            engine=object.__new__(type("E", (), {})),  # never touched here
            queue=StudyQueue(path=str(tmp_path / "q.jsonl")),
            web_searcher=object(),
            output_dir=str(tmp_path / "pending"),
        )

    def test_flag_off_never_builds_engine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORTEX_CURIOSITY", raising=False)
        studier = self._studier(tmp_path)
        with patch("src.core.curiosity.CuriosityEngine") as eng_cls:
            studier._maybe_reflect("topic", "briefing", None)
        eng_cls.assert_not_called()

    def test_flag_on_reflects_with_studier_queue(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_CURIOSITY", "1")
        studier = self._studier(tmp_path)
        with patch("src.core.curiosity.CuriosityEngine") as eng_cls:
            studier._maybe_reflect("topic", "briefing", "trace123")
        eng_cls.assert_called_once_with(queue=studier.queue)
        eng_cls.return_value.reflect.assert_called_once_with(
            "topic", "briefing", trace="trace123"
        )

    def test_reflect_exception_is_swallowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_CURIOSITY", "1")
        studier = self._studier(tmp_path)
        with patch("src.core.curiosity.CuriosityEngine") as eng_cls:
            eng_cls.return_value.reflect.side_effect = RuntimeError("boom")
            studier._maybe_reflect("topic", "briefing", None)  # no raise

    def test_study_one_calls_reflect_after_save(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_CURIOSITY", "1")
        studier = self._studier(tmp_path)
        monkeypatch.setattr(studier, "generate_queries", lambda t, **kw: ["q"])
        monkeypatch.setattr(studier, "gather_snippets",
                            lambda q, ml_context=False: "Source 1 (RAG basics):\nRAG retrieves documents to ground generated answers.\n[https://example.org/rag]")
        monkeypatch.setattr(studier, "_summarise",
                            lambda t, s, ml_context=False, **kw: "briefing md")
        monkeypatch.setattr(studier, "_apply_critic",
                            lambda t, b, trace=None, **kw: b)
        calls = {}
        monkeypatch.setattr(
            studier, "_maybe_reflect",
            lambda topic, briefing, trace: calls.update(
                topic=topic, briefing=briefing
            ),
        )
        path = studier._study_one("RAG basics")
        assert path  # briefing was saved first
        assert calls == {"topic": "RAG basics", "briefing": "briefing md"}


class TestQueueExtraMetadata:
    def test_extra_keys_persisted(self, queue):
        entry = queue.add("Python asyncio", extra={"origin": "curiosity",
                                                   "parent_topic": "Python"})
        assert entry["origin"] == "curiosity"
        stored = queue.list_pending()[0]
        assert stored["origin"] == "curiosity"
        assert stored["parent_topic"] == "Python"

    def test_extra_cannot_override_core_fields(self, queue):
        entry = queue.add("Python asyncio",
                          extra={"status": "consumed", "topic": "hijacked"})
        assert entry["status"] == "pending"
        assert entry["topic"] == "Python asyncio"

    def test_add_without_extra_unchanged(self, queue):
        entry = queue.add("Python asyncio")
        assert set(entry) == {"id", "topic", "added_at", "status"}
        assert len(entry["id"]) == 32 and all(c in "0123456789abcdef" for c in entry["id"])
