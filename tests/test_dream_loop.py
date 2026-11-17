"""
Tests for src/core/dream_loop.py — Dream Loop pipeline + store (scaffold #5).

Strategy: a FakePalace returning canned documents and an injected generate_fn,
plus a seeded RNG, so the whole pipeline is deterministic and never touches
Ollama or a real Palace.

Checks:
- dream() samples N chunks, builds a prompt with the fragments, stores pending
- empty Palace / empty generation -> None (no crash)
- DreamStore save/list/get/set_status/approve/reject round-trip
- pending() and latest() surfaces
- the feature flag is off by default
"""
import os
import random
from unittest.mock import patch

import pytest

from src.core.dream_loop import Dream, DreamLoop, DreamStore, _dream_enabled


class FakePalace:
    def __init__(self, docs):
        self._docs = docs

    def list_documents(self, tipo=None, limit=100):  # noqa: ARG002
        return list(self._docs[:limit])


def _docs(n):
    return [{"id": i, "texto": f"fragment number {i} about topic {i}",
             "tipo": "conversa"} for i in range(1, n + 1)]


@pytest.fixture
def store(tmp_path):
    return DreamStore(path=str(tmp_path / "dreams.jsonl"))


@pytest.fixture
def loop(store):
    palace = FakePalace(_docs(10))
    return DreamLoop(
        palace=palace,
        generate_fn=lambda prompt: "um sonho curioso conectando os fragmentos",
        store=store,
        n_chunks=3,
        rng=random.Random(42),
    )


class TestDreamPipeline:
    def test_dream_creates_pending(self, loop):
        d = loop.dream()
        assert isinstance(d, Dream)
        assert d.status == "pending"
        assert d.text == "um sonho curioso conectando os fragmentos"
        assert len(d.source_doc_ids) == 3
        assert len(d.source_snippets) == 3

    def test_dream_persisted(self, loop, store):
        d = loop.dream()
        assert store.get(d.id) is not None
        assert len(store.pending()) == 1

    def test_prompt_includes_fragments(self, store):
        captured = {}

        def gen(prompt):
            captured["prompt"] = prompt
            return "dream text"

        loop = DreamLoop(palace=FakePalace(_docs(5)), generate_fn=gen,
                         store=store, n_chunks=2, rng=random.Random(1))
        loop.dream()
        assert "Fragments:" in captured["prompt"]
        assert "fragment number" in captured["prompt"]

    def test_samples_min_of_n_and_pool(self, store):
        # Only 2 docs available but n_chunks=3 -> sample 2, no crash.
        loop = DreamLoop(palace=FakePalace(_docs(2)), generate_fn=lambda p: "x",
                         store=store, n_chunks=3, rng=random.Random(0))
        d = loop.dream()
        assert len(d.source_doc_ids) == 2

    def test_empty_palace_returns_none(self, store):
        loop = DreamLoop(palace=FakePalace([]), generate_fn=lambda p: "x",
                         store=store, rng=random.Random(0))
        assert loop.dream() is None

    def test_empty_generation_returns_none(self, store):
        loop = DreamLoop(palace=FakePalace(_docs(5)), generate_fn=lambda p: "   ",
                         store=store, rng=random.Random(0))
        assert loop.dream() is None

    def test_blank_docs_skipped(self, store):
        docs = [{"id": 1, "texto": "   "}, {"id": 2, "texto": ""}]
        loop = DreamLoop(palace=FakePalace(docs), generate_fn=lambda p: "x",
                         store=store, rng=random.Random(0))
        assert loop.dream() is None

    def test_generator_failure_is_soft(self, store):
        def boom(prompt):
            raise RuntimeError("model down")
        loop = DreamLoop(palace=FakePalace(_docs(5)), generate_fn=boom,
                         store=store, rng=random.Random(0))
        assert loop.dream() is None  # degrades, no crash


class TestReview:
    def test_approve_marks_kept(self, loop, store):
        d = loop.dream()
        assert loop.approve(d.id) is True
        assert store.get(d.id).status == "kept"
        assert store.pending() == []

    def test_reject_marks_discarded(self, loop, store):
        d = loop.dream()
        assert loop.reject(d.id) is True
        assert store.get(d.id).status == "discarded"

    def test_latest_returns_most_recent_pending(self, loop):
        d1 = loop.dream()
        d2 = loop.dream()
        latest = loop.latest()
        assert latest.id == d2.id
        loop.approve(d2.id)
        assert loop.latest().id == d1.id

    def test_set_status_unknown_id(self, store):
        assert store.set_status("nope", "kept") is False


class TestStore:
    def test_save_and_get(self, store):
        d = store.save("a dream", [1, 2], ["frag1", "frag2"])
        got = store.get(d.id)
        assert got.text == "a dream"
        assert got.source_doc_ids == [1, 2]
        assert got.status == "pending"

    def test_list_by_status(self, store):
        a = store.save("d1", [1], ["f"])
        b = store.save("d2", [2], ["f"])
        store.approve(a.id)
        assert [d.id for d in store.list_by_status("kept")] == [a.id]
        assert [d.id for d in store.list_by_status("pending")] == [b.id]

    def test_missing_file_empty(self, tmp_path):
        s = DreamStore(path=str(tmp_path / "none.jsonl"))
        assert s.pending() == []
        assert s.latest_pending() is None

    def test_tolerates_schema_drift(self, tmp_path):
        import json as _json
        path = tmp_path / "dreams.jsonl"
        rows = [
            # valid
            {"id": "a1", "created_at": "t", "source_doc_ids": [1],
             "source_snippets": ["s"], "text": "ok", "status": "pending"},
            # unknown future key -> tolerated (key dropped, row kept)
            {"id": "a2", "created_at": "t", "source_doc_ids": [2],
             "source_snippets": ["s"], "text": "ok2", "status": "pending",
             "future_field": 123},
            # missing required field 'text' -> skipped, not a crash
            {"id": "a3", "created_at": "t", "source_doc_ids": [3],
             "source_snippets": ["s"], "status": "pending"},
            "not json at all",  # garbage line -> skipped
        ]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write((r if isinstance(r, str) else _json.dumps(r)) + "\n")
        s = DreamStore(path=str(path))
        assert {d.id for d in s.pending()} == {"a1", "a2"}
        assert s.get("a2").text == "ok2"
        assert s.get("a3") is None


class TestFlag:
    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_DREAM_LOOP", None)
            assert _dream_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
    def test_on_values(self, val):
        with patch.dict(os.environ, {"CORTEX_DREAM_LOOP": val}):
            assert _dream_enabled() is True
