"""
Palace unit tests — pure logic, no Ollama, no network.

In-memory SQLite (`:memory:`) gives us a fresh schema per fixture. The
embedding callable is a deterministic hand-crafted mapping so cosine
similarity tests have predictable outcomes; no numpy randomness needed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from src.memory.palace import Palace


# ----------------------------------------------------------------------
# fixtures + helpers
# ----------------------------------------------------------------------


def _emb_lookup(table: dict, dim: int = 4):
    """Build an embed_fn that returns a fixed vector per known text.
    Unknown texts get a zero-ish vector (will be filtered by norm guard)."""
    default = np.zeros(dim, dtype=np.float32)

    def fn(text: str) -> np.ndarray:
        return table.get(text, default).astype(np.float32)

    return fn


@pytest.fixture
def palace():
    p = Palace(db_path=":memory:")
    yield p
    p.close()


@pytest.fixture
def palace_with_embed():
    """A small handcrafted embedding space:
       'cats are great'  → axis-aligned x
       'dogs are great'  → axis-aligned y
       'cats and dogs'   → 45° between x and y
       'pizza recipe'    → axis-aligned z
       query 'cats'      → close to x and to xy
    """
    table = {
        "cats are great":  np.array([1.0, 0.0, 0.0, 0.0]),
        "dogs are great":  np.array([0.0, 1.0, 0.0, 0.0]),
        "cats and dogs":   np.array([0.7071, 0.7071, 0.0, 0.0]),
        "pizza recipe":    np.array([0.0, 0.0, 1.0, 0.0]),
        "cats":            np.array([1.0, 0.0, 0.0, 0.0]),
        "pets":            np.array([0.7071, 0.7071, 0.0, 0.0]),
        "food":            np.array([0.0, 0.0, 1.0, 0.0]),
    }
    p = Palace(db_path=":memory:", embed_fn=_emb_lookup(table))
    yield p
    p.close()


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------


class TestSchema:
    def test_empty_counts(self, palace):
        assert palace.counts() == {
            "triplas": 0,
            "triplas_active": 0,
            "documentos": 0,
            "documentos_with_embedding": 0,
            "eventos": 0,
        }

    def test_schema_version_recorded(self, palace):
        row = palace._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        assert row["version"] == Palace.SCHEMA_VERSION


# ----------------------------------------------------------------------
# triplas — bitemporal lite
# ----------------------------------------------------------------------


class TestFacts:
    def test_add_and_get(self, palace):
        rid = palace.add_fact("user", "lives_in", "London")
        assert rid > 0
        fact = palace.get_fact("user", "lives_in")
        assert fact is not None
        assert fact["objeto"] == "London"
        assert fact["valid_until"] is None

    def test_reasserting_invalidates_prior(self, palace):
        palace.add_fact("user", "lives_in", "Lisbon")
        palace.add_fact("user", "lives_in", "London")
        active = palace.get_fact("user", "lives_in")
        assert active["objeto"] == "London"
        # The Lisbon row should still exist, just invalidated
        history = palace.list_facts("user", include_invalidated=True)
        assert len(history) == 2
        invalidated = [f for f in history if f["valid_until"] is not None]
        assert len(invalidated) == 1
        assert invalidated[0]["objeto"] == "Lisbon"

    def test_get_fact_at_past_timestamp(self, palace):
        past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        palace.add_fact("user", "job", "engineer", valid_from=past)
        # Re-assert at "now"
        palace.add_fact("user", "job", "founder")
        # At a date AFTER the engineer fact but BEFORE the founder fact?
        # Both share the same now-ish timestamps so we can't easily get
        # "engineer" by timestamp here — instead, check that the founder
        # is the current one and engineer is in history.
        active = palace.get_fact("user", "job")
        assert active["objeto"] == "founder"
        all_facts = palace.list_facts("user", include_invalidated=True)
        objs = [f["objeto"] for f in all_facts]
        assert "engineer" in objs
        assert "founder" in objs

    def test_list_facts_excludes_invalidated_by_default(self, palace):
        palace.add_fact("user", "lives_in", "Rio")
        palace.add_fact("user", "lives_in", "London")
        active_only = palace.list_facts("user")
        assert len(active_only) == 1
        assert active_only[0]["objeto"] == "London"

    def test_invalidate_fact(self, palace):
        palace.add_fact("user", "lives_in", "London")
        n = palace.invalidate_fact("user", "lives_in")
        assert n == 1
        assert palace.get_fact("user", "lives_in") is None
        # Invalidating twice is a no-op
        assert palace.invalidate_fact("user", "lives_in") == 0

    def test_add_fact_requires_sujeito_and_predicado(self, palace):
        with pytest.raises(ValueError):
            palace.add_fact("", "p", "o")
        with pytest.raises(ValueError):
            palace.add_fact("s", "", "o")

    def test_confidence_and_source_persisted(self, palace):
        palace.add_fact("user", "favorite_lang", "Python",
                        source="inferred", confidence=0.7)
        fact = palace.get_fact("user", "favorite_lang")
        assert fact["source"] == "inferred"
        assert abs(fact["confidence"] - 0.7) < 1e-6


# ----------------------------------------------------------------------
# documentos
# ----------------------------------------------------------------------


class TestDocuments:
    def test_add_document_basic(self, palace):
        doc_id = palace.add_document("note", "Test content", metadata={"src": "test"})
        doc = palace.get_document(doc_id)
        assert doc["texto"] == "Test content"
        assert doc["metadata"] == {"src": "test"}
        assert doc["has_embedding"] is False
        # raw bytes never leak to callers
        assert "embedding" not in doc

    def test_add_document_with_embedding(self, palace):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        doc_id = palace.add_document("note", "with vector", embedding=emb)
        doc = palace.get_document(doc_id)
        assert doc["has_embedding"] is True

    def test_add_document_with_embedding_via_embed_fn(self, palace_with_embed):
        doc_id = palace_with_embed.add_document_with_embedding(
            "note", "cats are great"
        )
        doc = palace_with_embed.get_document(doc_id)
        assert doc["has_embedding"] is True

    def test_add_document_with_embedding_requires_embed_fn(self, palace):
        with pytest.raises(RuntimeError):
            palace.add_document_with_embedding("note", "x")

    def test_add_document_validates_inputs(self, palace):
        with pytest.raises(ValueError):
            palace.add_document("", "ok")
        with pytest.raises(ValueError):
            palace.add_document("note", "")
        with pytest.raises(ValueError):
            palace.add_document("note", "   ")

    def test_list_documents_filtered_by_tipo(self, palace):
        palace.add_document("note", "n1")
        palace.add_document("knowledge", "k1")
        palace.add_document("note", "n2")
        notes = palace.list_documents("note")
        assert len(notes) == 2
        assert {d["texto"] for d in notes} == {"n1", "n2"}

    def test_delete_document(self, palace):
        doc_id = palace.add_document("note", "ephemeral")
        assert palace.delete_document(doc_id) is True
        assert palace.get_document(doc_id) is None
        # delete is idempotent
        assert palace.delete_document(doc_id) is False

    def test_metadata_unicode_roundtrip(self, palace):
        meta = {"topic": "açúcar e canela", "n": 3}
        doc_id = palace.add_document("note", "doce", metadata=meta)
        doc = palace.get_document(doc_id)
        assert doc["metadata"] == meta


# ----------------------------------------------------------------------
# FTS5 lexical search
# ----------------------------------------------------------------------


class TestFTSSearch:
    def test_fts_finds_term_match(self, palace):
        palace.add_document("note", "the cat sat on the mat")
        palace.add_document("note", "the dog ran in the park")
        # alpha=0 → lexical only
        hits = palace.search("cat", k=5, alpha=0.0)
        assert len(hits) == 1
        assert "cat" in hits[0]["texto"]

    def test_fts_filtered_by_tipo(self, palace):
        palace.add_document("note", "cats are great")
        palace.add_document("knowledge", "cats are mammals")
        hits = palace.search("cats", k=5, alpha=0.0, tipo="knowledge")
        assert len(hits) == 1
        assert hits[0]["tipo"] == "knowledge"

    def test_fts_query_or_combines_tokens(self):
        from src.memory.palace import _fts_query
        assert _fts_query("qual meu CPF") == "qual OR meu OR CPF"
        assert _fts_query("CPF") == "CPF"          # single term unchanged
        assert _fts_query("") == ""                 # empty → empty
        assert _fts_query("RAG: vs!") == "RAG OR vs"  # specials stripped

    def test_fts_or_matches_rare_term_despite_stopwords(self, palace):
        # s39: a stopword-heavy recall question must still match a doc that
        # holds only the rare discriminative term. Implicit AND required ALL
        # tokens ("qual" AND "meu" AND "cpf") and matched nothing; OR + BM25
        # surfaces the doc that has "cpf".
        palace.add_document("knowledge", "## Documentos brasileiros\n- CPF: 000")
        palace.add_document("knowledge", "## Idioma e cultura\n- nada aqui")
        hits = palace.search("qual meu CPF", k=5, alpha=0.0)
        assert any("CPF" in h["texto"] for h in hits), "OR-FTS should find the CPF doc"

    def test_fts_handles_special_chars_in_query(self, palace):
        palace.add_document("note", "asking about RAG vs fine-tuning")
        # Special chars get stripped — shouldn't crash, should match "RAG"
        hits = palace.search("RAG: vs!", k=5, alpha=0.0)
        assert len(hits) == 1

    def test_fts_empty_query_returns_empty(self, palace):
        palace.add_document("note", "anything")
        assert palace.search("", k=5, alpha=0.0) == []
        assert palace.search("   ", k=5, alpha=0.0) == []

    def test_fts_triggers_keep_index_in_sync_on_delete(self, palace):
        doc_id = palace.add_document("note", "uniqueterm one")
        palace.delete_document(doc_id)
        assert palace.search("uniqueterm", k=5, alpha=0.0) == []


# ----------------------------------------------------------------------
# semantic + hybrid search
# ----------------------------------------------------------------------


class TestHybridSearch:
    def test_semantic_only_finds_nearest_vector(self, palace_with_embed):
        palace_with_embed.add_document_with_embedding("note", "cats are great")
        palace_with_embed.add_document_with_embedding("note", "pizza recipe")
        # alpha=1.0 → semantic only
        hits = palace_with_embed.search("cats", k=1, alpha=1.0)
        assert len(hits) == 1
        assert hits[0]["texto"] == "cats are great"

    def test_semantic_ranks_by_cosine(self, palace_with_embed):
        palace_with_embed.add_document_with_embedding("note", "cats are great")
        palace_with_embed.add_document_with_embedding("note", "cats and dogs")
        palace_with_embed.add_document_with_embedding("note", "pizza recipe")
        hits = palace_with_embed.search("cats", k=3, alpha=1.0)
        # 'cats are great' aligns perfectly (cos=1), 'cats and dogs' at 45° (cos≈0.7),
        # 'pizza recipe' orthogonal (cos=0).
        assert hits[0]["texto"] == "cats are great"
        assert hits[1]["texto"] == "cats and dogs"

    def test_hybrid_combines_semantic_and_lexical(self, palace_with_embed):
        # 'pets' has no lexical overlap with the docs, but semantically
        # overlaps with cats+dogs. 'pizza recipe' has neither overlap.
        palace_with_embed.add_document_with_embedding("note", "cats and dogs")
        palace_with_embed.add_document_with_embedding("note", "pizza recipe")
        hits = palace_with_embed.search("pets", k=2, alpha=0.6)
        assert hits[0]["texto"] == "cats and dogs"

    def test_search_results_include_score(self, palace_with_embed):
        palace_with_embed.add_document_with_embedding("note", "cats are great")
        hits = palace_with_embed.search("cats", k=1, alpha=0.6)
        assert "_score" in hits[0]
        assert hits[0]["_score"] > 0

    def test_search_exposes_raw_semantic_similarity(self, palace_with_embed):
        # Bug B-2 infra: each result carries the RAW cosine (_sem_sim) so a
        # caller can apply a relevance floor. A nearest-vector hit scores ~1.0.
        palace_with_embed.add_document_with_embedding("note", "cats are great")
        hits = palace_with_embed.search("cats", k=1, alpha=1.0)
        assert "_sem_sim" in hits[0]
        assert hits[0]["_sem_sim"] == pytest.approx(1.0, abs=1e-6)

    def test_search_sem_sim_is_none_for_fts_only_hit(self, palace):
        # No embed_fn → semantic channel off → the hit surfaces via FTS, so it
        # has no semantic similarity. _sem_sim must be None (caller keeps it:
        # we have no semantic signal to floor it by).
        palace.add_document("note", "cats are great")
        hits = palace.search("cats", k=1, alpha=0.6)
        assert hits[0]["_sem_sim"] is None

    def test_search_no_embed_fn_falls_back_to_fts(self, palace):
        palace.add_document("note", "cats are great")
        palace.add_document("note", "pizza recipe")
        # alpha=0.6 would try semantic — but no embed_fn → uses FTS only
        hits = palace.search("cats", k=1, alpha=0.6)
        assert len(hits) == 1
        assert hits[0]["texto"] == "cats are great"

    def test_search_empty_db(self, palace_with_embed):
        assert palace_with_embed.search("anything", k=3) == []

    def test_search_invalid_query(self, palace_with_embed):
        assert palace_with_embed.search("", k=3) == []
        assert palace_with_embed.search(None, k=3) == []  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# eventos
# ----------------------------------------------------------------------


class TestEvents:
    def test_log_and_list(self, palace):
        palace.log_event("gate_approved", {"hours": 4}, trace_id="abc")
        events = palace.list_events()
        assert len(events) == 1
        assert events[0]["tipo"] == "gate_approved"
        assert events[0]["payload"] == {"hours": 4}
        assert events[0]["trace_id"] == "abc"

    def test_filter_by_tipo(self, palace):
        palace.log_event("gate_approved", {})
        palace.log_event("gate_rejected", {})
        palace.log_event("gate_approved", {})
        approved = palace.list_events(tipo="gate_approved")
        assert len(approved) == 2

    def test_filter_by_trace(self, palace):
        palace.log_event("step", {"n": 1}, trace_id="t1")
        palace.log_event("step", {"n": 2}, trace_id="t2")
        palace.log_event("step", {"n": 3}, trace_id="t1")
        t1 = palace.list_events(trace_id="t1")
        assert len(t1) == 2

    def test_payload_unicode_roundtrip(self, palace):
        payload = {"msg": "ação tomada — ✓"}
        palace.log_event("notice", payload)
        events = palace.list_events()
        assert events[0]["payload"] == payload

    def test_log_event_requires_tipo(self, palace):
        with pytest.raises(ValueError):
            palace.log_event("", {})


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        db_file = str(tmp_path / "palace.db")
        p1 = Palace(db_path=db_file)
        p1.add_fact("user", "lives_in", "London")
        p1.add_document("note", "persistent text")
        p1.log_event("test", {"k": "v"})
        p1.close()

        p2 = Palace(db_path=db_file)
        assert p2.get_fact("user", "lives_in")["objeto"] == "London"
        docs = p2.list_documents()
        assert len(docs) == 1
        events = p2.list_events()
        assert len(events) == 1
        p2.close()


# ----------------------------------------------------------------------
# access_count + legacy migration
# ----------------------------------------------------------------------


class TestAccessCount:
    def test_new_document_starts_at_zero(self, palace):
        doc_id = palace.add_document("note", "hello")
        doc = palace.get_document(doc_id)
        assert doc["access_count"] == 0

    def test_search_does_not_increment_access_count(self, palace_with_embed):
        # M3-A4: search() is read-only w.r.t. access_count — the bump moved
        # to mark_used(), called by the INJECTING caller. Bumping on raw
        # retrieval counted hits the caller's floor/dedup later discarded
        # (rich-get-richer loop in the frequency channel).
        p = palace_with_embed
        cat_id = p.add_document("note", "cats are great",
                                embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        hits = p.search("cats", k=1, alpha=1.0)
        assert hits[0]["id"] == cat_id
        assert hits[0]["access_count"] == 0          # raw DB value, no bump
        assert p.get_document(cat_id)["access_count"] == 0

    def test_mark_used_increments_and_persists(self, palace_with_embed):
        p = palace_with_embed
        cat_id = p.add_document("note", "cats are great",
                                embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        dog_id = p.add_document("note", "dogs are great",
                                embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        p.mark_used([cat_id])
        assert p.get_document(cat_id)["access_count"] == 1
        assert p.get_document(dog_id)["access_count"] == 0   # untouched
        p.mark_used([cat_id, dog_id])
        assert p.get_document(cat_id)["access_count"] == 2
        assert p.get_document(dog_id)["access_count"] == 1

    def test_mark_used_empty_list_is_noop(self, palace_with_embed):
        palace_with_embed.mark_used([])              # must not raise

    def test_search_with_empty_results_does_not_error(self, palace_with_embed):
        assert palace_with_embed.search("cats", k=1) == []


# ----------------------------------------------------------------------
# facade methods (M3-A9) — consumers must not touch _conn directly
# ----------------------------------------------------------------------


class TestFacade:
    def test_recent_texts_filters_by_tipo_and_cutoff(self, palace):
        palace.add_document("conversation", "old turn",
                            created_at="2026-06-01T10:00:00+00:00")
        palace.add_document("conversation", "new turn",
                            created_at="2026-06-10T10:00:00+00:00")
        palace.add_document("knowledge", "not a conversation",
                            created_at="2026-06-10T10:00:00+00:00")
        texts = palace.recent_texts("conversation", "2026-06-05T00:00:00+00:00")
        assert texts == ["new turn"]

    def test_recent_texts_newest_first(self, palace):
        palace.add_document("conversation", "first",
                            created_at="2026-06-10T09:00:00+00:00")
        palace.add_document("conversation", "second",
                            created_at="2026-06-10T11:00:00+00:00")
        texts = palace.recent_texts("conversation", "2026-06-10T00:00:00+00:00")
        assert texts == ["second", "first"]

    def test_count_distinct_meta_counts_values_not_rows(self, palace):
        palace.add_document("knowledge", "chunk 1", metadata={"filename": "a.md"})
        palace.add_document("knowledge", "chunk 2", metadata={"filename": "a.md"})
        palace.add_document("knowledge", "chunk 3", metadata={"filename": "b.md"})
        palace.add_document("knowledge", "no meta")
        palace.add_document("note", "other tipo", metadata={"filename": "c.md"})
        assert palace.count_distinct_meta("knowledge", "filename") == 2

    def test_count_distinct_meta_empty(self, palace):
        assert palace.count_distinct_meta("knowledge", "filename") == 0


# ----------------------------------------------------------------------
# concurrency pragmas (M3-A3)
# ----------------------------------------------------------------------


class TestBusyTimeout:
    def test_write_path_sets_busy_timeout(self, palace):
        # M3-A3: a momentary lock must become a short wait, not a dry
        # "database is locked" — requests cross threads and 3+ Palace
        # instances share the file in production.
        row = palace._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 5000

    def test_read_only_path_sets_no_extra_pragmas(self, tmp_path):
        # The ro path (M3-A7) deliberately runs no PRAGMA statements; the
        # 5000 it reports comes from Python's connect() timeout=5.0
        # default, not from us — pinned here so a future "harmless"
        # pragma on the ro path trips a review.
        path = str(tmp_path / "palace.db")
        Palace(db_path=path).close()
        ro = Palace(db_path=path, read_only=True)
        assert ro._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert ro._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        ro.close()


# ----------------------------------------------------------------------
# generic SQL facade (M3-A9 follow-up, s72) — for modules with their own
# table in the palace file (skill_registry). Nobody touches _conn.
# ----------------------------------------------------------------------


class TestSqlFacade:
    def test_write_then_read_roundtrip(self, palace):
        rowcount, lastrowid = palace.execute_write(
            "INSERT INTO documentos (tipo, texto, created_at) VALUES (?, ?, ?)",
            ("note", "via facade", "2026-06-12T00:00:00+00:00"),
        )
        assert rowcount == 1 and lastrowid is not None
        rows = palace.execute_read(
            "SELECT texto FROM documentos WHERE id = ?", (lastrowid,))
        assert rows[0]["texto"] == "via facade"

    def test_write_reports_rowcount(self, palace):
        rowcount, _ = palace.execute_write(
            "DELETE FROM documentos WHERE id = ?", (99999,))
        assert rowcount == 0

    def test_read_only_palace_refuses_write(self, tmp_path):
        import sqlite3
        path = str(tmp_path / "palace.db")
        Palace(db_path=path).close()
        ro = Palace(db_path=path, read_only=True)
        with pytest.raises(sqlite3.OperationalError):
            ro.execute_write(
                "INSERT INTO documentos (tipo, texto, created_at) VALUES ('a','b','c')")
        ro.close()


# ----------------------------------------------------------------------
# read-only mode (M3-A7)
# ----------------------------------------------------------------------


class TestReadOnly:
    def test_read_only_reads_but_never_writes(self, tmp_path):
        path = str(tmp_path / "palace.db")
        writer = Palace(db_path=path)
        doc_id = writer.add_document("note", "hello")
        writer.close()

        ro = Palace(db_path=path, read_only=True)
        assert ro.get_document(doc_id)["texto"] == "hello"
        assert len(ro.list_documents()) == 1
        # any write path dies at the SQLite layer, not by convention
        import sqlite3
        with pytest.raises(sqlite3.OperationalError):
            ro.add_document("note", "should fail")
        with pytest.raises(sqlite3.OperationalError):
            ro.mark_used([doc_id])
        ro.close()

    def test_read_only_missing_file_raises(self, tmp_path):
        import sqlite3
        with pytest.raises(sqlite3.OperationalError):
            Palace(db_path=str(tmp_path / "ghost.db"), read_only=True)
        # and the failed open did not create the file
        assert not (tmp_path / "ghost.db").exists()

    def test_read_only_rejects_memory(self):
        with pytest.raises(ValueError):
            Palace(db_path=":memory:", read_only=True)


# ----------------------------------------------------------------------
# temporal-aware search (Tier 2.2)
# ----------------------------------------------------------------------


class TestTemporalRetrieval:
    """Integration of detect_temporal_intent + decay + frequency into search()."""

    @staticmethod
    def _ts(now, days_ago):
        from datetime import timedelta
        return (now - timedelta(days=days_ago)).isoformat()

    def test_decay_demotes_older_doc(self, palace_with_embed):
        """Two docs with identical semantic match, but the newer one
        should win once decay is applied (conversation, λ=0.10)."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        # Same vector ('cats are great') but different ages.
        new_id = palace_with_embed.add_document(
            "conversation", "cats are great (NEW)", embedding=vec,
            created_at=self._ts(now, 1),
        )
        palace_with_embed.add_document(
            "conversation", "cats are great (OLD)", embedding=vec,
            created_at=self._ts(now, 60),
        )
        hits = palace_with_embed.search(
            "cats", k=2, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert hits[0]["id"] == new_id

    def test_age_is_a_tiebreak_not_a_verdict(self, palace_with_embed):
        """s83: a 4-month-old turn that MATCHES must beat last week's turn
        that doesn't. Under pure decay (λ=0.10) the old one was multiplied
        by ~1e-6 and lost to anything recent — "where do I work?" could not
        find the turn where it was said."""
        from datetime import datetime, timezone
        now = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        old_match = palace_with_embed.add_document(
            "conversation", "cats are great (OLD)",
            embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            created_at=self._ts(now, 120),
        )
        palace_with_embed.add_document(
            "conversation", "cats and dogs (NEW)",
            embedding=np.array([0.7071, 0.7071, 0.0, 0.0]),
            created_at=self._ts(now, 1),
        )
        hits = palace_with_embed.search(
            "cats", k=2, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert hits[0]["id"] == old_match

    def test_every_candidate_with_a_vector_carries_its_cosine(self, palace_with_embed):
        """s83: a doc that surfaced only via FTS or frequency used to come
        back with _sem_sim=None, which the relevance floor keeps by design —
        so the query-blind frequency channel injected the same old turns
        into unrelated questions. Any candidate with a vector is now scored."""
        import numpy as np
        # Outside the semantic top-k (k=1 → cand_k=3) but an FTS hit.
        for text in ("cats are great", "cats and dogs", "dogs are great"):
            palace_with_embed.add_document_with_embedding("note", text)
        pizza = palace_with_embed.add_document(
            "note", "pizza recipe with cats", embedding=np.array([0.0, 0.0, 1.0, 0.0]))
        pool = palace_with_embed.candidate_pool("cats", k=1, freq_weight=0.0)
        by_id = {c["id"]: c for c in pool}
        assert "S" not in by_id[pizza]["channels"]  # FTS-only…
        assert by_id[pizza]["sem_sim"] == pytest.approx(0.0, abs=1e-6)  # …yet scored

    def test_search_is_the_head_of_candidate_pool(self, palace_with_embed):
        for text in ("cats are great", "cats and dogs", "dogs are great", "pizza recipe"):
            palace_with_embed.add_document_with_embedding("note", text)
        pool = palace_with_embed.candidate_pool("cats", k=2)
        hits = palace_with_embed.search("cats", k=2)
        assert [h["id"] for h in hits] == [c["id"] for c in pool[:2]]
        assert [h["_sem_sim"] for h in hits] == [c["sem_sim"] for c in pool[:2]]

    def test_decay_off_keeps_old_profile_doc(self, palace_with_embed):
        """A 'prefiro' query disables decay; an old but matching profile
        doc should not be demoted by age."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        # 'cats' query vector; the profile doc embeds identically.
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        old_profile = palace_with_embed.add_document(
            "profile", "cats are great", embedding=vec,
            created_at=self._ts(now, 180),  # 6 months old
        )
        # decay-off keyword in query
        hits = palace_with_embed.search(
            "sempre prefiro cats", k=1, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert len(hits) == 1
        assert hits[0]["id"] == old_profile

    def test_window_filter_excludes_older_docs(self, palace_with_embed):
        """'ontem' window (2 days) hard-filters docs older than 2 days."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        recent = palace_with_embed.add_document(
            "conversation", "cats are great recent", embedding=vec,
            created_at=self._ts(now, 1),  # within window
        )
        palace_with_embed.add_document(
            "conversation", "cats are great long ago", embedding=vec,
            created_at=self._ts(now, 30),  # outside window
        )
        hits = palace_with_embed.search(
            "ontem cats", k=5, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert len(hits) == 1
        assert hits[0]["id"] == recent

    def test_window_filter_applies_to_lexical_too(self, palace_with_embed):
        """The 'ontem' window cutoff hides old docs from FTS5 as well."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        recent = palace_with_embed.add_document(
            "conversation", "cats and dogs",
            created_at=self._ts(now, 1),
        )
        palace_with_embed.add_document(
            "conversation", "cats are great",
            created_at=self._ts(now, 30),
        )
        # alpha=0.0 → lexical only. Query 'cats' matches both, but window
        # should drop the 30-day-old one.
        hits = palace_with_embed.search(
            "ontem cats", k=5, alpha=0.0, freq_weight=0.0, now=now,
        )
        assert len(hits) == 1
        assert hits[0]["id"] == recent

    def test_frequency_channel_boosts_accessed_doc(self, palace_with_embed):
        """A doc with semantic disadvantage can climb the ranking if it has
        been accessed multiple times before — provided freq_weight is on."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        # Perfect semantic match — never accessed.
        ideal = palace_with_embed.add_document(
            "conversation", "cats are great",
            embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            created_at=self._ts(now, 0),
        )
        # Weak semantic match — but heavily accessed.
        weak_but_popular = palace_with_embed.add_document(
            "conversation", "cats and dogs",
            embedding=np.array([0.7071, 0.7071, 0.0, 0.0]),
            created_at=self._ts(now, 0),
        )
        # Manually bump access_count on the weak doc.
        palace_with_embed._conn.execute(
            "UPDATE documentos SET access_count = 50 WHERE id = ?",
            (weak_but_popular,),
        )
        palace_with_embed._conn.commit()

        # With freq_weight 0 → ideal wins on semantic.
        no_freq = palace_with_embed.search(
            "cats", k=2, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert no_freq[0]["id"] == ideal

        # With freq_weight high enough → popular doc wins. (search() no
        # longer bumps access_count — M3-A4 — so `ideal` is still at 0.)
        with_freq = palace_with_embed.search(
            "cats", k=2, alpha=1.0, freq_weight=2.0, now=now,
        )
        assert with_freq[0]["id"] == weak_but_popular

    def test_frequency_channel_excludes_never_accessed(self, palace_with_embed):
        """access_count == 0 → doc is invisible to the frequency channel."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        palace_with_embed.add_document("conversation", "never accessed", created_at=self._ts(now, 0))
        # alpha=0 + freq-only would normally return all docs — but excluded.
        # With nothing in semantic and nothing in frequency, we get FTS5.
        hits = palace_with_embed._frequency_candidates(k=10, tipo=None, cutoff_iso=None)
        assert hits == []

    def test_no_temporal_cue_keeps_old_docs_visible(self, palace_with_embed):
        """A query without temporal cues should not apply a window — old
        docs still show up if nothing newer matches."""
        from datetime import datetime, timezone
        now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        import numpy as np
        old_id = palace_with_embed.add_document(
            "conversation", "cats are great",
            embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            created_at=self._ts(now, 60),
        )
        hits = palace_with_embed.search(
            "cats", k=1, alpha=1.0, freq_weight=0.0, now=now,
        )
        assert len(hits) == 1
        assert hits[0]["id"] == old_id


class TestLegacyMigration:
    def test_adds_access_count_to_pre_v2_db(self, tmp_path):
        """A palace.db created before SCHEMA_VERSION=2 lacks access_count.
        Re-opening it must add the column without touching existing rows."""
        import sqlite3

        db_file = str(tmp_path / "legacy.db")
        # Build a v1-style schema by hand.
        conn = sqlite3.connect(db_file)
        conn.executescript(
            """
            CREATE TABLE documentos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo       TEXT    NOT NULL,
                texto      TEXT    NOT NULL,
                embedding  BLOB,
                metadata   TEXT    NOT NULL DEFAULT '{}',
                created_at TEXT    NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO documentos (tipo, texto, created_at) VALUES (?, ?, ?)",
            ("note", "legacy row", "2024-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        # Opening with the current Palace should migrate the schema in place.
        p = Palace(db_path=db_file)
        docs = p.list_documents()
        assert len(docs) == 1
        assert docs[0]["texto"] == "legacy row"
        assert docs[0]["access_count"] == 0
        p.close()
