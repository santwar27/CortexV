"""
Tests for the nightly memory consolidator (Tier 3.5).

Same layered approach used in test_preferences.py:
  - Pure-logic functions with a stub `llm_chat_fn` so no Ollama is
    needed for the LLM round-trips.
  - Palace fixture is in-memory.
  - Studier integration test mocks ollama and points the studier at a
    tmp-path Palace.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.memory.consolidator import (
    archive_stale_documents,
    FactTriple,
    _coerce_fact_triple,
    extract_facts,
    persist_daily_summary,
    persist_facts,
    summarise_day,
)
from src.memory.palace import Palace


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def palace():
    p = Palace(db_path=":memory:")
    yield p
    p.close()


def _llm_returning(content: str):
    def fn(**kwargs):
        return {"message": {"content": content}}
    return fn


# ----------------------------------------------------------------------
# summarise_day
# ----------------------------------------------------------------------


class TestSummariseDay:
    def test_empty_input_returns_none(self):
        sentinel = MagicMock(side_effect=AssertionError("must not call LLM"))
        assert summarise_day("", sentinel, "any") is None
        assert summarise_day("   \t\n", sentinel, "any") is None
        assert summarise_day(None, sentinel, "any") is None  # type: ignore[arg-type]

    def test_strips_response(self):
        llm = _llm_returning("  resumo do dia  \n")
        assert summarise_day("alguma conversa", llm, "m") == "resumo do dia"

    def test_empty_model_response_returns_none(self):
        llm = _llm_returning("   ")
        assert summarise_day("alguma conversa", llm, "m") is None

    def test_llm_exception_returns_none(self):
        def boom(**kw):
            raise RuntimeError("ollama down")
        assert summarise_day("alguma conversa", boom, "m") is None

    def test_passes_temperature_zero(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": "ok"}}
        summarise_day("conversa", fn, "my-model")
        assert captured.get("options", {}).get("temperature") == 0.0
        assert captured.get("model") == "my-model"

    def test_input_truncated_to_max(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": "ok"}}
        huge = "x" * 20000
        summarise_day(huge, fn, "m", max_input_chars=5000)
        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        assert len(user_msg["content"]) <= 5000


# ----------------------------------------------------------------------
# _coerce_fact_triple — write-time sanitisation
# ----------------------------------------------------------------------


class TestCoerceFactTriple:
    def test_happy_path(self):
        t = _coerce_fact_triple({
            "sujeito": "user",
            "predicado": "vive_em",
            "objeto": "Londres",
            "confidence": 0.95,
            "evidence": "vive em Londres",
        })
        assert t is not None
        assert t.sujeito == "user"
        assert t.predicado == "vive_em"
        assert t.objeto == "Londres"

    def test_uppercase_predicate_normalised(self):
        t = _coerce_fact_triple({
            "sujeito": "USER",
            "predicado": "VIVE_EM",
            "objeto": "Londres",
            "confidence": 0.9,
        })
        assert t is not None
        assert t.sujeito == "user"
        assert t.predicado == "vive_em"

    @pytest.mark.parametrize("bad_predicado", [
        "vive em",      # space
        "vive-em",      # hyphen
        "viver_é",      # accent
        "",
    ])
    def test_invalid_predicado_rejected(self, bad_predicado):
        assert _coerce_fact_triple({
            "sujeito": "user",
            "predicado": bad_predicado,
            "objeto": "x",
            "confidence": 0.9,
        }) is None

    def test_invalid_sujeito_rejected(self):
        assert _coerce_fact_triple({
            "sujeito": "outra pessoa",  # space
            "predicado": "vive_em",
            "objeto": "x",
            "confidence": 0.9,
        }) is None

    def test_objeto_newline_rejected(self):
        assert _coerce_fact_triple({
            "sujeito": "user",
            "predicado": "vive_em",
            "objeto": "Londres\nattack",
            "confidence": 0.9,
        }) is None

    def test_objeto_control_char_rejected(self):
        assert _coerce_fact_triple({
            "sujeito": "user",
            "predicado": "vive_em",
            "objeto": "Londres\x00null",
            "confidence": 0.9,
        }) is None

    def test_objeto_authoritative_tag_rejected(self):
        for poison in ("[KNOWLEDGE] Londres", "foo [SYSTEM] bar", "[INSTRUCTION]"):
            assert _coerce_fact_triple({
                "sujeito": "user",
                "predicado": "vive_em",
                "objeto": poison,
                "confidence": 0.9,
            }) is None

    def test_objeto_too_long_rejected(self):
        assert _coerce_fact_triple({
            "sujeito": "user",
            "predicado": "vive_em",
            "objeto": "x" * 201,
            "confidence": 0.9,
        }) is None

    def test_confidence_clamped(self):
        t = _coerce_fact_triple({"sujeito": "user", "predicado": "vive_em",
                                  "objeto": "y", "confidence": 5.0})
        assert t is not None and t.confidence == 1.0
        t = _coerce_fact_triple({"sujeito": "user", "predicado": "vive_em",
                                  "objeto": "y", "confidence": -0.3})
        assert t is not None and t.confidence == 0.0

    def test_non_numeric_confidence_rejected(self):
        assert _coerce_fact_triple({"sujeito": "user", "predicado": "vive_em",
                                     "objeto": "y", "confidence": "alta"}) is None

    def test_non_dict_rejected(self):
        assert _coerce_fact_triple("nope") is None
        assert _coerce_fact_triple(None) is None
        assert _coerce_fact_triple([]) is None


# ----------------------------------------------------------------------
# extract_facts — LLM-shape contract
# ----------------------------------------------------------------------


class TestExtractFacts:
    def test_empty_input_short_circuits(self):
        sentinel = MagicMock(side_effect=AssertionError("must not call"))
        assert extract_facts("", sentinel, "m") == []

    def test_happy_path_clean_json(self):
        llm = _llm_returning(
            '[{"sujeito":"user","predicado":"vive_em","objeto":"Londres",'
            '"confidence":0.95,"evidence":"vive em Londres"}]'
        )
        out = extract_facts("digest about user in London", llm, "m")
        assert len(out) == 1
        assert out[0].objeto == "Londres"

    def test_multi_fact_output(self):
        llm = _llm_returning(
            '['
            '{"sujeito":"user","predicado":"vive_em","objeto":"Londres","confidence":0.95},'
            '{"sujeito":"user","predicado":"trabalha_como","objeto":"engenheiro","confidence":0.9}'
            ']'
        )
        out = extract_facts("digest", llm, "m")
        assert len(out) == 2

    def test_json_inside_markdown_fence(self):
        llm = _llm_returning(
            "```json\n"
            '[{"sujeito":"user","predicado":"vive_em","objeto":"Londres","confidence":0.95}]\n'
            "```"
        )
        assert len(extract_facts("d", llm, "m")) == 1

    def test_no_json_returns_empty(self):
        llm = _llm_returning("não há fatos extraíveis aqui.")
        assert extract_facts("d", llm, "m") == []

    def test_malformed_json_returns_empty(self):
        llm = _llm_returning('[{"sujeito":"user", BROKEN}]')
        assert extract_facts("d", llm, "m") == []

    def test_empty_array(self):
        assert extract_facts("d", _llm_returning("[]"), "m") == []

    def test_partial_bad_rows_dropped(self):
        llm = _llm_returning(
            '['
            '{"sujeito":"user","predicado":"vive_em","objeto":"Londres","confidence":0.9},'
            '{"sujeito":"user","predicado":"NOT VALID","objeto":"x","confidence":0.9},'
            '{"sujeito":"user","predicado":"usa","objeto":"Python","confidence":0.85}'
            ']'
        )
        out = extract_facts("d", llm, "m")
        preds = {t.predicado for t in out}
        assert preds == {"vive_em", "usa"}

    def test_llm_exception_returns_empty(self):
        def boom(**kw):
            raise RuntimeError("ollama down")
        assert extract_facts("d", boom, "m") == []

    def test_input_truncated(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": "[]"}}
        extract_facts("x" * 20000, fn, "m", max_input_chars=5000)
        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        assert len(user_msg["content"]) <= 5000


# ----------------------------------------------------------------------
# persist_daily_summary
# ----------------------------------------------------------------------


class TestPersistDailySummary:
    def test_writes_document_with_correct_tipo(self, palace):
        doc_id = persist_daily_summary(palace, "resumo do dia")
        assert doc_id is not None
        doc = palace.get_document(doc_id)
        assert doc["tipo"] == "daily_summary"
        assert "resumo do dia" in doc["texto"]

    def test_empty_summary_returns_none(self, palace):
        assert persist_daily_summary(palace, "") is None
        assert persist_daily_summary(palace, "   ") is None
        assert persist_daily_summary(palace, None) is None  # type: ignore[arg-type]

    def test_embed_failure_does_not_block_write(self, palace):
        # Attach an embed_fn that raises.
        palace.embed_fn = lambda txt: (_ for _ in ()).throw(RuntimeError("oops"))
        doc_id = persist_daily_summary(palace, "resumo do dia")
        # The document still landed; embedding is best-effort.
        assert doc_id is not None
        doc = palace.get_document(doc_id)
        # get_document hides raw bytes behind a has_embedding flag.
        assert doc["has_embedding"] is False

    def test_uses_embed_fn_when_available(self, palace):
        import numpy as np
        palace.embed_fn = lambda txt: np.array([1.0, 0.0, 0.0, 0.0])
        doc_id = persist_daily_summary(palace, "resumo do dia")
        doc = palace.get_document(doc_id)
        assert doc["has_embedding"] is True


# ----------------------------------------------------------------------
# persist_facts
# ----------------------------------------------------------------------


class TestPersistFacts:
    def test_writes_high_confidence(self, palace):
        triples = [
            FactTriple("user", "vive_em", "Londres", 0.9),
            FactTriple("user", "usa", "Python", 0.85),
        ]
        n = persist_facts(palace, triples)
        assert n == 2
        active = palace.list_facts(sujeito="user")
        assert {r["predicado"] for r in active} == {"vive_em", "usa"}

    def test_drops_low_confidence(self, palace):
        triples = [
            FactTriple("user", "vive_em", "Londres", 0.4),
            FactTriple("user", "usa", "Python", 0.9),
        ]
        n = persist_facts(palace, triples, min_confidence=0.55)
        assert n == 1
        assert palace.get_fact("user", "vive_em") is None
        assert palace.get_fact("user", "usa") is not None

    def test_idempotent_for_unchanged_value(self, palace):
        triples = [FactTriple("user", "vive_em", "Londres", 0.9)]
        assert persist_facts(palace, triples) == 1
        # Same value again → no churn.
        assert persist_facts(palace, triples) == 0
        all_rows = palace.list_facts(sujeito="user", include_invalidated=True)
        assert len(all_rows) == 1

    def test_replaces_value_via_bitemporal_closure(self, palace):
        persist_facts(palace, [FactTriple("user", "vive_em", "Londres", 0.9)])
        n = persist_facts(palace, [FactTriple("user", "vive_em", "São Paulo", 0.95)])
        assert n == 1
        active = palace.get_fact("user", "vive_em")
        assert active["objeto"] == "São Paulo"
        all_rows = palace.list_facts(sujeito="user", include_invalidated=True)
        assert len(all_rows) == 2

    def test_source_tag_persisted(self, palace):
        persist_facts(
            palace, [FactTriple("user", "vive_em", "Londres", 0.9)],
            source="studier_inferred",
        )
        row = palace.get_fact("user", "vive_em")
        assert row["source"] == "studier_inferred"

    def test_empty_list_no_op(self, palace):
        assert persist_facts(palace, []) == 0


# ----------------------------------------------------------------------
# Studier integration
# ----------------------------------------------------------------------


class TestStudierConsolidationIntegration:
    """_consolidate_memory should call ollama.chat twice (digest + facts),
    persist a daily_summary doc, and write facts as bitemporal triples."""

    def test_summary_and_facts_persisted(self, tmp_path, monkeypatch):
        from src.core.studier import NightlyStudier

        db_path = tmp_path / "palace.db"
        p = Palace(db_path=str(db_path))
        now_iso = datetime.now(timezone.utc).isoformat()
        # Both facts must be in the user's own words (s88): the digest alone is
        # not a source — it tells Cortex's side too.
        p.add_document("conversation",
                       "User: vivo em Londres e trabalho como engenheiro\nCortex: ok",
                       created_at=now_iso)
        p.close()

        digest_text = "O usuário vive em Londres e trabalha como engenheiro."
        facts_json = (
            '[{"sujeito":"user","predicado":"vive_em","objeto":"Londres",'
            '"confidence":0.95,"evidence":"vive em Londres"},'
            '{"sujeito":"user","predicado":"trabalha_como",'
            '"objeto":"engenheiro","confidence":0.9,"evidence":"trabalha como engenheiro"}]'
        )

        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", str(db_path))

        with patch("src.core.studier.ollama") as ollama_mock:
            # First call (summarise) returns digest; second call (extract) returns JSON
            ollama_mock.chat.side_effect = [
                {"message": {"content": digest_text}},
                {"message": {"content": facts_json}},
            ]
            result = studier._consolidate_memory()

        assert result["summary_id"] is not None
        assert result["facts_persisted"] == 2

        p2 = Palace(db_path=str(db_path))
        try:
            # Daily summary doc present
            row = p2._conn.execute(
                "SELECT tipo, texto FROM documentos WHERE tipo='daily_summary'"
            ).fetchone()
            assert row is not None
            assert digest_text in row["texto"]
            # Facts persisted with right source
            facts = p2.list_facts(sujeito="user")
            preds = {f["predicado"] for f in facts}
            assert preds == {"vive_em", "trabalha_como"}
            for f in facts:
                assert f["source"] == "studier_inferred"
        finally:
            p2.close()

    def test_no_palace_db_skips(self, tmp_path):
        from src.core.studier import NightlyStudier
        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        studier.PALACE_DB_PATH = str(tmp_path / "missing.db")
        result = studier._consolidate_memory()
        assert result == {"summary_id": None, "facts_persisted": 0}

    def test_summariser_crash_returns_zero(self, tmp_path, monkeypatch):
        """An LLM crash in the summary step must NOT propagate."""
        from src.core.studier import NightlyStudier

        db_path = tmp_path / "palace.db"
        p = Palace(db_path=str(db_path))
        p.add_document("conversation", "anything",
                       created_at=datetime.now(timezone.utc).isoformat())
        p.close()

        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", str(db_path))

        with patch("src.core.studier.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("boom")
            # Must not raise.
            result = studier._consolidate_memory()
            assert result["summary_id"] is None
            assert result["facts_persisted"] == 0

    def test_empty_corpus_skips(self, tmp_path, monkeypatch):
        """No conversation in the lookback window → skip the LLM round-trips."""
        from src.core.studier import NightlyStudier

        db_path = tmp_path / "palace.db"
        # Create an empty Palace (no documents).
        p = Palace(db_path=str(db_path))
        p.close()

        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", str(db_path))

        with patch("src.core.studier.ollama") as ollama_mock:
            result = studier._consolidate_memory()
            assert result == {"summary_id": None, "facts_persisted": 0}
            ollama_mock.chat.assert_not_called()


# ----------------------------------------------------------------------
# retention — archive_stale_documents (M3-A5, policy approved s72)
# ----------------------------------------------------------------------

import gzip
import json
import os


class TestRetention:
    _NOW = datetime(2026, 6, 12, 3, 0, tzinfo=timezone.utc)

    def _seed(self, palace):
        old = (self._NOW - timedelta(days=200)).isoformat()
        recent = (self._NOW - timedelta(days=10)).isoformat()
        stale_id = palace.add_document("conversation", "stale never used", created_at=old)
        used_id = palace.add_document("conversation", "old but injected", created_at=old)
        palace.mark_used([used_id])
        fresh_id = palace.add_document("conversation", "fresh", created_at=recent)
        knowledge_id = palace.add_document("knowledge", "old knowledge", created_at=old)
        return stale_id, used_id, fresh_id, knowledge_id

    def test_archives_only_old_and_never_used(self, palace, tmp_path):
        stale_id, used_id, fresh_id, knowledge_id = self._seed(palace)
        result = archive_stale_documents(
            palace, days=180, archive_dir=str(tmp_path), now=self._NOW)
        assert result["archived"] == 1
        # The stale row left SQLite; everything else is untouched.
        assert palace.get_document(stale_id) is None
        assert palace.get_document(used_id) is not None
        assert palace.get_document(fresh_id) is not None
        assert palace.get_document(knowledge_id) is not None

    def test_archive_file_holds_the_full_row_0600(self, palace, tmp_path):
        self._seed(palace)
        result = archive_stale_documents(
            palace, days=180, archive_dir=str(tmp_path), now=self._NOW)
        path = result["path"]
        assert path and path.endswith(".jsonl.gz")
        assert os.stat(path).st_mode & 0o777 == 0o600
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 1
        assert rows[0]["texto"] == "stale never used"
        assert rows[0]["tipo"] == "conversation"
        assert rows[0]["created_at"]

    def test_nothing_stale_writes_no_file(self, palace, tmp_path):
        palace.add_document("conversation", "fresh",
                            created_at=self._NOW.isoformat())
        result = archive_stale_documents(
            palace, days=180, archive_dir=str(tmp_path), now=self._NOW)
        assert result == {"archived": 0, "path": None}
        assert list(tmp_path.iterdir()) == []

    def test_days_zero_disables(self, palace, tmp_path):
        self._seed(palace)
        result = archive_stale_documents(
            palace, days=0, archive_dir=str(tmp_path), now=self._NOW)
        assert result["archived"] == 0
        assert list(tmp_path.iterdir()) == []

    def test_failure_is_non_fatal(self, tmp_path):
        class Broken:
            def stale_docs(self, *a, **k):
                raise RuntimeError("boom")
        result = archive_stale_documents(
            Broken(), days=180, archive_dir=str(tmp_path), now=self._NOW)
        assert result == {"archived": 0, "path": None}
