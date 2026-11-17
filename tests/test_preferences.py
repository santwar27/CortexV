"""
Tests for the implicit preference engine (Tier 2.3).

Three layers, no network:

  - extract_preferences with a stub `llm_chat_fn` so the LLM call is a
    pure function of the canned response. Covers happy path, JSON in
    markdown fence, no JSON, malformed JSON, partial bad rows, empty
    array, predicate slug validation, confidence clamp.

  - recent_conversations + persist_preferences against an in-memory
    Palace fixture (same trick the temporal tests use).

  - Palace.get_preferences() — the convenience wrapper.

A separate integration smoke test verifies the Studier wiring: it
mocks `ollama.chat`, runs `_consolidate_preferences()`, and asserts
the Palace ended up with the expected triples.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.memory.palace import Palace
from src.memory.preferences import (
    PreferenceTriple,
    _coerce_triple,
    extract_preferences,
    format_preferences_for_prompt,
    persist_preferences,
    recent_conversations,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def palace():
    p = Palace(db_path=":memory:")
    yield p
    p.close()


def _llm_returning(content: str):
    """Build a stub callable matching ollama.chat's signature."""
    def fn(**kwargs):
        return {"message": {"content": content}}
    return fn


# ----------------------------------------------------------------------
# _coerce_triple — schema gate
# ----------------------------------------------------------------------


class TestCoerceTriple:
    def test_happy_path(self):
        t = _coerce_triple({
            "predicado": "prefere_resposta_estilo",
            "objeto": "curta_direta",
            "confidence": 0.9,
            "evidence": "prefiro curta",
        })
        assert t is not None
        assert t.predicado == "prefere_resposta_estilo"
        assert t.objeto == "curta_direta"
        assert t.confidence == 0.9

    def test_predicado_must_start_with_prefere(self):
        assert _coerce_triple({
            "predicado": "gosta_de",
            "objeto": "x",
            "confidence": 0.9,
        }) is None

    def test_predicado_uppercase_is_normalised(self):
        # The LLM occasionally yells; coerce should lowercase rather
        # than throw away an otherwise valid triple.
        t = _coerce_triple({
            "predicado": "PREFERE_X",
            "objeto": "y",
            "confidence": 0.9,
        })
        assert t is not None
        assert t.predicado == "prefere_x"

    def test_predicado_with_invalid_chars_rejected(self):
        # Spaces, hyphens, accents — none of these belong in a slug.
        for bad in ("prefere x", "prefere-x", "prefere_é", "outro_prefix"):
            assert _coerce_triple({
                "predicado": bad,
                "objeto": "y",
                "confidence": 0.9,
            }) is None

    def test_empty_objeto_rejected(self):
        assert _coerce_triple({
            "predicado": "prefere_x",
            "objeto": "",
            "confidence": 0.9,
        }) is None

    def test_objeto_too_long_rejected(self):
        assert _coerce_triple({
            "predicado": "prefere_x",
            "objeto": "y" * 101,
            "confidence": 0.9,
        }) is None

    def test_confidence_clamped_to_unit_range(self):
        t = _coerce_triple({"predicado": "prefere_x", "objeto": "y",
                            "confidence": 5.0})
        assert t is not None and t.confidence == 1.0
        t = _coerce_triple({"predicado": "prefere_x", "objeto": "y",
                            "confidence": -0.3})
        assert t is not None and t.confidence == 0.0

    def test_non_numeric_confidence_rejected(self):
        assert _coerce_triple({"predicado": "prefere_x", "objeto": "y",
                               "confidence": "alta"}) is None

    def test_non_dict_rejected(self):
        assert _coerce_triple("not a dict") is None
        assert _coerce_triple(None) is None
        assert _coerce_triple(["array"]) is None


# ----------------------------------------------------------------------
# extract_preferences — LLM-shape contract
# ----------------------------------------------------------------------


class TestExtractPreferences:
    def test_empty_input_short_circuits(self):
        # The stub should not even be called for empty input.
        sentinel = MagicMock(side_effect=AssertionError("must not call LLM"))
        assert extract_preferences("", sentinel, "any-model") == []
        assert extract_preferences("   \n\t  ", sentinel, "any-model") == []

    def test_happy_path_clean_json(self):
        llm = _llm_returning(
            '[{"predicado":"prefere_resposta_estilo","objeto":"curta_direta",'
            '"confidence":0.9,"evidence":"prefiro curta"}]'
        )
        triples = extract_preferences("prefiro respostas curtas", llm, "m")
        assert len(triples) == 1
        assert triples[0].objeto == "curta_direta"

    def test_extracts_json_buried_in_prose(self):
        # Llama sometimes prepends a "Here is the JSON:" line.
        llm = _llm_returning(
            'Here is what I found:\n'
            '[{"predicado":"prefere_idioma","objeto":"pt-br","confidence":0.95}]\n'
            'Hope this helps!'
        )
        triples = extract_preferences("falo em português", llm, "m")
        assert len(triples) == 1
        assert triples[0].predicado == "prefere_idioma"

    def test_extracts_json_inside_markdown_fence(self):
        llm = _llm_returning(
            "```json\n"
            '[{"predicado":"prefere_idioma","objeto":"pt-br","confidence":0.95}]\n'
            "```"
        )
        triples = extract_preferences("...", llm, "m")
        assert len(triples) == 1

    def test_no_json_in_response_returns_empty(self):
        llm = _llm_returning("I cannot find any preferences here.")
        assert extract_preferences("...", llm, "m") == []

    def test_malformed_json_returns_empty(self):
        llm = _llm_returning('[{"predicado":"prefere_x", BROKEN }]')
        assert extract_preferences("...", llm, "m") == []

    def test_empty_array_returns_empty(self):
        llm = _llm_returning("[]")
        assert extract_preferences("...", llm, "m") == []

    def test_partial_bad_rows_dropped_good_rows_kept(self):
        llm = _llm_returning(
            '['
            '{"predicado":"prefere_idioma","objeto":"pt-br","confidence":0.9},'
            '{"predicado":"NOT_VALID","objeto":"x","confidence":0.9},'
            '{"predicado":"prefere_horario","objeto":"noite","confidence":0.8}'
            ']'
        )
        triples = extract_preferences("...", llm, "m")
        # The middle one is invalid → only two survive.
        assert {t.predicado for t in triples} == {
            "prefere_idioma", "prefere_horario"
        }

    def test_llm_exception_returns_empty(self):
        def boom(**kw):
            raise RuntimeError("ollama down")
        assert extract_preferences("...", boom, "m") == []

    def test_passes_temperature_zero_for_determinism(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": "[]"}}
        extract_preferences("...", fn, "my-model")
        assert captured.get("options", {}).get("temperature") == 0.0
        assert captured.get("model") == "my-model"

    def test_input_truncated_to_8k_chars(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": "[]"}}
        huge = "x" * 20000
        extract_preferences(huge, fn, "m")
        # The user-role message content should be capped.
        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        assert len(user_msg["content"]) <= 8000


# ----------------------------------------------------------------------
# recent_conversations
# ----------------------------------------------------------------------


class TestRecentConversations:
    def _add(self, p, texto, hours_ago, tipo="conversation"):
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        return p.add_document(tipo, texto, created_at=ts)

    def test_concatenates_within_window(self, palace):
        self._add(palace, "Pedro prefere noite", hours_ago=2)
        self._add(palace, "gosta de café", hours_ago=20)
        out = recent_conversations(palace, hours=24)
        assert "Pedro prefere noite" in out
        assert "gosta de café" in out

    def test_excludes_documents_outside_window(self, palace):
        self._add(palace, "old preference", hours_ago=200)
        out = recent_conversations(palace, hours=24)
        assert "old preference" not in out

    def test_excludes_non_conversation_tipo(self, palace):
        self._add(palace, "knowledge doc", hours_ago=1, tipo="knowledge")
        self._add(palace, "conv doc", hours_ago=1, tipo="conversation")
        out = recent_conversations(palace, hours=24)
        assert "knowledge doc" not in out
        assert "conv doc" in out

    def test_caps_at_max_chars_keeping_newest(self, palace):
        # newest first; each turn cut to a quarter of the budget (s88)
        self._add(palace, "NEW " + "x" * 100, hours_ago=1)
        self._add(palace, "OLD " + "y" * 100, hours_ago=10)
        out = recent_conversations(palace, hours=24, max_chars=120)
        assert len(out) <= 120
        assert out.index("NEW") < out.index("OLD")

    def test_one_long_turn_does_not_hide_the_day(self, palace):
        """The newest turn over the whole budget stopped the loop with
        nothing taken: an empty corpus, and both nightly passes skipped the
        day (s88, Codex lote 6, reproduced)."""
        self._add(palace, "User: prefiro respostas curtas\nCortex: anotado", hours_ago=3)
        self._add(palace, "User: resume esse artigo\nCortex: " + "x" * 9000, hours_ago=1)
        out = recent_conversations(palace, hours=24, max_chars=8000)
        assert "prefiro respostas curtas" in out
        assert "User: resume esse artigo" in out          # its head, cut
        assert len(out) <= 8000

    def test_zero_hours_returns_empty(self, palace):
        self._add(palace, "anything", hours_ago=1)
        assert recent_conversations(palace, hours=0) == ""

    def test_empty_palace_returns_empty(self, palace):
        assert recent_conversations(palace, hours=24) == ""


# ----------------------------------------------------------------------
# persist_preferences
# ----------------------------------------------------------------------


class TestPersistPreferences:
    def test_writes_high_confidence_triples(self, palace):
        triples = [
            PreferenceTriple("prefere_idioma", "pt-br", 0.9),
            PreferenceTriple("prefere_horario", "noite", 0.8),
        ]
        n = persist_preferences(palace, triples)
        assert n == 2
        active = palace.list_facts(sujeito="user")
        preds = {r["predicado"] for r in active}
        assert preds == {"prefere_idioma", "prefere_horario"}

    def test_drops_low_confidence(self, palace):
        triples = [
            PreferenceTriple("prefere_x", "y", 0.4),  # below floor
            PreferenceTriple("prefere_z", "w", 0.9),
        ]
        n = persist_preferences(palace, triples, min_confidence=0.55)
        assert n == 1
        assert palace.get_fact("user", "prefere_x") is None
        assert palace.get_fact("user", "prefere_z") is not None

    def test_idempotent_for_unchanged_value(self, palace):
        triples = [PreferenceTriple("prefere_idioma", "pt-br", 0.9)]
        assert persist_preferences(palace, triples) == 1
        # Second call with the same value should be a no-op (no churn).
        assert persist_preferences(palace, triples) == 0
        # And only ONE row exists.
        all_rows = palace.list_facts(sujeito="user", include_invalidated=True)
        assert len(all_rows) == 1

    def test_replaces_value_via_bitemporal_closure(self, palace):
        persist_preferences(palace, [
            PreferenceTriple("prefere_idioma", "pt-br", 0.9),
        ])
        # New value should close the old row and insert a new one.
        n = persist_preferences(palace, [
            PreferenceTriple("prefere_idioma", "en", 0.95),
        ])
        assert n == 1
        # Active = en; history = 2 rows.
        active = palace.get_fact("user", "prefere_idioma")
        assert active is not None
        assert active["objeto"] == "en"
        all_rows = palace.list_facts(sujeito="user", include_invalidated=True)
        assert len(all_rows) == 2

    def test_source_tag_persisted(self, palace):
        persist_preferences(palace, [
            PreferenceTriple("prefere_x", "y", 0.9),
        ], source="studier_inferred")
        row = palace.get_fact("user", "prefere_x")
        assert row["source"] == "studier_inferred"

    def test_empty_list_no_op(self, palace):
        assert persist_preferences(palace, []) == 0


# ----------------------------------------------------------------------
# Palace.get_preferences
# ----------------------------------------------------------------------


class TestPalaceGetPreferences:
    def test_returns_only_prefere_predicates(self, palace):
        palace.add_fact("user", "prefere_idioma", "pt-br")
        palace.add_fact("user", "nome", "Pedro")  # not a preference
        prefs = palace.get_preferences()
        preds = [p["predicado"] for p in prefs]
        assert "prefere_idioma" in preds
        assert "nome" not in preds

    def test_excludes_invalidated_by_default(self, palace):
        palace.add_fact("user", "prefere_x", "v1")
        palace.add_fact("user", "prefere_x", "v2")  # closes v1
        prefs = palace.get_preferences()
        assert len(prefs) == 1
        assert prefs[0]["objeto"] == "v2"

    def test_include_invalidated_flag(self, palace):
        palace.add_fact("user", "prefere_x", "v1")
        palace.add_fact("user", "prefere_x", "v2")
        prefs = palace.get_preferences(include_invalidated=True)
        assert len(prefs) == 2

    def test_min_confidence_filter(self, palace):
        palace.add_fact("user", "prefere_a", "x", confidence=0.4)
        palace.add_fact("user", "prefere_b", "y", confidence=0.9)
        prefs = palace.get_preferences(min_confidence=0.5)
        preds = [p["predicado"] for p in prefs]
        assert preds == ["prefere_b"]

    def test_other_sujeito_excluded(self, palace):
        palace.add_fact("user", "prefere_x", "u")
        palace.add_fact("system", "prefere_x", "s")
        prefs = palace.get_preferences()  # default sujeito="user"
        assert all(p["sujeito"] == "user" for p in prefs)


# ----------------------------------------------------------------------
# Studier integration
# ----------------------------------------------------------------------


class TestFormatPreferencesForPrompt:
    """The string block injected into the engine system prompt."""

    def test_empty_input_returns_empty_string(self):
        assert format_preferences_for_prompt([]) == ""

    def test_drops_rows_below_confidence_floor(self):
        rows = [
            {"predicado": "prefere_x", "objeto": "y", "confidence": 0.4},
            {"predicado": "prefere_z", "objeto": "w", "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows, min_confidence=0.55)
        assert "prefere_x" not in out
        assert "prefere_z → w" in out

    def test_all_below_floor_returns_empty(self):
        rows = [{"predicado": "prefere_x", "objeto": "y", "confidence": 0.1}]
        assert format_preferences_for_prompt(rows, min_confidence=0.5) == ""

    def test_format_shape(self):
        rows = [
            {"predicado": "prefere_idioma", "objeto": "pt-br", "confidence": 0.95},
            {"predicado": "prefere_horario", "objeto": "madrugada", "confidence": 0.8},
        ]
        out = format_preferences_for_prompt(rows)
        assert "USER PROFILE" in out
        assert "prefere_idioma → pt-br" in out
        assert "prefere_horario → madrugada" in out
        # Caveat hedge present so the LLM treats them as soft context
        assert "not absolutes" in out or "current turn always wins" in out
        # Trailing blank line so the block doesn't run into the next clause
        assert out.endswith("\n\n")

    def test_skips_rows_with_missing_fields(self):
        rows = [
            {"predicado": "", "objeto": "y", "confidence": 0.9},
            {"predicado": "prefere_z", "objeto": "", "confidence": 0.9},
            {"predicado": "prefere_ok", "objeto": "v", "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "prefere_ok → v" in out
        assert out.count("\n- ") == 1

    def test_handles_non_numeric_confidence_gracefully(self):
        rows = [
            {"predicado": "prefere_x", "objeto": "y", "confidence": "alta"},
            {"predicado": "prefere_z", "objeto": "w", "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        # "alta" falls through to 0.0 and gets dropped
        assert "prefere_x" not in out
        assert "prefere_z → w" in out

    # ----- top-N cap (M3-A8, review Fable 5) -----

    def test_caps_at_max_lines_keeping_highest_confidence(self):
        # The nightly extractor grows rows monotonically; without a cap
        # the block eats the num_ctx budget (M2-A1) line by line.
        rows = [
            {"predicado": f"prefere_p{i:02d}", "objeto": f"v{i}",
             "confidence": 0.55 + i * 0.01}
            for i in range(20)
        ]
        out = format_preferences_for_prompt(rows, max_lines=12)
        assert out.count("\n- ") == 12
        # Highest-confidence rows survive (the last 12 of the ramp) …
        assert "prefere_p19 → v19" in out
        assert "prefere_p08 → v8" in out
        # … the lowest are the ones dropped.
        assert "prefere_p07" not in out
        assert "prefere_p00" not in out

    def test_cap_ties_keep_palace_order(self):
        rows = [
            {"predicado": f"prefere_p{i}", "objeto": f"v{i}",
             "confidence": 0.9}
            for i in range(5)
        ]
        out = format_preferences_for_prompt(rows, max_lines=3)
        assert out.count("\n- ") == 3
        for i in (0, 1, 2):
            assert f"prefere_p{i} → v{i}" in out

    def test_under_cap_unaffected(self):
        rows = [
            {"predicado": "prefere_a", "objeto": "x", "confidence": 0.9},
            {"predicado": "prefere_b", "objeto": "y", "confidence": 0.7},
        ]
        out = format_preferences_for_prompt(rows, max_lines=12)
        assert out.count("\n- ") == 2

    # ----- read-path sanitization (defense in depth) -----

    def test_drops_predicate_failing_slug_regex(self):
        # Various predicados that bypass the write-time _PREDICADO_RE
        # check (manual edit, migration, future bug). All should be
        # rejected at read time too.
        rows = [
            {"predicado": "prefere x", "objeto": "y", "confidence": 0.9},     # space
            {"predicado": "prefere-x", "objeto": "y", "confidence": 0.9},     # hyphen
            {"predicado": "prefere_é", "objeto": "y", "confidence": 0.9},     # accent
            {"predicado": "outra_categoria", "objeto": "y", "confidence": 0.9},  # missing prefere_ prefix
            {"predicado": "prefere_ok", "objeto": "v", "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        # Only the well-formed row survives
        assert out.count("\n- ") == 1
        assert "prefere_ok → v" in out

    def test_drops_objeto_with_newline_injection(self):
        rows = [
            {"predicado": "prefere_x", "objeto": "valid\nattack",  "confidence": 0.9},
            {"predicado": "prefere_y", "objeto": "valid\r\ntoo",    "confidence": 0.9},
            {"predicado": "prefere_z", "objeto": "ok",              "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "attack" not in out
        assert "too" not in out
        assert "prefere_z → ok" in out

    def test_drops_objeto_with_control_chars(self):
        rows = [
            {"predicado": "prefere_x", "objeto": "valid\x00null",   "confidence": 0.9},
            {"predicado": "prefere_y", "objeto": "valid\x1bescape", "confidence": 0.9},
            {"predicado": "prefere_z", "objeto": "ok",              "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "null" not in out
        assert "escape" not in out
        assert "prefere_z → ok" in out

    def test_drops_objeto_with_authoritative_tag_injection(self):
        # The engine's system prompt reserves uppercase bracketed tags
        # like [KNOWLEDGE] for ground-truth markers. An adversarial
        # objeto with that shape would corrupt the prompt semantically.
        rows = [
            {"predicado": "prefere_x", "objeto": "[KNOWLEDGE] attack",   "confidence": 0.9},
            {"predicado": "prefere_y", "objeto": "foo [INSTRUCTION] bar","confidence": 0.9},
            {"predicado": "prefere_z", "objeto": "ok",                    "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "[KNOWLEDGE]" not in out
        assert "[INSTRUCTION]" not in out
        assert "prefere_z → ok" in out

    def test_keeps_lowercase_brackets_in_objeto(self):
        # Lowercase or mixed-case brackets are NOT engine markers — they're
        # legitimate notation. Don't over-reject.
        rows = [
            {"predicado": "prefere_estilo", "objeto": "curto [opcional]", "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "[opcional]" in out

    def test_drops_objeto_too_long(self):
        rows = [
            {"predicado": "prefere_x", "objeto": "a" * 101, "confidence": 0.9},
            {"predicado": "prefere_y", "objeto": "a" * 100, "confidence": 0.9},
        ]
        out = format_preferences_for_prompt(rows)
        assert "prefere_x" not in out
        assert "prefere_y" in out


# ----------------------------------------------------------------------
# CortexMemory.get_user_profile
# ----------------------------------------------------------------------


class TestMemoryGetUserProfile:
    """End-to-end: persisted triples come back as a formatted block."""

    def test_returns_empty_string_when_no_preferences(self, tmp_path):
        from src.core.memory import CortexMemory
        # palace= injection skips Ollama embedder construction
        mem = CortexMemory(palace=Palace(db_path=":memory:"))
        assert mem.get_user_profile() == ""
        mem._palace.close()

    def test_returns_block_when_preferences_persisted(self, tmp_path):
        from src.core.memory import CortexMemory
        p = Palace(db_path=":memory:")
        p.add_fact("user", "prefere_idioma", "pt-br", confidence=0.9)
        mem = CortexMemory(palace=p)
        out = mem.get_user_profile()
        assert "USER PROFILE" in out
        assert "prefere_idioma → pt-br" in out
        p.close()

    def test_filters_by_inject_confidence_floor(self):
        from src.core.memory import CortexMemory
        p = Palace(db_path=":memory:")
        p.add_fact("user", "prefere_low", "x", confidence=0.4)
        p.add_fact("user", "prefere_high", "y", confidence=0.9)
        mem = CortexMemory(palace=p)
        out = mem.get_user_profile()
        assert "prefere_low" not in out
        assert "prefere_high → y" in out
        p.close()

    def test_returns_empty_on_palace_exception(self, monkeypatch):
        """A bug in get_preferences must not break the system prompt."""
        from src.core.memory import CortexMemory
        p = Palace(db_path=":memory:")
        mem = CortexMemory(palace=p)
        monkeypatch.setattr(
            p, "get_preferences",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert mem.get_user_profile() == ""
        p.close()


# ----------------------------------------------------------------------
# Engine system prompt injection
# ----------------------------------------------------------------------


class TestEngineSystemPromptInjection:
    """The engine's _build_system_prompt must splice the USER PROFILE
    block between the temporal context and the static body."""

    def _make_engine(self, profile_text: str):
        """Build an engine without booting Ollama / a real Palace.
        Stubs memory.get_user_profile to return `profile_text`."""
        from src.core.engine import OllamaEngine
        e = OllamaEngine.__new__(OllamaEngine)
        # Re-run __init__ in a guarded try so anything network-y can fail
        # silently — we only need the system body, not a working memory.
        try:
            OllamaEngine.__init__(e)
        except Exception:
            pass
        # Force-replace memory with a stub regardless of init outcome.
        stub = MagicMock()
        stub.get_user_profile = MagicMock(return_value=profile_text)
        e.memory = stub
        return e

    def test_prompt_includes_profile_when_present(self):
        e = self._make_engine("USER PROFILE block\n\n")
        sp = e.system_prompt
        assert "USER PROFILE block" in sp["content"]
        # Profile sits AFTER temporal context, BEFORE static body
        idx_temporal = sp["content"].find("TEMPORAL CONTEXT")
        idx_profile = sp["content"].find("USER PROFILE block")
        idx_body = sp["content"].find("You are CORTEX")
        assert idx_temporal < idx_profile < idx_body

    def test_prompt_omits_profile_when_empty(self):
        e = self._make_engine("")
        content = e.system_prompt["content"]
        assert "USER PROFILE" not in content
        # And the body is still there.
        assert "You are CORTEX" in content

    def test_profile_exception_does_not_break_prompt(self):
        from src.core.engine import OllamaEngine
        e = OllamaEngine.__new__(OllamaEngine)
        try:
            OllamaEngine.__init__(e)
        except Exception:
            pass
        stub = MagicMock()
        stub.get_user_profile = MagicMock(side_effect=RuntimeError("boom"))
        e.memory = stub
        # Must not raise; static body must still appear.
        sp = e.system_prompt
        assert "You are CORTEX" in sp["content"]


class TestStudierPreferenceIntegration:
    """End-to-end: studier._consolidate_preferences runs against a real
    in-memory Palace path (we point PALACE_DB_PATH at a tmp file) and
    a mocked ollama.chat. Asserts the LLM was called and triples landed."""

    def test_extracts_and_persists_via_studier(self, tmp_path, monkeypatch):
        from src.core.studier import NightlyStudier

        db_path = tmp_path / "palace.db"
        # Seed conversations the extractor will see.
        p = Palace(db_path=str(db_path))
        now_iso = datetime.now(timezone.utc).isoformat()
        # The stored shape (store_interaction): the user's words, then Cortex's.
        p.add_document("conversation", "User: prefiro respostas curtas\nCortex: anotado",
                       created_at=now_iso)
        p.close()

        canned = (
            '[{"predicado":"prefere_resposta_estilo",'
            '"objeto":"curta_direta","confidence":0.9,'
            '"evidence":"prefiro respostas curtas"}]'
        )

        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", str(db_path))

        with patch("src.core.studier.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": canned}}
            written = studier._consolidate_preferences()

        assert written == 1
        # Verify the triple actually landed in the Palace.
        p2 = Palace(db_path=str(db_path))
        try:
            prefs = p2.get_preferences()
            assert len(prefs) == 1
            assert prefs[0]["predicado"] == "prefere_resposta_estilo"
            assert prefs[0]["objeto"] == "curta_direta"
            assert prefs[0]["source"] == "studier_inferred"
        finally:
            p2.close()

    def test_skips_when_no_palace_db(self, tmp_path):
        from src.core.studier import NightlyStudier
        studier = NightlyStudier.__new__(NightlyStudier)
        studier.engine = MagicMock()
        studier.engine.model_name = "cortex-lite"
        studier.PALACE_DB_PATH = str(tmp_path / "missing.db")
        assert studier._consolidate_preferences() == 0

    def test_extractor_crash_returns_zero_not_raises(self, tmp_path, monkeypatch):
        """A bad LLM, broken JSON, or Palace error in the preference
        path must NOT propagate — the briefing pipeline depends on it."""
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
            # Should not raise.
            assert studier._consolidate_preferences() == 0


class TestOnlyPedroSWordsReachTheProfile:
    """The nightly read "User: …\\nCortex: …" turns and kept whatever the 8B
    returned: an example Pedro asked for — "a Marta mora em Paris e prefere
    respostas longas" — became vive_em=Paris and prefere_resposta_estilo=
    longa for Pedro (s88, Codex lote 6, reproduced)."""

    TURN = ("User: me dá um exemplo fictício de preferências de alguém\n"
            "Cortex: Exemplo: a Marta mora em Paris e prefere respostas longas.\n"
            "User: (isto foi escrito pelo modelo, não pelo Pedro) prefiro respostas longas")

    def _studier(self, tmp_path, monkeypatch, turns):
        from src.core.studier import NightlyStudier
        db = tmp_path / "palace.db"
        p = Palace(db_path=str(db))
        for t in turns:
            p.add_document("conversation", t, created_at=datetime.now(timezone.utc).isoformat())
        p.close()
        s = NightlyStudier.__new__(NightlyStudier)
        s.engine = MagicMock()
        s.engine.model_name = "cortex-lite"
        monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", str(db))
        return s, db

    def test_a_preference_quoted_from_cortex_is_dropped(self, tmp_path, monkeypatch):
        s, db = self._studier(tmp_path, monkeypatch, [self.TURN])
        canned = ('[{"predicado":"prefere_resposta_estilo","objeto":"longa",'
                  '"confidence":0.9,"evidence":"prefiro respostas longas"}]')
        with patch("src.core.studier.ollama") as o:
            o.chat.return_value = {"message": {"content": canned}}
            assert s._consolidate_preferences() == 0
        p = Palace(db_path=str(db))
        assert p.get_fact("user", "prefere_resposta_estilo") is None
        p.close()

    def test_a_fact_only_cortex_said_is_dropped(self, tmp_path, monkeypatch):
        s, db = self._studier(tmp_path, monkeypatch, [self.TURN])
        with patch("src.core.studier.ollama") as o:
            o.chat.side_effect = [
                {"message": {"content": "Pedro pediu um exemplo; a Marta mora em Paris."}},
                {"message": {"content": '[{"sujeito":"user","predicado":"vive_em",'
                                        '"objeto":"Paris","confidence":0.9,'
                                        '"evidence":"mora em Paris"}]'}}]
            assert s._consolidate_memory()["facts_persisted"] == 0

    def test_what_pedro_said_still_lands(self, tmp_path, monkeypatch):
        s, db = self._studier(tmp_path, monkeypatch,
                              ["User: Prefiro respostas CURTAS, sério.\nCortex: anotado"])
        canned = ('[{"predicado":"prefere_resposta_estilo","objeto":"curta",'
                  '"confidence":0.9,"evidence":"prefiro respostas curtas"}]')
        with patch("src.core.studier.ollama") as o:
            o.chat.return_value = {"message": {"content": canned}}
            assert s._consolidate_preferences() == 1
