"""Self-tests for LLM-eval baseline storage + similarity scoring
(fase 2, sessão B).

These pin the PLUMBING that the opt-in `make eval-llm` relies on:
cosine maths, the drift threshold, baseline file naming, save/load
roundtrip, and the comparison that turns "current replies + recorded
baseline" into per-case drift flags.

Everything here is OFFLINE: the embedder is injected as a stub, so
these run inside `make check` with no Ollama. The real Ollama-backed
embedder is only exercised by the live `make eval-llm` smoke.
"""
from __future__ import annotations

import numpy as np
import pytest

from evals.core import EvalCase, EvalKind, EvalResult, EvalStatus
from evals.llm_baseline import (
    baseline_path,
    load_baseline,
    replies_from_results,
    save_baseline,
)
from evals.similarity import (
    SIMILARITY_THRESHOLD,
    baseline_drifts,
    cosine_similarity,
    score_against_baseline,
    text_similarity,
)


# =====================================================================
# cosine_similarity
# =====================================================================

class TestCosine:

    def test_identical_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert cosine_similarity(
            np.array([1.0, 0.0]), np.array([0.0, 1.0])
        ) == pytest.approx(0.0)

    def test_opposite_is_minus_one(self):
        assert cosine_similarity(
            np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        ) == pytest.approx(-1.0)

    def test_zero_vector_is_zero_not_nan(self):
        # All-zeros has undefined direction — must NOT produce nan and
        # poison the report; we define it as maximal drift (0.0).
        assert cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0

    def test_result_is_clamped(self):
        # Float error on an identical pair can nudge past 1.0.
        v = np.array([0.1, 0.2, 0.3])
        assert cosine_similarity(v, v) <= 1.0


# =====================================================================
# text_similarity (stub embedder — offline)
# =====================================================================

def _stub_embedder(table):
    """Return an embedder that maps known text -> fixed vector."""
    return lambda t: np.asarray(table[t], dtype=np.float32)


class TestTextSimilarity:

    def test_both_blank_is_one(self):
        # Never hands the embedder an empty string (it would raise).
        assert text_similarity("", "  ", embedder=lambda t: 1 / 0) == 1.0

    def test_one_blank_is_zero(self):
        assert text_similarity("hello", "", embedder=lambda t: 1 / 0) == 0.0

    def test_uses_injected_embedder(self):
        emb = _stub_embedder({"a": [1, 0, 0], "b": [0, 1, 0]})
        assert text_similarity("a", "b", embedder=emb) == pytest.approx(0.0)
        assert text_similarity("a", "a", embedder=emb) == pytest.approx(1.0)


class TestDriftThreshold:

    def test_below_threshold_drifts(self):
        assert baseline_drifts(SIMILARITY_THRESHOLD - 0.01) is True

    def test_at_threshold_does_not_drift(self):
        assert baseline_drifts(SIMILARITY_THRESHOLD) is False


# =====================================================================
# score_against_baseline
# =====================================================================

class TestScoreAgainstBaseline:

    def test_only_common_cases_scored(self):
        emb = _stub_embedder({"x": [1, 0], "y": [1, 0]})
        scores = score_against_baseline(
            current={"a": "x", "only_current": "y"},
            baseline={"a": "x", "only_baseline": "y"},
            embedder=emb,
        )
        assert set(scores) == {"a"}  # intersection only

    def test_match_is_not_drift(self):
        emb = _stub_embedder({"same": [1, 0, 0]})
        scores = score_against_baseline(
            {"a": "same"}, {"a": "same"}, embedder=emb
        )
        assert scores["a"]["similarity"] == pytest.approx(1.0)
        assert scores["a"]["drift"] is False

    def test_divergent_reply_drifts(self):
        emb = _stub_embedder({"cur": [0, 1, 0], "base": [1, 0, 0]})
        scores = score_against_baseline(
            {"a": "cur"}, {"a": "base"}, embedder=emb
        )
        assert scores["a"]["similarity"] == pytest.approx(0.0)
        assert scores["a"]["drift"] is True


# =====================================================================
# baseline file: path naming
# =====================================================================

class TestBaselinePath:

    def test_name_includes_model_and_version(self, tmp_path):
        p = baseline_path("cortex-lite", 1, base_dir=tmp_path)
        assert p.name == "cortex-lite__v1.json"

    def test_version_is_int_coerced(self, tmp_path):
        assert baseline_path("m", 2, base_dir=tmp_path).name == "m__v2.json"

    def test_unsafe_model_name_is_sanitised(self, tmp_path):
        # A registry-style name must not escape the baselines dir.
        p = baseline_path("library/cortex:gemma", 1, base_dir=tmp_path)
        assert "/" not in p.name
        assert p.parent == tmp_path

    def test_empty_model_falls_back(self, tmp_path):
        assert baseline_path("   ", 1, base_dir=tmp_path).name == "unknown__v1.json"

    def test_named_dormant_version_is_sanitised(self, tmp_path):
        # A dormant named prompt (CORTEX_PROMPT_VERSION=frontier, M2-B2) must
        # baseline like any int version — int() used to crash on it.
        p = baseline_path("cortex-lite-gemma", "frontier", base_dir=tmp_path)
        assert p.name == "cortex-lite-gemma__vfrontier.json"


# =====================================================================
# baseline file: save / load roundtrip
# =====================================================================

class TestBaselineRoundtrip:

    def test_save_then_load(self, tmp_path):
        replies = {"greeting": "Olá!", "math": "É 4."}
        save_baseline("cortex-lite", 1, replies, base_dir=tmp_path)
        loaded = load_baseline("cortex-lite", 1, base_dir=tmp_path)
        assert loaded == replies

    def test_load_missing_is_none(self, tmp_path):
        assert load_baseline("never-recorded", 9, base_dir=tmp_path) is None

    def test_cases_sorted_for_stable_diff(self, tmp_path):
        path = save_baseline(
            "m", 1, {"zebra": "z", "alpha": "a"}, base_dir=tmp_path
        )
        text = path.read_text(encoding="utf-8")
        assert text.index('"alpha"') < text.index('"zebra"')

    def test_doc_records_model_and_version(self, tmp_path):
        import json
        path = save_baseline("cortex-gemma", 3, {"c": "r"}, base_dir=tmp_path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["model"] == "cortex-gemma"
        assert doc["prompt_version"] == 3
        assert "generated_at" in doc

    def test_named_version_roundtrip(self, tmp_path):
        import json
        path = save_baseline("cortex-lite-gemma", "frontier", {"c": "r"},
                             base_dir=tmp_path)
        assert path.name == "cortex-lite-gemma__vfrontier.json"
        assert load_baseline("cortex-lite-gemma", "frontier",
                             base_dir=tmp_path) == {"c": "r"}
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["prompt_version"] == "frontier"

    def test_rebaseline_overwrites_same_pair(self, tmp_path):
        save_baseline("m", 1, {"c": "old"}, base_dir=tmp_path)
        save_baseline("m", 1, {"c": "new"}, base_dir=tmp_path)
        assert load_baseline("m", 1, base_dir=tmp_path) == {"c": "new"}

    def test_different_pair_is_separate_file(self, tmp_path):
        save_baseline("m", 1, {"c": "v1reply"}, base_dir=tmp_path)
        save_baseline("m", 2, {"c": "v2reply"}, base_dir=tmp_path)
        assert load_baseline("m", 1, base_dir=tmp_path) == {"c": "v1reply"}
        assert load_baseline("m", 2, base_dir=tmp_path) == {"c": "v2reply"}


# =====================================================================
# replies_from_results
# =====================================================================

def _result(name, **payload):
    case = EvalCase(name=name, kind=EvalKind.LLM_BEHAVIOR, category="llm_behavior")
    return EvalResult(case=case, status=EvalStatus.PASS, payload=payload)


class TestRepliesFromResults:

    def test_extracts_replies(self):
        results = [_result("a", reply="hello"), _result("b", reply="world")]
        assert replies_from_results(results) == {"a": "hello", "b": "world"}

    def test_skips_results_without_reply(self):
        results = [_result("a", reply="hi"), _result("b", input="no reply key")]
        assert replies_from_results(results) == {"a": "hi"}


# =====================================================================
# reporter — baseline similarity section
# =====================================================================

class TestReportBaselineSection:

    def test_renders_similarity_and_drift(self):
        from evals.report import format_terminal

        results = [
            _result("good", baseline_similarity=0.98, baseline_drift=False),
            _result("bad", baseline_similarity=0.40, baseline_drift=True),
            _result("fresh", baseline_status="new"),
        ]
        out = format_terminal(results)
        assert "Baseline similarity" in out
        assert "0.980" in out and "0.400" in out
        assert "DRIFT" in out
        assert "no baseline yet" in out

    def test_no_section_without_baseline_data(self):
        from evals.report import format_terminal

        out = format_terminal([_result("a", reply="hi")])
        assert "Baseline similarity" not in out


# =====================================================================
# CLI — _handle_baseline (offline: embedder/save monkeypatched)
# =====================================================================

class TestHandleBaseline:

    def _llm_result(self, name, reply, model="cortex-lite", version=1):
        return _result(name, reply=reply, model=model, prompt_version=version)

    def test_rebaseline_saves_replies(self, monkeypatch):
        import evals.__main__ as cli

        captured = {}
        monkeypatch.setattr(
            cli, "save_baseline",
            lambda m, v, replies: captured.update(model=m, version=v, replies=replies)
            or __import__("pathlib").Path("/tmp/x.json"),
        )
        results = [self._llm_result("a", "Olá!"), self._llm_result("b", "É 4.")]
        cli._handle_baseline(results, rebaseline=True)
        assert captured["model"] == "cortex-lite"
        assert captured["version"] == 1
        assert captured["replies"] == {"a": "Olá!", "b": "É 4."}

    def test_compare_annotates_results(self, monkeypatch):
        import evals.__main__ as cli

        monkeypatch.setattr(cli, "load_baseline", lambda m, v: {"a": "old"})
        monkeypatch.setattr(
            cli, "score_against_baseline",
            lambda cur, base: {"a": {"similarity": 0.5, "drift": True}},
        )
        results = [self._llm_result("a", "new reply")]
        cli._handle_baseline(results, rebaseline=False)
        assert results[0].payload["baseline_similarity"] == 0.5
        assert results[0].payload["baseline_drift"] is True

    def test_compare_marks_new_case(self, monkeypatch):
        import evals.__main__ as cli

        monkeypatch.setattr(cli, "load_baseline", lambda m, v: {"other": "x"})
        monkeypatch.setattr(cli, "score_against_baseline", lambda cur, base: {})
        results = [self._llm_result("a", "reply")]
        cli._handle_baseline(results, rebaseline=False)
        assert results[0].payload["baseline_status"] == "new"

    def test_no_reply_result_is_not_baselined(self, monkeypatch):
        # Tool-call results (no 'reply' payload) must be left untouched —
        # not marked "new", not scored.
        import evals.__main__ as cli

        monkeypatch.setattr(cli, "load_baseline", lambda m, v: {"chat": "old"})
        monkeypatch.setattr(
            cli, "score_against_baseline", lambda cur, base: {}
        )
        tool_result = _result("tool", model="cortex-lite", prompt_version=1,
                              tool_used="web_search")
        cli._handle_baseline([tool_result], rebaseline=False)
        assert "baseline_status" not in tool_result.payload
        assert "baseline_similarity" not in tool_result.payload

    def test_no_baseline_is_graceful(self, monkeypatch, capsys):
        import evals.__main__ as cli

        monkeypatch.setattr(cli, "load_baseline", lambda m, v: None)
        results = [self._llm_result("a", "reply")]
        cli._handle_baseline(results, rebaseline=False)
        assert "no baseline" in capsys.readouterr().out
