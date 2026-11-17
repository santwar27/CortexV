"""
Tests for src/core/semantic_router.py — intent routing by meaning (item 1a of
docs/design/m1_interpretation_plan.md).

Strategy: a deterministic FAKE embedder (stable hash → per-word unit vectors,
summed) so phrases sharing words land close in cosine space, WITHOUT touching
Ollama. This exercises the real routing logic — nearest-per-mode, the score
floor, the ambiguity margin — with full control over what's "similar".
"""
import hashlib

import numpy as np
import pytest

from src.core.semantic_router import (
    DEFAULT_CATALOG,
    IntentExamples,
    SemanticRouter,
    _semantic_router_enabled,
)

_DIM = 64


def _word_vec(word: str) -> np.ndarray:
    seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed).standard_normal(_DIM).astype(np.float32)


def fake_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedding: shared words → similar vectors."""
    v = np.zeros(_DIM, dtype=np.float32)
    for w in text.lower().split():
        v += _word_vec(w)
    return v


# A tiny, control-friendly catalogue for the logic tests.
CATALOG = (
    IntentExamples("PLAY_MUSIC", ("toca musica", "play song")),
    IntentExamples("SHOW_JOBS", ("mostra vagas", "show jobs")),
    IntentExamples("WEATHER", ("previsao tempo", "weather forecast")),
)


@pytest.fixture
def router():
    return SemanticRouter(fake_embed, catalog=CATALOG,
                          min_score=0.4, min_margin=0.04)


class TestRouting:
    def test_exact_example_routes_to_its_mode(self, router):
        m = router.route("toca musica")
        assert m is not None
        assert m.mode == "PLAY_MUSIC"
        assert m.score > 0.99  # identical to an example

    def test_paraphrase_shares_words_routes(self, router):
        # shares "musica" with PLAY_MUSIC's example, nothing with others
        m = router.route("quero musica")
        assert m is not None and m.mode == "PLAY_MUSIC"

    def test_english_example_routes(self, router):
        m = router.route("show jobs please")
        assert m is not None and m.mode == "SHOW_JOBS"

    def test_no_overlap_declines(self, router):
        # words share nothing with any example → below the score floor
        assert router.route("xyzzy plugh frobnicate") is None

    def test_empty_query_declines(self, router):
        assert router.route("") is None
        assert router.route("   ") is None

    def test_match_carries_score_margin_example(self, router):
        m = router.route("play song")
        assert m.mode == "PLAY_MUSIC"
        assert 0.0 <= m.margin <= 2.0
        assert m.example in ("toca musica", "play song")


class TestThresholds:
    def test_score_floor_blocks_weak_match(self):
        # A high floor rejects even a partial overlap.
        r = SemanticRouter(fake_embed, catalog=CATALOG,
                           min_score=0.98, min_margin=0.0)
        assert r.route("quero musica hoje") is None

    def test_margin_blocks_ambiguous(self):
        # Two modes share the SAME example word → a query of that word is
        # equidistant → margin ~0 → declined even though score is high.
        ambiguous = (
            IntentExamples("MODE_A", ("alpha",)),
            IntentExamples("MODE_B", ("alpha",)),
        )
        r = SemanticRouter(fake_embed, catalog=ambiguous,
                           min_score=0.4, min_margin=0.05)
        assert r.route("alpha") is None

    def test_overrides_default_thresholds(self):
        r = SemanticRouter(fake_embed, catalog=CATALOG,
                           min_score=0.1, min_margin=0.0)
        assert r.min_score == 0.1 and r.min_margin == 0.0


class TestConstruction:
    def test_examples_embedded_once_at_init(self):
        calls = {"n": 0}

        def counting_embed(text):
            calls["n"] += 1
            return fake_embed(text)

        r = SemanticRouter(counting_embed, catalog=CATALOG)
        n_examples = sum(len(e.phrases) for e in CATALOG)
        assert calls["n"] == n_examples  # one embed per example, no more
        calls["n"] = 0
        r.route("toca musica")
        assert calls["n"] == 1  # exactly one embed per query

    def test_empty_catalog_declines(self):
        r = SemanticRouter(fake_embed, catalog=())
        assert r.route("anything") is None

    def test_dim_mismatch_declines_not_crashes(self, router):
        # Query embedder returns a different dim → decline, never raise.
        r = SemanticRouter(fake_embed, catalog=CATALOG)
        r._embed = lambda t: np.ones(8, dtype=np.float32)
        assert r.route("toca musica") is None

    def test_failed_example_embed_is_skipped(self):
        def flaky(text):
            if "boom" in text:
                raise RuntimeError("embed down")
            return fake_embed(text)

        cat = (IntentExamples("A", ("boom phrase", "good phrase")),)
        r = SemanticRouter(flaky, catalog=cat)
        # the good phrase survived; routing to it still works
        assert r.route("good phrase").mode == "A"


class TestDefaultCatalog:
    def test_default_catalog_smoke(self):
        # The shipped catalogue embeds and routes a realistic phrase.
        r = SemanticRouter(fake_embed, min_score=0.3, min_margin=0.0)
        m = r.route("quais estudos estão na fila")
        # shares words with STUDIER_LIST examples
        assert m is None or m.mode in {e.mode for e in DEFAULT_CATALOG}

    def test_all_modes_are_known_dispatch_labels(self):
        # Guard: every catalogue mode must be a real dispatch routing label,
        # so a match can be handed straight to the existing handler.
        from src.core.dispatch import registry
        known = {
            r.mode
            for r in (*registry.DISPATCH_ROUTES, *registry.POST_PLANNER_ROUTES)
            if r.mode
        }
        for entry in DEFAULT_CATALOG:
            assert entry.mode in known, f"{entry.mode} not a dispatch mode"


class TestExplain:
    """explain() is what the SHADOW LOG calls: it must always report the
    candidate — including the one it just rejected — so near-misses can be
    told apart from genuine chat when calibrating the thresholds (item 1c)."""

    def test_accepted_matches_route(self, router):
        d = router.explain("toca musica")
        assert d.accepted is True and d.reason is None
        assert d.mode == "PLAY_MUSIC"
        m = router.route("toca musica")
        # the wrapper and the explanation can never disagree
        assert (m.mode, m.score, m.margin, m.example) == (
            d.mode, d.score, d.margin, d.example,
        )

    def test_decline_below_score_still_names_the_candidate(self):
        r = SemanticRouter(fake_embed, catalog=CATALOG,
                           min_score=0.98, min_margin=0.0)
        d = r.explain("quero musica hoje")
        assert d.accepted is False and d.reason == "below_score"
        # the whole point: the loser is reported, not thrown away
        assert d.mode == "PLAY_MUSIC"
        assert 0.0 < d.score < 0.98
        assert d.example in ("toca musica", "play song")

    def test_decline_below_margin_is_distinguishable(self):
        ambiguous = (
            IntentExamples("MODE_A", ("alpha",)),
            IntentExamples("MODE_B", ("alpha",)),
        )
        r = SemanticRouter(fake_embed, catalog=ambiguous,
                           min_score=0.4, min_margin=0.05)
        d = r.explain("alpha")
        # confident but ambiguous — a DIFFERENT fix from raising the floor
        assert d.accepted is False and d.reason == "below_margin"
        assert d.score > 0.9 and d.margin < 0.05

    def test_score_failure_wins_when_both_fail(self):
        # Both thresholds unmet → report 'below_score': raising the floor
        # subsumes it, so that is the primary fact for calibration.
        ambiguous = (
            IntentExamples("MODE_A", ("alpha",)),
            IntentExamples("MODE_B", ("alpha",)),
        )
        r = SemanticRouter(fake_embed, catalog=ambiguous,
                           min_score=0.99, min_margin=0.5)
        assert r.explain("alpha beta").reason == "below_score"

    def test_near_miss_is_distinguishable_from_a_random_query(self):
        # THE calibration case. Two declines, same reason, very different
        # meaning: one the router should rescue (floor is too high), one it
        # correctly ignored. Only the score tells them apart.
        loose = SemanticRouter(fake_embed, catalog=CATALOG,
                               min_score=0.0, min_margin=0.0)
        good = loose.explain("quero musica")   # real intent, partial overlap
        junk = loose.explain("xyzzy plugh frobnicate")  # genuine chat

        # Put the floor just above the real intent → it becomes a near-miss.
        r = SemanticRouter(fake_embed, catalog=CATALOG, min_margin=0.0,
                           min_score=good.score + 0.01)
        d_good, d_junk = r.explain("quero musica"), r.explain(
            "xyzzy plugh frobnicate")
        assert d_good.reason == d_junk.reason == "below_score"
        # ...and the log can still separate them, which is the whole point
        assert d_good.score > junk.score + 0.2
        assert d_good.mode == "PLAY_MUSIC"

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_has_no_candidate(self, router, query):
        d = router.explain(query)
        assert d.accepted is False and d.reason == "no_candidate"
        assert d.mode is None and d.example is None and d.score == 0.0

    def test_empty_catalog_has_no_candidate(self):
        d = SemanticRouter(fake_embed, catalog=()).explain("anything")
        assert d.reason == "no_candidate" and d.mode is None

    def test_embed_failure_is_reported_not_raised(self, router):
        router._embed = lambda t: (_ for _ in ()).throw(RuntimeError("down"))
        d = router.explain("toca musica")
        assert d.accepted is False and d.reason == "embed_failed"

    def test_dim_mismatch_is_reported_not_raised(self):
        r = SemanticRouter(fake_embed, catalog=CATALOG)
        r._embed = lambda t: np.ones(8, dtype=np.float32)
        d = r.explain("toca musica")
        assert d.accepted is False and d.reason == "dim_mismatch"

    def test_as_match_is_none_when_declined(self, router):
        assert router.explain("xyzzy plugh frobnicate").as_match() is None

    def test_explain_costs_one_embed(self, router):
        calls = {"n": 0}
        real = router._embed

        def counting(text):
            calls["n"] += 1
            return real(text)

        router._embed = counting
        router.explain("toca musica")
        assert calls["n"] == 1  # shadow must not cost more than route() did


class TestFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER", raising=False)
        assert _semantic_router_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
    def test_on_values(self, monkeypatch, val):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER", val)
        assert _semantic_router_enabled() is True


class TestTheDefaultFloorIsTheCalibratedOne:
    """The flag alone installs the floor the router was calibrated at (s83,
    0.74). It installed 0.62, the pre-calibration guess, and only .env's
    CORTEX_SEMANTIC_MIN_SCORE made production right — measured with the real
    bge-m3, "pode parar" scores SPOTIFY_PAUSE 0.657 (s88, Codex lote 8).

    In a fresh interpreter: this process has loaded .env already."""

    def test_without_the_env_var(self):
        import os
        import subprocess
        import sys
        env = {k: v for k, v in os.environ.items()
               if k != "CORTEX_SEMANTIC_MIN_SCORE"}
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, "-c",
             "import src.core.semantic_router as s; print(s._DEFAULT_MIN_SCORE)"],
            cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert out.stdout.strip() == "0.74", out.stderr
