"""
Tests for src/core/arbiter.py — CognitiveArbiter Layer-1 router.

Checks:
- Each Route fires on representative inputs at sensible confidence
- REFLEX delegates to the reflex layer (the two never disagree)
- LONGO is driven by INPUT size, not topic
- The bare fallback is SIMPLES at low confidence (the Layer-2 wake signal)
- classify() is pure: Layer 2 stays dormant unless CORTEX_ARBITER_CASCADE is on
- model_hint() maps every route
"""
import os
from unittest.mock import patch

import pytest

from src.core.arbiter import (
    ArbiterDecision,
    CognitiveArbiter,
    LONG_INPUT_CHARS,
    LONG_INPUT_WORDS,
    LOW_CONFIDENCE_THRESHOLD,
    Route,
    ROUTE_MODEL_HINT,
    arbiter,
)


@pytest.fixture
def ab():
    return CognitiveArbiter()


class TestReflexRoute:
    """Trivial inputs route to REFLEX via the shared reflex layer."""

    @pytest.mark.parametrize("text", [
        "oi", "bom dia", "que horas são", "obrigado", "ping",
        "what time is it", "hello",
    ])
    def test_reflex_hits(self, ab, text):
        d = ab.classify(text)
        assert d.route is Route.REFLEX
        assert d.confidence == 1.0

    def test_agrees_with_reflex_layer(self, ab):
        # Anything the reflex layer handles must be classified REFLEX.
        from src.core.reflex import reflex_layer
        for text in ("oi", "que horas são", "boa noite", "status"):
            assert reflex_layer.is_hit(text)
            assert ab.classify(text).route is Route.REFLEX


class TestLookupRoute:
    @pytest.mark.parametrize("text", [
        "qual o preço do bitcoin",
        "quanto custa um iphone 15",
        "me dá as últimas notícias",
        "qual a previsão do tempo em São Paulo hoje",
        "quem ganhou o jogo do Flamengo",
        "pesquisa na internet sobre fusão nuclear",
        "search for the best pizza",
        "abre https://example.com pra mim",
    ])
    def test_lookup_hits(self, ab, text):
        d = ab.classify(text)
        assert d.route is Route.LOOKUP, f"{text!r} -> {d}"


class TestMemoryRoute:
    @pytest.mark.parametrize("text", [
        "lembra do meu aniversário",
        "qual é o meu nome",
        "qual a minha cor favorita",
        "o que eu te disse ontem",
        "onde eu moro mesmo",
        "what did i tell you about my dog",
    ])
    def test_memory_hits(self, ab, text):
        d = ab.classify(text)
        assert d.route is Route.MEMORY, f"{text!r} -> {d}"


class TestComplexoRoute:
    @pytest.mark.parametrize("text", [
        "explique como o TCP estabelece conexão",
        "compare REST e GraphQL",
        "analise os prós e contras disso",
        "por que o céu é azul",
        "como funciona um motor a combustão",
        "escreve uma função python que ordena uma lista",
        "explain how transformers work",
    ])
    def test_complexo_hits(self, ab, text):
        d = ab.classify(text)
        assert d.route is Route.COMPLEXO, f"{text!r} -> {d}"


class TestCrossTablePrecedence:
    """Route is chosen by highest confidence ACROSS tables, not table order."""

    def test_higher_confidence_memory_beats_lower_lookup(self, ab):
        # "lembra" (MEMORY recall, 0.88) outranks "search" (LOOKUP, 0.82)
        # even though LOOKUP is the first table checked.
        d = ab.classify("lembra do meu search")
        assert d.route is Route.MEMORY
        assert d.confidence == 0.88

    def test_genuinely_stronger_lookup_still_wins(self, ab):
        # When LOOKUP has the stronger marker it still wins — the rule is
        # max-confidence, not "MEMORY always beats LOOKUP".
        d = ab.classify("qual o preço que você lembra")  # preço 0.9 > lembra 0.88
        assert d.route is Route.LOOKUP


class TestLongoRoute:
    def test_long_by_words(self, ab):
        text = "palavra " * (LONG_INPUT_WORDS + 5)
        d = ab.classify(text)
        assert d.route is Route.LONGO
        assert d.confidence >= 0.9

    def test_long_by_chars(self, ab):
        # A single long token: few words but past the char cap.
        text = "x" * (LONG_INPUT_CHARS + 10)
        assert ab.classify(text).route is Route.LONGO

    def test_long_beats_topic(self, ab):
        # A big paste that contains a COMPLEXO marker still routes by SIZE.
        text = "explique isto: " + ("contexto " * (LONG_INPUT_WORDS + 5))
        assert ab.classify(text).route is Route.LONGO

    def test_short_complex_is_not_longo(self, ab):
        # The reasoning verb on a short query stays COMPLEXO, not LONGO.
        assert ab.classify("explique a relatividade").route is Route.COMPLEXO


class TestSimplesFallback:
    @pytest.mark.parametrize("text", [
        "qual a capital da frança",
        "quem foi Napoleão",
        "conte uma curiosidade",
    ])
    def test_fallback_is_simples_low_conf(self, ab, text):
        d = ab.classify(text)
        assert d.route is Route.SIMPLES
        assert d.confidence < LOW_CONFIDENCE_THRESHOLD
        assert d.layer == "fallback"

    def test_empty_input(self, ab):
        d = ab.classify("")
        assert d.route is Route.SIMPLES
        assert d.confidence == 0.0
        assert d.layer == "fallback"

    def test_whitespace_only(self, ab):
        assert ab.classify("   ").route is Route.SIMPLES


class TestLayer2Dormant:
    """Layer 2 must never fire unless the flag is explicitly on."""

    def test_tiebreak_returns_none_by_default(self, ab):
        # Even on a low-confidence call, the tie-break is a no-op while off.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_ARBITER_CASCADE", None)
            d = ab.classify("qual a capital da frança")
            assert d.route is Route.SIMPLES
            assert d.layer == "fallback"  # cascade result, untouched

    def test_tiebreak_stub_is_noop_even_when_enabled(self, ab):
        # With the flag ON the stub is consulted but still returns None until
        # the real mini-model is wired on Mac Studio → cascade result stands.
        with patch.dict(os.environ, {"CORTEX_ARBITER_CASCADE": "1"}):
            d = ab.classify("qual a capital da frança")
            assert d.route is Route.SIMPLES

    def test_high_confidence_never_calls_tiebreak(self, ab):
        # A confident route must not even consult Layer 2, flag on or off.
        with patch.dict(os.environ, {"CORTEX_ARBITER_CASCADE": "1"}):
            with patch.object(ab, "_llm_tiebreak", side_effect=AssertionError(
                    "tie-break must not run on a confident route")):
                assert ab.classify("qual o preço do ouro").route is Route.LOOKUP


class TestPurityAndContract:
    def test_classify_is_idempotent(self, ab):
        text = "explique a fotossíntese"
        assert ab.classify(text) == ab.classify(text)

    def test_decision_is_frozen(self):
        d = ArbiterDecision(Route.SIMPLES, 0.5, "x")
        with pytest.raises(Exception):
            d.route = Route.LOOKUP  # type: ignore[misc]

    def test_route_is_str_valued(self):
        # str-Enum so it logs/serializes as plain text.
        assert Route.LOOKUP == "LOOKUP"
        assert ArbiterDecision(Route.LOOKUP, 0.9, "x").route.value == "LOOKUP"

    def test_model_hint_covers_every_route(self, ab):
        for route in Route:
            assert route in ROUTE_MODEL_HINT
            # model_hint returns the mapped tier (None only for REFLEX).
            hint = ab.model_hint(route)
            if route is Route.REFLEX:
                assert hint is None
            else:
                assert isinstance(hint, str)

    def test_singleton_exists(self):
        assert isinstance(arbiter, CognitiveArbiter)
