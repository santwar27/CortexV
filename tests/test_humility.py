"""
Tests for the Onda 3c humility layer — the volatile-fact detector wired into
_needs_realtime so fresh-fact questions route to the search offer instead of
a confabulated 8B answer.

The regex itself is locked by evals/cases/humility.py (REGEX cases). Here we
test the INTEGRATION inside _needs_realtime: the volatile fast-path returns
True without an LLM call, the offline-prefix fast-path still wins, and the
CORTEX_HUMILITY kill-switch falls through to the LLM classifier.
"""
from collections import OrderedDict

import pytest

import src.core.orchestrator as orch_mod
from src.core.dispatch.heuristics import VOLATILE_FACT_RE


def _orch():
    o = orch_mod.CortexOrchestrator.__new__(orch_mod.CortexOrchestrator)
    o._intent_cache = OrderedDict()
    # Stub engine so the LLM-classifier branch (only reached when humility is
    # off / not volatile) can build its prompt without a real engine.
    o.engine_analytical = type("E", (), {"model_name": "test-model"})()
    return o


# --- regex sanity (full lock is in the eval suite) -------------------------

def test_regex_matches_volatile():
    assert VOLATILE_FACT_RE.search("quem ganhou o jogo")
    assert VOLATILE_FACT_RE.search("qual a cotação do dólar")


def test_regex_matches_last_game_result():
    # Live miss (s80, the user): "resultado do ÚLTIMO jogo do flamengo" slipped
    # past the "resultado do jogo" branch because the adjective split it, so
    # the 8B classifier fabricated a score (real opponent: Chapecoense).
    assert VOLATILE_FACT_RE.search("qual foi o resultado do último jogo do flamengo?")
    assert VOLATILE_FACT_RE.search("quando é o próximo jogo do flamengo")
    assert VOLATILE_FACT_RE.search("resultado da última corrida")


def test_regex_ignores_stable():
    assert not VOLATILE_FACT_RE.search("quem foi Alan Turing")
    assert not VOLATILE_FACT_RE.search("qual a capital da França")
    # definitional/offline uses of "jogo" must NOT be flagged as volatile
    assert not VOLATILE_FACT_RE.search("o que é um jogo de tabuleiro")
    assert not VOLATILE_FACT_RE.search("explica as regras do jogo da velha")


# --- _needs_realtime integration -------------------------------------------

def test_volatile_returns_true_without_llm(monkeypatch):
    # If the LLM classifier were reached it would fail — proving the volatile
    # fast-path short-circuited to True first (and saved the call).
    monkeypatch.setattr(
        orch_mod.ollama, "generate",
        lambda **k: pytest.fail("LLM classifier should not be called for a volatile match"),
    )
    o = _orch()
    assert o._needs_realtime("quem ganhou o jogo do Flamengo") is True


def test_volatile_result_is_cached(monkeypatch):
    monkeypatch.setattr(orch_mod.ollama, "generate",
                        lambda **k: pytest.fail("must use cache / fast-path"))
    o = _orch()
    o._needs_realtime("quanto está o bitcoin")
    assert o._intent_cache["quanto está o bitcoin"] is True


def test_offline_prefix_beats_volatile(monkeypatch):
    # "o que é placar" contains 'placar' but the offline (definitional)
    # fast-path runs first → stays offline, no search offer.
    monkeypatch.setattr(orch_mod.ollama, "generate", lambda **k: {"response": "NO"})
    o = _orch()
    assert o._needs_realtime("o que é placar no futebol") is False


def test_kill_switch_falls_through_to_llm(monkeypatch):
    monkeypatch.setenv("CORTEX_HUMILITY", "0")
    called = {}

    def _gen(**k):
        called["llm"] = True
        return {"response": "NO"}

    monkeypatch.setattr(orch_mod.ollama, "generate", _gen)
    o = _orch()
    # With humility off, a volatile query no longer short-circuits — it goes
    # to the LLM classifier (here stubbed to NO).
    assert o._needs_realtime("quem ganhou o jogo") is False
    assert called.get("llm") is True


def test_stable_question_not_flagged(monkeypatch):
    monkeypatch.setattr(orch_mod.ollama, "generate", lambda **k: {"response": "NO"})
    o = _orch()
    assert o._needs_realtime("qual a capital da França") is False


def test_humility_enabled_default_on(monkeypatch):
    monkeypatch.delenv("CORTEX_HUMILITY", raising=False)
    assert orch_mod.CortexOrchestrator._humility_enabled() is True


@pytest.mark.parametrize("val,expected", [
    ("0", False), ("false", False), ("no", False), ("", False),
    ("1", True), ("yes", True),
])
def test_humility_flag_values(monkeypatch, val, expected):
    monkeypatch.setenv("CORTEX_HUMILITY", val)
    assert orch_mod.CortexOrchestrator._humility_enabled() is expected
