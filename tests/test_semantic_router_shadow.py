"""
Shadow wiring for the SemanticRouter (item 1b of m1_interpretation_plan.md):
the orchestrator observes what the router WOULD route, on LLM-bound queries,
without routing. Off unless CORTEX_SEMANTIC_ROUTER_SHADOW.

Built via __new__ (like the other orchestrator tests) so no Ollama/gatekeeper
boot; _semantic_router is monkeypatched to a fake so nothing embeds for real.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core.orchestrator import CortexOrchestrator
from src.core.semantic_router import (
    RouterDecision,
    _semantic_router_shadow_enabled,
)


def _accepted(mode="STUDIER_LIST", score=0.90, margin=0.18,
              example="me mostra minha lista de estudo") -> RouterDecision:
    return RouterDecision(mode=mode, score=score, margin=margin,
                          example=example, accepted=True, reason=None)


def _declined(mode="WEATHER", score=0.61, margin=0.09,
              example="vai chover amanhã",
              reason="below_score") -> RouterDecision:
    return RouterDecision(mode=mode, score=score, margin=margin,
                          example=example, accepted=False, reason=reason)


@pytest.fixture
def orch(monkeypatch):
    # The shadow stands down when the router is LIVE (s83) — it would embed the
    # same sentence twice. These tests are about the shadow itself, so the live
    # flag is cleared here instead of depending on whatever .env says.
    monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER", raising=False)
    return CortexOrchestrator.__new__(CortexOrchestrator)


class TestShadowFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER_SHADOW", raising=False)
        assert _semantic_router_shadow_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "ON"])
    def test_on_values(self, monkeypatch, val):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", val)
        assert _semantic_router_shadow_enabled() is True


class TestShadowRoute:
    def test_off_does_nothing(self, orch, monkeypatch):
        monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER_SHADOW", raising=False)
        built = MagicMock()
        monkeypatch.setattr(orch, "_semantic_router", built)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("me mostra a fila de estudo")
        built.assert_not_called()   # never even builds the router
        logged.assert_not_called()

    def test_on_logs_would_route_on_match(self, orch, monkeypatch):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.return_value = _accepted()
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("me de a lista de estudo")
        assert logged.call_count == 1
        args, kwargs = logged.call_args
        assert args[1] == "semantic_router" and args[2] == "would_route"
        assert kwargs["payload"]["mode"] == "STUDIER_LIST"
        assert kwargs["payload"]["score"] == 0.9
        assert "reason" not in kwargs["payload"]  # nothing was rejected

    def test_on_logs_declined_on_no_match(self, orch, monkeypatch):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.return_value = _declined()
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("qual a capital da frança")
        assert logged.call_count == 1
        args, _ = logged.call_args
        assert args[1] == "semantic_router" and args[2] == "declined"

    def test_decline_log_carries_the_rejected_candidate(self, orch, monkeypatch):
        # The calibration payload (item 1c): a decline must say WHAT it
        # rejected and by how much, or near-misses are invisible in the audit.
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.return_value = _declined(
            mode="JOBS_TOP", score=0.615, margin=0.12,
            example="tem vaga nova pra mim", reason="below_score",
        )
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("rolou alguma coisa nova de emprego")
        payload = logged.call_args.kwargs["payload"]
        assert payload["mode"] == "JOBS_TOP"
        assert payload["score"] == 0.615 and payload["margin"] == 0.12
        assert payload["reason"] == "below_score"
        assert payload["example"] == "tem vaga nova pra mim"

    def test_decline_with_no_candidate_logs_nulls(self, orch, monkeypatch):
        # Empty query / dead embedder: nothing was scored. Must not invent a
        # mode, and must not crash the log.
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.return_value = RouterDecision(
            mode=None, score=0.0, margin=0.0, example=None,
            accepted=False, reason="embed_failed",
        )
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("qualquer coisa")
        payload = logged.call_args.kwargs["payload"]
        assert payload["mode"] is None and payload["example"] is None
        assert payload["reason"] == "embed_failed"

    def test_router_exception_is_swallowed(self, orch, monkeypatch):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.side_effect = RuntimeError("embed down")
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log"):
            orch._shadow_semantic_route("anything")  # must not raise

    def test_shadow_never_routes(self, orch, monkeypatch):
        # The contract: shadow only LOGS. It returns None and changes nothing.
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        fake = MagicMock()
        fake.explain.return_value = _accepted(mode="JOBS_TOP")
        monkeypatch.setattr(orch, "_semantic_router", lambda: fake)
        with patch("src.core.orchestrator.audit.log"):
            assert orch._shadow_semantic_route("tem vaga") is None
        fake.route.assert_not_called()  # explain() observes; route() acts


class TestShadowStandsDownWhenLive:
    """With the router live, _semantic_route_live scores the same sentence and
    logs the same facts; a second embed is ~180ms of nothing (s83)."""

    def test_live_router_silences_the_shadow(self, orch, monkeypatch):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER", "1")
        called = []
        monkeypatch.setattr(orch, "_semantic_router",
                            lambda: called.append(1), raising=False)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("tem vaga nova?")
        assert called == [] and logged.call_count == 0

    def test_without_the_live_flag_the_shadow_still_logs(self, orch, monkeypatch):
        monkeypatch.setenv("CORTEX_SEMANTIC_ROUTER_SHADOW", "1")
        monkeypatch.delenv("CORTEX_SEMANTIC_ROUTER", raising=False)
        monkeypatch.setattr(
            orch, "_semantic_router",
            lambda: MagicMock(explain=MagicMock(return_value=_accepted())),
            raising=False)
        with patch("src.core.orchestrator.audit.log") as logged:
            orch._shadow_semantic_route("me mostra meus estudos")
        assert logged.call_count == 1
