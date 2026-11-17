"""
Prompt enrichment v1 (idea #30, s74).

What this pins, per the design Pedro approved:
  - opt-in only: the regex demands the explicit prefix + separator;
    normal sentences starting with "refina" don't trigger.
  - the expanded prompt is ALWAYS staged (sim/não) — never executed
    directly by the handler.
  - "sim" re-enters the FULL pipeline (_process_core) with the staged
    prompt and the staged language — refine is never a gate bypass.
  - LLM failure → honest error, nothing staged (no silent fall-through
    to free chat about "refina: X").
  - SOBERANIA: the meta-prompt carries only the user's request text.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.dispatch.refine import REFINE_RE
from src.core.i18n import I18nMixin
from src.core.refine_mixin import RefineMixin


class TestRefineRegex:
    @pytest.mark.parametrize("text,body", [
        ("refina: me ajuda com treino", "me ajuda com treino"),
        ("Refina, escreve um email pro meu chefe",
         "escreve um email pro meu chefe"),
        ("refine: help me plan my week", "help me plan my week"),
        ("melhora o prompt: organiza minhas notas",
         "organiza minhas notas"),
        ("enriquece o prompt: resumo do livro", "resumo do livro"),
    ])
    def test_matches_with_separator(self, text, body):
        m = REFINE_RE.match(text)
        assert m and m.group("request") == body

    @pytest.mark.parametrize("text", [
        "refina os detalhes da viagem",      # no separator → normal sentence
        "refinaria de petróleo no Brasil",   # different word
        "a refinação do açúcar",             # not anchored at start
        "refina:",                           # nothing after the prefix
        "me ajuda com treino",               # no prefix at all
    ])
    def test_normal_sentences_do_not_trigger(self, text):
        m = REFINE_RE.match(text)
        assert m is None or not m.group("request").strip()


def _host(monkeypatch, llm_reply=None, llm_exc=None):
    """Minimal RefineMixin host (the weather-test pattern)."""
    class O(RefineMixin, I18nMixin):
        pass
    o = O()
    o._reply_lang = "pt"
    o.engine_analytical = SimpleNamespace(model_name="test-model")
    o._pending_action = None
    o._persist_pending_action = MagicMock()
    calls = {}

    def fake_chat(**kw):
        calls.update(kw)
        if llm_exc is not None:
            raise llm_exc
        return {"message": {"content": llm_reply}}
    monkeypatch.setattr("src.core.refine_mixin.ollama.chat", fake_chat)
    return o, calls


class TestHandleRefine:
    def test_stages_the_expanded_prompt(self, monkeypatch):
        o, calls = _host(monkeypatch,
                         llm_reply="Monte um plano de treino semanal ...")
        reply, mode = o._handle_refine("me ajuda com treino")
        assert mode == "REFINE_STAGED"
        assert o._pending_action == {
            "type": "run_refined_prompt",
            "args": {"prompt": "Monte um plano de treino semanal ...",
                     "lang": "pt"},
        }
        o._persist_pending_action.assert_called_once()
        # The user must SEE what they're approving.
        assert "Monte um plano de treino semanal" in reply
        assert "sim" in reply and "não" in reply

    def test_meta_prompt_carries_only_the_request(self, monkeypatch):
        # SOBERANIA regra 1: no Palace/memory in the enrichment call —
        # the user content is the raw request, nothing else rides along.
        o, calls = _host(monkeypatch, llm_reply="x")
        o._handle_refine("me ajuda com treino")
        user_msgs = [m for m in calls["messages"] if m["role"] == "user"]
        assert user_msgs == [{"role": "user",
                              "content": "me ajuda com treino"}]
        assert calls["options"] == {"temperature": 0.0}

    def test_empty_request_falls_through(self, monkeypatch):
        o, _ = _host(monkeypatch, llm_reply="x")
        assert o._handle_refine("   ") is None
        assert o._pending_action is None

    def test_llm_failure_is_honest_and_stages_nothing(self, monkeypatch):
        o, _ = _host(monkeypatch, llm_exc=RuntimeError("ollama down"))
        reply, mode = o._handle_refine("me ajuda com treino")
        assert mode == "REFINE_ERROR"
        assert o._pending_action is None
        assert "sem o prefixo" in reply  # tells the user the way around

    def test_empty_llm_reply_is_an_error_too(self, monkeypatch):
        o, _ = _host(monkeypatch, llm_reply="   ")
        reply, mode = o._handle_refine("me ajuda com treino")
        assert mode == "REFINE_ERROR"
        assert o._pending_action is None


class TestApprovalRunsFullPipeline:
    def _orch(self):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = {
            "type": "run_refined_prompt",
            "args": {"prompt": "Monte um plano de treino semanal",
                     "lang": "pt"},
        }
        o._persist_pending_action = MagicMock()
        o._reply_lang = "pt"
        return o

    def test_sim_reenters_process_core_with_staged_prompt(self):
        o = self._orch()
        o._process_core = MagicMock(return_value=("feito", "OFFLINE"))
        reply, mode = o._execute_pending_action()
        # Slot cleared BEFORE the re-entry, so the inner turn can stage
        # its own S3 — refine must never bypass the gate.
        assert o._pending_action is None
        o._process_core.assert_called_once_with(
            "Monte um plano de treino semanal", language="pt")
        assert (reply, mode) == ("feito", "OFFLINE")

    def test_nao_cancels_without_executing(self):
        o = self._orch()
        o._process_core = MagicMock(
            side_effect=AssertionError("não must not execute"))
        reply, mode = o._cancel_pending_action()
        assert o._pending_action is None
        assert mode == "AGENT_ACTION_CANCELLED"
