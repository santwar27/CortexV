"""
Tests for NightlyStudier._apply_critic — the post-synthesis sceptical
fact-checker loop.

All tests mock ollama.generate so no local LLM is needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core.studier import NightlyStudier


CLAIM = "Todos os pedidos são sempre processados sem falhas."
EXPLANATION = "A afirmação promete ausência de falhas sem apresentar evidência."
PROBLEM = f"PROBLEMA | absoluta | {CLAIM} | {EXPLANATION}"
NO_PROOF = "O crítico não encontrou nenhum problema comprovado."


@pytest.fixture
def studier():
    """NightlyStudier instance with a minimal engine stub."""
    s = NightlyStudier.__new__(NightlyStudier)
    s.engine = MagicMock()
    s.engine.model_name = "test-model"
    return s


def _briefing(verdict_line="alta — ótimas fontes.", urls=None):
    """Build a minimal briefing in the expected format."""
    if urls is None:
        urls = [
            "https://arxiv.org/abs/1",
            "https://huggingface.co/docs",
            "https://pytorch.org/docs",
        ]
    fontes = "\n".join(f"- [{u}]" for u in urls)
    return (
        f"## Resumo executivo\n{CLAIM}\n\n"
        "## Pontos-chave\n- a\n- b\n\n"
        f"## Fontes consultadas\n{fontes}\n\n"
        f"## Confiança\n{verdict_line}\n"
    )


def _mock_ollama(verdict: str):
    """Return a patch target that makes ollama.generate return verdict."""
    mock = MagicMock()
    mock.return_value = {"response": verdict}
    return mock


class TestCriticOK:

    def test_ok_appends_critico_section(self, studier):
        with patch("src.core.studier.ollama.generate", _mock_ollama("OK | briefing passes review")):
            result = studier._apply_critic("RAG", _briefing())
        assert "## Crítico" in result
        assert NO_PROOF in result

    def test_ok_does_not_change_confianca(self, studier):
        briefing = _briefing("alta — ótimas fontes.")
        with patch("src.core.studier.ollama.generate", _mock_ollama("OK | all good")):
            result = studier._apply_critic("RAG", briefing)
        assert "rebaixado pelo crítico" not in result
        assert "alta — ótimas fontes." in result

    def test_ok_case_insensitive(self, studier):
        with patch("src.core.studier.ollama.generate", _mock_ollama("ok | looks fine")):
            result = studier._apply_critic("topic", _briefing())
        assert "## Crítico" in result


class TestCriticProblema:

    def test_problema_appends_critico_section(self, studier):
        verdict = PROBLEM
        with patch("src.core.studier.ollama.generate", _mock_ollama(verdict)):
            result = studier._apply_critic("RAG", _briefing())
        assert "## Crítico" in result
        assert f"- (absoluta) «{CLAIM}» — {EXPLANATION}" in result

    def test_problema_forces_confianca_baixa(self, studier):
        verdict = PROBLEM
        briefing = _briefing("alta — fontes excelentes.")
        with patch("src.core.studier.ollama.generate", _mock_ollama(verdict)):
            result = studier._apply_critic("RAG", briefing)
        assert "baixa — o crítico achou problemas (ver ## Crítico)." in result
        assert "fontes excelentes" not in result
        # 'alta' in the Confiança section must be gone
        import re
        confianca_block = re.search(r"## Confiança\n(.+)", result)
        assert confianca_block is not None
        assert "alta" not in confianca_block.group(1)
        assert "baixa" in confianca_block.group(1)

    def test_problema_already_baixa_not_doubled(self, studier):
        """If confiança was already baixa, critic just adds the section — no double-rewrite."""
        verdict = PROBLEM
        briefing = _briefing("baixa — fontes fracas.")
        with patch("src.core.studier.ollama.generate", _mock_ollama(verdict)):
            result = studier._apply_critic("topic", briefing)
        assert "## Crítico" in result
        # baixa should still be there
        assert "baixa" in result

    def test_problema_case_insensitive(self, studier):
        with patch("src.core.studier.ollama.generate", _mock_ollama(
                PROBLEM.replace("PROBLEMA", "problema"))):
            result = studier._apply_critic("topic", _briefing())
        assert "baixa — o crítico achou problemas (ver ## Crítico)." in result


class TestCriticFailSafe:

    def test_ollama_exception_returns_original_briefing(self, studier):
        """If Ollama crashes, _apply_critic returns the briefing unchanged."""
        original = _briefing()
        with patch("src.core.studier.ollama.generate", side_effect=RuntimeError("timeout")):
            result = studier._apply_critic("RAG", original)
        assert result == original
        assert "## Crítico" not in result

    def test_empty_response_reports_no_proven_problem(self, studier):
        with patch("src.core.studier.ollama.generate", _mock_ollama("")):
            result = studier._apply_critic("RAG", _briefing())
        # Empty verdict line → no PROBLEMA → no confidence rewrite
        assert "rebaixado pelo crítico" not in result
        assert NO_PROOF in result

    def test_multiline_garbage_is_ignored(self, studier):
        """Every line is checked; raw garbage never reaches the study."""
        multi = "OK | briefing passes\nExtra garbage line\nAnother line"
        with patch("src.core.studier.ollama.generate", _mock_ollama(multi)):
            result = studier._apply_critic("RAG", _briefing())
        assert f"## Crítico\n{NO_PROOF}\n" in result
        assert "Extra garbage line" not in result


class TestCriticIntegration:

    def test_critico_section_at_end_of_briefing(self, studier):
        """The critic section must be the LAST section in the briefing."""
        with patch("src.core.studier.ollama.generate", _mock_ollama("OK | all good")):
            result = studier._apply_critic("topic", _briefing())
        # After '## Crítico\n<verdict>\n' there should be nothing else
        idx = result.rfind("## Crítico")
        after = result[idx:].strip()
        lines = after.splitlines()
        assert lines[0] == "## Crítico"
        assert lines[1] == NO_PROOF

    def test_briefing_without_confianca_block_not_broken(self, studier):
        """A briefing that lacks the Confiança section must not crash."""
        bare = "## Resumo executivo\nFoo.\n\n## Fontes consultadas\n- [https://a.com]\n"
        with patch("src.core.studier.ollama.generate", _mock_ollama("PROBLEMA | missing section")):
            result = studier._apply_critic("topic", bare)
        # Should not raise; critic section still appended
        assert "## Crítico" in result
