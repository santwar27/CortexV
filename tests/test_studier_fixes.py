"""
Tests for the three NightlyStudier bug fixes shipped 2026-05-21:

  (a) query factuality filter — reject acronym hallucinations
  (c) DDG URL passthrough — include real URL in fallback output
  (d) confidence floor — downgrade verdict by unique source count

Pure logic only, no Ollama / Tavily / live DDG calls. The DDG path is
exercised via unittest.mock.patch over the ddgs context manager.
"""
from unittest.mock import patch, MagicMock

import pytest

from src.core.studier import NightlyStudier
from src.tools.web_search import WebSearcher


# --- Bug (a) --- query factuality filter ----------------------------------

class TestQueryFactualityFilter:
    """_query_looks_hallucinated catches fabricated acronym expansions."""

    @pytest.mark.parametrize("dirty", [
        "RAG = Regressive Attention Generator basics",
        "LLM = Large Linguistic Machine intro",
        "RAG stands for Regressive Attention Generator",
        "GQA means Glass Qualifications Authority",
        "MQA is short for Multiple Query Authority",
        "RAG short for Random Augmented Graph",
        "RAG (Regressive Attention Generator) explained",
        "GQA (Glass Qualifications Authority) overview",
        "MHA (Multi Head Attribution) papers",
    ])
    def test_acronym_expansion_rejected(self, dirty):
        assert NightlyStudier._query_looks_hallucinated(dirty) is True

    @pytest.mark.parametrize("clean", [
        "RAG vs fine-tuning comparison",
        "attention mechanism transformer LLM",
        "Llama 3.1 8B quantization benchmarks",
        "Qwen 2.5 Coder vs DeepSeek Coder",
        "RAG retrieval augmented generation tutorial",
        # Real-world queries shouldn't trip the filter even when they
        # mention several capitalised technical terms.
        "PyTorch vs TensorFlow attention head implementations",
        # Parentheses around a single short token are common and fine.
        "RAG (overview)",
    ])
    def test_clean_query_kept(self, clean):
        assert NightlyStudier._query_looks_hallucinated(clean) is False


# --- Bug (c) --- DDG URL passthrough --------------------------------------

class TestDDGUrlPassthrough:
    """_search_ddg emits 'href' in the same [url] bracket Tavily uses."""

    def _mock_ddgs(self, results):
        """Build a context-manager-compatible mock around ddgs.DDGS."""
        m = MagicMock()
        m.text.return_value = results
        cm = MagicMock()
        cm.__enter__.return_value = m
        cm.__exit__.return_value = False
        return cm

    def test_ddg_format_includes_url(self):
        results = [
            {"title": "Wiki RAG", "body": "RAG is retrieval augmented generation",
             "href": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"},
        ]
        searcher = WebSearcher()
        with patch("src.tools.web_search.DDGS", return_value=self._mock_ddgs(results)):
            out = searcher._search_ddg("rag", max_results=1)

        assert "Wiki RAG" in out
        assert "retrieval augmented generation" in out
        assert "[https://en.wikipedia.org/wiki/Retrieval-augmented_generation]" in out

    def test_ddg_format_handles_missing_url(self):
        results = [
            {"title": "Untitled", "body": "some body"},  # no 'href' key
        ]
        searcher = WebSearcher()
        with patch("src.tools.web_search.DDGS", return_value=self._mock_ddgs(results)):
            out = searcher._search_ddg("anything", max_results=1)

        # Empty brackets are fine — the synthesiser can see there was no URL
        # and won't hallucinate. Crucially: no crash.
        assert "[]" in out
        assert "some body" in out

    def test_ddg_empty_results_returns_system_notice(self):
        searcher = WebSearcher()
        with patch("src.tools.web_search.DDGS", return_value=self._mock_ddgs([])):
            out = searcher._search_ddg("nothing", max_results=3)
        assert "SYSTEM NOTICE" in out


# --- Bug (d) --- confidence floor -----------------------------------------

def _briefing(verdict: str, urls):
    """Build a minimal briefing in the expected synth format."""
    fontes = "\n".join(f"- [{u}]" for u in urls)
    return (
        "## Resumo executivo\nfoo bar.\n\n"
        "## Pontos-chave\n- a\n- b\n\n"
        f"## Fontes consultadas\n{fontes}\n\n"
        f"## Confiança\n{verdict} — placeholder.\n"
    )


class TestConfidenceFloor:
    """_enforce_confidence_floor rebaixa verdicts inflated by Llama 8B."""

    def test_three_unique_domains_alta_kept(self):
        b = _briefing("alta", [
            "https://arxiv.org/abs/1",
            "https://huggingface.co/docs",
            "https://pytorch.org/docs",
        ])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "alta" in out
        assert "rebaixado" not in out

    def test_two_unique_domains_alta_downgraded_to_media(self):
        b = _briefing("alta", [
            "https://arxiv.org/abs/1",
            "https://huggingface.co/docs",
        ])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "média" in out
        assert "média — poucas fontes independentes (2 domínio(s))." in out
        assert "placeholder" not in out
        # The original 'alta' verdict line should be gone (only the prompt
        # criteria mentions alta — outside the Confiança block).
        assert "Confiança\nalta" not in out

    def test_one_unique_domain_alta_downgraded_to_baixa(self):
        b = _briefing("alta", [
            "https://arxiv.org/abs/1",
            "https://arxiv.org/abs/2",  # same domain
        ])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "baixa" in out
        assert "baixa — poucas fontes independentes (1 domínio(s))." in out
        assert "placeholder" not in out

    def test_zero_urls_alta_downgraded_to_baixa(self):
        b = _briefing("alta", [])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "baixa" in out

    def test_two_domains_media_kept(self):
        """Already at média with 2 domains — verdict matches floor, no change."""
        b = _briefing("média", [
            "https://arxiv.org/abs/1",
            "https://huggingface.co/docs",
        ])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "média" in out
        assert "rebaixado" not in out

    def test_one_domain_media_downgraded_to_baixa(self):
        b = _briefing("média", ["https://arxiv.org/abs/1"])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "baixa" in out
        assert "baixa — poucas fontes independentes (1 domínio(s))." in out
        assert "placeholder" not in out

    def test_already_baixa_kept(self):
        b = _briefing("baixa", ["https://arxiv.org/abs/1"])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "baixa" in out
        assert "rebaixado" not in out  # never upgrades or annotates baixa

    def test_no_fontes_block_returns_unchanged(self):
        b = "## Resumo executivo\nfoo\n\n## Confiança\nalta — sem fontes.\n"
        out = NightlyStudier._enforce_confidence_floor(b)
        assert out == b

    def test_no_confianca_block_returns_unchanged(self):
        b = (
            "## Resumo executivo\nfoo\n\n"
            "## Fontes consultadas\n- [https://arxiv.org/abs/1]\n"
        )
        out = NightlyStudier._enforce_confidence_floor(b)
        assert out == b

    def test_high_english_verdict_also_handled(self):
        """Prompt accepts PT or EN — make sure 'high' is treated like 'alta'."""
        b = _briefing("high", ["https://arxiv.org/abs/1"])
        out = NightlyStudier._enforce_confidence_floor(b)
        assert "baixa" in out


class TestThinSourceFlag:
    """_flag_thin_sources prepends a loud banner when a briefing cites
    zero verifiable source domains — separate from the confidence floor."""

    def test_zero_domains_gets_banner(self):
        b = _briefing("baixa", [])  # Fontes block present but no URLs
        out = NightlyStudier._flag_thin_sources(b)
        assert out.startswith("> [POUCAS FONTES]")
        assert NightlyStudier._SOURCE_WARNING_MARKER in out

    def test_no_fontes_block_gets_banner(self):
        b = "## Resumo executivo\nfoo\n\n## Confiança\nbaixa.\n"
        out = NightlyStudier._flag_thin_sources(b)
        assert out.startswith("> [POUCAS FONTES]")

    def test_one_domain_no_banner(self):
        b = _briefing("baixa", ["https://arxiv.org/abs/1"])
        out = NightlyStudier._flag_thin_sources(b)
        assert "POUCAS FONTES" not in out
        assert out == b

    def test_idempotent(self):
        b = _briefing("baixa", [])
        once = NightlyStudier._flag_thin_sources(b)
        twice = NightlyStudier._flag_thin_sources(once)
        assert once == twice  # banner not duplicated

    def test_summarise_chains_floor_then_flag(self):
        """The two guards compose: a 0-source 'alta' briefing comes out
        both floored to baixa AND banner-flagged."""
        b = _briefing("alta", [])
        out = NightlyStudier._flag_thin_sources(
            NightlyStudier._enforce_confidence_floor(b)
        )
        assert out.startswith("> [POUCAS FONTES]")
        assert "baixa" in out
        assert "baixa — poucas fontes independentes (0 domínio(s))." in out
        assert "placeholder" not in out



class TestAFailedSynthesisIsNotABriefing:
    """An Ollama failure came back as "## Erro / Falha ao gerar resumo",
    was saved in pending/, counted as produced and never retried (s88, Codex
    lote 14 — invariant 7)."""

    @staticmethod
    def _studier(tmp_path):
        from unittest.mock import MagicMock
        from src.core.studier import NightlyStudier
        s = NightlyStudier.__new__(NightlyStudier)
        s.output_dir = str(tmp_path)
        s.engine = MagicMock(model_name="cortex-lite")
        s.generate_queries = lambda topic, **kw: ["q1"]
        s.gather_snippets = lambda queries, ml_context=False: (
            "Source 1:\nRegras do xadrez.\n[https://example.org/x]")
        s._apply_critic = lambda topic, briefing, trace=None, **kw: briefing
        s._maybe_reflect = lambda *a: None
        return s

    @pytest.mark.parametrize("generate", [
        lambda **kw: (_ for _ in ()).throw(ConnectionError("ollama down")),
        lambda **kw: {"response": "   "},
    ])
    def test_no_file_and_no_path(self, tmp_path, monkeypatch, generate):
        from src.core import studier as st
        monkeypatch.setattr(st.ollama, "generate", generate)
        s = self._studier(tmp_path)
        assert s._study_one("regras do xadrez") is None
        assert list(tmp_path.glob("*.md")) == []

    def test_a_real_briefing_is_still_saved(self, tmp_path, monkeypatch):
        from src.core import studier as st
        monkeypatch.setattr(st.ollama, "generate", lambda **kw: {
            "response": "## Resumo executivo\nXadrez.\n\n## Confiança\nbaixa"})
        path = self._studier(tmp_path)._study_one("regras do xadrez")
        assert path and "Xadrez." in open(path).read()
