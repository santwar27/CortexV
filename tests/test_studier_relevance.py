"""Code-only relevance gate: real provider formats, no network or model."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.deep_studier import DeepStudier
from src.core.studier import NightlyStudier


NEWS = "Principais acontecimentos de 2024 a 2026 no Reino Unido e no Brasil"
MACOS = "Novidades do macOS dos últimos dois anos"
RERANKER = ("reranker cross-encoder para RAG em Apple Silicon sem PyTorch: "
            "bge-reranker, ONNX Runtime e MLX")


def block(title, content, url="https://example.org/article", number=1):
    return f"Source {number} ({title}):\n{content}\n[{url}]"


@pytest.fixture
def studier(tmp_path):
    s = NightlyStudier.__new__(NightlyStudier)
    s.engine = SimpleNamespace(model_name="fake")
    s.web_searcher = Mock()
    s.output_dir = str(tmp_path)
    return s


# The same lexical evidence must be checked per block, never per query.
@pytest.mark.parametrize("topic,title,content,url", [
    (NEWS, "Connections Luxury", "Luxury travel and hotel collections.",
     "https://connectionsluxury.com/2024"),
    (NEWS, "Condé Nast Johansens", "Discover luxury hotels and spas.",
     "https://johansens.com/united-kingdom"),
    (NEWS, "McKinlay Kidd", "Tailor-made luxury holidays.",
     "https://mckinlaykidd.com/uk"),
    (NEWS, "Tripadvisor restaurants", "Best places to eat in London.",
     "https://tripadvisor.com/Restaurants"),
    (NEWS, "Time Out London", "The best restaurants to try this year.",
     "https://timeout.com/london/restaurants"),
    (NEWS, "Dallas Cowboys", "Team scores and fixtures for 2024 to 2026.",
     "https://dallascowboys.com/news"),
    (NEWS, "Wikipedia, the free encyclopedia", "Welcome to Wikipedia.",
     "https://en.wikipedia.org/wiki/Main_Page"),
    (NEWS, "VPN connection issue", "Tunnel fails after reconnecting.",
     "https://github.com/vendor/vpn/issues/123"),
    (MACOS, "Trading signals", "Latest market news and stock prices.",
     "https://trading.example/macos"),
    (MACOS, "Dolphin anty", "Browser profiles for managing accounts.",
     "https://dolphin-anty.com"),
    ("CSS cascade layers", "Flexbox application changelog",
     "Flexbox changed its subscription billing.", "https://example.org/flexbox"),
])
def test_real_noise_is_dropped(studier, topic, title, content, url):
    assert studier._filter_relevance(block(title, content, url), topic) == ("", 1)


@pytest.mark.parametrize("topic,title,content", [
    (NEWS, "Brazil election", "A report on the election outcome."),
    (NEWS, "UK election", "A report on the election outcome."),
    (NEWS, "United Kingdom budget", "A report on public spending."),
    (NEWS, "Britain's economy", "A report on public spending."),
    ("Brazil politics", "Brasil", "Political developments."),
    (MACOS, "macOS Sequoia", "Security and release notes."),
    (RERANKER, "cross encoder reranker", "Reordering retrieved documents."),
    ("cross-encoder", "Cross encoders", "Document scoring."),
    ("cross encoders", "Cross-encoder", "Document scoring."),
    ("SQLite WAL", "SQLite write-ahead logging", "Concurrency details."),
    ("WAL", "Checkpointing", "WAL files and readers."),
    ("OAuth PKCE", "OAuth authorization", "PKCE protects public clients."),
    ("PKCE", "Authorization", "PKCE protects public clients."),
    ("attention mechanism", "Attention Is All You Need", "Transformer paper."),
    ("mecanismo de atenção", "Attention mechanism", "Transformer paper."),
    ("mecanismo", "Mechanisms", "Explanation."),
    ("mechanisms", "Mecanismo", "Explicação."),
    ("ATENÇÃO", "Atencao", "Uma explicação."),
    ("transformers", "Transformer", "An architecture."),
    ("transformer", "Transformers", "Architectures."),
    ("avaliações", "Avaliacao", "Resultados."),
    ("redes neurais", "Rede neural", "Arquitetura."),
    ("security policies", "Security policy", "Configuration."),
    ("cross-encoder", "Cross\u2011encoders", "Document scoring."),
    ("notícias do Brasil", "Brazil news", "Election coverage."),
    ("CSS", "CSS layout", "Flexbox aligns items."),
])
def test_genuine_sources_survive_variants(studier, topic, title, content):
    source = block(title, content)
    assert studier._filter_relevance(source, topic) == (source, 0)


@pytest.mark.parametrize("content", [
    "2024 2025 2026", "2024-2026", "Principais novidades", "últimos anos", "ultimos anos",
    "Melhores práticas", "Latest news and events", "Acontecimentos e notícias",
])
def test_years_and_generic_words_are_not_evidence(studier, content):
    topic = NEWS + " 2024-2026 novidades últimos anos melhor prática best-practices"
    assert studier._filter_relevance(block("Overview", content), topic) == ("", 1)


def test_no_distinctive_terms_skips_relevance(studier):
    source = block("Overview", "2024 novidades")
    assert studier._filter_relevance(source, "Principais novidades de 2024") == (source, 0)


def test_whole_tokens_not_substrings(studier):
    assert studier._filter_relevance(block("Drawal", "Networking."), "WAL") == ("", 1)


def test_matching_source_does_not_rescue_its_neighbours(studier):
    good = block("macOS Sequoia", "Release notes.")
    bad = block("Dolphin anty", "Browser profiles.", number=2)
    assert studier._filter_relevance(good + "\n\n" + bad, MACOS) == (good, 1)


def test_ddg_single_newline_blocks_are_filtered_individually(studier):
    good = "Result 1 (macOS Sequoia): Release notes.\n[https://apple.com/macos]"
    bad = "Result 2 (Dolphin anty): Browser profiles.\n[https://dolphin-anty.com]"
    assert studier._filter_relevance(good + "\n" + bad + "\n", MACOS) == (good, 1)


def test_snippet_paragraphs_stay_with_their_source(studier):
    good = block("Release notes", "Introduction.\n\nmacOS Sequoia security.")
    assert studier._filter_relevance(good, MACOS) == (good, 0)


def test_useful_github_sources_are_not_blocked_by_domain(studier):
    good = block("SQLite WAL checkpoint", "Concurrency details.",
                 "https://github.com/example/sqlite/issues/123")
    studier.web_searcher.search.return_value = good
    assert good in studier.gather_snippets(["SQLite WAL"], topic="SQLite WAL")


def test_provider_answer_cannot_rescue_off_topic_sources(studier):
    answer = "QUICK ANSWER: macOS Sequoia release notes."
    studier.web_searcher.search.return_value = (
        answer + "\n\n"
        + block("Trading", "Stock prices.")
    )
    assert studier.gather_snippets(["macOS"], topic=MACOS) == "### Query: macOS\n" + answer


def test_blocklisted_query_is_skipped_before_relevance(studier, monkeypatch, caplog):
    studier.web_searcher.search.return_value = block(
        "macOS", "macOS release.", "https://etherscan.io/x")
    relevance = Mock(side_effect=AssertionError("blocklist must run first"))
    monkeypatch.setattr(studier, "_filter_relevance", relevance)
    counts = {}
    with caplog.at_level(logging.WARNING, logger="src.core.studier"):
        assert studier.gather_snippets(["macOS"], topic=MACOS, filter_counts=counts) == ""
    relevance.assert_not_called()
    assert counts == {"blocklist_dropped": 1, "off_topic_dropped": 0}
    assert "hit the blocklist" in caplog.text


def test_url_and_query_heading_cannot_supply_evidence(studier):
    bad = block("Trading", "Stock prices.", "https://macos.example/macos")
    assert studier._filter_relevance("### Query: macOS\n" + bad, MACOS) == ("", 1)


def test_all_off_topic_query_is_skipped_and_counts_are_per_topic(studier, caplog):
    bad = block("Restaurants", "Dining in London.")
    good = block("macOS Sequoia", "Release notes.")
    studier.web_searcher.search.side_effect = [bad, bad + "\n\n" + good]
    counts = {}
    with caplog.at_level(logging.INFO, logger="src.core.studier"):
        result = studier.gather_snippets(["wrong query", "release query"],
                                         topic=MACOS, filter_counts=counts)
    assert result == "### Query: release query\n" + good
    assert counts == {"blocklist_dropped": 0, "off_topic_dropped": 2}
    assert "all snippets for 'wrong query' are off-topic" in caplog.text
    lines = [r.message for r in caplog.records if "dropped 2 off-topic snippet(s)" in r.message]
    assert len(lines) == 1 and MACOS in lines[0]


def test_blocklist_runs_before_relevance_including_ddg(studier):
    studier.web_searcher.search.return_value = (
        "Result 1 (macOS mining): macOS.\n[https://etherscan.io/x]\n"
        "Result 2 (Trading): Stocks.\n[https://example.org/trading]\n"
        "Result 3 (macOS Sequoia): Release.\n[https://apple.com/macos]\n"
    )
    counts = {}
    result = studier.gather_snippets(["macOS"], topic=MACOS, filter_counts=counts)
    assert counts == {"blocklist_dropped": 1, "off_topic_dropped": 1}
    assert "macOS Sequoia" in result
    assert "etherscan" not in result and "Trading" not in result


def test_zero_kept_blocks_returns_none_and_audits_counts(studier, monkeypatch):
    from src.core.audit import audit

    rejected = block("Dolphin anty", "Browser profiles.")
    studier.web_searcher.search.return_value = rejected
    studier.generate_queries = Mock(return_value=["macOS", "macOS releases"])
    studier._apply_critic = Mock(side_effect=lambda topic, briefing, **kw: briefing)
    studier._maybe_reflect = Mock()
    generate = Mock(return_value={"response": (
        "## Resumo executivo\nSem fontes.\n\n"
        "## Fontes consultadas\n\n## Confiança\nalta — confident."
    )})
    monkeypatch.setattr("src.core.studier.ollama.generate", generate)
    audit_log = Mock()
    monkeypatch.setattr(audit, "log", audit_log)
    path = studier._study_one(MACOS)
    assert path is None
    generate.assert_not_called()
    payload = next(c.kwargs["payload"] for c in audit_log.call_args_list
                   if c.args[2] == "search_done")
    assert payload["off_topic_dropped"] == 2 and payload["snippet_chars"] == 0


def test_legacy_injected_gatherer_still_filters_and_skips_empty_study(studier):
    # Compatibility with the curiosity integration's old two-argument hook.
    studier.generate_queries = lambda topic, **kw: ["macOS"]
    studier.gather_snippets = lambda queries, ml_context=False: block("Trading", "Stocks.")
    summarise = Mock(return_value="Thin briefing")
    studier._summarise = summarise
    studier._apply_critic = lambda topic, briefing, **kw: briefing
    studier._maybe_reflect = Mock()
    assert studier._study_one(MACOS) is None
    summarise.assert_not_called()


def test_deep_study_uses_parent_topic_before_step_summary_and_citations(
    studier, tmp_path, monkeypatch,
):
    deep = DeepStudier(studier=studier, output_dir=str(tmp_path))
    deep._plan_subquestions = Mock(return_value=["Dolphin browser profiles"])
    studier.generate_queries = Mock(return_value=["Dolphin browser profiles"])
    studier.web_searcher.search.return_value = block("Dolphin anty", "Browser profiles.")
    deep._summarise_step = Mock(return_value="Should not be used")
    deep._synthesise_deep = Mock(return_value="Thin briefing")
    # Test the loop directly: no lease/network, and actual NightlyStudier gathering.
    report = deep._research_governed(MACOS, "test-trace")
    assert report.sources == [] and report.steps[0].snippet_chars == 0
    deep._summarise_step.assert_not_called()
    deep._synthesise_deep.assert_not_called()
