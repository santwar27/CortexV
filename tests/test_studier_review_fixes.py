"""Task 140d: regression coverage for the eight review fixes, without I/O services."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.deep_studier import DeepStudier
from src.core.studier import NightlyStudier


EXPLANATION = "A indicação apresentada precisa de confirmação segundo as fontes."


def block(text, number=1, url="https://example.org/article"):
    return f"Source {number} (Overview):\n{text}\n[{url}]"


def problem(rule, *quotes):
    return " | ".join(["PROBLEMA", rule, *quotes, EXPLANATION])


@pytest.fixture
def studier(tmp_path, monkeypatch):
    s = NightlyStudier.__new__(NightlyStudier)
    s.engine = SimpleNamespace(model_name="fake")
    s.web_searcher = Mock()
    s.output_dir = str(tmp_path)
    monkeypatch.setattr("src.core.studier.ollama.generate", Mock(return_value={"response": "OK"}))
    return s


def test_item1_all_off_topic_returns_none_without_synthesis(studier, monkeypatch, caplog):
    studier.generate_queries = Mock(return_value=["macOS"])
    studier.web_searcher.search.return_value = block("Trading stocks and shares.")
    studier._summarise = Mock(return_value="Must not be saved")
    studier._save = Mock()
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    with caplog.at_level(logging.WARNING):
        assert studier._study_one("macOS", trace="review") is None
    studier._summarise.assert_not_called()
    studier._save.assert_not_called()
    assert "all_off_topic" in caplog.text
    assert any(c.kwargs.get("payload", {}).get("reason") == "all_off_topic"
               for c in audit.call_args_list)


@pytest.mark.parametrize("search_text,reason", [
    (block("Trading stocks and shares."), "all_off_topic"),
    ("", "no_sources"),
    ("QUICK ANSWER: macOS release notes.", "no_sources"),
])
def test_item1_deep_zero_sources_skips_synthesis(studier, tmp_path, monkeypatch, search_text, reason):
    deep = DeepStudier(studier=studier, output_dir=str(tmp_path))
    deep._plan_subquestions = Mock(return_value=["macOS"])
    studier.generate_queries = Mock(return_value=["macOS"])
    studier.web_searcher.search.return_value = search_text
    deep._summarise_step = Mock(return_value="Must not be used")
    deep._synthesise_deep = Mock(return_value="Must not be used")
    audit = Mock()
    monkeypatch.setattr("src.core.deep_studier.audit.log", audit)
    report = deep._research_governed("macOS", "review")
    deep._summarise_step.assert_not_called()
    deep._synthesise_deep.assert_not_called()
    assert report.sources == [] and report.briefing.startswith("## Erro")
    assert any(c.kwargs.get("payload", {}).get("reason") == reason
               for c in audit.call_args_list)


@pytest.mark.parametrize("topic,acronym", [
    ("IA no dia a dia", "IA"), ("UX", "UX"), ("AI", "AI"),
    ("ML", "ML"), ("UI", "UI"), ("QA", "QA"), ("ABCDE", "ABCDE"),
])
def test_item2_capital_acronyms_are_specific_terms(studier, topic, acronym):
    good = block(f"Using {acronym} in everyday workflows.")
    bad = block("Local restaurants and hotels.", number=2)
    assert acronym in studier._distinctive_terms(topic)
    assert studier._filter_relevance(good + "\n\n" + bad, topic) == (good, 1)


@pytest.mark.parametrize("text", ["A média dos resultados melhora.", "ia", "ia en español",
                                       "XIA", "IA2", "ux", "abcde"])
def test_item2_acronyms_require_original_case_and_whole_token(studier, text):
    topic = "UX" if text == "ux" else "ABCDE" if text == "abcde" else "IA"
    assert studier._filter_relevance(block(text), topic) == ("", 1)


def test_item2_no_terms_keeps_nonblocklisted_results_and_logs_once(studier, caplog):
    good = block("Local restaurants and hotels.")
    bad = block("Mining rewards.", number=2, url="https://etherscan.io/x")
    studier.web_searcher.search.return_value = good + "\n\n" + bad
    counts = {}
    with caplog.at_level(logging.INFO):
        result = studier.gather_snippets(["one", "two"], topic="o que e", filter_counts=counts)
    assert result.count(good) == 2 and "etherscan" not in result
    assert counts == {"blocklist_dropped": 2, "off_topic_dropped": 0}
    assert sum("no distinctive terms" in r.message.lower() for r in caplog.records) == 1


def test_item3_contradictions_cannot_crowd_out_proven_problems(studier, monkeypatch):
    quotes = [f"O sistema número {i} apresenta melhorias para os utilizadores." for i in range(4)]
    dates = [f"O lançamento número {i} ocorreu em 2019 com melhorias importantes." for i in range(4)]
    reply = "\n".join([problem("contradicao", q, quotes[-1]) for q in quotes[:3]]
                      + [problem("data", q) for q in dates])
    monkeypatch.setattr("src.core.studier.ollama.generate", Mock(return_value={"response": reply}))
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    briefing = "\n".join(quotes + dates) + "\n\n## Confiança\nalta — fontes independentes."
    result = studier._apply_critic("sistema", briefing, trace="review")
    section = result.split("## Crítico\n")[1]
    assert section.count("- (data)") == 3 and section.count(" × ") == 2
    assert "## Confiança\nbaixa" in result
    audit.assert_called_once_with("review", "studier", "critic_verdict",
                                  payload={"proven": 3, "a_conferir": 2, "discarded": 2})


@pytest.mark.parametrize("answer,source,expected_answer,expected_source,counts", [
    ("macOS release notes.", "Local restaurant menus.", True, False, (0, 1)),
    ("Local restaurant menus.", "macOS release notes.", False, True, (0, 1)),
    ("macOS from https://etherscan.io/x", "macOS release notes.", False, True, (1, 0)),
])
def test_item4_quick_answer_is_an_independently_filtered_block(
        studier, answer, source, expected_answer, expected_source, counts):
    quick = "QUICK ANSWER: " + answer
    record = block(source)
    studier.web_searcher.search.return_value = quick + "\n\n" + record
    dropped = {}
    result = studier.gather_snippets(["macOS"], topic="macOS", filter_counts=dropped)
    assert (quick in result) is expected_answer
    assert (record in result) is expected_source
    assert dropped == dict(zip(("blocklist_dropped", "off_topic_dropped"), counts))


@pytest.mark.parametrize("text,accepted", [
    ("This may improve latency for all users.", False),
    ("They march together to improve the service.", False),
    ("An august institution maintains the service.", False),
    ("May 2026 brings changes for the service.", True),
    ("The release on 3 March brings changes.", True),
    ("March 3, 2025 brings changes for users.", True),
    ("August 3 brings changes for the service.", True),
    ("The release in maio brings changes for users.", True),
])
def test_item5_ordinary_month_words_need_date_context(studier, text, accepted):
    found, discarded, _ = studier._parse_critic_proof(problem("data", text), text)
    assert bool(found) is accepted and discarded == int(not accepted)


@pytest.mark.parametrize("marker,accepted", [
    ("8192 tokens", False), ("porta 8080", False), ("1899", False), ("2100", False),
    ("2019", True), ("1900", True), ("2099", True), ("06/11", True), ("6-11-26", True),
])
def test_item6_only_1900_to_2099_are_years(studier, marker, accepted):
    text = f"O resumo inclui a indicação {marker} para os utilizadores."
    found, discarded, _ = studier._parse_critic_proof(problem("data", text), text)
    assert bool(found) is accepted and discarded == int(not accepted)


@pytest.mark.parametrize("valid", [False, True])
def test_item7_chatter_is_ignored_but_chatter_only_is_malformed(studier, valid):
    quote = "O sistema nunca perde dados durante as operações."
    chatter = "Aqui estão os problemas encontrados:\n## Análise\n  \nNota adicional."
    reply = chatter + ("\n" + problem("absoluta", quote) if valid else "")
    found, discarded, malformed = studier._parse_critic_proof(reply, quote)
    assert bool(found) is valid and discarded == 0 and malformed is (not valid)
    _, discarded, _ = studier._parse_critic_proof(chatter + "\npRoBlEmA inválido", quote)
    assert discarded == 1


@pytest.mark.parametrize("rule", ["data", "contradicao"])
def test_item8_pipes_inside_quotes_are_accepted(studier, rule):
    first = "Em 2019 o comando cat entrada | sort | uniq produziu resultados."
    second = "O comando cat entrada | sort não produziu resultados para os utilizadores."
    quotes = [first, second] if rule == "contradicao" else [first]
    found, discarded, _ = studier._parse_critic_proof(problem(rule, *quotes), first + "\n" + second)
    assert found == [(rule, quotes, EXPLANATION)] and discarded == 0


def test_item8_pipe_split_does_not_rescue_missing_contradiction(studier):
    first = "Em 2019 o comando cat entrada | sort produziu resultados."
    absent = "Uma passagem ausente | com palavras suficientes para validar."
    found, discarded, _ = studier._parse_critic_proof(problem("contradicao", first, absent), first)
    assert found == [] and discarded == 1
