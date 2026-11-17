"""Tasks 140/140b/140c: rule-matched proof and tentative contradictions."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.studier import NightlyStudier
from src.core.study_time import EVENT_TIME_RULES


CLAIM = "Todos os pedidos são sempre processados sem falhas."
EXPLANATION = "A afirmação promete ausência de falhas sem apresentar evidência."
NO_PROOF = "O crítico não encontrou nenhum problema comprovado."
TO_CHECK = "A conferir (o crítico não prova contradição):"
EVENT_MARKERS = (
    "terá", "será", "previsto", "prevista", "previstos", "previstas",
    "agendado", "agendada", "aconteceu", "ocorreu", "acontecerá", "ocorrerá",
    "scheduled", "upcoming", "happened",
)


@pytest.fixture
def studier():
    s = NightlyStudier.__new__(NightlyStudier)
    s.engine = SimpleNamespace(model_name="test-model")
    return s


def briefing(body=CLAIM, confidence="alta"):
    return (f"## Resumo executivo\n{body}\n\n"
            f"## Confiança\n{confidence} — fontes independentes.\n")


def problem(quote=CLAIM, explanation=EXPLANATION, rule="absoluta", quote2=None):
    second = f" | {quote2}" if quote2 is not None else ""
    return f"PROBLEMA | {rule} | {quote}{second} | {explanation}"


def run(studier, monkeypatch, reply, text=None, topic="FastAPI", **kwargs):
    generate = Mock(return_value={"response": reply})
    monkeypatch.setattr("src.core.studier.ollama.generate", generate)
    result = studier._apply_critic(topic, text or briefing(), **kwargs)
    return result, generate


@pytest.mark.parametrize("complaint", [
    "Acronimo SOLID não é expandido corretamente.",
    "Acronimo 'STT' não expandido.",
    "Acronyms 'WAL' and 'I/O' are not expanded.",
    "A sigla usada no texto não foi explicada por extenso.",
    "Falta a expansão da sigla usada neste trecho.",
    "A sigla aparece sem expansão neste trecho.",
    "The acronym in this passage is left unexpanded.",
    "The acronym in this passage has no expansion.",
    "A sigla usada no trecho não foi definida.",
    "The acronym in this passage is not defined.",
    "O acrónimo usado no trecho carece de expansão.",
])
def test_unexpanded_acronyms_are_not_problems(studier, monkeypatch, complaint):
    # Even when a sufficiently long quote exists, this is outside rule 1.
    result, _ = run(studier, monkeypatch, problem(explanation=complaint))
    assert NO_PROOF in result
    assert "Descartados: 1." in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("reply", [
    "PROBLEMA | Acronimo SOLID não é expandido corretamente",
    "PROBLEMA | Acronimo 'STT' não expandido",
    "PROBLEMA | Acronyms 'WAL' and 'I/O' are not expanded",
    "PROBLEMA | Acronimo 'RAG' não expandido corretamente",
    "PROBLEMA | 1. RAG = Retrieval-Augmented Generation (expansão incorreta)",
    "PROBLEMA | Acronyms não são utilizados; afirmações fortes …",
])
def test_real_unproven_complaints_are_discarded(studier, monkeypatch, reply):
    result, _ = run(studier, monkeypatch, reply)
    assert NO_PROOF in result
    assert f"Descartados: {len(reply.split(';'))}." in result
    assert "alta — fontes independentes." in result


def test_rule_numbers_are_counted_and_discarded(studier, monkeypatch):
    result, _ = run(studier, monkeypatch, "PROBLEMA | 3 ; 4")
    assert NO_PROOF in result
    assert "Descartados: 2." in result
    assert "alta — fontes independentes." in result


def test_rag_quote_absent_from_fastapi_study_is_discarded(studier, monkeypatch):
    result, _ = run(studier, monkeypatch, problem(
        "RAG = Retrieval-Augmented Generation",
        "A expansão da sigla está incorreta neste trecho."))
    assert NO_PROOF in result
    assert "Descartados: 1." in result
    assert "alta — fontes independentes." in result


def test_proven_problem_is_readable_and_lowers_confidence(studier, monkeypatch):
    result, _ = run(studier, monkeypatch, problem())
    assert f"## Crítico\n- (absoluta) «{CLAIM}» — {EXPLANATION}\n" in result
    assert "## Confiança\nbaixa — o crítico achou problemas (ver ## Crítico)." in result
    assert "alta — fontes independentes." not in result


def test_wrong_expansion_is_still_a_valid_check(studier, monkeypatch):
    quote = "ABC significa Outra Coisa Inventada neste estudo."
    explanation = "A expansão apresentada contradiz a definição citada na fonte."
    result, _ = run(studier, monkeypatch, problem(quote, explanation, rule="sigla"), briefing(quote))
    assert f"- (sigla) «{quote}» — {explanation}" in result
    assert "## Confiança\nbaixa" in result


def test_quote_normalises_case_spacing_and_quote_marks(studier, monkeypatch):
    text = briefing('Todos os pedidos “seguros” são\n  SEMPRE processados sem falhas.')
    quote = '«todos  os pedidos "SEGUROS" são sempre processados sem falhas.»'
    result, _ = run(studier, monkeypatch, problem(quote), text)
    assert NO_PROOF not in result
    assert "## Confiança\nbaixa" in result
    assert "««" not in result


@pytest.mark.parametrize("quote", [
    "Todos", "2026", "extraordinariamenteincondicional",
    "2026 2027 2028 2029 2030 2031", "Todos os pedidos",
])
def test_short_or_number_only_quotes_are_rejected(studier, monkeypatch, quote):
    result, _ = run(studier, monkeypatch, problem(quote), briefing(quote))
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("explanation", [
    "3", "3 ; 4", "PROBLEMA", "Regra 3.", "Problema sem suporte",
    "Esta afirmação contém um problema sem evidência",  # no sentence ending
    "Regras três e quatro e cinco.",
])
def test_explanation_must_be_a_sentence(studier, monkeypatch, explanation):
    result, _ = run(studier, monkeypatch, problem(explanation=explanation))
    assert NO_PROOF in result
    assert "Descartados: 1." in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("location", ["topic", "topic_header", "critic", "instructions", "beyond_cap"])
def test_quote_must_be_in_the_body_actually_shown(studier, monkeypatch, location):
    quote = "Uma frase exclusiva que não pertence ao corpo."
    text, topic = briefing(), "FastAPI"
    if location == "topic":
        topic = quote
    elif location == "topic_header":
        text = f"# Topic: {quote}\n\n{text}"
    elif location == "critic":
        text += f"\n## Crítico\n- «{quote}» — {EXPLANATION}\n"
    elif location == "instructions":
        quote = EVENT_TIME_RULES.split(". ")[0]
    elif location == "beyond_cap":
        text = briefing("Uma frase neutra. " * 250 + quote)
    result, generate = run(studier, monkeypatch, problem(quote), text, topic)
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result
    if location in {"critic", "topic_header"}:
        assert quote not in generate.call_args.kwargs["prompt"]
    if location == "critic":
        assert result.count("## Crítico\n") == 1


def test_all_discarded_preserves_confidence_and_reports_count(studier, monkeypatch):
    reply = "\n".join([problem("Trecho ausente com palavras suficientes."),
                       problem(explanation="3 ; 4"),
                       problem(explanation="A sigla não foi expandida corretamente.")])
    result, _ = run(studier, monkeypatch, reply)
    assert f"## Crítico\n{NO_PROOF} Descartados: 3.\n" in result
    assert "alta — fontes independentes." in result


def test_empty_previous_critic_header_is_replaced(studier, monkeypatch):
    result, _ = run(studier, monkeypatch, "OK", briefing() + "\n## Crítico")
    assert result.count("## Crítico") == 1
    assert result.endswith(f"## Crítico\n{NO_PROOF}\n")


@pytest.mark.parametrize("reply, discarded", [("", 0), ("\n ", 0), ("garbage", 0)])
def test_empty_or_garbage_reply_has_no_proof_and_is_logged(
        studier, monkeypatch, caplog, reply, discarded):
    result, _ = run(studier, monkeypatch, reply)
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result
    assert "critic" in caplog.text.lower()
    assert "unparseable" in caplog.text.lower() or "empty" in caplog.text.lower()
    if discarded:
        assert f"Descartados: {discarded}." in result


def test_cap_and_audit_counts(studier, monkeypatch):
    quotes = [f"A operação número {i} nunca apresenta qualquer falha." for i in range(5)]
    reply = "\n".join(problem(q) for q in quotes) + "\n" + problem("Um trecho que não existe no corpo.")
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    result, _ = run(studier, monkeypatch, reply, briefing("\n".join(quotes)), trace="t140")
    section = result.split("\n## Crítico\n")[1]
    assert section.count("- (absoluta) «") == 3
    assert all(q in section for q in quotes[:3])
    assert all(q not in section for q in quotes[3:])
    audit.assert_called_once_with("t140", "studier", "critic_verdict",
                                  payload={"proven": 3, "a_conferir": 0, "discarded": 3})


@pytest.mark.parametrize("reply, counts", [
    ("OK", {"proven": 0, "a_conferir": 0, "discarded": 0}),
    ("", {"proven": 0, "a_conferir": 0, "discarded": 0}),
    ("PROBLEMA | 3 ; 4", {"proven": 0, "a_conferir": 0, "discarded": 2}),
    (problem(), {"proven": 1, "a_conferir": 0, "discarded": 0}),
])
def test_audit_counts_without_briefing(studier, monkeypatch, reply, counts):
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    run(studier, monkeypatch, reply, trace="t140")
    audit.assert_called_once_with("t140", "studier", "critic_verdict", payload=counts)


def test_prompt_requests_evidence_without_an_echoable_acronym_example(studier, monkeypatch):
    _, generate = run(studier, monkeypatch, "OK")
    prompt = generate.call_args.kwargs["prompt"]
    assert "RAG" not in prompt
    assert "Retrieval-Augmented" not in prompt
    assert "unexpanded acronym is NOT a problem" in prompt
    assert "Portuguese" in prompt
    for rule in ("sigla", "absoluta", "data"):
        assert f"PROBLEMA | {rule} | <exact quote copied from the briefing> | <one-sentence explanation>" in prompt
    assert "PROBLEMA | contradicao | <first exact quote> | <second exact quote> | <one-sentence explanation>" in prompt
    assert "non-overlapping" in prompt
    assert "quote itself" in prompt
    assert "quoted passages alone do not prove a contradiction" in prompt
    assert "or one of these event time/tense markers: " + ", ".join(EVENT_MARKERS) + "." in prompt
    assert "Ordinary auxiliary verbs alone do not qualify" in prompt
    assert EVENT_TIME_RULES in prompt
    assert generate.call_args.kwargs["options"]["num_predict"] >= 600


def test_multiple_lines_keep_only_proven_problems(studier, monkeypatch):
    reply = f"OK\n{problem()}\n{problem('Trecho ausente com palavras suficientes.')}"
    result, _ = run(studier, monkeypatch, reply)
    assert f"- (absoluta) «{CLAIM}» — {EXPLANATION}" in result
    assert "Trecho ausente" not in result
    assert "## Confiança\nbaixa" in result


@pytest.mark.parametrize("quote, explanation, rule", [
    ("O uso da injecção de dependência promove separação de preocupações e facilita os testes.",
     "Não há evidência de que a injecção sempre promova separação de preocupações.", "absoluta"),
    ("A aplicação dos princípios SOLID em Python melhora a manutenibilidade e flexibilidade do código.",
     "O trecho apresenta um forte julgamento sem evidências.", "absoluta"),
    ("A responsabilidade única (SRP) afirma que cada classe deve ter uma responsabilidade.",
     "A definição é incompleta e omite detalhes importantes.", "definicao"),
    ("A implementação do PKCE é recomendada por padrão em muitas plataformas modernas.",
     "A recomendação não se aplica em todas as plataformas.", "absoluta"),
])
def test_real_false_positives_preserve_confidence(studier, monkeypatch, quote, explanation, rule):
    result, _ = run(studier, monkeypatch, problem(quote, explanation, rule), briefing(quote))
    assert f"{NO_PROOF} Descartados: 1." in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("quote, topic", [
    ("O macOS High Sierra foi lançado em 2017 com várias melhorias de segurança.",
     "Novidades do macOS nos últimos dois anos"),
    ("O Reino Unido terá eventos culturais e festivais em 2026, incluindo o GAIA Mayfair jazz festival.",
     "Eventos culturais no Reino Unido em 2026"),
    ("A eleição presidencial de 2026 no Brasil inclui candidatos de vários partidos.",
     "A eleição presidencial de 2026 no Brasil"),
    ("O macOS Mojave teve a última atualização em 10.14.6 em 26 de setembro de 2019.",
     "Novidades do macOS nos últimos dois anos"),
])
def test_real_date_problems_survive(studier, monkeypatch, quote, topic):
    explanation = "A data ou o tempo verbal contradiz o período indicado para este estudo."
    result, _ = run(studier, monkeypatch, problem(quote, explanation, "data"), briefing(quote), topic)
    assert f"- (data) «{quote}» — {explanation}" in result
    assert "## Confiança\nbaixa" in result


def test_real_contradiction_displays_both_passages(studier, monkeypatch):
    first = "O macOS dos últimos dois anos não apresentou novidades significativas para os utilizadores."
    second = "As novidades incluem Apple Intelligence e melhorias significativas na integração com o iPhone."
    explanation = "A ausência de novidades contradiz a lista de melhorias significativas apresentada."
    result, _ = run(studier, monkeypatch, problem(first, explanation, "contradicao", second),
                    briefing(first + "\n" + second))
    assert f"{NO_PROOF}\n\n{TO_CHECK}\n- «{first}» × «{second}» — {explanation}" in result
    assert "## Confiança\nalta — fontes independentes." in result


@pytest.mark.parametrize("confidence", ["alta", "média", "baixa"])
@pytest.mark.parametrize("discarded", [0, 1])
def test_contradiction_only_is_tentative_and_preserves_confidence_and_audit(
        studier, monkeypatch, confidence, discarded):
    first = "A responsabilidade única (SRP) afirma que cada classe deve ter uma responsabilidade."
    second = "O princípio aberto/fechado (OCP) diz que o código deve permitir extensão."
    reply = problem(first, rule="contradicao", quote2=second)
    if discarded:
        reply += "\n" + problem("Uma passagem ausente com palavras suficientes.")
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    result, _ = run(studier, monkeypatch, reply, briefing(first + "\n" + second, confidence),
                    trace="t140c")
    no_proof = NO_PROOF + (f" Descartados: {discarded}." if discarded else "")
    assert result.endswith(
        f"## Crítico\n{no_proof}\n\n{TO_CHECK}\n- «{first}» × «{second}» — {EXPLANATION}\n")
    assert f"## Confiança\n{confidence} — fontes independentes." in result
    audit.assert_called_once_with("t140c", "studier", "critic_verdict",
                                  payload={"proven": 0, "a_conferir": 1, "discarded": discarded})


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_data_and_contradiction_have_separate_lists_and_audit(studier, monkeypatch, contradiction_first):
    first = "O evento Connections Luxury em Londres em 2026 reúne profissionais do setor."
    second = "A eleição presidencial de 2026 no Brasil inclui candidatos de vários partidos."
    replies = [problem(second, rule="data"), problem(first, rule="contradicao", quote2=second)]
    if contradiction_first:
        replies.reverse()
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    result, _ = run(studier, monkeypatch, "\n".join(replies), briefing(first + "\n" + second),
                    trace="t140c")
    assert result.endswith(
        f"## Crítico\n- (data) «{second}» — {EXPLANATION}\n\n"
        f"{TO_CHECK}\n- «{first}» × «{second}» — {EXPLANATION}\n")
    assert NO_PROOF not in result
    assert "## Confiança\nbaixa — o crítico achou problemas (ver ## Crítico)." in result
    audit.assert_called_once_with("t140c", "studier", "critic_verdict",
                                  payload={"proven": 1, "a_conferir": 1, "discarded": 0})


@pytest.mark.parametrize("data_position", [None, 0, 3])
def test_contradictions_have_separate_two_item_cap(studier, monkeypatch, data_position):
    quotes = [f"O sistema número {i} apresenta melhorias significativas para os utilizadores."
              for i in range(4)]
    replies = [problem(q, rule="contradicao", quote2=quotes[-1]) for q in quotes[:3]]
    data_quote = "O festival cultural está agendado para a próxima edição."
    if data_position is None:
        replies.append(problem(quotes[-1], rule="contradicao", quote2=quotes[0]))
    else:
        replies.insert(data_position, problem(data_quote, rule="data"))
    audit = Mock()
    monkeypatch.setattr("src.core.studier.audit.log", audit)
    result, generate = run(studier, monkeypatch, "\n".join(replies),
                           briefing("\n".join(quotes + [data_quote])), trace="t140c")
    section = result.split("\n## Crítico\n")[1]
    proven = int(data_position is not None)
    assert section.count(" × ") == 2
    assert section.count("- (data)") == proven
    assert "write at most 3 proven problems" in generate.call_args.kwargs["prompt"]
    assert "at most 2 tentative contradictions" in generate.call_args.kwargs["prompt"]
    if proven:
        assert NO_PROOF not in section
        assert "## Confiança\nbaixa" in result
    else:
        assert f"{NO_PROOF} Descartados: 2." in section
        assert "## Confiança\nalta — fontes independentes." in result
    audit.assert_called_once_with("t140c", "studier", "critic_verdict",
                                  payload={"proven": proven, "a_conferir": 2,
                                           "discarded": 2 - proven})


@pytest.mark.parametrize("word", [
    "sempre", "nunca", "todos", "todas", "tudo", "nenhum", "nenhuma", "jamais",
    "always", "never", "all", "every", "none", "SEMPRE", "NÚNCA", "JAMA\u0301IS",
])
def test_absolute_words_are_case_and_accent_insensitive(studier, monkeypatch, word):
    quote = f"O sistema usa a expressão {word} para garantir o resultado."
    result, _ = run(studier, monkeypatch, problem(quote), briefing(quote))
    assert f"- (absoluta) «{quote}»" in result
    assert "## Confiança\nbaixa" in result


@pytest.mark.parametrize("word", ["normally", "small", "overall", "nevermore", "everywhere", "nenhumas"])
def test_absolute_word_substrings_are_rejected(studier, monkeypatch, word):
    quote = f"O sistema usa a expressão {word} para descrever o resultado."
    result, _ = run(studier, monkeypatch, problem(quote), briefing(quote))
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("expansion", [
    "ABC (Another Bogus Concept)", "ABC = Another Bogus Concept", "ABC=Another Bogus Concept",
    "ABC stands for Another Bogus Concept", "ABC significa Outra Coisa Inventada",
    "ABC é a sigla de Outra Coisa Inventada", "Outra Coisa Inventada (ABC)",
    "ABC SIGNIFICA Outra Coisa Inventada", "ABC E A SIGLA DE Outra Coisa Inventada",
    "ABC (Another-Bogus Concept)",
])
def test_acronym_expansion_shapes_are_kept(studier, monkeypatch, expansion):
    quote = f"Neste estudo, {expansion} descreve o protocolo."
    explanation = "A expansão apresentada está errada segundo a definição da fonte."
    result, _ = run(studier, monkeypatch, problem(quote, explanation, "sigla"), briefing(quote))
    assert f"- (sigla) «{quote}» — {explanation}" in result
    assert "## Confiança\nbaixa" in result


@pytest.mark.parametrize("fragment", [
    "ABC permite processar dados", "abc significa Outra Coisa Inventada",
    "A significa Outra Coisa Inventada", "ABCDEFG significa Outra Coisa Inventada",
    "ABC () permite processar dados", "ABC = 1234 permite processar dados",
    "ABC melhora o sistema (com mais rapidez)",
])
def test_acronym_without_attached_expansion_is_rejected(studier, monkeypatch, fragment):
    quote = f"Neste estudo, {fragment} de forma eficiente."
    result, _ = run(studier, monkeypatch, problem(quote, rule="sigla"), briefing(quote))
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("marker", [
    "2017", "1900", "2099", "06/11", "6-11-26", "2026-11-06",
    "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
    "setembro", "outubro", "novembro", "dezembro",
    "January", "February", "3 March", "April", "May 3", "June", "July", "August 3",
    "September", "October", "November", "December", "MARÇO", "MARC\u0327O",
    *EVENT_MARKERS, "TERÁ", "SERA", "ACONTECERA\u0301",
])
def test_date_markers_are_case_and_accent_insensitive(studier, monkeypatch, marker):
    quote = f"O evento cultural inclui a indicação {marker} neste resumo."
    result, _ = run(studier, monkeypatch, problem(quote, rule="data"), briefing(quote))
    assert f"- (data) «{quote}»" in result
    assert "## Confiança\nbaixa" in result


@pytest.mark.parametrize("verb", ["foi", "vai", "was", "will", "é", "era", "teve", "has", "had"])
def test_ordinary_verbs_alone_are_not_date_markers(studier, monkeypatch, verb):
    quote = f"O texto contém a expressão {verb} para descrever o mecanismo de atenção."
    explanation = "O evento será realizado em março de 2026 segundo a fonte."
    result, _ = run(studier, monkeypatch, problem(quote, explanation, "data"), briefing(quote))
    assert f"{NO_PROOF} Descartados: 1." in result
    assert "## Confiança\nalta — fontes independentes." in result


def test_attention_paper_without_date_is_not_a_date_problem(studier, monkeypatch):
    quote = 'O mecanismo de atenção foi apresentado no artigo "Attention Is All You Need".'
    result, _ = run(studier, monkeypatch, problem(quote, rule="data"), briefing(quote))
    assert f"{NO_PROOF} Descartados: 1." in result
    assert "## Confiança\nalta — fontes independentes." in result


@pytest.mark.parametrize("fragment", ["maior", "marching", "willpower", "12345", "a data indicada"])
def test_date_rule_requires_marker_in_quote(studier, monkeypatch, fragment):
    quote = f"O festival cultural inclui {fragment} no resumo do estudo."
    explanation = "O evento será realizado em março de 2026 segundo a fonte."
    result, _ = run(studier, monkeypatch, problem(quote, explanation, "data"), briefing(quote))
    assert NO_PROOF in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("case", ["missing", "same", "same_normalised", "overlap", "one_quote"])
def test_contradiction_needs_two_distinct_nonoverlapping_found_passages(studier, monkeypatch, case):
    first = "O sistema apresenta várias melhorias significativas para os utilizadores."
    second = "As melhorias significativas para os utilizadores incluem uma interface nova."
    text = briefing(first + "\n" + second)
    if case == "missing":
        second = "Uma passagem ausente que o crítico inventou neste momento."
    elif case == "same":
        second = first
        text = briefing(first + "\n" + first)  # Duplicate text cannot prove opposition.
    elif case == "same_normalised":
        second = first.upper()
    elif case == "overlap":
        text = briefing("O sistema apresenta várias melhorias significativas para os utilizadores incluem uma interface nova.")
        first = "O sistema apresenta várias melhorias significativas para os utilizadores"
        second = "melhorias significativas para os utilizadores incluem uma interface nova."
    else:
        second = None
    result, _ = run(studier, monkeypatch, problem(first, rule="contradicao", quote2=second), text)
    assert f"{NO_PROOF} Descartados: 1." in result
    assert "alta — fontes independentes." in result


def test_contradiction_can_use_later_nonoverlapping_occurrence(studier, monkeypatch):
    first = "O sistema apresenta várias melhorias significativas para os utilizadores"
    second = "melhorias significativas para os utilizadores incluem uma interface nova."
    body = first + " incluem uma interface nova.\n" + second
    result, _ = run(studier, monkeypatch, problem(first, rule="contradicao", quote2=second), briefing(body))
    assert f"{TO_CHECK}\n- «{first}» × «{second}»" in result
    assert "## Confiança\nalta — fontes independentes." in result


@pytest.mark.parametrize("reply", [
    f"PROBLEMA | {CLAIM} | {EXPLANATION}",
    problem(rule=""), problem(rule="2"), problem(rule="desconhecida"),
    problem(rule="definicao"), problem(quote2=CLAIM),
])
def test_missing_unknown_rule_or_extra_field_is_rejected(studier, monkeypatch, reply):
    result, _ = run(studier, monkeypatch, reply)
    assert f"{NO_PROOF} Descartados: 1." in result
    assert "alta — fontes independentes." in result


@pytest.mark.parametrize("confidence", ["alta", "média", "baixa"])
def test_rule_mismatch_does_not_rewrite_confidence(studier, monkeypatch, confidence):
    quote = "A aplicação dos princípios melhora a flexibilidade do código."
    result, _ = run(studier, monkeypatch, problem(quote), briefing(quote, confidence))
    assert f"## Confiança\n{confidence} — fontes independentes." in result
    assert NO_PROOF in result


@pytest.mark.parametrize("proven", [False, True])
def test_review_last_study_digest_and_ui_with_new_section(studier, monkeypatch, tmp_path, proven):
    from datetime import datetime, timezone
    from src.core import digest
    from src.core.orchestrator import CortexOrchestrator
    from src.core.studies_review import StudiesReview
    from src.ui.aggregator import studier_snapshot

    result, _ = run(studier, monkeypatch, problem() if proven else "PROBLEMA | 3 ; 4")
    pending, approved = tmp_path / "pending", tmp_path / "approved"
    pending.mkdir()
    approved.mkdir()
    path = pending / "fastapi.md"
    path.write_text(f"# Topic: FastAPI\n\n{result}", encoding="utf-8")
    review = StudiesReview(str(pending), str(approved))
    expected = "baixa" if proven else "alta"
    assert review.list_pending()[0]["confidence"] == expected
    assert review.last_study()["summary"] == CLAIM
    assert review.get_briefing("1")[1].endswith(result)
    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch.studies_review, orch._reply_lang = review, "pt"
    reply, mode = orch._handle_last_study()
    assert mode == "STUDY_LAST"
    assert expected in reply and CLAIM in reply
    monkeypatch.setattr(digest, "_STUDIER_DIR", str(pending))
    assert digest.new_studies_since(datetime(2000, 1, 1, tzinfo=timezone.utc)) == [path.name]
    snapshot = studier_snapshot(str(tmp_path / "queue.jsonl"), str(tmp_path))
    assert snapshot["knowledge_pending"][0]["name"] == path.name
