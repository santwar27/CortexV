"""Task 129: local dates, deterministic periods, and code-owned confidence.

Only canned model/search responses and temporary output; no live services.
"""
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.deep_studier import DeepStudier
from src.core.studier import NightlyStudier


@pytest.fixture
def clock(monkeypatch):
    from src.core import study_time

    clock = Mock(return_value=date(2026, 11, 4))
    monkeypatch.setattr(study_time, "local_today", clock)
    return clock


@pytest.fixture
def studier(tmp_path):
    s = NightlyStudier.__new__(NightlyStudier)
    s.engine = SimpleNamespace(model_name="fake-model")
    s.output_dir = str(tmp_path)
    s._maybe_reflect = Mock()
    return s


@pytest.mark.parametrize("period, expected", [
    ("últimos dois anos", (2025, 2026)),
    ("últimos 2 anos", (2025, 2026)),
    ("últimos três anos", (2024, 2025, 2026)),
    ("últimos 4 anos", (2023, 2024, 2025, 2026)),
    ("last two years", (2025, 2026)),
    ("last 2 years", (2025, 2026)),
    ("last 4 years", (2023, 2024, 2025, 2026)),
    ("nos últimos anos", (2024, 2025, 2026)),
    ("este ano", (2026,)),
    ("this year", (2026,)),
    ("ano passado", (2025,)),
    ("last year", (2025,)),
    ("recentes", (2025, 2026)),
    ("recente", (2025, 2026)),
    ("novidades", (2025, 2026)),
    ("latest", (2025, 2026)),
    ("recent", (2025, 2026)),
    ("new features", (2025, 2026)),
    ("de 2024 a 2026", (2024, 2025, 2026)),
    ("2024–2026", (2024, 2025, 2026)),
    ("between 2024 and 2026", (2024, 2025, 2026)),
    ("novidades de 2020 a 2022", (2020, 2021, 2022)),
    ("moderno para landing pages", ()),
    ("latestish recently", ()),
])
def test_periods_are_resolved_before_the_model(clock, period, expected):
    from src.core.study_time import StudyTime

    assert StudyTime.for_topic(f"CSS {period}").years == expected


@pytest.mark.parametrize("period, expected", [
    ("este ano", (2027,)),
    ("last two years", (2026, 2027)),
])
def test_january_uses_the_local_year(clock, period, expected):
    from src.core.study_time import StudyTime

    clock.return_value = date(2027, 1, 1)
    assert StudyTime.for_topic(f"macOS {period}").years == expected


@pytest.mark.parametrize("answer", [
    "macOS High Sierra novidades\nmacOS Mojave atualizações\nmacOS 2025 features",
    "macOS 2017 novidades",
    "",
    "MACOS (Made Up Operating System)",
    RuntimeError("offline"),
])
def test_even_old_queries_and_fallbacks_get_a_range_year(
    clock, studier, monkeypatch, answer,
):
    def generate(**kwargs):
        assert "The topic covers 2025 and 2026." in kwargs["prompt"]
        if isinstance(answer, Exception):
            raise answer
        return {"response": answer}

    monkeypatch.setattr("src.core.studier.ollama.generate", generate)
    queries = studier.generate_queries("Novidades do macOS dos últimos dois anos")
    assert queries
    assert all("2025" in q or "2026" in q for q in queries)
    if isinstance(answer, str) and answer.startswith("macOS High Sierra"):
        assert queries[0] == "macOS High Sierra novidades 2026"
    if answer == "macOS 2017 novidades":
        assert queries == ["macOS 2017 novidades 2026"]
    if answer == "MACOS (Made Up Operating System)":
        assert queries == ["Novidades do macOS dos últimos dois anos 2026"]


def test_explicit_range_is_in_query_prompt(clock, studier, monkeypatch):
    model = Mock(return_value={"response": "Brasil acontecimentos"})
    monkeypatch.setattr("src.core.studier.ollama.generate", model)
    studier.generate_queries(
        "Principais acontecimentos de 2024 a 2026 no Reino Unido e no Brasil")
    assert "The topic covers 2024, 2025 and 2026." in model.call_args.kwargs["prompt"]


def test_undated_topic_queries_are_unchanged(clock, studier, monkeypatch):
    queries = ["CSS flexbox landing pages", "CSS grid landing pages", "CSS responsive landing pages"]
    model = Mock(return_value={"response": "\n".join(queries)})
    monkeypatch.setattr("src.core.studier.ollama.generate", model)
    assert studier.generate_queries("CSS moderno para landing pages") == queries
    # TODAY still belongs in every prompt; only the period/year injection is absent.
    assert "The topic covers" not in model.call_args.kwargs["prompt"]


def assert_dated(prompt):
    assert "TODAY: 2026-11-04 (Wednesday)." in prompt
    assert "knowledge ends around 2023" in prompt
    assert "only from the snippets" in prompt


def assert_event_rules(prompt):
    assert "up to today as past" in prompt
    assert '"planned for" a date before today is not evidence it happened' in prompt
    assert 'after today as "previsto"' in prompt


def test_one_clock_read_for_query_summary_and_critic(clock, studier, monkeypatch):
    prompts = []
    answers = iter(["macOS High Sierra novidades", "## Confiança\nalta — solid.", "OK | fine"])

    def generate(**kwargs):
        prompts.append(kwargs["prompt"])
        # Simulate crossing midnight after the first prompt.
        clock.return_value = date(2026, 11, 5)
        return {"response": next(answers)}

    monkeypatch.setattr("src.core.studier.ollama.generate", generate)
    studier.gather_snippets = Mock(return_value="A macOS release was planned for 2025.")
    assert studier._study_one("Novidades do macOS dos últimos dois anos")
    assert clock.call_count == 1
    assert len(prompts) == 3
    for prompt in prompts:
        assert_dated(prompt)
    assert_event_rules(prompts[1])
    assert_event_rules(prompts[2])


def test_deep_study_keeps_parent_date_and_period_in_every_step(
    clock, studier, tmp_path, monkeypatch,
):
    prompts = []
    answers = iter([
        "What changed in macOS?\nHow did macOS security change?",
        "macOS security", "A finding.",
        "macOS updates", "Another finding.", "Deep briefing.",
    ])

    def generate(**kwargs):
        prompts.append(kwargs["prompt"])
        clock.return_value = date(2026, 11, 5)
        return {"response": next(answers)}

    monkeypatch.setattr("src.core.studier.ollama.generate", generate)
    studier.gather_snippets = Mock(return_value="A macOS snippet.\n[https://example.org/macos]")
    deep = DeepStudier(studier=studier, output_dir=str(tmp_path))
    report = deep.research("Novidades do macOS dos últimos dois anos")
    assert clock.call_count == 1
    assert len(prompts) == 6
    for prompt in prompts:
        assert_dated(prompt)
    for index in (0, 1, 3):
        assert "The topic covers 2025 and 2026." in prompts[index]
    for index in (2, 4, 5):
        assert_event_rules(prompts[index])
    assert all("2026" in q for step in report.steps for q in step.queries)


@pytest.mark.parametrize("level, domains, expected", [
    ("alta", 0, "baixa"), ("alta", 1, "baixa"),
    ("média", 1, "baixa"), ("alta", 2, "média"),
])
def test_floor_replaces_the_entire_justification(level, domains, expected):
    sources = "\n".join(f"- [https://source{i}.org/a]" for i in range(domains))
    briefing = f"## Fontes consultadas\n{sources}\n\n## Confiança\n{level} — excellent sources.\n"
    result = NightlyStudier._enforce_confidence_floor(briefing)
    assert "excellent sources" not in result
    assert f"{expected} — poucas fontes independentes ({domains} domínio(s))." in result


def test_critic_replaces_justification(studier, monkeypatch):
    claim = "Todos os pedidos são sempre processados sem falhas."
    monkeypatch.setattr("src.core.studier.ollama.generate", Mock(
        return_value={"response": f"PROBLEMA | absoluta | {claim} | "
                      "A afirmação promete ausência de falhas sem apresentar evidência."}))
    result = studier._apply_critic(
        "CSS", f"## Resumo executivo\n{claim}\n\n## Confiança\nalta — excellent sources.\n")
    assert "excellent sources" not in result
    assert "## Confiança\nbaixa — o crítico achou problemas (ver ## Crítico)." in result
