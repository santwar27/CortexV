"""Prompt enrichment v1 (idea #30, s74).

What these cases pin, per the approved design:
  - the explicit prefix routes deterministically (mode REFINE) — the
    enrichment is opt-in per turn, so the ROUTE must be regex-first,
    never an LLM guess.
  - normal sentences that merely start with "refina"/contain the word
    do NOT trigger: the separator is part of the contract. Automatic
    enrichment sneaking in through a loose regex would reintroduce the
    latency/voice risk v1 was designed to avoid.

The staged sim/não behaviour itself is covered by tests
(tests/test_refine.py) — the handler calls the live LLM, so a
DISPATCH_REPLY here would violate SOBERANIA regra 6 (CI deterministic).
"""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    EvalCase(
        name="refine_route_pt", kind=EvalKind.DISPATCH, category="refine",
        input="refina: me ajuda com treino",
        expected_mode="REFINE", expected_matcher="REFINE_RE",
        description="Explicit prefix routes deterministically — no LLM guess.",
        tags=["regression"],
    ),
    EvalCase(
        name="refine_route_en", kind=EvalKind.DISPATCH, category="refine",
        input="refine: help me plan my week",
        expected_mode="REFINE", expected_matcher="REFINE_RE",
    ),
    EvalCase(
        name="refine_route_melhora_prompt", kind=EvalKind.DISPATCH,
        category="refine",
        input="melhora o prompt: escreve um email pro meu chefe",
        expected_mode="REFINE", expected_matcher="REFINE_RE",
    ),
    EvalCase(
        name="refine_no_separator_not_routed", kind=EvalKind.DISPATCH,
        category="refine",
        input="refina os detalhes da viagem",
        expected_mode="OFFLINE",
        description="No separator → normal sentence, never enrichment.",
        tags=["regression"],
    ),
    EvalCase(
        name="refine_different_word_not_routed", kind=EvalKind.DISPATCH,
        category="refine",
        input="refinaria de petróleo no Brasil",
        expected_mode="OFFLINE",
        description="'refinaria' is not the prefix — no false trigger.",
    ),
]
