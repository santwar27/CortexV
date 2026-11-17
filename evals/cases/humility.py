"""Humility cases (Onda 3c) — lock the volatile-fact detector.

VOLATILE_FACT_RE catches the fresh-fact categories the LLM realtime-classifier
misses (sports results, live prices, recent news), so they route to the
"quer que eu pesquise?" offer instead of a confabulated answer. These REGEX
cases bound BOTH sides: it must fire on volatile questions and must NOT fire
on stable general-knowledge (the false-positive cost is offering a needless
search). Deterministic → runs inside `make check`.

The definitional edge ("o que é placar" contains 'placar') is handled in
_needs_realtime by the OFFLINE_HINT_RE fast-path running first, not here —
that interaction is covered in tests/test_humility.py.
"""
from evals.core import EvalCase, EvalKind

_RE = "src.core.dispatch.heuristics.VOLATILE_FACT_RE"


def _vol(name, text, should_match, *, desc):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="humility",
        regex_dotted=_RE,
        regex_input=text,
        regex_use_search=True,
        expected_regex_match=should_match,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- MUST fire (volatile fact → offer search, don't confabulate) ----
    _vol("won_match", "quem ganhou o jogo do Flamengo", True,
         desc="Sports result without a time word — the classic miss."),
    _vol("placar", "qual foi o placar do jogo ontem", True,
         desc="Score question."),
    _vol("resultado_partida", "me diz o resultado da partida de ontem", True,
         desc="Match result phrasing."),
    _vol("cotacao", "qual a cotação do dólar", True,
         desc="Exchange rate / live quote."),
    _vol("preco_bitcoin", "quanto está o bitcoin", True,
         desc="Live crypto price."),
    _vol("preco_dolar", "qual o preço do dólar", True,
         desc="Live currency price."),
    _vol("who_won_en", "who won the game last night", True,
         desc="English sports result."),
    _vol("latest_news", "quais as últimas notícias sobre IA", True,
         desc="Recent news."),

    # ---- MUST NOT fire (stable knowledge → stay offline, no search offer) ----
    _vol("capital", "qual a capital da França", False,
         desc="Stable fact — never offer a search."),
    _vol("turing", "quem foi Alan Turing", False,
         desc="History — 'quem foi' must not look like 'quem ganhou'."),
    _vol("photosynthesis", "explica a fotossíntese", False,
         desc="Definitional/offline."),
    _vol("pointers", "como funcionam ponteiros em C", False,
         desc="Programming concept."),
    _vol("preco_generic", "qual o preço de um carro popular", False,
         desc="'preço' without a concrete asset token must not match."),
    _vol("math", "quanto é 15 vezes 7", False,
         desc="Arithmetic is not a volatile fact."),
]
