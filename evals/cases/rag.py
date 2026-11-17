"""RAG anti-hallucination cases (s73 — Copa 2026 bug, trace 48240c50).

The real failure: "qual o próximo jogo da seleção na Copa 2026?" answered
the SECOND fixture (Haiti, 19/06) from snippets that clearly listed the
debut (Marrocos, 13/06) first. The snippets were correct — the model
failed the temporal selection ("next" = earliest future event).

Two layers, mirroring SOBERANIA rule 6:
  - RAG_PROMPT (deterministic, runs in `make check`): pins the prompt
    anchors that fix the bug — TODAY'S DATE adjacent to the data and the
    explicit temporal-selection instruction. Pure string assembly.
  - LLM_BEHAVIOR (opt-in, `make eval-llm`): the behavioural truth — with
    the frozen snippets and a frozen date, the reply must name Marrocos
    and must NOT name Haiti. Fato presente no snippet ≠ fato na reply.

The snippet block below is condensed from the REAL Tavily replay of the
bug query (2026-06-12) and the date is frozen at the day of the bug, so
these cases never rot as the tournament moves on.
"""
from evals.core import EvalCase, EvalKind
from src.core.orchestrator import build_rag_prompt

COPA_QUERY = (
    "qual o proximo jogo da seleção brasileira na copa do mundo da fifa 2026"
)

COPA_TODAY = "2026-06-11 (Thursday)"

# Condensed from the real replay — all three fixtures present, debut first,
# exactly the shape that tripped the 8B.
COPA_SNIPPETS = (
    "QUICK ANSWER: Brasil estreia na Copa do Mundo 2026 no dia 13 de junho "
    "contra Marrocos. Os outros jogos são contra Haiti em 19 de junho e "
    "Escócia em 24 de junho.\n\n"
    "Source 1 (Copa do Mundo: tabela dos jogos do Brasil):\n"
    "O Brasil estreia no dia 13, contra o Marrocos. A estreia da Seleção "
    "será diante dos marroquinos, no dia 13 de junho (sábado). O duelo "
    "contra o Haiti será no dia 19 (sexta-feira), e o confronto com a "
    "Escócia, no dia 24 (quarta-feira).\n[https://ge.globo.com/...]\n\n"
    "Source 2 (Brasil na Copa do Mundo 2026):\n"
    "O Brasil está no Grupo C da Copa do Mundo 2026, ao lado de Marrocos, "
    "Haiti e Escócia. O Brasil joga nos dias 13 (sábado), 19 (sexta) e 24 "
    "(quarta) de junho de 2026.\n[https://www.gazetadopovo.com.br/...]"
)

# "último jogo" mirror of the Copa case (s80, the user): a Flamengo "resultado
# do último jogo" turn picked a game somewhat at random from a messy fixture
# list — the prompt taught "próximo" (earliest future) but said nothing about
# "último" (latest PAST). Frozen clean scenario: today 2026-07-20, the correct
# answer is Palmeiras (18/07, most recent already-played), NOT the future
# Corinthians fixture (25/07), NOT the older games, and NOT the misleading
# QUICK ANSWER that contradicts the dated list.
ULTIMO_QUERY = "qual foi o resultado do último jogo do flamengo"
ULTIMO_TODAY = "2026-07-20 (Monday)"
ULTIMO_SNIPPETS = (
    "QUICK ANSWER: O último jogo do Flamengo foi contra o Santos, empate 0 a 0.\n\n"
    "Source 1 (Últimos jogos do Flamengo - placar):\n"
    "Flamengo 2 - 1 Palmeiras — SÁB, 18/07 Campeonato Brasileiro\n"
    "Flamengo 0 - 0 Santos — SÁB, 11/07 Campeonato Brasileiro\n"
    "Flamengo 3 - 1 Grêmio — SÁB, 04/07 Campeonato Brasileiro\n"
    "[https://example.com/flamengo/ultimos]\n\n"
    "Source 2 (Próximos jogos do Flamengo):\n"
    "Flamengo x Corinthians — SÁB, 25/07 (próximo jogo)\n"
    "[https://example.com/flamengo/proximos]"
)

# Stale-'next' game (the user's real 2026-07-27 turn): the search index lags a
# day, so a game that was played YESTERDAY is still labelled "próximo" (next)
# and has no score yet. Today is 2026-07-27; São Paulo (26/07) already
# happened, so it — not the older Chapecoense (23/07) — is the most recent.
# The correct reply names São Paulo and admits the score isn't in the data.
STALE_QUERY = "qual foi o resultado do último jogo do flamengo"
STALE_TODAY = "2026-07-27 (Monday)"
STALE_SNIPPETS = (
    "QUICK ANSWER: O último jogo do Flamengo foi contra a Chapecoense, "
    "vitória por 4 a 0, em 23 de julho de 2026. O próximo jogo é contra o "
    "São Paulo em 26 de julho de 2026.\n\n"
    "Source 1 (365Scores): O último jogo do Flamengo foi Chapecoense 0 - 4 "
    "Flamengo.\n[https://example.com/365]\n\n"
    "Source 2 (Próximos jogos do Flamengo): Flamengo x São Paulo — "
    "próximo jogo, 26/07.\n[https://example.com/proximos]"
)

# Snippets that do NOT contain what the query asks for — the model must
# admit the gap instead of filling it from memory (s34 hedging contract).
GAPPED_SNIPPETS = (
    "Source 1 (Copa do Mundo 2026: sedes e estádios):\n"
    "A Copa do Mundo 2026 será realizada em 16 cidades dos EUA, México e "
    "Canadá. A final será no MetLife Stadium, em New Jersey, no dia 19 de "
    "julho.\n[https://example.org/sedes]"
)


CASES: list[EvalCase] = [
    # ------- deterministic (make check): pin the prompt anchors -------
    EvalCase(
        name="rag_prompt_compares_like_attributes_and_preserves_currency",
        kind=EvalKind.RAG_PROMPT,
        category="rag",
        input="compare A and B",
        rag_web_context="A: charging 60 W. B: battery 8 hours.",
        rag_today="2026-11-08 (Sunday)",
        expected_contains_all=[
            "COMPARISONS: Compare the same attribute; never call one better based on different attributes",
            "Each fact belongs only to the item the source names for it",
            "Never copy a number or feature to the other item",
            'use "também"/"both" only if the data states it for both',
            "say the data does not give it for the other",
            "Data: A charges at Y W; B lasts X hours.",
            "WRONG: A e B duram X horas.",
            "RIGHT: A carrega a Y W; B dura X horas; os dados não dão a duração de A, então não dá para dizer qual é melhor.",
            "Never convert or estimate currencies",
            "keep the currency stated by the source",
            "Use metric units consistently.",
        ],
        expected_not_contains="PRICES:",
        description="Pin comparison and currency policy without a configured market.",
        tags=["regression", "anti_hallucination"],
    ),
    EvalCase(
        name="rag_prompt_anchors_date_and_temporal_rule",
        kind=EvalKind.RAG_PROMPT,
        category="rag",
        input=COPA_QUERY,
        rag_web_context=COPA_SNIPPETS,
        rag_today=COPA_TODAY,
        expected_contains_all=[
            "today's date: 2026-06-11",
            "next/upcoming event",
            "próximo",
            "earliest",
            "live web data",
        ],
        description=(
            "The s73 fix anchors: TODAY'S DATE restated adjacent to the "
            "snippets + explicit 'pick the earliest future event' rule. "
            "If someone rewrites the RAG prompt and drops either anchor, "
            "this fails."
        ),
        tags=["regression", "anti_hallucination"],
    ),
    EvalCase(
        name="rag_prompt_rewrites_proximo_and_keeps_snippets",
        kind=EvalKind.RAG_PROMPT,
        category="rag",
        input=COPA_QUERY,
        rag_web_context=COPA_SNIPPETS,
        rag_today=COPA_TODAY,
        expected_contains_all=[
            "marrocos",
            "haiti",
            # The deterministic query rewrite — the piece that actually flips
            # the s73 answer (the 8B reads "próximo" as "second in the list").
            "primeiro (contando a partir de hoje) jogo da seleção brasileira",
        ],
        description=(
            "Snippets reach the prompt unmangled; the query reaches it with "
            "'proximo' rewritten to 'primeiro (contando a partir de hoje)'."
        ),
        tags=["regression", "anti_hallucination"],
    ),

    EvalCase(
        name="rag_prompt_anchors_ultimo_temporal_rule",
        kind=EvalKind.RAG_PROMPT,
        category="rag",
        input=ULTIMO_QUERY,
        rag_web_context=ULTIMO_SNIPPETS,
        rag_today=ULTIMO_TODAY,
        expected_contains_all=[
            "today's date: 2026-07-20",
            "last/most-recent event",
            "latest date that is on or before",
            "trust the dated list",
            # stale-'next' rule (the user, 2026-07-27): a fixture labelled next
            # whose date already passed has ALREADY been played.
            "already been played",
        ],
        description=(
            "s80 fix anchors: the RAG prompt must teach 'último' (latest PAST "
            "event, not a future fixture) alongside 'próximo', trust the dated "
            "list over a conflicting QUICK ANSWER, and treat a 'next' fixture "
            "whose date already passed as ALREADY PLAYED (web indexes lag)."
        ),
        tags=["regression", "anti_hallucination"],
    ),

    # ------- opt-in (make eval-llm): the behavioural truth -------
    EvalCase(
        name="rag_ultimo_game_is_palmeiras_not_future_or_quick_answer",
        kind=EvalKind.LLM_BEHAVIOR,
        category="rag",
        input=build_rag_prompt(ULTIMO_SNIPPETS, ULTIMO_QUERY, today_line=ULTIMO_TODAY),
        expected_contains_all=["palmeiras"],
        expected_not_contains="corinthians",   # never the future fixture
        language="pt",
        description=(
            "'resultado do último jogo', today 2026-07-20: MUST be Palmeiras "
            "(18/07, most recent already-played) — never the future Corinthians "
            "(25/07), and it must trust the dated list over the misleading "
            "QUICK ANSWER (Santos 0-0). The s80 temporal-selection fix."
        ),
        tags=["regression", "anti_hallucination"],
    ),
    EvalCase(
        name="rag_stale_next_game_names_sao_paulo_not_older",
        kind=EvalKind.LLM_BEHAVIOR,
        category="rag",
        input=build_rag_prompt(STALE_SNIPPETS, STALE_QUERY, today_line=STALE_TODAY),
        expected_contains_all=["são paulo"],
        language="pt",
        description=(
            "Pedro's real 2026-07-27 turn: the index lags, so São Paulo (26/07, "
            "already played) is still labelled 'próximo' with no score. Today is "
            "27/07, so São Paulo is the most recent game — the reply must NAME it "
            "(not confidently report the older Chapecoense 23/07 as the last one)."
        ),
        tags=["regression", "anti_hallucination"],
    ),
    EvalCase(
        name="rag_copa_next_game_is_marrocos_not_haiti",
        kind=EvalKind.LLM_BEHAVIOR,
        category="rag",
        input=build_rag_prompt(COPA_SNIPPETS, COPA_QUERY, today_line=COPA_TODAY),
        expected_contains_all=["marrocos"],
        expected_not_contains="haiti",
        language="pt",
        description=(
            "The exact s73 bug replay, frozen: snippets list three fixtures, "
            "today is 2026-06-11, 'próximo jogo' MUST be Marrocos (13/06) "
            "and MUST NOT be Haiti (19/06). Fact present in the snippet is "
            "not enough — it has to be the RIGHT fact for the question."
        ),
        tags=["regression", "anti_hallucination"],
    ),
    EvalCase(
        name="rag_admits_gap_when_snippets_lack_the_fact",
        kind=EvalKind.LLM_BEHAVIOR,
        category="rag",
        input=build_rag_prompt(GAPPED_SNIPPETS, COPA_QUERY, today_line=COPA_TODAY),
        expected_contains_any=[
            "não encontrei", "nao encontrei", "não consta", "não há",
            "não informam", "não menciona", "couldn't find",
        ],
        expected_not_contains="haiti",
        language="pt",
        description=(
            "Snippets about venues only — no fixture data. The reply must "
            "admit the gap (s34 hedging contract), never name an opponent "
            "from memory."
        ),
        tags=["anti_hallucination"],
    ),
]
