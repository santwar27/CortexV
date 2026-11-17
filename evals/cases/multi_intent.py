"""Multi-intent auto-routing to the planner (sessão 15).

The orchestrator detects compound requests (sequential conjunction +
≥2 distinct tool domain families) and routes them to the planner
WITHOUT requiring the explicit "planeja:" prefix. This file pins
the routing decisions so a future heuristic change can't silently
re-introduce the failure mode where Llama 8B saw a 2-intent input,
picked one tool, and lost the other.

Positives test the autoroute fires; negatives test we don't drag
casual conjunctions or single-domain queries into the planner.
"""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- Positives: should route to PLANNER via multi_intent_autoroute
    EvalCase(
        name="busca_jogos_e_adiciona_calendario",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input=(
            "busca os jogos da seleção brasileira e adiciona ao meu "
            "calendário"
        ),
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        description=(
            "The exact failure mode Pedro hit live in sessão 14: "
            "web_search + create_event combo. Auto-router must catch."
        ),
        tags=["jou-s15", "regression"],
    ),
    EvalCase(
        name="manda_email_e_cria_lembrete",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="manda email pro João e cria lembrete pras 18h",
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        tags=["jou-s15"],
    ),
    EvalCase(
        name="email_e_calendario_pt",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="envia um email pro joão e agenda reunião com ele amanhã",
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        description=(
            "email + calendar in one breath — two LLM-tool domain "
            "families, autorouter should send this to the planner."
        ),
    ),
    EvalCase(
        name="search_and_add_to_calendar_en",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="search the brazil game schedule and add to my calendar",
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        description="English compound request — bare 'and' as conjunction.",
        tags=["jou-s15"],
    ),
    EvalCase(
        name="le_documento_e_manda_resumo_email",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="lê o documento agentes e me manda um resumo por email",
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        description="read_document + send_email — two-step pipeline.",
        tags=["jou-s15"],
    ),
    EvalCase(
        name="then_chain_after_search",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="find out the football schedule then add the events",
        expected_mode="PLANNER",
        expected_matcher="multi_intent_autoroute",
        description="'then' conjunction (EN) instead of 'and'.",
        tags=["jou-s15"],
    ),

    # ------- Explicit "planeja:" still wins over autoroute
    EvalCase(
        name="explicit_planeja_prefix_wins",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="planeja: manda email pro João e cria lembrete",
        expected_mode="PLANNER",
        expected_matcher="PLANNER_TRIGGER_RE",
        description=(
            "When the explicit verb is present the explicit matcher "
            "must report, even though the body would also auto-route."
        ),
        tags=["jou-s15", "conflict"],
    ),

    # ------- Negatives: must NOT autoroute
    EvalCase(
        name="casual_conjunction_no_domains",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="vamos sair e tomar uma cerveja",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description="Conjunction present but zero tool domains — pure chat.",
        tags=["jou-s15", "false_positive_guard"],
    ),
    EvalCase(
        name="single_domain_repeated_no_autoroute",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="manda email pro João e responde o email do José",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Same domain mentioned twice ('email' x2) is one intent — "
            "the LLM can handle it without the planner."
        ),
        tags=["jou-s15", "false_positive_guard"],
    ),
    EvalCase(
        name="single_intent_no_autoroute",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="busca os jogos da seleção",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Single intent, no conjunction — pure chat (well, will "
            "ask for lease via the realtime path, but routing-wise "
            "this is the fallthrough)."
        ),
        tags=["jou-s15", "false_positive_guard"],
    ),
    EvalCase(
        name="comparison_and_not_sequential",
        kind=EvalKind.DISPATCH,
        category="multi_intent",
        input="Spotify and Apple Music são plataformas diferentes",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "'and' between proper nouns, no tool domains — comparative "
            "language, not a multi-step request."
        ),
        tags=["jou-s15", "false_positive_guard"],
    ),
]
