"""Nightly Studier verb dispatch."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- s83: taking topics OFF the queue -------
    EvalCase(
        name="last_study_is_read_not_remembered",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="o que voce aprendeu com seu ultimo estudo?",
        expected_mode="STUDY_LAST",
        description="Answered with SOLID and FastAPI — old approved knowledge — "
                    "because there was no route and the 8B improvised (s83).",
    ),
    EvalCase(
        name="approve_accepts_the_pt_imperative",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="aprova 1",
        expected_mode="STUDIER_APPROVE",
        description="The form the project tells the user to type did nothing: "
                    "the regex only took aprovar/aprove/approve (s83).",
    ),
    EvalCase(
        name="cancel_study_queue_bulk",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="limpar fila de estudos",
        expected_mode="STUDY_CANCELLED",
        description="Pedro changed his mind about three topics and this "
                    "phrasing had no route, so the 8B said it had tidied the "
                    "queue and had not.",
    ),
    EvalCase(
        name="cancel_study_by_number",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="exclui o estudo 2",
        expected_mode="STUDY_CANCELLED",
    ),
    EvalCase(
        name="adversarial_exclui_tudo_only_asks",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="exclui tudo",
        expected_mode="STUDY_LIST",
        description="Delete what — notes, emails, files, the queue? It asks "
                    "instead of acting, which is what the 8B should have done "
                    "instead of refusing with an invented policy.",
    ),
    EvalCase(
        name="study_add",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="estude rede neural",
        expected_mode="STUDIER_ADD",
    ),
    EvalCase(
        name="list_studies",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="lista os estudos",
        expected_mode="STUDIER_LIST",
    ),
    EvalCase(
        name="list_studies_courteous_lead_in",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="me de a lista de estudo que esta na fila",
        expected_mode="STUDIER_LIST",
        description="Pedro's exact live phrasing (s78, 2026-07-09) that fell "
                    "to the LLM and hallucinated — lead-in + 'lista DE estudo'.",
    ),
    EvalCase(
        name="list_studies_fila_singular",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="fila de estudo",
        expected_mode="STUDIER_LIST",
    ),
    EvalCase(
        name="review_studies",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="revisar estudos",
        expected_mode="STUDIER_REVIEW",
    ),
    EvalCase(
        name="show_study",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="ver estudo 1",
        expected_mode="STUDIER_SHOW",
        expected_matcher="SHOW_STUDY_RE",
        expected_capture="1",
        description="Open a specific study by ref — capture passes the ref through.",
    ),
    EvalCase(
        name="approve_study",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="aprovar 1",
        expected_mode="STUDIER_APPROVE",
        expected_matcher="APPROVE_STUDY_RE",
        expected_capture="1",
        description="Approving a pending study by ref — review workflow.",
    ),
    EvalCase(
        name="reject_study",
        kind=EvalKind.DISPATCH,
        category="studier",
        input="rejeitar 2",
        expected_mode="STUDIER_REJECT",
        expected_matcher="REJECT_STUDY_RE",
        expected_capture="2",
        description="Rejecting a pending study by ref — review workflow.",
    ),
]
