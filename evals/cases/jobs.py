"""Job Hunter surface dispatch behaviour (slice 4, s76)."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- ask for the top matches (read-only surface) -------
    EvalCase(
        name="jobs_top_vaga_nova_pt",
        kind=EvalKind.DISPATCH,
        category="jobs",
        input="tem vaga nova?",
        expected_mode="JOBS_TOP",
        expected_matcher="JOBS_TOP_RE",
        tags=["jobs"],
    ),
    EvalCase(
        name="jobs_top_melhores_pt",
        kind=EvalKind.DISPATCH,
        category="jobs",
        input="quais as minhas melhores vagas",
        expected_mode="JOBS_TOP",
        expected_matcher="JOBS_TOP_RE",
        tags=["jobs"],
    ),
    EvalCase(
        name="jobs_top_pra_mim_pt",
        kind=EvalKind.DISPATCH,
        category="jobs",
        input="vagas pra mim",
        expected_mode="JOBS_TOP",
        expected_matcher="JOBS_TOP_RE",
        tags=["jobs"],
    ),
    EvalCase(
        name="jobs_top_any_new_en",
        kind=EvalKind.DISPATCH,
        category="jobs",
        input="any new jobs for me",
        expected_mode="JOBS_TOP",
        expected_matcher="JOBS_TOP_RE",
        tags=["jobs"],
    ),
    EvalCase(
        name="jobs_top_best_en",
        kind=EvalKind.DISPATCH,
        category="jobs",
        input="my best jobs",
        expected_mode="JOBS_TOP",
        expected_matcher="JOBS_TOP_RE",
        tags=["jobs"],
    ),
]
