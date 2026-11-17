"""Golden-recall cases — guard the s39 retrieval fix (personal-field recall).

Runs each query through CortexMemory.retrieve_relevant_context over the
synthetic-profile fixture (evals/recall_fixture.py) and asserts the right
section/value entered the retrieved context. This is the permanent regression
net for the alpha / n_results / chunking tuning that took profile recall from
1/8 to 8/8.

Two tiers:
  • RETRIEVAL (deterministic, in `make check`) — queries share keywords with
    the target section, so lexical-only BM25/FTS5 retrieval is enough. A miss
    is a FAIL: the always-on smoke that catches a gross retrieval regression.
  • RETRIEVAL_SEMANTIC (opt-in, `make eval-llm`) — PARAPHRASE queries with no
    keyword overlap, so only the embedder can bridge them. Drift-tolerant
    (embeddings aren't bit-reproducible); the same cases will grade a stronger
    embedder later without edits.

Expected values come from PROFILE_CORPUS in the fixture (all synthetic).
"""
from evals.core import EvalCase, EvalKind


def _fts(name, query, expected, *, desc, polluted=False):
    return EvalCase(
        name=name,
        kind=EvalKind.RETRIEVAL,
        category="recall",
        retrieval_query=query,
        retrieval_expected=expected,
        retrieval_polluted=polluted,
        description=desc,
    )


def _sem(name, query, expected, *, desc):
    return EvalCase(
        name=name,
        kind=EvalKind.RETRIEVAL_SEMANTIC,
        category="recall",
        retrieval_query=query,
        retrieval_expected=expected,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- deterministic lexical smoke (keyword overlap) — runs in make check ----
    _fts(
        "fts_passport",
        "qual o número do meu passaporte?", "AB1234567",
        desc="Passport section must surface on a keyword-overlap query.",
    ),
    _fts(
        "fts_city",
        "qual a minha cidade?", "Cidade Exemplar",
        desc="Location section must surface.",
    ),
    _fts(
        "fts_role",
        "qual a minha função no trabalho?", "engenheiro",
        desc="Identity/role section must surface.",
    ),
    _fts(
        "fts_health",
        "qual o meu número de saúde?", "HX-999-000",
        desc="Health-ID section must surface.",
    ),

    # ---- M3-A2: polluted-convo battery (echo/crowding dynamic) ----
    # The fixture's polluted variant stores 6 verbose past copies of the
    # passport question as conversation turns (user-side echoes carry no
    # value, s40) plus a decoy knowledge section that outranks the real one.
    # Pre-M3-A2 (conversation first in the merge, no echo dedup) the real
    # value fell past the total context cap — asking again made recall worse,
    # and nothing pinned it.
    _fts(
        "fts_passport_polluted_convo",
        "qual o número do meu passaporte?", "AB1234567",
        polluted=True,
        desc="Passport value must survive repeated-question echoes + decoy "
             "section crowding the context cap (M3-A2 regression net).",
    ),
    _fts(
        "fts_health_polluted_convo",
        "qual o meu número de saúde?", "HX-999-000",
        polluted=True,
        desc="Unrelated pollution (passport echoes) must not break OTHER "
             "field recalls.",
    ),

    # ---- semantic battery (paraphrase, no keyword) — opt-in make eval-llm ----
    _sem(
        "sem_passport_paraphrase",
        "qual o número do meu documento de viagem?", "AB1234567",
        desc="'documento de viagem' → passport: needs embeddings (no keyword).",
    ),
    _sem(
        "sem_city_paraphrase",
        "em que lugar eu resido atualmente?", "Cidade Exemplar",
        desc="'lugar onde resido' → location section: semantic bridge.",
    ),
    _sem(
        "sem_health_paraphrase",
        "qual é o meu identificador do sistema de saúde?", "HX-999-000",
        desc="Paraphrase of the health-ID field: semantic bridge.",
    ),
]
