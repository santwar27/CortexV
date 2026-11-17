"""Digest cases (Onda 4a) — lock the "o que mudou" dispatch trigger.

WHATS_CHANGED_RE must fire on "catch me up" phrasings and must NOT eat normal
conversation (it routes to a deterministic pull-based digest, so a false
positive would hijack an unrelated turn). Deterministic → runs in `make check`.
"""
from evals.core import EvalCase, EvalKind

_RE = "src.core.dispatch.digest.WHATS_CHANGED_RE"


def _wc(name, text, should_match, *, desc):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="digest",
        regex_dotted=_RE,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- MUST fire ----
    _wc("o_que_mudou", "o que mudou", True, desc="Core PT trigger."),
    _wc("o_que_mudou_desde", "o que mudou desde ontem", True, desc="With a 'desde' tail."),
    _wc("novidades", "novidades?", True, desc="One-word PT trigger."),
    _wc("me_atualiza", "me atualiza", True, desc="'catch me up' PT."),
    _wc("o_que_perdi", "o que eu perdi", True, desc="'what did I miss' PT."),
    _wc("whats_new", "what's new", True, desc="EN trigger."),
    _wc("what_changed", "what changed", True, desc="EN trigger."),
    _wc("what_did_i_miss", "what did I miss", True, desc="EN trigger."),
    _wc("catch_me_up", "catch me up", True, desc="EN idiom."),

    # ---- MUST NOT fire ----
    _wc("explain_change", "explica o que mudou no clima desde a era do gelo", False,
        desc="A topic about change is not a catch-me-up command."),
    _wc("how_to_change", "como eu mudo a senha do wifi", False,
        desc="Unrelated 'change' verb."),
    _wc("new_project", "quero começar um projeto novo", False,
        desc="'novo' mid-sentence must not match 'novidades'."),
    _wc("greeting", "oi cortex tudo bem", False, desc="Plain greeting."),
]
