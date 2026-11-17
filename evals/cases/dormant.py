"""Dormant Fase-2 dispatch cases — lock the triggers for the scaffolds that
ship wired-but-OFF (DeepStudier deep-research; Dream Loop review).

The handlers are gated by CORTEX_DEEP_STUDIER / CORTEX_DREAM_LOOP (default off)
and fall through when off, so these only lock the REGEX shape: the deep/a-fundo
qualifier must be required (a plain 'pesquise X' stays with normal search).
Deterministic → runs in make check.
"""
from evals.core import EvalCase, EvalKind

_DEEP = "src.core.dispatch.deep_research.DEEP_RESEARCH_RE"
_DREAM = "src.core.dispatch.dream.DREAM_REVIEW_RE"


def _re(name, dotted, text, should_match, *, desc, group=None):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="dormant",
        regex_dotted=dotted,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        expected_regex_group=group,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- DeepStudier: deep-research trigger ----
    _re("deep_pt", _DEEP, "pesquise a fundo sobre transformers", True,
        group="transformers", desc="PT deep qualifier required + topic captured."),
    _re("deep_pt_profunda", _DEEP, "pesquisa profunda sobre o conflito", True,
        desc="'pesquisa profunda' phrasing."),
    _re("deep_en", _DEEP, "deep research on quantization", True,
        group="quantization", desc="EN trigger."),
    _re("deep_not_plain", _DEEP, "pesquise o clima de hoje", False,
        desc="Plain search (no deep qualifier) stays with normal routing."),
    _re("deep_not_chat", _DEEP, "oi cortex", False, desc="Plain chat."),

    # ---- Dream Loop: review trigger ----
    _re("dream_show", _DREAM, "me conta o último sonho", True, desc="Show dream."),
    _re("dream_keep", _DREAM, "guarda esse sonho", True, desc="Keep dream."),
    _re("dream_drop", _DREAM, "descarta o sonho", True, desc="Discard dream."),
    _re("dream_en", _DREAM, "show me the dream", True, desc="EN show."),
    _re("dream_not_statement", _DREAM, "sonho com viajar", False,
        desc="A statement about dreaming is not a review command."),
    _re("dream_not_chat", _DREAM, "oi cortex", False, desc="Plain chat."),
]
