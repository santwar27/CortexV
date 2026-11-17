"""Entity-brain dispatch cases (Onda 5) — lock the people-graph triggers.

Both regexes route to a deterministic people-graph reply, so a false positive
would hijack an unrelated turn. Lock both sides. Deterministic → make check.
"""
from evals.core import EvalCase, EvalKind

_LAST = "src.core.dispatch.entities.LAST_CONTACT_RE"
_RECENT = "src.core.dispatch.entities.RECENT_PEOPLE_RE"


def _re(name, dotted, text, should_match, *, desc, group=None):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="entities",
        regex_dotted=dotted,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        expected_regex_group=group,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- LAST_CONTACT must fire (+ capture the name) ----
    _re("last_basic", _LAST, "quando falei com João pela última vez", True,
        group="João", desc="Core PT trigger, captures the name."),
    _re("last_ultima_vez", _LAST, "quando foi a última vez que falei com Maria", True,
        group="Maria", desc="Longer PT phrasing."),
    _re("last_en", _LAST, "when did I last talk to Sarah", True,
        group="Sarah", desc="EN trigger."),

    # ---- LAST_CONTACT must NOT fire ----
    _re("last_imperative", _LAST, "fala com o João pra mim", False,
        desc="Imperative 'talk to' is a command, not a recall query."),
    _re("last_future", _LAST, "quero falar com a Maria amanhã", False,
        desc="Future intent, not a recall query."),

    # ---- RECENT_PEOPLE must fire ----
    _re("recent_com_quem", _RECENT, "com quem andei falando", True,
        desc="Core PT trigger."),
    _re("recent_pessoas", _RECENT, "pessoas recentes", True, desc="Short PT trigger."),
    _re("recent_en", _RECENT, "recent contacts", True, desc="EN trigger."),

    # ---- RECENT_PEOPLE must NOT fire ----
    _re("recent_greeting", _RECENT, "oi cortex tudo bem", False, desc="Plain greeting."),
]
