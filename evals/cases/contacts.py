"""Contact-add dispatch cases (s55) — lock the rich "adiciona contato" trigger.

CONTACT_ADD_RE routes to a deterministic contact-save flow, so it must fire on
the add phrasings and must NOT eat lookups / the narrow email-save / chat.
Deterministic → runs in make check.
"""
from evals.core import EvalCase, EvalKind

_RE = "src.core.dispatch.triggers.CONTACT_ADD_RE"


_INFO_RE = "src.core.dispatch.triggers.CONTACT_INFO_RE"


def _re(name, text, should_match, *, desc, dotted=_RE):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="contacts",
        regex_dotted=dotted,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        description=desc,
    )


def _info(name, text, should_match, *, desc):
    return _re(name, text, should_match, desc=desc, dotted=_INFO_RE)


CASES: list[EvalCase] = [
    # ---- must fire ----
    _re("add_pt", "adiciona contato nome: Lucas, email: lucas@x.com", True,
        desc="Core PT trigger with fields."),
    _re("add_pt_vamos", "vamos adicionar um contato. nome: João", True,
        desc="Conversational PT opener."),
    _re("add_pt_cadastra", "cadastra contato Pedro Henrique", True,
        desc="'cadastra contato' + bare name."),
    _re("add_en", "add contact name: Sarah email: sarah@x.com", True,
        desc="EN trigger."),
    _re("add_en_new", "new contact name: Bob", True, desc="EN 'new contact'."),

    # ---- must NOT fire ----
    _re("lookup", "qual o email da Maria", False,
        desc="Lookup is a different intent."),
    _re("narrow_save", "salva o email da lara como f@x.com", False,
        desc="The narrow email-save form (no word 'contato') stays with CONTACT_SAVE_RE."),
    _re("greeting", "oi cortex tudo bem", False, desc="Plain chat."),

    # ---- CONTACT_INFO (phone / full card) must fire ----
    _info("info_phone", "qual o telefone do Lucas", True, desc="Phone lookup."),
    _info("info_phone_bare", "telefone do Lucas", True, desc="Bare 'telefone do X'."),
    _info("info_card", "me fala do contato Maria", True, desc="Full-card lookup."),
    _info("info_show", "mostra o contato Bob", True, desc="'mostra o contato X'."),
    _info("info_en", "what is Lucas's phone", True, desc="EN phone lookup."),
    # ---- CONTACT_INFO must NOT fire (email goes to CONTACT_LOOKUP_RE) ----
    _info("info_not_email", "qual o email da lara", False,
        desc="Email lookup is a separate regex."),
    _info("info_not_chat", "oi cortex", False, desc="Plain chat."),
]
