"""Flow-control cases (M1-A4) — the confirmation reply seam.

Two layers pinned:
  • strip_vocatives must return a lone ack WHOLE ("ok." IS the message) —
    pre-fix it stripped to empty and both entry points answered
    "[SYSTEM ERROR] Empty input." with the action still staged.
  • APPROVAL_RE must claim the punctuated-ack family ("valeu!", "show!",
    "perfeito.", "okay,") — anything it misses falls to the anything-else
    branch, which DROPS the staged action silently (worse than the error).

Both are pure regex/normalisation → deterministic, run in make check.
"""
from evals.core import EvalCase, EvalKind

_APPROVAL = "src.core.dispatch.flow_control.APPROVAL_RE"
_DENIAL = "src.core.dispatch.flow_control.DENIAL_RE"


def _re(name, dotted, text, should_match, *, desc):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="flow_control",
        regex_dotted=dotted,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- punctuated acks must be approvals ----
    _re("approval_ok_dot", _APPROVAL, "ok.", True,
        desc="The canonical M1-A4 victim: 'ok.' is an approval."),
    _re("approval_okay_comma", _APPROVAL, "okay,", True,
        desc="Trailing comma is a legit ack shape."),
    _re("approval_beleza_bang", _APPROVAL, "beleza!", True,
        desc="PT-BR ack with bang."),
    _re("approval_valeu", _APPROVAL, "valeu!", True,
        desc="'valeu!' to a sim/não prompt reads as YES."),
    _re("approval_show", _APPROVAL, "show!", True,
        desc="'show!' is mineiro for yes."),
    _re("approval_perfeito", _APPROVAL, "perfeito.", True,
        desc="'perfeito.' approves."),
    _re("approval_blz", _APPROVAL, "blz", True,
        desc="SMS-grade ack."),
    _re("approval_got_it", _APPROVAL, "got it.", True,
        desc="EN ack family."),

    # ---- s76 smoke: the EN bare approvals stay covered ----
    # The live "yes didn't approve" bug was Whisper transcribing "yes" as
    # "Ah!" — APPROVAL_RE was innocent. Pin the innocence.
    _re("approval_yes_dot", _APPROVAL, "Yes.", True,
        desc="Bare EN 'Yes.' approves (s76: the regex was never the bug)."),
    _re("approval_not_garbled_ah", _APPROVAL, "Ah!", False,
        desc="Garbled STT noise must NOT approve — it triggers the "
             "one-shot re-prompt instead."),

    # ---- boundary: ack + MORE is NOT a bare approval ----
    _re("approval_not_ack_plus_command", _APPROVAL, "ok faz outra coisa", False,
        desc="Ack PREFIX with a new command must not auto-approve — the "
             "new query decides."),
    _re("approval_not_valeu_mas_cancela", _APPROVAL, "valeu, mas cancela", False,
        desc="Gratitude followed by a U-turn must not approve."),
    _re("approval_not_show_me_emails", _APPROVAL, "show me my emails", False,
        desc="EN 'show' as a verb is a command, never an approval."),

    # ---- denial side unaffected ----
    _re("denial_nao_dot", _DENIAL, "não.", True,
        desc="Punctuated denial keeps working."),
    _re("denial_esquece", _DENIAL, "esquece", True,
        desc="Colloquial cancel."),

    # ---- s75 (Onda 8 A): strip_courtesy must NOT eat approvals ----
    # "pode" is BOTH a courtesy prefix ("pode abrir o spotify") and an
    # approval ("pode crer"). strip_courtesy guards on APPROVAL_RE; these
    # pin that a bare approval routes as APPROVAL in the simulator instead
    # of being chopped to "crer"/"mandar" and falling to chat.
    EvalCase(
        name="courtesy_guard_pode_crer",
        kind=EvalKind.DISPATCH, category="flow_control",
        input="pode crer",
        expected_mode="APPROVAL",
        description="Approval survives the courtesy strip untouched.",
    ),
    EvalCase(
        name="courtesy_guard_pode_mandar",
        kind=EvalKind.DISPATCH, category="flow_control",
        input="pode mandar",
        expected_mode="APPROVAL",
        description="Approval survives the courtesy strip untouched.",
    ),
]
