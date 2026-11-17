"""System-status dispatch cases (Bug B-1, s57).

Locks the regex that routes "status do sistema?" to the deterministic status
handler instead of the LLM+retrieval path (where a weakly-relevant past turn
was woven into a fabricated status — ideas_backlog Bug B). The guard cases
matter most: it must NOT eat casual greetings ("tudo bem?", "como você está?").
Deterministic → runs in make check.
"""
from evals.core import EvalCase, EvalKind

_RE = "src.core.dispatch.system_status.SYSTEM_STATUS_RE"


def _re(name, text, should_match, *, desc):
    return EvalCase(
        name=name,
        kind=EvalKind.REGEX,
        category="system_status",
        regex_dotted=_RE,
        regex_input=text,
        regex_use_search=False,
        expected_regex_match=should_match,
        description=desc,
    )


CASES: list[EvalCase] = [
    # ---- should route to the status handler ----
    _re("status_sistema", "status do sistema", True, desc="The exact reported query."),
    _re("status_cortex", "status do cortex", True, desc="Cortex noun variant."),
    _re("qual_status", "qual o status do sistema", True, desc="'qual o status' phrasing."),
    _re("como_esta_sistema", "como está o sistema", True, desc="'como está o sistema'."),
    _re("como_anda_sistema", "como anda o sistema?", True, desc="'como anda o sistema'."),
    _re("tudo_funcionando", "está tudo funcionando?", True, desc="'tudo funcionando' = system phrasing."),
    _re("tudo_rodando", "tá tudo rodando", True, desc="'tudo rodando'."),
    _re("diagnostico", "diagnóstico do sistema", True, desc="Diagnostic word."),
    _re("saude", "saúde do sistema", True, desc="'saúde do sistema'."),
    _re("system_status_en", "system status", True, desc="EN system status."),
    _re("are_you_healthy", "are you healthy?", True, desc="EN health query."),
    _re("everything_ok_en", "is everything ok?", True, desc="EN everything-ok."),
    _re("health_check_en", "health check", True, desc="EN health check."),

    # ---- guards: must NOT eat greetings / unrelated chat ----
    _re("not_greeting_oi", "oi cortex", False, desc="Greeting, not a status query."),
    _re("not_tudo_bem", "tudo bem?", False, desc="Greeting 'tudo bem' (not 'funcionando')."),
    _re("not_como_voce_esta", "como você está?", False, desc="Casual greeting, no system noun."),
    _re("not_esta_tudo_bem", "está tudo bem?", False, desc="'bem' is a greeting, not a system word."),
    _re("not_status_jogo", "qual o status do jogo do Brasil", False, desc="Status of a game, not the system."),
    _re("not_working_on", "are you working on the report", False, desc="Trailing words → falls through."),
]
