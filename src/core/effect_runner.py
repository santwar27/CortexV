"""Record intent before an approved effect; never retry an uncertain outcome."""
from typing import Tuple

from src.core import effect_store
from src.core.audit import audit
from src.core.i18n import t


class EffectUncertain(Exception):
    """The effect may have happened even though its reply was lost."""


def run_effect(acao, args, call, *, origin=None, effect_id=None,
               cancelled=None, lang="pt", store=None) -> Tuple[str, str]:
    """Record, check Stop, mark running, then call so intent and cancellation precede effects."""
    store = effect_store.store if store is None else store
    unavailable = (t(lang,
        "Não consegui registrar essa ação antes de executar — não fiz nada.",
        "I couldn't record that action before running it — I did nothing."),
        "AGENT_ACTION_ERROR")
    effect = store.create(acao, args, origin, effect_id)
    payload = {"id": effect.id if effect is not None else effect_id, "action": acao}
    if effect is None:
        audit.log("–", "effects", "not_recorded", payload=payload)
        return unavailable
    if cancelled is not None and cancelled():
        store.transition(effect.id, "preparando", "cancelado")
        audit.log("–", "effects", "cancelled", payload=payload)
        return (t(lang, "Parei antes de executar — não fiz nada.",
                  "Stopped before running it — nothing was done."), "AGENT_ACTION_CANCELLED")
    if not store.transition(effect.id, "preparando", "executando"):
        audit.log("–", "effects", "not_recorded", payload=payload)
        return unavailable
    try:
        reply, mode = call()
        state = "feito" if mode == "AGENT_ACTION_DONE" else "falhou"
        detail = reply
    except Exception as exc:
        state, detail = "desconhecido", str(exc) or type(exc).__name__
        if isinstance(exc, EffectUncertain):
            reply = detail[:1].upper() + detail[1:] + t(lang,
                " Confere antes de pedir de novo — não vou repetir sozinho.",
                " Please check before asking again — I won't retry on my own.")
        else:
            reply = t(lang,
                f"Não sei se deu certo ({detail}). Confere antes de pedir de novo — "
                "não vou repetir sozinho.",
                f"I don't know whether it went through ({detail}). Please check before "
                "asking again — I won't retry on my own.")
        mode = "AGENT_ACTION_ERROR"
    # Losing the result must not change what the user hears about the effect.
    if not store.transition(effect.id, "executando", state, detail):
        audit.log("–", "effects", "result_not_recorded", payload=payload)
    return reply, mode
