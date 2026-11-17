"""Deterministic previous-answer usage for saved conversations (141a/b).

No estimates, current bucket, global last turn or model-generated wording.
The deliberately narrow whole-question grammar leaves mixed actions with
their existing owner and never searches quoted/document content for intent.
"""
from __future__ import annotations

import re
import unicodedata

from src.core.dispatch.normalisation import strip_courtesy, strip_vocatives
from src.core.prompt_safety import own_words


# Only these handlers provably answer without calling a model.
CODE_ONLY_MODES = frozenset({"USAGE_PREVIOUS", "SELF_MODEL", "SELF_PLAN", "SELF_AVATAR"})


_PT = re.compile(
    r"quantos tokens (?:"
    r"(?:voce |o cortex |cortex )?(?:gastou|usou|gasta|usa|gastaste|usaste)"
    r"(?: (?:na|nesta|nessa|para a) (?:ultima |sua )?resposta"
    r"| para responder (?:essa|esta|a) pergunta)?"
    r"|(?:essa|esta|a ultima|sua ultima) resposta (?:usou|gastou)"
    r"|foram)"
)
_EN = re.compile(
    r"how many (?:input/output |input |output )?tokens (?:"
    r"(?:did|do) you (?:use|spend)"
    r"(?: (?:in|on|for) (?:the |your )?(?:last |previous )?answer"
    r"| (?:to answer|answering) (?:this|that|the) question)?"
    r"|did (?:that|this|the last|your last|your previous) answer (?:use|spend))"
)


def normalise_own_question(text: str) -> str:
    """Normalise own words for narrow, whole-question code-owned replies."""
    text = strip_vocatives((own_words(text) or "").strip())
    text = re.sub(r"^(?:please|por favor)[,\s]+", "", text, flags=re.IGNORECASE)
    text = strip_courtesy(text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))
    return " ".join(text.split()).rstrip("?!. ")


def detect_usage_question(text: str) -> str | None:
    """Return the question's language, or decline; only own words may match.

    Full matching also excludes quotes, Markdown quotations/code, explanation
    requests and any extra clause (including an email/send request).
    """
    text = normalise_own_question(text)
    if _PT.fullmatch(text):
        return "pt"
    if _EN.fullmatch(text):
        return "en"
    return None


def _unknown(lang: str) -> str:
    return ("Não tenho a contagem da última resposta." if lang == "pt"
            else "I don't have the count for the last answer.")


def _zero_call(lang: str) -> str:
    return ("A última resposta foi dada pelo código, sem usar o modelo (0 tokens)."
            if lang == "pt" else
            "The last answer was given by code, without using the model (0 tokens).")


def format_usage(turn: dict, lang: str) -> str:
    """Render recorded totals or a code-only mode's known zero-call answer."""
    # These handlers never call a model; collect() saves no usage for their turns.
    if turn.get("status_mode") in CODE_ONLY_MODES:
        return _zero_call(lang)
    usage = turn.get("usage")
    if not isinstance(usage, dict):
        return _unknown(lang)
    counts = [usage.get(key) for key in ("prompt_eval_count", "eval_count", "calls")]
    if any(type(n) is not int or n < 0 for n in counts):
        return _unknown(lang)
    incoming, outgoing, calls = counts
    models = usage.get("models", [])
    if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
        return _unknown(lang)
    models = sorted(set(models))
    if calls == 0:
        if incoming or outgoing or models:
            return _unknown(lang)
        return _zero_call(lang)

    def number(n: int) -> str:
        value = f"{n:,}"
        return value.replace(",", ".") if lang == "pt" else value

    names = (" e " if lang == "pt" else " and ").join(models)
    if lang == "pt":
        target = (f" nos modelos {names}" if len(models) > 1 else
                  f" ao modelo {names}" if models else " ao modelo")
        return (
            f"A última resposta usou {number(incoming)} "
            f"{'token' if incoming == 1 else 'tokens'} de entrada e {number(outgoing)} de saída, "
            f"em {number(calls)} {'chamada' if calls == 1 else 'chamadas'}{target}.\n"
            "A contagem cobre apenas as chamadas ao modelo registradas pelo código."
        )
    target = (f" to models {names}" if len(models) > 1 else
              f" to model {names}" if models else " to the model")
    return (
        f"The last answer used {number(incoming)} input {'token' if incoming == 1 else 'tokens'} "
        f"and {number(outgoing)} output {'token' if outgoing == 1 else 'tokens'}, "
        f"in {number(calls)} {'call' if calls == 1 else 'calls'}{target}.\n"
        "The count covers only model calls recorded by the code."
    )


def previous_usage_answer(origin: dict | None, lang: str) -> str:
    """Read the channel's last saved turn before this answer is appended.

    Strict reads distinguish unavailable storage from an empty conversation.
    Never fall back to the process-wide conversation buffer.
    """
    from src.ui.chat_store import (
        NETWORK_LABEL, TERMINAL_LABEL, VOICE_LABEL, daily_title, store,
    )

    channel = origin.get("channel") if origin else None
    labels = {"terminal": TERMINAL_LABEL, "network": NETWORK_LABEL, "voice": VOICE_LABEL}
    conversation = origin.get("conversation") if origin and channel == "ui" else None
    if channel not in labels and not conversation:
        return ("Não consigo identificar a conversa deste turno." if lang == "pt"
                else "I can't identify this turn's conversation.")

    try:
        if channel in labels:
            conversation = store.find_titled(daily_title(labels[channel]), strict=True)
        turns = store.get_turns(conversation, limit=1, strict=True) if conversation else []
    except Exception:  # Storage failure must not turn into a fabricated zero.
        return _unknown(lang)
    if not turns:
        return ("Ainda não há uma resposta anterior nesta conversa." if lang == "pt"
                else "There is no previous answer in this conversation yet.")
    return format_usage(turns[-1], lang)
