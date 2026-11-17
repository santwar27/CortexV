"""Episodic no-record guard (s84).

A question about a moment of Pedro's life — "qual o nome do meu primeiro
cachorro?", "o que eu fiz no natal passado?", "como foi a minha formatura?"
— can only be answered from what Cortex was told. When nothing it was told
mentions the subject, the 8B answers anyway: measured on the real pipeline
(03/10, five runs per question), 4 specific inventions in 25 replies ("your
first dog was called Beto", a Christmas trip, a graduation course that does
not exist) and 3 more presupposing the event ("your graduation was a very
special moment!"). The system prompt already says "never invent specific
personal facts you don't have"; the model breaks it about one time in six.

So the model is not asked. When the question is about Pedro's own past AND
no word of its subject appears in what the model would see — the retrieved
PAST CONTEXT and the short-term thread — the reply is a fixed "I have no
record of that; want to tell me?", without an LLM call. Measured on the same
battery the subject word was absent from the context in every invented case
and present in the control ("onde eu trabalho?" → "trabalh").

What it cannot do: notice a subject written with a different word ("me
formei" for "formatura" — the stem catches "formação", not "formei"). That
direction ends in an honest "no record" when the data exists; the audit
event `episodic_no_record` is how to find out if it happens.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Iterable, List, Optional

from src.core.prompt_safety import CLOSE, OPEN

# The question is about the user's own past. Deliberately narrow: an opener or a
# possessive + past form, not every question with "meu".
_EPISODIC_CUE_RE = re.compile(
    r"\b(?:meu|minha)s?\s+primeir[oa]s?\b"
    r"|\bo\s+que\s+(?:eu|n[oó]s|a\s+gente)\s+(?:fiz|fizemos|fez)\b"
    r"|\bcomo\s+foi\s+(?:o\s+|a\s+)?(?:meu|minha|nosso|nossa)\b"
    # "lembra do/da/daquele/quando…" — not a bare "lembra de", which is how a
    # reminder is asked ("lembra de comprar pão").
    r"|^\s*(?:voc[eê]\s+|tu\s+)?(?:se\s+)?lembra\s+(?:d[oa]s?\b|daquel|quando\b|de\s+quando\b)"
    r"|^\s*me\s+lembra\s+de\s+quando\b"
    r"|\bmy\s+first\b"
    r"|\bwhat\s+did\s+(?:i|we)\s+do\b"
    r"|\bhow\s+was\s+my\b"
    # Only about OUR past: "Do you remember how merge sort works?" asks for a
    # concept and got "no record of that" (s88, Codex lote 3).
    r"|\bdo\s+you\s+remember\b(?=.*\b(?:i|we|me|us|my|our|when)\b)",
    re.IGNORECASE,
)

# Words that carry no subject: function words, the cue words themselves, and
# generic nouns ("nome", "ano", "dia") that would match anything. Accent-folded.
_NOT_SUBJECT = frozenset("""
o a os as um uma uns umas de da do das dos em no na nos nas ao aos e ou
que qual quais quem onde quando como porque por para pra pro com sem se
me te mim ti voce tu eu ele ela nos gente meu minha meus minhas seu sua
seus suas nosso nossa teu tua
lembra lembro lembrar lembre lembrou lembranca lembrancas
primeiro primeira primeiros primeiras ultimo ultima passado passada
ano anos dia dias vez vezes semana mes tempo hoje ontem agora
foi fui fiz fez fizemos era eram estava estive tinha teve fomos vamos ir
falei falou falamos disse disseste dissemos contei contou conversamos perguntei
aquele aquela aqueles aquelas daquele daquela daqueles daquelas
isso isto esse essa esses essas aquilo
nome chamava chama chamar chamou coisa coisas
sim nao yes okay ok
the an of my i we you do did was were remember first last how what
when where who your name thing things time
""".split())

_STEM_LEN = 5
_MIN_WORD = 3

NO_RECORD_REPLY = {
    "pt": "Não tenho registro disso nas nossas conversas. Quer me contar?",
    "en": "I don't have a record of that in our conversations. Want to tell me?",
}


def guard_enabled() -> bool:
    return os.getenv("CORTEX_EPISODIC_GUARD", "1").strip() not in ("0", "false", "no", "")


def _fold(text: str) -> str:
    """Lowercase, accents removed — "Criança" and "crianca" compare equal."""
    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def subject_stems(question: str) -> Optional[List[str]]:
    """Stems of the subject of an episodic question, or None if the question
    is not about Pedro's own past. An episodic question whose every word is
    generic ("como foi o meu dia?") returns [] — nothing to check, so the
    guard stands aside."""
    if not _EPISODIC_CUE_RE.search(question or ""):
        return None
    return content_stems(question)


def content_stems(text: str) -> List[str]:
    """Stems of the words in `text` that can carry a subject — any question,
    no episodic cue required. Also used by the engine's refusal retry (s84)."""
    stems: List[str] = []
    for w in re.findall(r"[a-z0-9]+", _fold(text)):
        if len(w) < _MIN_WORD or w in _NOT_SUBJECT:
            continue
        stem = w[:_STEM_LEN]
        if stem not in stems:
            stems.append(stem)
    return stems


def _with_synonyms(stems: List[str]) -> List[str]:
    """Add the stems of the profile's wording for a subject (s84: the
    profile says "data de nascimento" where Pedro says "aniversário").
    Same map retrieval expands with — knowledge_base.RETRIEVAL_SYNONYMS."""
    from src.core.knowledge_base import RETRIEVAL_SYNONYMS
    out = list(stems)
    for word, syn in RETRIEVAL_SYNONYMS.items():
        if word[:_STEM_LEN] in stems:
            for w in re.findall(r"[a-z]+", _fold(syn)):
                if len(w) >= _MIN_WORD and w not in _NOT_SUBJECT and w != "data":
                    if w[:_STEM_LEN] not in out:
                        out.append(w[:_STEM_LEN])
    return out


def subject_in(stems: Iterable[str], haystack: str) -> bool:
    """True if any stem — or the stem of a synonym the profile uses for it —
    starts a word of `haystack` (accent-folded)."""
    folded = _fold(haystack)
    return any(re.search(r"\b" + re.escape(s), folded)
               for s in _with_synonyms(list(stems)))


def no_record_reply(
    user_input: str,
    long_term_context: str,
    history: Iterable[dict],
    lang: str = "pt",
) -> Optional[str]:
    """The fixed reply when an episodic question has no subject on record,
    else None (let the model answer).

    `history` is the short-term thread BEFORE this turn. What Pedro said
    there counts as on record — except his own earlier questions, and earlier
    no-record replies, or asking twice would find the subject in the first
    ask and let the model invent on the second."""
    if not guard_enabled():
        return None
    if OPEN in (user_input or "") or CLOSE in (user_input or ""):
        return None  # document / web prompts bring their own context
    stems = subject_stems(user_input)
    if not stems:
        return None
    said = []
    for turn in history or []:
        content = (turn or {}).get("content") or ""
        role = (turn or {}).get("role")
        if role == "user" and "?" in content:
            continue
        if role == "user" and _fold(content).strip() == _fold(user_input).strip():
            continue
        if role == "assistant" and content in NO_RECORD_REPLY.values():
            continue
        if role in ("user", "assistant"):
            said.append(content)
    if subject_in(stems, (long_term_context or "") + "\n" + "\n".join(said)):
        return None
    return NO_RECORD_REPLY["en" if lang == "en" else "pt"]
