"""Code-owned replies and ephemeral guidance, shared by both entry points.

Loss allowlist (accent-folded, case-insensitive): morreu, morreram, faleceu,
falecimento, obito, velorio, enterro, funeral, cancer, esta internado/internada;
em/de luto or luto por/pelo/pela <relative>; perdi meu/minha <relative/pet>;
died, passed away, I lost my <relative/pet>. The relative/pet lists are below.
Death-verb clauses mentioning games or devices are excluded: "o jogo morreu
no fim" and "the battery died" are ordinary chat. Negation (or almost/quase)
only cancels a following death verb with at most two intervening adverbs.
Possessive loss requires a listed person/pet: keys, time and matches don't
qualify. Bare "luto" as a verb isn't enough. This deliberately favours missing
unusual names/forms over condolences on ordinary sentences; it is no semantic
classifier. Nothing here reads or writes Palace.
"""
from __future__ import annotations

import re
import unicodedata

from src.core.dispatch.normalisation import strip_vocatives
from src.core.i18n import t
from src.core.prompts import LOSS_NEWS_NOTE
from src.core.redaction import redact_text
from src.core.self_facts import needs_self_facts, self_fact_note
from src.core.taught_facts import REPLY_NOTE, teaching_is_empty


_PERSON_PT = (
    r"(?:pai|mae|avo|avos|irmao|irma|filho|filha|tio|tia|primo|prima|"
    r"sobrinho|sobrinha|marido|esposa|companheiro|companheira|parceiro|parceira|"
    r"namorado|namorada|amigo|amiga|cachorro|cachorra|cao|cadela|gato|gata)"
)
_PERSON_EN = (
    r"(?:father|mother|dad|mum|mom|parent|parents|grandfather|grandmother|"
    r"grandpa|grandma|brother|sister|son|daughter|uncle|aunt|cousin|nephew|niece|"
    r"husband|wife|partner|boyfriend|girlfriend|friend|dog|cat|pet)"
)
LOSS_RE = re.compile(
    r"\b(?:faleceu|falecimento|obito|velorio|enterro|funeral|cancer|"
    r"esta\s+internad[oa])\b"
    r"|\b(?:em|de)\s+luto\b"
    r"|\bluto\s+(?:por|pelo|pela)\s+(?:(?:meu|minha)\s+)?" + _PERSON_PT + r"\b"
    r"|\bperdi\s+(?:meu|minha)\s+" + _PERSON_PT + r"\b"
    r"|\bi\s+lost\s+my\s+" + _PERSON_EN + r"\b",
    re.IGNORECASE,
)
_DEATH_CLAUSE_RE = re.compile(
    r"(?:^|[.!?;\n])"
    r"(?![^.!?;\n]*\b(?:jogo|game|match|bateria|battery|motor|engine|"
    r"celular|phone|computador|computer)\b)[^.!?;\n]*",
    re.IGNORECASE,
)
_DEATH_VERB_RE = re.compile(r"\b(?:morreu|morreram|died|passed\s+away)\b", re.IGNORECASE)
_DEATH_NEGATION_RE = re.compile(
    r"\b(?:nao|nunca|not|never|quase|almost|\w+n['’]t)"
    r"(?:\s+(?:ainda|realmente|mesmo|yet|actually|really|ever)){0,2}\s*$",
    re.IGNORECASE,
)
RECALL_RE = re.compile(
    r"(?:(?:voce\s+)?(?:se\s+)?lembra(?:\s+(?:disso|disto|daquilo|dessa|desse))?"
    r"|(?:do\s+you|you)\s+remember\s+(?:this|that)|remember\s+that)\s*\??",
    re.IGNORECASE,
)
MAX_RECALL_CHARS = 300


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def reply_notes(text: str, taught=None, analytical=None, creative=None,
                self_text: str | None = None) -> str:
    """Compose receipt, loss and trusted self facts for this turn only."""
    notes = []
    if taught is not None and taught.status != "empty":
        notes.append(REPLY_NOTE)
    folded = _fold(text)
    if LOSS_RE.search(folded) or any(
        not _DEATH_NEGATION_RE.search(clause.group()[:verb.start()])
        for clause in _DEATH_CLAUSE_RE.finditer(folded)
        for verb in _DEATH_VERB_RE.finditer(clause.group())
    ):
        notes.append(LOSS_NEWS_NOTE)
    question = text if self_text is None else self_text
    if analytical is not None and creative is not None and needs_self_facts(question):
        notes.append(self_fact_note(analytical, creative))
    return "\n\n".join(notes)


def is_recall_question(text: str) -> bool:
    return RECALL_RE.fullmatch(_fold(strip_vocatives(text.strip()))) is not None


def recall_last(text: str, history: list, lang: str) -> str | None:
    """Quote the last earlier user turn from the engine's active channel history."""
    if not is_recall_question(text):
        return None
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        sentence = message.get("content", "").strip()
        if not sentence or is_recall_question(sentence) or teaching_is_empty(sentence):
            continue
        sentence = redact_text(sentence)
        if len(sentence) > MAX_RECALL_CHARS:
            sentence = sentence[:MAX_RECALL_CHARS - 1].rstrip() + "…"
        return t(lang, f'Lembro: você disse “{sentence}”.', f'I do: you said “{sentence}”.')
    return t(lang, "Lembrar do quê? Me diz o assunto.", "Remember what? Tell me the subject.")
