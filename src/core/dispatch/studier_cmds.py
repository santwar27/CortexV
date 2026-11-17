"""
Nightly Studier command regexes — queue management + review/approval.

Extracted from `src/core/orchestrator.py` in phase 2 of the split.
The orchestrator re-binds each constant as a `_NAME_RE` class attribute
on `CortexOrchestrator` for back-compat with existing tests. New code
should import from here directly:

    from src.core.dispatch.studier_cmds import STUDY_RE

Includes the legacy typo-tolerant commands and natural PT-BR/EN additions.
Edit here, not in the orchestrator.
"""
from __future__ import annotations

import re

from src.core.input_text import strip_paste_markers


# Expanded 2026-05-19: typo tolerance (etude, estud, etudo, estuda).
STUDY_RE = re.compile(
    r"^(?:estude|estuda|estudar|estudo|etude|estud|etudo|"
    r"study|studar|studie)\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)

# Require the study destination: these verbs also belong to Notes/Spotify.
# Both topic positions share this pattern with the streaming pure-chat veto.
_PT_ADD = (r"(?:adicion(?:ar?|e|ei)|coloc[ae]|colocar|p[õo]e|p[oô]r|"
           r"bota(?:r)?|inclu(?:[ií]r?|a)|acrescent[ae]|acrescentar)")
_PT_LIST = r"(?:na|[àa])\s+(?:(?:sua|minha|tua|a)\s+)?(?:lista|fila)\s+de\s+estudos?\b"
_EN_LIST = r"(?:to|on)\s+(?:(?:your|my|the)\s+)?study\s+(?:list|queue)\b"
ADD_STUDY_RE = re.compile(
    rf"^(?:{_PT_ADD}\s+(?:isso\s+)?{_PT_LIST}|(?:add|put)\s+{_EN_LIST})"
    r"\s*[:,\-]?\s*(?P<after>.+)$|"
    rf"^(?:{_PT_ADD}|add|put)\s+(?P<before>.+?)\s+(?:{_PT_LIST}|{_EN_LIST})"
    r"[.\s]*(?P<force>!!)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_study_topic(topic: str) -> str:
    """Clean delimiters without losing the queue's trailing !! force flag."""
    topic = strip_paste_markers(topic).strip().lstrip(":-,").strip()
    force = topic.endswith("!!")
    if force:
        topic = topic[:-2].rstrip()
    period_removed = topic.endswith(".")
    if period_removed:
        topic = topic[:-1].rstrip()
    for left, right in [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'), ('«', '»')]:
        if topic.startswith(left) and topic.endswith(right) and len(topic) >= 2:
            topic = topic[1:-1].strip()
            break
    if not period_removed and topic.endswith("."):
        topic = topic[:-1].rstrip()
    return topic + (" !!" if force else "")


def study_topic(match: re.Match) -> str:
    """Extract either topic position; timing words qualify the legacy verb."""
    if match.re is ADD_STUDY_RE:
        topic = match.group("after") or match.group("before")
        if match.group("force"):
            topic += " !!"
    else:
        topic = re.sub(
            r"^(?:(?:essa|esta)\s+noite|hoje\s+[àa]\s+noite|tonight)\s*[:,\-]?\s+",
            "", match.group(1), flags=re.IGNORECASE)
        if re.match(r"study\s", match.string, re.IGNORECASE):
            topic = re.sub(r"\s+tonight(?=\s*(?:!!)?\s*$)", "", topic, flags=re.IGNORECASE)
    return clean_study_topic(topic)


# Expanded 2026-05-28 (sessão 13, eval framework): accept 3rd-person
# imperative "lista" alongside the infinitive "listar", with optional
# article ("os") and singular/plural ("estudo"/"estudos"). Surfaced
# by an eval case where "lista os estudos" fell through to LLM —
# example of how the eval framework drives organic regex coverage.
#
# Expanded 2026-07-09 (s78, live miss by the user): "me de a lista de
# estudo que esta na fila" fell to the LLM, which hallucinated. Two
# gaps: the courteous lead-in ("me dê/dá/mostra …") before the verb,
# and "lista DE estudo(s)" (only "lista (os) estudos" was covered).
# Trailing text after the match ("… que esta na fila") was already
# fine — the regex is start-anchored, not full-match.
LIST_STUDIES_RE = re.compile(
    r"^(?:me\s+(?:d[êe]|d[áa]|mostra|mostre)\s+)?(?:a\s+)?"
    r"(?:lista(?:r)?\s+(?:de\s+)?(?:os\s+|todos\s+os\s+)?estudos?|"
    r"ver fila|ver filas|fila de estudos?|fila estudos|"
    r"estudos pendentes|estudo pendente|"
    r"list studies|study queue|studies queue|"
    r"mostra(?:r)? (?:a )?fila)\b",
    re.IGNORECASE,
)

# Review / approval
# "o que você aprendeu (com seu último estudo)?" — answered from the FILE,
# not from the model's memory. Asked this, Cortex described SOLID and
# FastAPI, which is old approved knowledge and not what it had just studied
# (s83). Anchored at the end so "o que você aprendeu com o livro de arquitetura"
# stays a conversation.
LAST_STUDY_RE = re.compile(
    r"^(?:"
    r"(?:o\s+)?que\s+(?:(?:voc[êe]|tu|vc)\s+)?aprendeu"
    r"(?:\s+(?:com\s+)?(?:o\s+|seu\s+|teu\s+|no\s+)?"
    r"(?:[úu]ltimo\s+)?(?:estudo|estudos|aprendizado)?)?"
    r"(?:\s+(?:hoje|ontem|essa\s+noite|esta\s+noite|de\s+madrugada))?"
    r"|"
    r"qual\s+(?:foi\s+)?(?:o\s+)?(?:seu\s+|teu\s+)?[úu]ltimo\s+estudo"
    r"|"
    r"what\s+did\s+you\s+learn(?:\s+(?:in\s+)?(?:your\s+)?"
    r"(?:last\s+)?(?:study|studies)?)?(?:\s+(?:today|last\s+night))?"
    r"|"
    r"(?:what\s+was\s+)?your\s+last\s+study"
    r")\s*[?!.]*$",
    re.IGNORECASE,
)

REVIEW_RE = re.compile(
    r"^(?:revis(?:ar|a|e)(?:\s+estudos?)?|ver estudos(?:\s+prontos)?|"
    r"estudos prontos|review studies|study review)\b",
    re.IGNORECASE,
)

# Ref ("1", "todos", or partial topic text) is OPTIONAL. When omitted
# the handler will either default to the single pending study or ask
# the user to clarify if there are several. The `(?:\b|$)` anchor
# after the verb prevents false matches against words like
# "aproveite" / "veracidade".
#
# NOTE (s60): "abrir"/"open" were REMOVED from the verb list. They are the
# canonical app-open verbs ("open spotify", "abrir o notion") and were
# greedily capturing the app name as a study ref, hijacking the macOS
# open-app route ("open spotify" → STUDIER_SHOW "spotify" not found). Studier
# show stays on "ver/mostrar/detalhar/show"; nobody says "open" to view a study.
# Nor does it take what another route owns (s88, Codex lote 9): "ver as notas"
# and "show my notes" answered "Estudo 'as notas' não encontrado" — this route
# runs before NOTE_LIST_RE and its handler never lets go. Said with "estudo",
# any topic is a study's.
SHOW_STUDY_RE = re.compile(
    r"^(?:ver|mostrar|detalhar|show)(?:\b|$)"
    r"(?!\s+(?:(?:as|os|a|o|my|the|meus?|minhas?)\s+)*"
    r"(?:notas?|notes?|e-?mails?|eventos?|events?|agenda|calend[áa]rio|vagas?|jobs?|"
    r"refei[çc][õo]es|macros|treinos?|lembretes?|reminders?|contatos?|contacts?)\b)"
    r"(?:\s+(?:estudo|study))?"
    r"\s*(.*)$",
    re.IGNORECASE,
)

APPROVE_STUDY_RE = re.compile(
    r"^(?:aprov(?:ar|e|a)|approve)(?:\b|$)"
    r"(?:\s+(?:o|os|estudo|estudos|study|the))?"
    r"\s*(.*)$",
    re.IGNORECASE,
)

# CANCEL — take a topic OFF the queue, or empty it. The queue could be added
# to, listed, shown and approved; it could never be emptied (s83). the user
# changed his mind about three topics and tried "limpar fila de estudos",
# "exclui tudo", "exclui o estudo 2" — none of which had a route, so the 8B
# answered, claimed it had tidied the queue, and had not.
#
# The queue noun is REQUIRED for the bulk forms: "exclui tudo" on its own is
# not a study command, and reading it as one would be the kind of wide
# pattern that answered "por que azul?" with pagode.
CANCEL_STUDY_RE = re.compile(
    r"^(?:exclu[ií](?:r)?|remov(?:e|er)|cancel(?:a|ar|e)|tira(?:r)?|"
    r"apaga(?:r)?|limpa(?:r)?|delete|remove|cancel|clear)\s+"
    r"(?:"
    # bulk: "… a fila de estudos" / "… todos os estudos" / "… the study queue"
    r"(?P<all>(?:a\s+|os\s+|as\s+|the\s+)?(?:fila\s+(?:de\s+)?estudos?|"
    r"estudos?\s+pendentes?|todos\s+os\s+estudos?|study\s+queue|"
    r"all\s+(?:the\s+)?stud(?:y|ies)))"
    r"|"
    # one by number: "… o estudo 2" / "… study 3"
    r"(?:o\s+|os\s+|the\s+)?(?:estudos?|stud(?:y|ies))\s*(?P<n>\d+(?:\s*,\s*\d+)*)"
    r")"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

# "exclui tudo" with no noun. Deliberately NOT an action: delete what —
# notes, emails, files, the queue? It routes to a question instead, which is
# harmless even when the guess is wrong, and is what the 8B should have done
# instead of inventing a policy to refuse with (s83). Anchored on the bare
# phrase, so "exclui tudo isso daqui" is still ordinary conversation.
CANCEL_VAGUE_RE = re.compile(
    r"^(?:exclu[ií](?:r)?|remov(?:e|er)|cancel(?:a|ar|e)|apaga(?:r)?|"
    r"limpa(?:r)?|deleta(?:r)?)\s+(?:tudo|todos|todas)[\s.!?]*$",
    re.IGNORECASE,
)

REJECT_STUDY_RE = re.compile(
    r"^(?:rejeit(?:ar|e|a)|descart(?:ar|e|a)|reject|discard)(?:\b|$)"
    r"(?:\s+(?:o|os|estudo|estudos|study|the))?"
    r"\s*(.*)$",
    re.IGNORECASE,
)
