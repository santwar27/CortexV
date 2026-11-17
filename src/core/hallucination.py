"""Detector for the Llama 3.1 8B Q5 "simulated action" failure mode.

When the LLM receives a request that should have used a tool (play
music, send email, lock screen, etc.) but chooses NOT to call the
tool, it sometimes generates a reply that NARRATES the action as if
it had happened — inventing track names, simulating audio with
"(pausa)", claiming an email was sent. The user thinks Cortex did
something it didn't.

This module flags those replies so the orchestrator/engine can swap
in a truthful "I couldn't do that" message before the user sees the
lie. It is intentionally **conservative**: a false positive here
costs the user a wrong "I can't help" on a legitimate knowledge
query, which is worse UX than missing some hallucinations. Only the
highest-precision patterns are encoded.

Four heuristic families:
  1. Parenthetical simulation tags — "(pausa)", "(música começa)",
     "(simulando reprodução)", "(pause)" — Llama 8B's signature
     narration habit. Almost never appears in a legitimate reply.
  2. First-line action claims — the reply opens with an explicit
     past-tense or present-progressive action verb that asserts
     something happened ("Tocando 'X' de Y", "Enviei o email para
     João", "Marcado na agenda"). Mid-sentence occurrences are
     ignored so essays about music history don't trip.
  3. First-line Mac control claims — past-tense actions paired with
     control objects (brightness, volume, modes, apps, screen lock,
     screenshots). These get a Mac-specific honest replacement.
  4. First-line study-list claims — paired addition verbs and a study
     destination, also checked on plain chat and streaming replies.

The caller is responsible for knowing whether a handler or tool ran.
The general detector covers no-tool turns; the study-specific detector
also covers plain chat, where a queue mutation cannot have happened.
"""

from __future__ import annotations

import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from src.core.dispatch.ps5 import PS5_WORDS


# Parenthetical simulation tags. Highly distinctive — these never
# appear in tool replies or legitimate prose. Matched ANYWHERE in
# the response because the LLM sometimes drops them mid-paragraph.
_PAREN_SIMULATION_RE = re.compile(
    r"\("
    r"\s*"
    r"(?:"
    # Portuguese
    r"p[aá]usa(?:\s+para\s+simular)?|"
    r"m[úu]sica\s+(?:come[çc]a|tocando|termina|termin[ao]u|para)|"
    r"simulando(?:\s+(?:reprodu[çc][aã]o|toque|envio|cria[çc][aã]o))?|"
    r"reprodu[çc][aã]o(?:\s+(?:come[çc]a|termina))?|"
    r"som\s+da\s+m[úu]sica|"
    r"toca\s+(?:a\s+)?m[úu]sica|"
    # English equivalents
    r"pause(?:\s+for(?:\s+simulation)?)?|"
    r"music\s+(?:starts|begins|stops|ends|playing)|"
    r"simulating(?:\s+(?:playback|sending|creation))?|"
    r"playing\s+music|"
    r"audio\s+plays?"
    r")"
    r"\s*"
    r"\)",
    re.IGNORECASE,
)


# First-line action claims. The pattern must anchor at the very
# start of the response (after optional whitespace) so we don't fire
# on essays. Each alternative ends with a word boundary so partial
# matches like "envido" don't trip "enviei".
#
# Covers the most common Llama 8B confabulations:
#   - "Tocando 'X' de Y" / "Tocou X"
#   - "Toquei a música X"
#   - "Pausada"/"Pausando" as a sentence-opener
#   - "Enviei o e-mail/mensagem para X"
#   - "Marcado para amanhã" / "Marquei na agenda"
#   - "Lembrete criado" / "Criei o lembrete"
#   - English: "Playing X by Y", "I have sent the email", etc.
_FIRST_LINE_ACTION_RE = re.compile(
    r"^\s*(?:"
    # Music — playback claims
    r"toc(?:ando|ou|ar[áa]|aram?|amos)\s+[\"'']?[a-z0-9]|"
    r"toquei\b|"
    r"pausad[ao]\b|"
    r"pausando\b|"
    r"pulei\b|"
    r"playing\s+[\"'']?[a-z0-9]|"
    r"now\s+playing\b|"
    # Email
    r"enviei\b|"
    r"e[- ]?mail\s+enviad[oa]\b|"
    r"mensagem\s+enviad[oa]\b|"
    r"(?:i\s+)?(?:have\s+)?sent\s+(?:the\s+|an\s+|your\s+)?(?:e[- ]?mail|message)\b|"
    # Calendar
    r"agendad[oa]\b|"
    r"marqu(?:ei|ado)\b|"
    r"evento\s+criad[oa]\b|"
    r"event\s+(?:created|scheduled)\b|"
    # Reminder
    r"lembrete\s+(?:criado|salvo|adicionado|marcado)\b|"
    r"criei\s+(?:o\s+|um\s+)?lembrete\b|"
    r"reminder\s+(?:created|set|added)\b|"
    # Generic completion claims (rare but seen)
    r"feito\.\s*$|done\.\s*$"
    r")",
    re.IGNORECASE,
)


# Keep Mac claims separate: the older families and their fallback stay intact.
# Only app names are case-sensitive — "Fechei a conta" is ordinary prose,
# whereas "Fechei o Safari" asserts an app action. Scoped flags preserve
# that distinction even though verbs and control nouns ignore case.
_MAC_APP_NAME = r"(?-i:[A-ZÀ-ÖØ-Þ][\w-]*)"
_KEYBOARD_LIGHT_PT = r"luz\s+do\s+teclado"
_KEYBOARD_LIGHT_EN = r"keyboard\s+(?:brightness|light|backlight)"
_MAC_FIRST_LINE_RE = re.compile(
    r"^(?:(?:pronto|feito|ok|done)\s*[,.:;!?—–-]+\s*)?(?:"
    # PT: adjustment verbs may have modifiers before the control object.
    r"(?:diminu[ií]|aumentei|abaixei|baixei|reduzi|subi|ajustei|coloquei|"
    r"deixei|mudei|alterei|configurei|diminuiu|aumentou|abaixou|baixou|"
    r"reduziu|subiu|ajustou|colocou|deixou|mudou|alterou|configurou)"
    r"\b[^\n]*?\b(?:brilho|volume|som|" + _KEYBOARD_LIGHT_PT + r")|"
    r"(?:apaguei|apagou|acendi|acendeu)\s+(?:a\s+)?" + _KEYBOARD_LIGHT_PT + r"|"
    r"(?:ativei|desativei|liguei|desliguei|ativou|desativou|ligou|desligou)\s+(?:(?:o|a)\s+)?"
    r"(?:modo\s+(?:escuro|claro|noturno)|tema\s+(?:escuro|claro)|"
    r"dark\s+mode|light\s+mode|mudo|som|" + _KEYBOARD_LIGHT_PT + r")|"
    r"silenciei|silenciou|(?:tirei|tirou)\s+o\s+som|(?:coloquei|colocou)\s+no\s+mudo|"
    r"(?:travei|bloqueei|travou|bloqueou)\s+(?:a\s+tela|o\s+mac|o\s+computador)|"
    r"(?:abri|fechei|encerrei|abriu|fechou|encerrou)\s+(?:o|a)\s+"
    r"(?:app|aplicativo|programa|janela|" + _MAC_APP_NAME + r")|"
    r"(?:tirei|fiz|capturei|tirou|fez|capturou)\s+(?:(?:um|uma)\s+)?"
    r"(?:print|screenshot|captura\s+de\s+tela)|"
    # PT/EN: presenting a screenshot also claims a tool-produced result.
    r"(?:aqui\s+est[áa]|segue)\s+(?:(?:o|a|um|uma|seu|sua)\s+)?"
    r"(?:print|screenshot|captura(?:\s+de\s+tela)?)|"
    r"(?:here's|here\s+is)\s+(?:(?:the|a|your)\s+)?screenshot|"
    r"brilho\s+(?:diminu[íi]do|reduzido|aumentado|ajustado)|"
    + _KEYBOARD_LIGHT_PT + r"\s+(?:diminu[íi]da|reduzida|aumentada|ajustada|apagada|acesa|ligada|desligada)|"
    r"volume\s+(?:ajustado|aumentado|diminu[íi]do|reduzido)|"
    r"modo\s+escuro\s+(?:ativado|desativado)|"
    r"tela\s+(?:travada|bloqueada)|"
    # EN: pair verbs with their control objects; no arbitrary prose gap.
    r"i\s+(?:have\s+)?(?:"
    r"(?:lowered|raised|dimmed|increased|decreased|set|adjusted|turned\s+(?:up|down))"
    r"\s+(?:(?:the|your)\s+)?(?:brightness|volume|sound|screen|" + _KEYBOARD_LIGHT_EN + r")|"
    r"(?:turned\s+(?:on|off)|enabled|disabled)\s+(?:(?:the|your)\s+)?"
    r"(?:dark\s+mode|light\s+mode|sound|screen|" + _KEYBOARD_LIGHT_EN + r")|"
    r"(?:muted|unmuted)\s+(?:(?:the|your)\s+)?(?:volume|sound)|"
    r"locked\s+(?:(?:the|your)\s+)?screen|"
    r"(?:opened|closed|quit)\s+(?:the\s+app|(?:the\s+)?" + _MAC_APP_NAME + r")|"
    r"took\s+a\s+screenshot)|"
    r"brightness\s+(?:lowered|set|adjusted)|"
    + _KEYBOARD_LIGHT_EN + r"\s+(?:lowered|raised|set|adjusted|on|off)|"
    r"volume\s+(?:set|adjusted|lowered|raised)|"
    r"dark\s+mode\s+(?:enabled|on|off|disabled)|"
    r"screen\s+locked"
    r")\b",
    re.IGNORECASE,
)


def _first_nonempty_line(reply: str) -> str:
    return next((line.strip() for line in (reply or "").split("\n") if line.strip()), "")


# These UI replies are built by code, never assistant prose. CODE_PROPOSAL
# can also report a failed save; the neutral marker asserts no pending state.
# AWAITING_CONFIRMATION is the strict-approval re-prompt, REFINE_STAGED the
# prompt-refinement preview. PDF, plans and tool previews use AGENT_*.
CONFIRMATION_HISTORY_MODES = frozenset({
    "AGENT_AWAITING_CONFIRMATION", "CODE_PROPOSAL", "REFINE_STAGED",
    "AWAITING_CONFIRMATION",
})
CONFIRMATION_HISTORY_STAND_IN = "[Non-model UI message omitted from history.]"

_CONFIRMATION_OPEN_RE = re.compile(r"^(?:confirmar|confirm)\b[^\n]*\?", re.I)
_CONFIRMATION_REPLY_RE = re.compile(
    r"\b(?:responda\s+[\"'‘“]?sim|reply\s+[\"'‘“]?yes)\b", re.I,
)
# Require the actual adjacent fields, plus the approval instruction below.
# A quoted email or an explanation beginning with prose is not a preview.
_CONFIRMATION_FIELDS_RE = re.compile(
    r"^(?:confirm[^\n]*\n)?[ \t]*(?:"
    r"Para:[^\n]*\n[ \t]*Assunto:[^\n]*\n[ \t]*Corpo:|"
    r"To:[^\n]*\n[ \t]*Subject:[^\n]*\n[ \t]*Body:|"
    r"T[íi]tulo:[^\n]*\n[ \t]*(?:In[íi]cio:[^\n]*\n[ \t]*Fim:|Quando:)|"
    r"Title:[^\n]*\n[ \t]*(?:Start:[^\n]*\n[ \t]*End:|When:))",
    re.I,
)
_STAGED_ACTION_CLAIM_RE = re.compile(
    r"^(?:(?:pronto|feito|ok|done)\s*[,.:;!?—–-]+\s*)?(?:"
    r"(?:lembrete|evento|reuni[ãa]o|e[- ]?mail)\s+"
    r"(?:criad[oa]|definid[oa]|agendad[oa]|marcad[oa]|enviad[oa]|salv[oa]|adicionad[oa])|"
    r"(?:eu\s+)?(?:criei|defini|agendei|marquei|enviei|salvei|adicionei)\s+"
    r"(?:(?:o|a|um|uma|seu|sua)\s+)?(?:lembrete|evento|reuni[ãa]o|e[- ]?mail)|"
    r"(?:reminder|event|meeting|e[- ]?mail)\s+(?:created|set|scheduled|sent|saved|added)|"
    r"(?:i\s+(?:have\s+)?)?(?:created|set|scheduled|sent|saved|added)\s+"
    r"(?:(?:a|an|the|your)\s+)?(?:reminder|event|meeting|e[- ]?mail)"
    r")\b(?!\s+(?:[ée]|significa|quer\s+dizer|is|means)\b)",
    re.I,
)

ACTION_HALLUCINATION_FALLBACK = {
    "pt": (
        "Nada foi feito. Peça diretamente um lembrete com horário, um evento "
        "com data e hora ou um email com destinatário e mensagem."
    ),
    "en": (
        "Nothing was done. Ask directly for a reminder with a time, an event "
        "with a date and time, or an email with a recipient and message."
    ),
}


def detect_invented_action(reply: str) -> bool:
    """Task 167: factual guard for model text on paths where no tool ran.

    Unlike the profile-gated legacy detector, this also covers plain chat.
    Only first-line claims or an approval block qualify, not quoted claims,
    explanations or a mention of an event later in ordinary prose.
    """
    head = _first_nonempty_line(reply)
    if head == CONFIRMATION_HISTORY_STAND_IN:
        return True
    if _STAGED_ACTION_CLAIM_RE.match(head):
        return True
    return bool(_CONFIRMATION_REPLY_RE.search(reply or "") and (
        _CONFIRMATION_OPEN_RE.match(head)
        or _CONFIRMATION_FIELDS_RE.match((reply or "").strip())))


# 146: failure claims are still invented when no handler/tool even tried.
# 146b: trigger/use may name a mechanism before an optional control verb.
# 146c: an action noun and connector may precede the control object too.
# 146f/h: a mechanism may reach a purpose clause in the first sentence,
# but its control verb must have an adjacent object. Direct connector chains
# remain covered by the adjacent-object branch below; bare device is not enough.
# 146g: temper only that gap: search/knowledge before the object is not control.
# A device/control object (or explicit command/request noun) stays mandatory.
_COMMAND_CONTROL_VERB_PT = (
    r"(?:acionar|ajustar|aumentar|diminuir|reduzir|baixar|tocar|reproduzir|"
    r"pausar|retomar|acender|apagar|ligar|desligar|abrir|fechar|ativar|desativar|"
    r"silenciar|iluminar|abaixar|clarear|escurecer|controlar)"
)
_COMMAND_CONTROL_VERB_EN = (
    r"(?:activate|adjust|increase|decrease|reduce|lower|raise|play|pause|resume|"
    r"turn\s+(?:on|off|up|down)|open|close|enable|disable|mute|unmute|control)"
)
_COMMAND_CONTROL_OBJECT = (
    r"(?:mac|computador|computer|tela|screen|display|brilho|brightness|"
    r"ilumin[âa]ncia|ilumina[çc][ãa]o\s+d[ao]\s+(?:tela|teclado)|teclado|keyboard|som|sound|volume|m[úu]sica|music|"
    r"spotify|modo\s+(?:escuro|claro|noturno)|dark\s+mode|light\s+mode|"
    r"app|aplicativo|programa)"
)
_COMMAND_SEARCH_KNOWLEDGE = (
    r"(?:busca|pesquisa|search|web|internet|encontrar|achar|buscar|pesquisar|"
    r"procurar|descobrir|informa[çc](?:[ãa]o|[õo]es)|artigo|pre[çc]o|"
    r"not[íi]cia|find|look\s+up|information|article|price|news)"
)
_COMMAND_FAILURE_RE = re.compile(
    r"^(?:(?:eu\s+)?n[ãa]o\s+consegui\s+"
    r"(?:executar|realizar|completar|cumprir|fazer)\s+"
    r"(?:o|esse|este|seu|teu)\s+(?:comando|pedido)|"
    r"i\s+could(?:n['’]t|\s+not)\s+(?:execute|run|carry\s+out|complete|do)\s+"
    r"(?:the|that|this|your)\s+(?:command|request))\b|"
    r"^(?:(?:eu\s+)?n[ãa]o\s+consegui\s+(?:acionar|ativar|usar)\s+"
    r"(?:a|o)\s+(?:ferramenta|fun[çc][ãa]o|op[çc][ãa]o|comando|controle|recurso)|"
    r"i\s+could(?:n['’]t|\s+not)\s+(?:trigger|activate|use)\s+"
    r"the\s+(?:tool|function|command|control))\b"
    rf"(?:(?!\b{_COMMAND_SEARCH_KNOWLEDGE}\b)[^.!?\n])*?\b"
    # Purpose-only additions leave the other failure branches unchanged.
    rf"(?:(?:para|pra)\s+(?:{_COMMAND_CONTROL_VERB_PT}|mutar)|"
    rf"to\s+(?:{_COMMAND_CONTROL_VERB_EN}|light\s+up|illuminate|brighten|dim|darken))\s+"
    r"(?:(?:o|a|os|as|um|uma|seu|sua|the|a|an|your)\s+)*"
    rf"{_COMMAND_CONTROL_OBJECT}\b|"
    r"^(?:(?:eu\s+)?n[ãa]o\s+consegui\s+"
    r"(?:(?:acionar|ativar|usar)\s+"
    r"(?:(?:a\s+(?:fun[çc][ãa]o|op[çc][ãa]o|ferramenta)|"
    r"o\s+(?:comando|controle|recurso))\s+(?:de|do|da|para|pra)\s+)?"
    rf"(?:{_COMMAND_CONTROL_VERB_PT}\s+)?|{_COMMAND_CONTROL_VERB_PT}\s+)"
    r"(?:(?:ajuste|controle|altera[çc][ãa]o|mudan[çc]a|regulagem)\s+(?:de|do|da)\s+)?|"
    r"i\s+could(?:n['’]t|\s+not)\s+"
    r"(?:(?:trigger|activate|use)\s+"
    r"(?:the\s+(?:function|command|control|option|feature|tool)\s+(?:to|for)\s+)?"
    rf"(?:{_COMMAND_CONTROL_VERB_EN}\s+)?|{_COMMAND_CONTROL_VERB_EN}\s+)"
    r"(?:(?:adjustment|control|change)\s+(?:of|for)\s+)?)"
    r"(?:(?:o|a|os|as|um|uma|seu|sua|the|a|an|your)\s+)*"
    rf"{_COMMAND_CONTROL_OBJECT}\b|"
    # Proper app names are only accepted with open/close, not arbitrary verbs.
    r"^(?:(?:eu\s+)?n[ãa]o\s+consegui\s+(?:abrir|fechar)|"
    r"i\s+could(?:n['’]t|\s+not)\s+(?:open|close))\s+"
    r"(?:(?:o|a|the)\s+)?" + _MAC_APP_NAME + r"\b",
    re.IGNORECASE,
)
_COMMAND_QUOTE_RE = re.compile(
    r"(?:'([^'\n]+)'|\"([^\"\n]+)\"|“([^”\n]+)”|‘([^’\n]+)’|«([^»\n]+)»)"
)
_COMMAND_SUGGESTION_RE = re.compile(
    r"\b(?:tent[ae](?:\s+(?:dizer|usar|digitar))?|"
    r"try(?:\s+(?:saying|using|typing))?|quer\s+que\s+eu\s+fa[çc]a|"
    r"do\s+you\s+want\s+me\s+to\s+do)\s*:?\s*"
    + _COMMAND_QUOTE_RE.pattern,
    re.IGNORECASE,
)
ROUTED_COMMAND_MODES = frozenset({
    "MAC_BRIGHTNESS", "MAC_KEYBOARD_BRIGHTNESS", "MAC_VOLUME", "MAC_MUTE",
    "MAC_UNMUTE", "MAC_DARK", "MAC_OPEN_APP", "SPOTIFY_PLAY",
    "SPOTIFY_PLAY_PLAYLIST", "SPOTIFY_PLAY_RADIO", "SPOTIFY_PAUSE",
    "SPOTIFY_RESUME", "SPOTIFY_NEXT", "SPOTIFY_PREV",
})
_COMMAND_ARGUMENT_MODES = frozenset({
    "SPOTIFY_PLAY", "SPOTIFY_PLAY_PLAYLIST", "SPOTIFY_PLAY_RADIO", "MAC_OPEN_APP",
})
COMMAND_FAILURE_FALLBACK = {
    "pt": "Não entendi esse comando, então nada foi feito.",
    "en": "I didn't understand that command, so nothing was done.",
}
# V9 can supply the honest opening itself, followed by unvetted advice.
# Match the complete fallback sentence, not a broader knowledge refusal.
_COMMAND_NOT_DONE_RE = re.compile(
    r"^(?:n[ãa]o\s+entendi\s+esse\s+comando[^\w\n]*ent[ãa]o\s+nada\s+foi\s+feito|"
    r"i\s+didn['’]?t\s+understand\s+that\s+command[^\w\n]*so\s+nothing\s+was\s+done)"
    r"(?=$|[^\w\s]|\s+(?:tent[ae]|try)\b)",
    re.IGNORECASE,
)
_COMMAND_FAKE_OFFER_RE = re.compile(
    r"\b(?:diga\s+sim|say\s+yes|quer\s+que\s+eu\s+fa[çc]a|"
    r"do\s+you\s+want\s+me\s+to\s+do)\b", re.IGNORECASE,
)
# Keep punctuation inside a quoted command from splitting its offer sentence.
# A quote cannot start inside a word (e.g. the apostrophe in "didn't").
_COMMAND_SENTENCE_RE = re.compile(
    rf"(?:(?<!\w){_COMMAND_QUOTE_RE.pattern}|[^.!?\n])+[.!?]*"
)
_command_stager = ContextVar("command_failure_stager", default=None)


@contextmanager
def command_failure_scope(stager):
    """A turn-local orchestrator callback; the engine never owns the slot."""
    token = _command_stager.set(stager)
    try:
        yield
    finally:
        _command_stager.reset(token)


def command_failure_stream(chunks, stager):
    """SSE pulls may change contexts/threads. Never keep a token over yield."""
    try:
        with command_failure_scope(stager):
            iterator = iter(chunks)
        while True:
            with command_failure_scope(stager):
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return
            yield chunk
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()


def command_failure_prefix(reply: str) -> bool:
    """Hold a possible failure opening before emitting even its first token."""
    head = re.sub(r"\s+", " ", _first_nonempty_line(reply).lower().replace("’", "'"))
    prefixes = ("não consegui ", "nao consegui ", "eu não consegui ",
                "eu nao consegui ", "i couldn't ", "i could not ",
                "não entendi esse comando", "nao entendi esse comando",
                "i didn't understand that command", "i didnt understand that command")
    return any(p.startswith(head) or head.startswith(p) for p in prefixes)


def _normalise_routed_text(text: str) -> str:
    """Exactly the router's vocative, sentence, courtesy, STT normalisation."""
    from src.core.dispatch.normalisation import (
        normalise_stt_aliases, strip_courtesy,
        strip_trailing_sentence_punctuation, strip_vocatives,
    )
    return normalise_stt_aliases(strip_courtesy(
        strip_trailing_sentence_punctuation(strip_vocatives(text.strip()))))


def routed_command_text(text: str) -> str:
    """Normalise a routed offer, including a Spotify-qualified pause."""
    text = _normalise_routed_text(text)
    pause = re.fullmatch(r"(pausa(?:r)?|pause)\s+(?:no|on)\s+spotify", text, re.IGNORECASE)
    return pause[1] if pause else text


def routed_command_match(text: str):
    """Resolve both registry segments, without running any handler."""
    from src.core.dispatch import registry
    return registry.simulate(routed_command_text(text),
                             registry.DISPATCH_ROUTES + registry.POST_PLANNER_ROUTES)


def _command_argument_in_input(argument: str | None, user_input: str) -> bool:
    """Compare the route's existing capture; never re-parse the suggestion."""
    def normalise(text: str) -> str:
        # Free arguments keep literal titles/app names, including pause phrases.
        folded = unicodedata.normalize("NFD", _normalise_routed_text(text).casefold())
        unaccented = "".join(c for c in folded if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", unaccented).strip()

    needle = normalise(argument or "")
    return bool(needle and re.search(
        r"(?<!\w)" + re.escape(needle) + r"(?!\w)", normalise(user_input)))


def command_failure_reply(lang: str, suggestion: str | None = None) -> str:
    reply = COMMAND_FAILURE_FALLBACK[lang]
    if suggestion is not None:
        reply += (f" Quer que eu faça: '{suggestion}'? Diga sim." if lang == "pt" else
                  f" Do you want me to do: '{suggestion}'? Say yes.")
    return reply


def guard_invented_command_failure(reply: str, user_input: str, lang: str) -> str:
    """Only call before persistence, when no tool or route ran this turn."""
    head = _first_nonempty_line(reply)
    failure = _COMMAND_FAILURE_RE.match(head)
    if asks_for_rewording(user_input) or not (
            failure or _COMMAND_NOT_DONE_RE.match(head)):
        return reply
    quoted = _COMMAND_SUGGESTION_RE.search(reply)
    suggestion, mode = None, "none"
    eligible = False
    rejections = []
    if quoted:
        # A suggestion list can start with an unroutable or disallowed command.
        # Stop at the first allowed route whose free argument the user supplied.
        for candidate in _COMMAND_QUOTE_RE.finditer(reply, quoted.start()):
            suggestion = next(s for s in candidate.groups() if s)
            match = routed_command_match(suggestion)
            mode = match.mode if match else "none"
            if mode not in ROUTED_COMMAND_MODES:
                continue
            # simulate() takes capture from the matched Route.capture group,
            # the same group its handler uses for query/playlist/seed/app.
            if mode in _COMMAND_ARGUMENT_MODES and not _command_argument_in_input(
                    match.capture, user_input):
                rejections.append({
                    "suggestion": suggestion, "mode": mode,
                    "reason": "suggestion_argument_not_in_input",
                })
                continue
            eligible = True
            break
    stager = _command_stager.get()
    offered = (suggestion if eligible and stager
               and stager(suggestion, mode, lang) else None)
    from src.core.audit import audit
    audit.log("–", "engine", "invented_command_failure", payload={
        "original_head": reply[:120], "user_input": user_input[:80],
        "suggestion": suggestion, "mode": mode,
        **({"suggestion_rejections": rejections} if rejections else {}),
    })
    if failure or offered is not None:
        return command_failure_reply(lang, offered)
    # V9's fallback is true. Preserve its content unless it promises an
    # approval that has no staged action behind it.
    return _COMMAND_SENTENCE_RE.sub(
        lambda sentence: "" if _COMMAND_FAKE_OFFER_RE.search(sentence[0]) else sentence[0],
        reply,
    )


_STUDY_FIRST_LINE_RE = re.compile(
    r"^(?:(?:pronto|feito|ok|done)\s*[,.:;!?—–-]+\s*)?(?:"
    r"(?:eu\s+)?(?:adicionei|coloquei|inclu[íi]|anotei|acrescentei)\b"
    r"[^\n]*?\b(?:lista|fila)\b[^\n]*?\bestudos?\b|"
    r"(?:i\s+(?:have\s+)?)?(?:added|put|queued)\b"
    r"[^\n]*?\b(?:study\s+(?:list|queue)|studies)\b)",
    re.IGNORECASE,
)

STUDY_HALLUCINATION_FALLBACK = {
    "pt": "Não coloquei nada na lista de estudos. Para colocar, diga: adiciona na lista de estudos: <tópico>",
    "en": "I didn't add anything to the study list. To add one, say: add to the study list: <topic>",
}


def detect_hallucinated_study_action(reply: str) -> bool:
    """First-line queue claims, checked only when no handler/tool ran."""
    return bool(_STUDY_FIRST_LINE_RE.match(_first_nonempty_line(reply)))


_PS5_FIRST_LINE_RE = re.compile(
    r"^(?:(?:pronto|feito|ok|done)\s*[,.:;!?—–-]+\s*)?"
    r"(?:(?:eu\s+)?(?:liguei|desliguei|acordei|abri)\b"
    r"[^\n]*?" + PS5_WORDS + r"|"
    r"(?:eu\s+)?(?:coloquei|pus|deixei)\b[^\n]*?" + PS5_WORDS
    + r"[^\n]*?\b(?:em\s+repouso|para\s+dormir|pra\s+dormir|dormindo)\b|"
    r"(?:i\s+(?:have\s+)?)?(?:turned\s+(?:on|off)|woke|opened)\b"
    r"[^\n]*?" + PS5_WORDS + r"|"
    r"(?:i\s+(?:have\s+)?)?put\b[^\n]*?" + PS5_WORDS
    + r"[^\n]*?\b(?:to\s+sleep|in(?:to)?\s+rest\s+mode)\b)", re.IGNORECASE,
)

PS5_HALLUCINATION_FALLBACK = {
    "pt": "Não fiz isso — não consegui acionar o controle do PS5. Tente: 'liga o ps5', 'desliga o ps5' ou 'o ps5 está ligado?'.",
    "en": "I didn't do that — I couldn't reach PS5 control. Try: 'turn on the ps5', 'turn off the ps5' or 'is the ps5 on?'.",
}


def detect_hallucinated_ps5_action(reply: str) -> bool:
    """First-line console claims use the same no-handler invariant as Mac."""
    return bool(_PS5_FIRST_LINE_RE.match(_first_nonempty_line(reply)))


def detect_hallucinated_mac_action(reply: str) -> bool:
    """Identify the Mac family for callers choosing its specific fallback.

    Same no-tool caller invariant as detect_hallucinated_action.
    """
    return bool(_MAC_FIRST_LINE_RE.match(_first_nonempty_line(reply)))


def detect_hallucinated_action(reply: str) -> bool:
    """True if `reply` looks like a simulated/claimed action.

    Caller invariant: only run this when the LLM had tools available
    and did NOT call one. Running it on tool replies or pure-chat
    paths would produce false positives — successful tool replies
    like "Tocando 'BACKSTAGE' de Matuê." are SUPPOSED to make these
    claims (the tool actually ran).
    """
    if not reply:
        return False
    if _PAREN_SIMULATION_RE.search(reply):
        return True
    # Look only at the first non-empty line, so essays don't trip
    # the action regex via a mid-paragraph sentence.
    first_line = _first_nonempty_line(reply)
    if _FIRST_LINE_ACTION_RE.match(first_line):
        return True
    return (detect_hallucinated_ps5_action(reply)
            or detect_hallucinated_mac_action(reply)
            or detect_hallucinated_study_action(reply))


def detect_fake_tool_call(reply: str, tool_names=None) -> bool:
    """True if `reply` WRITES a tool call as text instead of making one.

    Measured (s85, 15/10/2026, real 8B, isolated end to end): asked "cadê o
    pdf de punctuation? lê e responde a questão 5", the 8B answered 4 times
    in 6 with "I will use the `find_file` function… Here is the code:
    find_file(query=\"punctuation.pdf\", content=False)" — and called
    nothing. On 13/10 the same shape, read_file_at(…) with invented file
    names, was stored in the Palace as if it were a conversation. Three other
    tool requests: 0 in 18.

    The signature is a REAL tool name followed by "(": prose that merely
    names a tool ("I'll use the find_file function") is not flagged. Any of
    Cortex's tools counts, not only the ones offered this turn — the 8B
    writes calls to tools it was not given. Same caller invariant as
    detect_hallucinated_action: only run it when no tool was called."""
    if not reply:
        return False
    if tool_names is None:
        from src.core.tool_specs import TOOL_SPECS
        tool_names = [s["function"]["name"] for s in TOOL_SPECS]
    names = [n for n in tool_names if n]
    if not names:
        return False
    # "def send_email(" defines, it doesn't call (s88, Codex lote 3).
    pattern = (r"(?<!def )\b(?:" + "|".join(re.escape(n) for n in names)
               + r")\s*\(")
    return re.search(pattern, reply) is not None


# A request FOR code: its answer writes calls on purpose. "como faço uma
# função em python pra enviar email?" routes to the tool path (the word
# "email") and the example's send_email(...) was swapped for "couldn't fire
# the right tool" (s88, Codex lote 3). The s85 case asked for a file, not code.
_ASKS_FOR_CODE_RE = re.compile(
    r"\b(?:c[óo]digo|code|python|javascript|typescript|script|snippet"
    r"|programa(?:r|[çc][ãa]o)|programming)\b",
    re.IGNORECASE,
)


def asks_for_code(user_input: str) -> bool:
    """True when the user asked for code — detect_fake_tool_call stands down."""
    return bool(_ASKS_FOR_CODE_RE.search(user_input or ""))


# A request to translate or revise text the user hands over: the answer is that
# text reworded. "Traduza para português: I have sent the email." routes to
# the tool path (the word "email"), and the right answer "Enviei o email." was
# swapped for "couldn't fire the right tool" (s88, Codex lote 8). Only the
# instruction counts — what comes before ":" and outside quotes: "traduz e
# manda pro João: …" asks for an action, so the check stays on.
_QUOTED_RE = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»')


def asks_for_rewording(user_input: str) -> bool:
    """True when Pedro asked for text about text and his instruction names no
    action — detect_hallucinated_action stands down."""
    from src.core.dispatch.heuristics import TOOL_HINT_RE
    from src.core.fact_extractor import MATERIAL_CUE_RE
    head, colon, _material = (user_input or "").partition(":")
    instruction = _QUOTED_RE.sub(" ", head if colon else (user_input or ""))
    return bool(MATERIAL_CUE_RE.search(instruction)
                and not TOOL_HINT_RE.search(instruction))


# User-facing replacement message. Bilingual: orchestrator picks
# based on resolved language so the message is consistent with the
# rest of the turn.
HALLUCINATION_FALLBACK = {
    "pt": (
        "Não consegui acionar a ferramenta certa pra isso. Tenta "
        "ser mais direto — por exemplo, 'toca matue no spotify', "
        "'envia email pro João', 'modo escuro'."
    ),
    "en": (
        "I couldn't fire the right tool for that. Try a more direct "
        "phrasing — e.g. 'play matue on spotify', 'send email to "
        "John', 'dark mode'."
    ),
}


MAC_HALLUCINATION_FALLBACK = {
    "pt": (
        "Não fiz isso — não consegui acionar o controle do Mac. Tenta direto, "
        "por exemplo: 'diminua o brilho', 'volume 30', 'modo escuro', 'abre o Safari'."
    ),
    "en": (
        "I didn't do that — I couldn't reach the Mac control. Try it directly, "
        "e.g. 'lower the brightness', 'volume 30', 'dark mode', 'open Safari'."
    ),
}
