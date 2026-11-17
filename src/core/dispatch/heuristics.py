"""
Routing-decision heuristics — patterns that DO NOT match a command
or trigger an action; they just inform a downstream decision.

  - TOOL_HINT_RE         — lexical hint that the query is about the
    user's documents / notes / email / calendar / scheduling. When
    any token matches, the orchestrator routes through tool calling
    so the LLM can pick the right tool. Otherwise it skips tools
    entirely (Llama 3.1 8B Q4 has a strong over-use bias when given
    tools, especially read_document).
  - OFFLINE_HINT_RE      — fast-path for `_needs_realtime`: queries
    that unambiguously ask for stable conceptual content ("me
    explica RAG", "define X"). Matched at the START of the query
    so we don't catch incidental verbs mid-sentence.
  - REALTIME_MARKER_RE   — veto for the offline fast-path: even an
    "explain X" query needs live data if X is time-bound ("hoje",
    "agora", "atual"). If any marker appears, defer to the LLM.

These three drive routing choices, not behaviour. Edit here, not in
the orchestrator.
"""
from __future__ import annotations

import re


# Lexical hint that the query is about the user's documents/notes/files,
# email, calendar or scheduling. When any match, we route the query
# through tool calling so the LLM can pick the right tool. Otherwise
# we skip tools entirely (Llama 3.1 8B Q4 has a strong over-use bias
# when given tools, especially read_document).
# File-search phrasings (find_file, s81). "cadê meu currículo" carries none of
# the nouns below, so neither hint regex fired and the 8B answered from thin
# air. Defined ONCE and spliced into BOTH TOOL_HINT_RE (which vetoes the
# pure-chat shortcut) and REAL_TOOL_HINT_RE (which decides whether the tool
# menu is offered) — wiring it into only one is the bug this constant prevents.
#
# Anchored on a LOCATOR followed by a possessive/article, so "onde fica o
# Japão" (geography — note "fica" is deliberately excluded), "procura saber"
# and "acha que vai chover" stay out.
FILE_SEARCH_HINT = (
    r"(?:\bcad[êe]\s+(?:o|a|os|as|meu|minha|meus|minhas)\b|"
    r"\bonde\s+(?:est[áa]|t[áa])\s+(?:o|a|os|as|meu|minha|meus|minhas)\b|"
    r"\b(?:ach[ae]|acha[rm]?|procur[ae]|procurar|encontr[ae]|encontrar)\s+"
    r"(?:o|a|os|as|meu|minha|meus|minhas)\b|"
    r"\bwhere\s+(?:is|are)\s+(?:my|the)\b|"
    r"\b(?:find|locate)\s+(?:my|the)\b)"
)

# Narrow imperative shapes, shared by both decisions. "me lembra o que ..."
# asks for recall, as do "me lembra de quando/de como/de mim"; the existing
# deterministic reminder/recall routes still get first refusal.
REMINDER_REQUEST_HINT = (
    r"\b(?:me\s+lembr[ae]|lembr[ae][-–]me)\s+"
    r"(?:de\s+(?!\s*(?:quando|como|mim)\b)|"
    r"(?:em|daqui|amanh[ãa]|hoje|[àa]s|que)\b)"
)

TOOL_HINT_RE = re.compile(
    FILE_SEARCH_HINT + r"|" + REMINDER_REQUEST_HINT + r"|"
    r"\b(?:documento|document|arquivo|file|notas?|notes?|dossi[êe]|dossier|"
    r"vault|caderno|cofre|registro|anota[çc][aã]o|meus? arquivos?|"
    r"my (?:document|notes?|file)|"
    r"email|emails?|inbox|caixa de entrada|mensagens?|message|"
    r"mand(?:a|ar|e)|envi(?:e|ar|a)|send|reply|respond(?:er|a)?|forward|encaminhar|"
    r"calend[áa]rio|calendar|agenda|"
    r"evento|eventos|event|reuni[aã]o|meeting|compromisso|"
    r"agend(?:ar|a|e)|marcar|schedule|"
    r"lembre|lembrar|lembrete|lembretes|remind|reminder|reminders|"
    # Extended 2026-05-28 sessão 11 bloco H (fix D minimal). When the
    # input mentions music playback, Spotify, or macOS controls — but
    # the regex dispatch upstream missed the exact phrasing — we
    # still want chat_with_tools (not free chat) so the LLM at least
    # has a chance to call the right tool. The anti-hallucination
    # detector then catches the case where the LLM ignores the tools
    # and tries to fake it. False positives here cost ~500ms latency
    # on knowledge queries that incidentally mention these words.
    r"spotify|playlist|m[úu]sica|musica|music|"
    r"tocar|toque|toca|p[õo]e|p[õô]r|ponha|play|put on|ouvir|listen|escutar|"
    r"brilho|brightness|luz\s+do\s+teclado|keyboard\s+light|"
    r"modo escuro|tema escuro|dark mode|modo claro|tema claro|light mode|"
    r"trav(?:a|ar|ou|ei|am)|bloque(?:ar|ia|ie|io|ei|am)|lock|"
    r"print da tela|screenshot|captura(?:r)? (?:a )?tela|"
    r"abre|abrir|abra|lan[çc]a(?:r)?|open|launch"
    r")\b",
    re.IGNORECASE,
)


# Subset of TOOL_HINT_RE that maps to a tool that ACTUALLY EXISTS in
# TOOL_SPECS (documents / email / calendar / reminder). The music + macOS
# tokens are deliberately EXCLUDED: there is no spotify/macos tool in
# TOOL_SPECS, so when only those fire the orchestrator offers an EMPTY tool
# list — the anti-hallucination detector still runs, but the 8B can't
# fabricate an unrelated tool (e.g. a create_reminder from "amo música
# brasileira", which only matched on "música"). See the OFFLINE routing in
# orchestrator._process_core. s38, 2026-06-04.
REAL_TOOL_HINT_RE = re.compile(
    FILE_SEARCH_HINT + r"|" + REMINDER_REQUEST_HINT + r"|"
    r"\b(?:documento|document|arquivo|file|notas?|notes?|dossi[êe]|dossier|"
    r"vault|caderno|cofre|registro|anota[çc][aã]o|meus? arquivos?|"
    r"my (?:document|notes?|file)|"
    r"email|emails?|inbox|caixa de entrada|mensagens?|message|"
    r"mand(?:a|ar|e)|envi(?:e|ar|a)|send|reply|respond(?:er|a)?|forward|encaminhar|"
    r"calend[áa]rio|calendar|agenda|"
    r"evento|eventos|event|reuni[aã]o|meeting|compromisso|"
    r"agend(?:ar|a|e)|marcar|schedule|"
    r"lembre|lembrar|lembrete|lembretes|remind|reminder|reminders"
    r")\b",
    re.IGNORECASE,
)


# Multi-intent detector — routes natural compound requests to the
# planner without requiring the explicit "planeja:" prefix.
#
# Motivation (JOURNAL sessão 14 bloco D, sessão 15 bloco "decomposição"):
# The user's real-world test "busca os jogos da seleção brasileira E
# adiciona ao meu calendário" went to chat_with_tools, the 8B picked
# ONE tool, lost the second intent, and once even hallucinated
# "configure the calendar". The planner already handles multi-step
# decomposition cleanly — we just need to ROUTE compound phrases to
# it automatically.
#
# Trigger requires BOTH halves to fire (search-anywhere, ignore case):
#   1. _MULTI_INTENT_CONJUNCTION_RE — a sequential connective
#      ("e", "depois", "então", "and", "then"), guarded so the bare
#      "e" needs surrounding word boundaries to avoid matching every
#      "e" mid-word.
#   2. _MULTI_INTENT_DOMAINS_RE — distinct mentions from at least
#      two domain families: email, calendar/agenda, reminder, document,
#      web search/research. Counting distinct families (not raw matches)
#      keeps "manda email pro João e responde o email do José" from
#      being treated as multi-intent — it's still one domain.
#
# `multi_intent_match(text)` below combines the two checks. The
# orchestrator imports the helper, not the raw regexes, so the
# composition rule lives in one place.

_MULTI_INTENT_CONJUNCTION_RE = re.compile(
    r"\b(?:"
    r"e\s+(?:tamb[ée]m|depois|tamb[ée]m\s+depois)|"  # "e também", "e depois"
    r"depois\s+(?:disso|que)?|"                       # "depois (disso)"
    r"ent[ãa]o|em\s+seguida|"                         # "então", "em seguida"
    r"and\s+(?:then|also)|then|after(?:wards?)?|"     # "and then", "then"
    # Bare conjunctions LAST so the more specific ones above win first.
    # Both PT "e" and EN "and" need word boundaries to avoid mid-word
    # noise ("Spotify and Apple" wouldn't fire — `\band\b` matches
    # word "and"; `\bthen\b` matches the word "then"; etc.).
    r"e|and"
    r")\b",
    re.IGNORECASE,
)

# One alternative per domain family. The orchestrator counts how many
# DISTINCT families match — needs ≥2 to flag multi-intent. Order
# inside each family doesn't matter (we just check membership).
_MULTI_INTENT_DOMAIN_FAMILIES = {
    "email":     re.compile(
        r"\b(?:e[- ]?mail|emails?|inbox|caixa\s+de\s+entrada|"
        r"mensagens?(?:\s+do\s+gmail)?|message|gmail)\b",
        re.IGNORECASE,
    ),
    "calendar":  re.compile(
        r"\b(?:calend[áa]rio|calendar|agenda|evento|eventos|event|"
        r"reuni[aã]o|meeting|compromisso|agend(?:ar|a|e)|marcar|schedule)\b",
        re.IGNORECASE,
    ),
    "reminder":  re.compile(
        r"\b(?:lembre(?:te)?|lembrar|lembrete|lembretes|"
        r"remind(?:er)?s?)\b",
        re.IGNORECASE,
    ),
    "document":  re.compile(
        r"\b(?:documento|document|arquivo|file|notas?|notes?|"
        r"dossi[êe]|dossier|anota[çc][aã]o|registro)\b",
        re.IGNORECASE,
    ),
    "websearch": re.compile(
        r"\b(?:busca(?:r|s|m)?|busque|procura(?:r|s|m)?|procure|"
        r"pesquisa(?:r|s|m)?|pesquise|encontr[ae](?:r)?|"
        r"search\b|look\s+up|find\s+out|google)\b",
        re.IGNORECASE,
    ),
}


def multi_intent_match(text: str) -> bool:
    """True if `text` has a sequential conjunction AND mentions ≥2
    distinct tool domain families. Used by the orchestrator to route
    natural multi-intent requests to the planner automatically.

    Conservative on purpose: it's better to miss a multi-intent than
    to over-route knowledge queries to a slow planner decomposition.
    A false negative falls back to chat_with_tools (status quo);
    a false positive sends "vamos sair e tomar uma" into the LLM
    planner for 3-5 seconds of wasted compute.
    """
    if not text:
        return False
    if not _MULTI_INTENT_CONJUNCTION_RE.search(text):
        return False
    hits = sum(
        1 for rx in _MULTI_INTENT_DOMAIN_FAMILIES.values()
        if rx.search(text)
    )
    return hits >= 2


# Fast-path for _needs_realtime: queries that unambiguously ask for
# stable conceptual content. Matched at the START of the query so we
# don't catch incidental verbs mid-sentence.
#
# Extended 2026-05-27: "o que é / what is / what's / que é / o que
# significa / what does X mean" — very common definitional prefixes
# that were missing. The REALTIME_MARKER_RE veto downstream still
# vetoes time-bound versions ("o que é o jogo de hoje"), so adding
# these here is safe and saves a 1-2s LLM call per match.
OFFLINE_HINT_RE = re.compile(
    r"^\s*(?:"
    r"me\s+explica|explica[- ]me|explica|explique|"
    r"explain(?:\s+me|\s+to\s+me)?|"
    r"como\s+funciona|how\s+(?:does|do)\b.*?\bwork\b|"
    r"qual\s+(?:a|e|é)\s+diferen[çc]a\s+entre|what'?s?\s+the\s+difference\s+between|"
    r"defin[ae]\b|define\b|defini[çc][ãa]o\s+de|"
    r"o\s+que\s+(?:[eé]|significa)|que\s+[eé]|"
    r"what\s+(?:is|are)|what'?s|what\s+does\s+.+?\s+mean"
    r")\b",
    re.IGNORECASE,
)

# Pure arithmetic / math expression — "2+2", "15 * 7", "(3+1)/2".
# When the entire input is digits and operators (with optional spaces
# and a single trailing "?" or "="), there is no plausible reading
# that needs live web data. Acts as a second offline fast-path in
# _needs_realtime, saving another 1-2s LLM call.
# Each space belongs to one place (s88, Codex lote 9): the class holds \s and
# "=", and the old "\s*[?=]?\s*" around it fought over them — "1", 2000
# spaces and "x" took 10s to refuse. Same language, checked by fuzzing.
MATH_ONLY_RE = re.compile(
    r"^[\d+\-*/().\s,^=]+(?:\?\s*)?$",
)

# Veto for the fast-path: even an "explain X" query may need live data
# if X is time-bound. If any of these markers appear, defer to the LLM.
REALTIME_MARKER_RE = re.compile(
    r"\b(?:hoje|agora|atua(?:l|is|lmente)|recente|[úu]ltim[ao]s?|"
    r"today|now|current(?:ly)?|recent|latest|this\s+week|esta\s+semana)\b",
    re.IGNORECASE,
)

# Humility (Onda 3c): volatile-fact categories the LLM realtime-classifier
# tends to MISS when no time-marker word is present — exactly where the 8B
# confabulates (sports results, live prices/quotes, recent news). When this
# matches, _needs_realtime short-circuits to True so the query routes to the
# existing "quer que eu pesquise?" offer instead of a fabricated answer.
#
# Deliberately TIGHT (factual-question shapes, prices gated on a concrete
# asset token) to bound the false-positive cost of offering a search for
# something the model already knows. Definitional "o que é placar" still wins
# via the OFFLINE_HINT_RE fast-path, which runs first in _needs_realtime.
VOLATILE_FACT_RE = re.compile(
    r"(?:"
    # sports results
    r"\bquem\s+(?:ganhou|venceu|perdeu)\b"
    r"|\b(?:qual\s+(?:foi\s+)?o\s+)?placar\b"
    r"|\bresultado\s+d[oa]\s+(?:jogo|partida|cl[áa]ssico|corrida|gp|grande\s+pr[êe]mio)\b"
    # "resultado do ÚLTIMO jogo do flamengo" — an adjective between "do" and
    # "jogo" broke the branch above; "último/próximo jogo" is itself a live-
    # data shape (score just played / fixture just scheduled) the 8B fabricates.
    r"|\b(?:[úu]ltim[oa]|pr[óo]xim[oa])\s+(?:jogo|partida|corrida)\b"
    r"|\bquanto\s+(?:foi|terminou|ficou)\s+o\s+jogo\b"
    r"|\bwho\s+won\b|\bscore\s+of\s+the\s+(?:game|match)\b"
    # live prices / quotes (gated on a concrete asset to avoid over-matching)
    r"|\bcota[çc][ãa]o\b"
    r"|\b(?:pre[çc]o|quanto\s+(?:custa|est[áa]|vale))\b[^?]*\b"
    r"(?:d[óo]lar|euro|libra|bitcoin|btc|ethereum|cripto|a[çc][ãa]o|a[çc][õo]es|ibovespa|bolsa)\b"
    r"|\bhow\s+much\s+is\s+(?:the\s+)?(?:dollar|euro|bitcoin|btc)\b"
    # recent news
    r"|\b[úu]ltimas?\s+not[íi]cias?\b|\blatest\s+news\b"
    r")",
    re.IGNORECASE,
)
