"""
Generic command triggers — patterns that match new commands from the
user (not replies to a prompt). Each captures the verb plus its
argument(s) so the orchestrator can dispatch to the right handler.

  - READ_RE  / WRITE_RE  — local document I/O (`leia foo.txt`,
    `anote em bar.txt …`).
  - CREATIVE_RE          — forces the creative engine instead of the
    analytical one (`criativo: escreve um soneto sobre Bach`).
  - SEARCH_STRIP_RE      — strips a leading "pesquise / search /
    busque por X" verb so the bare query is what reaches the web
    searcher; lets the user be explicit without polluting the query.
  - LIST_EMAILS_RE       — "meus emails recentes" shortcut to skip
    LLM tool selection and call the Gmail integration directly.

Patterns kept verbatim from the previous `commands.py` grab-bag.
Edit here, not in the orchestrator.
"""
from __future__ import annotations

import re


# READ — accepts .md too (knowledge files), and tolerates common typos
# of "leia" (lei, le, lê, lia).
READ_RE = re.compile(
    r"^(?:read|reads|leia|lei|lia|leie|l[eê])\s+([\w.-]+\.(?:txt|pdf|docx|md))(?:\s+(.*))?",
    re.IGNORECASE | re.DOTALL,
)

# READ THIS FILE — "lê ESSE arquivo e me diz X", after a search has put a
# path on the table. Deterministic on purpose: asked to read a file it had
# just found, the 8B answered ABOUT the document without calling any tool
# (s83), which is the one failure mode a reading feature cannot have. A regex
# route cannot forget to open the file.
#
# The demonstrative is required. Without it "lê o documento agentes e me manda
# um resumo por email" — two intents, the planner's job — was swallowed by
# this route (caught by the dispatch parity test).
READ_THIS_FILE_RE = re.compile(
    r"^(?:"
    r"(?:l[eê]|leia|ler|abre|abrir|abra|read|open)\s+"
    r"(?:esse|este|essa|esta|aquele|aquela|this|that)\s+"
    r"(?:arquivo|pdf|documento|file|document)"
    r"|"
    r"(?:o\s+que\s+(?:diz|tem\s+(?:em|n[oa]))|resum[ae]|resuma|"
    r"me\s+explica|explica|what(?:'s| is)\s+in)\s+"
    r"(?:esse|este|essa|esta|this|that)\s+"
    r"(?:arquivo|pdf|documento|file|document)"
    r")\b(?:\s+(?P<question>.*))?$",
    re.IGNORECASE | re.DOTALL,
)

# A bare read verb. On its own it means nothing — "lê o punctuation" is only
# a read command because a file called Punctuation… is on the table. The
# orchestrator pairs this with a name check against the files the
# conversation has actually surfaced, so no wording can conjure a file.
READ_VERB_RE = re.compile(
    r"^(?:l[eê]|leia|ler|abre|abrir|abra|read|open)\b",
    re.IGNORECASE,
)

# WRITE — adds "jot down", "store", "record" and PT-BR typos like
# "escreve", "salva", "anota" (imperative variants).
WRITE_RE = re.compile(
    r"^(?:anote|anota|escreva|escreve|salve|salva|guarde|guarda|"
    r"jot down|store|record)"
    r"(?:\s+(?:em|no|na|o|arquivo|in|to))?\s+([\w.-]+)\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)

CREATIVE_RE = re.compile(
    r"^(?:criativo|criatvo|criativa|creative)[:\s]+(.*)",
    re.IGNORECASE | re.DOTALL,
)

SEARCH_STRIP_RE = re.compile(
    r"^(?:search|pesquise|pesquisa|pesquisar|pesqise|"
    r"procure|procura|procurar|"
    r"busque|busca|buscar|"
    r"look up|find out)"
    # "pesquise online ..." left the medium word dangling in the provider
    # query (s73 Copa bug trace: query started with "online qual o ...").
    r"(?:\s+(?:online|na internet|na web|on the internet|on the web))?"
    r"(?:\s+(?:for|sobre|por|about|on))?\s+",
    re.IGNORECASE,
)

# Gmail shortcut commands (read-only in Fase 8A). Expanded sessão 20
# after the s19 live smoke showed the bare regex missing every natural
# phrasing — "lista meus 3 últimos emails", "meus ultimos emails?",
# "o que tem na minha inbox?" all fell through to the LLM, which then
# picked list_events (wrong tool) and surfaced a fake "couldn't access
# Gmail" reply. Wider regex catches the common shapes; LLM fallback
# stays for genuinely complex queries.
LIST_EMAILS_RE = re.compile(
    r"^(?:"
    # ----- main branch -----
        # Optional opener verb
        r"(?:(?:liste?|lista|listar|mostra(?:r)?|me\s+mostra|me\s+d[áaã]|"
        r"quero\s+ver|ver|abre|abrir|"
        r"list|show(?:\s+me)?|check|read|give\s+me)\s+)?"
        # Optional possessive (PT/EN)
        r"(?:meus|os|my|the)?\s*"
        # Optional count BEFORE modifier ("3 últimos emails"). Captured
        # via named group `count_pre` so the handler can pass it through
        # — only ONE of count_pre / count_post fires per match (the
        # other is None) because the modifier sits between them.
        r"(?:(?P<count_pre>\d{1,2})\s+)?"
        # Optional modifier
        r"(?:[uú]ltimos?|recentes?|novos?|last|recent|new)?\s*"
        # Optional count AFTER modifier ("últimos 3 emails")
        r"(?:(?P<count_post>\d{1,2})\s+)?"
        # The noun — email/inbox/caixa de entrada
        r"(?:emails?|e-mails?|inbox|caixa\s+de\s+entrada)"
        # Optional trailing qualifier
        r"(?:\s+(?:recentes?|do\s+gmail|na\s+inbox))?"
        r"\s*\??"
    r"|"
    # ----- alternate branch: "o que tem na minha inbox/caixa de entrada" -----
        r"(?:o\s+que\s+tem|what(?:'s|\s+is)?\s+(?:in|on))"
        r"\s+(?:na\s+|in\s+|on\s+)?(?:minha\s+|my\s+|the\s+)?"
        r"(?:inbox|caixa\s+de\s+entrada)"
        r"\s*\??"
    r")$",
    re.IGNORECASE,
)


# Send-email "structured" form: the user provides explicit field labels
# (to/para, subject/assunto, body/corpo). Bypasses chat_with_tools so
# the 8B can't paraphrase the body or smuggle typos into the subject —
# transcription is LITERAL from named groups (sessão 20 fix after live
# bug where "eu te amo" → "Í eu te amo" and "milhões" → "milões").
# Free-form natural language ("envia email pro João sobre atraso") still
# routes through the LLM as before; this regex only fires on the
# structured shape with all three field markers present.
#
# Named groups: to (RFC-ish email), subject (text), body (text).
SEND_EMAIL_STRUCTURED_RE = re.compile(
    r"^"
    # Optional verb opener: mandar/envia/send + article + noun
    r"(?:(?:mand(?:a|ar|e)|envi(?:a|ar|e)|send)\s+)?"
    r"(?:um\s+|uma\s+|an\s+|a\s+)?"
    r"(?:e-?mail|mensagem|message)\s+"
    r"(?:para|pra|to)\s*:?\s*"
    r"(?P<to>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"\s*[,;\-–—]+\s*"
    r"(?:assunto|subject|t[ií]tulo|title)\s*:?\s*"
    r"(?P<subject>.+?)"
    r"\s*[,;\-–—]+\s*"
    r"(?:corpo|body|texto|conte[úu]do|content)\s*:?\s*"
    r"(?P<body>.+?)"
    r"\s*$",
    re.IGNORECASE | re.DOTALL,
)


# Contact book regexes (sessão 21). The contact book turns "manda email
# pra lara" into a structured send_email staged action by resolving the
# name to a stored address. Three operations are surface-driven:
#
#   1. SAVE  — "salva o email da X como Y@Z" / "save X's email as Y@Z"
#   2. LOOK  — "qual o email da X?" / "what's X's email?"
#   3. SEND  — same shape as SEND_EMAIL_STRUCTURED but with a NAME in
#              the "para:" slot instead of an email address. The
#              orchestrator resolves the name → address and stages the
#              same _pending_action shape downstream.
#
# Privacy: contact data lives in ~/.cortex/contacts.db, never Palace,
# never git. Lookups are case-insensitive and fuzzy-tolerant.

# Name token: letters / accents / hyphen / apostrophe / space. Keeps
# punctuation out of the captured name so "salva o email da Lara."
# strips the trailing period naturally.
_NAME_TOKEN = r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-' ]*?"

CONTACT_SAVE_RE = re.compile(
    r"^(?:"
    r"salva(?:r)?|guarda(?:r)?|adiciona(?:r)?|"
    r"save|store|add"
    r")"
    r"\s+(?:o\s+|the\s+)?(?:e-?mail|endere[çc]o)\s+"
    r"(?:d[ao]\s+|do\s+|de\s+|of\s+|for\s+)?"
    r"(?P<name>" + _NAME_TOKEN + r")"
    r"\s+(?:como|as|to|=)\s+"
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Rich "add a contact" — captures everything after the trigger as a field blob
# (parsed deterministically by contacts.parse_contact_fields: nome/email/
# telefone/notas). Distinct from CONTACT_SAVE_RE: that one is the narrow
# "salva o email da X como Y"; this one starts with "... contato" and carries
# multiple labelled fields (email optional). s55.
CONTACT_ADD_RE = re.compile(
    r"^(?:"
    # PT — "adiciona/cria/cadastra/novo/salva (um) contato …", "vamos adicionar um contato"
    r"(?:adiciona(?:r)?|cria(?:r)?|cadastra(?:r)?|nov[oa]|salva(?:r)?|guarda(?:r)?)"
    r"\s+(?:um\s+|uma\s+|o\s+|novo\s+)?contato"
    r"|vamos\s+adicionar\s+(?:um\s+)?contato"
    r"|"
    # EN — "add/create/new/save (a/new) contact …"
    r"(?:add|create|new|save)\s+(?:a\s+|new\s+)?contact"
    r")\b[\s:,.—-]*(?P<fields>.*)$",
    re.IGNORECASE | re.DOTALL,
)

CONTACT_LOOKUP_RE = re.compile(
    r"^(?:"
    # PT: "qual o email da X" / "qual é o email do X"
    r"qual\s+(?:[ée]\s+)?(?:o\s+)?(?:e-?mail|endere[çc]o)"
    r"\s+(?:d[ao]\s+|do\s+|de\s+)"
    r"(?P<name1>" + _NAME_TOKEN + r")"
    r"|"
    # EN: "what's X's email" / "what is X's email"
    r"what(?:'s|\s+is)\s+(?P<name2>" + _NAME_TOKEN + r")"
    r"(?:'s|s')\s+(?:e-?mail|address)"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# Contact INFO lookup (s55): phone / full card for a stored contact. Broader
# than the email-only CONTACT_LOOKUP_RE. "número" is included even though it's
# ambiguous (passport!) — the handler returns None when the name matches no
# contact, so an unrelated "qual o número do meu passaporte" falls through to
# normal routing instead of being hijacked.
CONTACT_INFO_RE = re.compile(
    r"^(?:"
    # PT — "(qual o) telefone/celular/fone/número/info/dados/contato d* X"
    r"(?:qual\s+(?:[ée]\s+)?)?(?:o\s+|a\s+|os\s+|as\s+)?"
    r"(?:telefone|celular|fone|n[úu]mero|info(?:rma[çc][õo]es)?|dados|contato)"
    r"\s+(?:d[ao]\s+|do\s+|de\s+)(?P<name1>" + _NAME_TOKEN + r")"
    r"|"
    # PT — "me fala/mostra/diz (sobre) (o) contato X"
    r"(?:me\s+)?(?:fala|mostra|mostre|diz|d[áa])\s+(?:sobre\s+|do\s+|da\s+|o\s+|a\s+)*contato\s+(?P<name2>" + _NAME_TOKEN + r")"
    r"|"
    # EN — "what's X's phone/number/info/details" / "show (the) contact X"
    r"what(?:'s|\s+is)\s+(?P<name3>" + _NAME_TOKEN + r")(?:'s|s')\s+(?:phone|number|info|contact|details)"
    r"|(?:show|show\s+me)\s+(?:the\s+)?contact\s+(?P<name4>" + _NAME_TOKEN + r")"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# Same structural shape as SEND_EMAIL_STRUCTURED_RE but with a name in
# the "para/to" slot. The orchestrator resolves the name → email via
# contact_book.lookup_fuzzy before staging.
SEND_EMAIL_BY_NAME_RE = re.compile(
    r"^"
    r"(?:(?:mand(?:a|ar|e)|envi(?:a|ar|e)|send)\s+)?"
    r"(?:um\s+|uma\s+|an\s+|a\s+)?"
    r"(?:e-?mail|mensagem|message)\s+"
    r"(?:para|pra|to)\s+(?:o\s+|a\s+|the\s+)?"
    r"(?P<name>" + _NAME_TOKEN + r")"
    r"\s*[,;\-–—]+\s*"
    r"(?:assunto|subject|t[ií]tulo|title)\s*:?\s*"
    r"(?P<subject>.+?)"
    r"\s*[,;\-–—]+\s*"
    r"(?:corpo|body|texto|conte[úu]do|content)\s*:?\s*"
    r"(?P<body>.+?)"
    r"\s*$",
    re.IGNORECASE | re.DOTALL,
)
