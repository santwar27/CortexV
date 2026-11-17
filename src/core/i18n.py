"""
Tiny bilingual response helper (s60, Claude-A).

Cortex's deterministic command handlers (mac / spotify / notes / reminder /
…) used to return strings hardcoded in Portuguese, regardless of the language
the user spoke. The voice listener (`listener.choose_voice_lang`, s53) picks the
TTS voice from the WORDS of the reply — so a PT reply to an English command was
either voiced in the wrong language or fell back to an English voice reading
Portuguese words (live bug, handoff s59).

The LLM/chat path already localises (the server injects a `response_language`
directive). This module gives the deterministic dispatch path the same ability
WITHOUT threading a `lang` argument through ~30 handler signatures:

  • `t(lang, pt, en)` — the one source of truth: returns the EN string for
    "en", the PT string otherwise (so any unexpected tag stays PT = the old
    behaviour).
  • `I18nMixin` — mixed into `CortexOrchestrator`; handler mixins reach
    `self._t(pt, en)` via the MRO (same pattern as `staging_mixin`). The
    orchestrator sets `self._reply_lang` once per turn in `_process_core`
    (after `_resolve_lang`), so handlers read the resolved language off the
    instance. Default "pt" preserves behaviour for any path that doesn't set it.

Integration clients (spotify_player, *_client) don't get the mixin — they take
an explicit `lang` kwarg and call `t(...)` directly; their dispatch wrappers
pass `o._reply_lang`.

This module also owns the language-detection heuristic (`_detect_lang` /
`_resolve_lang`), moved here from `orchestrator.py` in s69 (M10-A11) so the
UI layer can resolve the status-label language without importing the whole
orchestrator graph. The orchestrator re-imports them under the same names,
so existing references (and test monkeypatches) keep working.
"""
from __future__ import annotations

import re
import threading
from typing import Optional


def t(lang: str, pt: str, en: str) -> str:
    """Pick the reply string for `lang`. EN only for an explicit "en";
    everything else (incl. None / unknown) stays PT — the prior default."""
    return en if lang == "en" else pt


class I18nMixin:
    """Bilingual-reply helper for the orchestrator and its handler mixins.

    `_reply_lang` belongs to the executing thread. Outside a turn, the
    shared fallback preserves the last language set, defaulting to "pt"."""

    def _reply_lang_state(self) -> dict:
        # Lazy for __new__ instances; setdefault also handles concurrent first use.
        local = self.__dict__.setdefault("_reply_lang_local", threading.local())
        if not hasattr(local, "state"):
            local.state = {}
        return local.state

    @property
    def _reply_lang(self) -> str:
        local = getattr(self, "_reply_lang_local", None)
        state = getattr(local, "state", {})
        return state.get("lang", getattr(self, "_reply_lang_fallback", "pt"))

    @_reply_lang.setter
    def _reply_lang(self, value: str) -> None:
        self._reply_lang_state()["lang"] = value
        self._reply_lang_fallback = value

    def _clear_reply_lang(self) -> None:
        local = getattr(self, "_reply_lang_local", None)
        if local is not None:
            local.__dict__.pop("state", None)

    def _t(self, pt: str, en: str) -> str:
        return t(getattr(self, "_reply_lang", "pt"), pt, en)


# Bilingual prompt templates and language detection. Llama 3.1 8B Q5 does
# not reliably mirror the user's language even when told to, so the
# orchestrator picks the prompt template explicitly based on input.
_LANG_PT_PATTERN = re.compile(
    r"[ãõçáéíóúâêôà]|"
    r"\b(?:que|para|com|uma|um|isso|nao|não|sim|estou|está|estão|"
    # "quem" added s69 (M10-A11): the only PT question word missing from
    # this list — "quem descobriu o brasil?" read as EN. No EN collision.
    r"horas?|hoje|amanhã|ontem|meu|minha|seu|sua|qual|quando|onde|quem|"
    r"por\s+que|porque|obrigado|você|voce|eu|tu|nós|vocês|"
    # Added 2026-05-20: PT verbs/nouns that don't collide with EN. The
    # previous list missed unaccented PT queries like 'me explica RAG em
    # detalhes' so the streaming path defaulted to English directives.
    r"explica|explique|explicar|explicacao|"
    r"conta|conte|contar|"
    r"fala|fale|falar|"
    # Added s63 (M6-A1): the lease revoke verbs. "desliga a internet" read
    # as EN to this heuristic, so the revoke ack came back as "Internet
    # disconnected." spoken by an EN voice (the s53 class). None collide
    # with common English words.
    r"desliga|desligue|desligar|"
    r"desconecta|desconecte|desconectar|"
    r"corta|corte|cortar|"
    r"fica|ficar|fique|"
    # s88 (Codex lote 8): "me ajuda" and "pode continuar" got the English
    # directive and voice. PT-only forms — "continue" is English, kept out.
    r"ajuda|ajude|ajudar|pode|posso|podemos|continua|continuar|prossegue|"
    # ... and the grant mirrors (s63 follow-up). "liga" is not an English
    # word; "conecta/conecte" are PT-only conjugations.
    r"liga|ligue|ligar|"
    r"conecta|conecte|conectar|reconecta|reconecte|"
    r"diga|dizer|"
    # Added s84: the remembering family. "lembra daquele dia?" and "lembra de
    # mim?" read as EN — no accent, no listed word — and the 8B answered in
    # English. None of these is an English word.
    r"lembra|lembre|lembrar|lembro|lembrete|daquele|daquela|mim|comigo|"
    r"mostra|mostre|mostrar|"
    r"preciso|precisa|precisamos|"
    r"gosto|gosta|gostei|gostaria|"
    # The year reflex says "Estamos em 2026."; its reply selects the voice.
    r"sou|somos|estamos|"
    r"tambem|"
    r"entao|"
    r"sobre|"
    r"detalhes|"
    r"resumo|"
    r"agora|"
    # Informal/abbreviated PT (added 2026-06-02): the user types fast and
    # drops accents, especially on the laptop. These shortcuts read as
    # English to the old heuristic ("vc/pq/tb quero" → en), so a PT
    # question got an English reply. Only abbreviations that DON'T collide
    # with common English words are listed (e.g. 'to'/'no'/'da' excluded —
    # they're valid English). The voice path no longer depends on this
    # (it forwards Whisper's detected lang), but the text/UI path does.
    # Device/document commands (s83). the user asked "leia o arquivo e me diz o
    # prazo" and "abre o Notion" and got English back: the list had the lease
    # verbs but not the ones he uses every day for files, the screen and the
    # Mac. All PT-only — "trava", "tela", "arquivo", "leia" are not English
    # words; "read", "file", "open" and "lock" stay out for the same reason.
    r"leia|lei|l[eê]|ler|"
    r"abre|abra|abrir|"
    r"fecha|feche|fechar|"
    r"trava|travar|bloqueia|bloquear|"
    r"apaga|apague|apagar|"
    r"tira|tire|tirar|captura|"
    r"dorme|dormir|hiberna|hibernar|repouso|"
    r"arquivo|arquivos|documento|documentos|"
    r"tela|computador|"
    r"prazo|vaga|vagas|"
    r"cad[êe]|"
    # Writing answers on a PDF (s85): "escreve no pdf" came back in English
    # in the first end-to-end run. All PT-only words.
    r"escreve|escreva|escrever|preenche|preencha|preencher|coloca|coloque|"
    r"resposta|respostas|prova|provas|dois|duas|ambos|ambas|"
    # Study-queue and removal verbs (s83). the user tried five ways to empty the
    # queue — "limpar fila de estudos", "exclui tudo", "cancela o estudo 1" —
    # and every one came back in English. "remove", "cancel", "clear" and
    # "delete" stay out: those are English words.
    r"exclui|excluir|remova|remove[rm]|cancela|cancelar|"
    r"limpa|limpar|limpe|tira|tirar|apague|"
    r"fila|estudo|estudos|lembrete|lembretes|anota[çc][ãa]o|"
    # Study review verbs (s83): "revisar" came back in English, and the
    # reply to it was an English menu over a Portuguese session.
    r"revisar|revisa|revise|aprova|aprovar|aprove|rejeita|rejeitar|"
    r"aprendeu|aprendi|aprender|"
    # Everyday PT measured on the user's real sentences (s84). Of 622, 245 read
    # as EN — "toque matue no spotify", "aumenta o volume", "como fazer
    # gelo?", "tudo bem?", "comi 200g de frango" — so the reply came back in
    # English. Only words that are not English: "no", "do", "os" ("the OS"),
    # "nas" (a NAS) and "volume" stay out; every sentence the additions flip
    # from EN to PT was checked by hand against the same corpus.
    r"toque|toca|tocar|tocando|tocou|quero|ouvir|ouve|escutar|consegue|consigo|"
    r"sistema|tudo|bem|ol[aá]|manda|alterna|tema|modo|claro|escuro|"
    r"brilho|teclado|aumenta|aumentar|aumente|abaixa|abaixe|abaixar|diminui|diminuir|mudo|"
    r"pare|proxima|pr[oó]xim[oa]s?|meus|minhas|[uú]ltim[oa]s|lista|quais|s[aã]o|recentes|"
    r"configurar|como|fazer|faz|funciona|ganhar|encerra|encerrar|abres|"
    r"comi|corri|registra|corrida|frango|ovos|arroz|gramas|aveia|"
    r"contato|adiciona|tem|emprego|clima|porfavor|suspende|jogos|copa|mundo|"
    r"hor[aá]rios?|pelo|pela|muito|esse|essa|este|ele|ela|m[aá]ximo|supino|texto|"
    r"vc|voce|cê|"
    r"pq|pqp|"
    r"tb|tbm|"
    r"qto|qta|qts|qdo|qnd|"
    r"blz|"
    r"vlw|"
    r"obg|"
    r"msg|"
    r"mds|"
    r"cmg|ctg|"
    r"pra|pras|pro|pros|"
    r"dnv|"
    r"agr|"
    # PT demonstratives + reply/summarise verbs (s76, inovação #3 live
    # smoke): only "isso" was here, so "responde esse email" / "resume isto"
    # read as EN → the staging preview came back in English over a PT-drafted
    # body. All PT-only and boundary-safe (the group is \b(?:...)\b). "resume"
    # is deliberately EXCLUDED — it's a real English word; "isso/isto" already
    # carry "resume isso/isto".
    r"isto|esse|essa|este|esta|"
    r"responde|responda|responder|resuma|resumir|"
    r"depois)\b",
    re.IGNORECASE,
)


# A file path or a URL is a NAME, not language. the user asked in English —
# "read /Users/…/com~apple~CloudDocs/… and answer the question number five" —
# and was answered in Portuguese, because "com" inside the iCloud folder name
# com~apple~CloudDocs matched the Portuguese keyword list (s83). Anything
# inside a path is the filesystem's wording, never the user's.
_NOT_LANGUAGE_RE = re.compile(
    r"(?:https?://\S+)|(?:(?:~|/)(?:\\.|[^\s'\"<>])+)")


def _detect_lang(text: str) -> str:
    probe = _NOT_LANGUAGE_RE.sub(" ", text or "")
    return "pt" if _LANG_PT_PATTERN.search(probe) else "en"


def _resolve_lang(text: str, override: Optional[str]) -> str:
    """Pick the language to respond in.

    `override` wins when it is a valid tag ("pt" or "en"). Used by API
    clients that already KNOW the language they want — typically the
    iOS Shortcuts, where each shortcut is bound to a single language
    by the user. Falls back to the regex heuristic `_detect_lang` when
    no override is provided OR the override is invalid, so legacy
    clients (CLI, voice daemon, /ask without `language`) behave
    exactly as before.

    The heuristic alone misses common pt-BR inputs like "play michael
    jackson no spotify" (mixed EN verb + PT preposition) and "esta me
    ouvindo" (missing accent on "está") because the regex demands
    accented chars OR an exact PT keyword. Trusting the client is the
    right fix — only it actually knows whether the user is in PT or
    EN mode.
    """
    if override in ("pt", "en"):
        return override
    return _detect_lang(text)
