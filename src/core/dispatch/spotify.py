"""
Spotify regex dispatch patterns.

Extracted from `src/core/orchestrator.py` to keep that file focused on
routing rather than the per-skill pattern bodies. The orchestrator
class re-binds these module constants as class attributes so existing
test code (e.g. `CortexOrchestrator._SPOTIFY_PAUSE_RE`) keeps working
unchanged.

Strategy: regex-first dispatch in the orchestrator (zero LLM cost for
common commands like "pausa", "próxima", "spotify volume 50"), with the LLM
tool-call fallback reserved for complex queries that need
disambiguation. Achieves ~150ms latency vs 8-19s if routed through
Llama. See `src/tools/spotify_player.py` for execution side.
"""
from __future__ import annotations

import re

from src.core.dispatch.macos import DEVICE_TARGET_RE


# Play with optional device hint at the tail. Group(1) is the search
# query (track / artist / playlist). The device-hint suffix is matched
# but NOT captured here — see SPOTIFY_DEVICE_HINT_RE below.
#
# Verbs accepted, in three families:
#   1. Direct imperative — toca / coloca / põe / play / put on
#   2. Polite/permissive (added 2026-05-28, sessão 10 bloco F) —
#      "pode tocar X", "podes colocar X", "poderia tocar X", "podia
#      pôr X", "quero ouvir X", "queria ouvir X", "gostaria de ouvir
#      X", "could you play X", "can you put on X". These close the
#      gap that sent "ola cortex, pode tocar matue no spotify?" to
#      the LLM which then hallucinated a fake track list.
#   3. Trailing punctuation tolerance — voice-style chat ends with
#      "?", "!" or "." (e.g. "...no spotify?"). The original tail
#      `\s*$` rejected those; new tail `[?!.]*\s*$` accepts them.
# Fixed expressions that open with a play verb and are not about music.
# The adversarial sweep (s83) fed 89 ordinary sentences to the dispatch
# layer; these three actually STARTED music, which is the loud kind of false
# positive — "toca o barco" put on a track mid-conversation. Same shape as
# the "por que" fix: the verb stays, its idiomatic companions are excluded.
_NOT_ABOUT_MUSIC = (
    r"(?!(?:"
    r"toca(?:r)?\s+(?:o\s+barco|(?:pra|para)\s+frente|em\s+frente|fundo|"
    r"n[oe]sse\s+assunto|no\s+assunto|a\s+campainha|o\s+sino|a\s+vida)|"
    r"play\s+(?:it\s+(?:cool|safe|by\s+ear)|along|hardball|dumb|games|"
    r"the\s+(?:fool|victim)|devil'?s\s+advocate)|"
    # "coloca no calendário: dentista amanhã" and "coloca na nota Viagem: …"
    # were searched on Spotify, ahead of the calendar and notes routes (s88,
    # Codex lote 9). Putting something IN a place Cortex keeps is not music.
    r"(?:coloca(?:r)?|coloque|p[õo]e|p[ôõo]r|ponha)\s+"
    r"(?:(?:no|na|nos|nas|em|pro|pra|para\s+[oa]?)\s*(?:o\s+|a\s+|meu\s+|minha\s+)?"
    r"(?:calend[áa]rio|agenda|notas?|lembretes?|lista)|"
    r"(?:um|uma)\s+(?:lembrete|evento|nota|compromisso))"
    r")\b)"
    # Reserve control-shaped OBJECT PREFIXES, not words anywhere in a title.
    # "coloca a música modo escuro" remains music; "coloca o volume …" does not.
    r"(?!(?:(?:(?:pode|podes|podia|podiam|poderia|poderiam)\s+)?"
    r"(?:toca(?:r)?|toque|toquem|coloca(?:r)?|coloque|p[õo]e|p[ôõo]r|ponha|ponham|ouvir)|"
    r"(?:quero|queria|gostaria(?:\s+de)?)\s+ouvir|"
    r"(?:(?:can|could)\s+you\s+)?(?:play|put\s+on))\s+"
    r"(?:"
    r"(?:(?:o\s+mac|a\s+tela|o\s+computador)\s+)?"
    r"(?:(?:no|na|em|pro|pra|para(?:\s+[oa])?)\s+)?(?:o\s+)?"
    r"(?:modo|tema)\s+(?:escuro|claro|noturno|dark|light)|"
    r"(?:o\s+)?(?:brilho|volume)|o\s+som|a\s+luz\s+da\s+tela|"
    r"(?:o\s+mac|o\s+computador|a\s+tela)\s+(?:pra|para)\s+(?:dormir|descansar|travar)"
    r")\b)"
)

# An artist connector makes these objects explicit playback requests. Reuse
# the same object for bota's guard and the query prefix so video-editing
# requests ("bota uma música no meu vídeo") keep falling through.
_PLAY_PT_ARTIST_OBJECT = r"(?:(?:(?:a|uma)\s+)?m[úu]sica|(?:um\s+)?som)\s+(?:de|d[oa]s?)"
_PLAY_PT_DEVICE_TARGET = (
    r"(?:" + DEVICE_TARGET_RE + r"|sistema|iphone|celular|telefone|tv|carro|fones?)"
)

_PLAY_PT_SENDING_GUARDS = (
    # Recipients and sending channels anywhere after these verbs mean sending.
    r"(?!.*\s+(?!(?:pra|para)\s+(?:mim|tocar)\b)"
    r"(?:pra|para|pro|pros|pras)\s+\S)"
    r"(?!.*\s+(?:por|pelo|pela|no|na|via)\s+"
    r"(?:e-?mail|mensagem|msg|whatsapp|zap|sms|telegram|imessage|direct|dm)\b)"
)
_PLAY_PT_SENDING_PREFIX = (
    # Explicit artist-object branch for sending verbs. Consume the object
    # here: it cannot backtrack into a plain query.
    r"(?:manda|solta)" + _PLAY_PT_SENDING_GUARDS
    + r"(?:\s+a[ií])?"
    r"\s+(?:(?:um|o)\s+som|(?:uma|a)\s+m[úu]sica)\s+(?:de|d[oa]s?)"
    r"(?!\s+" + _PLAY_PT_DEVICE_TARGET + r"\b)"
)
_PLAY_VERB_PREFIX = (
    # 1. Direct imperative
    r"toca|toque|toquem|tocar|"
    r"coloca|coloque|colocar|"
    r"bota(?=\s+" + _PLAY_PT_ARTIST_OBJECT + r"\s+\S)|"
    # "por" without the circumflex is usually the PREPOSITION, and taking it
    # as the verb made every "por que …?" play music — "por que azul?" was
    # answered by putting on a pagode album (s83). It cannot simply be
    # dropped: the courtesy stripper turns the legitimate "pode por matuê"
    # into "por matuê". So the verb stays and the preposition's companions
    # are excluded — "por que/quê", "por favor", "por acaso", "por enquanto"
    # and the rest of the fixed phrases, none of which is a song anyone asks
    # for by that name.
    r"p[õo]e|p[ôõ]r|ponha|ponham|"
    r"por(?!\s+(?:qu[eê]|favor|acaso|enquanto|isso|isto|causa|exemplo|ora|"
    r"hora|aqui|ali|mim|ti|si|conta|volta|sorte|fim|obs[ée]quio|ventura|"
    r"vezes|[úu]ltimo|completo|cento|semana|m[êe]s|ano|dia)\b)|"
    r"play|put on|"
    # 2. Polite/permissive
    r"(?:pode|podes|podia|podiam|poderia|poderiam)\s+(?:tocar|colocar|p[õôo]r|ouvir)|"
    r"(?:quero|queria|gostaria(?:\s+de)?)\s+ouvir|"
    r"(?:can|could)\s+you\s+(?:play|put\s+on)"
)
_PLAY_PT_DEVICE_GUARD = (
    # Guard before the optional object: otherwise a rejected device artist
    # can backtrack into the plain query ("som do mac mais alto"). Only the
    # target immediately after de/do/da/dos/das is reserved; device tails stay valid.
    r"(?!\s+" + _PLAY_PT_ARTIST_OBJECT + r"\s+" + _PLAY_PT_DEVICE_TARGET + r"\b)"
)
_PLAY_ARTIST_PREFIX = (
    r"(?:" + _PLAY_PT_SENDING_PREFIX + r"|(?:" + _PLAY_VERB_PREFIX + r")"
    + _PLAY_PT_DEVICE_GUARD + r"\s+" + _PLAY_PT_ARTIST_OBJECT + r")"
)
_PLAY_TITLE_PREFIX = (
    r"(?:" + _PLAY_VERB_PREFIX + r")" + _PLAY_PT_DEVICE_GUARD
    + r"(?:\s+(?:a\s+|the\s+)?(?:m[úu]sica|song|aquela|aquele))?"
)

# Task 153: only explicit artist objects allow these conversational tails.
# Bare titles such as "volta pra mim" and "tô nem aí" must keep their words.
_PLAY_ARTIST_TAIL = r"\s+(?:a[ií]|(?:pra|para)\s+mim)"
# These tails still apply to every query, including bare titles. Device hints
# are extracted separately from the original text by SPOTIFY_DEVICE_HINT_RE.
_PLAY_COMMON_TAIL = (
    r"\s+(?:pra|para)\s+tocar|"
    r"\s+(?:no|na|on|in)\s+spotify|"
    r"\s+(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(?:iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)"
)
_PLAY_END = r"[?!.]*\s*$"
_PLAY_PREFIX = r"(?:" + _PLAY_ARTIST_PREFIX + r"|" + _PLAY_TITLE_PREFIX + r")"

SPOTIFY_PLAY_RE = re.compile(
    r"^" + _NOT_ABOUT_MUSIC
    # Query-first lookaheads keep the final query in public group(1); the
    # artist marker is group(2). Run twice: the first pass establishes that
    # marker, and the second extracts the query with the context-specific
    # tails. Both assertions are zero-width, at the start of the command.
    + r"(?:"
    + r"(?=" + _PLAY_PREFIX + r"\s+(.+?)"
    + r"(?:(?(2)" + _PLAY_ARTIST_TAIL + r"|(?!))|"
    + _PLAY_COMMON_TAIL + r")*" + _PLAY_END + r")"
    + r"(?:(?P<artist_object>(?=" + _PLAY_ARTIST_PREFIX + r"\s+\S))|)"
    + r"){2}"
    + r".+" + _PLAY_END,
    re.IGNORECASE | re.DOTALL,
)

# Task 154: this is only a candidate, never proof of playback intent. The
# execute-only registry handler must find an EXACT normalized library artist.
# Reserve send-object prefixes even if an artist happens to share their name;
# som/música also block generic requests like "manda uma música aí" from
# querying the library. Explicit artist objects use SPOTIFY_PLAY_RE first.
_SEND_OBJECT = (
    r"(?:e-?mail|mensagem|msg|oi|beijo|abra[çc]o|[áa]udio|foto|print|pix|"
    r"link|arquivo|resumo|lembrete|zap|recado|convite|v[íi]deo|documento|"
    r"sms|whatsapp|telegram|imessage|direct|dm|som|m[úu]sica)"
)
SPOTIFY_LIBRARY_ARTIST_RE = re.compile(
    r"^(?:manda|solta)" + _PLAY_PT_SENDING_GUARDS
    + r"(?:\s+a[ií])?\s+(?:um|uma)\s+"
    + r"(?!" + _SEND_OBJECT + r"\b)"
    # Bound X to 1–4 words before the playback-only tails. Sentence endings
    # are excluded from words so punctuation cannot become part of the name.
    + r"([^\s?!.]+(?:\s+[^\s?!.]+){0,3}?)"
    + r"(?:" + _PLAY_ARTIST_TAIL + r"|" + _PLAY_COMMON_TAIL + r")*"
    + _PLAY_END,
    re.IGNORECASE | re.DOTALL,
)

# Play a playlist by name. Two flavours kept as separate patterns so
# both stay readable:
#
#   1. PT-natural / EN-explicit:  "toca (minha|a) playlist <name>",
#      "play (my|the) playlist <name>". Playlist comes BEFORE the name.
#   2. EN-natural trailing-noun:  "play my <name> playlist",
#      "play the <name> playlist". Playlist comes AFTER the name —
#      e.g. "play my workout playlist". This is the most common EN
#      phrasing and won't match pattern (1).
#
# Both dispatched BEFORE SPOTIFY_PLAY_RE in the orchestrator (more
# specific first). Group(1) of either regex is the playlist name.
SPOTIFY_PLAY_PLAYLIST_RE = re.compile(
    r"^(?:"
    r"toca|toque|toquem|tocar|coloca|coloque|colocar|"
    r"p[õo]e|p[õo]r|ponha|ponham|play|put\s+on|"
    r"(?:pode|podes|podia|podiam|poderia|poderiam)\s+"
    r"(?:tocar|colocar|p[õôo]r|ouvir)"
    r")"
    r"\s+(?:a\s+|the\s+|minha\s+|my\s+)?playlist"
    r"\s+(.+?)"
    # Same trailing-fluff strip as SPOTIFY_PLAY_RE — without it the
    # playlist name swallows "no spotify" / device hints (e.g.
    # "toque minha playlist T(rap) no spotify" → name "T(rap) no
    # spotify", which never matches the user's real "T(rap)" list and
    # falls through to a junk public search). Device hint is captured
    # separately by SPOTIFY_DEVICE_HINT_RE on the original text.
    r"(?:"
    r"\s+(?:no|na|on|in)\s+spotify|"
    r"\s+(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(?:iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)"
    r")*"
    r"[?!.]*\s*$",
    re.IGNORECASE,
)

SPOTIFY_PLAY_PLAYLIST_EN_RE = re.compile(
    r"^(?:play|put\s+on)"
    r"\s+(?:my|the)"
    r"\s+(.+?)"
    r"\s+playlist"
    # Trailing-fluff strip (same rationale as SPOTIFY_PLAY_PLAYLIST_RE):
    # "play my workout playlist on my iphone" / "...on spotify".
    r"(?:"
    r"\s+(?:no|na|on|in)\s+spotify|"
    r"\s+(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(?:iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)"
    r")*"
    r"[?!.]*\s*$",
    re.IGNORECASE,
)

# Artist radio — "toca rádio do matuê", "põe uma radio de djonga",
# "play matue radio". An endless artist-seeded mix (recommendations
# beyond the artist's own catalogue), distinct from playlist and from
# plain artist-context play. Two flavours like the playlist pair:
#
#   1. PT/EN with "radio" BEFORE the name: "toca radio <name>",
#      "play radio <name>", optional "de/do/da/of" connector.
#   2. EN trailing-noun: "play <name> radio" — radio AFTER the name.
#
# Both need the literal "radio"/"rádio" word, dispatched BEFORE
# SPOTIFY_PLAY_RE so it doesn't swallow "radio <name>" as a track query.
# Group(1) is the artist name.
SPOTIFY_PLAY_RADIO_RE = re.compile(
    r"^(?:"
    r"toca|toque|toquem|tocar|coloca|coloque|colocar|"
    r"p[õo]e|p[õo]r|ponha|ponham|play|put\s+on|start|"
    r"(?:pode|podes|podia|podiam|poderia|poderiam)\s+"
    r"(?:tocar|colocar|p[õôo]r|ouvir)"
    r")"
    r"\s+(?:a\s+|the\s+|uma\s+|um\s+|minha\s+|my\s+)?r[áa]dio"
    r"(?:\s+(?:de|do|da|of))?"
    r"\s+(.+?)"
    r"(?:"
    r"\s+(?:no|na|on|in)\s+spotify|"
    r"\s+(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(?:iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)"
    r")*"
    r"[?!.]*\s*$",
    re.IGNORECASE,
)

SPOTIFY_PLAY_RADIO_EN_RE = re.compile(
    r"^(?:play|put\s+on|start)"
    r"\s+(?:a\s+|the\s+)?"
    r"(.+?)"
    r"\s+radio"
    r"(?:"
    r"\s+(?:no|na|on|in)\s+spotify|"
    r"\s+(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(?:iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)"
    r")*"
    r"[?!.]*\s*$",
    re.IGNORECASE,
)

# Detects the device hint anywhere in the play command so play_track
# can prefer that device (e.g. 'toque X no meu iphone' → wake iPhone,
# not the Mac). Group(1) is the normalised hint keyword.
SPOTIFY_DEVICE_HINT_RE = re.compile(
    r"\b(?:no|na|on|in)\s+(?:meu|minha|my)?\s*"
    r"(iphone|celular|telefone|phone|mac|macbook|computador|"
    r"computer|laptop|tv|caixinha|alto[-\s]?falante|speaker|"
    r"home\s*pod|homepod|fones?(?:\s+de\s+ouvido)?|headphones?)\b",
    re.IGNORECASE,
)

# EN coverage (sessão 11): the control verbs were PT-heavy. English
# now accepts the natural multi-word phrasings a user actually speaks
# ("pause the music", "next song", "skip this track", "what song is
# this", "save this song"), not just the bare keyword. PT forms are
# unchanged. All are full-match (^...$) and dispatched BEFORE the
# play regex, which needs a verb+argument and so never steals them.
SPOTIFY_PAUSE_RE = re.compile(
    r"^(?:"
    r"pausa(?:r)?(?:\s+a\s+m[úu]sica)?|"
    # Expanded 2026-05-28 sessão 13 bloco F bug 3. "pare" (3rd-person
    # imperative) was missing; "para de tocar" / "pare de tocar"
    # (with explicit "de tocar") were missing. Verified safe against
    # generic "para mim" / "para frente" — those have no trailing
    # "a música" / "de tocar" so they do not match.
    r"par(?:a|e|ar)(?:\s+(?:a\s+m[úu]sica|de\s+tocar(?:\s+a\s+m[úu]sica)?))?|"
    r"pause(?:\s+(?:the\s+)?(?:music|song|track|playback|spotify|it))?|"
    r"stop(?:\s+(?:the\s+)?(?:music|song|track|playback|spotify))?"
    r")$",
    re.IGNORECASE,
)

SPOTIFY_RESUME_RE = re.compile(
    r"^(?:"
    r"continua(?:r)?(?:\s+a\s+m[úu]sica)?|"
    r"despausa(?:r)?|"
    r"volta(?:r)?\s+a\s+tocar|"
    r"resume(?:\s+(?:the\s+)?(?:music|song|track|playback))?|"
    r"unpause|"
    r"continue(?:\s+(?:playing|the\s+(?:music|song|track)))?|"
    r"keep\s+playing|"
    # Bare verb with no argument = press play / resume (mirrors the
    # Spotify play button). Without this, "play" / "toca" alone matches
    # no dispatch regex and falls to the LLM, which leaked a raw
    # tool-call object ({"name": "spotify_control", ...}) to the user.
    r"play|toca|toque"
    r")$",
    re.IGNORECASE,
)

# "vai para a próxima" went to MAC_APP_SWITCH as an app named "próxima"
# (s88, Codex lote 9); going to the next one is skipping.
SPOTIFY_NEXT_RE = re.compile(
    r"^(?:"
    r"(?:(?:vai|passa)\s+(?:pro|pra|para)\s+(?:a\s+|o\s+)?)?pr[óo]xim[ao](?:\s+m[úu]sica)?|"
    r"pula(?:r)?(?:\s+essa(?:\s+m[úu]sica)?)?|"
    r"skip(?:\s+(?:this|the))?(?:\s+(?:song|track|one))?|"
    r"next(?:\s+(?:song|track|one))?"
    r")$",
    re.IGNORECASE,
)

SPOTIFY_PREV_RE = re.compile(
    r"^(?:"
    r"(?:(?:vai|volta)\s+(?:pro|pra|para)\s+(?:a\s+|o\s+)?)?(?:m[úu]sica\s+)?anterior|"
    r"volta(?:r)?(?:\s+(?:uma\s+)?m[úu]sica)?|"
    r"previous(?:\s+(?:song|track|one))?|prev|"
    r"go\s+back|"
    r"(?:the\s+)?last\s+(?:song|track)"
    r")$",
    re.IGNORECASE,
)

SPOTIFY_NOW_RE = re.compile(
    r"^(?:o\s+que\s+(?:est[áa]|t[áa])\s+tocando|"
    # s75: "que música está/tá tocando (agora)" — natural PT phrasing
    # caught falling to chat during the locked-screen battery.
    r"(?:que|qual)\s+m[úu]sica\s+(?:est[áa]|t[áa])\s+tocando(?:\s+agora)?|"
    r"que\s+m[úu]sica\s+[ée]\s+essa|"
    r"qual\s+(?:[ée]\s+)?(?:a\s+)?m[úu]sica\??|"
    r"now\s+playing|what(?:'s|\s+is)\s+playing|"
    r"what(?:'s|\s+is)\s+(?:(?:this|the)\s+)?(?:song|track)|"
    r"what\s+(?:song|track)\s+is\s+(?:this|playing)|"
    r"current\s+(?:song|track))\??$",
    re.IGNORECASE,
)

SPOTIFY_VOLUME_RE = re.compile(
    r"^(?:(?:volume|vol)\s+(?:do\s+)?spotify|spotify\s+(?:volume|vol))"
    r"\s+(?:(?:para|to|em|at|no)\s+)?(\d{1,3})%?$",
    re.IGNORECASE,
)

# Transfer active playback to another device ("troca pro iphone",
# "transfer to the macbook"). Group(1) is the device hint, fed straight
# to SpotifyPlayer.transfer_playback_to → _pick_device. Requires an
# explicit transfer verb + a known device noun, so it never collides
# with play_track's "toca X no iphone" (which is play-with-target, not
# move-current). Dispatched BEFORE play in the orchestrator. Device
# vocabulary mirrors SPOTIFY_DEVICE_HINT_RE / _HINT_TYPE_MAP.
SPOTIFY_TRANSFER_RE = re.compile(
    r"^(?:"
    r"(?:troca|trocar|passa|passar|transfere|transferir|manda|mandar|"
    r"joga|jogar|muda|mudar)\s+"
    r"(?:o\s+som\s+|a\s+m[úu]sica\s+|tudo\s+|isso\s+)?"
    r"(?:pro|pra|para\s+o|para\s+a|para)\s+(?:meu\s+|minha\s+|o\s+|a\s+)?|"
    r"(?:transfer|move|switch|cast)\s+"
    r"(?:playback\s+|the\s+music\s+|it\s+|this\s+)?"
    r"(?:to|onto)\s+(?:my\s+|the\s+)?"
    r")"
    r"(iphone|celular|telefone|phone|mac|macbook|computador|computer|laptop|"
    r"tv|caixinha|alto[-\s]?falante|speaker|home\s*pod|homepod|"
    r"fones?(?:\s+de\s+ouvido)?|headphones?)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

SPOTIFY_LIKE_RE = re.compile(
    r"^(?:"
    r"curte|curtir|salva(?:r)?|favorito|"
    r"adiciona(?:r)?(?:\s+aos\s+favoritos)?|"
    r"like|save|favou?rite|"
    r"add\s+to\s+(?:my\s+)?(?:favou?rites|library)"
    r")"
    r"(?:\s+(?:essa|esta)\s+m[úu]sica|\s+this\s+(?:song|track))?$",
    re.IGNORECASE,
)
