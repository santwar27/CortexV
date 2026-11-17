"""macOS system-control dispatch regexes (sessão 10 bloco G).

Same pattern as `src/core/dispatch/spotify.py`: module-level constants
that the orchestrator class re-binds as attributes so tests can
reference them via `CortexOrchestrator._MAC_X_RE` without spelunking
into private modules.

Conflict policy with Spotify:
  - Bare "volume 50" controls the Mac. Spotify volume requires Spotify
    to be named explicitly ("volume do spotify 50" / "spotify volume 50").
  - "aumenta o volume" / "abaixa o volume" / "volume up" / "volume
    down" / "mudo" / "mute" have no Spotify counterpart, so they
    route here without ambiguity.

Coverage (PT-BR primary, EN aliases secondary):
  - System volume — set / up / down / mute / unmute
  - Brightness — up / down / set
  - Keyboard backlight — up / down / set / on / off
  - Dark mode — toggle / on / off
  - Lock screen
  - Screenshot
  - App open

DND, app-quit, sleep, focus modes are deliberately NOT here — see
JOURNAL sessão 10 bloco G for the deferred list and rationale.
"""

from __future__ import annotations

import re

# Shared with the gatekeeper's revoke patterns (single source of truth) —
# see the note on NET_WORDS_RE in gatekeeper.py for why drift here is a bug.
from src.core.gatekeeper import NET_WORDS_RE
from src.core.dispatch.ps5 import PS5_WORDS


# ----------------------------------------------------------------------
# System volume
# ----------------------------------------------------------------------

_VOLUME_LOCATION = r"(?:\s+(?:do\s+mac|(?:do\s+)?sistema|do\s+computador))?"

MAC_MUTE_RE = re.compile(
    r"^(?:"
    r"mudo|silencia(?:r)?(?:\s+o\s+sistema)?|"
    r"silencia\s+tudo|"
    r"(?:tira(?:r)?|tire)\s+o\s+som" + _VOLUME_LOCATION + r"|"
    r"mute|mute\s+system"
    r")$",
    re.IGNORECASE,
)

MAC_UNMUTE_RE = re.compile(
    r"^(?:"
    r"(?:tira(?:r)?|desativa(?:r)?)\s+(?:o\s+)?mudo|"
    r"desmuta(?:r)?|"
    r"unmute"
    r")$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Brightness
# ----------------------------------------------------------------------

_BRIGHTNESS_DOWN_VERBS = (
    r"abaixa|abaixe|abaixar|baixa|baixe|baixar|diminui|diminua|diminuir|"
    r"reduz|reduza|reduzir|desce|des[cç]a|descer|escurece|escure[cç]a"
)
_BRIGHTNESS_UP_VERBS = r"aumenta|aumente|aumentar|sobe|suba|subir|clareia|clareie"
_BRIGHTNESS_DOWN_EN_VERBS = r"lower|decrease|dim|turn\s+down"
_BRIGHTNESS_UP_EN_VERBS = r"raise|increase|turn\s+up"
_BRIGHTNESS_SET_PREFIX = (
    r"(?:(?:" + _BRIGHTNESS_DOWN_VERBS + r"|" + _BRIGHTNESS_UP_VERBS +
    r"|p[õo]e|coloca|deixa|muda|ajusta|set|put)\s+)?"
)
_BRIGHTNESS_LOCATION = r"\s+(?:da\s+tela|do\s+mac|no\s+mac|do\s+monitor)"
# Light needs an explicit location in Portuguese; bare "luz" could be a room light.
_SCREEN_PT_LIGHT = r"(?:a\s+)?luz" + _BRIGHTNESS_LOCATION
_SCREEN_PT_OBJECT = (
    r"(?:(?:o\s+)?brilho(?:" + _BRIGHTNESS_LOCATION + r")?|" + _SCREEN_PT_LIGHT + r")"
)
_SCREEN_EN_LOCATION = r"\s+(?:of\s+the\s+screen|on\s+the\s+(?:mac|screen))"
_SCREEN_EN_DEVICE = r"(?:screen|display|mac(?:'s)?)\s+"
# Like "luz", "light" needs a screen object or location; "brightness" does not.
_SCREEN_EN_VALUE_OBJECT = (
    r"(?:the\s+)?(?:"
    r"(?:" + _SCREEN_EN_DEVICE + r")?brightness(?:" + _SCREEN_EN_LOCATION + r")?|"
    + _SCREEN_EN_DEVICE + r"light(?:" + _SCREEN_EN_LOCATION + r")?|"
    r"light" + _SCREEN_EN_LOCATION + r")"
)
# Preserve the legacy set grammar's Portuguese article before English objects.
_SCREEN_SET_EN_OBJECT = r"(?:o\s+)?" + _SCREEN_EN_VALUE_OBJECT
_SCREEN_EN_OBJECT = (
    r"(?:" + _SCREEN_EN_VALUE_OBJECT + r"|the\s+(?:screen|display)(?:" + _SCREEN_EN_LOCATION + r")?)"
)
_SCREEN_EN_TAIL = r"(?:\s+a\s+(?:bit|little))?"
_SCREEN_PT_DEVICE = r"a\s+tela"


def _screen_relative(verbs: str, en_verbs: str, screen_verbs: str, direction: str,
                     pt_direction: str, comparative: str, pt_comparative: str) -> re.Pattern:
    # Keep screen additions separate from the verb sets reused by keyboard/volume.
    return re.compile(
        r"^(?:(?:" + verbs + r")\s+" + _SCREEN_PT_OBJECT + r"(?:\s+um\s+pouco)?|"
        r"(?:" + screen_verbs + r")\s+" + _SCREEN_PT_DEVICE + r"(?:\s+um\s+pouco)?|"
        r"deixa\s+" + _SCREEN_PT_DEVICE + r"\s+mais\s+(?:" + pt_comparative + r")|"
        + _SCREEN_PT_OBJECT + r"\s+(?:" + pt_direction + r")|"
        r"(?:" + en_verbs + r")\s+" + _SCREEN_EN_OBJECT + _SCREEN_EN_TAIL + r"|"
        r"turn\s+" + _SCREEN_EN_OBJECT + r"\s+" + direction + _SCREEN_EN_TAIL + r"|"
        r"make\s+the\s+(?:screen|display)\s+(?:" + comparative + r")" + _SCREEN_EN_TAIL + r"|"
        + _SCREEN_EN_VALUE_OBJECT + r"\s+" + direction + _SCREEN_EN_TAIL + r"|"
        + (r"brighter" if direction == "up" else r"dimmer") + r")\s*\??$",
        re.IGNORECASE,
    )


MAC_BRIGHTNESS_UP_RE = _screen_relative(
    _BRIGHTNESS_UP_VERBS, _BRIGHTNESS_UP_EN_VERBS + r"|brighten", r"clareia|clareie",
    "up", r"mais|para\s+cima", "brighter", "clara|clarinha",
)
MAC_BRIGHTNESS_DOWN_RE = _screen_relative(
    _BRIGHTNESS_DOWN_VERBS, _BRIGHTNESS_DOWN_EN_VERBS + r"|reduce", r"escurece|escure[cç]a",
    "down", r"menos|para\s+baixo", "darker|dimmer", "escura|escurinha",
)

# Numeric capture stays group 1 for existing callers and the simulator.
# Named limit captures let the registry map min/max without guessing numbers.
MAC_BRIGHTNESS_SET_RE = re.compile(
    r"^"
    + _BRIGHTNESS_SET_PREFIX +
    r"(?:"
    r"(?:(?:o\s+)?brilho(?:" + _BRIGHTNESS_LOCATION + r"|\s+tela)?|"
    + _SCREEN_PT_LIGHT + r"|" + _SCREEN_SET_EN_OBJECT + r")"
    r"\s+(?:(?:para|em|a|no|to|at)\s+)?(\d+)%?|"
    r"(?P<minimum>" + _SCREEN_PT_OBJECT + r"\s+no\s+m[ií]nimo|"
    + _SCREEN_SET_EN_OBJECT + r"\s+(?:to\s+)?min(?:imum)?|"
    r"min(?:imum)?\s+" + _SCREEN_EN_VALUE_OBJECT + r")|"
    r"(?P<maximum>" + _SCREEN_PT_OBJECT + r"\s+no\s+m[áa]ximo|"
    + _SCREEN_SET_EN_OBJECT + r"\s+(?:to\s+)?max(?:imum)?|"
    r"max(?:imum)?\s+" + _SCREEN_EN_VALUE_OBJECT + r")"
    r")\s*\??$",
    re.IGNORECASE,
)


# Keyboard objects are mandatory here and absent from the screen grammar.
_KEYBOARD_PT_TARGET = r"teclado(?:\s+do\s+mac)?"
_KEYBOARD_PT_OBJECT = r"(?:brilho|luz)\s+do\s+" + _KEYBOARD_PT_TARGET
_KEYBOARD_PT_DEVICE = r"o\s+" + _KEYBOARD_PT_TARGET
_KEYBOARD_PT_LIGHT = r"(?:a\s+)?luz\s+do\s+" + _KEYBOARD_PT_TARGET
_KEYBOARD_OFF_VERBS = r"apaga|apague|apagar"
_KEYBOARD_ON_VERBS = r"acende|acenda|acender"
_KEYBOARD_EN_OBJECT = r"keyboard\s+(?:brightness|light)"
_KEYBOARD_OBJECT = r"(?:" + _KEYBOARD_PT_OBJECT + r"|" + _KEYBOARD_EN_OBJECT + r")"

MAC_KEYBOARD_BRIGHTNESS_SET_RE = re.compile(
    r"^" + _BRIGHTNESS_SET_PREFIX + r"(?:(?:o|a|the)\s+)?(?:"
    + _KEYBOARD_OBJECT + r"\s+(?:(?:para|em|a|no|to|at)\s+)?(\d+)%?|"
    r"(?P<minimum>" + _KEYBOARD_PT_OBJECT + r"\s+no\s+m[ií]nimo|"
    + _KEYBOARD_EN_OBJECT + r"\s+(?:to\s+)?min(?:imum)?)|"
    r"(?P<maximum>" + _KEYBOARD_PT_OBJECT + r"\s+no\s+m[áa]ximo|"
    + _KEYBOARD_EN_OBJECT + r"\s+(?:to\s+)?max(?:imum)?)|"
    r"(?P<off>(?:" + _KEYBOARD_OFF_VERBS + r"|desliga|desligue|desligar)\s+" + _KEYBOARD_PT_LIGHT + r"|"
    r"(?:" + _KEYBOARD_OFF_VERBS + r")\s+" + _KEYBOARD_PT_DEVICE + r"|keyboard\s+light\s+off|"
    r"turn\s+off\s+(?:the\s+)?keyboard\s+light)|"
    r"(?P<on>(?:" + _KEYBOARD_ON_VERBS + r"|liga|ligue|ligar)\s+" + _KEYBOARD_PT_LIGHT + r"|"
    r"(?:" + _KEYBOARD_ON_VERBS + r")\s+" + _KEYBOARD_PT_DEVICE + r"|keyboard\s+light\s+on|"
    r"turn\s+on\s+(?:the\s+)?keyboard\s+light)"
    r")$", re.IGNORECASE,
)


def _keyboard_relative(verbs: str, en_verbs: str, direction: str, pt_direction: str,
                       comparative: str) -> re.Pattern:
    return re.compile(
        r"^(?:(?:" + verbs + r")\s+(?:(?:o|a)\s+)?" + _KEYBOARD_PT_OBJECT +
        r"(?:\s+um\s+pouco)?|" + _KEYBOARD_PT_OBJECT + r"\s+(?:" + pt_direction + r")|"
        r"deixa\s+" + _KEYBOARD_PT_DEVICE + r"\s+mais\s+(?:" + comparative + r")|"
        + _KEYBOARD_EN_OBJECT + r"\s+" + direction + r"|"
        r"(?:" + en_verbs + r")\s+(?:the\s+)?" + _KEYBOARD_EN_OBJECT + r")$",
        re.IGNORECASE,
    )


MAC_KEYBOARD_BRIGHTNESS_UP_RE = _keyboard_relative(
    _BRIGHTNESS_UP_VERBS, _BRIGHTNESS_UP_EN_VERBS, "up", r"mais|para\s+cima", "forte|fortinho",
)
MAC_KEYBOARD_BRIGHTNESS_DOWN_RE = _keyboard_relative(
    _BRIGHTNESS_DOWN_VERBS, _BRIGHTNESS_DOWN_EN_VERBS, "down", r"menos|para\s+baixo", "fraco|fraquinho",
)


# The same natural verb families as brightness; explicit Spotify locations
# are intentionally absent. Numeric capture remains group 1 for callers.
_VOLUME_OBJECT = r"(?:vol(?:ume)?" + _VOLUME_LOCATION + r"|system\s+volume)"
_VOLUME_RELATIVE_TAIL = (
    r"\s+(?:o\s+|um\s+pou(?:co|quinho)\s+(?:o\s+|do\s+)?)?"
    r"(?:volume|som)" + _VOLUME_LOCATION + r"(?:\s+um\s+pou(?:co|quinho))?"
)

MAC_VOLUME_SET_RE = re.compile(
    r"^(?:(?:" + _BRIGHTNESS_DOWN_VERBS + r"|" + _BRIGHTNESS_UP_VERBS +
    r"|p[õo]e|coloca|deixa|muda|ajusta|set|put)\s+)?(?:o\s+)?"
    + _VOLUME_OBJECT + r"\s+(?:"
    r"(?:(?:para|em|no|a|to|at)\s+)?(\d+)%?|"
    r"(?P<minimum>no\s+m[ií]nimo|(?:to\s+)?min(?:imum)?)|"
    r"(?P<maximum>no\s+m[áa]ximo|(?:to\s+)?max(?:imum)?)"
    r")$", re.IGNORECASE,
)

MAC_VOLUME_UP_RE = re.compile(
    r"^(?:(?:" + _BRIGHTNESS_UP_VERBS + r")" + _VOLUME_RELATIVE_TAIL + r"|"
    r"deixa\s+(?:o\s+)?(?:volume|som)" + _VOLUME_LOCATION + r"\s+mais\s+(?:alto|altinho)|"
    r"volume\s+(?:up|mais|mais\s+alto)|"
    r"(?:raise|increase|turn\s+up)\s+(?:the\s+)?volume)$", re.IGNORECASE,
)

MAC_VOLUME_DOWN_RE = re.compile(
    r"^(?:(?:" + _BRIGHTNESS_DOWN_VERBS + r")" + _VOLUME_RELATIVE_TAIL + r"|"
    r"deixa\s+(?:o\s+)?(?:volume|som)" + _VOLUME_LOCATION + r"\s+mais\s+(?:baixo|baixinho)|"
    r"volume\s+(?:down|menos|mais\s+baixo)|"
    r"(?:lower|decrease|turn\s+down)\s+(?:the\s+)?volume)$", re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Dark mode
# ----------------------------------------------------------------------

# Toggle goes first in dispatch order (more specific verbs).
MAC_DARK_TOGGLE_RE = re.compile(
    r"^(?:"
    r"alterna(?:r)?\s+(?:o\s+)?(?:tema|modo)|"
    r"troca(?:r)?\s+(?:o\s+)?(?:tema|modo)|"
    r"inverte(?:r)?\s+(?:o\s+)?(?:tema|modo)|"
    r"toggle\s+dark\s+mode|"
    r"toggle\s+theme"
    r")$",
    re.IGNORECASE,
)

_DARK_PREFIX = (
    r"(?:(?:ativa(?:r)?|liga(?:r)?|coloca(?:r)?|p[õo]e(?:r)?|ponha|deixa|muda|troca)\s+)?"
    r"(?:(?:o\s+mac|a\s+tela|o\s+computador)\s+)?"
    r"(?:(?:no|na|em|pro|pra|para(?:\s+[oa])?)\s+)?(?:o\s+)?"
)

MAC_DARK_ON_RE = re.compile(
    r"^(?:"
    + _DARK_PREFIX + r"(?:modo|tema)\s+(?:escuro|noturno|dark)|"
    r"dark\s+mode(?:\s+(?:on|please))?|"
    r"(?:enable|switch\s+to|turn\s+on)\s+dark\s+mode"
    r")$",
    re.IGNORECASE,
)

MAC_DARK_OFF_RE = re.compile(
    r"^(?:"
    + _DARK_PREFIX + r"(?:modo|tema)\s+(?:claro|diurno|light)|"
    r"(?:desativa(?:r)?|desliga(?:r)?|tira(?:r)?)\s+(?:o\s+)?(?:modo|tema)\s+(?:escuro|noturno|dark)|"
    r"light\s+mode(?:\s+(?:on|please))?|"
    r"(?:disable|turn\s+off)\s+dark\s+mode|"
    r"(?:switch\s+to|turn\s+on)\s+light\s+mode"
    r")$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Lock screen
# ----------------------------------------------------------------------

# STT/accent aliases for "Mac": the Brazilian accent ("méqui") makes
# Whisper small transcribe "Mac" as "meck"/"meque"/"mec", so mac-control
# commands ("trava o mac", "dorme o mac") missed the regex. These are
# near-homophones with low collision risk, and every mac-control regex here
# is ^…$-anchored, so a longer word like "mecânico" can't false-match.
# Folded into the device alternation in the lock + sleep patterns below.
# Root fix is Whisper medium/large on the M4 Max; this is the cheap one.
#
# "Desliga a tela" means LOCK: the display sleeps and the Mac asks for the
# password at once (the user's screenLock delay is immediate). Without this the
# phrase fell into MAC_QUIT_APP_RE ("desligar" is a quit verb) and staged
# quitting an app called "tela" — seen live 2026-09-19 over Blink. Shared by
# MAC_LOCK_RE (accepts it) and MAC_QUIT_APP_RE (refuses it as an app name) so
# the two sets can't drift, the NET_WORDS_RE recipe. End-anchored where it is
# refused, so a real app like "Screen Sharing" still quits.
# The machine itself, as the user names it (STT aliases included). Shared by
# MAC_SLEEP_RE (accepts it) and MAC_QUIT_APP_RE (refuses it as an app name),
# same reason as SCREEN_TARGET_RE below: "desliga o mac" staged quitting an
# app called "mac". End-anchored where refused, so "quit MacUpdater" survives.
DEVICE_TARGET_RE = r"(?:mac|macbook|meck|meque|mec|computador|notebook)"

SCREEN_TARGET_RE = (
    r"(?:a\s+|o\s+|the\s+)?(?:tela|monitor|display|screen)"
    r"(?:\s+d[oa]\s+(?:mac|meck|meque|mec|computador|notebook))?"
)

MAC_LOCK_RE = re.compile(
    r"^(?:"
    r"trava(?:r)?(?:\s+(?:o\s+(?:mac|meck|meque|mec)|a\s+tela|o\s+computador))?|"
    r"bloqueia(?:r)?(?:\s+(?:a\s+tela|o\s+(?:mac|meck|meque|mec)|o\s+computador))?|"
    r"lock(?:\s+(?:screen|mac|meck|meque|mec|the\s+screen))?|"
    r"(?:desliga(?:r)?|desligue|apaga(?:r)?|apague|turn\s+off|switch\s+off)\s+"
    + SCREEN_TARGET_RE + r"|"
    r"(?:screen|display)\s+off"
    # Trailing punctuation: Whisper transcribes "Trava a tela." with the
    # period, which the bare `$` refused — open/close/quit already allow it.
    r")[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Screenshot
# ----------------------------------------------------------------------

MAC_SCREENSHOT_RE = re.compile(
    r"^(?:"
    r"tira(?:r)?\s+(?:um\s+)?print(?:\s+(?:da\s+tela|de\s+tela))?|"
    r"captura(?:r)?\s+(?:a\s+)?tela|"
    r"print\s+(?:da\s+|de\s+)?tela|"
    r"screenshot|"
    r"take\s+(?:a\s+)?screenshot|"
    r"capture\s+(?:the\s+)?screen"
    # Trailing punctuation — "tira print." from a Whisper transcript.
    r")[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# App open
# ----------------------------------------------------------------------

# App-open is BROADER than the others, so it lives last in the
# orchestrator dispatch order — after specific app-driven verbs
# (LIST_EMAILS, calendar trigger, planner, etc.) have had a chance
# to fire. Group(1) is the raw app name; `open -a` resolves common
# variations case-insensitively.
MAC_OPEN_APP_RE = re.compile(
    r"^(?![\s\S]*" + PS5_WORDS + r")"
    r"^(?:abre|abrir|abra|lan[çc]a(?:r)?|inicia(?:r)?|"
    r"open|launch|start)\s+(?:o\s+app\s+|a\s+aplica[çc][ãa]o\s+|the\s+app\s+|o\s+|a\s+|the\s+)?"
    r"(.+?)"
    r"(?:\s+app)?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# App close-window (NOT destructive — like clicking the red button / Cmd+W)
# ----------------------------------------------------------------------

# The user's mental model (2026-06-03): "fecha o X" = close the WINDOW (red
# button), the app keeps running; quit is a separate, heavier thing. So
# "fecha/feche/fechar" + EN "close" route here (no confirmation — it's the
# red button), while quit verbs route to MAC_QUIT_APP_RE below. The verb
# sets are disjoint, so dispatch order between them doesn't matter. Optional
# "a janela do / the window of" lets "fecha a janela do Spotify" work too.
MAC_CLOSE_WINDOW_RE = re.compile(
    r"^(?:"
    r"fecha(?:r)?|feche|"
    r"close"
    r")\s+(?:o\s+app\s+|a\s+aplica[çc][ãa]o\s+|the\s+app\s+|"
    r"a\s+janela\s+d[eo]\s+|the\s+window\s+of\s+|o\s+|a\s+|the\s+)?"
    r"(?P<app>.+?)"
    r"(?:\s+app)?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# App quit (S3 — staged behind sim/não in the orchestrator)
# ----------------------------------------------------------------------

# Quit = fully terminate the app (Cmd+Q), destructive (unsaved work). Verbs
# are disjoint from close-window (fecha/close), open (abre/open) and switch.
# The user reserves quit for explicit "quit/end" (EN) and "encerra/sai de/
# desligar" (PT). Group "app" → MacOSController.quit_app. Unlike close/open,
# this stages a pending action and waits for "sim".
# The negative lookahead after the verb keeps the lease-revoke phrasings out:
# "desliga a internet/net/wi-fi" is a STATE verb (gatekeeper revoke), not an
# app named "internet" — without it the quit staging eats the input ~50 lines
# before evaluate_gate ever runs and the revoke branch is dead code (M6-A1).
# The second lookahead does the same for the screen: "desliga a tela" is a
# lock (MAC_LOCK_RE), never an app called "tela".
MAC_QUIT_APP_RE = re.compile(
    r"^(?:"
    r"encerra(?:r)?|"
    r"sai\s+d[eo]|sair\s+d[eo]|"
    r"desliga(?:r)?|desligue|"
    r"quit|exit|end"
    r")\s+(?!(?:o\s+|a\s+|the\s+)?" + NET_WORDS_RE + r"\b)"
    r"(?!" + SCREEN_TARGET_RE + r"[\s.!?]*$)"
    r"(?!(?:o\s+|the\s+)?" + DEVICE_TARGET_RE + r"[\s.!?]*$)"
    r"(?:o\s+app\s+|a\s+aplica[çc][ãa]o\s+|the\s+app\s+|o\s+|a\s+|the\s+)?"
    r"(?P<app>.+?)"
    r"(?:\s+app)?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# App list / switch (S2, batch 3)
# ----------------------------------------------------------------------

# List the foreground apps. Anchored full-match phrases (no capture).
MAC_APP_LIST_RE = re.compile(
    r"^(?:"
    r"(?:quais|que)\s+(?:s[ãa]o\s+)?(?:os\s+)?apps?(?:\s+(?:est[ãa]o|t[ãa]o))?\s+abertos?|"
    r"(?:lista(?:r)?|mostra(?:r)?)\s+(?:os\s+)?apps?(?:\s+abertos?)?|"
    r"apps?\s+abertos?|"
    r"what\s+apps?\s+(?:are\s+)?(?:open|running)|"
    r"(?:list|show)\s+(?:open\s+|running\s+)?apps?|"
    r"which\s+apps?\s+(?:are\s+)?(?:open|running)"
    r")\??$",
    re.IGNORECASE,
)

# Switch focus to an app. Verbs (vai pro / foca / switch to / focus on /
# go to) are disjoint from open (abre/open) and quit (fecha/quit), so no
# collision. Deliberately omits "muda" to avoid overlap with the Spotify
# transfer verb. Group "app" → MacOSController.switch_to_app.
MAC_APP_SWITCH_RE = re.compile(
    r"^(?:"
    r"vai\s+(?:pro|pra|para\s+o|para\s+a|para)|"
    r"foca(?:r)?\s+(?:no|na|em|o|a)|"
    r"switch\s+to|focus\s+on|go\s+to"
    r")\s+(?:o\s+app\s+|a\s+aplica[çc][ãa]o\s+|the\s+app\s+|o\s+|a\s+|the\s+)?"
    # "próxima"/"anterior" is a track, not an app (s88, Codex lote 9).
    r"(?!(?:(?:o|a|the)\s+)?(?:pr[óo]xim[ao]|anterior|next|previous)\b)"
    r"(?P<app>.+?)"
    r"(?:\s+app)?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# System sleep (S3 — staged behind sim/não in the orchestrator)
# ----------------------------------------------------------------------

# "Put the mac to sleep" — destructive (current session work is paused,
# user might lose unsaved focus). Same staging pattern as MAC_QUIT_APP_RE.
# No capture group: the action is parameter-free.
#
# Verb shapes covered (PT + EN):
#   - "dorme o mac" / "dormir" / "põe o mac pra dormir"
#   - "coloca o mac pra dormir" / "manda o mac dormir"
#   - "suspende o mac"
#   - "sleep" / "sleep the mac" / "put the mac to sleep"
#
# BR accent / STT note: Whisper often drops the final "r" of the
# infinitive ("dormir" → "dormi"/"dormí"), so the verb is matched as
# `dorm[ií]r?` — covers dormi / dormir / dormí / dormír. Same family as
# the "Meck" device aliases below. The device qualifier + `$` anchor keep
# the bare-"dormi" form from false-positiving.
#
# Deliberately requires either the verb at sentence start OR an explicit
# "mac/sistema" qualifier — bare "dorme" / "sleep" alone are too easy
# to false-positive on conversational input ("hoje vou dormir cedo").
MAC_SLEEP_RE = re.compile(
    r"^(?:"
    # PT — "põe/coloca/manda o mac pra dormir" (STT may drop the final r)
    r"(?:p[õo]e|p[õo]r|coloca(?:r)?|manda(?:r)?)\s+"
    r"(?:o\s+|esse\s+)?(?:mac|macbook|meck|meque|mec|computador|sistema)"
    r"(?:\s+pra|\s+para)?\s+dorm[ií]r?"
    r"|"
    # PT — "dorme o mac" / "dormir mac" (STT may drop the final r)
    r"(?:dorme|dorm[ií]r?)\s+(?:o\s+)?(?:mac|macbook|meck|meque|mec|computador|sistema)"
    r"|"
    # PT — "suspende o mac" / "suspender mac"
    r"(?:suspende(?:r)?)\s+(?:o\s+)?(?:mac|macbook|meck|meque|mec|computador|sistema)"
    r"|"
    # EN — "sleep the mac" / "sleep mac" / "put the mac to sleep"
    r"(?:put\s+(?:the\s+)?(?:mac|macbook|meck|meque|mec|computer|system)\s+to\s+sleep)"
    r"|"
    r"(?:sleep)\s+(?:the\s+)?(?:mac|macbook|meck|meque|mec|computer|system)"
    r"|"
    # PT — "hiberna o mac" / "põe o mac em repouso" / "modo repouso".
    # "Repouso" is what the Apple menu calls sleep in pt-BR.
    r"(?:hiberna(?:r)?)\s+(?:o\s+)?" + DEVICE_TARGET_RE + r"|"
    r"(?:p[õo]e|p[õo]r|coloca(?:r)?|manda(?:r)?)\s+(?:o\s+|esse\s+)?"
    + DEVICE_TARGET_RE + r"\s+em\s+repouso|"
    r"(?:modo\s+)?repouso"
    r"|"
    # PT/EN — device first: "mac dorme" / "mac sleep"
    r"(?:o\s+)?" + DEVICE_TARGET_RE + r"\s+(?:dorme|sleep)"
    r"|"
    # EN — "go to sleep" / "sleep now". Without the first one "go to
    # sleep" was a SWITCH ("go to" is the focus verb) to an app "sleep".
    r"go\s+to\s+sleep|sleep\s+now"
    r"|"
    # PT — "desliga o mac" means sleep here (the user, 2026-09-19). Cortex
    # does not shut the Mac down: over Blink that is irreversible from
    # afar. The named group makes the confirmation say so out loud.
    r"(?P<shutdown>(?:desliga(?:r)?|desligue)\s+(?:o\s+|esse\s+)?"
    + DEVICE_TARGET_RE + r")"
    r")"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Active-context: reply to the email in front (inovação #3, Bloco A)
# ----------------------------------------------------------------------

# "responde esse email" → read the message SELECTED in Mail, draft a reply,
# stage it through the send_email gate. Parameter-free (no capture): the
# handler reads the selection itself. The verb is "responde/responder/reply",
# disjoint from send ("manda/envia"), so no collision with SEND_EMAIL_*_RE.
# Requires the "esse/este/o/this/that" + "email" tail so it can't steal a
# generic "responde isso" (that's the summarise/what-is family, later blocks).
# STT note: Whisper may drop the final "r" of "responder" → `responde[r]?`.
MAC_REPLY_EMAIL_RE = re.compile(
    r"^(?:"
    r"responde(?:r)?|responda|"
    r"reply(?:\s+to)?"
    r")\s+"
    r"(?:a\s+|to\s+)?"
    r"(?:esse|este|essa|esta|o|the|this|that)\s+"
    r"(?:e-?mail|mensagem|message)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Active-context: summarise what's in front (inovação #3, Bloco B)
# ----------------------------------------------------------------------

# "resume isso" → summarise the active browser page or the selected text.
# Requires an explicit object (isso/isto/aqui/a página/o texto/this/it) so it
# can't steal the bare "resume" that SPOTIFY_RESUME_RE owns (resume playback);
# Spotify's regex is anchored, so "resume isso" already fails its `$`. The
# optional leading "me " covers "me resume isso".
MAC_SUMMARISE_RE = re.compile(
    r"^(?:me\s+)?(?:"
    # PT — resume/resuma/resumir + the thing in front. The verb group
    # covers resume/resuma/resumir/resumar (the old `resum[ae]r?` MISSED
    # "resumir", so "resumir isso" fell through to the LLM and got
    # hallucinated — s76 live smoke). The object list grew the same way:
    # "resume esse pdf" / "resume o conteúdo na minha frente" are how the user
    # actually says it out loud, but the old list only had bare
    # demonstratives + página/texto/tela, so they leaked to chat too.
    r"resum(?:ir|[ae]r?)\s+(?:"
    r"isso|isto|aqui|"
    r"(?:esse|este|essa|esta|o|a)\s+"
    r"(?:p[áa]gina|texto|tela|pdf|documento|doc|arquivo|conte[úu]do)|"
    r"esse|este|essa|esta|"
    r"(?:o\s+)?conte[úu]do(?:\s+(?:d[ae]|n[ae])\s+(?:minha\s+)?frente)?"
    r")|"
    # PT — standalone "o que tem/tá/está/diz aqui / na frente / nessa página|tela"
    r"o\s+que\s+(?:tem|t[áa]|est[áa]|diz)\s+"
    r"(?:aqui|n[ae]\s+(?:minha\s+)?frente|(?:nessa|nesta)\s+(?:p[áa]gina|tela))|"
    # EN — summarise this / the page / it ; sum this up ; tldr
    r"summari[sz]e\s+(?:this|the\s+page|it)|"
    r"sum\s+this\s+up|"
    r"tl;?dr"
    r")[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Active-context: "o que é isso?" — identify what's in front (inovação #3, C)
# ----------------------------------------------------------------------

# Explain what's in front: text when there is text, else a screenshot through
# the local vision model. Tightly full-match anchored to the bare phrase so a
# conversational "o que é isso que você falou?" (extra words) does NOT hijack —
# only the standalone command fires. Distinct from MAC_SUMMARISE ("o que TEM
# aqui" → resume; here it's "o que É isso" → identify).
MAC_WHAT_IS_RE = re.compile(
    r"^(?:"
    # PT — "(o) que é isso/isto/essa imagem/essa tela"
    r"(?:o\s+)?que\s+(?:é|eh)\s+"
    r"(?:isso|isto|essa\s+imagem|esta\s+imagem|essa\s+tela|esta\s+tela)|"
    # PT — "explica/explique isso/isto/essa imagem"
    r"explica(?:r)?\s+(?:isso|isto|essa\s+imagem)|explique\s+(?:isso|isto)|"
    # EN — "what is this / what's this", "explain this"
    r"what(?:'s|\s+is)\s+this(?:\s+(?:image|screen))?|"
    r"explain\s+this"
    r")[\s.!?]*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Active-context: voice toggle of the screen-reading kill-switch (inovação #3)
# ----------------------------------------------------------------------

# "para de ver minha tela" — turn the screen-reading kill-switch OFF by voice,
# flipping the SAME flag file (`/tmp/cortex_screen_off`) that scripts/
# cortex-eyes.sh and the hotkey toggle. The voice path is the hands-free button
# The user asked about ("cadê esse botão?") — it must work even when audio is the
# only channel. Full-match anchored so a conversational "para de ver minha tela
# por um instante e..." doesn't hijack. The handler itself does NOT gate on the
# kill-switch (it CONTROLS it), so "off" is always honoured.
MAC_EYES_OFF_RE = re.compile(
    r"^(?:"
    # PT — "para/pare/parar de ver/olhar (a/minha) tela"; "não veja/olhe..."
    r"par[ae](?:r)?\s+de\s+(?:ver|olhar)\s+(?:(?:a|minha|pra)\s+)?tela|"
    r"n[ãa]o\s+(?:veja|olhe|olha|veja\s+mais)\s+(?:a\s+|minha\s+)?tela|"
    r"deixa\s+de\s+ver\s+(?:a\s+|minha\s+)?tela|"
    # EN — "stop watching/looking at my screen", "don't watch my screen"
    r"stop\s+(?:watching|looking\s+at|reading)\s+(?:my\s+|the\s+)?screen|"
    r"don'?t\s+(?:watch|look\s+at|read)\s+(?:my\s+|the\s+)?screen"
    r")[\s.!?]*$",
    re.IGNORECASE,
)


# "volta a ver minha tela" — turn the kill-switch back ON by voice (removes the
# flag). The ONLY hands-free way back, so it must NOT gate on the kill-switch.
# Disjoint from MAC_EYES_OFF (no "para/stop") and from the read commands.
# The verb covers `volt[a|e|ar]` (the user says the imperative "VOLTE a ver" out
# loud — s76 cont.31 live smoke: the first cut only had volta/voltar and the
# command silently fell through to chat) and an optional "pode (voltar)" prefix;
# the "a" before ver/olhar is optional so "volta ver" / "pode ver" also land.
MAC_EYES_ON_RE = re.compile(
    r"^(?:"
    # PT — "volta/volte/voltar/pode voltar (a) ver/olhar (a/minha) tela (de novo)"
    # "pode" REQUIRES the volt verb after it — bare "pode ver minha tela" is
    # ambiguous (it collides with STUDIER_SHOW) and isn't the natural command.
    r"(?:volt(?:a|e|ar)|pode\s+volt(?:a|e|ar))\s+"
    r"a?\s*(?:ver|olhar)\s+"
    r"(?:(?:a|minha)\s+)?tela(?:\s+(?:de\s+novo|novamente))?|"
    # EN — "start/resume watching my screen (again)", "you can watch ... again"
    r"(?:start|resume)\s+(?:watching|looking\s+at|reading)\s+"
    r"(?:my\s+|the\s+)?screen(?:\s+again)?|"
    r"you\s+can\s+(?:watch|see|read)\s+(?:my\s+|the\s+)?screen(?:\s+again)?"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
