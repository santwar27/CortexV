"""Regex coverage for src/core/dispatch/macos.py.

Verifies that each verb family matches its intended phrasings AND
that bare "volume 50" belongs to MAC_VOLUME_SET_RE, while Spotify
volume requires Spotify to be named explicitly.

The orchestrator integration (handlers, audit) is exercised via the
end-to-end smoke test, not here — these tests stay at the regex
level so they run in milliseconds and fail with a precise pointer
when a phrasing breaks.
"""
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from src.core.dispatch.macos import (
    MAC_APP_LIST_RE,
    MAC_APP_SWITCH_RE,
    MAC_BRIGHTNESS_DOWN_RE,
    MAC_CLOSE_WINDOW_RE,
    MAC_BRIGHTNESS_SET_RE,
    MAC_BRIGHTNESS_UP_RE,
    MAC_DARK_OFF_RE,
    MAC_DARK_ON_RE,
    MAC_DARK_TOGGLE_RE,
    MAC_LOCK_RE,
    MAC_MUTE_RE,
    MAC_OPEN_APP_RE,
    MAC_QUIT_APP_RE,
    MAC_SCREENSHOT_RE,
    MAC_SLEEP_RE,
    MAC_UNMUTE_RE,
    MAC_VOLUME_DOWN_RE,
    MAC_VOLUME_SET_RE,
    MAC_VOLUME_UP_RE,
)
from src.core.dispatch.spotify import SPOTIFY_VOLUME_RE


# =====================================================================
# Mac volume is the default
# =====================================================================

class TestSpotifyVsMacVolumeRouting:
    """Unqualified volume controls the Mac, not the Spotify account."""

    @pytest.mark.parametrize("text", [
        "volume 50",
        "volume 80",
        "volume to 60",
        "vol 30",
        "volume em 70",
        "volume 50%",
    ])
    def test_bare_volume_n_routes_to_mac(self, text):
        assert MAC_VOLUME_SET_RE.match(text), f"MAC should match: {text}"
        assert not SPOTIFY_VOLUME_RE.match(text), f"Spotify should NOT match: {text}"

    @pytest.mark.parametrize("text, expected_level", [
        ("volume sistema 50", "50"),
        ("volume do sistema 80", "80"),
        ("system volume 30", "30"),
        ("system volume to 60", "60"),
        ("volume sistema 100%", "100"),
    ])
    def test_qualified_volume_routes_to_mac(self, text, expected_level):
        m = MAC_VOLUME_SET_RE.match(text)
        assert m, f"MAC_VOLUME_SET should match: {text}"
        assert m.group(1) == expected_level
        # Spotify must NOT also match these (would create ambiguity).
        assert not SPOTIFY_VOLUME_RE.match(text), \
            f"Spotify must not match qualified: {text}"


# =====================================================================
# Volume up / down / mute / unmute
# =====================================================================

class TestVolumeRelative:

    @pytest.mark.parametrize("text", [
        "aumenta o volume",
        "aumenta volume",
        "aumentar o volume",
        "sobe o volume",
        "sobe volume",
        "volume up",
        "volume mais",
        "volume mais alto",
    ])
    def test_volume_up_matches(self, text):
        assert MAC_VOLUME_UP_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "abaixa o volume",
        "abaixar volume",
        "diminui o volume",
        "baixa o volume",
        "desce o volume",
        "volume down",
        "volume menos",
        "volume mais baixo",
    ])
    def test_volume_down_matches(self, text):
        assert MAC_VOLUME_DOWN_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "mudo",
        "silencia",
        "silenciar",
        "silencia o sistema",
        "silencia tudo",
        "mute",
        "mute system",
    ])
    def test_mute_matches(self, text):
        assert MAC_MUTE_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "tira o mudo",
        "tirar o mudo",
        "desativa o mudo",
        "desativar o mudo",
        "desmuta",
        "desmutar",
        "unmute",
    ])
    def test_unmute_matches(self, text):
        assert MAC_UNMUTE_RE.match(text), text


# =====================================================================
# Brightness
# =====================================================================

_DOWN_VERBS = (
    "abaixa abaixe abaixar baixa baixe baixar diminui diminua diminuir "
    "reduz reduza reduzir desce desça desca descer escurece escureça escureca"
).split()
_UP_VERBS = "aumenta aumente aumentar sobe suba subir clareia clareie".split()
_LOCATIONS = ("", " da tela", " do mac", " no mac", " do monitor")
_RELATIVE_CASES = [
    (f"{verb}{article} brilho{location}{suffix}", method, None)
    for verbs, method in ((_DOWN_VERBS, "brightness_down"), (_UP_VERBS, "brightness_up"))
    for verb, article, location, suffix in product(verbs, ("", " o"), _LOCATIONS, ("", " um pouco"))
] + [
    (text, "brightness_down", None) for text in (
        "brilho menos", "brilho para baixo", "brightness down", "dimmer", "dim the screen",
        *[f"{verb} {article}brightness" for verb, article in product(
            ("lower", "decrease", "dim", "turn down"), ("", "the "))],
    )
] + [
    (text, "brightness_up", None) for text in (
        "brilho mais", "brilho para cima", "brightness up", "brighter",
        *[f"{verb} {article}brightness" for verb, article in product(
            ("raise", "increase", "turn up"), ("", "the "))],
    )
] + [("Diminua brilho no mac", "brightness_down", None)]
_SET_CASES = [
    (f"{verb}{article}brilho{location}{prep}{level}{percent}", "set_brightness", level)
    for verb in ("", *[v + " " for v in _DOWN_VERBS + _UP_VERBS +
                         ["põe", "poe", "coloca", "deixa", "muda", "ajusta", "set", "put"]])
    for article, location, prep, level, percent in (
        ("", " no mac", " para ", 0, ""),
        ("o ", " da tela", " em ", 30, "%"),
        ("", " do monitor", " a ", 100, ""),
        ("o ", " do mac", " no ", 150, "%"),
        ("", "", " to ", 40, ""),
        ("o ", "", " at ", 50, "%"),
        ("", "", " ", 60, ""),
    )
] + [
    (f"brilho no {word}", "set_brightness", level)
    for word, level in (("mínimo", 0), ("minimo", 0), ("máximo", 100), ("maximo", 100))
] + [
    (f"brightness {prep}{word}", "set_brightness", level)
    for prep in ("", "to ")
    for word, level in (("min", 0), ("minimum", 0), ("max", 100), ("maximum", 100))
] + [("Diminua brilho no mac para 0", "set_brightness", 0)]


@pytest.mark.parametrize("text, method, level", _RELATIVE_CASES + _SET_CASES)
def test_brightness_contract_routes(text, method, level):
    from src.core.dispatch.registry import run

    macos = MagicMock()
    getattr(macos, method).return_value = "read-back reply"
    orch = SimpleNamespace(_reply_lang="pt", macos=macos)
    assert run(orch, text) == ("read-back reply", "MAC_BRIGHTNESS")
    args = () if level is None else (level,)
    assert macos.method_calls == [getattr(call, method)(*args, lang="pt")]


@pytest.mark.parametrize("text", [
    "o brilho do sol hoje", "qual o brilho ideal para ler?", "diminua brilho no mac para 0 e abra notas",
    "por que aumentar o brilho", "aumente o brilho amanhã", "brilho 30 ontem", "brightness minimum today",
    "brilho -1", "brilho 30.5", "dim the screen and play music",
])
def test_brightness_non_commands(text):
    assert not any(rx.match(text) for rx in (
        MAC_BRIGHTNESS_UP_RE, MAC_BRIGHTNESS_DOWN_RE, MAC_BRIGHTNESS_SET_RE,
    ))


class TestBrightness:

    @pytest.mark.parametrize("text", [
        "aumenta o brilho",
        "aumentar brilho",
        "sobe o brilho",
        "sobe brilho",
        "brilho mais",
        "brilho para cima",
        "brightness up",
        "brighter",
    ])
    def test_brightness_up_matches(self, text):
        assert MAC_BRIGHTNESS_UP_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "abaixa o brilho",
        "diminui o brilho",
        "baixa o brilho",
        "desce o brilho",
        "brilho menos",
        "brilho para baixo",
        "brightness down",
        "dimmer",
    ])
    def test_brightness_down_matches(self, text):
        assert MAC_BRIGHTNESS_DOWN_RE.match(text), text

    @pytest.mark.parametrize("text, expected", [
        ("brilho 50", "50"),
        ("brilho da tela 70", "70"),
        ("brilho para 30", "30"),
        ("brilho em 80", "80"),
        ("brightness 40", "40"),
        ("brightness to 60", "60"),
        ("brightness at 90", "90"),
    ])
    def test_brightness_set_captures(self, text, expected):
        m = MAC_BRIGHTNESS_SET_RE.match(text)
        assert m, text
        assert m.group(1) == expected


# =====================================================================
# Dark mode
# =====================================================================

class TestDarkMode:

    @pytest.mark.parametrize("text", [
        "alterna o tema",
        "alterna o modo",
        "alternar tema",
        "troca o tema",
        "trocar o modo",
        "inverte o tema",
        "toggle dark mode",
        "toggle theme",
    ])
    def test_toggle_matches(self, text):
        assert MAC_DARK_TOGGLE_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "modo escuro",
        "tema escuro",
        "ativa modo escuro",
        "ativar tema escuro",
        "liga o modo escuro",
        "modo noturno",
        "modo dark",
        "dark mode",
        "dark mode on",
        "dark mode please",
        "enable dark mode",
    ])
    def test_dark_on_matches(self, text):
        assert MAC_DARK_ON_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "modo claro",
        "tema claro",
        "ativa modo claro",
        "liga o modo claro",
        "modo diurno",
        "modo light",
        "light mode",
        "light mode on",
        "disable dark mode",
    ])
    def test_dark_off_matches(self, text):
        assert MAC_DARK_OFF_RE.match(text), text


# =====================================================================
# Lock
# =====================================================================

class TestLock:

    @pytest.mark.parametrize("text", [
        "trava",
        "travar",
        "trava o mac",
        "trava a tela",
        "trava o computador",
        "bloqueia",
        "bloqueia a tela",
        "bloqueia o mac",
        "lock",
        "lock screen",
        "lock mac",
        "lock the screen",
        # STT/accent aliases: "Mac" → "meck"/"meque"/"mec" (BR accent)
        "trava o meck",
        "trava o meque",
        "trava o mec",
        "bloqueia o meck",
        "lock meck",
        # "desliga a tela" = lock (screen sleeps, password at once). Before
        # 2026-09-19 it staged quitting an app called "tela".
        "desligar tela",
        "desliga a tela",
        "desligue a tela",
        "apaga a tela",
        "desliga o monitor",
        "desliga o display",
        "desligar tela do mac",
        "desliga a tela do computador",
        "turn off the screen",
        "switch off the display",
        "screen off",
        # Whisper leaves the period on; the bare `$` used to refuse it.
        "trava a tela.",
        "Desliga a tela.",
        "lock screen!",
    ])
    def test_lock_matches(self, text):
        assert MAC_LOCK_RE.match(text), text


# =====================================================================
# System sleep (S3) — incl. STT/accent aliases for "Mac"
# =====================================================================

class TestSleep:

    @pytest.mark.parametrize("text", [
        "dorme o mac",
        "dorme o macbook",
        "põe o mac pra dormir",
        "coloca o mac para dormir",
        "suspende o mac",
        "sleep the mac",
        "sleep mac",
        "put the mac to sleep",
        # STT/accent aliases: "Mac" → "meck"/"meque"/"mec" (BR accent)
        "dorme o meck",
        "dorme o meque",
        "dorme o mec",
        "põe o meck pra dormir",
        "suspende o mec",
        "sleep the meck",
        "put the meque to sleep",
        # s83 — the phrasings the user found missing (2026-09-19)
        "hiberna o mac",
        "hibernar o macbook",
        "põe o mac em repouso",
        "coloca o computador em repouso",
        "modo repouso",
        "repouso",
        "mac dorme",
        "mac sleep",
        "o mac dorme",
        "go to sleep",
        "sleep now",
        # "desliga o mac" = sleep: Cortex never shuts the Mac down, which
        # over Blink would be irreversible from afar.
        "desliga o mac",
        "desligar o mac",
        "desliga o computador",
        "desligue o notebook",
        "Desliga o mac.",
        # STT drops the final "r" of the infinitive (BR accent: dormir→dormi)
        "põe o mac pra dormi",
        "põe o mac pra dormí",
        "coloca o mac para dormi",
        "manda o mac dormi",
        "dormi o mac",
        "dormí o macbook",
    ])
    def test_sleep_matches(self, text):
        assert MAC_SLEEP_RE.match(text), text


# =====================================================================
# Screenshot
# =====================================================================

class TestScreenshot:

    @pytest.mark.parametrize("text", [
        "tira print",
        "tira um print",
        "tirar print",
        "tira print da tela",
        "tira print de tela",
        "captura a tela",
        "capturar tela",
        "print da tela",
        "print de tela",
        "screenshot",
        "screenshot!",          # Whisper leaves the punctuation on
        "tira print.",
        "take a screenshot",
        "take screenshot",
        "capture screen",
        "capture the screen",
    ])
    def test_screenshot_matches(self, text):
        assert MAC_SCREENSHOT_RE.match(text), text


# =====================================================================
# App open
# =====================================================================

class TestOpenApp:

    @pytest.mark.parametrize("text, expected_app", [
        ("abre Notion", "Notion"),
        ("abrir Notion", "Notion"),
        ("abra o app Notion", "Notion"),
        ("lança Notion", "Notion"),
        ("lançar Notion", "Notion"),
        ("inicia Notion", "Notion"),
        ("open Notion", "Notion"),
        ("launch Notion", "Notion"),
        ("start Notion", "Notion"),
        ("abre Spotify", "Spotify"),
        ("open the app Calendar", "Calendar"),
        ("abre a aplicação Mail", "Mail"),
        ("abre VS Code", "VS Code"),
        ("abre Notion.", "Notion"),
        ("abre Notion!", "Notion"),
        # PT article must be stripped (matches the quit regex). Live bug
        # 2026-06-03: "abre o terminal" tried to open an app named
        # "o terminal" and failed.
        ("abre o terminal", "terminal"),
        ("abre o Claude", "Claude"),
        ("abre a Calculadora", "Calculadora"),
        ("abre o Obsidian", "Obsidian"),
    ])
    def test_open_app_captures(self, text, expected_app):
        m = MAC_OPEN_APP_RE.match(text)
        assert m, text
        assert m.group(1).strip() == expected_app


class TestQuitApp:
    """Quit = fully terminate (Cmd+Q), S3-gated. Pedro reserves quit for
    'quit/exit/end' (EN) and 'encerra/sai de/desligar' (PT). 'fecha'/'close'
    are NOT quit anymore — they close the window (see TestCloseWindow)."""

    @pytest.mark.parametrize("text, expected_app", [
        ("encerra Mail", "Mail"),
        ("encerrar o Mail", "Mail"),
        ("sai do Notion", "Notion"),
        ("sair do Spotify", "Spotify"),
        ("desliga o Notion", "Notion"),
        ("desligar o Spotify", "Spotify"),
        ("quit Notion", "Notion"),
        ("exit Mail", "Mail"),
        ("quit the app Calendar", "Calendar"),
        ("end the Notion", "Notion"),
        ("quit the Calendar", "Calendar"),
        ("encerra o Obsidian", "Obsidian"),
    ])
    def test_quit_app_captures(self, text, expected_app):
        m = MAC_QUIT_APP_RE.match(text)
        assert m, text
        assert m.group("app").strip() == expected_app

    def test_quit_does_not_match_close_or_open_verbs(self):
        assert not MAC_QUIT_APP_RE.match("abre Notion")
        assert not MAC_QUIT_APP_RE.match("fecha Notion")   # → close window
        assert not MAC_QUIT_APP_RE.match("close Spotify")  # → close window

    @pytest.mark.parametrize("text", [
        "desligar tela",
        "desliga a tela",
        "desliga o monitor",
        "desliga o display",
        "desligar tela do mac",
        "quit the screen",
        "Desliga a tela.",
    ])
    def test_screen_is_never_an_app(self, text):
        """The live bug (2026-09-19, over Blink): "desligar tela" asked
        "Fechar o tela?" and answered "Fechando tela." — it's a lock."""
        assert not MAC_QUIT_APP_RE.match(text), text

    def test_machine_is_never_an_app(self):
        """"desliga o mac" staged quitting an app called "mac"; it sleeps."""
        for text in ("desliga o mac", "desligar o computador", "desligue o notebook"):
            assert not MAC_QUIT_APP_RE.match(text), text
        # …but a real app whose name merely starts with it still quits.
        m = MAC_QUIT_APP_RE.match("quit MacUpdater")
        assert m and m.group("app").strip() == "MacUpdater"

    @pytest.mark.parametrize("text, expected_app", [
        ("quit Screen Sharing", "Screen Sharing"),
        ("desliga o Screen Sharing", "Screen Sharing"),
        ("desliga a tv", "tv"),   # "TV" is Apple's app, not the screen
    ])
    def test_screen_guard_spares_real_apps(self, text, expected_app):
        m = MAC_QUIT_APP_RE.match(text)
        assert m, text
        assert m.group("app").strip() == expected_app


class TestCloseWindow:
    """Close window = the red button / Cmd+W, app keeps running, NOT gated.
    'fecha/feche/fechar' (PT) + 'close' (EN). Added 2026-06-03 after Pedro
    pointed out 'fecha o X' should close the window, not quit."""

    @pytest.mark.parametrize("text, expected_app", [
        ("fecha Notion", "Notion"),
        ("fechar Notion", "Notion"),
        ("feche o Notion", "Notion"),
        ("fecha o app Notion", "Notion"),
        ("fecha o Spotify", "Spotify"),
        ("fecha VS Code", "VS Code"),
        ("fecha o Notion.", "Notion"),
        ("fecha a janela do Code", "Code"),
        ("close Spotify", "Spotify"),
        ("close the Spotify", "Spotify"),
        ("close the window of Notion", "Notion"),
    ])
    def test_close_window_captures(self, text, expected_app):
        m = MAC_CLOSE_WINDOW_RE.match(text)
        assert m, text
        assert m.group("app").strip() == expected_app

    def test_close_disjoint_from_quit_and_open(self):
        assert not MAC_CLOSE_WINDOW_RE.match("encerra Notion")
        assert not MAC_CLOSE_WINDOW_RE.match("quit Notion")
        assert not MAC_CLOSE_WINDOW_RE.match("abre Notion")


# =====================================================================
# Negatives — false positives we explicitly want to avoid
# =====================================================================

class TestFalsePositives:
    """Phrases that look adjacent to a mac control verb but should
    NOT match — either belong to another route (Spotify, LIST_EMAILS)
    or are everyday non-command speech."""

    @pytest.mark.parametrize("re_const, text", [
        # Spotify must be named explicitly; Mac patterns cannot claim it.
        (MAC_VOLUME_SET_RE, "spotify volume 50"),
        (MAC_VOLUME_SET_RE, "volume do spotify 80"),
        # mute / unmute can't match non-commands
        (MAC_MUTE_RE, "ele ficou mudo de raiva"),
        # lock — long sentence shouldn't match the anchored single word
        (MAC_LOCK_RE, "ele vai travar a minha bike na rua"),
        # "mec" alias must not swallow longer words — anchoring protects us
        (MAC_LOCK_RE, "trava o mecânico"),
        # screen words only lock as a bare command, not in a sentence
        (MAC_LOCK_RE, "apaga a tela do meu celular que tá rachada"),
        (MAC_LOCK_RE, "o monitor desligou sozinho"),
        (MAC_SLEEP_RE, "dorme o mecânico"),
        (MAC_SLEEP_RE, "hoje vou dormir cedo"),
        # dropped-r "dormi" at sentence start still needs a device qualifier
        (MAC_SLEEP_RE, "dormi bem ontem"),
        (MAC_SLEEP_RE, "o mac dormiu sozinho ontem"),
        (MAC_SLEEP_RE, "vou de repouso médico amanhã"),
        # dark mode mid-sentence
        (MAC_DARK_ON_RE, "explica o que é o modo escuro"),
        # screenshot mid-sentence
        (MAC_SCREENSHOT_RE, "manda o print da tela pro João"),
    ])
    def test_non_command_does_not_match(self, re_const, text):
        assert not re_const.match(text), f"false positive: {text}"


class TestAppList:
    @pytest.mark.parametrize("text", [
        "quais apps estão abertos",
        "quais apps tão abertos",
        "quais são os apps abertos",
        "que apps estão abertos",
        "lista os apps",
        "lista os apps abertos",
        "mostra os apps abertos",
        "apps abertos",
        "what apps are open",
        "what apps are running",
        "list open apps",
        "show running apps",
        "which apps are running",
    ])
    def test_matches(self, text):
        assert MAC_APP_LIST_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "abre o app Notion",          # that's open, not list
        "quantos apps eu tenho instalados",
        "lista os estudos",           # studier, not apps
    ])
    def test_no_false_positive(self, text):
        assert not MAC_APP_LIST_RE.match(text), text


class TestAppSwitch:
    @pytest.mark.parametrize("text, app", [
        ("vai pro Notion", "Notion"),
        ("vai pra Spotify", "Spotify"),
        ("vai para o Notion", "Notion"),
        ("foca no Notion", "Notion"),
        ("foca na Calculadora", "Calculadora"),
        ("foca o Spotify", "Spotify"),
        ("focar no Mail", "Mail"),
        ("switch to Notion", "Notion"),
        ("focus on Spotify", "Spotify"),
        ("go to Calendar", "Calendar"),
        ("vai pro VS Code", "VS Code"),
        ("foca no Notion.", "Notion"),
    ])
    def test_matches_and_captures(self, text, app):
        m = MAC_APP_SWITCH_RE.match(text)
        assert m, text
        assert m.group("app").strip() == app

    def test_switch_verbs_disjoint_from_open_and_quit(self):
        assert not MAC_APP_SWITCH_RE.match("abre Notion")
        assert not MAC_APP_SWITCH_RE.match("fecha Notion")
        assert not MAC_OPEN_APP_RE.match("vai pro Notion")
        assert not MAC_QUIT_APP_RE.match("foca no Notion")

    @pytest.mark.parametrize("text", [
        "vai dormir",                 # not "vai pro X"
        "foca",                       # no app
    ])
    def test_no_false_positive(self, text):
        assert not MAC_APP_SWITCH_RE.match(text), text
