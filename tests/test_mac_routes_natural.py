"""Natural Mac commands must reach local controls, never music or a model."""
import os
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from src.core.dispatch import macos
from src.core.dispatch.normalisation import strip_courtesy
from src.core.dispatch.registry import run
from src.core.dispatch.spotify import SPOTIFY_DEVICE_HINT_RE, SPOTIFY_PLAY_RE, SPOTIFY_VOLUME_RE
from src.tools.macos_controller import MacOSController


_DARK = [
    (f"{verb} {device}{prep}{kind} {shade}", method, None)
    for verb, device, prep, kind in product(
        ("coloca", "põe", "poe", "deixa", "muda", "troca", "ativa", "liga"),
        ("", "o mac ", "a tela ", "o computador "),
        ("", "o ", "no ", "na ", "em ", "pro ", "pra ", "para o ", "para "),
        ("modo", "tema"),
    )
    for shade, method in (("escuro", "enable_dark_mode"), ("noturno", "enable_dark_mode"),
                          ("dark", "enable_dark_mode"), ("claro", "disable_dark_mode"),
                          ("light", "disable_dark_mode"))
]
# Exercise the full grammar above within one test per verb, keeping fixture cost low.
_DARK_SIMPLE = [
    (text, method, None) for text, method in (
        ("modo escuro", "enable_dark_mode"), ("modo claro", "disable_dark_mode"),
        ("desativa o modo escuro", "disable_dark_mode"),
        ("desliga o modo escuro", "disable_dark_mode"),
        ("tira o modo escuro", "disable_dark_mode"),
        ("switch to dark mode", "enable_dark_mode"),
        ("turn on dark mode", "enable_dark_mode"), ("enable dark mode", "enable_dark_mode"),
        ("switch to light mode", "disable_dark_mode"),
        ("turn on light mode", "disable_dark_mode"), ("turn off dark mode", "disable_dark_mode"),
        ("alterna o modo", "toggle_dark_mode"), ("troca o tema", "toggle_dark_mode"),
        ("toggle dark mode", "toggle_dark_mode"),
        ("coloca o mac no modo escuro", "enable_dark_mode"),
        ("põe a tela no modo noturno", "enable_dark_mode"),
        ("muda pro modo claro", "disable_dark_mode"),
    )
]
_VOLUME_SET = [
    (f"{verb}{article}volume{location}{prep}30{percent}", "set_system_volume", 30)
    for verb in ("", "põe ", "poe ", "coloca ", "deixa ", "muda ", "ajusta ",
                 "abaixa ", "aumenta ", "set ", "put ")
    for article, location, prep, percent in (
        ("o ", " do mac", " para ", "%"), ("", " do sistema", " em ", ""),
        ("o ", " do computador", " no ", ""), ("", "", " a ", ""),
        ("", "", " to ", ""), ("", "", " at ", "%"), ("o ", "", " ", ""),
    )
] + [(text, "set_system_volume", 30) for text in (
    "volume 30", "vol 30", "coloca o volume no 30", "system volume 30", "volume sistema 30",
)] + [
    (f"{verb}volume {limit}", "set_system_volume", level)
    for verb, (limit, level) in product(("", "coloca o ", "deixa o ", "put "), (
        ("no máximo", 100), ("no maximo", 100), ("no mínimo", 0), ("no minimo", 0),
        ("max", 100), ("to max", 100), ("min", 0), ("to min", 0),
    ))
]
_DOWN = "abaixa abaixe abaixar baixa baixe baixar diminui diminua diminuir reduz reduza reduzir desce desça desca descer escurece escureça escureca".split()
_UP = "aumenta aumente aumentar sobe suba subir clareia clareie".split()
_VOLUME_RELATIVE = [
    (f"{verb}{article} {noun}{location}{tail}", method, None)
    for verbs, method in ((_DOWN, "volume_down"), (_UP, "volume_up"))
    for verb in verbs
    for article, noun, location, tail in (
        (" o", "volume", "", ""), ("", "som", " do mac", " um pouco"),
        (" o", "som", " do sistema", " um pouquinho"),
        ("", "volume", " do computador", " um pouco"),
    )
] + [
    (f"deixa o volume mais {word}", method, None)
    for word, method in (("baixo", "volume_down"), ("baixinho", "volume_down"),
                         ("alto", "volume_up"), ("altinho", "volume_up"))
] + [(text, method, None) for text, method in (
    ("lower the volume", "volume_down"), ("lower volume", "volume_down"),
    ("turn down the volume", "volume_down"), ("turn down volume", "volume_down"),
    ("raise the volume", "volume_up"), ("raise volume", "volume_up"),
    ("turn up the volume", "volume_up"), ("turn up volume", "volume_up"),
    ("volume up", "volume_up"), ("volume down", "volume_down"),
    ("aumenta um pouco o volume", "volume_up"), ("abaixa um pouco do volume", "volume_down"),
)]
_BRIGHTNESS = [
    (f"{verb} o brilho{location} no {word}", "set_brightness", level)
    for verb, location, (word, level) in product(
        ("põe", "poe", "coloca", "deixa"), ("", " da tela", " do mac"),
        (("máximo", 100), ("maximo", 100), ("mínimo", 0), ("minimo", 0)),
    )
]


def assert_mac_route(text, method, level):
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock(), spotify=MagicMock(),
                           _SPOTIFY_DEVICE_HINT_RE=SPOTIFY_DEVICE_HINT_RE)
    getattr(orch.macos, method).return_value = "verified reply"
    result = run(orch, strip_courtesy(text))
    assert result is not None and result[0] == "verified reply", text
    args = () if level is None else (level,)
    assert orch.macos.method_calls == [getattr(call, method)(*args, lang="pt")], text
    assert not orch.spotify.mock_calls, text


@pytest.mark.parametrize("verb", ("coloca", "põe", "poe", "deixa", "muda", "troca", "ativa", "liga"))
def test_dark_grammar(verb):
    for text, method, level in _DARK:
        if text.startswith(verb + " "):
            assert_mac_route(text, method, level)


@pytest.mark.parametrize("suffix", ("", " por favor", " pfv", " please"))
@pytest.mark.parametrize("text,method,level", _DARK_SIMPLE + _VOLUME_SET + _VOLUME_RELATIVE + _BRIGHTNESS)
def test_natural_routes(text, method, level, suffix):
    assert_mac_route(text + suffix, method, level)


@pytest.mark.parametrize("verb", ("coloca", "põe", "poe", "toca", "play", "put on"))
@pytest.mark.parametrize("obj", (
    "o mac no modo escuro", "a tela no modo noturno", "o computador para o tema light",
    "modo escuro", "tema claro", "o brilho no máximo", "o volume no 30", "o som no mínimo",
    "a luz da tela no máximo", "o mac pra dormir", "o computador para descansar", "a tela pra travar",
))
def test_music_excludes_control_objects(verb, obj):
    assert not SPOTIFY_PLAY_RE.match(f"{verb} {obj}")


@pytest.mark.parametrize("text", (
    "poderia colocar o mac no modo escuro", "pode tocar o volume no 30",
    "could you play o mac no modo escuro", "coloca modo escuro blues",
    "coloca o volume da saudade",
))
def test_control_prefix_is_reserved_even_with_a_longer_title(text):
    assert not SPOTIFY_PLAY_RE.match(text)


@pytest.mark.parametrize("text", (
    "coloca matuê", "põe um som pra tocar", "coloca a música do modo avião",
    "coloca a música modo escuro", "coloca modo avião", "play dark horse",
    "coloca uma canção sobre o volume", "coloca a música o volume da saudade",
))
def test_music_still_plays(text):
    assert SPOTIFY_PLAY_RE.match(text)
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock(), spotify=MagicMock(),
                           _SPOTIFY_DEVICE_HINT_RE=SPOTIFY_DEVICE_HINT_RE)
    assert run(orch, text)[1] == "SPOTIFY_PLAY"
    orch.spotify.play_anything.assert_called_once()
    assert not orch.macos.mock_calls


@pytest.mark.parametrize("suffix", ("", " por favor", " pfv", " please"))
@pytest.mark.parametrize("text", ("volume do spotify 30", "spotify volume 30", "volume spotify to 30%"))
def test_named_spotify_volume(text, suffix):
    orch = SimpleNamespace(_reply_lang="en", macos=MagicMock(), spotify=MagicMock())
    assert run(orch, strip_courtesy(text + suffix))[1] == "SPOTIFY_VOLUME"
    orch.spotify.set_volume.assert_called_once_with(30, lang="en")
    assert not orch.macos.mock_calls


@pytest.mark.parametrize("text", ("volume 30", "vol 30", "coloca o volume no 30", "volume do mac 30"))
def test_spotify_volume_requires_name(text):
    assert not SPOTIFY_VOLUME_RE.match(text)


@pytest.mark.parametrize("text", ("volume -1", "volume 30.5", "volume 30 amanhã", "deixa o volume mais baixo amanhã",
                                 "hoje muda pro modo claro", "modo escuro e toca matuê"))
def test_control_non_commands(text):
    assert not any(rx.match(text) for rx in (macos.MAC_VOLUME_SET_RE, macos.MAC_VOLUME_UP_RE,
        macos.MAC_VOLUME_DOWN_RE, macos.MAC_DARK_ON_RE, macos.MAC_DARK_OFF_RE))


@pytest.mark.parametrize("stream", (False, True), ids=("process", "process_stream"))
@pytest.mark.parametrize("text,method,args,reply,mode", (
    ("coloca o mac no modo escuro por favor", "enable_dark_mode", (), "Modo escuro ativado.", "MAC_DARK"),
    ("coloca o volume no 30", "set_system_volume", (30,), "Volume do Mac em 30%.", "MAC_VOLUME"),
))
def test_public_entry_points(text, method, args, reply, mode, stream, monkeypatch):
    from src.core.orchestrator import CortexOrchestrator

    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch.engine_analytical = MagicMock()
    orch.engine_creative = MagicMock()
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).side_effect = AssertionError("must bypass the 8B")
    orch.macos = MagicMock()
    orch.spotify = MagicMock()
    orch.spotify.play_anything.side_effect = AssertionError("must not play music")
    orch._SPOTIFY_DEVICE_HINT_RE = SPOTIFY_DEVICE_HINT_RE
    getattr(orch.macos, method).return_value = reply
    orch._pending_action = None
    orch._pending_query = None
    orch._shadow_classify = MagicMock()
    orch._maybe_announce_pending = lambda reply, mode: reply
    monkeypatch.setattr(orch, "_handle_user_paths", MagicMock(side_effect=AssertionError("registry missed")))
    if stream:
        assert list(orch.process_stream(text)) == [(reply, True, mode, "pt")]
    else:
        assert orch.process(text) == (reply, mode)
    assert orch.macos.method_calls == [getattr(call, method)(*args, lang="pt")]
    assert not orch.spotify.mock_calls
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).assert_not_called()


_FAILURE = {"pt": "Não consegui mudar o volume do Mac — nada mudou.",
            "en": "I couldn't change the Mac volume — nothing changed."}


@pytest.mark.parametrize("lang", ("pt", "en"))
@pytest.mark.parametrize("method,args,before,after,target", (
    ("set_system_volume", (30,), 70, 30, 30), ("set_system_volume", (30,), 70, 28, 30),
    ("set_system_volume", (30,), 70, 32, 30), ("set_system_volume", (500,), 70, 100, 100),
    ("set_system_volume", (-20,), 70, 0, 0), ("volume_up", (), 40, 50, 50),
    ("volume_down", (), 40, 30, 30), ("volume_up", (), 95, 100, 100),
    ("volume_down", (), 5, 0, 0),
))
def test_volume_verified_reply(lang, method, args, before, after, target):
    with patch("src.tools.macos_controller._osascript", side_effect=[(True, str(before)), (True, ""), (True, str(after))]) as osa:
        reply = getattr(MacOSController(), method)(*args, lang=lang)
    assert reply == (f"Volume do Mac em {after}%." if lang == "pt" else f"Mac volume at {after}%.")
    assert [c.args[0] for c in osa.call_args_list] == [
        "output volume of (get volume settings)", f"set volume output volume {target}",
        "output volume of (get volume settings)"]


@pytest.mark.parametrize("lang", ("pt", "en"))
@pytest.mark.parametrize("read", ((False, "error"), (True, "???"), (True, "-1"), (True, "101"), (True, "")))
@pytest.mark.parametrize("method,args", (("set_system_volume", (30,)), ("volume_up", ()), ("volume_down", ())))
def test_volume_failed_initial_read_does_not_write(lang, read, method, args):
    with patch("src.tools.macos_controller._osascript", return_value=read) as osa:
        assert getattr(MacOSController(), method)(*args, lang=lang) == _FAILURE[lang]
    assert osa.call_count == 1


@pytest.mark.parametrize("lang", ("pt", "en"))
@pytest.mark.parametrize("write_ok,after", ((True, "70"), (False, "70"), (True, "???"), (True, ""), (True, "40"), (False, "40")))
def test_volume_failed_or_wrong_readback(lang, write_ok, after):
    with patch("src.tools.macos_controller._osascript", side_effect=[(True, "70"), (write_ok, "error"), (True, after)]):
        reply = MacOSController().set_system_volume(30, lang=lang)
    if after == "40":
        assert reply == ("Não consegui deixar em 30% — o volume do Mac está em 40%." if lang == "pt"
                         else "I couldn't set it to 30% — Mac volume is at 40%.")
    else:
        assert reply == _FAILURE[lang]


@pytest.mark.parametrize("lang", ("pt", "en"))
def test_volume_failed_readback(lang):
    with patch("src.tools.macos_controller._osascript", side_effect=[(True, "70"), (True, ""), (False, "error")]):
        assert MacOSController().set_system_volume(30, lang=lang) == _FAILURE[lang]


@pytest.mark.parametrize("lang", ("pt", "en"))
@pytest.mark.parametrize("method,current,pt,en", (
    ("volume_down", 0, "O volume já está no mínimo (0%).", "Volume is already at the minimum (0%)."),
    ("volume_up", 100, "O volume já está no máximo (100%).", "Volume is already at the maximum (100%)."),
))
def test_volume_at_limit(lang, method, current, pt, en):
    with patch("src.tools.macos_controller._osascript", return_value=(True, str(current))) as osa:
        assert getattr(MacOSController(), method)(lang=lang) == (pt if lang == "pt" else en)
    assert osa.call_count == 1


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_MAC_TESTS") != "1", reason="opt-in real Mac volume test")
def test_real_mac_volume_restore():
    controller = MacOSController()
    before = controller._get_system_volume()
    assert before is not None
    try:
        controller.set_system_volume(20)
        actual = controller._get_system_volume()
        assert actual is not None and abs(actual - 20) <= 2
    finally:
        controller.set_system_volume(before)
        restored = controller._get_system_volume()
        assert restored is not None and abs(restored - before) <= 2
