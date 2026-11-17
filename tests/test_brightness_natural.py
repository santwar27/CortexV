"""Tasks 119/119b: screen grammar through real entry points, no OS or model calls."""
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from src.core.dispatch.registry import run, simulate


DOWN = (
    "lower the brightness", "decrease brightness", "dim the screen",
    "turn down the brightness", "lower the screen brightness", "lower the screen light",
    "turn down the screen brightness", "reduce the brightness", "reduce screen brightness",
    "make the screen darker", "dim the display", "lower the display brightness",
    "screen brightness down", "turn the brightness down", "lower the mac brightness",
    "lower the brightness of the screen", "lower screen light", "dim the screen a bit",
    "lower the brightness on the mac",
    "dim the display light", "lower the light of the screen",
    "lower the mac light", "lower the mac's light", "lower the light on the screen",
    "lower the light on the mac",
)
UP = (
    "increase screen brightness", "raise the screen brightness", "brighten the screen",
    "make the screen brighter", "screen brightness up", "turn up the screen brightness",
    "turn the brightness up",
)
SET = (
    ("brightness 30", 30), ("brightness to 30%", 30), ("screen brightness 30", 30),
    ("set screen brightness to 30", 30), ("set the brightness to 30", 30),
    ("max brightness", 100), ("screen brightness max", 100), ("minimum brightness", 0),
    ("screen light 30", 30), ("set light on the screen to 30", 30),
    ("max display light", 100), ("light of the screen minimum", 0),
)
CASES = (
    [(text, "brightness_down", None, "en") for text in DOWN]
    + [(text, "brightness_up", None, "en") for text in UP]
    + [(text, "set_brightness", level, "en") for text, level in SET]
    + [(text, method, level, "pt") for text, method, level in (
        ("abaixa o brilho da tela", "brightness_down", None),
        ("abaixa a luz da tela", "brightness_down", None),
        ("aumenta a luz da tela", "brightness_up", None),
        ("luz da tela 30", "set_brightness", 30),
        ("escurece a tela", "brightness_down", None),
        ("clareia a tela", "brightness_up", None),
    )]
)
WRAPPERS = ("{}", "please {}", "{} please", "can you {}?", "can you {} please?")


@pytest.fixture
def orch(monkeypatch):
    from src.core.orchestrator import CortexOrchestrator

    instance = CortexOrchestrator.__new__(CortexOrchestrator)
    instance.macos = MagicMock()
    instance.spotify = MagicMock()
    instance.engine_analytical = MagicMock()
    instance.engine_creative = MagicMock()
    instance.gatekeeper = MagicMock()
    instance.gatekeeper.evaluate_gate.return_value.mode = "TEST"
    for engine in (instance.engine_analytical, instance.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).side_effect = AssertionError("must bypass the 8B")
    instance._pending_action = instance._pending_query = None
    instance._shadow_classify = MagicMock()
    instance._maybe_announce_pending = lambda reply, mode: reply
    monkeypatch.setattr(instance, "_handle_user_paths", MagicMock(
        side_effect=AssertionError("registry missed")))
    return instance


def assert_public_route(orch, text, method, level, lang, stream, mode="MAC_BRIGHTNESS"):
    getattr(orch.macos, method).return_value = "verified reply"
    if stream:
        assert list(orch.process_stream(text)) == [("verified reply", True, mode, lang)]
    else:
        assert orch.process(text) == ("verified reply", mode)
    args = () if level is None else (level,)
    assert orch.macos.method_calls == [getattr(call, method)(*args, lang=lang)]
    assert not orch.spotify.mock_calls
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).assert_not_called()


@pytest.mark.parametrize("stream", (False, True), ids=("process", "process_stream"))
@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("text,method,level,lang", CASES)
def test_reported_phrases(orch, text, method, level, lang, wrapper, stream):
    assert_public_route(orch, wrapper.format(text), method, level, lang, stream)


def assert_grammar_route(orch, text, method, level=None):
    orch.macos.reset_mock()
    getattr(orch.macos, method).return_value = "verified reply"
    assert run(orch, text) == ("verified reply", "MAC_BRIGHTNESS"), text
    args = () if level is None else (level,)
    assert orch.macos.method_calls == [getattr(call, method)(*args, lang="pt")], text
    prediction = simulate(text)
    assert prediction is not None and prediction.mode == "MAC_BRIGHTNESS", text
    assert prediction.capture == (str(level) if level not in (None, 0, 100) else None), text


@pytest.mark.parametrize("verbs,direction,comparatives", (
    (("lower", "decrease", "reduce", "dim", "turn down"), "down", ("darker", "dimmer")),
    (("raise", "increase", "brighten", "turn up"), "up", ("brighter",)),
))
def test_english_relative_grammar(verbs, direction, comparatives):
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock())
    method = "brightness_" + direction
    objects = [
        f"{article}{device}{noun}{location}"
        for article, device, noun, location in product(
            ("", "the "), ("", "screen ", "display ", "mac ", "mac's "),
            ("brightness", "light"), ("", " of the screen", " on the mac", " on the screen"))
        if noun == "brightness" or device or location
    ] + ["the screen", "the display"]
    for obj, tail in product(objects, ("", " a bit", " a little")):
        for verb in verbs:
            assert_grammar_route(orch, f"{verb} {obj}{tail}", method)
        assert_grammar_route(orch, f"turn {obj} {direction}{tail}", method)
    for device, comparative, tail in product(("screen", "display"), comparatives, ("", " a bit", " a little")):
        assert_grammar_route(orch, f"make the {device} {comparative}{tail}", method)
    assert_grammar_route(orch, f"screen brightness {direction}", method)


def test_english_absolute_grammar():
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock())
    for prefix, article, device, prep, percent in product(
        ("", "set "), ("", "the ", "o "), ("", "screen "), ("", "to "), ("", "%")):
        assert_grammar_route(orch, f"{prefix}{article}{device}brightness {prep}30{percent}", "set_brightness", 30)
    for word, level in (("min", 0), ("minimum", 0), ("max", 100), ("maximum", 100)):
        assert_grammar_route(orch, f"{word} brightness", "set_brightness", level)
        for obj, prep in product(("brightness", "screen brightness"), ("", "to ")):
            assert_grammar_route(orch, f"{obj} {prep}{word}", "set_brightness", level)


def test_portuguese_light_grammar():
    orch = SimpleNamespace(_reply_lang="pt", macos=MagicMock())
    for location, article in product(("da tela", "do mac", "no mac", "do monitor"), ("", "a ")):
        obj = f"{article}luz {location}"
        for verb, method in (("abaixa", "brightness_down"), ("aumenta", "brightness_up")):
            for tail in ("", " um pouco"):
                assert_grammar_route(orch, f"{verb} {obj}{tail}", method)
        for prefix, prep in product(("", "põe ", "poe ", "coloca ", "deixa "), ("", "para ", "em ")):
            assert_grammar_route(orch, f"{prefix}{obj} {prep}30%", "set_brightness", 30)
        for word, level in (("mínimo", 0), ("minimo", 0), ("máximo", 100), ("maximo", 100)):
            assert_grammar_route(orch, f"{obj} no {word}", "set_brightness", level)
    for verb, method in (("escurece", "brightness_down"), ("escureça", "brightness_down"),
                         ("escureca", "brightness_down"), ("clareia", "brightness_up"), ("clareie", "brightness_up")):
        assert_grammar_route(orch, f"{verb} a tela", method)


@pytest.mark.parametrize("text", (
    "make the screen dark", "deixa a tela escura", "dim the lights", "turn down the lights",
    "abaixa a luz", "luz 30", "max keyboard brightness", "screen brightness keyboard down",
    "lower screen brightness do teclado", "lower the brightness of the keyboard",
    "screen brightness -1", "screen brightness 30.5", "dim the display and play music",
    "lower the screen brightness tomorrow", "why reduce screen brightness",
))
def test_never_cross_or_accept_prose(orch, text, monkeypatch):
    # Stop at the existing registry fall-through; no planner, model or OS is run.
    monkeypatch.setattr(orch, "_handle_user_paths", lambda text: ("fall-through", "TEST_FALLTHROUGH"))
    assert orch.process(text) == ("fall-through", "TEST_FALLTHROUGH")
    assert not orch.macos.mock_calls
    assert not orch.spotify.mock_calls


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("text", (
    "lower the light", "turn down the light", "dim the light", "turn the light down",
    "light 30", "the light 50", "set light to 30", "max light",
    "lower light", "turn down light", "dim light", "turn light down",
    "set the light to 30", "max the light", "raise the light", "light up",
    "minimum light", "the light minimum", "light max",
))
def test_bare_english_light_is_not_a_screen(orch, text, wrapper, monkeypatch):
    text = wrapper.format(text)
    monkeypatch.setattr(orch, "_handle_user_paths", lambda text: ("fall-through", "TEST_FALLTHROUGH"))
    assert orch.process(text) == ("fall-through", "TEST_FALLTHROUGH")
    assert not orch.macos.mock_calls
    assert not orch.spotify.mock_calls
    assert simulate(text) is None


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,method,level,lang,mode", (
    ("abaixa a luz do teclado", "keyboard_brightness_down", None, "pt", "MAC_KEYBOARD_BRIGHTNESS"),
    ("lower the keyboard brightness", "keyboard_brightness_down", None, "en", "MAC_KEYBOARD_BRIGHTNESS"),
    ("keyboard brightness 30", "keyboard_brightness_set", 30, "en", "MAC_KEYBOARD_BRIGHTNESS"),
    ("lower the keyboard light", "keyboard_brightness_down", None, "en", "MAC_KEYBOARD_BRIGHTNESS"),
    ("keyboard light 30", "keyboard_brightness_set", 30, "en", "MAC_KEYBOARD_BRIGHTNESS"),
    ("modo escuro", "enable_dark_mode", None, "pt", "MAC_DARK"),
    ("tema escuro", "enable_dark_mode", None, "pt", "MAC_DARK"),
    ("dark mode", "enable_dark_mode", None, "en", "MAC_DARK"),
    ("coloca o mac no modo escuro", "enable_dark_mode", None, "pt", "MAC_DARK"),
    ("lower the volume", "volume_down", None, "en", "MAC_VOLUME"),
    ("volume 30", "set_system_volume", 30, "en", "MAC_VOLUME"),
))
def test_other_controls_keep_their_routes(orch, text, method, level, lang, mode, stream):
    assert_public_route(orch, text, method, level, lang, stream, mode)


@pytest.mark.parametrize("text", ("volume do spotify 30", "spotify volume 30", "play dark horse"))
def test_spotify_keeps_its_routes(orch, text):
    orch.spotify.set_volume.return_value = "music reply"
    orch.spotify.play_anything.return_value = "music reply"
    reply, mode = orch.process(text)
    assert reply == "music reply"
    assert mode == ("SPOTIFY_PLAY" if text.startswith("play") else "SPOTIFY_VOLUME")
    assert not orch.macos.mock_calls


@pytest.mark.parametrize("text,method,level,lang", CASES)
def test_voice_veto_and_simulator_share_screen_grammar(orch, text, method, level, lang):
    assert orch._is_pure_chat(text) is False
    assert any(getattr(orch, name).match(text) for name in orch._PURE_CHAT_BLOCKING_RES)
    prediction = simulate(text)
    assert prediction is not None and prediction.mode == "MAC_BRIGHTNESS"
