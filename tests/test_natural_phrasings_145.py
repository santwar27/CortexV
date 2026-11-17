"""Task 145: natural device commands, exact music queries, bounded ownership."""
from unittest.mock import call

import pytest

from evals.dispatch import simulate_dispatch
from src.core.dispatch import registry
from src.core.dispatch.normalisation import (
    normalise_stt_aliases,
    strip_courtesy,
    strip_trailing_sentence_punctuation,
    strip_vocatives,
)
from tests.test_taught_facts import orch


# text, integration, method, arguments, simulator mode, matcher, capture
NEW_MAC = [
    (f"deixa a tela mais {shade}", "macos", method, (), "MAC_BRIGHTNESS", matcher, None)
    for shades, method, matcher in (
        (("clara", "clarinha"), "brightness_up", "MAC_BRIGHTNESS_UP_RE"),
        (("escura", "escurinha"), "brightness_down", "MAC_BRIGHTNESS_DOWN_RE"),
    )
    for shade in shades
] + [
    (f"{verb} o teclado{location}", "macos", "keyboard_brightness_set", (level,),
     "MAC_KEYBOARD_BRIGHTNESS", "MAC_KEYBOARD_BRIGHTNESS_SET_RE", None)
    for verbs, level in ((("acende", "acenda", "acender"), 50),
                         (("apaga", "apague", "apagar"), 0))
    for verb in verbs
    for location in ("", " do mac")
] + [
    (f"deixa o teclado{location} mais {strength}", "macos", method, (),
     "MAC_KEYBOARD_BRIGHTNESS", matcher, None)
    for strengths, method, matcher in (
        (("forte", "fortinho"), "keyboard_brightness_up", "MAC_KEYBOARD_BRIGHTNESS_UP_RE"),
        (("fraco", "fraquinho"), "keyboard_brightness_down", "MAC_KEYBOARD_BRIGHTNESS_DOWN_RE"),
    )
    for strength in strengths
    for location in ("", " do mac")
] + [
    (f"{verb} o som{location}", "macos", "mute", (), "MAC_MUTE", "MAC_MUTE_RE", None)
    for verb in ("tira", "tire", "tirar")
    for location in ("", " do mac", " do sistema", " do computador")
]
NEW_SPOTIFY = [
    (f"{verb} {obj} {connector} {artist}{tail}", "spotify", "play_anything", (artist,),
     "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", artist)
    for verb in ("bota", "põe", "coloca")
    for obj in ("uma música", "música", "um som", "som")
    for connector, artist in (("do", "matuê"), ("da", "anitta"), ("de", "djonga"))
    for tail in ("", " pra tocar", " para tocar")
] + [
    ("bota uma música do matuê pra tocar no meu iphone", "spotify", "play_anything", ("matuê",),
     "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", "matuê"),
    ("bota música da Anitta no iphone", "spotify", "play_anything", ("Anitta",),
     "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", "Anitta"),
]
EXISTING = [
    ("abaixa um pouco o som", "macos", "volume_down", (), "MAC_VOLUME", "MAC_VOLUME_DOWN_RE", None),
    ("apaga a luz do teclado", "macos", "keyboard_brightness_set", (0,),
     "MAC_KEYBOARD_BRIGHTNESS", "MAC_KEYBOARD_BRIGHTNESS_SET_RE", None),
    ("escurece a tela", "macos", "brightness_down", (), "MAC_BRIGHTNESS", "MAC_BRIGHTNESS_DOWN_RE", None),
    ("coloca o mac no modo noturno", "macos", "enable_dark_mode", (), "MAC_DARK", "MAC_DARK_ON_RE", None),
    ("aumenta o som do mac", "macos", "volume_up", (), "MAC_VOLUME", "MAC_VOLUME_UP_RE", None),
    ("pula essa música", "spotify", "next_track", (), "SPOTIFY_NEXT", "SPOTIFY_NEXT_RE", None),
    ("põe o brilho no máximo", "macos", "set_brightness", (100,),
     "MAC_BRIGHTNESS", "MAC_BRIGHTNESS_SET_RE", None),
    ("make the screen brighter", "macos", "brightness_up", (), "MAC_BRIGHTNESS", "MAC_BRIGHTNESS_UP_RE", None),
    ("make the screen darker", "macos", "brightness_down", (), "MAC_BRIGHTNESS", "MAC_BRIGHTNESS_DOWN_RE", None),
    ("pode por matuê", "spotify", "play_anything", ("matuê",), "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", "matuê"),
    ("coloca a música modo escuro", "spotify", "play_anything", ("modo escuro",),
     "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", "modo escuro"),
]
CASES = NEW_MAC + NEW_SPOTIFY + EXISTING
FORMS = ("{}", "{}.", "Cortex, {}, por favor.")


def routing_copy(text):
    return normalise_stt_aliases(strip_courtesy(
        strip_trailing_sentence_punctuation(strip_vocatives(text))))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_simulate_mode_and_capture(case, form):
    text, _, _, _, mode, matcher, capture = case
    result = registry.simulate(routing_copy(form.format(text)))
    assert result is not None
    assert (result.mode, result.matcher, result.capture) == (mode, matcher, capture)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_process_calls_only_the_right_fake(orch, case, form):
    text, integration, method, args, mode, _, _ = case
    fake = getattr(orch, integration)
    getattr(fake, method).return_value = "Done."
    assert orch.process(form.format(text), language="pt") == ("Done.", mode)
    kwargs = {"lang": "pt"}
    if method == "play_anything":
        kwargs["device_hint"] = "iphone" if "iphone" in text else None
    assert fake.method_calls == [getattr(call, method)(*args, **kwargs)]
    other = orch.spotify if integration == "macos" else orch.macos
    assert other.method_calls == []
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).assert_not_called()


# All seven are currently registry misses / OFFLINE fallthrough. Only the
# video-music request carries TOOL_HINT_RE; this is advisory, not dispatch.
NEVER_STEAL = (
    ("como tiro o som de fundo de um vídeo?", False),
    ("tira o som do vídeo no iMovie", False),
    ("deixa a tela do site mais clara", False),
    ("o teclado está mais forte?", False),
    ("por que a tela fica mais clara?", False),
    ("bota uma música no meu vídeo", True),
    ("acende a luz da sala", False),
)

# Task 145b: device/system sound is not an artist, including when the
# optional artist-object prefix backtracks to the plain play capture.
# "põe um som" retains the existing advisory tool hint on fallthrough.
DEVICE_SOUND = (
    ("bota som do computador mais alto", False),
    ("coloca som do mac no máximo", False),
    ("põe um som do mac mais alto", True),
    ("bota um som do macbook", False),
    ("coloca som do sistema mais alto", False),
)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text,tools", NEVER_STEAL + DEVICE_SOUND)
def test_never_steal_keeps_current_fallthrough(orch, text, tools, form):
    text = form.format(text)
    routing = routing_copy(text)
    assert registry.simulate(routing) is None
    assert registry.simulate(routing, registry.POST_PLANNER_ROUTES) is None
    outcome = simulate_dispatch(text)
    assert (outcome.mode, outcome.matcher, outcome.capture, outcome.would_use_tools) == (
        "OFFLINE", "fallthrough", None, tools)
    assert orch.process(text, language="pt")[1] == "OFFLINE"
    assert orch.macos.method_calls == []
    assert orch.spotify.method_calls == []


@pytest.mark.parametrize("text", ("por que azul?", "por quê?", "por favor", "por acaso chove?"))
def test_por_preposition_guard_survives(text):
    assert registry.simulate(routing_copy(text)) is None


@pytest.mark.parametrize("device", (
    "mac", "macbook", "computador", "notebook", "sistema", "iphone",
    "celular", "telefone", "tv", "carro", "fone",
))
@pytest.mark.parametrize("connector", ("do", "de", "da"))
@pytest.mark.parametrize("obj", ("música", "uma música", "som", "um som"))
@pytest.mark.parametrize("verb", ("bota", "coloca", "põe"))
def test_device_is_not_an_artist_or_plain_sound_query(verb, obj, connector, device):
    text = routing_copy(f"{verb} {obj} {connector} {device} mais alto")
    assert registry.simulate(text) is None
