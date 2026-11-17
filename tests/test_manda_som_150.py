"""Task 150: artist-object manda/solta requests play without an LLM."""
from unittest.mock import call

import pytest

from evals.dispatch import simulate_dispatch
from src.core.dispatch import registry
from src.core.dispatch.spotify import SPOTIFY_PLAY_RE
from src.core.hallucination import routed_command_match
from tests.test_natural_phrasings_145 import FORMS, routing_copy
from tests.test_taught_facts import orch


# text, exact query, device hint
PLAY_EXAMPLES = [
    ("manda um som do veigh", "veigh", None),
    ("manda aí um som do veigh", "veigh", None),
    ("manda o som do veigh", "veigh", None),
    ("solta um som do veigh", "veigh", None),
    ("manda uma música do matuê", "matuê", None),
    ("manda um som do kendrick aí", "kendrick", None),
    ("manda uma música do matuê pra mim", "matuê", None),
    ("solta aí a música da Anitta pra tocar no meu iphone", "Anitta", "iphone"),
    ("manda o som de Kendrick Lamar no spotify aí pra mim", "Kendrick Lamar", None),
    ("manda um som do veigh pra mim no meu mac", "veigh", "mac"),
    ("manda uma música do matuê para mim", "matuê", None),
    ("manda ai um som do veigh", "veigh", None),
    ("manda um som do veigh ai", "veigh", None),
    # Task 153: restore pre-150 bare-title queries (no artist object).
    ("toca matuê aí", "matuê aí", None),
    ("toca matuê ai", "matuê ai", None),
    ("toca samurai", "samurai", None),
    ("toca bonsai", "bonsai", None),
]
PLAY = PLAY_EXAMPLES + [
    (f"{verb}{before} {obj} {connector} {artist}{tail}", artist, None)
    for verb in ("manda", "solta")
    for before in ("", " aí", " ai")
    for obj in ("um som", "o som", "uma música", "a música")
    for connector, artist in (("do", "veigh"), ("da", "anitta"), ("de", "djonga"))
    for tail in ("", " aí", " ai", " pra mim", " para mim", " pra tocar", " para tocar")
]


@pytest.mark.parametrize("text,query,device", PLAY)
def test_exact_artist_query_and_offer_route(text, query, device):
    match = SPOTIFY_PLAY_RE.match(text)
    assert match is not None
    assert match.group(1) == query
    outcome = simulate_dispatch(text)
    assert (outcome.mode, outcome.matcher, outcome.capture) == (
        "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", query)
    offer = routed_command_match(text)
    assert offer is not None
    assert (offer.mode, offer.capture) == ("SPOTIFY_PLAY", query)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text,query,device", PLAY_EXAMPLES)
def test_both_entry_points_play_only_the_exact_artist(orch, text, query, device, form, stream):
    orch.spotify.play_anything.return_value = "Playing."
    if stream:
        frames = list(orch.process_stream(form.format(text), language="pt"))
        assert frames[-1][1:3] == (True, "SPOTIFY_PLAY")
        reply = "".join(frame[0] for frame in frames)
    else:
        reply, mode = orch.process(form.format(text), language="pt")
        assert mode == "SPOTIFY_PLAY"
    assert reply == "Playing."
    assert orch.spotify.method_calls == [
        call.play_anything(query, device_hint=device, lang="pt")]
    assert orch.macos.method_calls == []
    for engine in (orch.engine_analytical, orch.engine_creative):
        for operation in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, operation).assert_not_called()


# Measured on the unmodified clone: every phrase is OFFLINE/fallthrough,
# with tools hinted for manda, but not solta. None has an existing route.
NEVER_STEAL = (
    ("manda um veigh aí", True),
    ("manda o som do veigh pro João", True),
    ("manda uma música do matuê para a Ana", True),
    ("manda um som do veigh pra Maria", True),
    ("manda o som do veigh por email", True),
    ("manda o som do veigh no whatsapp", True),
    ("manda um som do veigh por mensagem", True),
    ("manda um email pro João", True),
    ("manda uma mensagem pra Ana", True),
    ("manda a letra da música do veigh", True),
    ("manda o link do som do veigh", True),
    ("me manda um som do veigh por email", True),
    ("manda um áudio", True),
    ("solta o som", False),
    ("manda o som do mac mais alto", True),
    ("solta o som do computador", False),
)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text,tools", NEVER_STEAL)
def test_never_steal_preserves_measured_fallthrough(text, tools, form):
    text = form.format(text)
    routing = routing_copy(text)
    assert SPOTIFY_PLAY_RE.match(routing) is None
    assert registry.simulate(routing) is None
    assert registry.simulate(routing, registry.POST_PLANNER_ROUTES) is None
    outcome = simulate_dispatch(text)
    assert (outcome.mode, outcome.matcher, outcome.capture, outcome.would_use_tools) == (
        "OFFLINE", "fallthrough", None, tools)
    assert routed_command_match(text) is None


@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("recipient", ("pra Maria", "para a Ana", "pro João", "pros amigos", "pras meninas"))
def test_recipient_is_not_a_play_tail(verb, recipient):
    # These prepositions introduce recipients, not playback fluff.
    text = f"{verb} aí o som do veigh {recipient}"
    assert SPOTIFY_PLAY_RE.match(text) is None


@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("preposition", ("por", "pelo", "pela", "no", "na", "via"))
@pytest.mark.parametrize("channel", (
    "email", "e-mail", "mensagem", "msg", "whatsapp", "zap", "sms",
    "telegram", "imessage", "direct", "dm",
))
@pytest.mark.parametrize("template", (
    "{verb} {channel} ai o som do veigh",
    "{verb} ai o som do veigh {channel} remix",
    "{verb} ai o som do veigh para mim {channel} no meu iphone",
))
def test_sending_channel_anywhere_after_verb_is_not_playback(verb, preposition, channel, template):
    text = template.format(verb=verb, channel=f"{preposition} {channel}")
    assert SPOTIFY_PLAY_RE.match(text) is None


@pytest.mark.parametrize("text,query", (
    ("manda um som do veigh no whatsappinho", "veigh no whatsappinho"),
    ("manda um som do veigh viajem", "veigh viajem"),
    # Task 153: restore pre-150 bare-title queries (no artist object).
    ("toca samurai ai", "samurai ai"),
    ("toca bonsai aí", "bonsai aí"),
))
def test_channel_and_filler_guards_use_separate_words(text, query):
    match = SPOTIFY_PLAY_RE.match(text)
    assert match is not None
    assert match.group(1) == query


@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("obj", ("um som", "o som", "uma música", "a música"))
@pytest.mark.parametrize("device", ("mac", "computador", "macbook", "sistema", "iphone", "celular",
                                    "telefone", "tv", "carro", "fone", "notebook"))
def test_device_artist_cannot_backtrack_into_plain_query(verb, obj, device):
    assert SPOTIFY_PLAY_RE.match(f"{verb} aí {obj} do {device} mais alto") is None


@pytest.mark.parametrize("text", (
    "por que azul?", "por quê?", "por favor", "por acaso chove?",
    "toca o barco", "play it cool", "coloca no calendário: dentista amanhã",
    "manda som do veigh", "solta música do veigh", "manda um som", "solta veigh",
))
def test_non_music_and_required_article_guards(text):
    assert SPOTIFY_PLAY_RE.match(text) is None
