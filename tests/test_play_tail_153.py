"""Task 153: artist-only filler stripping and plural artist connectors."""
import pytest

from src.core.dispatch.spotify import SPOTIFY_PLAY_RE
from src.core.hallucination import routed_command_match


BARE_TITLES = [
    ("toca volta pra mim", "volta pra mim"),
    ("toca tô nem aí", "tô nem aí"),
    ("toca tá aí", "tá aí"),
    ("toca ai ai ai", "ai ai ai"),
    ("coloca liga pra mim", "liga pra mim"),
    ("toca a música volta pra mim", "volta pra mim"),
    ("play tô nem aí", "tô nem aí"),
    ("toca diz pra mim do Malta", "diz pra mim do Malta"),
    ("toca volta para mim", "volta para mim"),
    # Restore the pre-150 values: these have no explicit artist object.
    ("toca matuê aí", "matuê aí"),
    ("toca matuê ai", "matuê ai"),
    ("toca samurai ai", "samurai ai"),
    ("toca bonsai aí", "bonsai aí"),
    ("toca samurai", "samurai"),
]


# normalisation._COURTESY_SUFFIX_RE strips trailing pra/para mim: a known June limitation outside 153.
PIPELINE_BARE_TITLES = [
    ("toca tô nem aí", "tô nem aí"),
    ("toca tá aí", "tá aí"),
    ("toca ai ai ai", "ai ai ai"),
    ("play tô nem aí", "tô nem aí"),
    ("toca matuê aí", "matuê aí"),
]


def assert_regex_query(text, query):
    match = SPOTIFY_PLAY_RE.match(text)
    assert match is not None
    assert match.group(1) == query


def assert_play_query(text, query):
    assert_regex_query(text, query)
    route = routed_command_match(text)
    assert route is not None
    assert (route.mode, route.matcher, route.capture) == (
        "SPOTIFY_PLAY", "SPOTIFY_PLAY_RE", query)


@pytest.mark.parametrize("text,query", BARE_TITLES)
def test_bare_title_keeps_its_last_words(text, query):
    assert_regex_query(text, query)


@pytest.mark.parametrize("text,query", PIPELINE_BARE_TITLES)
def test_bare_title_keeps_its_last_words_in_pipeline(text, query):
    assert_play_query(text, query)


@pytest.mark.parametrize("verb", ("manda", "solta", "toca", "bota"))
@pytest.mark.parametrize("obj,connector,artist", (
    ("um som", "dos", "Racionais"),
    ("uma música", "das", "Spice Girls"),
))
@pytest.mark.parametrize("tail", ("", " aí", " ai", " pra mim", " para mim"))
def test_plural_artist_object_strips_only_filler(verb, obj, connector, artist, tail):
    assert_play_query(f"{verb} {obj} {connector} {artist}{tail}", artist)


@pytest.mark.parametrize("text", (
    "toca um som do matuê aí",
    "toca um som do matuê ai",
    "toca uma música do matuê pra mim",
    "toca uma música do matuê para mim",
    "toca um som do matuê no spotify aí pra mim",
    "toca um som do matuê pra mim no meu iphone",
    "toca\tum som dos\t Racionais\t aí",
))
def test_explicit_artist_object_allows_filler(text):
    assert_play_query(text, "Racionais" if "Racionais" in text else "matuê")


@pytest.mark.parametrize("tail", (
    " pra tocar", " para tocar", " no spotify", " no meu iphone",
    " pra tocar no spotify no meu mac",
))
@pytest.mark.parametrize("text,query", BARE_TITLES)
def test_common_tails_still_strip_after_bare_titles(text, query, tail):
    assert_regex_query(text + tail, query)


@pytest.mark.parametrize("tail", (
    " pra tocar", " para tocar", " no spotify", " no meu iphone",
    " pra tocar no spotify no meu mac",
))
@pytest.mark.parametrize("text,query", PIPELINE_BARE_TITLES)
def test_common_tails_still_strip_after_bare_titles_in_pipeline(text, query, tail):
    assert_play_query(text + tail, query)


@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("obj,connector,artist", (
    ("um som", "dos", "Racionais"),
    ("uma música", "das", "Spice Girls"),
))
@pytest.mark.parametrize("tail", (" pro João", " pra Maria", " para a Ana",
                                  " por email", " no whatsapp"))
def test_plural_recipient_and_channel_do_not_route(verb, obj, connector, artist, tail):
    text = f"{verb} {obj} {connector} {artist}{tail}"
    assert SPOTIFY_PLAY_RE.match(text) is None
    assert routed_command_match(text) is None


@pytest.mark.parametrize("verb", ("toca", "bota", "manda", "solta"))
@pytest.mark.parametrize("obj", ("o som", "um som", "uma música"))
@pytest.mark.parametrize("connector,device", (("dos", "fones"), ("das", "tv")))
def test_plural_device_artist_does_not_route(verb, obj, connector, device):
    text = f"{verb} {obj} {connector} {device}"
    assert SPOTIFY_PLAY_RE.match(text) is None
    assert routed_command_match(text) is None
