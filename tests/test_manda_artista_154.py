"""Task 154: bare manda/solta needs exact saved-artist membership."""
import logging
from unittest.mock import Mock

import pytest

from src.core.dispatch import registry
from src.core.dispatch.spotify import SPOTIFY_LIBRARY_ARTIST_RE, SPOTIFY_PLAY_RE
from src.core.hallucination import routed_command_match
from tests.test_natural_phrasings_145 import FORMS
from tests.test_taught_facts import orch


ARTISTS = ["Veigh", "Matuê", "Anitta", "Djonga", "Racionais MC's"]
SEND_OBJECTS = (
    "email", "e-mail", "mensagem", "msg", "oi", "beijo", "abraço", "abraco",
    "áudio", "audio", "foto", "print", "pix", "link", "arquivo", "resumo",
    "lembrete", "zap", "recado", "convite", "vídeo", "video", "documento",
    "sms", "whatsapp", "telegram", "imessage", "direct", "dm", "som", "música", "musica",
)
PLAYS = [
    ("manda um veigh aí", "Veigh", None),
    ("manda um Veigh", "Veigh", None),
    ("solta um matuê", "Matuê", None),
    ("manda uma anitta aí", "Anitta", None),
    ("manda aí um djonga", "Djonga", None),
    ("solta aí um veigh ai", "Veigh", None),
    ("solta ai um MATUE aí", "Matuê", None),
    ("manda um racionais mc's aí", "Racionais MC's", None),
    ("manda um veigh pra mim", "Veigh", None),
    ("solta um matue para tocar", "Matuê", None),
    ("manda um veigh aí no meu iphone", "Veigh", "iphone"),
    ("manda um veigh pra tocar no spotify no meu mac aí", "Veigh", "mac"),
]


class FakeSpotify:
    def __init__(self, names=ARTISTS):
        self._library_artist_names = Mock(return_value=list(names))
        self.play_anything = Mock(return_value="Playing.")


@pytest.fixture
def spotify(orch):
    orch.spotify = FakeSpotify()
    return orch.spotify


def turn(orch, text, stream, language="pt"):
    if stream:
        frames = list(orch.process_stream(text, language=language))
        assert frames[-1][1] is True
        return "".join(frame[0] for frame in frames), frames[-1][2]
    return orch.process(text, language=language)


def llm_calls(orch):
    return sum(getattr(engine, operation).call_count
               for engine in (orch.engine_analytical, orch.engine_creative)
               for operation in ("chat", "chat_stream", "chat_with_tools", "generate"))


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text,canonical,device", PLAYS)
def test_saved_artist_plays_without_llm(orch, spotify, text, canonical, device, form, stream):
    assert turn(orch, form.format(text), stream) == ("Playing.", "SPOTIFY_PLAY")
    spotify._library_artist_names.assert_called_once_with()
    spotify.play_anything.assert_called_once_with(canonical, device_hint=device, lang="pt")
    assert llm_calls(orch) == 0
    assert orch.macos.method_calls == []


@pytest.mark.parametrize("stream", (False, True))
def test_play_forwards_reply_language(orch, spotify, stream):
    assert turn(orch, "manda um veigh aí", stream, language="en") == ("Playing.", "SPOTIFY_PLAY")
    spotify.play_anything.assert_called_once_with("Veigh", device_hint=None, lang="en")


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,names", (
    ("manda um caetano aí", ARTISTS),
    ("manda um veigh aí", []),
    # No 0.7 fuzzy matching, prefixes, or punctuation removal for intent.
    ("manda um veig aí", ARTISTS),
    ("manda um matu aí", ARTISTS),
    ("manda um jonga aí", ARTISTS),
    ("manda um racionais aí", ARTISTS),
    ("manda um racionais mcs aí", ARTISTS),
))
def test_unknown_artist_reaches_llm(orch, spotify, text, names, stream):
    spotify._library_artist_names.return_value = names
    assert turn(orch, text, stream) == ("Model answer.", "OFFLINE")
    spotify._library_artist_names.assert_called_once_with()
    spotify.play_anything.assert_not_called()
    assert llm_calls(orch) > 0


@pytest.mark.parametrize("stream", (False, True))
def test_library_exception_falls_through_and_logs_only_type(orch, spotify, stream, caplog):
    spotify._library_artist_names.side_effect = RuntimeError("private exception detail")
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert turn(orch, "manda um veigh aí", stream) == ("Model answer.", "OFFLINE")
    spotify._library_artist_names.assert_called_once_with()
    spotify.play_anything.assert_not_called()
    assert llm_calls(orch) > 0
    records = [r for r in caplog.records if r.name == registry.__name__]
    assert len(records) == 1
    assert records[0].getMessage() == "Spotify library artist lookup failed: RuntimeError"
    assert records[0].exc_info is None
    assert "private exception detail" not in caplog.text


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("obj", SEND_OBJECTS)
@pytest.mark.parametrize("tail", (" aí", " pessoal aí"))
def test_send_objects_never_look_up_library(orch, spotify, obj, verb, tail, stream):
    # Even an exact library entry cannot override a sending object.
    spotify._library_artist_names.return_value = [obj, obj + " pessoal"]
    article = "uma" if obj in ("música", "musica") else "um"
    text = f"{verb} {article} {obj}{tail}"
    assert SPOTIFY_LIBRARY_ARTIST_RE.match(text) is None
    assert turn(orch, text, stream) == ("Model answer.", "OFFLINE")
    spotify._library_artist_names.assert_not_called()
    spotify.play_anything.assert_not_called()
    assert llm_calls(orch) > 0


RECIPIENTS = ("pra Maria", "pro João", "para a Ana", "pros amigos", "pras meninas")
CHANNELS = tuple(f"{prep} {channel}"
                 for prep in ("por", "pelo", "pela", "no", "na", "via")
                 for channel in ("email", "e-mail", "mensagem", "msg", "whatsapp", "zap",
                                 "sms", "telegram", "imessage", "direct", "dm"))


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("verb", ("manda", "solta"))
@pytest.mark.parametrize("refusal", RECIPIENTS + CHANNELS)
@pytest.mark.parametrize("template", (
    "{verb} {refusal} um veigh", "{verb} aí um veigh {refusal}",
    "{verb} aí um veigh {refusal} remix",
))
def test_recipients_and_channels_anywhere_never_look_up_library(
        orch, spotify, stream, verb, refusal, template):
    text = template.format(verb=verb, refusal=refusal)
    assert SPOTIFY_LIBRARY_ARTIST_RE.match(text) is None
    assert turn(orch, text, stream) == ("Model answer.", "OFFLINE")
    spotify._library_artist_names.assert_not_called()
    spotify.play_anything.assert_not_called()
    assert llm_calls(orch) > 0


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text", (
    "manda um artista com cinco palavras aqui aí",
    "solta aí uma banda com cinco palavras aqui",
    "manda veigh", "solta veigh", "manda o veigh", "me manda um veigh aí",
    "manda um", "solta uma", "como está o tempo?", "veigh é legal",
))
def test_non_candidates_never_look_up_library(orch, spotify, text, stream):
    assert SPOTIFY_LIBRARY_ARTIST_RE.match(text) is None
    assert turn(orch, text, stream)[1] == "OFFLINE"
    spotify._library_artist_names.assert_not_called()
    spotify.play_anything.assert_not_called()
    assert llm_calls(orch) > 0


@pytest.mark.parametrize("words", (1, 2, 3, 4))
def test_candidate_accepts_at_most_four_artist_words(orch, spotify, words):
    name = " ".join(("One", "Two", "Three", "Four")[:words])
    spotify._library_artist_names.return_value = [name]
    assert registry.run(orch, f"manda um {name.lower()} aí") == ("Playing.", "SPOTIFY_PLAY")
    spotify.play_anything.assert_called_once_with(name, device_hint=None, lang="pt")


@pytest.mark.parametrize("stream", (False, True))
def test_explicit_artist_object_keeps_earlier_route(orch, spotify, stream):
    text = "manda um som do veigh aí"
    assert SPOTIFY_PLAY_RE.match(text).group(1) == "veigh"
    assert turn(orch, text, stream) == ("Playing.", "SPOTIFY_PLAY")
    spotify.play_anything.assert_called_once_with("veigh", device_hint=None, lang="pt")
    spotify._library_artist_names.assert_not_called()
    assert llm_calls(orch) == 0


def test_simulator_and_offer_guard_cannot_claim_library_membership(orch, spotify):
    routes = registry.DISPATCH_ROUTES
    index = next(i for i, route in enumerate(routes) if route.regex is SPOTIFY_PLAY_RE)
    route = routes[index + 1]
    assert route.regex is SPOTIFY_LIBRARY_ARTIST_RE
    assert route.mode is None
    for text, _, _ in PLAYS:
        assert registry.simulate(text) is None
        assert routed_command_match(text) is None
    spotify._library_artist_names.assert_not_called()
