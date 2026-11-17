"""Task 155: Spotify resolves a bare title's ambiguous Portuguese courtesy tail."""
import logging
from unittest.mock import Mock, call

import pytest
import spotipy

from src.core.dispatch.normalisation import (
    removed_portuguese_courtesy_suffix,
    strip_courtesy,
)
from src.tools.spotify_player import SpotifyPlayer
from tests.test_manda_artista_154 import turn
from tests.test_taught_facts import orch


class FakeSpotify(SpotifyPlayer):
    """Real candidate search and URI playback, with every I/O boundary faked."""

    def __init__(self):
        super().__init__()
        self._client = Mock()
        self._client.search.return_value = {"tracks": {"items": []}}
        self.is_configured = Mock(return_value=True)
        self.play_anything = Mock(return_value="Playing fallback.")
        self.play_playlist = Mock(return_value="Playing playlist.")
        self.play_artist_radio = Mock(return_value="Playing radio.")
        self._library_artist_names = Mock(return_value=["Veigh"])
        self._start_playback_with_fallback = Mock()
        self._audit_pick = Mock()


@pytest.fixture
def spotify(orch):
    orch.spotify = FakeSpotify()
    return orch.spotify


def track(title, uri="spotify:track:exact"):
    return {"name": title, "uri": uri, "artists": [{"name": "Singer"}]}


TITLES = (
    ("toca volta pra mim", "volta", "volta pra mim", "Volta Pra Mim"),
    ("toca volta para mim", "volta", "volta para mim", "Volta Para Mim"),
    ("coloca liga pra mim", "liga", "liga pra mim", "Liga Pra Mim"),
    ("toca a música volta pra mim", "volta", "volta pra mim", "Volta Pra Mim"),
    ("toca voce pra mim", "voce", "voce pra mim", "VOCÊ PRA MIM"),
)


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,query,candidate,title", TITLES)
@pytest.mark.parametrize("form", ("{}", "Cortex, pode {}?!"))
def test_exact_title_plays_returned_track(orch, spotify, stream, text, query,
                                         candidate, title, form):
    # An unrelated first hit must not hide a later exact title.
    spotify.client.search.return_value = {"tracks": {"items": [
        track("Unrelated", "spotify:track:wrong"), track(title)]}}
    assert turn(orch, form.format(text), stream) == (
        f"Tocando '{title}' de Singer.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q=candidate, limit=5, type="track")
    spotify._start_playback_with_fallback.assert_called_once_with(
        uris=["spotify:track:exact"], device_hint=None)
    spotify.play_anything.assert_not_called()
    spotify._library_artist_names.assert_not_called()
    assert orch.macos.method_calls == []


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,query,candidate,title", TITLES + (
    ("toca matuê pra mim", "matuê", "matuê pra mim", "Matuê"),
))
def test_missing_exact_title_uses_original_query(orch, spotify, stream, text,
                                                query, candidate, title):
    # A near title or an artist/album equal to the candidate is insufficient.
    spotify.client.search.return_value = {"tracks": {"items": [{
        "name": query, "artists": [{"name": candidate}],
        "album": {"name": candidate}, "uri": "spotify:track:wrong"}]}}
    assert turn(orch, text, stream) == ("Playing fallback.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q=candidate, limit=5, type="track")
    spotify.play_anything.assert_called_once_with(query, device_hint=None, lang="pt")
    spotify._start_playback_with_fallback.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("name", (
    "Volta pra Mim - Ao Vivo", "Volta Pra Mim (feat. X)",
    "Volta pra Mim [Remastered]", "Volta pra Mim - Remasterizado 2019",
    "Volta Pra Mim - Remastered 2026", "Volta Pra Mim (feat. Singer)",
    "Volta Pra Mim (feat. X) [Remastered] - Ao Vivo",
    "Volta Pra Mim [Remastered] (feat. X)", "VÓLTA PRA MIM - Ao Vivo",
))
def test_base_title_plays_returned_version(orch, spotify, stream, name):
    spotify.client.search.return_value = {"tracks": {"items": [track(name)]}}
    assert turn(orch, "toca volta pra mim", stream) == (
        f"Tocando '{name}' de Singer.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    spotify._start_playback_with_fallback.assert_called_once_with(
        uris=["spotify:track:exact"], device_hint=None)
    spotify.play_anything.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("first,second", (
    ("Volta pra Mim - Ao Vivo", "Volta pra Mim"),
    ("Volta pra Mim", "Volta pra Mim - Ao Vivo"),
    ("Volta Pra Mim (feat. X)", "Volta pra Mim [Remastered]"),
))
def test_first_base_title_match_in_spotify_order_wins(
        orch, spotify, stream, first, second):
    spotify.client.search.return_value = {"tracks": {"items": [
        track("Volta", "spotify:track:wrong"),
        track("Volta pra Mim / Linda Demais - Ao Vivo", "spotify:track:medley"),
        track(first, "spotify:track:first"),
        track(second, "spotify:track:second"),
    ]}}
    assert turn(orch, "toca volta pra mim", stream) == (
        f"Tocando '{first}' de Singer.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    spotify._start_playback_with_fallback.assert_called_once_with(
        uris=["spotify:track:first"], device_hint=None)
    spotify.play_anything.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("name", (
    "Volta pra Mim / Linda Demais - Ao Vivo", "Você Vai Voltar pra Mim",
    "Volta", "Volta pra Mim Amor",
    "Volta Pra Mim!", "A Volta Pra Mim", "Volta Pra Mim Outra Vez",
    "Volta Para Mim", "Volta  Pra Mim",
    "Volta Pra Mim-Ao Vivo", "Volta Pra Mim- Ao Vivo", "Volta Pra Mim -Ao Vivo",
    "Volta (feat. X) Pra Mim", "Volta [Remastered] Pra Mim",
    "Volta Pra Mim (feat. X) Amor", "Volta Pra Mim (feat. X",
    "Volta Pra Mim [Remastered",
))
def test_base_title_equality_keeps_words_medleys_and_punctuation(
        orch, spotify, stream, name):
    spotify.client.search.return_value = {"tracks": {"items": [track(name)]}}
    assert turn(orch, "toca volta pra mim", stream) == ("Playing fallback.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    spotify.play_anything.assert_called_once_with("volta", device_hint=None, lang="pt")
    spotify._start_playback_with_fallback.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("text,method,query,mode", (
    ("toca volta please", "play_anything", "volta", "SPOTIFY_PLAY"),
    ("toca volta por favor", "play_anything", "volta", "SPOTIFY_PLAY"),
    ("toca volta pra mim please", "play_anything", "volta pra mim", "SPOTIFY_PLAY"),
    ("toca volta pra mim no meu iphone", "play_anything", "volta pra mim", "SPOTIFY_PLAY"),
    ("toca um som do matuê pra mim", "play_anything", "matuê", "SPOTIFY_PLAY"),
    ("manda um som do veigh pra mim", "play_anything", "veigh", "SPOTIFY_PLAY"),
    ("solta um som do veigh pra mim", "play_anything", "veigh", "SPOTIFY_PLAY"),
    ("manda um veigh pra mim", "play_anything", "Veigh", "SPOTIFY_PLAY"),
    ("solta um veigh pra mim", "play_anything", "Veigh", "SPOTIFY_PLAY"),
    ("toca playlist treino pra mim", "play_playlist", "treino", "SPOTIFY_PLAY_PLAYLIST"),
    ("toca rádio do matuê pra mim", "play_artist_radio", "matuê", "SPOTIFY_PLAY_RADIO"),
))
def test_other_phrases_keep_calls(orch, spotify, stream, text, method, query, mode):
    reply = getattr(spotify, method).return_value
    assert turn(orch, text, stream) == (reply, mode)
    device = "iphone" if "iphone" in text else None
    getattr(spotify, method).assert_called_once_with(query, device_hint=device, lang="pt")
    spotify.client.search.assert_not_called()
    spotify._start_playback_with_fallback.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
def test_open_spotify_keeps_calls(orch, spotify, stream):
    orch.macos.open_app.return_value = "Opened."
    assert turn(orch, "abre o spotify pra mim", stream) == ("Opened.", "MAC_OPEN_APP")
    orch.macos.open_app.assert_called_once_with("spotify", lang="pt")
    spotify.client.search.assert_not_called()
    spotify.play_anything.assert_not_called()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("error", (
    RuntimeError("private exception detail"),
    spotipy.SpotifyException(503, -1, "private exception detail"),
))
def test_search_error_falls_back_and_logs_type_only(orch, spotify, stream, error, caplog):
    spotify.client.search.side_effect = error
    with caplog.at_level(logging.WARNING, logger="src.tools.spotify_player"):
        assert turn(orch, "toca volta pra mim", stream) == ("Playing fallback.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    spotify.play_anything.assert_called_once_with("volta", device_hint=None, lang="pt")
    records = [r for r in caplog.records if r.name == "src.tools.spotify_player"]
    assert len(records) == 1
    assert records[0].getMessage() == (
        f"Spotify title candidate search failed: {type(error).__name__}")
    assert records[0].exc_info is None
    assert "private exception detail" not in caplog.text


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("exact", (False, True))
def test_device_and_language_are_preserved(orch, spotify, stream, exact):
    if exact:
        spotify.client.search.return_value = {"tracks": {"items": [track("Volta Pra Mim")]}}
    expected = "Playing 'Volta Pra Mim' by Singer." if exact else "Playing fallback."
    assert turn(orch, "toca volta no meu iphone pra mim", stream, language="en") == (
        expected, "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    if exact:
        spotify._start_playback_with_fallback.assert_called_once_with(
            uris=["spotify:track:exact"], device_hint="iphone")
        spotify.play_anything.assert_not_called()
    else:
        spotify.play_anything.assert_called_once_with("volta", device_hint="iphone", lang="en")


@pytest.mark.parametrize("stream", (False, True))
def test_empty_results_and_next_turn_do_not_reuse_suffix(orch, spotify, stream):
    assert turn(orch, "toca volta pra mim", stream) == ("Playing fallback.", "SPOTIFY_PLAY")
    assert turn(orch, "toca liga", stream) == ("Playing fallback.", "SPOTIFY_PLAY")
    spotify.client.search.assert_called_once_with(q="volta pra mim", limit=5, type="track")
    assert spotify.play_anything.call_args_list == [
        call("volta", device_hint=None, lang="pt"),
        call("liga", device_hint=None, lang="pt")]


@pytest.mark.parametrize("text,routing,suffix", (
    ("toca volta pra mim", "toca volta", "pra mim"),
    ("pode tocar volta PARA MIM?!", "tocar volta", "PARA MIM"),
    ("toca volta please", "toca volta", None),
    ("toca volta por favor", "toca volta", None),
    ("coloca na nota Mercado: arroz pra mim", "coloca na nota Mercado: arroz pra mim", None),
    ("pra mim", "pra mim", None),
    ("", "", None),
))
def test_helper_observes_existing_strip_contract(text, routing, suffix):
    assert strip_courtesy(text) == routing
    assert removed_portuguese_courtesy_suffix(text) == suffix
