"""
SpotifyPlayer reply localisation (s60, Claude-A).

The deterministic command path now replies in the SPOKEN language: the
dispatch wrappers pass `lang=o._reply_lang` into the player methods, which
render the reply via `src.core.i18n.t`. These tests prove EN replies when
`lang="en"` and that PT stays the default (no regression). No network — the
spotipy client is mocked.
"""
from unittest.mock import MagicMock

import pytest
import spotipy


@pytest.fixture
def player(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test_cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test_secret")
    from src.tools import spotify_player
    # is_configured() also requires the cached token (M4-A2).
    token = tmp_path / "spotify_token.json"
    token.write_text("{}")
    monkeypatch.setattr(spotify_player, "_TOKEN_PATH", str(token))
    p = spotify_player.SpotifyPlayer()
    p._client = MagicMock()
    return p


def _exc(status: int) -> spotipy.SpotifyException:
    return spotipy.SpotifyException(http_status=status, code=-1, msg="x")


class TestSimpleControlsEN:
    def test_pause_en(self, player):
        player._with_wake_on_404 = MagicMock(return_value=None)
        assert player.pause(lang="en") == "Paused."

    def test_pause_pt_default(self, player):
        player._with_wake_on_404 = MagicMock(return_value=None)
        assert player.pause() == "Pausado."
        assert player.pause(lang="pt") == "Pausado."

    def test_resume_en(self, player):
        player._client.start_playback = MagicMock(return_value=None)
        assert player.resume(lang="en") == "Resuming."

    def test_next_en(self, player):
        player._with_wake_on_404 = MagicMock(return_value=None)
        assert player.next_track(lang="en") == "Next track."

    def test_previous_en(self, player):
        player._with_wake_on_404 = MagicMock(return_value=None)
        assert player.previous_track(lang="en") == "Going back."

    def test_volume_en(self, player):
        player._with_wake_on_404 = MagicMock(return_value=None)
        assert player.set_volume(40, lang="en") == "Volume 40%."

    def test_volume_invalid_en(self, player):
        assert player.set_volume("loud", lang="en") == "Invalid volume. Use a number from 0 to 100."

    def test_volume_invalid_pt(self, player):
        assert player.set_volume("alto") == "Volume inválido. Use um número de 0 a 100."


class TestPlaybackRepliesEN:
    def test_play_track_not_found_en(self, player):
        player._client.search = MagicMock(return_value={"tracks": {"items": []}})
        assert player.play_track("zzz", lang="en") == "Couldn't find 'zzz' on Spotify."

    def test_play_track_not_found_pt(self, player):
        player._client.search = MagicMock(return_value={"tracks": {"items": []}})
        assert player.play_track("zzz") == "Não achei 'zzz' no Spotify."

    def test_play_track_ok_en(self, player):
        player._client.search = MagicMock(return_value={
            "tracks": {"items": [{"uri": "u", "name": "Song", "artists": [{"name": "Band"}]}]}
        })
        player._start_playback_with_fallback = MagicMock(return_value=None)
        assert player.play_track("song", lang="en") == "Playing 'Song' by Band."

    def test_now_playing_nothing_en(self, player):
        player._client.current_playback = MagicMock(return_value=None)
        assert player.now_playing(lang="en") == "Nothing playing right now."

    def test_now_playing_cache_is_per_language(self, player):
        # M4-A14d: the cache stores a LOCALISED string. A PT turn primes
        # it; an EN turn inside the 3s TTL must NOT be served the PT
        # message — lang mismatch refetches and re-caches in EN.
        playback = {"item": {"name": "Song", "artists": [{"name": "Band"}]}}
        player._client.current_playback = MagicMock(return_value=playback)
        assert player.now_playing(lang="pt") == "Tocando 'Song' de Band."
        assert player.now_playing(lang="en") == "Playing 'Song' by Band."
        # ...and back: the EN re-cache must not stick for PT either.
        assert player.now_playing(lang="pt") == "Tocando 'Song' de Band."
        assert player._client.current_playback.call_count == 3

    def test_now_playing_cache_still_serves_same_language(self, player):
        # The TTL cache keeps working when the language doesn't change:
        # second PT call within the TTL doesn't hit the API again.
        playback = {"item": {"name": "Song", "artists": [{"name": "Band"}]}}
        player._client.current_playback = MagicMock(return_value=playback)
        player.now_playing(lang="pt")
        player.now_playing(lang="pt")
        assert player._client.current_playback.call_count == 1

    def test_like_nothing_en(self, player):
        player._client.current_playback = MagicMock(return_value=None)
        assert player.like_current(lang="en") == "Nothing playing to like."


class TestErrorMessagesEN:
    def test_need_creds_en(self, monkeypatch):
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
        from src.tools.spotify_player import SpotifyPlayer
        p = SpotifyPlayer()
        msg = p.pause(lang="en")
        assert "isn't configured" in msg

    def test_friendly_error_404_en(self):
        from src.tools.spotify_player import SpotifyPlayer
        assert SpotifyPlayer._friendly_error(_exc(404), "en").startswith("No active Spotify device")

    def test_friendly_error_403_en(self):
        from src.tools.spotify_player import SpotifyPlayer
        assert SpotifyPlayer._friendly_error(_exc(403), "en") == "This command requires Spotify Premium."

    def test_friendly_error_404_pt_default(self):
        from src.tools.spotify_player import SpotifyPlayer
        assert SpotifyPlayer._friendly_error(_exc(404)).startswith("Nenhum dispositivo")


class TestAPausedTrackIsNotPlaying:
    """now_playing checked only that an item came back: a paused track was
    announced "Tocando..." (s88, Codex lote 13 — invariant 7)."""

    def test_paused_pt(self, player):
        player._client.current_playback = MagicMock(return_value={
            "is_playing": False, "item": {"name": "Song", "artists": [{"name": "Band"}]}})
        assert player.now_playing() == "Pausado em 'Song' de Band."

    def test_paused_en(self, player):
        player._client.current_playback = MagicMock(return_value={
            "is_playing": False, "item": {"name": "Song", "artists": [{"name": "Band"}]}})
        assert player.now_playing(lang="en") == "Paused on 'Song' by Band."

    def test_playing_is_still_playing(self, player):
        player._client.current_playback = MagicMock(return_value={
            "is_playing": True, "item": {"name": "Song", "artists": [{"name": "Band"}]}})
        assert player.now_playing() == "Tocando 'Song' de Band."
