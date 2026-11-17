"""
SpotifyPlayer 404 auto-transfer fallback.

The bug fixed here: only `play_track` and `resume` had a wake-device
retry when Spotify returned 404 ("no active device"). Saying "próxima"
or "pausa" while no client was active surfaced a generic error instead
of waking up whichever device was available and retrying.

These tests cover the `_with_wake_on_404` helper and prove that all
four newly-protected methods (`pause`, `next_track`, `previous_track`,
`set_volume`) actually retry after a wake. No network — the spotipy
client is mocked end-to-end.
"""
from unittest.mock import MagicMock

import pytest
import spotipy


@pytest.fixture
def player(monkeypatch, tmp_path):
    """SpotifyPlayer with creds env present and a mocked spotipy client.
    Avoids touching OAuth or the real filesystem. is_configured() also
    requires the cached token (M4-A2), so a fake one is created in tmp."""
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test_cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test_secret")

    from src.tools import spotify_player
    token = tmp_path / "spotify_token.json"
    token.write_text("{}")
    monkeypatch.setattr(spotify_player, "_TOKEN_PATH", str(token))

    p = spotify_player.SpotifyPlayer()
    p._client = MagicMock()
    return p


def _spotify_404(msg: str = "no active device") -> spotipy.SpotifyException:
    return spotipy.SpotifyException(http_status=404, code=-1, msg=msg)


def _spotify_403() -> spotipy.SpotifyException:
    return spotipy.SpotifyException(http_status=403, code=-1, msg="premium required")


class TestWakeOn404Helper:
    """Direct coverage of the helper that the four methods share."""

    def test_no_retry_when_first_call_succeeds(self, player):
        action = MagicMock(return_value=None)
        player._with_wake_on_404(action, "arg1", kw="kw1")
        action.assert_called_once_with("arg1", kw="kw1")

    def test_retry_after_wake_on_404(self, player):
        # First call raises 404, second call succeeds.
        action = MagicMock(side_effect=[_spotify_404(), None])
        # _wake_any_device finds a device and transfers playback.
        player._wake_any_device = MagicMock(return_value="dev_xyz")
        player._with_wake_on_404(action)
        assert action.call_count == 2
        player._wake_any_device.assert_called_once()

    def test_raises_404_when_no_device_to_wake(self, player):
        action = MagicMock(side_effect=_spotify_404())
        player._wake_any_device = MagicMock(return_value=None)
        with pytest.raises(spotipy.SpotifyException) as exc:
            player._with_wake_on_404(action)
        assert exc.value.http_status == 404

    def test_non_404_errors_propagate_without_retry(self, player):
        action = MagicMock(side_effect=_spotify_403())
        player._wake_any_device = MagicMock()
        with pytest.raises(spotipy.SpotifyException) as exc:
            player._with_wake_on_404(action)
        assert exc.value.http_status == 403
        # Wake must NOT be called for non-404 errors.
        player._wake_any_device.assert_not_called()
        action.assert_called_once()


class TestPauseFallback:
    def test_success_path(self, player):
        player.client.pause_playback.return_value = None
        assert player.pause() == "Pausado."
        player.client.pause_playback.assert_called_once()

    def test_404_triggers_wake_and_retries(self, player):
        player.client.pause_playback.side_effect = [_spotify_404(), None]
        player._wake_any_device = MagicMock(return_value="dev_iphone")
        assert player.pause() == "Pausado."
        assert player.client.pause_playback.call_count == 2
        player._wake_any_device.assert_called_once()

    def test_404_without_wakeable_device_surfaces_friendly_error(self, player):
        player.client.pause_playback.side_effect = _spotify_404()
        player._wake_any_device = MagicMock(return_value=None)
        reply = player.pause()
        assert "Nenhum dispositivo" in reply


class TestNextTrackFallback:
    def test_success_path(self, player):
        player.client.next_track.return_value = None
        assert player.next_track() == "Próxima música."

    def test_404_triggers_wake_and_retries(self, player):
        player.client.next_track.side_effect = [_spotify_404(), None]
        player._wake_any_device = MagicMock(return_value="dev_speaker")
        assert player.next_track() == "Próxima música."
        assert player.client.next_track.call_count == 2

    def test_now_cache_invalidated_on_success(self, player):
        player._now_cache_msg = "stale cache"
        player.client.next_track.return_value = None
        player.next_track()
        assert player._now_cache_msg is None


class TestPreviousTrackFallback:
    def test_success_path(self, player):
        player.client.previous_track.return_value = None
        assert player.previous_track() == "Voltando."

    def test_404_triggers_wake_and_retries(self, player):
        player.client.previous_track.side_effect = [_spotify_404(), None]
        player._wake_any_device = MagicMock(return_value="dev_mac")
        assert player.previous_track() == "Voltando."
        assert player.client.previous_track.call_count == 2


class TestVolumeFallback:
    def test_success_path(self, player):
        player.client.volume.return_value = None
        assert player.set_volume(50) == "Volume 50%."
        player.client.volume.assert_called_once_with(50)

    def test_404_triggers_wake_and_retries(self, player):
        player.client.volume.side_effect = [_spotify_404(), None]
        player._wake_any_device = MagicMock(return_value="dev_iphone")
        assert player.set_volume(30) == "Volume 30%."
        # Both attempts use the same argument.
        calls = player.client.volume.call_args_list
        assert len(calls) == 2
        assert calls[0] == calls[1]

    def test_volume_clamped_before_call(self, player):
        player.client.volume.return_value = None
        player.set_volume(150)
        player.client.volume.assert_called_once_with(100)

    def test_invalid_volume_rejected_before_api_call(self, player):
        reply = player.set_volume("loud")
        assert "inválido" in reply.lower()
        player.client.volume.assert_not_called()


class TestTransferPlaybackTo:
    """transfer_playback_to moves active playback to another device with
    force_play=True. Distinct from the internal _wake_any_device (404
    fallback, force_play=False)."""

    def _devices(self, *devs):
        return {"devices": list(devs)}

    def test_transfer_to_matching_device_forces_play(self, player):
        player.client.devices.return_value = self._devices(
            {"id": "dev_phone", "name": "Pedro's iPhone", "type": "Smartphone",
             "is_active": False, "is_restricted": False},
            {"id": "dev_mac", "name": "MacBook", "type": "Computer",
             "is_active": True, "is_restricted": False},
        )
        msg = player.transfer_playback_to("iphone")
        player.client.transfer_playback.assert_called_once_with(
            device_id="dev_phone", force_play=True
        )
        assert "iPhone" in msg

    def test_transfer_when_target_already_active_is_noop(self, player):
        player.client.devices.return_value = self._devices(
            {"id": "dev_phone", "name": "iPhone", "type": "Smartphone",
             "is_active": True, "is_restricted": False},
        )
        msg = player.transfer_playback_to("iphone")
        player.client.transfer_playback.assert_not_called()
        assert "já tá tocando" in msg.lower()

    def test_transfer_no_devices_available(self, player):
        player.client.devices.return_value = self._devices()
        msg = player.transfer_playback_to("iphone")
        player.client.transfer_playback.assert_not_called()
        assert "nenhum dispositivo" in msg.lower()

    def test_transfer_names_actual_destination_on_hint_miss(self, player):
        # Hint 'tv' but only a computer is connected → _pick_device falls
        # back to the first non-restricted device; reply must name where
        # the sound actually went (not silently claim the tv).
        player.client.devices.return_value = self._devices(
            {"id": "dev_mac", "name": "MacBook", "type": "Computer",
             "is_active": False, "is_restricted": False},
        )
        msg = player.transfer_playback_to("tv")
        player.client.transfer_playback.assert_called_once_with(
            device_id="dev_mac", force_play=True
        )
        assert "MacBook" in msg

    def test_transfer_not_configured(self, player, monkeypatch):
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        msg = player.transfer_playback_to("iphone")
        assert "não configurado" in msg.lower()
        player.client.devices.assert_not_called()


# =====================================================================
# play_anything — smart default for "toca X" (sessão 17 bloco F)
# =====================================================================

class TestPlayAnything:
    """Decision tests for the smart default. The bug being fixed:
    'toca djonga' used to play one track and stop because play_track
    sends uris=[track] (no autoplay context). play_anything routes to
    artist context when the name matches strongly, falling back to
    play_track otherwise."""

    def test_artist_match_routes_to_artist_context(self, player):
        # Search returns Djonga; query is "djonga"; lowercase equal.
        player._client.search = MagicMock(return_value={
            "artists": {"items": [{"name": "Djonga", "uri": "spotify:artist:abc"}]},
        })
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_anything("djonga")
        assert "Djonga" in result
        # Critical: context_uri was used (autoplay), NOT uris=[...].
        called = player._client.start_playback.call_args.kwargs
        assert called.get("context_uri") == "spotify:artist:abc"
        assert "uris" not in called

    def test_track_by_artist_phrasing_skips_artist_branch(self, player):
        # "fim da noite do matuê" — explicit track-by-artist form.
        # play_anything must NOT short-circuit to artist; it must
        # delegate to the track path for the specific song. (M4-A14e: it
        # lands on _play_track_corrected — the query was already corrected
        # once; going through play_track would correct it twice.)
        player._correct_artist_query = lambda q: q
        player._play_track_by_artist = MagicMock(return_value="ok")
        result = player.play_anything("fim da noite do matuê")
        assert result == "ok"
        # s84: split into title + artist (last separator) and played as a
        # song, never through the artist branch.
        player._play_track_by_artist.assert_called_once_with(
            "fim da noite", "matuê", "fim da noite do matuê",
            device_hint=None, lang="pt",
        )
        # The artist search must NOT have been called.
        player._client.search.assert_not_called()

    def test_no_artist_match_falls_back_to_track(self, player):
        # Search returns an artist whose name doesn't overlap the query
        # at all — we don't trust the match, fall back to the track path.
        player._client.search = MagicMock(return_value={
            "artists": {"items": [{"name": "Wildcard", "uri": "spotify:artist:zzz"}]},
        })
        player._play_track_corrected = MagicMock(return_value="track-fallback")
        result = player.play_anything("some obscure song title")
        assert result == "track-fallback"
        player._play_track_corrected.assert_called_once()

    def test_play_anything_corrects_query_exactly_once(self, player):
        # M4-A14e: play_anything corrected and then play_track corrected
        # AGAIN — harmless via the library cache, but log noise. Pin one
        # correction per turn.
        calls = []
        orig = player._correct_artist_query

        def counting(q):
            calls.append(q)
            return orig(q)

        player._correct_artist_query = counting
        player._client.search = MagicMock(return_value={
            "artists": {"items": []}, "tracks": {"items": []},
        })
        player.play_anything("some obscure song title")
        assert len(calls) == 1, f"corrected {len(calls)}x: {calls}"


class TestTrackByArtist:
    """s84: "Play daisies by Justin Bieber" played "Beauty And A Beat". The
    library correction ran on the whole phrase, which is 0.70 similar to
    "Justin Bieber" (cutoff 0.70), so it became an artist request."""

    @staticmethod
    def _track(name, artist, uri):
        return {"name": name, "uri": uri, "artists": [{"name": artist}]}

    def _wire(self, player, by_query, library=("Justin Bieber", "Guilherme Batista")):
        player._library_artist_names = lambda: list(library)
        seen = []

        def search(q, limit=5, type="track"):
            seen.append((q, type))
            return {"tracks": {"items": by_query.get(q, [])}, "artists": {"items": []}}
        player._client.search = MagicMock(side_effect=search)
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        return seen

    def test_the_reported_case_plays_the_song_not_the_artist(self, player):
        seen = self._wire(player, {
            'track:"daisies" artist:"Justin Bieber"': [
                self._track("DAISIES", "Justin Bieber", "spotify:track:daisies")],
        })
        msg = player.play_anything("daisies by Justin Bieber")
        assert "DAISIES" in msg
        assert player._client.start_playback.call_args.kwargs["uris"] == ["spotify:track:daisies"]
        assert all(t == "track" for _, t in seen)  # never an artist search

    def test_karaoke_versions_are_skipped(self, player):
        self._wire(player, {
            'track:"daisies" artist:"Justin Bieber"': [
                self._track("DAISIES (By Justin Bieber) - Instrumental Karaoke Version",
                            "Justin Bieber", "spotify:track:karaoke"),
                self._track("DAISIES", "Justin Bieber", "spotify:track:daisies")],
        })
        player.play_anything("daisies by Justin Bieber")
        assert player._client.start_playback.call_args.kwargs["uris"] == ["spotify:track:daisies"]

    def test_only_the_artist_part_is_corrected(self, player):
        # The correction's job: a mangled artist name inside a song request.
        seen = self._wire(player, {
            'track:"super ex" artist:"Guilherme Batista"': [
                self._track("Super Ex", "Guilherme Batista", "spotify:track:sx")],
        })
        player.play_anything("super ex de guilherme batistta")
        assert seen[0][0] == 'track:"super ex" artist:"Guilherme Batista"'
        assert player._client.start_playback.call_args.kwargs["uris"] == ["spotify:track:sx"]

    def test_wrong_song_is_reported_not_played(self, player):
        self._wire(player, {
            'track:"daisies" artist:"Justin Bieber"': [
                self._track("Beauty And A Beat", "Justin Bieber", "spotify:track:bab")],
        })
        msg = player.play_anything("daisies by Justin Bieber")
        assert "Não achei 'daisies' de Justin Bieber" in msg
        player._client.start_playback.assert_not_called()

    def test_a_split_with_no_real_artist_falls_back_to_the_whole_phrase(self, player):
        # "musica de ninar" has no artist: the field search finds nothing,
        # so the phrase is searched whole, as before.
        self._wire(player, {
            "musica de ninar": [self._track("Canção de Ninar", "Vários", "spotify:track:n")],
        }, library=())
        msg = player.play_anything("musica de ninar")
        assert "Canção de Ninar" in msg

    def test_karaoke_is_played_when_asked_for(self, player):
        self._wire(player, {
            'track:"daisies karaoke" artist:"Justin Bieber"': [
                self._track("DAISIES - Instrumental Karaoke Version", "Justin Bieber",
                            "spotify:track:k")],
        })
        # Title asked is "daisies karaoke" — a karaoke result is what's wanted.
        msg = player.play_anything("daisies karaoke by Justin Bieber")
        assert "Karaoke" in msg

    def test_mangled_title_of_a_real_artist_is_found_among_their_songs(self, player):
        # "Super-Ess" is how the transcript heard "Super Ex".
        self._wire(player, {
            'artist:"Guilherme Batista"': [
                self._track("Intro", "Guilherme Batista", "spotify:track:intro"),
                self._track("A Super Ex - Ao Vivo", "Guilherme Batista", "spotify:track:sx")],
        })
        player.play_anything("Super-Ess de Guilherme Batista")
        assert player._client.start_playback.call_args.kwargs["uris"] == ["spotify:track:sx"]

    def test_missing_title_of_a_real_artist_is_not_replaced_by_any_song(self, player):
        # Live s84: the whole-phrase fallback played "Kenny G" for "fim da
        # noite do matuê". With a real artist, say so instead.
        self._wire(player, {
            "fim da noite do matuê": [self._track("Kenny G", "Matuê", "spotify:track:kg")],
            'artist:"Matuê"': [self._track("Kenny G", "Matuê", "spotify:track:kg")],
        }, library=("Matuê",))
        msg = player.play_anything("fim da noite do matuê")
        assert "Não achei 'fim da noite' de Matuê" in msg
        player._client.start_playback.assert_not_called()

    def test_artist_search_never_trusts_a_single_result(self, player):
        # Live s84: limit=1 for "djonga" returned "BK"; with 5, Djonga first.
        player._library_artist_names = lambda: []
        calls = []

        def search(q, limit=5, type="artist"):
            calls.append(limit)
            items = [{"name": "BK", "uri": "u:bk"}] if limit == 1 else \
                [{"name": "Djonga", "uri": "u:dj"}, {"name": "BK", "uri": "u:bk"}]
            return {"artists": {"items": items}, "tracks": {"items": []}}
        player._client.search = MagicMock(side_effect=search)
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        player.play_anything("djonga")
        assert 1 not in calls
        assert player._client.start_playback.call_args.kwargs["context_uri"] == "u:dj"

    def test_artist_name_without_spaces_still_matches(self, player):
        player._client.search = MagicMock(return_value={
            "artists": {"items": [{"name": "Bad Bunny", "uri": "u:bb"}]}})
        assert player._search_artist("BadBunny")["uri"] == "u:bb"

    def _wire_bare(self, player, artist, top_track, library):
        player._library_artist_names = lambda: list(library)

        def search(q, limit=5, type="track"):
            if type == "artist":
                return {"artists": {"items": [{"name": artist, "uri": "u:artist"}]}}
            return {"tracks": {"items": [self._track(top_track, "Someone", "u:track")]}}
        player._client.search = MagicMock(side_effect=search)
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)

    def test_bare_word_that_is_a_top_song_plays_the_song(self, player):
        # "daisies": a band named Daisies exists, but the top song is DAISIES.
        self._wire_bare(player, "Daisies", "DAISIES", library=("Justin Bieber",))
        player.play_anything("daisies")
        assert player._client.start_playback.call_args.kwargs["uris"] == ["u:track"]

    def test_bare_artist_name_still_opens_the_artist(self, player):
        # "queen": top song is "Bohemian Rhapsody", not a song called Queen.
        self._wire_bare(player, "Queen", "Bohemian Rhapsody", library=())
        player.play_anything("queen")
        assert player._client.start_playback.call_args.kwargs["context_uri"] == "u:artist"

    def test_artist_in_the_library_always_wins(self, player):
        self._wire_bare(player, "Djonga", "Djonga", library=("Djonga",))
        player.play_anything("djonga")
        assert player._client.start_playback.call_args.kwargs["context_uri"] == "u:artist"

    def test_the_pick_is_audited(self, player, monkeypatch):
        from src.core import audit as audit_mod
        events = []
        monkeypatch.setattr(audit_mod.audit, "log", lambda *a, **k: events.append((a, k)))
        self._wire(player, {
            'track:"daisies" artist:"Justin Bieber"': [
                self._track("DAISIES", "Justin Bieber", "spotify:track:daisies")],
        })
        player.play_anything("daisies by Justin Bieber")
        (args, kw), = [e for e in events if e[0][1:3] == ("spotify", "track_pick")]
        assert kw["payload"]["picked"] == "DAISIES — Justin Bieber"


# =====================================================================
# play_playlist — user's own first, public fallback
# =====================================================================

class TestPlayPlaylist:

    def test_picks_user_playlist_by_name(self, player):
        # User has a "Gym" playlist; query "gym" finds it on first pass
        # without falling back to public search.
        player._client.current_user_playlists = MagicMock(return_value={
            "items": [
                {"name": "Chill", "uri": "spotify:playlist:c1"},
                {"name": "Gym", "uri": "spotify:playlist:g1"},
            ]
        })
        player._client.search = MagicMock()
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_playlist("gym")
        assert "Gym" in result
        called = player._client.start_playback.call_args.kwargs
        assert called.get("context_uri") == "spotify:playlist:g1"
        # Public search NOT called when user owns a match.
        player._client.search.assert_not_called()

    def test_falls_back_to_public_when_no_user_match(self, player):
        player._client.current_user_playlists = MagicMock(return_value={
            "items": [{"name": "Chill", "uri": "spotify:playlist:c1"}],
        })
        player._client.search = MagicMock(return_value={
            "playlists": {"items": [
                {"name": "Rock Classics", "uri": "spotify:playlist:r1"},
            ]},
        })
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_playlist("rock classics")
        assert "Rock Classics" in result
        called = player._client.start_playback.call_args.kwargs
        assert called.get("context_uri") == "spotify:playlist:r1"

    def test_no_match_returns_friendly_message(self, player):
        player._client.current_user_playlists = MagicMock(return_value={"items": []})
        player._client.search = MagicMock(return_value={
            "playlists": {"items": []},
        })
        result = player.play_playlist("nothing matches")
        assert "Não achei" in result
        assert "nothing matches" in result


# =====================================================================
# play_artist_radio — recommendations mix, artist-context fallback
# =====================================================================

class TestPlayArtistRadio:

    def test_plays_recommendation_uris(self, player):
        player._client.search = MagicMock(return_value={
            "artists": {"items": [{"name": "Matuê", "id": "a1", "uri": "spotify:artist:a1"}]},
        })
        player._client.recommendations = MagicMock(return_value={
            "tracks": [{"uri": "spotify:track:t1"}, {"uri": "spotify:track:t2"}],
        })
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_artist_radio("matue")
        assert "rádio de Matuê" in result
        called = player._client.start_playback.call_args.kwargs
        assert called.get("uris") == ["spotify:track:t1", "spotify:track:t2"]
        player._client.recommendations.assert_called_once_with(
            seed_artists=["a1"], limit=50,
        )

    def test_falls_back_to_artist_context_when_recs_unavailable(self, player):
        # recommendations endpoint restricted → SpotifyException → []
        player._client.search = MagicMock(return_value={
            "artists": {"items": [{"name": "Djonga", "id": "a2", "uri": "spotify:artist:a2"}]},
        })
        player._client.recommendations = MagicMock(side_effect=_spotify_404())
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_artist_radio("djonga")
        # Still a radio reply, now seeded on the artist context URI.
        assert "rádio de Djonga" in result
        called = player._client.start_playback.call_args.kwargs
        assert called.get("context_uri") == "spotify:artist:a2"
        assert "uris" not in called

    def test_no_artist_returns_friendly_message(self, player):
        player._client.search = MagicMock(return_value={"artists": {"items": []}})
        result = player.play_artist_radio("zzzznotanartist")
        assert "Não achei" in result
        assert "zzzznotanartist" in result


# =====================================================================
# PT-BR artist name correction — fuzzy-match STT output vs the library
# =====================================================================

def _saved_tracks(*artist_names):
    """Shape a current_user_saved_tracks payload from artist names."""
    return {"items": [
        {"track": {"artists": [{"name": n}]}} for n in artist_names
    ]}


class TestNameCorrection:

    def test_library_names_extracted_unique(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Matuê", "Djonga", "Matuê", "Anitta")
        )
        names = player._library_artist_names()
        # de-duped, order preserved
        assert names == ["Matuê", "Djonga", "Anitta"]

    def test_library_names_cached(self, player):
        m = MagicMock(return_value=_saved_tracks("Matuê"))
        player._client.current_user_saved_tracks = m
        player._library_artist_names()
        player._library_artist_names()
        m.assert_called_once()  # second call served from cache

    def test_library_fetch_failure_returns_empty_not_crash(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            side_effect=_spotify_404()
        )
        assert player._library_artist_names() == []

    def test_corrects_mangled_name(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Matuê", "Djonga")
        )
        # Whisper-style mangle of "Matuê"
        assert player._correct_artist_query("matue") == "Matuê"

    def test_leaves_correct_query_untouched_when_no_close_match(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Matuê", "Djonga")
        )
        # A real, different artist not in the library must pass through.
        assert player._correct_artist_query("radiohead") == "radiohead"

    def test_empty_library_passes_query_through(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks()
        )
        assert player._correct_artist_query("matue") == "matue"

    def test_track_phrase_not_rewritten(self, player):
        # "X de Y" is far longer than any artist name → no fuzzy match,
        # so play_anything's track-by-artist branch stays intact.
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Matuê")
        )
        q = "fim da noite do matuê"
        assert player._correct_artist_query(q) == q

    def test_play_track_searches_corrected_name(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Matuê")
        )
        player._client.search = MagicMock(return_value={
            "tracks": {"items": [
                {"name": "Kenny G", "uri": "spotify:track:t1",
                 "artists": [{"name": "Matuê"}]},
            ]},
        })
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        player.play_track("matue")
        # The Spotify search ran with the corrected name, not the mangle.
        assert player._client.search.call_args.kwargs["q"] == "Matuê"

    def test_play_anything_corrects_before_artist_search(self, player):
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Djonga")
        )
        player._client.search = MagicMock(return_value={
            "artists": {"items": [
                {"name": "Djonga", "id": "a1", "uri": "spotify:artist:a1"},
            ]},
        })
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_anything("jonga")
        assert player._client.search.call_args.kwargs["q"] == "Djonga"
        assert "Djonga" in result

    def test_play_artist_radio_corrects_before_artist_search(self, player):
        # s59: artist radio is artist-seeded, so it now applies the same
        # PT-BR fuzzy correction — "rádio do jonga" → searches "Djonga".
        player._client.current_user_saved_tracks = MagicMock(
            return_value=_saved_tracks("Djonga")
        )
        player._client.search = MagicMock(return_value={
            "artists": {"items": [
                {"name": "Djonga", "id": "a1", "uri": "spotify:artist:a1"},
            ]},
        })
        player._client.recommendations = MagicMock(return_value={"tracks": []})
        player._client.start_playback = MagicMock()
        player._wake_any_device = MagicMock(return_value=None)
        result = player.play_artist_radio("jonga")
        assert player._client.search.call_args.kwargs["q"] == "Djonga"
        assert "rádio de Djonga" in result


class TestSpotifyGuardFailSoft:
    """M4-A1: spotipy raises more than SpotifyException — a Wi-Fi drop
    surfaces requests' ConnectionError, a failed refresh SpotifyOauthError.
    Every public method must answer with a speakable line, never raise
    (pre-fix, a spoken "pausa" on flaky network became an HTTP 500)."""

    # All 12 public methods with minimal valid args.
    _CALLS = [
        ("play_track", ("matue",)),
        ("play_anything", ("matue",)),
        ("play_artist_radio", ("matue",)),
        ("play_playlist", ("gym",)),
        ("transfer_playback_to", ("iphone",)),
        ("pause", ()),
        ("resume", ()),
        ("next_track", ()),
        ("previous_track", ()),
        ("now_playing", ()),
        ("set_volume", (50,)),
        ("like_current", ()),
    ]

    class _NetworkDownClient:
        """Every spotipy call raises ConnectionError — NOT a SpotifyException."""
        def __getattr__(self, name):
            def _raise(*_a, **_k):
                import requests
                raise requests.exceptions.ConnectionError(f"net down: {name}")
            return _raise

    @pytest.mark.parametrize("method,args", _CALLS,
                             ids=[m for m, _ in _CALLS])
    def test_network_down_is_speakable_pt(self, player, method, args):
        player._client = self._NetworkDownClient()
        reply = getattr(player, method)(*args)
        assert isinstance(reply, str)
        assert "Sem conexão com o Spotify" in reply

    def test_network_down_lang_kwarg_en(self, player):
        player._client = self._NetworkDownClient()
        assert "Can't reach Spotify" in player.pause(lang="en")

    def test_network_down_lang_positional_en(self, player):
        # The guard binds against the real signature, so positional lang works.
        player._client = self._NetworkDownClient()
        assert "Can't reach Spotify" in player.pause("en")

    def test_no_raw_exception_repr_in_reply(self, player):
        # The old play_* generic handlers spoke "Erro Spotify: <repr>" aloud.
        player._client = self._NetworkDownClient()
        reply = player.play_track("matue")
        assert "ConnectionError" not in reply
        assert "net down" not in reply

    def test_spotify_exception_still_maps_friendly(self, player):
        # The decorator must not swallow the specific SpotifyException
        # mapping: 403 still becomes the Premium message.
        player._client.pause_playback = MagicMock(side_effect=_spotify_403())
        player._wake_any_device = MagicMock(return_value=None)
        reply = player.pause()
        assert "Premium" in reply



class TestNoTerminalPromptWithoutAToken:
    """open_browser=False still fell back to input("Enter the URL you were
    redirected to:") with an empty token cache: EOFError in the daemon, a
    waiting terminal under `cortex` (s88, Codex lote 13)."""

    def _oauth(self, monkeypatch, tmp_path, interactive):
        from src.tools import spotify_player as sp
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "fake")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "fake")
        token = tmp_path / "spotify_token.json"
        token.write_text("{}")
        monkeypatch.setattr(sp, "_TOKEN_PATH", str(token))
        monkeypatch.setattr(sp, "_CREDS_DIR", str(tmp_path))
        if interactive:
            monkeypatch.setenv("CORTEX_SPOTIFY_OAUTH_INTERACTIVE", "1")
        else:
            monkeypatch.delenv("CORTEX_SPOTIFY_OAUTH_INTERACTIVE", raising=False)
        return sp._build_oauth()

    def test_an_empty_cache_fails_without_reading_the_terminal(self, monkeypatch, tmp_path):
        from spotipy.oauth2 import SpotifyOauthError

        def no_input(*a, **k):
            raise AssertionError("the daemon must never prompt in a terminal")
        monkeypatch.setattr("builtins.input", no_input)
        oauth = self._oauth(monkeypatch, tmp_path, interactive=False)
        with pytest.raises(SpotifyOauthError):
            oauth.get_access_token(as_dict=False)

    def test_the_setup_script_keeps_its_consent_flow(self, monkeypatch, tmp_path):
        from spotipy.oauth2 import SpotifyOAuth
        from src.tools import spotify_player as sp
        oauth = self._oauth(monkeypatch, tmp_path, interactive=True)
        assert type(oauth) is SpotifyOAuth and not isinstance(oauth, sp._NoPromptOAuth)
