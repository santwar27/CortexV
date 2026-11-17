"""
Spotify OAuth must never go interactive inside the daemon (M4-A2).

With `open_browser=True`, a deleted/corrupted/revoked token cache made the
first authenticated call start the consent flow: a surprise browser on the
Mac session and a voice turn blocked forever. The interactive flow belongs
to scripts/spotify_oauth_setup.py ONLY (it sets the opt-in env var);
`is_configured()` treats a missing token cache as not-configured so the
handlers answer with the setup instruction instead of ever reaching OAuth.
"""
from unittest.mock import patch

import pytest

from src.tools import spotify_player


@pytest.fixture
def creds_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test_cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test_secret")
    monkeypatch.delenv("CORTEX_SPOTIFY_OAUTH_INTERACTIVE", raising=False)
    monkeypatch.setattr(
        spotify_player, "_TOKEN_PATH", str(tmp_path / "spotify_token.json")
    )
    monkeypatch.setattr(spotify_player, "_CREDS_DIR", str(tmp_path))
    return tmp_path


class TestRuntimeNeverOpensBrowser:
    def test_build_oauth_defaults_to_no_browser(self, creds_env):
        # The daemon's manager is the no-prompt subclass (s88, Codex lote 13):
        # no browser, and no input() fallback either.
        with patch.object(spotify_player, "_NoPromptOAuth") as oauth_cls, \
                patch.object(spotify_player, "SpotifyOAuth") as interactive_cls:
            spotify_player._build_oauth()
        assert oauth_cls.call_args.kwargs["open_browser"] is False
        interactive_cls.assert_not_called()

    def test_setup_script_env_var_opts_in(self, creds_env, monkeypatch):
        # scripts/spotify_oauth_setup.py sets this before building the
        # client — the one sanctioned interactive path.
        monkeypatch.setenv("CORTEX_SPOTIFY_OAUTH_INTERACTIVE", "1")
        with patch.object(spotify_player, "SpotifyOAuth") as oauth_cls:
            spotify_player._build_oauth()
        assert oauth_cls.call_args.kwargs["open_browser"] is True


class TestTokenCacheGatesConfiguration:
    def test_creds_without_token_is_not_configured(self, creds_env):
        player = spotify_player.SpotifyPlayer()
        assert not player.is_configured(), (
            "env vars alone must not count as configured — the first call "
            "would start the OAuth flow inside the daemon"
        )

    def test_creds_with_token_is_configured(self, creds_env):
        (creds_env / "spotify_token.json").write_text("{}")
        player = spotify_player.SpotifyPlayer()
        assert player.is_configured()

    def test_missing_token_yields_setup_instruction_not_oauth(self, creds_env):
        # A real handler path: with creds but no token, the user hears the
        # setup instruction; spotipy/OAuth is never constructed.
        player = spotify_player.SpotifyPlayer()
        with patch.object(spotify_player, "SpotifyOAuth") as oauth_cls:
            reply = player.pause()
        oauth_cls.assert_not_called()
        assert "spotify_oauth_setup" in reply
