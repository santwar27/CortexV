"""
First-time Spotify OAuth setup for Cortex.

Run ONCE after setting SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.
This opens your browser to the Spotify authorization page; click "Agree"
and the redirect to http://127.0.0.1:8765/callback caches your token.

After this, spotipy handles all token refresh automatically — you should
never need to run this again unless the cache file is lost or revoked.

Usage:
    cd $HOME/cortex_project
    ./venv/bin/python scripts/spotify_oauth_setup.py
"""
import os
import sys

# Make src.* importable when this script is invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# This script is the ONE place allowed to run the interactive consent flow
# (browser + callback). The runtime default is open_browser=False so a
# missing/revoked token can never pop a browser inside the daemon (M4-A2).
os.environ["CORTEX_SPOTIFY_OAUTH_INTERACTIVE"] = "1"

from src.tools.spotify_player import SpotifyPlayer


def main() -> None:
    player = SpotifyPlayer()
    # Env vars only — is_configured() also requires the cached token, which
    # is exactly what this script exists to create.
    if not (os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")):
        print(
            "ERROR: Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Opening browser for Spotify authorization...")
    print("Click 'Agree' on the Spotify page; you can close the browser tab when it says success.")
    print()

    try:
        # First property access triggers the lazy OAuth flow (browser + callback).
        client = player.client
        me = client.current_user()
    except Exception as e:
        print(f"ERROR during OAuth: {e}", file=sys.stderr)
        sys.exit(1)

    name = me.get("display_name") or me.get("id") or "(unknown)"
    email = me.get("email") or "(email scope not granted)"
    product = me.get("product") or "(unknown)"

    print(f"✅ Authorized as: {name}")
    print(f"   Email   : {email}")
    print(f"   Product : {product}  {'(✅ Premium)' if product == 'premium' else '(⚠️  not Premium — playback control won''t work)'}")
    print(f"   Token   : data/credentials/spotify_token.json (auto-refreshes)")


if __name__ == "__main__":
    main()
