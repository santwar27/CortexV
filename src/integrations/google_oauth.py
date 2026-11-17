"""
Google OAuth helper for the Cortex Agent integrations.

Flow:
  - First-time setup: user runs scripts/google_oauth_setup.py which calls
    authorise() with the desired scopes. A local webserver starts on a
    random port, opens the browser, the user grants consent in Google's
    UI, and we receive an authorisation code. Token is persisted to
    data/credentials/google_token.json.
  - Subsequent runs: load_credentials() reads the stored token, transparently
    refreshes it if expired (using the long-lived refresh_token), and
    returns ready-to-use credentials.

Security:
  - data/credentials/ is gitignored.
  - The token grants only the scopes that were originally consented to.
    If new scopes are added later (e.g. send-email in Fase 8B), the user
    must re-run the consent flow.
"""

import logging
import os
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

log = logging.getLogger(__name__)

CREDENTIALS_DIR = "data/credentials"
CLIENT_SECRETS_PATH = os.path.join(CREDENTIALS_DIR, "google_oauth_client.json")
TOKEN_PATH = os.path.join(CREDENTIALS_DIR, "google_token.json")


# Scopes used in the Agent layer. Each new capability appends to this
# list; bumping it requires a fresh consent flow (run setup script
# again so the user re-approves in the browser).
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",  # full calendar (read+write)
]


def has_client_secrets() -> bool:
    """True if the Google Cloud OAuth client JSON has been placed locally."""
    return os.path.isfile(CLIENT_SECRETS_PATH)


def has_token() -> bool:
    """True if a stored user token exists on disk."""
    return os.path.isfile(TOKEN_PATH)


def authorise(scopes: Optional[List[str]] = None) -> Credentials:
    """
    Run the interactive OAuth flow (opens browser). Use this on first
    setup or when scopes change. Persists the token to disk and returns
    the fresh credentials.
    """
    scopes = scopes or DEFAULT_SCOPES
    if not has_client_secrets():
        raise FileNotFoundError(
            f"Missing {CLIENT_SECRETS_PATH}. Download the OAuth client JSON "
            f"from Google Cloud Console and place it there."
        )
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, scopes)
    # port=0 → an unused random local port (avoids conflicts with our 8000)
    creds = flow.run_local_server(port=0)
    _save(creds)
    log.info(f"OAuth: token saved to {TOKEN_PATH} for scopes {scopes}")
    return creds


def load_credentials(scopes: Optional[List[str]] = None) -> Optional[Credentials]:
    """
    Load credentials from disk; refresh if expired. Returns None if no
    token exists yet (caller should prompt user to run setup).
    """
    scopes = scopes or DEFAULT_SCOPES
    if not has_token():
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, scopes)
    except Exception as e:
        log.warning(f"OAuth: failed to load stored token: {e}")
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds)
            log.info("OAuth: token refreshed silently.")
        except Exception as e:
            log.warning(f"OAuth: refresh failed: {e}")
            return None
    return creds


def _save(creds: Credentials) -> None:
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    # Permissions: token is sensitive — restrict to user only.
    try:
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass
