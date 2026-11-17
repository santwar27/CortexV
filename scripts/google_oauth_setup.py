#!/usr/bin/env python3
"""
Google OAuth first-time setup for the Cortex Agent.

Prerequisites (done in the browser, ONE TIME):
  1. Go to https://console.cloud.google.com/
  2. Create (or pick) a project.
  3. APIs & Services → Enabled APIs → enable "Gmail API"
     (Calendar will be enabled later in Fase 8C).
  4. APIs & Services → OAuth consent screen:
       - User Type: External
       - App name: Cortex (or anything)
       - Add YOUR email as a "test user"
  5. APIs & Services → Credentials → Create Credentials →
       OAuth client ID → Application type: Desktop app.
  6. Download the JSON, save it as
       data/credentials/google_oauth_client.json

Then run this script:
    ./venv/bin/python scripts/google_oauth_setup.py

It opens your browser, you approve, and the resulting token lands at
data/credentials/google_token.json. From then on, the Cortex can read
your Gmail without further prompts (token auto-refreshes).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.google_oauth import (  # noqa: E402
    CLIENT_SECRETS_PATH,
    TOKEN_PATH,
    authorise,
    has_client_secrets,
    has_token,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("oauth_setup")

    if not has_client_secrets():
        print(
            f"\n❌ MISSING: {CLIENT_SECRETS_PATH}\n\n"
            "You need to create an OAuth client in Google Cloud Console "
            "first. See the docstring at the top of this script for the "
            "step-by-step.\n"
        )
        return 1

    if has_token():
        print(
            f"⚠️  An existing token already exists at {TOKEN_PATH}.\n"
            "    Running this will REPLACE it (re-consent in the browser).\n"
        )
        answer = input("    Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "s", "sim"):
            print("Aborted.")
            return 0

    log.info("Starting OAuth flow — your browser should open in a moment.")
    creds = authorise()
    log.info(f"✓ OAuth complete. Token persisted to {TOKEN_PATH}.")
    log.info(f"  Scopes granted: {list(creds.scopes or [])}")
    print(
        "\n🎉 Done. The Cortex can now read your Gmail. Try:\n"
        "   curl -X POST http://localhost:8000/ask \\\n"
        "       -H 'X-Cortex-Key: cortex-omega-2026' \\\n"
        "       -H 'Content-Type: application/json' \\\n"
        "       -d '{\"user_input\":\"meus emails\"}'\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
