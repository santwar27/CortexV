"""Read-only web dashboard at /ui.

The dashboard renders snapshot views of Cortex state — lease, recent
audit events, Studier queue, launchd services — so Pedro has visual
visibility without polling the chat. See `routes.py` for the FastAPI
router and `aggregator.py` for the data sources.

Auth model: same X-Cortex-Key as /ask, but tolerant to browser flow.
Browsers can't easily send custom headers when navigating, so the
helper `verify_ui_access` also accepts a `cortex_ui` cookie set by
the sign-in form at /ui/login (POST; the key never rides in the URL). IP allowlist still applies (middleware runs
ahead of any route).
"""
