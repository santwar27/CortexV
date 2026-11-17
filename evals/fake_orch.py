"""
Lightweight fake orchestrator for the DISPATCH_REPLY eval kind (s60).

The DISPATCH eval kind only checks ROUTING (simulate_dispatch, no execution).
To get a permanent regression shield over the spoken-language REPLY of the
deterministic command path (the s60 i18n work), we need to actually EXECUTE the
matched handler and read its reply text — but WITHOUT the real orchestrator's
cost/side-effects (Palace, Ollama, ChromaDB), mirroring why evals/dispatch.py
simulates rather than constructs.

This builds a tiny object that mixes ONLY the handler mixins whose localisation
lives in the mixin itself (notes, studier) plus the I18nMixin, backed by fixed
in-memory stubs. The controller-localised families (spotify/macos) and the pure
assemblers (digest/system_status/entities) are covered by deterministic pytest
unit tests instead — they need subprocess/network mocking the eval harness
doesn't provide.

One deliberate exception (M4-A1): `spotify` is a REAL SpotifyPlayer whose
spotipy client raises ConnectionError on every call — the "Wi-Fi dropped
mid-turn" world. Spotify DISPATCH_REPLY cases therefore exercise the real
route lambda → real player → `_spotify_guard` chain and pin that a spoken
"pausa" on a dead network yields a speakable reply, never an exception.

`run_reply(text, lang)` runs the real dispatch registry against the fake orch
and returns the executed (reply, mode), or None if nothing routed.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from src.core.dispatch import registry as dispatch_registry
from src.core.dispatch.flow_control import ALL_REF_RE
from src.core.i18n import I18nMixin
from src.core.notes_mixin import NotesMixin
from src.core.studier_mixin import StudierMixin
from src.core.weather_mixin import WeatherMixin


class _StubNotes:
    """Fixed two-note store; search is a case-insensitive title substring."""
    _NOTES = [{"title": "Projeto X"}, {"title": "Compras da semana"}]

    def list_notes(self, limit: int = 10) -> List[dict]:
        return list(self._NOTES)

    def search_notes(self, query: str, limit: int = 10) -> List[dict]:
        q = (query or "").lower()
        return [n for n in self._NOTES if q in n["title"].lower()]

    def create_note(self, title: str, body: str = "",
                    folder: Optional[str] = None) -> None:  # pragma: no cover
        return None

    def append_to_note(self, title: str, text: str) -> dict:  # pragma: no cover
        return {"title": title}


class _StubQueue:
    """Studier queue with one pending topic (so 'estude X' → '1 pending')."""
    def add(self, topic: str) -> dict:
        return {"topic": topic.lstrip("!").strip()}

    def list_pending(self) -> List[dict]:
        return [{"topic": "física quântica"}]


class _StubReview:
    """Studier review with NO pending briefings — exercises the deterministic
    empty/'not found' branches (both localised)."""
    def list_pending(self) -> List[dict]:
        return []

    def get_briefing(self, ref: str):
        return None  # nothing staged → handlers hit the localised 'not found'

    def approve(self, ref: str):
        return None

    def reject(self, ref: str):
        return None

    def approve_all(self) -> List[dict]:
        return []

    def reject_all(self) -> List[dict]:
        return []


class _NetworkDownClient:
    """Stands in for spotipy.Spotify with the network OUT: every API call
    raises requests' ConnectionError — which is NOT a SpotifyException, the
    exact class of failure M4-A1 guards against."""
    def __getattr__(self, name: str):
        def _raise(*_a, **_k):
            import requests
            raise requests.exceptions.ConnectionError(f"network down (eval): {name}")
        return _raise


def _network_down_spotify():
    """A REAL SpotifyPlayer that believes it's configured but whose client
    can't reach the network. No env vars or token involved."""
    import logging

    from src.tools.spotify_player import SpotifyPlayer

    # The guard's log.exception on the PLANNED ConnectionError would smear
    # tracebacks over the eval report. Eval-process only — production
    # logging is untouched.
    logging.getLogger("src.tools.spotify_player").setLevel(logging.CRITICAL)
    player = SpotifyPlayer()
    player._client = _NetworkDownClient()
    player.is_configured = lambda: True  # creds are fine; the NETWORK is out
    return player


class _StubGatekeeper:
    """No lease, ever — so the weather route exercises its OFFLINE degrade
    branch (the deterministic 'needs internet' offer), never the network."""
    def is_lease_active(self) -> bool:
        return False


class FakeOrch(NotesMixin, StudierMixin, WeatherMixin, I18nMixin):
    """Mixin composite + stubs. No Palace/Ollama/Chroma. Per-turn _reply_lang
    is set by run_reply()."""
    _ALL_REF_RE = ALL_REF_RE

    def __init__(self) -> None:
        self.notes = _StubNotes()
        self.study_queue = _StubQueue()
        self.studies_review = _StubReview()
        self.spotify = _network_down_spotify()
        self.gatekeeper = _StubGatekeeper()
        self.lease_hours = 4
        self._pending_query = None
        self._pending_engine = None


def run_reply(text: str, lang: str = "pt") -> Optional[Tuple[str, str]]:
    """Execute the real dispatch registry against a fresh fake orch with
    `_reply_lang` forced. Returns the matched handler's (reply, mode) or None."""
    orch: Any = FakeOrch()
    orch._reply_lang = lang
    return dispatch_registry.run(orch, text)
