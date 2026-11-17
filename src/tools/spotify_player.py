"""
Spotify Web API wrapper for Cortex voice/text commands.

Strategy: regex-first dispatch in the orchestrator (zero LLM cost for
common commands like "pausa", "próxima", "volume 50"), with the LLM
tool-call fallback reserved for complex queries that need disambiguation.

Auth: spotipy handles OAuth 2.0 + token refresh transparently. First-time
setup runs scripts/spotify_oauth_setup.py once to open the browser and
cache the token to data/credentials/spotify_token.json (gitignored).

Prerequisites for end-user actions:
- SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET in .env
- Spotify Premium (required by API for playback control)
- An active device (desktop/mobile/web Spotify client open)
"""
import difflib
import functools
import inspect
import logging
import os
import re
import time
import unicodedata
from typing import List, Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError

from src.core.i18n import t


# Detects the "track BY artist" phrasing — "fim da noite do matuê",
# "redemption song by marley". Triggers track-mode in play_anything so
# the user's specific request wins over the artist-radio default.
_TRACK_BY_ARTIST_RE = re.compile(r"\s+(?:de|do|da|by)\s+", re.IGNORECASE)

# Versions nobody means when they name a song (s84): the literal search for
# "daisies by Justin Bieber" ranked an instrumental karaoke DAISIES first,
# and the next five held two more karaoke tracks. Skipped unless the request
# itself asks for one ("toca a versão karaokê de...").
_DERIVATIVE_RE = re.compile(
    r"\b(?:karaok[eê]|instrumental|cover|tribute|in the style of|"
    r"originally performed|made famous|lullaby|8[- ]?bit)\b",
    re.IGNORECASE,
)


def _split_track_artist(query: str):
    """("title", "artist") from "title by/de/do/da artist", split on the LAST
    separator so a title that has one survives ("garota de ipanema de tom
    jobim"); None when there is no separator or a side is empty."""
    seps = list(_TRACK_BY_ARTIST_RE.finditer(query or ""))
    if not seps:
        return None
    m = seps[-1]
    title, artist = query[:m.start()].strip(), query[m.end():].strip()
    return (title, artist) if title and artist else None

log = logging.getLogger(__name__)

# spotipy logs every HTTP error at WARNING before raising the exception.
# Those 404s ("No active device found") are already handled by _wake_any_device
# retry logic — leaking them to the console creates false alarms that look
# like unhandled failures when the retry actually succeeds.
logging.getLogger("spotipy").setLevel(logging.ERROR)
logging.getLogger("spotipy.client").setLevel(logging.ERROR)

_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-library-modify",
    "user-library-read",
    # Required by current_user_playlists for the "toca minha playlist X"
    # path. Without this, only public/owned-as-public playlists show up,
    # and the user's private "Gym" / "Sertanejo" lists become invisible.
    # Adding this scope invalidates the cached token — re-run
    # scripts/spotify_oauth_setup.py once after deploy to re-authorise.
    "playlist-read-private",
    "playlist-read-collaborative",
])

_REDIRECT_URI = "http://127.0.0.1:8765/callback"
_CREDS_DIR = "data/credentials"
_TOKEN_PATH = os.path.join(_CREDS_DIR, "spotify_token.json")


def _spotify_guard(method):
    """Fail-soft contract for every public SpotifyPlayer action (M4-A1).

    spotipy raises more than SpotifyException: a Wi-Fi drop mid-turn surfaces
    requests' ConnectionError/ReadTimeout, and a failed token refresh raises
    SpotifyOauthError — none of them subclasses of SpotifyException, so they
    escaped the per-method handlers and a spoken "pausa" on flaky network
    became an HTTP 500 / dead SSE turn. SpotifyException keeps the specific
    `_friendly_error` mapping; anything else becomes a speakable
    "can't reach Spotify" line, with the traceback preserved in the log.
    """
    sig = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        def _lang() -> str:
            # `lang` may arrive positionally (pause("en")) or as a kwarg;
            # bind against the real signature instead of guessing.
            try:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                return bound.arguments.get("lang", "pt")
            except TypeError:
                return "pt"

        try:
            return method(self, *args, **kwargs)
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, _lang())
        except Exception:
            log.exception(f"SpotifyPlayer.{method.__name__} failed (non-Spotify error)")
            return t(
                _lang(),
                "Sem conexão com o Spotify agora. Tenta de novo em instantes.",
                "Can't reach Spotify right now. Try again in a moment.",
            )

    return wrapper


class _NoPromptOAuth(SpotifyOAuth):
    """SpotifyOAuth that never asks for the redirect URL. open_browser=False
    alone still falls back to input("Enter the URL you were redirected to:")
    when the cache holds no token (an empty or damaged spotify_token.json):
    EOFError inside the daemon, and in a `cortex` terminal a turn that waits
    for typing (s88, Codex lote 13 — a doubt confirmed with an empty cache).
    The consent flow belongs to scripts/spotify_oauth_setup.py only."""

    def get_auth_response(self, open_browser=None):
        raise SpotifyOauthError(
            "Spotify token missing or invalid — run scripts/spotify_oauth_setup.py")


def _build_oauth() -> SpotifyOAuth:
    cid = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise RuntimeError(
            "Spotify credentials missing. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env (see .env.example)."
        )
    os.makedirs(_CREDS_DIR, exist_ok=True)
    interactive = os.getenv("CORTEX_SPOTIFY_OAUTH_INTERACTIVE") == "1"
    oauth_cls = SpotifyOAuth if interactive else _NoPromptOAuth
    return oauth_cls(
        client_id=cid,
        client_secret=secret,
        redirect_uri=_REDIRECT_URI,
        scope=_SCOPES,
        cache_path=_TOKEN_PATH,
        # The interactive consent flow belongs to scripts/spotify_oauth_setup.py
        # ONLY (it sets the env var). Inside the uvicorn daemon a missing or
        # revoked token must fail fast — open_browser=True popped a surprise
        # browser on the Mac session and blocked the voice turn forever
        # waiting for consent (M4-A2, review Fable 5).
        open_browser=interactive,
    )


class SpotifyPlayer:
    """Lazy-loaded Spotify client. Safe to instantiate without credentials."""

    _NOW_PLAYING_CACHE_TTL_S = 3.0
    # The user's library (artist names) changes slowly and the lookup is
    # on the play hot-path, so cache it. Refreshed past this TTL.
    _LIBRARY_CACHE_TTL_S = 3600.0
    # difflib ratio floor for accepting a library name as the "real" one
    # behind an STT-mangled query. Deliberately high: a correctly-heard
    # name that doesn't closely match anything in the library must pass
    # through unchanged, never get rewritten to a similar-but-wrong name.
    _NAME_MATCH_CUTOFF = 0.7

    def __init__(self):
        self._client: Optional[spotipy.Spotify] = None
        self._now_cache_msg: Optional[str] = None
        self._now_cache_at: float = 0.0
        # The cached message is a LOCALISED string, so it is only valid
        # for the language it was built in (M4-A14d): an EN turn inside
        # the TTL must not be served a PT message cached by a voice turn
        # seconds earlier (and vice versa). Lang mismatch = refetch.
        self._now_cache_lang: Optional[str] = None
        self._library_names: Optional[List[str]] = None
        self._library_names_at: float = 0.0

    @property
    def client(self) -> spotipy.Spotify:
        if self._client is None:
            self._client = spotipy.Spotify(auth_manager=_build_oauth())
        return self._client

    def is_configured(self) -> bool:
        # The token-cache check matters as much as the env vars: with creds
        # set but no cached token, the first authenticated call would start
        # the OAuth consent flow inside the daemon (M4-A2). _need_creds_msg
        # already points at scripts/spotify_oauth_setup.py, which is the one
        # place allowed to run that flow.
        return (
            bool(os.getenv("SPOTIFY_CLIENT_ID"))
            and bool(os.getenv("SPOTIFY_CLIENT_SECRET"))
            and os.path.exists(_TOKEN_PATH)
        )

    @staticmethod
    def _need_creds_msg(lang: str = "pt") -> str:
        return t(
            lang,
            "Spotify não configurado. Defina SPOTIFY_CLIENT_ID e "
            "SPOTIFY_CLIENT_SECRET no .env e rode "
            "scripts/spotify_oauth_setup.py uma vez.",
            "Spotify isn't configured. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env and run "
            "scripts/spotify_oauth_setup.py once.",
        )

    @staticmethod
    def _friendly_error(e: spotipy.SpotifyException, lang: str = "pt") -> str:
        status = getattr(e, "http_status", 0)
        if status == 404:
            return t(
                lang,
                "Nenhum dispositivo Spotify ativo. Abre o Spotify no Mac, celular ou web e tenta de novo.",
                "No active Spotify device. Open Spotify on your Mac, phone or the web and try again.",
            )
        if status == 403:
            return t(lang, "Esse comando requer Spotify Premium.",
                     "This command requires Spotify Premium.")
        if status == 401:
            return t(lang, "Token Spotify expirou. Roda scripts/spotify_oauth_setup.py de novo.",
                     "Spotify token expired. Run scripts/spotify_oauth_setup.py again.")
        return t(lang, f"Erro Spotify ({status}): {getattr(e, 'msg', e)}",
                 f"Spotify error ({status}): {getattr(e, 'msg', e)}")

    # Maps user-spoken hint words to Spotify device type categories.
    # Spotify reports device["type"] as 'Computer', 'Smartphone', 'Speaker',
    # 'TV', 'AVR', 'STB', 'AudioDongle', 'GameConsole', 'CastVideo',
    # 'CastAudio', 'Automobile', 'Unknown'.
    _HINT_TYPE_MAP = {
        "iphone": "smartphone", "celular": "smartphone",
        "telefone": "smartphone", "phone": "smartphone",
        "mac": "computer", "macbook": "computer",
        "computador": "computer", "computer": "computer",
        "laptop": "computer",
        "tv": "tv",
        "caixinha": "speaker", "speaker": "speaker",
        "alto-falante": "speaker", "alto falante": "speaker",
        "homepod": "speaker", "home pod": "speaker",
        "fone": "audiodongle", "fones": "audiodongle",
        "fones de ouvido": "audiodongle",
        "headphone": "audiodongle", "headphones": "audiodongle",
    }

    def _pick_device(self, devices: list, hint: Optional[str]) -> Optional[dict]:
        """
        Choose the best device from the available list. If a hint is given,
        prefer matching by type, then by name substring. Otherwise return
        the first non-restricted device.
        """
        if not devices:
            return None
        if hint:
            target_type = self._HINT_TYPE_MAP.get(hint.lower())
            if target_type:
                for d in devices:
                    if (d.get("type") or "").lower() == target_type:
                        return d
            # Fallback: substring match against the device name (catches
            # custom-named devices like "the user's iPhone").
            for d in devices:
                if hint.lower() in (d.get("name") or "").lower():
                    return d
            # Hint didn't match any device — fall through to default.
        return next((d for d in devices if not d.get("is_restricted")), devices[0])

    def _wake_any_device(self, hint: Optional[str] = None) -> Optional[str]:
        """
        If no device matching the hint is currently active, transfer
        playback (without forcing play). Returns the device_id woken up,
        or None if nothing was available. Used both as a 404 fallback in
        start_playback and proactively when the user names a specific
        device ('no meu iphone').
        """
        try:
            payload = self.client.devices()
        except Exception:
            return None
        devices = (payload or {}).get("devices") or []
        if not devices:
            return None
        picked = self._pick_device(devices, hint)
        if not picked:
            return None
        device_id = picked.get("id")
        if not device_id:
            return None
        # If the picked device is already active AND there's no hint
        # contradiction, no transfer needed.
        if picked.get("is_active"):
            return device_id
        try:
            self.client.transfer_playback(device_id=device_id, force_play=False)
            # Spotify needs a beat to actually transfer state.
            time.sleep(0.6)
            return device_id
        except spotipy.SpotifyException:
            return None

    # ---------- ACTIONS ----------

    def _with_wake_on_404(self, action, *args, **kwargs) -> None:
        """Run `action(*args, **kwargs)`. If Spotify replies 404 ("no
        active device"), wake any available device and retry once.

        Re-raises any non-404 SpotifyException so callers can map it
        with `_friendly_error`. Used by pause / next / previous /
        volume which previously surfaced a generic 404 instead of
        transferring playback to an available device.
        """
        try:
            action(*args, **kwargs)
            return
        except spotipy.SpotifyException as e:
            if getattr(e, "http_status", 0) != 404:
                raise
        # 404 path — wake a device and retry once.
        device_id = self._wake_any_device()
        if not device_id:
            raise spotipy.SpotifyException(
                http_status=404,
                code=-1,
                msg="No active device available even after wake attempt",
            )
        action(*args, **kwargs)

    def _start_playback_with_fallback(
        self,
        *,
        device_hint: Optional[str] = None,
        uri: Optional[str] = None,
        uris: Optional[list] = None,
    ) -> None:
        """Common play path: optional target-device wake, 404 retry on
        any available device. Pass `uri` for context (album/artist/
        playlist; Spotify autoplays the whole context) OR `uris` for a
        single-track list. Raises spotipy.SpotifyException if no device
        is available even after the wake attempt — callers map with
        `_friendly_error`."""
        kwargs = {}
        if uri is not None:
            kwargs["context_uri"] = uri
        if uris is not None:
            kwargs["uris"] = uris
        target_id = self._wake_any_device(hint=device_hint) if device_hint else None
        try:
            if target_id:
                self.client.start_playback(device_id=target_id, **kwargs)
            else:
                self.client.start_playback(**kwargs)
            return
        except spotipy.SpotifyException as e:
            if getattr(e, "http_status", 0) != 404:
                raise
        # 404 = no active device. Wake one and retry once.
        device_id = self._wake_any_device(hint=device_hint)
        if not device_id:
            raise spotipy.SpotifyException(
                http_status=404,
                code=-1,
                msg="No active device available even after wake attempt",
            )
        self.client.start_playback(device_id=device_id, **kwargs)

    def _library_artist_names(self) -> List[str]:
        """Artist names pulled from the user's saved tracks. Used to
        repair STT-mangled names ("matwe" → "Matuê") — the artists the
        user actually listens to are the ones worth correcting toward.

        Uses `user-library-read` (already granted; no re-auth). Cached
        `_LIBRARY_CACHE_TTL_S` since it's on the play hot-path and the
        library moves slowly. A fetch failure returns the stale cache
        (or []) without poisoning it — it'll retry next window."""
        now = time.time()
        if (
            self._library_names is not None
            and (now - self._library_names_at) < self._LIBRARY_CACHE_TTL_S
        ):
            return self._library_names
        try:
            saved = self.client.current_user_saved_tracks(limit=50) or {}
        except Exception:
            log.exception("library fetch for name correction failed")
            return self._library_names or []
        seen: set = set()
        names: List[str] = []
        for item in saved.get("items") or []:
            track = item.get("track") or {}
            for artist in track.get("artists") or []:
                name = (artist.get("name") or "").strip()
                key = name.lower()
                if name and key not in seen:
                    seen.add(key)
                    names.append(name)
        self._library_names = names
        self._library_names_at = now
        return names

    @staticmethod
    def _normalize_name(s: str) -> str:
        """Casefold + strip diacritics for matching. Crucial here: STT
        drops the accent ("Matuê"→"matue"), and difflib is case- and
        accent-sensitive, so "matue" vs "Matuê" scores only 0.6 raw but
        1.0 normalized. We match on normalized forms, return the original."""
        decomposed = unicodedata.normalize("NFKD", s)
        return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()

    def _correct_artist_query(self, query: str) -> str:
        """If `query` closely matches an artist in the user's library,
        return that canonical name; otherwise the query unchanged.

        This is the PT-BR-name fix: Whisper hears Brazilian artist names
        "in English" ("Matuê"→"matue", "Djonga"→"jonga"), and the raw
        text then misses on Spotify search. Matching against the library
        recovers the real name. Conservative cutoff (`_NAME_MATCH_CUTOFF`)
        so a correctly-heard name is never rewritten."""
        q = (query or "").strip()
        if not q:
            return query
        names = self._library_artist_names()
        if not names:
            return query
        # Map normalized form → canonical (first occurrence wins).
        norm_to_orig: dict = {}
        for n in names:
            norm_to_orig.setdefault(self._normalize_name(n), n)
        qn = self._normalize_name(q)
        match = difflib.get_close_matches(
            qn, list(norm_to_orig.keys()), n=1, cutoff=self._NAME_MATCH_CUTOFF
        )
        if not match:
            return query
        canonical = norm_to_orig[match[0]]
        if self._normalize_name(canonical) != qn:
            log.info("spotify: corrected query %r → %r via library", q, canonical)
        return canonical

    @_spotify_guard
    def play_track(self, query: str, device_hint: Optional[str] = None,
                   lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        query = self._correct_artist_query(query)
        return self._play_track_corrected(query, device_hint=device_hint,
                                          lang=lang)

    @_spotify_guard
    def play_with_title_candidate(self, query: str, title_candidate: str,
                                 device_hint: Optional[str] = None,
                                 lang: str = "pt") -> str:
        """Probe an ambiguous courtesy tail once, then use the usual play path.

        Require the whole normalized base title, ignoring only version tails
        after " - " and trailing parenthesized or square-bracketed parts.
        Play the returned URI directly so an exact hit needs no second search.
        """
        if self.is_configured():
            try:
                tracks = self._search_tracks(title_candidate)
            except Exception as exc:
                log.warning("Spotify title candidate search failed: %s", type(exc).__name__)
            else:
                wanted = self._normalize_name(title_candidate)
                for track in tracks:
                    base = (track.get("name") or "").split(" - ", 1)[0]
                    base = re.sub(r"(?:\s*\([^()]*\)|\s*\[[^\[\]]*\])+\s*$", "", base)
                    if self._normalize_name(base) == wanted:
                        self._audit_pick(title_candidate, None, title_candidate, track)
                        return self._play_track_uri(track, device_hint, lang)
        return self.play_anything(query, device_hint=device_hint, lang=lang)

    def _search_tracks(self, q: str, limit: int = 5) -> list:
        result = self.client.search(q=q, limit=limit, type="track")
        return (result.get("tracks") or {}).get("items") or []

    @classmethod
    def _wanted(cls, track: dict, title: Optional[str] = None,
                artist: Optional[str] = None, allow_derivative: bool = False) -> bool:
        """Is `track` the song asked for? Title starts with the requested
        title (so "DAISIES - Remastered" counts), an artist matches, and it
        isn't a karaoke/instrumental/cover version unless that was asked."""
        name = track.get("name") or ""
        if not allow_derivative and _DERIVATIVE_RE.search(name):
            return False
        if title is not None:
            tn, nn = cls._normalize_name(title).strip(), cls._normalize_name(name).strip()
            if allow_derivative:
                # "daisies karaoke": the version word isn't part of the song's
                # title — look for the rest anywhere in the name.
                tn = " ".join(_DERIVATIVE_RE.sub(" ", tn).split())
                if not re.search(r"\b" + re.escape(tn) + r"\b", nn):
                    return False
            elif not (nn == tn or re.match(re.escape(tn) + r"\b", nn)):
                return False
        if artist is not None:
            an = cls._normalize_name(artist).strip()
            names = [cls._normalize_name(a.get("name") or "") for a in track.get("artists") or []]
            if not any(an == n or an in n or n in an for n in names if n):
                return False
        return True

    def _play_track_uri(self, track: dict, device_hint: Optional[str], lang: str) -> str:
        self._start_playback_with_fallback(uris=[track["uri"]], device_hint=device_hint)
        self._now_cache_msg = None  # invalidate cache
        artist = track["artists"][0]["name"] if track.get("artists") else "?"
        return t(lang, f"Tocando '{track['name']}' de {artist}.",
                 f"Playing '{track['name']}' by {artist}.")

    @staticmethod
    def _audit_pick(title: Optional[str], artist: Optional[str], query: str,
                    track: Optional[dict]) -> None:
        """What was searched and what was picked — the turn's audit only said
        SPOTIFY_PLAY, so "it played another song" couldn't be traced (s84)."""
        try:
            from src.core.audit import audit
            picked = None
            if track:
                picked = f"{track.get('name')} — " + ", ".join(
                    a.get("name", "") for a in track.get("artists") or [])
            audit.log("–", "spotify", "track_pick", payload={
                "title": title, "artist": artist, "query": query, "picked": picked,
            })
        except Exception:  # noqa: BLE001 — telemetry must never break playback
            pass

    _TITLE_MATCH_CUTOFF = 0.72
    _LEADING_ARTICLE_RE = re.compile(r"^(?:a|o|as|os|um|uma|the)\s+")

    @classmethod
    def _title_key(cls, name: str) -> str:
        """Base title for fuzzy comparison: no " - Ao Vivo"/"(feat. …)" tail,
        no leading article, no punctuation."""
        base = re.split(r"\s+[-(\[]", name or "")[0]
        base = " ".join(re.sub(r"[^a-z0-9]+", " ", cls._normalize_name(base)).split())
        return cls._LEADING_ARTICLE_RE.sub("", base)

    def _closest_title(self, tracks: list, title: str, artist: str,
                       allow: bool) -> Optional[dict]:
        """The artist's song whose title is closest to a mangled `title`, if
        close enough. Measured (s84): "Super-Ess" vs "A Super Ex" scores 0.82
        once the article goes; wrong songs stayed under 0.48."""
        want = self._title_key(title)
        best, best_r = None, 0.0
        for x in tracks:
            if not self._wanted(x, artist=artist, allow_derivative=allow):
                continue
            r = difflib.SequenceMatcher(None, want, self._title_key(x.get("name"))).ratio()
            if r > best_r:
                best, best_r = x, r
        return best if best_r >= self._TITLE_MATCH_CUTOFF else None

    def _search_artist(self, name: str, fallback_first: bool = False) -> Optional[dict]:
        """The artist whose name matches `name`, among Spotify's first five.

        Never limit=1 (s84, measured live): Spotify ranks differently by page
        size, and for "djonga" limit=1 returned "BK" while limit≥2 put Djonga
        first. "toca djonga" fell through to a single track, and "rádio do
        djonga" would have played BK's radio. Also: the API rejects limit>10.
        Loose match — equal, or one contains the other, also with spaces
        dropped ("BadBunny" is "Bad Bunny"). `fallback_first` keeps the old
        artist-radio behaviour: with no name match, the top hit."""
        res = self.client.search(q=name, limit=5, type="artist")
        items = (res.get("artists") or {}).get("items") or []
        q = self._normalize_name(name or "").strip()
        qs = q.replace(" ", "")
        for a in items:
            n = self._normalize_name(a.get("name") or "").strip()
            ns = n.replace(" ", "")
            if n and (n == q or n in q or q in n or ns == qs):
                return a
        return items[0] if (fallback_first and items) else None

    def _in_library(self, artist: dict) -> bool:
        n = self._normalize_name(artist.get("name") or "").strip()
        return any(self._normalize_name(x).strip() == n for x in self._library_artist_names())

    def _is_real_artist(self, artist: str) -> bool:
        """In the user's library, or Spotify's top artist hit for the name."""
        an = self._normalize_name(artist).strip()
        if any(self._normalize_name(n).strip() == an for n in self._library_artist_names()):
            return True
        try:
            return self._search_artist(artist) is not None
        except spotipy.SpotifyException:
            return False

    def _play_track_by_artist(self, title: str, artist: str, original: str,
                              device_hint: Optional[str] = None,
                              lang: str = "pt") -> str:
        """Play `title` by `artist` (s84). Field search first — for
        "daisies"/"Justin Bieber" it returns the original DAISIES five times,
        where the literal phrase returned karaoke first. If the field search
        finds nothing at all, the split was probably wrong ("música de
        ninar" has no artist), so the whole phrase is searched as before. If
        it finds songs but none is the title asked for, Cortex says so
        instead of playing a different one."""
        allow = bool(_DERIVATIVE_RE.search(original))
        try:
            q = f'track:"{title}" artist:"{artist}"'
            pick = next((x for x in self._search_tracks(q)
                         if self._wanted(x, title, artist, allow)), None)
            if pick is None:
                q = f"{title} {artist}"
                pick = next((x for x in self._search_tracks(q)
                             if self._wanted(x, title, artist, allow)), None)
            if pick is None and self._is_real_artist(artist):
                # The artist exists, so the title is what went wrong — most
                # often the transcript ("Super-Ess" for "Super Ex"). Look for
                # it among the artist's songs; never fall back to the whole
                # phrase, which plays whatever ranks first (s84: "fim da
                # noite do matuê" → "Kenny G").
                q = f'artist:"{artist}"'
                pick = self._closest_title(self._search_tracks(q, limit=10),
                                           title, artist, allow)
            elif pick is None:
                # No such artist: the split was probably wrong ("música de
                # ninar" has none). Search the phrase whole, as before.
                q = original
                pick = next((x for x in self._search_tracks(q)
                             if self._wanted(x, allow_derivative=allow)), None)
            self._audit_pick(title, artist, q, pick)
            if pick is None:
                return t(lang,
                         f"Não achei '{title}' de {artist} no Spotify.",
                         f"Couldn't find '{title}' by {artist} on Spotify.")
            return self._play_track_uri(pick, device_hint, lang)
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    def _play_track_corrected(self, query: str,
                              device_hint: Optional[str] = None,
                              lang: str = "pt") -> str:
        """Track search+play for a query ALREADY passed through
        _correct_artist_query. play_anything corrects once and lands here
        directly (M4-A14e: it used to call play_track and correct twice —
        harmless via the cache, but noisy in the logs)."""
        try:
            tracks = self._search_tracks(query)
            allow = bool(_DERIVATIVE_RE.search(query))
            # First result that isn't a karaoke/instrumental/cover version
            # (s84); if every result is one, the first — something plays.
            track = next((x for x in tracks if self._wanted(x, allow_derivative=allow)),
                         tracks[0] if tracks else None)
            self._audit_pick(None, None, query, track)
            if track is None:
                return t(lang, f"Não achei '{query}' no Spotify.",
                         f"Couldn't find '{query}' on Spotify.")
            return self._play_track_uri(track, device_hint, lang)
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        # Anything non-Spotify (network drop, malformed payload) is the
        # _spotify_guard decorator's job — one speakable line, not a raw
        # exception repr read aloud (M4-A1).

    def _play_artist_uri(self, artist: dict, device_hint: Optional[str] = None,
                         lang: str = "pt") -> str:
        """Start playback in artist context. Spotify autoplays the
        artist's top tracks then bleeds into recommendations — the same
        UX as tapping the artist's profile in the app, which is the fix
        for the long-standing 'toca djonga → 1 song then dead silence'
        bug (single track URI gave no context for autoplay)."""
        try:
            self._start_playback_with_fallback(
                uri=artist["uri"], device_hint=device_hint,
            )
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        self._now_cache_msg = None
        return t(lang, f"Tocando {artist.get('name') or 'artista'}.",
                 f"Playing {artist.get('name') or 'artist'}.")

    @_spotify_guard
    def play_anything(self, query: str, device_hint: Optional[str] = None,
                      lang: str = "pt") -> str:
        """Default dispatch for bare 'toca X'. Picks between artist
        context (autoplay continues) and single-track playback by query
        shape: 'X de/do/da/by Y' reads as a specific track-by-artist
        request; anything else is presumed to be a bare artist/topic
        name and gets the artist-context treatment when a strong name
        match exists. Falls back to single-track search otherwise."""
        if not self.is_configured():
            return self._need_creds_msg(lang)
        # Repair an STT-mangled artist name against the user's library
        # before searching. A "X de/do/da Y" track phrase is much longer
        # than any single artist name, so it won't fuzzy-match and passes
        # through untouched into the track branch below.
        # "X by/de Y" names a song (s84). The library correction used to run
        # on the WHOLE phrase, and with a short title and a long name the
        # phrase itself scored as the artist: "daisies by Justin Bieber" is
        # 0.70 similar to "Justin Bieber" (cutoff 0.70), was rewritten to the
        # artist, and played the artist's top track — "Beauty And A Beat".
        # Now only the artist part is corrected, which is what the
        # correction is for ("super ex de guilherme batista").
        split = _split_track_artist(query)
        if split:
            title, artist = split
            artist = self._correct_artist_query(artist)
            return self._play_track_by_artist(title, artist, query,
                                              device_hint=device_hint, lang=lang)
        query = self._correct_artist_query(query)
        try:
            # Loose name match (in _search_artist) stops the artist branch
            # from hijacking queries that happen to share a token with a
            # real artist name.
            artist = self._search_artist(query)
            if artist is not None and not self._in_library(artist):
                # A bare word can be a song AND an artist ("daisies": Justin
                # Bieber's DAISIES, and a band called Daisies). Spotify no
                # longer returns popularity, so the signal is search order:
                # when the top song IS that title and the artist isn't one
                # The user listens to, he meant the song (s84). "queen" still
                # opens Queen — its top song is "Bohemian Rhapsody".
                tracks = self._search_tracks(query)
                if tracks and self._title_key(tracks[0].get("name")) == self._title_key(query):
                    self._audit_pick(query, None, query, tracks[0])
                    return self._play_track_uri(tracks[0], device_hint, lang)
            if artist is not None:
                return self._play_artist_uri(artist, device_hint=device_hint, lang=lang)
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        except Exception:
            log.exception("play_anything artist-search failed")
            # Fall through to track — search failure shouldn't block play.
        return self._play_track_corrected(query, device_hint=device_hint,
                                          lang=lang)

    def _artist_radio_uris(self, artist_id: str, limit: int = 50) -> list:
        """Recommendation track URIs seeded on an artist, or [] if the
        endpoint is unavailable. Isolated so a failure (Spotify
        restricted the recommendations endpoint for some apps in late
        2024) degrades to artist-context play rather than an error."""
        try:
            rec = self.client.recommendations(seed_artists=[artist_id], limit=limit)
        except spotipy.SpotifyException:
            return []
        except Exception:
            log.exception("recommendations call failed")
            return []
        tracks = (rec or {}).get("tracks") or []
        return [t["uri"] for t in tracks if t.get("uri")]

    @_spotify_guard
    def play_artist_radio(self, query: str, device_hint: Optional[str] = None,
                          lang: str = "pt") -> str:
        """Play an endless artist-seeded radio for `query`. Searches the
        artist, asks Spotify for recommendations seeded on it (a mix that
        reaches beyond the artist's own catalogue — the 'radio' feel),
        and plays them. Falls back to plain artist-context play (top
        tracks + Spotify autoplay, also endless) if recommendations are
        unavailable or empty, so the command always plays something."""
        if not self.is_configured():
            return self._need_creds_msg(lang)
        # Artist radio is artist-seeded, so the same PT-BR fuzzy correction as
        # play_track/play_anything applies — "rádio do jonga" → "Djonga". (s59:
        # this path previously searched the mangled query verbatim and missed.)
        query = self._correct_artist_query(query)
        try:
            artist = self._search_artist(query, fallback_first=True)
            if artist is None:
                return t(lang, f"Não achei o artista '{query}' no Spotify.",
                         f"Couldn't find the artist '{query}' on Spotify.")
            name = artist.get("name") or query
            uris = self._artist_radio_uris(artist["id"])
            if uris:
                self._start_playback_with_fallback(uris=uris, device_hint=device_hint)
            else:
                # Graceful fallback: artist context still autoplays into
                # similar music, so the user gets a radio-like stream.
                self._start_playback_with_fallback(
                    uri=artist["uri"], device_hint=device_hint,
                )
            self._now_cache_msg = None
            return t(lang, f"Tocando rádio de {name}.", f"Playing {name} radio.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        # Non-Spotify exceptions → _spotify_guard (M4-A1).

    @_spotify_guard
    def play_playlist(self, query: str, device_hint: Optional[str] = None,
                      lang: str = "pt") -> str:
        """Play a playlist whose name matches `query`. Prefers the
        user's own playlists (the common ask is 'toca minha playlist
        gym'); falls back to public search if nothing matches. Plays
        with context_uri so Spotify cycles through the whole list."""
        if not self.is_configured():
            return self._need_creds_msg(lang)
        q_lower = query.strip().lower()
        try:
            chosen: Optional[dict] = None
            # First pass: the user's own playlists. current_user_playlists
            # needs the playlist-read-private scope (added in this same
            # commit) — if the cached token predates that, the call will
            # still succeed but only return public/owned-as-public lists,
            # so we'll silently fall through to the public search below.
            user_pls = self.client.current_user_playlists(limit=50) or {}
            for pl in user_pls.get("items") or []:
                name = (pl.get("name") or "").lower()
                if name and (q_lower == name or q_lower in name or name in q_lower):
                    chosen = pl
                    break
            if chosen is None:
                public = self.client.search(q=query, limit=1, type="playlist")
                items = (public.get("playlists") or {}).get("items") or []
                if items:
                    chosen = items[0]
            if chosen is None:
                return t(lang, f"Não achei a playlist '{query}'.",
                         f"Couldn't find the playlist '{query}'.")
            self._start_playback_with_fallback(
                uri=chosen["uri"], device_hint=device_hint,
            )
            self._now_cache_msg = None
            return t(lang, f"Tocando playlist '{chosen.get('name') or query}'.",
                     f"Playing playlist '{chosen.get('name') or query}'.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        # Non-Spotify exceptions → _spotify_guard (M4-A1).

    @_spotify_guard
    def transfer_playback_to(self, hint: Optional[str] = None,
                             lang: str = "pt") -> str:
        """Move active playback to another device ('troca pro iphone').

        Distinct from the internal `_wake_any_device` (which wakes a
        device WITHOUT forcing play, as a 404 fallback): this is the
        user explicitly asking to move what they're listening to, so we
        `force_play=True` to keep the audio going on the destination.

        The reply names the device we actually transferred to, so even
        if `_pick_device` falls back (hint didn't match any connected
        device) the user sees exactly where the sound went rather than a
        silent surprise."""
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            payload = self.client.devices()
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        devices = (payload or {}).get("devices") or []
        if not devices:
            return t(
                lang,
                "Nenhum dispositivo Spotify disponível. Abre o app no "
                "aparelho de destino e tenta de novo.",
                "No Spotify device available. Open the app on the target "
                "device and try again.",
            )
        picked = self._pick_device(devices, hint)
        if not picked or not picked.get("id"):
            return t(lang,
                     "Não achei esse dispositivo. Confere se o Spotify tá aberto nele.",
                     "Couldn't find that device. Make sure Spotify is open on it.")
        name = picked.get("name") or t(lang, "dispositivo", "device")
        if picked.get("is_active"):
            return t(lang, f"Já tá tocando no {name}.", f"Already playing on {name}.")
        try:
            self.client.transfer_playback(device_id=picked["id"], force_play=True)
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
        self._now_cache_msg = None  # playback moved; invalidate now-playing cache
        return t(lang, f"Passando pro {name}.", f"Moving to {name}.")

    @_spotify_guard
    def pause(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            self._with_wake_on_404(self.client.pause_playback)
            return t(lang, "Pausado.", "Paused.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def resume(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            try:
                self.client.start_playback()
            except spotipy.SpotifyException as e:
                if getattr(e, "http_status", 0) == 404:
                    device_id = self._wake_any_device()
                    if device_id:
                        self.client.start_playback(device_id=device_id)
                    else:
                        return self._friendly_error(e, lang)
                else:
                    raise
            return t(lang, "Continuando.", "Resuming.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def next_track(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            self._with_wake_on_404(self.client.next_track)
            self._now_cache_msg = None
            return t(lang, "Próxima música.", "Next track.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def previous_track(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            self._with_wake_on_404(self.client.previous_track)
            self._now_cache_msg = None
            return t(lang, "Voltando.", "Going back.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def now_playing(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        now = time.time()
        if (self._now_cache_msg
                and self._now_cache_lang == lang
                and (now - self._now_cache_at) < self._NOW_PLAYING_CACHE_TTL_S):
            return self._now_cache_msg
        try:
            current = self.client.current_playback()
            if not current or not current.get("item"):
                msg = t(lang, "Nada tocando agora.", "Nothing playing right now.")
            else:
                item = current["item"]
                artist = item["artists"][0]["name"] if item.get("artists") else "?"
                if current.get("is_playing") is False:
                    # An item comes back for a paused track too; it was
                    # announced as playing (s88, Codex lote 13).
                    msg = t(lang, f"Pausado em '{item['name']}' de {artist}.",
                            f"Paused on '{item['name']}' by {artist}.")
                else:
                    msg = t(lang, f"Tocando '{item['name']}' de {artist}.",
                            f"Playing '{item['name']}' by {artist}.")
            self._now_cache_msg = msg
            self._now_cache_at = now
            self._now_cache_lang = lang
            return msg
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def set_volume(self, percent: int, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            percent = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            return t(lang, "Volume inválido. Use um número de 0 a 100.",
                     "Invalid volume. Use a number from 0 to 100.")
        try:
            self._with_wake_on_404(self.client.volume, percent)
            return t(lang, f"Volume {percent}%.", f"Volume {percent}%.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)

    @_spotify_guard
    def like_current(self, lang: str = "pt") -> str:
        if not self.is_configured():
            return self._need_creds_msg(lang)
        try:
            current = self.client.current_playback()
            if not current or not current.get("item"):
                return t(lang, "Nada tocando pra curtir.", "Nothing playing to like.")
            track_id = current["item"]["id"]
            track_name = current["item"]["name"]
            self.client.current_user_saved_tracks_add([track_id])
            return t(lang, f"Curtida: '{track_name}'.", f"Liked: '{track_name}'.")
        except spotipy.SpotifyException as e:
            return self._friendly_error(e, lang)
