"""
Dispatch registry — declarative routing table (refactor, s57 → s58).

The orchestrator's `_process_core` grew a new inline `if RE.match(text): return
self._handle_X(...)` block per feature (~60 of them). That bloats a 2300-line
god-object every time a feature lands. This registry replaces that pattern: a
route is ONE data entry, and `_process_core` runs a single loop over it.

It is ALSO the single source of truth for the eval simulator. `evals/dispatch.py`
used to keep a hand-maintained parallel table of the same routing order — which
drifted (it never had the s57 routes). Now both consumers read THIS list:
  • `run(orch, text)`   — EXECUTES the matching route's handler (the orchestrator).
  • `simulate(text)`    — reports the routing decision WITHOUT executing (evals).

Contract (uniform — why no special cases are needed): every handler returns
`Optional[(reply, mode)]`, where None means "not mine, fall through". So `run`
is: first regex that matches AND whose handler returns non-None wins; otherwise
fall through to the next route / the rest of `_process_core`.

A Route carries both halves so the table stays single-sourced:
  • `run`    — execution (orchestrator); its handler returns the data-dependent
               outcome mode (e.g. CONTACT_LOOKUP vs CONTACT_LOOKUP_MISS).
  • `mode`   — the ROUTING-label mode the simulator reports (e.g. "NOTE_CREATE").
               Distinct from the handler's outcome mode ON PURPOSE: the simulator
               does not execute, so it reports routing fidelity, not the runtime
               result. `mode=None` means "execute in _process_core but stay
               INVISIBLE to the simulator" — used for dormant routes (no DISPATCH
               eval) and for a route's secondary regex whose simulator coverage
               we deliberately keep as-is.
  • `capture`— group spec (int/str) the simulator reports as `capture`.

Migration is incremental + bottom-up: families move from the inline
`_process_core` chain into this list one at a time (s58: notes, calendar, …),
each deleting its inline block AND its `evals/dispatch.py` table in the same
commit. Order is preserved by prepending each higher (earlier-running) family.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from src.core.audit import audit
from src.core.i18n import t

from src.core.clock import local_now
from src.core.dispatch.calendar import (
    CALENDAR_TRIGGER_CONTENT_RE, CALENDAR_TRIGGER_RE, CALENDAR_VERB_RE,
)
from src.core.dispatch.flow_control import SILENCE_RE
from src.core.dispatch.macos import (
    MAC_APP_LIST_RE,
    MAC_APP_SWITCH_RE,
    MAC_BRIGHTNESS_DOWN_RE,
    MAC_BRIGHTNESS_SET_RE,
    MAC_BRIGHTNESS_UP_RE,
    MAC_KEYBOARD_BRIGHTNESS_DOWN_RE,
    MAC_KEYBOARD_BRIGHTNESS_SET_RE,
    MAC_KEYBOARD_BRIGHTNESS_UP_RE,
    MAC_CLOSE_WINDOW_RE,
    MAC_DARK_OFF_RE,
    MAC_DARK_ON_RE,
    MAC_DARK_TOGGLE_RE,
    MAC_EYES_OFF_RE,
    MAC_EYES_ON_RE,
    MAC_LOCK_RE,
    MAC_MUTE_RE,
    MAC_OPEN_APP_RE,
    MAC_QUIT_APP_RE,
    MAC_REPLY_EMAIL_RE,
    MAC_SCREENSHOT_RE,
    MAC_SUMMARISE_RE,
    MAC_WHAT_IS_RE,
    MAC_SLEEP_RE,
    MAC_UNMUTE_RE,
    MAC_VOLUME_DOWN_RE,
    MAC_VOLUME_SET_RE,
    MAC_VOLUME_UP_RE,
)
from src.core.dispatch.reminders import REMINDER_TRIGGER_RE
from src.core.dispatch.studier_cmds import (
    CANCEL_STUDY_RE,
    CANCEL_VAGUE_RE,
    LAST_STUDY_RE,
    APPROVE_STUDY_RE,
    LIST_STUDIES_RE,
    REJECT_STUDY_RE,
    REVIEW_RE,
    SHOW_STUDY_RE,
    STUDY_RE,
    ADD_STUDY_RE,
    study_topic,
)
from src.core.dispatch.triggers import (
    CONTACT_ADD_RE,
    CONTACT_INFO_RE,
    CONTACT_LOOKUP_RE,
    CONTACT_SAVE_RE,
    LIST_EMAILS_RE,
    SEND_EMAIL_BY_NAME_RE,
    SEND_EMAIL_STRUCTURED_RE,
)
from src.core.dispatch.spotify import (
    SPOTIFY_LIBRARY_ARTIST_RE,
    SPOTIFY_LIKE_RE,
    SPOTIFY_NEXT_RE,
    SPOTIFY_NOW_RE,
    SPOTIFY_PAUSE_RE,
    SPOTIFY_PLAY_PLAYLIST_EN_RE,
    SPOTIFY_PLAY_PLAYLIST_RE,
    SPOTIFY_PLAY_RADIO_EN_RE,
    SPOTIFY_PLAY_RADIO_RE,
    SPOTIFY_PLAY_RE,
    SPOTIFY_PREV_RE,
    SPOTIFY_RESUME_RE,
    SPOTIFY_TRANSFER_RE,
    SPOTIFY_VOLUME_RE,
)
from src.tools.spotify_player import SpotifyPlayer
from src.core.dispatch.normalisation import removed_portuguese_courtesy_suffix
from src.core.dispatch.notes import (
    NOTE_APPEND_RE,
    NOTE_APPEND_TAIL_RE,
    NOTE_CREATE_FOLDER_RE,
    NOTE_CREATE_RE,
    NOTE_LIST_RE,
    NOTE_SEARCH_RE,
)
from src.core.dispatch.refine import REFINE_RE
from src.core.dispatch.undo import UNDO_RE
from src.core.dispatch.weather import WEATHER_RE
from src.core.dispatch.digest import WHATS_CHANGED_RE
from src.core.dispatch.system_status import SYSTEM_STATUS_RE
from src.core.dispatch.deep_research import DEEP_RESEARCH_RE
from src.core.dispatch.dream import DREAM_REVIEW_RE
from src.core.dispatch.entities import LAST_CONTACT_RE, RECENT_PEOPLE_RE
from src.core.dispatch.jobs import JOBS_RERANK_RE, JOBS_TOP_RE
from src.core.dispatch.fitness import (
    FITNESS_LOG_MEAL_RE, FITNESS_MACROS_TODAY_RE,
    FITNESS_LIST_TODAY_RE, FITNESS_UNDO_MEAL_RE, FITNESS_CLEAR_TODAY_RE,
    FITNESS_LOG_RUN_RE, FITNESS_LOG_LIFT_RE,
    FITNESS_RUN_TREND_RE, FITNESS_LIFT_TREND_RE,
    FITNESS_UNDO_WORKOUT_RE, FITNESS_CLEAR_WORKOUTS_RE, FITNESS_DELETE_GUARD_RE,
)

log = logging.getLogger(__name__)

# A route's `run` adapter does any per-route extraction (named groups, derived
# args) and calls the handler. It returns the (reply, mode) tuple, or None to
# fall through (a gated/dormant feature, or an unparseable body).
RunFn = Callable[[Any, "re.Match"], Optional[Tuple[str, str]]]


@dataclass(frozen=True)
class Route:
    name: str                              # matcher name (simulator + debug)
    regex: "re.Pattern"                    # compiled regex — matched directly
    run: RunFn                             # execution (orchestrator)
    mode: Optional[str] = None             # routing-label mode for the simulator
    capture: Optional[Union[int, str]] = None  # group spec for the simulator


@dataclass(frozen=True)
class RouteMatch:
    """Routing-only result for the eval simulator (no execution)."""
    mode: str
    matcher: str
    capture: Optional[str] = None


def _dream_action(m: "re.Match") -> str:
    return "show" if m.group("show") else "keep" if m.group("keep") else "drop"


def _mac_quit_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # Quitting can lose unsaved work (S3): the handler STAGES a pending action
    # and waits for the user's "sim" instead of acting immediately.
    return o._handle_mac_quit_regex(m.group("app").strip())


def _contact_info_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # The name can land in any of four alternation groups. If it matches no
    # stored contact the handler returns None and we FALL THROUGH — so "qual o
    # número do meu passaporte" isn't hijacked (goes to memory/knowledge).
    name = (m.group("name1") or m.group("name2")
            or m.group("name3") or m.group("name4")).strip()
    return o._handle_contact_info(name)


def _list_emails_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # Honour a user-given count (count_pre "3 últimos" / count_post "últimos 3";
    # only one fires), default 5, clamp to [1, 50] (Gmail caps maxResults and a
    # 2-digit count over 50 is almost certainly a misparse).
    raw_count = (m.groupdict().get("count_pre") or m.groupdict().get("count_post"))
    try:
        count = int(raw_count) if raw_count else 5
    except (TypeError, ValueError):
        count = 5
    count = max(1, min(count, 50))
    return o._handle_list_emails(count=count)


def _spotify_device_hint(o: Any, m: "re.Match") -> Optional[str]:
    # The device hint ("no iphone", "na sala") is a side-channel searched over
    # the WHOLE input (m.string == the text the regex matched), not the route's
    # own capture. Mirrors the old inline `_SPOTIFY_DEVICE_HINT_RE.search(text)`.
    hint = o._SPOTIFY_DEVICE_HINT_RE.search(m.string)
    return hint.group(1).lower() if hint else None


def _spotify_playlist_run(o: Any, m: "re.Match") -> Tuple[str, str]:
    return (o.spotify.play_playlist(m.group(1).strip(),
                                    device_hint=_spotify_device_hint(o, m),
                                    lang=o._reply_lang),
            "SPOTIFY_PLAY_PLAYLIST")


def _spotify_radio_run(o: Any, m: "re.Match") -> Tuple[str, str]:
    return (o.spotify.play_artist_radio(m.group(1).strip(),
                                        device_hint=_spotify_device_hint(o, m),
                                        lang=o._reply_lang),
            "SPOTIFY_PLAY_RADIO")


def _spotify_play_run(o: Any, m: "re.Match", *, query: Optional[str] = None,
                      pre_strip_text: Optional[str] = None) -> Tuple[str, str]:
    # Only the bare-title route can recover this ambiguous courtesy tail.
    # Library artists call this adapter with an explicit canonical query.
    if (query is None and m.re is SPOTIFY_PLAY_RE
            and m.group("artist_object") is None):
        suffix = removed_portuguese_courtesy_suffix(pre_strip_text or "")
        if suffix:
            query = m.group(1).strip()
            return (o.spotify.play_with_title_candidate(
                        query, f"{query} {suffix}",
                        device_hint=_spotify_device_hint(o, m), lang=o._reply_lang),
                    "SPOTIFY_PLAY")
    # play_anything picks artist-context vs track-by-artist by query shape.
    return (o.spotify.play_anything(query if query is not None else m.group(1).strip(),
                                    device_hint=_spotify_device_hint(o, m),
                                    lang=o._reply_lang),
            "SPOTIFY_PLAY")


def _spotify_library_artist_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # A wrong positive here sends an ordinary sending request to Spotify.
    # Reuse the saved-track cache and normalization, but NO fuzzy correction
    # (play_anything's 0.7 cutoff is suitable only after intent is established).
    query = SpotifyPlayer._normalize_name(m.group(1).strip())
    try:
        canonical = next((name for name in o.spotify._library_artist_names()
                          if SpotifyPlayer._normalize_name(name) == query), None)
    except Exception as exc:
        log.warning("Spotify library artist lookup failed: %s", type(exc).__name__)
        return None
    if canonical is None:
        return None
    return _spotify_play_run(o, m, query=canonical)


def _reminder_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # Mirrors the old inline block: stage a reminder from the parsed body; an
    # unparseable body returns None → fall through to the LLM tool-call path.
    # Local zone by name, not UTC (s85): reminders honour the offset, so a
    # UTC "now" turned "às 10" into an 11:00 reminder in British Summer Time.
    return o._handle_reminder_regex(m.group("body"), local_now())


def _calendar_run(o: Any, m: "re.Match") -> Optional[Tuple[str, str]]:
    # Mirrors the old inline block: stage a create_event from the parsed body;
    # an unparseable body returns None → fall through to the LLM tool-call path.
    return o._handle_calendar_regex(m.group("body"), local_now())


# Order is preserved exactly from the previous inline chain. Adding a feature =
# append one Route here (and its handler) — `_process_core` is untouched.
# Families migrated from the inline chain are PREPENDED (they ran earlier).
DISPATCH_ROUTES: List[Route] = [
    # ---- Studier / Contacts / Email (migrated from inline s59 — the last,
    #      interleaved block). Order = the old _process_core execution order:
    #      studier REVIEW/APPROVE/REJECT first, then the contact CRUD, then the
    #      email verbs, then studier SHOW/LIST/STUDY. LIST_EMAILS stays BEFORE
    #      SHOW_STUDY (the over-greedy "show <anything>" form — s20). The
    #      simulator `mode`s carry the STUDIER_ prefix (routing labels, not the
    #      handlers' outcome modes). CONTACT_ADD/INFO are execute-only
    #      (mode=None): they were never simulated, and CONTACT_INFO falls
    #      through (None) when the name matches no contact. ----
    Route("REVIEW_RE", REVIEW_RE,
          lambda o, m: o._handle_review_studies(), mode="STUDIER_REVIEW"),
    Route("APPROVE_STUDY_RE", APPROVE_STUDY_RE,
          lambda o, m: o._handle_approve_study(m.group(1).strip()),
          mode="STUDIER_APPROVE", capture=1),
    Route("REJECT_STUDY_RE", REJECT_STUDY_RE,
          lambda o, m: o._handle_reject_study(m.group(1).strip()),
          mode="STUDIER_REJECT", capture=1),
    Route("CONTACT_ADD_RE", CONTACT_ADD_RE,
          lambda o, m: o._handle_contact_add(m.group("fields"))),
    Route("CONTACT_SAVE_RE", CONTACT_SAVE_RE,
          lambda o, m: o._handle_contact_save(
              name=m.group("name").strip(), email=m.group("email").strip()),
          mode="CONTACT_SAVE"),
    Route("CONTACT_LOOKUP_RE", CONTACT_LOOKUP_RE,
          lambda o, m: o._handle_contact_lookup(
              (m.group("name1") or m.group("name2")).strip()),
          mode="CONTACT_LOOKUP"),
    Route("CONTACT_INFO_RE", CONTACT_INFO_RE, _contact_info_run),
    Route("SEND_EMAIL_STRUCTURED_RE", SEND_EMAIL_STRUCTURED_RE,
          lambda o, m: o._handle_send_email_structured(
              to=m.group("to"), subject=m.group("subject").strip(),
              body=m.group("body").strip()),
          mode="SEND_EMAIL_STRUCTURED"),
    Route("SEND_EMAIL_BY_NAME_RE", SEND_EMAIL_BY_NAME_RE,
          lambda o, m: o._handle_send_email_by_name(
              name=m.group("name").strip(), subject=m.group("subject").strip(),
              body=m.group("body").strip()),
          mode="SEND_EMAIL_BY_NAME"),
    Route("LIST_EMAILS_RE", LIST_EMAILS_RE, _list_emails_run, mode="LIST_EMAILS"),
    Route("SHOW_STUDY_RE", SHOW_STUDY_RE,
          lambda o, m: o._handle_show_study(m.group(1).strip()),
          mode="STUDIER_SHOW", capture=1),
    Route("LAST_STUDY_RE", LAST_STUDY_RE,
          lambda o, m: o._handle_last_study(), mode="STUDY_LAST"),
    Route("CANCEL_STUDY_RE", CANCEL_STUDY_RE,
          lambda o, m: o._handle_cancel_studies(
              None if m.group("all") else
              [int(x) for x in m.group("n").replace(" ", "").split(",")]),
          mode="STUDY_CANCELLED"),
    Route("CANCEL_VAGUE_RE", CANCEL_VAGUE_RE,
          lambda o, m: o._handle_cancel_vague(), mode="STUDY_LIST"),
    Route("LIST_STUDIES_RE", LIST_STUDIES_RE,
          lambda o, m: o._handle_list_studies(), mode="STUDIER_LIST"),
    Route("STUDY_RE", STUDY_RE,
          lambda o, m: o._handle_add_study(study_topic(m)),
          mode="STUDIER_ADD", capture=1),
    # Before Spotify's broad coloca/põe/bota and Notes' addition routes;
    # the mandatory study-list destination leaves their commands untouched.
    Route("ADD_STUDY_RE", ADD_STUDY_RE,
          lambda o, m: o._handle_add_study(study_topic(m)),
          mode="STUDIER_ADD"),

    # ---- Silence + Mac sleep (migrated from inline s59). Both run BEFORE
    #      Spotify: MAC_SLEEP needs the "mac/computador" qualifier but "põe o
    #      mac pra dormir" would otherwise be eaten by SPOTIFY_PLAY's permissive
    #      "põe X" verb. Direct handler calls, no capture. ----
    Route("SILENCE_RE", SILENCE_RE,
          lambda o, m: o._handle_silence(), mode="SILENCE"),
    Route("MAC_SLEEP_RE", MAC_SLEEP_RE,
          lambda o, m: o._handle_mac_sleep_regex(
              phrased_as_shutdown=bool(m.groupdict().get("shutdown"))),
          mode="MAC_SLEEP"),

    # Absolute brightness precedes Spotify's broad "põe/coloca X" play route.
    Route("MAC_KEYBOARD_BRIGHTNESS_SET_RE", MAC_KEYBOARD_BRIGHTNESS_SET_RE,
          lambda o, m: (o.macos.keyboard_brightness_set(
              int(m.group(1)) if m.group(1) is not None else
              0 if m.group("minimum") or m.group("off") else 50 if m.group("on") else 100,
              lang=o._reply_lang), "MAC_KEYBOARD_BRIGHTNESS"),
          mode="MAC_KEYBOARD_BRIGHTNESS", capture=1),
    Route("MAC_KEYBOARD_BRIGHTNESS_UP_RE", MAC_KEYBOARD_BRIGHTNESS_UP_RE,
          lambda o, m: (o.macos.keyboard_brightness_up(lang=o._reply_lang), "MAC_KEYBOARD_BRIGHTNESS"),
          mode="MAC_KEYBOARD_BRIGHTNESS"),
    Route("MAC_KEYBOARD_BRIGHTNESS_DOWN_RE", MAC_KEYBOARD_BRIGHTNESS_DOWN_RE,
          lambda o, m: (o.macos.keyboard_brightness_down(lang=o._reply_lang), "MAC_KEYBOARD_BRIGHTNESS"),
          mode="MAC_KEYBOARD_BRIGHTNESS"),
    Route("MAC_BRIGHTNESS_SET_RE", MAC_BRIGHTNESS_SET_RE,
          lambda o, m: (o.macos.set_brightness(
              int(m.group(1)) if m.group(1) is not None else (0 if m.group("minimum") else 100),
              lang=o._reply_lang), "MAC_BRIGHTNESS"),
          mode="MAC_BRIGHTNESS", capture=1),

    # ---- Spotify (migrated from inline s58). Order is load-bearing: single-
    #      word verbs first (so PLAY's broad capture can't swallow them), then
    #      transfer, then playlist/radio (need the literal word) BEFORE generic
    #      play. VOLUME/TRANSFER and the play/playlist/radio targets report
    #      group(1) as the simulator capture; the device hint is a search-based
    #      side-channel (see _spotify_device_hint). ----
    Route("SPOTIFY_PAUSE_RE", SPOTIFY_PAUSE_RE,
          lambda o, m: (o.spotify.pause(lang=o._reply_lang), "SPOTIFY_PAUSE"), mode="SPOTIFY_PAUSE"),
    Route("SPOTIFY_RESUME_RE", SPOTIFY_RESUME_RE,
          lambda o, m: (o.spotify.resume(lang=o._reply_lang), "SPOTIFY_RESUME"), mode="SPOTIFY_RESUME"),
    Route("SPOTIFY_NEXT_RE", SPOTIFY_NEXT_RE,
          lambda o, m: (o.spotify.next_track(lang=o._reply_lang), "SPOTIFY_NEXT"), mode="SPOTIFY_NEXT"),
    Route("SPOTIFY_PREV_RE", SPOTIFY_PREV_RE,
          lambda o, m: (o.spotify.previous_track(lang=o._reply_lang), "SPOTIFY_PREV"), mode="SPOTIFY_PREV"),
    Route("SPOTIFY_NOW_RE", SPOTIFY_NOW_RE,
          lambda o, m: (o.spotify.now_playing(lang=o._reply_lang), "SPOTIFY_NOW"), mode="SPOTIFY_NOW"),
    Route("SPOTIFY_LIKE_RE", SPOTIFY_LIKE_RE,
          lambda o, m: (o.spotify.like_current(lang=o._reply_lang), "SPOTIFY_LIKE"), mode="SPOTIFY_LIKE"),
    Route("SPOTIFY_VOLUME_RE", SPOTIFY_VOLUME_RE,
          lambda o, m: (o.spotify.set_volume(int(m.group(1)), lang=o._reply_lang), "SPOTIFY_VOLUME"),
          mode="SPOTIFY_VOLUME", capture=1),
    Route("SPOTIFY_TRANSFER_RE", SPOTIFY_TRANSFER_RE,
          lambda o, m: (o.spotify.transfer_playback_to(m.group(1).lower(), lang=o._reply_lang), "SPOTIFY_TRANSFER"),
          mode="SPOTIFY_TRANSFER", capture=1),
    Route("SPOTIFY_PLAY_PLAYLIST_RE", SPOTIFY_PLAY_PLAYLIST_RE,
          _spotify_playlist_run, mode="SPOTIFY_PLAY_PLAYLIST", capture=1),
    Route("SPOTIFY_PLAY_PLAYLIST_EN_RE", SPOTIFY_PLAY_PLAYLIST_EN_RE,
          _spotify_playlist_run, mode="SPOTIFY_PLAY_PLAYLIST", capture=1),
    Route("SPOTIFY_PLAY_RADIO_RE", SPOTIFY_PLAY_RADIO_RE,
          _spotify_radio_run, mode="SPOTIFY_PLAY_RADIO", capture=1),
    Route("SPOTIFY_PLAY_RADIO_EN_RE", SPOTIFY_PLAY_RADIO_EN_RE,
          _spotify_radio_run, mode="SPOTIFY_PLAY_RADIO", capture=1),
    Route("SPOTIFY_PLAY_RE", SPOTIFY_PLAY_RE,
          _spotify_play_run, mode="SPOTIFY_PLAY", capture=1),
    # Library-dependent like CONTACT_INFO: simulation/hallucination offers
    # cannot establish membership, so this route deliberately has mode=None.
    Route("SPOTIFY_LIBRARY_ARTIST_RE", SPOTIFY_LIBRARY_ARTIST_RE,
          _spotify_library_artist_run),

    # ---- macOS system control (migrated from inline s58). Direct calls into
    #      MacOSController; VOLUME_SET carries an int level in group(1).
    #      Spotify requires an explicit name; bare volume belongs to the Mac.
    #      Play excludes Mac control objects. Order within the block matters
    #      (dark toggle before on/off). ----
    Route("MAC_VOLUME_SET_RE", MAC_VOLUME_SET_RE,
          lambda o, m: (o.macos.set_system_volume(
              0 if m.group("minimum") else 100 if m.group("maximum") else int(m.group(1)),
              lang=o._reply_lang), "MAC_VOLUME"),
          mode="MAC_VOLUME", capture=1),
    Route("MAC_VOLUME_UP_RE", MAC_VOLUME_UP_RE,
          lambda o, m: (o.macos.volume_up(lang=o._reply_lang), "MAC_VOLUME"), mode="MAC_VOLUME"),
    Route("MAC_VOLUME_DOWN_RE", MAC_VOLUME_DOWN_RE,
          lambda o, m: (o.macos.volume_down(lang=o._reply_lang), "MAC_VOLUME"), mode="MAC_VOLUME"),
    Route("MAC_MUTE_RE", MAC_MUTE_RE,
          lambda o, m: (o.macos.mute(lang=o._reply_lang), "MAC_MUTE"), mode="MAC_MUTE"),
    Route("MAC_UNMUTE_RE", MAC_UNMUTE_RE,
          lambda o, m: (o.macos.unmute(lang=o._reply_lang), "MAC_UNMUTE"), mode="MAC_UNMUTE"),
    Route("MAC_BRIGHTNESS_UP_RE", MAC_BRIGHTNESS_UP_RE,
          lambda o, m: (o.macos.brightness_up(lang=o._reply_lang), "MAC_BRIGHTNESS"), mode="MAC_BRIGHTNESS"),
    Route("MAC_BRIGHTNESS_DOWN_RE", MAC_BRIGHTNESS_DOWN_RE,
          lambda o, m: (o.macos.brightness_down(lang=o._reply_lang), "MAC_BRIGHTNESS"), mode="MAC_BRIGHTNESS"),
    Route("MAC_DARK_TOGGLE_RE", MAC_DARK_TOGGLE_RE,
          lambda o, m: (o.macos.toggle_dark_mode(lang=o._reply_lang), "MAC_DARK"), mode="MAC_DARK"),
    Route("MAC_DARK_ON_RE", MAC_DARK_ON_RE,
          lambda o, m: (o.macos.enable_dark_mode(lang=o._reply_lang), "MAC_DARK"), mode="MAC_DARK"),
    Route("MAC_DARK_OFF_RE", MAC_DARK_OFF_RE,
          lambda o, m: (o.macos.disable_dark_mode(lang=o._reply_lang), "MAC_DARK"), mode="MAC_DARK"),
    Route("MAC_LOCK_RE", MAC_LOCK_RE,
          lambda o, m: (o.macos.lock_screen(lang=o._reply_lang), "MAC_LOCK"), mode="MAC_LOCK"),
    Route("MAC_SCREENSHOT_RE", MAC_SCREENSHOT_RE,
          lambda o, m: (o.macos.screenshot(lang=o._reply_lang), "MAC_SCREENSHOT"), mode="MAC_SCREENSHOT"),
    Route("MAC_APP_LIST_RE", MAC_APP_LIST_RE,
          lambda o, m: (o.macos.list_apps(lang=o._reply_lang), "MAC_APP_LIST"), mode="MAC_APP_LIST"),

    # ---- Active-context: reply to the email in front (inovação #3, Bloco A).
    #      Reads the Mail-selected message, drafts a reply, stages it through
    #      the send_email gate (nothing sends without "sim"). Parameter-free. ----
    Route("MAC_REPLY_EMAIL_RE", MAC_REPLY_EMAIL_RE,
          lambda o, m: o._handle_reply_email(), mode="MAC_REPLY_EMAIL"),
    Route("MAC_SUMMARISE_RE", MAC_SUMMARISE_RE,
          lambda o, m: o._handle_summarise_screen(), mode="MAC_SUMMARISE"),
    Route("MAC_WHAT_IS_RE", MAC_WHAT_IS_RE,
          lambda o, m: o._handle_what_is_this(), mode="MAC_WHAT_IS"),
    # Voice toggle of the screen-reading kill-switch (#3): same flag file as
    # scripts/cortex-eyes.sh + the hotkey. The handlers do NOT gate on the
    # kill-switch — they control it. Parameter-free; full-match anchored.
    Route("MAC_EYES_OFF_RE", MAC_EYES_OFF_RE,
          lambda o, m: o._handle_screen_eyes_off(), mode="MAC_EYES_OFF"),
    Route("MAC_EYES_ON_RE", MAC_EYES_ON_RE,
          lambda o, m: o._handle_screen_eyes_on(), mode="MAC_EYES_ON"),

    # ---- Reminder (migrated from inline s58). Single front-door regex; an
    #      unparseable body returns None → falls through to the LLM path. ----
    Route("REMINDER_TRIGGER_RE", REMINDER_TRIGGER_RE, _reminder_run, mode="REMINDER"),

    # ---- Calendar (migrated from inline s58). Two front-door regexes feed one
    #      handler (keyword-first, then content-first). The CONTENT regex is
    #      execute-only (mode=None) to preserve the simulator's existing
    #      coverage (it only ever simulated TRIGGER). An unparseable body
    #      returns None → falls through to the LLM tool-call path. ----
    Route("CALENDAR_TRIGGER_RE", CALENDAR_TRIGGER_RE, _calendar_run, mode="CALENDAR"),
    Route("CALENDAR_TRIGGER_CONTENT_RE", CALENDAR_TRIGGER_CONTENT_RE, _calendar_run),
    # s85 (fila 4.2): verb-first with no calendar keyword, "marca dentista
    # amanhã às 10". Execute-only like CONTENT; same None-on-no-time contract.
    Route("CALENDAR_VERB_RE", CALENDAR_VERB_RE, _calendar_run),

    # ---- Weather (idea #27, s73). Deterministic: Open-Meteo + template,
    #      NO LLM. Lease checked inside the handler (no lease → the same
    #      offer-to-search flow; approval re-dispatches deterministically).
    Route("WEATHER_RE", WEATHER_RE,
          lambda o, m: o._handle_weather(m.group("city"), raw_text=m.string),
          mode="WEATHER"),

    # ---- Apple Notes (migrated from inline s58). S2: executes directly. ----
    Route("NOTE_LIST_RE", NOTE_LIST_RE,
          lambda o, m: o._handle_note_list(),
          mode="NOTE_LIST"),
    Route("NOTE_SEARCH_RE", NOTE_SEARCH_RE,
          lambda o, m: o._handle_note_search(m.group("query")),
          mode="NOTE_SEARCH"),
    # v1.1 (s73): append + folder targeting. Both append shapes and the
    # folder create run BEFORE the plain create — "adiciona nota na pasta
    # X: Y" also matches NOTE_CREATE_RE (folder swallowed into the body).
    Route("NOTE_APPEND_RE", NOTE_APPEND_RE,
          lambda o, m: o._handle_note_append(m.group("title"), m.group("text")),
          mode="NOTE_APPEND"),
    Route("NOTE_APPEND_TAIL_RE", NOTE_APPEND_TAIL_RE,
          lambda o, m: o._handle_note_append(
              m.group("title"), m.group("text") or m.group("text_en")),
          mode="NOTE_APPEND"),
    Route("NOTE_CREATE_FOLDER_RE", NOTE_CREATE_FOLDER_RE,
          lambda o, m: o._handle_note_create(
              m.group("body"), folder=m.group("folder")),
          mode="NOTE_CREATE"),
    Route("NOTE_CREATE_RE", NOTE_CREATE_RE,
          lambda o, m: o._handle_note_create(m.group("body")),
          mode="NOTE_CREATE"),

    # ---- Fitness Engine v1 (slice 1: meals + macros, s76). Deterministic:
    #      parsed food → local food table → arithmetic → template, NO LLM.
    #      S2 local-only. LOG_MEAL returns None (falls through) when it
    #      recognises no known food and has nothing to report. ----
    Route("FITNESS_MACROS_TODAY_RE", FITNESS_MACROS_TODAY_RE,
          lambda o, m: o._handle_fitness_macros_today(raw_text=m.string),
          mode="FITNESS_MACROS"),
    Route("FITNESS_LIST_TODAY_RE", FITNESS_LIST_TODAY_RE,
          lambda o, m: o._handle_fitness_list_today(raw_text=m.string),
          mode="FITNESS_LIST"),
    # LOG comes AFTER the read routes on purpose (s82 cont.): its trigger is
    # a SEARCH for the eating verb anywhere in the sentence, which is broad by
    # design — so every "what did I eat" question must get first refusal.
    Route("FITNESS_LOG_MEAL_RE", FITNESS_LOG_MEAL_RE,
          lambda o, m: o._handle_fitness_log_meal(m.group("body"), raw_text=m.string),
          mode="FITNESS_MEAL", capture="body"),
    Route("FITNESS_UNDO_MEAL_RE", FITNESS_UNDO_MEAL_RE,
          lambda o, m: o._handle_fitness_undo_meal(raw_text=m.string),
          mode="FITNESS_UNDO"),
    Route("FITNESS_CLEAR_TODAY_RE", FITNESS_CLEAR_TODAY_RE,
          lambda o, m: o._handle_fitness_clear_today(raw_text=m.string),
          mode="FITNESS_CLEAR"),
    # slice 2: workouts. LOG before TREND; RUN_TREND before LIFT_TREND so
    # "como tá minha corrida" goes to running, not the generic exercise capture.
    Route("FITNESS_LOG_RUN_RE", FITNESS_LOG_RUN_RE,
          lambda o, m: o._handle_fitness_log_run(m.group("dist"), m.group("dur"),
                                                 raw_text=m.string),
          mode="FITNESS_RUN_LOG"),
    Route("FITNESS_LOG_LIFT_RE", FITNESS_LOG_LIFT_RE,
          lambda o, m: o._handle_fitness_log_lift(
              m.group("ex"), m.group("sets"), m.group("reps"), m.group("weight"),
              raw_text=m.string),
          mode="FITNESS_LIFT_LOG"),
    Route("FITNESS_RUN_TREND_RE", FITNESS_RUN_TREND_RE,
          lambda o, m: o._handle_fitness_run_trend(raw_text=m.string),
          mode="FITNESS_RUN_TREND"),
    # mode=None: the handler falls through when the captured word isn't a
    # logged exercise ("como tá meu dia" → LLM), so it stays invisible to the
    # simulator (same pattern as CONTACT_INFO_RE). Covered by unit tests.
    Route("FITNESS_LIFT_TREND_RE", FITNESS_LIFT_TREND_RE,
          lambda o, m: o._handle_fitness_lift_trend(m.group("ex"), raw_text=m.string)),
    # The kind said goes along: "apaga a última corrida" removed the bench
    # press logged after it (s88, Codex lote 9).
    Route("FITNESS_UNDO_WORKOUT_RE", FITNESS_UNDO_WORKOUT_RE,
          lambda o, m: o._handle_fitness_undo_workout(
              raw_text=m.string,
              kind=next((k for k in m.group("kind", "kind2", "kind3", "kind4") if k), None)),
          mode="FITNESS_WORKOUT_UNDO"),
    Route("FITNESS_CLEAR_WORKOUTS_RE", FITNESS_CLEAR_WORKOUTS_RE,
          lambda o, m: o._handle_fitness_clear_workouts(raw_text=m.string),
          mode="FITNESS_WORKOUT_CLEAR"),
    # AFTER every fitness deletion: what reaches it was not carried out, and
    # gets a fixed "nothing deleted" (s88, Codex lote 9 — SOBERANIA 4).
    Route("FITNESS_DELETE_GUARD_RE", FITNESS_DELETE_GUARD_RE,
          lambda o, m: o._handle_fitness_delete_guard(raw_text=m.string),
          mode="FITNESS_DELETE_NONE"),

    # ---- Job Hunter surface (slice 4, s76). Read-only: reads the top-N
    #      already-ranked matches from the local ledger (no LLM/embedding in the
    #      turn). Shared registry → surfaces in /ask AND voice. ----
    #      Re-rank is registered FIRST: a "re-rankeia minhas vagas" carries a
    #      rank verb and must not be read as a plain "minhas vagas" lookup.
    Route("JOBS_RERANK_RE", JOBS_RERANK_RE,
          lambda o, m: o._handle_jobs_rerank(raw_text=m.string),
          mode="JOBS_RERANK"),
    Route("JOBS_TOP_RE", JOBS_TOP_RE,
          lambda o, m: o._handle_jobs_top(raw_text=m.string),
          mode="JOBS_TOP"),

    # ---- Onda 4/5 + Fase 2 (s57). Dormant/new → mode=None (no DISPATCH eval;
    #      covered by their own REGEX evals + test_dispatch_registry). ----
    Route("undo", UNDO_RE,
          lambda o, m: (setattr(o, "_reply_lang", "en" if m.group(0).lower().startswith("undo") else "pt"),
                        o._handle_undo())[1]),
    Route("whats_changed", WHATS_CHANGED_RE,
          lambda o, m: o._handle_whats_changed()),
    Route("system_status", SYSTEM_STATUS_RE,
          lambda o, m: o._handle_system_status()),
    Route("deep_research", DEEP_RESEARCH_RE,
          lambda o, m: o._handle_deep_research(m.group("topic"))),
    Route("dream_review", DREAM_REVIEW_RE,
          lambda o, m: o._handle_dream_review(_dream_action(m))),
    Route("last_contact", LAST_CONTACT_RE,
          lambda o, m: o._handle_last_contact(m.group("name"))),
    Route("recent_people", RECENT_PEOPLE_RE,
          lambda o, m: o._handle_recent_people()),

    # ---- Prompt enrichment v1 (idea #30, s74). Opt-in by explicit prefix;
    #      the handler stages the expanded prompt for sim/não — "sim" runs
    #      it through the full pipeline (never a gate bypass). Bare prefix
    #      (empty request) returns None → falls through. ----
    Route("REFINE_RE", REFINE_RE,
          lambda o, m: o._handle_refine(m.group("request")),
          mode="REFINE", capture="request"),
]


# Second segment: the broad macOS app-control verbs (quit/open/switch/close).
# These run LATER than DISPATCH_ROUTES — AFTER the planner / multi-intent
# autoroute — because their verbs ("abre X", "fecha X") are permissive and must
# not pre-empt a multi-intent phrase or a specific earlier dispatch. So they
# live in their own list, invoked by a second `run`/`simulate` call positioned
# after the planner in both _process_core and simulate_dispatch (s59).
POST_PLANNER_ROUTES: List[Route] = [
    # Close window (red button / Cmd+W) — non-destructive, acts immediately.
    Route("MAC_CLOSE_WINDOW_RE", MAC_CLOSE_WINDOW_RE,
          lambda o, m: (o.macos.close_window(m.group("app").strip(), lang=o._reply_lang), "MAC_CLOSE_WINDOW"),
          mode="MAC_CLOSE_WINDOW", capture="app"),
    # Quit (S3 — can lose unsaved work) → staged, waits for "sim".
    Route("MAC_QUIT_APP_RE", MAC_QUIT_APP_RE, _mac_quit_run,
          mode="MAC_QUIT_APP", capture="app"),
    # Switch focus (S2 — reversible).
    Route("MAC_APP_SWITCH_RE", MAC_APP_SWITCH_RE,
          lambda o, m: (o.macos.switch_to_app(m.group("app").strip(), lang=o._reply_lang), "MAC_APP_SWITCH"),
          mode="MAC_APP_SWITCH", capture="app"),
    # Open app (broad verb — last, so "abre meus emails" already routed to
    # LIST_EMAILS in the first segment).
    Route("MAC_OPEN_APP_RE", MAC_OPEN_APP_RE,
          lambda o, m: (o.macos.open_app(m.group(1).strip(), lang=o._reply_lang), "MAC_OPEN_APP"),
          mode="MAC_OPEN_APP", capture=1),
]


def run(orch: Any, text: str, routes: Optional[List[Route]] = None, *,
        pre_strip_text: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Try each route in order. Returns the first non-None handler result, or
    None if no route claimed the text (caller continues `_process_core`).
    `routes` defaults to the pre-planner DISPATCH_ROUTES; pass POST_PLANNER_ROUTES
    for the second segment.
    `pre_strip_text` is execution-only context for the bare Spotify title
    ambiguity; all route matching and simulation keep the stripped text.

    Per-route barrier (M1-A11): every handler PROMISES to be fail-soft, but a
    promise isn't a contract — an osascript timeout, a new spotipy edge or a
    mixin bug would propagate to the API and become an HTTP 500 / dead SSE
    turn instead of a speakable reply. A crash is audited (`route_crash`,
    traceback in the server log) and answered in the turn's language. The
    route DID claim the text, so we return a reply rather than falling
    through to the LLM with a command it can't execute."""
    for route in (DISPATCH_ROUTES if routes is None else routes):
        match = route.regex.match(text)
        if match is None:
            continue
        try:
            if route.run is _spotify_play_run:
                result = _spotify_play_run(orch, match, pre_strip_text=pre_strip_text)
            else:
                result = route.run(orch, match)
        except Exception as e:
            log.exception(f"route {route.name} crashed")
            audit.log("–", "dispatch", "route_crash", payload={
                "route": route.name, "error": f"{type(e).__name__}: {e}"[:200],
            })
            lang = getattr(orch, "_reply_lang", "pt") or "pt"
            return (
                t(lang,
                  "Não consegui executar esse comando agora. O detalhe ficou no log.",
                  "I couldn't run that command right now. Details are in the log."),
                "ROUTE_ERROR",
            )
        if result is not None:
            return result
    return None


def simulate(text: str, routes: Optional[List[Route]] = None) -> Optional[RouteMatch]:
    """Routing-only walk for the eval simulator: the first route (with a
    simulator `mode`) whose regex matches, reporting its routing mode + capture.
    Routes with `mode=None` are invisible here (executed by `run`, not
    simulated). Returns None if no simulated route matches — the caller (the
    eval `simulate_dispatch`) then continues to its remaining checks. `routes`
    defaults to DISPATCH_ROUTES; pass POST_PLANNER_ROUTES for the second
    segment."""
    for route in (DISPATCH_ROUTES if routes is None else routes):
        if route.mode is None:
            continue
        match = route.regex.match(text)
        if match is None:
            continue
        captured: Optional[str] = None
        if route.capture is not None:
            try:
                raw = match.group(route.capture)
                if raw is not None:
                    captured = raw.strip()
            except Exception:
                pass
        return RouteMatch(mode=route.mode, matcher=route.name, capture=captured)
    return None
