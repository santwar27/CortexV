import json
import logging
import os
import re
import queue
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Tuple
from dotenv import load_dotenv
import ollama
from src.core.ollama_client import KEEP_ALIVE as _KEEP_ALIVE, THINK as _THINK

from src.core.audit import audit
from src.core.clock import local_wall_time
from src.core.build_guides import detect_guide, guide_text
from src.core.dialog_scope import PerConversation
from src.core.action_ledger import ledger as _action_ledger
from src.core.arbiter import arbiter
from src.core.gate_queue import gate_queue
from src.core.reflex import reflex_layer
from src.core.gatekeeper import LogicGate
from src.core.engine import SKIP_FOLLOWUP_MARKER, LiteralToolReply, OllamaEngine
from src.core.stream_reply import ReplyReplacement
from src.core.input_text import strip_paste_markers
from src.core.studies_review import StudiesReview
from src.core.study_queue import StudyQueue
from src.core.tool_risk import build_preview as _build_dry_run_preview
from src.core.model_profile import guard_active
from src.core.doc_retrieval import select_relevant
from src.core import path_route
from src.core.prompt_safety import (
    MAX_DOC_CHARS, clip_to_window, document_prompt, fence, guard_line,
    note_false_read,
)
from src.core.tool_specs import TOOL_SPECS
from src.core import planner as _planner
from src.core import code_roots
from src.core.taught_facts import (
    keep_taught_fact, prepare_taught_fact, taught_turn, teaching_is_empty,
)
from src.core.turn_notes import is_recall_question, recall_last, reply_notes
from src.core.usage_answer import detect_usage_question, previous_usage_answer
from src.core.self_facts import detect_self_question, self_answer
from src.core.save_code import (
    is_finder_request, parse_request as code_save_request, save_last_code, show_saved_in_finder,
    is_saved_location_request, last_saved_folder, saved_location, save_web_document,
)

# Regex dispatch patterns extracted to src/core/dispatch/ so this file
# stays focused on routing. The constants are re-bound as class
# attributes below for back-compat with tests reaching in via
# `CortexOrchestrator._SPOTIFY_PAUSE_RE` etc.
from src.core.dispatch.spotify import (
    SPOTIFY_DEVICE_HINT_RE,
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
from src.core.dispatch.reminders import REMINDER_BODY_RE, REMINDER_TRIGGER_RE
from src.core.dispatch.notes import (
    NOTE_APPEND_RE,
    NOTE_APPEND_TAIL_RE,
    NOTE_CREATE_FOLDER_RE,
    NOTE_CREATE_RE,
    NOTE_LIST_RE,
    NOTE_SEARCH_RE,
)
from src.core.dispatch.calendar import (
    CALENDAR_TRIGGER_CONTENT_RE,
    CALENDAR_VERB_RE,
    CALENDAR_TRIGGER_RE,
    VAGUE_CALENDAR_RE,
)
from src.core.dispatch.undo import UNDO_RE
from src.core.dispatch.digest import WHATS_CHANGED_RE
from src.core.dispatch.system_status import SYSTEM_STATUS_RE
from src.core.dispatch.entities import LAST_CONTACT_RE, RECENT_PEOPLE_RE
from src.core.dispatch.deep_research import DEEP_RESEARCH_RE
from src.core.dispatch.dream import DREAM_REVIEW_RE
from src.core.dispatch import registry as dispatch_registry
from src.core.dispatch.planner_trigger import PLANNER_TRIGGER_RE
from src.core.dispatch.ps5 import PS5_RE, ps5_phrase_language
from src.core.dispatch.flow_control import (
    ALL_REF_RE,
    APPROVAL_RE,
    DENIAL_RE,
    SILENCE_RE,
)
from src.core.dispatch.heuristics import (
    MATH_ONLY_RE,
    OFFLINE_HINT_RE,
    REALTIME_MARKER_RE,
    REAL_TOOL_HINT_RE,
    TOOL_HINT_RE,
    VOLATILE_FACT_RE,
    multi_intent_match,
)
from src.core.dispatch.triggers import (
    READ_THIS_FILE_RE,
    READ_VERB_RE,
    CONTACT_ADD_RE,
    CONTACT_INFO_RE,
    CONTACT_LOOKUP_RE,
    CONTACT_SAVE_RE,
    CREATIVE_RE,
    LIST_EMAILS_RE,
    READ_RE,
    SEARCH_STRIP_RE,
    SEND_EMAIL_BY_NAME_RE,
    SEND_EMAIL_STRUCTURED_RE,
    WRITE_RE,
)
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
)
from src.core.dispatch.normalisation import (
    normalise_stt_aliases,
    strip_courtesy,
    removed_portuguese_courtesy_suffix,
    strip_trailing_sentence_punctuation,
    strip_vocatives,
)
from src.core.dispatch.macos import (
    MAC_BRIGHTNESS_DOWN_RE,
    MAC_BRIGHTNESS_SET_RE,
    MAC_BRIGHTNESS_UP_RE,
    MAC_KEYBOARD_BRIGHTNESS_DOWN_RE,
    MAC_KEYBOARD_BRIGHTNESS_SET_RE,
    MAC_KEYBOARD_BRIGHTNESS_UP_RE,
    MAC_DARK_OFF_RE,
    MAC_DARK_ON_RE,
    MAC_DARK_TOGGLE_RE,
    MAC_LOCK_RE,
    MAC_APP_LIST_RE,
    MAC_REPLY_EMAIL_RE,
    MAC_SUMMARISE_RE,
    MAC_WHAT_IS_RE,
    MAC_EYES_OFF_RE,
    MAC_EYES_ON_RE,
    MAC_APP_SWITCH_RE,
    MAC_CLOSE_WINDOW_RE,
    MAC_MUTE_RE,
    MAC_OPEN_APP_RE,
    MAC_QUIT_APP_RE,
    MAC_SCREENSHOT_RE,
    MAC_SLEEP_RE,
    MAC_UNMUTE_RE,
    MAC_VOLUME_DOWN_RE,
    MAC_VOLUME_SET_RE,
    MAC_VOLUME_UP_RE,
)
from src.core.staging_mixin import StagingMixin
from src.core.notes_mixin import NotesMixin
from src.core.weather_mixin import WeatherMixin
from src.core.dispatch.weather import WEATHER_RE
from src.core.dispatch.fitness import (
    FITNESS_LOG_MEAL_RE, FITNESS_MACROS_TODAY_RE,
    FITNESS_LIST_TODAY_RE, FITNESS_UNDO_MEAL_RE, FITNESS_CLEAR_TODAY_RE,
    FITNESS_LOG_RUN_RE, FITNESS_LOG_LIFT_RE,
    FITNESS_RUN_TREND_RE, FITNESS_LIFT_TREND_RE,
    FITNESS_UNDO_WORKOUT_RE, FITNESS_CLEAR_WORKOUTS_RE, FITNESS_DELETE_GUARD_RE,
)
from src.core.dispatch.jobs import JOBS_RERANK_RE, JOBS_TOP_RE
from src.core.jobs_mixin import JobsMixin
from src.core.refine_mixin import RefineMixin
from src.core.dispatch.refine import REFINE_RE
from src.core.contacts_mixin import ContactsMixin
from src.core.studier_mixin import StudierMixin
from src.core.email_mixin import EmailMixin
from src.core.screen_context_mixin import ScreenContextMixin
from src.core.entities_mixin import EntitiesMixin
from src.core.features_mixin import FeaturesMixin
from src.core.fitness_mixin import FitnessMixin
from src.core.focus_mixin import FocusMixin
from src.core.projects_mixin import ProjectsMixin
from src.core.named_file_mixin import NamedFileMixin
from src.core.scheduling_mixin import SchedulingMixin
from src.core.system_control_mixin import SystemControlMixin
from src.core.i18n import I18nMixin
from src.tools.local_reader import LocalFileReader
from src.tools.local_writer import LocalFileWriter
from src.tools.web_search import WebSearcher
from src.tools.vision_processor import VisionProcessor
from src.tools.spotify_player import SpotifyPlayer
from src.tools.macos_controller import MacOSController
from src.integrations import notes_client

load_dotenv()

log = logging.getLogger(__name__)


def _lease_hours_from_env() -> int:
    """CORTEX_LEASE_HOURS as int, defaulting to 4. M1-A12b: a malformed
    value ("4h", "quatro", "") must not crash the BOOT of the whole
    server — fall back and log, don't raise inside __init__."""
    raw = os.getenv("CORTEX_LEASE_HOURS", "4")
    try:
        return int(raw)
    except ValueError:
        log.warning("CORTEX_LEASE_HOURS malformed (%r) — using 4", raw)
        return 4


# Language detection lives in src.core.i18n since s69 (M10-A11) so the
# UI layer can share it without importing this module. Re-imported under
# the original private names: every in-module reference and external
# `orchestrator._detect_lang` import keeps working unchanged.
from src.core.i18n import _detect_lang, _resolve_lang  # noqa: E402


_PROMPTS = {
    "needs_internet": {
        "pt": "Essa pergunta precisa de dados atuais da internet. Posso liberar acesso por {h}h? (responda 'sim' ou 'não')",
        "en": "This question needs current internet data. May I grant access for {h} hours? (reply 'yes' or 'no')",
    },
    "lease_granted": {
        "pt": "Internet liberada até {time}{tz_suffix}.",
        "en": "Internet granted until {time}{tz_suffix}.",
    },
    "lease_revoked": {
        "pt": "Internet desligada.",
        "en": "Internet disconnected.",
    },
    "stale_action_discarded": {
        "pt": "Tinha uma ação antiga aguardando confirmação e descartei "
              "por segurança (passou de {m} minutos). Me pede de novo se "
              "ainda quiser.",
        "en": "There was an old action awaiting confirmation and I "
              "discarded it for safety (it was over {m} minutes old). "
              "Ask me again if you still want it.",
    },
    "already_offline": {
        "pt": "Já estou offline. Nada a desligar.",
        "en": "Already offline. Nothing to disconnect.",
    },
    "search_needs_gate": {
        "pt": "Pesquisa requer acesso à internet. Posso liberar por {h}h? (responda 'sim' ou 'não')",
        "en": "Web search needs internet access. May I grant it for {h} hours? (reply 'yes' or 'no')",
    },
    "offline_denied": {
        "pt": "Entendido, sigo offline.",
        "en": "Got it, staying offline.",
    },
    # Spoken when a reply to a pending internet-approval prompt is too
    # short/garbled to read as sim OR não (s76: Whisper turned a spoken
    # "yes" into "Ah!"). Re-asks instead of silently dropping the pending
    # query; AWAITING_APPROVAL reopens the voice follow-up window.
    "approval_unclear": {
        "pt": "Não entendi — sim ou não?",
        "en": "I didn't catch that — yes or no?",
    },
    "auto_search_prefix": {
        "pt": "[Internet liberada até {time} UTC] ",
        "en": "[Internet granted until {time} UTC] ",
    },
    # Spoken when the engine dies mid-stream. This text reaches the TTS,
    # so it must be a short speakable sentence — the technical detail goes
    # to the audit log, never to the voice (M5-A3/M10-A5).
    "engine_error": {
        "pt": "Tive um problema pra gerar a resposta. Tenta de novo.",
        "en": "I hit a problem generating the reply. Please try again.",
    },
}


# Bug s73 (Copa 2026): the 8B reads "próximo jogo" as "the SECOND item in
# the list" (next-after-the-first), not "the event closest to today" — it
# answered Haiti (2nd fixture) over snippets that clearly led with the
# Marrocos debut. Probing showed NO prompt instruction fixes this (date
# anchor, worked example, clarifier note, even code-annotated day deltas all
# still answered Haiti; "primeiro a partir de hoje" answers Marrocos on every
# seed). So the word itself is rewritten deterministically before injection.
# SINGULAR only ("próximos jogos" asks for a list — no selection to get
# wrong) and never before a time unit ("próxima semana" must survive).
_NEXT_EVENT_TIME_UNITS = (
    r"semana|m[eê]s|ano|dia|vez|hora|minuto|"
    r"segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado|domingo|feira"
)
_NEXT_EVENT_PT_RE = re.compile(
    rf"\bpr[oó]xim(o|a)\b(?!s)(?!\s+(?:{_NEXT_EVENT_TIME_UNITS})\b)",
    re.IGNORECASE,
)


def rewrite_next_event_query(text: str) -> str:
    """Replace singular "próximo/próxima <event>" with the phrasing the 8B
    actually resolves correctly: "primeiro/primeira (contando a partir de
    hoje) <event>". Pure; returns the text unchanged when nothing matches."""
    def _repl(m: "re.Match[str]") -> str:
        first = "primeira" if m.group(1).lower() == "a" else "primeiro"
        return f"{first} (contando a partir de hoje)"
    return _NEXT_EVENT_PT_RE.sub(_repl, text)


def build_rag_prompt(
    web_context: str, text: str, today_line: Optional[str] = None, market=None,
) -> str:
    """The ONLINE_RAG synthesis prompt — pure, so evals can build it with
    frozen snippets/date and grade the model on it (anti-hallucination RAG).
    market is an optional frozen SearchMarket; configuration is read by the caller.

    Bug s73 (Copa 2026: "próximo jogo" → answered the SECOND fixture): the
    snippets were correct, but selecting "the next one" from a list of dated
    events needs today's date ADJACENT to the data. The engine system prompt
    does carry the date, but far from the snippets — the 8B didn't anchor on
    it. Re-state it here, instruct the temporal selection explicitly, and —
    the part that actually flips the answer — rewrite "próximo" in the query
    (see rewrite_next_event_query above).
    """
    if today_line is None:
        today_line = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
    text = rewrite_next_event_query(text)
    policy = (
        "COMPARISONS: Compare the same attribute; never call one better based "
        "on different attributes. Each fact belongs only to the item the source "
        "names for it. Never copy a number or feature to the other item; use "
        "\"também\"/\"both\" only if the data states it for both. "
        "If an attribute is given for only one, state it and say the data does "
        "not give it for the other. Data: A charges at Y W; B lasts X hours. "
        "WRONG: A e B duram X horas. RIGHT: A carrega a Y W; B dura X horas; "
        "os dados não dão a duração de A, então não dá para dizer qual é melhor."
    )
    if market is None:
        policy += (
            "\nCURRENCIES: Never convert or estimate currencies; keep the currency "
            "stated by the source."
        )
    else:
        policy = (
            f"PRICES: The user's market is {market.name_en} ({market.currency}, "
            f"{market.symbol}). State a price only if the data gives it for that "
            "market in that currency. Omit other countries' or currencies' "
            f"prices; instead say PT: \"Não achei o preço {market.location_pt}.\" / "
            f"EN: \"I couldn't find the price in {market.name_en}.\" "
            "Never convert or estimate currencies.\n"
        ) + policy
    # Web snippets are UNTRUSTED external content (OWASP LLM01): a scraped page
    # can carry text aimed at the model. Fence it + a guard line (same helper
    # the studier/screen-context use); fence() strips forged markers too. The
    # temporal-selection instructions below are unchanged.
    return (
        f"TODAY'S DATE: {today_line}\n\n"
        f"{guard_line('en')}\n\n"
        f"LIVE WEB DATA:\n{fence(web_context)}\n\n"
        f"USER QUERY: {text}\n"
        "INSTRUCTION: Answer the query using ONLY the provided live web data. "
        "When the query asks for the NEXT/upcoming event (PT: \"próximo\") and "
        "the data lists several dated events, compare each date against "
        "TODAY'S DATE above and pick the EARLIEST one that has not happened "
        "yet — never a later one. "
        "When the query asks for the LAST/most-recent event (PT: \"último\", "
        "e.g. \"resultado do último jogo\") and the data lists several dated "
        "events, pick the one with the LATEST date that is ON OR BEFORE "
        "TODAY'S DATE — the most recent that has ALREADY happened — never an "
        "older one and never a future fixture. If a short 'QUICK ANSWER' "
        "conflicts with the dated list of results, TRUST THE DATED LIST. "
        "IMPORTANT — stale 'next' games: a fixture the data labels as "
        "UPCOMING/NEXT (PT: \"próximo\") whose date is ALREADY ON OR BEFORE "
        "TODAY'S DATE has in fact ALREADY been played — web indexes lag a day "
        "behind fresh results. For a LAST/most-recent query, THAT game is the "
        "most recent one: name it and its date. If its score/result is NOT in "
        "the data, SAY the result isn't in the results yet (PT: \"o jogo mais "
        "recente foi contra X em DATA, mas o placar ainda não aparece nos "
        "resultados\") rather than reporting an OLDER game as if it were the "
        "last. "
        "Do NOT invent or guess any specific fact — dates, times, fixtures, "
        "opponents, venues, scores, standings, names, numbers or prices — that "
        "is not explicitly stated in the LIVE WEB DATA above. If the data does "
        "not clearly contain the exact detail the query asks for, say so plainly "
        "(PT: \"Não encontrei essa informação nos resultados.\" / EN: \"I couldn't "
        "find that in the results.\") instead of filling it in from memory. It is "
        "better to admit the gap than to state an unverified fact as if it were true. "
        "Reply in the exact same language as the USER QUERY. Be concise. "
        "ALWAYS USE METRIC UNITS in your answer regardless of what the "
        "source provides: Celsius for temperature (NEVER Fahrenheit), "
        "km/h for wind speed (NEVER mph), mm for precipitation, km for "
        "distance, kg for weight, hPa/mb for pressure. If the source "
        "shows both metric and imperial, just pick the metric value. "
        "If only imperial is given, convert. Use metric units consistently."
        f"\n{policy}"
    )


# "… in London?" / "… em Londres" — read ONLY after the semantic router has
# decided the intent is weather (see _semantic_weather).
_CITY_TAIL_RE = re.compile(
    r"\b(?:in|em|no|na)\s+(?P<city>[A-Za-zÀ-ÿ][\w'\-]*"
    r"(?:\s+(?!hoje\b|agora\b|amanh|today\b|now\b|tomorrow\b)"
    r"[A-Za-zÀ-ÿ][\w'\-]*)?)"
    r"(?:\s+(?:hoje|agora|amanh[ãa]|today|now|tomorrow))?[\s.!?]*$",
    re.IGNORECASE,
)


def _drop_unusable_file_read(specs: list, seen_paths) -> list:
    """Hide read_file_at until the conversation HAS a path.

    Offered from the first turn it out-competed find_file: "cadê o pdf de
    punctuation?" made the 8B invent /Users/.../punctuation.pdf and ask to
    open it (s83). A tool that cannot legally be used yet — the executor
    refuses a path nobody has seen — has no business on the menu. Same move
    as the s38 anti-fabrication gate: the model cannot pick what it is not
    offered. A path the user types is harvested before dispatch, so stating
    one still works on the first turn."""
    if seen_paths:
        return specs
    return [spec for spec in specs
            if spec.get("function", {}).get("name") != "read_file_at"]


def _partial_read_note(how: str, lang: str = "pt") -> str:
    """Said out loud whenever the answer came from PART of a document — and
    which part. An answer drawn from a slice of a longer file is a partial
    answer, and the model has no way to know it was given a slice (s83).

    `how` comes from doc_retrieval.select_relevant: "whole" (nothing to say),
    "retrieved" (the passages closest to the question) or "clipped" (the
    beginning, because retrieval could not run)."""
    if how == "retrieved":
        return ("\n\n(O documento é maior do que cabe de uma vez, então "
                "busquei nele os trechos que respondem à sua pergunta.)"
                if lang != "en" else
                "\n\n(The document is longer than fits at once, so I searched "
                "it for the passages that answer your question.)")
    if how == "clipped":
        return ("\n\n(Li só a primeira parte do documento — ele é maior do que "
                "cabe na janela do modelo. Me pergunta por uma seção específica "
                "se o que você precisa estiver mais adiante.)"
                if lang != "en" else
                "\n\n(I read only the first part of the document — it is longer "
                "than the model's window. Ask me about a specific section if what "
                "you need is further in.)")
    return ""


class CortexOrchestrator(
    StagingMixin, NotesMixin, ContactsMixin, StudierMixin, EmailMixin,
    ScreenContextMixin, EntitiesMixin, FeaturesMixin, SchedulingMixin,
    SystemControlMixin, WeatherMixin, FitnessMixin, JobsMixin, RefineMixin,
    FocusMixin, ProjectsMixin, NamedFileMixin, I18nMixin,
):
    """
    Single brain shared by the CLI (src/main.py) and the HTTP API (src/api.py).
    Routes each request to the right tool/engine and returns (reply, status_mode).

    Inherits the full staged Agent-action surface from `StagingMixin`
    (`src/core/staging_mixin.py`):
      - load / persist / execute / cancel / preview of pending actions
      - tool dispatch (`_execute_tool`, `_execute_tool_unstaged`)
      - planner execution (`_execute_staged_plan`)
    """

    # Generic command regexes live in src/core/dispatch/ — split by
    # function: triggers.py (read/write/creative/search/list-emails),
    # flow_control.py (approval/denial/silence/all-ref), and
    # heuristics.py (tool/offline/realtime hints). Studier verbs in
    # studier_cmds.py. All re-bound here as class attrs so existing
    # tests reaching in via `CortexOrchestrator._READ_RE`,
    # `._APPROVAL_RE`, etc. keep working unchanged.
    # Edit the patterns in their modules — NOT here.
    _CANCEL_STUDY_RE = CANCEL_STUDY_RE
    _CANCEL_VAGUE_RE = CANCEL_VAGUE_RE
    _LAST_STUDY_RE = LAST_STUDY_RE
    _READ_RE = READ_RE
    _READ_THIS_FILE_RE = READ_THIS_FILE_RE
    _READ_VERB_RE = READ_VERB_RE
    _WRITE_RE = WRITE_RE
    _CREATIVE_RE = CREATIVE_RE
    _SEARCH_STRIP_RE = SEARCH_STRIP_RE
    _APPROVAL_RE = APPROVAL_RE
    _PS5_RE = PS5_RE
    _DENIAL_RE = DENIAL_RE
    _TOOL_HINT_RE = TOOL_HINT_RE
    _REAL_TOOL_HINT_RE = REAL_TOOL_HINT_RE
    _OFFLINE_HINT_RE = OFFLINE_HINT_RE
    _REALTIME_MARKER_RE = REALTIME_MARKER_RE
    _MATH_ONLY_RE = MATH_ONLY_RE
    _VOLATILE_FACT_RE = VOLATILE_FACT_RE
    _ALL_REF_RE = ALL_REF_RE
    _LIST_EMAILS_RE = LIST_EMAILS_RE
    _SEND_EMAIL_STRUCTURED_RE = SEND_EMAIL_STRUCTURED_RE
    _SEND_EMAIL_BY_NAME_RE = SEND_EMAIL_BY_NAME_RE
    _CONTACT_SAVE_RE = CONTACT_SAVE_RE
    _CONTACT_ADD_RE = CONTACT_ADD_RE
    _CONTACT_INFO_RE = CONTACT_INFO_RE
    _CONTACT_LOOKUP_RE = CONTACT_LOOKUP_RE
    _SILENCE_RE = SILENCE_RE

    # Nightly Studier verbs.
    _STUDY_RE = STUDY_RE
    _ADD_STUDY_RE = ADD_STUDY_RE
    _LIST_STUDIES_RE = LIST_STUDIES_RE
    _REVIEW_RE = REVIEW_RE
    _SHOW_STUDY_RE = SHOW_STUDY_RE
    _APPROVE_STUDY_RE = APPROVE_STUDY_RE
    _REJECT_STUDY_RE = REJECT_STUDY_RE

    # Spotify intent regexes (added 2026-05-19). Regex-first dispatch keeps
    # the common "pausa / próxima / volume X" path zero-LLM: ~500ms total
    # latency vs 8-19s if routed through Llama. Complex queries that don't
    # match these can still go through tool-call fallback later.
    # Regex constants live in src/core/dispatch/ — re-bound here as
    # class attrs so existing tests (CortexOrchestrator._SPOTIFY_PAUSE_RE,
    # ._REMINDER_TRIGGER_RE, etc.) keep working unchanged. Edit the
    # patterns in src/core/dispatch/{spotify,reminders,calendar,
    # planner_trigger}.py — NOT here.
    _SPOTIFY_PLAY_RE = SPOTIFY_PLAY_RE
    _SPOTIFY_LIBRARY_ARTIST_RE = SPOTIFY_LIBRARY_ARTIST_RE
    _SPOTIFY_PLAY_PLAYLIST_RE = SPOTIFY_PLAY_PLAYLIST_RE
    _SPOTIFY_PLAY_PLAYLIST_EN_RE = SPOTIFY_PLAY_PLAYLIST_EN_RE
    _SPOTIFY_PLAY_RADIO_RE = SPOTIFY_PLAY_RADIO_RE
    _SPOTIFY_PLAY_RADIO_EN_RE = SPOTIFY_PLAY_RADIO_EN_RE
    _SPOTIFY_DEVICE_HINT_RE = SPOTIFY_DEVICE_HINT_RE
    _SPOTIFY_PAUSE_RE = SPOTIFY_PAUSE_RE
    _SPOTIFY_RESUME_RE = SPOTIFY_RESUME_RE
    _SPOTIFY_NEXT_RE = SPOTIFY_NEXT_RE
    _SPOTIFY_PREV_RE = SPOTIFY_PREV_RE
    _SPOTIFY_NOW_RE = SPOTIFY_NOW_RE
    _SPOTIFY_VOLUME_RE = SPOTIFY_VOLUME_RE
    _SPOTIFY_LIKE_RE = SPOTIFY_LIKE_RE
    _SPOTIFY_TRANSFER_RE = SPOTIFY_TRANSFER_RE

    _REMINDER_TRIGGER_RE = REMINDER_TRIGGER_RE
    _REMINDER_BODY_RE = REMINDER_BODY_RE
    _NOTE_CREATE_RE = NOTE_CREATE_RE
    _NOTE_LIST_RE = NOTE_LIST_RE
    _NOTE_SEARCH_RE = NOTE_SEARCH_RE
    _NOTE_APPEND_RE = NOTE_APPEND_RE
    _NOTE_APPEND_TAIL_RE = NOTE_APPEND_TAIL_RE
    _NOTE_CREATE_FOLDER_RE = NOTE_CREATE_FOLDER_RE
    _WEATHER_RE = WEATHER_RE
    _FITNESS_LOG_MEAL_RE = FITNESS_LOG_MEAL_RE
    _FITNESS_MACROS_TODAY_RE = FITNESS_MACROS_TODAY_RE
    _FITNESS_LIST_TODAY_RE = FITNESS_LIST_TODAY_RE
    _FITNESS_UNDO_MEAL_RE = FITNESS_UNDO_MEAL_RE
    _FITNESS_CLEAR_TODAY_RE = FITNESS_CLEAR_TODAY_RE
    _FITNESS_LOG_RUN_RE = FITNESS_LOG_RUN_RE
    _FITNESS_LOG_LIFT_RE = FITNESS_LOG_LIFT_RE
    _FITNESS_RUN_TREND_RE = FITNESS_RUN_TREND_RE
    _FITNESS_LIFT_TREND_RE = FITNESS_LIFT_TREND_RE
    _FITNESS_UNDO_WORKOUT_RE = FITNESS_UNDO_WORKOUT_RE
    _FITNESS_CLEAR_WORKOUTS_RE = FITNESS_CLEAR_WORKOUTS_RE
    _FITNESS_DELETE_GUARD_RE = FITNESS_DELETE_GUARD_RE
    _JOBS_TOP_RE = JOBS_TOP_RE
    _JOBS_RERANK_RE = JOBS_RERANK_RE

    _UNDO_RE = UNDO_RE
    _WHATS_CHANGED_RE = WHATS_CHANGED_RE
    _SYSTEM_STATUS_RE = SYSTEM_STATUS_RE
    _LAST_CONTACT_RE = LAST_CONTACT_RE
    _RECENT_PEOPLE_RE = RECENT_PEOPLE_RE
    _DEEP_RESEARCH_RE = DEEP_RESEARCH_RE
    _DREAM_REVIEW_RE = DREAM_REVIEW_RE

    _PLANNER_TRIGGER_RE = PLANNER_TRIGGER_RE
    _REFINE_RE = REFINE_RE

    _CALENDAR_TRIGGER_RE = CALENDAR_TRIGGER_RE
    _CALENDAR_TRIGGER_CONTENT_RE = CALENDAR_TRIGGER_CONTENT_RE
    _CALENDAR_VERB_RE = CALENDAR_VERB_RE
    _VAGUE_CALENDAR_RE = VAGUE_CALENDAR_RE

    # macOS system control (sessão 10 bloco G — computer control
    # batch 1, caminho A skill-based). Regex-first dispatch keeps
    # each verb at osascript latency (~150ms) without going through
    # the LLM. All actions are S2 (audit-only, no confirmation) —
    # reversible (volume/dark/lock) or purely additive (screenshot,
    # open app). S3 verbs (app-quit, sleep) are deferred — see
    # JOURNAL bloco G. Source regexes in src/core/dispatch/macos.py.
    _MAC_VOLUME_SET_RE = MAC_VOLUME_SET_RE
    _MAC_VOLUME_UP_RE = MAC_VOLUME_UP_RE
    _MAC_VOLUME_DOWN_RE = MAC_VOLUME_DOWN_RE
    _MAC_MUTE_RE = MAC_MUTE_RE
    _MAC_UNMUTE_RE = MAC_UNMUTE_RE
    _MAC_BRIGHTNESS_UP_RE = MAC_BRIGHTNESS_UP_RE
    _MAC_BRIGHTNESS_DOWN_RE = MAC_BRIGHTNESS_DOWN_RE
    _MAC_BRIGHTNESS_SET_RE = MAC_BRIGHTNESS_SET_RE
    _MAC_KEYBOARD_BRIGHTNESS_UP_RE = MAC_KEYBOARD_BRIGHTNESS_UP_RE
    _MAC_KEYBOARD_BRIGHTNESS_DOWN_RE = MAC_KEYBOARD_BRIGHTNESS_DOWN_RE
    _MAC_KEYBOARD_BRIGHTNESS_SET_RE = MAC_KEYBOARD_BRIGHTNESS_SET_RE
    _MAC_DARK_TOGGLE_RE = MAC_DARK_TOGGLE_RE
    _MAC_DARK_ON_RE = MAC_DARK_ON_RE
    _MAC_DARK_OFF_RE = MAC_DARK_OFF_RE
    _MAC_LOCK_RE = MAC_LOCK_RE
    _MAC_SCREENSHOT_RE = MAC_SCREENSHOT_RE
    _MAC_OPEN_APP_RE = MAC_OPEN_APP_RE
    _MAC_CLOSE_WINDOW_RE = MAC_CLOSE_WINDOW_RE
    _MAC_QUIT_APP_RE = MAC_QUIT_APP_RE
    _MAC_SLEEP_RE = MAC_SLEEP_RE
    _MAC_APP_LIST_RE = MAC_APP_LIST_RE
    _MAC_APP_SWITCH_RE = MAC_APP_SWITCH_RE
    _MAC_REPLY_EMAIL_RE = MAC_REPLY_EMAIL_RE
    _MAC_SUMMARISE_RE = MAC_SUMMARISE_RE
    _MAC_WHAT_IS_RE = MAC_WHAT_IS_RE
    _MAC_EYES_OFF_RE = MAC_EYES_OFF_RE
    _MAC_EYES_ON_RE = MAC_EYES_ON_RE

    # Status modes for which the morning announcement of pending studies
    # is suppressed: administrative commands, error paths, and approval
    # flows where injecting an unrelated reminder would derail the UX.
    _SILENT_ANNOUNCE_MODES = frozenset({
        "STUDY_QUEUED", "STUDY_LIST",
        "STUDIES_REVIEW", "STUDY_SHOW",
        "STUDY_APPROVE", "STUDY_REJECT", "STUDY_AMBIGUOUS",
        "STUDY_NOT_FOUND", "STUDY_APPROVE_ERROR", "STUDY_REJECT_ERROR",
        "AWAITING_APPROVAL", "APPROVAL_DENIED",
        "STALE_ACTION_DISCARDED",  # administrative ack (M1-A9)
        "SYSTEM_UPDATE",  # lease grant / revoke ack
        "INPUT_ERROR", "INTERNAL_ERROR",
        "FILE_ERROR", "FILE_LIST", "FILE_OPEN", "SEARCH_ERROR",
        "NOTHING_PENDING",
        "RECALL_LAST", "SAVE_CODE", "SHOW_CODE_FINDER", "SHOW_CODE_LOCATION",  # literal replies
        "USAGE_PREVIOUS",  # fixed telemetry reply; no unrelated postscript
        "SELF_MODEL", "SELF_PLAN", "SELF_AVATAR",  # fixed product facts
        # Focus sessions (s86): short fixed replies at the moment of starting
        # or stopping — a morning postscript there is the distraction the
        # feature exists to remove.
        "FOCUS_ASK_STEP", "FOCUS_STARTED", "FOCUS_CANCELLED",
        "FOCUS_ASK_SWITCH", "FOCUS_KEPT", "FOCUS_CLOSED", "FOCUS_ASK_NEXT",
        "FOCUS_NEXT_SAVED", "FOCUS_NONE", "FOCUS_WHERE", "FOCUS_ERROR",
        "PROJECT_ACTIVE", "PROJECT_LIST", "PROJECT_CLEARED", "PROJECT_ERROR",
        "PROJECT_NOTED", "PROJECT_NONE", "PROJECT_WHERE", "PROJECT_RENAMED", "PROJECT_TAKEN",
        # Agent (Gmail / Calendar) confirmation flow
        "AGENT_AWAITING_CONFIRMATION",
        "AGENT_ACTION_DONE", "AGENT_ACTION_CANCELLED", "AGENT_ACTION_ERROR",
        "AGENT_STAGE_CONFLICT", "AGENT_ACTION_STALE",
        "GMAIL_LIST", "GMAIL_NOT_CONFIGURED", "GMAIL_ERROR",
        "CALENDAR_LIST", "CALENDAR_ERROR",
        # Spotify intent dispatch — fire-and-forget action acks
        "SPOTIFY_PAUSE", "SPOTIFY_RESUME", "SPOTIFY_NEXT", "SPOTIFY_PREV",
        "SPOTIFY_NOW", "SPOTIFY_VOLUME", "SPOTIFY_LIKE", "SPOTIFY_PLAY",
        "SPOTIFY_PLAY_PLAYLIST", "SPOTIFY_PLAY_RADIO", "SPOTIFY_TRANSFER",
        # Skill Fast Path — registry-driven deterministic replies
        "SKILL",
        # Speech interrupt
        "SPEECH_INTERRUPTED",
        # Prompt enrichment staging (idea #30) — approval flow, same
        # rationale as AGENT_AWAITING_CONFIRMATION.
        "REFINE_STAGED", "REFINE_ERROR",
    })

    # Tools that must NEVER auto-execute live behind `is_destructive()`
    # from src/core/tool_risk.py — single source of truth for the
    # S1/S2/S3 classification (Tier 3.4). The dispatch branch in
    # `_execute_tool` calls _is_destructive_tool(name); previews are
    # built by _build_dry_run_preview(name, args) (Tier 3.3).

    _LAST_ANNOUNCE_PATH = "data/last_announce.txt"

    # _pending_action persistence — survives uvicorn restarts so a staged
    # email/event/reminder doesn't vanish when the user replies "sim" / "não".
    _PENDING_ACTION_FILE = "data/pending_action.json"

    _INTENT_CACHE_MAX = 256

    # Questions and permissions belong to the conversation that received them.
    _offer = PerConversation("_offer", None)
    _pending_query = PerConversation("_pending_query", None)
    _pending_engine = PerConversation("_pending_engine", None)
    _pending_query_reprompted = PerConversation("_pending_query_reprompted", False)
    _last_listed = PerConversation("_last_listed", None)
    _pdf_split_question = PerConversation("_pdf_split_question", None)

    def __init__(self):
        self.gatekeeper = LogicGate()
        self.engine_analytical = OllamaEngine(
            model_name=os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite"),
            max_history_pairs=5,
        )
        self.engine_creative = OllamaEngine(
            model_name=os.getenv("CORTEX_MODEL_CREATIVE", "cortex-criativo"),
            max_history_pairs=5,
            # The Modelfile's intended 0.7 — the per-request option used to
            # hardcode the analytical 0.3 here too, neutralising creative
            # mode since s38 (M2-A6).
            default_temperature=float(os.getenv("CORTEX_CREATIVE_TEMPERATURE", "0.7")),
        )
        self.reader = LocalFileReader(base_dir="data/documents")
        self.writer = LocalFileWriter(base_dir="data/documents")
        self.web_searcher = WebSearcher()
        self.vision = VisionProcessor(
            model_name=os.getenv("CORTEX_MODEL_VISION", "llava")
        )
        self.lease_hours = _lease_hours_from_env()
        # Cognitive Arbiter shadow mode: when on, classify
        # every query and log the route distribution WITHOUT changing
        # routing. Off by default — dormant skeleton ahead of Mac Studio.
        self._arbiter_shadow: bool = os.getenv(
            "CORTEX_ARBITER_SHADOW", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._intent_cache: "OrderedDict[str, bool]" = OrderedDict()
        # Pending query: stored when classifier says SIM but no lease is active.
        # Resolved on the next turn if the user approves or denies.
        self._pending_query: Optional[str] = None
        self._pending_engine: Optional[OllamaEngine] = None
        # One re-prompt per pending query when the reply is too garbled to
        # read as sim/não (s76 smoke — see the branch in _process_core).
        self._pending_query_reprompted: bool = False
        # Pending Agent action (send_email / create_event). Set when the LLM
        # picks a destructive tool; resolved on the next turn by the user's
        # "sim" or "não". Takes precedence over _pending_query.
        self._pending_action: Optional[dict] = self._load_pending_action()
        self._abandoned_effects_lock = threading.Lock()
        self._abandoned_effects = []
        try:
            from src.core import effect_store
            effect_store.store.reconcile_abandoned()
            self._abandoned_effects = effect_store.store.pending_notices()
        except Exception:
            log.exception("orchestrator: could not reconcile abandoned effects")
        # Background mode: set True by non-interactive callers (a nightly
        # scheduled task, future Dream Loop). When True, a destructive (S3)
        # tool the LLM picks is routed to the gate batch queue for morning
        # review instead of the inline "sim/não" slot — there's no human to
        # answer at 3am, and it must not clobber an interactive user's
        # staged action. `background_origin` labels who produced it.
        self.background: bool = False
        self.background_origin: str = "background"
        # Live activity feed (s36): an optional callback the streaming /ui
        # endpoint installs to observe pipeline stages in real time
        # ("web_search", "thinking", "reading", …) and surface a Claude-style
        # "what I'm doing now" line. The listener is THREAD-LOCAL: each turn's
        # process() runs in its own worker thread and installs its emitter
        # there, so a concurrent voice turn (different thread) never sees the
        # UI's emitter — no cross-talk, no shared-state clobber. Unset →
        # _emit_status is a no-op, so every non-UI caller behaves EXACTLY as
        # before. Presentation (emoji labels) lives in the UI layer; the
        # orchestrator only emits semantic stage names.
        self._status_local = threading.local()
        # Nightly Studier — queue of topics the user asked to study overnight
        self.study_queue = StudyQueue()
        # Nightly Studier — review/approval layer over generated briefings
        self.studies_review = StudiesReview()
        # Spotify Web API wrapper (lazy auth — safe to construct without creds)
        self.spotify = SpotifyPlayer()
        # macOS system controller (volume / brightness / dark mode /
        # lock / screenshot / app launch). Stateless wrapper around
        # osascript + open + screencapture. See JOURNAL sessão 10
        # bloco G for the verb list and deferred items.
        self.macos = MacOSController()
        # Apple Notes (local AppleScript, S2). Module bound as an attribute so
        # tests can swap it for a fake without touching osascript.
        self.notes = notes_client
        # Skill Fast Path Registry (Tier 3.2) — dynamic dispatch table
        # backed by the Palace `skills` table. Loaded lazily so a
        # missing Palace (test env, fresh install) doesn't crash boot.
        self.skill_registry = self._load_skill_registry()

    def _load_skill_registry(self):
        """Build a SkillRegistry sharing the engine's Palace. Returns
        None on any failure — the orchestrator must boot even if the
        registry is broken; skills are a fast-path optimisation, not a
        load-bearing dependency."""
        try:
            from src.core.skill_registry import SkillRegistry
            palace = getattr(self.engine_analytical.memory, "_palace", None)
            if palace is None:
                return None
            return SkillRegistry(palace)
        except Exception:
            log.exception("orchestrator: could not initialise SkillRegistry")
            return None

    # _load_pending_action / _persist_pending_action live on
    # StagingMixin (src/core/staging_mixin.py).

    @contextmanager
    def _conversation_scope(self, conversation_history: "Optional[list]"):
        """Bind a UI conversation's recent turns as the engines' short-term
        history for the duration of a turn (UI 2c), then unbind.

        `conversation_history` is a pre-loaded list of engine-message dicts
        (the ui layer reads it from the durable chat_store and hands it in, so
        core never depends on the ui/storage layer). **None vs []** matters:
          - None → no conversation context (voice, CLI, /ask, evals) → a no-op
            scope, keeping the global history exactly as before.
          - [] → a conversation that has no prior turns yet (its first message).
            Still bound — as an EMPTY, isolated override — so that first turn
            does NOT leak the global thread (which may hold another
            conversation's turns). Isolation is the whole point of 2c, and it
            has to hold from turn one, not only once a conversation has history.
        Bound on BOTH engines because the dispatch may route a turn to either;
        the override is thread-local, so this never disturbs a concurrent turn.
        Nesting is safe: process_stream binds, then its non-streamable path
        defers to process() with None → the inner scope is a no-op and leaves
        the outer binding intact."""
        if conversation_history is None:
            yield
            return
        engines = (self.engine_analytical, self.engine_creative)
        for e in engines:
            e.push_history_override(conversation_history)
        try:
            yield
        finally:
            for e in engines:
                try:
                    e.pop_history_override()
                except Exception:  # noqa: BLE001 — cleanup must never raise
                    pass

    @contextmanager
    def _project_scope(self, text: str):
        """Bind read-only project notes on both engines, on the executing thread.

        The streaming worker reopens this via process(), after restoring origin.
        """
        engines = []
        local = self.__dict__.setdefault("_project_local", threading.local())
        previous = getattr(local, "project", None)
        previous_context = getattr(local, "context", "")
        previous_request = getattr(local, "request_text", "")
        local.request_text = text
        kind = detect_guide(text)
        local.project = None
        local.context = ""

        def cleanup():
            while engines:
                e = engines.pop()
                try:
                    pop = getattr(e, "pop_turn_context", None)
                    if pop:
                        pop()
                    budget = getattr(e, "push_build_kind", None)
                    if budget:
                        budget(None)
                except Exception:  # Cleanup must never break a turn.
                    pass

        try:
            try:
                local.project = self._resolve_turn_project(text)
                context = self._project_context(text)
            except Exception:  # A project-store failure must not hide the guide.
                context = ""
            try:
                if kind:
                    context = (context + "\n\n" if context else "") + guide_text(kind)
                for name in ("engine_analytical", "engine_creative"):
                    e = getattr(self, name, None)
                    push = getattr(e, "push_turn_context", None)
                    if push:
                        # Include a push that binds context before raising.
                        engines.append(e)
                        push(context)
                        budget = getattr(e, "push_build_kind", None)
                        if budget:
                            budget(kind)
                local.context = context
            except Exception:  # Project context is optional for every turn.
                cleanup()
            with _action_ledger.observing(self._observe_action):
                yield
        finally:
            cleanup()
            local.project = previous
            local.context = previous_context
            local.request_text = previous_request

    @contextmanager
    def _web_request_scope(self, text: str):
        """Carry the original request through approved plans, without a global."""
        local = self.__dict__.setdefault("_project_local", threading.local())
        previous = getattr(local, "request_text", "")
        local.request_text = text
        try:
            yield
        finally:
            local.request_text = previous

    def _web_request_text(self) -> str:
        return getattr(getattr(self, "_project_local", None), "request_text", "")

    @contextmanager
    def _turn_reply_scope(self, text, taught, *, self_text=None):
        """Compose receipt/loss/self notes with the per-turn engine context."""
        note = reply_notes(text, taught,
                           getattr(self, "engine_analytical", None),
                           getattr(self, "engine_creative", None), self_text)
        if not note:
            yield
            return
        local = getattr(self, "_project_local", None)
        previous = getattr(local, "context", "")
        context = (previous + "\n\n" if previous else "") + note
        engines = []
        try:
            for name in ("engine_analytical", "engine_creative"):
                engine = getattr(self, name, None)
                push = getattr(engine, "push_turn_context", None)
                if push:
                    engines.append(engine)
                    push(context)
            yield
        finally:
            for engine in reversed(engines):
                try:
                    if previous:
                        engine.push_turn_context(previous)
                    else:
                        engine.pop_turn_context()
                except Exception:  # Cleanup must never break a turn.
                    pass

    def _stage_routed_command(self, text: str, mode: str, lang: str) -> bool:
        """Stage only the exact quoted command that the guard just simulated."""
        action = {"type": "routed_command", "args": {
            "text": text, "mode": mode, "lang": lang,
        }}
        self._pending_action = action
        self._persist_pending_action()
        staged = (self._pending_action is action
                  and not getattr(self._pending_turn(), "failed", False)
                  and not getattr(self._pending_turn(), "conflict", False))
        if staged:
            # Start before a pure stream's first yield: subsequent SSE pulls
            # no longer have the request's delivery bound to their thread.
            self._watch_delivery()
        return staged

    def process(
        self,
        text: str,
        image_b64: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: "Optional[list]" = None,
    ) -> Tuple[str, str]:
        """
        Public entry point. Delegates the routing to _process_core and then
        optionally appends a morning announcement of pending studies once
        per natural day.

        `language` overrides the regex heuristic when the caller knows
        which language the user is in (iOS Shortcut bound to PT or EN).
        Valid values: "pt", "en", or None (auto-detect).

        `conversation_history` (UI 2c): when the caller is the UI serving a
        specific chat conversation, this is that conversation's recent turns as
        engine messages — bound as the short-term thread for this turn only.
        None (voice/CLI/eval) keeps the global history.
        """
        text = strip_paste_markers(text or "")
        trace = audit.new_trace()
        audit.log(trace, "orchestrator", "request",
                  payload={
                      "input": (text or "")[:80],
                      "has_image": bool(image_b64),
                      "lang_override": language,
                  })
        t0 = time.perf_counter()
        # The pending slot comes back from disk and this turn remembers which
        # action it saw (s88, option A — staging_mixin). A staging refused
        # because another action got there first replaces the reply: the
        # preview built for it would show the OTHER action.
        self._begin_pending_turn()
        from src.core.hallucination import command_failure_scope
        try:
            with command_failure_scope(self._stage_routed_command), taught_turn(text, reuse=True) as taught, self._conversation_scope(conversation_history), self._project_scope(text):
                try:
                    reply, mode = self._process_core(text, image_b64, language=language)
                except Exception as exc:
                    if taught.result is None:
                        raise
                    audit.log(trace, "engine", "taught_turn_error",
                              payload={"error": str(exc)[:200]})
                    reply, mode = (_PROMPTS["engine_error"][taught.result.lang],
                                   "INTERNAL_ERROR")
                # A streamed client that never takes this preview gets its
                # action withdrawn (s88, lote 7 — staging_mixin).
                if getattr(self._pending_turn(), "failed", False):
                    self._pending_action = self._read_pending_raw()
                    reply, mode = (self._t(
                        "Não consegui guardar essa ação — me pede de novo.",
                        "I couldn't save that action — please ask again."),
                        "AGENT_STAGE_ERROR")
                else:
                    self._watch_delivery()
                if self._end_pending_turn():
                    reply, mode = self._stage_conflict_reply()
                if taught.result is None or taught.result.status != "empty":
                    reply = self._maybe_announce_pending(reply, mode)
                    reply = self._maybe_announce_abandoned(reply, mode)
                reply = taught.append_receipt(reply)
        finally:
            self._end_pending_turn()
            self._clear_reply_lang()
        audit.log(trace, "orchestrator", "response",
                  latency_ms=(time.perf_counter() - t0) * 1000,
                  payload={"mode": mode})
        return reply, mode

    def set_status_emitter(
        self, cb: "Optional[Callable[[str, str], None]]"
    ) -> None:
        """Install (or clear, with None) a live-activity listener for the
        CURRENT thread. Thread-local by design: the streaming /ui endpoint
        runs each turn in its own worker thread and calls this there, so a
        concurrent voice/UI turn in another thread is never affected. Lazily
        creates the thread-local for instances built via __new__ in tests."""
        local = getattr(self, "_status_local", None)
        if local is None:
            local = threading.local()
            self._status_local = local
        local.emitter = cb

    def set_token_emitter(
        self, cb: "Optional[Callable[[str], None]]"
    ) -> None:
        """Install (or clear, with None) a per-CHUNK listener for the CURRENT
        thread, so a slow LLM-synthesis step (ONLINE_RAG web search) can stream
        token by token to the UI instead of landing all at once after ~20s.

        Thread-local by the same reasoning as set_status_emitter: the streaming
        /ui endpoint runs each turn in its own worker thread and installs this
        there; a concurrent voice/UI turn in another thread is untouched. When
        no emitter is installed (CLI, /ask, voice) the synthesis stays a single
        blocking call — behaviour unchanged."""
        local = getattr(self, "_token_local", None)
        if local is None:
            local = threading.local()
            self._token_local = local
        local.emitter = cb

    def _token_emitter(self) -> "Optional[Callable[[str], None]]":
        local = getattr(self, "_token_local", None)
        return getattr(local, "emitter", None) if local is not None else None

    def _status_emitter(self) -> "Optional[Callable[[str, str], None]]":
        """This thread's live-activity listener, or None. The mirror of
        _token_emitter — needed so the streaming path can carry BOTH emitters
        across a thread boundary (see _process_stream_impl's deferral)."""
        local = getattr(self, "_status_local", None)
        return getattr(local, "emitter", None) if local is not None else None

    def _synthesise(self, engine, injected: str, lang: str, mode: str):
        """Run the LLM synthesis step of a RAG path, streaming it token by
        token when a token emitter is installed and blocking otherwise.

        Shared by the web path (_do_web_search) and the local-document path
        (_handle_read) so the two can't drift — and they had: the web path
        streamed while _handle_read ended in a blocking engine.chat(), so
        "resume esse documento" landed whole after the read plus the whole
        synthesis, with the identical code one screen above it streaming.

        One place also means the voice path only has to be taught about
        streaming once. The full text is always returned, so the drawer
        buffer, the chat store and every blocking caller are unchanged.
        """
        from src.core.market_price_guard import (
            MarketPriceGuard, check_market_prices, web_market_context,
        )
        from src.tools.web_search import SearchMarket

        emit = self._token_emitter()
        # Only _do_web_search supplies this immutable market/query snapshot.
        market, user_query = web_market_context()
        price_guard = (MarketPriceGuard(market, user_query, lang)
                       if mode == "ONLINE_RAG" and isinstance(market, SearchMarket) else None)

        def audit_prices(checked):
            if checked.count:
                audit.log("–", "orchestrator", "market_price_guard", payload={
                    "reason": "foreign_price_removed",
                    "currencies": sorted(checked.currencies), "count": checked.count,
                })

        # RAG already fetched data: the no-tool capability guard must not
        # rewrite its synthesis (or hold back its live stream).
        # Self facts use only the fenced prompt's own query; retrieved content
        # never supplies intent. Existing loss/receipt routing stays unchanged.
        with self._turn_reply_scope("", None, self_text=injected):
            if emit is None:
                reply = engine.chat(injected, response_language=lang, tool_backed=True)
                if price_guard is not None:
                    price_guard.feed(reply)
                    price_guard.finish()
                    audit_prices(price_guard)
                    reply = price_guard.reply
                return (reply, mode)
            parts: list = []
            replacement = None
            for piece in engine.chat_stream(injected, response_language=lang, tool_backed=True):
                if isinstance(piece, ReplyReplacement):
                    # Match plain chat: a late guard correction is the
                    # authoritative final, never another live token.
                    replacement = piece
                elif piece:
                    safe = price_guard.feed(piece) if price_guard is not None else piece
                    if safe:
                        parts.append(safe)
                        emit(safe)
            if price_guard is not None:
                if replacement is not None:
                    # Other guards still own whole-reply corrections (148a).
                    # Do not flush the superseded, partially buffered sentence.
                    checked = check_market_prices(replacement, market, user_query, lang)
                    price_guard.count += checked.count
                    price_guard.currencies.update(checked.currencies)
                    if checked.reply != replacement:
                        replacement = ReplyReplacement(checked.reply)
                else:
                    tail = price_guard.finish()
                    if tail:
                        parts.append(tail)
                        emit(tail)
                audit_prices(price_guard)
            return (replacement if replacement is not None else "".join(parts), mode)

    def _shadow_classify(self, text: str) -> None:
        """Cognitive Arbiter SHADOW MODE: classify the query and
        log its route + confidence to the audit, WITHOUT changing routing —
        the dormant skeleton, here only to measure the real decision
        distribution before Mac Studio. Off unless CORTEX_ARBITER_SHADOW.

        Must run ONCE PER TURN across all entry points: `process` reaches it
        via `_process_core`, but `process_stream`'s reflex-hit and pure-chat
        shortcuts bypass `_process_core` (the same 2-entry-points trap the
        reflex layer hit), so they call this directly. Wrapped so a classifier
        bug can never break the live path."""
        if not getattr(self, "_arbiter_shadow", False):
            return
        try:
            _d = arbiter.classify(text)
            audit.log("–", "arbiter", "classify", payload={
                "route": _d.route.value,
                "confidence": _d.confidence,
                "reason": _d.reason,
                "layer": _d.layer,
                "input": text[:60],
            })
        except Exception:  # noqa: BLE001 — shadow must never crash routing
            pass

    def _semantic_router(self):
        """Lazy SemanticRouter singleton for shadow measurement (item 1b),
        built on first use with the same bge-m3 embedder the Palace uses."""
        r = getattr(self, "_sem_router_instance", None)
        if r is None:
            from src.core.semantic_router import SemanticRouter
            from src.memory.embeddings import OllamaEmbedder
            r = SemanticRouter(OllamaEmbedder().embed)
            self._sem_router_instance = r
        return r

    # Intents the router may ACT on from the intent alone. The router reports
    # a mode, never slots: "manda matue tocando" is confidently SPOTIFY_PLAY
    # and says nothing about WHICH track, so acting on it would play
    # something arbitrary. Everything that needs an argument is left to the
    # regexes, which capture it, or to the LLM (s83).
    _SEMANTIC_LIVE_HANDLERS = {
        "JOBS_TOP": lambda o, text: o._handle_jobs_top(raw_text=text),
        "JOBS_RERANK": lambda o, text: o._handle_jobs_rerank(raw_text=text),
        "NOTE_LIST": lambda o, text: o._handle_note_list(),
        "STUDIER_LIST": lambda o, text: o._handle_list_studies(),
        "FITNESS_MACROS": lambda o, text: o._handle_fitness_macros_today(
            raw_text=text),
        "SPOTIFY_PAUSE": lambda o, text: (
            o.spotify.pause(lang=o._reply_lang), "SPOTIFY_PAUSE"),
        "WEATHER": lambda o, text: o._semantic_weather(text),
    }

    def _semantic_weather(self, text: str):
        """Weather needs a city and the handler never guesses one. WEATHER_RE
        wants the weather noun FIRST, so "How's the weather like in London?"
        misses it entirely — hence the router catching the intent. Only once
        the intent is settled is it safe to read a city off the tail; without
        one this falls through to chat, exactly as the handler would."""
        match = _CITY_TAIL_RE.search(text or "")
        if not match:
            return None
        return self._handle_weather(match.group("city"), raw_text=text)

    def _semantic_route_live(self, text: str):
        """The semantic router ACTING (item 1c), after every deterministic
        layer has declined. Returns (reply, mode) or None to keep talking.

        Calibrated offline on 614 real sentences (scripts/router_replay.py):
        at a 0.74 floor it routes 14 of the 340 that reach it, all of them
        real rescues, with the first false positive at 0.719. Never raises —
        a router fault must cost a chat reply, not the turn."""
        from src.core.semantic_router import _semantic_router_enabled
        if not _semantic_router_enabled() or not (text or "").strip():
            return None
        try:
            decision = self._semantic_router().explain(text)
        except Exception:  # noqa: BLE001
            log.exception("semantic router: explain failed (non-fatal)")
            return None
        if not decision.accepted:
            # The calibration data the shadow used to collect: a near-miss and
            # a genuine chat query are different facts, and only the first says
            # where the threshold belongs.
            audit.log("–", "semantic_router", "declined", payload={
                "input": text[:80], "mode": decision.mode,
                "score": round(decision.score, 3),
                "margin": round(decision.margin, 3), "reason": decision.reason,
                "example": decision.example, "live": True,
            })
            return None
        handler = self._SEMANTIC_LIVE_HANDLERS.get(decision.mode or "")
        if handler is None:
            # Confident about an intent it cannot act on without slots. Logged
            # because this is the list of things a richer catalogue or a slot
            # extractor would unlock.
            audit.log("–", "semantic_router", "no_handler", payload={
                "mode": decision.mode, "score": round(decision.score, 3),
                "input": text[:80],
            })
            return None
        try:
            result = handler(self, text)
        except Exception:  # noqa: BLE001
            log.exception("semantic router: handler %s failed", decision.mode)
            return None
        if not result:
            audit.log("–", "semantic_router", "handler_declined", payload={
                "mode": decision.mode, "input": text[:80],
            })
            return None
        audit.log("–", "semantic_router", "routed", payload={
            "mode": decision.mode, "score": round(decision.score, 3),
            "margin": round(decision.margin, 3), "example": decision.example,
            "input": text[:80],
        })
        return result

    def _shadow_semantic_route(self, text: str) -> None:
        """SemanticRouter SHADOW MODE (item 1b, m1_interpretation_plan.md):
        on a query about to hit the LLM (i.e. the regex/deterministic layers
        all MISSED), log what the router WOULD have routed to — WITHOUT
        routing. This is the calibration data: real misses the router could
        rescue, and genuine chat/knowledge it must NOT hijack, so min_score/
        min_margin can be tuned (item 1c) before it ever changes behaviour.

        Off unless CORTEX_SEMANTIC_ROUTER_SHADOW. Costs one embed (~180ms) on
        the LLM-bound turn only, negligible against the LLM itself. Wrapped so
        a router bug can never break the live path (mirrors _shadow_classify)."""
        from src.core.semantic_router import (
            _semantic_router_enabled, _semantic_router_shadow_enabled,
        )

        if not _semantic_router_shadow_enabled():
            return
        if _semantic_router_enabled():
            # Live, so _semantic_route_live is about to score this very
            # sentence and logs the same facts. Two embeds of one query is
            # ~360ms of nothing (s83).
            return
        try:
            d = self._semantic_router().explain(text)
            # explain(), not route(): a decline must carry the candidate it
            # REJECTED and by how much. Logging only the input made 31 of the
            # first 34 shadow events unusable — a near-miss (the router should
            # have rescued it) looked identical to a genuine chat query. The
            # near-misses are what place the threshold (item 1c).
            payload = {
                "input": text[:80], "mode": d.mode,
                "score": round(d.score, 3), "margin": round(d.margin, 3),
                "example": d.example,
            }
            if d.accepted:
                audit.log("–", "semantic_router", "would_route", payload=payload)
            else:
                audit.log("–", "semantic_router", "declined",
                          payload={**payload, "reason": d.reason})
        except Exception:  # noqa: BLE001 — shadow must never crash routing
            pass

    def _emit_status(self, stage: str, detail: str = "") -> None:
        """Report a pipeline stage to this thread's live-activity listener,
        if one is installed. No-op otherwise — so this is safe to sprinkle
        through the hot path without affecting any other caller. Never
        raises: a broken UI feed must not break a reply."""
        local = getattr(self, "_status_local", None)
        cb = getattr(local, "emitter", None) if local is not None else None
        if cb is None:
            return
        try:
            cb(stage, detail)
        except Exception:  # noqa: BLE001 — telemetry must never break a turn
            pass

    def process_stream(
        self,
        text: str,
        image_b64: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: "Optional[list]" = None,
    ):
        """Public streaming entry point. Binds a UI conversation's recent turns
        as the short-term history for this turn (UI 2c) — see process() /
        _conversation_scope — then delegates to the streaming implementation.
        `conversation_history` None (voice/CLI) keeps the global history. The
        scope stays open across every yield and unbinds when the generator is
        exhausted or closed. The pending slot is reloaded from disk for the
        shortcut checks (s88, staging_mixin); a deferred turn runs process()
        on its own thread, which keeps its own view."""
        text = strip_paste_markers(text or "")
        delivery = self._turn_delivery()
        parts = []
        self._begin_pending_turn()
        try:
            with self._conversation_scope(conversation_history), self._project_scope(text):
                for frame in self._process_stream_impl(
                    text, image_b64, language=language,
                    conversation_history=conversation_history,
                ):
                    if delivery is not None and not delivery._outcome_worker:
                        if isinstance(frame[0], ReplyReplacement):
                            parts.clear()
                        parts.append(frame[0])
                        if frame[1]:
                            delivery.outcome = ("".join(parts), frame[2])
                    yield frame
        finally:
            self._end_pending_turn()
            self._clear_reply_lang()
            if delivery is not None and not delivery._outcome_worker:
                delivery.finished.set()

    def _process_stream_impl(
        self,
        text: str,
        image_b64: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: "Optional[list]" = None,
    ):
        """
        Streaming variant of process(). Yields (chunk, is_final, mode, lang)
        tuples:
          - (text, False, None, lang) for partial chunks during streaming
          - ("", True, mode, lang)    on the final event for streamable paths
          - (text, True, mode, lang)  as a single emission for non-streamable
            paths
          - (ReplyReplacement(text), True, mode, lang) replaces all prior
            text for storage and final events, without another token emission

        `lang` is the turn's reply language ("pt"/"en") — the same value the
        deterministic handlers localise with. It rides on EVERY tuple so the
        SSE layer can announce it before the first chunk and the voice
        listener can pick the TTS voice the server already decided, instead
        of re-deriving it from a hand-kept PT word list (M5-A4/M10-A3).

        Streaming only kicks in for the pure OFFLINE chat path (no tool
        calls, no special regexes, no pending state, no realtime classifier).
        Everything else returns the full reply as one chunk — same behaviour
        as process(), just wrapped in the same generator interface so the
        caller doesn't need a separate code path.
        """
        # Latency clock for the SHORTCUT exits (reflex / pure-chat / engine
        # error). They bypass process() — the only place that logged the
        # `response` event — so the voice path never landed in the
        # latency_report buckets and the reported p95 came from a biased,
        # text-only population (M7-A3/M11-A5). It's also the instrument the
        # swap runbook §5 relies on. The deferral branch logs via process().
        t0 = time.perf_counter()
        text_norm = strip_vocatives((text or "").strip())
        # Routing-side variant (s75, Onda 8 A) — mirrors _process_core's
        # route_text. The pure-chat probe and the reflex probe below must
        # see through a courtesy wrapper ("can you open spotify please")
        # exactly like the deferred dispatch will, or the stream chats
        # right past the handler (the two-entry-points lesson, again).
        # The chat_stream call itself keeps text_norm: the LLM answers
        # the user's actual sentence, not the stripped one.
        reply_text = strip_trailing_sentence_punctuation(text_norm)
        route_text = normalise_stt_aliases(strip_courtesy(reply_text))
        # Only what the user wrote counts as "seen" — never a fenced attachment
        # riding in the same message (s85 review, see _pedros_words).
        own = self._pedros_words(text_norm)
        route_words = self._pedros_words(reply_text)
        self.remember_paths_in(own)            # both entry points, always
        self.remember_emails_in(own)           # both entry points, always
        # Resolved here for the shortcut paths below (reflex / pure chat),
        # which bypass _process_core and so would otherwise leave the turn
        # without a language. The deferral path recomputes via _process_core
        # and its value wins (it also honours pending-state nuances).
        lang = _resolve_lang(text_norm, language)
        if not text_norm and not image_b64:
            self._pdf_split_question = None
            yield ("[SYSTEM ERROR] Empty input.", True, "INPUT_ERROR", lang)
            return

        # Reflex layer — MUST run on the streaming path too. The voice
        # pipeline only ever calls process_stream, and the _is_pure_chat
        # shortcut below routes trivial queries ("que horas são", "oi")
        # straight to the LLM — which is slow and, worse, hallucinates
        # ("não posso fornecer a hora atual"). _process_core runs the
        # reflex, but this shortcut bypasses _process_core entirely.
        # Mirror it here, guarded by pending state so a "sim/não" reply
        # to a staged action still flows through the full pipeline.
        # A focus turn (s86) skips the reflex too: "oi" or "que horas são"
        # can't be a first step, but the veto is the same predicate the
        # pure-chat check and _process_core use — one rule, three places.
        # A yes while Cortex's question is live goes to _process_core too:
        # "valeu" is a farewell to the reflex and a yes to "escrevo no PDF?"
        # (s88, Codex lote 1). Unanswered there, it reaches the same reflex.
        if (
            not image_b64
            and not getattr(self, "_abandoned_effects", None)
            and self._pending_action is None
            and self._pending_query is None
            and self._pdf_split_question is None
            and self._turn_pending_id() is None
            and not self._is_focus_turn(route_words)
            and not self._is_project_command(route_words)
            and not code_roots.parse_command(route_words)
            and not self._is_named_file_turn(route_words)
            and not (getattr(self, "_offer", None)
                     and self._APPROVAL_RE.match(text_norm))
        ):
            _reflex_reply = reflex_layer.try_respond(route_text)
            if _reflex_reply is not None:
                self._shadow_classify(text_norm)  # bypasses _process_core
                # Reflex replies are written in the matched PATTERN's
                # language (they predate the i18n layer), which can differ
                # from the input-derived lang — "oi" reads as EN to the
                # heuristic but the reply is PT. The announced language
                # must match the WORDS the TTS will read (the s53 lesson),
                # so detect it from the reply itself.
                lang = _detect_lang(_reflex_reply)
                self._reply_lang = lang  # shortcut bypasses _process_core
                self._offer = None  # a question lasts ONE turn, this one too
                audit.log("–", "reflex", "hit", payload={"input": text_norm[:60]})
                audit.log("–", "orchestrator", "response",
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          payload={"mode": "REFLEX", "stream": True})
                yield (_reflex_reply, True, "REFLEX", lang)
                return

        # Pending crash warnings need process()'s deterministic suffix in the worker.
        if (not image_b64 and not getattr(self, "_abandoned_effects", None)
                and self._turn_pending_id() is None
                and self._is_pure_chat(route_text)):
            self._shadow_classify(text_norm)  # bypasses _process_core
            self._shadow_semantic_route(route_text)  # bypasses _process_core
            self._reply_lang = lang  # shortcut bypasses _process_core
            self._offer = None  # a question lasts ONE turn, this one too
            # The router acts here AND in _process_core's chat fall-through —
            # the two-entry-points rule: wiring a cross-cutting layer into one
            # of them is the bug this codebase keeps re-learning.
            _semantic = self._semantic_route_live(route_text)
            if _semantic is not None:
                _reply, _mode = _semantic
                audit.log("–", "orchestrator", "response",
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          payload={"mode": _mode, "stream": True})
                yield (_reply, True, _mode, lang)
                return
            taught = keep_taught_fact(
                text, lambda: self.engine_analytical.memory.knowledge, lang)
            if taught is not None and taught.status == "empty":
                yield (taught.receipt, True, "TAUGHT_FACT", lang)
                return
            replacement = None
            from src.core.hallucination import command_failure_stream
            try:
                with self._turn_reply_scope(text, taught):
                    for chunk in command_failure_stream(self.engine_analytical.chat_stream(
                        text_norm, response_language=lang
                    ), self._stage_routed_command):
                        if isinstance(chunk, ReplyReplacement):
                            replacement = chunk
                        else:
                            yield (chunk, False, None, lang)
            except Exception as e:
                # The raw exception would be FED TO THE TTS if emitted as a
                # chunk ("[ENGINE ERROR: ...]" spoken aloud — M10-A5). Speak
                # a short localised sentence instead; detail goes to audit.
                audit.log("–", "engine", "stream_error",
                          payload={"error": str(e)[:200]})
                audit.log("–", "orchestrator", "response",
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          payload={"mode": "INTERNAL_ERROR", "stream": True})
                yield (_PROMPTS["engine_error"][lang], taught is None,
                       "INTERNAL_ERROR", lang)
                if taught is not None:
                    yield ("\n\n" + taught.receipt, True, "INTERNAL_ERROR", lang)
                return
            mode = ("AGENT_AWAITING_CONFIRMATION"
                    if (self._pending_action or {}).get("type") == "routed_command"
                    else "OFFLINE")
            audit.log("–", "orchestrator", "response",
                      latency_ms=(time.perf_counter() - t0) * 1000,
                      payload={"mode": mode, "stream": True})
            final_text = ("\n\n" + taught.receipt) if taught is not None else ""
            if replacement is not None:
                if final_text:
                    yield (final_text, False, None, lang)
                final_text = ReplyReplacement(replacement + final_text)
            yield (final_text, True, mode, lang)
            return

        # Deferred path: the normal pipeline, run in a WORKER THREAD so a
        # step that already knows how to stream — both RAG syntheses, via
        # _synthesise — can push its tokens out through THIS generator
        # instead of landing whole at the end. That is what left ONLINE_RAG
        # silent BY VOICE for the entire search plus synthesis (the project
        # baseline is p95 ~47s), with the spoken "Só um momento" as the only
        # mitigation. Paths that never call the emitter (tools, dispatch,
        # reflex) stream nothing and behave exactly as before.
        #
        # Every piece of per-turn context here is THREAD-LOCAL and does not
        # cross the boundary on its own; each is re-installed in the worker,
        # and each would fail SILENTLY if it weren't:
        #   - the conversation history (UI 2c) — handed to process(), which
        #     opens its own scope inside the worker. Without it the turn
        #     quietly answers with the GLOBAL history instead of this
        #     conversation's.
        #   - the status emitter — the UI's live-activity feed would simply
        #     stop reporting "web_search"/"thinking" mid-turn.
        #   - the token emitter — forwarded, so the UI's existing side
        #     channel keeps working byte for byte as before.
        #   - the unverified-speaker mark (s88, Codex lote 12) — without it a
        #     visitor's words to the room microphone become facts about the user
        #     on every turn that takes this path.
        #   - the UI's token bucket (s88, Codex lote 13) — without it every
        #     token spent here is dropped and the turn shows no usage.
        outer_status = self._status_emitter()
        outer_token = self._token_emitter()
        outer_delivery = self._turn_delivery()   # the route's (s88, lote 7)
        outer_origin = self._turn_origin()
        outer_pending_id = self._turn_pending_id()
        from src.core.memory import speaker_unverified, unverified_speaker
        from src.core import turn_metrics
        outer_unverified = speaker_unverified()
        outer_usage = turn_metrics.current()
        bridge: "queue.Queue" = queue.Queue()
        box: dict = {}
        _DONE = object()

        def _bridge_emit(piece: str) -> None:
            if not piece:
                return
            bridge.put((piece, self._reply_lang))
            if outer_token is not None:
                try:
                    outer_token(piece)
                except Exception:  # noqa: BLE001 — a UI feed must not break the turn
                    pass

        def _worker() -> None:
            # Retain this turn's state before process() detaches it in cleanup.
            reply_state = self._reply_lang_state()
            try:
                self.set_status_emitter(outer_status)
                self.set_token_emitter(_bridge_emit)
                self.set_turn_delivery(outer_delivery)
                self.set_turn_origin(
                    outer_origin["channel"] if outer_origin else None,
                    outer_origin.get("conversation") if outer_origin else None)
                self.set_turn_pending_id(outer_pending_id)
                turn_metrics.attach(outer_usage)
                with taught_turn(text) as taught, (unverified_speaker() if outer_unverified
                                                  else nullcontext()):
                    box["result"] = self.process(
                        text, image_b64, language=language,
                        conversation_history=conversation_history,
                    )
                    box["taught"] = taught.result
                if outer_delivery is not None:
                    outer_delivery.outcome = box["result"]
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                box["error"] = exc
                if outer_delivery is not None:
                    outer_delivery.outcome = (
                        _PROMPTS["engine_error"][reply_state.get("lang") or lang],
                        "INTERNAL_ERROR")
            finally:
                box["lang"] = reply_state.get("lang") or lang
                self._clear_reply_lang()
                self.set_status_emitter(None)
                self.set_token_emitter(None)
                self.set_turn_delivery(None)
                self.set_turn_origin(None)
                self.set_turn_pending_id(None)
                turn_metrics.attach(None)
                if outer_delivery is not None:
                    outer_delivery.finished.set()
                bridge.put(_DONE)

        # A closed stream must leave completion to its still-running worker.
        if outer_delivery is not None:
            outer_delivery._outcome_worker = True
        worker = threading.Thread(
            target=_worker, name="cortex-stream-turn", daemon=True)
        worker.start()
        streamed: list = []
        while True:
            piece = bridge.get()
            if piece is _DONE:
                break
            piece, piece_lang = piece
            streamed.append(piece)
            # _process_core resolves the turn language (it sees pending state
            # the shortcut-level resolve can't). By the time a synthesis token
            # exists it has already run, so its decision is what rides out.
            yield (piece, False, None, piece_lang)
        worker.join()
        if "error" in box:
            raise box["error"]
        reply, mode = box["result"]
        lang = box["lang"]
        taught = box.get("taught")
        if taught is not None:
            # process() has already added the receipt after announcements.
            # Keep it as its own final chunk even for deferred/tool turns.
            suffix = taught.receipt
            if reply != suffix:
                suffix = "\n\n" + suffix
            body = reply[:-len(suffix)]
            sent = "".join(streamed)
            if body.startswith(sent):
                tail = body[len(sent):]
            else:
                audit.log("–", "orchestrator", "stream_reply_diverged",
                          payload={"mode": mode, "streamed": len(sent),
                                   "final": len(reply)})
                tail = body
            if tail:
                yield (tail, False, None, lang)
            yield (suffix, True, mode, lang)
            return
        if not streamed:
            yield (reply, True, mode, lang)
            return
        # Tokens already went out. The final tuple must carry only what the
        # consumer has NOT seen, or the UI stores the answer twice and the
        # voice listener speaks it twice.
        sent = "".join(streamed)
        if reply.startswith(sent):
            # The normal case, and the only mutation the pipeline makes after
            # a synthesis: _maybe_announce_pending appends its "💡 Por sinal"
            # postscript. That tail — usually empty — is what is still owed.
            yield (reply[len(sent):], True, mode, lang)
            return
        # Defensive: some future step rewrote the reply after it streamed, so
        # the two have diverged. Correct CONTENT wins over a duplicate — the
        # consumer gets the authoritative text — and the audit says it
        # happened, because nothing else would ever surface it.
        audit.log("–", "orchestrator", "stream_reply_diverged",
                  payload={"mode": mode, "streamed": len(sent),
                           "final": len(reply)})
        yield (reply, True, mode, lang)

    # Every dispatch route the registry can fire must ALSO veto the
    # streaming pure-chat shortcut, or the voice path streams an LLM
    # reply right past the handler (the s24/s73 two-entry-points bug).
    # tests/test_pure_chat_registry_guard.py enforces the invariant
    # registry↔this-list automatically — add a Route, the guard fails
    # until its regex name lands here (or in the test's documented
    # exceptions). Order in this list doesn't matter (any hit disables
    # pure-chat); order in the registry DOES matter.
    _PURE_CHAT_BLOCKING_RES = (
        "_PS5_RE",
        "_APPROVAL_RE", "_DENIAL_RE",
        "_READ_RE", "_WRITE_RE", "_CREATIVE_RE",
        "_REVIEW_RE", "_APPROVE_STUDY_RE", "_REJECT_STUDY_RE",
        "_SHOW_STUDY_RE", "_LIST_STUDIES_RE", "_STUDY_RE", "_ADD_STUDY_RE",
        "_CANCEL_STUDY_RE", "_CANCEL_VAGUE_RE", "_LAST_STUDY_RE",
        "_LIST_EMAILS_RE", "_SEND_EMAIL_STRUCTURED_RE",
        "_SEND_EMAIL_BY_NAME_RE", "_CONTACT_SAVE_RE", "_CONTACT_ADD_RE",
        "_CONTACT_LOOKUP_RE", "_CONTACT_INFO_RE",
        "_NOTE_CREATE_RE", "_NOTE_LIST_RE", "_NOTE_SEARCH_RE",
        "_NOTE_APPEND_RE", "_NOTE_APPEND_TAIL_RE", "_NOTE_CREATE_FOLDER_RE",
        "_WEATHER_RE", "_REFINE_RE",
        "_FITNESS_LOG_MEAL_RE", "_FITNESS_MACROS_TODAY_RE",
        "_FITNESS_LIST_TODAY_RE", "_FITNESS_UNDO_MEAL_RE", "_FITNESS_CLEAR_TODAY_RE",
        "_FITNESS_LOG_RUN_RE", "_FITNESS_LOG_LIFT_RE",
        "_FITNESS_RUN_TREND_RE", "_FITNESS_LIFT_TREND_RE",
        "_FITNESS_UNDO_WORKOUT_RE", "_FITNESS_CLEAR_WORKOUTS_RE",
        "_FITNESS_DELETE_GUARD_RE",
        "_JOBS_TOP_RE", "_JOBS_RERANK_RE",
        # s74 guard findings: SILENCE was a REAL gap (no TOOL_HINT word in
        # "silêncio"/"cala" — spoken, the stream would chat over the shush);
        # reminder/calendar were only covered by TOOL_HINT_RE by accident.
        "_SILENCE_RE", "_REMINDER_TRIGGER_RE",
        "_CALENDAR_TRIGGER_RE", "_CALENDAR_TRIGGER_CONTENT_RE",
        "_CALENDAR_VERB_RE",
        "_UNDO_RE", "_WHATS_CHANGED_RE", "_SYSTEM_STATUS_RE",
        "_LAST_CONTACT_RE", "_RECENT_PEOPLE_RE", "_DEEP_RESEARCH_RE",
        "_DREAM_REVIEW_RE",
        # Inline in _process_core, between the two registries, so the guard
        # test missed it: "planeja: organizar a semana" streamed chat (s88).
        "_PLANNER_TRIGGER_RE",
        "_SPOTIFY_PAUSE_RE", "_SPOTIFY_RESUME_RE", "_SPOTIFY_NEXT_RE",
        "_SPOTIFY_PREV_RE", "_SPOTIFY_NOW_RE", "_SPOTIFY_LIKE_RE",
        "_SPOTIFY_VOLUME_RE", "_SPOTIFY_TRANSFER_RE", "_SPOTIFY_PLAY_RE",
        "_SPOTIFY_LIBRARY_ARTIST_RE",
        "_SPOTIFY_PLAY_PLAYLIST_RE", "_SPOTIFY_PLAY_PLAYLIST_EN_RE",
        "_SPOTIFY_PLAY_RADIO_RE", "_SPOTIFY_PLAY_RADIO_EN_RE",
        "_MAC_VOLUME_SET_RE", "_MAC_VOLUME_UP_RE", "_MAC_VOLUME_DOWN_RE",
        "_MAC_MUTE_RE", "_MAC_UNMUTE_RE",
        "_MAC_BRIGHTNESS_UP_RE", "_MAC_BRIGHTNESS_DOWN_RE", "_MAC_BRIGHTNESS_SET_RE",
        "_MAC_KEYBOARD_BRIGHTNESS_UP_RE", "_MAC_KEYBOARD_BRIGHTNESS_DOWN_RE",
        "_MAC_KEYBOARD_BRIGHTNESS_SET_RE",
        "_MAC_DARK_TOGGLE_RE", "_MAC_DARK_ON_RE", "_MAC_DARK_OFF_RE",
        "_MAC_LOCK_RE", "_MAC_SCREENSHOT_RE", "_MAC_OPEN_APP_RE",
        "_MAC_CLOSE_WINDOW_RE", "_MAC_QUIT_APP_RE", "_MAC_SLEEP_RE",
        "_MAC_APP_LIST_RE", "_MAC_APP_SWITCH_RE", "_MAC_REPLY_EMAIL_RE",
        "_MAC_SUMMARISE_RE", "_MAC_WHAT_IS_RE",
        "_MAC_EYES_OFF_RE", "_MAC_EYES_ON_RE",
    )

    def _is_pure_chat(self, text: str) -> bool:
        """
        True iff this input will end up at engine.chat (OFFLINE plain
        conversation) — i.e. no regex/state/tool dispatch in the way.
        Used by process_stream to decide whether to stream tokens.

        Note: this DOES invoke _needs_realtime which is one LLM call (~1-2s)
        for the classifier. Acceptable cost since we save 5-15s downstream
        via streaming when the answer is YES (pure chat).
        """
        if (self._pending_action is not None or self._pending_query is not None
                or self._pdf_split_question is not None):
            return False

        # Defer to the normal pipeline: registry/state ownership wins, then
        # recall quotes this channel's history as a normal single-chunk reply.
        # No realtime classifier, retrieval or model is needed for this query.
        if (is_recall_question(text) or code_save_request(text) is not None
                or detect_usage_question(text) is not None
                or detect_self_question(text) is not None
                or is_finder_request(text)
                or (is_saved_location_request(text)
                    and last_saved_folder(self.engine_analytical.history) is not None)):
            return False

        # A path — or "os dois" right after a folder listing — is settled by
        # the path route in _process_core, never by streamed chat (s85). Same
        # predicate as the route, so the two can't disagree.
        # A document the user attached is answered by _process_core's attachment
        # route (no tools). Said explicitly: the "ARQUIVO ANEXADO" label only
        # vetoed this by accident, via the word "arquivo" in TOOL_HINT.
        if self._is_attachment_turn(text):
            return False

        own = self._pedros_words(text)
        if self._mentions_files(own):
            return False

        # "foco:", "parei:", "onde parei?" — and the answer to the focus
        # cycle's own question — are settled by _process_core's focus route.
        if self._is_focus_turn(own):
            return False
        if self._is_project_command(own):
            return False
        if code_roots.parse_command(own):
            return False

        # "lê meu currículo", "cadê o pdf de X" — and the answer to its
        # numbered list — are settled by _process_core's named-file route.
        if self._is_named_file_turn(own):
            return False

        # The address Cortex just asked for ("qual o email do João?") is the
        # answer to that question, not chat (s85, fila 4.1).
        if self._answers_email_question(own):
            return False

        for rx_name in self._PURE_CHAT_BLOCKING_RES:
            if getattr(self, rx_name).match(text):
                return False

        if self._TOOL_HINT_RE.search(text):
            return False

        # peek=True: this is a routing PROBE, not the turn's gate pass. The
        # default would revoke/grant right here and the deferred
        # _process_core would re-evaluate against the already-mutated state
        # ("desliga a internet" answered with "já estamos offline" — M6-A1).
        status = self.gatekeeper.evaluate_gate(text, peek=True)
        if status.mode != "OFFLINE":
            return False

        if teaching_is_empty(text):
            return True

        if self._needs_realtime(text):
            return False

        return True

    def _process_core(
        self,
        text: str,
        image_b64: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, str]:
        # Strip a leading vocative ("olá cortex, ...", "ei jarvis -")
        # so the `^`-anchored dispatch regexes match polite phrasings.
        # The salutation carries no routing intent; the system prompt
        # already establishes who the assistant is. See JOURNAL
        # sessão 10 bloco F for the regression that motivated this.
        raw_text = text or ""  # Exact-yes grants require the whole typed reply.
        text = strip_vocatives(raw_text.strip())
        if not text and not image_b64:
            self._pdf_split_question = None
            return ("[SYSTEM ERROR] Empty input.", "INPUT_ERROR")

        # Normalise commands and denials only. Every approval keeps HEAD's
        # punctuation-preserving text (raw_text for exact-yes grants):
        # stripping the question mark from "sim?" would create authority.
        reply_text = strip_trailing_sentence_punctuation(text)
        route_text = normalise_stt_aliases(strip_courtesy(reply_text))

        # Client override (e.g. iOS Shortcut bound to PT or EN) wins over
        # the regex heuristic, which misses mixed-language inputs like
        # "play michael jackson no spotify" and PT inputs without accents.
        lang = _resolve_lang(text, language)
        if language not in ("pt", "en"):
            # PS5 forms own their language ("status do ps5" is PT).
            # The stream defers here and carries this resolved value out.
            lang = ps5_phrase_language(route_text) or lang
        # Request-scoped: deterministic dispatch handlers read this (via the
        # I18nMixin `self._t`) to reply in the SPOKEN language instead of
        # hardcoded PT (s60). Set BEFORE the dispatch registry runs below.
        self._reply_lang = lang

        # Consume the choice on the very next turn in this conversation,
        # before normalisation can turn prose into a numbered answer.
        choice = self._answer_pdf_split_question(raw_text)
        if choice is not None:
            return choice

        # Cognitive Arbiter — SHADOW MODE (see _shadow_classify).
        self._shadow_classify(text)

        # What Cortex asked LAST turn ("write it on the PDF?", "which file?"),
        # if anything. Taken now, so it answers only this turn; a handler
        # below may leave a new one for the next.
        offer = getattr(self, "_offer", None)
        self._offer = None
        # What the user wrote, without a fenced attachment (s85 review): the UI
        # sends an attached document and his question as ONE text, and the
        # routes below trust "what the user wrote" — a path or address inside
        # the document must not count as his.
        own = self._pedros_words(text)
        route_words = self._pedros_words(reply_text)

        # Agent action confirmation (send_email / create_event) — highest
        # precedence: the user is replying to a staged destructive action.
        expected = self._turn_pending_id()
        if (expected is not None
                and (self._APPROVAL_RE.match(text) or self._DENIAL_RE.match(reply_text))
                and (self._read_pending_raw() or {}).get("id") != expected):
            return self._preview_stale_reply()
        if self._pending_action is not None:
            # TTL (M1-A9): a "sim" minutes-to-days late must not fire a
            # forgotten action — the existing drop-on-other-input guard
            # only helps if there WAS an intermediate turn. Expired slot
            # is discarded; an approval gets told (silently eating a "sim"
            # would read as a missed command), anything else just routes.
            if self._pending_action_expired(self._pending_action):
                stale_type = self._pending_action.get("type", "?")
                self._pending_action = None
                self._persist_pending_action()
                audit.log("–", "pending", "stale_discarded_on_turn",
                          payload={"type": stale_type, "input": text[:60]})
                if self._APPROVAL_RE.match(text) or self._DENIAL_RE.match(reply_text):
                    return (
                        _PROMPTS["stale_action_discarded"][lang].format(
                            m=int(self._PENDING_TTL_MIN)
                        ),
                        "STALE_ACTION_DISCARDED",
                    )
                # Not an approval — fall through and route normally.
            elif (self._APPROVAL_RE.match(text)
                  or (self._pending_action.get("type") in self._STRICT_CODE_APPROVALS
                      and re.match(r"^\s*(?:sim|yes)\b", raw_text, re.I))):
                if self._pending_action.get("type") in self._STRICT_CODE_APPROVALS:
                    # Code authority requires the whole typed reply, never a loose yes.
                    refusal = self._pending_approval_reply(self._pending_action)
                    if refusal:
                        return refusal
                    if re.fullmatch(r"\s*(?:sim|yes)[!.]?\s*", raw_text, re.I):
                        return self._execute_pending_action()
                    if self._pending_action.get("type") in ("run_code_tests", "pdf_merge", "pdf_split", "pdf_png", "pdf_images"):
                        self._reply_lang = self._pending_action["args"].get("lang") or self._reply_lang
                    return (self._t(
                        "Para tirar as imagens, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To take the images out, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "pdf_images" else self._t(
                        "Para salvar as imagens, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To save the images, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "pdf_png" else self._t(
                        "Para separar as páginas, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To split the pages, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "pdf_split" else self._t(
                        "Para juntar os PDFs, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To merge the PDFs, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "pdf_merge" else self._t(
                        "Para autorizar uma pasta de código, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To authorise a code folder, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "code_root_add" else self._t(
                        "Para rodar os testes, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To run the tests, reply exactly 'yes' (or 'no' to cancel).")
                        if self._pending_action.get("type") == "run_code_tests" else self._t(
                        "Para aplicar uma mudança de código, responda exatamente 'sim' (ou 'não' para cancelar).",
                        "To apply a code change, reply exactly 'yes' (or 'no' to cancel)."),
                        "AWAITING_CONFIRMATION")
                return self._execute_pending_action()
            elif self._DENIAL_RE.match(reply_text):
                return self._cancel_pending_action()
            elif self._pending_same_origin():
                # Anything else: drop the staged action and let the new
                # query flow normally. We do NOT silently execute on an
                # unrelated turn.
                self._pending_action = None
                self._persist_pending_action()

        # Approval/denial of a pending search proposal takes precedence over
        # any other routing — the user is responding to the previous turn.
        if self._pending_query is not None:
            if self._APPROVAL_RE.match(text):
                pending_text = self._pending_query
                pending_engine = self._pending_engine or self.engine_analytical
                # The pending query was captured BEFORE the override was
                # known. Respect the same override for the resumed call —
                # the user is on the same shortcut on the same iPhone.
                pending_lang = _resolve_lang(pending_text, language)
                self._pending_query = None
                self._pending_engine = None
                expires_at = self.gatekeeper.grant_lease(self.lease_hours)
                # Weather stays deterministic end-to-end (idea #27): if the
                # pending query is a weather one, the approval re-dispatches
                # through the template route — NOT the LLM RAG path. Same
                # lease-rollback contract on failure (M1-A10).
                pending_route_text = normalise_stt_aliases(strip_courtesy(
                    strip_trailing_sentence_punctuation(pending_text)))
                _wm = self._WEATHER_RE.match(pending_route_text)
                if _wm:
                    self._reply_lang = pending_lang
                    _weather = self._handle_weather(
                        _wm.group("city"), raw_text=pending_route_text)
                    if _weather is not None:
                        reply, search_mode = _weather
                        if search_mode == "WEATHER_ERROR":
                            self.gatekeeper.revoke_lease()
                            audit.log("–", "gate", "lease_rolled_back",
                                      payload={"reason": "approved_weather_failed",
                                               "query": pending_text[:60]})
                            return (reply, "WEATHER_ERROR")
                        # Prefix in the approval-turn language, body in the
                        # pending query's — same split _do_web_search gets.
                        prefix = _PROMPTS["auto_search_prefix"][lang].format(
                            time=expires_at.strftime("%H:%M")
                        )
                        return (prefix + reply, "WEATHER")
                reply, search_mode = self._do_web_search(
                    pending_engine, pending_text, pending_lang
                )
                # M1-A10: the lease is granted BEFORE the search (the
                # search needs it). If the search then fails, don't keep
                # 4h of internet open for a search that never happened —
                # revoke, report the REAL mode, and skip the cheerful
                # "[Internet liberada até …]" prefix that masked the
                # error as success.
                if search_mode == "SEARCH_ERROR":
                    self.gatekeeper.revoke_lease()
                    audit.log("–", "gate", "lease_rolled_back",
                              payload={"reason": "approved_search_failed",
                                       "query": pending_text[:60]})
                    return (reply, "SEARCH_ERROR")
                prefix = _PROMPTS["auto_search_prefix"][lang].format(
                    time=expires_at.strftime("%H:%M")
                )
                return (prefix + reply, "AUTO_SEARCH")
            if self._DENIAL_RE.match(reply_text):
                self._pending_query = None
                self._pending_engine = None
                return (_PROMPTS["offline_denied"][lang], "APPROVAL_DENIED")
            # s76 smoke: Whisper transcribed a spoken "yes" follow-up as
            # "Ah!", which landed here, silently dropped the pending query
            # and routed the noise itself to the LLM (it hallucinated an
            # answer out of PAST CONTEXT). A reply this short is far more
            # likely garbled STT than a new command, so re-ask ONCE and
            # keep the pending; AWAITING_APPROVAL reopens the listener's
            # follow-up window. Fails closed: nothing is approved without
            # an APPROVAL_RE match (SOBERANIA r4 untouched). Anything
            # longer — or a second miss — reads as a genuine new command
            # and flows exactly as before. A focus command ("parei",
            # "onde parei?") is short AND unmistakable: it drops the pending
            # and reaches the focus route (s88, Codex lote 1).
            if (text and not image_b64
                    and len(text) <= 12 and len(text.split()) <= 2
                    and not self._is_focus_command(route_words)
                    and not self._is_project_command(route_words)
                    and not code_roots.parse_command(route_words)
                    and not getattr(self, "_pending_query_reprompted", False)):
                self._pending_query_reprompted = True
                return (_PROMPTS["approval_unclear"][lang], "AWAITING_APPROVAL")
            # Anything else: treat as a new query, drop the pending state.
            self._pending_query = None
            self._pending_engine = None
            self._pending_query_reprompted = False

        # Declarative dispatch registry (dispatch/registry.py) — the single
        # ordered routing table, shared with the eval simulator. The whole
        # PRE-planner regex-first command phase now lives here (migrated
        # s58/s59: studier, contacts, email, silence, mac_sleep, Spotify, macOS,
        # reminder, calendar, notes, + the Onda 4/5/Fase 2 routes). Dormant
        # features (deep/dream) return None when their flag is off and fall
        # through. Order is preserved exactly. Adding a pre-planner route = one
        # entry in the registry, zero lines here.
        #
        # A SECOND registry segment (POST_PLANNER_ROUTES) runs after the planner
        # below for the broad macOS app-control verbs. What stays inline (genuine
        # control-flow, not table-driven dispatch): the planner / multi-intent
        # autoroute, and the engine-dependent READ/WRITE/CREATIVE routing.
        #
        # `route_text` (s75, Onda 8 A): courtesy wrapper stripped FOR THE
        # DISPATCH PHASE ONLY — "Can you open Spotify please?" must hit
        # MAC_OPEN_APP like "open spotify" does (the verb-first regexes
        # can't see through the polite envelope; smoke finding 1, s74).
        # The LLM fallback below keeps the ORIGINAL `text`: "can you hear
        # me?" is a different sentence from "hear me?".
        # A "sim" that nothing is waiting for (s85, fila item 3): measured,
        # 14 of 14 such replies right after Cortex's own question ("write it
        # on the PDF?", "which one?") got a bare "Sim." back and did nothing.
        # Every staged action and pending query was handled above, so here
        # the yes can only answer that question — or nothing at all.
        # A document the user attached in the chat: answered from the document,
        # with NO tools (s85). Measured: 18 of 18 such turns called a tool,
        # 5 of 6 "resume esse arquivo" wrote an unasked file into the vault
        # (write_document needs no confirmation — the attachment's text would
        # be read back later as a document), and 6 of 6 with a hidden order in
        # it answered "document not found" with the document right there.
        if not image_b64 and self._is_attachment_turn(text):
            return self._answer_attachment(text, lang)

        # This deictic request uses only the active channel's assistant history.
        # Settle it before path parsing, file lookup, planning or any model call:
        # a requested filename is a basename, never authority to access a path.
        if not image_b64 and code_save_request(route_text) is not None:
            return (save_last_code(route_text, self.engine_analytical.history, lang), "SAVE_CODE")
        if not image_b64 and is_finder_request(route_text):
            return (show_saved_in_finder(self.engine_analytical.history, lang), "SHOW_CODE_FINDER")
        if not image_b64 and is_saved_location_request(route_text):
            location = saved_location(self.engine_analytical.history, lang)
            if location is not None:
                self.engine_analytical.history.extend([
                    {'role': 'user', 'content': text}, {'role': 'assistant', 'content': location}])
                return (location, "SHOW_CODE_LOCATION")

        # An explicit project command must not become a focus first step.
        if not image_b64:
            _project = self._handle_projects(route_words)
            if _project is not None:
                return _project
            command = code_roots.parse_command(route_words)
            if command:
                action, path, command_lang = command
                if action == "add":
                    return self._stage_code_root(path, command_lang)
                if action == "list":
                    authorised = code_roots.roots()
                    return (("Pastas autorizadas para código:\n" +
                             "\n".join(f"- {r}" for r in authorised)
                             if authorised else "Nenhuma pasta autorizada para código."),
                            "CODE_ROOT_LIST")
                try:
                    ok, reply = code_roots.remove(path)
                except OSError:
                    ok, reply = False, "Não foi possível guardar as autorizações para código."
                return reply, "CODE_ROOT_REMOVED" if ok else "CODE_ROOT_REFUSED"

        # Focus sessions (s86): a command, or the answer to the cycle's own
        # question, settled by fixed rules — measured, 1 useful reply in 30
        # from the 8B. BEFORE the registry: the first step "abrir o
        # formulário" tried to open an app called "formulário" (4 of 4).
        # process_stream gets here through the reflex and pure-chat vetoes.
        if not image_b64:
            # Answers to a live offer may grant work; preserve their text.
            _focus = self._handle_focus(own if offer else route_words, offer)
            if _focus is not None:
                return _focus

        # The answer to a named-file list ("2", "abre o primeiro", or "sim"
        # to "quer que eu leia?") — before the yes-with-nothing-pending check
        # below would call that "sim" an orphan (s87, 4.3).
        if not image_b64:
            # File choices and yes replies retain HEAD's literal grammar.
            _picked = self._answer_named_file_offer(own, offer)
            if _picked is not None:
                return _picked

        if not image_b64 and (offer or {}).get("kind") == "email_address":
            _resumed = self._resume_email(own, offer)
            if _resumed is not None:
                return _resumed

        if not image_b64 and self._APPROVAL_RE.match(text):
            _yes = self._handle_unpending_yes(text, offer)
            if _yes is not None:
                return _yes

        self.remember_paths_in(own)            # both entry points, always
        self.remember_emails_in(own)           # both entry points, always
        # Streaming defers command execution here too. Keep the pre-courtesy
        # text local to this dispatch, so a previous turn cannot supply a tail.
        if removed_portuguese_courtesy_suffix(reply_text):
            _routed = dispatch_registry.run(self, route_text, pre_strip_text=reply_text)
        else:
            _routed = dispatch_registry.run(self, route_text)
        if _routed is not None:
            return _routed

        # Saved UI usage is a code-owned fact. Pending work and explicit
        # commands above keep ownership; full-question matching declines mixed
        # actions and planner/app commands. Before generic path parsing too:
        # "input/output tokens" is not a path. The shared pure-chat veto
        # defers streaming here BEFORE tool hints / the realtime classifier
        # (including the word "responder").
        usage_lang = detect_usage_question(own) if not image_b64 else None
        if usage_lang is not None:
            self._reply_lang = usage_lang
            return (previous_usage_answer(self._turn_origin(), usage_lang), "USAGE_PREVIOUS")

        # Same whole-question/own-words boundary as usage; existing owners win.
        self_question = detect_self_question(own) if not image_b64 else None
        if self_question is not None:
            self._reply_lang = self_question.lang
            return (self_answer(self_question, self.engine_analytical, self.engine_creative),
                    self_question.mode)

        # A path in the message never reaches the 8B on its own (s85): 18 of
        # 18 such turns pretended to a read that never happened. After the
        # deterministic registry, before the planner — the planner is the 8B
        # too. process_stream gets here through _is_pure_chat's veto.
        if not image_b64:
            _by_path = self._handle_user_paths(own)
            if _by_path is not None:
                return _by_path

        # A file asked for by NAME, without a path (s87, 4.3): measured, 1
        # useful reply in 28 — the 8B searched by content every time, and
        # nothing knew currículo = CV = resume. After the registry ("lê meu
        # email" is the email route's) and the typed-path route; before the
        # planner and the app opener ("abre meu perfil" is a file).
        if not image_b64:
            _by_name = self._handle_named_file(own)
            if _by_name is not None:
                return _by_name

        # Plan-Execute dispatch (Tier 3.1) — explicit "planeja:" prefix
        # kicks off multi-step decomposition. The decomposed plan is
        # staged in _pending_action as "execute_plan"; user reviews the
        # preview, sim/não resolves it. Falls through to LLM on
        # decompose failure so a typo'd "planeja" doesn't dead-end.
        planner_match = self._PLANNER_TRIGGER_RE.match(route_text)
        if planner_match:
            _result = self._handle_planner_request(
                planner_match.group("objective").strip()
            )
            if _result is not None:
                return _result

        # Auto-planner for natural multi-intent (sessão 15). Real-world
        # input from the user that motivated this:
        #   "busca os jogos da seleção brasileira E adiciona ao meu
        #    calendário"
        # Without explicit "planeja:" prefix it used to fall to
        # chat_with_tools — Llama 3.1 8B Q4 picks ONE tool, loses the
        # second intent, and has been observed hallucinating
        # ("configure the calendar" with calendar already authorised).
        # `multi_intent_match` requires a sequential conjunction
        # ("e"/"and"/"depois"/"then"/…) PLUS mentions of two distinct
        # domain families (email, calendar, reminder, document, search)
        # so casual "e"s don't drag knowledge queries into a 3-5s
        # planner decomposition. Falls through to chat_with_tools on
        # decompose failure (same contract as the explicit path above).
        if multi_intent_match(route_text):
            _result = self._handle_planner_request(route_text)
            if _result is not None:
                return _result

        # Post-planner macOS app-control verbs (close/quit/switch/open) — broad
        # verbs that must run AFTER the planner so a multi-intent phrase or an
        # earlier specific dispatch wins first ("abre meus emails" → LIST_EMAILS
        # in the first segment, not "launch an app called 'meus emails'"). They
        # live in the registry's POST_PLANNER_ROUTES, invoked here as the second
        # segment (s59). Quit is S3 (staged, waits for "sim"); the rest act
        # immediately. Unknown app → `open -a` surfaces "app não encontrado".
        _mac_app = dispatch_registry.run(self, route_text, dispatch_registry.POST_PLANNER_ROUTES)
        if _mac_app is not None:
            return _mac_app

        # Reflex layer — deterministic zero-LLM responses for trivial inputs
        # (greetings, time/date queries, ping). Must come after all stateful
        # checks (pending_action, pending_query) and command routing (Studier,
        # Spotify, etc.) so it only intercepts queries headed for the LLM.
        _reflex_reply = reflex_layer.try_respond(route_text)
        if _reflex_reply is not None:
            audit.log("–", "reflex", "hit", payload={"input": text[:60]})
            return (_reflex_reply, "REFLEX")
        audit.log("–", "reflex", "miss", payload={"input": text[:60]})

        # Skill Fast Path (Tier 3.2) — dynamic Palace-persisted skills.
        # Sits AFTER reflex (which handles universal hardcoded patterns)
        # and BEFORE the heavy LLM path. A skill match returns a
        # deterministic reply with zero LLM cost; bumps the skill's
        # invocation_count as a side effect. Skipping execute() on the
        # rare failure path (handler_payload corrupted, unknown handler
        # type) so the user falls through to the normal LLM dispatch
        # rather than getting a broken reply.
        if self.skill_registry is not None:
            skill_hit = self.skill_registry.match(route_text)
            if skill_hit is not None:
                # The two callbacks are the tool_sequence gate plumbing
                # (M8-A1): S1/S2 macro steps run through the per-tool
                # dispatch, an S3 step stages in _pending_action exactly
                # like a direct S3 call (sim/não before anything fires).
                # static_response skills ignore both.
                skill_reply = self.skill_registry.execute(
                    skill_hit, text,
                    run_step=self._skill_macro_run_step,
                    stage_step=self._skill_macro_stage_step,
                )
                if skill_reply is not None:
                    audit.log("–", "skill", "hit",
                              payload={"name": skill_hit.name, "input": text[:60]})
                    return (skill_reply, "SKILL")
                audit.log("–", "skill", "execute_failed",
                          payload={"name": skill_hit.name})

        # SemanticRouter SHADOW (item 1b): every deterministic layer above
        # (dispatch registry, planner, reflex, skills) MISSED — this query is
        # LLM-bound. Log what the router WOULD have routed it to, without
        # routing. Exactly the population to calibrate on. No-op unless shadow.
        self._shadow_semantic_route(route_text)

        engine = self.engine_analytical
        creative_match = self._CREATIVE_RE.match(route_text)
        if creative_match:
            engine = self.engine_creative
            text = creative_match.group(1).strip()

        if image_b64:
            prompt = text or "Analise esta imagem e descreva os seus elementos principais detalhadamente."
            return (self.vision.analyze_image(prompt, image_b64), "VISION_ANALYSIS")

        read_match = self._READ_RE.match(route_text)
        if read_match:
            return self._handle_read(engine, read_match.group(1).strip(), read_match.group(2), lang)

        # "lê esse arquivo…" (the noun) or "lê o punctuation…" (the name of a
        # file already on the table). The second needs the runtime check: a
        # read verb alone is not a command to open anything.
        if (self._READ_THIS_FILE_RE.match(route_text)
                or (self._READ_VERB_RE.match(route_text)
                    and self._names_a_known_file(route_text))):
            return self._handle_read_this_file(route_text, lang)

        write_match = self._WRITE_RE.match(route_text)
        if write_match:
            return self._handle_write(engine, write_match.group(1).strip(), write_match.group(2).strip(), lang)

        status = self.gatekeeper.evaluate_gate(text)

        # Every existing command (including read/write, skills and gate
        # controls) owns its turn first. Keep ORIGINAL words at this LLM
        # boundary, before the model runs; the public scope appends the
        # deterministic receipt after announcements, even on model failure.
        # A regex remains owned even if its handler declined (e.g. disabled
        # studies/planner), or it selects an engine instead of returning
        # (CREATIVE_RE). Reuse the streaming ownership list rather than
        # maintaining a second set. Explicit search stays with the gate.
        taught = None
        if status.mode == "OFFLINE" and not any(
            getattr(self, name).match(route_text)
            for name in self._PURE_CHAT_BLOCKING_RES
        ):
            if is_recall_question(text):
                return (recall_last(text, engine.history, lang), "RECALL_LAST")
            taught = prepare_taught_fact(
                lambda: self.engine_analytical.memory.knowledge, lang)
            if taught is not None and taught.status == "empty":
                return ("", "TAUGHT_FACT")

        if status.mode == "OFFLINE":
            # Taught context already existed on this whole branch. Loss context
            # is installed only at the local engine calls below, leaving the
            # realtime classifier and internet synthesis path unchanged.
            with self._turn_reply_scope("", taught):
                # Tool hint has PRECEDENCE over the realtime classifier. If the
                # query references the user's own data (documents, email,
                # calendar, etc.), route it straight to tool calling — the
                # realtime classifier doesn't know about our tools and would
                # mis-flag e.g. "meus eventos próximos 7 dias" as needing
                # internet when it's really a Calendar question.
                if self._TOOL_HINT_RE.search(text):
                    # Offer the real tools ONLY when a hint for a tool that exists
                    # in TOOL_SPECS fired (documents/email/calendar/reminder). If
                    # ONLY music/macOS hinted (no such tool exists), offer an empty
                    # list: the path still runs so the anti-hallucination detector
                    # can catch a faked action, but the 8B can't fabricate an
                    # unrelated tool (e.g. create_reminder from "amo música
                    # brasileira", which only matched on "música"). s38.
                    specs = self._tool_specs_for(text)
                    with self._turn_reply_scope(raw_text, taught):
                        reply, tool_used = engine.chat_with_tools(
                            text, specs, self._execute_tool,
                            response_language=lang,
                        )
                    if self._pending_action is not None:
                        # CRITICAL: do NOT use the LLM-generated reply for the
                        # confirmation prompt. The LLM's second round sometimes
                        # paraphrases or translates the staged args (e.g.
                        # "Reunión" → "Reunião"), making the user agree to
                        # something different from what will actually run.
                        # Build the prompt deterministically from the real
                        # staged args so the user sees exactly what will happen.
                        return (
                            self._human_confirmation_prompt(self._pending_action),
                            "AGENT_AWAITING_CONFIRMATION",
                        )
                    mode = ("SAVE_CODE" if isinstance(reply, LiteralToolReply) else
                            f"TOOL_{tool_used.upper()}" if tool_used else "OFFLINE")
                    return (reply, mode)

                # No tool hint — fall back to the realtime classifier.
                if self._needs_realtime(text):
                    if self.gatekeeper.is_lease_active():
                        return self._do_web_search(engine, text, lang)
                    self._pending_query = text
                    self._pending_engine = engine
                    self._pending_query_reprompted = False
                    return (
                        _PROMPTS["needs_internet"][lang].format(h=self.lease_hours),
                        "AWAITING_APPROVAL",
                    )
                semantic = self._semantic_route_live(route_text)
                if semantic is not None:
                    return semantic
                with self._turn_reply_scope(raw_text, taught):
                    reply = engine.chat(text, response_language=lang)
                return (reply, "AGENT_AWAITING_CONFIRMATION"
                        if (self._pending_action or {}).get("type") == "routed_command"
                        else "OFFLINE")

        if status.mode == "LEASE_GRANTED":
            local_when = status.expires_at.astimezone()
            tz_label = local_when.tzname() or ""
            suffix = f" ({tz_label})" if tz_label else ""
            return (
                _PROMPTS["lease_granted"][lang].format(
                    time=local_when.strftime("%H:%M"), tz_suffix=suffix
                ),
                "SYSTEM_UPDATE",
            )

        if status.mode == "LEASE_REVOKED":
            return (_PROMPTS["lease_revoked"][lang], "SYSTEM_UPDATE")

        if status.mode == "ALREADY_OFFLINE":
            return (_PROMPTS["already_offline"][lang], "SYSTEM_UPDATE")

        if status.mode == "GATE_CLOSED":
            # Explicit "pesquise X" without lease — same approval flow.
            self._pending_query = text
            self._pending_engine = engine
            self._pending_query_reprompted = False
            return (
                _PROMPTS["search_needs_gate"][lang].format(h=self.lease_hours),
                "AWAITING_APPROVAL",
            )

        if status.mode == "ONLINE_REQUEST":
            return self._do_web_search(engine, text, lang)

        return ("[SYSTEM ERROR] Unknown gate state.", "INTERNAL_ERROR")

    @staticmethod
    def _humility_enabled() -> bool:
        """Kill-switch for the Onda 3c volatile-fact humility layer. On by
        default; CORTEX_HUMILITY=0 disables it. Read per-call so it can be
        toggled without a restart (mirrors _passive_facts_enabled)."""
        return os.getenv("CORTEX_HUMILITY", "1").strip().lower() not in (
            "0", "false", "no", "",
        )

    def _tool_specs_for(self, text: str) -> list:
        """Specs offered to chat_with_tools on a tool-hinted turn. The
        empty-list branch is the s38 anti-fabrication gate: when only
        music/macOS hinted (no such tool in TOOL_SPECS), the 8B got an
        empty menu so it couldn't invent create_reminder from "amo música".
        A stronger profile drops the gate and offers the full menu
        (guard `tool_specs_hint_gate`)."""
        if not guard_active("tool_specs_hint_gate"):
            return _drop_unusable_file_read(TOOL_SPECS,
                                            getattr(self, "_seen_paths", None))
        specs = TOOL_SPECS if self._REAL_TOOL_HINT_RE.search(text) else []
        return _drop_unusable_file_read(specs,
                                        getattr(self, "_seen_paths", None))

    def _drop_unusable_file_read(self, specs: list) -> list:
        return _drop_unusable_file_read(specs, getattr(self, "_seen_paths", None))

    def _unused_drop_doc(self) -> None:
        """Hide read_file_at until the conversation HAS a path.

        Offered from the first turn, it out-competed find_file: "cadê o pdf de
        punctuation?" made the 8B invent /Users/.../punctuation.pdf and ask to
        open it (s83). A tool that cannot legally be used yet — the executor
        refuses a path nobody has seen — has no business on the menu. Same
        move as the s38 anti-fabrication gate: the model cannot pick what it
        is not offered. A path the user types is harvested before dispatch,
        so stating one still works on the first turn."""

    def _needs_realtime(self, text: str) -> bool:
        """LLM-based binary classifier: does this query need live web data?"""
        key = text.lower().strip()
        if key in self._intent_cache:
            self._intent_cache.move_to_end(key)
            return self._intent_cache[key]

        # Fast-path 1: "explica X", "como funciona Y", "o que é Z", etc. —
        # unambiguously offline unless a realtime marker ("hoje", "now", ...)
        # is present.
        # Fast-path 2: pure arithmetic — "2+2", "15*7", "(3+1)/2 ?".
        # Both paths skip the 1-2s LLM classifier call.
        is_offline_prefix = (
            self._OFFLINE_HINT_RE.match(text)
            and not self._REALTIME_MARKER_RE.search(text)
        )
        is_math_only = self._MATH_ONLY_RE.match(text)
        if is_offline_prefix or is_math_only:
            if len(self._intent_cache) >= self._INTENT_CACHE_MAX:
                self._intent_cache.popitem(last=False)
            self._intent_cache[key] = False
            return False

        # Humility (Onda 3c): volatile-fact categories the LLM classifier
        # tends to miss (sports results, live prices, recent news) when no
        # time-marker word is present. Treat as realtime so the query routes
        # to the existing "quer que eu pesquise?" offer instead of letting the
        # 8B confabulate. Runs AFTER the offline fast-path (so definitional
        # "o que é placar" still stays offline). Default-on; CORTEX_HUMILITY=0
        # disables it (read per-call, no restart). Also saves the LLM call.
        if self._humility_enabled() and self._VOLATILE_FACT_RE.search(text):
            if len(self._intent_cache) >= self._INTENT_CACHE_MAX:
                self._intent_cache.popitem(last=False)
            self._intent_cache[key] = True
            return True

        prompt = (
            "Task: Classify whether answering the question below requires "
            "fetching CURRENT real-time data from the internet.\n\n"
            "Reply YES for: news, current events, recent results, weather, "
            "sports scores, stock prices, exchange rates, current promotions, "
            "today's prices, anything time-sensitive, anything happening now.\n\n"
            "Reply NO for: general knowledge, history, science, definitions, "
            "programming, mathematics, language, creative writing, opinions, "
            "explanations of stable concepts.\n\n"
            "Examples:\n"
            "Q: who was Alan Turing → NO\n"
            "Q: what is the weather in London right now → YES\n"
            "Q: what time is it → NO\n"
            "Q: que horas são → NO\n"
            "Q: what date is today → NO\n"
            "Q: que dia é hoje → NO\n"
            "Q: explain pointers in C → NO\n"
            "Q: latest news about AI → YES\n"
            "Q: how do quantised LLMs work → NO\n"
            "Q: result of yesterday's Flamengo match → YES\n\n"
            "Reply with ONLY the word YES or NO, nothing else.\n\n"
            f"Q: {text}\n"
            "Answer:"
        )
        try:
            response = ollama.generate(
                model=self.engine_analytical.model_name,
                prompt=prompt,
                # Gemma used 6 tokens for a label; 12 leaves room for leading tokens.
                options={"num_predict": 12, "temperature": 0.0},
                keep_alive=_KEEP_ALIVE, think=False,  # M2-A7: don't downgrade the warm timer
            )
            reply = (response.get("response") or "").strip().upper()
            result = reply.startswith("SIM") or reply.startswith("YES")
            if not result and not re.match(r"NO\b", reply):
                audit.log("–", "orchestrator", "realtime_classifier_unclear", payload={
                    "reason": "empty" if not reply else "unparseable",
                    "reply_length": len(reply),
                })
        except Exception:
            result = False  # failsafe — when in doubt, stay offline

        if len(self._intent_cache) >= self._INTENT_CACHE_MAX:
            self._intent_cache.popitem(last=False)
        self._intent_cache[key] = result
        return result

    def _do_web_search(
        self, engine: OllamaEngine, text: str, lang: str = "pt"
    ) -> Tuple[str, str]:
        """Strip search trigger words, query the web, inject into the LLM."""
        search_query = self._SEARCH_STRIP_RE.sub("", text)
        self._emit_status("web_search", search_query.strip()[:60])
        web_context = self.web_searcher.search(search_query)
        # Bug s73: the snippets were not persisted anywhere, so diagnosing a
        # wrong ONLINE_RAG answer required a paid replay of the search. Log
        # them here — capped (rotation handles growth, but one event must not
        # dwarf the file) and redacted at the audit.log border like any sink.
        audit.log("–", "web_search", "snippets", payload={
            "query": search_query.strip()[:200],
            "context": web_context[:3000],
        })
        if web_context.startswith("[SYSTEM"):
            return (web_context, "SEARCH_ERROR")
        self._emit_status("thinking")
        market = self.web_searcher.get_market()
        injected = build_rag_prompt(web_context, text, market=market)
        from src.core.market_price_guard import web_market_scope
        with web_market_scope(market, text):
            return self._synthesise(engine, injected, lang, "ONLINE_RAG")

    def _names_a_known_file(self, text: str) -> bool:
        """True when the sentence names one of the files on the table."""
        return bool(self._match_candidates(text))

    def _match_candidates(self, text: str) -> list:
        """Files on the table whose name shows up in what the user just said."""
        lowered = (text or "").lower()
        out = []
        for path in self.readable_candidates():
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            words = [w for w in re.split(r"[\s_.\-]+", stem) if len(w) > 3]
            if any(w in lowered for w in words):
                out.append(path)
        return out

    def _handle_read_this_file(self, raw_text: str, lang: str = "pt"):
        """"lê esse arquivo e me diz X" — the file being whatever the
        conversation has on the table (a search result, or a path the user
        typed). Stages the same S3 confirmation the tool does, so the gate is
        identical whichever way the read was asked for."""
        candidates = self.readable_candidates()
        if not candidates:
            return (self._t(
                "qual arquivo? Me diz o nome que eu procuro primeiro.",
                "which file? Tell me its name and I'll find it first."),
                "FILE_ERROR")
        # "lê o punctuation" — a word of the filename in what the user just
        # said picks the file, before falling back to "the one on the table".
        named = self._match_candidates(raw_text)
        if len(named) == 1:
            path = named[0]
        elif len(candidates) == 1:
            path = candidates[0]
        else:
            names = ", ".join(os.path.basename(c) for c in candidates[:5])
            return (self._t(f"qual deles? {names}", f"which one? {names}"),
                    "FILE_ERROR")
        return self._stage_file_read(path, raw_text)

    def _stage_file_read(self, path: str, question: str):
        """Ask once per file, then answer: the staged read every deterministic
        read route shares, so the gate is identical however it was asked."""
        args = {"path": path, "question": question}
        if self._read_file_approved(os.path.realpath(path)):
            return self._run_read_file_at(args)
        self._pending_action = {"type": "read_file_at", "args": args}
        self._persist_pending_action()
        return (self._human_confirmation_prompt(self._pending_action),
                "AGENT_AWAITING_CONFIRMATION")

    def _stage_files_read(self, paths: list, question: str):
        """Several files under ONE confirmation (s85) — the folder twin of
        _stage_file_read. The language rides along: the "sim" that approves
        it reads as Portuguese whatever the conversation was in."""
        for path in paths:
            self._remember_seen_path(path)
        args = {"paths": list(paths), "question": question,
                "lang": getattr(self, "_reply_lang", "pt") or "pt"}
        if all(self._read_file_approved(os.path.realpath(p)) for p in paths):
            return self._run_read_files(args)
        self._pending_action = {"type": "read_files", "args": args}
        self._persist_pending_action()
        return (self._human_confirmation_prompt(self._pending_action),
                "AGENT_AWAITING_CONFIRMATION")

    # How many files a folder listing names before "and N more".
    _FOLDER_LIST_MAX = 10
    # How many files may be read together under one confirmation. Two PDFs
    # of an exam are the case; ten files would not fit the window anyway.
    _READ_TOGETHER_MAX = 3
    # A folder message that also asks for a read — "read them and answer
    # every question" (13/10) — goes straight to one confirmation for its
    # files. Checked on the message WITHOUT the path, so a folder called
    # "read" does not count as asking.
    _READ_INTENT_RE = re.compile(
        r"\b(?:read|l[eê]|leia|ler|answer|respond[ae]|responder|"
        r"resolv[ae]|resolver|solve|summari[sz]e|resum[ae]|resuma)\b",
        re.IGNORECASE)
    # "os dois" / "both" right after a listing (s85: 0 of 12 such turns read
    # anything; "both" was answered from an old Qwen-vs-DeepSeek chat, since
    # the listing is a fixed reply the 8B never sees). The WHOLE message has
    # to be that, optionally with "e responde…": "todos os dias eu corro"
    # is not a read command.
    _LISTED_ALL_RE = re.compile(
        r"^(?:(?:pode\s+)?(?:l[eê]|leia|ler|abre|abra|read|open)\s+)?"
        r"(?:os\s+dois|as\s+duas|ambos|ambas|todos|todas|tudo|both|all)"
        r"(?:\s+(?:of\s+them|deles|delas|os\s+arquivos|os\s+pdfs|"
        r"the\s+files|files|pdfs))?"
        r"\s*[.!]?\s*(?:(?:e|and)\s+(?P<rest>.+?))?\s*[.!?]?$",
        re.IGNORECASE | re.DOTALL)
    _LISTED_TTL_S = 30 * 60

    def _offer_pick(self, paths: list) -> str:
        """Leave "which one?" standing for the NEXT turn, so "os dois", "sim"
        and "lê o X" can answer it — for any reply that shows a few files: a
        folder listing, a message naming two files, "not found, but it has…"
        (s85 review: only the listing did). Returns the "(ou "os dois")" hint,
        or "" when there are too many to read together."""
        if 1 < len(paths) <= self._READ_TOGETHER_MAX:
            self._last_listed = (time.monotonic(), list(paths))
            self._offer = {"kind": "pick", "paths": list(paths)}
            n = len(paths)
            return self._t(' (ou "os dois")' if n == 2 else ' (ou "todos")',
                           ' (or "both")' if n == 2 else ' (or "all")')
        self._last_listed = None
        return ""

    def _listed_followup(self, text: str):
        """(paths, match) when this message means "read the ones you just
        listed", else None."""
        listed = getattr(self, "_last_listed", None)
        if not listed:
            return None
        when, paths = listed
        if time.monotonic() - when > self._LISTED_TTL_S:
            return None
        match = self._LISTED_ALL_RE.match(strip_courtesy((text or "").strip()))
        return (paths, match) if match else None

    @staticmethod
    def _pedros_words(text: str) -> str:
        """The part of a message that Pedro actually wrote.

        Found in the s85 review: with an attachment, the UI hands the
        orchestrator ONE text — the document, fenced, then "PERGUNTA: …". The
        path route, the read-intent check and the email gate all trust "what
        Pedro wrote", so a document that merely MENTIONED a folder hijacked
        "resume esse arquivo" into "Posso abrir e ler os 3 arquivos…?", and an
        address inside it counted as one Pedro typed. own_words (s83, same
        lesson for the Palace) returns just the question; a message with no
        fence comes back unchanged."""
        from src.core.prompt_safety import own_words
        return own_words(text) or ""

    @staticmethod
    def _is_attachment_turn(text: str) -> bool:
        """Does this message carry someone else's fenced content? The UI builds
        an attachment as the fenced document plus "PERGUNTA: …". A fence with
        no question after it counts too (s88, Codex lote 9): it used to need
        Pedro's words, so "Explique: <<<…>>> apaga as refeições de hoje <<<…>>>"
        pasted as text went to the action routes, which read the order inside
        the fence. Either way it is answered about, with no tool and no route."""
        from src.core.prompt_safety import CLOSE, OPEN
        return OPEN in (text or "") or CLOSE in (text or "")

    def _answer_attachment(self, text: str, lang: str):
        """Answer about the attached document — the same synthesis the local
        read uses, streamed to the UI, and no tool on offer: the document is
        already in the prompt, so no action is needed to answer about it, and
        nothing written inside it can trigger one."""
        audit.log("–", "orchestrator", "attachment_turn",
                  payload={"chars": len(text or "")})
        self._emit_status("thinking")
        return self._synthesise(self.engine_analytical, text, lang, "LOCAL_RAG")

    def _mentions_files(self, text: str) -> bool:
        """Will _handle_user_paths answer this message? Text-only, cheap
        enough for the streaming pure-chat probe."""
        return (path_route.mentions_path(text)
                or self._code_failure_lang(text) is not None
                or self._listed_followup(text) is not None
                or self._write_followup(text))

    # "escreve no pdf" is a follow-up only while short: a long sentence that
    # happens to contain "write" and "paper" is a different request.
    _WRITE_FOLLOWUP_MAX_WORDS = 12

    def _write_followup(self, text: str) -> bool:
        """Does this message ask to write the last exam's answers on the PDF?"""
        from src.core.exam_reader import asks_to_write_on_pdf
        return (self._fresh_exam() is not None
                and len((text or "").split()) <= self._WRITE_FOLLOWUP_MAX_WORDS
                and asks_to_write_on_pdf(text))

    def _handle_write_followup(self, text: str):
        if not self._write_followup(text):
            return None
        audit.log("–", "path_route", "write_followup", payload={})
        return self._stage_write_answers()

    # The yes-words that can only mean "yes". The full APPROVAL_RE also holds
    # thanks and acks — "valeu", "show", "beleza", "kk" — which, with nothing
    # asked, are conversation, not a confirmation, and stay with the chat.
    _EXPLICIT_YES_RE = re.compile(
        r"^[\"'`(\[]?(?:sim|ssim|yes|yse|yess|ys|claro|pode|autorizo|"
        r"autoriza|confirmo|confirma)[\"'`)\]]?[.!,]?$", re.IGNORECASE)

    def _answers_email_question(self, text: str) -> bool:
        offer = getattr(self, "_offer", None) or {}
        return (offer.get("kind") == "email_address"
                and bool(self._EMAIL_IN_TEXT_RE.search(text or "")))

    def _resume_email(self, text: str, offer: dict):
        """The address Cortex asked for: stage the kept draft to it — through
        the same gate (the address is now in Pedro's own message)."""
        found = self._EMAIL_IN_TEXT_RE.search(text or "")
        if not found:
            return None
        # This runs before the turn's own harvest further down, and the gate
        # only trusts addresses the user wrote — so record this one first.
        self.remember_emails_in(text)
        args = {**(offer.get("args") or {}), "to": found.group(0).rstrip(".")}
        audit.log("–", "flow", "email_address_given", payload={})
        brief = self._execute_tool("send_email", args)
        if self._pending_action is not None:
            return (self._human_confirmation_prompt(self._pending_action),
                    "AGENT_AWAITING_CONFIRMATION")
        return (str(brief).replace(SKIP_FOLLOWUP_MARKER, ""), "AGENT_ACTION_ERROR")

    def _handle_unpending_yes(self, text: str, offer: Optional[dict]):
        """A yes with no staged action: the answer to Cortex's last question,
        or an honest "nothing to confirm". None leaves it to the chat."""
        kind = (offer or {}).get("kind")
        if kind == "write_pdf" and self._fresh_exam() is not None:
            audit.log("–", "flow", "yes_to_offer", payload={"offer": kind})
            return self._stage_write_answers()
        if kind == "pick" and (offer or {}).get("paths"):
            audit.log("–", "flow", "yes_to_offer", payload={"offer": kind})
            names = [os.path.basename(p) for p in offer["paths"]]
            listed = self._t(" ou ", " or ").join(
                [", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)
            both = (self._t("os dois", "both") if len(names) == 2
                    else self._t("todos", "all of them"))
            self._offer = offer                    # still asking which
            return (self._t(f"Qual deles? {listed}? Ou {both}?",
                            f"Which one? {listed}? Or {both}?"), "FILE_LIST")
        if self._EXPLICIT_YES_RE.match(text):
            audit.log("–", "flow", "nothing_pending", payload={"input": text[:20]})
            return (self._t(
                "Não tem nada pendente pra confirmar agora — o que você quer "
                "que eu faça?",
                "There's nothing waiting for a yes right now — what would you "
                "like me to do?"), "NOTHING_PENDING")
        return None

    def _handle_listed_followup(self, text: str):
        hit = self._listed_followup(text)
        if hit is None:
            return None
        paths, match = hit
        self._last_listed = None
        audit.log("–", "path_route", "listed_followup",
                  payload={"files": len(paths)})
        question = (match.group("rest") or "").strip()
        if len(paths) == 1:
            return self._stage_file_read(paths[0], question or text)
        return self._stage_files_read(paths, question)

    def _read_folder(self, folder: str, question: str):
        """Stage the read of a folder's files, or None when there are too
        many to read together (the caller lists them instead)."""
        from src.tools.file_finder import folder_gate, readable_files_in

        if not folder_gate(folder)[0]:
            return None
        found = [p for p, _size in readable_files_in(folder)]
        if not found or len(found) > self._READ_TOGETHER_MAX:
            return None
        if len(found) == 1:
            self._remember_seen_path(found[0])
            return self._stage_file_read(found[0], question)
        return self._stage_files_read(found, question)

    def _handle_code_change_phrase(self, text: str, mentions: list):
        from src.tools.file_finder import is_readable_path

        first = mentions[0]
        prefix, _, rest = text.partition(first.typed)
        prefix = prefix.lstrip().lower()
        if prefix not in ("muda em ", "mude em ", "muda no ", "muda na ", "change ", "change in "):
            return None
        lang = "en" if prefix.startswith("change ") else "pt"
        self._reply_lang = lang
        request = re.sub(r"^\s*:\s*", "", rest).strip()
        if ((self._turn_origin() or {}).get("channel") not in ("terminal", "ui")
                or getattr(self, "background", False)):
            return self._code_change_typed_refusal(lang), "CODE_PROPOSAL_REFUSED"
        if not request:
            return (self._t(
                f"Me diz o que mudar: muda em {first.typed}: o que mudar.",
                f"Tell me what to change: change {first.typed}: what to change."), "CODE_PROPOSAL_REFUSED")
        if len(mentions) > 1 or first.kind == path_route.FOLDER:
            return (self._t(
                "Uma mudança por vez, em UM arquivo .py: muda em ~/pasta/arquivo.py: o que mudar.",
                "One change at a time, in ONE .py file: change ~/folder/file.py: what to change."), "CODE_PROPOSAL_REFUSED")
        if first.kind == path_route.MISSING:
            return self._say_path_missing(first)
        if not os.path.realpath(first.path).lower().endswith(".py"):
            return (self._t(
                "Só proponho mudança em arquivo .py de uma pasta autorizada para código.",
                "I only propose changes to a .py file in a folder authorised for code."), "CODE_PROPOSAL_REFUSED")
        ok, real = is_readable_path(first.path)
        if not ok:
            name = os.path.basename(first.path)
            return (self._t(f"não posso mudar {name}: {real}.",
                            f"I can't change {name}: {real}."), "FILE_ERROR")
        return self._stage_code_change(real, request, lang)

    def _handle_code_syntax_phrase(self, text: str, mentions: list):
        from src.tools.file_finder import is_readable_path

        first = mentions[0]
        prefix, _, rest = text.partition(first.typed)
        prefix = prefix.lstrip().lower()
        if not re.fullmatch(r"(?:(?:checa|confere) a sintaxe (?:em|de|do|da)|check the syntax (?:of|in)) ", prefix):
            return None
        # More than one path has its own refusal; extra prose keeps the old route.
        if len(mentions) == 1 and not re.fullmatch(r"[\s.?!]*", rest):
            return None
        self._reply_lang = "en" if prefix.startswith("check ") else "pt"
        if len(mentions) > 1:
            return (self._t(
                "Uma pasta ou um arquivo .py por vez: checa a sintaxe em ~/pasta.",
                "One folder or one .py file at a time: check the syntax of ~/folder."), "CODE_SYNTAX_REFUSED")
        if first.kind == path_route.MISSING:
            return self._say_path_missing(first)
        real = os.path.realpath(first.path)
        name = os.path.basename(real)
        folder = first.kind == path_route.FOLDER
        if folder:
            if real not in code_roots.roots() and not code_roots.inside(real):
                return (self._t(
                    f"não posso conferir a pasta {name}: ela não está autorizada para código.",
                    f"I can't check the folder {name}: it isn't authorised for code."), "FILE_ERROR")
        else:
            if not real.lower().endswith(".py"):
                return (self._t(
                    "Só confiro a sintaxe de arquivos .py de uma pasta autorizada para código.",
                    "I only check the syntax of .py files in a folder authorised for code."), "CODE_SYNTAX_REFUSED")
            ok, real = is_readable_path(first.path)
            if not ok:
                name = os.path.basename(first.path)
                return (self._t(f"não posso conferir {name}: {real}.",
                                f"I can't check {name}: {real}."), "FILE_ERROR")
        return self._check_code_syntax(real, folder)

    def _handle_code_tests_phrase(self, text: str, mentions: list):
        first = mentions[0]
        prefix, _, rest = text.partition(first.typed)
        prefix = prefix.lstrip().lower()
        if not re.fullmatch(r"(?:(?:roda|rode) os testes (?:em|de|do|da)|run the tests (?:in|of)) ", prefix):
            return None
        # Extra prose keeps the old route; multiple paths get a fixed refusal.
        if len(mentions) == 1 and not re.fullmatch(r"[\s.?!]*", rest):
            return None
        self._reply_lang = "en" if prefix.startswith("run ") else "pt"
        if ((self._turn_origin() or {}).get("channel") not in ("terminal", "ui")
                or getattr(self, "background", False)):
            return self._code_tests_typed_refusal(), "CODE_TESTS_REFUSED"
        if len(mentions) > 1:
            return (self._t("Uma pasta por vez: roda os testes em ~/pasta.",
                            "One folder at a time: run the tests in ~/folder."), "CODE_TESTS_REFUSED")
        if first.kind == path_route.MISSING:
            return self._say_path_missing(first)
        if first.kind != path_route.FOLDER:
            return (self._t("Me aponta a pasta do projeto: roda os testes em ~/pasta.",
                            "Point me to the project folder: run the tests in ~/folder."), "CODE_TESTS_REFUSED")
        real = os.path.realpath(first.path)
        name = os.path.basename(real)
        if real not in code_roots.roots() and not code_roots.inside(real):
            return (self._t(
                f"não posso rodar testes em {name}: a pasta não está autorizada para código.",
                f"I can't run tests in {name}: the folder isn't authorised for code."), "FILE_ERROR")
        return self._stage_code_tests(real)

    @staticmethod
    def _parse_pdf_pages(spec: str):
        """Parse ranges without expanding untrusted, potentially huge numbers."""
        spec = re.sub(r"^(?:(?:a|as)\s+)?páginas?\s+|^pages?\s+", "", spec.strip().lower())
        number = r"[0-9]+"
        part = rf"{number}(?:\s*(?:-|a|to)\s*{number})?"
        if not re.fullmatch(rf"{part}(?:\s*(?:,|e|and)\s*{part})*", spec):
            raise ValueError("invalid pages")
        ranges = []
        for item in re.split(r"\s*(?:,|and|e)\s*", spec):
            values = re.split(r"\s*(?:-|to|a)\s*", item)
            start, end = int(values[0]), int(values[-1])
            if start < 1 or end < start:
                raise ValueError("invalid range")
            ranges.append((start, end))
        return ranges

    def _answer_pdf_split_question(self, text: str):
        question = self._pdf_split_question
        self._pdf_split_question = None
        if (not question or not self._pdf_merge_typed()
                or question["origin"] != self._turn_origin()
                or time.time() >= question["expires"]
                or not re.fullmatch(r"\s*[12]\.?\s*", text)):
            return None
        self._reply_lang = question["lang"]
        # This is already a resolved path, not another natural-language turn.
        # Re-parsing it would lose escapes and impose the finder's word limit.
        path = question["path"]
        kind = (path_route.FOLDER if os.path.isdir(path) else
                path_route.FILE if os.path.exists(path) else path_route.MISSING)
        mention = path_route.PathMention(path, path, kind)
        return self._prepare_pdf_split(mention, [(p, p) for p in question["pages"]],
                                       "extract" if text.strip().rstrip(".") == "1" else "remove")

    def _handle_pdf_images_phrase(self, text: str, mentions: list):
        if len(mentions) != 1:
            return None
        prefix, _, suffix = text.partition(mentions[0].typed)
        if not re.fullmatch(r"[ .!?]*", suffix):
            return None
        patterns = [
            (r"\s*(?:tira|tire|extrai|extraia|salva|salve|pega|pegue)\s+"
             r"(?:(?:todas\s+)?as\s+)?(?:imagens|fotos|figuras)"
             r"(?:\s+das\s+(?P<spec>.+?))?\s+(?:de|do|da)\s+", "pt"),
            (r"\s*(?:extract|save|get)\s+(?:all\s+)?(?:the\s+)?(?:images|pictures)"
             r"(?:\s+(?:from|of)\s+(?P<spec>.+?))?\s+(?:from|of)\s+", "en"),
        ]
        for pattern, lang in patterns:
            match = re.fullmatch(pattern, prefix, re.I)
            if match:
                break
        else:
            return None
        self._reply_lang = lang
        if not self._pdf_merge_typed():
            return self._pdf_images_typed_refusal(), "PDF_IMAGES_REFUSED"
        try:
            ranges = self._parse_pdf_pages(match.group("spec")) if match.group("spec") else []
        except ValueError:
            return (self._t(
                "Não entendi as páginas. Exemplo: tira as imagens das páginas 3 a 5 de ~/x.pdf.",
                "I didn't understand the pages. Example: extract the images from pages 3 to 5 of ~/x.pdf."),
                "PDF_IMAGES_REFUSED")
        return self._prepare_pdf_images(mentions[0], ranges)

    def _prepare_pdf_images(self, mention, ranges):
        from src.core import pdf_info, pdf_publish, pdf_runner
        from src.tools.file_finder import is_readable_path

        if mention.kind == path_route.MISSING:
            return self._say_path_missing(mention)
        if mention.kind == path_route.FOLDER:
            return (self._t("Só tiro imagens de um arquivo .pdf.",
                            "I only take images out of a .pdf file."), "PDF_IMAGES_REFUSED")
        ok, path = is_readable_path(mention.path)
        if not ok:
            source = os.path.basename(mention.path)
            return (self._t(f"não posso ler {source}: {path}.",
                            f"I can't read {source}: {path}."), "FILE_ERROR")
        if not path.lower().endswith(".pdf"):
            return (self._t("Só tiro imagens de um arquivo .pdf.",
                            "I only take images out of a .pdf file."), "PDF_IMAGES_REFUSED")
        source = os.path.basename(path)
        try:
            [digest] = self._pdf_merge_hashes([path])
        except ValueError:
            return (self._t(f"{source} passa de 50 MB — grande demais para tirar imagens.",
                            f"{source} exceeds 50 MB — too large to take images out."), "PDF_IMAGES_REFUSED")
        except OSError:
            return self._pdf_images_error()
        info = pdf_info.inspect(path, cancelled=lambda: StagingMixin._turn_cancelled(self))
        if info.status != "ok":
            replies = {
                "protected": (f"{source} tem senha — não tiro imagens de PDF com senha.",
                              f"{source} has a password — I don't take images out of password-protected PDFs."),
                "broken": (f"{source} parece danificado — não consegui abrir.",
                           f"{source} looks damaged — I couldn't open it."),
                "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.",
                                "The PDF tools (Poppler) aren't available on this Mac."),
            }
            return (self._t(*replies.get(info.status, (
                f"Não consegui conferir {source} — nada foi feito.",
                f"I couldn't check {source} — nothing was done."))), "PDF_IMAGES_REFUSED")
        for start, end in ranges:
            if end > info.pages:
                missing = max(start, info.pages + 1)
                return (self._t(f"{source} tem {info.pages} páginas — a página {missing} não existe.",
                                f"{source} has {info.pages} pages — page {missing} doesn't exist."), "PDF_IMAGES_REFUSED")
        pages = sorted({p for a, b in ranges for p in range(a, b + 1)})
        runs = [list(run) for run in self._pdf_page_runs(pages)]
        cancelled = lambda: StagingMixin._turn_cancelled(self)
        try:
            with pdf_runner.PdfJob() as job:
                [staged] = job.stage([path])
                if self._pdf_merge_hashes([staged]) != [digest]:
                    return self._pdf_merge_changed(path, "PDF_IMAGES_REFUSED")
                inventory = self._pdf_images_inventory(job, staged, runs, info.pages, cancelled)
                if inventory is None or cancelled():
                    raise ValueError("inventory failed")
            keep = self._pdf_images_keep(inventory)
        except (OSError, ValueError):
            return (self._t(f"Não consegui conferir as imagens de {source} — nada foi feito.",
                            f"I couldn't check the images in {source} — nothing was done."), "PDF_IMAGES_REFUSED")
        if len(keep) > 50:
            return (self._t(
                f"{source} tem {len(keep)} imagens — consigo até 50 por vez. Escolha páginas, por exemplo: tira as imagens das páginas 3 a 5 de ~/x.pdf.",
                f"{source} has {len(keep)} images — I can do up to 50 at a time. Pick pages, e.g.: extract the images from pages 3 to 5 of ~/x.pdf."),
                "PDF_IMAGES_REFUSED")
        if sum(map(len, inventory)) > 100:
            return (self._t("Essas páginas têm itens demais para tirar de uma vez — escolha menos páginas.",
                            "Those pages hold too many items to take at once — pick fewer pages."), "PDF_IMAGES_REFUSED")
        if not keep:
            return (self._t(f"Não achei imagens em {source} (fora máscaras e ícones pequenos) — nada a fazer.",
                            f"I found no images in {source} (leaving out masks and tiny icons) — nothing to do."), "PDF_IMAGES_REFUSED")
        label = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
        action = (("imagens extraídas" + (f" das páginas {label}" if label else "")) if self._reply_lang == "pt" else
                  ("extracted images" + (f" of pages {label}" if label else "")))
        name = pdf_publish.next_free(pdf_publish.folder_name(source, action, datetime.now().date()))
        if name is None:
            return (self._t(
                "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo.",
                "There are already 99 folders with that name in ~/Documents/Cortex/PDF — delete some and ask again."),
                "PDF_IMAGES_REFUSED")
        self._pending_action = {"type": "pdf_images", "args": {
            "input": {"path": path, "sha256": digest, "pages": info.pages},
            "runs": runs, "inventory": inventory, "keep": keep,
            "label": label, "name": name, "lang": self._reply_lang}}
        self._persist_pending_action()
        return self._human_confirmation_prompt(self._pending_action), "AGENT_AWAITING_CONFIRMATION"

    def _handle_pdf_split_phrase(self, text: str, mentions: list):
        if len(mentions) != 1:
            return None
        prefix, _, suffix = text.partition(mentions[0].typed)
        match = re.fullmatch(
            r"\s*(separa|separe|extrai|extraia|extract|remove|apaga|apague|delete|tira|tire|take out)"
            r"\s+(.+?)\s+(de|do|da|of|from)\s+", prefix, re.I)
        if not match or not re.fullmatch(r"[ .!?]*", suffix):
            return None
        verb, spec, join = (v.lower() for v in match.groups())
        self._reply_lang = ("pt" if re.search(r"\bpáginas?\b", spec) or join in ("de", "do", "da")
                            else "en")
        if not self._pdf_merge_typed():
            return self._pdf_split_typed_refusal(), "PDF_SPLIT_REFUSED"
        try:
            ranges = self._parse_pdf_pages(spec)
        except ValueError:
            return (self._t(
                "Não entendi as páginas. Exemplo: separa as páginas 3 a 5 de ~/x.pdf.",
                "I didn't understand the pages. Example: extract pages 3 to 5 from ~/x.pdf."),
                "PDF_SPLIT_REFUSED")
        mode = ("ask" if verb in ("tira", "tire", "take out") else
                "remove" if verb in ("remove", "apaga", "apague", "delete") else "extract")
        return self._prepare_pdf_split(mentions[0], ranges, mode)

    def _handle_pdf_png_phrase(self, text: str, mentions: list):
        if len(mentions) != 1:
            return None
        prefix, _, suffix = text.partition(mentions[0].typed)
        # The image word is mandatory. Keep the two positions separate so
        # arbitrary prose after a path never becomes a rendering request.
        patterns = [
            (r"\s*(?:(?P<verb>salva|salve|transforma|transforme|converte|converta|exporta|exporte)\s+)?"
             r"(?P<spec>.+?)\s+(?:de|do|da)\s+", r"\s+(?:como|em)\s+(?:imagem|imagens|png)[ .!?]*", "pt"),
            (r"\s*(?:(?P<verb>save|export|convert)\s+)?(?P<spec>.+?)\s+(?:of|from)\s+",
             r"\s+(?:as\s+(?:image|images|png)|to\s+png)[ .!?]*", "en"),
            (r"\s*(?:imagem|imagens|png|pngs)\s+(?:da|das)\s+(?P<spec>.+?)\s+(?:de|do|da)\s+",
             r"[ .!?]*", "pt"),
            (r"\s*(?:image|images|png)\s+of\s+(?P<spec>.+?)\s+(?:of|from)\s+", r"[ .!?]*", "en"),
        ]
        for before, after, lang in patterns:
            match = re.fullmatch(before, prefix, re.I)
            if match and re.fullmatch(after, suffix, re.I):
                break
        else:
            return None
        spec = match.group("spec")
        if ("verb" in match.groupdict() and match.group("verb") is None
                and not re.match(r"(?:(?:a|as)\s+)?páginas?\b|pages?\b|[0-9]", spec, re.I)):
            return None
        self._reply_lang = lang
        if not self._pdf_merge_typed():
            return self._pdf_png_typed_refusal(), "PDF_PNG_REFUSED"
        try:
            ranges = self._parse_pdf_pages(spec)
        except ValueError:
            return (self._t(
                "Não entendi as páginas. Exemplo: salva as páginas 3 a 5 de ~/x.pdf como imagem.",
                "I didn't understand the pages. Example: save pages 3 to 5 from ~/x.pdf as images."),
                "PDF_PNG_REFUSED")
        return self._prepare_pdf_png(mentions[0], ranges)

    def _prepare_pdf_png(self, mention, ranges):
        from src.core import pdf_info, pdf_publish
        from src.tools.file_finder import is_readable_path

        if mention.kind == path_route.MISSING:
            return self._say_path_missing(mention)
        if mention.kind == path_route.FOLDER:
            return (self._t("Só salvo páginas de um arquivo .pdf como imagem.",
                            "I only save pages from a .pdf file as images."), "PDF_PNG_REFUSED")
        ok, path = is_readable_path(mention.path)
        if not ok:
            source = os.path.basename(mention.path)
            return (self._t(f"não posso ler {source}: {path}.",
                            f"I can't read {source}: {path}."), "FILE_ERROR")
        if not path.lower().endswith(".pdf"):
            return (self._t("Só salvo páginas de um arquivo .pdf como imagem.",
                            "I only save pages from a .pdf file as images."), "PDF_PNG_REFUSED")
        source = os.path.basename(path)
        try:
            [digest] = self._pdf_merge_hashes([path])
        except ValueError:
            return (self._t(f"{source} passa de 50 MB — grande demais para salvar páginas como imagem.",
                            f"{source} exceeds 50 MB — too large to save pages as images."), "PDF_PNG_REFUSED")
        except OSError:
            return self._pdf_png_error()
        info = pdf_info.inspect(path, cancelled=lambda: StagingMixin._turn_cancelled(self))
        if info.status != "ok":
            replies = {
                "protected": (f"{source} tem senha — não salvo páginas de PDF com senha como imagem.",
                              f"{source} has a password — I don't save pages from password-protected PDFs as images."),
                "broken": (f"{source} parece danificado — não consegui abrir.",
                           f"{source} looks damaged — I couldn't open it."),
                "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.",
                                "The PDF tools (Poppler) aren't available on this Mac."),
            }
            return (self._t(*replies.get(info.status, (
                f"Não consegui conferir {source} — nada foi feito.",
                f"I couldn't check {source} — nothing was done."))), "PDF_PNG_REFUSED")
        # Check physical bounds before expansion, as split does.
        for start, end in ranges:
            if end > info.pages:
                missing = max(start, info.pages + 1)
                return (self._t(f"{source} tem {info.pages} páginas — a página {missing} não existe.",
                                f"{source} has {info.pages} pages — page {missing} doesn't exist."), "PDF_PNG_REFUSED")
        pages = sorted({p for a, b in ranges for p in range(a, b + 1)})
        if len(pages) > 10:
            return (self._t(f"Faço até 10 páginas por vez como imagem — você pediu {len(pages)}.",
                            f"I do up to 10 pages at a time as images — you asked for {len(pages)}."), "PDF_PNG_REFUSED")
        label = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in self._pdf_page_runs(pages))
        action = ((f"imagem da página {label}" if len(pages) == 1 else f"imagens das páginas {label}")
                  if self._reply_lang == "pt" else
                  (f"image of page {label}" if len(pages) == 1 else f"images of pages {label}"))
        name = pdf_publish.next_free(pdf_publish.folder_name(source, action, datetime.now().date()))
        if name is None:
            return (self._t(
                "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo.",
                "There are already 99 folders with that name in ~/Documents/Cortex/PDF — delete some and ask again."),
                "PDF_PNG_REFUSED")
        self._pending_action = {"type": "pdf_png", "args": {
            "input": {"path": path, "sha256": digest, "pages": info.pages},
            "keep": pages, "label": label, "name": name, "lang": self._reply_lang}}
        self._persist_pending_action()
        return self._human_confirmation_prompt(self._pending_action), "AGENT_AWAITING_CONFIRMATION"

    def _prepare_pdf_split(self, mention, ranges, mode):
        from src.core import pdf_info, pdf_publish
        from src.tools.file_finder import is_readable_path

        if mention.kind == path_route.MISSING:
            return self._say_path_missing(mention)
        if mention.kind == path_route.FOLDER:
            return (self._t("Só separo páginas de um arquivo .pdf.",
                            "I only split pages from a .pdf file."), "PDF_SPLIT_REFUSED")
        ok, path = is_readable_path(mention.path)
        if not ok:
            name = os.path.basename(mention.path)
            return (self._t(f"não posso ler {name}: {path}.",
                            f"I can't read {name}: {path}."), "FILE_ERROR")
        if not path.lower().endswith(".pdf"):
            return (self._t("Só separo páginas de um arquivo .pdf.",
                            "I only split pages from a .pdf file."), "PDF_SPLIT_REFUSED")
        source = os.path.basename(path)
        try:
            [digest] = self._pdf_merge_hashes([path])
        except ValueError:
            return (self._t(f"{source} passa de 50 MB — grande demais para separar páginas.",
                            f"{source} exceeds 50 MB — too large to split pages."), "PDF_SPLIT_REFUSED")
        except OSError:
            return self._pdf_split_error()
        info = pdf_info.inspect(path, cancelled=lambda: StagingMixin._turn_cancelled(self))
        status = info.status
        if status != "ok":
            replies = {
                "protected": (f"{source} tem senha — não separo páginas de PDF com senha.",
                              f"{source} has a password — I don't split password-protected PDFs."),
                "broken": (f"{source} parece danificado — não consegui abrir.",
                           f"{source} looks damaged — I couldn't open it."),
                "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.",
                                "The PDF tools (Poppler) aren't available on this Mac."),
            }
            return (self._t(*replies.get(status, (
                f"Não consegui conferir {source} — nada foi feito.",
                f"I couldn't check {source} — nothing was done."))), "PDF_SPLIT_REFUSED")
        for start, end in ranges:
            if end > info.pages:
                missing = max(start, info.pages + 1)
                return (self._t(f"{source} tem {info.pages} páginas — a página {missing} não existe.",
                                f"{source} has {info.pages} pages — page {missing} doesn't exist."),
                        "PDF_SPLIT_REFUSED")
        pages = sorted({p for a, b in ranges for p in range(a, b + 1)})
        selected = set(pages)
        removed = [p for p in range(1, info.pages + 1) if p not in selected]
        # Both choices must be feasible before offering the ambiguous choice.
        for keep in ([pages, removed] if mode == "ask" else [pages if mode == "extract" else removed]):
            if not keep:
                return (self._t("Sem essas páginas não sobra nada — nada foi feito.",
                                "Without those pages nothing is left — nothing was done."), "PDF_SPLIT_REFUSED")
            if len(keep) > 99:
                return (self._t(f"O resultado teria {len(keep)} páginas — consigo até 99 por vez.",
                                f"The result would have {len(keep)} pages — I can do up to 99 at a time."),
                        "PDF_SPLIT_REFUSED")
        label = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in self._pdf_page_runs(pages))
        if mode == "ask":
            self._pdf_split_question = {"path": path, "pages": pages, "lang": self._reply_lang,
                                        "origin": self._turn_origin(), "expires": time.time() + 15 * 60}
            return (self._t(
                f"Em {source} ({info.pages} {'página' if info.pages == 1 else 'páginas'}), você quer:\n"
                f"1 — um PDF só com as páginas {label}\n2 — uma cópia sem as páginas {label}\nResponda 1 ou 2.",
                f"In {source} ({info.pages} {'page' if info.pages == 1 else 'pages'}), do you want:\n"
                f"1 — a PDF with only pages {label}\n2 — a copy without pages {label}\nReply 1 or 2."),
                "PDF_SPLIT_QUESTION")
        action = (f"páginas {label}" if mode == "extract" else f"sem as páginas {label}") if self._reply_lang == "pt" else (
            f"pages {label}" if mode == "extract" else f"without pages {label}")
        name = pdf_publish.next_free(pdf_publish.folder_name(source, action, datetime.now().date()))
        if name is None:
            return (self._t(
                "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo.",
                "There are already 99 folders with that name in ~/Documents/Cortex/PDF — delete some and ask again."),
                "PDF_SPLIT_REFUSED")
        self._pending_action = {"type": "pdf_split", "args": {
            "input": {"path": path, "sha256": digest, "pages": info.pages}, "mode": mode,
            "keep": pages if mode == "extract" else removed, "label": label,
            "name": name, "lang": self._reply_lang}}
        self._persist_pending_action()
        return self._human_confirmation_prompt(self._pending_action), "AGENT_AWAITING_CONFIRMATION"

    def _handle_pdf_merge_phrase(self, text: str, mentions: list):
        from src.core import pdf_info, pdf_publish
        from src.tools.file_finder import is_readable_path

        prefix, first_path, rest = text.partition(mentions[0].typed)
        prefix = prefix.lstrip().lower()
        if prefix not in ("junta ", "junte ", "juntar ", "merge "):
            return None
        # The shared finder deduplicates paths. Consume one mention at a time
        # so repeated files remain distinct inputs, in exactly the typed order.
        rest = re.sub(r",(?=~/|/)", ", ", first_path + rest)
        ordered = []
        while rest:
            found = path_route.find_paths(rest)
            if not found or not rest.startswith(found[0].typed):
                return None
            first = found[0]
            ordered.append(first)
            rest = rest[len(first.typed):]
            if re.fullmatch(r"[ .!?]*", rest):
                break
            separator = re.match(r" *(?:e|and|,) *", rest)
            if not separator or not rest[separator.end():]:
                return None
            rest = rest[separator.end():]
        if not ordered:
            return None
        self._reply_lang = "en" if prefix == "merge " else "pt"
        if not self._pdf_merge_typed():
            return self._pdf_merge_typed_refusal(), "PDF_MERGE_REFUSED"
        if not 2 <= len(ordered) <= 3 or any(m.kind == path_route.FOLDER for m in ordered):
            return (self._t("Junto de 2 a 3 PDFs por vez: junta ~/a.pdf e ~/b.pdf.",
                            "I merge 2 to 3 PDFs at a time: merge ~/a.pdf and ~/b.pdf."), "PDF_MERGE_REFUSED")
        paths = []
        for mention in ordered:
            if mention.kind == path_route.MISSING:
                return self._say_path_missing(mention)
            ok, real = is_readable_path(mention.path)
            if not ok:
                name = os.path.basename(mention.path)
                return (self._t(f"não posso ler {name}: {real}.",
                                f"I can't read {name}: {real}."), "FILE_ERROR")
            if not real.lower().endswith(".pdf"):
                return self._t("Só junto arquivos .pdf.", "I only merge .pdf files."), "PDF_MERGE_REFUSED"
            paths.append(real)
        inputs = []
        for path in paths:
            name = os.path.basename(path)
            info = pdf_info.inspect(path, cancelled=lambda: StagingMixin._turn_cancelled(self))
            if info.status != "ok":
                replies = {
                    "protected": (f"{name} tem senha — não junto PDF com senha.",
                                  f"{name} has a password — I don't merge password-protected PDFs."),
                    "broken": (f"{name} parece danificado — não consegui abrir.",
                               f"{name} looks damaged — I couldn't open it."),
                    "unavailable": ("As ferramentas de PDF (Poppler) não estão disponíveis neste Mac.",
                                    "The PDF tools (Poppler) aren't available on this Mac."),
                }
                return (self._t(*replies.get(info.status, (
                    f"Não consegui conferir {name} — nada foi feito.",
                    f"I couldn't check {name} — nothing was done."))), "PDF_MERGE_REFUSED")
            inputs.append({"path": path, "pages": info.pages})
        try:
            hashes = self._pdf_merge_hashes(paths)
        except ValueError:
            return (self._t("Os PDFs passam de 50 MB juntos — grande demais para juntar.",
                            "The PDFs exceed 50 MB together — too large to merge."), "PDF_MERGE_REFUSED")
        except OSError:
            return self._pdf_merge_error()
        for item, digest in zip(inputs, hashes):
            item["sha256"] = digest
        name = pdf_publish.next_free(pdf_publish.folder_name(
            os.path.basename(paths[0]), "merged" if self._reply_lang == "en" else "juntado",
            datetime.now().date()))
        if name is None:
            return (self._t(
                "Já existem 99 pastas com esse nome em ~/Documents/Cortex/PDF — apague algumas e peça de novo.",
                "There are already 99 folders with that name in ~/Documents/Cortex/PDF — delete some and ask again."),
                "PDF_MERGE_REFUSED")
        self._pending_action = {"type": "pdf_merge", "args": {
            "inputs": inputs, "name": name, "lang": self._reply_lang}}
        self._persist_pending_action()
        return self._human_confirmation_prompt(self._pending_action), "AGENT_AWAITING_CONFIRMATION"

    def _handle_user_paths(self, text: str):
        """A path in the message, settled before any model sees it (s85).

        File → the staged read (confirmation first). Folder → its readable
        files, and which one to read. Not there → said so, with what IS in
        the folder when the folder exists. None when the message has no path.
        """
        from src.tools.file_finder import is_readable_path

        failure = self._handle_code_failure_phrase(text)
        if failure is not None:
            return failure
        mentions = path_route.find_paths(text)
        if not mentions:
            return (self._handle_listed_followup(text)
                    or self._handle_write_followup(text))
        proposal = self._handle_code_change_phrase(text, mentions)
        if proposal is not None:
            return proposal
        syntax = self._handle_code_syntax_phrase(text, mentions)
        if syntax is not None:
            return syntax
        tests = self._handle_code_tests_phrase(text, mentions)
        if tests is not None:
            return tests
        merge = self._handle_pdf_merge_phrase(text, mentions)
        if merge is not None:
            return merge
        images = self._handle_pdf_images_phrase(text, mentions)
        if images is not None:
            return images
        split = self._handle_pdf_split_phrase(text, mentions)
        if split is not None:
            return split
        png = self._handle_pdf_png_phrase(text, mentions)
        if png is not None:
            return png
        files = [m for m in mentions if m.kind == path_route.FILE]
        folders = [m for m in mentions if m.kind == path_route.FOLDER]
        audit.log("–", "path_route", "hit", payload={
            "files": len(files), "folders": len(folders),
            "missing": len(mentions) - len(files) - len(folders),
        })
        if len(files) > 1:
            # Drop typed paths entirely: filename digits are not question numbers.
            question = text
            for m in mentions:
                question = question.replace(m.typed, " ")
            question = " ".join(question.split())
            if (len(files) <= self._READ_TOGETHER_MAX
                    and self._READ_INTENT_RE.search(question)):
                for m in files:
                    ok, why = is_readable_path(m.path)
                    if not ok:
                        name = os.path.basename(m.path)
                        return (self._t(f"não posso ler {name}: {why}.",
                                        f"I can't read {name}: {why}."), "FILE_ERROR")
                return self._stage_files_read([m.path for m in files], question)
            for m in files:
                self._remember_seen_path(m.path)
            names = ", ".join(os.path.basename(m.path) for m in files[:5])
            hint = self._offer_pick([m.path for m in files])
            return (self._t(f"qual deles? {names}{hint}",
                            f"which one? {names}{hint}"), "FILE_ERROR")
        if files:
            path = files[0].path
            name = os.path.basename(path)
            ok, why = is_readable_path(path)
            if not ok:
                return (self._t(f"não posso ler {name}: {why}.",
                                f"I can't read {name}: {why}."), "FILE_ERROR")
            # The harvest stops at a plain space; the route does not.
            self._remember_seen_path(path)
            return self._stage_file_read(
                path, text.replace(files[0].typed, name))
        if folders:
            folder = folders[0]
            if self._READ_INTENT_RE.search(text.replace(folder.typed, " ")):
                name = os.path.basename(folder.path.rstrip(os.sep)) or folder.typed
                real = os.path.realpath(folder.path)
                if real in code_roots.roots() or code_roots.inside(real):
                    return self._stage_code_project(real, text.replace(folder.typed, name))
                staged = self._read_folder(
                    folder.path, text.replace(folder.typed, name))
                if staged is not None:
                    return staged
            return self._list_folder(folder.path)
        return self._say_path_missing(mentions[0])

    def _list_folder(self, folder: str):
        from src.tools.file_finder import folder_gate, human_size, readable_files_in

        name = os.path.basename(folder.rstrip(os.sep)) or folder
        ok, why = folder_gate(folder)
        if not ok:
            return (self._t(f"não posso abrir a pasta {name}: {why}.",
                            f"I can't open the folder {name}: {why}."),
                    "FILE_ERROR")
        found = readable_files_in(folder)
        if not found:
            return (self._t(
                f"a pasta {name} não tem nenhum arquivo que eu consiga ler "
                "(pdf, txt, md, docx).",
                f"the folder {name} has no file I can read "
                "(pdf, txt, md, docx)."), "FILE_LIST")
        for path, _size in found:
            self._remember_seen_path(path)
        # "os dois" / "sim" next turn mean these — when few enough to read together.
        together = self._offer_pick([p for p, _size in found])
        shown = found[: self._FOLDER_LIST_MAX]
        lines = "\n".join(f"- {os.path.basename(p)} ({human_size(size)})"
                          for p, size in shown)
        more = len(found) - len(shown)
        if more:
            lines += self._t(f"\n… e mais {more}.", f"\n… and {more} more.")
        n = len(found)
        return (self._t(
            f"Na pasta {name} tem {n} arquivo{'s' if n > 1 else ''} que eu "
            f"consigo ler:\n{lines}\nQual deles eu leio?{together}",
            f"The folder {name} has {n} file{'s' if n > 1 else ''} I can "
            f"read:\n{lines}\nWhich one should I read?{together}"), "FILE_LIST")

    def _say_path_missing(self, mention):
        from src.tools.file_finder import folder_gate, readable_files_in

        target = mention.path.rstrip(os.sep)
        name = os.path.basename(target) or mention.typed
        parent = os.path.dirname(target)
        if parent and folder_gate(parent)[0]:
            where = os.path.basename(parent) or parent
            found = readable_files_in(parent)
            if found:
                for path, _size in found:
                    self._remember_seen_path(path)
                names = ", ".join(os.path.basename(p) for p, _ in found[:5])
                hint = self._offer_pick([p for p, _ in found])
                ask = self._t(f" Quer ler algum?{hint}",
                              f" Want me to read one?{hint}") if hint else ""
                return (self._t(
                    f"não achei {name} na pasta {where}. Nela tem: {names}.{ask}",
                    f"I couldn't find {name} in the folder {where}. It has: "
                    f"{names}.{ask}"), "FILE_ERROR")
            return (self._t(f"não achei {name} na pasta {where}.",
                            f"I couldn't find {name} in the folder {where}."),
                    "FILE_ERROR")
        return (self._t(f"não achei esse caminho: {mention.typed}",
                        f"I couldn't find that path: {mention.typed}"),
                "FILE_ERROR")

    def _handle_read(
        self,
        engine: OllamaEngine,
        filename: str,
        specific_query: Optional[str],
        lang: str = "pt",
    ) -> Tuple[str, str]:
        self._emit_status("reading", filename)
        file_content = self.reader.read_document(filename)
        if file_content.startswith("[SYSTEM ERROR]"):
            return (file_content, "FILE_ERROR")
        query = specific_query.strip() if specific_query else "Summarise the key points of this document."
        # A local document is UNTRUSTED external content (OWASP LLM01): a file
        # can hide instructions aimed at the model ("ignore the query and…").
        # A live lab confirmed the old raw `--- {content} ---` framing let the
        # 8B obey such a hidden order. Fence it + a guard line (the same
        # prompt_safety helper the studier/screen-context already use); fence()
        # also strips forged markers so injected text can't break out. The
        # human gate on destructive tools stays the real backstop — this just
        # makes the model far less likely to obey injected text.
        file_content, how = select_relevant(
            file_content, specific_query, budget=MAX_DOC_CHARS)
        injected = document_prompt(file_content, query, lang)
        self._emit_status("thinking")
        reply, mode = self._synthesise(engine, injected, lang, "LOCAL_RAG")
        note_false_read(reply, "local_rag")
        return (reply + _partial_read_note(how, lang), mode)

    def _handle_write(
        self,
        engine: OllamaEngine,
        filename: str,
        content: str,
        lang: str = "pt",
    ) -> Tuple[str, str]:
        # This regex route can receive HTML directly, or generate it during
        # refinement. Both must use the same publisher as tool dispatch.
        if re.match(r'\s*(?:<!doctype\s+html\b|<html\b|```)', content, re.I):
            web = save_web_document(filename, content, lang, request_text=self._web_request_text())
            if web is not None:
                engine.history.extend([{'role': 'user', 'content': f'salve {filename} {content}'},
                                       {'role': 'assistant', 'content': web}])
                return (web, "SAVE_CODE")
        display_name = filename if filename.lower().endswith(('.html', '.htm', '.css', '.js', '.txt')) else filename + '.txt'
        self._emit_status("writing", display_name)
        refinement_prompt = (
            "Execute the following instruction with maximum technical detail. "
            "Format the response professionally. Do not include conversational text. "
            "Reply in the SAME language as the instruction below. "
            f"Instruction: '{content}'"
        )
        refined = engine.chat(refinement_prompt, response_language=lang, tool_backed=True)
        web = save_web_document(filename, refined, lang, request_text=self._web_request_text())
        if web is not None:
            # chat() already recorded the refinement; replace that assistant
            # turn so Finder and subsequent code requests see the receipt.
            if engine.history and engine.history[-1].get('role') == 'assistant':
                engine.history[-1]['content'] = web
            else:
                engine.history.extend([{'role': 'user', 'content': refinement_prompt},
                                       {'role': 'assistant', 'content': web}])
            return (web, "SAVE_CODE")
        if not filename.lower().endswith(".txt"):
            filename += ".txt"
        feedback = self.writer.write_document(filename, refined)
        ack = "Anotado." if lang == "pt" else "Noted."
        return (f"{ack} {feedback}", "LOCAL_WRITE")

    def _maybe_announce_abandoned(self, reply: str, mode: str) -> str:
        """Consume once, after synthesis and outside confirmation previews."""
        if mode in self._SILENT_ANNOUNCE_MODES or mode.startswith("AGENT_STAGE_"):
            return reply
        lock = getattr(self, "_abandoned_effects_lock", None)
        if lock is None:
            return reply
        with lock:
            effects = getattr(self, "_abandoned_effects", [])
            if not effects:
                return reply
            from src.core import effect_store
            claimed = set(effect_store.store.claim_notices([effect.id for effect in effects]))
            effects = [effect for effect in effects if effect.id in claimed]
            self._abandoned_effects = []
        if not effects:
            return reply
        labels = (self._UNDO_ACTION_LABELS_EN if getattr(self, "_reply_lang", "pt") == "en"
                  else self._UNDO_ACTION_LABELS)
        items = []
        for effect in effects[:3]:
            item = labels.get(effect.action, effect.action)
            if effect.action == "send_email":
                if effect.args.get("to"):
                    item += self._t(" para ", " to ") + str(effect.args["to"])
                if effect.args.get("subject"):
                    item += f' "{effect.args["subject"]}"'
            elif effect.action in {"create_event", "create_reminder"}:
                detail = effect.args.get("summary" if effect.action == "create_event" else "title")
                if detail:
                    item += f' "{detail}"'
            elif effect.action == "mac_quit_app" and effect.args.get("app_name"):
                item += " " + str(effect.args["app_name"])
            items.append(item)
        shown = "; ".join(items)
        if len(effects) > 3:
            shown += f" (+{len(effects) - 3})"
        return reply + self._t(
            f"\n\n⚠️ Antes de reiniciar, eu estava no meio de: {shown}. Não sei se foi feito — "
            "confere antes de pedir de novo; não vou repetir sozinho.",
            f"\n\n⚠️ Before restarting, I was in the middle of: {shown}. I don't know whether it "
            "went through — please check before asking again; I won't retry on my own.")

    def _maybe_announce_pending(self, reply: str, mode: str) -> str:
        """
        Append a morning reminder about pending work — Nightly Studier
        briefings AND background actions queued for approval — at most once
        per natural day. Suppressed on administrative and error modes (see
        _SILENT_ANNOUNCE_MODES).
        """
        if mode in self._SILENT_ANNOUNCE_MODES:
            return reply

        # M1-A8: the announce follows the turn's language via self._t
        # (_reply_lang is set by _process_core just before this runs).
        # EN users used to get a PT postscript glued onto an EN reply.
        parts: list[str] = []

        # 1. Nightly Studier briefings ready to review.
        topics = [p["topic"] for p in self.studies_review.list_pending()]
        if topics:
            n = len(topics)
            if n <= 3:
                shown, extra = ", ".join(topics), ""
            else:
                shown = ", ".join(topics[:3])
                extra = self._t(f" (+{n - 3} mais)", f" (+{n - 3} more)")
            if n == 1:
                body = self._t(f"1 estudo pronto pra revisar: {shown}",
                               f"1 study ready for review: {shown}")
            else:
                body = self._t(f"{n} estudos prontos pra revisar: {shown}{extra}",
                               f"{n} studies ready for review: {shown}{extra}")
            parts.append(self._t(
                f"você tem {body}. Diga 'revisar' quando quiser",
                f"you have {body}. Say 'review studies' whenever you like",
            ))

        # 2. Destructive (S3) actions queued by background work overnight,
        #    waiting for your approval (gate batch queue — design-reference ch. 13).
        #    The CLI is still the ONLY review surface for the gate queue
        #    (no voice route / no /ui panel yet — flagged in the backlog),
        #    so the honest instruction names it, phrased as a sentence
        #    instead of a bare command dump.
        try:
            queued = gate_queue.pending()
        except Exception:
            queued = []
        if queued:
            n = len(queued)
            kinds = ", ".join(q.get("type", "?") for q in queued[:3])
            extra = self._t(f" (+{n - 3} mais)", f" (+{n - 3} more)") if n > 3 else ""
            if n == 1:
                head = self._t("1 ação aguardando sua aprovação",
                               "1 action awaiting your approval")
            else:
                head = self._t(f"{n} ações aguardando sua aprovação",
                               f"{n} actions awaiting your approval")
            parts.append(head + f": {kinds}{extra}. " + self._t(
                "Pra revisar, roda o gate review no Mac "
                "(python -m src.gate_review)",
                "To review, run the gate review on the Mac "
                "(python -m src.gate_review)",
            ))

        if not parts:
            return reply

        today = datetime.now().strftime("%Y-%m-%d")
        if self._read_last_announce() == today:
            return reply

        suffix = (
            self._t("\n\n💡 Por sinal: ", "\n\n💡 By the way: ")
            + "; ".join(parts) + "."
        )
        self._write_last_announce(today)
        return reply + suffix

    def _read_last_announce(self) -> Optional[str]:
        try:
            with open(self._LAST_ANNOUNCE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip() or None
        except OSError:
            return None

    def _write_last_announce(self, value: str) -> None:
        try:
            os.makedirs(os.path.dirname(self._LAST_ANNOUNCE_PATH), exist_ok=True)
            with open(self._LAST_ANNOUNCE_PATH, "w", encoding="utf-8") as f:
                f.write(value)
        except OSError:
            pass

    # --- Agent action staging -----------------------------------------

    @staticmethod
    def _parse_reminder_due(
        time_expr: str, now: datetime
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert a relative time expression to an ISO 8601 datetime string.

        Handles "amanhã às 4h", "hoje às 18h", "amanhã", "às 16h", etc.
        Returns `(iso, assumed)`: `iso` is None when no parseable time is
        present; `assumed` carries the resolved "HH:MM" when the parser
        GUESSED the hour (M1-A12c) so the confirmation preview can say so
        — None when the user was explicit.

        Ambiguous hours (1–6) without an explicit "da manhã/tarde/noite"
        qualifier are assumed to be PM — in everyday pt-BR "às 4 horas"
        almost always means 4 da tarde. The confirmation prompt always shows
        the resolved time so the user can correct before saving.
        """
        if not time_expr:
            return None, None
        s = time_expr.strip().lower()

        # Base date
        tz = now.tzinfo or timezone.utc
        if re.search(r"\b(?:amanhã|amanha)\b", s):
            base = (now + timedelta(days=1)).date()
        else:
            base = now.date()

        # Extract hour and optional minute
        hour_m = re.search(
            r"(?:[àa]s?\s+)?(?P<h>\d{1,2})(?:[h:](?P<m>\d{1,2}))?"
            r"\s*(?:da\s+(?P<period>manh[ãa]|manha|tarde|noite|madrugada))?",
            s,
        )
        if not hour_m:
            # Only a day word ("amanhã") — anchor to 09:00 as a sensible
            # default; user will see and confirm this in the prompt.
            try:
                iso = datetime(base.year, base.month, base.day, 9, 0,
                               tzinfo=tz).isoformat()
                return iso, "09:00"
            except ValueError:
                return None, None

        h = int(hour_m.group("h"))
        m = int(hour_m.group("m") or 0)
        period = (hour_m.group("period") or "").lower()
        assumed: Optional[str] = None

        if "tarde" in period or "noite" in period:
            if h < 12:
                h += 12
        elif "manh" in period or "madrug" in period:
            pass  # keep literal
        else:
            # No qualifier: 1–6 h → assume tarde (16:00 beats 04:00 for
            # typical scheduling). 7–11 → keep as morning. 12–23 → correct.
            if 1 <= h <= 6:
                h += 12
                assumed = f"{h:02d}:{m:02d}"

        try:
            iso = datetime(base.year, base.month, base.day, h, m,
                           tzinfo=tz).isoformat()
            return iso, assumed
        except ValueError:
            return None, None

    @staticmethod
    def _calendar_end_iso(start_iso: str, default_hours: int = 1) -> str:
        """Return start_iso + `default_hours` as ISO 8601. The Calendar API
        requires an explicit end; without one we'd have to fall back to the
        LLM. Default of 1h matches Google Calendar's own UI default for new
        events. The user sees the resolved end in the confirmation prompt
        and can cancel + re-phrase for a different duration."""
        dt = datetime.fromisoformat(start_iso)
        return (dt + timedelta(hours=default_hours)).isoformat()

    # Tools the planner is allowed to compose plans over. Subset of
    # TOOL_SPECS allowed inside a Plan. spotify_control is intentionally
    # excluded (regex dispatch, not a TOOL_SPEC). web_search WAS
    # excluded until sessão 15 — the auto-multi-intent router (see
    # `multi_intent_match` above) needs it so phrases like "busca os
    # jogos E adiciona ao calendário" can sequence a web_search step
    # before the create_event step. _execute_tool_unstaged ("web_search"
    # branch in staging_mixin.py) returns "[SYSTEM] Web search is
    # unavailable: no active internet lease..." when no lease is open;
    # the planner's _ERROR_MARKERS now recognises that substring so the
    # plan fail-stops cleanly instead of pretending the search ran.
    _PLANNER_ALLOWED_TOOLS = frozenset({
        "read_document", "write_document",
        "list_emails", "send_email",
        "list_events", "create_event",
        "create_reminder",
        "web_search",
    })

    def _handle_planner_request(
        self, objective: str
    ) -> Optional[Tuple[str, str]]:
        """Decompose an objective into a Plan, stage it for user
        approval. Returns None when decomposition fails so the caller
        falls through to the LLM path (typo'd 'planeja' or empty body
        shouldn't dead-end).

        The plan is staged in _pending_action with type='execute_plan'
        and resolved on the next turn by the existing sim/não machinery.
        Plan-level approval covers all S3 steps inside — the preview
        lists every action so the gate stays honest."""
        if not objective:
            return None
        try:
            plan = _planner.decompose(
                objective,
                # M2-A7: bind keep_alive into the injected fn (late-bound on
                # ollama.chat so test patches still apply).
                llm_chat_fn=lambda **kw: ollama.chat(**kw, keep_alive=_KEEP_ALIVE, think=_THINK),
                model=self.engine_analytical.model_name,
                allowed_tools=sorted(self._PLANNER_ALLOWED_TOOLS),
            )
        except Exception:
            log.exception("planner: decompose raised")
            return None
        if plan is None or not plan.passos:
            return None

        # Serialise the plan for _pending_action persistence — we
        # rehydrate it on the next turn using the same shape. `lang` rides
        # along (M1-A8): the approving "sim/yes" arrives in a NEW turn whose
        # language heuristic can diverge ("sim" reads as PT regardless of
        # the conversation) — the summary must come back in the language
        # the plan was REQUESTED in, same lesson as the auto-search resume.
        self._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": plan.objetivo,
                "request_text": self._web_request_text() or objective,
                "lang": self._reply_lang,
                "passos": [
                    {
                        "id": s.id,
                        "tool": s.tool,
                        "args": s.args,
                        "depende_de": s.depende_de,
                        "descricao": s.descricao,
                    }
                    for s in plan.passos
                ],
            },
        }
        self._persist_pending_action()
        audit.log("–", "planner", "plan_staged", payload={
            "n_steps": len(plan.passos),
            "tools": [s.tool for s in plan.passos],
        })
        return (
            _planner.render_preview(plan, lang=self._reply_lang),
            "AGENT_AWAITING_CONFIRMATION",
        )

    # _execute_staged_plan / _execute_tool_unstaged live on
    # StagingMixin (src/core/staging_mixin.py).

    # Human labels for naming an action when undo can't reverse it.
    _UNDO_ACTION_LABELS = {
        "run_code_tests": "rodar os testes",
        "apply_code_change": "aplicar uma mudança de código",
        "send_email": "enviar um e-mail",
        "create_event": "criar um evento",
        "create_reminder": "criar um lembrete",
        "create_note": "criar uma nota",
        "append_to_note": "acrescentar a uma nota",
        "undo": "desfazer uma ação",
        "mac_quit_app": "fechar um app",
        "mac_sleep": "colocar o Mac pra dormir",
    }
    # EN parallel (s60) — picked by `_handle_undo` when _reply_lang == "en".
    _UNDO_ACTION_LABELS_EN = {
        "run_code_tests": "run the tests",
        "apply_code_change": "apply a code change",
        "send_email": "send an email",
        "create_event": "create an event",
        "create_reminder": "create a reminder",
        "create_note": "create a note",
        "append_to_note": "add to a note",
        "undo": "undo an action",
        "mac_quit_app": "quit an app",
        "mac_sleep": "put the Mac to sleep",
    }


    @staticmethod
    def _format_iso_for_display(iso: str) -> str:
        """
        Render an ISO 8601 datetime for the user in a friendly form
        ("seg 18/05 17:00 (BST)"). We strip any timezone suffix the LLM
        emitted, since the calendar client will force local timezone
        anyway — what the user will SEE on the calendar is the wall-clock
        time in their local zone.
        """
        s = (iso or "").strip()
        if not s:
            return "?"
        # The wall clock in the zone of the event's DATE — said on 24/10, 25/10
        # is GMT — by the rule the Reminders client uses (clock.py).
        dt = local_wall_time(s)
        if dt is None:
            return s
        local_tz = dt.tzname() or ""
        # Portuguese-ish weekday abbreviations
        weekdays = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
        wd = weekdays[dt.weekday()] if 0 <= dt.weekday() <= 6 else dt.strftime("%a")
        out = f"{wd} {dt.strftime('%d/%m')} {dt.strftime('%H:%M')}"
        if local_tz:
            out += f" ({local_tz})"
        return out

    def _format_confirmation_brief(self, name: str, args: dict) -> str:
        """LLM-facing English brief shown as the tool result when an
        S3 action is staged. Tells the model: action NOT executed,
        ask the user for sim/não confirmation. Delegates to the
        central dry-run builder so the per-tool layout stays
        symmetric with the human-facing prompt."""
        return _build_dry_run_preview(name, args)["llm"]

    # _execute_pending_action / _cancel_pending_action / _execute_tool /
    # _execute_tool_unstaged / _execute_staged_plan live on StagingMixin
    # (src/core/staging_mixin.py).

