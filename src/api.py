import hmac
import ipaddress
import json
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from src.core.orchestrator import CortexOrchestrator, _resolve_lang
from src.core.stream_reply import ReplyReplacement
from src.core import turn_metrics
from src.core.text_utils import strip_markdown_for_speech
from src.ui.conversation import buffer as conversation_buffer
from src.ui.routes import (
    router as ui_router, mount_static as ui_mount_static,
    _daily_history_limit, _daily_history_msgs,
)
from src.ui.chat_store import NETWORK_LABEL, TERMINAL_LABEL, VOICE_LABEL

load_dotenv()

app = FastAPI(title="CORTEX Network Interface")

SECRET_API_KEY = os.getenv("CORTEX_API_KEY")
if not SECRET_API_KEY:
    raise RuntimeError("CORTEX_API_KEY is not set. Copy .env.example to .env and fill it in.")

api_key_header = APIKeyHeader(name="X-Cortex-Key", auto_error=True)

# Rate limiting — per-IP throttle. the user is the only legitimate user,
# so the per-IP limit doubles as a brute-force / runaway-client guard.
# Default: 60 requests/min per IP across all /ask* endpoints. Override
# with CORTEX_RATE_LIMIT env var ("120/minute", "10/second", etc.).
_RATE_LIMIT = os.getenv("CORTEX_RATE_LIMIT", "60/minute").strip()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
auth_logger = logging.getLogger("cortex.auth")
auth_logger.setLevel(logging.INFO)
auth_logger.propagate = False
if not auth_logger.handlers:
    _h = logging.FileHandler(_LOG_DIR / "auth.log")
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    auth_logger.addHandler(_h)


# ---------- IP allowlist (Tailscale-friendly) ----------
# CORTEX_IP_ALLOWLIST is a comma-separated list of IPs or CIDRs. When
# set, requests from any other IP are rejected before auth runs — so a
# leaked API key still can't be used from outside the allowed range.
#
# Defence-in-depth, not replacement for Tailscale ACLs (which gate at
# the network layer). Both should be in place: Tailscale ACL says
# "only this device can even REACH the API"; allowlist below says
# "even if you got in, your IP must be in the Tailnet".
#
# Default empty = allow all (back-compat with the pre-allowlist
# deployment). Typical production value: "100.64.0.0/10" (entire
# Tailscale CGNAT range) plus "127.0.0.1/32" for local development.
def _parse_allowlist(raw: str) -> List[ipaddress._BaseNetwork]:
    """Parse a comma-separated list of IPs/CIDRs into network objects.
    Bare IPs are treated as /32 (v4) or /128 (v6). Invalid entries are
    skipped with a logged warning rather than failing startup — typo
    in env var shouldn't take down the server."""
    nets: List[ipaddress._BaseNetwork] = []
    for chunk in (raw or "").split(","):
        s = chunk.strip()
        if not s:
            continue
        try:
            nets.append(ipaddress.ip_network(s, strict=False))
        except ValueError:
            logging.getLogger("cortex.api").warning(
                "CORTEX_IP_ALLOWLIST: invalid entry %r (skipped)", s
            )
    return nets


_ALLOWLIST_RAW = os.getenv("CORTEX_IP_ALLOWLIST", "")
_ALLOWLIST = _parse_allowlist(_ALLOWLIST_RAW)

# Fail-open guard: if the env var was set (operator INTENDED to restrict
# access) but every entry was invalid, the parsed list is empty and the
# server would silently accept all IPs — the exact opposite of what was
# intended. Log a loud error so it shows up in startup logs even when
# the operator forgot to tail `cortex.api` warnings. Doesn't crash on
# purpose (see _parse_allowlist docstring) but makes the failure mode
# very visible.
if _ALLOWLIST_RAW.strip() and not _ALLOWLIST:
    logging.getLogger("cortex.api").error(
        "CORTEX_IP_ALLOWLIST was set (%r) but no valid entries were parsed. "
        "Server is running with NO IP restriction. Fix the env var and "
        "restart, or unset it to acknowledge open access.",
        _ALLOWLIST_RAW,
    )


def _ip_allowed(client_ip: str) -> bool:
    """True if the allowlist is empty (back-compat) or the IP falls in
    any configured network. Treats parse failures as denial — better
    to lock out a malformed proxy header than to silently allow.

    IPv4-mapped IPv6 unwrap: when the OS reports a client as
    "::ffff:100.64.5.5" (dual-stack socket accepting IPv4 connections
    via IPv6), the address parses as `IPv6Address` and would fail
    membership tests against an IPv4 CIDR like "100.64.0.0/10". We
    unwrap to the embedded IPv4 first so the allowlist behaves
    consistently regardless of how the OS surfaced the connection.
    """
    if not _ALLOWLIST:
        return True
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return any(addr in net for net in _ALLOWLIST)


class IPAllowlistMiddleware(BaseHTTPMiddleware):
    """Rejects requests from IPs outside CORTEX_IP_ALLOWLIST before
    any route handler or auth dependency runs.

    Runs ahead of `verify_access` so even a request with a valid API
    key is dropped if the IP is wrong. Skips entirely when the
    allowlist is empty (env var unset) for back-compat.

    Source-IP policy: deliberately uses `request.client.host` (the raw
    TCP source from Starlette) and NOT `X-Forwarded-For` / `X-Real-IP`.
    There is no trusted reverse proxy in the current deployment —
    accepting those headers would let any client spoof a Tailnet IP
    and bypass the allowlist. If a proxy is ever added in front of
    the API, this dispatch needs to (a) read XFF, (b) verify the
    immediate hop is the trusted proxy IP, and (c) strip any
    client-supplied XFF values past the last trusted hop.
    """

    async def dispatch(self, request: Request, call_next):
        if not _ALLOWLIST:
            return await call_next(request)
        client_ip = request.client.host if request.client else ""
        if _ip_allowed(client_ip):
            return await call_next(request)
        auth_logger.info(
            f"BLOCK ip={client_ip or 'unknown'} "
            f"path={request.url.path} reason=ip-not-allowlisted"
        )
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied: IP not in allowlist."},
        )


app.add_middleware(IPAllowlistMiddleware)


def verify_access(request: Request, api_key: str = Security(api_key_header)):
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    # Independent brute-force guard. slowapi's decorator runs AFTER the
    # auth dependency resolves, so an attacker spamming wrong keys would
    # be 403'd on every request without being throttled. This pre-check
    # closes that gap with a per-IP sliding window over failed attempts.
    if _failed_auth_exceeded(client_ip):
        auth_logger.info(f"BLOCK ip={client_ip} path={path} reason=brute-force")
        raise HTTPException(
            status_code=429,
            detail="Too many failed auth attempts — slow down."
        )
    # Constant-time compare: `!=` short-circuits on the first divergent
    # byte, leaking the key prefix through response timing (M10-A1).
    # Bytes, not str: compare_digest raises TypeError on non-ASCII str,
    # which would turn attacker-controlled header bytes into a 500.
    if not hmac.compare_digest(
        api_key.encode("utf-8"), SECRET_API_KEY.encode("utf-8")
    ):
        _record_failed_auth(client_ip)
        auth_logger.info(f"FAIL ip={client_ip} path={path}")
        raise HTTPException(status_code=403, detail="[SYSTEM ERROR] Access Denied. Invalid API Key.")
    auth_logger.info(f"OK   ip={client_ip} path={path}")
    return api_key


# ---------- failed-auth brute-force guard ----------
# Per-IP sliding window over failed auth attempts. Independent of slowapi
# because slowapi's @limiter.limit fires AFTER FastAPI resolves the
# verify_access dependency — so 403'd requests escape the rate limit.
_FAILED_AUTH_WINDOW_S = 60.0
_FAILED_AUTH_LIMIT = int(os.getenv("CORTEX_FAILED_AUTH_LIMIT", "10"))
_failed_auth: Dict[str, "deque[float]"] = {}


def _failed_auth_exceeded(client_ip: str) -> bool:
    """Returns True iff this IP has hit _FAILED_AUTH_LIMIT failures in
    the last _FAILED_AUTH_WINDOW_S seconds. Side effect: trims expired
    timestamps so the deque doesn't grow unbounded."""
    now = time.monotonic()
    cutoff = now - _FAILED_AUTH_WINDOW_S
    attempts = _failed_auth.get(client_ip)
    if attempts is None:
        return False
    while attempts and attempts[0] < cutoff:
        attempts.popleft()
    if not attempts:
        _failed_auth.pop(client_ip, None)  # GC empty entries
        return False
    return len(attempts) >= _FAILED_AUTH_LIMIT


def _record_failed_auth(client_ip: str) -> None:
    """Append a failed-auth timestamp for the given IP."""
    _failed_auth.setdefault(client_ip, deque()).append(time.monotonic())


orchestrator = CortexOrchestrator()


# ---------- Dashboard UI ----------
# Mounted AFTER the IP allowlist middleware (registered above) so that
# any /ui* request is filtered by the same allowlist as /ask. The
# router uses its own cookie/header auth.
#
# Orchestrator is exposed on app.state so ui routes (specifically
# /ui/chat/stream) can call it without a circular import back to this
# module. Tests can replace it with a mock before TestClient(app).
#
# Rate limit: /ui/chat/stream ends up calling orchestrator.process() which
# is just as expensive as /ask. The slowapi limiter buckets per
# endpoint, so /ui/chat/stream does NOT share /ask's budget. Acceptable for
# V1 because the UI is gated by IP allowlist + SameSite=Strict
# cookie (no cross-site POSTs reach it). Backlog: explicit per-IP
# limit on /ui/chat/stream if Cortex ever leaves the Tailnet.
app.state.orchestrator = orchestrator
app.include_router(ui_router)
ui_mount_static(app)


class CortexRequest(BaseModel):
    user_input: str
    image_b64: Optional[str] = None
    # Optional explicit language hint. Valid: "pt" / "en" (canonical),
    # plus a few common aliases ("pt-br", "english"). When provided, the
    # orchestrator skips its regex heuristic and uses this directly.
    # Clients that ALREADY KNOW the language (iOS Shortcuts bound to PT
    # or EN) should send this — the heuristic misses mixed-language
    # inputs ("play michael jackson no spotify") and unaccented PT
    # ("esta me ouvindo?"). Unrecognised values quietly fall back to
    # the heuristic so the server never errors on a typo.
    language: Optional[str] = None


class CortexResponse(BaseModel):
    response: str
    status_mode: str


class RunLogRequest(BaseModel):
    # Apple Watch → iOS Shortcuts automation → here. start_date is the
    # workout's start; it doubles as the dedup key so a daily automation that
    # re-sends today's runs never double-logs (import_run skips an existing ts).
    # distance/duration accept the raw Workout properties WITH units ("5 km",
    # "25 min", "0:25:30") so the Shortcut can dump them straight in — the
    # endpoint normalises. See apple_health.parse_distance_km/parse_duration_min.
    distance_km: Union[str, float]
    duration_min: Union[str, float]
    start_date: Optional[str] = None


def _normalise_language(raw: Optional[str]) -> Optional[str]:
    """Canonicalise the language tag sent by a client to 'pt' / 'en'
    or None. Quietly degrades on garbage input — the orchestrator's
    fallback heuristic still works."""
    if not raw:
        return None
    tag = raw.strip().lower()
    if tag in ("pt", "pt-br", "pt_br", "ptbr", "pt-pt", "portuguese",
               "portugues", "português"):
        return "pt"
    if tag in ("en", "en-gb", "en_gb", "en-us", "en_us", "english"):
        return "en"
    return None


@app.post("/ask", response_model=CortexResponse)
@limiter.limit(_RATE_LIMIT)
def process_query(request: Request, payload: CortexRequest, _api_key: str = Depends(verify_access)):
    if not payload.user_input.strip() and not payload.image_b64:
        raise HTTPException(status_code=400, detail="Input cannot be entirely empty.")

    set_origin = getattr(orchestrator, "set_turn_origin", None)
    turn_metrics.reset()
    try:
        host = request.client.host if request.client else None
        if set_origin is not None:
            set_origin("terminal" if host in _LOCAL_HOSTS else "network")
        reply, status_mode = orchestrator.process(
            payload.user_input,
            payload.image_b64,
            language=_normalise_language(payload.language),
            conversation_history=_daily_history_msgs(
                TERMINAL_LABEL if host in _LOCAL_HOSTS else NETWORK_LABEL,
                _daily_history_limit(orchestrator)),
        )
        _record_ask_turn(request, payload.user_input, reply, status_mode,
                         usage=turn_metrics.collect())
        return CortexResponse(response=reply, status_mode=status_mode)
    except Exception:
        # M10-A8: str(e) can carry internal paths/secrets — the client
        # gets a generic detail; the traceback goes to the server log
        # (before this, it was silently lost).
        ref = uuid.uuid4().hex[:8]
        logging.getLogger("cortex.api").exception("/ask orchestrator failure ref=%s", ref)
        raise HTTPException(status_code=500, detail="internal error",
                            headers={"X-Cortex-Ref": ref})
    finally:
        turn_metrics.collect()
        if set_origin is not None:
            set_origin(None)


@app.post("/ask_voice", response_class=PlainTextResponse)
@limiter.limit(_RATE_LIMIT)
def process_query_voice(request: Request, payload: CortexRequest, _api_key: str = Depends(verify_access)):
    """
    Voice-friendly endpoint for iOS Shortcuts and other speech-only clients.

    Same input contract as /ask but returns the reply as plain text
    (Content-Type: text/plain), with markdown syntax stripped so it
    is speech-ready. iOS Shortcuts can pipe the body straight into
    "Speak Text" without any dictionary / magic-variable parsing.
    """
    if not payload.user_input.strip() and not payload.image_b64:
        raise HTTPException(status_code=400, detail="Input cannot be entirely empty.")

    set_origin = getattr(orchestrator, "set_turn_origin", None)
    turn_metrics.reset()
    try:
        if set_origin is not None:
            set_origin("voice")
        reply, status_mode = orchestrator.process(
            payload.user_input,
            payload.image_b64,
            language=_normalise_language(payload.language),
            conversation_history=_daily_history_msgs(
                VOICE_LABEL, _daily_history_limit(orchestrator)),
        )
        # Mirror the spoken turn into the dashboard drawer (store the raw
        # reply, not the speech-stripped one, so the UI shows full text).
        _record_voice_turn(payload.user_input, reply, status_mode,
                           usage=turn_metrics.collect())
        return strip_markdown_for_speech(reply)
    except Exception:
        # Same hygiene as /ask (M10-A8): generic detail out, traceback in.
        ref = uuid.uuid4().hex[:8]
        logging.getLogger("cortex.api").exception(
            "/ask_voice orchestrator failure ref=%s", ref)
        raise HTTPException(status_code=500, detail="internal error",
                            headers={"X-Cortex-Ref": ref})
    finally:
        turn_metrics.collect()
        if set_origin is not None:
            set_origin(None)


def _run_ts(raw: Optional[str]) -> str:
    """Normalise a Shortcut-supplied workout start ("2026-06-16T07:30:00",
    "2026-06-16 07:30:00 -0300", …) to "YYYY-MM-DD HH:MM:SS" for the dedup
    key. Falls back to now when absent/unparseable (still logs, just not
    idempotent for that one)."""
    from src.integrations.apple_health import run_start
    return run_start(raw) or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@app.post("/fitness/run", response_class=PlainTextResponse)
@limiter.limit(_RATE_LIMIT)
async def fitness_log_run(request: Request, payload: RunLogRequest,
                          _api_key: str = Depends(verify_access)):
    """Log/sync ONE run — the Apple Watch auto-sync sink (iOS Shortcuts
    automation POSTs here). De-duplicated by the workout's start_date so a
    re-running daily automation never double-logs. Plain-text status back."""
    from src.integrations.apple_health import parse_distance_km, parse_duration_min
    from src.integrations.fitness import get_log
    from src.core.fitness_mixin import _pace_str
    try:
        km = parse_distance_km(payload.distance_km)
        mn = parse_duration_min(payload.duration_min)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400,
                            detail="distance_km / duration_min unparseable")
    if km <= 0 or mn <= 0:
        raise HTTPException(status_code=400,
                            detail="distance_km and duration_min must be > 0")
    ts = _run_ts(payload.start_date)
    from src.integrations.apple_health import run_instant
    result = get_log().import_run_result(km, mn, ts, start_utc=run_instant(payload.start_date))
    if result == "error":
        return PlainTextResponse(
            "Não consegui gravar a corrida agora — nada foi gravado. Tenta de novo daqui a pouco.",
            status_code=503,
        )
    if result == "invalid":
        raise HTTPException(status_code=400,
                            detail="distance_km and duration_min must be > 0")
    pace = _pace_str(mn / km)
    if result == "added":
        return (f"Corrida registrada: {km:g}km em {mn:g}min ({pace}/km).")
    return "Corrida já estava registrada (sem mudança)."


@app.post("/fitness/health-export", response_class=PlainTextResponse)
@limiter.limit(_RATE_LIMIT)
async def fitness_health_export(request: Request,
                                _api_key: str = Depends(verify_access)):
    """Sink for the 'Health Auto Export' app's REST automation — it POSTs a
    JSON batch of workouts; we import the RUNS (dedup by start). The raw body
    is saved to logs/health_export_last.json so the exact schema can be
    inspected/refined after the first real send."""
    raw = await request.body()
    try:  # capture the payload for schema verification — best-effort
        (_LOG_DIR / "health_export_last.json").write_bytes(raw[:2_000_000])
    except Exception:
        pass
    try:
        payload = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    from src.integrations.apple_health import iter_export_runs
    from src.integrations.fitness import get_log
    flog = get_log()
    imported = skipped = failed = total = 0
    for run in iter_export_runs(payload):
        total += 1
        result = flog.import_run_result(run["distance_km"], run["duration_min"], run["ts"],
                                        start_utc=run.get("start_utc"))
        if result == "added":
            imported += 1
        elif result == "duplicate":
            skipped += 1
        else:
            failed += 1
    if failed:
        return PlainTextResponse(
            f"Corridas: {imported} novas, {skipped} já existiam, "
            f"{failed} não consegui gravar (de {total} no payload).",
            status_code=503,
        )
    return (f"Corridas: {imported} novas, {skipped} já existiam "
            f"(de {total} no payload).")


# Spoken by the voice listener when the stream dies before the
# orchestrator could produce a reply (e.g. an exception escaping the
# generator). Short and speakable on purpose: the technical detail rides
# in the `error` field for logs/UI, never in the spoken text (M5-A3).
_STREAM_FAILURE_REPLY = {
    "pt": "Deu um erro aqui no servidor. Tenta de novo.",
    "en": "Something went wrong on the server. Please try again.",
}


def _record_voice_turn(user: str, reply: str, status_mode: str,
                       usage: Optional[dict] = None) -> None:
    """Mirror a VOICE turn (the listener's /ask_stream, iOS Shortcuts'
    /ask_voice) into the /ui conversation buffer so the dashboard drawer
    shows — in writing — what Cortex answered out loud. The typed path
    (/ui/chat/stream) already appends its own turns; these two voice
    entrypoints didn't, so a spoken reply left no readable trace (Pedro's
    ask: "ver a resposta quando me responde por voz").

    Tagged `source="voice"` so the drawer styles it with the 🎙️ marker.
    Best-effort: a buffer failure must NEVER break the spoken turn, so we
    swallow and log. Skips empty turns (image-only with no transcript, or
    an empty reply) rather than storing a blank bubble.

    s82 — it ALSO persists to the durable chat store now. The buffer above is
    an in-memory deque that dies with the process; when the chat page moved to
    the durable store (fatia 2), the voice mirror was left pointing at the old
    drawer, so spoken replies stopped appearing in the chat. Pedro noticed
    ("antigamente ficava no chat, e agora não") and could not tell whether he
    had asked for that — he hadn't. It was drift, not a decision.

    Both surfaces are written on purpose: the drawer is the live dashboard
    view, the store is the history that survives a restart and that a later
    "what did it actually answer?" can be read from."""
    user = (user or "").strip()
    reply = (reply or "").strip()
    if not user or not reply:
        return
    try:
        conversation_buffer.append(
            user=user,
            reply=reply,
            status_mode=status_mode or "OK",
            source="voice",
        )
    except Exception:
        logging.getLogger("cortex.api").exception(
            "voice turn buffer append failed")
    _persist_voice_turn(user, reply, status_mode, usage=usage)


def _voice_conversation_title(now=None) -> str:
    """One conversation per DAY, not per utterance — a sidebar with a row for
    every "toca uma música" would be unusable."""
    from src.ui.chat_store import VOICE_LABEL, daily_title
    return daily_title(VOICE_LABEL, now)


def _persist_voice_turn(user: str, reply: str, status_mode: str,
                        usage: Optional[dict] = None) -> None:
    """Append a spoken turn to the durable chat store, in the day's voice
    conversation (created on the first utterance of the day). The per-day
    logic is shared with the terminal turns (s85) — chat_store.append_daily."""
    from src.ui.chat_store import VOICE_LABEL, append_daily
    append_daily("voice", VOICE_LABEL, user, reply, status_mode, usage=usage)


# A call from this Mac — `cx` in a terminal, a curl — versus one over the
# network (Tailscale: the iPhone Shortcut, another machine).
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _record_ask_turn(request: Request, user: str, reply: str,
                     status_mode: str, usage: Optional[dict] = None) -> None:
    """Mirror a /ask turn into the chat history (s85): `cx` typed in a
    terminal left no trace once the terminal closed. Labelled by where it
    came from, so an iPhone call over Tailscale is not called "Terminal"."""
    from src.ui.chat_store import NETWORK_LABEL, TERMINAL_LABEL, append_daily
    host = getattr(getattr(request, "client", None), "host", "") or ""
    if host in _LOCAL_HOSTS:
        append_daily("terminal", TERMINAL_LABEL, user, reply, status_mode, usage=usage)
    else:
        append_daily("network", NETWORK_LABEL, user, reply, status_mode, usage=usage)


def _sse_serialise(events):
    """Turn `process_stream` (chunk, is_final, mode, lang) tuples into SSE
    `data: {...}` lines.

    Contract (s62, C8):
      data: {"language": "pt"}                    ← prelude, before 1st chunk
      data: {"chunk": "<token text>"}             ← zero or more
      data: {"error": ..., "reply": <speakable>, "language": ...}  ← on error
      data: {"done": true, "status_mode": ..., "language": ...}    ← always last

    The language prelude exists so the voice listener can pick the TTS
    voice the server already decided (orchestrator._reply_lang) instead of
    re-deriving it from a hand-maintained PT word list (M5-A4/M10-A3). It
    is emitted before any text so the voice is known by the first sentence.
    Older clients ignore unknown fields/frames — backward compatible.

    A FINAL event can carry the whole reply text: every non-streamable
    path — reflex ("São 20:01."), Spotify ("Tocando Matuê"), tool calls,
    lease grants, confirmation prompts — yields `(reply, True, mode)`.
    The voice listener only speaks `chunk` payloads, so we MUST emit that
    text as a chunk BEFORE the done marker or the assistant goes silent
    (the bug the voice telemetry panel surfaced: reflex turns with a
    `replied` but no spoken audio). The LLM streaming path ends with an
    empty final tuple, so `if chunk` skips the redundant emit there.

    A ReplyReplacement final supersedes earlier text in the done frame's
    `reply`, without another speakable chunk unless no text was emitted.
    Already played speech cannot be retracted; clients that only speak
    chunks keep their live behaviour.

    An INTERNAL_ERROR final is the engine dying mid-stream: it must NOT
    go out as a speakable chunk mixed with real content (M10-A5). It
    becomes a structured `error` frame whose `reply` carries the short
    localised sentence the orchestrator built for the TTS."""
    lang_announced = False
    text_emitted = False
    for chunk, is_final, mode, lang in events:
        if not lang_announced and lang:
            yield f"data: {json.dumps({'language': lang}, ensure_ascii=False)}\n\n"
            lang_announced = True
        if is_final:
            if mode == "INTERNAL_ERROR":
                err = {
                    "error": "engine_stream_failure",
                    "reply": chunk,
                    "language": lang,
                }
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            elif chunk and (not isinstance(chunk, ReplyReplacement) or not text_emitted):
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            if mode is None:
                # M10-A11b: a None mode from a new path would silently
                # masquerade as OFFLINE — keep the fallback (clients rely
                # on a string) but leave a trace of the routing gap.
                logging.getLogger("cortex.api").warning(
                    "SSE done frame with mode=None — a routing path "
                    "didn't set a status_mode"
                )
            done = {"done": True, "status_mode": mode or "OFFLINE",
                    "language": lang}
            if isinstance(chunk, ReplyReplacement):
                done["reply"] = str(chunk)
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        else:
            if not chunk:
                continue
            text_emitted = True
            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"


@app.post("/ask_stream")
@limiter.limit(_RATE_LIMIT)
async def process_query_stream(request: Request, payload: CortexRequest, _api_key: str = Depends(verify_access)):
    """
    Server-Sent Events streaming endpoint. The voice listener consumes
    this to speak each sentence as the LLM produces it (perceived
    latency: 8-19s → 2-3s).

    Event format (one per line, with the leading 'data: ' prefix):
      data: {"language": "pt"}                       ← prelude (1st frame)
      data: {"chunk": "<token text>"}
      data: {"error": ..., "reply": "<speakable>", "language": "pt"}
      data: {"done": true, "status_mode": "OFFLINE", "language": "pt"}

    A done event may also carry `reply` to replace the final transcript.

    For non-streamable paths (Spotify, Studies, tool calls, lease grants,
    confirmation prompts, etc.), the full reply arrives as a single chunk
    followed by the done event — same semantics as /ask but in SSE form.
    Error frames always carry a short localised `reply` the voice client
    can speak — a bare {"error"} used to mean total silence on the voice
    surface (M5-A3/M10-A5).
    """
    if not payload.user_input.strip() and not payload.image_b64:
        raise HTTPException(status_code=400, detail="Input cannot be entirely empty.")

    parts: list = []
    captured = {"mode": "OFFLINE"}

    def event_stream():
        # Accumulate the reply as it streams so the whole turn can be
        # mirrored into the /ui drawer once it's done (voice echo). For the
        # LLM path the reply joins tokens unless a final correction replaces
        # them; for a non-streamable path it's the single final chunk. `_tap`
        # sits between process_stream and the SSE serialiser, passing tuples
        # through untouched while capturing text + the final status_mode.
        def _tap(events):
            for tup in events:
                chunk, is_final, mode, _lang = tup
                if is_final:
                    captured["mode"] = mode or "OFFLINE"
                    if mode == "INTERNAL_ERROR" or isinstance(chunk, ReplyReplacement):
                        # Record the authoritative final reply, not partial
                        # tokens or the capability denial already streamed.
                        parts.clear()
                        if chunk:
                            parts.append(chunk)
                    elif chunk:
                        parts.append(chunk)
                elif chunk:
                    parts.append(chunk)
                yield tup

        # The listener dropping the stream before the confirmation must not
        # leave an approvable action (s88, Codex lote 7): the turn gets a
        # Delivery, taken once the last frame was asked past. It reaches the
        # turn's worker, created on the FIRST pull — set and cleared within
        # that pull, in its thread (pool threads serve other requests).
        from src.core.staging_mixin import Delivery
        delivery = Delivery()
        set_delivery = getattr(orchestrator, "set_turn_delivery", None)
        set_origin = getattr(orchestrator, "set_turn_origin", None)

        def _delivered(events):
            if set_delivery is not None:
                set_delivery(delivery)
            if set_origin is not None:
                set_origin("voice")
            try:
                first = next(events, None)
            finally:
                if set_delivery is not None:
                    set_delivery(None)
                if set_origin is not None:
                    set_origin(None)
            if first is None:
                return
            yield first
            yield from events

        # Only the room listener calls this endpoint, and it answers whoever
        # says the wake word: the turn is remembered but never becomes a fact
        # about the user (s88, Codex lote 12 — invariant 3; see memory.py).
        from src.core.memory import unverified_speaker_stream

        try:
            yield from _sse_serialise(_tap(_delivered(unverified_speaker_stream(
                orchestrator.process_stream(
                    payload.user_input,
                    payload.image_b64,
                    language=_normalise_language(payload.language),
                    conversation_history=_daily_history_msgs(
                        VOICE_LABEL, _daily_history_limit(orchestrator)),
                )))))
            delivery.taken.set()
        except Exception:
            # The generator died before/while producing frames. Same lang
            # resolution the orchestrator would have used (client override
            # wins, heuristic fallback) — the orchestrator may never have
            # gotten far enough to decide it.
            # str(e) went to the client: a path or an internal detail in the
            # SSE frame (s88, Codex lote 7). A fixed code out, the traceback
            # in the server log — the M10-A8 rule of /ask and /ask_voice.
            ref = uuid.uuid4().hex[:8]
            logging.getLogger("cortex.api").exception(
                "/ask_stream orchestrator failure ref=%s", ref)
            lang = _resolve_lang(payload.user_input,
                                 _normalise_language(payload.language))
            parts.clear()
            parts.append(_STREAM_FAILURE_REPLY[lang])
            captured["mode"] = "INTERNAL_ERROR"
            err = json.dumps({
                "error": "internal error",
                "ref": ref,
                "reply": _STREAM_FAILURE_REPLY[lang],
                "language": lang,
            }, ensure_ascii=False)
            yield f"data: {err}\n\n"
            done = json.dumps({"done": True, "status_mode": "INTERNAL_ERROR",
                               "language": lang}, ensure_ascii=False)
            yield f"data: {done}\n\n"
        finally:
            if not delivery.taken.is_set():
                delivery.cancelled.set()

    def metered_stream():
        # Starlette calls next() through AnyIO's thread pool for EVERY pull;
        # there is no thread affinity. Keep one bucket for the iterator and
        # attach it only while pulling/closing, never across a yield. The
        # orchestrator's deferred worker already inherits this same dict.
        previous = turn_metrics.current()
        turn_metrics.reset()
        bucket = turn_metrics.current()
        turn_metrics.attach(previous)
        events = event_stream()
        try:
            while True:
                previous = turn_metrics.current()
                turn_metrics.attach(bucket)
                try:
                    frame = next(events)
                except StopIteration:
                    return
                finally:
                    turn_metrics.attach(previous)
                yield frame
        finally:
            previous = turn_metrics.current()
            turn_metrics.attach(bucket)
            try:
                events.close()
            finally:
                usage = turn_metrics.collect()
                turn_metrics.attach(previous)
            # Best-effort mirror — never let a buffer hiccup break the stream.
            _record_voice_turn(payload.user_input, "".join(parts),
                               captured["mode"], usage=usage)

    return StreamingResponse(metered_stream(), media_type="text/event-stream")
