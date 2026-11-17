"""FastAPI router for the Cortex UI.

Page entry points and snapshot:
  GET  /ui/login                Sign-in form
  POST /ui/login                key in the BODY -> cookie, redirect to /ui/novo
  GET  /, /ui, /ui/chat          303 to /ui/novo or /ui/login, according to auth
  GET  /ui/novo                  HTML UI (server-rendered)
  GET  /ui/data                  JSON snapshot for the Central

Auth model — dual:
  - X-Cortex-Key header (for curl / scripts), OR
  - cortex_ui cookie set by /ui/login (for browsers)

The header carries the API key; the HttpOnly cookie carries a revocable
session token whose digest is keyed with that key. SameSite=Lax protects
against cross-site POSTs. max_age = 30 days; sign in again to renew.

IP allowlist still applies: IPAllowlistMiddleware runs in front of
every route on the app, including these — so a leaked cookie used
from outside the Tailnet is dropped before this router is reached.

Rate limit: not applied here. Slowapi's per-IP throttle on /ask*
already covers brute-force on the API key; /ui is just a different
surface to the same secret. If this surface ever becomes a vector
(e.g. exposed to the open internet) the right fix is the IP allowlist,
not adding another throttle.
"""

import hmac
import json
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Literal, Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from src.core.audit import audit
from src.core.stream_reply import ReplyReplacement
from src.core.prompt_safety import note_false_read
from src.core.i18n import _resolve_lang, t
from src.core import effect_store, focus, projects, turn_metrics
from src.core.redaction import redact_payload, redact_text
from src.core.save_code import folder_link_key, read_saved_file
from src.tools.local_reader import LocalFileReader
from src.ui.aggregator import build_snapshot
from src.ui import session_store
from src.ui.chat_store import store as chat_store
from src.ui.conversation import buffer as conversation_buffer
from src.voice import stt

# The cookie carries a random session token; logout revokes it server-side.
_COOKIE_NAME = "cortex_ui"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30

# SSE streaming (/ui/chat/stream): how often to emit a keepalive comment
# while the worker runs without emitting a status (keeps proxies/clients from
# idling out, and bounds the q.get wait so a wedged turn can't block the
# stream thread forever), and how long to wait on the worker at teardown.
_SSE_KEEPALIVE_S = float(os.getenv("CORTEX_SSE_KEEPALIVE_S", "10"))
_SSE_JOIN_TIMEOUT_S = float(os.getenv("CORTEX_SSE_JOIN_TIMEOUT_S", "120"))
_STOP_OUTCOME_WAIT_S = float(os.getenv("CORTEX_STOP_OUTCOME_WAIT_S", "60"))

# Templates and static files live alongside this module.
_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

router = APIRouter()

# Mounted as a sub-app in src/api.py. Sub-app prefix wouldn't help
# the dependency loop, so we mount static files on the parent app
# instead — see src/api.py.

logger = logging.getLogger("cortex.ui")


def _expected_key() -> str:
    """Re-read at call time so tests can monkeypatch the env var.
    Production reads it once at module import in src/api.py; this
    indirection costs nothing and avoids a circular import."""
    key = os.getenv("CORTEX_API_KEY")
    if not key:
        # Should never happen — src/api.py refuses to start without
        # the env var. Defence-in-depth.
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: CORTEX_API_KEY not set.",
        )
    return key


def _require_auth(request: Request) -> None:
    """Raise 401 without a valid browser session or script header key.
    401 (not 403) so browsers don't treat it as a hard wall — Pedro
    can hit /ui/login and try again."""
    header_key = request.headers.get("X-Cortex-Key")
    cookie = request.cookies.get(_COOKIE_NAME)
    expected = _expected_key()
    # compare_digest: constant-time, no timing oracle on the key (M10-A1).
    # Bytes, not str: compare_digest raises TypeError on non-ASCII str.
    if header_key:
        authenticated = hmac.compare_digest(
            header_key.encode("utf-8"), expected.encode("utf-8"))
        reason = "ui-bad-key"
    else:
        authenticated = session_store.store.valid(cookie, expected)
        reason = "ui-bad-session" if cookie else "ui-no-key"
    if not authenticated:
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        auth_logger = logging.getLogger("cortex.auth")
        auth_logger.info(f"FAIL ip={client_ip} path={path} reason={reason}")
        raise HTTPException(
            status_code=401,
            # Never spell the key into a URL here: this message was the
            # instruction that taught the ?key= habit whose leak into the
            # access log and browser history s82 found.
            detail=(
                "Not authenticated. Open /ui/login in a browser and sign in, "
                "or send the X-Cortex-Key header."
            ),
        )


def _login_page(message: str = "", tone: str = "") -> str:
    """The sign-in form. Deliberately dependency-free HTML — this page has to
    render when nothing else about the session works."""
    banner = ""
    if message:
        colour = "#b3261e" if tone == "error" else "#7a5900"
        banner = (f'<p style="color:{colour};margin:0 0 14px;font-size:13px;'
                  f'line-height:1.45">{message}</p>')
    return (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Cortex — entrar</title></head>"
        "<body style=\"margin:0;min-height:100vh;display:flex;align-items:center;"
        "justify-content:center;font:14px -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;background:#faf9f7;color:#1a1a1a\">"
        "<form method='post' action='/ui/login' style=\"width:min(360px,90vw);"
        "padding:28px;border:1px solid #e3e0da;border-radius:12px;"
        "background:#fff\">"
        "<h1 style='margin:0 0 6px;font-size:18px'>Cortex</h1>"
        "<p style='margin:0 0 18px;color:#6b6b6b;font-size:13px'>"
        "Entra com a tua chave de API.</p>"
        + banner +
        "<input type='password' name='key' autocomplete='current-password' "
        "autofocus required placeholder='chave' style=\"width:100%;"
        "box-sizing:border-box;padding:10px 12px;border:1px solid #d5d1c9;"
        "border-radius:8px;font-size:16px\">"
        "<button type='submit' style=\"width:100%;margin-top:12px;padding:10px;"
        "border:0;border-radius:8px;background:#1a1a1a;color:#fff;font-size:14px;"
        "cursor:pointer\">Entrar</button>"
        "</form></body></html>"
    )


@router.get("/ui/login", response_class=HTMLResponse)
def ui_login_form(request: Request, key: str = "") -> Response:
    """Render the sign-in form. **No longer authenticates from the query
    string.**

    The old flow was GET /ui/login?key=<key>, and uvicorn logs the full request
    line — so the API key ended up in cleartext in logs/uvicorn.out.log (40
    times, found in s82) and in the browser history of every machine that ever
    signed in. Both are readable by anything that can read a file or a profile,
    long after the tab is closed.

    A key arriving in the query is now IGNORED, not honoured — but the page
    says so instead of failing blankly, because Pedro has a shell alias built
    around the old URL and a silent no-op would look like the server was down
    (which is exactly how the s82 401s felt). Scripts are unaffected: they
    authenticate with the X-Cortex-Key header, which never touched this route.
    """
    if key:
        client_ip = request.client.host if request.client else "unknown"
        logging.getLogger("cortex.auth").info(
            f"FAIL ip={client_ip} path=/ui/login reason=key-in-url-refused"
        )
        return HTMLResponse(_login_page(
            "A chave na URL não é mais aceita — ela vazava para o histórico do "
            "navegador e para o log de acesso. Cola ela aqui embaixo; o cookie "
            "vale 30 dias.<br><b>Troca o teu atalho para abrir só "
            "<code>/ui/login</code>, sem <code>?key=</code>.</b>"))
    return HTMLResponse(_login_page())


@router.post("/ui/login")
async def ui_login(request: Request) -> Response:
    """Verify the key from the POST body and issue the cookie.

    Verifies BEFORE issuing so an attacker can't plant an arbitrary cookie
    value. The body is parsed by hand rather than via fastapi.Form because
    Form pulls in python-multipart, and a urlencoded body is three lines of
    the standard library — this project keeps its dependency list short on
    purpose.
    """
    client_ip = request.client.host if request.client else "unknown"
    auth_logger = logging.getLogger("cortex.auth")

    # CSRF: a page on another origin can POST a form to this endpoint even
    # though it can't read the response, which is enough to log the user into a
    # key of the attacker's choosing, or to grind guesses at the local API.
    # SameSite=lax protects the SESSION cookie, not this request. Browsers
    # always send Origin on a cross-site POST, so a present-and-foreign Origin
    # is a reliable reject; an absent one is a non-browser client (curl), which
    # gains nothing from CSRF anyway.
    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc != request.headers.get("host", ""):
            auth_logger.info(
                f"FAIL ip={client_ip} path=/ui/login reason=cross-origin-post")
            raise HTTPException(status_code=403, detail="Cross-origin POST refused.")

    raw = (await request.body()).decode("utf-8", "replace")
    key = (parse_qs(raw).get("key") or [""])[0]

    expected = _expected_key()
    # compare_digest: constant-time, no timing oracle on the key (M10-A1).
    # Bytes, not str: compare_digest raises TypeError on non-ASCII str.
    if not key or not hmac.compare_digest(
        key.encode("utf-8"), expected.encode("utf-8")
    ):
        auth_logger.info(
            f"FAIL ip={client_ip} path=/ui/login reason=login-bad-key")
        # The form again, not a bare 401 body: a wrong paste is the likeliest
        # reason to be here, and a dead-end page means retyping the URL.
        return HTMLResponse(_login_page("Chave inválida.", tone="error"),
                            status_code=401)

    token = session_store.store.issue(key)
    if token is None:
        return HTMLResponse(_login_page(
            "Não consegui abrir a sessão — tenta de novo.", tone="error"), status_code=503)

    # Since 21/10, /ui/novo is the everyday surface (the user's decision).
    resp = RedirectResponse(url="/ui/novo", status_code=303)
    resp.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        # `lax` (was: `strict`, sessão 21 fix): Safari mobile dropped
        # the cookie when `strict` was set, because following the
        # `/ui/login` → `/ui` redirect without a Referer header made
        # the navigation look "origin-less" to Safari's Strict
        # interpretation. Lax allows top-level GET navigations
        # (exactly this flow) while still blocking cross-site POST —
        # which is what CSRF protection actually wants for a
        # session cookie. It also survives the 303 this POST returns.
        samesite="lax",
        # Trusted proxy headers expose HTTPS through Tailscale Serve.
        secure=request.url.scheme == "https",
        path="/",
    )
    auth_logger.info(
        f"OK   ip={client_ip} path=/ui/login event=cookie-issued")
    return resp


@router.get("/ui/logout")
def ui_logout(request: Request) -> Response:
    """Revoke and clear the cookie. Unauthenticated — clearing your own cookie
    doesn't need proof. Redirects to a tiny acknowledgement page."""
    cookie = request.cookies.get(_COOKIE_NAME)
    if cookie:
        session_store.store.revoke(cookie, _expected_key())
    resp = HTMLResponse(
        "<html><body><p>Logged out. "
        "<a href='/ui/login'>Sign in again</a>.</p></body></html>"
    )
    resp.delete_cookie(_COOKIE_NAME, path="/", secure=request.url.scheme == "https")
    return resp


@router.get("/")
@router.get("/ui")
@router.get("/ui/chat")
def ui_entry(request: Request) -> Response:
    """Keep bare and old addresses usable with the same auth check and log."""
    try:
        _require_auth(request)
    except HTTPException as exc:
        if exc.status_code != 401:
            raise
        return RedirectResponse(url="/ui/login", status_code=303)
    return RedirectResponse(url="/ui/novo", status_code=303)


def _local_now() -> datetime:
    """Use the Mac's local clock, independently patchable from other UI reads."""
    return datetime.now()


@router.get("/ui/novo", response_class=HTMLResponse)
def ui_novo_page(request: Request) -> Response:
    _require_auth(request)
    hour = _local_now().hour
    greeting = "Bom dia" if 5 <= hour < 12 else "Boa tarde" if 12 <= hour < 18 else "Boa noite"
    name = os.getenv("CORTEX_USER_NAME", "Pedro")
    if name:
        greeting += f", {name}"
    return templates.TemplateResponse(
        request=request, name="novo.html", context={"greeting": greeting}
    )


@router.get("/ui/data")
def ui_data(request: Request) -> Response:
    _require_auth(request)
    return JSONResponse(build_snapshot())


_CODE_TYPES = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
               '.json': 'application/json', '.md': 'text/plain', '.txt': 'text/plain'}
_CODE_HEADERS = {
    'Content-Security-Policy': "sandbox allow-scripts allow-forms; default-src 'self' 'unsafe-inline' data:; connect-src 'none'",
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'Cache-Control': 'no-store',
}


@router.get('/ui/codigo/{key}/{folder}/{file}')
def ui_codigo(key: str, folder: str, file: str) -> Response:
    try:
        # Opaque-origin sandbox subresources have no login cookie. The link
        # grants only this folder, including its relative CSS/JS requests.
        if not hmac.compare_digest(key.encode('utf-8'),
                                   folder_link_key(folder).encode('ascii')):
            raise ValueError('invalid saved-code link')
        content = read_saved_file(folder, file)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail='Not Found') from None
    return Response(content, media_type=_CODE_TYPES[Path(file).suffix.lower()],
                    headers=_CODE_HEADERS)


def _read_record(record, fields):
    if record is None:
        return None
    values = {name: getattr(record, name) for name in fields.split()}
    return {k: v.astimezone(timezone.utc).isoformat() if isinstance(v, datetime) else v
            for k, v in values.items()}


def _project_view(store, project, active):
    return dict(_read_record(project, "id name objective created_at updated_at"),
                active=active is not None and active.id == project.id,
                conversations=store.conversations(project.id))


def _progress_view(entries):
    # Same rule as _project_where, without arming its focus offer or auditing.
    return {
        "last_done": next((e for e in entries if e["kind"] == "feito"), None),
        "todos": [e for e in entries if e["kind"] == "falta" and e["closed_at"] is None],
        "last_decision": next((e for e in entries if e["kind"] == "decisao"), None),
        "next_step": next((e for e in entries if e["kind"] == "proximo" and e["closed_at"] is None), None),
    }


def _project_entries(store, project):
    entries = store.where_entries(project.id)
    if entries is None:
        raise ValueError("Project entries unavailable")
    return [_read_record(e, "id kind text channel conversation_id created_at closed_at") for e in entries]


def _focus_view():
    empty = {"available": False, "current": None, "last": None}
    store = focus.store
    try:
        if not store.available():
            return empty
        now = datetime.now(timezone.utc)

        def session_view(session):
            if session is None:
                return None
            return dict(_read_record(session, "id task first_step started_at ends_at closed_at next_step"),
                        open=session.is_open, time_is_up=session.time_is_up(now),
                        reminder="failed" if session.reminder == focus.REMINDER_FAILED else
                        "unknown" if session.reminder == focus.REMINDER_UNKNOWN else
                        "none" if session.reminder is None else "created")

        return {"available": True, "current": session_view(store.current()), "last": session_view(store.last())}
    except Exception:
        return empty


@router.get("/ui/projects")
def ui_projects(request: Request) -> Response:
    _require_auth(request)
    empty = {"available": False, "active_id": None, "projects": []}
    store = projects.store
    try:
        if not store.available():
            return JSONResponse(empty)
        active = store.active()
        return JSONResponse({"available": True, "active_id": active.id if active else None,
                             "projects": [_project_view(store, p, active) for p in store.list()]})
    except Exception:
        return JSONResponse(empty)


@router.get("/ui/projects/{project_id}")
def ui_project(request: Request, project_id: int) -> Response:
    _require_auth(request)
    empty = {"available": False, "project": None, "progress": _progress_view([]),
             "entries": [], "observations": []}
    store = projects.store
    try:
        if not store.available():
            return JSONResponse(empty)
        project = store.get(project_id)
        if project is not None:
            entries = _project_entries(store, project)
            return JSONResponse({"available": True, "project": _project_view(store, project, store.active()),
                                 "progress": _progress_view(entries), "entries": entries,
                                 "observations": [_read_record(o, "id kind tool reference channel conversation_id created_at")
                                                  for o in store.observations(project.id, limite=20)]})
    except Exception:
        return JSONResponse(empty)
    raise HTTPException(status_code=404, detail="Project not found")


def _project_to_delete(project_id):
    if not projects.store.available():
        raise RuntimeError("Project store unavailable")
    project = projects.store.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _project_delete_plan(request, project_id):
    """Finish every read before erasing anything, using the live engine Palace."""
    linked = projects.store.conversations(project_id, strict=True)
    if linked is None:
        raise RuntimeError("Project links unavailable")
    conversations, turns = [], []
    for cid in linked:
        if chat_store.get_conversation(cid, strict=True) is not None:
            conversations.append(cid)
            turns.extend(chat_store.get_turns(cid, strict=True))
    memory = request.app.state.orchestrator.engine_analytical.memory
    return conversations, memory, memory.conversation_memory_ids(turns)


@router.get("/ui/projects/{project_id}/delete-preview")
def ui_project_delete_preview(request: Request, project_id: int) -> Response:
    _require_auth(request)
    try:
        project = _project_to_delete(project_id)
        conversations, _, memories = _project_delete_plan(request, project_id)
        return JSONResponse({"name": project.name, "conversations": len(conversations),
                             "memories": len(memories)})
    except HTTPException:
        raise
    except Exception:
        logger.exception("ui: project delete preview failed")
        return JSONResponse({"ok": False}, status_code=503)


@router.post("/ui/projects/{project_id}/delete")
def ui_project_delete(request: Request, project_id: int) -> Response:
    _require_auth(request)
    result = {"ok": False, "conversations": 0, "memories": 0}
    try:
        _project_to_delete(project_id)
        conversations, memory, memories = _project_delete_plan(request, project_id)
        for mid in memories:
            if not memory._palace.delete_document(mid):
                raise RuntimeError("Memory deletion failed")
            result["memories"] += 1
        for cid in conversations:
            if not chat_store.delete_conversation(cid):
                raise RuntimeError("Conversation deletion failed")
            result["conversations"] += 1
        if projects.store.delete_project(project_id) is None:
            raise RuntimeError("Project deletion failed")
        result["ok"] = True
    except HTTPException:
        raise
    except Exception:
        logger.exception("ui: project deletion failed")
    return JSONResponse(result)


@router.get("/ui/resume")
def ui_resume(request: Request, conversation_id: Optional[str] = None) -> Response:
    _require_auth(request)
    result = {"available": False, "project": None, "via": None,
              "progress": _progress_view([]), "focus": _focus_view()}
    store = projects.store
    try:
        if not store.available():
            return JSONResponse(result)
        project = store.progress_project(conversation_id)
        if project is False:
            return JSONResponse(result)
        if project is not None:
            entries = _project_entries(store, project)
            view = _project_view(store, project, store.active())
            linked = store.project_for_conversation(conversation_id)
            result.update(project=view, via="conversation" if linked else "active", progress=_progress_view(entries))
        result["available"] = True
    except Exception:
        pass  # Store failures must remain distinct from a successful empty read.
    return JSONResponse(result)


@router.get("/xr", response_class=HTMLResponse)
def xr_scene(request: Request) -> Response:
    """WebXR Fase A — cena A-Frame para o passthrough do Quest 3.

    Mesma cookie-auth do resto do /ui (URLs de WebXR também vazam em
    history/logs → query-string nunca autentica aqui; o cookie de
    /ui/login é a credencial). A cena é só LEITURA: faz fetch de
    /ui/conversation + /ui/data com `credentials: same-origin`, então
    o cookie viaja sozinho. SOBERANIA: A-Frame é vendorado local
    (/ui/static/vendor), zero CDN; não captura áudio (voz segue pela
    pipeline nativa). immersive-ar exige secure context → servido por
    HTTPS via `tailscale serve`.
    """
    _require_auth(request)
    return templates.TemplateResponse(request=request, name="xr.html", context={})


# ---------- Chat / conversation ----------

class AttachmentPayload(BaseModel):
    name: str = Field(..., max_length=200)
    text: str = Field(..., max_length=12000)


class ChatPayload(BaseModel):
    """Body for POST /ui/chat/stream. Same shape as /ask minus images and
    with a default language (browser users typing PT-BR is the common
    case; the orchestrator's heuristic still runs on the text when
    language=None).

    conversation_id (fatia 2): when present, the turn is persisted to that
    conversation in the durable chat_store; when absent, it lands in the
    ephemeral ConversationBuffer (the pre-fatia-2 behaviour, kept so the
    drawer and any un-migrated client keep working)."""
    text: str = Field(..., min_length=1, max_length=4000)
    language: Optional[str] = None
    conversation_id: Optional[str] = Field(None, max_length=64)
    pending_id: Optional[str] = Field(None, max_length=64)
    # Attachment (s81): text already extracted by POST /ui/upload from a file
    # The user attached. Fenced as UNTRUSTED content before it reaches the model
    # (the indirect-injection work) — a document can carry instructions aimed at
    # the LLM. attachment_name is shown in the bubble; not trusted for anything.
    attachment: Optional[str] = Field(None, max_length=12000)
    attachment_name: Optional[str] = Field(None, max_length=200)
    attachments: Optional[List[AttachmentPayload]] = Field(None, max_length=2)

    @model_validator(mode="after")
    def one_attachment_form(self):
        if self.attachments is not None and (
                self.attachment is not None or self.attachment_name is not None):
            raise ValueError("Use attachments or the legacy attachment fields, not both.")
        return self


class PushPayload(BaseModel):
    """Body for POST /ui/conversation/push — used by the voice
    daemon (when it returns post-hardware) to surface its turns in
    the dashboard drawer."""
    user: str = Field(..., min_length=1, max_length=4000)
    reply: str = Field(..., min_length=1, max_length=20000)
    status_mode: str = "OK"
    source: str = "voice"  # voice daemon is the only foreseen caller


def _get_orchestrator(request: Request):
    """Resolve the orchestrator from app.state. Returning a clear
    error here (rather than letting the AttributeError surface as a
    500) helps when running the ui router on a minimal app without
    the full pipeline — exactly the situation our tests are in by
    default. Tests opt in to chat coverage by setting
    `app.state.orchestrator = MockOrchestrator()`."""
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not available on this app (no app.state.orchestrator).",
        )
    return orch


# Stage → human label (emoji + PT/EN, M10-A11). Presentation lives HERE,
# in the UI layer — the orchestrator only emits semantic stage names. New
# stages degrade gracefully to a generic "processando…"/"processing…".
# `lang` defaults to "pt" (the pre-i18n behaviour) so any caller that
# doesn't resolve a language keeps the old labels.
def _status_label(stage: str, detail: str = "", lang: str = "pt") -> str:
    d = (detail or "").strip()
    if stage == "web_search":
        if d:
            return t(lang, f"🔍 buscando: {d}", f"🔍 searching: {d}")
        return t(lang, "🔍 buscando na web…", "🔍 searching the web…")
    if stage == "reading":
        if d:
            return t(lang, f"📖 lendo {d}…", f"📖 reading {d}…")
        return t(lang, "📖 lendo…", "📖 reading…")
    if stage == "writing":
        if d:
            return t(lang, f"✍️ escrevendo {d}…", f"✍️ writing {d}…")
        return t(lang, "✍️ escrevendo…", "✍️ writing…")
    if stage == "remembering":
        return t(lang, "🧠 lembrando…", "🧠 remembering…")
    if stage == "thinking":
        return t(lang, "🧠 pensando…", "🧠 thinking…")
    return t(lang, "🧠 processando…", "🧠 processing…")


def _sse(obj: dict) -> str:
    """One Server-Sent Event frame. ensure_ascii=False keeps PT-BR + emoji
    intact over the wire."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _conversation_history_msgs(conversation_id: str, limit: int) -> list:
    """Rebuild a conversation's short-term thread from the durable chat_store,
    as engine-message pairs, for the orchestrator to bind as this turn's
    history (UI 2c). Reopening an old conversation then gets ITS last turns as
    immediate context instead of the global (voice/CLI) thread — while the
    Palace long-term memory stays global and always present.

    The ui layer owns this read so the core orchestrator never depends on the
    chat_store. Only turns with both a user and a reply are included (the 8B is
    sensitive to role alternation — a dangling half-turn would break it).
    `limit` is the caller's half-message budget, not a raw-row tail length.
    Read all turns to count complete exchanges at their absolute positions;
    this costs O(conversation size) I/O and memory, without a schema change.
    Fail-soft: any error → [] (an empty, isolated conversation history)."""
    from src.core.history_window import HistoryWindow
    from src.core.staging_mixin import CODE_TESTS_OUTPUT
    from src.core.hallucination import (
        CONFIRMATION_HISTORY_MODES, CONFIRMATION_HISTORY_STAND_IN,
    )

    try:
        window = HistoryWindow(limit * 2 if isinstance(limit, int)
                               and not isinstance(limit, bool) else None)
        turns = chat_store.get_turns(conversation_id)
        msgs: list = []
        for turn in turns:
            user = (turn.get("user") or "").strip()
            reply = (turn.get("reply") or "").strip()
            if (turn.get("status_mode") or "").startswith("CODE_TESTS_"):
                # Keep the saved/screen reply whole; project output must not reach the 8B.
                for marker in CODE_TESTS_OUTPUT.values():
                    reply = reply.split(marker, 1)[0]
            if not user or not reply:
                continue
            if turn.get("status_mode") in CONFIRMATION_HISTORY_MODES:
                # Code owns the preview, not the model. Keep storage/display
                # intact, without teaching the next turn to imitate that UI.
                reply = CONFIRMATION_HISTORY_STAND_IN
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": reply})
        return msgs[2 * window.start(len(msgs) // 2):]
    except Exception:
        logger.exception("ui: could not load conversation history for %s",
                         conversation_id)
        return []


def _daily_history_limit(orchestrator) -> int:
    """Use half the engine's positive integer history budget, or five turns."""
    try:
        max_history = getattr(getattr(orchestrator, "engine_analytical", None),
                              "max_history", 10)
        if (not isinstance(max_history, int) or isinstance(max_history, bool)
                or max_history <= 0):
            max_history = 10
        return max(1, max_history // 2)
    except Exception:
        return 5


def _daily_history_msgs(label: str, limit: int) -> list:
    """Read this channel's thread for today; even an empty list isolates it."""
    from src.ui.chat_store import daily_title

    try:
        cid = chat_store.find_titled(daily_title(label))
        return _conversation_history_msgs(cid, limit) if cid else []
    except Exception:
        logger.exception("ui: could not load daily history for %s", label)
        return []


_UPLOAD_MAX_BYTES = 8 * 1024 * 1024          # 8 MB raw file
_UPLOAD_MAX_CHARS = 12000                     # extracted text cap (context-safe)
_UPLOAD_EXTS = (".txt", ".md", ".pdf", ".docx")


@router.post("/ui/voice/transcribe")
async def ui_voice_transcribe(request: Request) -> Response:
    """Transcribe browser audio locally; retain neither the file nor the text."""
    _require_auth(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not content_type.startswith("audio/"):
        return JSONResponse({"error": "unsupported_media_type"}, status_code=415)
    import tempfile
    tmp = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(delete=False) as audio_file:
            tmp = audio_file.name  # Set before writing so partial writes are removed.
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 10 * 1024 * 1024:
                    return JSONResponse({"error": "too_large"}, status_code=413)
                audio_file.write(chunk)
        # Loading/inference can be slow; keep the API event loop responsive.
        text, lang = await run_in_threadpool(stt.transcribe_file, tmp)
        audit.log("–", "ui", "voice_transcribed", payload={
            "duration_s": time.perf_counter() - started, "lang": lang,
        })
        return JSONResponse({"text": text, "lang": lang})
    except stt.TooLong:
        return JSONResponse({"error": "too_long"}, status_code=413)
    except stt.BadAudio:
        return JSONResponse({"error": "bad_audio"}, status_code=400)
    except stt.STTUnavailable:
        return JSONResponse({"error": "stt_unavailable"}, status_code=503)
    finally:
        if tmp:
            os.remove(tmp)


class UploadPayload(BaseModel):
    """Body for POST /ui/upload — the file as base64 in JSON, so no
    python-multipart dependency (the project stays dependency-lean; the rest of
    the UI is JSON too). data_b64 may carry a data: URL prefix — stripped."""
    name: str = Field(..., min_length=1, max_length=200)
    data_b64: str = Field(..., max_length=12 * 1024 * 1024)   # ~8MB × 4/3 + slack


@router.post("/ui/upload")
def ui_upload(payload: UploadPayload, request: Request) -> Response:
    """Extract text from a file Pedro attached in the chat, so he can ask about
    it. The file itself is NEVER stored — it is read once, into text, and
    discarded; only the extracted text goes back to the browser, which sends it
    with the next message. The text is fenced as untrusted before it reaches the
    model (done on the stream route). .txt/.md/.pdf/.docx only, size-capped."""
    _require_auth(request)
    name = os.path.basename(payload.name or "arquivo")
    if not name.lower().endswith(_UPLOAD_EXTS):
        raise HTTPException(status_code=400,
                            detail="Formato não suportado. Use .txt, .md, .pdf ou .docx.")
    import base64
    b64 = payload.data_b64.split(",", 1)[-1]     # drop a data:...;base64, prefix
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo inválido.")
    if len(raw) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo grande demais (máx 8 MB).")

    import tempfile
    suffix = os.path.splitext(name)[1]
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            # Named BEFORE the write: a write failing half-way (disk full) left
            # part of the user's document in the temp dir (s88, Codex lote 7).
            tmp = f.name
            f.write(raw)
        orch = getattr(request.app.state, "orchestrator", None)
        reader = getattr(orch, "reader", None) or LocalFileReader(base_dir="data/documents")
        text = reader.read_path(tmp)
    except Exception:
        logger.exception("ui_upload: extraction failed")
        raise HTTPException(status_code=422, detail="Não consegui ler esse arquivo.")
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass

    if text.startswith("[SYSTEM ERROR]"):
        raise HTTPException(status_code=422, detail="Não consegui ler esse arquivo.")
    text = text.strip()
    truncated = len(text) > _UPLOAD_MAX_CHARS
    if truncated:
        text = text[:_UPLOAD_MAX_CHARS]
    return JSONResponse({"name": name, "text": text,
                         "chars": len(text), "truncated": truncated})


def _compose_attachment_prompt(question: str, attachment: str | List[AttachmentPayload],
                               name: str = "arquivo", lang: str = "pt", *,
                               with_how: bool = False) -> str | tuple[str, str]:
    """Share the measured window across files, then fence them as one block.
    Keep the string-only interface for existing single-document callers;
    the stream also needs HOW it was read, so it can say when it read part."""
    from src.core.doc_retrieval import select_relevant
    from src.core.prompt_safety import CLOSE, OPEN, MAX_DOC_CHARS, fence, guard_line
    items = ([(name, attachment)] if isinstance(attachment, str) else
             [(a.name, a.text) for a in attachment])
    names, sections = [], []
    for name, text in items:
        if not text.strip():
            continue
        # Names outside the fence are plain, single-line labels only.
        name = " ".join(name.replace(OPEN, "").replace(CLOSE, "").split()) or "arquivo"
        names.append(name)
        sections.append(f"=== {name} ===\n{text.strip()}")
    content, how = select_relevant("\n\n".join(sections), question, budget=MAX_DOC_CHARS)
    # Retrieval's visible gap separators count against the window too.
    content = content[:MAX_DOC_CHARS]
    prompt = (
        f"{guard_line(lang)}\n\n"
        f"ARQUIVOS ANEXADOS ({', '.join(names)}):\n{fence(content)}\n\n"
        f"PERGUNTA: {question}"
    )
    return (prompt, how) if with_how else prompt


_active_turns: dict = {}
_active_turns_lock = threading.Lock()


class CancelPayload(BaseModel):
    id: str


@router.post("/ui/chat/cancel")
def ui_chat_cancel(payload: CancelPayload, request: Request) -> Response:
    _require_auth(request)
    with _active_turns_lock:
        delivery = _active_turns.get(payload.id)
        if delivery is not None:
            delivery.cancelled.set()
    return JSONResponse({"cancelled": delivery is not None})


def _queue_preview(preview: str) -> str:
    # Chat replies don't decide queue items; instructions live in tool_risk.py builders and planner.render_preview.
    head, newline, last = preview.rpartition("\n")
    for instruction in ("Responda 'sim'", "Reply 'yes'"):
        start = last.find(instruction)
        if start != -1:
            return (head + newline + last[:start]).rstrip()
    return preview


@router.get("/ui/gate")
def ui_gate(request: Request, limit: int = 50, lang: str = "pt") -> Response:
    _require_auth(request)
    from src.core.gate_queue import gate_queue

    # Pending approvals must remain visible beyond the recent page.
    history = gate_queue.history(None)
    if history is None:
        return JSONResponse({"available": False, "pending": [], "recent": []})
    lang = lang if lang in ("pt", "en") else "pt"
    orch = getattr(request.app.state, "orchestrator", None)
    labels = getattr(orch, "_UNDO_ACTION_LABELS_EN" if lang == "en" else "_UNDO_ACTION_LABELS", {})
    store = effect_store.store

    def stamp(value):
        try:
            value = datetime.fromisoformat(value)
            return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            return None

    def item(entry):
        queued = entry["queued"]
        action = queued.get("type", "")
        action = action if isinstance(action, str) else ""
        args = redact_payload(queued.get("args", {}))
        preview = None
        if orch is not None:
            try:
                preview = orch._human_confirmation_prompt({"type": action, "args": args}, lang=lang)
                preview = _queue_preview(preview)
            except Exception:
                pass  # A damaged historical action must not hide the queue.
        effect = store.get(queued["id"])
        return redact_payload({
            "id": queued["id"], "type": action, "label": labels.get(action, action),
            "status": entry["status"], "effect_state": effect.state if effect else None,
            # Probe only: a dead owner's effect must remain untouched by polling.
            "effect_owner_alive": store.owner_alive(effect.owner)
                if effect and effect.state in ("preparando", "executando") else None,
            "origin": redact_payload(queued.get("origin")),
            "reason": redact_text(queued.get("reason")), "trace_id": queued.get("trace_id"),
            "queued_at": stamp(queued.get("ts")), "args": args, "preview": redact_text(preview),
            "events": [{"event": e["event"], "at": stamp(e.get("ts")),
                        "detail": redact_text(e.get("detail"))} for e in entry["events"]],
        })

    recent = [item(entry) for entry in history[:max(1, min(200, limit))]]
    pending = sorted((item(e) for e in history if e["status"] == "queued"),
                     key=lambda e: e["queued_at"] or "")
    return JSONResponse({"available": True, "pending": pending, "recent": recent})


class GateDecisionPayload(BaseModel):
    id: str = Field(..., strict=True, min_length=1, max_length=64)
    decision: Literal["approve", "reject"]
    lang: Literal["pt", "en"] = "pt"


@router.post("/ui/gate/decide")
def ui_gate_decide(payload: GateDecisionPayload, request: Request) -> Response:
    _require_auth(request)
    from src import gate_review
    from src.core.gate_queue import gate_queue

    result = "not_recorded"
    try:
        runner = None
        if payload.decision == "approve":
            # Resolve before deciding: unavailable execution must leave the item pending.
            orch = _get_orchestrator(request)
            def runner(kind, args):
                setters = [getattr(orch, name, None) for name in (
                    "set_turn_origin", "set_turn_delivery", "set_turn_pending_id")]
                try:
                    for setter in setters:
                        if setter:
                            setter(None)
                    # Pin only this pooled thread; the language setter changes the fallback.
                    orch._reply_lang_state()["lang"] = payload.lang
                    return orch._run_action(kind, args, effect_id=payload.id)
                finally:
                    orch._clear_reply_lang()
                    for setter in setters:
                        if setter:
                            setter(None)
        outcome = gate_review.decide(
            payload.id, "approved" if payload.decision == "approve" else "rejected",
            queue=gate_queue, runner=runner)
        result = outcome["result"]
        code = {"not_found": 404, "already_decided": 409, "not_recorded": 503}.get(result, 200)
        body = {"decided": code == 200, "result": result, "status": outcome["status"]}
        if code == 200:
            body["reply"] = redact_text(outcome["reply"])
        return JSONResponse(body, status_code=code)
    finally:
        audit.log("–", "gate", "ui_decision", payload={
            "id": payload.id, "decision": payload.decision, "result": result})


@router.get("/ui/effects")
def ui_effects(request: Request, limit: int = 50, lang: str = "pt",
               conversation_id: Optional[str] = None) -> Response:
    _require_auth(request)
    if conversation_id is not None and len(conversation_id) > 64:
        raise HTTPException(status_code=422, detail="conversation_id exceeds 64 characters")
    store = effect_store.store
    if not store.available():
        body = {"available": False, "effects": [], "notices": []}
        if conversation_id is not None:
            body["conversation"] = []
        return JSONResponse(body)
    limit = max(1, min(200, limit))
    orch = getattr(request.app.state, "orchestrator", None)
    labels = getattr(orch, "_UNDO_ACTION_LABELS_EN" if lang == "en" else "_UNDO_ACTION_LABELS", {})

    def stamp(value):
        return value.astimezone(timezone.utc).isoformat() if value is not None else None

    def turn_link(origin):
        if (isinstance(origin, dict)
                and isinstance(origin.get("conversation"), str) and origin["conversation"]
                and isinstance(origin.get("turn"), str) and origin["turn"]):
            return {"conversation_id": origin["conversation"], "turn_id": origin["turn"]}
        return None

    def item(effect):
        history = store.history(effect.id)
        origin = effect.origin if isinstance(effect.origin, dict) else {}
        return {
            "id": effect.id, "action": effect.action,
            "label": labels.get(effect.action, effect.action), "state": effect.state,
            "origin": redact_payload(effect.origin), "args": redact_payload(effect.args),
            "turn": turn_link(origin) if origin.get("channel") == "ui" else None,
            "asked_turn": turn_link(origin.get("asked")),
            "detail": redact_text(effect.detail), "created_at": stamp(effect.created_at),
            "updated_at": stamp(effect.updated_at), "seen_at": stamp(effect.seen_at),
            "restart": any(h.to_state == "desconhecido" and h.detail == effect_store.RESTART_DETAIL
                           for h in history),
            # Probe only: polling must never reconcile a dead owner's effect or lock.
            "owner_alive": store.owner_alive(effect.owner)
                if effect.state in ("preparando", "executando") else None,
            "chat_notice_pending": effect.notice_pending == 1,
            "history": [{"from": h.from_state, "to": h.to_state,
                         "detail": redact_text(h.detail), "at": stamp(h.created_at)} for h in history],
        }

    body = {"available": True, "effects": [item(e) for e in store.recent(limit)],
            "notices": [item(e) for e in store.unseen_unknown(limit)]}
    if conversation_id is not None:
        body["conversation"] = [item(e) for e in store.unknown_for_conversation(conversation_id, limit)]
    return JSONResponse(body)


class EffectsSeenPayload(BaseModel):
    ids: List[Annotated[str, Field(strict=True, max_length=64)]] = Field(
        ..., min_length=1, max_length=50)


@router.post("/ui/effects/seen")
def ui_effects_seen(payload: EffectsSeenPayload, request: Request) -> Response:
    _require_auth(request)
    marked = effect_store.store.mark_seen(payload.ids)
    audit.log("–", "effects", "seen", payload={"ids": marked})
    return JSONResponse({"seen": marked})


@router.get("/ui/pending")
def ui_pending(request: Request, conversation_id: Optional[str] = None,
               lang: Optional[str] = None) -> Response:
    _require_auth(request)
    orch = _get_orchestrator(request)
    # Raw disk read: opening a preview must never clean up or approve it.
    action = orch._read_pending_raw()
    if action is None:
        return JSONResponse({"pending": None})
    orch.set_turn_origin("ui", conversation_id)
    try:
        here = orch._pending_same_origin(action, allow_voice=True)
    finally:
        orch.set_turn_origin(None)
    if lang is None and action.get("type") == "execute_plan":
        lang = (action.get("args") or {}).get("lang")
    lang = lang if lang in ("pt", "en") else "pt"
    expires_at = orch._pending_expires_at(action)
    return JSONResponse({"pending": {
        "id": action.get("id"), "type": action.get("type"),
        "preview": orch._human_confirmation_prompt(action, lang=lang),
        "origin": action.get("origin"), "staged_at": action.get("staged_at"),
        "expires_at": expires_at,
        "expired": expires_at is None or orch._pending_action_expired(action), "here": here,
    }})


class PendingTransferPayload(BaseModel):
    pending_id: str = Field(..., strict=True, max_length=64)
    conversation_id: str = Field(..., strict=True, max_length=64)


@router.post("/ui/pending/transfer")
def ui_pending_transfer(payload: PendingTransferPayload, request: Request) -> Response:
    _require_auth(request)
    if chat_store.get_conversation(payload.conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    reason, _record = _get_orchestrator(request).transfer_pending_to_ui(
        payload.pending_id, payload.conversation_id)
    # Use the same read-only preview contract, including a concurrent settlement.
    view = json.loads(ui_pending(request, conversation_id=payload.conversation_id).body)
    return JSONResponse({"transferred": reason == "transferred", "reason": reason, **view})


@router.post("/ui/chat/stream")
def ui_chat_stream(payload: ChatPayload, request: Request) -> Response:
    """THE chat route — feeds both the dashboard drawer and the dedicated
    /ui/chat page (fatia 1). Emits SSE frames:

      {"type":"status","stage":..,"label":..}  activity feed (🔍/🧠)
      {"type":"token","text":..}               reply text, token by token
      {"type":"final","turn":{..}}             the persisted turn (reconcile)
      {"type":"error","detail":..}             localised failure sentence

    The orchestrator is sync, so we drive process_stream() in a worker
    thread and drain a queue it feeds. For the pure-chat path process_stream
    yields the reply token by token → a stream of "token" frames (the bubble
    types out like Claude); for tool/web/reflex paths it fires the 🔍/🧠
    status feed via the emitter and lands the whole reply as one final chunk.
    The emitter is installed THREAD-LOCALLY inside that worker
    (set_status_emitter), so overlapping turns — e.g. a voice turn running at
    the same time — never cross-talk or clobber each other's feed.

    Older clients (the drawer in dashboard.js) ignore unknown "token" frames
    and still render status → the full reply from the "final" turn, so this
    stays backward compatible.
    """
    _require_auth(request)
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    orch = _get_orchestrator(request)
    language = payload.language
    # Status labels stream BEFORE the orchestrator resolves the reply
    # language, so resolve it here with the same heuristic + override
    # rule the orchestrator applies (M10-A11): the labels and the reply
    # come out in the same language.
    label_lang = _resolve_lang(text, language)
    asked_ts = datetime.now(timezone.utc).isoformat()

    # Attachment (s81): the model sees the fenced document + the question; the
    # CONVERSATION stores only the question plus a 📎 marker (the fenced blob
    # would bloat the sidebar and re-feed on every future turn). So `model_text`
    # (to the LLM) and `stored_user` (to chat_store) diverge here.
    stored_user = text
    model_text = text
    has_attachment = False
    attachment_how = "whole"
    attachments = payload.attachments or []
    if payload.attachment is not None:
        attachments = [AttachmentPayload(name=payload.attachment_name or "arquivo",
                                         text=payload.attachment)]
    attachments = [a for a in attachments if a.text.strip()]
    if attachments:
        model_text, attachment_how = _compose_attachment_prompt(
            text, attachments, lang=label_lang, with_how=True)
        names = ", ".join(a.name.strip() or "arquivo" for a in attachments)
        stored_user = f"{text}{ATTACHMENT_MARK}{names}"
        has_attachment = True

    # UI 2c: when this turn belongs to a saved conversation, load that
    # conversation's recent turns so the orchestrator binds them as the
    # short-term thread (see _conversation_history_msgs). Loaded once here (not
    # per token) and closed over by the worker below. Absent id → None → the
    # orchestrator keeps the global history (voice/CLI behaviour).
    cid = payload.conversation_id
    history_msgs = None
    if cid:
        _max_hist = getattr(getattr(orch, "engine_analytical", None),
                            "max_history", 10)
        history_msgs = _conversation_history_msgs(cid, max(1, _max_hist // 2))

    result: dict = {}
    persist_lock = threading.Lock()
    worker = None

    def persist_turn(delivery):
        # Stop's worker and the live generator share one receipt.
        with persist_lock:
            if "turn" not in result:
                fields = dict(
                    user=stored_user, reply=result.get("reply", ""),
                    status_mode=result.get("mode", "OK"), source="ui",
                    asked_ts=asked_ts, usage=result.get("usage"),
                    latency_ms=result.get("latency_ms"))
                turn = chat_store.append_turn(
                    conversation_id=cid, turn_id=delivery.id, **fields) if cid else None
                if turn is None:
                    turn = conversation_buffer.append(**fields)
                result["turn"] = turn
            return result["turn"]

    def event_stream(delivery):
        nonlocal worker
        q: "queue.Queue" = queue.Queue()
        sentinel = object()
        # Taken once the browser has the final frame; an action this turn
        # stages and the browser never takes (Stop) is withdrawn (s88).
        set_delivery = getattr(orch, "set_turn_delivery", None)
        set_origin = getattr(orch, "set_turn_origin", None)
        set_pending_id = getattr(orch, "set_turn_pending_id", None)

        streamed = {"tokens": False, "parts": []}

        def emitter(stage: str, detail: str = "") -> None:
            q.put(("status", stage, detail))

        def token_emitter(piece: str) -> None:
            # A slow LLM-synthesis step (ONLINE_RAG web search) pushes its
            # tokens here as they're produced, so the reply types out instead
            # of landing all at once after ~20s. Marks the turn as already
            # streamed so the final tuple isn't re-emitted as a duplicate.
            if piece and not delivery.cancelled.is_set():
                streamed["tokens"] = True
                streamed["parts"].append(piece)
                q.put(("token", piece))

        def run() -> None:
            # Install the emitters on THIS worker thread only; clear them in
            # finally. Thread-local → no other thread's turn is touched.
            # Same thread-local story for the token bucket (idea #26):
            # the whole turn — every engine call — runs right here.
            #
            # Token-a-token (fatia 1 da UI de chat): we drive process_stream,
            # not process(). Pure-chat YIELDS the reply token by token; the
            # ONLINE_RAG web-search path streams its synthesis via the token
            # EMITTER (a side channel drained in the main thread, like the
            # status feed); tool/reflex paths land the whole reply as one final
            # chunk. All flow through the same queue in THIS thread, so their
            # order on the wire is exactly their order here.
            orch.set_status_emitter(emitter)
            orch.set_token_emitter(token_emitter)
            if set_delivery is not None:
                set_delivery(delivery)
            if set_origin is not None:
                set_origin("ui", cid)
            if set_pending_id is not None:
                set_pending_id(payload.pending_id)
            turn_metrics.reset()
            t0 = time.perf_counter()
            parts: list = []
            stream = None
            try:
                if delivery.cancelled.is_set():
                    audit.log("–", "turn", "cancelled", payload={
                        "at": "ui_chat_stream.run", "tool": None})
                    return
                # Voice should inherit the project the user last worked in here.
                if cid:
                    try:
                        project = projects.store.project_for_conversation(cid)
                        active = projects.store.active()
                        if project and (not active or active.id != project.id):
                            if projects.store.set_active(project.id):
                                audit.log("–", "project", "ui_active")
                    except Exception:
                        pass  # Storage must not interrupt an ordinary UI turn.
                stream = orch.process_stream(
                    model_text, None, language=language,
                    conversation_history=history_msgs,
                )
                for chunk, is_final, mode, _lang in stream:
                    if delivery.cancelled.is_set():
                        audit.log("–", "turn", "cancelled", payload={
                            "at": "ui_chat_stream.run", "tool": None})
                        break
                    if is_final and mode == "INTERNAL_ERROR":
                        # Engine degraded mid-turn: process_stream already
                        # built a short localised sentence (never str(exc) —
                        # M10-A5/A8). Surface THAT as the error, not partial
                        # tokens, and don't fold it into the reply buffer.
                        result["error"] = chunk or "internal error"
                        break
                    if is_final and isinstance(chunk, ReplyReplacement):
                        # saved() renders turn.reply on the existing final
                        # event; live tokens need no browser-side protocol.
                        parts[:] = [str(chunk)]
                    elif chunk:
                        # If the emitter already streamed this turn's tokens
                        # (web-search synthesis), the final tuple carries the
                        # SAME full reply — keep it for the buffer but don't
                        # re-emit it as a duplicate token frame.
                        if not streamed["tokens"]:
                            q.put(("token", chunk))
                        parts.append(chunk)
                    if is_final:
                        result["mode"] = mode or "OK"
            except Exception:  # surfaced as an SSE error frame
                # M10-A8: same hygiene as the non-stream route — the
                # frame reaches the browser, so no str(exc) in it.
                result["ref"] = uuid.uuid4().hex[:8]
                logger.exception("ui_chat_stream orchestrator failure ref=%s", result["ref"])
                result["error"] = "internal error"
            finally:
                # Close plain chat now; only an already-running action may
                # finish in process_stream's worker after this consumer stops.
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
                orch.set_status_emitter(None)
                orch.set_token_emitter(None)
                if set_delivery is not None:
                    set_delivery(None)
                if set_origin is not None:
                    set_origin(None)
                if set_pending_id is not None:
                    set_pending_id(None)
                if delivery.cancelled.is_set():
                    if delivery._outcome_worker:
                        delivery.finished.wait(_STOP_OUTCOME_WAIT_S)
                    reply = "".join(streamed["parts"] if streamed["tokens"] else parts)
                    outcome = delivery.outcome if delivery.finished.is_set() else None
                    if outcome and outcome[0] and outcome[0] != reply:
                        reply = "\n\n".join(p for p in (reply, outcome[0]) if p)
                    result.pop("error", None)
                    result["reply"] = reply or ("— stopped —" if label_lang == "en" else "— parado —")
                    result["mode"] = outcome[1] if outcome else "AGENT_ACTION_CANCELLED"
                elif "error" not in result:
                    result["reply"] = "".join(parts)
                if "error" not in result:
                    if has_attachment:
                        note_false_read(result["reply"], "ui_attachment")
                    result.setdefault("mode", "OK")
                    result["usage"] = turn_metrics.collect()
                    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                    if delivery.cancelled.is_set():
                        persist_turn(delivery)
                q.put(sentinel)

        # Baseline activity within ~0ms so the bubble is never silent,
        # even for a plain-chat turn that emits no specific stage.
        yield _sse({"type": "status", "stage": "thinking",
                    "label": _status_label("thinking", lang=label_lang)})
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        got_sentinel = False
        try:
            while True:
                try:
                    item = q.get(timeout=_SSE_KEEPALIVE_S)
                except queue.Empty:
                    # Worker busy with no status for a while: keep the
                    # connection warm with a comment frame (clients ignore
                    # ': ...' lines), or stop if the worker died WITHOUT
                    # enqueuing its sentinel (so a wedged turn can't block
                    # the stream thread forever).
                    if not worker.is_alive():
                        break
                    yield ": keepalive\n\n"
                    continue
                if item is sentinel:
                    got_sentinel = True
                    break
                if item[0] == "token":
                    yield _sse({"type": "token", "text": item[1]})
                    continue
                _, stage, detail = item
                yield _sse({"type": "status", "stage": stage,
                            "label": _status_label(stage, detail,
                                                   lang=label_lang)})
        finally:
            if not got_sentinel:
                delivery.cancelled.set()
            worker.join(timeout=_SSE_JOIN_TIMEOUT_S)

        if "error" in result:
            frame = {"type": "error", "detail": result["error"]}
            if "ref" in result:
                frame["ref"] = result["ref"]
            yield _sse(frame)
            return
        if not got_sentinel or "reply" not in result:
            # run()'s finally always enqueues the sentinel, so this only
            # trips if the worker was killed mid-flight — never emit a blank
            # turn into the buffer.
            yield _sse({"type": "error",
                        "detail": "stream ended unexpectedly"})
            return
        if not delivery.cancelled.is_set() and attachment_how != "whole":
            from src.core.orchestrator import _partial_read_note
            note = _partial_read_note(attachment_how, label_lang)
            result["reply"] += note
            yield _sse({"type": "token", "text": note})
        frame = {"type": "final", "turn": persist_turn(delivery)}
        read_pending = getattr(orch, "_read_pending_raw", None)
        pending = read_pending() if read_pending is not None else None
        if pending is not None and pending.get("by") == delivery.id:
            frame["pending"] = {"id": pending["id"],
                                "expires_at": orch._pending_expires_at(pending)}
        yield _sse(frame)
        # Resumed: the server sent the final frame and asked for more — the
        # browser was still there to take the preview.
        delivery.taken.set()

    def registered_stream():
        from src.core.staging_mixin import Delivery
        delivery = Delivery()
        with _active_turns_lock:
            _active_turns[delivery.id] = delivery
        try:
            yield _sse({"type": "turn", "id": delivery.id})
            yield from event_stream(delivery)
        finally:
            if not delivery.taken.is_set():
                delivery.cancelled.set()
            # The transport can close before run() starts or after it exits.
            if worker is None:
                result["reply"] = "— stopped —" if label_lang == "en" else "— parado —"
                result["mode"] = "AGENT_ACTION_CANCELLED"
            if (worker is None or not worker.is_alive()) and "reply" in result:
                persist_turn(delivery)
            with _active_turns_lock:
                _active_turns.pop(delivery.id, None)

    return StreamingResponse(
        registered_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable proxy/nginx response buffering so the feed streams
            # live even behind a reverse proxy (no-op direct via Tailscale).
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/ui/conversation")
def ui_conversation(request: Request, limit: int = 50) -> Response:
    """Return the recent conversation buffer for the drawer to
    backfill when it opens. limit is clamped to [1, 200]."""
    _require_auth(request)
    limit = max(1, min(int(limit), 200))
    return JSONResponse({"turns": conversation_buffer.snapshot(limit=limit)})


@router.post("/ui/conversation/clear")
def ui_conversation_clear(request: Request) -> Response:
    """Wipe the drawer's in-memory conversation buffer on demand.

    Safe by design: this clears only the EPHEMERAL UI mirror. Anything
    worth keeping was already persisted per-turn — every substantive reply
    flows through the engine's store_interaction (Palace conversation rows
    + passive fact extraction), so long-term memory is untouched. Returns
    how many turns were dropped, for an honest confirmation message.
    """
    _require_auth(request)
    dropped = conversation_buffer.clear()  # atomic count + wipe
    return JSONResponse({"ok": True, "dropped": dropped})


@router.post("/ui/conversation/push")
def ui_conversation_push(payload: PushPayload, request: Request) -> Response:
    """Append a turn produced outside this process — primarily the
    voice listener daemon, which runs as its own service and POSTs
    here once it has STT'd the user and received the orchestrator
    reply. By going through the API the dashboard stays in sync
    regardless of which entrypoint produced the turn.

    Auth: same X-Cortex-Key / cookie as the rest of /ui. The voice
    daemon already has the key for /ask_voice — no new credential."""
    _require_auth(request)
    turn = conversation_buffer.append(
        user=payload.user,
        reply=payload.reply,
        status_mode=payload.status_mode or "OK",
        source=payload.source or "voice",
    )
    return JSONResponse(turn)


# ---------- Conversations (fatia 2 — durable chat history) ----------
# CRUD over the SQLite chat_store so the chat page gets a Claude-style
# sidebar of past conversations that survives uvicorn restarts. Same cookie
# auth as the rest of /ui. This is UI HISTORY, not memory — the Palace stays
# the single source of long-term memory (see chat_store.py docstring).

class ConversationCreatePayload(BaseModel):
    """Optional title on create; if omitted, the first user turn names the
    conversation (chat_store.append_turn)."""
    title: Optional[str] = Field(None, max_length=200)


class ConversationRenamePayload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# What a stored turn carries instead of its document (the fenced blob is not
# kept). A turn with it can't be regenerated faithfully — see pop_last.
ATTACHMENT_MARK = "\n\n📎 "


class PopLastPayload(BaseModel):
    """The turn the page shows as its last — pop it only if it still is."""
    turn_id: Optional[str] = Field(None, max_length=64)


class ConversationBulkDeletePayload(BaseModel):
    """Ids to delete in one go (s82: selection mode in the sidebar).

    Capped at 200: the sidebar lists at most 100, so a larger request is not
    something the UI can produce — refusing it keeps a stray script from
    turning one call into an unbounded delete loop."""
    ids: List[str] = Field(..., min_length=1, max_length=200)


@router.get("/ui/conversations")
def ui_conversations_list(request: Request, limit: int = 100) -> Response:
    """All conversations, newest-updated first (id, title, updated_ts, turn
    count, preview) for the sidebar."""
    _require_auth(request)
    return JSONResponse({"conversations": chat_store.list_conversations(limit=limit)})


@router.post("/ui/conversations")
def ui_conversations_create(payload: ConversationCreatePayload,
                            request: Request) -> Response:
    """Create a new (empty) conversation and return it."""
    _require_auth(request)
    convo = chat_store.create_conversation(title=payload.title)
    if convo is None:
        return JSONResponse({"ok": False, "error": "create failed"}, status_code=500)
    return JSONResponse(convo)


@router.get("/ui/conversations/{conversation_id}")
def ui_conversation_get(conversation_id: str, request: Request) -> Response:
    """A conversation's metadata + its turns (chat order), for the page to
    load when you open it from the sidebar. 404 if it doesn't exist."""
    _require_auth(request)
    convo = chat_store.get_conversation(conversation_id)
    if convo is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"conversation": convo,
                         "turns": chat_store.get_turns(conversation_id)})


@router.post("/ui/conversations/{conversation_id}/rename")
def ui_conversation_rename(conversation_id: str,
                           payload: ConversationRenamePayload,
                           request: Request) -> Response:
    """Rename a conversation. 404 if it doesn't exist."""
    _require_auth(request)
    ok = chat_store.rename_conversation(conversation_id, payload.title)
    if not ok:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "id": conversation_id,
                         "title": payload.title.strip()[:200]})


@router.post("/ui/conversations/{conversation_id}/delete")
def ui_conversation_delete(conversation_id: str, request: Request) -> Response:
    """Delete a conversation and all its turns. 404 if there was nothing to
    delete. This removes only the UI history — the Palace (memory) is
    untouched, mirroring /ui/conversation/clear's contract."""
    _require_auth(request)
    removed = chat_store.delete_conversation(conversation_id)
    if not removed:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "id": conversation_id})


@router.post("/ui/conversations/delete")
def ui_conversations_bulk_delete(
    payload: ConversationBulkDeletePayload, request: Request,
) -> Response:
    """Delete several conversations in one request (selection mode).

    Reports `deleted` and `missing` separately rather than 404-ing the whole
    batch: with several ids, "one of them was already gone" is not a failure
    of the request — the user's intent (make these disappear) is satisfied
    either way, and the sidebar just reloads. Duplicate ids are collapsed so a
    repeated id can't inflate the count.

    Same contract as the single delete: this removes UI history only, never
    the Palace."""
    _require_auth(request)
    seen, deleted, missing = set(), 0, 0
    for cid in payload.ids:
        if cid in seen:
            continue
        seen.add(cid)
        if chat_store.delete_conversation(cid):
            deleted += 1
        else:
            missing += 1
    audit.log("–", "ui", "conversations_bulk_delete",
              payload={"requested": len(seen), "deleted": deleted})
    return JSONResponse({"ok": True, "deleted": deleted, "missing": missing})


@router.post("/ui/conversations/{conversation_id}/pop_last")
def ui_conversation_pop_last(conversation_id: str, request: Request,
                             payload: Optional[PopLastPayload] = None) -> Response:
    """Remove the conversation's most recent turn and return it (fatia 3:
    regenerate). The chat page calls this before re-asking the same question so
    the new answer REPLACES the rejected one rather than stacking a duplicate —
    and so the short-term history bound for the retry (UI 2c) doesn't include
    the answer being discarded. 404 when there is no turn to pop.

    UI history only: the Palace already recorded the original interaction and
    is append-only — regenerating doesn't retract it."""
    _require_auth(request)
    from src.ui.chat_store import NotTheLastTurn
    expected = (payload.turn_id if payload is not None else None) or None
    # The document of an attached turn is not kept: re-asked, the question
    # would go without it, and the answer is gone already (s88, Codex lote 7).
    last = chat_store.get_turns(conversation_id, limit=1)
    if last and ATTACHMENT_MARK in (last[-1].get("user") or ""):
        return JSONResponse({"ok": False, "error": "attachment"}, status_code=409)
    try:
        turn = chat_store.pop_last_turn(conversation_id, expected_id=expected)
    except NotTheLastTurn:
        # A newer turn arrived since the page loaded: delete nothing, let the
        # page reload the conversation (s88, Codex lote 7).
        return JSONResponse({"ok": False, "error": "not last"}, status_code=409)
    if turn is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "turn": turn})


@router.get("/ui/costs")
def ui_costs(request: Request, since: str = "", limit: int = 20) -> Response:
    """Paid-API spend ledger. `since` is an optional ISO timestamp lower
    bound (e.g. '2026-05-01' for month-to-date); `limit` caps the recent
    rows returned alongside the aggregate summary. Cortex is local-first,
    so the only metered provider today is Tavily search."""
    _require_auth(request)
    # Late import on purpose — same singleton-rebinding rationale as
    # aggregator.cost_snapshot (M10-A11a, considered and kept).
    from src.core.cost_tracker import cost_tracker
    since_arg = since.strip() or None
    return JSONResponse({
        "summary": cost_tracker.summary(since=since_arg),
        "recent": cost_tracker.recent(limit=limit),
    })


# ---------- Contacts (hamburger menu → Contatos panel) ----------
# CRUD over the local SQLite address book so adding a contact stops being
# a voice-only chore ("adiciona contato nome: X email: Y"). Same cookie
# auth as the rest of /ui; the book is local-only, so nothing leaves the
# machine (SOBERANIA r1). The orchestrator's voice/chat path uses the same
# `contact_book` singleton, so edits here are visible to "manda email pra X"
# immediately.

class ContactPayload(BaseModel):
    """Body for POST /ui/contacts. Only `name` is required — email/phone/
    notes are optional, mirroring ContactBook.add (a contact can exist with
    just a name and be filled in later)."""
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field("", max_length=320)
    phone: Optional[str] = Field(None, max_length=60)
    notes: Optional[str] = Field(None, max_length=2000)


class ContactDeletePayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


def _contact_to_dict(c) -> dict:
    return {
        "name": c.name, "email": c.email,
        "phone": c.phone or "", "notes": c.notes or "",
    }


@router.get("/ui/contacts")
def ui_contacts_list(request: Request) -> Response:
    """All saved contacts (name-sorted by the book), for the panel table."""
    _require_auth(request)
    from src.integrations.contacts import contact_book
    return JSONResponse({
        "contacts": [_contact_to_dict(c) for c in contact_book.list_all()],
    })


@router.post("/ui/contacts")
def ui_contacts_add(payload: ContactPayload, request: Request) -> Response:
    """Add a contact, or UPDATE it in place if the name already exists —
    the panel has one 'save' form, so a repeated name is an edit, not an
    error (mirrors the voice handler's add-or-update in contacts_mixin)."""
    _require_auth(request)
    from src.integrations.contacts import contact_book
    name = payload.name.strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
    if contact_book.lookup(name) is not None:
        ok = contact_book.update(
            name, email=payload.email.strip(),
            phone=payload.phone, notes=payload.notes,
        )
        action = "updated"
    else:
        ok = contact_book.add(
            name, payload.email.strip(),
            phone=payload.phone, notes=payload.notes,
        )
        action = "added"
    if not ok:
        return JSONResponse({"ok": False, "error": "save failed"}, status_code=500)
    return JSONResponse({"ok": True, "action": action, "name": name})


@router.post("/ui/contacts/delete")
def ui_contacts_delete(payload: ContactDeletePayload, request: Request) -> Response:
    """Remove a contact by exact name. Returns ok=False (404) when there
    was nothing to delete, so the UI can tell 'gone' from 'never existed'."""
    _require_auth(request)
    from src.integrations.contacts import contact_book
    removed = contact_book.delete(payload.name.strip())
    if not removed:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "name": payload.name.strip()})


class _RevalidatingStatic(StaticFiles):
    """StaticFiles that adds `Cache-Control: no-cache` to every asset.

    Vanilla StaticFiles sends ETag + Last-Modified but NO Cache-Control, so
    browsers apply *heuristic* caching and can serve stale UI assets for a
    while WITHOUT revalidating — exactly the "I updated the UI but the old
    behaviour is still there" trap. `no-cache` means "always
    revalidate before use": with the ETag already present, that's a cheap
    conditional GET → 304 when unchanged, a full 200 the moment a file changes.
    Assets are tiny and same-Tailnet, so the per-load revalidation is free."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def mount_static(app) -> None:
    """Mount /ui/static -> src/ui/static on the parent FastAPI app.
    Called once from src/api.py after the router is included."""
    static_dir = _HERE / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/ui/static", _RevalidatingStatic(directory=str(static_dir)),
              name="ui-static")
