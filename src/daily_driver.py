"""
Daily Driver — spoken morning briefing (Onda 4c).

At 08:00 (via launchd) Cortex assembles a DETERMINISTIC briefing — today's
agenda, unread-email triage, yesterday's Studier output, and pending approvals
— writes it to data/briefings/, speaks it, and posts a one-line macOS
notification. Template + data, no LLM synthesis → zero hallucination (the whole
point of leaning on the M1's strength, not its weakness).

The spoken text is plain pt-BR prose (no emoji/bullets — `say` would stumble on
them). Audio output only: no microphone, so no mic-permission grant needed,
which is why this works fine from an unattended launchd job.

Run:  venv/bin/python -m src.daily_driver --stdout    (print, don't speak — test)
      venv/bin/python -m src.daily_driver --speak --notify   (what launchd runs)
      make daily                                          (print only)

`assemble`/`headline` are pure (synthetic sources); `gather`/`speak`/`main` are
fail-soft I/O, mirroring src/weekly_review.py + src/core/digest.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.digest import _parse_ts  # shared tz-aware ISO parser
from src.integrations.ats import format_deadline  # job deadline phrasing (pure)

_BRIEFINGS_DIR = "data/briefings"

# 💼 Job Hunter surface (slice 5): how many top matches to read out, and the
# score floor below which a match isn't worth surfacing. Scores compress in the
# 0.78–0.83 band (the ranker's known limitation), so 0.75 floors out the clearly
# weak ones without demanding an unreachable absolute.
_JOBS_TOP_N = 3
_JOBS_MIN_SCORE = 0.75


def _daily_refresh_enabled() -> bool:
    """Whether the briefing auto-fetches fresh jobs (slice 5 v2). On by default;
    CORTEX_JOBS_DAILY_REFRESH=0 disables (e.g. offline, or to save the FAA quota)."""
    return (os.getenv("CORTEX_JOBS_DAILY_REFRESH", "1").strip().lower()
            not in ("0", "false", "off", "no"))


@dataclass
class DailySources:
    events_today: List[Dict[str, Any]] = field(default_factory=list)
    unread: List[Dict[str, Any]] = field(default_factory=list)
    studies_yesterday: List[str] = field(default_factory=list)
    pending: List[Dict[str, Any]] = field(default_factory=list)
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    # Sources that could not be read this morning. An empty list from a failed
    # read is not "nothing there": with the Google token broken the briefing
    # said "Você não tem eventos na agenda de hoje… Nada urgente" (s88, Codex
    # lote 10 — invariant 7).
    failed: List[str] = field(default_factory=list)


def gather(now: datetime) -> DailySources:
    """Fetch each briefing source, fail-soft (a broken source → empty)."""
    out = DailySources()
    today = now.astimezone().date()

    # Proactive external reads are AUDITED (M7-A4, review Fable 5) —
    # this job runs under launchd with nobody in the loop; the breadcrumb
    # makes its Google API traffic visible. Read-only: audit, not gate.
    try:
        from src.core.audit import audit
        audit.log("–", "background_read", "daily_gather", payload={
            "sources": ["calendar.list_events", "gmail.list_recent", "jobs.list"],
        })
    except Exception:  # noqa: BLE001 — audit must never break the briefing
        pass

    # 📅 Today's events.
    try:
        from src.integrations.calendar_client import list_events
        for ev in list_events(max_results=15):
            ts = _parse_ts(ev.get("start"))
            if ts is not None and ts.astimezone().date() == today:
                out.events_today.append(ev)
    except Exception:  # noqa: BLE001
        out.failed.append("calendar")

    # 📥 Unread email triage.
    try:
        from src.integrations.gmail_client import list_recent
        out.unread = list_recent(count=10, query="is:unread") or []
    except Exception:  # noqa: BLE001
        out.failed.append("email")

    # 🧠 Studier briefings produced in the last ~24h. Same definition of
    # "new study" as the digest (M7-A6) — only the window differs (fixed
    # 24h here vs the moving last_seen marker there).
    try:
        from datetime import timedelta

        from src.core.digest import new_studies_since
        out.studies_yesterday = new_studies_since(now - timedelta(hours=24))
    except Exception:  # noqa: BLE001
        out.failed.append("studies")

    # ✅ Pending approvals (background gate queue).
    try:
        from src.core.gate_queue import gate_queue
        out.pending = gate_queue.pending() or []
    except Exception:  # noqa: BLE001
        out.failed.append("pending")

    # 💼 New top job matches (Job Hunter slice 5). Read-only surface of what the
    # ranker already scored against the résumé; the briefing is the proactive
    # push. Only status="new" (so a match surfaces once, then main() marks it
    # seen) and only ranked matches above the score floor. Fail-soft.
    try:
        from src.integrations.jobs import get_log
        # list_OPEN, same rule as the conversation handler (s82): never push a
        # vacancy that has already closed. Both surfaces share ats.is_closed so
        # they cannot drift apart about what "closed" means.
        rows = get_log().list_open(limit=_JOBS_TOP_N, status="new",
                                   order="score")
        out.jobs = [
            {"id": r["id"], "title": r["title"], "location": r["location"],
             "score": r["score"], "closing_at": r["closing_at"]}
            for r in rows
            if r["score"] is not None and r["score"] >= _JOBS_MIN_SCORE
        ]
    except Exception:  # noqa: BLE001
        out.failed.append("jobs")

    return out


def _greeting(now: datetime) -> str:
    h = now.astimezone().hour
    if h < 12:
        return "Bom dia"
    if h < 19:
        return "Boa tarde"
    return "Boa noite"


def assemble(sources: DailySources, now: datetime) -> str:
    """Render the briefing as plain spoken pt-BR prose. Pure."""
    parts = [f"{_greeting(now)}, Pedro."]

    n_ev = len(sources.events_today)
    if "calendar" in sources.failed:
        parts.append("Não consegui consultar a sua agenda hoje — confira por lá.")
    elif n_ev == 0:
        parts.append("Você não tem eventos na agenda de hoje.")
    else:
        titles = []
        for ev in sources.events_today[:4]:
            summary = (ev.get("summary") or "evento sem título").strip()
            ts = _parse_ts(ev.get("start"))
            when = ts.astimezone().strftime("%H:%M") if ts else ""
            titles.append(f"{summary}{(' às ' + when) if when else ''}")
        plural = "evento" if n_ev == 1 else "eventos"
        parts.append(f"Hoje você tem {n_ev} {plural}: " + "; ".join(titles) + ".")

    n_un = len(sources.unread)
    if "email" in sources.failed:
        parts.append("Também não consegui ver os seus e-mails."
                     if "calendar" in sources.failed
                     else "Não consegui ver os seus e-mails.")
    elif n_un:
        sender = (sources.unread[0].get("from") or "").split("<")[0].strip()
        tail = f", o mais recente de {sender}" if sender else ""
        plural = "e-mail não lido" if n_un == 1 else "e-mails não lidos"
        parts.append(f"Você tem {n_un} {plural}{tail}.")

    if sources.studies_yesterday:
        n = len(sources.studies_yesterday)
        plural = "estudo novo" if n == 1 else "estudos novos"
        parts.append(f"O Studier preparou {n} {plural} desde ontem.")

    if sources.pending:
        n = len(sources.pending)
        plural = "ação aguardando" if n == 1 else "ações aguardando"
        parts.append(f"E há {n} {plural} sua aprovação. Diga 'make gate' pra revisar.")

    if sources.jobs:
        n = len(sources.jobs)
        titles = []
        for j in sources.jobs[:_JOBS_TOP_N]:
            # Speak the city only — the postcode tail ("GU14 7LP") reads as noise.
            city = (j.get("location") or "").split(",")[0].strip()
            # Deadline is the actionable bit: "fecha em 4 dias" prompts applying.
            deadline = format_deadline(j.get("closing_at"))
            where = f" em {city}" if city else ""
            when = f", {deadline}" if deadline else ""
            titles.append(f"{j['title']}{where}{when}")
        plural = "vaga nova alinhada" if n == 1 else "vagas novas alinhadas"
        parts.append(f"Encontrei {n} {plural} ao seu perfil: " + "; ".join(titles)
                     + ". Diga 'tem vaga nova' pra ver os detalhes.")

    if "jobs_refresh" in sources.failed:
        parts.append("Não consegui buscar vagas novas hoje.")

    if (n_ev == 0 and not n_un and not sources.studies_yesterday
            and not sources.pending and not sources.jobs
            and not sources.failed):
        parts.append("Nada urgente por aqui. Bom dia tranquilo.")

    return " ".join(parts)


def headline(sources: DailySources) -> str:
    """One-line summary for the macOS notification banner."""
    bits = []
    if sources.events_today:
        bits.append(f"{len(sources.events_today)} evento(s)")
    if sources.unread:
        bits.append(f"{len(sources.unread)} não lidos")
    if sources.studies_yesterday:
        bits.append(f"{len(sources.studies_yesterday)} estudo(s)")
    if sources.pending:
        bits.append(f"{len(sources.pending)} aprovação(ões)")
    if sources.jobs:
        bits.append(f"{len(sources.jobs)} vaga(s)")
    if "calendar" in sources.failed:
        bits.insert(0, "agenda indisponível")
    if "email" in sources.failed:
        bits.append("e-mail indisponível")
    if bits:
        return ", ".join(bits)
    return "fontes indisponíveis" if sources.failed else "nada urgente"


def speak(text: str, lang: str = "pt") -> bool:
    """Speak `text` via the macOS `say` builtin (local, no deps, no mic).
    Uses the resolved TTSConfig voice when available. Fail-soft."""
    cmd = ["say"]
    try:
        from src.voice.tts_config import resolve_tts_config
        voice = resolve_tts_config().say_voice_for(lang)
        if voice:
            cmd += ["-v", voice]
    except Exception:  # noqa: BLE001 — voice is optional
        pass
    try:
        result = subprocess.run([*cmd, text], timeout=180, check=False)
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — never crash the job
        return False


def _write_briefing(text: str, now: datetime) -> Optional[str]:
    try:
        os.makedirs(_BRIEFINGS_DIR, exist_ok=True)
        path = os.path.join(_BRIEFINGS_DIR, f"{now.astimezone().date()}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return path
    except OSError:
        return None


def post_to_chat(text: str, now: datetime) -> Optional[str]:
    """Drop the morning briefing into the chat page in the day's conversation, so
    opening /ui/novo shows the day at the top of the sidebar — phrased as if
    Pedro had asked "o que eu tenho pra hoje?" and Cortex answered — and he can
    just keep typing to follow up on it (the Palace context carries over).

    Reuses the same durable chat_store the UI reads, so no new storage. One
    conversation per day. Fail-soft: a chat-store hiccup never breaks the spoken
    / notified briefing that already ran. Returns the conversation id or None."""
    try:
        from src.ui.chat_store import store as chat_store

        title = "☀️ Bom dia · " + now.astimezone().strftime("%a %d/%m")
        cid = chat_store.conversation_titled_on(title, now)
        if cid is None:
            return None
        turn = chat_store.append_turn(
            conversation_id=cid,
            user="o que eu tenho pra hoje?",
            reply=text,
            status_mode="DAILY",
            source="daily",
        )
        return cid if turn is not None else None
    except Exception:  # noqa: BLE001 — chat is a bonus, never breaks the briefing
        return None


def main(argv: Optional[List[str]] = None) -> int:
    # The 08h launchd job runs `python -m src.daily_driver` with a bare env (only
    # PATH from the plist) — nothing imports the orchestrator here, so .env is
    # never loaded otherwise. Without it the FAA key (CORTEX_FAA_KEY) and the
    # job-search overrides (CORTEX_JOBS_*) are invisible and the slice-5 v2 refresh
    # silently no-ops. Load it here so the scheduled refresh actually fetches.
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Spoken morning briefing.")
    parser.add_argument("--speak", action="store_true", help="speak the briefing aloud")
    parser.add_argument("--notify", action="store_true", help="post a macOS notification headline")
    parser.add_argument("--stdout", action="store_true", help="print the briefing (default when nothing else asked)")
    parser.add_argument("--to-chat", action="store_true", help="also open a /ui/novo conversation with the briefing")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)

    # 💼 Proactive pull (slice 5 v2): fetch + rank fresh apprenticeships BEFORE
    # assembling, so the "new matches" are genuinely new — not just whatever was
    # ranked by hand. Only on a real delivery (the 08h launchd run passes
    # --speak/--notify); a --stdout preview stays read-only and offline. The
    # whole step is fail-soft inside refresh_jobs → never blocks the briefing.
    refresh_failed = False
    if (args.speak or args.notify) and _daily_refresh_enabled():
        try:
            from src.integrations.job_hunter import refresh_jobs
            refresh_failed = bool((refresh_jobs() or {}).get("error"))
        except Exception:  # noqa: BLE001 — never crash the job
            refresh_failed = True

    sources = gather(now)
    if refresh_failed:
        sources.failed.append("jobs_refresh")
    text = assemble(sources, now)
    _write_briefing(text, now)

    if args.stdout or not (args.speak or args.notify):
        print(text)
    notify_ok = chat_ok = spoke_ok = None
    if args.notify:
        try:
            from src.core.heartbeat import notify
            notify_ok = bool(notify("🌅 Bom dia, Pedro", headline(sources)))
        except Exception:  # noqa: BLE001
            notify_ok = False
    if args.to_chat:
        try:
            chat_ok = post_to_chat(text, now) is not None
        except Exception:  # noqa: BLE001
            chat_ok = False
    if args.speak:
        try:
            spoke_ok = bool(speak(text))
        except Exception:  # noqa: BLE001
            spoke_ok = False

    # Mark matches "seen" only when the briefing text reached chat or speech.
    # A notification headline or --stdout preview must not consume them.
    jobs_seen = 0
    if (chat_ok or spoke_ok) and sources.jobs:
        try:
            from src.integrations.jobs import get_log
            log = get_log()
            for j in sources.jobs:
                if log.set_status(j["id"], "seen"):
                    jobs_seen += 1
        except Exception:  # noqa: BLE001 — never crash the job
            pass
    if args.notify or args.to_chat or args.speak:
        try:
            from src.core.audit import audit
            audit.log("–", "daily", "delivered", payload={
                "notify": notify_ok, "chat": chat_ok, "speak": spoke_ok,
                "jobs": len(sources.jobs), "jobs_seen": jobs_seen,
            })
        except Exception:  # noqa: BLE001 — audit must never break the briefing
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
