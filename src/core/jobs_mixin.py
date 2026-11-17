"""
Job Hunter "surface" handler for `CortexOrchestrator` (slice 4, s76).

Same mixin pattern as fitness/notes. The contract this path pins:
  - READ-ONLY: reads the top-N already-scored jobs from ~/.cortex/jobs.db.
    No LLM, no embedding here (ranking is the CLI's job — too slow for a turn),
    no network. The reply is a deterministic template over the local ledger.
  - Voice-friendly WITHOUT hiding the link (s82): the title is emitted as a
    MARKDOWN link. The UI renders it clickable (`inlineMd`, sanitised href) and
    `strip_markdown_for_speech` — already applied at BOTH voice entry points,
    /ask_voice and the listener's per-sentence path — reduces `[title](url)` to
    just `title`, so the TTS never reads a URL aloud. One chokepoint formats for
    speech instead of every handler asking "am I being spoken?".
    ⚠️ The corollary: emit links as MARKDOWN, never as a bare URL — a bare URL
    survives the stripper and would be read out.
  - Fail-soft + honest: an empty or unranked ledger degrades to a clear "run
    the ranker" sentence, never a traceback or a guessed job.

Relies on the host providing `self._reply_lang` (set by _process_core) but, like
the fitness handlers, picks the reply language from the TRIGGER so text AND
voice answer in kind even when the heuristic would misfire.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from src.core.audit import audit
from src.integrations.ats import format_deadline


def _row_get(row, key, default=None):
    """sqlite3.Row has no .get(); be defensive so an older row (pre-migration
    in a test) never raises into a spoken turn."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default

# How many matches to read back in a spoken/looked-up turn. Deliberately small:
# the CLI (`rank_jobs.py --top N`) is the place for a long list; a conversation
# wants the headline few.
_TOP_N = 3

# The trigger itself is a reliable language signal (EN phrasings vs PT), more so
# than the accent-blind heuristic — mirror the fitness approach.
_EN_TRIGGER_PREFIXES = (
    "any job", "any new job", "my top job", "my best job", "my new job",
    "top job", "best job", "new job", "jobs for me", "job for me", "job matches",
)


def _lang_from_trigger(raw_text: str) -> str:
    return "en" if (raw_text or "").strip().lower().startswith(_EN_TRIGGER_PREFIXES) else "pt"


def _pct(score: float) -> str:
    return f"{score * 100:.0f}%"


class JobsMixin:
    """Surface the ranked Job Hunter matches in conversation (read-only)."""

    def _handle_jobs_top(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"tem vaga nova?" / "my top jobs" → the best-scored matches from the
        local ledger. Honest when nothing's ranked yet."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.jobs import get_log
        log = get_log()

        # order="score" → best first, with unscored jobs falling to the end; we
        # only surface the ones actually ranked against the résumé.
        # list_OPEN: a closed vacancy would take one of the 3 slots and send
        # The user to a dead page (s82 — the ledger was 95/182 closed).
        rows = [r for r in log.list_open(limit=_TOP_N, order="score")
                if r["score"] is not None]

        if not rows:
            total = log.count()
            # Distinguish "nothing ranked" from "everything already closed" —
            # the second is the common case with a stale ledger, and the
            # next step is a re-fetch, not a re-rank.
            ranked = [r for r in log.list(limit=log.count() or 1, order="score")
                      if r["score"] is not None]
            if ranked:
                audit.log("–", "jobs", "top_all_closed",
                          payload={"ranked": len(ranked)})
                return (
                    t(f"Suas {len(ranked)} vagas rankeadas já encerraram. "
                      "Roda o fetch_jobs.py pra buscar vagas novas.",
                      f"All {len(ranked)} ranked jobs have closed. Run "
                      "fetch_jobs.py to pull fresh ones."),
                    "JOBS_ALL_CLOSED",
                )
            audit.log("–", "jobs", "top_empty", payload={"stored": total})
            if total == 0:
                return (
                    t("Não tenho vagas guardadas ainda. Roda o fetch_jobs.py "
                      "pra buscar e o rank_jobs.py pra rankear vs seu currículo.",
                      "No jobs stored yet. Run fetch_jobs.py to pull some and "
                      "rank_jobs.py to score them against your résumé."),
                    "JOBS_EMPTY",
                )
            return (
                t(f"Tenho {total} vagas guardadas, mas nenhuma rankeada ainda. "
                  "Roda o rank_jobs.py --resume <cv> pra eu ordenar.",
                  f"I've got {total} jobs stored but none ranked yet. Run "
                  "rank_jobs.py --resume <cv> and I'll order them."),
                "JOBS_UNRANKED",
            )

        audit.log("–", "jobs", "top_read", payload={"shown": len(rows)})
        lines = []
        for i, r in enumerate(rows, 1):
            loc = f" em {r['location']}" if r["location"] else ""
            if lang == "en":
                loc = f" in {r['location']}" if r["location"] else ""
            dist = ""
            if r["distance"] is not None:
                dist = t(f" (a {r['distance']:.0f} milhas)",
                         f" ({r['distance']:.0f} miles away)")
            # Deadline is the most actionable bit — speak it when present.
            deadline = format_deadline(_row_get(r, "closing_at"), lang)
            tail = f" — {deadline}" if deadline else ""
            # Markdown link, not a bare URL: clickable in the UI, and silently
            # dropped by the speech stripper (see the module contract). Guard
            # the url anyway — a hand-inserted row could carry an empty one,
            # and "[title]()" would render as dead text.
            url = (_row_get(r, "url") or "").strip()
            title = f"[{r['title']}]({url})" if url else r["title"]
            lines.append(f"{i}. {title}{loc}{dist} — {_pct(r['score'])}{tail}")
        header = t("Suas melhores vagas agora:", "Your best matches right now:")
        return ("\n".join([header, *lines]), "JOBS_TOP")

    # --- re-rank on demand (slice: voice re-rank) -------------------------
    # The ledger is scored against the résumé by the slow embedding pass
    # (rank_jobs). Too slow to block a spoken turn, so a voice "re-rankeia
    # minhas vagas" kicks it off in the BACKGROUND and acks immediately; the
    # fresh scores show up the next time the user asks "tem vaga nova".

    _rerank_lock = threading.Lock()   # guards the in-progress flag (one orch)

    def _handle_jobs_rerank(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"re-rankeia minhas vagas" / "rank my jobs" → re-score the ledger
        against the stored résumé in the background. Honest when there's no
        résumé / no jobs; no-ops politely if a re-rank is already running."""
        lang = "pt" if "vaga" in (raw_text or "").lower() else "en"

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.jobs import get_log
        log = get_log()

        if log.get_resume() is None:
            return (
                t("Não tenho seu currículo guardado ainda. Rode uma vez "
                  "rank_jobs.py --resume <arquivo> pra eu poder rankear.",
                  "I don't have your résumé stored yet. Run rank_jobs.py "
                  "--resume <file> once so I can rank."),
                "JOBS_RERANK_NO_RESUME",
            )
        if log.count() == 0:
            return (
                t("Não tenho vagas guardadas pra rankear ainda.",
                  "No jobs stored to rank yet."),
                "JOBS_RERANK_EMPTY",
            )

        with JobsMixin._rerank_lock:
            if getattr(self, "_rerank_running", False):
                return (
                    t("Já estou re-rankeando suas vagas — só um instante.",
                      "Already re-ranking your jobs — one moment."),
                    "JOBS_RERANK_BUSY",
                )
            self._rerank_running = True

        audit.log("–", "jobs", "rerank_start", payload={})
        self._spawn_rerank()
        return (
            t("Ok, re-rankeando suas vagas em segundo plano. Pergunte 'tem vaga "
              "nova' em instantes pra ver a ordem atualizada.",
              "On it — re-ranking your jobs in the background. Ask 'any new "
              "jobs' in a moment for the refreshed order."),
            "JOBS_RERANK",
        )

    def _spawn_rerank(self) -> None:
        """Start the background re-rank thread. A seam so tests stub it out
        instead of spinning a real Ollama-bound thread."""
        threading.Thread(target=self._rerank_worker, daemon=True).start()

    def _rerank_worker(self) -> None:
        """Re-score the whole ledger against the résumé (embeddings — slow).
        Fail-soft, and ALWAYS clears the in-progress flag so a failed run never
        wedges the feature."""
        try:
            from src.integrations.jobs import get_log
            from src.integrations.job_ranker import rank_jobs
            n = rank_jobs(get_log())
            audit.log("–", "jobs", "rerank_done", payload={"scored": n})
        except Exception:
            audit.log("–", "jobs", "rerank_failed", payload={})
        finally:
            with JobsMixin._rerank_lock:
                self._rerank_running = False
