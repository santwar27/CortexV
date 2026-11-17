"""
NightlyStudier — overnight knowledge acquisition pipeline.

For each pending topic in the StudyQueue:
  1. Generate 3-5 focused search query variations using the LLM.
  2. Run those queries through the web searcher (DuckDuckGo).
  3. Aggregate the resulting snippets.
  4. Ask the LLM for a structured Markdown briefing.
  5. Save to data/knowledge/pending/<slug>_<timestamp>.md.
  6. Mark the queue entry as consumed (atomic).

Why this design:
  - Pending → approved flow: the morning's manual approval is the
    integrity gate. Nothing enters the permanent RAG without the user
    reading the briefing first.
  - One file per topic: easy to read, edit, delete, version, share.
  - Markdown over JSON/SQLite: the user can `cat` it in Finder or vim.
  - All stages bounded: query gen ≤ 200 tokens, summary ≤ 1500 tokens,
    snippets capped at 10k chars before going to the LLM.
"""

import fcntl
import inspect
import logging
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import ollama
from src.core.ollama_client import KEEP_ALIVE as _KEEP_ALIVE, THINK as _THINK

from src.core.audit import audit

from src.core.engine import OllamaEngine
from src.core.study_queue import StudyQueue
from src.core.study_time import EVENT_TIME_RULES, StudyTime
from src.memory.consolidator import (
    extract_facts,
    persist_daily_summary,
    persist_facts,
    summarise_day,
)
from src.memory.preferences import (
    extract_preferences,
    persist_preferences,
    recent_conversations,
    recent_user_words,
    said_by_user,
    value_said_by_user,
)
from src.tools.web_search import WebSearcher

log = logging.getLogger(__name__)


class NightlyStudier:
    """Consumes the study queue, researches each topic, saves Markdown briefings."""

    SNIPPET_CHAR_LIMIT = 10_000  # bound the context window we feed to the LLM
    MAX_QUERIES_PER_TOPIC = 5
    SEARCH_RESULTS_PER_QUERY = 3

    # Topics containing any of these keywords get auto-disambiguated by
    # injecting "LLM" or "machine learning" into the generated queries.
    # Stops Tavily/DDG from returning Audi Q4 / fiscal Q4 / Q8 gasolineira
    # / Rick Roll for topics like "Q4 vs Q5 vs Q8".
    _ML_CONTEXT_TRIGGERS = re.compile(
        r"\b(?:LLM|transformer|neural\s*network|model|attention|"
        r"quantization|quantiza[çc][ãa]o|"
        r"fine[-\s]?tuning|RAG|embedding|tokeniz|"
        r"GGUF|GGML|safetensors|"
        r"Q[2-8](?:_K_M|_0)?|"  # Q4_K_M, Q5, Q8_0, etc.
        r"KV[-\s]?cache|GQA|MHA|MQA|"
        r"Llama|Mistral|Qwen|Gemma|Phi|DeepSeek|Claude|GPT|"
        r"attention\s+head|multi[-\s]?head|"
        r"pre[-\s]?train|inference)\b",
        re.IGNORECASE,
    )

    # Domain blocklist for known off-topic results that contaminate ML
    # briefings. Tavily/DDG return these for short ambiguous terms (Q4
    # = Audi/fiscal/gasolineira; KV = abbreviation; GQA = Glass Qualifications;
    # dQw4w9WgXcQ = Rick Roll). Snippets from these domains are dropped
    # before synthesis when the topic is ML-related.
    _ML_DOMAIN_BLOCKLIST = (
        # Rick Roll specifically — the synthesizer cited this once
        "youtube.com/watch?v=dqw4w9wgxcq",
        # Investor relations / unrelated Q4 product / fiscal
        "q4inc.com",
        "investopedia.com/fiscal-quarter",
        "q4dates.com",
        # Audi / car models
        "audiusa.com",
        "caranddriver.com",
        "audi.com",
        # Q8 gasoline stations
        "q8.it",
        "q8.com",
        # Generic "KV" abbreviation pages
        "en.wikipedia.org/wiki/kv",
        "kv.com",
        # Glass Qualifications Authority (not Grouped Query Attention)
        "gqa.co.uk",
        "gqa.com",
    )

    # General junk domains — blockchain explorers, crypto exchanges and
    # mining sites that Tavily returns for short/technical terms but are
    # NEVER a legitimate research source for the user's topics. Observed live
    # 2026-06-03: a "constrained decoding" study cited etherscan.io,
    # iqmining.com and explorer.xprnetwork.org. Unlike _ML_DOMAIN_BLOCKLIST,
    # this list is dropped for EVERY topic, not just ML ones.
    _GENERAL_DOMAIN_BLOCKLIST = (
        # blockchain explorers
        "etherscan.io",
        "bscscan.com",
        "polygonscan.com",
        "solscan.io",
        "blockchair.com",
        "blockstream.info",
        "xprnetwork.org",
        # crypto mining
        "iqmining.com",
        "nicehash.com",
        "minerstat.com",
        "f2pool.com",
    )

    # Detect when the LLM tried to expand an acronym in its query — the
    # recurring failure mode is Llama inventing "RAG = Regressive Attention
    # Generator" or similar, which then poisons the search results. Match
    # either form 'XYZ = words' / 'XYZ stands for words' / 'XYZ means words'
    # or parenthetical expansion 'XYZ (Word Word Word)'. We reject these
    # queries and fall back to the literal topic if every candidate is dirty.
    _ACRONYM_EXPANSION_RE = re.compile(
        r"\b[A-Z]{2,6}\s*(?:=|stands\s+for|means|is\s+short\s+for|short\s+for)\s+\S+",
        re.IGNORECASE,
    )
    _PARENTHETICAL_EXPANSION_RE = re.compile(
        r"\b[A-Z]{2,6}\s*\(\s*[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){1,5}\s*\)"
    )

    # Confidence-floor regex set. Llama 8B Q5 defaults to "alta" even on
    # briefings citing just 1-2 thin sources, so we rewrite the verdict
    # deterministically based on the number of unique source domains
    # actually cited in 'Fontes consultadas'.
    _FONTES_BLOCK_RE = re.compile(
        r"##\s*Fontes\s+consultadas\s*\n(.*?)(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    _CONFIANCA_LINE_RE = re.compile(
        r"(##\s*Confian[çc]a\s*\n)(alta|m[ée]dia|high|medium)(\b[^\n]*)",
        re.IGNORECASE,
    )
    _URL_IN_BLOCK_RE = re.compile(r"https?://([^/\s\]\)>]+)")

    # Thin-source banner. The confidence floor already rebaixa a verdict
    # when sourcing is thin, but a briefing citing ZERO verifiable
    # domains is essentially unsourced and shouldn't be approved into
    # knowledge by a quick skim. We prepend a loud banner so review
    # can't miss it. Marker doubles as the idempotency guard.
    _SOURCE_WARNING_MARKER = "POUCAS FONTES"
    _SOURCE_WARNING_BANNER = (
        "> [POUCAS FONTES] este briefing não cita nenhum domínio "
        "verificável em 'Fontes consultadas' — revise com ceticismo "
        "antes de aprovar para o conhecimento.\n\n"
    )

    def __init__(
        self,
        engine: Optional[OllamaEngine] = None,
        queue: Optional[StudyQueue] = None,
        web_searcher: Optional[WebSearcher] = None,
        output_dir: str = "data/knowledge/pending",
    ):
        self.engine = engine or OllamaEngine(
            model_name=os.getenv("CORTEX_MODEL_ANALYTICAL", "cortex-lite"),
            max_history_pairs=5,
        )
        self.queue = queue or StudyQueue()
        self.web_searcher = web_searcher or WebSearcher()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    PALACE_DB_PATH = "data/palace.db"
    BACKUP_DIR = "data/backups"

    def _backup_palace(self) -> Optional[str]:
        """Snapshot the Palace DB to data/backups/YYYY-MM-DD.db before any
        write happens tonight. Returns the backup path, or None if the
        Palace doesn't exist yet (first-run before migration). Errors are
        logged but never raised — backup failure must not block the run.

        Uses SQLite's hot-backup API (sqlite3.Connection.backup) rather
        than a raw filesystem copy: it respects WAL, holds proper locks
        for the duration of the transfer, and produces a consistent
        single-file snapshot even if the engine has writes in flight.
        Same-day re-runs overwrite the destination cleanly.
        """
        src_path = Path(self.PALACE_DB_PATH)
        if not src_path.exists():
            log.info("Nightly Studier: no palace.db yet — skipping backup.")
            return None

        dst_dir = Path(self.BACKUP_DIR)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dst_path = dst_dir / f"{stamp}.db"

        src_conn = None
        dst_conn = None
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Drop a stale same-day backup so the destination starts as a
            # fresh DB — avoids any leftover WAL/SHM sidecars confusing
            # the backup API.
            if dst_path.exists():
                dst_path.unlink()
            src_conn = sqlite3.connect(str(src_path))
            dst_conn = sqlite3.connect(str(dst_path))
            src_conn.backup(dst_conn)
            log.info(f"Nightly Studier: palace.db backed up → {dst_path}")
            return str(dst_path)
        except (OSError, sqlite3.Error) as e:
            log.warning(f"Nightly Studier: palace backup failed: {e}")
            return None
        finally:
            for c in (src_conn, dst_conn):
                if c is not None:
                    try:
                        c.close()
                    except sqlite3.Error:
                        pass

    PREFERENCE_LOOKBACK_HOURS = 24
    PREFERENCE_MAX_CHARS = 8000
    PREFERENCE_MIN_CONFIDENCE = 0.55

    MEMORY_LOOKBACK_HOURS = 24
    MEMORY_MAX_CHARS = 8000
    FACT_MIN_CONFIDENCE = 0.55

    # Retention window (M3-A5, policy approved s72). Env override
    # CORTEX_RETENTION_DAYS; <= 0 disables the pass entirely.
    RETENTION_DAYS = 180

    def _retention_pass(self) -> dict:
        """Archive conversation docs older than RETENTION_DAYS that were
        never injected (access_count = 0) into ~/.cortex/archive/ —
        archive always, delete never (see consolidator.
        archive_stale_documents). Non-fatal by design."""
        result = {"archived": 0, "path": None}
        try:
            raw = os.environ.get("CORTEX_RETENTION_DAYS", "")
            try:
                days = int(raw) if raw.strip() else self.RETENTION_DAYS
            except ValueError:
                days = self.RETENTION_DAYS
            if days <= 0:
                log.info("Studier retention: disabled (days=%s).", days)
                return result
            from src.memory.palace import Palace
            if not os.path.exists(self.PALACE_DB_PATH):
                return result
            palace = Palace(db_path=self.PALACE_DB_PATH)
            try:
                from src.memory.consolidator import archive_stale_documents
                result = archive_stale_documents(palace, days=days)
            finally:
                palace.close()
            return result
        except Exception:
            log.exception("Studier retention: pass failed (non-fatal)")
            return result

    def _consolidate_memory(self) -> dict:
        """Tier 3.5: summarise the day's conversation and extract
        structured facts, persisting both into the Palace.

        Returns a status dict so the audit log can see what happened
        even when nothing was persisted. Wrapped in try/except — any
        failure logs and returns the zero-state dict, never propagates.
        Must come BEFORE _consolidate_preferences in run() so the
        summary is available if the preference pipeline ever wants to
        reuse it (currently independent)."""
        result = {"summary_id": None, "facts_persisted": 0}
        try:
            from src.memory.palace import Palace
            if not os.path.exists(self.PALACE_DB_PATH):
                log.info("Studier memory: no palace.db — skipping.")
                return result
            palace = Palace(db_path=self.PALACE_DB_PATH)
            try:
                corpus = recent_conversations(
                    palace,
                    hours=self.MEMORY_LOOKBACK_HOURS,
                    max_chars=self.MEMORY_MAX_CHARS,
                )
                if not corpus:
                    log.info("Studier memory: no recent conversation — skipping.")
                    return result

                digest = summarise_day(
                    corpus,
                    llm_chat_fn=lambda **kw: ollama.chat(**kw, keep_alive=_KEEP_ALIVE, think=_THINK),
                    model=self.engine.model_name,
                )
                if not digest:
                    log.warning("Studier memory: summariser returned empty.")
                    return result

                summary_id = persist_daily_summary(palace, digest)
                result["summary_id"] = summary_id
                log.info("Studier memory: daily summary persisted (id=%s, %d chars).",
                         summary_id, len(digest))

                facts = extract_facts(
                    digest,
                    llm_chat_fn=lambda **kw: ollama.chat(**kw, keep_alive=_KEEP_ALIVE, think=_THINK),
                    model=self.engine.model_name,
                )
                # Only what the user said: the digest also tells Cortex's side
                # (s88, Codex lote 6 — see preferences.recent_user_words).
                own = recent_user_words(palace, hours=self.MEMORY_LOOKBACK_HOURS)
                facts = [f for f in facts if value_said_by_user(own, f.objeto)]
                written = persist_facts(
                    palace,
                    facts,
                    source="studier_inferred",
                    min_confidence=self.FACT_MIN_CONFIDENCE,
                )
                result["facts_persisted"] = written
                log.info(
                    "Studier memory: %d candidate fact(s) → %d persisted.",
                    len(facts), written,
                )
                return result
            finally:
                palace.close()
        except Exception:
            log.exception("Studier memory: consolidation failed (non-fatal)")
            return result

    def _consolidate_preferences(self) -> int:
        """Run the implicit preference extractor over the last 24h of
        conversation. Persisted triples (via Palace.add_fact) become the
        long-running profile that the orchestrator can inject into the
        system prompt.

        Every step is wrapped: a Palace path missing, an empty corpus,
        a malformed LLM response, or an extractor crash must NOT stop
        the rest of the nightly run. Returns the number of triples
        persisted (0 on any short-circuit)."""
        try:
            from src.memory.palace import Palace  # local import to dodge
            # any circular-import risk if studier ever gets pulled into
            # the engine boot path.
            if not os.path.exists(self.PALACE_DB_PATH):
                log.info("Studier preferences: no palace.db — skipping.")
                return 0
            palace = Palace(db_path=self.PALACE_DB_PATH)
            try:
                corpus = recent_conversations(
                    palace,
                    hours=self.PREFERENCE_LOOKBACK_HOURS,
                    max_chars=self.PREFERENCE_MAX_CHARS,
                )
                if not corpus:
                    log.info("Studier preferences: no recent conversation — skipping.")
                    return 0
                triples = extract_preferences(
                    corpus,
                    llm_chat_fn=lambda **kw: ollama.chat(**kw, keep_alive=_KEEP_ALIVE, think=_THINK),
                    model=self.engine.model_name,
                )
                # Only what the user said (s88, Codex lote 6): the corpus has
                # Cortex's replies too, as context — never as evidence.
                own = recent_user_words(palace, hours=self.PREFERENCE_LOOKBACK_HOURS)
                triples = [t for t in triples if said_by_user(own, t.evidence)]
                written = persist_preferences(
                    palace,
                    triples,
                    source="studier_inferred",
                    min_confidence=self.PREFERENCE_MIN_CONFIDENCE,
                )
                log.info(
                    "Studier preferences: %d candidate(s) → %d persisted.",
                    len(triples), written,
                )
                return written
            finally:
                palace.close()
        except Exception:
            log.exception("Studier preferences: extraction failed (non-fatal)")
            return 0

    def run(self) -> List[str]:
        """Drain the queue and produce a briefing per pending topic."""
        # First action of the night: snapshot the Palace. Runs even when
        # the queue is empty so daily backups don't depend on workload.
        self._backup_palace()

        try:
            from src.core.studies_review import StudiesReview
            StudiesReview(
                pending_dir=self.output_dir,
                approved_dir=os.path.join(os.path.dirname(self.output_dir), "approved"),
            ).reconcile_approved()
        except Exception:
            log.exception("Nightly Studier: approval reconciliation failed (non-fatal)")

        # Retention AFTER backup, before any other write: every doc the
        # pass archives tonight also exists in tonight's snapshot, so
        # even an archive-file bug can't lose data (M3-A5, s72).
        self._retention_pass()

        # Second action: consolidate the day's memory — daily summary
        # document + structured facts as bitemporal triples. Runs
        # before _consolidate_preferences so the summary is available
        # if preferences ever wants to reuse it. Non-fatal by design.
        self._consolidate_memory()

        # Third action: extract implicit preferences from the day's
        # conversations. Runs unconditionally (independent of queue) so
        # the user's habits stay current even on nights with no study
        # topics. Errors here MUST NOT block the briefing pipeline.
        self._consolidate_preferences()

        with ExitStack() as run_lock:
            if not run_lock.enter_context(self._queue_run_lock()):
                try:
                    audit.log(audit.new_trace(), "studier", "run_busy")
                except OSError:
                    log.exception("Nightly Studier: run_busy audit failed (non-fatal)")
                return []
            trace = audit.new_trace()
            try:
                for entry in self.queue.recover_interrupted():
                    if entry["status"] == "consumed":
                        payload = {"linked": True}
                    else:
                        payload = {"retry": entry["retries"]}
                    payload.update(id=entry["id"], topic=entry.get("topic", ""))
                    audit.log(trace, "studier", "topic_interrupted", payload=payload)
                    if entry["status"] == "failed":
                        audit.log(trace, "studier", "retry_exhausted",
                                  payload={"topic": entry.get("topic", "")})
                pending = self.queue.pending_ids()
            except Exception:
                log.exception("Nightly Studier: queue recovery/snapshot failed (non-fatal)")
                pending = []
            # The deep queue (M8-A2) counts as work too — a night with only
            # deep topics must not short-circuit here. Peek (not consume): the
            # actual consume happens inside the lease window below.
            from src.core.deep_studier import _deep_enabled, deep_queue

            deep_waiting = _deep_enabled() and bool(deep_queue().list_pending())
            if not pending and not deep_waiting:
                log.info("Nightly Studier: queue empty — nothing to do.")
                return []

            log.info(f"Nightly Studier: processing {len(pending)} topic(s).")
            audit.log(trace, "studier", "run_start", payload={"n_topics": len(pending)})
            t_run = time.perf_counter()
            produced: List[str] = []
            # Background governance (P-2 decision, 2026-06-10): the nightly's
            # web traffic runs under a scoped, audited, bounded SystemLease —
            # before this, the Studier was the one producer touching the
            # internet with no lease, no cap and no per-query audit (M6-A2).
            # The raw searcher is swapped for the governed wrapper only for
            # the duration of the run; gather_snippets() already treats a
            # denied search like any failed query (logs and moves on), so an
            # exhausted lease degrades to thinner briefings, never a crash.
            from src.core.system_lease import GovernedWebSearcher, SystemLease

            raw_searcher = self.web_searcher
            lease = SystemLease.acquire(
                scope="web_search",
                origin="studier-nightly",
                minutes=int(os.getenv("CORTEX_SYSTEM_LEASE_MINUTES", "90")),
                max_calls=int(os.getenv("CORTEX_SYSTEM_LEASE_MAX_CALLS", "60")),
            )
            self.web_searcher = GovernedWebSearcher(raw_searcher, lease)
            try:
                for entry_id in pending:
                    try:
                        entry = self.queue.transition(
                            entry_id, ("pending",), "studying",
                            claimed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception:
                        log.exception("  topic claim failed (non-fatal)")
                        continue
                    if entry is None:
                        continue
                    topic = (entry.get("topic") or "").strip()
                    path = None
                    self._claimed_entry_id = entry_id
                    try:
                        if topic:
                            path = self._study_one(topic, trace=trace)
                    except Exception:
                        log.exception(f"  ✗ '{topic}' raised an exception")
                        audit.log(trace, "studier", "topic_error", payload={"topic": topic})
                    finally:
                        self._claimed_entry_id = None
                    try:
                        fields = {"briefing": path} if path else {}
                        consumed = self.queue.transition(
                            entry_id, ("studying", "saving"), "consumed",
                            consumed_at=datetime.now(timezone.utc).isoformat(), **fields,
                        )
                    except Exception:
                        log.exception("  topic completion failed (non-fatal)")
                        continue
                    if consumed is None:
                        continue
                    if path:
                        produced.append(path)
                        log.info(f"  ✓ '{topic}' → {path}")
                    else:
                        log.warning(f"  ✗ '{topic}' produced no output")
                        self._requeue_failed(entry, trace)

                # The ordinary study queue is finished; deep research keeps its
                # existing queue/lease behaviour outside this run lock.
                run_lock.close()

                # Deep-research queue (M8-A2): "pesquise a fundo X" enqueues
                # instead of blocking the request thread for minutes; the
                # topics run HERE, inside this already-governed lease window
                # (DeepStudier.research adds its own budget-pinned lease on
                # top — both count). Dormant: flag off → queue untouched, and
                # nothing was enqueued in the first place.
                from src.core.deep_studier import DeepStudier, _deep_enabled, deep_queue

                if _deep_enabled():
                    deep_pending = deep_queue().consume_all()
                    if deep_pending:
                        log.info(
                            "Nightly Studier: %d deep topic(s).", len(deep_pending),
                        )
                    for entry in deep_pending:
                        topic = (entry.get("topic") or "").strip()
                        if not topic:
                            continue
                        try:
                            path = DeepStudier(
                                studier=self, output_dir=self.output_dir,
                            ).run_and_save(topic, trace=trace)
                            if path:
                                produced.append(path)
                                log.info(f"  ✓ deep '{topic}' → {path}")
                        except Exception:
                            log.exception(f"  ✗ deep '{topic}' raised an exception")
                            audit.log(trace, "studier", "deep_topic_error",
                                      payload={"topic": topic})
            finally:
                self.web_searcher = raw_searcher
                lease.release()
            audit.log(trace, "studier", "run_end",
                      latency_ms=(time.perf_counter() - t_run) * 1000,
                      payload={"produced": len(produced)})
            return produced

    # --- internals ---------------------------------------------------

    @contextmanager
    def _queue_run_lock(self):
        """A live holder owns recovery and claims; never wait for that holder."""
        fd = None
        acquired = False
        try:
            fd = os.open(self.queue.path + ".run.lock", os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            log.info("Nightly Studier: another run owns the queue — skipping.")
        except OSError:
            log.exception("Nightly Studier: run lock unavailable — skipping queue.")
        try:
            yield acquired
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    log.exception("Nightly Studier: run lock close failed (non-fatal)")

    def _requeue_failed(self, entry: Dict, trace: Optional[str]) -> None:
        """Give a topic that produced nothing tonight ONE more chance.

        Seen live 2026-07-10: the web lease expired mid-run and 4 of 6
        topics hit no_snippets — consumed from the queue, briefing never
        produced, silently LOST. Re-enqueue with the '!!' force flag (the
        topic already exists as consumed, so the duplicate guard would
        veto it) and a retry counter so a topic that fails twice is
        dropped for good instead of looping forever. Non-fatal by design."""
        topic = (entry.get("topic") or "").strip()
        retries = int(entry.get("retries") or 0)
        if not topic:
            return
        if retries >= 1:
            audit.log(trace or audit.new_trace(), "studier",
                      "retry_exhausted", payload={"topic": topic})
            return
        try:
            self.queue.add(
                topic + " !!",
                extra={"retries": retries + 1, "requeued": "failed_run"},
            )
            audit.log(trace or audit.new_trace(), "studier",
                      "topic_requeued",
                      payload={"topic": topic, "retry": retries + 1})
        except Exception:
            log.exception("  requeue failed (non-fatal)")

    def _study_one(self, topic: str, trace: Optional[str] = None) -> Optional[str]:
        study_time = StudyTime.for_topic(topic)
        trace = trace or audit.new_trace()
        audit.log(trace, "studier", "topic_start", payload={"topic": topic})

        # ml_context still drives the DOMAIN BLOCKLIST (dropping Audi/fiscal
        # pages from an ML briefing); disambiguation is a narrower decision
        # since s83. The log used to claim "disambig active" off the first
        # flag, so it said active on a run where it was off — a log that
        # misreports what ran is worse than no log.
        ml_context = self.is_ml_topic(topic)
        if ml_context:
            log.info("  topic '%s' tagged as ML-context (blocklist on, "
                     "disambig %s)", topic,
                     "on" if self._needs_disambiguation(topic) else
                     "off — topic is specific enough")

        t0 = time.perf_counter()
        queries = self.generate_queries(topic, study_time=study_time)
        audit.log(trace, "studier", "query_gen",
                  latency_ms=(time.perf_counter() - t0) * 1000,
                  payload={"n_queries": len(queries), "queries": queries})
        log.info(f"  queries for '{topic}': {queries}")

        t1 = time.perf_counter()
        filter_counts = {}
        snippets = self._gather_for_topic(queries, topic=topic, ml_context=ml_context,
                                         filter_counts=filter_counts)
        audit.log(trace, "studier", "search_done",
                  latency_ms=(time.perf_counter() - t1) * 1000,
                  payload={"snippet_chars": len(snippets), **filter_counts})
        if not snippets.strip():
            reason = "all_off_topic" if filter_counts.get("off_topic_dropped", 0) else "no_results"
            log.warning("  no snippets gathered for '%s' (%s)", topic, reason)
            audit.log(trace, "studier", "no_snippets",
                      payload={"topic": topic, "reason": reason})
            # Empty context invites a briefing from model memory. Let run()
            # give this topic its existing single retry, without restoring text.
            return None

        t2 = time.perf_counter()
        briefing = self._summarise(topic, snippets, ml_context=ml_context,
                                   study_time=study_time)
        if briefing is None:
            # No briefing, so no file and no "produced": run() gives the
            # topic its one retry (s88, Codex lote 14 — invariant 7).
            audit.log(trace, "studier", "synthesis_failed",
                      latency_ms=(time.perf_counter() - t2) * 1000,
                      payload={"topic": topic})
            return None
        audit.log(trace, "studier", "synthesis_done",
                  latency_ms=(time.perf_counter() - t2) * 1000,
                  payload={"briefing_chars": len(briefing)})

        t3 = time.perf_counter()
        briefing = self._apply_critic(topic, briefing, trace=trace, study_time=study_time)
        audit.log(trace, "studier", "critic_done",
                  latency_ms=(time.perf_counter() - t3) * 1000)

        path = self._save(topic, briefing, queries)
        audit.log(trace, "studier", "topic_saved", payload={"path": path})
        self._maybe_reflect(topic, briefing, trace)
        return path

    def _maybe_reflect(self, topic: str, briefing: str,
                       trace: Optional[str]) -> None:
        """REFLECT edge (curiosity loop, autonomous_self_improvement.md
        §A.3 step 6): propose follow-up topics from tonight's briefing and
        re-enqueue them. DORMANT behind CORTEX_CURIOSITY (default off) —
        flag off means this is a no-op and tonight's run is unchanged.
        Proposals land as pending queue entries for the NEXT night (run()
        already snapshotted the queue via consume_all), so a reflection can
        never extend the current run. Failures never block the briefing."""
        from src.core.curiosity import CuriosityEngine, _curiosity_enabled

        if not _curiosity_enabled():
            return
        try:
            CuriosityEngine(queue=self.queue).reflect(topic, briefing,
                                                      trace=trace)
        except Exception:
            log.exception("  curiosity reflect failed (non-fatal)")

    def is_ml_topic(self, topic: str) -> bool:
        """True when the topic mentions ML/LLM-specific terms — used to
        auto-disambiguate queries and filter off-topic search results."""
        return bool(self._ML_CONTEXT_TRIGGERS.search(topic or ""))

    # Below this many words a topic is short enough to be ambiguous ("Q4 vs
    # Q5", "KV cache") and worth disambiguating. Above it, the topic names
    # what it wants and the disambiguation DILUTES it: asked about "reranker
    # cross-encoder para RAG em Apple Silicon sem PyTorch: bge-reranker, ONNX
    # Runtime e MLX", the model produced "machine learning LLM on Apple
    # Silicon without PyTorch" — the word "reranker" gone from the query, and
    # with it the subject (s83). Generic queries bought generic snippets and
    # the briefing never answered the question.
    _DISAMBIG_MAX_WORDS = 6

    def _needs_disambiguation(self, topic: str) -> bool:
        return (self.is_ml_topic(topic)
                and len((topic or "").split()) <= self._DISAMBIG_MAX_WORDS)

    # Deliberately small vocabulary from the reported news / attention
    # topics, not a translation dictionary. The first spelling is canonical.
    # News/events are also generic below: they cannot prove relevance alone.
    _RELEVANCE_ALIASES = (
        ("brasil", "brazil"),
        ("uk", "reino unido", "united kingdom", "britain"),
        ("noticia", "noticias", "news"),
        ("acontecimento", "acontecimentos", "event", "events"),
        ("atencao", "attention"),
        ("mecanismo", "mechanism"),
    )
    # Function words plus framing/time/quality words from these topics.
    # Both this list and topic terms undergo the SAME normalisation.
    _RELEVANCE_GENERIC = (
        "para sobre como sem com e o a os as de da do das dos em no na nos nas "
        "que the and for with without how to in on of vs usar quando cada "
        "um uma mais melhor melhores principal principais novidade novidades "
        "ultimo ultimos ultima ultimas ano anos dois duas pratica praticas "
        "noticia noticias acontecimento acontecimentos news events "
        "best better main major new latest last recent year years practice practices"
    ).split()

    @staticmethod
    def _singular_term(word: str) -> str:
        """Bounded PT/EN plural folding, not linguistic stemming.

        Preserve macOS and -ss/-us/-is endings (CSS, cross, analysis).
        Numeric model/version terms keep their digits; plain numbers are
        excluded separately by _distinctive_terms.
        """
        if len(word) > 4:
            for ending, singular in (("oes", "ao"), ("aes", "ao"),
                                     ("ais", "al"), ("ies", "y")):
                if word.endswith(ending):
                    return word[:-len(ending)] + singular
        if word == "macos" or word.endswith(("ss", "us", "is")):
            return word
        return word[:-1] if len(word) > 3 and word.endswith("s") else word

    @classmethod
    def _normalise_relevance(cls, text: str, *, compounds: bool = False) -> str:
        text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                       if not unicodedata.combining(c))
        text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
        replacements = []
        for aliases in cls._RELEVANCE_ALIASES:
            pattern = "|".join(re.escape(a).replace(r"\ ", r"[\s-]+")
                               for a in sorted(aliases, key=len, reverse=True))
            pattern = r"\b(?:" + pattern + r")\b"
            replacements.append((pattern, aliases[0]))
            text = re.sub(pattern, aliases[0], text)
        pattern = r"[a-z0-9]+(?:-[a-z0-9]+)*" if compounds else r"[a-z0-9]+"
        text = " ".join("-".join(cls._singular_term(p) for p in word.split("-"))
                        for word in re.findall(pattern, text))
        # Fold translated plurals too: mechanisms -> mechanism -> mecanismo.
        for pattern, canonical in replacements:
            text = re.sub(pattern, canonical, text)
        return text

    @classmethod
    def _distinctive_terms(cls, topic: str) -> List[str]:
        """Shared query/result vocabulary: non-generic tokens of >=3 chars,
        original 2–5-letter capital acronyms or alphanumeric codes (Q4).
        Acronyms retain their case for matching against original text.
        Numeric-only terms never count, including year ranges. Keep hyphenated names intact,
        so cross-encoder matches the whole phrase.
        """
        stop = {cls._normalise_relevance(w, compounds=True)
                for w in cls._RELEVANCE_GENERIC}
        acronyms = re.findall(r"(?<!\w)[A-ZÀ-ÖØ-Þ]{2,5}(?!\w)", topic or "")
        acronym_terms = {cls._normalise_relevance(w, compounds=True) for w in acronyms}
        terms = [
            w for w in cls._normalise_relevance(topic or "", compounds=True).split()
            if any(p not in stop and not p.isdigit() for p in w.split("-"))
            and (len(w) >= 3 or w == "uk" or (len(w) == 2 and any(c.isdigit() for c in w)))
            and w not in acronym_terms
        ]
        return list(dict.fromkeys(terms + acronyms))

    @classmethod
    def _matches_topic(cls, text: str, terms: List[str]) -> bool:
        normalised = " " + cls._normalise_relevance(text) + " "
        return any(bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text))
                   if term.isupper() else " " + term.replace("-", " ") + " " in normalised
                   for term in terms)

    def _keep_on_topic(self, queries: List[str], topic: str) -> List[str]:
        """Drop queries that share no distinctive term with the topic.

        A query is a proxy for the question; one that kept none of the
        question's own words is researching something else. If that would
        empty the list, the topic itself is the fallback query — better a
        narrow search than a wrong one."""
        terms = self._distinctive_terms(topic)
        if not terms:
            log.info("  no distinctive terms for topic '%s'; keeping queries", topic)
            return queries
        kept = [q for q in queries if self._matches_topic(q, terms)]
        dropped = [q for q in queries if q not in kept]
        if dropped:
            log.info("  dropped %d off-topic quer%s: %s",
                     len(dropped), "y" if len(dropped) == 1 else "ies", dropped)
        return kept or [topic]

    def generate_queries(self, topic: str, *, study_time: Optional[StudyTime] = None) -> List[str]:
        """Ask the LLM for 3-5 focused search queries."""
        study_time = study_time or StudyTime.for_topic(topic)
        disambig_clause = ""
        if self._needs_disambiguation(topic):
            disambig_clause = (
                "- This topic is about MACHINE LEARNING / LLMs. Always "
                "include a contextual word like 'LLM', 'transformer', "
                "'machine learning', or 'neural network' in EVERY query "
                "to avoid getting back results about cars, fiscal "
                "quarters, gas stations, or unrelated abbreviations.\n"
            )
        prompt = (
            "Generate 3 to 5 distinct, focused web-search queries to "
            "research the topic below. Return ONLY the queries, one per "
            "line, no numbering, no quotes, no explanation.\n\n"
            + study_time.prompt_block() + "\n"
            "CRITICAL RULES:\n"
            "- Do NOT expand acronyms or define terms in the queries.\n"
            "- Do NOT guess what a term means — use the EXACT name the "
            "user gave.\n"
            "- Do NOT invent facts, organisations, or attributions.\n"
            "- Queries should be short search phrases, not sentences.\n"
            "- If the topic covers explicit years, include those years in the queries. "
            "For past events, search for what happened, not event calendars or announcements.\n"
            "- EVERY query must keep at least one distinctive word from the "
            "topic — the specific name, library or technique it asks about. "
            "A query without it researches something else.\n"
            + disambig_clause +
            "\n"
            f"Topic: {topic}\n"
            "Queries:"
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 200},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M2-A7: nightly must not downgrade the timer
            )
        except Exception as e:
            log.warning(f"  query generation failed: {e}")
            return study_time.anchor_queries([topic])

        text = (response.get("response") or "").strip()
        candidates = [
            line.strip().lstrip("-•*").strip().strip('"').rstrip("?.,;:")
            for line in text.splitlines()
            if line.strip()
        ]

        # Factuality filter: drop queries that contain a fabricated acronym
        # expansion ("RAG = Regressive Attention Generator", "RAG (Random
        # Augmented Graph)"). These poison the search results. If every
        # candidate is dirty, fall back to the literal topic so we still
        # do a search rather than skipping the topic entirely.
        clean = [q for q in candidates if not self._query_looks_hallucinated(q)]
        if candidates and not clean:
            log.warning(
                f"  all generated queries for '{topic}' contained suspect "
                f"acronym expansions; falling back to literal topic"
            )
            return study_time.anchor_queries([topic])

        queries = [q for q in clean if 5 < len(q) < 200][: self.MAX_QUERIES_PER_TOPIC]

        # A query that kept none of the topic's own distinctive words is
        # researching a different subject (s83).
        queries = self._keep_on_topic(queries, topic)

        # Belt and braces: if the LLM ignored the disambig clause, force
        # an ML keyword into queries that don't already have one. Only for
        # the short, genuinely ambiguous topics — on a specific one this is
        # what replaced "reranker" with "machine learning".
        if self._needs_disambiguation(topic) and queries:
            queries = [self._inject_ml_context(q) for q in queries]

        return study_time.anchor_queries(queries or [topic])

    @classmethod
    def _query_looks_hallucinated(cls, query: str) -> bool:
        """True when the query contains a fabricated acronym expansion."""
        return bool(
            cls._ACRONYM_EXPANSION_RE.search(query) or
            cls._PARENTHETICAL_EXPANSION_RE.search(query)
        )

    @staticmethod
    def _inject_ml_context(query: str) -> str:
        """Add 'LLM' to the query if it doesn't already contain an ML
        disambiguator. Conservative: only adds when clearly missing."""
        ml_words = ("llm", "transformer", "neural", "machine learning",
                    "deep learning", "model", "ai ", "attention")
        q_lower = query.lower()
        if any(w in q_lower for w in ml_words):
            return query
        return f"{query} LLM"

    def _gather_for_topic(self, queries: List[str], *, topic: str,
                          ml_context: bool, filter_counts: Dict[str, int]) -> str:
        """Adapt older injected gatherers without masking errors inside them.

        Real gathering accepts the topic and filters each query. Older
        replacements may accept only queries/ml_context; check their output
        here as well so the compatibility path cannot bypass either gate.
        """
        kwargs = dict(topic=topic, ml_context=ml_context, filter_counts=filter_counts)
        try:
            inspect.signature(self.gather_snippets).bind(queries, **kwargs)
        except TypeError:
            result = self.gather_snippets(queries, ml_context=ml_context)
            filtered, blocked = self._filter_blocklist(result, include_ml=ml_context)
            filtered, off_topic = self._filter_relevance(filtered, topic)
            filter_counts.update(blocklist_dropped=blocked, off_topic_dropped=off_topic)
            log.info("  dropped %d off-topic snippet(s) for topic '%s'", off_topic, topic)
            return filtered
        return self.gather_snippets(queries, **kwargs)

    def gather_snippets(self, queries: List[str], ml_context: bool = False, *,
                        topic: Optional[str] = None,
                        filter_counts: Optional[Dict[str, int]] = None) -> str:
        """Blocklist, then lexical relevance, before any query text is added.

        Pipeline callers supply the user's topic (not an LLM-generated
        query). Legacy standalone callers are checked against each query.
        filter_counts is a per-call output for the existing audit events.
        """
        chunks: List[str] = []
        dropped = 0
        off_topic = 0
        skip_relevance = topic is not None and not self._distinctive_terms(topic)
        if skip_relevance:
            log.info("  no distinctive terms for topic '%s'; skipping relevance filter", topic)
        for q in queries:
            try:
                result = self.web_searcher.search(q, max_results=self.SEARCH_RESULTS_PER_QUERY)
            except Exception as e:
                log.warning(f"  search failed for '{q}': {e}")
                continue
            if result.startswith("[SYSTEM"):
                log.warning(f"  search returned system notice for '{q}': {result[:80]}")
                continue
            # Always drop general junk (crypto explorers, mining); add the
            # ML-specific noise list only for ML topics.
            filtered, n_dropped = self._filter_blocklist(result, include_ml=ml_context)
            dropped += n_dropped
            if not filtered.strip():
                log.warning(f"  all snippets for '{q}' hit the blocklist — skipped")
                continue
            if skip_relevance:
                n_off_topic = 0
            else:
                filtered, n_off_topic = self._filter_relevance(filtered, topic if topic is not None else q)
            off_topic += n_off_topic
            if not filtered.strip():
                log.warning(f"  all snippets for '{q}' are off-topic — skipped")
                continue
            chunks.append(f"### Query: {q}\n{filtered}")
        if dropped:
            log.info(f"  dropped {dropped} blocklisted snippet(s)")
        log.info("  dropped %d off-topic snippet(s) for topic '%s'",
                 off_topic, topic if topic is not None else " / ".join(queries))
        if filter_counts is not None:
            filter_counts.update(blocklist_dropped=dropped, off_topic_dropped=off_topic)
        return "\n\n".join(chunks)

    @staticmethod
    def _result_blocks(snippets_text: str) -> List[str]:
        """Split both Tavily Source and DDG Result records, keeping content
        paragraphs with their URL. Keep a provider QUICK ANSWER as its own
        block; query headings cannot supply evidence for any block.
        Blank-line splitting remains for older unlabelled snippet callers.
        """
        text = re.sub(r"(?m)^### Query:[^\n]*\n?", "", snippets_text)
        starts = list(re.finditer(r"(?m)^(?:Source|Result)\s+\d+\s*\(", text))
        if starts:
            offsets = [m.start() for m in starts] + [len(text)]
            preamble = text[:offsets[0]].strip()
            return ([preamble] if preamble else []) + [
                text[a:b].strip() for a, b in zip(offsets, offsets[1:])]
        return [b.strip() for b in text.split("\n\n") if b.strip()]

    @classmethod
    def _filter_relevance(cls, snippets_text: str, topic: str) -> tuple:
        """Keep a block iff its title/snippet matches a distinctive topic
        term. Strip URLs (including paths), labels and query headings from
        the evidence. With no terms, skip relevance while retaining the
        preceding blocklist gate.
        """
        terms = cls._distinctive_terms(topic)
        if not terms:
            log.info("  no distinctive terms for topic '%s'; skipping relevance filter", topic)
            return snippets_text, 0
        kept = []
        dropped = 0
        for block in cls._result_blocks(snippets_text):
            evidence = re.sub(r"https?://[^\s\]<>]+", "", block)
            evidence = re.sub(r"^(?:Source|Result)\s+\d+\s*", "", evidence)
            if cls._matches_topic(evidence, terms):
                kept.append(block)
            else:
                dropped += 1
        return "\n\n".join(kept), dropped

    @classmethod
    def _filter_blocklist(cls, snippets_text: str, *, include_ml: bool = False) -> tuple:
        """Drop result blocks whose URL matches a junk domain. Returns
        (filtered_text, dropped_count).

        The general list (crypto explorers, mining) applies to EVERY topic;
        the ML-specific list (Audi/fiscal/Rick-Roll noise from short
        ambiguous terms) is added only when include_ml=True."""
        # Tavily-style output blocks look like:
        #   Source N (Title):
        #   <content>
        #   [https://url/path]
        # DDG uses Result N and single newlines; share the relevance splitter.
        blocklist = cls._GENERAL_DOMAIN_BLOCKLIST + (
            cls._ML_DOMAIN_BLOCKLIST if include_ml else ()
        )
        blocks = cls._result_blocks(snippets_text)
        kept = []
        dropped = 0
        for block in blocks:
            block_lower = block.lower()
            if any(bad in block_lower for bad in blocklist):
                dropped += 1
                continue
            kept.append(block)
        return ("\n\n".join(kept), dropped)

    def _summarise(self, topic: str, snippets: str,
                   ml_context: bool = False, *,
                   study_time: Optional[StudyTime] = None) -> Optional[str]:
        """Ask the LLM for a structured Markdown briefing. None when it could
        not be written: an Ollama failure used to come back as "## Erro /
        Falha ao gerar resumo: …", saved as tonight's briefing, counted as
        produced and never retried (s88, Codex lote 14)."""
        if len(snippets) > self.SNIPPET_CHAR_LIMIT:
            snippets = snippets[: self.SNIPPET_CHAR_LIMIT] + "\n\n[truncated]"

        ml_guard = ""
        if ml_context:
            ml_guard = (
                "\nML/LLM TOPIC GUARD: this topic is about machine learning. "
                "If you see ANY source that is clearly off-topic (Audi cars, "
                "fiscal quarters, gas stations, Rick Astley YouTube videos, "
                "investor relations, unrelated abbreviations), you MUST:\n"
                "1. NOT cite that source in Fontes consultadas.\n"
                "2. NOT extract any facts from it.\n"
                "3. Cap Confiança at 'baixa' and explain in the "
                "justification that sources were contaminated.\n"
                "4. If 50%+ of sources are off-topic, write exactly: "
                "'## Confiança\\nbaixa — fontes contaminadas, re-estudar "
                "com query mais específica.'\n"
            )

        from src.core.prompt_safety import fence, guard_line

        study_time = study_time or StudyTime.for_topic(topic)

        prompt = (
            "You are CORTEX preparing an overnight research briefing for the "
            "user. Synthesise the following web snippets into a clear, "
            "structured Markdown report. The user reads this in the morning "
            "and decides whether to keep the knowledge permanently.\n\n"
            + study_time.prompt_block() + EVENT_TIME_RULES + "\n"
            + guard_line("en") + "\n\n"
            f"Topic: {topic}\n\n"
            "Web snippets to synthesise:\n"
            f"{fence(snippets)}\n\n"
            + ml_guard +
            "\nOutput STRICTLY in this Markdown format (do not deviate):\n\n"
            "## Resumo executivo\n"
            "<3-5 sentences. Write the prose in the topic's language "
            "(Brazilian Portuguese PT-BR, never PT-PT, or UK English), but "
            "keep technical terms in English — see RULES.>\n\n"
            "## Pontos-chave\n"
            "- <bullet with a concrete fact, not a vague claim>\n"
            "- <at least 5 bullets if the topic is broad>\n\n"
            "## Fontes consultadas\n"
            "- <FULL URL exactly as it appears in the snippets inside [ ] brackets>\n"
            "- <List every unique URL from the snippets. Do NOT replace them "
            "with domain names or labels like 'Result 1'. Copy the exact URL.>\n\n"
            "## Confiança\n"
            "<alta | média | baixa> — <1-sentence justification>\n\n"
            "CONFIDENCE CRITERIA (follow strictly):\n"
            "- alta: 3+ independent reputable sources agree on the key facts, "
            "no contradictions found, topic is well-established. Reputable "
            "means: arxiv papers, official docs (vendor / framework), "
            "established outlets, peer-reviewed. NOT: YouTube random videos, "
            "random blog domains, marketing pages.\n"
            "- média: fewer than 3 sources, OR sources are user-generated "
            "(Wikipedia, Reddit, TikTok, forums), OR some facts could not be "
            "cross-verified.\n"
            "- baixa: only 1 source, OR sources contradict each other, OR the "
            "snippets are tangential / off-topic for the query, OR sources "
            "look fake / random / off-domain.\n\n"
            "RULES:\n"
            "- Stick to what the snippets ACTUALLY say. Do NOT fabricate facts.\n"
            "- Keep technical terms, library/framework/model names, and "
            "acronyms in their original English form — do NOT translate them. "
            "Write 'constrained decoding', 'structured outputs', 'GBNF', "
            "'MLX', 'KV cache', model names verbatim, never a Portuguese "
            "rendering like 'decodificação construída'.\n"
            "- ANSWER THE SPECIFIC QUESTION in the Topic above. If it asks to "
            "compare or decide ('X vs Y', 'when to use X', 'tradeoffs'), "
            "deliver that comparison — do NOT pad with generic background "
            "('what X is') that sidesteps the question.\n"
            "- NEVER invent specific numbers (sizes, percentages, dates). If "
            "the snippet doesn't have the number, omit the claim.\n"
            "- If sources disagree, flag the divergence in 'Pontos-chave'.\n"
            "- Do NOT output more than one briefing — produce exactly ONE set "
            "of sections (Resumo, Pontos-chave, Fontes, Confiança).\n"
            "- Be concise but substantive. Shallow platitudes lower the rating."
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 1500},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M2-A7
            )
            briefing = (response.get("response") or "").strip()
            if not briefing:
                log.warning("  summary generation returned nothing")
                return None
            briefing = self._enforce_confidence_floor(briefing)
            return self._flag_thin_sources(briefing)
        except Exception as e:
            log.exception(f"  summary generation failed: {e}")
            return None

    @classmethod
    def _enforce_confidence_floor(cls, briefing: str) -> str:
        """Downgrade Confiança 'alta' / 'média' when the briefing cites
        fewer unique source domains than the verdict warrants.

        - 3+ unique domains: any verdict allowed
        - 2 unique domains: floor at 'média' (rewrite alta → média)
        - 0-1 unique domains: floor at 'baixa' (rewrite alta/média → baixa)

        Deterministic guard against Llama 8B's tendency to default to 'alta'.
        Only rewrites — never upgrades. If the LLM already said 'baixa',
        we trust it.
        """
        m_fontes = cls._FONTES_BLOCK_RE.search(briefing)
        if not m_fontes:
            return briefing
        domains = {
            d.lower() for d in cls._URL_IN_BLOCK_RE.findall(m_fontes.group(1))
        }
        n = len(domains)
        if n >= 3:
            return briefing  # any verdict allowed

        m_conf = cls._CONFIANCA_LINE_RE.search(briefing)
        if not m_conf:
            return briefing
        current = m_conf.group(2).lower()
        if current in ("baixa", "low"):
            return briefing  # never upgrade

        new_level = "média" if n >= 2 else "baixa"
        # If we're already at média and domains >= 2, no change needed.
        if current in ("média", "media", "medium") and new_level == "média":
            return briefing

        return cls._CONFIANCA_LINE_RE.sub(
            lambda mm: (f"{mm.group(1)}{new_level} — poucas fontes independentes "
                        f"({n} domínio(s))."),
            briefing,
            count=1,
        )

    @classmethod
    def _flag_thin_sources(cls, briefing: str) -> str:
        """Prepend a loud banner when the briefing cites ZERO verifiable
        source domains (no 'Fontes consultadas' block, or a block with no
        URLs). Separate from the confidence floor on purpose: the floor
        tunes the verdict word, this makes an unsourced briefing
        impossible to approve on a skim. Idempotent via the marker."""
        if cls._SOURCE_WARNING_MARKER in briefing:
            return briefing
        m_fontes = cls._FONTES_BLOCK_RE.search(briefing)
        domains = set()
        if m_fontes:
            domains = {
                d.lower() for d in cls._URL_IN_BLOCK_RE.findall(m_fontes.group(1))
            }
        if not domains:
            return cls._SOURCE_WARNING_BANNER + briefing
        return briefing

    # PROBLEMA | rule | quote [| second quote for contradicao] | PT sentence.
    _CRITIC_MAX_PROBLEMS = 3
    _CRITIC_MAX_CONTRADICTIONS = 2
    _CRITIC_MIN_QUOTE_CHARS = 24
    _CRITIC_MIN_QUOTE_WORDS = 4
    _CRITIC_SECTION_RE = re.compile(
        r"^[ \t]*##[ \t]+Cr[ií]tico[ \t]*(?:\r?\n|$).*?(?=^[ \t]*##[ \t]+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    _CRITIC_QUOTE_MARKS = str.maketrans("", "", "\"'‘’“”«»‹›„‟‚‛")
    _CRITIC_NO_PROOF = "O crítico não encontrou nenhum problema comprovado."
    _CRITIC_TO_CHECK = "A conferir (o crítico não prova contradição):"
    _CRITIC_RULES = {"sigla": "sigla", "absoluta": "absoluta",
                     "contradicao": "contradição", "data": "data"}
    # Deliberately short, whole-word vocabularies. Match case/accent-insensitively
    # in the QUOTE only; explanation wording cannot supply a missing trigger.
    _CRITIC_ABSOLUTE_WORDS = (
        "sempre", "nunca", "todos", "todas", "tudo", "nenhum", "nenhuma", "jamais",
        "always", "never", "all", "every", "none",
    )
    _CRITIC_MONTHS = (
        "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
        "setembro", "outubro", "novembro", "dezembro",
        "january", "february", "april", "june", "july",
        "september", "october", "november", "december",
    )
    # Also ordinary English verbs/adjectives: require adjacent date context.
    _CRITIC_AMBIGUOUS_MONTH_RE = re.compile(
        r"\b(?:march|may|august)\s+(?:0?[1-9]|[12]\d|3[01]|(?:19|20)\d{2})\b"
        r"|\b(?:0?[1-9]|[12]\d|3[01]|(?:19|20)\d{2})\s+(?:march|may|august)\b",
        re.IGNORECASE,
    )
    _CRITIC_EVENT_WORDS = (
        "terá", "será", "previsto", "prevista", "previstos", "previstas",
        "agendado", "agendada", "aconteceu", "ocorreu", "acontecerá", "ocorrerá",
        "scheduled", "upcoming", "happened",
    )
    # A standalone 1900–2099 year, or numeric day/month (slash or hyphen),
    # optionally followed by a two-/four-digit year. This is a shape check,
    # not calendar validation or a deterministic finding that a date is wrong.
    _CRITIC_DATE_RE = re.compile(
        r"\b(?:19|20)\d{2}\b|\b(?:0?[1-9]|[12]\d|3[01])[/\-](?:0?[1-9]|1[0-2])"
        r"(?:[/\-](?:\d{4}|\d{2}))?\b"
    )
    # Acronym casing is significant. Look for at least two alphabetic words
    # (up to 12, allowing hyphens) in explicit, adjacent expansion syntax.
    _CRITIC_ACRONYM = r"(?<!\w)[A-ZÀ-ÖØ-Þ]{2,6}(?!\w)"
    _CRITIC_EXPANSION = r"[^\W\d_]+(?:[-\s]+[^\W\d_]+){1,11}"

    @classmethod
    def _normalise_critic_quote(cls, text: str) -> str:
        """Ignore only Unicode compatibility, case, quote marks and spacing."""
        text = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(text.translate(cls._CRITIC_QUOTE_MARKS).split())

    @staticmethod
    def _critic_fold(text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                       if not unicodedata.combining(c))

    @classmethod
    def _critic_has_words(cls, quote: str, vocabulary: tuple) -> bool:
        words = set(re.findall(r"\b\w+\b", cls._critic_fold(quote)))
        return bool(words.intersection(cls._critic_fold(w) for w in vocabulary))

    @classmethod
    def _critic_has_expansion(cls, quote: str) -> bool:
        text = unicodedata.normalize("NFKC", quote)
        expansion = cls._CRITIC_EXPANSION
        for acronym in re.finditer(cls._CRITIC_ACRONYM, text):
            after = text[acronym.end():]
            before = text[:acronym.start()]
            if re.match(rf"\s*\(\s*{expansion}\s*\)", after):
                return True
            if re.match(rf"\s*(?:=\s*|(?i:stands\s+for|significa|[ée]\s+a\s+sigla\s+de)\s+)"
                        rf"{expansion}", after):
                return True
            if (re.match(r"\s*\)", after)
                    and re.search(rf"{expansion}\s*\(\s*$", before)):
                return True
        return False

    @classmethod
    def _critic_rule_fits(cls, rule: str, quotes: list[str], body: str) -> bool:
        if rule == "absoluta":
            return cls._critic_has_words(quotes[0], cls._CRITIC_ABSOLUTE_WORDS)
        if rule == "sigla":
            return cls._critic_has_expansion(quotes[0])
        if rule == "data":
            return (bool(cls._CRITIC_DATE_RE.search(quotes[0]))
                    or bool(cls._CRITIC_AMBIGUOUS_MONTH_RE.search(quotes[0]))
                    or cls._critic_has_words(quotes[0], cls._CRITIC_MONTHS + cls._CRITIC_EVENT_WORDS))
        if rule == "contradicao":
            first, second = map(cls._normalise_critic_quote, quotes)
            if first == second:
                return False
            # Consider every occurrence: the first matches may overlap even
            # when a later occurrence supplies a separate passage.
            spans = [[(m.start(), m.start() + len(q))
                      for m in re.finditer(rf"(?={re.escape(q)})", body)]
                     for q in (first, second)]
            return any(end1 <= start2 or end2 <= start1
                       for start1, end1 in spans[0] for start2, end2 in spans[1])
        return False

    @staticmethod
    def _critic_sentence(explanation: str) -> bool:
        """A conservative readability gate, not a grammatical fact-checker."""
        text = "".join(c for c in unicodedata.normalize("NFKD", explanation.casefold())
                       if not unicodedata.combining(c))
        words = re.findall(r"[^\W\d_]+", text, re.UNICODE)
        labels = {
            "problema", "problemas", "regra", "regras", "numero", "numeros",
            "rule", "rules", "number", "numbers", "e", "a", "de", "and", "to",
            "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        }
        return (len(explanation) >= 24 and len(words) >= 5
                and explanation.endswith((".", "!", "?"))
                and any(w not in labels for w in words))

    @staticmethod
    def _critic_missing_expansion(explanation: str) -> bool:
        """Missing acronym expansions are outside rule 1, even with a quote."""
        text = "".join(c for c in unicodedata.normalize("NFKD", explanation.casefold())
                       if not unicodedata.combining(c))
        return bool(re.search(
            r"\bunexpanded\b|"
            r"\b(?:nao|not|never)\b[^.;!?]{0,80}\b(?:expand\w*|spell\w*\s+out)\b|"
            r"\bnao\b[^.;!?]{0,40}\b(?:explicad\w*|definid\w*)\b[^.;!?]*\bextenso\b|"
            r"\b(?:sigla\w*|acron\w*)\b[^.;!?]{0,40}\b(?:nao|not)\b"
            r"[^.;!?]{0,40}\b(?:definid\w*|defined|explicad\w*|explained)\b|"
            r"\b(?:sem|without|no|missing)\s+(?:uma?\s+|an?\s+)?(?:expans\w*|explicacao)\b|"
            r"\b(?:falta\w*|carece\w*|ausencia)\b[^.;!?]{0,40}\bexpans\w*\b|"
            r"\b(?:sigla\w*|acron\w*)\b[^.;!?]{0,40}\bpor\s+expandir\b",
            text,
        ))

    @classmethod
    def _parse_critic_proof(cls, reply: str, critic_input: str) -> tuple[list, int, bool]:
        """Return accepted (rule, quotes, sentence), discard count and bad format.

        Only PROBLEMA lines count as complaints. Legacy semicolon items still
        count individually. Proven complaints and tentative contradictions have
        separate display caps; valid complaints beyond either are discarded.
        Pipes inside quotes are preserved, testing every two-quote boundary.
        """
        body = cls._normalise_critic_quote(critic_input)
        accepted, discarded, malformed = [], 0, not reply.strip()
        counts = {False: 0, True: 0}
        chatter, valid_line = False, False
        for line in reply.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"OK(?:\s*\|.*)?", line, re.IGNORECASE):
                valid_line = True
                continue
            if not re.match(r"PROBLEMA\b", line, re.IGNORECASE):
                chatter = True
                continue
            fields = line.split("|", 2)
            if len(fields) < 3 or "|" not in fields[2] or fields[0].strip().casefold() != "problema":
                legacy = re.match(r"PROBLEMA\s*\|(.*)", line, re.IGNORECASE)
                discarded += (max(1, len([p for p in legacy[1].split(";") if p.strip()]))
                              if legacy and len(fields) == 2 else 1)
                malformed = True
                continue
            rule = fields[1].strip().casefold()
            if rule not in cls._CRITIC_RULES:
                discarded += 1
                continue
            quoted, explanation = fields[2].rsplit("|", 1)
            explanation = explanation.strip()
            if (not cls._critic_sentence(explanation)
                    or cls._critic_missing_expansion(explanation)):
                discarded += 1
                continue
            if rule == "contradicao":
                candidates = ([quoted[:m.start()], quoted[m.end():]]
                              for m in re.finditer(r"\|", quoted))
            else:
                candidates = ([quoted],)
            # Remove optional surrounding quote marks for clean «…» rendering.
            pairs = {'"': '"', "'": "'", '“': '”', '‘': '’', '«': '»', '‹': '›'}
            quotes = None
            for candidate in candidates:
                candidate = [q.strip() for q in candidate]
                for i, quote in enumerate(candidate):
                    if len(quote) >= 2 and pairs.get(quote[0]) == quote[-1]:
                        candidate[i] = quote[1:-1].strip()
                normalised = [cls._normalise_critic_quote(q) for q in candidate]
                if (all(len(q) >= cls._CRITIC_MIN_QUOTE_CHARS
                        and len(re.findall(r"[^\W\d_]+", q, re.UNICODE)) >= cls._CRITIC_MIN_QUOTE_WORDS
                        and q in body for q in normalised)
                        and cls._critic_rule_fits(rule, candidate, body)):
                    quotes = candidate
                    break
            tentative = rule == "contradicao"
            cap = cls._CRITIC_MAX_CONTRADICTIONS if tentative else cls._CRITIC_MAX_PROBLEMS
            if quotes is None or counts[tentative] >= cap:
                discarded += 1
                continue
            valid_line = True
            counts[tentative] += 1
            accepted.append((rule, quotes, explanation))
        return accepted, discarded, malformed or (chatter and not valid_line)

    def _apply_critic(
        self, topic: str, briefing: str, trace: Optional[str] = None, *,
        study_time: Optional[StudyTime] = None,
    ) -> str:
        """Append rule-matched complaints and tentative contradictions.

        Only sigla, absoluta and data complaints lower confidence.
        An Ollama exception leaves the original briefing available for review.
        Empty/unparseable replies instead produce an explicit no-proof section.
        """
        # Neither old critic prose nor the topic heading can prove a complaint.
        body = self._CRITIC_SECTION_RE.sub("", briefing).rstrip()
        critic_input = re.sub(r"^[ \t]*#?[ \t]*Topic:[^\n]*(?:\n|$)", "", body,
                              flags=re.IGNORECASE | re.MULTILINE)[:3000]
        study_time = study_time or StudyTime.for_topic(topic)

        prompt = (
            "You are a SCEPTICAL FACT-CHECKER reviewing a research briefing.\n\n"
            + study_time.prompt_block() + "\n"
            f"Topic: {topic}\n\n"
            "Briefing to review:\n"
            f"---\n{critic_input}\n---\n\n"
            "Report ONLY these four rules, using their exact labels:\n"
            "sigla: An acronym expanded WRONGLY. An unexpanded acronym is NOT a problem; "
            "do not request missing expansions. The quote itself must contain a 2–6 capital-letter "
            "acronym with its expansion: acronym (expansion), acronym = expansion, "
            "acronym stands for expansion, acronym significa expansion, "
            "acronym é a sigla de expansion, or expansion (acronym).\n"
            "absoluta: An unsupported absolute claim. The quote itself must contain one of: "
            + ", ".join(self._CRITIC_ABSOLUTE_WORDS) + ". "
            "Match whole words, ignoring case/accents. Ordinary statements about benefits, "
            "recommendations or incomplete definitions do not qualify. An absolute word "
            "only in your explanation does not qualify.\n"
            "contradicao: Two directly contradictory statements in the briefing. "
            "Copy TWO different, non-overlapping passages; both must exist in the body. "
            "These are suggestions to check; quoted passages alone do not prove a contradiction.\n"
            "data: Stale recency claims or event dates/tense violating these rules: "
            + EVENT_TIME_RULES + "\n"
            "The data quote itself must contain a year from 1900 to 2099, a numeric day/month date, "
            "a full Portuguese/English month name (English may, march and august require "
            "an adjacent day number or year), or one of these event time/tense markers: "
            + ", ".join(self._CRITIC_EVENT_WORDS) + ". "
            "Ordinary auxiliary verbs alone do not qualify.\n"
            "Reply OK if you find no problem under these four checks. Otherwise, "
            f"write at most {self._CRITIC_MAX_PROBLEMS} proven problems (sigla, absoluta, data) "
            f"and at most {self._CRITIC_MAX_CONTRADICTIONS} tentative contradictions, one per line. "
            "Use the matching format (no other rules or extra fields; preserve pipes inside quotes):\n"
            "PROBLEMA | sigla | <exact quote copied from the briefing> | <one-sentence explanation>\n"
            "PROBLEMA | absoluta | <exact quote copied from the briefing> | <one-sentence explanation>\n"
            "PROBLEMA | data | <exact quote copied from the briefing> | <one-sentence explanation>\n"
            "PROBLEMA | contradicao | <first exact quote> | <second exact quote> | <one-sentence explanation>\n"
            "Copy a complete passage from the briefing body, not from the topic, "
            "these instructions or a previous critic. EACH quote must contain at least "
            f"{self._CRITIC_MIN_QUOTE_CHARS} characters and {self._CRITIC_MIN_QUOTE_WORDS} words. "
            "Explain the specific error in plain Portuguese, in one complete sentence "
            "of at least five words ending with punctuation. Rule numbers are not explanations. "
            "Do not invent quotes or report problems you cannot prove with a quote."
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 700},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M2-A7
            )
            reply = (response.get("response") or "").strip()
        except Exception as e:
            log.warning(f"  critic call failed for '{topic}': {e}")
            return briefing

        accepted, discarded, malformed = self._parse_critic_proof(reply, critic_input)
        proven = [item for item in accepted if item[0] != "contradicao"]
        to_check = [item for item in accepted if item[0] == "contradicao"]
        if malformed:
            log.warning("  critic reply empty or unparseable for '%s' "
                        "(proven=%d, a_conferir=%d, discarded=%d)",
                        topic, len(proven), len(to_check), discarded)
        log.info("  critic verdict for '%s': proven=%d, a_conferir=%d, discarded=%d",
                 topic, len(proven), len(to_check), discarded)
        if trace:
            audit.log(trace, "studier", "critic_verdict",
                      payload={"proven": len(proven), "a_conferir": len(to_check),
                               "discarded": discarded})

        if proven:
            verdict = "\n".join(
                f"- ({self._CRITIC_RULES[rule]}) "
                + " × ".join(f"«{q}»" for q in quotes) + f" — {explanation}"
                for rule, quotes, explanation in proven
            )
        else:
            verdict = self._CRITIC_NO_PROOF
            if discarded:
                verdict += f" Descartados: {discarded}."
        if to_check:
            verdict += "\n\n" + self._CRITIC_TO_CHECK + "\n" + "\n".join(
                "- " + " × ".join(f"«{q}»" for q in quotes) + f" — {explanation}"
                for _, quotes, explanation in to_check
            )
        briefing = body + f"\n\n## Crítico\n{verdict}\n"

        # The domain-count floor remains independent; never raise confidence.
        if proven:
            briefing = self._CONFIANCA_LINE_RE.sub(
                lambda mm: (
                    f"{mm.group(1)}baixa — o crítico achou problemas (ver ## Crítico)."
                ),
                briefing,
                count=1,
            )
            log.warning(f"  critic found problems in '{topic}' — confidence forced baixa")

        return briefing

    def _save(self, topic: str, briefing: str, queries: List[str]) -> str:
        slug = self._slugify(topic)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{slug}_{ts}_{uuid4().hex}.md"
        path = os.path.join(self.output_dir, filename)

        header = (
            f"# Topic: {topic}\n\n"
            f"*Generated by CORTEX Nightly Studier on "
            f"{datetime.now(timezone.utc).isoformat()}*\n\n"
            f"**Queries used:**\n"
            + "\n".join(f"- {q}" for q in queries)
            + "\n\n---\n\n"
        )

        entry_id = getattr(self, "_claimed_entry_id", None)
        if entry_id is not None:
            saving = self.queue.transition(
                entry_id, ("studying",), "saving", briefing=path,
            )
            if saving is None:
                raise OSError("Study claim no longer available for saving")

        fd, tmp_path = tempfile.mkstemp(prefix=".study-", suffix=".tmp",
                                       dir=self.output_dir)
        try:
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(header + briefing + "\n")
            os.replace(tmp_path, path)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        return path

    @staticmethod
    def _slugify(topic: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
        return slug[:50] or "untitled"
