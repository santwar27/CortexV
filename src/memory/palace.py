"""
Palace — structured + semantic memory store on a single SQLite file.

Three tables form three memory natures:

  triplas    — atomic facts ⟨sujeito, predicado, objeto⟩ with bitemporal
               lite validity (valid_from / valid_until). Re-asserting a
               fact does NOT overwrite: the prior row is closed (valid_until
               = now) and a new row is inserted. Enables "what did I think
               about X last March?" queries even when the current answer
               differs.

  documentos — narrative chunks (conversation turns, knowledge briefings,
               notes). Optional embedding BLOB (float32 numpy bytes) lets
               cosine search run against the local nomic-embed-text model.
               Mirrored by an FTS5 virtual table for BM25 lexical search.

  eventos    — append-only event log (gate decisions, tool calls, audit
               trail). Holds JSON payloads, indexed by timestamp and tipo.

`search()` runs hybrid retrieval: semantic candidates + FTS5 candidates
combined with Reciprocal Rank Fusion (RRF). Weight `alpha` controls the
semantic/lexical mix (default 0.6 semantic).

Reference: design-reference ch. 6 + Bônus Palace. This implementation is Cortex-
original, single-DB (vs the reference's 16-file split), and standalone — it does
NOT replace CortexMemory in this commit. Integration happens in the next
session after the schema has been reviewed under real usage.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.memory.temporal import (
    TemporalIntent,
    detect_temporal_intent,
    recency_weight,
)


log = logging.getLogger(__name__)


# Float32 dtype keeps cosine math fast and the BLOB compact
# (768 dims * 4 bytes = 3 KB per doc).
_EMB_DTYPE = np.float32


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_embedding(emb: np.ndarray) -> bytes:
    return np.ascontiguousarray(emb, dtype=_EMB_DTYPE).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=_EMB_DTYPE)


# FTS5 reserves these characters as operators; we tokenise to \w+ runs so a
# raw user query can't trip a syntax error.
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _fts_query(text: str) -> str:
    """Build an FTS5 MATCH expression — terms OR-combined, NOT the FTS5
    default implicit AND.

    Why OR: a recall question like "qual meu CPF" must still lexically match
    a chunk that holds "CPF" but not the stopwords "qual"/"meu". Implicit AND
    required ALL tokens, so such queries matched NOTHING — sinking
    exact-but-stopword-wrapped field lookups below retrieval (s39 bug: CPF /
    RG / CEP recall failed; the field's section sat at rank 7-13). BM25 still
    ranks by relevance, so the rare discriminative term ("CPF") dominates and
    common words contribute ~nothing; the hybrid's semantic channel + RRF
    keep precision despite the wider lexical candidate pool."""
    tokens = _FTS_TOKEN_RE.findall(text or "")
    return " OR ".join(tokens) if tokens else ""


def fact_write_allowed(
    existing: Optional[Dict[str, Any]],
    objeto: str,
    source: str,
    confidence: float,
) -> bool:
    """No-downgrade guard for every fact persister (M3-A1, review Fable 5).

    `add_fact` closes the active triple unconditionally, so each writer
    must decide BEFORE calling it. The rule existed in one path only (the
    passive consolidator); the nightly persisters skipped just the
    identical-objeto case, letting a 0.55 studier inference close a fact
    Pedro stated himself. One helper, three callers, one rule:

      - no existing active triple        → write
      - same objeto                      → skip (no-op churn would roll
                                           the bitemporal log)
      - existing is user_explicit        → only another user_explicit
                                           write may close it
      - existing has HIGHER confidence   → skip (equal confidence may
                                           overwrite: newer wins ties,
                                           matching the passive guard)
    """
    if existing is None:
        return True
    if existing.get("objeto") == objeto:
        return False
    if existing.get("source") == "user_explicit" and source != "user_explicit":
        return False
    if float(existing.get("confidence", 0.0)) > float(confidence):
        return False
    return True


class Palace:
    """SQLite-backed multi-modal memory. One file, three table families."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        db_path: str = "data/palace.db",
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        read_only: bool = False,
    ):
        """
        db_path:  ':memory:' for tests, or a file path. Parent dir is
                  created if missing.
        embed_fn: callable(text) -> np.ndarray. If None, add_document_with_
                  embedding() raises and search() falls back to FTS5-only.
        read_only: open via SQLite URI `mode=ro` and skip _init_schema —
                  for auditors (memory_doctor) that must never mutate the
                  patient, not even the no-op CREATE/INSERT of a healthy
                  init. A missing DB raises instead of being created.
                  Not supported for ':memory:' (nothing to audit there).
        """
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.read_only = read_only
        # Threads sharing a connection also share its transaction.
        self._write_lock = threading.RLock()

        if read_only:
            if db_path == ":memory:":
                raise ValueError("read_only Palace needs a file, not ':memory:'")
            self._conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            # No journal_mode/foreign_keys pragmas: journal_mode would try to
            # WRITE the header on a non-WAL file, and FK enforcement only
            # matters for writes this connection can't do anyway.
            return

        if db_path != ":memory:":
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # check_same_thread=False because requests genuinely cross threads
        # (/ask on the event loop, /ask_stream in worker threads) and 3+
        # Palace instances share this file (engine + studier). WAL keeps
        # readers and one writer concurrent; busy_timeout makes a moment
        # of lock contention a short wait instead of a dry
        # "database is locked" OperationalError (M3-A3). Python's connect()
        # already defaults to timeout=5.0 (same 5000ms) — the explicit
        # PRAGMA pins the contract instead of leaning on that default.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._init_schema()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS triplas (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                sujeito      TEXT    NOT NULL,
                predicado    TEXT    NOT NULL,
                objeto       TEXT    NOT NULL,
                source       TEXT    NOT NULL DEFAULT 'user_explicit',
                confidence   REAL    NOT NULL DEFAULT 1.0,
                valid_from   TEXT    NOT NULL,
                valid_until  TEXT,
                created_at   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_triplas_sp
                ON triplas(sujeito, predicado);
            CREATE INDEX IF NOT EXISTS idx_triplas_active
                ON triplas(sujeito, predicado) WHERE valid_until IS NULL;

            CREATE TABLE IF NOT EXISTS documentos (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo         TEXT    NOT NULL,
                texto        TEXT    NOT NULL,
                embedding    BLOB,
                metadata     TEXT    NOT NULL DEFAULT '{}',
                created_at   TEXT    NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_documentos_tipo
                ON documentos(tipo);

            CREATE VIRTUAL TABLE IF NOT EXISTS documentos_fts
                USING fts5(
                    texto,
                    content='documentos',
                    content_rowid='id',
                    tokenize='unicode61'
                );

            CREATE TRIGGER IF NOT EXISTS documentos_ai
                AFTER INSERT ON documentos
                BEGIN
                    INSERT INTO documentos_fts(rowid, texto)
                        VALUES (new.id, new.texto);
                END;

            CREATE TRIGGER IF NOT EXISTS documentos_ad
                AFTER DELETE ON documentos
                BEGIN
                    INSERT INTO documentos_fts(documentos_fts, rowid, texto)
                        VALUES ('delete', old.id, old.texto);
                END;

            CREATE TRIGGER IF NOT EXISTS documentos_au
                AFTER UPDATE ON documentos
                BEGIN
                    INSERT INTO documentos_fts(documentos_fts, rowid, texto)
                        VALUES ('delete', old.id, old.texto);
                    INSERT INTO documentos_fts(rowid, texto)
                        VALUES (new.id, new.texto);
                END;

            CREATE TABLE IF NOT EXISTS eventos (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                tipo      TEXT    NOT NULL,
                payload   TEXT    NOT NULL DEFAULT '{}',
                trace_id  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_eventos_ts ON eventos(timestamp);
            CREATE INDEX IF NOT EXISTS idx_eventos_tipo ON eventos(tipo);
            CREATE INDEX IF NOT EXISTS idx_eventos_trace ON eventos(trace_id);

            CREATE TABLE IF NOT EXISTS skills (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT    NOT NULL UNIQUE,
                trigger_regex    TEXT    NOT NULL,
                handler_type     TEXT    NOT NULL,
                handler_payload  TEXT    NOT NULL DEFAULT '{}',
                enabled          INTEGER NOT NULL DEFAULT 1,
                invocation_count INTEGER NOT NULL DEFAULT 0,
                last_used_at     TEXT,
                created_at       TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_enabled ON skills(enabled);

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            """
        )
        try:
            c.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_triplas_active "
                "ON triplas(sujeito, predicado) WHERE valid_until IS NULL"
            )
        except sqlite3.IntegrityError:
            # Preserve legacy facts; resolving duplicates needs a human decision.
            duplicates = c.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT 1 FROM triplas WHERE valid_until IS NULL "
                "GROUP BY sujeito, predicado HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            log.warning(
                "Could not create uq_triplas_active: %d duplicate active fact pairs; "
                "existing rows left unchanged", duplicates,
            )
        self._migrate_legacy_schema()
        c.execute(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )
        c.commit()

    def _migrate_legacy_schema(self) -> None:
        """Upgrade older palace.db files in place. Idempotent: each step is
        guarded so re-running on a current-version DB is a no-op."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(documentos)").fetchall()
        }
        if "access_count" not in cols:
            self._conn.execute(
                "ALTER TABLE documentos ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
            )

    # ------------------------------------------------------------------
    # facts (triplas) — bitemporal lite
    # ------------------------------------------------------------------

    def add_fact(
        self,
        sujeito: str,
        predicado: str,
        objeto: str,
        source: str = "user_explicit",
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
    ) -> int:
        """
        Assert (sujeito, predicado) → objeto. If a prior active row exists
        for the same (sujeito, predicado), it is closed (valid_until = now)
        before the new row is inserted. Returns the new row id.
        """
        if not sujeito or not predicado:
            raise ValueError("add_fact: sujeito and predicado are required")
        now = _now_iso()
        vf = valid_from or now
        with self._write_lock, self._conn:
            self._conn.execute(
                """
                UPDATE triplas SET valid_until = ?
                 WHERE sujeito = ? AND predicado = ? AND valid_until IS NULL
                """,
                (now, sujeito, predicado),
            )
            cur = self._conn.execute(
                """
                INSERT INTO triplas
                    (sujeito, predicado, objeto, source, confidence,
                     valid_from, valid_until, created_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (sujeito, predicado, objeto, source, float(confidence),
                 vf, now),
            )
        return cur.lastrowid

    def get_fact(
        self,
        sujeito: str,
        predicado: str,
        at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the active triple for (sujeito, predicado), or the one valid
        at the ISO timestamp `at`. None if no match.
        """
        if at is None:
            row = self._conn.execute(
                """
                SELECT * FROM triplas
                 WHERE sujeito = ? AND predicado = ? AND valid_until IS NULL
                 ORDER BY id DESC LIMIT 1
                """,
                (sujeito, predicado),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT * FROM triplas
                 WHERE sujeito = ? AND predicado = ?
                   AND valid_from <= ?
                   AND (valid_until IS NULL OR valid_until > ?)
                 ORDER BY valid_from DESC LIMIT 1
                """,
                (sujeito, predicado, at, at),
            ).fetchone()
        return dict(row) if row else None

    def list_facts(
        self,
        sujeito: Optional[str] = None,
        include_invalidated: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list = []
        if sujeito is not None:
            clauses.append("sujeito = ?")
            params.append(sujeito)
        if not include_invalidated:
            clauses.append("valid_until IS NULL")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM triplas{where} ORDER BY id DESC", params
        ).fetchall()
        return [dict(r) for r in rows]

    def get_preferences(
        self,
        sujeito: str = "user",
        min_confidence: float = 0.0,
        include_invalidated: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return active preference triples (`predicado LIKE 'prefere_%'`)
        for `sujeito`, optionally filtered by confidence floor.

        Convenience wrapper over the triplas table — preferences are just
        triples with a naming convention. Used by the orchestrator to
        inject Pedro's long-running habits into the system prompt, and by
        the Studier's nightly extractor to avoid re-asserting unchanged
        rows."""
        clauses = ["sujeito = ?", "predicado LIKE 'prefere_%'"]
        params: list = [sujeito]
        if not include_invalidated:
            clauses.append("valid_until IS NULL")
        if min_confidence > 0.0:
            clauses.append("confidence >= ?")
            params.append(float(min_confidence))
        where = " WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM triplas{where} ORDER BY predicado ASC, id DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def invalidate_fact(
        self,
        sujeito: str,
        predicado: str,
        valid_until: Optional[str] = None,
    ) -> int:
        """Close the active triple without inserting a new one. Returns rows updated."""
        ts = valid_until or _now_iso()
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE triplas SET valid_until = ?
                 WHERE sujeito = ? AND predicado = ? AND valid_until IS NULL
                """,
                (ts, sujeito, predicado),
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # documents (documentos + FTS5)
    # ------------------------------------------------------------------

    def add_document(
        self,
        tipo: str,
        texto: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a document. embedding is stored as float32 BLOB if given.

        `created_at` lets callers preserve a historical timestamp — needed by
        the migration script (which copies ChromaDB rows with their original
        write times) and by tests that need deterministic ages for the
        temporal decay code. When omitted, the current UTC time is stamped.
        """
        if not tipo:
            raise ValueError("add_document: tipo is required")
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("add_document: texto must be a non-empty string")
        blob = _serialize_embedding(embedding) if embedding is not None else None
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        ts = created_at or _now_iso()
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO documentos (tipo, texto, embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tipo, texto, blob, meta_json, ts),
            )
        return cur.lastrowid

    def add_document_with_embedding(
        self,
        tipo: str,
        texto: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Convenience: compute the embedding via self.embed_fn and insert."""
        if self.embed_fn is None:
            raise RuntimeError(
                "Palace.add_document_with_embedding requires embed_fn at construction"
            )
        emb = self.embed_fn(texto)
        return self.add_document(tipo, texto, metadata, embedding=emb)

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM documentos WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_doc(row)

    def list_documents(
        self,
        tipo: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if tipo is None:
            rows = self._conn.execute(
                "SELECT * FROM documentos ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documentos WHERE tipo = ? ORDER BY id DESC LIMIT ?",
                (tipo, limit),
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def recent_texts(self, tipo: str, since_iso: str,
                     exclude_meta: Optional[Tuple[str, str]] = None) -> List[str]:
        """Texts of docs of `tipo` created at/after `since_iso`, newest
        first. Facade for consumers that used to query _conn directly
        (M3-A9) — keeps every SQL touchpoint inside the Palace.
        `exclude_meta=(key, value)` leaves out the docs whose metadata.<key>
        is that value; a doc without the key stays."""
        sql = "SELECT texto FROM documentos WHERE tipo = ? AND created_at >= ?"
        params: list = [tipo, since_iso]
        if exclude_meta is not None:
            key, value = exclude_meta
            sql += " AND json_extract(metadata, ?) IS NOT ?"
            params += [f"$.{key}", value]
        rows = self._conn.execute(
            sql + " ORDER BY created_at DESC", params).fetchall()
        return [r["texto"] or "" for r in rows]

    def count_distinct_meta(self, tipo: str, key: str) -> int:
        """Count distinct values of metadata.<key> among docs of `tipo`.
        NULL metadata/key is ignored by COUNT(DISTINCT). Facade (M3-A9)."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT json_extract(metadata, ?)) "
            "FROM documentos WHERE tipo = ?",
            (f"$.{key}", tipo),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def stale_docs(self, tipo: str, before_iso: str) -> List[Dict[str, Any]]:
        """Docs of `tipo` created BEFORE `before_iso` that were never
        injected into a prompt (access_count = 0) — the retention job's
        candidate set (M3-A5, policy approved s72). Oldest first, so an
        interrupted archive run resumes deterministically."""
        rows = self._conn.execute(
            """
            SELECT * FROM documentos
             WHERE tipo = ? AND access_count = 0 AND created_at < ?
             ORDER BY created_at ASC
            """,
            (tipo, before_iso),
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM documentos WHERE id = ?", (doc_id,)
            )
        return cur.rowcount > 0

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except json.JSONDecodeError:
            d["metadata"] = {}
        # Don't leak raw bytes to callers — expose `has_embedding` instead.
        d["has_embedding"] = d.get("embedding") is not None
        d.pop("embedding", None)
        return d

    # ------------------------------------------------------------------
    # hybrid search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.6,
        freq_weight: float = 0.3,
        tipo: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with temporal awareness. Three ranked candidate
        lists are combined via Reciprocal Rank Fusion:

          1. semantic — cosine similarity, weighted by age through
                        `temporal.recency_weight` — a tiebreak, at most a
                        RECENCY_WEIGHT share of the score (skipped when the
                        query implies "sempre / prefiro / always").
          2. lexical  — BM25 via FTS5.
          3. frequency— rank by access_count desc; docs never injected
                        before are excluded so the signal stays honest.

        Search itself never bumps access_count (M3-A4): the caller decides
        which hits were actually INJECTED into a prompt (after relevance
        floor / echo dedup) and reports them via `mark_used(ids)`. Bumping
        here would also count hits the caller later discards, feeding a
        rich-get-richer loop in the frequency channel.

        Temporal cues in the query ("ontem", "semana passada", "last
        month", etc.) trigger a hard date window that filters all three
        channels before ranking. Cues like "sempre / prefiro" disable
        the decay modulator on the semantic side but keep the channels.

        Parameters:
          k            number of final results to return
          alpha        0.0 = lexical only, 1.0 = semantic only, 0.6 default
          freq_weight  contribution of the frequency channel to RRF
                       (0.0 disables it; default 0.3 = moderate boost)
          tipo         restrict to a single document tipo, or None for all
          now          reference time for decay (default: current UTC).
                       Tests pass an explicit datetime for deterministic
                       behavior; production leaves None.
        """
        pool = self.candidate_pool(query, k=k, alpha=alpha,
                                   freq_weight=freq_weight, tipo=tipo, now=now)
        if not pool:
            return []
        head = pool[:k]
        top_ids = [c["id"] for c in head]
        placeholders = ",".join("?" * len(top_ids))
        rows = self._conn.execute(
            f"SELECT * FROM documentos WHERE id IN ({placeholders})", top_ids
        ).fetchall()
        by_id = {r["id"]: self._row_to_doc(r) for r in rows}
        results: List[Dict[str, Any]] = []
        for c in head:
            doc = by_id.get(c["id"])
            if doc is not None:
                doc["_score"] = c["score"]
                doc["_sem_sim"] = c["sem_sim"]  # raw cosine, or None
                results.append(doc)

        return results

    def candidate_pool(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.6,
        freq_weight: float = 0.3,
        tipo: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Every candidate `search()` fused, in RRF order — `search()` is the
        head of this list. Each entry: id, score, sem_sim (raw cosine or
        None), channels (which of semantic/FTS/frequency surfaced it).

        Public so audit tools (scripts/palace_retrieval_probe.py) read the
        same pool production ranks, instead of rebuilding the fusion and
        risking a copy that drifts.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        intent = detect_temporal_intent(query)
        # The cue word ("ontem", "sempre") is metadata, not content. Strip
        # it before handing the text to the semantic and lexical channels
        # so BM25 does not AND the cue against every document.
        content_query = intent.effective_query(query)
        ref_now = now if now is not None else datetime.now(timezone.utc)
        cutoff_iso: Optional[str] = None
        if intent.window_days is not None:
            cutoff_iso = (ref_now - timedelta(days=intent.window_days)).isoformat()

        # Pull more candidates than k from each side so the RRF mix has
        # room to swap items in. 3*k is the standard rule of thumb.
        cand_k = max(k * 3, k)

        sem_hits, raw_sims = (
            self._semantic_candidates(content_query, cand_k, tipo, intent, ref_now, cutoff_iso)
            if (self.embed_fn is not None and alpha > 0.0)
            else ([], {})
        )
        fts_hits = (
            self._fts_candidates(content_query, cand_k, tipo, cutoff_iso)
            if alpha < 1.0 else []
        )
        freq_hits = (
            self._frequency_candidates(cand_k, tipo, cutoff_iso)
            if freq_weight > 0.0 else []
        )

        if not sem_hits and not fts_hits and not freq_hits:
            return []

        rrf_k = 60  # standard RRF constant from Cormack et al.
        scores: Dict[int, float] = {}
        channels: Dict[int, str] = {}
        for rank, (doc_id, _sim) in enumerate(sem_hits):
            scores[doc_id] = scores.get(doc_id, 0.0) + alpha / (rrf_k + rank + 1)
            channels[doc_id] = channels.get(doc_id, "") + "S"
        for rank, doc_id in enumerate(fts_hits):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) / (rrf_k + rank + 1)
            channels[doc_id] = channels.get(doc_id, "") + "F"
        for rank, doc_id in enumerate(freq_hits):
            scores[doc_id] = scores.get(doc_id, 0.0) + freq_weight / (rrf_k + rank + 1)
            channels[doc_id] = channels.get(doc_id, "") + "Q"

        # The raw cosine travels with EVERY candidate that has a vector, not
        # only the semantic top-k (s83). Before, a doc that surfaced via FTS
        # or frequency had no score, and the caller's relevance floor kept it
        # by design — so the frequency channel, which does not look at the
        # query at all, injected the same old turns into unrelated questions
        # (one music request into 38 of them). A doc with no usable vector
        # still comes back as None, and the floor still keeps it.
        ordered = sorted(scores, key=lambda i: -scores[i])
        return [
            {"id": i, "score": scores[i], "sem_sim": raw_sims.get(i),
             "channels": channels[i]}
            for i in ordered
        ]

    def mark_used(self, ids: List[int]) -> None:
        """Bump access_count for docs that were actually INJECTED into a
        prompt. Callers invoke this after their own filtering (relevance
        floor, echo dedup) so the frequency channel learns from real usage,
        not from raw retrieval (M3-A4). Empty list is a no-op."""
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        with self._write_lock, self._conn:
            self._conn.execute(
                f"UPDATE documentos SET access_count = access_count + 1 "
                f"WHERE id IN ({ph})",
                list(ids),
            )

    def _semantic_candidates(
        self,
        query: str,
        k: int,
        tipo: Optional[str],
        intent: TemporalIntent,
        now: datetime,
        cutoff_iso: Optional[str],
    ) -> Tuple[List[Tuple[int, float]], Dict[int, float]]:
        """Cosine similarity over embedded docs, modulated by tipo-specific
        decay unless the intent disables it. Honors window cutoff and tipo
        filter. Returns (top-k as (id, raw_cosine) pairs, {id: raw_cosine} for
        every doc scored). The top-k is ranked by the recency-weighted
        similarity; both carry the RAW cosine so a caller's relevance floor
        measures topical relevance, not age."""
        try:
            q_emb = self.embed_fn(query)
        except Exception:
            return [], {}
        q = np.asarray(q_emb, dtype=_EMB_DTYPE)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return [], {}
        q_unit = q / q_norm

        clauses = ["embedding IS NOT NULL"]
        params: list = []
        if tipo is not None:
            clauses.append("tipo = ?")
            params.append(tipo)
        if cutoff_iso is not None:
            clauses.append("created_at >= ?")
            params.append(cutoff_iso)
        where = " AND ".join(clauses)
        cur = self._conn.execute(
            f"SELECT id, tipo, created_at, embedding FROM documentos WHERE {where}",
            params,
        )

        scored: List = []
        for row in cur.fetchall():
            emb = _deserialize_embedding(row["embedding"])
            if emb.shape != q_unit.shape:
                continue  # mixed dims — skip rather than crash
            n = float(np.linalg.norm(emb))
            if n < 1e-9:
                continue
            raw_sim = float(np.dot(q_unit, emb / n))
            ranked_sim = raw_sim
            if not intent.decay_off:
                ranked_sim *= recency_weight(row["tipo"], row["created_at"], now)
            scored.append((ranked_sim, raw_sim, row["id"]))
        raw_all = {doc_id: raw_sim for _, raw_sim, doc_id in scored}
        scored.sort(key=lambda t: -t[0])
        return [(doc_id, raw_sim) for _, raw_sim, doc_id in scored[:k]], raw_all

    def _fts_candidates(
        self,
        query: str,
        k: int,
        tipo: Optional[str],
        cutoff_iso: Optional[str] = None,
    ) -> List[int]:
        """BM25 ranking via FTS5 with optional tipo + window cutoff. Returns top-k ids."""
        q = _fts_query(query)
        if not q:
            return []
        clauses = ["documentos_fts MATCH ?"]
        params: list = [q]
        if tipo is not None:
            clauses.append("d.tipo = ?")
            params.append(tipo)
        if cutoff_iso is not None:
            clauses.append("d.created_at >= ?")
            params.append(cutoff_iso)
        where = " AND ".join(clauses)
        params.append(k)
        try:
            rows = self._conn.execute(
                f"""
                SELECT d.id FROM documentos d
                  JOIN documentos_fts fts ON fts.rowid = d.id
                 WHERE {where}
                 ORDER BY rank LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query slipped past _fts_query — give up gracefully.
            return []
        return [r["id"] for r in rows]

    def _frequency_candidates(
        self,
        k: int,
        tipo: Optional[str] = None,
        cutoff_iso: Optional[str] = None,
    ) -> List[int]:
        """Top-k docs by access_count descending. Docs that have never been
        injected (access_count = 0 — see `mark_used`) are excluded — without
        prior real usage, the frequency signal would be noise. Honors tipo +
        window filters."""
        clauses = ["access_count > 0"]
        params: list = []
        if tipo is not None:
            clauses.append("tipo = ?")
            params.append(tipo)
        if cutoff_iso is not None:
            clauses.append("created_at >= ?")
            params.append(cutoff_iso)
        where = " AND ".join(clauses)
        params.append(k)
        rows = self._conn.execute(
            f"""
            SELECT id FROM documentos
             WHERE {where}
             ORDER BY access_count DESC, id DESC
             LIMIT ?
            """,
            params,
        ).fetchall()
        return [r["id"] for r in rows]

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def log_event(
        self,
        tipo: str,
        payload: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        if not tipo:
            raise ValueError("log_event: tipo is required")
        ts = timestamp or _now_iso()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO eventos (timestamp, tipo, payload, trace_id) VALUES (?, ?, ?, ?)",
                (ts, tipo, payload_json, trace_id),
            )
        return cur.lastrowid

    def list_events(
        self,
        tipo: Optional[str] = None,
        trace_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list = []
        if tipo is not None:
            clauses.append("tipo = ?")
            params.append(tipo)
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM eventos{where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.get("payload") or "{}")
            except json.JSONDecodeError:
                d["payload"] = {}
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # generic SQL facade (M3-A9 follow-up, s72)
    # ------------------------------------------------------------------
    # For modules that own their OWN table inside the palace file (today:
    # skill_registry's `skills`). Domain-specific facade methods
    # (recent_texts, count_distinct_meta) fit Palace-owned tables; for a
    # foreign domain table a method per query would leak that domain in
    # here. The generic pair keeps the actual contract — the Palace owns
    # the connection, its lock semantics and the transaction boundary —
    # without anyone touching `_conn` directly.

    def execute_read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Run one read statement and return all rows."""
        return self._conn.execute(sql, params).fetchall()

    def execute_write(self, sql: str, params: tuple = ()) -> tuple:
        """Run one write statement in its own transaction. Returns
        (rowcount, lastrowid). Raises on read_only connections (SQLite
        enforces mode=ro at its own layer)."""
        with self._write_lock, self._conn:
            cur = self._conn.execute(sql, params)
        return cur.rowcount, cur.lastrowid

    # ------------------------------------------------------------------
    # admin
    # ------------------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        return {
            "triplas": self._conn.execute(
                "SELECT COUNT(*) FROM triplas"
            ).fetchone()[0],
            "triplas_active": self._conn.execute(
                "SELECT COUNT(*) FROM triplas WHERE valid_until IS NULL"
            ).fetchone()[0],
            "documentos": self._conn.execute(
                "SELECT COUNT(*) FROM documentos"
            ).fetchone()[0],
            "documentos_with_embedding": self._conn.execute(
                "SELECT COUNT(*) FROM documentos WHERE embedding IS NOT NULL"
            ).fetchone()[0],
            "eventos": self._conn.execute(
                "SELECT COUNT(*) FROM eventos"
            ).fetchone()[0],
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
