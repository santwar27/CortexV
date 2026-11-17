"""
Knowledge base — persistent, semantic memory of APPROVED Nightly Studier
briefings.

Thin wrapper over `Palace`. Each approved Markdown file is split into
SECTION CHUNKS at H2 (`##`) headings and stored as one `documentos` row
per chunk (`tipo='knowledge'`), each with its own embedding. Chunking
keeps a specific section (e.g. '## Passaporte brasileiro') retrievable on
its own instead of being diluted inside one big whole-file embedding —
which was sinking exact-but-specific lookups below `n_results` (s37 bug:
"qual é o número do meu passaporte" failed to retrieve the profile doc
that literally held it). Idempotency by filename: re-indexing a path
deletes all prior chunks for that basename first, never duplicates.

This module deliberately mirrors the previous ChromaDB-backed API so the
Studier approval flow (`src/core/studies_review.py`) requires zero changes:

    index_file(path)            → True/False
    reindex_all()               → int (count indexed)
    remove_file(filename)       → bool
    retrieve_relevant(query, n) → List[str]
    count()                     → int

The legacy `db_path` argument is accepted for call-site compatibility
but is now informational — the Palace lives at `data/palace.db` by
default and stays there regardless. Tests can override via `palace_db`.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from src.memory.embeddings import OllamaEmbedder
from src.memory.palace import Palace

log = logging.getLogger(__name__)

# Retrieval-only synonyms for the user's own data (s84). The profile says "Data
# de nascimento"; the user asks "quando é meu aniversário?". bge-m3 doesn't tie
# the two: for "lembra do meu aniversário do ano passado?" the Identificação
# chunk wasn't in the top 10, the neighbouring "Tradições e datas importantes"
# (with the start of his relationship) came first, and the 8B answered that
# date 5 times in 5. Appending the profile's own wording to the SEARCH — the
# model still sees the question as asked — puts Identificação at #1.
# Keyed by accent-folded word. Shared with src/core/episodic_guard.py, so a
# subject found through a synonym counts as on record there too.
# Only measured gaps go here; a bigger map would be guessing.
RETRIEVAL_SYNONYMS = {
    "aniversario": "data de nascimento",
    "birthday": "data de nascimento",
}


# Two changes to how knowledge is ranked (s84), measured on 14 questions whose
# right chunk is known (11 about the user's profile, 3 about study briefings):
#
#   No frequency channel. It doesn't look at the question, and here every hit
#   returned is marked used, so the profile chunks retrieved most keep
#   climbing. "como funciona o attention mechanism?" had its briefing (cosine
#   0.71) at #5 behind profile chunks at 0.40 — outside the 900-char knowledge
#   budget. The right chunk reached the prompt 12 times in 14; without the
#   channel, 14 in 14. Conversation retrieval keeps it.
#
#   A relevance floor of 0.40. The weakest right chunk measured 0.433 ("onde
#   eu nasci?"); a study briefing about rerankers came into a birthday
#   question at 0.32-0.34. A floor cannot tell a right section from its
#   siblings (those reach 0.67) — it only cuts the off-topic tail. 0.45 would
#   already lose "onde eu nasci?". A hit with no semantic score is kept.
_KNOWLEDGE_FREQ_WEIGHT = 0.0
_KNOWLEDGE_SIM_FLOOR = float(os.getenv("CORTEX_KNOWLEDGE_SIM_FLOOR", "0.40"))

# How much wider the search goes when a study chunk gives up its slot (s88).
# With an Einstein briefing approved, "onde eu nasci?" ranked five Einstein
# chunks in the top 6 and the profile chunk holding the user's birthplace 8th.
_STUDY_BACKFILL_FACTOR = 3


def expand_for_retrieval(query: str) -> str:
    """`query` plus the profile wording for any word it uses differently."""
    import unicodedata
    folded = "".join(c for c in unicodedata.normalize("NFKD", (query or "").lower())
                     if not unicodedata.combining(c))
    words = set(re.findall(r"[a-z]+", folded))
    extra = [syn for word, syn in RETRIEVAL_SYNONYMS.items()
             if word in words and syn not in folded]
    extra = list(dict.fromkeys(extra))
    return f"{query} {' '.join(extra)}" if extra else query


class KnowledgeBase:
    """RAG layer over approved Nightly Studier briefings, backed by Palace."""

    _TOPIC_RE = re.compile(r"^#\s*Topic:\s*(.+)$", re.MULTILINE)
    # Every Nightly Studier briefing opens with "# Topic: …" (studier.py
    # writes it as line one), and _chunk_markdown prefixes each section with
    # that H1, so every chunk of a study starts with it. the user's own profile
    # documents open with "# the user — …". The `source` metadata can't tell the
    # two apart: both are "nightly_studier", because reindex_all indexes the
    # whole approved folder the same way.
    _STUDY_CHUNK_RE = re.compile(r"\s*#\s*Topic:")
    DOC_TIPO = "knowledge"
    DEFAULT_PALACE_DB = "data/palace.db"

    def __init__(
        self,
        db_path: Optional[str] = None,           # legacy, ignored
        approved_dir: str = "data/knowledge/approved",
        palace_db: Optional[str] = None,
        palace: Optional[Palace] = None,
    ):
        self.approved_dir = approved_dir
        os.makedirs(self.approved_dir, exist_ok=True)

        if palace is not None:
            self._palace = palace
            self._owns_palace = False
        else:
            self._palace = Palace(
                db_path=palace_db or self.DEFAULT_PALACE_DB,
                embed_fn=OllamaEmbedder(),
            )
            self._owns_palace = True

    # --- write side -------------------------------------------------

    def index_file(self, path: str) -> bool:
        """Index a Markdown briefing. Idempotent on filename: re-indexing
        the same basename replaces the prior row."""
        if not os.path.isfile(path):
            log.warning(f"KnowledgeBase: file not found: {path}")
            return False
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except OSError as e:
            log.warning(f"KnowledgeBase: cannot read {path}: {e}")
            return False

        filename = os.path.basename(path)
        topic = self._extract_topic(content) or filename

        # Drop any prior row for this filename so re-indexing upserts
        # instead of duplicating. Cheap: only `knowledge` docs scanned.
        self._delete_by_filename(filename)

        # Explicit teaching has one entry per UTC heading. the user's literal
        # text can itself contain Markdown headings or fences; those must
        # never create more chunks (or turn his words into a study).
        from src.core.taught_facts import FILENAME, TITLE, taught_chunks
        taught = filename == FILENAME and content.startswith(TITLE + "\n")
        chunks = taught_chunks(content) if taught else self._chunk_markdown(content)
        if not chunks:
            log.warning(f"KnowledgeBase: nothing to index in {path}")
            return False

        indexed_at = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": "pedro_taught" if taught else "nightly_studier",
                "topic": topic,
                "filename": filename,
                "indexed_at": indexed_at,
                "chunk_index": i,
                "chunk_count": len(chunks),
            }
            embedding = self._safe_embed(chunk)
            self._palace.add_document(
                tipo=self.DOC_TIPO,
                texto=chunk,
                metadata=metadata,
                embedding=embedding,
            )
        log.info(
            f"KnowledgeBase: indexed '{topic}' as {len(chunks)} chunk(s) ({path})"
        )
        return True

    def reindex_all(self) -> int:
        """Re-index every .md under approved_dir. Returns count."""
        if not os.path.isdir(self.approved_dir):
            return 0
        count = 0
        for filename in sorted(os.listdir(self.approved_dir)):
            if filename.endswith(".md"):
                if self.index_file(os.path.join(self.approved_dir, filename)):
                    count += 1
        return count

    def remove_file(self, filename: str) -> bool:
        """Drop a previously indexed briefing by its basename."""
        try:
            return self._delete_by_filename(filename) > 0
        except Exception as e:
            log.warning(f"KnowledgeBase: delete failed for {filename}: {e}")
            return False

    # --- read side --------------------------------------------------

    def is_fully_indexed(self, filename: str) -> bool:
        """Whether all declared chunks of this filename are in the Palace."""
        chunks = []
        for doc in self._palace.list_documents(tipo=self.DOC_TIPO, limit=-1):
            meta = doc.get("metadata") or {}
            if meta.get("filename") == filename:
                chunks.append(meta)
        if not chunks:
            return False
        count = chunks[0].get("chunk_count")
        if type(count) is not int or count < 1:
            return False
        indices = set()
        for meta in chunks:
            index = meta.get("chunk_index")
            if (type(meta.get("chunk_count")) is not int
                    or meta["chunk_count"] != count
                    or type(index) is not int or not 0 <= index < count):
                return False
            indices.add(index)
        return len(indices) == count

    def retrieve_relevant(self, query: str, n_results: int = 3) -> List[str]:
        """Return up to n_results briefing snippets ranked by hybrid search.

        A chunk of an approved study enters only when the question names the
        study's subject (`study_named_in`). A slot it gives up is filled from
        a wider search, so the profile chunk that ranked just below gets in.
        With no study turned away, this is the single search it always was."""
        try:
            hits = self._search(query, n_results)
            kept = [h for h in hits if self._may_enter(h["texto"], query)]
            if len(kept) < len(hits):
                seen = {h["id"] for h in hits}
                for h in self._search(query, n_results * _STUDY_BACKFILL_FACTOR):
                    if len(kept) >= n_results:
                        break
                    if h["id"] not in seen and self._may_enter(h["texto"], query):
                        kept.append(h)
                    seen.add(h["id"])
            # No dedup downstream — every hit kept here is injected, so all
            # of those count as used (M3-A4).
            self._palace.mark_used([h["id"] for h in kept])
        except Exception as e:
            log.warning(f"KnowledgeBase: query failed: {e}")
            return []
        return [h["texto"] for h in kept]

    def _search(self, query: str, k: int) -> list:
        hits = self._palace.search(expand_for_retrieval(query), k=k,
                                   tipo=self.DOC_TIPO,
                                   freq_weight=_KNOWLEDGE_FREQ_WEIGHT)
        return [h for h in hits
                if h.get("_sem_sim") is None or h["_sem_sim"] >= _KNOWLEDGE_SIM_FLOOR]

    @classmethod
    def _may_enter(cls, chunk: str, query: str) -> bool:
        return not cls.is_study_chunk(chunk) or cls.study_named_in(chunk, query)

    @classmethod
    def study_named_in(cls, chunk: str, query: str) -> bool:
        """Does `query` name the subject of this study chunk — a word of its
        "# Topic:" line?

        Why (s88, Codex lote 14 — invariant 3): a study is research about its
        subject, never about Pedro, yet it ranked high for questions about
        him and the 8B took its facts for his. Measured with the real
        pipeline: "qual minha profissão?" → "Sua profissão é arquiteta." (the
        studied person's job) 3 times in 3. Tagging and fencing the study,
        with or without a prompt rule, only moved the mistake: "Você nasceu
        na Polônia.", 2 times in 6. Kept out of questions that don't name
        it, it can't be read as his. Known limit: a generic topic word
        ("carreira" in "X: biografia e carreira") still lets it in."""
        from src.core.episodic_guard import content_stems, subject_in
        m = cls._TOPIC_RE.search(chunk or "")
        stems = content_stems(m.group(1)) if m else []
        return bool(stems) and subject_in(stems, query)

    def count(self) -> int:
        """Number of distinct briefings (files), NOT chunks — since one file
        now indexes as several section rows, counting rows would overcount.
        Counts distinct filenames in SQL (NULL metadata is ignored by
        COUNT(DISTINCT)), so it stays O(1)-ish instead of loading every row."""
        try:
            return self._palace.count_distinct_meta(self.DOC_TIPO, "filename")
        except Exception:
            return 0

    # --- helpers ----------------------------------------------------

    def _delete_by_filename(self, filename: str) -> int:
        """Delete all knowledge docs whose metadata.filename matches.
        Returns rows deleted. Small N — linear scan is fine."""
        deleted = 0
        for doc in self._palace.list_documents(tipo=self.DOC_TIPO, limit=10_000):
            meta = doc.get("metadata") or {}
            if meta.get("filename") == filename:
                if self._palace.delete_document(doc["id"]):
                    deleted += 1
        return deleted

    def _safe_embed(self, text: str):
        """Compute embedding via Palace's embed_fn; return None on failure
        so a transient Ollama outage doesn't drop the write."""
        fn = self._palace.embed_fn
        if fn is None:
            return None
        try:
            return fn(text)
        except Exception as e:
            log.warning(f"KnowledgeBase: embedding failed ({e}); storing without vector")
            return None

    @classmethod
    def is_study_chunk(cls, chunk: str) -> bool:
        """True for a chunk of an approved web study, False for Pedro's own
        profile data. Read from the chunk itself, so rows indexed before
        this existed need no reindex."""
        return bool(cls._STUDY_CHUNK_RE.match(chunk or ""))

    @classmethod
    def _extract_topic(cls, content: str) -> str:
        m = cls._TOPIC_RE.search(content)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _chunk_markdown(content: str) -> List[str]:
        """Split a briefing into section chunks at H2 (`##`) headings.

        Each section chunk is prefixed with the document's H1 title (if any)
        so a section stays disambiguated ACROSS files — '## Passaporte'
        appears in both the Brazilian and Italian profiles, and the H1
        ('# Pedro — Perfil Brasileiro') is what lets a 'passaporte
        brasileiro' query pick the right one. The preamble (text before the
        first `##`) already carries the H1, so it isn't double-prefixed. A
        document with no H2 headings stays a single whole-file chunk, so
        short studier briefings behave exactly as before.
        """
        if not content or not content.strip():
            return []
        lines = content.splitlines()

        # Single pass: capture the first real H1 and split at H2 boundaries,
        # while IGNORING '#'/'##' lines inside fenced code blocks (``` or ~~~)
        # — a '## step' in a bash snippet is not a section heading.
        h1 = ""
        sections: List[List[str]] = []
        current: List[str] = []
        in_fence = False
        fence_marker = ""
        for ln in lines:
            stripped = ln.lstrip()

            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence, fence_marker = True, marker
                elif stripped.startswith(fence_marker):
                    in_fence, fence_marker = False, ""
                current.append(ln)
                continue

            if not in_fence and stripped.startswith("## "):
                if current:
                    sections.append(current)
                current = [ln]
                continue

            if (not in_fence and not h1
                    and stripped.startswith("# ")
                    and not stripped.startswith("## ")):
                h1 = ln.strip()
            current.append(ln)

        if current:
            sections.append(current)

        # No H2 → one chunk (whole file), preserving legacy behaviour.
        if len(sections) <= 1:
            return [content.strip()]

        chunks: List[str] = []
        for sec in sections:
            text = "\n".join(sec).strip()
            if not text:
                continue
            if h1 and not text.startswith(h1):
                text = f"{h1}\n{text}"
            chunks.append(text)
        return chunks or [content.strip()]
