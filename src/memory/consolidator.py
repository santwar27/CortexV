"""
Nightly memory consolidator — Tier 3.5.

Extends the Studier's nightly pass with two operations that sit
naturally on top of the Tier 2 stack:

  1. Summarise the day. Concatenated conversation turns (via
     `preferences.recent_conversations`) become a single digest of
     ~400-600 words. The digest is persisted as a document
     `tipo='daily_summary'` so retrieval and future consolidation
     passes can search across days without re-reading raw turns.

  2. Extract structured facts. Distinct from preferences: facts are
     stable statements about the user or world ("vive_em → Londres",
     "trabalha_em → empresa_X"), not stylistic preferences. They land
     as bitemporal triples via `Palace.add_fact` with
     `source='studier_inferred'`.

The third operation design-reference ch. 0 lists for the consolidator — updating
the preference engine — was already shipped in `e4069ea` as
`studier._consolidate_preferences`. The new `_consolidate_memory` hook
runs BEFORE preferences so the day's summary is available if the
preference pipeline ever wants to reuse it (out of scope for this
ticket).

Design mirrors `preferences.py`: pure logic, no SQLite touch, LLM
client injected for testability. Sanitisation is applied at write
time — `_coerce_fact_triple` rejects rows whose predicado would fail
slug validation or whose objeto contains control characters / reserved
all-caps bracket tags. This is the same defense-in-depth posture from
the preferences read-path sanitisation (`_safe_to_inject`); fact
storage gets it at the write step so the triple table never holds a
row that's later unsafe to surface.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FactTriple:
    """One extracted fact candidate.

    sujeito    — entity the statement is about. Usually "user" but
                 could be "cortex" or any named entity from the corpus.
    predicado  — verb/relation slug (lowercase, no prefix requirement
                 — preferences use `prefere_*`, facts can be anything).
                 Examples: `vive_em`, `trabalha_em`, `nasceu_em`.
    objeto     — the value, free-form text (not a slug). Examples:
                 `Reino Unido`, `Londres`, `engenheiro de software`.
    confidence — extractor's self-rated confidence, 0..1.
    evidence   — short quote from the digest supporting the triple.
    """
    sujeito: str
    predicado: str
    objeto: str
    confidence: float
    evidence: str = ""


# ---------------------------------------------------------------------
# slug + injection-safety patterns (shared with preferences sanitisation)
# ---------------------------------------------------------------------

# Predicado: lowercase verb/relation slug, no fixed prefix unlike preferences.
_FACT_PREDICADO_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
# Sujeito: same shape, with the same length bounds.
_FACT_SUJEITO_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
# Objeto: bound length and reject the same engine-marker shape we reject
# in preferences. Facts may eventually be surfaced in prompts too, so
# defending at write time keeps the triples table clean.
_FACT_OBJETO_MAX = 200
_FACT_SUSPICIOUS_TAG_RE = re.compile(r"\[[A-Z]{3,}\]")


# ---------------------------------------------------------------------
# daily summary
# ---------------------------------------------------------------------


_SUMMARY_SYSTEM = """You compact a day of conversation between a user and an assistant into a single concise digest in Brazilian Portuguese.

Write 300-600 words in plain prose. Capture:
  - What the user worked on (topics, projects, code modules)
  - Decisions reached
  - Open questions or follow-ups
  - Notable personal facts the user mentioned (location, schedule, identity) — for the assistant's awareness

Do NOT:
  - Quote raw turns verbatim
  - Repeat the same point in different words
  - Add anything not present in the source
  - Use bullet lists — write as a paragraph or two

Return ONLY the digest. No headers, no markdown, no preface.
"""


def summarise_day(
    text: str,
    llm_chat_fn: Callable[..., Any],
    model: str,
    max_input_chars: int = 8000,
) -> Optional[str]:
    """Produce a single digest for the day's conversation.

    Returns None for empty input, LLM failure, or empty model response.
    Otherwise returns the digest as a stripped string.

    `text` is typically the output of `preferences.recent_conversations`
    — same shape, same window. We bound the input to keep latency
    predictable; the 8k default fits comfortably in Llama 3.1 8B's
    context with room for the system prompt and response.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        response = llm_chat_fn(
            model=model,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": text[:max_input_chars]},
            ],
            options={"temperature": 0.0},
        )
    except Exception as e:
        log.warning("consolidator.summarise_day: LLM call failed: %s", e)
        return None
    msg = (response or {}).get("message") or {}
    content = (msg.get("content") or "").strip()
    return content or None


def persist_daily_summary(
    palace: Any,
    summary: str,
    created_at: Optional[str] = None,
) -> Optional[int]:
    """Insert the digest as a document with `tipo='daily_summary'`.

    Embedding is best-effort: when `palace.embed_fn` is configured we
    embed the digest so future searches can pull it via semantic
    retrieval. A failed embedder doesn't block the write — FTS5 can
    still index the text.

    Returns the new document id, or None on empty input. The caller
    is expected to wrap in try/except for any deeper failure (corrupt
    palace, disk full).
    """
    if not isinstance(summary, str) or not summary.strip():
        return None
    embedding = None
    fn = getattr(palace, "embed_fn", None)
    if fn is not None:
        try:
            embedding = fn(summary)
        except Exception as e:
            log.warning(
                "consolidator.persist_daily_summary: embed failed (continuing): %s",
                e,
            )
    return palace.add_document(
        tipo="daily_summary",
        texto=summary,
        embedding=embedding,
        created_at=created_at,
    )


# ---------------------------------------------------------------------
# fact extraction
# ---------------------------------------------------------------------


_FACT_EXTRACTION_SYSTEM = """You extract STRUCTURED FACTS about the user or the world from a digest of a day's conversation. You return ONLY a JSON array. No prose, no markdown fences, nothing outside the array.

Most digests contain SEVERAL stable facts — location AND work AND hardware AND language all in the same paragraph. Extract ALL OF THEM — one triple per distinct fact. Sweep these categories every time:
  - location: where the user lives, was born, traveled
  - work: occupation, employer, role
  - hardware: machines, devices, equipment they use
  - languages: programming languages, spoken languages
  - tools: software, frameworks, services the user works with
  - identity: name, nationality, age range (only if stated)
  - relationships: family, colleagues mentioned by relation

Each item is an object with these exact keys:
  - sujeito:   entity the fact is about ("user", "cortex", or a named entity), lowercase ASCII slug
  - predicado: verb/relation, lowercase ASCII, no spaces, e.g. "vive_em", "trabalha_como", "nasceu_em", "usa_hardware", "usa_linguagem", "usa_ferramenta"
  - objeto:    the value as a short noun phrase (proper case allowed): "Londres", "Reino Unido", "Python", "engenheiro de software", "MacBook Pro M1"
  - confidence: float between 0.0 and 1.0
  - evidence:  the shortest substring from the digest that justifies the triple

Extract ONLY stable facts. SKIP:
  - preferences ("prefiro X", "gosto de Y", "geralmente") — another pipeline handles those
  - ephemeral events ("hoje fez um commit", "ontem foi ao mercado")
  - things the user is unsure about
  - facts about other people unrelated to the user

Good examples (input → output):

Digest: "O usuário vive em Londres e trabalha como engenheiro de software."
[{"sujeito":"user","predicado":"vive_em","objeto":"Londres","confidence":0.95,"evidence":"vive em Londres"},{"sujeito":"user","predicado":"trabalha_como","objeto":"engenheiro de software","confidence":0.9,"evidence":"trabalha como engenheiro de software"}]

Digest: "Hoje o usuário corrigiu um bug no Spotify e enviou um email."
[]

Digest: "Pedro mencionou que tem um Mac M1 Pro e usa Python na maior parte do tempo."
[{"sujeito":"user","predicado":"usa_hardware","objeto":"Mac M1 Pro","confidence":0.9,"evidence":"tem um Mac M1 Pro"},{"sujeito":"user","predicado":"usa_linguagem","objeto":"Python","confidence":0.9,"evidence":"usa Python na maior parte do tempo"}]

Multi-fact digest (location + work + hardware + language all in one paragraph — extract all four):

Digest: "O usuário vive em Londres, trabalha como engenheiro de software, usa um MacBook Pro M1 e desenvolve principalmente em Python."
[{"sujeito":"user","predicado":"vive_em","objeto":"Londres","confidence":0.95,"evidence":"vive em Londres"},{"sujeito":"user","predicado":"trabalha_como","objeto":"engenheiro de software","confidence":0.9,"evidence":"trabalha como engenheiro de software"},{"sujeito":"user","predicado":"usa_hardware","objeto":"MacBook Pro M1","confidence":0.9,"evidence":"usa um MacBook Pro M1"},{"sujeito":"user","predicado":"usa_linguagem","objeto":"Python","confidence":0.9,"evidence":"desenvolve principalmente em Python"}]

Return [] when no stable fact qualifies. NEVER invent. RETURN ALL distinct facts in the array — sweep all the categories above before you stop.
"""


# Same JSON-array recovery regex as preferences.py — match outermost [...]
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def _coerce_fact_triple(raw: Any) -> Optional[FactTriple]:
    """Validate one raw dict-from-JSON for facts. Returns None on any
    deviation rather than raising — a bad row shouldn't kill the batch.

    Applies write-time sanitisation that mirrors the read-time guards
    used by preferences (newlines, control chars, suspicious all-caps
    bracket tags). A row that fails any of these never lands in the
    triplas table, so downstream consumers can trust the storage."""
    if not isinstance(raw, dict):
        return None
    sujeito = str(raw.get("sujeito", "")).strip().lower()
    predicado = str(raw.get("predicado", "")).strip().lower()
    objeto = str(raw.get("objeto", "")).strip()
    if not _FACT_SUJEITO_RE.match(sujeito):
        return None
    if not _FACT_PREDICADO_RE.match(predicado):
        return None
    if not objeto or len(objeto) > _FACT_OBJETO_MAX:
        return None
    # Reject control chars / line breaks / suspicious tags — same defense
    # the preference read-path uses, applied at write time here.
    for ch in objeto:
        if ch in ("\n", "\r"):
            return None
        if ord(ch) < 32 and ch != "\t":
            return None
    if _FACT_SUSPICIOUS_TAG_RE.search(objeto):
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    evidence = str(raw.get("evidence", ""))[:200]
    return FactTriple(
        sujeito=sujeito,
        predicado=predicado,
        objeto=objeto,
        confidence=confidence,
        evidence=evidence,
    )


def extract_facts(
    digest: str,
    llm_chat_fn: Callable[..., Any],
    model: str,
    max_input_chars: int = 8000,
) -> List[FactTriple]:
    """Run the LLM once on `digest` and return validated fact triples.

    `digest` is typically the output of `summarise_day`, but extracting
    from a raw conversation chunk also works — the prompt is robust
    either way. Operating on the digest is cheaper (fewer tokens).

    Returns [] on:
      - empty input
      - LLM exception
      - response without a JSON array
      - JSON parse failure

    Bad rows inside a valid array are silently dropped.
    """
    if not isinstance(digest, str) or not digest.strip():
        return []
    try:
        response = llm_chat_fn(
            model=model,
            messages=[
                {"role": "system", "content": _FACT_EXTRACTION_SYSTEM},
                {"role": "user", "content": digest[:max_input_chars]},
            ],
            options={"temperature": 0.0},
        )
    except Exception as e:
        log.warning("consolidator.extract_facts: LLM call failed: %s", e)
        return []
    msg = (response or {}).get("message") or {}
    content = msg.get("content") or ""
    match = _JSON_ARRAY_RE.search(content)
    if not match:
        return []
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    triples: List[FactTriple] = []
    for raw in rows:
        triple = _coerce_fact_triple(raw)
        if triple is not None:
            triples.append(triple)
    return triples


def persist_facts(
    palace: Any,
    facts: List[FactTriple],
    source: str = "studier_inferred",
    min_confidence: float = 0.55,
) -> int:
    """Write fact triples to the Palace via `add_fact`. Drops candidates
    below `min_confidence`; defers to `fact_write_allowed` for the no-op
    skip and the no-downgrade rule (inferred never closes explicit or
    higher-confidence — M3-A1). Returns the count actually persisted.

    Same churn-avoidance as `persist_preferences`: a re-assertion of an
    unchanged fact doesn't roll the bitemporal log.
    """
    if not facts:
        return 0
    from src.memory.palace import fact_write_allowed

    written = 0
    for f in facts:
        if f.confidence < min_confidence:
            continue
        existing = palace.get_fact(f.sujeito, f.predicado)
        # No-downgrade guard (M3-A1): besides the no-op skip, an inferred
        # nightly fact must never close a user_explicit or higher-confidence
        # triple — a badly summarised digest used to rebase "mora_em" said
        # by the user onto whatever the 8B inferred at 03:00.
        if not fact_write_allowed(existing, f.objeto, source, f.confidence):
            continue
        palace.add_fact(
            sujeito=f.sujeito,
            predicado=f.predicado,
            objeto=f.objeto,
            source=source,
            confidence=f.confidence,
        )
        written += 1
    return written


# ---------------------------------------------------------------------
# retention (M3-A5 — policy approved by the user, s72)
# ---------------------------------------------------------------------

def archive_stale_documents(
    palace: Any,
    *,
    tipo: str = "conversation",
    days: int = 180,
    archive_dir: Optional[str] = None,
    now: Optional[Any] = None,
) -> dict:
    """Move stale `tipo` docs out of the Palace into a gzip JSONL.

    Policy (approved s72): a doc qualifies only when it is older than
    `days` AND was never injected into a prompt (access_count = 0).
    Archive, never delete — the row leaves SQLite but every field
    survives in ~/.cortex/archive/palace_<tipo>_<stamp>.jsonl.gz
    (0600), restorable by hand. Embeddings are NOT exported: they are
    derived data, recomputable from `texto` on restore. Knowledge,
    triplas and eventos are untouched by design.

    Ordering is the safety: the archive file is written and closed
    BEFORE any row is deleted, so a crash mid-run can duplicate a doc
    into a later archive but can never lose one. Returns
    {"archived": n, "path": str|None}; never raises (nightly contract).
    """
    from datetime import datetime, timedelta, timezone

    result: dict = {"archived": 0, "path": None}
    try:
        if days <= 0:
            return result
        ref = now if now is not None else datetime.now(timezone.utc)
        cutoff = (ref - timedelta(days=days)).isoformat()
        rows = palace.stale_docs(tipo, cutoff)
        if not rows:
            return result

        import gzip
        import os

        out_dir = archive_dir or os.path.expanduser("~/.cortex/archive")
        os.makedirs(out_dir, exist_ok=True)
        stamp = ref.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(out_dir, f"palace_{tipo}_{stamp}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

        deleted = 0
        for row in rows:
            if palace.delete_document(row["id"]):
                deleted += 1
        result["archived"] = deleted
        result["path"] = path
        log.info(
            "retention: %d %r doc(s) older than %dd archived to %s",
            deleted, tipo, days, path,
        )
        return result
    except Exception:
        log.exception("retention: archive pass failed (non-fatal)")
        return result
