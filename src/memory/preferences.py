"""
Implicit preference engine — Tier 2.3.

Extracts long-running facts about the user from recent conversation text
("prefere respostas curtas", "usa dark mode", "trabalha de noite") and
persists them as bitemporal triples in the Palace. The Studier calls
into this module during its nightly consolidation pass.

Pipeline:

  1. recent_conversations(palace, hours=24) — concatenate the day's
     `tipo='conversation'` documents into a single chunk capped at
     ~8k chars so the extraction prompt fits in any 8B model's window.

  2. extract_preferences(text, llm_chat_fn) — single LLM call with a
     JSON-only system prompt. The model returns a list of candidate
     triples with a confidence score and a short evidence quote. We
     parse defensively: malformed JSON returns []; partial lists are
     filtered to entries that have all required keys.

  3. persist_preferences(palace, triples, min_confidence=0.55) — drops
     low-confidence candidates and calls Palace.add_fact for each
     survivor. The add_fact bitemporal semantics handle replacement:
     a new value for the same (sujeito, predicado) closes the prior
     row automatically.

Design choices:

  - sujeito is always "user". Cortex is single-user by definition; a
    multi-tenant future would re-key on user_id.

  - predicado uses a `prefere_<slug>` convention so consumers can
    filter with `WHERE predicado LIKE 'prefere_%'` without needing a
    separate "type" column.

  - source='studier_inferred' marks these as automatically extracted —
    consumers who want only explicit user statements can filter on
    source='user_explicit' (the default for add_fact).

  - Confidence floor is conservative: 0.55. Llama 3.1 8B has a known
    overconfidence bias on inferred facts; below 0.55 the noise rate
    in trials was too high to be useful.
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
class PreferenceTriple:
    """One extracted preference candidate.

    predicado  — short slug, prefixed `prefere_`. Examples:
                 `prefere_resposta_estilo`, `prefere_idioma`,
                 `prefere_modo_trabalho`.
    objeto     — the value the predicate resolves to, e.g. `curta_direta`,
                 `pt-br`, `noturno`. Slug-style is preferred for triples
                 the orchestrator will pattern-match later.
    confidence — extractor's self-rated confidence, 0..1.
    evidence   — short quote from the source text supporting the triple.
                 Stored only on the in-memory dataclass for logging /
                 debugging; not persisted to the triple itself.
    """
    predicado: str
    objeto: str
    confidence: float
    evidence: str = ""


# ---------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------


# JSON-only system prompt. We keep it terse and example-rich because
# Llama 3.1 8B Q4 starts hallucinating field names on long prompts.
_EXTRACTION_SYSTEM = """You extract LONG-RUNNING preferences from a chunk of conversation between a user and an assistant. You return ONLY a JSON array. No prose, no markdown fences, nothing outside the array.

The input is usually MULTIPLE turns and may contain SEVERAL distinct preferences. Extract ALL of them — one triple per independent preference statement. Do not stop at the first one.

Each item is an object with these exact keys:
  - predicado: a short slug prefixed `prefere_`, lowercase, ASCII only
  - objeto:    the value the predicate resolves to, slug-style
  - confidence: float between 0.0 and 1.0
  - evidence:  the shortest substring of the user's text that justifies the triple

Extract ONLY stable, long-running preferences. SKIP:
  - one-off requests ("toque essa música agora")
  - factual questions ("o que é GGUF")
  - momentary moods
  - things stated about other people

Good examples (input → output):

User: "Prefiro respostas curtas e diretas, sem rodeio."
[{"predicado":"prefere_resposta_estilo","objeto":"curta_direta","confidence":0.95,"evidence":"Prefiro respostas curtas e diretas"}]

User: "Eu sempre trabalho de madrugada, costumo dormir às 6 da manhã."
[{"predicado":"prefere_horario_trabalho","objeto":"madrugada","confidence":0.9,"evidence":"sempre trabalho de madrugada"}]

User: "Pode falar em português comigo, prefiro."
[{"predicado":"prefere_idioma","objeto":"pt-br","confidence":0.95,"evidence":"prefiro falar em português"}]

Multiple-preference example (one input, several triples in the output):

User: "Prefiro café preto sem açúcar. Geralmente uso dark mode no terminal."
[{"predicado":"prefere_cafe","objeto":"preto_sem_acucar","confidence":0.9,"evidence":"prefiro café preto sem açúcar"},{"predicado":"prefere_tema_terminal","objeto":"dark_mode","confidence":0.85,"evidence":"uso dark mode no terminal"}]

User: "que horas são?" → []

Return [] when nothing qualifies. NEVER invent preferences that aren't supported by the text. When the chunk contains several independent preferences, RETURN ALL OF THEM in the array.
"""


# Match a JSON array even when the model wraps it in markdown fences or
# leading prose. Captures the outermost `[...]` segment.
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)

# A predicate must look like `prefere_<lowercase_with_underscores>`.
_PREDICADO_RE = re.compile(r"^prefere_[a-z][a-z0-9_]{0,40}$")


def _coerce_triple(raw: Any) -> Optional[PreferenceTriple]:
    """Validate one raw dict-from-JSON. Returns None on any deviation
    rather than raising — bad rows shouldn't kill the whole batch."""
    if not isinstance(raw, dict):
        return None
    predicado = str(raw.get("predicado", "")).strip().lower()
    objeto = str(raw.get("objeto", "")).strip()
    if not _PREDICADO_RE.match(predicado):
        return None
    if not objeto or len(objeto) > 100:
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    evidence = str(raw.get("evidence", ""))[:200]
    return PreferenceTriple(
        predicado=predicado,
        objeto=objeto,
        confidence=confidence,
        evidence=evidence,
    )


def extract_preferences(
    text: str,
    llm_chat_fn: Callable[..., Any],
    model: str,
) -> List[PreferenceTriple]:
    """Call the LLM once on `text` and return validated preference triples.

    `llm_chat_fn` is anything with the ollama.chat signature: it accepts
    `model`, `messages`, `options` kwargs and returns a dict with the
    OpenAI-style `{"message": {"content": "..."}}` shape. Injecting it
    keeps the module unit-testable without an Ollama process.

    Returns [] on:
      - empty / whitespace-only input
      - LLM call exception
      - response content missing a JSON array
      - JSON parse failure

    Bad rows inside a valid array are silently dropped — the rest of
    the batch survives.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        response = llm_chat_fn(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": text[:8000]},
            ],
            # Determinism: a preference extracted twice on the same chunk
            # should produce the same triples. The bitemporal upsert
            # already handles duplicates, but stable output makes the
            # audit log readable.
            options={"temperature": 0.0},
        )
    except Exception as e:
        log.warning("preferences.extract: LLM call failed: %s", e)
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

    triples: List[PreferenceTriple] = []
    for raw in rows:
        triple = _coerce_triple(raw)
        if triple is not None:
            triples.append(triple)
    return triples


# ---------------------------------------------------------------------
# Palace helpers
# ---------------------------------------------------------------------


def recent_conversations(
    palace: Any,
    hours: int = 24,
    max_chars: int = 8000,
    now_iso: Optional[str] = None,
) -> str:
    """Concatenate `tipo='conversation'` documents created within the last
    `hours` into a single chunk, newest first, capped at `max_chars`.

    Why newest-first: when we hit the cap, the freshest signal survives.
    Stale snippets at the tail get dropped, which is the right bias for
    a preference extractor that needs to reflect the user's current
    habits, not last year's.

    Pure read — no access_count bump, no FTS query. Direct SQL keeps
    the cost negligible even on a year of conversation history.
    """
    from datetime import datetime, timedelta, timezone

    if hours <= 0 or max_chars <= 0:
        return ""
    if now_iso is None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    else:
        from datetime import datetime as _dt
        ref = _dt.fromisoformat(now_iso)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff = (ref - timedelta(hours=hours)).isoformat()

    texts = palace.recent_texts("conversation", cutoff)

    # One turn gets at most a quarter of the budget, cut at its END — the user's
    # words open every stored turn. The newest turn over the whole budget (a
    # long document answer) used to stop the loop with nothing taken: an
    # EMPTY corpus, and both nightly passes skipped the day (s88, Codex
    # lote 6).
    per_turn = max(1, max_chars // 4)
    pieces: List[str] = []
    used = 0
    for t in texts:
        if not t:
            continue
        if len(t) > per_turn:
            t = t[:per_turn] + " […]"
        # +2 accounts for the "\n\n" separator we'll join on.
        if used + len(t) + 2 > max_chars:
            continue
        pieces.append(t)
        used += len(t) + 2
    return "\n\n".join(pieces)


# ---------------------------------------------------------------------
# Who said it (s88, Codex lote 6)
# ---------------------------------------------------------------------
# The nightly passes read "User: …\nCortex: …" turns. A preference or a
# fact the 8B found on Cortex's side — "a Marta mora em Paris e prefere
# respostas longas", an example the user asked for — went into the user's profile
# (reproduced: vive_em=Paris, prefere_resposta_estilo=longa). What reaches
# the profile must be in the user's own words.


def _fold(text: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text or "")
    plain = "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()
    return re.sub(r"\s+", " ", plain).strip()


# The mark on a conversation turn heard by the room microphone (s88, Codex
# lote 12 — invariant 3): anyone who says the wake word is answered, and
# speaker-ID is still shadow only, so the words may be a visitor's.
UNVERIFIED_SPEAKER = ("speaker", "unverified")


def recent_user_words(palace: Any, hours: int = 24,
                      now_iso: Optional[str] = None) -> str:
    """Pedro's side of the same window recent_conversations reads: the part
    of each stored turn before its first "\nCortex: ", one turn at a time —
    a "User:" the model wrote inside a reply never counts. Folded (case,
    accents, spaces) for the checks below. A turn marked UNVERIFIED_SPEAKER
    is left out: what a visitor said to the room microphone is no evidence
    of what Pedro said."""
    from datetime import datetime, timedelta, timezone
    if hours <= 0:
        return ""
    ref = (datetime.fromisoformat(now_iso) if now_iso
           else datetime.now(timezone.utc))
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    cutoff = (ref - timedelta(hours=hours)).isoformat()
    own = []
    for t in palace.recent_texts("conversation", cutoff,
                                 exclude_meta=UNVERIFIED_SPEAKER):
        head = (t or "").split("\nCortex: ", 1)[0]
        if head.startswith("User: "):
            own.append(head[len("User: "):])
    return _fold("\n".join(own))


def said_by_user(own_words: str, evidence: str) -> bool:
    """The evidence is a stretch of what Pedro wrote — the extractor's own
    contract ("the shortest substring of the user's text")."""
    quote = _fold(evidence).strip(" .,;:!?\"'")
    return bool(quote) and quote in own_words


def value_said_by_user(own_words: str, value: str) -> bool:
    """Every word of the value (3+ letters) appears in what Pedro wrote —
    for facts read off the day's digest, whose evidence quotes the digest."""
    words = [w for w in re.split(r"[^0-9a-z]+", _fold(value)) if len(w) >= 3]
    return bool(words) and all(w in own_words for w in words)


# Rows returned by Palace.get_preferences must NOT be trusted to splice
# directly into a system prompt. The write path enforces _PREDICADO_RE and
# objeto bounds via _coerce_triple, but a Palace row can also land via
# migrations, manual SQL, or a bug in some future extractor. A poisoned
# objeto (newline + leading `[KNOWLEDGE]`) would inject what looks like
# authoritative content into the engine's instructions — that tag means
# "treat as ground truth" in the system prompt's KNOWLEDGE block. Same
# logic for any all-caps bracket-tag, since the engine reserves that
# shape for its own markers.
_INJECTION_SUSPICIOUS_TAG_RE = re.compile(r"\[[A-Z]{3,}\]")
_OBJETO_MAX_INJECT_LEN = 100


def _safe_to_inject(predicado: str, objeto: str) -> bool:
    """Return True iff this (predicado, objeto) pair is safe to splice
    into the system prompt. Mirrors the write-side validation in
    `_coerce_triple` but applied at read time, defending against rows
    that bypassed the writer (manual edit, migration, future bug)."""
    if not _PREDICADO_RE.match(predicado):
        return False
    if not objeto or len(objeto) > _OBJETO_MAX_INJECT_LEN:
        return False
    # Reject control characters that would break the prompt structure.
    # Tab is allowed because it has no impact on LLM tokenisation; newline
    # and carriage return would let an objeto split into a new line of
    # the bullet list (or worse, slip a `[KNOWLEDGE]` tag onto its own
    # line where the model treats it as a section header).
    for ch in objeto:
        if ch in ("\n", "\r"):
            return False
        if ord(ch) < 32 and ch != "\t":
            return False
    if _INJECTION_SUSPICIOUS_TAG_RE.search(objeto):
        return False
    return True


def format_preferences_for_prompt(
    triples: List[Any],
    min_confidence: float = 0.55,
    max_lines: int = 12,
) -> str:
    """Render persisted preference triples as a compact USER PROFILE block
    suitable for prepending to the system prompt.

    Accepts the dict shape returned by `Palace.get_preferences()` or
    `Palace.list_facts()` — each row must expose `predicado`, `objeto`,
    and optionally `confidence`. Returns the empty string when no row
    survives the confidence floor, so the caller can concatenate
    unconditionally without an `if`.

    Format is intentionally terse — Llama 3.1 8B Q4 doesn't reward
    verbose framing — and uses a `→` arrow because that's the most
    compact way the model parses key/value pairs reliably in the
    triples it sees during training. The trailing blank line
    separates this block from whatever follows.

    The 'treat as durable context, not absolutes' caveat is deliberate:
    these triples are inferred, not stated. Pedro may have said
    something offhandedly that the extractor picked up too eagerly,
    and we don't want the model to defend the wrong preference if
    Pedro contradicts it in the current turn.

    `max_lines` caps the block at the top-N rows by confidence (M3-A8,
    review Fable 5): the nightly extractor adds rows monotonically, and
    an unbounded block compounds the num_ctx 8192 budget squeeze (M2-A1)
    one short line at a time. Ties keep the original (Palace) order.
    """
    if not triples:
        return ""
    scored: List[tuple] = []
    for row in triples:
        predicado = (row.get("predicado") or "").strip()
        objeto = (row.get("objeto") or "").strip()
        if not predicado or not objeto:
            continue
        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_confidence:
            continue
        if not _safe_to_inject(predicado, objeto):
            # A row that fails the read-side check almost certainly came
            # from outside the normal write path. Log so we notice if
            # this starts firing in production.
            log.warning(
                "format_preferences_for_prompt: dropped unsafe row "
                "predicado=%r objeto=%r",
                predicado, objeto,
            )
            continue
        scored.append((conf, f"- {predicado} → {objeto}"))
    if not scored:
        return ""
    # Stable sort: ties keep insertion (Palace) order.
    scored.sort(key=lambda pair: -pair[0])
    lines = [line for _, line in scored[:max_lines]]
    return (
        "USER PROFILE (preferences inferred from past conversations — "
        "treat as durable context, not absolutes; the current turn "
        "always wins on conflict):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def persist_preferences(
    palace: Any,
    triples: List[PreferenceTriple],
    source: str = "studier_inferred",
    min_confidence: float = 0.55,
) -> int:
    """Write triples to the Palace via add_fact. Drops candidates below
    `min_confidence`. Returns the number of triples actually persisted.

    add_fact's bitemporal semantics mean a new (predicado, objeto) for
    the same predicate auto-closes the prior row — so the write decision
    happens HERE, via `fact_write_allowed`: skip the no-op same-objeto
    case (audit churn) and never let an inferred triple close a
    user_explicit or higher-confidence one (M3-A1).
    """
    if not triples:
        return 0
    from src.memory.palace import fact_write_allowed

    written = 0
    for t in triples:
        if t.confidence < min_confidence:
            continue
        existing = palace.get_fact("user", t.predicado)
        # No-downgrade guard (M3-A1): same-objeto skip plus "an inferred
        # write never closes an explicit or higher-confidence triple".
        if not fact_write_allowed(existing, t.objeto, source, t.confidence):
            continue
        palace.add_fact(
            sujeito="user",
            predicado=t.predicado,
            objeto=t.objeto,
            source=source,
            confidence=t.confidence,
        )
        written += 1
    return written
