"""
The Hippocampus — persistent conversation memory, now backed by Palace.

CortexMemory was historically a ChromaDB wrapper with two collections
(`cortex_long_term` for chat history, `cortex_knowledge` for briefings).
It is now a thin facade over `Palace`, where:

    conversation turns  → documentos(tipo='conversation')
    approved briefings  → documentos(tipo='knowledge')  via KnowledgeBase

The public API is unchanged so the engine (src/core/engine.py) and any
other caller continues to work without modification:

    store_interaction(user_query, assistant_response)
    retrieve_relevant_context(current_query, n_results=3,
                              include_knowledge=True) -> str

Error-marker filtering on writes is preserved verbatim — we don't want
to poison long-term memory with engine-failure replies.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from src.core.knowledge_base import KnowledgeBase
from src.core.prompt_safety import clip_keeping_fence, fence, own_words
from src.core.prompts import KNOWLEDGE_TAG, STUDY_TAG
from src.memory.embeddings import OllamaEmbedder
from src.memory.palace import Palace
from src.memory.preferences import UNVERIFIED_SPEAKER

log = logging.getLogger(__name__)

# A turn heard by the room microphone (s88, Codex lote 12 — invariant 3):
# the listener answers anyone who says the wake word and speaker-ID is
# still shadow only, so "meu hobby é escalada" may be a visitor's. Such a
# turn is kept as conversation, marked, and never becomes a fact about
# The user — not by the passive regex below, not by the nightly passes (their
# "said by the user" check skips the mark). Thread-local: /ask_stream sets
# it around each pull of the stream, since pool threads serve other
# requests, and the orchestrator carries it into its stream worker.
_speaker = threading.local()


def speaker_unverified() -> bool:
    return bool(getattr(_speaker, "unverified", False))


@contextmanager
def unverified_speaker():
    before = speaker_unverified()
    _speaker.unverified = True
    try:
        yield
    finally:
        _speaker.unverified = before


def unverified_speaker_stream(events):
    """`events`, each pull made with the mark set on the thread pulling it."""
    events = iter(events)
    try:
        while True:
            with unverified_speaker():
                try:
                    item = next(events)
                except StopIteration:
                    return
            yield item
    finally:
        with unverified_speaker():
            close = getattr(events, "close", None)
            if close is not None:
                close()


class CortexMemory:
    """Conversation-history store backed by Palace, with KnowledgeBase mounted."""

    DOC_TIPO = "conversation"
    DEFAULT_PALACE_DB = "data/palace.db"

    # Refuse to persist outputs that look like engine-level error replies —
    # otherwise the long-term memory gets poisoned with failure messages.
    _ERROR_MARKERS = (
        "Insufficient data due to engine failure",
        "[SYSTEM ERROR]",
        "[OFFLINE]",
    )

    # Refusal phrases: when the 8B wrongly declines a harmless general-
    # knowledge / creative request, storing that reply poisons memory —
    # next time the same question retrieves the refusal as "PAST CONTEXT"
    # and the model refuses AGAIN, digging the hole deeper (observed live
    # 2026-06: "quem descobriu o Brasil?" → escalating refusals). These
    # are matched only at the START of the reply (after light stripping)
    # so a legitimate answer that merely contains "desculpe" mid-sentence
    # is still stored. See _is_refusal.
    # Matched within the OPENING of the reply (first ~80 chars, lowered),
    # not just the very start, because refusals often carry a greeting
    # first ("Olá! Sinto muito, mas não posso..."). A refusal is never
    # useful future context — even a legitimate "I can't do that action"
    # only adds noise to retrieval — so the bar for storing is: not a
    # refusal opener. First-person "não posso"/"cannot" in the opening is
    # a strong refusal signal that a normal informative answer won't trip.
    _REFUSAL_PHRASES = (
        "não posso",
        "nao posso",
        "sinto muito",
        "peço desculpas",
        "peco desculpas",
        "lamento, mas",
        "não consigo ajudar",
        "nao consigo ajudar",
        "i cannot",
        "i can't",
        "i'm sorry",
        "i am sorry",
        "i'm not able to",
        "i'm unable to",
        "i am unable to",
    )
    _REFUSAL_SCAN_CHARS = 80

    @classmethod
    def _is_refusal(cls, reply: str) -> bool:
        """True if the reply's opening contains a refusal phrase. Scans the
        first ~80 chars (lowered) so a leading greeting before the refusal
        ('Olá! Sinto muito, mas...') is still caught. Tuned to first-person
        decline phrasing so a normal informative answer isn't suppressed."""
        head = (reply or "").strip().lower()[: cls._REFUSAL_SCAN_CHARS]
        return any(phrase in head for phrase in cls._REFUSAL_PHRASES)

    def __init__(
        self,
        db_path: Optional[str] = None,        # legacy, ignored
        palace_db: Optional[str] = None,
        palace: Optional[Palace] = None,
    ):
        if palace is not None:
            self._palace = palace
            self._owns_palace = False
        else:
            self._palace = Palace(
                db_path=palace_db or self.DEFAULT_PALACE_DB,
                embed_fn=OllamaEmbedder(),
            )
            self._owns_palace = True

        # Approved nightly studies live in the same Palace, exposed via the
        # KnowledgeBase wrapper. We share the Palace instance so writes from
        # one side are visible to the other without a re-open.
        self.knowledge = KnowledgeBase(palace=self._palace)

    def conversation_memory_ids(self, turns: list[dict]) -> list[int]:
        """Find legacy turn documents, which have no conversation ID metadata.

        Match the write format exactly. Only when a turn has no exact match
        can its long reply identify a document, and only if that reply occurs
        in exactly one conversation document across the whole Palace.
        Read errors propagate so callers cannot erase chat before its memories.
        """
        queries = []
        for turn in turns:
            user = own_words(turn["user"])
            reply = turn["reply"]
            if user is not None or len(reply) >= 80:
                text = f"User: {user}\nCortex: {reply}" if user is not None else None
                queries.append((text, reply))
        if not queries:
            return []

        documents = self._palace.list_documents(tipo=self.DOC_TIPO, limit=-1)
        by_text, by_reply = {}, {}
        for doc in documents:
            text = doc["texto"]
            by_text.setdefault(text, []).append(doc["id"])
            _, separator, reply = text.partition("\nCortex: ")
            if separator:
                by_reply.setdefault(reply, []).append(doc)

        matches = set()
        for text, reply in queries:
            exact = by_text.get(text, [])
            if exact:
                matches.update(exact)
            elif len(reply) >= 80:
                candidates = by_reply.get(reply, [])
                if len(candidates) == 1 and candidates[0]["texto"].startswith("User: "):
                    matches.add(candidates[0]["id"])
        return [doc["id"] for doc in documents if doc["id"] in matches]

    def store_interaction(self, user_query: str, assistant_response: str):
        """Persist a turn as a single conversation document. Embedding is
        best-effort: if the Ollama call fails, the row is still stored
        (FTS5 retrieval still works on the text)."""
        if any(marker in assistant_response for marker in self._ERROR_MARKERS):
            return
        # Don't persist refusals — they poison future retrievals (see
        # _REFUSAL_PREFIXES). A wrongly-declined harmless question must not
        # teach the model to decline it again.
        if self._is_refusal(assistant_response):
            return
        # Every caller passes the engine's input, which on the web, document
        # and attachment paths is the whole assembled prompt, fenced external
        # content included. Reduce it to the user's own words here, the one place
        # all of them pass through, rather than at each of the builders.
        user_query = own_words(user_query)
        if user_query is None:
            return

        text_to_store = f"User: {user_query}\nCortex: {assistant_response}"
        metadata = {
            "source": "conversation",
            "interaction_id": f"int_{os.urandom(4).hex()}",
        }
        unverified = speaker_unverified()
        if unverified:
            key, value = UNVERIFIED_SPEAKER
            metadata[key] = value
        embedding = self._safe_embed(text_to_store)
        self._palace.add_document(
            tipo=self.DOC_TIPO,
            texto=text_to_store,
            metadata=metadata,
            embedding=embedding,
        )

        # Passive profile-fact extraction. Runs on the same
        # per-turn chokepoint as the conversation write, so it inherits the
        # error/refusal guards above and fires across every engine path.
        # Not for a turn whose speaker is unverified (see unverified_speaker).
        if not unverified:
            self._consolidate_passive_facts(user_query)

    # Kill-switch: passive extraction is on by default but can be disabled
    # with CORTEX_PASSIVE_FACTS=0 if a regex ever starts mis-capturing in
    # the wild. Read per-call so it can be toggled without a restart.
    @staticmethod
    def _passive_facts_enabled() -> bool:
        return os.getenv("CORTEX_PASSIVE_FACTS", "1").strip() not in ("0", "false", "no", "")

    def _consolidate_passive_facts(self, user_query: str) -> int:
        """Extract stable profile facts from the user's turn and persist
        them to the Palace. Best-effort: any failure is swallowed so a bad
        turn never breaks the reply path. Returns the number written.

        No-downgrade guard: a passive (confidence 0.6, source 'passive_regex')
        hit must never clobber a stronger fact. We skip when the active
        triple is explicit, higher-confidence, or already the same value —
        mirroring `preferences.persist_preferences`."""
        if not self._passive_facts_enabled():
            return 0
        try:
            from src.core.fact_extractor import (
                PASSIVE_CONFIDENCE, PASSIVE_SOURCE, extract_facts,
            )
            facts = extract_facts(user_query)
        except Exception as e:  # noqa: BLE001 — extraction must never break a turn
            log.warning("passive fact extraction failed: %s", e)
            return 0

        from src.memory.palace import fact_write_allowed

        written = 0
        for f in facts:
            try:
                existing = self._palace.get_fact("user", f.predicado)
                # The shared no-downgrade guard (M3-A1) — this path is where
                # the rule was born; the nightly persisters now share it.
                if not fact_write_allowed(
                    existing, f.objeto, PASSIVE_SOURCE, PASSIVE_CONFIDENCE
                ):
                    continue
                self._palace.add_fact(
                    sujeito="user",
                    predicado=f.predicado,
                    objeto=f.objeto,
                    source=PASSIVE_SOURCE,
                    confidence=PASSIVE_CONFIDENCE,
                )
                written += 1
            except Exception as e:  # noqa: BLE001
                log.warning("passive fact persist failed (%s): %s", f.predicado, e)
        return written

    PREFERENCE_INJECT_MIN_CONFIDENCE = 0.55

    def get_user_profile(self) -> str:
        """Return a pre-formatted USER PROFILE block ready to splice into
        the system prompt. Reads active preference triples from the Palace
        (`predicado LIKE 'prefere_%'`), filters by confidence, formats via
        preferences.format_preferences_for_prompt.

        Returns an empty string on ANY failure (no Palace, malformed row,
        unexpected exception) — the system prompt must always build. The
        block is regenerated on every call rather than cached so a
        preference change between turns (e.g. the Studier wrote new rows
        last night) takes effect on the very next reply."""
        try:
            from src.memory.preferences import format_preferences_for_prompt
            rows = self._palace.get_preferences(
                min_confidence=self.PREFERENCE_INJECT_MIN_CONFIDENCE,
            )
            prefs_block = format_preferences_for_prompt(
                rows,
                min_confidence=self.PREFERENCE_INJECT_MIN_CONFIDENCE,
            )
        except Exception as e:
            log.warning("CortexMemory.get_user_profile (preferences) failed: %s", e)
            prefs_block = ""

        # Passively-extracted profile facts (name, city, profession) ride
        # alongside the preference block but through a SEPARATE formatter:
        # the preference injector's predicate regex only accepts `prefere_*`,
        # and these aren't preferences. Independent try/except so a profile
        # failure never costs us the preference block (or vice-versa).
        try:
            from src.core.fact_extractor import (
                PROFILE_INJECT_KEYS, format_profile_facts_for_prompt,
            )
            user_facts = [
                r for r in self._palace.list_facts("user")
                if r.get("predicado") in PROFILE_INJECT_KEYS
                and float(r.get("confidence", 0.0)) >= self.PREFERENCE_INJECT_MIN_CONFIDENCE
            ]
            facts_block = format_profile_facts_for_prompt(user_facts)
        except Exception as e:
            log.warning("CortexMemory.get_user_profile (facts) failed: %s", e)
            facts_block = ""

        return prefs_block + facts_block

    # Per-chunk and total caps on injected context. The model reprocesses
    # this whole block on EVERY query (it changes each turn, so Ollama's
    # prompt cache can't help), and prompt-eval on the M1 8B is ~22 tok/s
    # → an 11k-char context cost ~19.6s of latency BEFORE the first token.
    # Measured TTFT vs context size: 11k→19.6s, 3k→2.5s, 1.5k→1.8s, 0.5k→1s.
    # Capping each chunk (the query + the head of the reply carries the
    # retrieval value; the long tail doesn't) and the total keeps TTFT
    # ~2s without losing the relevant signal. Tunable via env for testing.
    _CHUNK_CHAR_CAP = int(os.getenv("CORTEX_CTX_CHUNK_CAP", "400"))
    _CONTEXT_CHAR_CAP = int(os.getenv("CORTEX_CTX_TOTAL_CAP", "1500"))
    # Reserved slice of the total cap for [KNOWLEDGE] (M3-A2). Knowledge is
    # authoritative — it's where the passport/CPF VALUE lives — while convo
    # snippets are colour; yet the old merge put conversation first and
    # head-truncated the total, so knowledge was the first thing cut. Worse,
    # every successful recall stores "User: qual meu X" as a conversation
    # turn; on the next identical question those echoes rank ~1.0, pass the
    # floor and crowd the cap WITHOUT carrying the value (re-injection is
    # user-side only, s40) — so asking again made recall worse. Knowledge now
    # leads the merge under this budget; conversation fills only the slack
    # the knowledge actually leaves (full cap when there's no knowledge, so
    # the include_knowledge=False tool path is unchanged).
    _KNOWLEDGE_BUDGET = int(os.getenv("CORTEX_CTX_KNOWLEDGE_BUDGET", "900"))
    # What fencing adds to one study chunk over the plain [KNOWLEDGE] layout
    # (48 chars) — paid outside the caps, see retrieve_relevant_context.
    _STUDY_LABEL_COST = (len(f"{STUDY_TAG}\n{fence('')}")
                         - len(f"{KNOWLEDGE_TAG}\n"))
    # Knowledge gets a WIDER candidate pool than conversation memory. After
    # section-chunking (s37) a field lives in a small section that can rank
    # mid-pool (a multi-field section like "Documentos brasileiros" ranks 7-13
    # for "qual meu CPF"); since chunks are now small, pulling more of them is
    # cheap and the total is still capped by _CONTEXT_CHAR_CAP. Paired with the
    # OR-based FTS query (s39), this lands tail field lookups (CEP/RG/…).
    _KNOWLEDGE_N_RESULTS = int(os.getenv("CORTEX_CTX_KNOWLEDGE_N", "6"))
    # Relevance floor for CONVERSATION retrieval (Bug B-2). RRF returns the
    # top-k by RANK regardless of absolute relevance, so a generic query with
    # nothing relevant still drags in the least-irrelevant past turn (the "fala
    # do jogo em 'status do sistema'" contamination). The raw cosine the Palace
    # now exposes (`_sem_sim`) is an absolute relevance signal; a convo hit
    # whose semantic similarity is BELOW this floor is dropped.
    #
    # CONSERVATIVE backstop, not a precise separator: nomic-embed-text compresses
    # cosine into a high, narrow band (live calibration s58 — unrelated turns
    # ~0.59, genuine follow-ups ~0.71, bands OVERLAP), so the floor sits safely
    # below the observed legit-follow-up minimum (~0.64) to avoid regressing real
    # recall. It removes only the clearly-orthogonal tail (math, off-topic).
    # Knowledge is untouched (different code path) → the s39 gain is safe by
    # construction. A hit with NO semantic score is KEPT: we have no semantic
    # signal to judge it. Since s83 every candidate with a vector carries its
    # cosine, whichever channel surfaced it, so this only applies to a turn with
    # no usable vector. Raise the floor when a better embedder widens the band.
    # Kill-switch: set to 0.
    _CONVO_SIM_FLOOR = float(os.getenv("CORTEX_CONVO_SIM_FLOOR", "0.55"))

    # A user side this short ("sim", "c", "Ah!", "play", "no Spotify.") carries
    # nothing once its reply is dropped on re-injection. And the floor can't
    # catch it: bge-m3 scores such fragments 0.50-0.65 against almost any query.
    # The s83 probe found them in about a quarter of all injections. Measured
    # on the echo key (casefolded, label and punctuation gone), so "Spotify."
    # and "spotify" count the same. Kill-switch: set to 0.
    _MIN_USER_CHARS = int(os.getenv("CORTEX_CTX_MIN_USER_CHARS", "15"))

    # Don't re-inject a past user turn that is a QUESTION (s84). Only the user
    # side comes back, so an old question never carries its answer — and the
    # 8B reads it as context anyway: with "pense em uma cor... por que azul?"
    # in the prompt it answered "your favourite colour is blue", while the
    # turn where the user said "green" sat further down. Detected by "?" alone,
    # on purpose: interrogative openers (qual/como/que…) would also cut
    # statements like "como eu disse, trabalho no escritório". Probe: injected
    # questions 597 → 84, memory recall unchanged at 12/14. What's lost is
    # the "you asked me this before" signal. Kill-switch: set to 0.
    @staticmethod
    def _skip_questions() -> bool:
        return os.getenv("CORTEX_CTX_SKIP_QUESTIONS", "1").strip() not in ("0", "false", "no", "")

    # How many conversation candidates to fetch per slot (s84). The floor, the
    # echo dedup and the short-turn cut all run AFTER the search, so with
    # k = n_results a past copy of the current question took a slot and was
    # then dropped, and nothing from below moved up. Every time the user asked,
    # the question was stored, and the next time its echo pushed the answer
    # out: asked twice in a row on the real 8B, "qual o meu prato brasileiro
    # favorito?" was answered from the stored turn the first time and with an
    # invented dish the second. Now cuts skip a candidate instead of spending
    # a slot, and injection stops at n_results survivors.
    _CONVO_FETCH_FACTOR = 3

    @classmethod
    def convo_fetch_k(cls, n_results: int) -> int:
        """How many conversation candidates retrieval asks the Palace for —
        shared with scripts/palace_retrieval_probe.py so it ranks the same."""
        return max(n_results, n_results * cls._CONVO_FETCH_FACTOR)

    @staticmethod
    def _truncate(text: str, cap: int) -> str:
        # Never leaves a [STUDY] fence open: a cut through one closes it, so
        # the conversation and the question after it stay outside.
        return clip_keeping_fence(text, cap, "…")

    @classmethod
    def _tag_knowledge(cls, chunk: str) -> str:
        """One retrieved knowledge chunk as it enters PAST CONTEXT: Pedro's
        own profile data as [KNOWLEDGE]; an approved web study as [STUDY],
        fenced as third-party content, so what it says about someone else
        never reads as a fact about him (invariant 3, Codex lote 14)."""
        text = cls._truncate(chunk, cls._CHUNK_CHAR_CAP * 2)
        if KnowledgeBase.is_study_chunk(chunk):
            return f"{STUDY_TAG}\n{fence(text)}"
        return f"{KNOWLEDGE_TAG}\n{text}"

    @staticmethod
    def _user_side(turn_text: str) -> str:
        """Return just the 'User:' half of a stored 'User: …\\nCortex: …' turn.
        Falls back to the whole text if the separator is absent (malformed or
        legacy row). See retrieve_relevant_context for why."""
        head, _sep, _tail = (turn_text or "").partition("\nCortex:")
        return head

    @staticmethod
    def _conversation_date_label(created_at: Optional[str]) -> str:
        """Stored UTC timestamp as a Mac-local day; unknown dates stay bare.

        Offset-free legacy timestamps mean UTC, as in Palace's age helpers.
        Convert at the turn's date so historical daylight saving is respected.
        """
        try:
            stored = datetime.fromisoformat(created_at)
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            return f"[Conversa de {stored.astimezone():%d/%m/%Y}] "
        except (TypeError, ValueError, OverflowError, OSError):
            return ""

    @staticmethod
    def _echo_key(text: str) -> str:
        """Normalise a snippet for echo comparison: casefold, drop the
        'User:' label, collapse punctuation/whitespace. Two snippets with
        the same key are the same question for retrieval purposes — the
        second copy adds zero information to the injected context."""
        text = (text or "").casefold()
        text = re.sub(r"^\s*user\s*:\s*", "", text)
        return re.sub(r"[\W_]+", " ", text).strip()

    def retrieve_relevant_context(
        self,
        current_query: str,
        n_results: int = 3,
        include_knowledge: bool = True,
    ) -> str:
        """
        Retrieve relevant context from conversation history and,
        optionally, the approved-knowledge base, merged into a single
        context block.

        Conversation snippets carry their stored local day; Pedro's own profile
        documents are tagged [KNOWLEDGE], and approved web studies [STUDY]
        and fenced (see _tag_knowledge).

        `include_knowledge=False` skips the knowledge base entirely. The
        tool-calling path (`chat_with_tools`) passes this: once the
        orchestrator has decided the turn is an ACTION (TOOL_HINT matched),
        the user's personal profile briefings don't help pick or fill a
        tool — and worse, an 8B model latches onto a semantically-matched
        [KNOWLEDGE] block and asks irrelevant clarifying questions instead
        of acting (JOURNAL sessão 14, bug 5). Conversation memory is still
        retrieved because recent turns can carry a tool argument.
        """
        # Knowledge FIRST (M3-A2): the total cap below keeps the HEAD of the
        # merged block, so whatever comes first survives truncation. Knowledge
        # is the authoritative category — it leads, under its own budget.
        knowledge_block = ""
        label_room = 0
        if include_knowledge:
            try:
                knowledge_chunks = self.knowledge.retrieve_relevant(
                    current_query,
                    n_results=max(n_results, self._KNOWLEDGE_N_RESULTS),
                )
                # Knowledge briefings can be authoritative personal facts —
                # give them a larger slice so a value (passport, NHS, etc.)
                # isn't cut, but still bounded.
                tagged = [self._tag_knowledge(c) for c in knowledge_chunks]
                # The fence around a study is a label, not content: it gets
                # room of its own on top of the caps, so fencing never pushes
                # out what fit before. Measured live (s88): without this, two
                # fenced studies cut the user's birthplace chunk from 140 to 45
                # chars and "onde eu nasci?" went from right 3/3 to refused 3/3.
                label_room = self._STUDY_LABEL_COST * sum(
                    KnowledgeBase.is_study_chunk(c) for c in knowledge_chunks)
                if tagged:
                    knowledge_block = self._truncate(
                        "\n---\n".join(tagged),
                        min(self._KNOWLEDGE_BUDGET, self._CONTEXT_CHAR_CAP)
                        + label_room,
                    )
            except Exception as e:
                log.warning(f"CortexMemory: knowledge search failed: {e}")

        convo_chunks: list[str] = []
        try:
            convo_hits = self._palace.search(
                current_query, k=self.convo_fetch_k(n_results), tipo=self.DOC_TIPO
            )
            # Relevance floor (Bug B-2): drop convo hits whose semantic
            # similarity is below the floor. RRF returns the top-k by rank even
            # when nothing is genuinely relevant, so a generic query otherwise
            # drags in the least-irrelevant turn. A hit with no semantic score
            # (_sem_sim is None — it surfaced via FTS/frequency, i.e. keyword or
            # usage relevance) is KEPT; only a present-but-weak cosine is cut.
            floor = self._CONVO_SIM_FLOOR
            if floor > 0.0:
                convo_hits = [
                    h for h in convo_hits
                    if h.get("_sem_sim") is None or h["_sem_sim"] >= floor
                ]
            # Re-inject only the USER's side of each past turn. The assistant's
            # own reply is dropped on retrieval so a confused/hallucinated turn
            # ("Cortex: the user é você", "minha configuração de perfil pessoal")
            # can't re-enter PAST CONTEXT and prime the 8B to echo it — the
            # conversation analogue of the _is_refusal store guard. Storage and
            # fact-extraction keep the full turn; only what we RE-INJECT changes.
            # Short-term continuity is covered by the in-memory history.
            #
            # Echo dedup (M3-A2): past copies of the CURRENT question, and
            # near-identical copies of each other, carry zero retrieval value
            # (the answer side was dropped above) — they only crowd the cap.
            seen_keys = {self._echo_key(current_query)}
            injected_ids: list[int] = []
            for h in convo_hits:
                if len(injected_ids) >= n_results:
                    break
                user_side = self._user_side(h["texto"])
                chunk = self._truncate(user_side, self._CHUNK_CHAR_CAP)
                key = self._echo_key(chunk)
                if not key or key in seen_keys:
                    continue
                if len(key) < self._MIN_USER_CHARS:
                    continue
                if "?" in user_side and self._skip_questions():
                    continue
                seen_keys.add(key)
                # Date only the emitted chunk: dedup/short-turn keys retain
                # their original unlabelled truncation, even across days.
                # The label spends the existing chunk and total budgets.
                label = self._conversation_date_label(h.get("created_at"))
                convo_chunks.append(self._truncate(label + user_side, self._CHUNK_CHAR_CAP))
                injected_ids.append(h["id"])
            # Frequency feedback (M3-A4): only the hits that SURVIVED the
            # floor + dedup — i.e. that actually enter the prompt — count as
            # used. The later char-budget truncation is a SIZE cut, not a
            # relevance cut, so it deliberately doesn't subtract from this.
            self._palace.mark_used(injected_ids)
        except Exception as e:
            log.warning(f"CortexMemory: conversation search failed: {e}")

        # Conversation fills the slack the knowledge left free — never the
        # other way around. With no knowledge (or include_knowledge=False)
        # the conversation keeps the full cap, as before.
        parts = []
        if knowledge_block:
            parts.append(knowledge_block)
        if convo_chunks:
            sep_cost = len("\n---\n") if knowledge_block else 0
            convo_budget = (self._CONTEXT_CHAR_CAP + label_room
                            - len(knowledge_block) - sep_cost)
            if convo_budget > 0:
                convo_block = self._truncate("\n---\n".join(convo_chunks), convo_budget)
                if convo_block:
                    parts.append(convo_block)

        if not parts:
            return ""
        merged = "\n---\n".join(parts)
        # Final guard: cap the total so TTFT stays bounded no matter how
        # many/large the chunks were.
        return self._truncate(merged, self._CONTEXT_CHAR_CAP + label_room)

    # --- helpers ----------------------------------------------------

    def _safe_embed(self, text: str):
        fn = self._palace.embed_fn
        if fn is None:
            return None
        try:
            return fn(text)
        except Exception as e:
            log.warning(f"CortexMemory: embedding failed ({e}); storing without vector")
            return None
