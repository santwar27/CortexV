"""
CuriosityEngine — the REFLECT edge that closes the study loop.

Design: docs/design/autonomous_self_improvement.md (Parte A.2/A.3, step 6).
Today the loop is OPEN: Pedro seeds every topic ("estuda sobre X" →
StudyQueue.add) and nothing re-feeds the queue. This module adds the missing
edge: after a briefing is produced, reflect on what was learned, propose 1-3
follow-up topics, score them against the study charter, and re-enqueue the
best ones — so the queue never dries up, while the HUMAN GATE stays exactly
where it was (briefings land in data/knowledge/pending/; nothing enters the
permanent RAG without approval, and every auto-proposed queue entry carries
origin="curiosity" provenance so it can be vetoed by deleting the line).

Two layers, same split as the Arbiter/DeepStudier scaffolds:

  • Layer 1 — BUILT NOW, deterministic and fully testable with an injected
    generate_fn: charter parsing, candidate parsing, relevance × novelty
    scoring, the hard cap per round, transient/duplicate rejection (reuses
    StudyQueue's own guards) and the provenance-tagged enqueue.

  • Layer 2 — DORMANT, model-bound: the QUALITY of the proposed topics.
    The default generator runs on the local analytical model (shallow on
    the 8B, shines on the Mac Studio model). The whole edge sits behind
    CORTEX_CURIOSITY (default off): the NightlyStudier only calls reflect()
    when the flag is on, so tonight's behaviour is byte-identical until
    Mac Studio day = flip the flag.

Safeguards wired in (doc §A.7):
  - charter required: no data/study_charter.md with goals → the engine
    refuses to run (curiosity without an anchor drifts into trivia);
  - hard cap of proposals per reflection (CORTEX_CURIOSITY_MAX, default 2);
  - novelty via StudyQueue.find_similar + the queue's own transient/dup
    ValueError guards (a proposal is never force-added);
  - global queue cap (CORTEX_CURIOSITY_QUEUE_CAP, default 10): reflection
    refuses to enqueue when the pending queue is already at the cap — the
    per-reflection cap alone would still allow compounding growth across
    nights (6 topics → 12 proposals → 24 …);
  - proposals enqueued during a nightly run are NOT studied that same
    night — NightlyStudier.run() snapshots the queue via consume_all()
    before studying, so auto-proposed topics wait for the next round.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import ollama

from src.core.audit import audit
from src.core.ollama_client import KEEP_ALIVE as _KEEP_ALIVE, THINK as _THINK
from src.core.study_queue import StudyQueue

log = logging.getLogger(__name__)

DEFAULT_CHARTER_PATH = "data/study_charter.md"


def _curiosity_enabled() -> bool:
    """True when the curiosity edge is ON. DORMANT default: off until Mac
    Studio. Guards the NightlyStudier wiring, not this module's own logic —
    reflect() always works, it just isn't reachable from the nightly run."""
    return os.environ.get("CORTEX_CURIOSITY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Minimal PT+EN stopword set for the relevance overlap. Deliberately tiny:
# the charter is keyword-dense, so we only need to stop glue words from
# producing fake overlap ("de", "the"), not do real NLP.
_STOPWORDS = frozenset(
    "a o e os as um uma de da do das dos em no na nos nas com para por que "
    "como sobre entre sem mais menos seu sua meu minha "
    "the and for with from into onto over under about between without "
    "how what when why which this that these those its our your "
    "vs versus of in on to at is are be an".split()
)


def _tokens(text: str) -> frozenset:
    """Accent-stripped lowercase alnum tokens minus stopwords, so
    'programação' matches 'programacao' and glue words never count."""
    norm = unicodedata.normalize("NFD", (text or "").lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    # Plain alnum runs: hyphens/underscores split ('fine-tuning' ≡
    # 'fine tuning', 'keep_alive' ≡ 'keep alive'), mirroring the
    # StudyQueue's own _normalise_for_match behaviour.
    words = re.findall(r"[a-z0-9]+", norm)
    return frozenset(w for w in words if len(w) >= 2 and w not in _STOPWORDS)


def load_charter(path: Optional[str] = None) -> List[str]:
    """Read the study charter and return its goal lines.

    Format: plain Markdown; every non-empty line that isn't a heading (#),
    a blockquote (>) or an HTML comment counts as one goal. Bullets are
    stripped. Missing/empty file → [] (the engine then refuses to run)."""
    path = path or os.environ.get("CORTEX_STUDY_CHARTER", DEFAULT_CHARTER_PATH)
    if not os.path.exists(path):
        return []
    goals: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ">", "<!--")):
                continue
            line = re.sub(r"^[-*+]\s+", "", line).strip()
            if line:
                goals.append(line)
    return goals


@dataclass
class TopicProposal:
    """One candidate follow-up topic with its deterministic scores."""
    topic: str
    relevance: float  # charter-token overlap, 0..1
    novelty: float    # 1.0 unless the queue already knows something similar

    @property
    def score(self) -> float:
        return self.relevance * self.novelty


class CuriosityEngine:
    """Reflects on a finished study and re-enqueues follow-up topics."""

    # At most this many raw candidates are considered per reflection, no
    # matter how chatty the model gets.
    MAX_CANDIDATES = 5
    DEFAULT_MAX_PROPOSALS = 2

    # Sanity bounds for a single proposed topic string.
    _TOPIC_MIN_CHARS = 4
    _TOPIC_MAX_CHARS = 160

    # Briefings can be long; the reflection prompt must fit num_ctx=8192
    # alongside its instructions, so the briefing body is capped.
    BRIEFING_CHAR_CAP = 3000

    _REFLECT_PROMPT = (
        "You are CORTEX reflecting on a study you just completed, to decide "
        "what to study next.\n\n"
        "Topic studied: {topic}\n\n"
        "Briefing produced:\n---\n{briefing}\n---\n\n"
        "Propose 1-3 FOLLOW-UP study topics: open questions, gaps or "
        "adjacent concepts this briefing exposes. Stay at the EDGE of what "
        "was just learned — not trivially contained in the briefing, not "
        "unrelated to it.\n"
        "Rules:\n"
        "- one topic per line, nothing else (no bullets, numbering, prose)\n"
        "- each topic under 12 words, in the briefing's language\n"
        "- stable concepts only — never news, prices, weather or events"
    )

    def __init__(
        self,
        queue: Optional[StudyQueue] = None,
        charter_path: Optional[str] = None,
        max_proposals: Optional[int] = None,
        queue_cap: Optional[int] = None,
        generate_fn: Optional[Callable[[str], str]] = None,
        model_name: Optional[str] = None,
    ):
        self.queue = queue or StudyQueue()
        self.charter_path = charter_path or os.environ.get(
            "CORTEX_STUDY_CHARTER", DEFAULT_CHARTER_PATH
        )
        if max_proposals is None:
            max_proposals = int(os.environ.get("CORTEX_CURIOSITY_MAX", "2"))
        self.max_proposals = max(0, max_proposals)
        if queue_cap is None:
            queue_cap = int(os.environ.get("CORTEX_CURIOSITY_QUEUE_CAP", "10"))
        self.queue_cap = max(0, queue_cap)
        # Injectable for tests / the Mac Studio wiring; None → lazy local
        # ollama on the analytical model (dormant quality — 8B is shallow).
        self.generate_fn = generate_fn
        self.model_name = model_name or os.getenv(
            "CORTEX_MODEL_ANALYTICAL", "cortex-lite"
        )

    # --- the REFLECT step ---------------------------------------------

    def reflect(
        self, topic: str, briefing: str, trace: Optional[str] = None
    ) -> List[Dict]:
        """Propose follow-up topics for a finished study and enqueue the
        best ones. Returns the queue entries created (possibly empty).
        Never raises: every failure degrades to 'no proposals tonight'."""
        trace = trace or audit.new_trace()
        audit.log(trace, "curiosity", "reflect_start",
                  payload={"parent_topic": topic})

        goals = load_charter(self.charter_path)
        if not goals:
            # Safeguard A.7#3: curiosity without an anchor drifts. Refuse.
            log.warning(
                "Curiosity: no charter at %s — refusing to propose topics.",
                self.charter_path,
            )
            audit.log(trace, "curiosity", "no_charter",
                      payload={"path": self.charter_path})
            return []
        # Relevance anchor = charter ∪ PARENT TOPIC tokens. First live run
        # (2026-07-10) rejected every genuine candidate as "irrelevant":
        # follow-ups naturally speak the parent's vocabulary ("fixture",
        # "monkeypatch"), not the charter's ("pytest", "testing"). The
        # parent was itself charter-vetted, so extending the anchor to it
        # doesn't open drift — it IS the "borda do que já sabe" (doc A.3.1).
        anchor_tokens = (
            frozenset().union(*(_tokens(g) for g in goals)) | _tokens(topic)
        )

        # Global backpressure: if the pending queue is already at the cap,
        # don't even spend a model call. Bounds compounding growth across
        # nights — without this, every studied topic could add 2 more
        # forever (the per-reflection cap alone doesn't stop that).
        pending_now = len(self.queue.list_pending())
        if pending_now >= self.queue_cap:
            log.info(
                "Curiosity: queue at cap (%d ≥ %d) — skipping reflection.",
                pending_now, self.queue_cap,
            )
            audit.log(trace, "curiosity", "queue_full",
                      payload={"pending": pending_now,
                               "cap": self.queue_cap})
            return []

        candidates = self._parse_candidates(self._generate(topic, briefing))
        if not candidates:
            audit.log(trace, "curiosity", "no_candidates")
            return []
        audit.log(trace, "curiosity", "candidates",
                  payload={"topics": candidates})

        proposals: List[TopicProposal] = []
        for cand in candidates:
            relevance = self._relevance(cand, anchor_tokens)
            if relevance <= 0:
                audit.log(trace, "curiosity", "skipped",
                          payload={"topic": cand, "reason": "irrelevant"})
                continue
            if self.queue.find_similar(cand):
                audit.log(trace, "curiosity", "skipped",
                          payload={"topic": cand, "reason": "duplicate"})
                continue
            proposals.append(TopicProposal(cand, relevance, 1.0))
        proposals.sort(key=lambda p: p.score, reverse=True)

        enqueued: List[Dict] = []
        for prop in proposals[: self.max_proposals]:
            try:
                entry = self.queue.add(
                    prop.topic,
                    extra={
                        "origin": "curiosity",
                        "parent_topic": topic,
                        "score": round(prop.score, 3),
                    },
                )
            except ValueError as e:
                # The queue's own transient/duplicate guards have the final
                # word — a proposal is never force-added ('!!').
                audit.log(trace, "curiosity", "skipped",
                          payload={"topic": prop.topic,
                                   "reason": str(e)[:120]})
                continue
            enqueued.append(entry)
            audit.log(trace, "curiosity", "enqueued",
                      payload={"topic": prop.topic, "parent_topic": topic,
                               "score": round(prop.score, 3)})

        audit.log(trace, "curiosity", "reflect_done",
                  payload={"n_enqueued": len(enqueued)})
        return enqueued

    # --- internals -----------------------------------------------------

    def _generate(self, topic: str, briefing: str) -> str:
        """Ask the model for follow-up topics. Injected fn wins; default is
        the local analytical model (Layer 2 — quality is model-bound)."""
        body = (briefing or "")[: self.BRIEFING_CHAR_CAP]
        prompt = self._REFLECT_PROMPT.format(topic=topic, briefing=body)
        try:
            if self.generate_fn is not None:
                return self.generate_fn(prompt) or ""
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.4, "num_predict": 150},
                keep_alive=_KEEP_ALIVE, think=_THINK,
            )
            return (response.get("response") or "").strip()
        except Exception:
            log.exception("Curiosity: topic generation failed (non-fatal)")
            return ""

    @classmethod
    def _parse_candidates(cls, raw: str) -> List[str]:
        """One topic per line → cleaned, deduped, bounded list."""
        seen = set()
        out: List[str] = []
        for line in (raw or "").splitlines():
            line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", line)
            # LLM preamble ("Aqui estão minhas sugestões:") leaks as a
            # line ending in ':' — seen live 2026-07-10; never a topic.
            if line.rstrip().endswith(":"):
                continue
            line = line.strip("\"'. ")
            if not (cls._TOPIC_MIN_CHARS <= len(line) <= cls._TOPIC_MAX_CHARS):
                continue
            key = " ".join(sorted(_tokens(line)))
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(line)
            if len(out) >= cls.MAX_CANDIDATES:
                break
        return out

    @staticmethod
    def _relevance(candidate: str, charter_tokens: frozenset) -> float:
        """Charter-token overlap ratio: |cand ∩ charter| / |cand|, 0..1.
        Zero overlap means the topic has no anchor in Pedro's goals."""
        cand_tokens = _tokens(candidate)
        if not cand_tokens:
            return 0.0
        return len(cand_tokens & charter_tokens) / len(cand_tokens)
