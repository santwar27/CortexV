"""
DeepStudier — multi-step research, ahead of the hardware. the reference "Deep web
research" (see ideas-backlog); scaffold #2 of the build-now-wire-later batch.

The NightlyStudier does ONE round per topic: queries -> search -> synthesis.
The DeepStudier does SEVERAL chained rounds under a budget: it decomposes a
topic into sub-questions, researches each, tracks citations across steps, and
weaves the findings into one deep briefing.

Two layers, by design (same split as the Cognitive Arbiter):

  • Layer 1 — BUILT NOW, deterministic orchestration, fully testable with a
    mocked engine/searcher: the planner, the per-step research loop, the
    SEARCH BUDGET (so it can never fan out into an unbounded number of Tavily
    calls), the citation aggregation, and the save format.

  • Layer 2 — DORMANT, model-bound: the synthesis QUALITY. _synthesise_deep
    runs today on the 8B (shallow), and shines on Gemma 4. Nothing here is
    blocked from running — it's just not wired into any live command yet.
    The whole feature sits behind CORTEX_DEEP_STUDIER (default off); the
    orchestrator does NOT route to it until Mac Studio day = flip the flag +
    add the "pesquise profundamente X" dispatch + point at the strong model.

Reuse over duplication: DeepStudier *composes* a NightlyStudier and reuses
its already-hardened primitives (`generate_queries`, `gather_snippets`,
`is_ml_topic`, the blocklist) so the deep loop only owns what is genuinely
new — planning, the budgeted loop, citations, and the deep synthesis.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import ollama
from src.core.ollama_client import KEEP_ALIVE as _KEEP_ALIVE, THINK as _THINK

from src.core.audit import audit
from src.core.engine import OllamaEngine
from src.core.studier import NightlyStudier
from src.core.study_time import EVENT_TIME_RULES, StudyTime

log = logging.getLogger(__name__)


def _deep_enabled() -> bool:
    """True when the DeepStudier feature is ON. DORMANT default: off until
    Mac Studio. Guards the (future) orchestrator dispatch, not this module's
    own logic — research() always works, it just isn't reachable live yet."""
    return os.environ.get("CORTEX_DEEP_STUDIER", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def deep_queue():
    """The deep-research topic queue (M8-A2). 'pesquise a fundo X' ENQUEUES
    here and returns immediately — 15 searches + N syntheses never run
    inline in the request thread (they'd block every other route for
    minutes, M1-A6/M5-A3). The NightlyStudier consumes this queue inside
    its governed lease window. Reuses StudyQueue, so deep topics get the
    same transient/duplicate validation as regular study topics."""
    from src.core.study_queue import StudyQueue
    return StudyQueue(
        path=os.environ.get("CORTEX_DEEP_QUEUE", "data/deep_queue.jsonl")
    )


@dataclass
class ResearchStep:
    """One sub-question and what researching it produced."""
    subquestion: str
    queries: List[str]
    snippet_chars: int
    finding: str
    sources: List[str] = field(default_factory=list)


@dataclass
class DeepStudyReport:
    """The full result of a deep study: the plan, every step, the woven
    briefing, and the deduped citation list across all steps."""
    topic: str
    subquestions: List[str]
    steps: List[ResearchStep]
    briefing: str
    sources: List[str] = field(default_factory=list)


class DeepStudier:
    """Budgeted multi-step researcher composed over a NightlyStudier."""

    MIN_SUBQUESTIONS = 2
    MAX_SUBQUESTIONS = 5
    # Hard ceiling on total web searches across ALL steps. A web search has
    # cost/rate limits (Tavily), so the loop must never fan out unbounded.
    MAX_SEARCHES_TOTAL = 15
    # Per-step query cap: deep research goes WIDE across sub-questions, not
    # deep on any single one. MAX_SUBQUESTIONS * STEP_MAX_QUERIES == budget.
    STEP_MAX_QUERIES = 3
    STEP_SNIPPET_LIMIT = 6_000
    DEEP_SNIPPET_LIMIT = 12_000

    # URLs in Tavily-style snippets appear as "[https://...]".
    _URL_RE = re.compile(r"\[(https?://[^\]\s]+)\]")

    def __init__(
        self,
        studier: Optional[NightlyStudier] = None,
        engine: Optional[OllamaEngine] = None,
        output_dir: str = "data/knowledge/deep",
    ):
        # Own directory, NOT data/knowledge/pending (M8-A4): digest, daily
        # and studies_review glob pending/*.md blindly — a deep study dropped
        # there would show up as "estudo novo do Studier" no briefing e entrar
        # na fila de review sem nenhum consumidor distinguir o prefixo deep_.
        # Surfacing/review de deep studies é wiring do dia do flip (runbook).
        self.studier = studier or NightlyStudier(engine=engine, output_dir=output_dir)
        self.engine = engine or self.studier.engine
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # -- public API --------------------------------------------------------

    def research(self, topic: str, trace: Optional[str] = None) -> DeepStudyReport:
        """Plan -> budgeted research loop -> deep synthesis. Pure of I/O
        except the (mockable) engine + searcher and audit logging. Raises
        ValueError on an empty topic (fail fast instead of an error report)."""
        topic = (topic or "").strip()
        if not topic:
            raise ValueError("deep study topic is required")
        trace = trace or audit.new_trace()
        audit.log(trace, "deep_studier", "start", payload={"topic": topic})

        # M8-A3: the deep loop inherits the Studier's WebSearcher, which
        # has no gate of its own — without this, one command fans out into
        # up to MAX_SEARCHES_TOTAL ungoverned searches. Same decision as
        # the nightly (P-2/M6-A2): a scoped, bounded, audited SystemLease
        # for the duration of the run, with max_calls pinned to the search
        # budget so the lease and the budget can never drift apart. When
        # called FROM the nightly the searcher is already governed — the
        # second wrapper just means both leases count (defence in depth).
        # gather_snippets treats a denied search like a failed query, so
        # an exhausted lease degrades to thinner findings, never a crash.
        from src.core.system_lease import GovernedWebSearcher, SystemLease

        raw_searcher = getattr(self.studier, "web_searcher", None)
        lease = SystemLease.acquire(
            scope="web_search",
            origin="deep-studier",
            minutes=30,
            max_calls=self.MAX_SEARCHES_TOTAL,
        )
        if raw_searcher is not None:
            self.studier.web_searcher = GovernedWebSearcher(raw_searcher, lease)
        try:
            return self._research_governed(topic, trace)
        finally:
            if raw_searcher is not None:
                self.studier.web_searcher = raw_searcher
            lease.release()

    def _research_governed(self, topic: str, trace: str) -> DeepStudyReport:
        """The research loop proper — runs with the studier's searcher
        already swapped for the governed wrapper by research()."""
        study_time = StudyTime.for_topic(topic)
        subquestions = self._plan_subquestions(topic, study_time=study_time)
        audit.log(trace, "deep_studier", "plan",
                  payload={"n": len(subquestions), "subquestions": subquestions})

        steps: List[ResearchStep] = []
        searches_used = 0
        off_topic_dropped = 0
        for subq in subquestions:
            remaining = self.MAX_SEARCHES_TOTAL - searches_used
            if remaining <= 0:
                log.info("DeepStudier: search budget exhausted at %d", searches_used)
                audit.log(trace, "deep_studier", "budget_exhausted",
                          payload={"searches_used": searches_used})
                break

            ml = self.studier.is_ml_topic(subq)
            queries = self.studier.generate_queries(subq, study_time=study_time)
            # Trim to the per-step cap AND the remaining global budget.
            queries = queries[: min(self.STEP_MAX_QUERIES, remaining)]
            if not queries:
                continue

            filter_counts = {}
            # The real studier checks the ORIGINAL topic, so a drifting
            # planner cannot make an unrelated subquestion its own evidence.
            # Preserve the minimal gather API of injected studier adapters.
            if isinstance(self.studier, NightlyStudier):
                snippets = self.studier._gather_for_topic(
                    queries, topic=topic, ml_context=ml, filter_counts=filter_counts,
                )
            else:
                snippets = self.studier.gather_snippets(queries, ml_context=ml)
            searches_used += len(queries)
            off_topic_dropped += filter_counts.get("off_topic_dropped", 0)
            sources = self._extract_sources(snippets)
            finding = (
                self._summarise_step(subq, snippets, ml_context=ml, study_time=study_time)
                if snippets.strip() and sources else ""
            )
            steps.append(ResearchStep(
                subquestion=subq, queries=queries,
                snippet_chars=len(snippets), finding=finding, sources=sources,
            ))
            audit.log(trace, "deep_studier", "step_done", payload={
                "subq": subq[:60], "queries": len(queries),
                "sources": len(sources), "searches_used": searches_used,
                **filter_counts,
            })

        all_sources = self._dedup(
            [src for step in steps for src in step.sources]
        )
        if not all_sources:
            reason = "all_off_topic" if off_topic_dropped else "no_sources"
            log.warning("DeepStudier: no sources kept for '%s' (%s)", topic, reason)
            audit.log(trace, "deep_studier", "no_sources",
                      payload={"topic": topic, "reason": reason})
            briefing = "## Erro\nNenhuma fonte de pesquisa foi mantida para síntese."
        else:
            briefing = self._synthesise_deep(topic, steps, study_time=study_time)
            audit.log(trace, "deep_studier", "synthesis_done",
                      payload={"briefing_chars": len(briefing), "n_steps": len(steps)})

        return DeepStudyReport(
            topic=topic, subquestions=subquestions, steps=steps,
            briefing=briefing, sources=all_sources,
        )

    def run_and_save(self, topic: str, trace: Optional[str] = None) -> str:
        """research() + persist the report to output_dir. Returns the path."""
        report = self.research(topic, trace=trace)
        return self._save(report)

    # -- Layer 1: planning + citations (deterministic / mockable) ----------

    def _plan_subquestions(self, topic: str, *,
                           study_time: Optional[StudyTime] = None) -> List[str]:
        """Decompose the topic into 2-5 focused sub-questions. Falls back to
        the literal topic if the LLM fails or returns nothing usable."""
        study_time = study_time or StudyTime.for_topic(topic)
        prompt = (
            "Break the research topic below into "
            f"{self.MIN_SUBQUESTIONS} to {self.MAX_SUBQUESTIONS} focused "
            "sub-questions that, answered together, would cover it deeply. "
            "Return ONLY the sub-questions, one per line, no numbering, no "
            "explanation.\n\n"
            + study_time.prompt_block() + "\n"
            "RULES:\n"
            "- Each sub-question is a distinct angle (definition, mechanism, "
            "trade-offs, alternatives, real-world use). No duplicates.\n"
            "- Do NOT expand acronyms or invent terms — use the topic's exact "
            "wording.\n"
            "- Keep technical terms in English.\n\n"
            "- Keep the topic's time period in every sub-question.\n\n"
            f"Topic: {topic}\n"
            "Sub-questions:"
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 300},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M8-A6: scaffold must not downgrade the timer
            )
        except Exception as e:  # noqa: BLE001 — degrade, never crash a run
            log.warning("DeepStudier: planning failed (%s); using literal topic", e)
            return [topic] if topic else []

        text = (response.get("response") or "").strip()
        subs: List[str] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•*").strip()
            line = re.sub(r"^\d+[.)]\s*", "", line).strip().strip('"')
            if 5 < len(line) < 200:
                subs.append(line)
        subs = self._dedup(subs)[: self.MAX_SUBQUESTIONS]
        return subs or ([topic] if topic else [])

    def _extract_sources(self, snippets: str) -> List[str]:
        """Pull every [https://...] URL out of the snippet block, deduped."""
        return self._dedup(self._URL_RE.findall(snippets or ""))

    @staticmethod
    def _dedup(items: List[str]) -> List[str]:
        """Order-preserving, case-sensitive dedup (used for both URLs and
        sub-questions)."""
        seen: set[str] = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    # -- Layer 2: synthesis (model-bound; shines on Gemma 4) ---------------

    def _summarise_step(self, subquestion: str, snippets: str,
                        ml_context: bool = False, *,
                        study_time: Optional[StudyTime] = None) -> str:
        """Concise answer to ONE sub-question from its snippets. Model-bound
        quality (8B now, Gemma 4 later); structurally part of the loop."""
        if len(snippets) > self.STEP_SNIPPET_LIMIT:
            snippets = snippets[: self.STEP_SNIPPET_LIMIT] + "\n\n[truncated]"
        study_time = study_time or StudyTime.for_topic(subquestion)
        prompt = (
            "Answer the sub-question below using ONLY the web snippets. Be "
            "concise (3-6 sentences), concrete, and do NOT fabricate facts. "
            "Keep technical terms in English; write prose in the question's "
            "language (PT-BR or English).\n\n"
            + study_time.prompt_block() + EVENT_TIME_RULES + "\n"
            f"Sub-question: {subquestion}\n\n"
            f"Snippets:\n---\n{snippets}\n---\n"
            "Answer:"
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 400},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M8-A6
            )
            return (response.get("response") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("DeepStudier: step synthesis failed (%s)", e)
            return ""

    def _synthesise_deep(self, topic: str, steps: List[ResearchStep], *,
                         study_time: Optional[StudyTime] = None) -> str:
        """Weave the per-step findings into one deep briefing. THE model-bound
        part — quality is why this whole thing waits for Gemma 4. Builds the
        prompt from the structured findings (no raw snippets re-fed)."""
        if not steps:
            return "## Erro\nNenhum passo de pesquisa produziu resultados."
        study_time = study_time or StudyTime.for_topic(topic)

        findings_block = "\n\n".join(
            f"### Sub-pergunta: {s.subquestion}\n{s.finding or '(sem achados)'}"
            for s in steps
        )
        if len(findings_block) > self.DEEP_SNIPPET_LIMIT:
            findings_block = findings_block[: self.DEEP_SNIPPET_LIMIT] + "\n\n[truncated]"
        all_sources = self._dedup(
            [src for step in steps for src in step.sources]
        )
        sources_block = "\n".join(f"- [{u}]" for u in all_sources) or "- (nenhuma)"

        prompt = (
            "You are CORTEX producing a DEEP research briefing. You are given "
            "answers to several sub-questions about one topic; synthesise them "
            "into a single coherent report. Connect the findings — do not just "
            "concatenate them.\n\n"
            + study_time.prompt_block() + EVENT_TIME_RULES + "\n"
            f"Topic: {topic}\n\n"
            f"Findings per sub-question:\n---\n{findings_block}\n---\n\n"
            f"All sources gathered:\n{sources_block}\n\n"
            "Output STRICTLY in this Markdown format:\n\n"
            "## Resumo executivo\n<3-6 sentences tying the findings together>\n\n"
            "## Pontos-chave\n- <concrete facts; group by sub-question theme>\n\n"
            "## Fontes consultadas\n- <copy each FULL URL above, verbatim>\n\n"
            "## Confiança\n<alta | média | baixa> — <1-sentence justification>\n\n"
            "RULES:\n"
            "- Stick to the findings. Do NOT invent facts, numbers, or dates.\n"
            "- Keep technical terms / library / model names in English.\n"
            "- If findings disagree, flag it in Pontos-chave.\n"
            "- Produce exactly ONE report."
        )
        try:
            response = ollama.generate(
                model=self.engine.model_name,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 2000},
                keep_alive=_KEEP_ALIVE, think=_THINK,  # M8-A6
            )
            return (response.get("response") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.exception("DeepStudier: deep synthesis failed: %s", e)
            return f"## Erro\nFalha ao gerar briefing profundo: {e}"

    # -- persistence -------------------------------------------------------

    def _save(self, report: DeepStudyReport) -> str:
        slug = NightlyStudier._slugify(report.topic)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"deep_{slug}_{ts}.md")

        plan_block = "\n".join(f"- {q}" for q in report.subquestions)
        header = (
            f"# Deep study: {report.topic}\n\n"
            f"*Generated by CORTEX DeepStudier on "
            f"{datetime.now(timezone.utc).isoformat()}*\n\n"
            f"**Sub-questions researched ({len(report.steps)} steps):**\n"
            f"{plan_block}\n\n---\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + report.briefing + "\n")
        return path


# No module-level singleton on purpose: building one eagerly would construct
# a NightlyStudier (hence an OllamaEngine) at import time. Mirrors studier.py
# — instantiate DeepStudier() where it's used (behind CORTEX_DEEP_STUDIER).
