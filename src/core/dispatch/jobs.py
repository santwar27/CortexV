"""
Job Hunter regex dispatch — the on-demand "surface" (slice 4, s76).

The fetch + rank halves were CLI-only (scripts/fetch_jobs.py, rank_jobs.py).
This route lets the already-ranked top matches surface in the conversation —
text (/ask) AND voice, since both share the dispatch registry. It only READS
the local ledger (the top-N already scored by rank_jobs.py); re-ranking stays
an explicit CLI op (embedding 100s of jobs is too slow for a spoken turn).

Same regex-first pattern as fitness/weather: deterministic routing, no LLM
deciding whether this is a job query. Kept narrow + full-match anchored so a
conversational "tem vaga no estacionamento?" (a parking spot) can't hijack —
every PT branch carries a job-sense qualifier (nova/melhores/minhas/de emprego).
"""
from __future__ import annotations

import re


# "tem vaga nova?" / "minhas melhores vagas" / "vagas pra mim" / EN twins.
# Full-match anchored (^…[\s.?!]*$) so only the standalone question fires, not
# "minhas vagas estão ruins" mid-sentence. Each branch requires a job-sense
# word, so "tem vaga?" alone (a parking space) does NOT match.
#
# NB: the dispatch path matches the COURTESY-STRIPPED text (strip_courtesy), so
# "vagas pra mim" arrives here as just "vagas" — hence the bare-plural branch.
# Bare singular "vaga" is deliberately NOT accepted (too easily a parking spot).
JOBS_TOP_RE = re.compile(
    r"^(?:"
    # --- PT ---
    r"(?:tem|h[áa]|tem\s+alguma|alguma)\s+vagas?\s+nova?s?(?:\s+pra\s+mim)?|"
    r"vagas?\s+nova?s?(?:\s+pra\s+mim)?|"
    r"(?:quais\s+(?:s[ãa]o\s+)?(?:as\s+)?)?(?:as\s+|minhas\s+)?melhores\s+vagas"
    r"(?:\s+pra\s+mim)?|"
    r"(?:quais\s+(?:s[ãa]o\s+)?(?:as\s+)?)?minhas\s+vagas|"
    r"vagas?\s+(?:de\s+)?(?:emprego|trabalho)|"
    r"vagas?\s+pra\s+mim|"
    r"vagas|"     # bare plural (e.g. "vagas pra mim" after courtesy-strip)
    # --- EN ---
    r"any\s+(?:new\s+)?jobs?(?:\s+for\s+me)?|"
    r"(?:my\s+)?(?:top|best|new)\s+jobs?(?:\s+for\s+me)?|"
    r"jobs?\s+for\s+me|"
    r"job\s+matches"
    r")[\s.?!]*$",
    re.IGNORECASE,
)


# "re-rankeia minhas vagas" / "rankeia as vagas de novo" / "re-rank my jobs".
# A DIFFERENT intent from JOBS_TOP: this RE-SCORES the ledger against the stored
# résumé (slow — embeddings), so the handler runs it in the background and acks.
# Both regexes are full-match anchored and disjoint (TOP never starts with a
# rank verb), but this is registered FIRST so the rank verb always wins. Every
# branch carries "vagas"/"jobs" so a generic "rankeia ..." (a playlist, a list)
# can't hijack it.
JOBS_RERANK_RE = re.compile(
    r"^(?:"
    # --- PT (always says "vagas") ---
    r"re-?rankei[ao]\s+(?:as\s+|minhas\s+|todas\s+as\s+)?vagas(?:\s+de\s+novo|\s+novamente)?|"
    r"rankei[ao]\s+(?:as\s+|minhas\s+|todas\s+as\s+)?vagas(?:\s+de\s+novo|\s+novamente)?|"
    r"re-?rankear\s+(?:as\s+|minhas\s+)?vagas|"
    r"recalcula(?:r)?\s+(?:as\s+|minhas\s+)?vagas|"
    r"atualiza(?:r)?\s+o\s+ranking\s+das\s+vagas|"
    # --- EN (always says "jobs") ---
    r"re-?rank\s+(?:my\s+|the\s+|all\s+)?jobs(?:\s+again)?|"
    r"rank\s+(?:my\s+|the\s+|all\s+)?jobs(?:\s+again)?|"
    r"re-?score\s+(?:my\s+|the\s+)?jobs|"
    r"refresh\s+(?:my\s+|the\s+)?job\s+ranking"
    r")[\s.?!]*$",
    re.IGNORECASE,
)
