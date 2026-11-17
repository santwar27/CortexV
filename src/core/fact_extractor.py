"""
Passive fact extraction — design-reference ch. 16 ("maybeExtractFacts").

A cheap, deterministic regex pass over each user turn that captures stable
PROFILE facts the user states in passing — their name, where they live,
what they do — and persists them to the Palace without asking. It
complements `src/memory/preferences.py`, which runs a heavier LLM pass over
the nightly batch: this one is per-turn, costs ~0 (no model call), and only
goes after high-signal, low-ambiguity profile statements.

Design bias: FALSE POSITIVES ARE THE ENEMY. A regex that mis-captures
"sou capaz de fazer isso" as identity="capaz de fazer isso" poisons the
profile. So the patterns are narrow, negations are rejected, questions are
skipped entirely, and captured values are trimmed at clause boundaries and
length-capped. We would rather miss a fact than store a wrong one — the
nightly LLM extractor is the recall safety net.

The module is pure (no I/O, no Palace) so it unit-tests trivially. The
caller (`CortexMemory.store_interaction`) owns persistence and the
no-downgrade guard.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

from src.core.redaction import is_sensitive_attr


@dataclass(frozen=True)
class ExtractedFact:
    """One captured profile fact. `predicado`/`objeto` map onto a Palace
    tripla (sujeito is always 'user'); `evidence` is the trimmed snippet
    that produced it, kept for the audit trail."""
    predicado: str
    objeto: str
    evidence: str


# Confidence stamped on every passively-extracted fact. Deliberately below
# an explicitly-stated fact (1.0) and below the nightly LLM extractor's
# typical scores, so the no-downgrade guard in the persistence layer never
# lets a regex hit clobber a stronger source.
PASSIVE_CONFIDENCE = 0.6

# Source tag written to the Palace `source` column. Makes passive rows
# filterable and auditable, and lets the no-downgrade guard recognise its
# own writes.
PASSIVE_SOURCE = "passive_regex"

# Only these predicates are stable + high-value enough to splice into the
# system prompt. Everything else the extractor captures (generic "meu X é
# Y") is still STORED — queryable, promotable by the nightly consolidator —
# but not auto-injected, which bounds prompt growth and blast radius.
#
# Curated, not open-ended (s31 cont. widened it from the original 3): a slug
# is added here only when it's (a) STABLE over time, (b) high personalisation
# value, (c) low harm if slightly wrong. Birthday/team/sign/favourites
# qualify; volatile attrs (idade, humor) or anything sensitive deliberately
# do NOT — they stay stored-but-not-surfaced. The generic "meu/minha X é Y"
# pattern already produces these slugs; this only decides which get injected.
PROFILE_INJECT_KEYS = (
    # core identity (original three)
    "nome", "mora_em", "profissao",
    # stable personalisation facts (surfaced since s31 cont.)
    "aniversario", "time", "signo", "hobby",
    "cor_favorita", "comida_favorita",
)

# Value capture stops at the first clause boundary so we don't swallow a
# whole sentence into a "fact". e.g. "moro em São Paulo mas quero mudar"
# captures "São Paulo", not the rest. Conjunctions ("e"/"ou") are
# boundaries too, so a turn that chains facts ("me chamo the user e trabalho
# como X") doesn't let the first pattern's greedy capture swallow the
# second — the small cost is clipping a genuinely compound value like
# "preto e branco" to "preto", which we accept to avoid run-on captures.
# Temporal markers (desde / há / faz / desde que) also end a value: they
# never begin part of a fact ("torço pelo Flamengo desde criança" → the team
# is "Flamengo"; "vivo em Londres faz 5 anos" → "Londres"). Added s32 cont.
# for the natural-phrasing patterns; harmless to the existing ones.
_BOUNDARY_RE = re.compile(
    r"\s+(?:e|ou|mas|porém|porem|porque|pois|que|então|entao|desde|há|ha|faz)\s+"
    r"|[.,;:!?]"
)

# A captured value longer than this (after trimming) is almost certainly a
# run-on mis-capture, so we drop it.
_MAX_VALUE_CHARS = 50
_MAX_VALUE_WORDS = 6

# Tokens that signal the "value" is actually filler / not a real fact.
_FILLER_VALUES = {
    "", "que", "o", "a", "um", "uma", "isso", "aquilo", "aqui", "ali",
    "muito", "meio", "bem", "tipo", "assim", "tão", "tao",
}

# Text the user hands over to be worked on is not the user speaking (s88, Codex
# lote 6). 'Traduza: "moro em Paris"' wrote mora_em=Paris, and 'Revisa esse
# texto: "Meu nome é Ana e moro em Porto."' put "nome → Ana, mora_em → Porto"
# in the profile every prompt carries. Quoted spans are dropped; a request to
# translate, fix, revise or exemplify skips the turn — losing a fact said in
# passing is the conservative side, inventing one is not.
_QUOTED_RE = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»')
# Shared with hallucination.asks_for_rewording: the same sentence is not an
# action Cortex claims either (s88, Codex lote 8).
MATERIAL_CUE_RE = re.compile(
    r"\b(?:tradu[zç]\w*|translat\w*|corrig\w*|corrij\w*|revis[ae]\w*|"
    r"reescrev\w*|rewrite|proofread\w*|parafrase\w*|paraphras\w*|"
    r"gram[aá]tica|grammar|exemplo|example|como se diz|how do (?:i|you) say)\b",
    re.IGNORECASE,
)

# High-signal patterns. Each captures the value in group 'v'. The negative
# lookbehinds reject a negated statement ("não moro em SP"); both "não " and
# "nao " are 4 chars wide, which Python's fixed-width lookbehind requires.
_NEG = r"(?<!não )(?<!nao )"

_PATTERNS = [
    # name: "meu nome é X" / "me chamo X" / "pode me chamar de X".
    # _NEG on each alternative — "não me chamo the user" must not capture.
    (re.compile(rf"\b(?:{_NEG}meu nome (?:é|eh)|{_NEG}me chamo|{_NEG}pode me chamar de)\s+(?P<v>.+)",
                re.IGNORECASE), "nome"),
    # lives in: "(eu) moro em/no/na X"
    (re.compile(rf"\b{_NEG}(?:eu\s+)?moro\s+(?:em|no|na|nos|nas)\s+(?P<v>.+)",
                re.IGNORECASE), "mora_em"),
    # profession: "(eu) trabalho como/de X"
    (re.compile(rf"\b{_NEG}(?:eu\s+)?trabalho\s+(?:como|de)\s+(?P<v>.+)",
                re.IGNORECASE), "profissao"),
    # football team: "torço pro/pelo/para o X". The verb 'torcer' is
    # unambiguous (unlike "sou X-ista/ense", which collides with profession
    # 'jornalista' / origin 'paulista' / sign 'capricorniano' — deliberately
    # NOT matched). Surfaced (predicate 'time' is whitelisted).
    (re.compile(rf"\b{_NEG}tor[çc]o\s+(?:pro|pelo|pra|para\s+o)\s+(?P<v>.+)",
                re.IGNORECASE), "time"),
    # lives in (synonym of 'moro em'): "(eu) vivo em/no/na X"
    (re.compile(rf"\b{_NEG}(?:eu\s+)?vivo\s+(?:em|no|na|nos|nas)\s+(?P<v>.+)",
                re.IGNORECASE), "mora_em"),
    # age: "tenho N anos" → idade. Tight (digits only) and rejects
    # "tenho 2 anos DE experiência/idade/casa" via the lookahead. Stored but
    # NOT surfaced (idade is volatile — off PROFILE_INJECT_KEYS on purpose).
    (re.compile(rf"\b{_NEG}tenho\s+(?P<v>\d{{1,3}})\s+anos?\b(?!\s+de\b)",
                re.IGNORECASE), "idade"),
]

# Generic "meu/minha <attr> é/eh <value>" — the reference's "meu X é Y" pattern.
# attr becomes the predicate (slugified); 1-3 words keeps it sane.
_GENERIC_RE = re.compile(
    r"\b(?:meu|minha)\s+(?P<attr>[a-zà-úç]+(?:\s+[a-zà-úç]+){0,2})\s+"
    r"(?:é|eh)\s+(?P<v>.+)",
    re.IGNORECASE,
)

# Generic attrs that are handled by a dedicated pattern or are non-facts.
_GENERIC_ATTR_SKIP = {"nome"}  # 'meu nome é' → handled by the name pattern

# "nasci em X" is ambiguous: a DATE ("10 de maio") is a birthday, a PLACE
# ("São Paulo") is origin. Disambiguated by content below. "sou de X" is
# origin only. Both map origin to `natural_de`, which is NOT surfaced in the
# prompt (stored + queryable only) — so even a rare mis-capture is low harm.
_MONTHS = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|"
    "agosto|setembro|outubro|novembro|dezembro"
)
# A value is "date-like" if it carries a number or a month name.
_DATE_HINT_RE = re.compile(rf"\b(?:\d{{1,2}}|{_MONTHS})\b", re.IGNORECASE)
_NASCI_EM_RE = re.compile(rf"\b{_NEG}nasci\s+em\s+(?P<v>.+)", re.IGNORECASE)
_SOU_DE_RE = re.compile(rf"\b{_NEG}sou\s+de\s+(?P<v>.+)", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _slugify_attr(attr: str) -> str:
    """Turn an attribute phrase into an ASCII predicate slug:
    'cor favorita' → 'cor_favorita'."""
    ascii_attr = _strip_accents(attr.lower().strip())
    ascii_attr = re.sub(r"[^a-z0-9]+", "_", ascii_attr).strip("_")
    return ascii_attr


def _trim_value(raw: str) -> str:
    """Cut a raw capture down to the actual value: stop at the first clause
    boundary, strip surrounding punctuation/quotes, collapse whitespace."""
    m = _BOUNDARY_RE.search(raw)
    value = raw[: m.start()] if m else raw
    value = value.strip().strip("\"'`.,;:!?()[]{}").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _is_reasonable_value(value: str) -> bool:
    if not value or value.lower() in _FILLER_VALUES:
        return False
    if len(value) > _MAX_VALUE_CHARS:
        return False
    if len(value.split()) > _MAX_VALUE_WORDS:
        return False
    # Reject any value carrying a negation token — a sign we captured across
    # a "não" the lookbehind didn't cover (e.g. inside the generic pattern).
    low = f" {value.lower()} "
    if " não " in low or " nao " in low:
        return False
    return True


def extract_facts(text: str) -> List[ExtractedFact]:
    """Return profile facts stated in `text`. Empty list is the common case
    — most turns state no durable fact, and that's fine.

    Conservative by construction: questions are skipped wholesale (so
    "qual é o meu nome?" never writes anything), negations are rejected,
    and each predicate is emitted at most once (first match wins)."""
    if not text or not text.strip():
        return []
    # A question is asking, not stating — never extract from it.
    if "?" in text:
        return []
    # Material to translate / revise / exemplify is someone else's sentence.
    if MATERIAL_CUE_RE.search(text):
        return []
    text = _QUOTED_RE.sub(" ", text)

    facts: List[ExtractedFact] = []
    seen: set[str] = set()

    def _add(predicado: str, raw_value: str) -> None:
        if predicado in seen:
            return
        value = _trim_value(raw_value)
        if not _is_reasonable_value(value):
            return
        seen.add(predicado)
        facts.append(ExtractedFact(predicado=predicado, objeto=value, evidence=text.strip()[:160]))

    for pattern, predicado in _PATTERNS:
        m = pattern.search(text)
        if m:
            _add(predicado, m.group("v"))

    for m in _GENERIC_RE.finditer(text):
        attr = m.group("attr").strip().lower()
        if attr in _GENERIC_ATTR_SKIP:
            continue
        # Sensitive attr ("minha senha do banco é 1234") → REFUSE the
        # write, not just the injection (M3-A6). The tripla would land in
        # unencrypted SQLite, queryable forever; not storing is the only
        # redaction that works for a fact store.
        if is_sensitive_attr(attr):
            continue
        slug = _slugify_attr(attr)
        # The slug must be a usable predicate and not collide with a
        # dedicated one we already emitted.
        if slug and slug not in seen:
            _add(slug, m.group("v"))

    # "nasci em X": a date → birthday (surfaced); a proper-noun place →
    # origin (natural_de, not surfaced). A lowercase / non-date value is
    # ambiguous noise ("nasci em casa") → skipped.
    m = _NASCI_EM_RE.search(text)
    if m:
        val = _trim_value(m.group("v"))
        if val and _DATE_HINT_RE.search(val):
            _add("aniversario", m.group("v"))
        elif val[:1].isupper():
            _add("natural_de", m.group("v"))

    # "sou de <Proper>": origin only. Proper-noun guard (uppercase initial)
    # keeps out "sou de boa / de esquerda / de TI". natural_de isn't
    # surfaced, so the residual risk is stored-only, low harm.
    m = _SOU_DE_RE.search(text)
    if m:
        val = _trim_value(m.group("v"))
        if val[:1].isupper():
            _add("natural_de", m.group("v"))

    return facts


def format_profile_facts_for_prompt(rows: List[dict], max_lines: int = 8) -> str:
    """Render whitelisted profile facts as a compact block for the system
    prompt. `rows` are Palace tripla dicts (predicado/objeto). Only the
    PROFILE_INJECT_KEYS surface, and each value is sanity-checked the same
    way the preference injector checks its rows. Returns "" when nothing
    qualifies, so the caller can concatenate unconditionally.

    `max_lines` caps the block at the top-N rows by confidence (M3-A8,
    review Fable 5) so passive extraction can't grow the system prompt
    monotonically and eat the num_ctx budget (M2-A1). Ties keep the
    original (Palace) order."""
    scored: List[tuple] = []
    for row in rows:
        predicado = (row.get("predicado") or "").strip()
        objeto = (row.get("objeto") or "").strip()
        if predicado not in PROFILE_INJECT_KEYS or not objeto:
            continue
        if len(objeto) > _MAX_VALUE_CHARS:
            continue
        # No control chars / injected section tags — same guard the
        # preference block applies before splicing into the prompt.
        if any(ch in objeto for ch in ("\n", "\r")) or re.search(r"\[[A-Z]{3,}\]", objeto):
            continue
        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        scored.append((conf, f"- {predicado} → {objeto}"))
    if not scored:
        return ""
    scored.sort(key=lambda pair: -pair[0])
    lines = [line for _, line in scored[:max_lines]]
    return (
        "USER FACTS (stated in passing — treat as durable context, not "
        "absolutes; the current turn always wins on conflict):\n"
        + "\n".join(lines)
        + "\n\n"
    )
