"""Code-owned product facts and whole-question replies (141d).

Engine aliases and repository configuration are read on demand. Everything
else is curated here and changed only through code review, never retrieval,
chat, studies or voice. Configured bases/windows are not runner observations.
Open self questions receive a code-built ephemeral fact note (141e).
No model calls, external lookups or persistent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import unicodedata

from src.core.usage_answer import normalise_own_question


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Bilingual:
    pt: str
    en: str

    def in_language(self, lang: str) -> str:
        return self.pt if lang == "pt" else self.en


@dataclass(frozen=True)
class CuratedFacts:
    planned_alias: str
    planned_label: Bilingual
    planned_architecture: Bilingual
    plan_status: Bilingual
    plan_reason: Bilingual
    host: Bilingual
    avatar: Bilingual
    tastes: Bilingual
    makers: tuple[tuple[str, str], ...]


# The user's reviewed product decision, 07/11/2026. Avatar behaviour checked in
# avatar.js: 145–146, 349, 449–469, 994–1030, 1106–1137, 1178–1185.
# This is product description only: no profile, personal records or history.
CURATED = CuratedFacts(
    planned_alias="gemma4:26b",
    planned_label=Bilingual("Gemma 4 26B", "Gemma 4 26B"),
    planned_architecture=Bilingual("mistura de especialistas", "mixture of experts"),
    plan_status=Bilingual("é um plano, ainda sem data.",
                          "it is a plan, with no date yet."),
    plan_reason=Bilingual(
        "No teste em uma máquina alugada em 11/09/2026, venceu em 30 de 32 casos. "
        "A família Qwen foi vetada porque fingiu ter usado uma ferramenta.",
        "In the rented-machine test on 11/09/2026, it won 30 of 32 cases. "
        "The Qwen family was vetoed because it pretended to have used a tool."),
    host=Bilingual(
        "localmente no seu Mac (Apple Silicon M1) pelo Ollama.",
        "locally on your Mac (Apple Silicon M1) through Ollama."),
    avatar=Bilingual(
        "Meu avatar é um pato robô em pixel art na tela do chat. "
        "Ele anda, acena e muda de estado enquanto eu penso, busco na web, leio ou respondo. "
        "Quando está visível e em repouso, dorme após 3 minutos sem atividade.",
        "My avatar is a pixel-art robot duck in the chat screen. "
        "It walks, waves and changes state while I think, search the web, read or reply. "
        "When visible and idle, it falls asleep after 3 minutes without activity."),
    tastes=Bilingual("Não tenho gostos próprios.", "I don't have tastes of my own."),
    makers=(("Llama", "Meta"), ("Gemma", "Google"), ("Qwen", "Alibaba")),
)


@dataclass(frozen=True)
class ModelFacts:
    alias: str | None
    base: str | None
    num_ctx: int | None


@dataclass(frozen=True)
class SelfFacts:
    analytical: ModelFacts
    creative: ModelFacts


def read_model_config(path: Path) -> tuple[str | None, int | None]:
    """Read FROM/num_ctx only; missing/unreadable/invalid fields stay unknown.

    Ignore comments and multiline bodies (e.g. SYSTEM), whose prose must not
    become configuration. Do not substitute an engine/env/default window.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None, None
    base, num_ctx, multiline = None, None, False
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        if multiline:
            if '"""' in line:
                multiline = False
            continue
        if '"""' in line:
            multiline = line.count('"""') % 2 == 1
            continue
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        if len(parts) == 2 and parts[0].upper() == "FROM":
            base = parts[1] or None
        if len(parts) == 3 and [p.lower() for p in parts[:2]] == ["parameter", "num_ctx"]:
            num_ctx = int(parts[2]) if parts[2].isascii() and parts[2].isdigit() else None
            if num_ctx is not None and num_ctx <= 0:
                num_ctx = None
    return base, num_ctx


def read_self_facts(analytical, creative) -> SelfFacts:
    """Use constructed engines, not environment guesses or import-time values."""
    def model(engine, filename):
        alias = getattr(engine, "model_name", None)
        alias = alias.strip() if isinstance(alias, str) and alias.strip() else None
        return ModelFacts(alias, *read_model_config(REPO_ROOT / filename))

    return SelfFacts(model(analytical, "Modelfile"), model(creative, "Modelfile-creative"))


@dataclass(frozen=True)
class SelfQuestion:
    mode: str
    lang: str
    detail: bool = False  # plan rationale for model choice; taste caveat for avatar


_QUESTIONS = tuple((SelfQuestion(mode, lang, detail), re.compile(pattern))
                  for mode, lang, detail, pattern in (
    ("SELF_MODEL", "pt", False,
     r"(?:(?:qual|que) modelo(?: llm)? voce usa|qual (?:e (?:o )?)?seu modelo|"
     r"qual modelo voce e|voce usa qual modelo|que llm voce e)"),
    ("SELF_MODEL", "en", False,
     r"(?:(?:what|which) model are you|what model do you use|which llm do you use)"),
    ("SELF_MODEL", "pt", True,
     r"qual (?:modelo(?: llm)? e mais viavel|o melhor modelo) para voce"),
    ("SELF_MODEL", "en", True, r"what model is best for you"),
    ("SELF_PLAN", "pt", False, r"o que (?:voce )?(?:pensa|acha) (?:do|sobre o) gemma ?4"),
    ("SELF_PLAN", "en", False, r"what do you think (?:of|about) gemma ?4"),
    ("SELF_AVATAR", "pt", False, r"(?:qual (?:e )?|como e )o seu avatar"),
    ("SELF_AVATAR", "en", False, r"what is your avatar"),
    ("SELF_AVATAR", "pt", True, r"gostou do seu avatar"),
    ("SELF_AVATAR", "en", True, r"do you like your avatar"),
))


def detect_self_question(text: str) -> SelfQuestion | None:
    """Decline quotes, mixed actions and general model/avatar questions."""
    question = normalise_own_question(text)
    return next((intent for intent, pattern in _QUESTIONS if pattern.fullmatch(question)), None)


# Match only the user's unquoted words, before retrieval/enrichment. A model
# family mention is sufficient; other topics need a relation to Cortex.
_QUOTED = re.compile(
    r"```[^\n]*\n.*?(?:```|$)|~~~[^\n]*\n.*?(?:~~~|$)"
    r"|`[^`\n]*`|\"[^\"]*\"|“[^”]*”|‘[^’]*’"
    r"|(?<!\w)'[^'\n]*'(?!\w)|^[ \t]*>[^\n]*",
    re.MULTILINE | re.DOTALL,
)
_NAMED_MODEL = re.compile(r"\b(?:(?:llama|gemma|qwen)(?:\d+(?:\.\d+)*)?"
                          r"|cortex-(?:lite|criativo))\b")
_TOPIC = (
    # Company-qualified topics still need one of the self relations below.
    r"(?:(?:modelos?|models?|ia|ai)\s+(?:de|do|da|from|by|of)\s+"
    r"(?:meta|google|openai|alibaba)|modelos?(?!\s+(?:de|do|da)\b)|"
    r"models?|llm|ia|ai|cerebro|brain|"
    r"janela|window|contexto|context|tokens?|memoria|memory|maquina|machine|"
    r"mac|computador|computer|nuvem|cloud|local(?:mente|ly)?|internet|"
    r"avatar|pato|duck|ferramentas?|tools?)\b"
)
_SELF_RELATIONS = tuple(re.compile(pattern) for pattern in (
    r"\b(?:seus?|suas?|teus?|tuas?|your)\s+" + _TOPIC,
    r"\b" + _TOPIC + r"\s+(?:de|d[oa]|of)\s+(?:o\s+)?cortex\b",
    r"\b(?:voce|(?:o\s+)?cortex)\s+"
    r"(?:(?:nao|ainda|vai|pode|um|uma)\s+){0,3}"
    r"(?:usa|utiliza|acessa|tem|possui|e|muda|mudar|troca|trocar)\s+"
    r"(?:(?:um|uma|o|a|outro|outra|qual|quantos|quantas|de)\s+){0,3}" + _TOPIC,
    r"\b" + _TOPIC + r"\s+(?:voce|(?:o\s+)?cortex)\s+(?:usa|utiliza|tem|possui|e)\b",
    r"\b(?:voce|(?:o\s+)?cortex)\s+(?:roda|funciona|executa|opera)\b",
    r"\b(?:o que\s+(?:voce|cortex)\s+(?:sabe|pode)\s+fazer|"
    r"por que\s+(?:voce|cortex)\s+nao\s+usa)\b",
    r"\bdiferenca\s+entre\s+(?:voce|cortex)\s+e\b",
    r"\b(?:you|cortex)\s+(?:(?:do|don't|do not|will|can|not)\s+){0,2}"
    r"(?:use|have|are|change|switch)\s+"
    r"(?:(?:a|an|the|another|your|to)\s+){0,3}" + _TOPIC,
    r"\b" + _TOPIC + r"\s+(?:(?:do|does|are)\s+)?(?:you|cortex)\s+(?:use|have)\b",
    r"\b(?:are\s+you|is\s+cortex)\s+(?:(?:a|an|the)\s+)?" + _TOPIC,
    r"\b(?:you|cortex)\s+(?:run|operate)\b",
    r"\b(?:what\s+can\s+you\s+do|difference\s+between\s+you\s+and)\b",
))


def needs_self_facts(text: str) -> bool:
    """Deterministic PT/EN relations, never quoted/attached/retrieved intent.

    Not a route: whole-question code replies and existing command owners still
    win. No question-mark requirement (voice commonly omits punctuation).
    """
    from src.core.prompt_safety import own_words

    own = _QUOTED.sub(" ", own_words(text) or "")
    # Unlike whole-question dispatch, keep vocatives and courtesy prefixes:
    # stripping them loses "Cortex usa ...", "cortex-lite" and "will you ...".
    question = " ".join("".join(
        c for c in unicodedata.normalize("NFKD", own.lower())
        if not unicodedata.combining(c)).split())
    return bool(_NAMED_MODEL.search(question) or any(
        pattern.search(question) for pattern in _SELF_RELATIONS))


def self_fact_note(analytical, creative) -> str:
    """Trusted, language-neutral sheet; configured values are read each turn.

    Short subject-first sentences distinguish self facts from general model
    knowledge. No personality or user data.
    """
    facts = read_self_facts(analytical, creative)
    current = facts.analytical
    window = f"{current.num_ctx} tokens" if current.num_ctx is not None else "unknown"
    makers = dict(CURATED.makers)
    base = _base_label(current.base, "unknown")
    family = re.match(r"(llama|qwen|gemma)(?=\d|:|$)", current.base or "", re.IGNORECASE)
    maker = makers.get(family[1].capitalize()) if family else None
    made_by = f" (made by {maker})" if maker else ""
    plan = ("" if _is_planned_base(current.base) else
            f"Your planned next model is {CURATED.planned_label.en} "
            f"({CURATED.planned_alias}, made by {makers['Gemma']}, "
            f"{CURATED.planned_architecture.en}). It is a plan with no date.\n")
    return (
        "[CORTEX_FACTS]\n"
        "You are Cortex, a private assistant built by Pedro.\n"
        "Self facts come only from this note. General model questions (e.g. coding): "
        "use general knowledge; never contradict this note; say when unsure.\n"
        f"You run on {current.alias or 'unknown'}, which is {base}{made_by}.\n"
        f"Your configured window is {window}.\n"
        f"You run {CURATED.host.en}\n"
        f"Your creative model is {facts.creative.alias or 'unknown'}.\n"
        f"{plan}"
        "Pedro tested Gemma 4 26B on a rented machine on 11 September 2026. "
        "Gemma 4 26B won 30 of 32 cases.\n"
        f"Pedro rejected the Qwen models (made by {makers['Qwen']}) because "
        "Qwen pretended to have used a tool. Nobody vetoed you.\n"
        "Your avatar is a pixel-art robot duck. It walks, waves and "
        "changes state with your activity. "
        "It sleeps after 3 idle minutes while visible.\n"
        "[/CORTEX_FACTS]"
    )


def _plan(lang: str, reason: bool, current: ModelFacts) -> str:
    if _is_planned_base(current.base):
        result = (f"O {CURATED.planned_label.pt} já é minha base atual." if lang == "pt"
                  else f"{CURATED.planned_label.en} is already my current base.")
        return result + (" " + CURATED.plan_reason.in_language(lang) if reason else "")
    choice = (f"{CURATED.planned_label.in_language(lang)} ({CURATED.planned_alias}, "
              f"{CURATED.planned_architecture.in_language(lang)}); ")
    lead = ("Para a máquina nova, o escolhido é o " if lang == "pt"
            else "The choice for the new machine is ")
    result = lead + choice + CURATED.plan_status.in_language(lang)
    return result + (" " + CURATED.plan_reason.in_language(lang) if reason else "")


_BASE_MODEL = re.compile(
    r"(llama|qwen|gemma)(\d+(?:\.\d+)*):(\d+(?:\.\d+)?)b(?:-[\w.-]+)?",
    re.IGNORECASE)


def _is_planned_base(base: str | None) -> bool:
    """Match the live base tag, including variants of the same family/version/size."""
    if base is None:
        return False
    if base.casefold() == CURATED.planned_alias.casefold():
        return True
    live = _BASE_MODEL.fullmatch(base)
    planned = _BASE_MODEL.fullmatch(CURATED.planned_alias)
    return bool(live and planned and
                tuple(part.casefold() for part in live.groups()) ==
                tuple(part.casefold() for part in planned.groups()))


def _base_label(base: str | None, unknown: str) -> str:
    """Derive a human name only when the tag supplies family, version and size."""
    if base is None:
        return unknown
    match = _BASE_MODEL.fullmatch(base)
    if match is None:
        return base
    family, version, size = match.groups()
    return f"{family.capitalize()} {version} {size}B ({base})"


def _current_summary(model: ModelFacts, lang: str) -> str:
    unknown = "não sei essa parte" if lang == "pt" else "I don't know that part"

    base = _base_label(model.base, unknown)
    window = f"{model.num_ctx} tokens" if model.num_ctx is not None else unknown
    if lang == "pt":
        current = f"Hoje eu rodo no {model.alias or unknown}, que é o {base}, com janela de {window}, "
    else:
        current = (f"Today I run on {model.alias or unknown}, which is {base}, "
                   f"with a context window of {window}, ")
    return current + CURATED.host.in_language(lang)


def self_answer(intent: SelfQuestion, analytical, creative) -> str:
    """Fixed wording; the question selects facts, never supplies/updates them."""
    lang = intent.lang
    if intent.mode == "SELF_AVATAR":
        return CURATED.avatar.in_language(lang) + (
            " " + CURATED.tastes.in_language(lang) if intent.detail else "")

    facts = read_self_facts(analytical, creative)
    if intent.mode == "SELF_PLAN":
        return _plan(lang, reason=True, current=facts.analytical)
    unknown = "não sei essa parte" if lang == "pt" else "I don't know that part"
    model = facts.analytical
    current = _current_summary(model, lang)

    creative_base = _base_label(facts.creative.base, unknown)
    same_base = model.base is not None and model.base == facts.creative.base
    if lang == "pt":
        basis = "com a mesma base" if same_base else f"com base em {creative_base}"
        creative_reply = f"Para textos criativos uso o {facts.creative.alias or unknown}, {basis}."
    else:
        basis = "with the same base" if same_base else f"based on {creative_base}"
        creative_reply = f"For creative writing I use {facts.creative.alias or unknown}, {basis}."

    return " ".join((
        current,
        creative_reply,
        _plan(lang, reason=intent.detail, current=model),
    ))
