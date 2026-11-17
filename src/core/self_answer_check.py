"""141f: narrow deterministic checks of note-scoped Cortex/model assertions.

Not a general fact checker. Quotes/code, negations and other models' specs are
left alone. Unknown names need an explicit self-model assertion; user-named
models can be echoed. Makers need a named subject in the same clause. Dates
need test language. Unrecognised paraphrases deliberately pass (false positives
cost more than misses). No reply-derived allowlist, model calls or disk reads.

The trusted ephemeral note is the snapshot: parse only its code-owned field
lines, never user/history/retrieval content. This also prevents a changed
Modelfile during generation from changing the validation/fallback facts.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.core.prompt_safety import own_words
from src.core.self_facts import CURATED, ModelFacts, _base_label, _current_summary, _plan, _QUOTED


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower().replace("…", " "))
                   if not unicodedata.combining(c)).replace("’", "'")


@dataclass(frozen=True)
class AnswerFacts:
    current: ModelFacts
    creative_alias: str | None


class AnswerRules(list[str]):
    """Keep the audit's rule list, with the subject needed for correction."""

    def __init__(self, rules=(), *, cortex_maker: bool = False):
        super().__init__(sorted(rules))
        self.cortex_maker = cortex_maker


def note_facts(context: str) -> AnswerFacts | None:
    """Only the per-turn system context, never a scan of the whole prompt."""
    block = re.search(r"(?:^|\n)\[CORTEX_FACTS\]\n(.*?)\n\[/CORTEX_FACTS\](?:$|\n)",
                      context, re.DOTALL)
    if block is None:
        return None
    text = block[1]
    current = re.search(r"^You run on (.+?), which is (.+?)\.$", text, re.MULTILINE)
    window = re.search(r"^Your configured window is (unknown|\d+ tokens)\.$", text, re.MULTILINE)
    creative = re.search(r"^Your creative model is (.+?)\.$", text, re.MULTILINE)
    if not (current and window and creative):
        return None
    base = re.sub(r" \(made by [^)]+\)$", "", current[2])
    tag = re.search(r"\(([^()]+)\)$", base)
    base = tag[1] if tag else base
    return AnswerFacts(ModelFacts(
        None if current[1] == "unknown" else current[1],
        None if base == "unknown" else base,
        None if window[1] == "unknown" else int(window[1].split()[0])),
        None if creative[1] == "unknown" else creative[1])


# Version/size are optional so shorthand "Llama 3.1" is accepted against the
# configured tag, but any supplied version/size must agree. Bare family names
# are not an invented version. GPT/Mistral need a self assertion as well.
_MODEL = re.compile(
    r"\b(?P<family>llama|gemma|qwen|gpt|mistral|chatgpt)"
    r"(?:[- ]?(?P<version>\d+(?:\.\d+)*))?"
    r"(?:(?::|\s+)(?P<size>\d+(?:\.\d+)?)\s*b)?"
    r"(?:-[\w.-]+)?\b")
_SELF_MODEL = re.compile(
    r"\b(?:(?:eu |cortex )?(?:rodo|uso|utilizo|estou rodando)|"
    r"(?:i|cortex) (?:am |am currently |currently )?(?:run(?:ning)?|use)|"
    r"(?:meu|o meu|my) (?:(?:modelo (?:atual|base)|current model)|modelo|model)|"
    r"(?:eu sou|i am|i'm|cortex (?:e|is)).*?\b(?:basead[oa]|based)|"
    r"(?:eu sou|sou|i am|i'm)|cortex (?:usa|uses|runs on))\b")
# Only grammatical links may connect an assertion to its model name. This
# excludes "I am Cortex, and GPT-4 is another model" and model preferences.
_MODEL_LINK = re.compile(
    r"(?:[\s,*_:()-]|\b(?:e|is|o|a|um|uma|an|the|no|na|on|em|in|de|do|da|"
    r"modelo|model|basead[oa]|based|chamado|called|currently|running|rodando|"
    r"planejad[oa]|planned|proxim[oa]|next|to|be|upgraded)\b)*")
_NEGATION = re.compile(r"\b(?:nao|nunca|not|never|don't|doesn't|isn't|wasn't|nao e)\b")
_CLAUSES = re.compile(r"(?<!\d)[.!?;\n]+|[.!?;\n]+(?!\d)|\b(?:enquanto|whereas|while|but|mas)\b")
_QUESTION_INTRO = re.compile(r"\b(?:se|if|whether)\b")
_MAKER = re.compile(
    r"\b(?:desenvolvid[oa]|criad[oa]|feit[oa]|construid[oa]|produzid[oa]|"
    r"developed|created|made|built)\s+"
    r"(?:pela?|por|by)\s+(?:a |o |the )?(meta|google|openai|alibaba)\b")
_SELF_SUBJECT = re.compile(r"\b(?:cortex|eu|sou|i)\b")
_BASE_SUBJECT = re.compile(r"\b(?:(?:meu|my) (?:modelo|model)|(?:minha|my) base)\b")
# An answer particle followed by a comma is not a denial of the next claim.
# Apply only to maker checks; keep the other rules' negation behaviour intact.
_ANSWER_PARTICLE = re.compile(r"^\s*(?:nao|sim|no|yes)\s*,\s*")
_MAKER_NAME = r"(?:meta|google|openai|alibaba)"
_IDENTITY_MAKER = re.compile(
    r"\b(?:(?:eu\s+)?sou|cortex\s+e)\s+"
    r"(?:(?:um|uma)\s+)?(?:(?:ia|modelo|assistente|inteligencia artificial)\s+)?"
    r"(?:da|do|de)\s+(?:a\s+|o\s+)?" + _MAKER_NAME + r"\b|"
    r"\b(?:i am|i'm|cortex is)\s+(?:a|an)\s+"
    r"(?:" + _MAKER_NAME + r"\s+ai\b|(?:ai|model|assistant)\s+"
    r"(?:from|by|of)\s+" + _MAKER_NAME + r"\b)")
_WINDOW = re.compile(
    r"\b(?:janela(?: de (?:contexto|processamento))?|(?:context )?window)\s*"
    r"(?:(?:e|is|de|of|configured|configurada|tem|has|com|with|a|uma)\s+)*"
    r"(?P<number>[\d][\d.,]*(?: \d{3})*)\s*"
    r"(?P<multiplier>mil|thousand|k)?\b(?:\s*(?P<unit>tokens?)\b)?")
_VETO = re.compile(r"\b(?:(?:eu )?fui (?:vetad[oa]|rejeitad[oa])|"
                   r"i was (?:vetoed|rejected)|cortex (?:foi|was) (?:vetado|rejeitado|vetoed|rejected))\b")
_DATE = re.compile(r"\b(?:09/11/2026|0?9 de novembro(?: de 2026)?|november 0?9(?:,? 2026)?)\b")


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


def _clauses(text: str):
    """Keep the same clause boundaries, retaining each closing delimiter."""
    start = 0
    for delimiter in _CLAUSES.finditer(text):
        yield text[start:delimiter.start()], delimiter[0]
        start = delimiter.end()
    yield text[start:], ""


def _questioned(prefix: str, delimiter: str) -> bool:
    """Echoed/reported questions are not Cortex identity or maker claims."""
    return "?" in delimiter or bool(_QUESTION_INTRO.search(prefix))


def _self_assertion(clause, model, models) -> bool:
    prefix = clause[:model.start()]
    assertions = list(_SELF_MODEL.finditer(prefix))
    if not assertions or _negated(prefix):
        return False
    assertion = assertions[-1]
    return bool(_MODEL_LINK.fullmatch(prefix[assertion.end():])) and not any(
        earlier.start() >= assertion.end() and earlier.end() <= model.start()
        for earlier in models)


def _compatible_name(model, allowed) -> bool:
    return model['family'] == allowed['family'] and all(
        model[field] is None or model[field] == allowed[field]
        for field in ('version', 'size'))


def _allowed_name(match, facts: AnswerFacts, question: str, clause: str) -> bool:
    name = match[0]
    if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question):
        return True
    if match['family'] == 'qwen' and re.search(r"vet|reject|fing|pretend|tool|ferrament", clause):
        return True
    for source in (facts.current.alias, facts.current.base,
                   facts.creative_alias, CURATED.planned_alias):
        for allowed in _MODEL.finditer(_fold(source or "")):
            if _compatible_name(match, allowed):
                return True
    return False


def check_self_answer(reply: str, question: str, facts: AnswerFacts) -> AnswerRules:
    """Stable audit rule names; supplied facts come exclusively from the note."""
    rules = set()
    cortex_maker = False
    question = _fold(own_words(question) or "")
    makers = {family.lower(): maker.lower() for family, maker in CURATED.makers}
    planned = _MODEL.search(_fold(CURATED.planned_alias))
    current_names = list(_MODEL.finditer(_fold(
        (facts.current.base or "") + " " + (facts.current.alias or ""))))
    for clause, delimiter in _clauses(_fold(_QUOTED.sub(" ", reply))):
        models = list(_MODEL.finditer(clause))
        for model in models:
            prefix = clause[:model.start()]
            if _self_assertion(clause, model, models):
                if not _allowed_name(model, facts, question, clause):
                    rules.add("model_name")
                # Future/planned clauses are not a current-model assertion.
                if (planned is not None and _compatible_name(model, planned)
                        and not re.search(r"\b(?:plan\w*|proxim[oa]|next|future|futur[oa]|upgrade\w*|will|vou|vai)\b", prefix)
                        and not any(_compatible_name(model, current) for current in current_names)):
                    rules.add("planned_as_current")
        maker_clause = _ANSWER_PARTICLE.sub("", clause)
        for identity in _IDENTITY_MAKER.finditer(maker_clause):
            if _questioned(maker_clause[:identity.start()], delimiter):
                continue
            if not _negated(maker_clause[:identity.end()]):
                rules.add("maker")
                cortex_maker = True
        for maker in _MAKER.finditer(maker_clause):
            prefix = maker_clause[:maker.start()]
            maker_models = list(_MODEL.finditer(prefix))
            subjects = [(m.start(), m['family']) for m in maker_models]
            subjects += [(m.start(), 'cortex') for m in _SELF_SUBJECT.finditer(prefix)]
            subjects += [(m.start(), 'base') for m in _BASE_SUBJECT.finditer(prefix)]
            if not subjects:
                continue
            subject = max(subjects)[1]
            if subject == 'cortex' and _questioned(prefix, delimiter):
                continue
            if _negated(maker_clause[:maker.start()]):
                continue
            if ((subject in makers and maker[1] != makers[subject]
                 and (subject != 'qwen' or maker[1] in ('meta', 'google')))
                    or subject == 'cortex'):
                rules.add("maker")
                cortex_maker |= subject == 'cortex'
        for window in _WINDOW.finditer(clause):
            prefix = clause[:window.start()]
            # Other named subjects' windows are outside this guard's scope.
            previous_models = [m for m in models if m.end() <= window.start()]
            own_window = re.search(r"\b(?:minha|my)\s+(?:configured\s+)?$", prefix)
            if (previous_models and not own_window
                    and not _self_assertion(clause, previous_models[-1], models)):
                continue
            if re.search(r"\b(?:browser|navegador|pixels?|px)\b", clause):
                continue
            if _negated(prefix):
                continue
            configured = facts.current.num_ctx
            if configured is None:
                continue
            if window['multiplier']:
                # Integer thousands allow decimal or binary rounding (33k/32k).
                # Separators with a multiplier are ambiguous; decline to judge.
                if not window['number'].isdigit():
                    continue
                consistent = int(window['number']) in (
                    round(configured / 1000), round(configured / 1024))
            else:
                # An unrecognised unit (e.g. million) must not become an exact 8.
                if not window['unit'] and re.match(r"\s*[a-z]", clause[window.end():]):
                    continue
                number = int(re.sub(r"[., ]", "", window['number']))
                consistent = number == configured
            if not consistent:
                rules.add("window")
        for veto in _VETO.finditer(clause):
            if not _negated(clause[:veto.end()]):
                rules.add("veto_inversion")
        if (_DATE.search(clause) and re.search(r"\b(?:test\w*|teste\w*)\b", clause)
                and not _negated(clause)):
            rules.add("test_date")
    return AnswerRules(rules, cortex_maker=cortex_maker)


def corrected_self_answer(facts: AnswerFacts, lang: str, *, rules: AnswerRules | None = None) -> str:
    """Correct Cortex ownership explicitly; other failures retain 141d wording."""
    if rules is not None and rules.cortex_maker:
        family = re.match(r"(llama|qwen|gemma)(?=\d|:|$)", facts.current.base or "", re.IGNORECASE)
        maker = dict(CURATED.makers).get(family[1].capitalize()) if family else None
        base = _base_label(facts.current.base, "").split(" (")[0]
        if lang == "pt":
            identity = "Não. Eu sou o Cortex, um assistente feito pelo Pedro"
            if maker:
                identity += f"; só a minha base, o {base}, é da {maker}"
        else:
            identity = "No. I am Cortex, an assistant built by Pedro"
            if maker:
                identity += f"; only my base, {base}, is from {maker}"
        return identity + ".\n" + _current_summary(facts.current, lang)
    unsure = "Não sei essa parte." if lang == "pt" else "I don't know that part."
    return (unsure + "\n" + _current_summary(facts.current, lang) + " " +
            _plan(lang, reason=True, current=facts.current))
