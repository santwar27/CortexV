"""
Fitness Engine regex dispatch patterns (slice 1: meals + macros, s76).

Same regex-first pattern as notes/weather: deterministic routing, no LLM
deciding whether this is a meal log. S2 (local-only SQLite, recoverable) →
the orchestrator executes the match directly, no sim/não gate.

Kept narrow so they don't steal neighbouring intents:
  - LOG_MEAL fires on "comi …" / "registra refeição …" / "ate …", NOT on
    "anota que comi …" (that stays with Apple Notes).
  - MACROS_TODAY is a full-match on the standalone question.
"""
from __future__ import annotations

import re


# Verbs that mean DELETE. The read pattern below must refuse them: a broad
# read regex happily matches "apaga tudo que comi hoje" (it contains "comi"),
# and answering a delete with a listing leaves the user thinking nothing
# happened (s82 cont. — caught while checking how the user would undo a meal).
_DELETE_VERB = (r"(?:apaga(?:r)?|limpa(?:r)?|zera(?:r)?|esquec(?:e|er|a)|"
                r"remove(?:r)?|deleta(?:r)?|desfa(?:z(?:er)?|ça)|"
                r"clear|delete|forget|undo|reset|wipe)")
# What may come before a delete verb in an order, once courtesy is stripped.
_DELETE_LEAD = (r"^(?:(?:ent[ãa]o|agora|e|tamb[ée]m|"
                r"(?:eu\s+)?(?:quero|queria|preciso))\s+)*")
_NEGATION = r"\b(?:n[ãa]o|nunca|jamais|don'?t|do\s+not|never|not)\b"
_NO_NEGATION = r"(?!.*" + _NEGATION + r")"
# What a fitness deletion is about — for the guard below.
_FITNESS_DELETABLE = (r"\b(?:refei[çc](?:[ãa]o|[õo]es)|macros|comidas?|comi|"
                      r"registros?|meals?|treinos?|exerc[íi]cios?|corridas?|"
                      r"workouts?)\b")

# "comi 200g de frango e 2 ovos" / "no café da manhã de hoje eu comi …" /
# "às 12h eu comi …" / "registra refeição: …" / "I ate …". Group `body` is the
# food list, handed to FitnessLog.parse_meal.
#
# s82 — TWO rounds of this bug, and the second is the instructive one. The
# pattern was anchored at "^comi", so "hoje eu comi …" never matched. The
# first fix ENUMERATED prefixes (hoje|ontem|eu|…) — and promptly failed on
# "no café da manhã de hoje eu comi" and "às 12h eu comi". Enumerating the
# ways a person can open a sentence is the same mistake at a larger scale:
# that list has no end.
#
# So the trigger now SEARCHES for the eating verb anywhere in the sentence
# and takes everything after it as the food list. Three things make that
# safe rather than reckless:
#   1. ORDER — this route is registered AFTER the read routes (MACROS, LIST),
#      so "o que eu comi hoje" and "quantos macros eu comi hoje" are claimed
#      by them first. Notes ("anota que comi …") is earlier still.
#   2. NEGATION guard — "não comi nada" must not open a meal log.
#   3. The handler FALLS THROUGH (returns None) when the body contains no
#      recognisable food, so a stray "comi" costs nothing.
# Bare "ate" stays anchored at the start: unaccented "até" ("vou ate a loja")
# is a real Portuguese word and a mid-sentence search on it would misfire.
# An interrogative appearing BEFORE the eating verb means this is a QUESTION
# about food, not a report of eating it. Registry order alone cannot protect
# the read intents, because the read patterns only cover the phrasings their
# author anticipated — "o que eu comi no café da manhã de hoje?" matched none
# of them and the broad log trigger swallowed it (s82 cont., caught in live
# use). This guard makes the WRITE side refuse questions regardless of whether
# a read pattern happens to cover them.
_MEAL_QUESTION_GUARD = (
    r"(?!.*\b(?:o\s+que|que\s+que|quanto|quantos|quantas|quais|qual|"
    r"me\s+lembra|lembra|what|how\s+many|how\s+much|which)\b"
    r".*\b(?:comi|ate)\b)"
)

FITNESS_LOG_MEAL_RE = re.compile(
    r"^(?!.*\b(?:n[ãa]o|nunca|jamais)\s+(?:comi|ate)\b)"      # not a negation
    + _MEAL_QUESTION_GUARD +                                    # not a question
    # not a deletion either: "apaga tudo que comi hoje" contains "comi", and a
    # verb SEARCH would happily read it as a meal to log (s82 cont.).
    r"(?!.*\b" + _DELETE_VERB + r"\b.*\b(?:comi|ate)\b)"
    r"(?:"
    r".*?\b(?:comi|registra(?:r)?\s+(?:a\s+)?refei[çc][ãa]o)|"  # PT, anywhere
    r".*?\bi\s+ate|"                                           # EN "I ate"
    r"(?:ate|log\s+(?:a\s+)?meal)"                              # EN, at the start
    r")\b[:\s]+(?P<body>.+)$",
    re.IGNORECASE,
)


# "meus macros hoje" / "quanto comi hoje" / "macros do dia" / EN twins.
# Full-match anchored so a conversational "meus macros hoje estão altos?"
# doesn't hijack — only the standalone question fires.
FITNESS_MACROS_TODAY_RE = re.compile(
    r"^(?:"
    r"(?:quais\s+(?:os\s+)?)?(?:meus\s+)?macros\s+(?:de\s+)?hoje|"
    r"macros\s+do\s+dia|"
    r"quanto(?:s)?\s+(?:macros\s+)?(?:eu\s+)?(?:comi|calorias?)\s+hoje|"
    r"quantas\s+calorias\s+(?:eu\s+)?(?:comi\s+)?hoje|"
    r"minhas\s+calorias\s+(?:de\s+)?hoje|"
    r"resumo\s+(?:da\s+)?(?:alimenta[çc][ãa]o|comida)\s+(?:de\s+)?hoje|"
    r"my\s+macros\s+(?:for\s+)?today|"
    r"(?:how\s+many\s+)?calories\s+today|"
    r"macros\s+today"
    r")[\s.?!]*$",
    re.IGNORECASE,
)


# "o que comi hoje" / "minhas refeições hoje" → LIST the day's meals (the
# breakdown that answers "where did 5077 kcal come from"). Distinct from
# MACROS ("quanto comi" = totals): here it's "o QUE comi" = the list.
# "o que eu comi hoje" / "o que eu comi no café da manhã de hoje?" / "me
# lembra o que comi" / "quais foram minhas refeições" / EN twins.
#
# s82 cont. — this was a closed full-match ("o que eu comi hoje" and nothing
# else), which is how a plain question about breakfast slipped past every read
# route and got swallowed by the log trigger. Reads are the SAFE side to be
# generous on: the handler only lists what is already stored, so a false
# positive costs a listing, never a wrong write.
#
# The one thing it must not do is answer about the wrong DAY: the handler
# lists TODAY, so an explicit "ontem"/"yesterday" is refused rather than
# answered with today's meals.
FITNESS_LIST_TODAY_RE = re.compile(
    r"^(?!.*\b(?:ontem|yesterday|semana|week)\b)"
    r"(?!.*\b" + _DELETE_VERB + r"\b)"
    r".*?\b(?:"
    r"(?:o\s+que|que)\s+(?:eu\s+)?comi|"
    r"(?:quais|que)\s+(?:foram\s+)?(?:as\s+)?(?:minhas\s+)?refei[çc][õo]es|"
    r"lista(?:r)?\s+(?:as\s+)?(?:minhas\s+)?refei[çc][õo]es|"
    r"minhas\s+refei[çc][õo]es|"
    r"what\s+did\s+i\s+eat|my\s+meals"
    r")\b.*$",
    re.IGNORECASE,
)


# Which meal is said (s88, Codex lote 9 — SOBERANIA 4): "apaga o registro"
# removed the last meal without naming a meal, and "apaga a refeição" doesn't
# say which. "desfaz" names it on its own — undoing is of the last thing done.
FITNESS_UNDO_MEAL_RE = re.compile(
    r"^(?:"
    r"desfa(?:z(?:er)?|ça)\s+(?:a\s+|minha\s+)?(?:[úu]ltima\s+)?refei[çc][ãa]o|"
    r"(?:apaga(?:r)?|remove(?:r)?|tira(?:r)?)\s+(?:a\s+|minha\s+)?[úu]ltima\s+refei[çc][ãa]o|"
    r"undo\s+(?:the\s+|my\s+)?(?:last\s+)?meal|remove\s+(?:the\s+|my\s+)?last\s+meal"
    r")[\s.?!]*$",
    re.IGNORECASE,
)


# "limpa as refeições de hoje" / "zera os macros de hoje" → wipe TODAY's log
# (e.g. clear a batch of test entries). Explicit trigger; today only.
# "limpa as refeições de hoje" / "apaga tudo que comi hoje" / "esquece o que
# eu comi hoje". Broadened alongside the read pattern (s82 cont.): the natural
# way to ask for this is "esquece", and that used to route to a LISTING.
# Still requires an explicit "hoje"/"today" — wiping the day is destructive
# enough that it should never be inferred from a vague sentence.
#
# The verb comes FIRST (s88, Codex lote 9): the pattern began with "^.*?", so
# "não apaga as refeições de hoje" wiped the day — the "não" was just more of
# the prefix. Courtesy and vocatives are stripped before routing; what may
# still precede a real order is in _DELETE_LEAD. A negation anywhere and
# nothing is deleted ("limpa as de hoje, não as de ontem" too — the
# conservative side); FITNESS_DELETE_GUARD_RE answers instead.
FITNESS_CLEAR_TODAY_RE = re.compile(
    _NO_NEGATION + _DELETE_LEAD + _DELETE_VERB + r"\b"
    r"(?=.*\b(?:hoje|today)\b)"
    # One way to read each run of determiners: "todas as" was "todas as" or
    # "todas" + "as", so 24 of them took 8.5s to refuse (s88, Codex lote 9).
    r"\s*(?:(?:as|os|a|o|tudo|todas?|minhas|meus|todos\s+os)\s+)*"
    r"(?:(?:o\s+)?que\s+(?:eu\s+)?comi|refei[çc][õo]es|macros|comidas?|"
    r"registros?|meals?)"
    r"\b.*$|"
    # The English forms say the day too: "clear meals" wiped today's and
    # "reset today" never said what (s88, Codex lote 9).
    r"^(?:(?:clear|reset|delete|wipe)\s+today'?s?\s+meals|"
    r"(?:clear|reset|delete|wipe)\s+(?:the\s+|my\s+)?meals\s+(?:from\s+|for\s+)?today)"
    r"[\s.?!]*$",
    re.IGNORECASE,
)


# A deletion of meals or workouts that no route above carried out: a negated
# one ("não apaga as refeições de hoje"), or one that never said what or which
# day ("clear meals", "apaga o registro" — SOBERANIA 4). Answered with a fixed
# "nothing deleted, say exactly what" — left to the 8B it could say "apaguei"
# (s88, Codex lote 9). "não esquece de registrar a corrida" is a reminder.
FITNESS_DELETE_GUARD_RE = re.compile(
    r"^(?!.*\besquec(?:e|er|a)\s+d[aeo]s?\b)"
    r"(?=.*\b" + _DELETE_VERB + r"\b)"
    r"(?=.*" + _FITNESS_DELETABLE + r").*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Fitness Engine v1 slice 2: workouts (runs + lifts) + trends
# ----------------------------------------------------------------------

# "registra corrida 5km em 25min" / "corri 5km em 25 minutos". dist then dur.
FITNESS_LOG_RUN_RE = re.compile(
    r"^(?:registra(?:r)?\s+|anota(?:r)?\s+|log\s+)?"
    r"(?:corrida|corri|correr|run|ran)\s+"
    r"(?:de\s+)?(?P<dist>\d+(?:[.,]\d+)?)\s*(?:km|k|quil[ôo]metros?)\s+"
    r"(?:em\s+|in\s+)?(?P<dur>\d+(?:[.,]\d+)?)\s*min(?:utos?)?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

# "supino 3x10 80kg" / "registra agachamento 4x8 100kg" / "supino reto 3x10
# com 80kg". The NxM + kg shape is distinctive, so the exercise name (anything
# non-digit before the sets×reps) can be free-form.
FITNESS_LOG_LIFT_RE = re.compile(
    r"^(?:registra(?:r)?\s+|anota(?:r)?\s+|log\s+)?"
    r"(?P<ex>[^\d\n]+?)\s+"
    r"(?P<sets>\d+)\s*x\s*(?P<reps>\d+)\s+"
    r"(?:com\s+|with\s+|@\s*)?(?P<weight>\d+(?:[.,]\d+)?)\s*(?:kg|quilos?|kilos?)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

# "como tá meu supino?" / "evoluí no supino" / "progresso no agachamento".
# The "como tá" form REQUIRES the possessive "meu/minha" so it can't swallow
# "como tá o tempo?" (weather). Even so, the handler falls through (None) when
# the captured word has no history ("como tá meu dia" → LLM), so this route is
# registered with mode=None (invisible to the eval simulator, like contact_info).
FITNESS_LIFT_TREND_RE = re.compile(
    r"^(?:"
    r"como\s+(?:t[áa]|est[áa]|anda|vai)\s+(?:meu\s+|minha\s+)|"
    r"evolu[íi]\s+(?:no|na|em|com)\s+(?:o\s+|a\s+|meu\s+|minha\s+)?|"
    r"(?:minha\s+)?evolu[çc][ãa]o\s+(?:no|na|em|do|da)\s+(?:meu\s+|minha\s+)?|"
    r"progresso\s+(?:no|na|em|do|da)\s+(?:meu\s+|minha\s+)?"
    r")(?P<ex>[^\d\n?.!]+?)[\s.?!]*$",
    re.IGNORECASE,
)

# "meu pace" / "como tá minha corrida" / "evoluí na corrida". Checked BEFORE
# the lift trend so "corrida"/"pace" go here, not into the generic capture.
FITNESS_RUN_TREND_RE = re.compile(
    r"^(?:"
    r"(?:como\s+(?:t[áa]|est[áa]|anda|vai)\s+)?(?:o\s+|a\s+|meu\s+|minha\s+)?"
    r"(?:pace|ritmo|corrida)(?:\s+melhorou|\s+t[áa]\s+melhor)?|"
    r"evolu[íi]\s+(?:na|no)\s+corrida|"
    r"my\s+(?:pace|running)|how'?s\s+my\s+(?:pace|running)"
    r")[\s.?!]*$",
    re.IGNORECASE,
)

# "desfaz o último treino" / "limpa os treinos de hoje" — cleanup for workouts,
# mirroring the meal undo/clear. "treino" keeps them distinct from the meal
# ones; end-anchored "o último treino" doesn't collide with the Notes undo.
# Same rule as the meal undo: "apaga o treino" doesn't say which one.
FITNESS_UNDO_WORKOUT_RE = re.compile(
    r"^(?:"
    r"desfa(?:z(?:er)?|ça)\s+"
    r"(?:o\s+|a\s+|meu\s+|minha\s+)?(?:[úu]ltim[ao]\s+)?(?P<kind>treino|exerc[íi]cio|corrida)|"
    r"(?:apaga(?:r)?|remove(?:r)?|tira(?:r)?)\s+"
    r"(?:o\s+|a\s+|meu\s+|minha\s+)?[úu]ltim[ao]\s+(?P<kind2>treino|exerc[íi]cio|corrida)|"
    r"undo\s+(?:the\s+|my\s+)?(?:last\s+)?(?P<kind3>workout|run)|"
    r"remove\s+(?:the\s+|my\s+)?last\s+(?P<kind4>workout|run)"
    r")[\s.?!]*$",
    re.IGNORECASE,
)
FITNESS_CLEAR_WORKOUTS_RE = re.compile(
    r"^(?:"
    r"(?:limpa(?:r)?|apaga(?:r)?|zera(?:r)?)\s+"
    r"(?:os\s+|as\s+|tudo\s+|todos\s+os\s+)?"
    r"(?:treinos?|exerc[íi]cios?|corridas?)\s+(?:de\s+)?hoje|"
    r"(?:clear|reset)\s+today'?s?\s+workouts|"
    r"(?:clear|reset)\s+(?:the\s+|my\s+)?workouts\s+(?:from\s+|for\s+)?today"
    r")[\s.?!]*$",
    re.IGNORECASE,
)
