"""
Spoken-language reply localisation (s60, Claude-A) — DISPATCH_REPLY kind.

Executes the matched deterministic handler against the fake orchestrator
(evals/fake_orch.py) with `language` forced, and asserts the REPLY is in that
language. This is the permanent regression shield for the s60 i18n work on the
command path: the bug was English speech → Portuguese command replies.

Scope here = mixin-localised families that the fake orch can drive
deterministically (notes, studier). Controller families (spotify/macos) and the
pure assemblers (digest/system_status/entities) are covered by deterministic
pytest unit tests (test_spotify_i18n, test_macos_i18n, test_i18n_digests); the
planner preview/summary/announce (M1-A8 — decompose is LLM-bound, so the fake
orch can't drive it) by TestPlannerI18n*/TestAnnounceI18n in test_planner.py.

Each behaviour is asserted in BOTH languages so PT can never silently win.
"""
from evals.core import EvalCase, EvalKind


def _c(name, text, lang, contains, not_contains=None):
    return EvalCase(
        name=name, kind=EvalKind.DISPATCH_REPLY, category="i18n",
        input=text, language=lang,
        expected_contains=contains, expected_not_contains=not_contains,
    )


CASES: list[EvalCase] = [
    # ----- notes: list -----
    _c("notes_list_pt", "minhas notas", "pt", "Suas notas:", not_contains="Your notes"),
    _c("notes_list_en", "list my notes", "en", "Your notes:", not_contains="Suas notas"),

    # ----- notes: search miss -----
    _c("notes_search_miss_pt", "acha a nota sobre zzz", "pt", "Não achei nenhuma nota"),
    _c("notes_search_miss_en", "find my notes about zzz", "en", "I found no notes"),

    # ----- notes: search hit -----
    _c("notes_search_hit_pt", "acha a nota sobre projeto", "pt", 'Notas sobre "projeto"'),
    _c("notes_search_hit_en", "find my notes about projeto", "en", 'Notes about "projeto"'),

    # ----- studier: queue a topic -----
    _c("studier_add_pt", "estude física quântica", "pt", "Vou estudar", not_contains="I'll study"),
    _c("studier_add_en", "study quantum physics", "en", "I'll study", not_contains="Vou estudar"),

    # ----- studier: review (empty queue) -----
    _c("studier_review_empty_pt", "revisar", "pt", "Nenhum estudo pronto para revisar"),
    _c("studier_review_empty_en", "review studies", "en", "No studies ready to review"),

    # ----- studier: list queue -----
    _c("studier_list_pt", "fila de estudos", "pt", "Fila de estudos"),

    # ----- studier: show a missing briefing (not-found branch) -----
    _c("studier_show_missing_pt", "ver 1", "pt", "não encontrado"),
    _c("studier_show_missing_en", "show 1", "en", "not found"),
]
