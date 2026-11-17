"""Fitness Engine v1 dispatch behaviour (slice 1: meals + macros, s76)."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- log a meal -------
    EvalCase(
        name="log_meal_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="comi 200g de frango e 2 ovos",
        expected_mode="FITNESS_MEAL",
        expected_matcher="FITNESS_LOG_MEAL_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="log_meal_registra_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="registra refeição: 150g de arroz",
        expected_mode="FITNESS_MEAL",
        expected_matcher="FITNESS_LOG_MEAL_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="log_meal_en",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="I ate 100g of rice",
        expected_mode="FITNESS_MEAL",
        expected_matcher="FITNESS_LOG_MEAL_RE",
        tags=["fitness"],
    ),
    # ------- ask today's macros -------
    EvalCase(
        name="macros_today_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="meus macros hoje",
        expected_mode="FITNESS_MACROS",
        expected_matcher="FITNESS_MACROS_TODAY_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="macros_quanto_comi_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="quanto comi hoje",
        expected_mode="FITNESS_MACROS",
        expected_matcher="FITNESS_MACROS_TODAY_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="macros_today_en",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="my macros today",
        expected_mode="FITNESS_MACROS",
        expected_matcher="FITNESS_MACROS_TODAY_RE",
        tags=["fitness"],
    ),
    # ------- list / undo / clear (s76 cont.35: added after the live "where did
    #         5077 kcal come from" — needed a breakdown + cleanup). ----
    EvalCase(
        name="list_today_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="o que comi hoje",
        expected_mode="FITNESS_LIST",
        expected_matcher="FITNESS_LIST_TODAY_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="list_today_refeicoes_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="minhas refeições hoje",
        expected_mode="FITNESS_LIST",
        expected_matcher="FITNESS_LIST_TODAY_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="undo_meal_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="desfaz a última refeição",
        expected_mode="FITNESS_UNDO",
        expected_matcher="FITNESS_UNDO_MEAL_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="clear_today_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="limpa as refeições de hoje",
        expected_mode="FITNESS_CLEAR",
        expected_matcher="FITNESS_CLEAR_TODAY_RE",
        tags=["fitness"],
    ),
    # ------- collision guards -------
    # "anota que comi …" stays with Apple Notes (trigger is "anota", not "comi").
    EvalCase(
        name="notes_not_fitness_guard",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="anota que comi pão hoje",
        expected_mode="NOTE_CREATE",
        tags=["fitness"],
    ),
    # "quanto comi hoje" is the TOTAL (MACROS), not the LIST.
    EvalCase(
        name="quanto_is_macros_not_list",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="quanto comi hoje",
        expected_mode="FITNESS_MACROS",
        tags=["fitness"],
    ),
    # ------- slice 2: workouts (runs + lifts + trends) -------
    EvalCase(
        name="log_run_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="registra corrida 5km em 25min",
        expected_mode="FITNESS_RUN_LOG",
        expected_matcher="FITNESS_LOG_RUN_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="log_run_corri_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="corri 5km em 25 minutos",
        expected_mode="FITNESS_RUN_LOG",
        expected_matcher="FITNESS_LOG_RUN_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="log_lift_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="supino 3x10 80kg",
        expected_mode="FITNESS_LIFT_LOG",
        expected_matcher="FITNESS_LOG_LIFT_RE",
        tags=["fitness"],
    ),
    # (lift-trend routing has mode=None — it falls through when the captured
    #  word isn't a logged exercise — so it's covered by unit tests, not a
    #  DISPATCH eval, like CONTACT_INFO.)
    EvalCase(
        name="run_trend_pace_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="meu pace",
        expected_mode="FITNESS_RUN_TREND",
        expected_matcher="FITNESS_RUN_TREND_RE",
        tags=["fitness"],
    ),
    # guard: "como tá minha corrida" → RUN trend, not the generic lift capture.
    EvalCase(
        name="corrida_is_run_trend_not_lift",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="como tá minha corrida",
        expected_mode="FITNESS_RUN_TREND",
        tags=["fitness"],
    ),
    # workout cleanup (mirror the meal undo/clear)
    EvalCase(
        name="undo_workout_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="desfaz o último treino",
        expected_mode="FITNESS_WORKOUT_UNDO",
        expected_matcher="FITNESS_UNDO_WORKOUT_RE",
        tags=["fitness"],
    ),
    EvalCase(
        name="clear_workouts_pt",
        kind=EvalKind.DISPATCH,
        category="fitness",
        input="limpa os treinos de hoje",
        expected_mode="FITNESS_WORKOUT_CLEAR",
        expected_matcher="FITNESS_CLEAR_WORKOUTS_RE",
        tags=["fitness"],
    ),
]
