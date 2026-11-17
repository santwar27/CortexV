"""Unit tests for the Fitness Engine v1 store (src/integrations/fitness.py)."""
import pytest

from src.core import fitness_mixin
from src.integrations import fitness as fitness_mod
from src.integrations.fitness import FitnessLog, LoggedItem


@pytest.fixture
def fl(tmp_path):
    return FitnessLog(db_path=str(tmp_path / "fitness.db"))


# ---------- seeding + food lookup ----------

def test_seed_populates_foods(fl):
    assert fl.lookup_food("frango") is not None
    assert fl.lookup_food("arroz") is not None


def test_lookup_accent_and_case_insensitive(fl):
    # "maçã" seeded with accent — query without it / in caps still resolves.
    assert fl.lookup_food("maca").name == "maçã"
    assert fl.lookup_food("MAÇÃ").name == "maçã"


def test_lookup_by_alias(fl):
    assert fl.lookup_food("peito de frango").name == "frango"
    assert fl.lookup_food("ovos").name == "ovo"


def test_lookup_fuzzy(fl):
    assert fl.lookup_food("frnago").name == "frango"  # typo
    assert fl.lookup_food("arrloz").name == "arroz"


def test_lookup_miss(fl):
    assert fl.lookup_food("dragonfruit do cerrado") is None
    assert fl.lookup_food("") is None


# ---------- grams resolution ----------

def test_resolve_grams_units(fl):
    """Returns (grams, assumed) — `assumed` marks a weight WE chose because
    the user named none, so the reply can admit it (s82)."""
    egg = fl.lookup_food("ovo")
    chicken = fl.lookup_food("frango")
    assert fl._resolve_grams(200, "g", chicken) == (200, False)
    assert fl._resolve_grams(1, "kg", chicken) == (1000, False)
    assert fl._resolve_grams(250, "ml", fl.lookup_food("leite")) == (250, False)
    # "2 ovos" — countable, no unit, small qty → count × unit_g (50)
    assert fl._resolve_grams(2, None, egg) == (100, False)
    # "200 arroz" — no unit, not small → grams
    assert fl._resolve_grams(200, None, fl.lookup_food("arroz")) == (200, False)
    # explicit "unidade"
    assert fl._resolve_grams(3, "unidades", egg) == (150, False)
    # no qty at all → the food's typical portion, and flagged as assumed
    assert fl._resolve_grams(None, None, chicken) == (150, True)   # portion_g
    assert fl._resolve_grams(None, None, egg) == (50, True)        # unit_g


def test_resolve_grams_falls_back_when_no_portion(fl):
    # A food with neither portion_g nor unit_g still gets the 100 g default,
    # and it is still flagged: the user did not choose it, we did.
    fl.add_food("blargfruit", kcal=10, protein=1, carb=1, fat=0)
    grams, assumed = fl._resolve_grams(None, None, fl.lookup_food("blargfruit"))
    assert (grams, assumed) == (100.0, True)


# ---------- parsing ----------

def test_parse_meal_multi_item_macros(fl):
    items, unknown = fl.parse_meal("200g de frango e 2 ovos e 150g de arroz")
    assert unknown == []
    by = {it.food: it for it in items}
    assert by["frango"].kcal == 330.0 and by["frango"].protein == 62.0
    assert by["ovo"].grams == 100.0 and by["ovo"].kcal == 143.0
    assert by["arroz"].carb == 42.0


def test_parse_meal_collects_unknown(fl):
    items, unknown = fl.parse_meal("100g de frango e 50g de blarg")
    assert [it.food for it in items] == ["frango"]
    assert unknown == ["blarg"]


def test_parse_meal_empty(fl):
    assert fl.parse_meal("") == ([], [])


def test_parse_meal_connectors(fl):
    # comma / plus / "com" all split items
    items, _ = fl.parse_meal("100g de arroz, 100g de feijão + 100g de frango")
    assert {it.food for it in items} == {"arroz", "feijão", "frango"}


def test_parse_meal_en_connectors_and_aliases(fl):
    # "and"/"with" split EN meals; EN food names resolve via aliases
    items, _ = fl.parse_meal("100g of rice and 2 eggs")
    assert {it.food for it in items} == {"arroz", "ovo"}


# ---------- logging + daily totals ----------

def test_log_and_today_totals(fl):
    items, _ = fl.parse_meal("200g de frango e 150g de arroz")
    assert fl.log_meal(items) is True
    t = fl.today_totals()
    assert t["kcal"] == 525.0      # 330 + 195
    assert t["protein"] == 66.1    # 62 + 4.1
    assert t["n"] == 2


def test_today_totals_accumulates_across_meals(fl):
    fl.log_meal(fl.parse_meal("100g de frango")[0])
    fl.log_meal(fl.parse_meal("100g de arroz")[0])
    t = fl.today_totals()
    assert t["n"] == 2
    assert t["kcal"] == pytest.approx(165 + 130, abs=0.1)


def test_log_meal_empty_is_false(fl):
    assert fl.log_meal([]) is False


def test_today_totals_empty_db(fl):
    t = fl.today_totals()
    assert t == {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0, "n": 0}


# ---------- extending the food table ----------

def test_add_food_then_lookup(fl):
    assert fl.add_food("paçoca", 470, 12, 55, 22, unit_g=20) is True
    f = fl.lookup_food("paçoca")
    assert f is not None and f.kcal == 470 and f.unit_g == 20


def test_add_food_upsert_overwrites(fl):
    fl.add_food("tapioca", 240, 0, 60, 0)
    fl.add_food("tapioca", 999, 1, 2, 3)  # corrected values
    assert fl.lookup_food("tapioca").kcal == 999


# ---------- fail-soft ----------

def test_unwritable_db_degrades(tmp_path):
    # A path under a file (not a dir) can't be opened → fail-soft, no raise.
    bad = tmp_path / "afile"
    bad.write_text("x")
    fl = FitnessLog(db_path=str(bad / "nested.db"))
    assert fl.lookup_food("frango") is None
    assert fl.log_meal([LoggedItem("frango", 100, 165, 31, 0, 3.6)]) is False
    assert fl.today_totals()["n"] == 0


# ---------- orchestrator handlers (mixin) ----------

class _StubOrch(fitness_mixin.FitnessMixin):
    """Minimal host: the mixin only needs `_t` (I18nMixin) in PT."""
    _reply_lang = "pt"

    def _t(self, pt, en):
        return pt


@pytest.fixture
def orch(tmp_path, monkeypatch):
    log = FitnessLog(db_path=str(tmp_path / "f.db"))
    monkeypatch.setattr(fitness_mod, "_log_singleton", log)
    return _StubOrch()


def test_handler_log_meal_reports_and_totals(orch):
    reply, mode = orch._handle_fitness_log_meal("200g de frango e 2 ovos")
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "frango" in reply and "kcal" in reply and "proteína" in reply
    # the day total now reflects the logged meal
    reply2, mode2 = orch._handle_fitness_macros_today()
    assert mode2 == "FITNESS_MACROS_TODAY"
    assert "473" in reply2  # 330 (frango) + 143 (2 ovos)


def test_handler_unknown_food(orch):
    reply, mode = orch._handle_fitness_log_meal("50g de blarg")
    assert mode == "FITNESS_MEAL_UNKNOWN"
    assert "blarg" in reply


def test_handler_en_trigger_replies_en_and_maps_alias(orch):
    # EN trigger → EN reply; "rice" alias resolves to the PT canonical "arroz".
    reply, mode = orch._handle_fitness_log_meal("100g of rice", raw_text="I ate 100g of rice")
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "Logged" in reply and "protein" in reply  # English
    assert "arroz" in reply                            # canonical food name


def test_lang_follows_trigger_not_heuristic(orch):
    # Unaccented PT input that the heuristic would call EN still replies PT,
    # because the trigger "comi" is unambiguous.
    reply, _ = orch._handle_fitness_log_meal("200g de frango e arroz",
                                             raw_text="comi 200g de frango e arroz")
    assert "proteína" in reply  # PT


def test_handler_partial_unknown_still_logs_known(orch):
    reply, mode = orch._handle_fitness_log_meal("100g de frango e 50g de blarg")
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "frango" in reply and "blarg" in reply  # logged + noted unknown


def test_handler_empty_body_falls_through(orch):
    assert orch._handle_fitness_log_meal("") is None


def test_handler_macros_empty_day(orch):
    reply, mode = orch._handle_fitness_macros_today()
    assert mode == "FITNESS_MACROS_EMPTY"


# ---------- list / undo / clear (store) ----------

def test_list_today_groups_by_meal(fl):
    fl.log_meal(fl.parse_meal("200g de frango e 150g de arroz")[0])
    fl.log_meal(fl.parse_meal("1 banana")[0])
    meals = fl.list_today()
    assert len(meals) == 2  # two separate log events
    foods = {f for g, f in meals[0]["items"]}
    assert foods == {"frango", "arroz"}
    assert meals[0]["kcal"] == 525.0


def test_undo_last_meal_removes_latest_group(fl):
    fl.log_meal(fl.parse_meal("200g de frango e 150g de arroz")[0])  # 2 items
    fl.log_meal(fl.parse_meal("1 banana")[0])                        # 1 item
    assert fl.undo_last_meal() == 1            # the banana meal
    assert fl.today_totals()["n"] == 2
    assert fl.undo_last_meal() == 2            # the frango+arroz meal
    assert fl.today_totals()["n"] == 0
    assert fl.undo_last_meal() == 0            # nothing left


def test_clear_today_wipes_day(fl):
    fl.log_meal(fl.parse_meal("200g de frango e 150g de arroz")[0])
    fl.log_meal(fl.parse_meal("1 banana")[0])
    assert fl.clear_today() == 3
    assert fl.today_totals()["n"] == 0
    assert fl.clear_today() == 0


# ---------- list / undo / clear (handlers) ----------

def test_handler_list_today(orch):
    orch._handle_fitness_log_meal("200g de frango", raw_text="comi 200g de frango")
    reply, mode = orch._handle_fitness_list_today(raw_text="o que comi hoje")
    assert mode == "FITNESS_LIST_TODAY"
    assert "frango" in reply and "Total" in reply


def test_handler_list_empty(orch):
    reply, mode = orch._handle_fitness_list_today()
    assert mode == "FITNESS_LIST_EMPTY"


def test_handler_undo_then_empty(orch):
    orch._handle_fitness_log_meal("200g de frango", raw_text="comi 200g de frango")
    _, mode = orch._handle_fitness_undo_meal(raw_text="desfaz a última refeição")
    assert mode == "FITNESS_UNDO"
    _, mode2 = orch._handle_fitness_undo_meal()
    assert mode2 == "FITNESS_UNDO_EMPTY"


def test_handler_clear_today(orch):
    orch._handle_fitness_log_meal("200g de frango e arroz",
                                  raw_text="comi 200g de frango e arroz")
    _, mode = orch._handle_fitness_clear_today(raw_text="limpa as refeições de hoje")
    assert mode == "FITNESS_CLEAR"
    assert orch._handle_fitness_macros_today()[1] == "FITNESS_MACROS_EMPTY"


# ---------- workouts: store ----------

def test_log_run_and_history_pace(fl):
    assert fl.log_run(5, 25) is True
    runs = fl.run_history()
    assert len(runs) == 1
    assert runs[0]["pace_min_km"] == 5.0  # 25 / 5


def test_log_lift_and_exercise_history(fl):
    fl.log_lift("supino", 3, 10, 70)
    fl.log_lift("supino reto", 3, 10, 80)  # "supino" query matches both
    hist = fl.exercise_history("supino")
    assert len(hist) == 2
    assert hist[-1]["weight_kg"] == 80
    assert hist[-1]["volume"] == 2400.0  # 3 * 10 * 80


def test_exercise_history_miss(fl):
    assert fl.exercise_history("supino") == []
    assert fl.exercise_history("") == []


def test_log_run_lift_reject_bad(fl):
    assert fl.log_run(0, 25) is False
    assert fl.log_lift("", 3, 10, 80) is False
    assert fl.log_lift("supino", 0, 10, 80) is False


# ---------- workouts: pace helper ----------

def test_pace_str():
    from src.core.fitness_mixin import _pace_str
    assert _pace_str(5.0) == "5:00"
    assert _pace_str(4.8) == "4:48"
    assert _pace_str(0) == "—"


# ---------- workouts: handlers ----------

def test_handler_log_run(orch):
    reply, mode = orch._handle_fitness_log_run("5", "25",
                                               raw_text="registra corrida 5km em 25min")
    assert mode == "FITNESS_RUN_LOGGED"
    assert "5:00/km" in reply


def test_handler_log_lift(orch):
    reply, mode = orch._handle_fitness_log_lift("supino", "3", "10", "80",
                                                raw_text="supino 3x10 80kg")
    assert mode == "FITNESS_LIFT_LOGGED"
    assert "2400" in reply  # volume = 3*10*80


def test_handler_lift_trend_progression(orch):
    orch._handle_fitness_log_lift("supino", "3", "10", "70", raw_text="supino 3x10 70kg")
    orch._handle_fitness_log_lift("supino", "3", "10", "80", raw_text="supino 3x10 80kg")
    reply, mode = orch._handle_fitness_lift_trend("supino", raw_text="como tá meu supino")
    assert mode == "FITNESS_LIFT_TREND"
    assert "70" in reply and "80" in reply


def test_handler_lift_trend_falls_through_when_no_history(orch):
    # "como tá meu dia" → ex="dia", no history → None → LLM handles it
    assert orch._handle_fitness_lift_trend("dia", raw_text="como tá meu dia") is None


def test_handler_run_trend_empty(orch):
    reply, mode = orch._handle_fitness_run_trend(raw_text="meu pace")
    assert mode == "FITNESS_RUN_TREND_EMPTY"


def test_handler_run_trend_shows_runs(orch):
    orch._handle_fitness_log_run("5", "26", raw_text="corri 5km em 26min")
    orch._handle_fitness_log_run("5", "25", raw_text="corri 5km em 25min")
    reply, mode = orch._handle_fitness_run_trend(raw_text="meu pace")
    assert mode == "FITNESS_RUN_TREND"
    assert "km" in reply


# ---------- workout cleanup: store ----------

def test_undo_last_workout_returns_and_deletes(fl):
    fl.log_run(5, 25)
    fl.log_lift("supino", 3, 10, 80)
    w = fl.undo_last_workout()          # the lift (most recent)
    assert w["kind"] == "lift" and w["name"] == "supino"
    assert fl.exercise_history("supino") == []
    assert len(fl.run_history()) == 1   # the run survives
    w2 = fl.undo_last_workout()
    assert w2["kind"] == "run"
    assert fl.undo_last_workout() is None


def test_clear_workouts_today(fl):
    fl.log_run(5, 25)
    fl.log_lift("supino", 3, 10, 80)
    assert fl.clear_workouts_today() == 2
    assert fl.run_history() == []
    assert fl.clear_workouts_today() == 0


# ---------- workout cleanup: handlers ----------

def test_handler_undo_workout(orch):
    orch._handle_fitness_log_run("5", "25", raw_text="registra corrida 5km em 25min")
    reply, mode = orch._handle_fitness_undo_workout(raw_text="desfaz o último treino")
    assert mode == "FITNESS_WORKOUT_UNDO"
    assert "corrida" in reply
    _, mode2 = orch._handle_fitness_undo_workout()
    assert mode2 == "FITNESS_WORKOUT_UNDO_EMPTY"


def test_handler_clear_workouts(orch):
    orch._handle_fitness_log_lift("supino", "3", "10", "80", raw_text="supino 3x10 80kg")
    _, mode = orch._handle_fitness_clear_workouts(raw_text="limpa os treinos de hoje")
    assert mode == "FITNESS_WORKOUT_CLEAR"
    assert orch._handle_fitness_run_trend(raw_text="meu pace")[1] == "FITNESS_RUN_TREND_EMPTY"


# ---------- sentence punctuation between items (s82) ----------
# The user logged "55 gramas de aveia. 100 gramas de yoghurt grego e 80 gramas
# de banana" and the oats VANISHED: "." was not a separator, so the first two
# items stayed one chunk, the item regex read qty=55, and the food lookup
# fuzzy-matched the last food named. The yoghurt was logged at 55g instead of
# 100g — with `unknown` empty, so nothing signalled the loss.

def test_parse_meal_splits_on_full_stop(fl):
    items, unknown = fl.parse_meal("200g de frango. 150g de arroz")
    assert unknown == []
    assert sorted(it.food for it in items) == ["arroz", "frango"]
    by = {it.food: it for it in items}
    assert by["frango"].grams == 200.0 and by["arroz"].grams == 150.0


def test_parse_meal_full_stop_and_connector_together(fl):
    # The exact shape of the real sentence: "A. B e C".
    items, _ = fl.parse_meal("200g de frango. 150g de arroz e 2 ovos")
    assert sorted(it.food for it in items) == ["arroz", "frango", "ovo"]


def test_parse_meal_decimal_comma_is_not_a_separator(fl):
    # The reason "." and "," need digit lookarounds: a naive split would cut
    # "1,5 kg de frango" into "1" and "5 kg de frango".
    items, unknown = fl.parse_meal("1,5 kg de frango")
    assert unknown == []
    assert len(items) == 1
    assert items[0].food == "frango" and items[0].grams == 1500.0


def test_parse_meal_decimal_point_is_not_a_separator(fl):
    items, unknown = fl.parse_meal("1.5 kg de arroz")
    assert unknown == []
    assert len(items) == 1 and items[0].grams == 1500.0


def test_parse_meal_never_drops_a_chunk_silently(fl):
    # An unparseable chunk must SHOW UP, not disappear: a quietly wrong total
    # is worse than an honest "I didn't recognise this".
    items, unknown = fl.parse_meal("200g de frango. ????")
    assert [it.food for it in items] == ["frango"]
    assert unknown == ["????"]


def test_parse_meal_trailing_full_stop_is_harmless(fl):
    items, unknown = fl.parse_meal("200g de frango.")
    assert unknown == [] and [it.food for it in items] == ["frango"]


# ---------- the trigger's leading prefix (s82) ----------
# The regex was anchored at "^comi", so the way the user actually types it —
# "hoje eu comi …" — never routed, fell through to the LLM, and the 8B
# invented the macros (it claimed 235g of carbs for a ~59g breakfast).

from src.core.dispatch.registry import simulate  # noqa: E402


@pytest.mark.parametrize("phrase", [
    "comi 55 gramas de aveia",
    "eu comi 55 gramas de aveia",
    "hoje comi 55 gramas de aveia",
    "hoje eu comi 55 gramas de aveia",
    "ontem eu comi frango",
    "agora comi uma banana",
    "i ate two eggs",
    "today i ate two eggs",
    "registra refeição: 200g de frango",
    # Round two (s82 cont.): the first fix enumerated prefixes and these
    # still failed. Enumerating sentence openings has no end — the trigger
    # searches for the verb instead.
    "cortex, no café da manhã de hoje eu comi 55 gramas de aveia",
    "no café da manhã de hoje eu comi aveia",
    "as 12h eu comi um sanduiche com peito de frango",
    "no almoço eu comi arroz e feijao",
    "no jantar de ontem eu comi 200g de frango",
    "depois do treino eu comi uma banana",
])
def test_meal_trigger_accepts_natural_prefixes(phrase):
    m = simulate(phrase)
    assert m is not None and m.mode == "FITNESS_MEAL", phrase


@pytest.mark.parametrize("phrase,expected", [
    # A verb SEARCH is broad, so what protects the neighbours is registry
    # ORDER: the reading routes are registered before the logging one.
    ("anota que comi pizza", "NOTE_CREATE"),      # Apple Notes keeps this
    ("quanto eu comi hoje", "FITNESS_MACROS"),    # reading, not logging
    ("quantos macros eu comi hoje", "FITNESS_MACROS"),
    ("o que eu comi hoje", "FITNESS_LIST"),       # listing, not logging
])
def test_meal_trigger_does_not_hijack_neighbours(phrase, expected):
    m = simulate(phrase)
    assert m is not None and m.mode == expected, phrase


@pytest.mark.parametrize("phrase", [
    "nao comi nada hoje",
    "não comi nada",
    "nunca comi sushi",
    # unaccented "até" is a real PT word — a mid-sentence search on bare
    # "ate" would misfire, so that trigger stays anchored at the start
    "vou ate a loja comprar pao",
    "fiquei ate as 10 da noite",
])
def test_meal_trigger_ignores_negation_and_ate_as_ate(phrase):
    m = simulate(phrase)
    assert m is None or m.mode != "FITNESS_MEAL", phrase


def test_the_real_breakfast_routes_and_totals(orch):
    """End to end on the exact sentence that failed in production: it must
    route to the meal handler AND come back with the food table's numbers,
    not the model's imagination."""
    sentence = ("hoje eu comi 55 gramas de aveia. 100 gramas de yoghurt "
                "grego e 80 gramas de banana")
    m = simulate(sentence)
    assert m is not None and m.mode == "FITNESS_MEAL"
    body = m.capture          # the food list, minus the trigger + prefix
    assert body.startswith("55 gramas de aveia")
    reply, mode = orch._handle_fitness_log_meal(body, raw_text=sentence)
    assert mode == "FITNESS_MEAL_LOGGED"
    # All three foods survive, each with the quantity actually said — the
    # regression was the oats vanishing and the yoghurt taking their 55g.
    assert "aveia" in reply and "banana" in reply and "iogurte" in reply
    assert "55g de aveia" in reply
    assert "100g de iogurte" in reply
    assert "80g de banana" in reply


def test_the_real_second_attempt_routes(orch):
    """Round two of the production sentence — the one that STILL failed after
    the prefix-enumeration fix. Vocative, meal name, and a date phrase all
    sit between the start and the verb."""
    sentence = ("cortex, no café da manhã de hoje eu comi 55 gramas de aveia, "
                "100 gramas de yoghurt grego, e 80 gramas de banana")
    m = simulate(sentence)
    assert m is not None and m.mode == "FITNESS_MEAL"
    reply, mode = orch._handle_fitness_log_meal(m.capture, raw_text=sentence)
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "55g de aveia" in reply and "80g de banana" in reply


def test_weightless_meal_uses_flagged_portions_not_invented_numbers(orch):
    """The other real sentence gave no weights and asked for an average. The
    model had answered "aproximadamente 350 gramas" — a number it made up.
    Now: known foods log at their typical portion, each marked as assumed,
    and the trailing clause is not paraded as an unknown food."""
    sentence = ("as 12h eu comi peito de frango e tomate. "
                "nao sei o peso, mas faça na media")
    m = simulate(sentence)
    assert m is not None and m.mode == "FITNESS_MEAL"
    reply, mode = orch._handle_fitness_log_meal(m.capture, raw_text=sentence)
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "150g de frango*" in reply and "80g de tomate*" in reply
    assert "porção média" in reply          # the assumption is admitted
    assert "nao sei o peso" not in reply    # chatter isn't a food


# ---------- lookup specificity, portions, table growth (s82) ----------

def test_qualified_food_beats_a_generic_alias(fl):
    """The bug: "yoghurt grego" resolved to *iogurte natural* on a fresh DB,
    because the plain yoghurt's "yoghurt" alias is a substring of the query
    and its row comes first. 40% less protein, silently."""
    assert fl.lookup_food("yoghurt grego").name == "iogurte grego"
    assert fl.lookup_food("iogurte grego").name == "iogurte grego"
    assert fl.lookup_food("greek yogurt").name == "iogurte grego"
    # ...and the unqualified word still resolves to the plain one
    assert fl.lookup_food("yoghurt").name == "iogurte natural"
    assert fl.lookup_food("iogurte").name == "iogurte natural"


def test_specificity_does_not_break_ordinary_lookups(fl):
    assert fl.lookup_food("peito frango").name == "frango"
    assert fl.lookup_food("peito de frango").name == "frango"
    assert fl.lookup_food("batata doce").name == "batata doce"
    assert fl.lookup_food("batata").name == "batata"
    assert fl.lookup_food("arroz integral").name == "arroz integral"
    assert fl.lookup_food("arroz").name == "arroz"


def test_typical_portion_used_when_no_weight_given(fl):
    items, unknown = fl.parse_meal("arroz e frango")
    assert unknown == []
    by = {it.food: it for it in items}
    assert by["arroz"].grams == 150.0 and by["arroz"].assumed is True
    assert by["frango"].grams == 150.0 and by["frango"].assumed is True


def test_stated_weight_is_never_marked_assumed(fl):
    items, _ = fl.parse_meal("200g de frango")
    assert items[0].grams == 200.0 and items[0].assumed is False


def test_reply_flags_assumed_portions(orch):
    reply, mode = orch._handle_fitness_log_meal("arroz e frango",
                                                raw_text="comi arroz e frango")
    assert mode == "FITNESS_MEAL_LOGGED"
    assert "*" in reply and "porção média" in reply


def test_reply_has_no_footnote_when_every_weight_was_given(orch):
    reply, _ = orch._handle_fitness_log_meal("200g de frango e 150g de arroz",
                                             raw_text="comi 200g de frango")
    assert "porção média" not in reply and "*" not in reply


def test_new_everyday_foods_resolve(fl):
    # Each of these came back "não reconheci" in real use.
    for name in ("milho", "pepino", "presunto", "cheddar", "wrap", "manga",
                 "cogumelos", "grão de bico", "espinafre", "mel"):
        assert fl.lookup_food(name) is not None, name


def test_seed_syncs_into_an_existing_db(tmp_path):
    """The seed used to bail out when the table was non-empty, so an existing
    install never received a new food or a corrected alias — which is why
    Pedro's months-old DB still carried the original two-alias rows."""
    path = str(tmp_path / "f.db")
    first = FitnessLog(db_path=path)
    conn = first._connect()
    conn.execute("DELETE FROM foods WHERE name = 'milho'")
    conn.execute("UPDATE foods SET aliases = 'apenas-isto' WHERE name = 'frango'")
    conn.commit()
    first.close() if hasattr(first, "close") else None

    reopened = FitnessLog(db_path=path)          # re-init runs the sync
    assert reopened.lookup_food("milho") is not None          # re-added
    assert reopened.lookup_food("chicken").name == "frango"   # alias restored


def test_seed_sync_preserves_user_corrected_macros(tmp_path):
    # add_food is how a user fixes a wrong figure; re-seeding must not undo it.
    path = str(tmp_path / "f.db")
    first = FitnessLog(db_path=path)
    first.add_food("frango", kcal=999, protein=99, carb=9, fat=9)
    reopened = FitnessLog(db_path=path)
    assert reopened.lookup_food("frango").kcal == 999


def test_sentence_fragments_are_not_reported_as_unknown_foods(fl):
    """The honest "não reconheci X" is for things that look like food. The
    clause the user wrapped around the list ("nao sei o peso, mas faça na
    media") is not a food claim and reporting it is just noise."""
    items, unknown = fl.parse_meal(
        "peito de frango e tomate. nao sei o peso, mas faça na media")
    assert sorted(it.food for it in items) == ["frango", "tomate"]
    assert unknown == []


def test_a_real_unknown_food_is_still_reported(fl):
    items, unknown = fl.parse_meal("200g de frango e um sanduiche")
    assert [it.food for it in items] == ["frango"]
    assert unknown == ["um sanduiche"]


# ---------- meal NAMES are context, not food (s82) ----------
# "hoje eu comi no café da manhã; 55 gramas de aveia…" logged 200g de CAFÉ:
# the ";" made "no cafe da manha" its own chunk and the lookup matched the
# word "café" inside it. the user spotted it precisely because he HAD drunk
# coffee but never said so — the system invented a food from a time label.

@pytest.mark.parametrize("label", [
    "no cafe da manha", "no café da manhã", "café da manhã",
    "no almoço", "no jantar", "na janta", "no lanche", "no lanche da tarde",
    "de manhã", "à noite", "at breakfast", "for lunch", "dinner",
])
def test_meal_labels_are_not_logged_as_food(fl, label):
    items, unknown = fl.parse_meal(f"{label}; 200g de frango")
    assert [it.food for it in items] == ["frango"], label
    assert unknown == [], label


def test_the_reported_sentence_logs_only_the_three_foods(fl):
    items, unknown = fl.parse_meal(
        "no cafe da manha; 55 gramas de aveia, 100 gramas de iorgurte "
        "e 80 gramas de banana")
    assert [it.food for it in items] == ["aveia", "iogurte natural", "banana"]
    assert [it.grams for it in items] == [55.0, 100.0, 80.0]
    assert unknown == []


def test_coffee_the_drink_still_logs(fl):
    """Only the full "café da manhã" phrase is a meal name. Coffee itself is
    a thing people log, so the bare word must survive."""
    items, _ = fl.parse_meal("200ml de cafe")
    assert [it.food for it in items] == ["café"]
    items, _ = fl.parse_meal("um cafe e uma banana")
    assert sorted(it.food for it in items) == ["banana", "café"]


# ---------- delete intents must not be answered with a listing (s82) ----------
# Broadening the READ pattern made it swallow "apaga tudo que comi hoje" (it
# contains "comi"), and the broad LOG trigger swallowed it too. Answering a
# deletion with a listing — or worse, logging it as a meal — leaves the user
# believing nothing happened.

@pytest.mark.parametrize("phrase", [
    "esquece o que eu comi hoje",
    "esquece minhas refeições de hoje",
    "apaga tudo que comi hoje",
    "apaga as refeições de hoje",
    "limpa as refeições de hoje",
    "zera as refeicoes de hoje",
])
def test_clearing_the_day_is_not_read_as_listing_or_logging(phrase):
    m = simulate(phrase)
    assert m is not None and m.mode == "FITNESS_CLEAR", phrase


@pytest.mark.parametrize("phrase", [
    "desfaz a última refeição",
    "apaga a última refeição",
    "remove a ultima refeicao",
    "undo last meal",
])
def test_undo_last_meal_still_routes(phrase):
    m = simulate(phrase)
    assert m is not None and m.mode == "FITNESS_UNDO", phrase


def test_clearing_requires_an_explicit_today():
    # Wiping the day is destructive; it must never be inferred from a vague
    # sentence that never says which day.
    m = simulate("esquece as refeicoes")
    assert m is None or m.mode != "FITNESS_CLEAR"


@pytest.mark.parametrize("phrase,expected", [
    ("o que eu comi hoje", "FITNESS_LIST"),
    ("o que eu comi no café da manhã de hoje?", "FITNESS_LIST"),
    ("quantos macros eu comi hoje", "FITNESS_MACROS"),
    ("hoje eu comi aveia", "FITNESS_MEAL"),
    ("no cafe da manha eu comi 55g de aveia", "FITNESS_MEAL"),
])
def test_the_delete_guard_leaves_the_other_intents_alone(phrase, expected):
    m = simulate(phrase)
    assert m is not None and m.mode == expected, phrase


def test_a_meal_that_was_not_saved_is_not_reported_logged(orch, monkeypatch):
    """With another process holding the write, log_meal returned False and the
    reply still said "Registrei: … 330 kcal" (s88, Codex lote 11)."""
    from src.integrations import fitness as fm
    events = []
    monkeypatch.setattr("src.core.fitness_mixin.audit.log",
                        lambda *a, **k: events.append(a[2]))
    monkeypatch.setattr(fm.get_log(), "log_meal", lambda items, **_k: False)
    reply, mode = orch._handle_fitness_log_meal("200g de frango",
                                                raw_text="comi 200g de frango")
    assert mode == "FITNESS_MEAL_ERROR"
    assert "Registrei" not in reply and "nada foi gravado" in reply
    assert "meal_logged" not in events and "meal_log_failed" in events
    assert fm.get_log().today_totals()["n"] == 0



def test_undo_takes_the_last_logged_across_the_clock_change(tmp_path, monkeypatch):
    """25/10/2026, clocks back at 02:00 BST: a meal at 01:50 BST, then one at
    01:10 GMT an hour later. ts is local wall time, so "desfaz a última" took
    the 01:50 one (s88, Codex lote 11). The same for workouts."""
    from datetime import datetime
    from src.integrations import fitness as fm

    class Clock(datetime):
        at = None

        @classmethod
        def now(cls, tz=None):
            return cls.at
    monkeypatch.setattr(fm, "datetime", Clock)
    log = FitnessLog(db_path=str(tmp_path / "f.db"))
    Clock.at = datetime(2026, 10, 25, 1, 50)
    log.log_meal(log.parse_meal("200g de frango")[0])
    log.log_run(5, 25)
    Clock.at = datetime(2026, 10, 25, 1, 10)
    log.log_meal(log.parse_meal("100g de arroz")[0])
    log.log_lift("supino", 3, 10, 80)
    log.undo_last_meal()
    removed = log.undo_last_workout()
    conn = log._connect()
    assert [r[0] for r in conn.execute("SELECT food FROM meals")] == ["frango"]
    assert removed["kind"] == "lift"
    assert [r[0] for r in conn.execute("SELECT kind FROM workouts")] == ["run"]


class TestTheDayAMealWasEaten:
    """"ontem eu comi 200g de frango" at 00:30 on 18/10 went into 18/10 and
    raised today's macros: raw_text only chose the language (s88, Codex lote
    11 — invariant 1)."""

    @pytest.fixture
    def clock(self, monkeypatch):
        from datetime import datetime
        from src.integrations import fitness as fm

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 10, 18, 0, 30)
        monkeypatch.setattr(fm, "datetime", Clock)

    def _dates(self):
        conn = fitness_mod.get_log()._connect()
        return [r[0][:10] for r in conn.execute("SELECT ts FROM meals")]

    @pytest.mark.parametrize("text,date", [
        ("ontem eu comi 200g de frango", "2026-10-17"),
        ("anteontem eu comi 200g de frango", "2026-10-16"),
        ("comi 200g de frango", "2026-10-18"),
        ("hoje eu comi 200g de frango", "2026-10-18"),
    ])
    def test_it_is_logged_on_that_day(self, orch, clock, text, date):
        reply, mode = orch._handle_fitness_log_meal("200g de frango", raw_text=text)
        assert mode == "FITNESS_MEAL_LOGGED" and self._dates() == [date]

    def test_yesterday_leaves_today_alone(self, orch, clock):
        reply, _ = orch._handle_fitness_log_meal("200g de frango",
                                                 raw_text="ontem eu comi 200g de frango")
        assert fitness_mod.get_log().today_totals()["n"] == 0
        assert "Registrei em ontem" in reply and "Ontem no total: 330 kcal" in reply

    @pytest.mark.parametrize("text", ["na segunda eu comi 200g de frango",
                                      "semana passada eu comi 200g de frango",
                                      "no dia 12 eu comi 200g de frango"])
    def test_another_past_day_is_refused_not_logged_today(self, orch, clock, text):
        _reply, mode = orch._handle_fitness_log_meal("200g de frango", raw_text=text)
        assert mode == "FITNESS_MEAL_DATE_UNSUPPORTED" and self._dates() == []


class TestAHouseholdMeasureIsNotGrams:
    """"comi 2 colheres de azeite" logged 2 g, 17.7 kcal, unmarked: azeite
    matched inside "colheres de azeite" and the 2 became grams (s88, Codex
    lote 11 — invariant 1)."""

    @pytest.mark.parametrize("body", ["2 colheres de azeite", "1 xícara de arroz",
                                      "meia xícara de aveia", "1 copo de leite",
                                      "2 tbsp of olive oil"])
    def test_nothing_is_logged_and_grams_are_asked(self, orch, body):
        reply, mode = orch._handle_fitness_log_meal(body, raw_text=f"comi {body}")
        assert mode == "FITNESS_MEAL_MEASURE" and "gramas" in reply
        assert fitness_mod.get_log().today_totals()["n"] == 0

    def test_the_rest_of_the_meal_is_logged_and_the_measure_named(self, orch):
        reply, mode = orch._handle_fitness_log_meal(
            "200g de frango e 2 colheres de azeite",
            raw_text="comi 200g de frango e 2 colheres de azeite")
        assert mode == "FITNESS_MEAL_LOGGED"
        assert "não registrei 2 colheres de azeite" in reply
        assert fitness_mod.get_log().today_totals()["kcal"] == 330

    @pytest.mark.parametrize("body,grams", [("15g de azeite", 15), ("2 ovos", None),
                                            ("200 arroz", 200)])
    def test_weights_and_counts_still_log(self, orch, body, grams):
        items, _unknown, unmeasured = fitness_mod.get_log().parse_meal_detailed(body)
        assert items and not unmeasured
        if grams is not None:
            assert items[0].grams == grams


# --- two processes at once (s88, Codex lote 11) -----------------------------
# Two FitnessLog instances on one file stand for two processes; a barrier
# right after each SELECT forces both to read before either writes. Before:
# the same run stored twice, and both undos said they removed the one
# workout there was.
import threading as _threading  # noqa: E402


class _Paused:
    """Connection proxy: after a SELECT, wait for the other process."""
    def __init__(self, conn, barrier):
        self._c, self._b = conn, barrier
    def execute(self, sql, *a):
        cur = self._c.execute(sql, *a)
        if sql.lstrip().upper().startswith("SELECT") and "workouts" in sql:
            cur = list(cur.fetchall())
            try: self._b.wait()
            except _threading.BrokenBarrierError: pass
            return _Rows(cur)
        return cur
    def __getattr__(self, k): return getattr(self._c, k)

class _Rows:
    def __init__(self, rows): self.rows = rows
    def fetchone(self): return self.rows[0] if self.rows else None
    def fetchall(self): return self.rows


def _pair(tmp_path):
    db = str(tmp_path / "f.db")
    a, b = FitnessLog(db_path=db), FitnessLog(db_path=db)
    barrier = _threading.Barrier(2, timeout=2)
    for log in (a, b):
        real = log._connect()
        log._connect = lambda real=real: _Paused(real, barrier)
    return a, b


def _together(fa, fb):
    out = [None, None]
    ts = [_threading.Thread(target=lambda: out.__setitem__(0, fa())),
          _threading.Thread(target=lambda: out.__setitem__(1, fb()))]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


def test_two_processes_importing_one_run_store_it_once(tmp_path):
    a, b = _pair(tmp_path)
    r = _together(lambda: a.import_run(5, 25, "2026-10-18 07:30:00", start_utc="2026-10-18T06:30:00Z"),
                  lambda: b.import_run(5, 25, "2026-10-18 07:30:00", start_utc="2026-10-18T06:30:00Z"))
    n = FitnessLog(db_path=str(tmp_path / "f.db"))._connect().execute("SELECT COUNT(*) FROM workouts").fetchone()[0]
    assert n == 1


def test_two_undos_at_once_never_announce_the_same_removal(tmp_path):
    seed = FitnessLog(db_path=str(tmp_path / "f.db")); seed.log_run(5, 25)
    a, b = _pair(tmp_path)
    r = _together(a.undo_last_workout, b.undo_last_workout)
    assert sum(bool(x) for x in r) == 1
