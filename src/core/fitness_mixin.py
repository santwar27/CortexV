"""
Fitness Engine handlers for `CortexOrchestrator` (slice 1: meals + macros, s76).

Same mixin pattern as notes/weather. The contract this path pins:
  - NO LLM: parsed food → local food table → arithmetic → template string.
    Macros are computed, never guessed (the whole reason for a structured
    store instead of RAG). The eval pins the routing mode.
  - S2 local-only: logs to ~/.cortex/fitness.db, nothing leaves the machine.
  - Fail-soft: a store error degrades to an honest "couldn't log", never a
    traceback into the request path.

Relies on the host providing `self._t` (I18nMixin) and `self._reply_lang`.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from src.core.audit import audit


def _g(x: float) -> str:
    """Grams/number without a pointless trailing .0 (200.0 → "200")."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _pace_str(pace_min_km: float) -> str:
    """min/km (float) → "M:SS" per km (5.0 → "5:00", 4.8 → "4:48")."""
    if not pace_min_km or pace_min_km <= 0:
        return "—"
    m = int(pace_min_km)
    s = int(round((pace_min_km - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"


# The dispatch trigger is itself language-specific ("comi"/"meus macros hoje"
# are PT; "I ate"/"my macros today" are EN), so it's a MORE reliable language
# signal than the regex heuristic — which misfires on unaccented PT like "comi
# 200g de frango e arroz" (no accent, no PT keyword → guessed EN). We pick the
# reply language from the trigger so text AND voice both answer in kind.
_EN_TRIGGER_PREFIXES = (
    "i ate", "ate ", "log meal",
    "my macros", "macros today", "calories today", "how many calories",
    "what did i eat", "my meals",
    "undo", "remove the last meal", "remove last meal",
    "clear", "reset today",
    "run ", "ran ", "my pace", "my running", "how's my", "how s my",
)


def _lang_from_trigger(raw_text: str) -> str:
    return "en" if (raw_text or "").strip().lower().startswith(_EN_TRIGGER_PREFIXES) else "pt"


# Which day a meal was eaten, by a fixed rule on the user's words (s88, Codex lote
# 11 — invariant 1): "ontem eu comi" was logged as today and raised today's
# macros. Today, yesterday and the day before are understood; any other past
# day is refused rather than guessed into today.
_DAY_BEFORE_RE = re.compile(r"\b(?:anteontem|day\s+before\s+yesterday)\b", re.IGNORECASE)
_YESTERDAY_RE = re.compile(r"\b(?:ontem|yesterday|last\s+night)\b", re.IGNORECASE)
_OTHER_PAST_RE = re.compile(
    r"\b(?:semana\s+passada|m[êe]s\s+passado|last\s+(?:week|month)|"
    r"n[ao]\s+(?:[úu]ltim[ao]\s+)?(?:segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado|domingo)|"
    r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:no\s+)?dia\s+\d{1,2}\b|\d+\s+dias\s+atr[áa]s|\d+\s+days\s+ago)\b",
    re.IGNORECASE,
)


def _meal_days_ago(raw_text: str) -> Optional[int]:
    """0 today, 1 yesterday, 2 the day before; None for a past day not
    understood."""
    text = raw_text or ""
    if _DAY_BEFORE_RE.search(text):
        return 2
    if _YESTERDAY_RE.search(text):
        return 1
    if _OTHER_PAST_RE.search(text):
        return None
    return 0


class FitnessMixin:
    """Meal-logging + macro handlers. Numbers come from the local food table;
    nothing here calls an LLM or leaves the machine."""

    def _handle_fitness_log_meal(self, body: str,
                                 raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"comi 200g de frango e 2 ovos" → log it, report meal + day macros."""
        body = (body or "").strip()
        if not body:
            return None  # nothing after the trigger — fall through

        # Reply language from the trigger, not the (accent-blind) heuristic.
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        days_ago = _meal_days_ago(raw_text)
        if days_ago is None:
            return (
                t("Só sei registrar refeições de hoje, de ontem ou de anteontem — "
                  "não registrei nada. Diz, por exemplo, \"ontem eu comi …\".",
                  "I can only log meals for today, yesterday or the day before — "
                  "nothing was logged. Say e.g. \"yesterday I ate …\"."),
                "FITNESS_MEAL_DATE_UNSUPPORTED",
            )

        from src.integrations.fitness import get_log
        fl = get_log()
        items, unknown, unmeasured = fl.parse_meal_detailed(body)

        if not items and unmeasured:
            return (
                t(f"Não registrei: não sei quanto pesa {', '.join(unmeasured)}. "
                  f"Me diz em gramas (ex.: \"comi 15g de azeite\").",
                  f"Not logged: I don't know how much {', '.join(unmeasured)} "
                  f"weighs. Tell me in grams (e.g. \"I ate 15g of olive oil\")."),
                "FITNESS_MEAL_MEASURE",
            )

        if not items:
            # Recognised the intent ("comi …") but no known food. If there were
            # candidates we couldn't match, say so (honest, and a hook to teach
            # the engine later); otherwise fall through to the LLM.
            if unknown:
                names = ", ".join(unknown)
                return (
                    t(
                        f"Não reconheci {names} na minha base de alimentos ainda. "
                        f"Tenta com gramas (ex.: \"comi 150g de arroz\").",
                        f"I don't know {names} in my food table yet. Try grams "
                        f"(e.g. \"I ate 150g of rice\")."),
                    "FITNESS_MEAL_UNKNOWN",
                )
            return None

        if not fl.log_meal(items, days_ago=days_ago):
            # The write failed (another process holding the database, a disk
            # error): it said "Registrei" with the macros and stored nothing
            # (s88, Codex lote 11, reproduced with a held write lock).
            audit.log("–", "fitness", "meal_log_failed", payload={"items": len(items)})
            return (
                t("Não consegui registrar essa refeição agora — nada foi gravado. "
                  "Tenta de novo daqui a pouco.",
                  "I couldn't log that meal just now — nothing was saved. "
                  "Try again in a moment."),
                "FITNESS_MEAL_ERROR",
            )
        # READ audit: item count + meal kcal only, never re-log the food text.
        meal_kcal = round(sum(it.kcal for it in items), 1)
        audit.log("–", "fitness", "meal_logged",
                  payload={"items": len(items), "kcal": meal_kcal,
                           "unknown": len(unknown)})

        de = "de" if lang == "pt" else "of"
        # Mark every weight we ASSUMED. the user asked for "faça na média" and got
        # an invented number back with no hint it was invented (s82); a
        # standard serving is a fine answer, but it has to look like one.
        eaten = ", ".join(
            f"{_g(it.grams)}g {de} {it.food}" + ("*" if it.assumed else "")
            for it in items)
        assumed_note = ""
        if any(it.assumed for it in items):
            assumed_note = t(
                " (* porção média — me diz o peso se quiser exato)",
                " (* typical portion — tell me the weight for an exact figure)")
        mp = round(sum(it.protein for it in items), 1)
        mc = round(sum(it.carb for it in items), 1)
        mf = round(sum(it.fat for it in items), 1)
        day = fl.today_totals(days_ago=days_ago)
        when_pt, when_en = {0: ("Hoje", "Today"), 1: ("Ontem", "Yesterday"),
                            2: ("Anteontem", "The day before")}[days_ago]

        tail = ""
        if unknown:
            names = ", ".join(unknown)
            tail = t(f" (não conheço {names} ainda)", f" (don't know {names} yet)")
        if unmeasured:
            names = ", ".join(unmeasured)
            tail += t(f" (não registrei {names} — me diz em gramas)",
                      f" (didn't log {names} — tell me in grams)")

        return (
            t(
                f"Registrei{' em ' + when_pt.lower() if days_ago else ''}: {eaten}{tail}{assumed_note}. "
                f"Essa refeição: {_g(meal_kcal)} kcal, "
                f"{_g(mp)}g proteína, {_g(mc)}g carbo, {_g(mf)}g gordura. "
                f"{when_pt} no total: {_g(day['kcal'])} kcal, {_g(day['protein'])}g proteína.",
                f"Logged{' for ' + when_en.lower() if days_ago else ''}: {eaten}{tail}{assumed_note}. "
                f"This meal: {_g(meal_kcal)} kcal, "
                f"{_g(mp)}g protein, {_g(mc)}g carb, {_g(mf)}g fat. "
                f"{when_en + ' so far' if not days_ago else when_en + ' in total'}: "
                f"{_g(day['kcal'])} kcal, {_g(day['protein'])}g protein."),
            "FITNESS_MEAL_LOGGED",
        )

    def _handle_fitness_macros_today(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"meus macros hoje" → today's running totals from the local store."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        day = get_log().today_totals()
        if not day["n"]:
            return (
                t(
                    "Você ainda não registrou nada hoje. Diz \"comi …\" pra eu somar.",
                    "You haven't logged anything today yet. Say \"I ate …\" and I'll total it."),
                "FITNESS_MACROS_EMPTY",
            )
        return (
            t(
                f"Hoje: {_g(day['kcal'])} kcal, {_g(day['protein'])}g proteína, "
                f"{_g(day['carb'])}g carbo, {_g(day['fat'])}g gordura "
                f"(em {day['n']} {'item' if day['n'] == 1 else 'itens'}).",
                f"Today: {_g(day['kcal'])} kcal, {_g(day['protein'])}g protein, "
                f"{_g(day['carb'])}g carb, {_g(day['fat'])}g fat "
                f"({day['n']} {'item' if day['n'] == 1 else 'items'})."),
            "FITNESS_MACROS_TODAY",
        )

    def _handle_fitness_list_today(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"o que comi hoje" → the day's meals, broken down (the answer to
        "where did that total come from")."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        meals = get_log().list_today()
        if not meals:
            return (
                t("Você ainda não registrou nada hoje.",
                  "You haven't logged anything today yet."),
                "FITNESS_LIST_EMPTY",
            )
        de = "de" if lang == "pt" else "of"
        lines = []
        tot_k = tot_p = 0.0
        for ml in meals:
            items = ", ".join(f"{_g(g)}g {de} {f}" for g, f in ml["items"])
            lines.append(f"{ml['time']} — {items}: {_g(ml['kcal'])} kcal, "
                         f"{_g(ml['protein'])}g {'proteína' if lang == 'pt' else 'protein'}")
            tot_k += ml["kcal"]
            tot_p += ml["protein"]
        header = t(f"Hoje ({len(meals)} refeições):", f"Today ({len(meals)} meals):")
        footer = t(f"Total: {_g(round(tot_k, 1))} kcal, {_g(round(tot_p, 1))}g proteína.",
                   f"Total: {_g(round(tot_k, 1))} kcal, {_g(round(tot_p, 1))}g protein.")
        return ("\n".join([header, *lines, footer]), "FITNESS_LIST_TODAY")

    def _handle_fitness_undo_meal(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"desfaz a última refeição" → remove the most recent meal."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        result = get_log().undo_last_meal_result()
        if result.error:
            audit.log("–", "fitness", "undo_last_meal", payload={"removed": 0, "error": True})
            return (
                t("Não consegui desfazer a refeição agora — deu erro no registro, nada foi apagado. Tenta de novo daqui a pouco.",
                  "I couldn't undo the meal just now — the log failed, nothing was deleted. Try again in a moment."),
                "FITNESS_UNDO_ERROR",
            )
        n = result.removed
        audit.log("–", "fitness", "undo_last_meal", payload={"removed": n})
        if not n:
            return (t("Não há refeição pra desfazer.", "No meal to undo."),
                    "FITNESS_UNDO_EMPTY")
        return (
            t(f"Desfeito: removi a última refeição ({n} {'item' if n == 1 else 'itens'}).",
              f"Undone: removed the last meal ({n} item{'' if n == 1 else 's'})."),
            "FITNESS_UNDO",
        )

    def _handle_fitness_clear_today(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"limpa as refeições de hoje" → wipe today's log (test cleanup)."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        result = get_log().clear_today_result()
        if result.error:
            audit.log("–", "fitness", "clear_today", payload={"removed": 0, "error": True})
            return (
                t("Não consegui apagar as refeições de hoje — deu erro no registro, nada foi apagado.",
                  "I couldn't clear today's meals — the log failed, nothing was deleted."),
                "FITNESS_CLEAR_ERROR",
            )
        n = result.removed
        audit.log("–", "fitness", "clear_today", payload={"removed": n})
        if not n:
            return (t("Não havia nada registrado hoje.", "Nothing was logged today."),
                    "FITNESS_CLEAR_EMPTY")
        return (
            t(f"Zerado: apaguei as refeições de hoje ({n} {'item' if n == 1 else 'itens'}).",
              f"Cleared: removed today's meals ({n} item{'' if n == 1 else 's'})."),
            "FITNESS_CLEAR",
        )

    def _handle_fitness_delete_guard(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """A meal/workout deletion no route carried out: nothing was deleted,
        and the reply says so — fixed, never the 8B's (s88, Codex lote 9).
        The turn's language: "don't clear…" opens with no trigger verb."""
        lang = "en" if getattr(self, "_reply_lang", "pt") == "en" else "pt"
        audit.log("–", "fitness", "delete_refused", payload={"chars": len(raw_text or "")})
        return (
            ("Não apaguei nada. Pra apagar, diga exatamente o quê: 'desfaz a "
             "última refeição', 'limpa as refeições de hoje', 'desfaz o último "
             "treino' ou 'limpa os treinos de hoje'.") if lang == "pt" else
            ("I didn't delete anything. To delete, say exactly what: 'undo "
             "last meal', 'clear today's meals', 'undo last workout' or "
             "'clear today's workouts'."),
            "FITNESS_DELETE_NONE",
        )

    # ---------- workouts (slice 2: runs + lifts + trends) ----------

    def _handle_fitness_log_run(self, dist, dur, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"registra corrida 5km em 25min" → log it, report the pace."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        try:
            d = float(str(dist).replace(",", "."))
            m = float(str(dur).replace(",", "."))
        except (TypeError, ValueError):
            return None
        from src.integrations.fitness import get_log
        result = get_log().log_run_result(d, m)
        if result == "invalid":
            return None
        if result == "error":
            audit.log("–", "fitness", "run_log_failed", payload={})
            return (
                t("Não consegui registrar essa corrida agora — nada foi gravado. Tenta de novo daqui a pouco.",
                  "I couldn't log that run just now — nothing was saved. Try again in a moment."),
                "FITNESS_RUN_ERROR",
            )
        audit.log("–", "fitness", "run_logged", payload={"km": d, "min": m})
        pace = _pace_str(m / d if d else 0)
        return (
            t(f"Registrei: corrida de {_g(d)}km em {_g(m)}min — pace {pace}/km.",
              f"Logged: {_g(d)}km run in {_g(m)}min — pace {pace}/km."),
            "FITNESS_RUN_LOGGED",
        )

    def _handle_fitness_log_lift(self, ex, sets, reps, weight,
                                 raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"supino 3x10 80kg" → log a strength set, report the volume."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        ex = (ex or "").strip()
        try:
            s = int(sets)
            r = int(reps)
            w = float(str(weight).replace(",", "."))
        except (TypeError, ValueError):
            return None
        if not ex:
            return None
        from src.integrations.fitness import get_log
        result = get_log().log_lift_result(ex, s, r, w)
        if result == "invalid":
            return None
        if result == "error":
            audit.log("–", "fitness", "lift_log_failed", payload={})
            return (
                t("Não consegui registrar esse treino agora — nada foi gravado. Tenta de novo daqui a pouco.",
                        "I couldn't log that workout just now — nothing was saved. Try again in a moment."),
                "FITNESS_LIFT_ERROR",
            )
        vol = round(s * r * w, 1)
        audit.log("–", "fitness", "lift_logged",
                  payload={"ex": ex[:40], "sets": s, "reps": r, "kg": w})
        return (
            t(f"Registrei: {ex} {s}x{r} {_g(w)}kg (volume {_g(vol)}kg).",
              f"Logged: {ex} {s}x{r} {_g(w)}kg (volume {_g(vol)}kg)."),
            "FITNESS_LIFT_LOGGED",
        )

    def _handle_fitness_lift_trend(self, ex, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"como tá meu supino?" → the exercise's progression. Falls through
        (None) when the captured word has no history, so an accidental match
        ("como tá meu dia") goes to the LLM instead of a wrong answer."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        hist = get_log().exercise_history((ex or "").strip())
        if not hist:
            return None
        last = hist[-1]
        name = (last["name"] or "").capitalize()
        if len(hist) == 1:
            return (
                t(f"{name}: 1 sessão registrada — {last['sets']}x{last['reps']} "
                  f"{_g(last['weight_kg'])}kg (volume {_g(last['volume'])}).",
                  f"{name}: 1 session logged — {last['sets']}x{last['reps']} "
                  f"{_g(last['weight_kg'])}kg (volume {_g(last['volume'])})."),
                "FITNESS_LIFT_TREND",
            )
        first = hist[0]
        dw = round(last["weight_kg"] - first["weight_kg"], 1)
        sign = f"+{_g(dw)}" if dw > 0 else _g(dw)
        verb = t("subiu", "up") if dw > 0 else (t("caiu", "down") if dw < 0 else t("estável", "flat"))
        return (
            t(f"{name}: {len(hist)} sessões. De {_g(first['weight_kg'])}kg ({first['date']}) "
              f"pra {_g(last['weight_kg'])}kg ({last['date']}) — {verb} {sign}kg. "
              f"Última: {last['sets']}x{last['reps']} {_g(last['weight_kg'])}kg.",
              f"{name}: {len(hist)} sessions. From {_g(first['weight_kg'])}kg ({first['date']}) "
              f"to {_g(last['weight_kg'])}kg ({last['date']}) — {verb} {sign}kg. "
              f"Last: {last['sets']}x{last['reps']} {_g(last['weight_kg'])}kg."),
            "FITNESS_LIFT_TREND",
        )

    def _handle_fitness_run_trend(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"meu pace" / "como tá minha corrida" → recent runs + pace direction."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        runs = get_log().run_history(5)  # newest first
        if not runs:
            return (
                t("Ainda não registrei nenhuma corrida. Diz \"corri 5km em 25min\".",
                  "No runs logged yet. Say \"I ran 5km in 25min\"."),
                "FITNESS_RUN_TREND_EMPTY",
            )
        parts = [f"{_g(r['distance_km'])}km {_pace_str(r['pace_min_km'])}/km ({r['date']})"
                 for r in runs[:3]]
        note = ""
        if len(runs) >= 2:
            newest, oldest = runs[0]["pace_min_km"], runs[-1]["pace_min_km"]
            if newest < oldest:
                note = t(" — pace melhorando", " — pace improving")
            elif newest > oldest:
                note = t(" — pace subindo", " — pace slipping")
        return (
            t(f"Últimas corridas: {', '.join(parts)}{note}.",
              f"Recent runs: {', '.join(parts)}{note}."),
            "FITNESS_RUN_TREND",
        )

    def _handle_fitness_undo_workout(self, raw_text: str = "",
                                     kind: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """"desfaz o último treino" → remove the most recent workout.
        "corrida"/"run" removes the most recent RUN; "treino"/"exercício" any."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        only = "run" if (kind or "").lower() in ("corrida", "run") else None
        result = get_log().undo_last_workout_result(kind=only)
        if result.error:
            audit.log("–", "fitness", "undo_last_workout",
                      payload={"removed": 0, "error": True, "kind": only})
            return (
                t("Não consegui desfazer o treino agora — deu erro no registro, nada foi apagado. Tenta de novo daqui a pouco.",
                  "I couldn't undo the workout just now — the log failed, nothing was deleted. Try again in a moment."),
                "FITNESS_WORKOUT_UNDO_ERROR",
            )
        w = result.workout
        audit.log("–", "fitness", "undo_last_workout",
                  payload={"removed": 1 if w else 0, "kind": only})
        if not w:
            if only == "run":
                return (t("Não há corrida pra desfazer.", "No run to undo."),
                        "FITNESS_WORKOUT_UNDO_EMPTY")
            return (t("Não há treino pra desfazer.", "No workout to undo."),
                    "FITNESS_WORKOUT_UNDO_EMPTY")
        if w["kind"] == "run":
            desc = t(f"corrida de {_g(w['distance_km'])}km", f"{_g(w['distance_km'])}km run")
        else:
            desc = f"{w['name']} {w['sets']}x{w['reps']} {_g(w['weight_kg'])}kg"
        return (
            t(f"Desfeito: removi o último treino ({desc}).",
              f"Undone: removed the last workout ({desc})."),
            "FITNESS_WORKOUT_UNDO",
        )

    def _handle_fitness_clear_workouts(self, raw_text: str = "") -> Optional[Tuple[str, str]]:
        """"limpa os treinos de hoje" → wipe today's workouts (test cleanup)."""
        lang = _lang_from_trigger(raw_text)

        def t(pt: str, en: str) -> str:
            return pt if lang == "pt" else en

        from src.integrations.fitness import get_log
        result = get_log().clear_workouts_today_result()
        if result.error:
            audit.log("–", "fitness", "clear_workouts_today", payload={"removed": 0, "error": True})
            return (
                t("Não consegui apagar os treinos de hoje — deu erro no registro, nada foi apagado.",
                  "I couldn't clear today's workouts — the log failed, nothing was deleted."),
                "FITNESS_WORKOUT_CLEAR_ERROR",
            )
        n = result.removed
        audit.log("–", "fitness", "clear_workouts_today", payload={"removed": n})
        if not n:
            return (t("Não havia treino registrado hoje.", "No workouts logged today."),
                    "FITNESS_WORKOUT_CLEAR_EMPTY")
        return (
            t(f"Zerado: apaguei os treinos de hoje ({n}).",
              f"Cleared: removed today's workouts ({n})."),
            "FITNESS_WORKOUT_CLEAR",
        )
