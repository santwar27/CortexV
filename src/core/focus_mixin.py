"""Focus-session routes for `CortexOrchestrator` (s86 bloco 2a).

The words and the record live in src/core/focus.py; this mixin answers them,
always with a fixed text — no 8B anywhere in the cycle. Measured before
(real 8B, both entry points): 1 useful reply in 30. "foco: candidatura…"
became a job search, the first step "abrir o formulário" tried to open an
app called "formulário", and "onde parei?" came back as "Curitiba, Minas
Gerais."

The cycle's own questions ("qual o primeiro passo?", "fecho X e começo Y?",
"mais 25 min?", "qual o próximo passo?") ride on the one-turn `_offer` (s85).
Both entry points reach `_handle_focus`: _process_core calls it before the
dispatch registry (so "abrir o formulário" is a first step, not an app), and
process_stream's reflex and pure-chat shortcuts step aside through
`_is_focus_turn` — one predicate for both, so they can't disagree.

The audit gets counts and durations only — never the text of a task: the
week of use is measured from it, and the tasks are Pedro's day.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Tuple

from src.core import focus
from src.core.audit import audit
from src.core.clock import local_now
from src.core.redaction import redact_text
from src.core.reflex import reflex_layer

log = logging.getLogger(__name__)

# Every reply in the cycle, as (pt, en). The fields in braces are filled in
# by the route; a text may drop a field, but the test in test_focus_routes
# names the ones each text must keep.
# The PT side is the one the user reads: short, the next action first, an
# invitation rather than a demand — these land at the moment of starting and
# of coming back, the two moments that are hard.
FOCUS_TEXTS = {
    # "foco: X" — the one question before starting.
    "ask_step": (
        "Bora! Qual o primeiro passo de 5 minutos em {task}?",
        "What's a 5-minute first step for {task}?"),
    # The session is open.
    "started": (
        "Valendo até {until}. Começa por: {first_step}. "
        "Se parar no meio, me diz: parei: <próximo passo>.",
        "Focus until {until} on {task}. First step: {first_step}. "
        "When you stop, write: parei: <next step>."),
    # "não sei" to the first-step question.
    "default_step": (
        "abrir o que você precisa para começar",
        "open what you need to start"),
    # "não" to the first-step question.
    "cancelled": (
        "Tranquilo, fica pra depois. Quando quiser: foco: <tarefa>.",
        "All right, no focus for now. Whenever you like: foco: <task>."),
    # A new "foco:" while another is still inside its time (the user's rule).
    "ask_switch": (
        "Você ainda está em {old_task} até {until}. Troco para {task}? (sim)",
        "You're still on {old_task} until {until}. Close it and start {task}? (yes)"),
    "kept": (
        "Beleza, segue em {old_task} até {until}.",
        "OK, carrying on with {old_task} until {until}."),
    # "parei: Y".
    "closed": (
        "Anotado. Na volta, começa por: {next_step}. É só perguntar: onde parei?",
        "Saved. Next step: {next_step}. When you're back, ask: onde parei?"),
    # A bare "parei".
    "ask_next": (
        "Pausei {task}. Qual vai ser o próximo passo quando você voltar?",
        "Stopped the focus on {task}. What's the next step, for when you're back?"),
    "next_saved": (
        "Anotado: {next_step}.",
        "Saved: {next_step}."),
    "next_skipped": (
        "Tranquilo. Na volta, é só perguntar: onde parei?",
        "All right. When you're back, ask: onde parei?"),
    "nothing_open": (
        "Não tem foco rolando agora. Pra começar: foco: <tarefa>.",
        "There's no focus running right now. To start one: foco: <task>."),
    # "onde parei?"
    "where_running": (
        "Você está em {task} até {until}. O passo era: {first_step}.",
        "You're on {task} until {until}. First step: {first_step}."),
    "where_time_up": (
        "Você estava em {task}; o tempo acabou {when}. Bora mais {minutes} min? (sim)",
        "You were on {task}; the time ran out {when}. {minutes} more minutes? (yes)"),
    "where_next": (
        "Você estava em {task} ({when}). Próximo passo: {next_step}. "
        "Bora mais {minutes} min? (sim)",
        "You were on {task} ({when}). Next step: {next_step}. "
        "{minutes} more minutes? (yes)"),
    "where_no_next": (
        "Você estava em {task} ({when}). Bora mais {minutes} min? (sim)",
        "You were on {task} ({when}). {minutes} more minutes? (yes)"),
    "where_none": (
        "Ainda não tem nenhum foco guardado. Pra começar: foco: <tarefa>.",
        "No focus saved yet. To start one: foco: <task>."),
    "error": (
        "Não consegui guardar o foco agora — deu erro no registro. Tenta de novo?",
        "I couldn't save the focus just now — the record failed."),
    # The Reminders item for the end of the session (iCloud → the iPhone).
    # The user, 16/10: the task goes in the title; the next step never does.
    "reminder_title": (
        "Fim do foco: {task}",
        "End of focus: {task}"),
    "reminder_body": (
        "No Cortex: parei: <próximo passo>",
        "In Cortex: parei: <next step>"),
    # The nudge: a second reminder 10 min past the end (the user, 16/10).
    "nudge_title": (
        "Ainda em {task}?",
        "Still on {task}?"),
    "nudge_body": (
        "Se parou, no Cortex: parei: <próximo passo>",
        "If you stopped, in Cortex: parei: <next step>"),
    # Added to "onde parei?" when Reminders.app failed to create it.
    "reminder_failed": (
        "(O lembrete no celular não foi criado — o Lembretes não respondeu.)",
        "(The phone reminder wasn't created — Reminders didn't respond.)"),
    "reminder_unknown": (
        "(Não sei se o lembrete no celular foi criado — o Lembretes não confirmou.)",
        "(I don't know whether the phone reminder was created — Reminders didn't confirm.)"),
}

_PT_DAYS = ("na segunda", "na terça", "na quarta", "na quinta", "na sexta",
            "no sábado", "no domingo")
_EN_DAYS = ("on Monday", "on Tuesday", "on Wednesday", "on Thursday",
            "on Friday", "on Saturday", "on Sunday")


# ----------------------------------------------------------------------
# the reminder, off the reply path
# ----------------------------------------------------------------------
# Reminders.app over AppleScript: create ≈ 2 s, complete ≈ 1 s, and it has
# timed out at 30 s (s86 bloco 0) — the reply never waits for it. ONE worker,
# so a process's calls run in order. Disk claims and completion intent also
# cover stops and duplicate jobs from another process.
_REMINDERS_WORKER = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix="focus-reminders")


def _in_background(job, *args) -> None:
    _REMINDERS_WORKER.submit(job, *args)


def _create_reminders_job(session_id: int, end: tuple, nudge: tuple) -> None:
    """The end-of-focus reminder, then the nudge 10 min later — each a
    (title, body, due_iso)."""
    from src.integrations import reminders_client as rc
    for which, args, step in (("end", end, "create"), ("nudge", nudge, "create_nudge")):
        claimed = focus.store.claim_reminder(session_id, which)
        if claimed is None:
            _reminder_not_saved("claim_" + which)
            return
        if not claimed:
            if which == "end":
                session = focus.store.get(session_id)
                if (session is not None and session.reminder_state == "done"
                        and session.reminder
                        and session.reminder not in (focus.REMINDER_FAILED, focus.REMINDER_UNKNOWN)):
                    continue  # a saved end allows the still-pending nudge claim
            return
        ref, outcome = focus.REMINDER_FAILED, "failed"
        ready = True
        if which == "end":
            try:
                rc.ensure_list(focus.REMINDERS_LIST)
            except Exception as e:  # noqa: BLE001 — a background job never raises
                _reminder_failed(step, e)
                ready = False  # list setup never attempted the reminder create
        if ready:
            try:
                ref = _make_reminder(rc, *args)
                outcome = "done"
            except Exception as e:  # noqa: BLE001
                _reminder_failed(step, e)
                if getattr(e, "uncertain", False):
                    ref, outcome = focus.REMINDER_UNKNOWN, "unknown"
        if not focus.store.finish_reminder(session_id, which, outcome, ref):
            _reminder_not_saved(step)
            if which == "end" and not focus.store.skip_nudge(session_id):
                _reminder_not_saved("skip_nudge")
            return
        # stop() may have committed in another process during the AppleScript.
        # Its worker left running creates alone; the creator now owns cleanup.
        session = focus.store.get(session_id)
        if session is not None and session.completion_wanted:
            _complete_reminders_job(session_id)
        if outcome != "done" and which == "end":
            return
        if which == "nudge":
            audit.log("–", "focus", "reminder_created", payload={"nudge": outcome == "done"})


def _reminder_not_saved(step: str) -> None:
    audit.log("–", "focus", "reminder_not_saved", payload={"step": step})


def _complete_reminders_job(session_id: int) -> None:
    """Complete known ids, including legacy/planted ids. A running create
    belongs to its creator; recorded successful completions are never repeated."""
    from src.integrations import reminders_client as rc
    session = focus.store.get(session_id)
    if session is None:
        return
    done = 0
    for which, step, ref, state, completion in (
        ("end", "complete", session.reminder, session.reminder_state, session.reminder_completion),
        ("nudge", "complete_nudge", session.nudge_reminder, session.nudge_state, session.nudge_completion),
    ):
        if (not ref or ref in (focus.REMINDER_FAILED, focus.REMINDER_UNKNOWN)
                or state == "running" or completion == "done"):
            continue
        try:
            found = rc.complete_reminder(ref, focus.REMINDERS_LIST)
        except Exception as e:  # noqa: BLE001 — a background job never raises
            _reminder_failed(step, e)
            result = "failed"
        else:
            result = "done" if found else "not_found"
            if not found:       # moved or deleted on the iPhone: still rings
                _reminder_failed(step, LookupError("not in its list"))
            else:
                done += 1
        if not focus.store.record_completion(session_id, which, ref, result):
            _reminder_not_saved(step)
    if done:
        audit.log("–", "focus", "reminder_completed", payload={"count": done})


def _make_reminder(rc, title: str, body: str, due_iso: str) -> str:
    # Redacted at the border, like every sink: the title carries the task,
    # it syncs to iCloud and shows on the locked iPhone, and nobody confirms
    # it — "foco: …; senha do banco é X" went out whole (s88, Codex lote 4).
    made = rc.create_reminder(redact_text(title), due_iso, body=redact_text(body),
                              list_name=focus.REMINDERS_LIST)
    ref = (made or {}).get("id")
    if not ref:
        error = rc.RemindersUnavailable("no reminder id in the reply")
        error.uncertain = True
        raise error
    return ref


def _reminder_failed(step: str, error: Exception) -> None:
    log.warning("focus: reminder %s failed: %s", step, type(error).__name__)
    # The error's type only: its message can carry the script, and so the
    # task's name.
    payload = {"step": step, "error": type(error).__name__}
    if getattr(error, "uncertain", False):
        payload["uncertain"] = True
    audit.log("–", "focus", "reminder_failed", payload=payload)


class _Fields(dict):
    """A field a text names but the route doesn't fill stays visible as
    "{name}" instead of crashing the reply."""

    def __missing__(self, key):
        return "{" + key + "}"


class FocusMixin:
    """Focus-session routes. Mixed into the orchestrator; uses the host's
    `_offer`, `_reply_lang` (via `_t`), `_APPROVAL_RE` and `_DENIAL_RE`."""

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------
    def _is_focus_turn(self, own: str) -> bool:
        """True when `_handle_focus` may answer this turn — the veto the
        stream's shortcuts check, so "abrir o formulário" (a first step)
        never reaches the reflex, the streamed chat or the app opener."""
        if self._is_focus_command(own):
            return True
        kind = (getattr(self, "_offer", None) or {}).get("kind", "")
        return kind.startswith("focus_") and self._focus_may_answer(own)

    @staticmethod
    def _is_focus_command(own: str) -> bool:
        return bool(focus.parse_start(own) or focus.parse_stop(own) is not None
                    or focus.asks_where_was_i(own))

    def _handle_focus(self, own: str, offer: Optional[dict]) -> Optional[Tuple[str, str]]:
        """A focus command, or the answer to the cycle's last question.
        None leaves the turn to the normal routes."""
        now = local_now()
        if self._is_focus_command(own):
            self._reply_lang = focus.command_lang(own)
        started = focus.parse_start(own)
        if started:
            return self._focus_start(*started, now=now)
        if focus.asks_where_was_i(own):
            return self._focus_where(now)
        next_step = focus.parse_stop(own)
        if next_step is not None:
            return self._focus_stop(next_step, now)

        kind = (offer or {}).get("kind", "")
        if not kind.startswith("focus_") or not self._focus_may_answer(own):
            return None
        # An answer is in the QUESTION's language: "ok" or "beleza" read as
        # English to the detector, and the PT cycle answered in English.
        self._reply_lang = offer.get("lang") or getattr(self, "_reply_lang", "pt")
        if kind == "focus_step":
            return self._focus_step_answer(own, offer, now)
        if kind == "focus_switch":
            return self._focus_switch_answer(own, offer, now)
        if kind == "focus_more":
            return self._focus_more_answer(own, offer, now)
        if kind == "focus_next":
            return self._focus_next_answer(own, offer)
        return None

    # ------------------------------------------------------------------
    # starting
    # ------------------------------------------------------------------
    def _focus_start(self, task: str, minutes: int, now: datetime):
        current = focus.store.current()
        if (current is not None
                and focus.on_start_while_open(current, now) == focus.ASK_FIRST):
            self._focus_ask({"kind": "focus_switch", "task": task, "minutes": minutes})
            audit.log("–", "focus", "ask_switch", payload={})
            return (self._focus_text("ask_switch", task=task, old_task=current.task,
                                     until=self._focus_hhmm(current.ends_at, now)),
                    "FOCUS_ASK_SWITCH")
        return self._focus_ask_step(task, minutes)

    def _focus_ask_step(self, task: str, minutes: int):
        self._focus_ask({"kind": "focus_step", "task": task, "minutes": minutes})
        audit.log("–", "focus", "ask_step", payload={"minutes": minutes})
        return (self._focus_text("ask_step", task=task), "FOCUS_ASK_STEP")

    def _focus_open(self, task: str, first_step: str, minutes: int,
                    now: datetime, resumed: bool = False):
        # The sessions this start closed — whoever opened them, this process
        # or another one a moment ago — have their alarms completed.
        session, closed = focus.store.start_closing(task, first_step, minutes, now)
        if session is None:
            return (self._focus_text("error"), "FOCUS_ERROR")
        for closed_id in closed:
            _in_background(_complete_reminders_job, closed_id)
        _in_background(
            _create_reminders_job, session.id,
            (self._focus_text("reminder_title", task=session.task),
             self._focus_text("reminder_body"),
             session.ends_at.isoformat()),
            (self._focus_text("nudge_title", task=session.task),
             self._focus_text("nudge_body"),
             focus.later(session.ends_at, focus.NUDGE_AFTER).isoformat()))
        audit.log("–", "focus", "resume" if resumed else "start",
                  payload={"minutes": minutes})
        return (self._focus_text("started", task=session.task,
                                 first_step=session.first_step,
                                 until=self._focus_hhmm(session.ends_at, now)),
                "FOCUS_STARTED")

    def _focus_step_answer(self, own: str, offer: dict, now: datetime):
        if self._DENIAL_RE.match(own.strip()):
            audit.log("–", "focus", "cancelled", payload={})
            return (self._focus_text("cancelled"), "FOCUS_CANCELLED")
        # "não sei" — and a bare "ok"/"beleza", which says "go" rather than
        # naming a step — start with the default step instead of a dead end.
        vague = focus.doesnt_know(own) or self._APPROVAL_RE.match(own.strip())
        step = self._focus_text("default_step") if vague else own.strip()
        if not step:
            return None
        return self._focus_open(offer["task"], step, offer["minutes"], now)

    def _focus_switch_answer(self, own: str, offer: dict, now: datetime):
        if self._APPROVAL_RE.match(own.strip()):
            return self._focus_ask_step(offer["task"], offer["minutes"])
        if self._DENIAL_RE.match(own.strip()):
            current = focus.store.current()
            if current is None:
                return (self._focus_text("cancelled"), "FOCUS_CANCELLED")
            audit.log("–", "focus", "switch_kept", payload={})
            return (self._focus_text("kept", old_task=current.task,
                                     until=self._focus_hhmm(current.ends_at, now)),
                    "FOCUS_KEPT")
        return None

    def _focus_more_answer(self, own: str, offer: dict, now: datetime):
        if self._APPROVAL_RE.match(own.strip()):
            if offer.get("first_step"):
                return self._focus_open(offer["task"], offer["first_step"],
                                        focus.DEFAULT_MINUTES, now, resumed=True)
            return self._focus_ask_step(offer["task"], focus.DEFAULT_MINUTES)
        if self._DENIAL_RE.match(own.strip()):
            return (self._focus_text("cancelled"), "FOCUS_CANCELLED")
        return None

    # ------------------------------------------------------------------
    # stopping
    # ------------------------------------------------------------------
    def _focus_stop(self, next_step: str, now: datetime):
        session = focus.store.stop(next_step, now)
        if session is None:
            return (self._focus_text("nothing_open"), "FOCUS_NONE")
        _in_background(_complete_reminders_job, session.id)
        audit.log("–", "focus", "close",
                  payload={"next_step": bool(session.next_step),
                           "early": not session.time_is_up(now),
                           "minutes": session.ran_minutes(now)})
        if session.next_step:
            return (self._focus_text("closed", next_step=session.next_step)
                    + self._focus_note_project(session.next_step),
                    "FOCUS_CLOSED")
        self._focus_ask({"kind": "focus_next", "session_id": session.id})
        return (self._focus_text("ask_next", task=session.task), "FOCUS_ASK_NEXT")

    def _focus_next_answer(self, own: str, offer: dict):
        if self._DENIAL_RE.match(own.strip()) or focus.doesnt_know(own):
            return (self._focus_text("next_skipped"), "FOCUS_NEXT_SAVED")
        step = own.strip()
        if not step or not focus.store.set_next_step(offer["session_id"], step):
            return None
        audit.log("–", "focus", "next_saved", payload={})
        return (self._focus_text("next_saved", next_step=focus.clip_text(step))
                + self._focus_note_project(focus.clip_text(step)),
                "FOCUS_NEXT_SAVED")

    def _focus_note_project(self, step: str) -> str:
        try:
            project, _ = self._note_project_progress("proximo", step)
            if project:
                return self._t(f' Anotei também em “{project.name}”.',
                               f' Also noted in “{project.name}”.')
        except Exception:
            audit.log("–", "project", "error")
        return ""

    # ------------------------------------------------------------------
    # coming back
    # ------------------------------------------------------------------
    def _focus_where(self, now: datetime):
        last = focus.store.last()
        if last is None:
            audit.log("–", "focus", "where", payload={"state": "none"})
            return (self._focus_text("where_none"), "FOCUS_WHERE")
        minutes = focus.DEFAULT_MINUTES
        if last.is_open and not last.time_is_up(now):
            state, text = "running", self._focus_text(
                "where_running", task=last.task, first_step=last.first_step,
                until=self._focus_hhmm(last.ends_at, now))
        elif last.is_open:
            state, text = "time_up", self._focus_text(
                "where_time_up", task=last.task, minutes=minutes,
                when=self._focus_when(last.ends_at, now))
            self._focus_ask({"kind": "focus_more", "task": last.task, "first_step": None})
        elif last.next_step:
            state, text = "next_step", self._focus_text(
                "where_next", task=last.task, next_step=last.next_step,
                minutes=minutes, when=self._focus_when(last.closed_at, now))
            self._focus_ask({"kind": "focus_more", "task": last.task,
                           "first_step": last.next_step})
        else:
            state, text = "no_next_step", self._focus_text(
                "where_no_next", task=last.task, minutes=minutes,
                when=self._focus_when(last.closed_at, now))
            self._focus_ask({"kind": "focus_more", "task": last.task, "first_step": None})
        # Running OR time up: after the 25 min is exactly when the phone
        # should have rung and didn't (s88, Codex lote 4).
        if last.is_open and last.reminder == focus.REMINDER_FAILED:
            text += " " + self._focus_text("reminder_failed")
        elif last.is_open and last.reminder == focus.REMINDER_UNKNOWN:
            text += " " + self._focus_text("reminder_unknown")
        audit.log("–", "focus", "where", payload={"state": state})
        return (text, "FOCUS_WHERE")

    # ------------------------------------------------------------------
    # text helpers
    # ------------------------------------------------------------------
    def _focus_ask(self, offer: dict) -> None:
        """Leave the cycle's question for the next turn (one turn only), with
        the language it was asked in."""
        self._offer = {**offer, "lang": getattr(self, "_reply_lang", "pt")}

    @staticmethod
    def _focus_may_answer(own: str) -> bool:
        """Whether this message can be the answer to the cycle's question.
        A question can't ("que horas são?" after "qual o primeiro passo?"),
        and neither can what the reflex already answers — "que horas são"
        typed without the "?", "oi", "obrigado" are never a first step. Both
        go to the normal routes, and the question lapses (one turn)."""
        text = (own or "").strip()
        return bool(text) and not text.endswith("?") and not reflex_layer.is_hit(text)

    def _focus_text(self, key: str, **fields) -> str:
        return self._t(*FOCUS_TEXTS[key]).format_map(_Fields(fields))

    @staticmethod
    def _focus_hhmm(dt: datetime, now: datetime) -> str:
        return dt.astimezone(now.tzinfo).strftime("%H:%M")

    def _focus_when(self, dt: datetime, now: datetime) -> str:
        """ "hoje às 15:40", "ontem às 15:40", "na terça às 15:40", "em 03/10"."""
        local = dt.astimezone(now.tzinfo)
        days = (now.date() - local.date()).days
        hhmm = local.strftime("%H:%M")
        if days <= 0:
            return self._t(f"hoje às {hhmm}", f"today at {hhmm}")
        if days == 1:
            return self._t(f"ontem às {hhmm}", f"yesterday at {hhmm}")
        if days < 7:
            wd = local.weekday()
            return self._t(f"{_PT_DAYS[wd]} às {hhmm}", f"{_EN_DAYS[wd]} at {hhmm}")
        return self._t(f"em {local:%d/%m}", f"on {local:%d/%m}")
