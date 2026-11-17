"""
Higher-level feature handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. Groups the self-referential / meta + Fase-2 dormant
feature handlers the dispatch registry routes to:

  - undo (Onda 3 — reverse the last ledgered action)
  - whats_changed (Onda 4 — deterministic delta digest)
  - system_status (Bug B-1 — speakable health summary)
  - dream_review / deep_research (Fase 2 — DORMANT, gated; return None when
    their flag is off so the caller falls through)

All bodies use local imports, so no orchestrator-module state moves with them.
Mixed in so `self._handle_*(...)` resolves through the MRO; undo reaches the
`_UNDO_ACTION_LABELS` class attr and `_run_reverse` (StagingMixin) via the MRO.
"""
from __future__ import annotations

from typing import Optional, Tuple


class FeaturesMixin:
    """Meta/observability (undo, digest, status) + Fase-2 dormant handlers."""

    @staticmethod
    def _undo_now():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def _handle_undo(self) -> Tuple[str, str]:
        """Reverse the last reversible action recorded in the Action Ledger.

        'isso' = the most recent thing I did. If that last action is
        irreversible (a sent email), say so honestly instead of silently
        undoing something older. The undo is itself recorded so a second
        'desfaz' can't double-undo it."""
        from datetime import datetime, timezone
        from src.core.action_ledger import last_action, ledger
        from src.core import effect_store
        from src.core.effect_runner import run_effect

        _UNDO_WINDOW_S = 30 * 60
        now = self._undo_now()
        entries = ledger.load_entries()
        target = last_action(entries)
        if target is None:
            return (self._t("Não há nada recente pra desfazer.",
                            "There's nothing recent to undo."), "UNDO_NONE")

        # An undated action cannot safely be treated as recent.
        try:
            ts = datetime.fromisoformat(target.get("ts"))
        except (TypeError, ValueError):
            return (self._t("Não há nada recente pra desfazer.",
                            "There's nothing recent to undo."), "UNDO_NONE")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (now - ts).total_seconds()

        action = target.get("action", "?")
        label = self._UNDO_ACTION_LABELS.get(action, action)
        label_en = self._UNDO_ACTION_LABELS_EN.get(action, action)
        # Check age before reversibility or effects; clock skew stays recent.
        if age_s > _UNDO_WINDOW_S:
            if age_s < 120 * 60:
                age = age_en = f"{int(age_s // 60)} min"
            elif age_s < 48 * 60 * 60:
                h = int(age_s // 3600)
                age = f"{h} hora" + ("s" if h != 1 else "")
                age_en = f"{h} hour" + ("s" if h != 1 else "")
            else:
                d = int(age_s // 86400)
                age = f"{d} dia" + ("s" if d != 1 else "")
                age_en = f"{d} day" + ("s" if d != 1 else "")
            return (
                self._t(
                    f"Não há nada recente pra desfazer — a última ação ({label}) foi há {age}. "
                    "O desfaz vale até 30 minutos depois.",
                    f"There's nothing recent to undo — the last action ({label_en}) was {age_en} ago. "
                    "Undo works up to 30 minutes after."),
                "UNDO_TOO_OLD",
            )
        if not target.get("reversible") or not target.get("reverse"):
            return (
                self._t(
                    f"A última coisa que fiz foi {label}, e isso não dá pra desfazer.",
                    f"The last thing I did was {label_en}, and that can't be undone."),
                "UNDO_IRREVERSIBLE",
            )

        # The target's identity survives a lost ledger write or a concurrent undo.
        base = f"undo:{target.get('ts')}:{action}"

        def next_attempt():
            states = set()
            attempt, effect_id = 1, base
            while (effect := effect_store.store.get(effect_id)) is not None:
                states.add(effect.state)
                attempt += 1
                effect_id = f"{base}:{attempt}"
            if "feito" in states:
                reply = self._t("Isso já foi desfeito.", "That was already undone.")
            elif states & {"preparando", "esperando", "executando"}:
                reply = self._t("Isso está sendo desfeito agora.",
                                "That is being undone right now.")
            elif "desconhecido" in states:
                reply = self._t(
                    "Não sei se desfiz isso — confere antes de pedir de novo; não vou repetir sozinho.",
                    "I don't know whether I undid that — please check before asking again; I won't retry on my own.")
            else:
                return effect_id, None
            return effect_id, (reply, "UNDO_NONE")

        effect_id, duplicate = next_attempt()
        if duplicate is not None:
            return duplicate
        called = False

        def reverse():
            nonlocal called
            called = True
            reply, mode = self._run_reverse(target.get("reverse"))
            if mode == "AGENT_ACTION_DONE":
                ledger.record(
                    "undo", args={"target": action}, status="done", source="undo",
                    extra={"undone_ts": target.get("ts")},
                )
            return reply, mode

        reply, mode = run_effect(
            "undo", target.get("reverse"), reverse, effect_id=effect_id,
            origin=getattr(self, "_effect_origin", getattr(self, "_turn_origin", lambda: None))(),
            cancelled=lambda: getattr(self, "_turn_cancelled", lambda: False)(),
            lang=self._reply_lang)
        # Another caller can claim after the precheck; run_effect is the atomic guard.
        if not called and mode == "AGENT_ACTION_ERROR":
            _, duplicate = next_attempt()
            if duplicate is not None:
                return duplicate
        if mode == "AGENT_ACTION_DONE":
            return (self._t(f"Pronto, desfiz: {reply}", f"Done, undid: {reply}"), "UNDO_DONE")
        return (reply, mode)

    def _handle_whats_changed(self) -> Tuple[str, str]:
        """Return the deterministic delta digest and advance the marker.
        Thin wrapper — all the pull/assembly logic lives in src.core.digest,
        which is pure-testable. Fail-soft: a broken source degrades to an
        empty section, never an error reply."""
        from src.core import digest
        return (digest.whats_changed(lang=self._reply_lang), "WHATS_CHANGED")

    def _handle_system_status(self) -> Tuple[str, str]:
        """Return a deterministic, speakable system-health summary (Bug B-1).
        Thin wrapper — all the pull/assembly lives in src.core.system_status,
        which is pure-testable and fail-soft. This keeps "status do sistema?"
        off the LLM+retrieval path, where a weakly-relevant past turn was being
        woven into a fabricated status (ideas_backlog Bug B)."""
        from src.core import system_status
        return (system_status.summary(lang=self._reply_lang), "SYSTEM_STATUS")

    def _handle_dream_review(self, action: str) -> Optional[Tuple[str, str]]:
        """Review the latest pending Dream (Fase 2). DORMANT: returns None when
        CORTEX_DREAM_LOOP is off so the caller falls through. Reads the
        DreamStore (deterministic — no model) for show/keep/drop. Fail-soft."""
        from src.core.dream_loop import DreamStore, _dream_enabled
        if not _dream_enabled():
            return None
        try:
            store = DreamStore()
            latest = store.latest_pending()
            if latest is None:
                return (self._t("Não há nenhum sonho pendente pra revisar.",
                                "There's no pending dream to review."), "DREAM_NONE")
            if action == "keep":
                store.approve(latest.id)
                return (self._t("Guardei esse sonho. 🌙", "Kept that dream. 🌙"), "DREAM_KEPT")
            if action == "drop":
                store.reject(latest.id)
                return (self._t("Descartei esse sonho.", "Discarded that dream."), "DREAM_DROPPED")
            return (self._t(f"Último sonho:\n{latest.text}",
                            f"Latest dream:\n{latest.text}"), "DREAM_SHOW")
        except Exception as e:  # noqa: BLE001 — never crash the turn
            return (self._t(f"[SYSTEM ERROR] Falha ao acessar os sonhos: {e}",
                            f"[SYSTEM ERROR] Failed to access dreams: {e}"), "DREAM_ERROR")

    def _handle_deep_research(self, topic: str) -> Optional[Tuple[str, str]]:
        """Queue a deep-research topic (Fase 2). DORMANT: returns None when
        CORTEX_DEEP_STUDIER is off so the caller falls through to normal
        search — the multi-step synthesis only earns its cost with a strong
        model (Mac Studio).

        When enabled, ENQUEUES and returns immediately (M8-A2): the deep
        loop is 15 searches + several LLM syntheses — minutes of work that
        used to run synchronously INSIDE the request thread, blocking every
        other route and hanging the voice client without feedback. The
        NightlyStudier consumes the queue inside its governed SystemLease
        window (and research() itself acquires a lease — M8-A3 — so no
        caller can fan out ungoverned searches). Fail-soft."""
        from src.core.deep_studier import _deep_enabled, deep_queue
        if not _deep_enabled():
            return None  # dormant → normal search/LLM handles "pesquise …"
        topic = (topic or "").strip()
        if not topic:
            return None
        try:
            deep_queue().add(topic)
        except ValueError as e:
            # StudyQueue validation: transient topic or near-duplicate.
            return (self._t(f"Não enfileirei a pesquisa profunda: {e}",
                            f"Didn't queue the deep research: {e}"),
                    "DEEP_RESEARCH_REJECTED")
        except Exception as e:  # noqa: BLE001 — never crash the turn
            return (self._t(f"[SYSTEM ERROR] Pesquisa profunda falhou: {e}",
                            f"[SYSTEM ERROR] Deep research failed: {e}"), "DEEP_RESEARCH_ERROR")
        return (
            self._t(f"Pesquisa profunda sobre '{topic}' na fila — rodo de "
                    "madrugada junto com o estudo noturno e o resultado "
                    "aparece nos estudos pendentes.",
                    f"Deep research on '{topic}' queued — it runs overnight "
                    "with the nightly study and lands in pending studies."),
            "DEEP_RESEARCH_QUEUED",
        )
