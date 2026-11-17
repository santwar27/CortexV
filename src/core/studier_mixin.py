"""
Nightly Studier skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. The dispatch registry routes "estude X" / "revisar" /
"ver|aprovar|rejeitar N" here. Mixin so `self._handle_*study*(...)` still
resolves through the MRO. These handlers reach into orchestrator state the host
provides — `self.study_queue`, `self.studies_review` — and the `_ALL_REF_RE`
class attr (resolved via the MRO). `_resolve_default_ref` is the shared helper
that turns an empty ref into "1" (single pending) or an ambiguity message.
"""
from __future__ import annotations

from typing import Optional, Tuple

from src.core.audit import audit
from src.core.study_queue import StudyTopicRejected


_STUDY_STATUS_LABELS = {
    "pending": ("na fila", "in the queue"),
    "studying": ("em estudo", "being studied"),
    "saving": ("sendo salvo", "being saved"),
    "consumed": ("já estudado", "already studied"),
    "done": ("já estudado", "already studied"),
    "failed": ("falhou", "failed"),
}


class StudierMixin:
    """Nightly Studier queue + review handlers. Mixed into the orchestrator."""

    def _handle_add_study(self, topic: str) -> Tuple[str, str]:
        """Queue a topic for the Nightly Studier to process overnight."""
        if not topic:
            return (self._t("Não entendi o tópico de estudo. Repita por favor.",
                            "I didn't catch the study topic. Please repeat."), "INPUT_ERROR")
        try:
            entry = self.study_queue.add(topic)
        except StudyTopicRejected as e:
            if e.reason == "TRANSIENT":
                return (self._t(
                    f'Não coloquei na lista: "{e.topic}" muda com o tempo (tem "{e.word}"). '
                    'Pergunte direto que eu busco na internet. Para colocar mesmo assim, termine com !!',
                    f'I didn\'t add it: "{e.topic}" changes over time (it has "{e.word}"). '
                    "Ask me directly and I'll look it up online. To add it anyway, end it with !!"),
                    "STUDY_REJECTED_TRANSIENT")
            status_pt, status_en = _STUDY_STATUS_LABELS.get(
                e.status, ("no histórico", "in the history"))
            return (self._t(
                f'Já tem um estudo parecido: "{e.existing}" ({status_pt}). '
                'Para estudar de novo mesmo assim, termine com !!',
                f'There is already a similar study: "{e.existing}" ({status_en}). '
                'To study it anyway, end it with !!'),
                "STUDY_REJECTED_DUPLICATE")
        except ValueError as e:
            return (str(e), "INPUT_ERROR")
        count = len(self.study_queue.list_pending())
        # Use entry["topic"] (already stripped of force-flag '!!') instead
        # of the raw input so the confirmation shows the canonical name.
        topic_name = entry["topic"]
        if count == 1:
            reply = self._t(
                f"Anotado. Vou estudar '{topic_name}' à noite. Fila com 1 estudo pendente.",
                f"Noted. I'll study '{topic_name}' tonight. Queue has 1 pending study.")
        else:
            reply = self._t(
                f"Anotado. Vou estudar '{topic_name}' à noite. Fila com {count} estudos pendentes.",
                f"Noted. I'll study '{topic_name}' tonight. Queue has {count} pending studies.")
        return (reply, "STUDY_QUEUED")

    def _handle_cancel_studies(self, indexes=None) -> Tuple[str, str]:
        """Take topics off the study queue (s83). `indexes` are 1-based as the
        listing numbers them; None means all of them.

        Cancelling one executes directly — it is undone by asking for the
        topic again. Cancelling the WHOLE queue says what it removed, because
        those are topics Pedro typed and would otherwise vanish silently."""
        pending = self.study_queue.list_pending()
        if not pending:
            return (self._t("A fila de estudos já está vazia.",
                            "The study queue is already empty."), "STUDY_LIST")
        cancelled = self.study_queue.cancel(indexes)
        if not cancelled:
            n = len(pending)
            return (self._t(
                f"não achei esse estudo na fila — tem {n} pendente"
                f"{'s' if n > 1 else ''}. Pede 'lista estudos' pra ver os números.",
                f"I couldn't find that study — there are {n} pending. Ask for "
                "'list studies' to see the numbers."), "STUDY_LIST")
        names = "; ".join(e.get("topic", "?")[:60] for e in cancelled)
        if len(cancelled) == 1:
            reply = self._t(f"Tirei da fila: {names}.",
                            f"Removed from the queue: {names}.")
        else:
            reply = self._t(
                f"Tirei {len(cancelled)} estudos da fila: {names}.",
                f"Removed {len(cancelled)} studies from the queue: {names}.")
        left = len(self.study_queue.list_pending())
        reply += self._t(
            f" Sobrou {left} na fila." if left != 1 else " Sobrou 1 na fila.",
            f" {left} left in the queue." if left != 1 else " 1 left in the queue.")
        audit.log("–", "studier", "cancelled",
                  payload={"count": len(cancelled), "left": left})
        return (reply, "STUDY_CANCELLED")

    def _handle_cancel_vague(self) -> Tuple[str, str]:
        """"exclui tudo" — tudo o quê? Ask, never guess (s83). The 8B answered
        this with a refusal and then invented a policy to justify it."""
        pending = len(self.study_queue.list_pending())
        if pending:
            return (self._t(
                f"Tudo o quê? Se é a fila de estudos ({pending} pendente"
                f"{'s' if pending > 1 else ''}), diz 'limpa a fila de estudos'.",
                f"All of what? If you mean the study queue ({pending} pending), "
                "say 'clear the study queue'."), "STUDY_LIST")
        return (self._t(
            "Tudo o quê? Me diz o que apagar — a fila de estudos está vazia.",
            "All of what? Tell me what to clear — the study queue is empty."),
            "STUDY_LIST")

    def _handle_list_studies(self) -> Tuple[str, str]:
        """Return the current Nightly Studier queue."""
        entries = self.study_queue.list_pending()
        if not entries:
            return (self._t("Fila de estudos vazia.", "Study queue is empty."), "STUDY_LIST")
        lines = [f"{i + 1}. {e['topic']}" for i, e in enumerate(entries)]
        n = len(entries)
        header = self._t(
            f"Fila de estudos ({n} {'pendente' if n == 1 else 'pendentes'}):",
            f"Study queue ({n} pending):")
        return (header + "\n" + "\n".join(lines), "STUDY_LIST")

    def _handle_last_study(self) -> Tuple[str, str]:
        """"o que você aprendeu?" — from the briefing on disk (s83).

        Says which state it is in, because an unreviewed briefing is not
        knowledge yet, and repeats the confidence, because the critic
        demoting a study to 'baixa' is the most useful thing on the page."""
        item = self.studies_review.last_study()
        if not item:
            return (self._t(
                "Ainda não estudei nada. Manda 'estuda <assunto>' que eu "
                "pesquiso.",
                "I haven't studied anything yet. Say 'study <topic>' and "
                "I'll research it."), "STUDY_LAST")
        topic, confidence = item["topic"], item["confidence"]
        summary = item.get("summary") or ""
        if len(summary) > 700:
            summary = summary[:700].rstrip() + "…"
        if item["state"] == "pending":
            head = self._t(
                f"Meu último estudo foi '{topic}' (confiança: {confidence}) — "
                "ainda não revisado, então não entrou no meu conhecimento.",
                f"My last study was '{topic}' (confidence: {confidence}) — not "
                "reviewed yet, so it hasn't entered my knowledge.")
            tail = self._t("\n\nDiga 'revisar' pra decidir se aprova.",
                           "\n\nSay 'review' to decide whether to approve it.")
        else:
            head = self._t(
                f"Meu último estudo foi '{topic}' (confiança: {confidence}), "
                "já aprovado.",
                f"My last study was '{topic}' (confidence: {confidence}), "
                "already approved.")
            tail = ""
        body = f"\n\n{summary}" if summary else ""
        audit.log("–", "studier", "last_study_read",
                  payload={"state": item["state"], "confidence": confidence})
        return (head + body + tail, "STUDY_LAST")

    def _handle_review_studies(self) -> Tuple[str, str]:
        """List briefings produced by the Nightly Studier and waiting for triage."""
        items = self.studies_review.list_pending()
        if not items:
            return (
                self._t(
                    "Nenhum estudo pronto para revisar. Adicione tópicos com "
                    "'estude X' e eu trabalho na próxima madrugada.",
                    "No studies ready to review. Add topics with 'study X' and "
                    "I'll work on them overnight."),
                "STUDIES_REVIEW",
            )
        n = len(items)
        example = "1" if n == 1 else "1, 2..."
        pt_lines = [
            f"{it['index']}. {it['topic']} (confiança: {it['confidence']})"
            for it in items
        ]
        en_lines = [
            f"{it['index']}. {it['topic']} (confidence: {it['confidence']})"
            for it in items
        ]
        pt = (
            f"Você tem {n} {'estudo' if n == 1 else 'estudos'} pronto(s) pra revisar:\n"
            + "\n".join(pt_lines)
            + f"\n\nDiga 'ver {example}' para ler, 'aprovar {example}' ou "
            f"'rejeitar {example}' para decidir, ou 'aprovar todos' / "
            "'rejeitar todos'. Também aceita 'primeiro', 'segundo', etc."
        )
        en = (
            f"You have {n} {'study' if n == 1 else 'studies'} ready to review:\n"
            + "\n".join(en_lines)
            + f"\n\nSay 'show {example}' to read, 'approve {example}' or "
            f"'reject {example}' to decide, or 'approve all' / 'reject all'. "
            "Ordinals like 'first', 'second' also work."
        )
        return (self._t(pt, en), "STUDIES_REVIEW")

    def _resolve_default_ref(self, ref: str, verb: str) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
        """
        Translate an empty `ref` into "1" when there is exactly one pending
        study, or into an ambiguity message when there are several.
        Returns (resolved_ref, error_tuple). On success error_tuple is None.
        """
        ref = (ref or "").strip()
        if ref:
            return ref, None
        # EN command verbs the studier regexes also accept (show/approve/reject).
        en_verb = {"ver": "show", "aprovar": "approve", "rejeitar": "reject"}.get(verb, verb)
        verb_noun = {"ver": "view", "aprovar": "approve", "rejeitar": "reject"}.get(verb, verb)
        items = self.studies_review.list_pending()
        if not items:
            return None, (
                self._t(f"Nenhum estudo pendente para {verb}.",
                        f"No pending studies to {verb_noun}."),
                "STUDY_NOT_FOUND",
            )
        if len(items) == 1:
            return "1", None
        n = len(items)
        return None, (
            self._t(
                f"Qual estudo {verb}? Você tem {n} pendentes. "
                f"Diga '{verb} 1' (ou outro número), '{verb} todos', "
                "ou rode 'revisar' para ver a lista.",
                f"Which study to {verb_noun}? You have {n} pending. "
                f"Say '{en_verb} 1' (or another number), '{en_verb} all', "
                "or run 'review' to see the list."),
            "STUDY_AMBIGUOUS",
        )

    def _handle_show_study(self, ref: str) -> Tuple[str, str]:
        """Return the full Markdown of a single briefing."""
        ref, err = self._resolve_default_ref(ref, "ver")
        if err:
            return err
        result = self.studies_review.get_briefing(ref)
        if not result:
            return (
                self._t(f"Estudo '{ref}' não encontrado. Use 'revisar' para listar.",
                        f"Study '{ref}' not found. Use 'review' to list."),
                "STUDY_NOT_FOUND",
            )
        item, content = result
        return (
            self._t(f"Estudo {item['index']} — {item['topic']}:\n\n{content}",
                    f"Study {item['index']} — {item['topic']}:\n\n{content}"),
            "STUDY_SHOW",
        )

    def _handle_approve_study(self, ref: str) -> Tuple[str, str]:
        """Move a pending briefing into approved/ (indexed into RAG)."""
        if self._ALL_REF_RE.match(ref or ""):
            moved, failed = self.studies_review.approve_all()
            if not moved and not failed:
                return (self._t("Nenhum estudo pendente para aprovar.",
                                "No pending studies to approve."), "STUDY_APPROVE")
            if failed:
                names = ", ".join(f"'{t}'" for t in failed)
                return (
                    self._t(
                        f"Aprovei {len(moved)} estudos. Não consegui guardar na "
                        f"memória {names} — continuam pendentes; aprova de novo "
                        "daqui a pouco.",
                        f"Approved {len(moved)} studies. I couldn't store {names} "
                        "in memory — still pending; approve again shortly."),
                    "STUDY_APPROVE_ERROR",
                )
            return (
                self._t(
                    f"Aprovados {len(moved)} estudos. Agora fazem parte da "
                    "memória permanente.",
                    f"Approved {len(moved)} studies. They're now part of "
                    "permanent memory."),
                "STUDY_APPROVE",
            )

        ref, err = self._resolve_default_ref(ref, "aprovar")
        if err:
            return err

        result = self.studies_review.get_briefing(ref)
        if not result:
            return (
                self._t(f"Estudo '{ref}' não encontrado. Use 'revisar' para listar.",
                        f"Study '{ref}' not found. Use 'review' to list."),
                "STUDY_NOT_FOUND",
            )
        item, _ = result
        from src.core.studies_review import IndexFailed
        try:
            new_path = self.studies_review.approve_item(item)
        except IndexFailed:
            # Not "memória permanente": it is not in the Palace (s88, lote 14).
            return (
                self._t(f"Não consegui guardar '{item['topic']}' na memória agora — "
                        "ele continua pendente. Aprova de novo daqui a pouco.",
                        f"I couldn't store '{item['topic']}' in memory just now — "
                        "it's still pending. Approve it again shortly."),
                "STUDY_APPROVE_ERROR",
            )
        if not new_path:
            return (
                self._t(f"Não aprovei: '{item['topic']}' já não está entre os pendentes "
                        "(foi aprovado ou descartado agora há pouco). Use 'revisar' "
                        "para ver a lista.",
                        f"Not approved: '{item['topic']}' is no longer pending (it was "
                        "just approved or discarded). Use 'review' to see the list."),
                "STUDY_APPROVE_ERROR",
            )
        return (
            self._t(f"Aprovado. '{item['topic']}' agora faz parte da memória permanente.",
                    f"Approved. '{item['topic']}' is now part of permanent memory."),
            "STUDY_APPROVE",
        )

    def _handle_reject_study(self, ref: str) -> Tuple[str, str]:
        """Delete a pending briefing."""
        if self._ALL_REF_RE.match(ref or ""):
            deleted = self.studies_review.reject_all()
            if not deleted:
                return (self._t("Nenhum estudo pendente para rejeitar.",
                                "No pending studies to reject."), "STUDY_REJECT")
            return (
                self._t(f"Descartados {len(deleted)} estudos pendentes.",
                        f"Discarded {len(deleted)} pending studies."),
                "STUDY_REJECT",
            )

        ref, err = self._resolve_default_ref(ref, "rejeitar")
        if err:
            return err

        result = self.studies_review.get_briefing(ref)
        if not result:
            return (
                self._t(f"Estudo '{ref}' não encontrado. Use 'revisar' para listar.",
                        f"Study '{ref}' not found. Use 'review' to list."),
                "STUDY_NOT_FOUND",
            )
        item, _ = result
        deleted = self.studies_review.reject_item(item)
        if not deleted:
            return (
                self._t(f"Não descartei: '{item['topic']}' já não está entre os "
                        "pendentes. Use 'revisar' para ver a lista.",
                        f"Not discarded: '{item['topic']}' is no longer pending. "
                        "Use 'review' to see the list."),
                "STUDY_REJECT_ERROR",
            )
        return (
            self._t(f"Descartado. '{item['topic']}' foi removido.",
                    f"Discarded. '{item['topic']}' was removed."),
            "STUDY_REJECT",
        )
