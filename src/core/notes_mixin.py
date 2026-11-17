"""
Apple Notes skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) — the first of the per-family
handler mixins that follow the dispatch-registry refactor. The registry routes
"anota …" / "minhas notas" / "acha a nota …" to these methods; carving them into
a mixin shrinks the orchestrator god-object while `self._handle_note_*(...)`
still resolves through the MRO (mirrors `staging_mixin.py`). The methods reach
into orchestrator state — `self.notes`, the Notes client the host provides.
"""
from __future__ import annotations

from typing import Optional, Tuple
from uuid import uuid4

from src.core.action_ledger import ledger as _action_ledger
from src.core.effect_runner import run_effect
from src.integrations.notes_client import NotesUnavailable, _NOTE_ID_RE


class NotesMixin:
    """Apple Notes handlers (create / list / search). Mixed into the
    orchestrator; relies on the host providing `self.notes`."""

    def _handle_note_create(
        self, body: str, folder: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Create a note from a regex-parsed body. The first line is the
        title (Notes derives it from the body's first line); the rest is the
        content. `folder` (v1.1) targets/creates that folder. Returns None on
        an empty body (fall through to the LLM)."""
        body = (body or "").strip()
        if not body:
            return None
        first, _, rest = body.partition("\n")
        title = first.strip()
        note_body = rest.strip()
        folder = (folder or "").strip() or None
        def create():
            try:
                result = self.notes.create_note(title=title, body=note_body, folder=folder)
            except NotesUnavailable as e:
                # A timeout may already have written the note; never invite a retry.
                if getattr(e, "uncertain", False):
                    raise
                return (self._t(f"[SYSTEM ERROR] Não consegui criar a nota: {e}",
                                f"[SYSTEM ERROR] Couldn't create the note: {e}"), "AGENT_ACTION_ERROR")
            args = {"title": title}
            note_id = result.get("id") if isinstance(result, dict) else None
            if isinstance(note_id, str) and _NOTE_ID_RE.fullmatch(note_id):
                args["note_id"] = note_id
            _action_ledger.record("create_note", args, source="dispatch")
            if folder:
                return (self._t(f'Anotado: "{title}" (pasta {folder}).',
                                f'Noted: "{title}" (folder {folder}).'), "AGENT_ACTION_DONE")
            return (self._t(f'Anotado: "{title}".', f'Noted: "{title}".'), "AGENT_ACTION_DONE")

        reply, mode = run_effect(
            "create_note", {"title": title, "body": note_body, "folder": folder}, create,
            effect_id=uuid4().hex, origin=getattr(self, "_effect_origin", getattr(self, "_turn_origin", lambda: None))(),
            cancelled=lambda: getattr(self, "_turn_cancelled", lambda: False)(),
            lang=self._reply_lang)
        return reply, ("NOTE_CREATE" if mode == "AGENT_ACTION_DONE" else
                       "AGENT_ACTION_CANCELLED" if mode == "AGENT_ACTION_CANCELLED" else "NOTE_ERROR")

    def _handle_note_append(
        self, title: str, text: str,
    ) -> Optional[Tuple[str, str]]:
        """Append text to an existing note (v1.1, find-then-modify). A miss
        (no note with that title) is an honest 'não achei' — never a silent
        create. Recorded in the ledger as irreversible (an append can't be
        precisely un-appended; reverse_spec doesn't know this action)."""
        title = (title or "").strip()
        text = (text or "").strip()
        if not title or not text:
            return None
        missed = False

        def append():
            nonlocal missed
            try:
                result = self.notes.append_to_note(title=title, text=text)
            except NotesUnavailable as e:
                if getattr(e, "uncertain", False):
                    raise
                if "no note found" in str(e).lower():
                    missed = True
                    return (self._t(f'Não achei nenhuma nota chamada "{title}".',
                                    f'I found no note called "{title}".'), "AGENT_ACTION_ERROR")
                return (self._t(f"[SYSTEM ERROR] Não consegui editar a nota: {e}",
                                f"[SYSTEM ERROR] Couldn't edit the note: {e}"), "AGENT_ACTION_ERROR")
            matched = result.get("title", title)
            _action_ledger.record("append_to_note", {"title": matched}, source="dispatch")
            return (self._t(f'Adicionado à nota "{matched}".',
                            f'Added to note "{matched}".'), "AGENT_ACTION_DONE")

        reply, mode = run_effect(
            "append_to_note", {"title": title, "text": text}, append,
            effect_id=uuid4().hex, origin=getattr(self, "_effect_origin", getattr(self, "_turn_origin", lambda: None))(),
            cancelled=lambda: getattr(self, "_turn_cancelled", lambda: False)(),
            lang=self._reply_lang)
        return reply, ("NOTE_APPEND" if mode == "AGENT_ACTION_DONE" or missed else
                       "AGENT_ACTION_CANCELLED" if mode == "AGENT_ACTION_CANCELLED" else "NOTE_ERROR")

    def _handle_note_list(self) -> Tuple[str, str]:
        try:
            notes = self.notes.list_notes(limit=10)
        except Exception as e:
            return (self._t(f"[SYSTEM ERROR] Não consegui listar as notas: {e}",
                            f"[SYSTEM ERROR] Couldn't list the notes: {e}"), "NOTE_ERROR")
        if not notes:
            return (self._t("Você não tem notas.", "You have no notes."), "NOTE_LIST")
        lines = "\n".join(f"• {n['title']}" for n in notes)
        return (self._t(f"Suas notas:\n{lines}", f"Your notes:\n{lines}"), "NOTE_LIST")

    def _handle_note_search(self, query: str) -> Optional[Tuple[str, str]]:
        query = (query or "").strip()
        if not query:
            return None
        try:
            hits = self.notes.search_notes(query, limit=10)
        except Exception as e:
            return (self._t(f"[SYSTEM ERROR] Não consegui buscar notas: {e}",
                            f"[SYSTEM ERROR] Couldn't search notes: {e}"), "NOTE_ERROR")
        if not hits:
            return (self._t(f'Não achei nenhuma nota sobre "{query}".',
                            f'I found no notes about "{query}".'), "NOTE_SEARCH")
        lines = "\n".join(f"• {h['title']}" for h in hits)
        return (self._t(f'Notas sobre "{query}":\n{lines}',
                        f'Notes about "{query}":\n{lines}'), "NOTE_SEARCH")
