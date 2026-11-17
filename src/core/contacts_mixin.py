"""
Contact-book skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. The dispatch registry routes "adiciona contato …" /
"salva o email da X" / "qual o email da X" / "telefone do X" here. Mixin so
`self._handle_contact_*(...)` still resolves through the MRO. These handlers
import `contact_book` / `parse_contact_fields` locally from
`src.integrations.contacts`, so they carry no orchestrator-module state.

CONTACT_INFO falls through (returns None) when the name matches no contact, so
an ambiguous query ("número do meu passaporte") reaches normal routing instead
of a wrong "no contact" reply.
"""
from __future__ import annotations

from typing import Optional, Tuple


class ContactsMixin:
    """Contact CRUD + lookup handlers. Mixed into the orchestrator."""

    def _handle_contact_save(self, *, name: str, email: str) -> Tuple[str, str]:
        """Persist a name→email mapping in the contact book. Idempotent:
        if the name already exists, call update(); otherwise add()."""
        from src.integrations.contacts import contact_book
        existing = contact_book.lookup(name)
        if existing:
            ok = contact_book.update(name, email=email)
        else:
            ok = contact_book.add(name, email)
        if not ok:
            fail = self._t("não pude atualizar" if existing else "não pude salvar",
                           "couldn't update" if existing else "couldn't save")
            return (
                self._t(f"[SYSTEM ERROR] Contato {name}: {fail}. Tenta de novo.",
                        f"[SYSTEM ERROR] Contact {name}: {fail}. Try again."),
                "CONTACT_SAVE_ERROR",
            )
        verb = self._t("atualizado" if existing else "salvo",
                       "updated" if existing else "saved")
        return (self._t(f"Contato {verb}: {name} → {email}.",
                        f"Contact {verb}: {name} → {email}."), "CONTACT_SAVE")

    def _handle_contact_add(self, fields_text: str) -> Tuple[str, str]:
        """Rich contact add (s55): parse labelled fields (nome/email/telefone/
        notas) and upsert. Email is optional — a name-only contact is fine
        (good for voice). Deterministic parse → no fabricated fields."""
        from src.integrations.contacts import contact_book, parse_contact_fields

        fields = parse_contact_fields(fields_text or "")
        name = (fields.get("name") or "").strip()
        if not name:
            return (
                self._t(
                    "Pra adicionar um contato me diga ao menos o nome. Ex.: "
                    "'adiciona contato nome: Lucas, email: lucas@x.com, "
                    "telefone: +44..., notas: amigo da facul'.",
                    "To add a contact give me at least the name. E.g.: "
                    "'add contact name: Lucas, email: lucas@x.com, "
                    "phone: +44..., notes: friend from uni'."),
                "CONTACT_ADD_HELP",
            )
        email = (fields.get("email") or "").strip()
        phone = fields.get("phone")
        notes = fields.get("notes")

        if contact_book.lookup(name) is not None:
            ok = contact_book.update(
                name, email=email or None, phone=phone, notes=notes)
            existing = True
        else:
            ok = contact_book.add(name, email, phone=phone, notes=notes)
            existing = False
        if not ok:
            fail = self._t("não pude atualizar" if existing else "não pude salvar",
                           "couldn't update" if existing else "couldn't save")
            return (self._t(f"[SYSTEM ERROR] Contato {name}: {fail}.",
                            f"[SYSTEM ERROR] Contact {name}: {fail}."), "CONTACT_ADD_ERROR")

        verb = self._t("atualizado" if existing else "salvo",
                       "updated" if existing else "saved")
        bits = [name]
        if email:
            bits.append(email)
        if phone:
            bits.append(self._t(f"tel {phone}", f"phone {phone}"))
        if notes:
            bits.append(self._t(f"nota: {notes}", f"note: {notes}"))
        tail = "" if email else self._t(" (sem email — dá pra adicionar depois)",
                                        " (no email — you can add one later)")
        return (self._t(f"Contato {verb}: {', '.join(bits)}.{tail}",
                        f"Contact {verb}: {', '.join(bits)}.{tail}"), "CONTACT_ADD")

    def _handle_contact_info(self, name: str) -> Optional[Tuple[str, str]]:
        """Return the stored card (phone/email/notes) for a contact. Returns
        None when no contact matches the name — the caller FALLS THROUGH so an
        ambiguous query ("número do meu passaporte") reaches normal routing
        instead of a wrong 'no contact' reply. Never invents a field."""
        from src.integrations.contacts import contact_book
        results = contact_book.lookup_fuzzy(name)
        if not results:
            return None  # not a known contact → let normal routing handle it
        if len(results) > 1:
            listed = "\n".join(f"  · {c.name}" for c in results)
            return (self._t(f"Achei mais de um contato:\n{listed}\nDiz o nome exato.",
                            f"I found more than one contact:\n{listed}\nTell me the exact name."),
                    "CONTACT_INFO_AMBIGUOUS")
        c = results[0]
        bits = []
        if c.phone:
            bits.append(self._t(f"telefone {c.phone}", f"phone {c.phone}"))
        if c.email:
            bits.append(f"email {c.email}")
        if c.notes:
            bits.append(self._t(f"nota: {c.notes}", f"note: {c.notes}"))
        if not bits:
            return (self._t(f"Tenho {c.name} salvo, mas sem telefone/email/nota.",
                            f"I have {c.name} saved, but with no phone/email/note."),
                    "CONTACT_INFO_EMPTY")
        return (f"{c.name}: {'; '.join(bits)}.", "CONTACT_INFO")

    def _handle_contact_lookup(self, name: str) -> Tuple[str, str]:
        """Return the email for `name` (exact then fuzzy). Honest refusal
        when nothing matches — never invent an address."""
        from src.integrations.contacts import contact_book
        results = contact_book.lookup_fuzzy(name)
        if not results:
            return (
                self._t(f"Não tenho email salvo pra {name}. Tenta "
                        f"'salva o email da {name} como X@Y.com'.",
                        f"I don't have an email saved for {name}. Try "
                        f"'save {name}'s email as X@Y.com'."),
                "CONTACT_LOOKUP_MISS",
            )
        if len(results) == 1:
            c = results[0]
            # Contact saved without an email is a feature (s55 voice add)
            # — but answering "Email de Lucas: ." with the empty field was
            # not (M4-A10). Honest branch + the fix instruction.
            if not (c.email or "").strip():
                return (
                    self._t(f"Tenho {c.name} salvo, mas sem email. Tenta "
                            f"'salva o email da {c.name} como X@Y.com'.",
                            f"I have {c.name} saved, but with no email. Try "
                            f"'save {c.name}'s email as X@Y.com'."),
                    "CONTACT_LOOKUP_NO_EMAIL",
                )
            return (self._t(f"Email de {c.name}: {c.email}.",
                            f"{c.name}'s email: {c.email}."), "CONTACT_LOOKUP")
        # Multiple close matches — let the user disambiguate.
        listed = "\n".join(f"  · {c.name}: {c.email}" for c in results)
        return (
            self._t(f"Achei mais de um:\n{listed}\nDiz o nome exato pra ver só esse.",
                    f"I found more than one:\n{listed}\nTell me the exact name to see just that one."),
            "CONTACT_LOOKUP_AMBIGUOUS",
        )
