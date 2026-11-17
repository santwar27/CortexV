"""
Email skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. The dispatch registry routes the structured send
("manda email para X: …"), send-by-name ("manda email pra Lara: …") and the
inbox shortcut ("meus últimos N emails") here. Mixin so the call-sites resolve
through the MRO; the staging handlers reach into orchestrator/StagingMixin
state (`self._pending_action`, `self._persist_pending_action`,
`self._human_confirmation_prompt`). `audit` is the shared singleton (patching
`audit.log` anywhere patches it here too). Send paths stage an S3 action and
wait for the user's "sim"; nothing leaves without confirmation.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.core.audit import audit

log = logging.getLogger(__name__)


class EmailMixin:
    """Gmail send (structured / by-name) + inbox list handlers."""

    def _handle_send_email_by_name(
        self, *, name: str, subject: str, body: str,
    ) -> Tuple[str, str]:
        """Resolve <name> → email via the contact book, then stage the
        same _pending_action shape as the address-explicit path. The
        S3 confirmation prompt downstream shows the RESOLVED address so
        the user can catch a wrong match before the email goes out."""
        from src.integrations.contacts import contact_book
        results = contact_book.lookup_fuzzy(name)
        if not results:
            return (
                self._t(f"Não tenho email salvo pra {name}. Tenta "
                        f"'salva o email da {name} como X@Y.com' antes.",
                        f"I don't have an email saved for {name}. Try "
                        f"'save {name}'s email as X@Y.com' first."),
                "CONTACT_LOOKUP_MISS",
            )
        if len(results) > 1:
            listed = "\n".join(f"  · {c.name}: {c.email}" for c in results)
            return (
                self._t(f"Mais de um contato bate com '{name}':\n{listed}\n"
                        f"Diz o nome exato e tento de novo.",
                        f"More than one contact matches '{name}':\n{listed}\n"
                        f"Tell me the exact name and I'll try again."),
                "CONTACT_LOOKUP_AMBIGUOUS",
            )
        # Contact exists but has no email (saved by voice without one —
        # s55 feature): refuse BEFORE staging. The old path staged a
        # send_email with to="" that only failed in execution, AFTER the
        # user said "sim" to an empty recipient (M4-A10).
        if not (results[0].email or "").strip():
            return (
                self._t(f"Tenho {results[0].name} salvo, mas sem email — "
                        f"não dá pra enviar. Tenta 'salva o email da "
                        f"{results[0].name} como X@Y.com' antes.",
                        f"I have {results[0].name} saved, but with no email "
                        f"— I can't send. Try 'save {results[0].name}'s "
                        f"email as X@Y.com' first."),
                "CONTACT_LOOKUP_NO_EMAIL",
            )
        # Single match — delegate to the structured path so the
        # confirmation flow + S3 staging are identical to a direct-
        # address send.
        return self._handle_send_email_structured(
            to=results[0].email, subject=subject, body=body,
        )

    def _handle_send_email_structured(
        self, *, to: str, subject: str, body: str,
    ) -> Tuple[str, str]:
        """Stage a send_email action with LITERAL fields parsed by
        SEND_EMAIL_STRUCTURED_RE — bypasses chat_with_tools so the 8B
        can't paraphrase the body or smuggle typos into the subject.

        Shape of `_pending_action` mirrors the LLM-tool path exactly
        (same `{"type": "send_email", "args": {...}}` slot, same
        persistence, same S3 confirmation prompt downstream) — so the
        existing sim/não handler executes it without any extra plumbing.
        """
        self._pending_action = {
            "type": "send_email",
            "args": {"to": to, "subject": subject, "body": body},
        }
        self._persist_pending_action()
        audit.log(
            "–", "tool", "stage",
            payload={
                "name": "send_email",
                "source": "regex_structured",  # vs "llm_tool" for the chat_with_tools path
                "to": to,
            },
        )
        return (
            self._human_confirmation_prompt(self._pending_action),
            "AGENT_AWAITING_CONFIRMATION",
        )

    def _handle_list_emails(self, count: int = 5, query: Optional[str] = None) -> Tuple[str, str]:
        """Surface recent emails via the Gmail integration."""
        # Lazy import — only pay the cost when actually used.
        try:
            from src.integrations.gmail_client import (
                GmailUnavailable, format_email_list, list_recent,
            )
        except Exception as e:
            # The exception goes to the log, never into the result: the follow-up is
            # told to repeat a tool error verbatim, and the 8B read it out 6 times in
            # 6 ("O erro foi: EOF marker not found" — s88, Codex lote 8).
            log.warning("gmail import failed: %r", e)
            return (self._t("[SYSTEM ERROR] Integração Gmail indisponível.",
                            "[SYSTEM ERROR] Gmail integration unavailable."), "GMAIL_ERROR")

        try:
            emails = list_recent(count=count, query=query)
        except GmailUnavailable as e:
            log.warning("list_emails: %r", e)
            return (
                self._t(
                    "Acesso ao Gmail ainda não foi configurado. "
                    "Rode './venv/bin/python scripts/google_oauth_setup.py' uma vez "
                    "para autorizar.",
                    "Gmail access isn't configured yet. Run "
                    "'./venv/bin/python scripts/google_oauth_setup.py' once "
                    "to authorise."),
                "GMAIL_NOT_CONFIGURED",
            )
        except Exception as e:
            log.warning("list_emails failed: %r", e)
            return (self._t("[SYSTEM ERROR] Falha ao consultar Gmail.",
                            "[SYSTEM ERROR] Failed to query Gmail."), "GMAIL_ERROR")

        if not emails:
            return (self._t("Caixa de entrada vazia.", "Inbox is empty."), "GMAIL_LIST")
        body = format_email_list(emails, lang=self._reply_lang)
        return (self._t(f"Seus {len(emails)} emails mais recentes:\n{body}",
                        f"Your {len(emails)} most recent emails:\n{body}"), "GMAIL_LIST")
