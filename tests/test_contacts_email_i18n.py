"""
Contacts + email mixin reply localisation (s60 follow-up, Claude-A scope C).

Closes the i18n regression gap for the two command families whose
localisation lives in the mixin but had no EN-path shield: contacts
(`_handle_contact_info`) and email (`_handle_list_emails`).

Both handlers reach a module-level integration singleton at call time
(`contact_book` / `gmail_client.list_recent`) rather than a `self.*` stub,
so they don't fit the fake_orch DISPATCH_REPLY eval (which deliberately
avoids network/file mocking). The clean home is a unit test that patches
the integration — the same pattern as test_macos_i18n / test_spotify_i18n.

Every behaviour is asserted in BOTH languages so PT (the default) can
never silently win the EN path back.
"""
from unittest.mock import patch

from src.core.contacts_mixin import ContactsMixin
from src.core.email_mixin import EmailMixin
from src.core.i18n import I18nMixin
from src.integrations.contacts import Contact


class _ContactsHost(ContactsMixin, I18nMixin):
    pass


class _EmailHost(EmailMixin, I18nMixin):
    pass


def _contact(name, *, email="", phone=None, notes=None):
    return Contact(id=1, name=name, email=email, phone=phone, notes=notes,
                   created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")


class TestContactInfoEN:
    def test_ambiguous_en(self):
        host = _ContactsHost()
        host._reply_lang = "en"
        two = [_contact("Ana Silva"), _contact("Ana Souza")]
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=two):
            reply, mode = host._handle_contact_info("ana")
        assert mode == "CONTACT_INFO_AMBIGUOUS"
        assert "I found more than one contact" in reply
        assert "mais de um contato" not in reply

    def test_ambiguous_pt_default(self):
        host = _ContactsHost()
        two = [_contact("Ana Silva"), _contact("Ana Souza")]
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=two):
            reply, mode = host._handle_contact_info("ana")
        assert mode == "CONTACT_INFO_AMBIGUOUS"
        assert "Achei mais de um contato" in reply

    def test_empty_card_en(self):
        host = _ContactsHost()
        host._reply_lang = "en"
        one = [_contact("Bob")]  # no phone/email/notes → empty-card branch
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=one):
            reply, mode = host._handle_contact_info("bob")
        assert mode == "CONTACT_INFO_EMPTY"
        assert "I have Bob saved" in reply
        assert "salvo" not in reply

    def test_empty_card_pt_default(self):
        host = _ContactsHost()
        one = [_contact("Bob")]
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=one):
            reply, mode = host._handle_contact_info("bob")
        assert mode == "CONTACT_INFO_EMPTY"
        assert "Tenho Bob salvo" in reply

    def test_field_labels_en(self):
        host = _ContactsHost()
        host._reply_lang = "en"
        one = [_contact("Bob", phone="+55 11 9", notes="gym buddy")]
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=one):
            reply, mode = host._handle_contact_info("bob")
        assert mode == "CONTACT_INFO"
        assert "phone" in reply and "note:" in reply
        assert "telefone" not in reply and "nota:" not in reply

    def test_field_labels_pt_default(self):
        host = _ContactsHost()
        one = [_contact("Bob", phone="+55 11 9", notes="parça da academia")]
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=one):
            reply, mode = host._handle_contact_info("bob")
        assert mode == "CONTACT_INFO"
        assert "telefone" in reply and "nota:" in reply

    def test_no_match_falls_through(self):
        """No contact → None so normal routing can take over (lang-agnostic)."""
        host = _ContactsHost()
        host._reply_lang = "en"
        with patch("src.integrations.contacts.contact_book.lookup_fuzzy", return_value=[]):
            assert host._handle_contact_info("passport number") is None


class TestListEmailsEN:
    def test_empty_inbox_en(self):
        host = _EmailHost()
        host._reply_lang = "en"
        with patch("src.integrations.gmail_client.list_recent", return_value=[]):
            reply, mode = host._handle_list_emails()
        assert mode == "GMAIL_LIST"
        assert reply == "Inbox is empty."

    def test_empty_inbox_pt_default(self):
        host = _EmailHost()
        with patch("src.integrations.gmail_client.list_recent", return_value=[]):
            reply, mode = host._handle_list_emails()
        assert mode == "GMAIL_LIST"
        assert reply == "Caixa de entrada vazia."

    def test_list_en(self):
        host = _EmailHost()
        host._reply_lang = "en"
        emails = [
            {"from": "Ana <ana@x.com>", "subject": "Lunch?"},
            {"from": "Bob <bob@y.com>", "subject": "Invoice"},
        ]
        with patch("src.integrations.gmail_client.list_recent", return_value=emails):
            reply, mode = host._handle_list_emails(count=2)
        assert mode == "GMAIL_LIST"
        assert reply.startswith("Your 2 most recent emails:")
        assert "mais recentes" not in reply

    def test_list_pt_default(self):
        host = _EmailHost()
        emails = [
            {"from": "Ana <ana@x.com>", "subject": "Almoço?"},
            {"from": "Bob <bob@y.com>", "subject": "Nota fiscal"},
        ]
        with patch("src.integrations.gmail_client.list_recent", return_value=emails):
            reply, mode = host._handle_list_emails(count=2)
        assert mode == "GMAIL_LIST"
        assert reply.startswith("Seus 2 emails mais recentes:")
