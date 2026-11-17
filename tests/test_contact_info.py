"""
Tests for the contact INFO lookup (s55) — "qual o telefone do X" + full card.

Covers CONTACT_INFO_RE and orchestrator._handle_contact_info, including the
None-fallthrough safety so an ambiguous "número do meu passaporte" is NOT
hijacked when no contact matches.
"""
import pytest

from src.integrations.contacts import ContactBook
from src.core.dispatch.triggers import CONTACT_INFO_RE


def _name(m):
    return next((m.group(g) for g in ("name1", "name2", "name3", "name4")
                 if m.group(g)), None)


# --- regex -----------------------------------------------------------------

@pytest.mark.parametrize("text,name", [
    ("qual o telefone do Lucas", "Lucas"),
    ("telefone do Lucas", "Lucas"),
    ("qual o número do Lucas", "Lucas"),
    ("me fala do contato Lucas", "Lucas"),
    ("mostra o contato Maria", "Maria"),
    ("info do João", "João"),
    ("dados da Ana", "Ana"),
    ("what is Lucas's phone", "Lucas"),
    ("show contact Bob", "Bob"),
])
def test_info_re_matches(text, name):
    m = CONTACT_INFO_RE.match(text)
    assert m and _name(m).strip() == name


@pytest.mark.parametrize("text", [
    "qual o email da lara",          # email → CONTACT_LOOKUP_RE, not this
    "oi cortex",
])
def test_info_re_ignores(text):
    assert not CONTACT_INFO_RE.match(text)


def test_info_re_passport_captures_but_handler_will_drop():
    # The regex DOES capture "meu passaporte" (número is ambiguous); the safety
    # is in the handler (returns None → fall through). See handler tests below.
    m = CONTACT_INFO_RE.match("qual o número do meu passaporte")
    assert m and _name(m).strip() == "meu passaporte"


# --- handler ---------------------------------------------------------------

def _orch(tmp_path, monkeypatch, seed=None):
    import src.integrations.contacts as contacts_mod
    book = ContactBook(db_path=str(tmp_path / "c.db"))
    if seed:
        book.add(**seed)
    monkeypatch.setattr(contacts_mod, "contact_book", book)
    from src.core.orchestrator import CortexOrchestrator
    return CortexOrchestrator.__new__(CortexOrchestrator)


def test_info_returns_full_card(tmp_path, monkeypatch):
    o = _orch(tmp_path, monkeypatch, seed=dict(
        name="Lucas Lima", email="lucas@x.com", phone="+44 7700", notes="irmão"))
    reply, mode = o._handle_contact_info("Lucas")
    assert mode == "CONTACT_INFO"
    assert "+44 7700" in reply and "lucas@x.com" in reply and "irmão" in reply


def test_info_phone_only_when_thats_all(tmp_path, monkeypatch):
    o = _orch(tmp_path, monkeypatch, seed=dict(name="Bia", phone="119999"))
    reply, mode = o._handle_contact_info("Bia")
    assert mode == "CONTACT_INFO" and "119999" in reply


def test_info_unknown_contact_returns_none(tmp_path, monkeypatch):
    # The passport-collision safety: no contact → None → caller falls through.
    o = _orch(tmp_path, monkeypatch)
    assert o._handle_contact_info("meu passaporte") is None


def test_info_contact_with_no_fields(tmp_path, monkeypatch):
    o = _orch(tmp_path, monkeypatch, seed=dict(name="Vazio"))  # name only
    reply, mode = o._handle_contact_info("Vazio")
    assert mode == "CONTACT_INFO_EMPTY"


# ----------------------------------------------------------------------
# M4-A10: contact saved WITHOUT email (s55 voice-add feature) — the
# consumers must cover the empty field instead of leaking it.
# ----------------------------------------------------------------------


class TestContactWithoutEmail:
    def test_lookup_no_email_is_honest(self, tmp_path, monkeypatch):
        # Was: "Email de Lucas: ." — the empty field rendered as a dot.
        o = _orch(tmp_path, monkeypatch, seed=dict(name="Lucas", phone="119"))
        reply, mode = o._handle_contact_lookup("Lucas")
        assert mode == "CONTACT_LOOKUP_NO_EMAIL"
        assert "sem email" in reply
        assert "salva o email da Lucas" in reply
        assert "Email de Lucas: ." not in reply

    def test_lookup_no_email_en(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch, seed=dict(name="Lucas", phone="119"))
        o._reply_lang = "en"
        reply, mode = o._handle_contact_lookup("Lucas")
        assert mode == "CONTACT_LOOKUP_NO_EMAIL"
        assert "no email" in reply

    def test_send_by_name_refuses_before_staging(self, tmp_path, monkeypatch):
        # Was: staged send_email with to="" — only failed in execution,
        # AFTER the user said "sim" to an empty recipient.
        o = _orch(tmp_path, monkeypatch, seed=dict(name="Lucas", phone="119"))
        o._pending_action = None
        reply, mode = o._handle_send_email_by_name(
            name="Lucas", subject="oi", body="tudo bem?")
        assert mode == "CONTACT_LOOKUP_NO_EMAIL"
        assert "não dá pra enviar" in reply
        assert o._pending_action is None, "must refuse BEFORE staging"

    def test_send_by_name_with_email_still_stages(self, tmp_path, monkeypatch):
        o = _orch(tmp_path, monkeypatch, seed=dict(
            name="Lucas", email="lucas@x.com"))
        o._pending_action = None
        monkeypatch.setattr(
            type(o), "_PENDING_ACTION_FILE",
            str(tmp_path / "pending.json"), raising=False)
        reply, mode = o._handle_send_email_by_name(
            name="Lucas", subject="oi", body="tudo bem?")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert o._pending_action["args"]["to"] == "lucas@x.com"
