"""
Tests for the rich "adiciona contato …" flow (s55):
  - contacts.parse_contact_fields (deterministic labelled-field parser)
  - ContactBook.add now allows email-optional (name-only contact)
  - CONTACT_ADD_RE dispatch trigger
  - orchestrator._handle_contact_add (upsert + email-optional)
"""
import pytest

from src.integrations.contacts import ContactBook, parse_contact_fields
from src.core.dispatch.triggers import CONTACT_ADD_RE


# --- parse_contact_fields --------------------------------------------------

def test_parse_full_labelled_with_commas():
    f = parse_contact_fields("nome: Lucas, email: lucas@x.com, telefone: +44 7700, notas: amigo da facul")
    assert f == {"name": "Lucas", "email": "lucas@x.com", "phone": "+44 7700", "notes": "amigo da facul"}


def test_parse_labels_without_colons():
    f = parse_contact_fields("nome Maria email maria@x.com notas chefe consular")
    assert f["name"] == "Maria" and f["email"] == "maria@x.com" and f["notes"] == "chefe consular"


def test_parse_name_only():
    f = parse_contact_fields("Pedro Henrique")
    assert f["name"] == "Pedro Henrique" and f["email"] is None


def test_parse_english_labels():
    f = parse_contact_fields("name: Sarah email: sarah@x.com")
    assert f["name"] == "Sarah" and f["email"] == "sarah@x.com"


def test_parse_email_lowercased():
    assert parse_contact_fields("nome: X email: LUCAS@X.COM")["email"] == "lucas@x.com"


def test_parse_email_fallback_without_label():
    # email captured even when not explicitly labelled
    f = parse_contact_fields("nome: João joao@x.com")
    assert f["email"] == "joao@x.com"
    assert "joao@x.com" not in (f["name"] or "")   # stripped out of the name


def test_parse_empty():
    assert parse_contact_fields("") == {"name": None, "email": None, "phone": None, "notes": None}


# --- ContactBook.add email-optional ----------------------------------------

def test_add_allows_empty_email(tmp_path):
    book = ContactBook(db_path=str(tmp_path / "c.db"))
    assert book.add("Lucas") is True            # name only
    c = book.lookup("Lucas")
    assert c is not None and c.email == ""


def test_add_still_requires_name(tmp_path):
    book = ContactBook(db_path=str(tmp_path / "c.db"))
    assert book.add("", "x@y.com") is False


# --- dispatch regex --------------------------------------------------------

@pytest.mark.parametrize("text,fields_start", [
    ("adiciona contato nome: Lucas", "nome: Lucas"),
    ("adicionar um contato nome Maria", "nome Maria"),
    ("vamos adicionar um contato. nome: João", "nome: João"),
    ("cadastra contato Pedro", "Pedro"),
    ("add contact name: Sarah", "name: Sarah"),
    ("new contact name: Bob", "name: Bob"),
])
def test_contact_add_re_matches(text, fields_start):
    m = CONTACT_ADD_RE.match(text)
    assert m and m.group("fields").startswith(fields_start)


@pytest.mark.parametrize("text", [
    "qual o email da Maria",          # lookup, not add
    "salva o email da lara como f@x.com",   # the narrow email-save form
    "oi cortex",
])
def test_contact_add_re_ignores(text):
    assert not CONTACT_ADD_RE.match(text)


# --- handler integration ---------------------------------------------------

def _orch_with_book(tmp_path, monkeypatch):
    import src.integrations.contacts as contacts_mod
    book = ContactBook(db_path=str(tmp_path / "c.db"))
    monkeypatch.setattr(contacts_mod, "contact_book", book)
    from src.core.orchestrator import CortexOrchestrator
    return CortexOrchestrator.__new__(CortexOrchestrator), book


def test_handle_add_full(tmp_path, monkeypatch):
    o, book = _orch_with_book(tmp_path, monkeypatch)
    reply, mode = o._handle_contact_add("nome: Lucas, email: lucas@x.com, notas: amigo")
    assert mode == "CONTACT_ADD"
    c = book.lookup("Lucas")
    assert c.email == "lucas@x.com" and c.notes == "amigo"


def test_handle_add_name_only_notes_optional_email(tmp_path, monkeypatch):
    o, book = _orch_with_book(tmp_path, monkeypatch)
    reply, mode = o._handle_contact_add("nome: Bia, telefone: 119999")
    assert mode == "CONTACT_ADD"
    assert "sem email" in reply
    c = book.lookup("Bia")
    assert c.email == "" and c.phone == "119999"


def test_handle_add_upsert_updates(tmp_path, monkeypatch):
    o, book = _orch_with_book(tmp_path, monkeypatch)
    o._handle_contact_add("nome: Ana, email: ana@old.com")
    reply, mode = o._handle_contact_add("nome: Ana, email: ana@new.com")
    assert "atualizado" in reply
    assert book.lookup("Ana").email == "ana@new.com"


def test_handle_add_no_name_asks(tmp_path, monkeypatch):
    o, _ = _orch_with_book(tmp_path, monkeypatch)
    reply, mode = o._handle_contact_add("")
    assert mode == "CONTACT_ADD_HELP"
    assert "nome" in reply.lower()
