"""
Tests for scripts/add_contact.py — the add/update-contact CLI.

Runs against a temp contacts DB (--db) so nothing touches ~/.cortex.
"""
import importlib.util
from pathlib import Path

from src.integrations.contacts import ContactBook

# Load the script as a module (it lives in scripts/, not a package).
_SPEC = importlib.util.spec_from_file_location(
    "add_contact",
    Path(__file__).resolve().parent.parent / "scripts" / "add_contact.py",
)
add_contact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(add_contact)


def _db(tmp_path):
    return str(tmp_path / "contacts.db")


def test_adds_contact_with_info(tmp_path, capsys):
    db = _db(tmp_path)
    rc = add_contact.main(["--name", "João Silva", "--email", "joao@x.com",
                           "--notes", "colega do trabalho", "--db", db])
    assert rc == 0
    assert "adicionado" in capsys.readouterr().out
    c = ContactBook(db_path=db).lookup("João Silva")
    assert c.email == "joao@x.com" and c.notes == "colega do trabalho"


def test_upsert_updates_existing(tmp_path, capsys):
    db = _db(tmp_path)
    add_contact.main(["--name", "Ana", "--email", "ana@old.com", "--db", db])
    rc = add_contact.main(["--name", "Ana", "--email", "ana@new.com",
                           "--phone", "+44 123", "--db", db])
    assert rc == 0
    assert "atualizado" in capsys.readouterr().out
    c = ContactBook(db_path=db).lookup("Ana")
    assert c.email == "ana@new.com" and c.phone == "+44 123"


def test_list_contacts(tmp_path, capsys):
    db = _db(tmp_path)
    add_contact.main(["--name", "Bob", "--email", "bob@x.com", "--notes", "amigo", "--db", db])
    rc = add_contact.main(["--list", "--db", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Bob" in out and "bob@x.com" in out and "amigo" in out


def test_list_empty(tmp_path, capsys):
    rc = add_contact.main(["--list", "--db", _db(tmp_path)])
    assert rc == 0
    assert "Nenhum contato" in capsys.readouterr().out


def test_requires_name_and_email(tmp_path):
    import pytest
    with pytest.raises(SystemExit):  # argparse error → exit
        add_contact.main(["--name", "Só Nome", "--db", _db(tmp_path)])
