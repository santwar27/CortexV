"""
ContactBook — local name→email ledger.

Pedro asked for this after the s20 send_email bug: "manda email pra lara"
should resolve without him typing the full address each time. The tracker
is fail-soft (DB error must never crash dispatch) and lives outside the
project tree (~/.cortex/contacts.db), so the unit tests use an in-memory
or tmp-path DB to stay isolated.
"""
import pytest

from src.integrations.contacts import Contact, ContactBook


@pytest.fixture
def book(tmp_path):
    """Fresh ContactBook on a tmp SQLite file — never touches ~/.cortex/."""
    return ContactBook(db_path=str(tmp_path / "contacts.db"))


class TestAdd:
    def test_adds_a_contact(self, book):
        assert book.add("lara", "lara@example.com") is True
        c = book.lookup("lara")
        assert c is not None
        assert c.name == "lara"
        assert c.email == "lara@example.com"

    def test_duplicate_name_rejected(self, book):
        # The user-facing key (name) is UNIQUE — second insert should
        # fail (caller must call update() instead).
        assert book.add("lara", "a@b.com") is True
        assert book.add("lara", "different@b.com") is False
        # Original is preserved.
        assert book.lookup("lara").email == "a@b.com"

    def test_a_duplicate_does_not_hold_the_write_lock(self, tmp_path):
        """Server and terminal each hold a connection to the same file. The
        IntegrityError skipped commit(): the transaction stayed open and the
        other process's writes timed out as "locked" (s88, Codex lote 2)."""
        db = str(tmp_path / "contacts.db")
        server, terminal = ContactBook(db_path=db), ContactBook(db_path=db)
        assert server.add("Ana", "ana@x.com") is True
        assert terminal.add("Ana", "outra@x.com") is False      # the race
        assert terminal._connect().in_transaction is False
        server._connect().execute("PRAGMA busy_timeout = 200")
        assert server.add("Bruno", "bruno@x.com") is True
        assert server.update("Ana", email="ana2@x.com") is True

    def test_case_insensitive_dedup(self, book):
        # COLLATE NOCASE on the unique index — "Lara"/"lara"/"LARA"
        # are the same row.
        assert book.add("Lara", "a@b.com") is True
        assert book.add("lara", "x@y.com") is False
        assert book.add("LARA", "x@y.com") is False

    @pytest.mark.parametrize("name,email", [
        ("", "a@b.com"),
        ("  ", "a@b.com"),
    ])
    def test_rejects_blank_name(self, book, name, email):
        # Name is required; a blank name is always rejected.
        assert book.add(name, email) is False

    @pytest.mark.parametrize("email", ["", "  "])
    def test_allows_blank_email(self, book, email):
        # Email is now OPTIONAL (s55): a name-only contact is valid (voice
        # flow, where email dictation is unreliable). Stored as "".
        assert book.add("Lara", email) is True
        assert book.lookup("Lara").email == ""


class TestUpdate:
    def test_updates_email(self, book):
        book.add("lara", "old@x.com")
        assert book.update("lara", email="new@x.com") is True
        assert book.lookup("lara").email == "new@x.com"

    def test_updates_phone_and_notes(self, book):
        book.add("lara", "a@b.com")
        assert book.update("lara", phone="+44 7000 000 000",
                           notes="moved to London") is True
        c = book.lookup("lara")
        assert c.phone == "+44 7000 000 000"
        assert c.notes == "moved to London"

    def test_update_missing_returns_false(self, book):
        assert book.update("noone", email="x@y.com") is False

    def test_no_fields_provided_returns_false(self, book):
        book.add("lara", "a@b.com")
        # Caller passed no kwargs — nothing to do, signal that explicitly.
        assert book.update("lara") is False


class TestDelete:
    def test_removes_contact(self, book):
        book.add("lara", "a@b.com")
        assert book.delete("lara") is True
        assert book.lookup("lara") is None

    def test_delete_missing_returns_false(self, book):
        assert book.delete("ghost") is False


class TestLookupFuzzy:
    def test_exact_match_wins(self, book):
        book.add("lara", "lara@example.com")
        book.add("fred", "fred@x.com")
        # Exact match always comes back as a single-item list — the
        # caller doesn't have to disambiguate.
        results = book.lookup_fuzzy("lara")
        assert len(results) == 1
        assert results[0].email == "lara@example.com"

    def test_typo_resolves(self, book):
        book.add("lara", "lara@example.com")
        results = book.lookup_fuzzy("laraa")  # one extra 'a'
        assert len(results) >= 1
        assert results[0].name == "lara"

    def test_case_insensitive_fuzzy(self, book):
        book.add("Lara", "a@b.com")
        results = book.lookup_fuzzy("larA")
        assert len(results) >= 1
        assert results[0].name == "Lara"

    def test_no_match_returns_empty(self, book):
        book.add("lara", "a@b.com")
        results = book.lookup_fuzzy("ghost-name-that-does-not-exist")
        assert results == []

    def test_multiple_close_names_all_returned(self, book):
        # Disambiguation case — orchestrator can ask the user which one.
        book.add("anna", "anna@x.com")
        book.add("ana", "ana@x.com")
        book.add("hannah", "hannah@x.com")
        results = book.lookup_fuzzy("anna", max_results=3, cutoff=0.5)
        names = [c.name for c in results]
        # "anna" exact wins; "ana" close behind.
        assert "anna" in names
        assert len(results) >= 1

    def test_empty_query_returns_empty(self, book):
        book.add("lara", "a@b.com")
        assert book.lookup_fuzzy("") == []
        assert book.lookup_fuzzy("   ") == []


class TestListAll:
    def test_returns_all_alphabetical(self, book):
        book.add("zoe", "z@x.com")
        book.add("anna", "a@x.com")
        book.add("mike", "m@x.com")
        names = [c.name for c in book.list_all()]
        assert names == ["anna", "mike", "zoe"]

    def test_empty_book_returns_empty_list(self, book):
        assert book.list_all() == []


class TestFailSoft:
    """A broken DB path must NEVER raise into the caller's request flow —
    every method degrades to a safe empty/False result and logs."""

    def test_unwritable_path_does_not_crash(self):
        # /dev/null isn't a directory — _connect() will fail; methods
        # should silently return None/False/[].
        broken = ContactBook(db_path="/dev/null/contacts.db")
        assert broken.add("x", "y@z.com") is False
        assert broken.lookup("x") is None
        assert broken.lookup_fuzzy("x") == []
        assert broken.update("x", email="a@b.com") is False
        assert broken.delete("x") is False
        assert broken.list_all() == []
