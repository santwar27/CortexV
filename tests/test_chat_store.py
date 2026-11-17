"""Tests for the durable chat conversation store (fatia 2, src/ui/chat_store.py).

Covers the CRUD surface, the auto-title-from-first-turn behaviour, chat
ordering, persistence across a fresh store instance (the uvicorn-restart
case the ephemeral ConversationBuffer couldn't survive), and the fail-soft
contract.
"""
import os

import pytest

from src.ui.chat_store import ChatStore


@pytest.fixture
def store(tmp_path):
    return ChatStore(db_path=str(tmp_path / "chat.db"))


class TestConversationCrud:

    def test_create_returns_placeholder_title(self, store):
        c = store.create_conversation()
        assert c["title"] == "Nova conversa"
        assert c["turn_count"] == 0
        assert c["id"] and c["created_ts"] == c["updated_ts"]

    def test_create_with_explicit_title(self, store):
        c = store.create_conversation(title="Planejamento")
        assert c["title"] == "Planejamento"

    def test_get_missing_conversation_is_none(self, store):
        assert store.get_conversation("nope") is None

    def test_rename(self, store):
        c = store.create_conversation()
        assert store.rename_conversation(c["id"], "Novo nome") is True
        assert store.get_conversation(c["id"])["title"] == "Novo nome"

    def test_rename_missing_returns_false(self, store):
        assert store.rename_conversation("ghost", "x") is False

    def test_rename_blank_rejected(self, store):
        c = store.create_conversation(title="mantém")
        assert store.rename_conversation(c["id"], "   ") is False
        assert store.get_conversation(c["id"])["title"] == "mantém"

    def test_delete_removes_conversation_and_turns(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], "oi", "olá")
        assert store.delete_conversation(c["id"]) is True
        assert store.get_conversation(c["id"]) is None
        assert store.get_turns(c["id"]) == []

    def test_delete_missing_returns_false(self, store):
        assert store.delete_conversation("ghost") is False


class TestTurns:

    def test_first_turn_names_the_conversation(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], "qual a capital da França?", "Paris.")
        assert store.get_conversation(c["id"])["title"] == "qual a capital da França?"

    def test_first_turn_title_is_trimmed(self, store):
        c = store.create_conversation()
        long_q = "x" * 200
        store.append_turn(c["id"], long_q, "ok")
        assert len(store.get_conversation(c["id"])["title"]) <= 48

    def test_explicit_title_not_overwritten_by_first_turn(self, store):
        c = store.create_conversation(title="Meu título")
        store.append_turn(c["id"], "primeira pergunta", "resposta")
        assert store.get_conversation(c["id"])["title"] == "Meu título"

    def test_turns_returned_in_chat_order(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], "um", "1")
        store.append_turn(c["id"], "dois", "2")
        store.append_turn(c["id"], "três", "3")
        assert [t["user"] for t in store.get_turns(c["id"])] == ["um", "dois", "três"]

    def test_get_turns_limit_keeps_last_in_order(self, store):
        c = store.create_conversation()
        for i in range(5):
            store.append_turn(c["id"], f"q{i}", f"a{i}")
        last2 = store.get_turns(c["id"], limit=2)
        assert [t["user"] for t in last2] == ["q3", "q4"]

    def test_turn_shape_matches_buffer(self, store):
        c = store.create_conversation()
        t = store.append_turn(
            c["id"], "oi", "olá", status_mode="OFFLINE",
            asked_ts="2026-07-24T00:00:00+00:00",
            usage={"eval_count": 20, "calls": 1}, latency_ms=1234,
        )
        for k in ("id", "ts", "asked_ts", "user", "reply",
                  "status_mode", "source", "usage", "latency_ms"):
            assert k in t
        assert t["usage"] == {"eval_count": 20, "calls": 1}
        assert t["latency_ms"] == 1234
        assert t["status_mode"] == "OFFLINE"

    def test_append_to_missing_conversation_returns_none(self, store):
        assert store.append_turn("ghost", "x", "y") is None

    def test_append_bumps_updated_ts_and_reorders_list(self, store):
        a = store.create_conversation(title="A")
        b = store.create_conversation(title="B")
        # B is newest so far; a turn on A must float A to the top.
        store.append_turn(a["id"], "oi", "olá")
        titles = [c["title"] for c in store.list_conversations()]
        assert titles[0] == "A"


class TestListing:

    def test_list_newest_updated_first_with_counts(self, store):
        c1 = store.create_conversation(title="primeira")
        c2 = store.create_conversation(title="segunda")
        store.append_turn(c1["id"], "a", "1")
        store.append_turn(c1["id"], "b", "2")
        store.append_turn(c2["id"], "c", "3")
        lst = store.list_conversations()
        by_id = {c["id"]: c for c in lst}
        assert by_id[c1["id"]]["turn_count"] == 2
        assert by_id[c2["id"]]["turn_count"] == 1
        assert by_id[c1["id"]]["preview"] == "b"  # last user message

    def test_empty_store_lists_nothing(self, store):
        assert store.list_conversations() == []


class TestPersistence:

    def test_survives_reopen(self, tmp_path):
        """The whole point of fatia 2: history outlives the process."""
        path = str(tmp_path / "chat.db")
        s1 = ChatStore(db_path=path)
        c = s1.create_conversation()
        s1.append_turn(c["id"], "lembra disso", "lembrado")
        # A brand-new instance against the same file = uvicorn restart.
        s2 = ChatStore(db_path=path)
        turns = s2.get_turns(c["id"])
        assert len(turns) == 1
        assert turns[0]["user"] == "lembra disso"
        assert s2.get_conversation(c["id"]) is not None


class TestFailSoft:

    def test_unwritable_path_degrades_quietly(self, tmp_path):
        # Point the store at a path whose parent is a FILE, so connect fails.
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        s = ChatStore(db_path=str(blocker / "nested" / "chat.db"))
        # No exception — every method returns an empty/false/None result.
        assert s.create_conversation() is None
        assert s.list_conversations() == []
        assert s.get_conversation("x") is None
        assert s.get_turns("x") == []
        assert s.append_turn("x", "a", "b") is None
        assert s.rename_conversation("x", "y") is False
        assert s.delete_conversation("x") is False


class TestPopLastTurn:
    """fatia 3 (regenerate): the rejected turn is removed so the new answer
    replaces it, instead of stacking a duplicate — and so the short-term
    history bound for the retry (UI 2c) excludes the discarded answer."""

    def test_pops_and_returns_the_newest_turn(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], user="primeira", reply="r1")
        store.append_turn(c["id"], user="segunda", reply="r2")
        popped = store.pop_last_turn(c["id"])
        assert popped["user"] == "segunda"
        assert popped["reply"] == "r2"
        remaining = store.get_turns(c["id"])
        assert [t["user"] for t in remaining] == ["primeira"]

    def test_pop_on_empty_conversation_is_none(self, store):
        c = store.create_conversation()
        assert store.pop_last_turn(c["id"]) is None

    def test_pop_on_missing_conversation_is_none(self, store):
        assert store.pop_last_turn("nope") is None

    def test_does_not_touch_other_conversations(self, store):
        a = store.create_conversation()
        b = store.create_conversation()
        store.append_turn(a["id"], user="da A", reply="ra")
        store.append_turn(b["id"], user="da B", reply="rb")
        store.pop_last_turn(a["id"])
        assert [t["user"] for t in store.get_turns(b["id"])] == ["da B"]
        assert store.get_turns(a["id"]) == []

    def test_title_survives_popping_the_first_turn(self, store):
        # A regenerated first turn re-asks the SAME question, so the auto-title
        # stays accurate — popping must not reset it to the placeholder.
        c = store.create_conversation()
        store.append_turn(c["id"], user="como faço café", reply="assim")
        assert store.get_conversation(c["id"])["title"] == "como faço café"
        store.pop_last_turn(c["id"])
        assert store.get_conversation(c["id"])["title"] == "como faço café"

    def test_repeated_pop_drains_then_stops(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], user="u1", reply="r1")
        assert store.pop_last_turn(c["id"])["user"] == "u1"
        assert store.pop_last_turn(c["id"]) is None

    def test_append_after_pop_replaces_cleanly(self, store):
        c = store.create_conversation()
        store.append_turn(c["id"], user="pergunta", reply="ruim")
        store.pop_last_turn(c["id"])
        store.append_turn(c["id"], user="pergunta", reply="boa")
        turns = store.get_turns(c["id"])
        assert len(turns) == 1
        assert turns[0]["reply"] == "boa"


class TestTwoProcessesOnOneFile:
    """The server and the terminal each hold a ChatStore on chat.db (s88,
    Codex lote 7). Both races reproduced; both closed by taking the write
    lock (BEGIN IMMEDIATE) before the lookup, so the other process waits."""

    class _Hooked:
        """A connection that runs `hook` right after a matching SELECT."""
        def __init__(self, real, needle, hook):
            self.real, self.needle, self.hook, self.done = real, needle, hook, False

        def execute(self, sql, *a):
            cur = self.real.execute(sql, *a)
            if not self.done and self.needle in sql:
                self.done = True
                self.hook()
            return cur

        def __getattr__(self, name):
            return getattr(self.real, name)

    def test_the_first_turns_of_the_day_share_one_conversation(self, tmp_path, monkeypatch):
        import threading
        import src.ui.chat_store as cs
        db = str(tmp_path / "chat.db")
        terminal, api = cs.ChatStore(db_path=db), cs.ChatStore(db_path=db)
        title = cs.daily_title(cs.TERMINAL_LABEL)
        other = threading.Thread(target=lambda: api.conversation_titled(title))
        real = terminal._connect()
        terminal._conn = self._Hooked(real, "SELECT id FROM conversations WHERE title",
                                      other.start)       # /ask arrives right then
        a = terminal.conversation_titled(title)
        other.join(10)
        assert [c["id"] for c in api.list_conversations()] == [a]

    def test_a_turn_is_never_stored_onto_a_deleted_conversation(self, tmp_path):
        import threading
        import src.ui.chat_store as cs
        db = str(tmp_path / "chat.db")
        terminal, api = cs.ChatStore(db_path=db), cs.ChatStore(db_path=db)
        cid = terminal.create_conversation(title="x")["id"]
        delete = threading.Thread(target=lambda: api.delete_conversation(cid))
        real = terminal._connect()
        def deleted_right_then():
            delete.start()
            delete.join(1.0)    # old code: done by now; fixed: waits for the lock
        terminal._conn = self._Hooked(real, "SELECT title FROM conversations",
                                      deleted_right_then)
        terminal.append_turn(conversation_id=cid, user="oi", reply="ok")
        delete.join(10)
        orphans = api._connect().execute(
            "SELECT COUNT(*) FROM turns WHERE conversation_id NOT IN "
            "(SELECT id FROM conversations)").fetchone()[0]
        assert orphans == 0
