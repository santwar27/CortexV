"""Morning briefing retries share a conversation only within the local day."""
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src import daily_driver as dd
from src.ui import chat_store as cs


NOW = datetime(2026, 10, 23, 8, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def local_timezone(monkeypatch):
    # Exercise UTC/local date differences and the 25-hour autumn day reliably.
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "Europe/London")
            time.tzset()
            yield
    finally:
        time.tzset()


@pytest.fixture
def store(tmp_path, monkeypatch):
    ledger = cs.ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cs, "store", ledger)
    yield ledger
    if ledger._conn is not None:
        ledger._conn.close()


def title_for(now):
    return "☀️ Bom dia · " + now.astimezone().strftime("%a %d/%m")


def set_created_at(store, cid, instant):
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE conversations SET created_ts = ? WHERE id = ?",
                     (instant.astimezone(timezone.utc).isoformat(), cid))


@pytest.mark.parametrize("delay", [timedelta(), timedelta(hours=2)])
def test_same_day_posts_append_to_one_conversation(store, delay):
    first = dd.post_to_chat("First briefing", NOW)
    second = dd.post_to_chat("Updated briefing", NOW + delay)

    assert first is not None
    assert second == first
    conversations = store.list_conversations()
    assert len(conversations) == 1
    assert conversations[0]["title"] == title_for(NOW)
    assert conversations[0]["turn_count"] == 2
    turns = store.get_turns(first)
    assert [turn["reply"] for turn in turns] == [
        "First briefing", "Updated briefing",
    ]
    for turn in turns:
        assert turn["user"] == "o que eu tenho pra hoje?"
        assert turn["status_mode"] == "DAILY"
        assert turn["source"] == "daily"


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("created", [NOW.replace(year=2025),
                                     NOW - timedelta(days=1),
                                     NOW + timedelta(days=1)])
def test_same_title_from_another_date_is_preserved(store, renamed, created):
    old = store.create_conversation(
        title="Pedro's conversation" if renamed else title_for(NOW))
    assert store.append_turn(old["id"], "Old question", "Old reply") is not None
    if renamed:
        assert store.rename_conversation(old["id"], title_for(NOW))
    set_created_at(store, old["id"], created)
    old_turns = store.get_turns(old["id"])

    today = dd.post_to_chat("Today's briefing", NOW)

    assert today is not None and today != old["id"]
    assert len(store.list_conversations()) == 2
    assert store.get_turns(old["id"]) == old_turns
    assert store.get_turns(today)[0]["reply"] == "Today's briefing"


def test_simultaneous_posts_share_one_conversation(store):
    start = threading.Barrier(2)

    def post(text):
        start.wait(timeout=5)
        return dd.post_to_chat(text, NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(post, "First concurrent briefing")
        second = pool.submit(post, "Second concurrent briefing")
        ids = [first.result(timeout=10), second.result(timeout=10)]

    assert ids[0] is not None and ids[1] == ids[0]
    assert len(store.list_conversations()) == 1
    assert {turn["reply"] for turn in store.get_turns(ids[0])} == {
        "First concurrent briefing", "Second concurrent briefing",
    }


def test_independent_stores_get_or_create_atomically(store):
    other = cs.ChatStore(db_path=store.db_path)
    # Establish independent connections before releasing both writers.
    assert store._connect() is not None
    assert other._connect() is not None
    start = threading.Barrier(2)

    def get_conversation(ledger):
        start.wait(timeout=5)
        return ledger.conversation_titled_on(title_for(NOW), NOW)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(get_conversation, store)
            second = pool.submit(get_conversation, other)
            ids = [first.result(timeout=10), second.result(timeout=10)]
        assert ids[0] is not None and ids[1] == ids[0]
        assert len(store.list_conversations()) == 1
    finally:
        other._conn.close()


def test_deleted_conversation_is_recreated(store):
    first = dd.post_to_chat("First briefing", NOW)
    assert first is not None
    assert store.delete_conversation(first)

    second = dd.post_to_chat("Briefing after deletion", NOW)

    assert second is not None and second != first
    assert [c["id"] for c in store.list_conversations()] == [second]
    assert store.get_turns(first) == []
    assert store.get_turns(second)[0]["reply"] == "Briefing after deletion"


@pytest.mark.parametrize("raises", [False, True])
def test_failed_append_returns_none_and_retry_reuses_empty_conversation(
        store, monkeypatch, raises):
    def failed_append(*args, **kwargs):
        if raises:
            raise RuntimeError("append unavailable")
        return None

    with monkeypatch.context() as patch:
        patch.setattr(store, "append_turn", failed_append)
        assert dd.post_to_chat("Unsaved briefing", NOW) is None
    empty = store.list_conversations()
    assert len(empty) == 1 and empty[0]["turn_count"] == 0

    cid = dd.post_to_chat("Retry briefing", NOW)

    assert cid == empty[0]["id"]
    assert len(store.list_conversations()) == 1
    assert [turn["reply"] for turn in store.get_turns(cid)] == ["Retry briefing"]


@pytest.mark.parametrize("now,created,reuse", [
    # BST: local midnight is 23:00 UTC on the preceding date.
    (NOW, datetime(2026, 10, 22, 23, tzinfo=timezone.utc), True),
    (NOW, datetime(2026, 10, 22, 22, 59, 59, tzinfo=timezone.utc), False),
    (NOW, datetime(2026, 10, 23, 23, tzinfo=timezone.utc), False),
    # The autumn clock change makes this local day 25 hours long.
    (datetime(2026, 10, 25, 8, tzinfo=timezone.utc),
     datetime(2026, 10, 24, 23, tzinfo=timezone.utc), True),
    (datetime(2026, 10, 25, 8, tzinfo=timezone.utc),
     datetime(2026, 10, 25, 23, 30, tzinfo=timezone.utc), True),
    (datetime(2026, 10, 25, 8, tzinfo=timezone.utc),
     datetime(2026, 10, 26, 0, tzinfo=timezone.utc), False),
    # Use the system's local date, rather than the date in now's input zone.
    (datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc),
     datetime(2026, 10, 23, 8, tzinfo=timezone.utc), True),
])
def test_reuse_is_limited_to_the_local_calendar_day(store, now, created, reuse):
    existing = store.create_conversation(title=title_for(now))
    set_created_at(store, existing["id"], created)

    cid = dd.post_to_chat("Briefing", now)

    assert cid is not None
    assert (cid == existing["id"]) is reuse
    assert len(store.list_conversations()) == (1 if reuse else 2)


def test_database_failure_is_fail_soft_and_releases_transaction(store):
    conn = store._connect()
    conn.execute("""CREATE TRIGGER reject_conversation BEFORE INSERT ON conversations
                    BEGIN SELECT RAISE(ABORT, 'storage unavailable'); END""")
    conn.commit()

    assert dd.post_to_chat("Unsaved briefing", NOW) is None
    assert store.list_conversations() == []
    assert not conn.in_transaction

    conn.execute("DROP TRIGGER reject_conversation")
    conn.commit()
    assert dd.post_to_chat("Retry briefing", NOW) is not None
    assert len(store.list_conversations()) == 1
