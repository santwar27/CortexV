"""Stable block windows across durable threads and all engine entry points."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.core import engine as mod
from src.core.history_window import HistoryWindow
from src.core.staging_mixin import CODE_TESTS_OUTPUT
from src.ui import routes
from src.ui.chat_store import ChatStore, VOICE_LABEL, daily_title


SIZES = [0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3]
STARTS = [0, 0, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9]


def pairs(start, end):
    return [message for i in range(start, end) for message in (
        {"role": "user", "content": f"Question {i}."},
        {"role": "assistant", "content": f"Answer {i}."})]


def assert_alternating(messages):
    assert [m["role"] for m in messages] == [
        "user" if i % 2 == 0 else "assistant" for i in range(len(messages))]


def test_exact_rule_and_boundaries():
    window = HistoryWindow()
    assert (window.min_pairs, window.max_pairs, window.step) == (2, 4, 3)
    starts = [window.start(n) for n in range(13)]
    assert starts == STARTS
    assert [n - starts[n] for n in range(13)] == SIZES
    assert [n for n in range(1, 13) if starts[n] != starts[n - 1]] == [5, 8, 11]


@pytest.mark.parametrize("budget", [None, False, True, "10", -10, 0, 1, 2, 3,
                                         4, 5, 6, 8, 10, 12, 20])
def test_invalid_and_small_budgets_are_bounded(budget):
    window = HistoryWindow(budget)
    assert window.start(0) == 0
    for n in range(1, 30):
        assert 1 <= n - window.start(n) <= window.max_pairs
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        assert (window.min_pairs, window.max_pairs) == (2, 4)
    elif budget < 6:
        assert (window.min_pairs, window.max_pairs) == (1, 1)


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(routes, "chat_store", store)
    yield store
    if store._conn is not None:
        store._conn.close()


@pytest.mark.parametrize("daily", [False, True])
def test_ui_absolute_windows_ignore_half_turns_and_extend(store, daily):
    cid = store.conversation_titled(daily_title(VOICE_LABEL))
    read = (lambda: routes._daily_history_msgs(VOICE_LABEL, 5)) if daily else (
        lambda: routes._conversation_history_msgs(cid, 5))
    assert read() == []
    previous = []
    for n in range(1, 13):
        # Neither raw row count nor the recent raw-row tail is the position.
        store.append_turn(cid, "no reply", "")
        store.append_turn(cid, "whitespace reply", " \n ")
        store.append_turn(cid, "", "no user")
        store.append_turn(cid, f"  Question {n - 1}.  ", f"  Answer {n - 1}.  ")
        for _ in range(2):
            store.append_turn(cid, "unfinished tail", "")
        messages = read()
        assert messages == pairs(STARTS[n], n)
        assert_alternating(messages)
        assert read() == messages  # A read cannot advance the anchor.
        if STARTS[n] == STARTS[n - 1]:
            assert messages[:len(previous)] == previous
            if previous:
                assert messages[0] == previous[0]
        previous = messages


def test_ui_code_tests_cut_is_preserved_in_block_window(store):
    cid = store.create_conversation()["id"]
    for i in range(5):
        store.append_turn(cid, "unfinished tests", "", status_mode="CODE_TESTS_FAILED")
        store.append_turn(cid, f"Question {i}.",
                          f"Answer {i}." + CODE_TESTS_OUTPUT["pt"] + "PRIVATE",
                          status_mode="CODE_TESTS_PASSED")
    assert routes._conversation_history_msgs(cid, 5) == pairs(3, 5)
    assert "PRIVATE" in store.get_turns(cid)[-1]["reply"]


@pytest.mark.parametrize("failure", ["read", "build"])
def test_ui_fail_soft_includes_message_building(store, monkeypatch, failure):
    cid = store.create_conversation()["id"]
    if failure == "read":
        monkeypatch.setattr(store, "get_turns", Mock(side_effect=RuntimeError("read")))
    else:
        monkeypatch.setattr(store, "get_turns", lambda *_: [{"user": object()}])
    assert routes._conversation_history_msgs(cid, 5) == []


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("CORTEX_MODEL_PROFILE", "8b")
    memory = SimpleNamespace(
        get_user_profile=Mock(return_value=""),
        retrieve_relevant_context=Mock(return_value=""),
        store_interaction=Mock(),
    )
    with patch.object(mod, "CortexMemory", return_value=memory):
        return mod.OllamaEngine()


@pytest.mark.parametrize("path", ["chat", "stream", "tools"])
def test_twelve_engine_turns_send_the_ui_window(engine, store, monkeypatch, path):
    cid = store.create_conversation()["id"]
    seen = []

    def fake_chat(**kwargs):
        n = len(seen)
        seen.append(deepcopy(kwargs["messages"]))
        response = {"message": {"content": f"Answer {n}."}}
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(mod.ollama, "chat", fake_chat)
    for n in range(12):
        ui_window = routes._conversation_history_msgs(cid, 5)
        question = f"Question {n}."
        if path == "stream":
            reply = "".join(engine.chat_stream(question))
        elif path == "tools":
            reply, tool = engine.chat_with_tools(question, [], Mock())
            assert tool is None
        else:
            reply = engine.chat(question)
        assert reply == f"Answer {n}."
        payload = seen[-1]
        assert payload[0]["role"] == "system"
        assert payload[1:-1] == ui_window == pairs(STARTS[n], n)
        assert payload[-1]["role"] == "user"
        assert payload[-1]["content"].startswith(question)
        assert_alternating(payload[1:])
        assert_alternating(engine.history)
        store.append_turn(cid, question, reply)

    # Once the first exchange is history, compare the actual first message.
    # Turn zero has only an in-flight user with the existing clock suffix.
    assert [n for n in range(2, 12) if seen[n][1] != seen[n - 1][1]] == [5, 8, 11]
    assert all(payload[0] == seen[0][0] for payload in seen)


@pytest.mark.parametrize("budget", [1, 4, 5, 6, 10, 12, 20])
def test_ui_window_is_not_pruned_again(engine, store, budget):
    engine.max_history = budget
    cid = store.create_conversation()["id"]
    for n in range(1, 25):
        store.append_turn(cid, f"Question {n - 1}.", f"Answer {n - 1}.")
        ui_window = routes._conversation_history_msgs(cid, max(1, budget // 2))
        current_user = {"role": "user", "content": f"Question {n}."}
        engine.push_history_override(ui_window)
        try:
            engine.history.append(current_user)
            engine._prune_memory()
            assert engine.history == ui_window + [current_user]
        finally:
            engine.pop_history_override()
        assert engine.history == []


def test_finish_no_record_jumps_after_complete_pairs(engine):
    for n in range(1, 13):
        reply = f"Answer {n - 1}."
        assert engine._finish_no_record(f"Question {n - 1}.", reply) == reply
        assert engine.history == pairs(STARTS[n], n)
        assert_alternating(engine.history)
    engine.memory.store_interaction.assert_not_called()


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 12])
def test_prune_preserves_complete_pairs_and_trailing_user(engine, count, dangling):
    tail = [{"role": "user", "content": "Current."}] if dangling else []
    engine.history = pairs(0, count) + tail
    engine._prune_memory()
    start = count - 2 if count > 4 else 0
    assert engine.history == pairs(start, count) + tail
    assert_alternating(engine.history)
    before = deepcopy(engine.history)
    engine._prune_memory()
    assert engine.history == before


@pytest.mark.parametrize("count", range(5))
@pytest.mark.parametrize("trailing_user", [False, True])
def test_prune_is_noop_between_jumps(engine, count, trailing_user):
    tail = [{"role": "user", "content": "Current."}] if trailing_user else []
    engine.history = pairs(0, count) + tail
    original = engine.history
    before = deepcopy(original)
    engine._prune_memory()
    assert engine.history is original
    assert engine.history == before


def test_old_dangling_user_and_other_messages_stay_between_jumps(engine):
    engine.history = [
        {"role": "assistant", "content": "Orphan."},
        {"role": "user", "content": "Withheld reply."},
    ] + pairs(0, 4) + [
        {"role": "user", "content": "Interrupted."},
        {"role": "system", "content": "Interruption."},
        {"role": "assistant", "content": "Not adjacent to a user."},
        {"role": "tool", "content": "Tool result."},
        {"role": "user", "content": "Current."},
    ]
    original = engine.history
    before = deepcopy(original)
    engine._prune_memory()
    assert engine.history is original
    assert engine.history == before


def test_jump_keeps_entire_suffix_from_second_last_complete_exchange(engine):
    between = [
        {"role": "assistant", "content": "Orphan between exchanges."},
        {"role": "user", "content": "Withheld reply."},
        {"role": "system", "content": "Interruption."},
        {"role": "assistant", "content": "Not adjacent to a user."},
    ]
    tail = [
        {"role": "user", "content": "Withheld reply."},
        {"role": "user", "content": "Current."},
        {"role": "tool", "content": "Tool result."},
    ]
    suffix = pairs(3, 4) + between + pairs(4, 5) + tail
    engine.history = [{"role": "assistant", "content": "Leading orphan."}] + (
        pairs(0, 3) + suffix)
    engine._prune_memory()
    assert engine.history == suffix
    assert engine.history[0]["role"] == "user"
    original = engine.history
    engine._prune_memory()
    assert engine.history is original
    assert engine.history == suffix
