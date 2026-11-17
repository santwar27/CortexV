"""Task 117: retrieved conversations retain their local calendar date."""
import time

import numpy as np
import pytest

from src.core.memory import CortexMemory
from src.memory.palace import Palace


@pytest.fixture
def memory(monkeypatch):
    # Restore libc's timezone as well as the environment, even on failure.
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "UTC0")
            time.tzset()
            palace = Palace(db_path=":memory:", embed_fn=lambda _: np.ones(4, dtype=np.float32))
            try:
                yield CortexMemory(palace=palace)
            finally:
                palace.close()
    finally:
        time.tzset()


def store(memory, text, created_at="2026-05-22T12:00:00+00:00"):
    return memory._palace.add_document(
        tipo="conversation",
        texto=f"User: {text}\nCortex: ASSISTANT_REPLY_MUST_NOT_RETURN",
        embedding=np.ones(4, dtype=np.float32),
        created_at=created_at,
    )


def test_old_turn_has_date_before_user_and_drops_assistant(memory):
    doc_id = store(memory, "me lembre da ida ao lago amanha as 4horas")
    context = memory.retrieve_relevant_context("você lembra disso?")
    assert context == (
        "[Conversa de 22/05/2026] User: me lembre da ida ao lago amanha as 4horas"
    )
    assert "Cortex:" not in context
    assert memory._palace.get_document(doc_id)["access_count"] == 1
    assert "ASSISTANT_REPLY_MUST_NOT_RETURN" in memory._palace.get_document(doc_id)["texto"]


@pytest.mark.parametrize(
    "zone, timestamp, day",
    [
        ("JST-9", "2026-05-22T23:30:00+00:00", "23/05/2026"),
        ("JST-9", "2026-05-22T23:30:00Z", "23/05/2026"),
        ("JST-9", "2026-05-22T23:30:00", "23/05/2026"),
        ("EST5", "2026-05-22T01:30:00+00:00", "21/05/2026"),
        ("Europe/London", "2026-05-22T23:30:00+00:00", "23/05/2026"),
        ("Europe/London", "2026-01-22T23:30:00+00:00", "22/01/2026"),
    ],
)
def test_date_uses_mac_local_timezone(memory, monkeypatch, zone, timestamp, day):
    with monkeypatch.context() as patch:
        patch.setenv("TZ", zone)
        time.tzset()
        store(memory, "I like walking around the lake", timestamp)
        context = memory.retrieve_relevant_context("lake")
        assert context == f"[Conversa de {day}] User: I like walking around the lake"


@pytest.mark.parametrize("timestamp", [None, "", "garbled", "2026-02-30T12:00:00Z", 123, []])
def test_missing_or_bad_date_has_no_label_and_does_not_drop_other_hits(
    memory, monkeypatch, timestamp
):
    bad_id = store(memory, "I like walking around the lake")
    good_id = store(memory, "I like swimming across the lake")
    bad = memory._palace.get_document(bad_id)
    bad["created_at"] = timestamp
    good = memory._palace.get_document(good_id)
    monkeypatch.setattr(memory._palace, "search", lambda *a, **kw: [bad, good])
    assert memory.retrieve_relevant_context("lake", include_knowledge=False) == (
        "User: I like walking around the lake\n---\n"
        "[Conversa de 22/05/2026] User: I like swimming across the lake"
    )
    assert memory._palace.get_document(bad_id)["access_count"] == 1
    assert memory._palace.get_document(good_id)["access_count"] == 1


def test_absent_created_at_key_has_no_label(memory, monkeypatch):
    doc_id = store(memory, "I like walking around the lake")
    hit = memory._palace.get_document(doc_id)
    del hit["created_at"]
    monkeypatch.setattr(memory._palace, "search", lambda *a, **kw: [hit])
    assert memory.retrieve_relevant_context("lake", include_knowledge=False) == (
        "User: I like walking around the lake"
    )


@pytest.mark.parametrize("cap", [400, 65])
def test_dedup_across_days_keeps_original_unlabelled_key(memory, monkeypatch, cap):
    monkeypatch.setattr(memory, "_CHUNK_CHAR_CAP", cap)
    sentence = "I like walking around the lake " * 3
    first = store(memory, sentence, "2026-05-22T12:00:00Z")
    second = store(memory, sentence.upper() + "!", "2026-05-23T12:00:00Z")
    context = memory.retrieve_relevant_context("lake")
    assert context.count("[Conversa de ") == 1
    assert sum(memory._palace.get_document(i)["access_count"] for i in (first, second)) == 1
    # The date must not turn an echo of today's query into a survivor.
    if cap == 400:
        assert memory.retrieve_relevant_context(sentence) == ""


@pytest.mark.parametrize("chunk_cap, total_cap", [(100, 1500), (400, 150)])
def test_date_label_spends_existing_character_caps(memory, monkeypatch, chunk_cap, total_cap):
    monkeypatch.setattr(memory, "_CHUNK_CHAR_CAP", chunk_cap)
    monkeypatch.setattr(memory, "_CONTEXT_CHAR_CAP", total_cap)
    for i in range(3):
        store(memory, f"Lake activity number {i}: " + "walking " * 100)
    context = memory.retrieve_relevant_context("lake", n_results=3)
    assert context.startswith("[Conversa de 22/05/2026] User:")
    assert len(context) <= total_cap + 1  # Existing truncation adds an ellipsis.
    assert all(len(chunk) <= chunk_cap + 1 for chunk in context.split("\n---\n"))
    assert context.endswith("…")
