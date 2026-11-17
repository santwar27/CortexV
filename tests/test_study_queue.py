"""
StudyQueue: intake filter + duplicate detection + basic CRUD.

Covers the two ValueError tags (TRANSIENT_TOPIC, DUPLICATE_TOPIC) and
the '!!' force flag that bypasses both.
"""
import pytest

from src.core.study_queue import StudyTopicRejected


class TestIntakeFilter:
    """Transient topics should be rejected with a friendly tag."""

    def test_rejects_current_affairs(self, tmp_study_queue):
        with pytest.raises(ValueError, match="TRANSIENT_TOPIC"):
            tmp_study_queue.add("current affairs in britain")

    def test_rejects_weather(self, tmp_study_queue):
        with pytest.raises(ValueError, match="TRANSIENT_TOPIC"):
            tmp_study_queue.add("weather in london")

    def test_rejects_pricing(self, tmp_study_queue):
        with pytest.raises(ValueError, match="TRANSIENT_TOPIC"):
            tmp_study_queue.add("preço do bitcoin")

    def test_rejects_scores(self, tmp_study_queue):
        with pytest.raises(ValueError, match="TRANSIENT_TOPIC"):
            tmp_study_queue.add("resultado de Flamengo")

    def test_rejects_trending(self, tmp_study_queue):
        with pytest.raises(ValueError, match="TRANSIENT_TOPIC"):
            tmp_study_queue.add("trending on twitter")

    def test_accepts_stable_topic(self, tmp_study_queue):
        entry = tmp_study_queue.add("RAG vs fine-tuning")
        assert entry["topic"] == "RAG vs fine-tuning"
        assert entry["status"] == "pending"

    def test_force_flag_bypasses_transient(self, tmp_study_queue):
        entry = tmp_study_queue.add("current affairs in britain !!")
        assert entry["topic"] == "current affairs in britain"
        assert entry["status"] == "pending"


class TestDuplicateDetection:
    """Fuzzy duplicate detection on add()."""

    @pytest.mark.parametrize("topic", ["RAG vs fine-tuning", "RAG vs finetuning"])
    def test_rejection_carries_existing_topic_and_status(self, tmp_study_queue, topic):
        existing = tmp_study_queue.add("RAG vs fine-tuning")
        tmp_study_queue.consume_all()
        with pytest.raises(StudyTopicRejected) as rejected:
            tmp_study_queue.add(topic)
        assert rejected.value.reason == "DUPLICATE"
        assert rejected.value.topic == topic
        assert rejected.value.existing == existing["topic"]
        assert rejected.value.status == "consumed"
        assert len(tmp_study_queue._read_all()) == 1

    def test_exact_match_rejected(self, tmp_study_queue):
        tmp_study_queue.add("RAG vs fine-tuning")
        with pytest.raises(ValueError, match="DUPLICATE_TOPIC"):
            tmp_study_queue.add("RAG vs fine-tuning")

    def test_case_insensitive(self, tmp_study_queue):
        tmp_study_queue.add("transformer architecture")
        with pytest.raises(ValueError, match="DUPLICATE_TOPIC"):
            tmp_study_queue.add("Transformer Architecture")

    def test_fuzzy_hyphen_variant(self, tmp_study_queue):
        tmp_study_queue.add("RAG vs fine-tuning")
        with pytest.raises(ValueError, match="DUPLICATE_TOPIC"):
            tmp_study_queue.add("RAG vs finetuning")

    def test_genuinely_different_topic_accepted(self, tmp_study_queue):
        tmp_study_queue.add("transformer architecture")
        # Distinct subject — should be accepted.
        entry = tmp_study_queue.add("PyTorch attention")
        assert entry["status"] == "pending"

    def test_force_flag_bypasses_dup(self, tmp_study_queue):
        tmp_study_queue.add("RAG vs fine-tuning")
        entry = tmp_study_queue.add("RAG vs fine-tuning !!")
        assert entry["topic"] == "RAG vs fine-tuning"

    def test_dup_check_includes_consumed(self, tmp_study_queue):
        """A consumed (past) topic should still block a re-add unless forced."""
        tmp_study_queue.add("quantum computing basics")
        tmp_study_queue.consume_all()  # marks all as consumed
        with pytest.raises(ValueError, match="DUPLICATE_TOPIC"):
            tmp_study_queue.add("quantum computing basics")
        # But force should let it through for re-study.
        entry = tmp_study_queue.add("quantum computing basics !!")
        assert entry["status"] == "pending"


class TestCRUD:
    """Basic queue invariants."""

    def test_empty_input_rejected(self, tmp_study_queue):
        with pytest.raises(ValueError, match="empty"):
            tmp_study_queue.add("")

    def test_whitespace_only_rejected(self, tmp_study_queue):
        with pytest.raises(ValueError, match="empty"):
            tmp_study_queue.add("   ")

    def test_force_flag_alone_rejected(self, tmp_study_queue):
        with pytest.raises(ValueError, match="empty"):
            tmp_study_queue.add("!!")

    def test_list_pending(self, tmp_study_queue):
        tmp_study_queue.add("topic one")
        tmp_study_queue.add("topic two")
        pending = tmp_study_queue.list_pending()
        assert len(pending) == 2
        assert {p["topic"] for p in pending} == {"topic one", "topic two"}

    def test_consume_all_transitions_status(self, tmp_study_queue):
        tmp_study_queue.add("topic one")
        consumed = tmp_study_queue.consume_all()
        assert len(consumed) == 1
        assert tmp_study_queue.list_pending() == []

    def test_consume_idempotent(self, tmp_study_queue):
        tmp_study_queue.add("topic one")
        tmp_study_queue.consume_all()
        # Second consume should return nothing — no pending left.
        assert tmp_study_queue.consume_all() == []



def test_a_topic_added_while_the_nightly_consumes_is_kept(tmp_path):
    """consume_all read the file, decided, and replaced it whole: a topic the
    API appended in between got "Anotado" and vanished with the replace
    (s88, Codex lote 14). Every writer now holds one lock between processes.
    The second instance stands in for the API process."""
    import threading
    from src.core.study_queue import StudyQueue
    p = str(tmp_path / "q.jsonl")
    nightly, api = StudyQueue(path=p), StudyQueue(path=p)
    api.add("SQLite WAL internals")
    read_done, resume = threading.Event(), threading.Event()
    real_read = nightly._read_all

    def paused_read():
        out = real_read()
        read_done.set()
        resume.wait(5)
        return out
    nightly._read_all = paused_read
    worker = threading.Thread(target=nightly.consume_all)
    worker.start()
    assert read_done.wait(5)
    added = threading.Thread(target=lambda: api.add("história do jazz modal"))
    added.start()
    added.join(0.3)
    assert added.is_alive(), "the add must wait for the consume to finish"
    resume.set()
    worker.join(5)
    added.join(5)
    topics = {e["topic"]: e["status"] for e in StudyQueue(path=p)._read_all()}
    assert topics == {"SQLite WAL internals": "consumed",
                      "história do jazz modal": "pending"}
