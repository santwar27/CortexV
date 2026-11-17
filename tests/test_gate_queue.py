"""
GateQueue — append-only JSONL event log for background S3 actions.

Mirrors the test style of audit / study_queue: an injectable path (here a
tmp file) so no real ~/.cortex write happens, and the contract the
background gate + the gate_review CLI rely on. The interesting behaviour
is the FOLD: state is derived by replaying events per id, not stored.
"""
import json

import pytest

from src.core.gate_queue import GateQueue


@pytest.fixture
def queue(tmp_path):
    return GateQueue(str(tmp_path / "gate_queue.jsonl"))


class TestEnqueueAndPending:
    def test_enqueue_returns_id_and_shows_pending(self, queue):
        gid = queue.enqueue("send_email", {"to": "a@b", "subject": "s"})
        assert isinstance(gid, str) and len(gid) == 12
        pend = queue.pending()
        assert len(pend) == 1
        assert pend[0]["id"] == gid
        assert pend[0]["type"] == "send_email"
        assert pend[0]["args"] == {"to": "a@b", "subject": "s"}
        assert pend[0]["origin"] == "background"

    def test_args_are_copied_not_referenced(self, queue):
        args = {"to": "a@b"}
        queue.enqueue("send_email", args)
        args["to"] = "mutated"
        assert queue.pending()[0]["args"] == {"to": "a@b"}

    def test_custom_origin_and_reason(self, queue):
        queue.enqueue("create_event", {}, origin="studier", reason="weekly digest")
        e = queue.pending()[0]
        assert e["origin"] == "studier"
        assert e["reason"] == "weekly digest"

    def test_pending_oldest_first(self, queue):
        a = queue.enqueue("send_email", {})
        b = queue.enqueue("create_event", {})
        ids = [e["id"] for e in queue.pending()]
        assert ids == [a, b]

    def test_empty_queue_pending_is_empty(self, queue):
        assert queue.pending() == []

    def test_file_created_owner_only(self, queue, tmp_path):
        # M6-A6: S3 args live here — 0600 from birth.
        queue.enqueue("send_email", {"to": "a@b"})
        mode = (tmp_path / "gate_queue.jsonl").stat().st_mode & 0o777
        assert mode == 0o600

    def test_existing_file_tightened_on_init(self, tmp_path):
        path = tmp_path / "gate_queue.jsonl"
        path.write_text("")
        path.chmod(0o644)
        GateQueue(str(path))
        assert path.stat().st_mode & 0o777 == 0o600

    def test_args_redacted_on_write(self, queue, tmp_path):
        # M6-A6: the queue is plaintext JSONL AND the source of truth for
        # execution — redaction happens at enqueue, so what the reviewer
        # sees, what persists and what later runs are the same.
        queue.enqueue("send_email", {
            "to": "a@b",
            "body": "minha senha do banco é 1234",
            "api_token": "sk-999",
        })
        raw = (tmp_path / "gate_queue.jsonl").read_text()
        assert "1234" not in raw and "sk-999" not in raw
        args = queue.pending()[0]["args"]
        assert args["to"] == "a@b"
        assert args["api_token"] == "[REDACTED]"
        assert "[REDACTED]" in args["body"]


class TestFold:
    def test_approved_removes_from_pending(self, queue):
        gid = queue.enqueue("send_email", {})
        assert queue.resolve(gid, "approved") is True
        assert queue.pending() == []

    def test_rejected_removes_from_pending(self, queue):
        gid = queue.enqueue("send_email", {})
        assert queue.resolve(gid, "rejected") is True
        assert queue.pending() == []

    def test_executed_is_not_pending_after_approve(self, queue):
        gid = queue.enqueue("send_email", {})
        queue.resolve(gid, "approved")
        queue.mark_executed(gid, ok=True, detail="sent")
        assert queue.pending() == []

    def test_one_pending_one_decided(self, queue):
        a = queue.enqueue("send_email", {})
        b = queue.enqueue("create_event", {})
        queue.resolve(a, "rejected")
        pend = queue.pending()
        assert [e["id"] for e in pend] == [b]


class TestResolveGuards:
    def test_invalid_decision_raises(self, queue):
        gid = queue.enqueue("send_email", {})
        with pytest.raises(ValueError):
            queue.resolve(gid, "maybe")

    def test_resolve_unknown_id_is_false(self, queue):
        assert queue.resolve("deadbeef0000", "approved") is False

    def test_double_resolve_is_idempotent(self, queue):
        # The second approve must NOT succeed — guards against running an
        # action twice (fail-closed).
        gid = queue.enqueue("send_email", {})
        assert queue.resolve(gid, "approved") is True
        assert queue.resolve(gid, "approved") is False
        assert queue.resolve(gid, "rejected") is False


class TestStatusAndGet:
    def test_status_lifecycle(self, queue):
        gid = queue.enqueue("send_email", {})
        assert queue.status(gid) == "queued"
        queue.resolve(gid, "approved")
        assert queue.status(gid) == "approved"
        queue.mark_executed(gid, ok=True)
        assert queue.status(gid) == "executed"

    def test_status_failed(self, queue):
        gid = queue.enqueue("send_email", {})
        queue.resolve(gid, "approved")
        queue.mark_executed(gid, ok=False, detail="boom")
        assert queue.status(gid) == "failed"

    def test_status_rejected(self, queue):
        gid = queue.enqueue("send_email", {})
        queue.resolve(gid, "rejected")
        assert queue.status(gid) == "rejected"

    def test_status_unknown_is_none(self, queue):
        assert queue.status("nope") is None

    def test_get_returns_queued_record(self, queue):
        gid = queue.enqueue("create_reminder", {"title": "x"})
        rec = queue.get(gid)
        assert rec["type"] == "create_reminder"
        assert rec["args"] == {"title": "x"}

    def test_get_unknown_is_none(self, queue):
        assert queue.get("nope") is None


class TestPersistenceAndRobustness:
    def test_jsonl_round_trip_across_instances(self, tmp_path):
        path = str(tmp_path / "q.jsonl")
        q1 = GateQueue(path)
        gid = q1.enqueue("send_email", {"to": "x"})
        # A fresh instance over the same file sees the same state.
        q2 = GateQueue(path)
        assert [e["id"] for e in q2.pending()] == [gid]
        q2.resolve(gid, "approved")
        assert GateQueue(path).pending() == []

    def test_one_line_per_event(self, queue):
        gid = queue.enqueue("send_email", {})
        queue.resolve(gid, "approved")
        queue.mark_executed(gid, ok=True)
        with open(queue.path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 3
        assert [json.loads(ln)["event"] for ln in lines] == [
            "queued", "approved", "executed",
        ]

    def test_malformed_line_is_skipped(self, queue):
        gid = queue.enqueue("send_email", {})
        with open(queue.path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")
        # Still reads the valid line, ignores the junk.
        assert [e["id"] for e in queue.pending()] == [gid]

    def test_env_override(self, tmp_path, monkeypatch):
        path = str(tmp_path / "from_env.jsonl")
        monkeypatch.setenv("CORTEX_GATE_QUEUE", path)
        q = GateQueue()  # no explicit path → reads env
        assert q.path == path

    def test_explicit_path_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_GATE_QUEUE", str(tmp_path / "env.jsonl"))
        explicit = str(tmp_path / "explicit.jsonl")
        assert GateQueue(explicit).path == explicit


def test_two_queues_approving_at_once_run_it_once(tmp_path):
    """Two terminals approved the same id: both read it as pending before
    either wrote, both got True and both ran it — the lock was per instance,
    so per process (s88, Codex lote 10). A barrier right after the read forces
    the interleaving; with the decision lock the second sees it decided."""
    import threading
    from src.core.gate_queue import GateQueue
    path = str(tmp_path / "q.jsonl")
    gid = GateQueue(path).enqueue("send_email", {"to": "a@b.com"})
    a, b = GateQueue(path), GateQueue(path)
    barrier = threading.Barrier(2, timeout=1)
    for q in (a, b):
        read = q._pending_ids

        def after_reading(read=read):
            ids = read()
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass  # the other one is waiting on the lock — as it should
            return ids
        q._pending_ids = after_reading
    results = []
    threads = [threading.Thread(target=lambda q=q: results.append(q.resolve(gid, "approved")))
               for q in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False, True]
