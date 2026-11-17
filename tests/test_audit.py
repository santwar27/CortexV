"""
Tests for src/core/audit.py — AuditLogger append-only JSONL.

All tests use tmp_path to avoid touching ~/.cortex on disk.
"""
import gzip
import json
import threading
import time

import pytest

from src.core.audit import AuditLogger


@pytest.fixture
def logger(tmp_path):
    return AuditLogger(path=str(tmp_path / "audit.jsonl"))


class TestAuditLogger:

    def test_log_creates_file(self, logger, tmp_path):
        logger.log("abc123", "orchestrator", "request")
        assert (tmp_path / "audit.jsonl").exists()

    def test_log_valid_json_per_line(self, logger, tmp_path):
        logger.log("abc123", "orchestrator", "request", payload={"input": "oi"})
        line = (tmp_path / "audit.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["trace_id"] == "abc123"
        assert record["stage"] == "orchestrator"
        assert record["event"] == "request"
        assert record["payload"]["input"] == "oi"

    def test_log_with_latency(self, logger, tmp_path):
        logger.log("t1", "studier", "synthesis_done", latency_ms=123.45)
        record = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert record["latency_ms"] == 123.45

    def test_log_without_payload_omits_key(self, logger, tmp_path):
        logger.log("t1", "reflex", "hit")
        record = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert "payload" not in record

    def test_file_created_owner_only(self, logger, tmp_path):
        # M6-A6: the JSONL is personal data — 0600 from birth.
        logger.log("t1", "orchestrator", "request")
        mode = (tmp_path / "audit.jsonl").stat().st_mode & 0o777
        assert mode == 0o600

    def test_existing_file_tightened_on_init(self, tmp_path):
        # One-shot for files created before M6-A6 (e.g. a 644 audit.jsonl).
        path = tmp_path / "audit.jsonl"
        path.write_text("")
        path.chmod(0o644)
        AuditLogger(path=str(path))
        assert path.stat().st_mode & 0o777 == 0o600

    def test_rotation_archives_gz_and_never_deletes(self, tmp_path, monkeypatch):
        # Retention policy (s72): >threshold → gzip into archive/, 0600,
        # fresh active file; every record survives in active+archive.
        monkeypatch.setenv("CORTEX_AUDIT_MAX_BYTES", "500")
        logger = AuditLogger(path=str(tmp_path / "audit.jsonl"))
        for i in range(30):
            logger.log("t1", "stage", "event", payload={"i": i, "pad": "x" * 40})
        archive = tmp_path / "archive"
        gzs = sorted(archive.glob("audit-*.jsonl.gz"))
        assert gzs, "expected at least one rotated archive"
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in gzs)
        # No uncompressed leftover (gzip succeeded). The active file may
        # exceed the threshold by at most ONE record (rotation runs before
        # the append that crossed it).
        assert not list(archive.glob("audit-*.jsonl"))
        assert (tmp_path / "audit.jsonl").stat().st_size < 500 + 200
        # Nothing lost: archived + active records add up to all 30.
        total = len((tmp_path / "audit.jsonl").read_text().strip().splitlines())
        for gz in gzs:
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                total += len(f.read().strip().splitlines())
        assert total == 30

    def test_rotation_threshold_env_malformed_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_AUDIT_MAX_BYTES", "muito")
        logger = AuditLogger(path=str(tmp_path / "audit.jsonl"))
        logger.log("t1", "stage", "event")  # must not raise nor rotate
        assert not (tmp_path / "archive").exists()

    def test_payload_redacted_at_border(self, logger, tmp_path):
        # M6-A6: sensitive keys and free-text secrets never reach the JSONL.
        logger.log("t1", "tool", "call", payload={
            "api_key": "sk-123",
            "input": "minha senha do banco é 1234",
            "to": "a@b.c",
        })
        raw = (tmp_path / "audit.jsonl").read_text()
        assert "sk-123" not in raw and "1234" not in raw
        record = json.loads(raw.strip())
        assert record["payload"]["api_key"] == "[REDACTED]"
        assert record["payload"]["to"] == "a@b.c"

    def test_multiple_logs_appended(self, logger, tmp_path):
        logger.log("t1", "orchestrator", "request")
        logger.log("t1", "orchestrator", "response", latency_ms=500.0)
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["event"] == "response"

    def test_new_trace_unique(self, logger):
        traces = {logger.new_trace() for _ in range(100)}
        assert len(traces) == 100

    def test_new_trace_is_8_chars(self, logger):
        assert len(logger.new_trace()) == 8

    def test_read_tail_returns_last_n(self, logger):
        for i in range(10):
            logger.log(f"t{i}", "stage", "event")
        tail = logger.read_tail(3)
        assert len(tail) == 3
        assert tail[-1]["trace_id"] == "t9"

    def test_read_trace_filters_by_id(self, logger):
        logger.log("aaa", "orchestrator", "request")
        logger.log("bbb", "studier", "run_start")
        logger.log("aaa", "orchestrator", "response")
        records = logger.read_trace("aaa")
        assert len(records) == 2
        assert all(r["trace_id"] == "aaa" for r in records)

    def test_read_tail_empty_file(self, logger, tmp_path):
        # File doesn't exist yet — should return empty list, not raise.
        assert logger.read_tail(10) == []

    def test_thread_safety(self, logger, tmp_path):
        """Concurrent writes must not corrupt the JSONL file."""
        errors = []

        def write_many(tid):
            try:
                for i in range(20):
                    logger.log(f"t{tid}", "stage", f"event_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_many, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 100  # 5 threads × 20 writes
        for line in lines:
            json.loads(line)  # must all be valid JSON

    def test_span_context_manager_logs_start_and_end(self, logger, tmp_path):
        trace = logger.new_trace()
        with logger.span(trace, "studier", "synthesis_start", "synthesis_end"):
            time.sleep(0.01)

        records = logger.read_trace(trace)
        events = [r["event"] for r in records]
        assert "synthesis_start" in events
        assert "synthesis_end" in events
        end_rec = next(r for r in records if r["event"] == "synthesis_end")
        assert end_rec["latency_ms"] >= 10  # slept 10ms

    def test_span_logs_error_on_exception(self, logger):
        trace = logger.new_trace()
        with pytest.raises(ValueError):
            with logger.span(trace, "test", "start", "end"):
                raise ValueError("boom")

        records = logger.read_trace(trace)
        error_rec = next((r for r in records if r["event"] == "error"), None)
        assert error_rec is not None
        assert "boom" in error_rec["payload"]["error"]

    def test_log_never_raises_on_bad_path(self, tmp_path):
        """Even with an unwritable path, log() must not crash the pipeline."""
        bad_logger = AuditLogger(path="/nonexistent_root/deep/audit.jsonl")
        # Should not raise — audit failure is silent.
        bad_logger.log("x", "test", "event")


def test_a_record_opened_before_another_rotation_is_kept(tmp_path, monkeypatch):
    """Two processes on one audit: B had opened the file when A crossed the
    limit, renamed, gzipped and removed it — B then wrote into a deleted
    inode and its record was gone (s88, Codex lote 10). Two loggers on one
    path stand for two processes: their threading locks never meet."""
    import builtins
    import glob
    import gzip
    import threading
    import time
    from src.core import audit as audit_mod

    path = str(tmp_path / "audit.jsonl")
    a, b = audit_mod.AuditLogger(path), audit_mod.AuditLogger(path)
    a._max_bytes = b._max_bytes = 2000
    a.log("t", "s", "seed", payload={"x": "y" * 50})
    opened, release = threading.Event(), threading.Event()
    real_open = builtins.open

    def open_then_pause_b(p, mode="r", *args, **kw):
        f = real_open(p, mode, *args, **kw)
        if threading.current_thread().name == "B" and "a" in mode and p == path:
            opened.set()
            release.wait(5)
        return f
    monkeypatch.setattr(audit_mod, "open", open_then_pause_b, raising=False)
    tb = threading.Thread(target=lambda: b.log("t", "s", "B_SENTINEL"), name="B")
    tb.start()
    opened.wait(5)
    ta = threading.Thread(target=lambda: [a.log("t", "s", f"A{i}", payload={"x": "y" * 50})
                                          for i in range(60)])
    ta.start()
    time.sleep(0.3)
    release.set()
    tb.join(5)
    ta.join(10)
    text = ""
    for f in [path] + glob.glob(str(tmp_path / "archive" / "*")):
        with (gzip.open if f.endswith(".gz") else real_open)(f, "rt", encoding="utf-8") as h:
            text += h.read()
    assert text.count("B_SENTINEL") == 1
    assert all(f"\"A{i}\"" in text for i in range(60))


class TestReadersSeeTheRotatedWeek:
    """The weekly review and the policy detector read only the active file:
    after a rotation, a 7-day window lost the days before it (s88, Codex
    lote 10)."""

    def _audit(self, tmp_path):
        import json
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        arch = tmp_path / "archive"
        arch.mkdir()

        def rec(days_ago, event, **payload):
            return json.dumps({"ts": (now - timedelta(days=days_ago)).isoformat(),
                               "trace_id": "t", "stage": "gate", "event": event,
                               "payload": payload}) + "\n"
        rot = (now - timedelta(days=2)).strftime("%Y%m%d-%H%M%S")
        with gzip.open(arch / f"audit-{rot}-aaaaaa.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(rec(9, "OLD_ONE"))          # outside the window
            f.write(rec(4, "ARCHIVED_IN_WINDOW"))
        old = (now - timedelta(days=20)).strftime("%Y%m%d-%H%M%S")
        with gzip.open(arch / f"audit-{old}-bbbbbb.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(rec(25, "ANCIENT"))
        (tmp_path / "audit.jsonl").write_text(rec(1, "ACTIVE_ONE"), encoding="utf-8")
        return str(tmp_path / "audit.jsonl"), now

    def test_the_weekly_window_spans_the_rotation(self, tmp_path):
        from datetime import timedelta
        from src.weekly_review import load_records
        path, now = self._audit(tmp_path)
        events = [r["event"] for r in load_records(path, now - timedelta(days=7))]
        assert events == ["ARCHIVED_IN_WINDOW", "ACTIVE_ONE"]

    def test_the_detector_sees_an_archived_decision(self, tmp_path):
        from datetime import timedelta
        from src.core.policy_detector import load_records
        path, now = self._audit(tmp_path)
        events = {r["event"] for r in load_records(path, since=now - timedelta(days=7))}
        assert "ARCHIVED_IN_WINDOW" in events and "ANCIENT" not in events
