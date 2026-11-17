"""Crash recovery and per-topic claims, with only temporary queue/files."""
import builtins
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.core.audit import audit
from src.core.studier import NightlyStudier
from src.core.study_queue import StudyQueue


class ProcessDied(BaseException):
    pass


TOPICS = ["botany", "black holes", "Roman history"]


def studier(tmp_path, monkeypatch):
    s = NightlyStudier(
        engine=MagicMock(), queue=StudyQueue(path=tmp_path / "queue.jsonl"),
        web_searcher=MagicMock(), output_dir=str(tmp_path / "pending"),
    )
    for name in ("_backup_palace", "_retention_pass", "_consolidate_memory",
                 "_consolidate_preferences"):
        monkeypatch.setattr(s, name, lambda: None)
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "0")
    monkeypatch.setenv("CORTEX_CURIOSITY", "0")
    monkeypatch.setattr(s, "generate_queries", lambda topic, **kw: [topic])
    monkeypatch.setattr(s, "gather_snippets", lambda *a, **kw: "source text")
    monkeypatch.setattr(s, "_summarise", lambda *a, **kw: "complete briefing")
    monkeypatch.setattr(s, "_apply_critic", lambda topic, text, **kw: text)
    return s


def records(queue):
    return [json.loads(line) for line in Path(queue.path).read_text().splitlines()]


def seed(queue, **fields):
    queue.add(TOPICS[0])
    entries = records(queue)
    entries[0].update(fields)
    Path(queue.path).write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def events(name):
    return [e for e in map(json.loads, Path(audit.path).read_text().splitlines())
            if e["stage"] == "studier" and e["event"] == name]


def assert_utc(value):
    assert datetime.fromisoformat(value).utcoffset() == timezone.utc.utcoffset(None)


def test_death_mid_run_recovers_only_claimed_topic(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    for topic in TOPICS:
        s.queue.add(topic)
    original = s._study_one

    def die_on_second(topic, **kw):
        if topic == TOPICS[1]:
            raise ProcessDied()
        return original(topic, **kw)

    monkeypatch.setattr(s, "_study_one", die_on_second)
    with pytest.raises(ProcessDied):
        s.run()
    first, second, third = records(s.queue)
    assert first["status"] == "consumed"
    assert Path(first["briefing"]).read_text().endswith("complete briefing\n")
    assert_utc(first["consumed_at"])
    assert second["status"] == "studying"
    assert_utc(second["claimed_at"])
    assert third["status"] == "pending" and "claimed_at" not in third

    resumed = studier(tmp_path, monkeypatch)
    studied = MagicMock(wraps=resumed._study_one)
    monkeypatch.setattr(resumed, "_study_one", studied)
    assert len(resumed.run()) == 2
    assert [c.args[0] for c in studied.call_args_list] == TOPICS[1:]
    after = records(resumed.queue)
    assert after[0] == first
    assert after[1]["retries"] == 1 and after[1]["status"] == "consumed"
    assert_utc(after[1]["interrupted_at"])
    assert events("topic_interrupted")[0]["payload"]["retry"] == 1


def test_interrupted_twice_exhausts_retry(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    seed(s.queue, status="studying", retries=1)
    study = MagicMock()
    monkeypatch.setattr(s, "_study_one", study)
    assert s.run() == []
    assert records(s.queue)[0]["status"] == "failed"
    study.assert_not_called()
    assert events("retry_exhausted")
    assert events("topic_interrupted")[0]["payload"]["retry"] == 1


def test_saved_but_not_marked_is_linked(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    path = Path(s.output_dir) / "finished.md"
    path.write_text("finished")
    seed(s.queue, status="saving", briefing=str(path), retries=1)
    study = MagicMock()
    monkeypatch.setattr(s, "_study_one", study)
    assert s.run() == []
    entry = records(s.queue)[0]
    assert entry["status"] == "consumed" and entry["briefing"] == str(path)
    assert_utc(entry["consumed_at"])
    study.assert_not_called()
    assert events("topic_interrupted")[0]["payload"]["linked"] is True


@pytest.mark.parametrize("empty_file", [False, True])
def test_unsaved_saving_returns_to_pending(tmp_path, monkeypatch, empty_file):
    s = studier(tmp_path, monkeypatch)
    path = Path(s.output_dir) / "missing.md"
    if empty_file:
        path.touch()
    seed(s.queue, status="saving", briefing=str(path))
    # Stop after recovery and before the next claim to inspect pending state.
    from src.core.system_lease import SystemLease
    monkeypatch.setattr(SystemLease, "acquire", MagicMock(side_effect=ProcessDied))
    with pytest.raises(ProcessDied):
        s.run()
    entry = records(s.queue)[0]
    assert entry["status"] == "pending" and entry["retries"] == 1
    assert "briefing" not in entry
    assert_utc(entry["interrupted_at"])


def test_atomic_briefing_half_write_never_publishes(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    real_open = builtins.open

    class HalfWrite:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def write(self, text):
            self.file.write(text[:len(text) // 2])
            self.file.flush()
            raise OSError("disk write interrupted")

    def half_open(path, mode="r", *args, **kw):
        file = real_open(path, mode, *args, **kw)
        if "w" in mode and Path(path).parent == Path(s.output_dir):
            return HalfWrite(file)
        return file

    monkeypatch.setattr("src.core.studier.open", half_open, raising=False)
    with pytest.raises(OSError, match="disk write interrupted"):
        s._save(TOPICS[0], "complete briefing", ["query"])
    assert list(Path(s.output_dir).glob("*.md")) == []


@pytest.mark.parametrize("published", [False, True])
def test_death_at_publish_boundary(tmp_path, monkeypatch, published):
    s = studier(tmp_path, monkeypatch)
    s.queue.add(TOPICS[0])
    import src.core.studier as module
    replace = module.os.replace

    def interrupted_publish(source, destination):
        if Path(destination).suffix == ".md":
            entry = records(s.queue)[0]
            assert entry["status"] == "saving"
            assert entry["briefing"] == str(destination)
            assert Path(source).parent == Path(destination).parent
            assert not Path(destination).exists()
            if published:
                replace(source, destination)
            raise ProcessDied()
        replace(source, destination)

    monkeypatch.setattr(module.os, "replace", interrupted_publish)
    with pytest.raises(ProcessDied):
        s.run()
    assert records(s.queue)[0]["status"] == "saving"
    monkeypatch.setattr(module.os, "replace", replace)
    resumed = studier(tmp_path, monkeypatch)
    study = MagicMock(wraps=resumed._study_one)
    monkeypatch.setattr(resumed, "_study_one", study)
    assert len(resumed.run()) == (0 if published else 1)
    assert study.call_count == (0 if published else 1)
    entry = records(resumed.queue)[0]
    assert entry["status"] == "consumed"
    assert Path(entry["briefing"]).read_text().endswith("complete briefing\n")
    if not published:
        assert entry["retries"] == 1


def test_run_lock_busy_leaves_queue_untouched(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    seed(s.queue, status="studying")
    before = Path(s.queue.path).read_bytes()
    with open(str(s.queue.path) + ".run.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert s.run() == []
        assert Path(s.queue.path).read_bytes() == before
    assert events("run_busy")


def test_run_lock_held_while_studying(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    s.queue.add(TOPICS[0])
    contender = studier(tmp_path, monkeypatch)
    original = s._study_one

    def competing_run(topic, **kw):
        before = Path(s.queue.path).read_bytes()
        assert contender.run() == []
        assert Path(s.queue.path).read_bytes() == before
        return original(topic, **kw)

    monkeypatch.setattr(s, "_study_one", competing_run)
    assert len(s.run()) == 1
    assert len(events("run_busy")) == 1


@pytest.mark.parametrize("operation", ["open", "flock"])
def test_run_lock_error_is_fail_soft(tmp_path, monkeypatch, operation):
    s = studier(tmp_path, monkeypatch)
    s.queue.add(TOPICS[0])
    before = Path(s.queue.path).read_bytes()
    target = "src.core.studier." + ("os.open" if operation == "open" else "fcntl.flock")
    if operation == "open":
        import os
        real_open = os.open

        def failing_lock(path, *args, **kw):
            if str(path).endswith(".run.lock"):
                raise OSError("lock unavailable")
            return real_open(path, *args, **kw)
    else:
        # A system-wide flock error also reaches the audit sink's lock.
        failing_lock = MagicMock(side_effect=OSError("lock unavailable"))

    monkeypatch.setattr(target, failing_lock)
    assert s.run() == []
    assert Path(s.queue.path).read_bytes() == before


def test_add_during_run_waits_for_next_night(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    s.queue.add(TOPICS[0])
    other = StudyQueue(path=str(tmp_path / "queue.jsonl"))
    original = s._study_one

    def add_then_study(topic, **kw):
        other.add(TOPICS[1])
        return original(topic, **kw)

    study = MagicMock(side_effect=add_then_study)
    monkeypatch.setattr(s, "_study_one", study)
    assert len(s.run()) == 1
    assert [e["topic"] for e in other.list_pending()] == [TOPICS[1]]
    assert study.call_count == 1


def test_cancel_during_run_skips_claim(tmp_path, monkeypatch):
    s = studier(tmp_path, monkeypatch)
    for topic in TOPICS:
        s.queue.add(topic)
    other = StudyQueue(path=str(tmp_path / "queue.jsonl"))
    original = s._study_one

    def cancel_then_study(topic, **kw):
        if topic == TOPICS[0]:
            # Topic 1 is already studying; pending positions are now 2, 3.
            cancelled = other.cancel([2])
            assert [e["topic"] for e in cancelled] == [TOPICS[2]]
        return original(topic, **kw)

    study = MagicMock(side_effect=cancel_then_study)
    monkeypatch.setattr(s, "_study_one", study)
    assert len(s.run()) == 2
    assert [c.args[0] for c in study.call_args_list] == TOPICS[:2]
    assert records(s.queue)[2]["status"] == "cancelled"
    assert "claimed_at" not in records(s.queue)[2]


def test_legacy_ids_added_on_rewrite_and_stable(tmp_path):
    queue = StudyQueue(path=str(tmp_path / "queue.jsonl"))
    Path(queue.path).write_text(
        json.dumps({"topic": TOPICS[0], "status": "pending"}) + "\n"
        + "malformed line\n"
        + json.dumps({"topic": TOPICS[1], "status": "consumed"}) + "\n"
    )
    queue.cancel()
    ids = [e["id"] for e in records(queue)]
    assert len(set(ids)) == 2
    assert all(UUID(value).hex == value for value in ids)
    new = queue.add(TOPICS[2], extra={"id": "must-not-win"})
    assert UUID(new["id"]).hex == new["id"]
    queue.consume_all()
    assert [e["id"] for e in records(queue)][:2] == ids


@pytest.mark.parametrize("failure", ["no_snippets", "no_synthesis", "exception"])
def test_normal_failure_consumes_and_requeues_once(tmp_path, monkeypatch, failure):
    s = studier(tmp_path, monkeypatch)
    s.queue.add(TOPICS[0])
    if failure == "no_snippets":
        monkeypatch.setattr(s, "gather_snippets", lambda *a, **kw: "")
    elif failure == "no_synthesis":
        monkeypatch.setattr(s, "_summarise", lambda *a, **kw: None)
    else:
        monkeypatch.setattr(s, "_study_one", MagicMock(side_effect=RuntimeError("failed")))
    assert s.run() == []
    first, retry = records(s.queue)
    assert first["status"] == "consumed"
    assert retry["status"] == "pending" and retry["retries"] == 1
    assert retry["topic"] == TOPICS[0] and retry["requeued"] == "failed_run"
    assert s.run() == []
    assert len(records(s.queue)) == 2
    assert all(e["status"] == "consumed" for e in records(s.queue))
    assert len(events("topic_requeued")) == 1
    assert len(events("retry_exhausted")) == 1
