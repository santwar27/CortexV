"""Interrupted approvals recover using only temporary files and a fake embedder."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.audit import audit
from src.core.knowledge_base import KnowledgeBase
from src.core.studier import NightlyStudier
from src.core.studies_review import StudiesReview
from src.core.study_queue import StudyQueue
from src.memory.palace import Palace


BRIEFING = "# Topic: Botany\nIntroduction.\n\n## Leaves\nGreen.\n\n## Roots\nUnderground.\n"


class ProcessDied(BaseException):
    pass


@pytest.fixture
def review(tmp_path):
    palace = Palace(db_path=str(tmp_path / "palace.db"),
                    embed_fn=MagicMock(return_value=np.ones(4, dtype=np.float32)))
    kb = KnowledgeBase(approved_dir=str(tmp_path / "approved"), palace=palace)
    yield StudiesReview(pending_dir=str(tmp_path / "pending"),
                        approved_dir=kb.approved_dir, knowledge_base=kb)
    palace.close()


def write_study(folder, filename="botany.md"):
    path = Path(folder) / filename
    path.write_text(BRIEFING, encoding="utf-8")
    return path


def rows(review, filename="botany.md"):
    return [d for d in review.knowledge_base._palace.list_documents(tipo="knowledge")
            if d["metadata"].get("filename") == filename]


def seed_rows(review, metadata, filename="botany.md", tipo="knowledge"):
    for meta in metadata:
        review.knowledge_base._palace.add_document(
            tipo=tipo, texto="synthetic chunk", metadata={"filename": filename, **meta})


def study_events():
    return [(e["event"], e.get("payload")) for e in audit.read_tail()
            if e["stage"] == "studies"]


def test_death_after_move_is_completed_by_new_review(review):
    path = write_study(review.pending_dir)
    palace = review.knowledge_base._palace
    palace.embed_fn.side_effect = [np.ones(4, dtype=np.float32), ProcessDied()]

    with pytest.raises(ProcessDied):
        review.approve("1")

    approved = Path(review.approved_dir) / path.name
    assert approved.is_file() and not path.exists()
    assert [d["metadata"]["chunk_count"] for d in rows(review)] == [3]
    palace.embed_fn.side_effect = None
    # Reopen the database too: the partial row survived the old process.
    palace.close()
    restarted_palace = Palace(db_path=palace.db_path, embed_fn=palace.embed_fn)
    try:
        restarted = StudiesReview(
            pending_dir=review.pending_dir, approved_dir=review.approved_dir,
            knowledge_base=KnowledgeBase(approved_dir=review.approved_dir,
                                         palace=restarted_palace))
        assert restarted.reconcile_approved() == [path.name]
        assert len(rows(restarted)) == 3
        assert {d["metadata"]["chunk_index"] for d in rows(restarted)} == {0, 1, 2}
        assert study_events() == [("approval_completed", {"count": 1})]
        palace.embed_fn.reset_mock()
        assert restarted.reconcile_approved() == []
        palace.embed_fn.assert_not_called()
        assert study_events() == [("approval_completed", {"count": 1})]
    finally:
        restarted_palace.close()


def test_death_before_any_row_is_completed(review):
    path = write_study(review.approved_dir)
    assert rows(review) == []
    assert review.reconcile_approved() == [path.name]
    assert len(rows(review)) == 3
    assert study_events() == [("approval_completed", {"count": 1})]


def test_complete_file_does_not_embed_or_audit(review):
    path = write_study(review.approved_dir)
    review.knowledge_base.index_file(str(path))
    embed = review.knowledge_base._palace.embed_fn
    embed.reset_mock()
    assert review.reconcile_approved() == []
    embed.assert_not_called()
    assert study_events() == []


@pytest.mark.parametrize("action", ["approve", "approve_all", "approve_item"])
def test_approval_completes_older_file_first(review, monkeypatch, action):
    old = write_study(review.approved_dir, "older.md")
    seed_rows(review, [{"chunk_index": 0, "chunk_count": 3}], filename=old.name)
    new = write_study(review.pending_dir, "new.md")
    original = review.knowledge_base.index_file
    indexed = []

    def index(path):
        indexed.append(Path(path).name)
        if Path(path).name == new.name:
            assert len(rows(review, old.name)) == 3
        return original(path)

    monkeypatch.setattr(review.knowledge_base, "index_file", index)
    if action == "approve_all":
        assert review.approve_all() == ([str(Path(review.approved_dir) / new.name)], [])
    elif action == "approve_item":
        assert review.approve_item(review.find("1")) == str(Path(review.approved_dir) / new.name)
    else:
        assert review.approve("1") == str(Path(review.approved_dir) / new.name)
    assert indexed == [old.name, new.name]
    assert not new.exists()
    assert len(rows(review, new.name)) == 3
    assert study_events() == [("approval_completed", {"count": 1})]


@pytest.mark.parametrize("failure", ["exception", "false", "partial"])
def test_failed_file_stays_approved_and_does_not_stop_others(review, monkeypatch, failure):
    bad = write_study(review.approved_dir, "a_bad.md")
    good = write_study(review.approved_dir, "z_good.md")
    original = review.knowledge_base.index_file

    def index(path):
        if Path(path).name == bad.name:
            if failure == "exception":
                raise RuntimeError("synthetic write failure")
            if failure == "partial":
                seed_rows(review, [{"chunk_index": 0, "chunk_count": 3}], filename=bad.name)
                return True
            return False
        return original(path)

    monkeypatch.setattr(review.knowledge_base, "index_file", index)
    assert review.reconcile_approved() == [good.name]
    assert bad.read_text() == BRIEFING and good.is_file()
    assert list(Path(review.pending_dir).iterdir()) == []
    assert len(rows(review, good.name)) == 3
    assert sorted(study_events()) == [
        ("approval_completed", {"count": 1}), ("approval_incomplete", {"count": 1})]


def test_completeness_read_failure_is_per_file(review, monkeypatch):
    write_study(review.approved_dir, "a_bad.md")
    write_study(review.approved_dir, "z_good.md")
    original = getattr(review.knowledge_base, "is_fully_indexed", None)

    def read(filename):
        if filename == "a_bad.md":
            raise RuntimeError("synthetic read failure")
        return original(filename)

    monkeypatch.setattr(review.knowledge_base, "is_fully_indexed", read, raising=False)
    assert review.reconcile_approved() == ["z_good.md"]
    assert sorted(study_events()) == [
        ("approval_completed", {"count": 1}), ("approval_incomplete", {"count": 1})]


def test_audit_failure_does_not_escape_reconciliation(review, monkeypatch):
    write_study(review.approved_dir)
    monkeypatch.setattr(audit, "log", MagicMock(side_effect=RuntimeError("audit unavailable")))
    assert review.reconcile_approved() == ["botany.md"]
    assert len(rows(review)) == 3


def test_directory_failure_does_not_escape_approval(review, monkeypatch):
    import src.core.studies_review as module

    original = module.os.listdir

    def listdir(path):
        if str(path) == review.approved_dir:
            raise OSError("approved directory unavailable")
        return original(path)

    monkeypatch.setattr(module.os, "listdir", listdir)
    assert review.reconcile_approved() == []
    assert review.approve("1") is None


def test_only_markdown_files_are_reconciled(review):
    write_study(review.approved_dir, "ignored.txt")
    (Path(review.approved_dir) / "directory.md").mkdir()
    assert review.reconcile_approved() == []
    review.knowledge_base._palace.embed_fn.assert_not_called()
    assert study_events() == []


@pytest.mark.parametrize("metadata, expected", [
    ([], False),
    ([{"chunk_index": 0, "chunk_count": 3}], False),
    ([{"chunk_index": 0, "chunk_count": 2}, {"chunk_index": 1, "chunk_count": 3},
      {"chunk_index": 2, "chunk_count": 3}], False),
    ([{"chunk_index": 0, "chunk_count": 2}, {"chunk_index": 0, "chunk_count": 2}], False),
    ([{}, {}], False),
    ([{"chunk_index": 0, "chunk_count": 0}], False),
    ([{"chunk_index": 0, "chunk_count": 1}], True),
    ([{"chunk_index": i, "chunk_count": 3} for i in range(3)], True),
    ([{"chunk_index": i, "chunk_count": 2} for i in [0, 1, 1]], True),
])
def test_is_fully_indexed_checks_all_chunk_metadata(review, metadata, expected):
    seed_rows(review, metadata)
    seed_rows(review, [{"chunk_index": 0, "chunk_count": 5}], filename="unrelated.md")
    seed_rows(review, [{"chunk_index": 0, "chunk_count": 5}], tipo="conversation")
    assert review.knowledge_base.is_fully_indexed("botany.md") is expected
    review.knowledge_base._palace.embed_fn.assert_not_called()


@pytest.mark.parametrize("raises", [False, True])
def test_nightly_reconciles_once_after_backup_and_continues(tmp_path, monkeypatch, caplog, raises):
    studier = NightlyStudier(
        engine=MagicMock(), queue=StudyQueue(path=str(tmp_path / "queue.jsonl")),
        web_searcher=MagicMock(), output_dir=str(tmp_path / "pending"))
    monkeypatch.setenv("CORTEX_DEEP_STUDIER", "0")
    order = []
    for name in ("_backup_palace", "_retention_pass", "_consolidate_memory",
                 "_consolidate_preferences"):
        monkeypatch.setattr(studier, name, lambda name=name: order.append(name))

    def reconcile(self):
        assert self.pending_dir == studier.output_dir
        assert self.approved_dir == str(tmp_path / "approved")
        order.append("reconcile")
        if raises:
            raise RuntimeError("synthetic reconciliation failure")
        return []

    reconcile_mock = MagicMock(side_effect=reconcile)
    monkeypatch.setattr(StudiesReview, "reconcile_approved",
                        lambda self: reconcile_mock(self), raising=False)
    assert studier.run() == []
    reconcile_mock.assert_called_once()
    assert order == ["_backup_palace", "reconcile", "_retention_pass",
                     "_consolidate_memory", "_consolidate_preferences"]
    if raises:
        assert "synthetic reconciliation failure" in caplog.text
