"""StudiesReview — which briefing a reference names, and what approving it
promises (s88, Codex lote 14)."""
import os
import time
from unittest.mock import MagicMock

import pytest

from src.core.studies_review import StudiesReview


@pytest.fixture
def review(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    for i, (fn, topic) in enumerate([("sqlite.md", "SQLite"), ("oauth.md", "OAuth"),
                                     ("ww2.md", "Segunda Guerra Mundial")]):
        p = pending / fn
        p.write_text(f"# Topic: {topic}\n\n## Confiança\nalta\n\nCorpo.\n")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    return StudiesReview(pending_dir=str(pending), approved_dir=str(tmp_path / "approved"),
                         knowledge_base=MagicMock())


class TestAnOrdinalInsideATitle:
    """Any ordinal word anywhere in the reference was read as a position:
    "aprova Segunda Guerra Mundial" approved study nº 2, OAuth."""

    def test_the_title_wins(self, review):
        assert review.find("Segunda Guerra Mundial")["topic"] == "Segunda Guerra Mundial"

    def test_approving_it_moves_that_file(self, review):
        path = review.approve("Segunda Guerra Mundial")
        assert os.path.basename(path) == "ww2.md"
        assert sorted(os.listdir(review.pending_dir)) == ["oauth.md", "sqlite.md"]

    @pytest.mark.parametrize("ref, topic", [
        ("segundo", "OAuth"), ("o segundo", "OAuth"), ("segundo estudo", "OAuth"),
        ("o segundo estudo", "OAuth"), ("2º", "OAuth"), ("2", "OAuth"),
        ("estudo 3", "Segunda Guerra Mundial"), ("the second one", "OAuth"),
        ("primeiro", "SQLite"), ("third", "Segunda Guerra Mundial"),
    ])
    def test_positions_still_work(self, review, ref, topic):
        assert review.find(ref)["topic"] == topic



class TestApprovedMeansInThePalace:
    """A failed index was only logged: the file left pending/ and Pedro
    heard "agora faz parte da memória permanente" (s88, Codex lote 14 —
    invariant 7)."""

    @staticmethod
    def _host(review):
        from src.core.i18n import I18nMixin
        from src.core.orchestrator import CortexOrchestrator
        from src.core.studier_mixin import StudierMixin

        class H(StudierMixin, I18nMixin):
            _ALL_REF_RE = CortexOrchestrator._ALL_REF_RE
        h = H()
        h._reply_lang = "pt"
        h.studies_review = review
        return h

    def test_a_locked_palace_is_said_and_the_study_stays_pending(self, review):
        import sqlite3
        review.knowledge_base.index_file.side_effect = sqlite3.OperationalError(
            "database is locked")
        reply, mode = self._host(review)._handle_approve_study("SQLite")
        assert mode == "STUDY_APPROVE_ERROR"
        assert "memória permanente" not in reply and "continua pendente" in reply
        assert "sqlite.md" in os.listdir(review.pending_dir)
        assert os.listdir(review.approved_dir) == []
        review.knowledge_base.remove_file.assert_called_once_with("sqlite.md")

    def test_nothing_indexed_is_a_failure_too(self, review):
        review.knowledge_base.index_file.return_value = False
        _reply, mode = self._host(review)._handle_approve_study("SQLite")
        assert mode == "STUDY_APPROVE_ERROR"

    def test_approving_again_retries(self, review):
        import sqlite3
        review.knowledge_base.index_file.side_effect = [
            sqlite3.OperationalError("database is locked"), True]
        h = self._host(review)
        h._handle_approve_study("SQLite")
        reply, mode = h._handle_approve_study("SQLite")
        assert mode == "STUDY_APPROVE" and "memória permanente" in reply
        assert os.listdir(review.approved_dir) == ["sqlite.md"]

    def test_approve_all_names_what_was_not_stored(self, review):
        import sqlite3
        review.knowledge_base.index_file.side_effect = [
            True, sqlite3.OperationalError("database is locked"), True]
        reply, mode = self._host(review)._handle_approve_study("todos")
        assert mode == "STUDY_APPROVE_ERROR"
        assert "Aprovei 2 estudos" in reply and "'OAuth'" in reply
        assert os.listdir(review.pending_dir) == ["oauth.md"]

    def test_the_happy_path_is_unchanged(self, review):
        review.knowledge_base.index_file.return_value = True
        reply, mode = self._host(review)._handle_approve_study("SQLite")
        assert mode == "STUDY_APPROVE" and "memória permanente" in reply



class TestOneReferenceOneStudy:
    """The handler read A's details, then approve(ref) resolved "1" again —
    after the terminal had approved A, "1" was B: B approved, A announced
    (s88, Codex lote 14 — reproduced)."""

    def _race(self, review, verb):
        host = TestApprovedMeansInThePalace._host(review)
        other = StudiesReview(pending_dir=review.pending_dir,
                              approved_dir=review.approved_dir,
                              knowledge_base=MagicMock())
        real = review.get_briefing

        def then_the_terminal_approves(ref):
            out = real(ref)
            other.approve("1")
            return out
        review.get_briefing = then_the_terminal_approves
        handler = (host._handle_approve_study if verb == "approve"
                   else host._handle_reject_study)
        return handler("1")

    def test_approve_does_not_take_the_next_one(self, review):
        reply, mode = self._race(review, "approve")
        assert mode == "STUDY_APPROVE_ERROR" and "SQLite" in reply
        assert "já não está entre os pendentes" in reply
        assert sorted(os.listdir(review.pending_dir)) == ["oauth.md", "ww2.md"]

    def test_reject_does_not_take_the_next_one(self, review):
        reply, mode = self._race(review, "reject")
        assert mode == "STUDY_REJECT_ERROR"
        assert sorted(os.listdir(review.pending_dir)) == ["oauth.md", "ww2.md"]
