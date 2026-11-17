"""Three defects from one real session with the Studier (s83).

Pedro queued a very specific topic — "reranker cross-encoder para RAG em
Apple Silicon sem PyTorch: bge-reranker, ONNX Runtime e MLX" — and the
briefing came back generic and self-contradicting. The log showed why: the
ML disambiguation, built to stop "Q4" returning Audi, forced "machine
learning LLM" into every query and the word "reranker" fell out of the
first one. Generic queries, generic snippets, a briefing that never
answered the question.

Then "o que você aprendeu com seu último estudo?" was answered with SOLID
and FastAPI — old approved knowledge — because there was no route and the
8B improvised. And "revisar" came back in English.
"""
import pytest

from src.core.dispatch.studier_cmds import (
    APPROVE_STUDY_RE, LAST_STUDY_RE, REJECT_STUDY_RE, REVIEW_RE,
)
from src.core.i18n import _detect_lang
from src.core.studier import NightlyStudier
from src.core.studies_review import StudiesReview

SPECIFIC = ("reranker cross-encoder para RAG em Apple Silicon sem PyTorch: "
            "bge-reranker, ONNX Runtime e MLX")


@pytest.fixture
def studier():
    return NightlyStudier.__new__(NightlyStudier)


class TestDisambiguationIsForAmbiguousTopics:

    @pytest.mark.parametrize("topic", [
        "Q4 vs Q5 vs Q8",
        "KV cache",
        "attention mechanism",
        "quantização Q4_K_M",
    ])
    def test_short_ml_topics_still_get_disambiguated(self, studier, topic):
        assert studier._needs_disambiguation(topic)

    @pytest.mark.parametrize("topic", [
        SPECIFIC,
        "estratégias de chunking para RAG: semantic chunking, late chunking e "
        "parent-document retrieval",
    ])
    def test_a_specific_topic_is_left_alone(self, studier, topic):
        """It already says what it wants; adding "machine learning" to every
        query is what lost the subject."""
        assert not studier._needs_disambiguation(topic)

    def test_a_non_ml_topic_is_never_disambiguated(self, studier):
        assert not studier._needs_disambiguation("a história de Paraná")


class TestQueriesStayOnTopic:

    def test_the_distinctive_words_are_the_specific_ones(self, studier):
        terms = studier._distinctive_terms(SPECIFIC)
        assert "reranker" in terms and "bge-reranker" in terms
        # function words carry no subject
        assert "para" not in terms and "sem" not in terms

    def test_an_off_topic_query_is_dropped(self, studier):
        queries = ["bge-reranker cross encoder RAG", "best pizza in london"]
        assert studier._keep_on_topic(queries, SPECIFIC) == [
            "bge-reranker cross encoder RAG"]

    def test_dropping_everything_falls_back_to_the_topic(self, studier):
        """Better a narrow search than a wrong one."""
        assert studier._keep_on_topic(["best pizza in london"], SPECIFIC) == [
            SPECIFIC]

    def test_a_topic_with_no_distinctive_words_keeps_its_queries(self, studier):
        queries = ["o que e isso", "algo mais"]
        assert studier._keep_on_topic(queries, "o que e") == queries


class TestWhatDidYouLearn:

    @pytest.fixture
    def review(self, tmp_path):
        pending = tmp_path / "pending"
        approved = tmp_path / "approved"
        pending.mkdir()
        approved.mkdir()
        return StudiesReview(pending_dir=str(pending), approved_dir=str(approved))

    def _briefing(self, folder, topic, confidence="alta", summary="O resumo."):
        path = folder / f"{topic[:20].replace(' ', '_')}.md"
        path.write_text(
            f"# Topic: {topic}\n\n## Resumo executivo\n{summary}\n\n"
            f"## Pontos-chave\n- um\n\n## Confiança\n{confidence} — porque sim.\n",
            encoding="utf-8")
        return path

    def test_nothing_studied_yet_is_said_plainly(self, review):
        assert review.last_study() is None

    def test_it_finds_the_newest_across_both_folders(self, review, tmp_path):
        import os, time
        old = self._briefing(tmp_path / "approved", "FastAPI dependency injection")
        time.sleep(0.01)
        new = self._briefing(tmp_path / "pending", "reranker cross-encoder",
                             confidence="baixa")
        os.utime(old, (1, 1))          # make the ordering unambiguous
        item = review.last_study()
        assert item["topic"] == "reranker cross-encoder"
        assert item["state"] == "pending"
        assert item["confidence"] == "baixa"
        assert item["summary"] == "O resumo."

    def test_an_approved_study_is_reported_as_approved(self, review, tmp_path):
        self._briefing(tmp_path / "approved", "SOLID na prática")
        assert review.last_study()["state"] == "approved"

    @pytest.fixture
    def orch(self, review):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.studies_review = review
        o._reply_lang = "pt"
        return o

    def test_the_reply_comes_from_the_file(self, orch, review, tmp_path):
        self._briefing(tmp_path / "pending", "reranker cross-encoder",
                       confidence="baixa", summary="Cross-encoders reordenam.")
        reply, mode = orch._handle_last_study()
        assert mode == "STUDY_LAST"
        assert "reranker cross-encoder" in reply
        assert "Cross-encoders reordenam." in reply

    def test_a_pending_study_is_not_called_knowledge(self, orch, tmp_path):
        """The critic demoted it and nobody reviewed it: saying "I learned"
        would be the same overclaim the 8B made."""
        self._briefing(tmp_path / "pending", "reranker", confidence="baixa")
        reply, _ = orch._handle_last_study()
        assert "ainda não revisado" in reply and "baixa" in reply
        assert "revisar" in reply

    def test_nothing_studied_says_how_to_start(self, orch):
        reply, _ = orch._handle_last_study()
        assert "Ainda não estudei nada" in reply and "estuda" in reply


class TestTheWayPedroActuallyTypes:
    """"aprova 1" is what this project TELLS the user to type, and the regex
    only accepted "aprovar"/"aprove"/"approve" (s83)."""

    @pytest.mark.parametrize("text", ["aprova 1", "aprovar 1", "aprove 1",
                                      "approve 1", "aprova todos"])
    def test_approve_accepts_the_imperative(self, text):
        assert APPROVE_STUDY_RE.match(text), text

    @pytest.mark.parametrize("text", ["rejeita 1", "rejeitar 1", "rejeite 1",
                                      "descarta 1", "reject 1"])
    def test_reject_accepts_the_imperative(self, text):
        assert REJECT_STUDY_RE.match(text), text

    @pytest.mark.parametrize("text", ["revisa", "revisar", "revise",
                                      "revisa estudos", "review studies"])
    def test_review_accepts_the_imperative(self, text):
        assert REVIEW_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "revisar", "revisa", "aprova 1", "rejeita 1",
        "o que você aprendeu com seu último estudo?",
    ])
    def test_these_are_portuguese(self, text):
        assert _detect_lang(text) == "pt", text

    @pytest.mark.parametrize("text", ["approve 1", "reject 1",
                                      "review studies",
                                      "what did you learn?"])
    def test_and_the_english_ones_stay_english(self, text):
        assert _detect_lang(text) == "en", text


class TestLastStudyPhrasing:

    @pytest.mark.parametrize("text", [
        "o que voce aprendeu com seu ultimo estudo?",
        "o que você aprendeu?",
        "que aprendeu hoje",
        "qual foi seu último estudo?",
        "what did you learn?",
        "what did you learn in your last study",
    ])
    def test_matches(self, text):
        assert LAST_STUDY_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "o que você aprendeu com o livro de arquitetura",
        "o que eu aprendi hoje",
        "o que você acha disso?",
    ])
    def test_does_not_swallow_a_conversation(self, text):
        assert not LAST_STUDY_RE.match(text), text
