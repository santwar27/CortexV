"""Taking topics OFF the study queue (s83).

The queue could be added to, listed, shown and approved — never emptied.
Pedro changed his mind about three topics and tried five phrasings:
"limpar fila de estudos", "exclui tudo", "exclui o estudo 2", "remove o
estudo 3", "cancela o estudo 1". None had a route, so the 8B answered,
said "I'll just tidy up your study queue", and tidied nothing — then
refused the next attempt and invented a policy to justify the refusal.
"""
import pytest

from src.core.dispatch.studier_cmds import CANCEL_STUDY_RE, CANCEL_VAGUE_RE
from src.core.study_queue import StudyQueue


@pytest.fixture
def queue(tmp_path):
    q = StudyQueue(path=str(tmp_path / "study_queue.jsonl"))
    for topic in ("reranker cross-encoder para RAG",
                  "estratégias de chunking para RAG",
                  "roteamento de intenção por embedding"):
        q.add(topic)
    return q


class TestTheStore:

    def test_cancel_one_by_its_listed_number(self, queue):
        cancelled = queue.cancel([2])
        assert [e["topic"] for e in cancelled] == ["estratégias de chunking para RAG"]
        assert [e["topic"] for e in queue.list_pending()] == [
            "reranker cross-encoder para RAG",
            "roteamento de intenção por embedding"]

    def test_cancel_several(self, queue):
        queue.cancel([1, 3])
        assert len(queue.list_pending()) == 1

    def test_cancel_everything(self, queue):
        assert len(queue.cancel()) == 3
        assert queue.list_pending() == []

    def test_a_number_out_of_range_cancels_nothing(self, queue):
        assert queue.cancel([9]) == []
        assert len(queue.list_pending()) == 3

    def test_the_line_survives_as_history(self, queue):
        queue.cancel([1])
        statuses = [e.get("status") for e in queue._read_all()]
        assert "cancelled" in statuses and len(statuses) == 3

    def test_a_cancelled_topic_can_be_added_again(self, queue):
        """find_similar only blocks on pending and consumed, so changing your
        mind twice is allowed."""
        queue.cancel([1])
        again = queue.add("reranker cross-encoder para RAG")
        assert again["status"] == "pending"

    def test_cancelling_does_not_touch_consumed_history(self, queue):
        queue.consume_all()
        assert queue.cancel() == []
        assert all(e.get("status") == "consumed" for e in queue._read_all())


class TestThePhrasings:

    @pytest.mark.parametrize("text", [
        "limpar fila de estudos",
        "limpa a fila de estudos",
        "apaga a fila de estudos",
        "cancela todos os estudos",
        "limpa os estudos pendentes",
        "exclui a fila de estudos",
        "clear the study queue",
        "remove all the studies",
    ])
    def test_the_bulk_forms_match_and_mean_everything(self, text):
        m = CANCEL_STUDY_RE.match(text)
        assert m and m.group("all"), text

    @pytest.mark.parametrize("text, numbers", [
        ("exclui o estudo 2", "2"),
        ("remove o estudo 3", "3"),
        ("cancela o estudo 1", "1"),
        ("tira o estudo 2", "2"),
        ("exclui os estudos 1,3", "1,3"),
        ("remove study 2", "2"),
    ])
    def test_the_numbered_forms_capture_the_number(self, text, numbers):
        m = CANCEL_STUDY_RE.match(text)
        assert m and not m.group("all")
        assert m.group("n").replace(" ", "") == numbers

    @pytest.mark.parametrize("text", [
        # the queue noun is required: a wide pattern here is how "por que
        # azul?" came to play pagode
        "exclui tudo isso daqui",
        "limpa a casa",
        "cancela a reunião de amanhã",
        "tira uma dúvida minha",
        "remove o Spotify",
        "apaga essa lembrança",
    ])
    def test_ordinary_sentences_are_not_queue_commands(self, text):
        assert not CANCEL_STUDY_RE.match(text), text

    @pytest.mark.parametrize("text", ["exclui tudo", "limpa tudo",
                                      "cancela todos", "apaga tudo"])
    def test_a_bare_everything_only_asks(self, text):
        """Delete what — notes, emails, files, the queue? It asks."""
        assert CANCEL_VAGUE_RE.match(text), text

    @pytest.mark.parametrize("text", ["exclui tudo isso daqui",
                                      "exclui tudo que eu disse"])
    def test_the_vague_form_stays_anchored(self, text):
        assert not CANCEL_VAGUE_RE.match(text), text


class TestTheReplies:

    @pytest.fixture
    def orch(self, queue):
        from src.core.orchestrator import CortexOrchestrator
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o.study_queue = queue
        o._reply_lang = "pt"
        return o

    def test_it_says_what_it_removed(self, orch):
        reply, mode = orch._handle_cancel_studies([2])
        assert mode == "STUDY_CANCELLED"
        assert "chunking" in reply and "Sobrou 2" in reply

    def test_clearing_everything_names_the_topics(self, orch):
        """They are topics Pedro typed; they should not vanish silently."""
        reply, _ = orch._handle_cancel_studies(None)
        assert "reranker" in reply and "chunking" in reply and "roteamento" in reply

    def test_an_empty_queue_says_so(self, orch):
        orch._handle_cancel_studies(None)
        reply, _ = orch._handle_cancel_studies(None)
        assert "já está vazia" in reply

    def test_a_bad_number_points_at_the_listing(self, orch):
        reply, _ = orch._handle_cancel_studies([9])
        assert "não achei" in reply and "lista estudos" in reply

    def test_the_vague_form_offers_the_queue(self, orch):
        reply, _ = orch._handle_cancel_vague()
        assert "Tudo o quê?" in reply and "fila de estudos" in reply
