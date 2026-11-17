"""Answering an exam from its own structure (s85, Pedro's design).

Baseline, measured against the real 8B before this existed: "lê os dois"
after a folder listing read nothing (0/6, one reply invented two PDFs about
OAuth), "both" was answered from an unrelated old chat (6/6), and the paper
read in one go was refused 3/3 ("I can't read external content").
"""
import os

import pytest

from src.core import exam_reader
from src.core.exam_reader import (
    Question, declared_ranges, parse_exam, requested_numbers, split_questions,
    split_texts, texts_for,
)
from src.core.orchestrator import CortexOrchestrator

PAPER = """Pearson Edexcel Functional Skills
Set 2
Information
• Questions 1 to 2 are based on Text A
• Questions 3 to 3 are based on Text B
SECTION A
Read Text A and answer Questions 1 to 2.
1 Using Text A, identify two ways that businesses benefit.
(Total for Question 1 = 2 marks)
4 97 9A0312
2 Which word best describes the style?
A serious
B humorous
(Total for Question 2 = 1 mark)
SECTION B
• 3 What does the college offer its staff?
(Total for Question 3 = 1 mark)
4 Compare Text A and Text B.
(Total for Question 4 = 4 marks)
BLANK PAGE
"""

BOOKLET = """Text Booklet
Text A
Article
Flexible working makes workers more productive.
Text B
Webpage
Bresley College offers a wellbeing programme.
"""


class TestParsing:

    def test_texts_split_at_their_headings(self):
        texts = split_texts(BOOKLET)
        assert sorted(texts) == ["A", "B"]
        assert "productive" in texts["A"] and "wellbeing" not in texts["A"]
        assert "wellbeing" in texts["B"]

    def test_a_text_mentioned_in_a_sentence_is_not_a_heading(self):
        assert split_texts("Read Text A and answer Questions 1 to 2.") == {}

    def test_questions_in_order_ignoring_set_numbers_and_barcodes(self):
        qs = split_questions(PAPER)
        assert [q.number for q in qs] == [1, 2, 3, 4]
        # "Set 2" did not start question 2, the barcode did not start 4
        assert qs[1].text.startswith("2 Which word")
        assert qs[3].text.startswith("4 Compare")

    def test_a_question_ends_at_its_marks_line(self):
        q1 = split_questions(PAPER)[0]
        assert q1.text.endswith("(Total for Question 1 = 2 marks)")
        assert "4 97" not in q1.text

    def test_a_stray_bullet_before_the_number_is_fine(self):
        assert split_questions(PAPER)[2].text.startswith("• 3 What")

    def test_one_lost_stem_does_not_stop_the_count(self):
        doc = "1 First?\n(Total for Question 1 = 1 mark)\n3 Third?\n"
        assert [q.number for q in split_questions(doc)] == [1, 3]

    def test_declared_ranges_both_wordings(self):
        assert declared_ranges(PAPER) == {1: "A", 2: "A", 3: "B"}

    def test_a_named_text_wins_then_the_declared_one_then_all(self):
        ranges = {1: "A"}
        assert texts_for(Question(9, "Compare Text A and Text B"), ranges,
                         ["A", "B", "C"]) == ["A", "B"]
        assert texts_for(Question(1, "Which word?"), ranges,
                         ["A", "B"]) == ["A"]
        assert texts_for(Question(7, "Which word?"), ranges,
                         ["A", "B"]) == ["A", "B"]

    def test_the_whole_exam(self):
        exam = parse_exam([PAPER, BOOKLET])
        assert [q.texts for q in exam.questions] == [["A"], ["A"], ["B"], ["A", "B"]]

    def test_the_order_of_the_files_does_not_matter(self):
        assert parse_exam([BOOKLET, PAPER]).questions[3].texts == ["A", "B"]

    @pytest.mark.parametrize("docs", [
        ["Just a letter.\nDear Pedro,\nThanks."],
        [BOOKLET],                                  # texts, no questions
        [PAPER.replace("Total for Question", "Marks")],   # not marked
        [],
    ])
    def test_not_an_exam_is_none(self, docs):
        assert parse_exam(docs) is None

    @pytest.mark.parametrize("text,want", [
        ("answer every questions", []),
        ("responde a questão 13", [13]),
        ("answer questions 1 to 4", [1, 2, 3, 4]),
        ("questões 2, 3 e 5", [2, 3, 5]),
        ("question 12-15", [12, 13, 14, 15]),
        ("there is two pdf in this folder 13-10, read them", []),
    ])
    def test_requested_numbers(self, text, want):
        assert requested_numbers(text) == want


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def exam_dir(home):
    d = home / "Downloads" / "college" / "13-10"
    d.mkdir(parents=True)
    (d / "question.pdf").write_bytes(b"%PDF-q")
    (d / "text.pdf").write_bytes(b"%PDF-t")
    return d


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._pending_query = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o.background = False
    o.background_origin = "background"
    o._reply_lang = "pt"
    contents = {"question.pdf": PAPER, "text.pdf": BOOKLET}
    monkeypatch.setattr(o, "reader", type("R", (), {
        "read_path": lambda _s, p: contents.get(os.path.basename(p),
                                                "[SYSTEM ERROR] nope")})(),
        raising=False)
    o.engine_analytical = type("E", (), {"model_name": "cortex-lite"})()
    return o


@pytest.fixture
def llm(monkeypatch):
    """The bare exam call, captured. Answers 'resposta N'."""
    calls = []

    def fake_chat(**kw):
        calls.append(kw)
        user = kw["messages"][1]["content"]
        n = user.rsplit("Answer question ", 1)[1].rstrip(".")
        return {"message": {"content": f"resposta {n}"}}

    monkeypatch.setattr("src.core.staging_mixin.ollama.chat", fake_chat)
    return calls


class TestTheRoutes:

    def test_a_folder_that_asks_for_a_read_confirms_both_at_once(self, orch,
                                                                exam_dir):
        reply, mode = orch._handle_user_paths(
            f"there is two pdf in this folder {exam_dir} , read them and "
            "answer every questions")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "read_files"
        assert [os.path.basename(p) for p in orch._pending_action["args"]["paths"]] \
            == ["question.pdf", "text.pdf"]
        assert "question.pdf e text.pdf" in reply

    def test_a_folder_named_read_is_not_a_request(self, orch, home):
        d = home / "read"
        d.mkdir()
        (d / "a.pdf").write_bytes(b"x")
        (d / "b.pdf").write_bytes(b"x")
        _reply, mode = orch._handle_user_paths(f"o que tem em {d}?")
        assert mode == "FILE_LIST"

    def test_too_many_files_are_listed_not_read(self, orch, home):
        d = home / "muitos"
        d.mkdir()
        for i in range(5):
            (d / f"f{i}.pdf").write_bytes(b"x")
        _reply, mode = orch._handle_user_paths(f"lê {d}")
        assert mode == "FILE_LIST" and orch._pending_action is None
        assert orch._listed_followup("todos") is None   # five is too many

    @pytest.mark.parametrize("follow", ["os dois", "lê os dois", "both",
                                        "Both.", "lê os dois e responde as questões",
                                        "read both of them", "ambos"])
    def test_both_after_a_listing_confirms_the_listed_files(self, orch,
                                                            exam_dir, follow):
        orch._handle_user_paths(f"o que tem em {exam_dir}?")
        assert orch._mentions_files(follow)            # the stream steps aside
        _reply, mode = orch._handle_user_paths(follow)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "read_files"

    @pytest.mark.parametrize("text", ["todos os dias eu corro", "tudo bem?",
                                      "both teams played well", "all good"])
    def test_ordinary_sentences_are_not_a_read(self, orch, exam_dir, text):
        orch._handle_user_paths(f"o que tem em {exam_dir}?")
        assert not orch._mentions_files(text)
        assert orch._handle_user_paths(text) is None

    def test_both_means_nothing_without_a_listing(self, orch):
        assert orch._handle_user_paths("both") is None

    def test_the_listing_expires(self, orch, exam_dir, monkeypatch):
        orch._handle_user_paths(f"o que tem em {exam_dir}?")
        when, paths = orch._last_listed
        orch._last_listed = (when - orch._LISTED_TTL_S - 1, paths)
        assert orch._handle_user_paths("os dois") is None

    def test_the_preview_names_both_files(self):
        from src.core.tool_risk import build_preview, is_destructive
        assert is_destructive("read_files")
        human = build_preview("read_files", {"paths": [
            "/Users/p/13-10/question.pdf", "/Users/p/13-10/text.pdf"]},
            lang="en")["human"]
        assert human.startswith(
            "May I open and read these 2 files: question.pdf and text.pdf?")
        assert "/Users/p/13-10" in human


class TestReadingAndAnswering:

    def _approve(self, orch, exam_dir, text="read them and answer every questions"):
        orch._handle_user_paths(f"two pdf in {exam_dir} , {text}")
        return orch._execute_pending_action()

    def test_one_call_per_question_each_with_only_its_texts(self, orch,
                                                            exam_dir, llm):
        reply, mode = self._approve(orch, exam_dir)
        assert mode == "LOCAL_RAG"
        assert len(llm) == 4
        users = [c["messages"][1]["content"] for c in llm]
        assert "productive" in users[0] and "wellbeing" not in users[0]   # Q1: A
        assert "wellbeing" in users[2] and "productive" not in users[2]   # Q3: B
        assert "productive" in users[3] and "wellbeing" in users[3]       # Q4: A+B
        for n in (1, 2, 3, 4):
            assert f"**{n}.** resposta {n}" in reply

    def test_the_file_text_stays_fenced_and_the_task_is_ours(self, orch,
                                                             exam_dir, llm):
        self._approve(orch, exam_dir)
        system = llm[0]["messages"][0]["content"]
        user = llm[0]["messages"][1]["content"]
        assert "Pedro gave you the exam files" in system
        assert user.count("<<<UNTRUSTED_CONTENT>>>") == 2    # text + question
        assert "tools" not in llm[0]                          # nothing to act with

    def test_no_memory_no_history(self, orch, exam_dir, llm):
        """Fifteen exam answers are not conversation: the bare call never
        reaches the engine, so nothing is retrieved or stored."""
        self._approve(orch, exam_dir)          # engine stub has no chat()
        assert all(len(c["messages"]) == 2 for c in llm)

    def test_only_the_questions_asked_for(self, orch, exam_dir, llm):
        reply, _ = self._approve(orch, exam_dir, "answer question 3")
        assert len(llm) == 1 and "**3.** resposta 3" in reply

    def test_a_question_the_paper_does_not_have(self, orch, exam_dir, llm):
        """Asked for 20 on a 4-question paper, it used to answer all four."""
        reply, _ = self._approve(orch, exam_dir, "answer question 20")
        assert llm == [] and reply == "Essa prova vai da questão 1 à 4 — não tem a 20."

    def test_some_asked_numbers_exist(self, orch, exam_dir, llm):
        reply, _ = self._approve(orch, exam_dir, "answer questions 3 and 20")
        assert len(llm) == 1 and "**3.**" in reply
        assert "A questão 20 não existe nesta prova." in reply

    def test_the_answers_stream_and_are_not_sent_twice(self, orch, exam_dir,
                                                       llm):
        got = []
        orch.set_token_emitter(got.append)
        reply, _ = self._approve(orch, exam_dir)
        orch.set_token_emitter(None)
        assert "".join(got) == reply

    def test_a_failed_call_does_not_sink_the_rest(self, orch, exam_dir,
                                                  monkeypatch):
        calls = []

        def flaky(**kw):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("ollama down")
            return {"message": {"content": "ok"}}

        monkeypatch.setattr("src.core.staging_mixin.ollama.chat", flaky)
        reply, _ = self._approve(orch, exam_dir)
        assert len(calls) == 4 and "**2.** (não consegui" in reply

    def test_the_answer_language_is_the_request_not_the_sim(self, orch,
                                                            exam_dir, llm):
        orch._reply_lang = "en"
        orch._handle_user_paths(f"two pdf in {exam_dir} , read them")
        orch._reply_lang = "pt"                 # the "sim" reads as PT
        reply, _ = orch._execute_pending_action()
        assert reply.startswith("I read question.pdf and text.pdf")
        assert "Answer in English." in llm[0]["messages"][0]["content"]

    def test_not_an_exam_gets_one_answer_over_both(self, orch, exam_dir,
                                                   monkeypatch, llm):
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _s, p: f"conteudo de {os.path.basename(p)}"})(),
            raising=False)
        seen = {}

        def synth(_engine, injected, lang, mode):
            seen["prompt"] = injected
            return ("resumo", "LOCAL_RAG")

        monkeypatch.setattr(orch, "_synthesise", synth, raising=False)
        reply, mode = self._approve(orch, exam_dir, "resume them")
        assert reply.startswith("resumo") and llm == []
        assert "conteudo de question.pdf" in seen["prompt"]
        assert "conteudo de text.pdf" in seen["prompt"]

    def test_a_file_that_cannot_be_read_is_named(self, orch, exam_dir,
                                                 monkeypatch, llm):
        monkeypatch.setattr(orch, "reader", type("R", (), {
            "read_path": lambda _s, p: PAPER if p.endswith("question.pdf")
            else "[SYSTEM ERROR] broken"})(), raising=False)
        monkeypatch.setattr(orch, "_synthesise",
                            lambda *a, **k: ("x", "LOCAL_RAG"), raising=False)
        reply, _ = self._approve(orch, exam_dir)
        assert "text.pdf" in reply.rsplit("(", 1)[-1]

    def test_an_unseen_path_is_never_read(self, orch, exam_dir, llm):
        reply, mode = orch._run_read_files(
            {"paths": [str(exam_dir / "question.pdf")], "question": "x"})
        assert mode == "FILE_ERROR" and llm == []


class TestEndToEndWithoutTheModel:

    def test_folder_then_sim_answers_every_question(self, tmp_path, exam_dir,
                                                    monkeypatch, llm):
        from tests.test_dispatch_parity import _build_real_orch
        o = _build_real_orch(tmp_path)
        o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
        o.background = False
        contents = {"question.pdf": PAPER, "text.pdf": BOOKLET}
        o.reader.read_path.side_effect = lambda p: contents[os.path.basename(p)]
        _r, mode = o._process_core(
            f"cortex, there is two pdf in this folder {exam_dir} , read them "
            "and answer every questions")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        reply, mode = o._process_core("yes")
        assert mode == "LOCAL_RAG" and len(llm) == 4
        assert "**4.** resposta 4" in reply
        o.engine_analytical.chat.assert_not_called()
        o.engine_analytical.chat_with_tools.assert_not_called()
