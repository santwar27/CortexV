"""Exam answers written on a COPY of the paper (s85).

Pedro asked for both: the answers in the chat, and the same answers on the
PDF, where a candidate would write them. The original is never touched, and
a question with no safe place to write is left blank and reported.
"""
import hashlib
import os

import numpy as np
import pytest

from src.tools import ocr
from src.tools.pdf_answer_writer import (
    answered_copy_path, chosen_letters, clean_answer, dotted_lines, parts_of,
    table_values,
)

S = 2.0


class TestAnswerParsing:

    def test_reasoning_is_not_written(self):
        assert clean_answer("D friendly\n\nReasoning: it sounds warm.") == "D friendly"

    def test_parts_at_their_markers(self):
        assert parts_of("(1) More productive.\n(2) Larger profits.") == [
            "More productive.", "Larger profits."]
        assert parts_of("Text A: one\n\nText B: two\n\nText C: three") == [
            "one", "two", "three"]
        assert parts_of("(1) Language feature: Idiom\nExample: 'fly by'") == [
            "Idiom", "'fly by'"]
        assert parts_of("7 (a) College Gazette\n7 (b) A footnote") == [
            "College Gazette", "A footnote"]

    def test_parts_without_markers_split_at_blank_lines(self):
        assert parts_of("first idea\n\nsecond idea") == ["first idea", "second idea"]

    def test_chosen_letters(self):
        assert chosen_letters("B 'to sweeten the deal'\nD 'more time'",
                              ["A", "B", "C", "D", "E"]) == ["B", "D"]
        assert chosen_letters("D friendly", ["A", "B", "C", "D"]) == ["D"]
        assert chosen_letters("A", ["A", "B"]) == ["A"]
        assert chosen_letters("Both texts use statistics.", ["A", "B"]) == []

    def test_table_values(self):
        assert table_values("1. Key findings (Paragraph 1)\n"
                            "3. Role of IT (Paragraph 3)") == {
            "key findings": "1", "role of it": "3"}
        assert table_values("Working hours - 2") == {"working hours": "2"}


class TestCopyPath:

    def test_beside_the_paper(self, tmp_path):
        paper = tmp_path / "question.pdf"
        assert answered_copy_path(str(paper)) == str(tmp_path / "question - respostas.pdf")
        assert answered_copy_path(str(paper), "en") == str(tmp_path / "question - answers.pdf")

    def test_never_an_existing_file(self, tmp_path):
        paper = tmp_path / "question.pdf"
        (tmp_path / "question - respostas.pdf").write_bytes(b"x")
        (tmp_path / "question - respostas (2).pdf").write_bytes(b"x")
        assert answered_copy_path(str(paper)).endswith("question - respostas (3).pdf")


def _page(rows=400, cols=1224):
    return np.full((rows, cols), 255, dtype=np.uint8)


class TestDottedLines:
    """A dotted answer line: thin, light, even, with gaps. Measured on the
    real paper — and the first real run also found a table border passing
    as one, which is why gaps are required."""

    W, H = 612.0, 200.0

    def test_a_dotted_row_is_found(self):
        a = _page()
        a[200:202, 160:1100:4] = 120            # a dot every 2 pt
        [(y, x0, x1)] = dotted_lines(a, self.W, self.H)
        assert abs(y - (self.H - 202 / S)) < 1.5
        assert x0 < 90 and x1 > 520

    def test_a_solid_rule_is_not_an_answer_line(self):
        a = _page()
        a[200:201, 160:1100] = 200               # thin grey border
        assert dotted_lines(a, self.W, self.H) == []

    def test_text_is_not_an_answer_line(self):
        a = _page()
        rng = np.random.default_rng(1)
        a[190:212, 160:1100] = np.where(rng.random((22, 940)) < 0.3, 40, 255)
        assert dotted_lines(a, self.W, self.H) == []

    def test_a_short_run_of_dots_is_not_a_line(self):
        a = _page()
        a[200:202, 160:300:4] = 120              # 35 pt
        assert dotted_lines(a, self.W, self.H) == []


def _make_paper(path):
    """A one-page exam in the Pearson layout: an answer question with two
    labelled dotted lines, a multiple-choice question with tick boxes."""
    import AppKit
    import Quartz
    from Foundation import NSURL

    box = Quartz.CGRectMake(0, 0, 612, 792)
    ctx = Quartz.CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(str(path)), box, None)
    Quartz.CGContextBeginPage(ctx, box)
    ns = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(ns)
    attrs = {AppKit.NSFontAttributeName: AppKit.NSFont.fontWithName_size_("Helvetica", 11)}

    def text(x, y, s):
        AppKit.NSString.stringWithString_(s).drawAtPoint_withAttributes_((x, y), attrs)

    dots = "." * 140
    text(72, 700, "1  Using Text A, identify two ways that businesses benefit.")
    text(72, 670, "1"); text(84, 670, dots)
    text(84, 645, dots); text(520, 645, "(1)")
    text(72, 620, "2"); text(84, 620, dots)
    text(84, 595, dots); text(520, 595, "(1)")
    text(330, 570, "(Total for Question 1 = 2 marks)")
    text(72, 530, "2  Which word best describes the style of Text A?")
    for y, option in ((505, "A  serious"), (480, "B  humorous")):
        Quartz.CGContextSetRGBStrokeColor(ctx, 0.3, 0.3, 0.3, 1)
        Quartz.CGContextStrokeRect(ctx, Quartz.CGRectMake(92, y + 1, 9, 9))
        text(112, y, option)
    text(330, 455, "(Total for Question 2 = 1 mark)")
    AppKit.NSGraphicsContext.restoreGraphicsState()
    Quartz.CGContextEndPage(ctx)
    Quartz.CGPDFContextClose(ctx)
    del ctx


@pytest.mark.skipif(not ocr.available(), reason="needs macOS Vision")
class TestWritingACopy:

    def _write(self, tmp_path, answers):
        from src.tools.pdf_answer_writer import write_answers
        paper = tmp_path / "question.pdf"
        _make_paper(paper)
        before = hashlib.sha256(paper.read_bytes()).hexdigest()
        out = tmp_path / "question - respostas.pdf"
        result = write_answers(str(paper), answers, str(out))
        assert hashlib.sha256(paper.read_bytes()).hexdigest() == before   # untouched
        return result, out

    def test_answers_land_on_their_lines_and_the_box_is_crossed(self, tmp_path):
        result, out = self._write(tmp_path, {
            1: "(1) More productive workers.\n(2) Larger profits for companies.",
            2: "B humorous",
        })
        assert result.written == [1, 2] and result.skipped == [] and result.crosses == 1
        from pypdf import PdfReader
        found = {}

        def visit(text, _cm, tm, _font, _size):
            if text.strip():      # the PDF keeps "fi" as a ligature
                found[text.strip().replace("\ufb01", "fi")] = tm[5]

        PdfReader(str(out)).pages[0].extract_text(visitor_text=visit)
        first = next(y for t, y in found.items() if t.startswith("More productive"))
        second = next(y for t, y in found.items() if t.startswith("Larger profits"))
        assert 660 < first < 680          # on the line labelled 1
        assert 610 < second < 630         # on the line labelled 2
        assert any("confira" in t for t in found)

    def test_the_cross_is_inside_the_chosen_box(self, tmp_path):
        import Quartz
        from Foundation import NSURL
        _result, out = self._write(tmp_path, {2: "B humorous"})
        doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(out)))
        img = ocr._render_page(Quartz, Quartz.CGPDFDocumentGetPage(doc, 1), 2.0)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        bpr = Quartz.CGImageGetBytesPerRow(img)
        px = np.frombuffer(bytes(data), dtype=np.uint8).reshape(-1, bpr)
        px = px[:, : Quartz.CGImageGetWidth(img) * 4].reshape(px.shape[0], -1, 4)

        def blue(x0, x1, y0, y1):          # points, PDF origin bottom-left
            H = px.shape[0]
            region = px[int(H - y1 * 2):int(H - y0 * 2), int(x0 * 2):int(x1 * 2)]
            return int(((region[..., 2] > 150) & (region[..., 0] < 90)).sum())

        assert blue(90, 103, 479, 492) > 20       # B is crossed
        assert blue(90, 103, 504, 517) == 0       # A is not

    def test_a_question_with_nowhere_to_write_is_left_blank(self, tmp_path):
        result, _out = self._write(tmp_path, {1: "(1) a\n(2) b", 7: "anything"})
        assert result.written == [1] and result.skipped == [7]

    def test_never_over_an_existing_file(self, tmp_path):
        from src.tools.pdf_answer_writer import write_answers
        paper = tmp_path / "question.pdf"
        _make_paper(paper)
        out = tmp_path / "taken.pdf"
        out.write_bytes(b"mine")
        with pytest.raises(FileExistsError):
            write_answers(str(paper), {1: "x"}, str(out))
        assert out.read_bytes() == b"mine"

    def test_never_over_a_file_made_while_it_reads(self, tmp_path, monkeypatch):
        """The exists() check comes seconds before the drawing — a second run
        writing the same name in between was overwritten (s88, Codex lote 3)."""
        from src.tools import pdf_answer_writer as w
        paper = tmp_path / "question.pdf"
        _make_paper(paper)
        out = tmp_path / "question - respostas.pdf"
        real = w.read_pages

        def read_pages_while_another_run_writes(p):
            out.write_bytes(b"the other run")
            return real(p)
        monkeypatch.setattr(w, "read_pages", read_pages_while_another_run_writes)
        with pytest.raises(FileExistsError):
            w.write_answers(str(paper), {1: "x"}, str(out))
        assert out.read_bytes() == b"the other run"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "question - respostas.pdf", "question.pdf"]      # no temp left behind


# ---------------------------------------------------------------- the routes

from tests.test_exam_reader import BOOKLET, PAPER  # noqa: E402


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
    from src.core.orchestrator import CortexOrchestrator
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._pending_query = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o.background = False
    o.background_origin = "background"
    o._reply_lang = "pt"
    contents = {"question.pdf": PAPER, "text.pdf": BOOKLET}
    monkeypatch.setattr(o, "reader", type("R", (), {
        "read_path": lambda _s, p: contents[os.path.basename(p)]})(), raising=False)
    o.engine_analytical = type("E", (), {"model_name": "cortex-lite"})()
    monkeypatch.setattr("src.core.staging_mixin.ollama.chat",
                        lambda **kw: {"message": {"content": "B humorous"}})
    return o


def _answered(orch, exam_dir, ask="read them and answer every question"):
    orch._handle_user_paths(f"two pdf in {exam_dir} , {ask}")
    return orch._execute_pending_action()


class TestBothOptions:

    def test_the_answers_end_with_the_offer(self, orch, exam_dir):
        reply, mode = _answered(orch, exam_dir)
        assert mode == "LOCAL_RAG"
        assert reply.rstrip().endswith('"sim" (ou, mais tarde, "escreve no pdf").')
        assert orch._last_exam["paper"].endswith("question.pdf")
        assert sorted(orch._last_exam["answers"]) == [1, 2, 3, 4]

    def test_asked_up_front_the_copy_is_offered_for_confirmation(self, orch,
                                                                  exam_dir):
        reply, mode = _answered(
            orch, exam_dir, "read them, answer every question and write the "
                            "answers on the pdf")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["type"] == "write_answers_pdf"
        assert "question - respostas.pdf" in reply and "não é alterado" in reply

    @pytest.mark.parametrize("ask", ["escreve no pdf", "escreve as respostas no pdf",
                                     "fill in the pdf", "write them on the pdf",
                                     "preenche a prova"])
    def test_asked_afterwards(self, orch, exam_dir, ask):
        _answered(orch, exam_dir)
        assert orch._mentions_files(ask)           # the stream steps aside
        _reply, mode = orch._handle_user_paths(ask)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        args = orch._pending_action["args"]
        assert args["out"].endswith("question - respostas.pdf")
        assert sorted(args["answers"]) == ["1", "2", "3", "4"]

    def test_not_without_an_exam(self, orch):
        assert not orch._mentions_files("escreve no pdf")
        assert orch._handle_user_paths("escreve no pdf") is None

    def test_a_long_sentence_is_not_the_follow_up(self, orch, exam_dir):
        _answered(orch, exam_dir)
        long = ("can you explain how I should write a good answer for this kind "
                "of exam question next time I practise")
        assert not orch._mentions_files(long)

    def test_the_offer_expires(self, orch, exam_dir):
        _answered(orch, exam_dir)
        orch._last_exam["when"] -= orch._LAST_EXAM_TTL_S + 1
        assert orch._handle_user_paths("escreve no pdf") is None


class TestTheWriteAction:

    def _staged(self, orch, exam_dir):
        _answered(orch, exam_dir)
        orch._handle_user_paths("escreve no pdf")
        return orch._pending_action["args"]

    def test_sim_writes_with_the_answers_shown(self, orch, exam_dir, monkeypatch):
        from src.tools.pdf_answer_writer import WriteResult
        seen = {}

        def fake_write(paper, answers, out, lang):
            seen.update(paper=paper, answers=answers, out=out)
            return WriteResult(path=out, written=[1, 2, 3], skipped=[4], crosses=1)

        monkeypatch.setattr("src.tools.pdf_answer_writer.write_answers", fake_write)
        args = self._staged(orch, exam_dir)
        reply, mode = orch._execute_pending_action()
        assert mode == "AGENT_ACTION_DONE"
        assert seen["out"] == args["out"] and seen["answers"][2] == "B humorous"
        assert "escrevi 3 respostas" in reply and "questão 4" in reply

    def test_an_existing_copy_is_never_overwritten(self, orch, exam_dir,
                                                   monkeypatch):
        from src.tools.pdf_answer_writer import WriteResult
        monkeypatch.setattr("src.tools.pdf_answer_writer.write_answers",
                            lambda p, a, out, l: WriteResult(path=out, written=[1]))
        args = self._staged(orch, exam_dir)
        (exam_dir / "question - respostas.pdf").write_bytes(b"made by hand")
        reply, _ = orch._execute_pending_action()
        assert (exam_dir / "question - respostas.pdf").read_bytes() == b"made by hand"
        assert "(2).pdf" in reply

    def test_the_copy_must_sit_beside_the_paper(self, orch, exam_dir, home):
        args = self._staged(orch, exam_dir)
        args["out"] = str(home / "elsewhere.pdf")
        orch._pending_action["args"] = args
        _reply, mode = orch._execute_pending_action()
        assert mode == "AGENT_ACTION_ERROR"

    def test_an_unseen_paper_is_refused(self, orch, home):
        _reply, mode = orch._run_write_answers_pdf({
            "paper": str(home / "x.pdf"), "out": str(home / "x - respostas.pdf"),
            "answers": {"1": "a"}})
        assert mode == "AGENT_ACTION_ERROR"

    def test_the_preview(self):
        from src.core.tool_risk import build_preview, is_destructive
        assert is_destructive("write_answers_pdf")
        human = build_preview("write_answers_pdf", {
            "paper": "/Users/p/13-10/question.pdf",
            "out": "/Users/p/13-10/question - respostas.pdf",
            "answers": {"1": "a", "2": "b"}}, lang="pt")["human"]
        assert human.startswith("Posso criar 'question - respostas.pdf' na pasta 13-10?")
        assert "o original (question.pdf) não é alterado" in human


def test_a_name_taken_before_the_yes_is_asked_again(tmp_path, monkeypatch):
    """The preview said "question - respostas.pdf"; another file took that
    name before the "sim", and the copy went to "(2)" unapproved (s88,
    Codex lote 2). Now it asks again with the free name."""
    import time
    from tests.test_dispatch_parity import _build_real_orch
    import src.tools.pdf_answer_writer as w
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / "Downloads"
    d.mkdir()
    paper = d / "question.pdf"
    paper.write_bytes(b"%PDF-1.4")
    o = _build_real_orch(tmp_path)
    o.background = False
    o._reply_lang = "pt"
    o._remember_seen_path(str(paper))
    o._last_exam = {"when": time.monotonic(), "paper": str(paper),
                    "answers": {1: "a"}, "lang": "pt"}
    o._stage_write_answers()
    (d / "question - respostas.pdf").write_bytes(b"someone else's")
    wrote = []

    def fake_write(paper, answers, out, lang="pt"):
        wrote.append(os.path.basename(out))
        return type("R", (), {"path": out, "written": [1], "skipped": [], "crosses": 0})()
    monkeypatch.setattr(w, "write_answers", fake_write)

    reply, mode = o.process("sim")
    assert mode == "AGENT_AWAITING_CONFIRMATION" and wrote == []
    assert "question - respostas (2).pdf" in reply
    _reply, mode = o.process("sim")
    assert mode == "AGENT_ACTION_DONE" and wrote == ["question - respostas (2).pdf"]


class TestTheHeaderNeverCoversPrint:
    """"Respostas do Cortex — confira antes de usar" was drawn at the top
    whatever was printed there — over a paper's own title (s88, Codex
    lote 3). It takes the top band if free, else the bottom one."""

    def _titled(self, tmp_path):
        import AppKit
        import Quartz
        from Foundation import NSURL
        base = tmp_path / "base.pdf"
        _make_paper(base)
        paper = tmp_path / "question.pdf"
        page = Quartz.CGPDFDocumentGetPage(
            Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(base))), 1)
        box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
        ctx = Quartz.CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(str(paper)), box, None)
        Quartz.CGContextBeginPage(ctx, box)
        Quartz.CGContextDrawPDFPage(ctx, page)
        ns = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
        AppKit.NSGraphicsContext.saveGraphicsState()
        AppKit.NSGraphicsContext.setCurrentContext_(ns)
        AppKit.NSString.stringWithString_("Functional Skills English — Level 2").drawAtPoint_withAttributes_(
            (61, 770), {AppKit.NSFontAttributeName: AppKit.NSFont.fontWithName_size_("Helvetica", 10)})
        AppKit.NSGraphicsContext.restoreGraphicsState()
        Quartz.CGContextEndPage(ctx)
        Quartz.CGPDFContextClose(ctx)
        return paper

    @staticmethod
    def _ink(path, top: bool) -> int:
        import Quartz
        from Foundation import NSURL
        from src.tools import pdf_answer_writer as w
        page = Quartz.CGPDFDocumentGetPage(
            Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(path))), 1)
        a, W, _H = w._gray(Quartz, page)
        rows = slice(0, int(30 * w._S)) if top else slice(a.shape[0] - int(30 * w._S), None)
        return int((a[rows, :int(0.7 * W * w._S)] < 160).sum())

    def test_a_printed_title_sends_it_to_the_bottom(self, tmp_path):
        from src.tools.pdf_answer_writer import write_answers
        paper = self._titled(tmp_path)
        out = tmp_path / "question - respostas.pdf"
        write_answers(str(paper), {1: "x", 2: "B"}, str(out))
        assert self._ink(out, top=True) == self._ink(paper, top=True)
        assert self._ink(out, top=False) > self._ink(paper, top=False)

    def test_a_free_top_band_keeps_it_on_top(self, tmp_path):
        from src.tools.pdf_answer_writer import write_answers
        paper = tmp_path / "question.pdf"
        _make_paper(paper)
        out = tmp_path / "question - respostas.pdf"
        write_answers(str(paper), {1: "x", 2: "B"}, str(out))
        assert self._ink(out, top=True) > self._ink(paper, top=True)
        assert self._ink(out, top=False) == self._ink(paper, top=False)
