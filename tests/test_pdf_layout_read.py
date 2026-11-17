"""Fixed reader choice, with real PDF text layers and a contained-runner double."""
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from pypdf import PdfReader
from pypdf._page import PageObject

from src.core import pdf_runner, pdf_sandbox
from src.core.audit import audit
from src.core.exam_reader import parse_exam
from src.tools.local_reader import LocalFileReader, PDF_NO_TEXT, read_by_ocr


NOTICE = "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"
WORDS = ["word" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(100)]


def _pdf(path, *pages):
    """Real Helvetica pages, following test_exam_text_layer's PDF helper."""
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                 for line in text.split("\n")]
        stream = ("BT /F1 10 Tf 12 TL 40 800 Td\n" +
                  "\n".join(f"({line}) Tj T*" for line in lines) + "\nET").encode()
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                        "/Resources << /Font << /F1 3 0 R >> >> "
                        f"/Contents {5 + 2 * i} 0 R >>").encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() +
                       stream + b"\nendstream")
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for n, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{n} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend((f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                 f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(data)
    return str(path)


def _run(*chunks, status="exited", code=0, stdout=None):
    if stdout is None:
        stdout = "".join(chunk + "\f" for chunk in chunks).encode()
    return pdf_runner.ToolRun(status, code, .01, stdout, "private diagnostic", [])


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_PDF_POPPLER", "1")
    monkeypatch.setattr(LocalFileReader, "_ocr_text", staticmethod(lambda _p: ""))
    tool, events = Mock(), Mock()
    monkeypatch.setattr(pdf_runner, "run_tool", tool)
    monkeypatch.setattr(audit, "log", events)
    return LocalFileReader(str(tmp_path)), tool, events


def _off(reader, path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("CORTEX_PDF_POPPLER", "0")
        return reader.read_path(path)


def _event(events, *, reader="pypdf", poppler="ok", pages=1, words=100, lost=0, glued=0):
    events.assert_called_once_with("–", "pdf", "read_choice", payload={
        "reader": reader, "poppler": poppler, "pages": pages,
        "words": words, "lost": lost, "glued": glued})
    payload = events.call_args.kwargs["payload"]
    assert all(type(payload[key]) is int for key in ("pages", "words", "lost", "glued"))


@pytest.mark.parametrize("long", [False, True])
def test_lost_words_choose_exact_cleaned_squeezed_and_capped_text(fake, tmp_path, long):
    reader, tool, events = fake
    path = _pdf(tmp_path / "paper.pdf", " ".join(WORDS[:80]), " ".join(WORDS[:20]))
    first = "    ".join(WORDS) + "\n....DO NOT WRITE IN THIS AREA\t\t\t\tEND\n"
    expected = "   ".join(WORDS) + "\n …   END\n"
    if long:
        first += " tail" * 4000
        expected += " tail" * 4000
    second = "  ".join(WORDS[:20])
    tool.return_value = _run(first, second)
    expected += "\n" + second + "\n"
    if long:
        expected = expected[:15000] + NOTICE
    assert reader.read_path(path) == expected
    assert tool.call_count == events.call_count == 1


@pytest.mark.parametrize("lost,glued,chosen", [
    (5, 0, "poppler"), (0, 10, "poppler"),
    (4, 0, "pypdf"), (0, 9, "pypdf"), (4, 9, "pypdf"), (0, 0, "pypdf"),
])
def test_exact_thresholds_and_audit(fake, tmp_path, monkeypatch, lost, glued, chosen):
    reader, tool, events = fake
    raw = " ".join(WORDS[lost + glued:])
    if glued:
        raw += " " + "".join(WORDS[lost:lost + glued])
    path = _pdf(tmp_path / "threshold.pdf", raw)
    baseline = _off(reader, path, monkeypatch)
    poppler = " ".join(WORDS)
    tool.return_value = _run(poppler)
    assert reader.read_path(path) == (poppler + "\n" if chosen == "poppler" else baseline)
    _event(events, reader=chosen, lost=lost, glued=glued)


def test_below_fifty_distinct_words_keeps_pypdf(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    path = _pdf(tmp_path / "short.pdf", " ".join(WORDS[:25]))
    tool.return_value = _run((" ".join(WORDS[:49]) + " ") * 3)
    assert reader.read_path(path) == _off(reader, path, monkeypatch)
    _event(events, words=49, lost=24)


def test_same_words_are_byte_identical_one_staged_run_and_one_extraction(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    pages = ["    ".join(WORDS[:50]), " ".join(WORDS[50:])]
    path = _pdf(tmp_path / "same.pdf", *pages)
    baseline = _off(reader, path, monkeypatch)
    stage = Mock(wraps=pdf_runner.PdfJob.stage)
    monkeypatch.setattr(pdf_runner.PdfJob, "stage", lambda job, paths: stage(job, paths))
    extract = PageObject.extract_text
    seen = []

    def capture(page, *args, **kwargs):
        seen.append(page)
        return extract(page, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", capture)
    tool.return_value = _run(*(" ".join(page.split()) for page in pages))
    assert reader.read_path(path) == baseline
    assert len(seen) == 2 and seen[0] is not seen[1]
    tool.assert_called_once()
    job, name, args = tool.call_args.args
    stage.assert_called_once_with(job, [path])
    assert name == "pdftotext"
    assert args == ["-layout", "-enc", "UTF-8", "-eol", "unix", "-f", "1", "-l", "2",
                    str(job.input_dir / "input-1.pdf"), "-"]
    assert tool.call_args.kwargs == {}
    assert not job.root.exists()
    _event(events, pages=2)


@pytest.mark.parametrize("failure", [
    _run(status="unavailable", code=None), _run(status="timeout", code=None),
    _run(status="stopped", code=None), _run(status="too_big", code=None),
    _run(status="error", code=None),
    _run(code=1), _run(stdout=b"\xff\f"), _run(stdout=b"no terminator"),
    _run(stdout=b"one\ftwo\f"), _run(stdout=b"one\ftrailing"),
    RuntimeError("unexpected runner exception"),
])
def test_failures_keep_original_and_audit_once(fake, tmp_path, monkeypatch, caplog, failure):
    reader, tool, events = fake
    path = _pdf(tmp_path / "failure.pdf", "    ".join(WORDS))
    baseline = _off(reader, path, monkeypatch)
    if isinstance(failure, Exception):
        tool.side_effect = failure
    else:
        tool.return_value = failure
    assert reader.read_path(path) == baseline
    tool.assert_called_once()
    if isinstance(failure, Exception):
        assert any(record.levelname == "WARNING" for record in caplog.records)
    _event(events, poppler="unavailable" if getattr(failure, "status", "") == "unavailable"
           else "failed", words=0)


@pytest.mark.parametrize("phase", ["stage", "enter", "exit"])
def test_job_exceptions_fail_soft_with_warning(fake, tmp_path, monkeypatch, caplog, phase):
    reader, tool, events = fake
    path = _pdf(tmp_path / "refused.pdf", " ".join(WORDS[:80]))
    baseline = _off(reader, path, monkeypatch)
    method = {"stage": "stage", "enter": "__enter__", "exit": "__exit__"}[phase]
    original = getattr(pdf_runner.PdfJob, method)

    def fail(job, *args):
        if phase == "exit":
            original(job, *args)
        raise ValueError("fictitious staging or lifecycle refusal")

    monkeypatch.setattr(pdf_runner.PdfJob, method, fail)
    tool.return_value = _run(" ".join(WORDS))
    assert reader.read_path(path) == baseline
    assert tool.call_count == (1 if phase == "exit" else 0)
    assert any(record.levelname == "WARNING" for record in caplog.records)
    _event(events, poppler="failed", words=0)


@pytest.mark.parametrize("switch", ["0", "false", "off", " FALSE ", "Off"])
def test_switch_off_never_attempts(fake, tmp_path, monkeypatch, switch):
    reader, tool, events = fake
    path = _pdf(tmp_path / "off.pdf", " ".join(WORDS))
    monkeypatch.setenv("CORTEX_PDF_POPPLER", switch)
    reader.read_path(path)
    tool.assert_not_called()
    events.assert_not_called()


def test_switch_defaults_on_at_call_time(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    path = _pdf(tmp_path / "default.pdf", " ".join(WORDS[:80]))
    baseline = _off(reader, path, monkeypatch)
    monkeypatch.delenv("CORTEX_PDF_POPPLER")
    tool.return_value = _run(" ".join(WORDS))
    assert reader.read_path(path) != baseline
    _event(events, reader="poppler", lost=20)


@pytest.mark.parametrize("count", [0, 20, 21])
def test_visited_page_limit(fake, tmp_path, monkeypatch, count):
    reader, tool, events = fake
    path = _pdf(tmp_path / "many.pdf", *([" ".join(WORDS)] * count))
    baseline = _off(reader, path, monkeypatch)
    tool.return_value = _run(*([" ".join(WORDS)] * count))
    assert reader.read_path(path) == baseline
    assert tool.call_count == events.call_count == (1 if count == 20 else 0)


def test_cap_on_third_of_five_pages_reads_only_three(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    raw = "ordinary " * 600
    path = _pdf(tmp_path / "cap.pdf", *([raw] * 5))
    baseline = _off(reader, path, monkeypatch)
    tool.return_value = _run(raw, raw, raw)
    assert reader.read_path(path) == baseline
    assert tool.call_args.args[2][8] == "3"
    _event(events, pages=3, words=3)


def test_visited_pages_continue_past_the_raw_ocr_sample_cap(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    first = " ".join(WORDS) + ". " * 8000
    rest = " ".join(WORDS)
    path = _pdf(tmp_path / "raw-cap.pdf", first, "", rest)
    baseline = _off(reader, path, monkeypatch)
    tool.return_value = _run(rest, "", rest)
    assert reader.read_path(path) == baseline
    assert tool.call_args.args[2][8] == "3"
    _event(events, pages=3, words=200)


@pytest.mark.parametrize("length", [14999, 15000])
def test_poppler_cap_boundary_includes_page_newline(fake, tmp_path, length):
    reader, tool, events = fake
    path = _pdf(tmp_path / "boundary.pdf", " ".join(WORDS[:80]))
    prefix = " ".join(WORDS) + " "
    chunk = prefix + "z" * (length - len(prefix))
    tool.return_value = _run(chunk)
    expected = chunk + "\n"
    if len(expected) > 15000:
        expected = expected[:15000] + NOTICE
    assert reader.read_path(path) == expected
    _event(events, reader="poppler", words=101, lost=21)


@pytest.mark.parametrize("raw,ocr", [("", ""), ("", "recognised words " * 20),
                                        ("barcode", "recognised words " * 20)])
def test_ocr_results_never_compare(fake, tmp_path, monkeypatch, raw, ocr):
    reader, tool, events = fake
    path = _pdf(tmp_path / "scan.pdf", raw)
    monkeypatch.setattr(reader, "_ocr_text", lambda _p: ocr)
    result = reader.read_path(path)
    assert read_by_ocr(result) if ocr else result == PDF_NO_TEXT
    tool.assert_not_called()
    events.assert_not_called()


def test_compare_raw_pages_not_cleaned_and_sum_per_page(fake, tmp_path):
    reader, tool, events = fake
    raw = " ".join(WORDS[:21]) + " DO NOT WRITE IN THIS AREA"
    path = _pdf(tmp_path / "raw.pdf", raw, raw)
    tool.return_value = _run(raw.lower(), raw.lower())
    reader.read_path(path)
    _event(events, pages=2, words=50)


def test_words_on_other_pages_do_not_hide_loss(fake, tmp_path):
    reader, tool, events = fake
    first, second = " ".join(WORDS[:50]), " ".join(WORDS[50:])
    path = _pdf(tmp_path / "pages.pdf", first, second)
    tool.return_value = _run(second, first)
    assert reader.read_path(path) == second + "\n" + first + "\n"
    _event(events, reader="poppler", pages=2, lost=100)


def test_nfkc_case_and_word_definition(fake, tmp_path):
    reader, tool, events = fake
    raw = " ".join(WORDS).upper() + " office 123 __ ab a1b"
    path = _pdf(tmp_path / "unicode.pdf", raw)
    # Compatibility ligature, full-width letters, digits, underscores and short words.
    poppler = " ".join(WORDS) + " oﬃce １２３ __ ａｂ a1b"
    tool.return_value = _run(poppler)
    expected = reader.read_path(path)
    assert expected == PdfReader(path).pages[0].extract_text() + "\n"
    _event(events, words=101)


def test_glued_fingerprint_removes_nonword_but_keeps_digits_and_underscores(fake, tmp_path):
    reader, tool, events = fake
    raw = " ".join(WORDS[:80]) + " word-dc word_zz word9zy"
    path = _pdf(tmp_path / "fingerprint.pdf", raw)
    # worddc is present across a hyphen; wordzz and wordzy are lost across _ and 9.
    tool.return_value = _run(" ".join(WORDS[:80]) + " worddc wordzz wordzy")
    reader.read_path(path)
    _event(events, words=83, lost=2, glued=1)


def test_audit_failure_does_not_change_chosen_text(fake, tmp_path):
    reader, tool, events = fake
    path = _pdf(tmp_path / "audit.pdf", " ".join(WORDS[:80]))
    text = " ".join(WORDS)
    tool.return_value = _run(text)
    events.side_effect = RuntimeError("audit unavailable")
    assert reader.read_path(path) == text + "\n"
    assert events.call_count == 1


def test_exam_switches_paper_only_and_preserves_structure(fake, tmp_path, monkeypatch):
    reader, tool, events = fake
    paper = ("Practice paper\nQuestions 1 to 2 are based on Text A.\n"
             "Questions 3 to 4 are based on Text B.\n" + " ".join(WORDS[:80]) + "\n" +
             "\n".join(f"{n} Which word is used?\n(Total for Question {n} = 1 mark)"
                       for n in range(1, 5)))
    booklet = ("Text AWebpage\n" + " ".join(WORDS[:50]) +
               "\nText BArticle\n" + " ".join(WORDS[50:]))
    paths = [_pdf(tmp_path / "paper.pdf", paper), _pdf(tmp_path / "booklet.pdf", booklet)]
    baseline_docs = [_off(reader, path, monkeypatch) for path in paths]
    baseline = parse_exam(baseline_docs)
    # Extra paper labels precede the first question; booklet's layout loses headings.
    paper_layout = paper.replace("Practice paper", "Practice paper " + " ".join(WORDS[80:]))
    tool.side_effect = [_run(paper_layout), _run(booklet.replace("Text A", "Text     A")
                                               .replace("Text B", "Text     B"))]
    docs = [reader.read_path(path) for path in paths]
    exam = parse_exam(docs)
    assert docs[0] == paper_layout + "\n"
    assert docs[1] == baseline_docs[1]
    assert baseline is not None and exam is not None
    assert exam.questions == baseline.questions
    assert exam.texts == baseline.texts
    assert [(q.number, q.texts) for q in exam.questions] == [
        (1, ["A"]), (2, ["A"]), (3, ["B"]), (4, ["B"])]
    assert [call.kwargs["payload"]["reader"] for call in events.call_args_list] == [
        "poppler", "pypdf"]


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                    or not Path(pdf_sandbox.SANDBOX_EXEC).exists()
                    or pdf_sandbox.discover() is None,
                    reason="requires opt-in real sandbox-exec and Poppler")
def test_real_ordinary_pdf_keeps_pypdf(tmp_path, monkeypatch):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    path = _pdf(tmp_path / "fictitious.pdf", "\n".join(
        " ".join(WORDS[i:i + 10]) for i in range(0, 100, 10)))
    reader = LocalFileReader(str(tmp_path))
    baseline = _off(reader, path, monkeypatch)
    events = Mock()
    monkeypatch.setattr(audit, "log", events)
    monkeypatch.setenv("CORTEX_PDF_POPPLER", "1")
    assert reader.read_path(path) == baseline
    choices = [call for call in events.call_args_list if call.args[1:3] == ("pdf", "read_choice")]
    assert len(choices) == 1
    captured = Mock()
    captured(*choices[0].args, **choices[0].kwargs)
    _event(captured)
