"""Fixed PDF facts, bounded page coverage, and opt-in contained Poppler checks."""
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import pdf_info, pdf_runner, pdf_sandbox


TEXT = "Private fictitious content with at least forty visible characters."
NORMAL = (b"Title: private internal filename.pdf\nPages:           3\nEncrypted:       no\n"
          b"Page size:       612 x 792 pts (letter)\n")
OWNER = b"Pages: 3\nEncrypted: yes (print:no copy:no change:yes addNotes:yes algorithm:RC4)\n"
DETAILS = {
    "too_big": "PDF inspection limit exceeded.",
    "timeout": "PDF inspection timed out.",
    "stopped": "PDF inspection stopped.",
    "unavailable": "Contained PDF tools unavailable.",
    "error": "PDF inspection failed.",
}


def run(stdout=b"", *, status="exited", code=0, stderr=""):
    return pdf_runner.ToolRun(status, code, 0.01, stdout, stderr, [], "private runner detail")


def info(status="ok", pages=None, encrypted=None, restrictions=(), classes=(), detail=""):
    return pdf_info.PdfInfo(status, pages, encrypted, restrictions, classes, len(classes), detail)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    source = tmp_path / "private-source.pdf"
    source.write_bytes(b"fictitious input; no uncontained parser")
    tool = Mock()
    monkeypatch.setattr(pdf_runner, "run_tool", tool)

    def inspect(*runs, cancelled=None):
        tool.side_effect = runs
        result = pdf_info.inspect(source, cancelled=cancelled)
        assert TEXT not in repr(result)
        assert "private" not in repr(result)
        assert "Incorrect password" not in repr(result)
        assert all(not call.args[0].root.exists() for call in tool.call_args_list)
        return result

    return source, tool, inspect


def test_normal_and_one_contained_job(fake, monkeypatch):
    source, tool, inspect = fake
    stage = pdf_runner.PdfJob.stage
    staged_calls = []

    def capture(job, paths):
        staged_calls.append((job, paths))
        return stage(job, paths)

    monkeypatch.setattr(pdf_runner.PdfJob, "stage", capture)
    cancelled = lambda: False
    result = inspect(run(NORMAL), run((TEXT + "\f").encode() * 3), cancelled=cancelled)
    assert result == info(pages=3, encrypted=False, classes=("text",) * 3,
                          detail="PDF inspected.")
    assert staged_calls == [(tool.call_args_list[0].args[0], [source])]
    first, second = tool.call_args_list
    job = first.args[0]
    staged = str(job.input_dir / "input-1.pdf")
    assert first.args == (job, "pdfinfo", [staged])
    assert second.args == (job, "pdftotext", ["-enc", "UTF-8", "-eol", "unix",
                                            "-f", "1", "-l", "3", staged, "-"])
    assert first.kwargs == second.kwargs == {"cancelled": cancelled}
    with pytest.raises(FrozenInstanceError):
        result.pages = 5


@pytest.mark.parametrize("metadata,expected", [
    (NORMAL, info(pages=3, encrypted=False, classes=("text", "thin", "none"),
                  detail="PDF inspected.")),
    (OWNER, info(pages=3, encrypted=True, restrictions=("copy", "print"),
                 classes=("text", "thin", "none"), detail="PDF inspected.")),
])
def test_mixed_and_owner_locked(fake, metadata, expected):
    result = fake[2](run(metadata), run((TEXT + "\fshort\f \n\t\f").encode()))
    assert result == expected
    assert pdf_info.scanned_pages(result) == (2, 3)
    assert pdf_info.has_text_layer(result) is False


@pytest.mark.parametrize("stderr,status,detail", [
    ("Command Line Error: Incorrect password", "protected", "PDF requires a password."),
    ("Syntax Warning: May not be a PDF\nSyntax Error: Couldn't find trailer dictionary",
     "broken", "PDF is broken."),
    ("Syntax Error: invalid xref", "broken", "PDF is broken."),
    ("Syntax Error (123): invalid object", "broken", "PDF is broken."),
    ("private arbitrary diagnostic", "error", DETAILS["error"]),
])
def test_pdfinfo_failures(fake, stderr, status, detail):
    assert fake[2](run(code=1, stderr=stderr)) == info(status, detail=detail)
    assert fake[1].call_count == 1


@pytest.mark.parametrize("pages", [None, "unknown", "3 junk", "0", "-1", "1.5", "٣", "3\nPages: 4"])
def test_invalid_page_count_never_checks_text(fake, pages):
    stdout = b"Encrypted: no\n" + (b"" if pages is None else f"Pages: {pages}\n".encode())
    assert fake[2](run(stdout)) == info("error", encrypted=False,
                                      detail="PDF page count unavailable.")
    assert fake[1].call_count == 1


@pytest.mark.parametrize("value,encrypted,restrictions", [
    ("no", False, ()), ("yes", True, ()), ("maybe", None, ()),
    ("no junk", None, ()), ("yes (copy:notno)", True, ()),
    ("yes (print:no copy:no change:no addNotes:no algorithm:RC4)", True,
     ("addNotes", "change", "copy", "print")),
])
def test_encryption_is_parsed_strictly(fake, value, encrypted, restrictions):
    metadata = f"Pages: 1\nEncrypted: {value}\n".encode()
    assert fake[2](run(metadata), run(b"\f")) == info(
        pages=1, encrypted=encrypted, restrictions=restrictions, classes=("none",),
        detail="PDF inspected.")


def test_missing_encryption_is_unknown(fake):
    assert fake[2](run(b"Pages: 1\n"), run(b"\f")) == info(
        pages=1, classes=("none",), detail="PDF inspected.")


def test_absurd_count(fake):
    assert pdf_info.MAX_DECLARED_PAGES == 2000
    assert fake[2](run(b"Pages: 2001\nEncrypted: no\n")) == info(
        "too_big", pages=2001, encrypted=False, detail="PDF declared page limit exceeded.")
    assert fake[1].call_count == 1


def test_excessive_integer_digits_are_bounded(fake):
    assert fake[2](run(b"Pages: " + b"9" * 5000 + b"\n")) == info(
        "too_big", detail="PDF declared page limit exceeded.")
    assert fake[1].call_count == 1


@pytest.mark.parametrize("pages", [20, 25, 2000])
def test_page_cap_and_helpers(fake, pages):
    assert pdf_info.MAX_PAGES_READ == 20
    result = fake[2](run(f"Pages: {pages}\nEncrypted: no\n".encode()),
                     run((TEXT + "\f").encode() * 20))
    assert result == info(pages=pages, encrypted=False, classes=("text",) * 20,
                          detail="PDF inspected.")
    assert fake[1].call_args.args[2][7] == "20"
    assert pdf_info.scanned_pages(result) == ()
    assert pdf_info.has_text_layer(result) is True
    assert pdf_info.has_text_layer(info()) is False


def test_visible_thresholds(fake):
    text = "\x00\u200b\t\n " + "é " * 40
    stdout = (text + "\f" + "x " * 39 + "\f" + "\x00\u200b\t\n \f").encode()
    assert fake[2](run(NORMAL), run(stdout)) == info(
        pages=3, encrypted=False, classes=("text", "thin", "none"), detail="PDF inspected.")


@pytest.mark.parametrize("status", list(DETAILS))
@pytest.mark.parametrize("phase", ["info", "text"])
def test_runner_statuses_keep_known_facts(fake, status, phase):
    failure = run(status=status, code=None, stderr="private diagnostic Incorrect password")
    runs = (failure,) if phase == "info" else (run(OWNER), failure)
    expected = info(status, detail=DETAILS[status]) if phase == "info" else info(
        status, 3, True, ("copy", "print"), detail="PDF text check failed. " + DETAILS[status])
    assert fake[2](*runs) == expected
    assert fake[1].call_count == len(runs)


@pytest.mark.parametrize("failure,status,detail", [
    (run(code=1, stderr="private diagnostic"), "error", "PDF text check failed."),
    (run(code=1, stderr="Command Line Error: Incorrect password"),
     "protected", "PDF text check failed. PDF requires a password."),
    (run(code=1, stderr="Syntax Error: Couldn't find trailer dictionary"),
     "broken", "PDF text check failed. PDF is broken."),
    (run(b""), "error", "PDF text check failed."),
    (run(b"short\f"), "error", "PDF text check failed."),
    (run(b"\f\f\f\f"), "error", "PDF text check failed."),
    (run(b"\f\f\fprivate trailing content"), "error", "PDF text check failed."),
    (run(b"\xff\f\f\f"), "error", "PDF text check failed."),
])
def test_text_failure_and_malformed_success(fake, failure, status, detail):
    assert fake[2](run(OWNER), failure) == info(status, 3, True, ("copy", "print"), detail=detail)


@pytest.mark.parametrize("message,status,detail", [
    ("PDF input bytes exceed the job limit", "too_big", "PDF input size limit exceeded."),
    ("PDF input must be a regular file", "error", "PDF input must be a regular file."),
    ("private unexpected staging failure", "error", "PDF input staging failed."),
])
def test_stage_refusals(fake, monkeypatch, message, status, detail):
    monkeypatch.setattr(pdf_runner.PdfJob, "stage", Mock(side_effect=ValueError(message)))
    assert fake[2]() == info(status, detail=detail)
    fake[1].assert_not_called()


@pytest.mark.parametrize("phase", ["enter", "info", "text", "exit"])
def test_exceptions_never_escape(fake, monkeypatch, phase):
    failure = RuntimeError("private exception text")
    if phase in ("enter", "exit"):
        method = "__enter__" if phase == "enter" else "__exit__"
        original = getattr(pdf_runner.PdfJob, method)

        def fail(job, *args):
            if phase == "exit":
                original(job, *args)
            raise failure

        monkeypatch.setattr(pdf_runner.PdfJob, method, fail)
    runs = (failure,) if phase == "info" else (run(NORMAL), failure)
    if phase == "exit":
        runs = (run(NORMAL), run((TEXT + "\f").encode() * 3))
    expected = info("error", detail=DETAILS["error"])
    if phase in ("text", "exit"):
        expected = info("error", 3, False, detail="PDF text check failed.")
    assert fake[2](*runs) == expected


def two_page_pdf():
    stream = f"BT /F1 12 Tf 20 100 Td ({TEXT}) Tj ET\n".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>"]
    pdf, offsets = b"%PDF-1.4\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(pdf)
    pdf += b"xref\n0 7\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return pdf + f"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


@pytest.mark.skipif(os.environ.get("CORTEX_REAL_SANDBOX_TESTS") != "1"
                    or not Path(pdf_sandbox.SANDBOX_EXEC).exists()
                    or pdf_sandbox.discover() is None,
                    reason="requires opt-in real sandbox-exec and Poppler")
@pytest.mark.parametrize("case", ["mixed", "broken", "password"])
def test_real_inspection(tmp_path, case):
    if not pdf_sandbox.available():
        pytest.skip("real sandbox unavailable in this environment")
    source = tmp_path / "fictitious.pdf"
    source.write_bytes(two_page_pdf() if case != "broken" else b"broken fictitious file")
    if case == "password":
        from pypdf import PdfWriter

        writer = PdfWriter(clone_from=source)
        writer.encrypt(user_password="fictitious-test-only")
        with source.open("wb") as output:
            writer.write(output)
    result = pdf_info.inspect(source)
    if case == "mixed":
        assert result == info(pages=2, encrypted=False, classes=("text", "none"),
                              detail="PDF inspected.")
    elif case == "broken":
        assert result == info("broken", detail="PDF is broken.")
    else:
        assert result == info("protected", detail="PDF requires a password.")
