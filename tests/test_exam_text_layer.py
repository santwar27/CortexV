"""Fictitious Pearson-shaped text layers: keep words, then split by marks."""
import pytest
from pypdf import PdfReader

from src.core.exam_reader import parse_exam, requested_numbers, split_questions, split_texts
from src.tools.local_reader import LocalFileReader


TEXTS = """Text Booklet
Read Text A and answer the questions.
Text AWebpage
The town library lends gardening tools.
Text BArticle about sharing
Neighbours share seeds at the library.
Text CLeaflet for visitors
Bring a bag to carry your seeds home.
"""
INTRO = """Functional Skills practice paper
Questions 1 to 2 are based on Text A.
Questions 3 to 4 are based on Text B.
Answer ALL questions. Write your answers in the spaces provided.
"""
STEMS = [
    "Read Text A and answer Questions 1 to 2.1 Which tools can you borrow?",
    "2 Which word describes the style?",
    "3 What do neighbours share?",
    "4 Why might sharing help neighbours?",
    "5 Using Text C, what should visitors bring?",
    "6 Compare Text A and Text C.",
]
MARKERS = [f"(Total for Question {n} = 1 mark)" for n in range(1, 7)]
PAPER = INTRO + "".join(stem + mark for stem, mark in zip(STEMS, MARKERS))


def _pdf(path, *pages):
    """A real Helvetica PDF, with an uncompressed text stream per page."""
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


def test_glued_headings_and_false_headings():
    texts = split_texts(TEXTS)
    assert list(texts) == ["A", "B", "C"]
    assert texts["A"] == "Text AWebpage\nThe town library lends gardening tools."
    assert texts["B"].endswith("Neighbours share seeds at the library.")
    assert texts["C"].startswith("Text CLeaflet")
    assert split_texts("Text Booklet\nRead Text A and answer") == {}


def test_glued_stems_split_at_their_own_markers():
    exam = parse_exam([PAPER, TEXTS])
    assert exam is not None
    assert [q.number for q in exam.questions] == list(range(1, 7))
    assert [q.text for q in exam.questions] == [
        stem + mark for stem, mark in zip(STEMS, MARKERS)]
    assert [q.texts for q in exam.questions] == [
        ["A"], ["A"], ["B"], ["B"], ["C"], ["A", "C"]]


def test_scanned_line_start_split_is_unchanged():
    doc = ("Set 2\n1 First question?\n(Total for Question 1 = 1 mark)\n"
           "4 97 9A0312\nSECTION B\n• 2 Second question?\n"
           "(Total for Question 2 = 2 marks)\nBLANK PAGE")
    assert [(q.number, q.text) for q in split_questions(doc)] == [
        (1, "1 First question?\n(Total for Question 1 = 1 mark)"),
        (2, "• 2 Second question?\n(Total for Question 2 = 2 marks)"),
    ]


@pytest.mark.parametrize("numbers, expected", [
    ([1, 3], [1, 3]), ([1, 1, 2], [1]), ([2, 1], [2]), ([1, 4], [1]),
])
def test_marker_order_uses_the_existing_one_gap_rule(numbers, expected):
    doc = "".join(f"{n} Which word?(Total for Question {n} = 2 marks)"
                  for n in numbers)
    assert [q.number for q in split_questions(doc)] == expected


def test_invalid_marker_order_keeps_the_stem_result():
    doc = ("1 First?\n(Total for Question 1 = 1 mark)"
           "3 Third?(Total for Question 3 = 1 mark)"
           "2 Second?(Total for Question 2 = 1 mark)")
    assert [(q.number, q.text) for q in split_questions(doc)] == [(1, doc)]


def test_pdf_answer_lines_do_not_eat_the_cap(tmp_path):
    padding = (". .... ... .. .. " * 200) + "DO NOT WRITE IN THIS AREA\n"
    padded = INTRO + "".join(stem + padding + mark
                             for stem, mark in zip(STEMS, MARKERS))
    path = _pdf(tmp_path / "paper.pdf", padded)
    raw = PdfReader(path).pages[0].extract_text()
    assert len(raw) > 15000
    content = LocalFileReader(str(tmp_path))._read_pdf(path)
    assert all(mark in content for mark in MARKERS)
    assert "Document truncated" not in content
    assert "DO NOT WRITE IN THIS AREA" not in content
    assert " … " in content


def test_pdf_real_words_still_hit_the_cap(tmp_path):
    path = _pdf(tmp_path / "words.pdf", "ordinary words " * 1200)
    raw = PdfReader(path).pages[0].extract_text()
    content = LocalFileReader(str(tmp_path))._read_pdf(path)
    assert content == raw[:15000] + (
        "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]")


def test_pdf_cleaning_changes_only_dot_runs_and_margin_notes(tmp_path, monkeypatch):
    text = "Keep 1.25 ... and . . .; collapse .... and . .. .\nDO NOT WRITE IN THIS AREA"
    path = _pdf(tmp_path / "dots.pdf", text)
    reader = LocalFileReader(str(tmp_path))
    monkeypatch.setattr(reader, "_ocr_text", lambda _p: pytest.fail("raw page is not thin"))
    assert reader._read_pdf(path) == (
        "Keep 1.25 ... and . . .; collapse  …  and  … \n\n\n")


def test_cleaning_keeps_the_raw_ocr_sampling_and_comparison(tmp_path, monkeypatch):
    # The original cap samples the first two pages: one thin is not a majority.
    path = _pdf(tmp_path / "pages.pdf", "barcode", ". " * 8000, "x")
    reader = LocalFileReader(str(tmp_path))
    monkeypatch.setattr(reader, "_ocr_text", lambda _p: pytest.fail("raw sample is not mostly thin"))
    reader._read_pdf(path)
    path = _pdf(tmp_path / "thin.pdf", ". " * 10)
    monkeypatch.setattr(reader, "_ocr_text", lambda _p: "recognised text" * 2)
    assert reader._read_pdf(path) == " …  \n\n"  # OCR is shorter than twice the raw text.


def test_two_pdf_documents_select_only_question_five_without_a_model(tmp_path):
    reader = LocalFileReader(str(tmp_path))
    docs = [reader._read_pdf(_pdf(tmp_path / name, text))
            for name, text in [("paper.pdf", PAPER), ("texts.pdf", TEXTS)]]
    exam = parse_exam(docs)
    assert exam is not None and len(exam.questions) == 6
    asked = requested_numbers("answer question 5")
    selected = [q for q in exam.questions if q.number in asked]
    assert [(q.number, q.texts) for q in selected] == [(5, ["C"])]
