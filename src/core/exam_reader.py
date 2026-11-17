"""Answer an exam from its own structure (s85).

Pedro's design: the 8B cannot hold a whole exam, so split it into parts that
fit and answer part by part — and be smart about WHICH parts, because the
exam says so itself. His Functional Skills paper declares "Questions 1 to 4
are based on Text A", the text booklet has a heading per text, and a question
like 13 names its texts ("Compare … Text A and Text C"). Every question is
answered with its own text(s) only, one call each.

Why not "read it all in chunks, summarise, merge": the paper asks for exact
quotations (question 3: which quotations are informal; 12: a word that
replaces a quotation). A summary loses exactly that, and every merge step is
one more chance for the 8B to invent.

Measured before this existed (14/10/2026, real 8B): asked to answer the
paper from one read, the 8B refused 3 of 3 times ("I can't read external
content") — an exam is full of imperatives, and the injection guard tells it
never to obey the file. The answering prompt here states the task itself and
keeps the file text fenced as data.

This module only parses. It never calls a model, so every rule in it is
testable without Ollama.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Text layers glue "Text AWebpage"; "Text Booklet" is still a word, not B.
_TEXT_HEADING_RE = re.compile(r"(?m)^[ \t]*Text ([A-Z])(?:[ \t]*$|(?=[A-Z][a-z]))")
# A question stem: its number at the start of a line (OCR sometimes puts a
# stray "•" in front), then a word or "(a)". "4 97 9A0312" — a barcode the
# OCR read as numbers — is a digit after the space, so it does not count.
_STEM_RE = re.compile(r"(?m)^[ \t•·*]*(\d{1,2})[ \t]+(?=\(?[A-Za-z'‘\"“])")
# Where a question ends: its own marks line.
_TOTAL_RE = r"\(?\s*Total for Question {n}\b[^\n]*"
_TOTAL_MARKER_RE = re.compile(r"\(\s*Total for Question\s+(\d{1,2})\b[^()]*\)", re.I)
_ANSWER_INSTRUCTION_RE = re.compile(
    r"Answer\s+ALL\s+questions\b.*?spaces\s+provided\.", re.I | re.S)
# "Questions 1 to 4 are based on Text A" / "Read Text A and answer Questions 1 to 4"
_RANGE_RES = (
    re.compile(r"Questions?\s+(\d{1,2})\s*(?:to|-|–|and)\s*(\d{1,2})\s+are\s+based\s+on\s+Text\s+([A-Z])", re.I),
    re.compile(r"Read\s+Text\s+([A-Z])\s+and\s+answer\s+Questions?\s+(\d{1,2})\s*(?:to|-|–|and)\s*(\d{1,2})", re.I),
)
_NAMED_TEXT_RE = re.compile(r"\bText ([A-Z])\b")
# A paper is a paper when it marks its questions.
_MARKS_RE = re.compile(r"Total for Question \d", re.I)


@dataclass
class Question:
    number: int
    text: str
    texts: List[str] = field(default_factory=list)   # letters, e.g. ["A", "C"]


@dataclass
class Exam:
    questions: List[Question]
    texts: Dict[str, str]
    paper: int = 0           # index, in the documents given, of the question paper


def split_texts(doc: str) -> Dict[str, str]:
    """{"A": "Text A\\nArticle\\n…", …} — each from its heading to the next."""
    marks = list(_TEXT_HEADING_RE.finditer(doc or ""))
    out: Dict[str, str] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(doc)
        letter = mark.group(1)
        if letter not in out:                  # the first heading wins
            out[letter] = doc[mark.start():end].strip()
    return out


def split_questions(doc: str) -> List[Question]:
    """The numbered questions, in order. A number counts only when it follows
    the previous one (one gap allowed, for a stem the OCR lost), so a stray
    "2" from "Set 2" or a page number cannot start a question."""
    starts = []
    last = 0
    for match in _STEM_RE.finditer(doc or ""):
        n = int(match.group(1))
        if last < n <= last + 2:
            starts.append((n, match.start()))
            last = n
    out = []
    for i, (n, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(doc)
        block = doc[start:end]
        total = re.search(_TOTAL_RE.format(n=n), block, re.I)
        if total:
            block = block[:total.end()]
        out.append(Question(number=n, text=block.strip()))
    # pypdf glued all 15 stems in the measured paper; their own marks survived.
    # Scans with enough line-start stems keep exactly the split above.
    markers = list(_TOTAL_MARKER_RE.finditer(doc or ""))
    if len(out) < len(markers):
        instruction = _ANSWER_INSTRUCTION_RE.search(doc[:markers[0].start()])
        start = instruction.end() if instruction else 0
        last = 0
        by_marker = []
        for marker in markers:
            n = int(marker.group(1))
            if not last < n <= last + 2:
                break
            by_marker.append(Question(number=n, text=doc[start:marker.end()].strip()))
            start = marker.end()
            last = n
        else:
            return by_marker
    return out


def declared_ranges(doc: str) -> Dict[int, str]:
    """Question number → the text the paper says it is based on."""
    out: Dict[int, str] = {}
    for rx in _RANGE_RES:
        for match in rx.finditer(doc or ""):
            groups = match.groups()
            if rx is _RANGE_RES[0]:
                lo, hi, letter = int(groups[0]), int(groups[1]), groups[2]
            else:
                letter, lo, hi = groups[0], int(groups[1]), int(groups[2])
            for n in range(lo, hi + 1):
                out.setdefault(n, letter.upper())
    return out


def texts_for(question: Question, ranges: Dict[int, str],
              available: List[str]) -> List[str]:
    """The texts a question needs: the ones it names, else the one the paper
    assigns it, else all of them (the caller trims to the window)."""
    named = sorted(set(_NAMED_TEXT_RE.findall(question.text)) & set(available))
    if named:
        return named
    assigned = ranges.get(question.number)
    if assigned in available:
        return [assigned]
    return sorted(available)


def parse_exam(docs: List[str]) -> Optional[Exam]:
    """An exam out of the documents read, or None when they do not look like
    one (then the caller answers the plain way). The question paper is the
    document that marks its questions; texts come from the others, or from
    the same one when a single file carries both."""
    present = [i for i, d in enumerate(docs) if d and d.strip()]
    if not present:
        return None
    paper_i = max(present, key=lambda i: len(_MARKS_RE.findall(docs[i])))
    paper = docs[paper_i]
    if len(_MARKS_RE.findall(paper)) < 2:
        return None
    questions = split_questions(paper)
    texts: Dict[str, str] = {}
    for i in present:
        if i != paper_i or len(present) == 1:
            for letter, body in split_texts(docs[i]).items():
                texts.setdefault(letter, body)
    if len(questions) < 2 or not texts:
        return None
    ranges = declared_ranges(paper)
    letters = sorted(texts)
    for q in questions:
        q.texts = texts_for(q, ranges, letters)
    return Exam(questions=questions, texts=texts, paper=paper_i)


# "responde a questão 13", "answer questions 1 to 4", "questões 2, 3 e 5".
_ASKED_RE = re.compile(
    r"\b(?:quest(?:ão|ao|ões|oes)|questions?|q)\s*\.?\s*"
    r"((?:\d{1,2})(?:\s*(?:,|e|and|a|to|até|-|–)\s*\d{1,2})*)", re.I)


def requested_numbers(request: str) -> List[int]:
    """The question numbers the user asked for, or [] for "all of them"."""
    out: List[int] = []
    for match in _ASKED_RE.finditer(request or ""):
        spec = match.group(1)
        for lo, hi in re.findall(r"(\d{1,2})\s*(?:a|to|até|-|–)\s*(\d{1,2})", spec):
            out.extend(range(int(lo), int(hi) + 1))
        spec = re.sub(r"(\d{1,2})\s*(?:a|to|até|-|–)\s*(\d{1,2})", " ", spec)
        out.extend(int(x) for x in re.findall(r"\d{1,2}", spec))
    return sorted(set(out))


# "escreve as respostas no pdf", "fill in the pdf", "preenche a prova" — the
# ask to have the answers written on a copy of the paper (s85). Shared by the
# read request ("lê e escreve no pdf") and the follow-up after the answers.
WRITE_ON_PDF_RE = re.compile(
    r"\b(?:escrev\w*|preench\w*|coloc\w*|p[õo]e|write|fill(?:\s+in)?|put)\b"
    r"[^.?!\n]{0,40}?\b(?:pdf|prova|paper|exam|folha)\b", re.I)


def asks_to_write_on_pdf(text: str) -> bool:
    return bool(WRITE_ON_PDF_RE.search(text or ""))
