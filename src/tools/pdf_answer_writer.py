"""Write exam answers onto a COPY of the paper (s85).

Pedro asked for both: the answers in the chat, and the same answers written
on the PDF where a candidate would write them — over the dotted answer
lines, a cross in the tick box of a multiple-choice question, a number in a
table cell. The original file is never touched.

Everything here is found, not guessed by a model:
  - question areas come from the OCR boxes: a question runs from its
    numbered stem to its "(Total for Question N …)" line;
  - dotted answer lines come from the pixels: a thin row (white two points
    above and below — text has height) of light, even ink (a solid rule is
    far darker), at least 120 points long;
  - a tick box is the ink just left of an option line ("A serious");
  - a label at the start of a line ("1", "Example", "Text A") opens a new
    slot, and so does a sub-question stem ("(b) Which organisational…").
The answer is split into parts at its own markers ("(1)", "(a)", "Text B:")
and the parts go to the slots in order. When a part does not fit, its first
paragraph is written (the answer without the reasoning); when even that
does not fit, it is cut with "…".

Measured on Pedro's Functional Skills paper (12 pages, 15 questions): every
answer landed in its own space — 6 crosses inside their boxes, 3 table
cells, labelled lines matched — and nothing was written over printed text.
That is ONE layout. A question with no safe place to write is left blank
and reported, never written somewhere else.

macOS only (Quartz/AppKit, already dependencies via pyobjc). No new package.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_S = 2.0                         # pixel analysis scale
_FONT_SIZE = 8.5
_TABLE_FONT_SIZE = 10.0
_MIN_LINE_PT = 120               # a dotted answer line is at least this long
_INK = (0.05, 0.2, 0.75)         # blue, like a pen

_STEM_RE = re.compile(r"^[\s•·*]*(\d{1,2})\s+(?=\(?[A-Za-z'‘\"“])")
_TOTAL_RE = re.compile(r"Total for Question (\d{1,2})", re.I)
# The tick box itself is sometimes OCR'd as a bullet: "• A serious".
_OPTION_RE = re.compile(r"^[\s•·*□☐☒■◻]*([A-E])\s+\S")
_SUB_RE = re.compile(r"^\s*\(([a-e])\)\s+\S")
_DOTS = set(".…·")

_STOP_RE = re.compile(
    r"(?i)^(reasoning|note|this is a fact because|this option shows|"
    r"the text uses|both text [a-c] and text [a-c] use)\b")
_PART_RE = re.compile(
    r"(?m)^\s*(?:\((\d|[a-e])\)|(\d)\.|(?:\d{1,2}\s+)?\(([a-e])\)|"
    r"(Text [A-C])\s*:|(Language feature)\s*:|(Example)\s*:|"
    r"(Similarity \d)\s*:|\*\s+)")
_TABLE_ROW_RE = re.compile(
    r"(?m)^\s*(?:\d\.\s*)?([A-Z][A-Za-z ]+?)\s*(?:\(Paragraph (\d)\)|[-:–]\s*(\d))\s*$")


@dataclass
class WriteResult:
    path: str
    written: List[int] = field(default_factory=list)
    skipped: List[int] = field(default_factory=list)
    crosses: int = 0


def answered_copy_path(paper: str, lang: str = "pt") -> str:
    """"question.pdf" → "question - respostas.pdf" beside it, never an
    existing file: "(2)", "(3)"… until the name is free."""
    folder, name = os.path.split(paper)
    stem = os.path.splitext(name)[0]
    tag = "respostas" if lang != "en" else "answers"
    candidate = os.path.join(folder, f"{stem} - {tag}.pdf")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem} - {tag} ({n}).pdf")
        n += 1
    return candidate


# ---------------------------------------------------------------- answers

def clean_answer(answer: str) -> str:
    """What goes on paper: the answer, not the reasoning or the preamble."""
    kept = []
    for line in re.sub(r"\*\*", "", answer or "").splitlines():
        s = line.strip()
        if _STOP_RE.match(s):
            break
        if re.match(r"(?i)^(here are|\(no marks)", s):
            continue
        kept.append(s)
    return "\n".join(kept).strip()


def parts_of(text: str) -> List[str]:
    """An answer split at its own markers — "(1)", "(a)", "Text B:",
    "Language feature:" — or at blank lines when it has none."""
    text = re.sub(r"(?m)^\s*\(\d\)\s+(?=(?:Language feature|Example|Text [A-C])\s*:)",
                  "", text)
    marks = list(_PART_RE.finditer(text))
    if not marks:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    head = text[:marks[0].start()].strip()
    if head:
        out.append(head)
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = re.sub(r"^\d\.\s+", "", text[m.end():end].strip())
        label = next((g for g in m.groups() if g), "")
        if label.startswith("Similarity"):
            chunk = f"{label}: {chunk}"
        if chunk:
            out.append(chunk)
    return out


def chosen_letters(answer: str, offered: List[str]) -> List[str]:
    """The option letters an answer picks: "B 'to sweeten the deal'\\nD …"."""
    found = re.findall(r"(?m)^\s*([A-E])\b(?=[\s'\".:)‘]|$)", answer or "")
    return [x for x in dict.fromkeys(found) if x in offered]


def table_values(answer: str) -> Dict[str, str]:
    """"Key findings (Paragraph 1)" / "Role of IT - 3" → {"key findings": "1"}."""
    out = {}
    for m in _TABLE_ROW_RE.finditer(answer or ""):
        out[m.group(1).strip().lower()] = m.group(2) or m.group(3)
    return out


# ------------------------------------------------------------------ pixels

def _gray(Quartz, page):
    import numpy as np
    from src.tools.ocr import MAX_PIXELS
    r = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
    w, h = int(r.size.width * _S), int(r.size.height * _S)
    # The analysis is in _S pixels per point, so a giant page is refused, not
    # shrunk (s88, Codex lote 3). The caller says it couldn't write the PDF.
    if w * h > MAX_PIXELS:
        raise ValueError(f"page too large to analyse: {w}x{h} px")
    ctx = Quartz.CGBitmapContextCreate(None, w, h, 8, w,
                                       Quartz.CGColorSpaceCreateDeviceGray(),
                                       Quartz.kCGImageAlphaNone)
    Quartz.CGContextSetGrayFillColor(ctx, 1, 1)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, w, h))
    Quartz.CGContextScaleCTM(ctx, _S, _S)
    Quartz.CGContextDrawPDFPage(ctx, page)
    img = Quartz.CGBitmapContextCreateImage(ctx)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    bpr = Quartz.CGImageGetBytesPerRow(img)
    a = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, bpr)[:, :w]
    return a, float(r.size.width), float(r.size.height)   # row 0 = top


def dotted_lines(a, W: float, H: float) -> List[Tuple[float, float, float]]:
    """[(y, x0, x1)] in PDF points (origin bottom-left) of the dotted answer
    lines in a page bitmap (rendered at _S, row 0 at the top)."""
    import numpy as np
    D = 255.0 - a.astype(np.float32)
    k = 24
    c = np.cumsum(np.pad(D, ((0, 0), (1, 0))), axis=1)
    sm = (c[:, k:] - c[:, :-k]) / k                  # window mean along x
    x_lo, x_hi = int(0.08 * W * _S), int(0.90 * W * _S)
    found = []
    for r in range(5, sm.shape[0] - 5):
        row = sm[r, x_lo:x_hi]
        m = ((row > 12) & (row < 120)
             & (sm[r - 4, x_lo:x_hi] < 10) & (sm[r + 4, x_lo:x_hi] < 10))
        if m.sum() < _MIN_LINE_PT * _S:
            continue
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        i = int(np.argmax(en - st))
        if en[i] - st[i] < _MIN_LINE_PT * _S:
            continue
        # Dots have gaps; a table border or a rule is unbroken ink. Seen on
        # the first real run: the borders of question 4's table read as
        # answer lines and the numbers went over the table (s85).
        raw = D[r, x_lo + st[i]:x_lo + en[i]]
        if (raw < 30).mean() < 0.3:
            continue
        found.append((r, x_lo + st[i], x_lo + en[i] + k // 2))
    lines: list = []
    for r, x0, x1 in found:
        if lines and r - lines[-1][1] <= 3:
            r0, _r1, lx0, lx1 = lines[-1]
            lines[-1] = (r0, r, min(lx0, x0), max(lx1, x1))
        else:
            lines.append((r, r, x0, x1))
    return [(H - (r1 + 1) / _S, x0 / _S, x1 / _S) for _r0, r1, x0, x1 in lines]


def _ink_box(a, H: float, x0: float, x1: float, y0: float, y1: float,
             level: int) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box (points) of the ink darker than `level` in a region."""
    import numpy as np
    c0, c1 = max(int(x0 * _S), 0), max(int(x1 * _S), 0)
    r0, r1 = max(int((H - y1) * _S), 0), max(int((H - y0) * _S), 0)
    region = a[r0:r1, c0:c1] < level
    ys, xs = np.where(region)
    if len(xs) < 10:
        return None
    return ((c0 + xs.min()) / _S, H - (r0 + ys.max() + 1) / _S,
            (xs.max() - xs.min() + 1) / _S, (ys.max() - ys.min() + 1) / _S)


def _tick_box(a, H: float, b) -> Optional[Tuple[float, float, float, float]]:
    """The tick box of an option line: the FIRST blob of ink, left to right,
    around the start of the line. Where the OCR box starts at the letter, the
    box is to its left; where the OCR read the box as "•", it is inside. The
    leftmost blob is the box either way — the letter always comes after it."""
    import numpy as np
    x, y, w, h = b
    c0, c1 = max(int((x - 40) * _S), 0), int((x + 12) * _S)
    r0, r1 = max(int((H - (y + h + 3)) * _S), 0), int((H - (y - 3)) * _S)
    region = a[r0:r1, c0:c1] < 215
    cols = region.any(axis=0)
    d = np.diff(np.concatenate([[0], cols.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for st, en in zip(starts, ends):
        if not 4 * _S <= en - st <= 16 * _S:
            continue
        ys = np.where(region[:, st:en].any(axis=1))[0]
        return ((c0 + st) / _S, H - (r0 + ys.max() + 1) / _S,
                (en - st) / _S, (ys.max() - ys.min() + 1) / _S)
    return None


# ------------------------------------------------------------------ layout

@dataclass
class _Page:
    a: object
    W: float
    H: float
    boxes: list          # [(text, (x, y, w, h))] in points
    dots: list           # [(y, x0, x1)]


def _label_at(P: _Page, line) -> Optional[str]:
    """The printed label at the start of a dotted line, if any."""
    y, x0, _x1 = line
    for t, b in P.boxes:
        core = re.sub(r"[.…·\s]+$", "", t).strip()
        if not core or set(core) <= _DOTS:
            continue
        if abs((b[1] + b[3] / 2) - (y + 4)) < 8 and b[0] < x0 + 2:
            return core
    # A lone "1" or "2" is often not OCR'd at all: ink just left of the dots.
    if _ink_box(P.a, P.H, max(x0 - 30, P.W * 0.08), x0 - 1, y - 1, y + 11, 120):
        return "#"
    return None


def _questions(pages: List[_Page]):
    """Walk the pages in reading order; give every dotted line, option line
    and sub-question stem to the question it sits in."""
    events = []
    last = 0
    for pi, P in enumerate(pages):
        ordered = sorted(P.boxes, key=lambda tb: (-(tb[1][1] + tb[1][3]), tb[1][0]))
        for t, b in ordered:
            m = _STEM_RE.match(t)
            if m and last < int(m.group(1)) <= last + 2 and b[0] < P.W * 0.3:
                last = int(m.group(1))
                events.append((pi, -(b[1] + b[3]), "stem", last))
                continue
            m = _TOTAL_RE.search(t)
            if m:
                events.append((pi, -(b[1] + b[3]), "total", int(m.group(1))))
                continue
            if _SUB_RE.match(t) and b[0] < P.W * 0.3 and len(t) > 12:
                events.append((pi, -(b[1] + b[3] / 2), "sub", None))
                continue
            m = _OPTION_RE.match(t)
            if m and P.W * 0.12 < b[0] < P.W * 0.4:
                events.append((pi, -(b[1] + b[3] / 2), "option", (m.group(1), b)))
        for d in P.dots:
            events.append((pi, -d[0], "dots", d))
    events.sort(key=lambda e: (e[0], e[1]))
    lines: Dict[int, list] = {}
    options: Dict[int, list] = {}
    regions: Dict[int, list] = {}
    current = None
    for pi, negy, kind, val in events:
        if kind == "stem":
            current = val
            regions.setdefault(current, []).append((pi, -negy))
        elif kind == "total":
            current = None
        elif current is None:
            continue
        elif kind == "dots":
            lines.setdefault(current, []).append((pi, val))
        elif kind == "sub":
            lines.setdefault(current, []).append((pi, None))
        elif kind == "option":
            options.setdefault(current, []).append((pi, val))
    return lines, options, regions


def _slots(pages: List[_Page], lines: list) -> List[list]:
    slots: List[list] = []
    opened = False
    for pi, d in lines:
        if d is None:
            opened = True
            continue
        if _label_at(pages[pi], d) or opened or not slots:
            slots.append([(pi, d)])
            opened = False
        else:
            slots[-1].append((pi, d))
    return slots


def _attrs(size: float):
    import AppKit
    return {
        AppKit.NSFontAttributeName: AppKit.NSFont.fontWithName_size_("Helvetica", size),
        AppKit.NSForegroundColorAttributeName:
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*_INK, 1),
    }


def _width(text: str, attrs) -> float:
    import AppKit
    return AppKit.NSString.stringWithString_(text).sizeWithAttributes_(attrs).width


def _wrap(text: str, width: float, attrs) -> List[str]:
    rows: List[str] = []
    for chunk in [c for c in text.split("\n") if c.strip()]:
        cur = ""
        for word in chunk.split():
            nxt = (cur + " " + word).strip()
            if not cur or _width(nxt, attrs) <= width:
                cur = nxt
            else:
                rows.append(cur)
                cur = word
        if cur:
            rows.append(cur)
    return rows


def _fit(text: str, lines: list, attrs) -> Tuple[List[str], bool]:
    """Rows for these lines: the whole part, else its first paragraph, else
    cut with "…". → (rows, shortened)."""
    width = min(d[2] - d[1] for _pi, d in lines) - 14
    rows = _wrap(text, width, attrs)
    if len(rows) <= len(lines):
        return rows, False
    rows = _wrap(re.split(r"\n\s*\n", text.strip())[0], width, attrs)
    if len(rows) > len(lines):
        rows = rows[:len(lines)]
        rows[-1] = rows[-1].rstrip(" .,;") + "…"
    return rows, True


def _table_header(pages: List[_Page], region: list):
    """The "… number" column header below a question's stem, if any."""
    for pi, top in region:
        for t, b in pages[pi].boxes:
            if b[1] < top and "number" in t.lower() and len(t) < 30:
                return pi, b
    return None


def plan(pages: List[_Page], answers: Dict[int, str]):
    """→ (draws, result). draws: {page_index: [("text", (x, y, s, size)) |
    ("cross", (x, y, w, h))]}. Pure layout — nothing is drawn here."""
    attrs = _attrs(_FONT_SIZE)
    lines, options, regions = _questions(pages)
    draws: Dict[int, list] = {i: [] for i in range(len(pages))}
    result = WriteResult(path="")
    for qn in sorted(answers):
        answer = clean_answer(answers[qn])
        opts = options.get(qn, [])
        q_lines = [x for x in lines.get(qn, []) if x[1] is not None]
        if not answer:
            result.skipped.append(qn)
            continue
        # multiple choice: tick boxes, no writing
        if opts and not q_lines:
            letters = chosen_letters(answer, [o[1][0] for o in opts])
            crosses = 0
            for pi, (letter, b) in opts:
                if letter not in letters:
                    continue
                P = pages[pi]
                box = _tick_box(P.a, P.H, b)
                if box:
                    draws[pi].append(("cross", box))
                    crosses += 1
            if crosses:
                result.crosses += crosses
                result.written.append(qn)
            else:
                result.skipped.append(qn)
            continue
        # a table with a "… number" column: one value per named row
        values = table_values(answer)
        if values and _table_header(pages, regions.get(qn, [])):
            done = 0
            for pi, head in [_table_header(pages, regions.get(qn, []))]:
                P = pages[pi]
                cx = head[0] + head[2] / 2
                for t, b in P.boxes:
                    name = t.strip().lower()
                    if name not in values or b[1] >= head[1]:
                        continue
                    filled = any(re.fullmatch(r"\d{1,2}", t2.strip())
                                 and abs(b2[1] - b[1]) < 8
                                 and abs(b2[0] + b2[2] / 2 - cx) < 40
                                 for t2, b2 in P.boxes)
                    if not filled:
                        draws[pi].append(("text", (cx - 3, b[1] + 1,
                                                   values[name], _TABLE_FONT_SIZE)))
                        done += 1
            (result.written if done else result.skipped).append(qn)
            continue
        if not q_lines:
            result.skipped.append(qn)
            continue
        slots = _slots(pages, lines.get(qn, []))
        parts = parts_of(answer)
        if len(parts) == len(slots):
            pairs = list(zip(parts, slots))
        else:
            pairs = [("\n".join(parts), [ln for s in slots for ln in s])]
        for text, slot in pairs:
            rows, _short = _fit(text, slot, attrs)
            for (pi, (y, x0, _x1)), row in zip(slot, rows):
                draws[pi].append(("text", (x0 + 5, y + 1.5, row, _FONT_SIZE)))
        result.written.append(qn)
    return draws, result


# ------------------------------------------------------------------ output

def read_pages(paper: str) -> List[_Page]:
    import Quartz
    from Foundation import NSURL
    from src.tools.ocr import page_observations

    doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(paper))
    if doc is None:
        return []
    pages = []
    for n in range(1, Quartz.CGPDFDocumentGetNumberOfPages(doc) + 1):
        page = Quartz.CGPDFDocumentGetPage(doc, n)
        a, W, H = _gray(Quartz, page)
        boxes = [(t, (b[0] * W, b[1] * H, b[2] * W, b[3] * H))
                 for t, b in page_observations(page)]
        pages.append(_Page(a=a, W=W, H=H, boxes=boxes, dots=dotted_lines(a, W, H)))
    return pages


def write_answers(paper: str, answers: Dict[int, str], out: str,
                  lang: str = "pt") -> WriteResult:
    """Draw the answers on a copy of `paper` saved as `out`. Raises only on
    a path that already exists — the caller picks a free one."""
    import AppKit
    import Quartz
    from Foundation import NSURL

    if os.path.exists(out):
        raise FileExistsError(out)
    pages = read_pages(paper)
    draws, result = plan(pages, answers)
    result.path = out
    doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(paper))
    first = Quartz.CGPDFPageGetBoxRect(Quartz.CGPDFDocumentGetPage(doc, 1),
                                       Quartz.kCGPDFMediaBox)
    # Drawn under a hidden name, then linked to `out`: link() refuses a name
    # that exists, so a file made there during the seconds of reading — by a
    # second run, say — is never overwritten (s88, Codex lote 3).
    tmp = os.path.join(os.path.dirname(out) or ".",
                       f".{os.path.basename(out)}.{os.getpid()}.{id(pages)}.tmp")
    header = ("Respostas do Cortex — confira antes de usar" if lang != "en"
              else "Answers by Cortex — check before use")
    size = AppKit.NSString.stringWithString_(header).sizeWithAttributes_(
        _attrs(_FONT_SIZE))
    spot = _header_spot(pages[0], (size.width, size.height))
    try:
        _draw(AppKit, Quartz, NSURL, doc, pages, draws, first, tmp,
              header if spot else None, spot)
        os.link(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return result


def _header_spot(page: _Page, size: Tuple[float, float]
                 ) -> Optional[Tuple[float, float]]:
    """Where the "check before use" line fits on the first page without
    touching print — the top band, else the bottom one; None, no line. It
    was drawn over a paper's own title in the top band (s88, Codex lote 3)."""
    w, h = size
    x = page.W * 0.1
    for y in (page.H - 22, 8.0):
        r0 = max(int((page.H - y - h) * _S) - 4, 0)       # row 0 = top
        r1 = int((page.H - y) * _S) + 4
        c0, c1 = max(int(x * _S) - 4, 0), int((x + w) * _S) + 4
        region = page.a[r0:r1, c0:c1]
        if region.size and not (region < 200).any():
            return (x, y)
    return None


def _draw(AppKit, Quartz, NSURL, doc, pages, draws, first, out: str,
          header: Optional[str], spot: Optional[Tuple[float, float]]) -> None:
    ctx = Quartz.CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(out), first, None)
    try:
        for i in range(len(pages)):
            page = Quartz.CGPDFDocumentGetPage(doc, i + 1)
            box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
            Quartz.CGContextBeginPage(ctx, box)
            Quartz.CGContextDrawPDFPage(ctx, page)
            ns = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
            AppKit.NSGraphicsContext.saveGraphicsState()
            AppKit.NSGraphicsContext.setCurrentContext_(ns)
            if i == 0 and header:
                AppKit.NSString.stringWithString_(header).drawAtPoint_withAttributes_(
                    spot, _attrs(_FONT_SIZE))
            for kind, val in draws[i]:
                if kind == "text":
                    x, y, text, size = val
                    AppKit.NSString.stringWithString_(text).drawAtPoint_withAttributes_(
                        (x, y), _attrs(size))
                else:
                    x, y, w, h = val
                    Quartz.CGContextSetRGBStrokeColor(ctx, *_INK, 1)
                    Quartz.CGContextSetLineWidth(ctx, 1.6)
                    Quartz.CGContextMoveToPoint(ctx, x, y)
                    Quartz.CGContextAddLineToPoint(ctx, x + w, y + h)
                    Quartz.CGContextMoveToPoint(ctx, x, y + h)
                    Quartz.CGContextAddLineToPoint(ctx, x + w, y)
                    Quartz.CGContextStrokePath(ctx)
            AppKit.NSGraphicsContext.restoreGraphicsState()
            Quartz.CGContextEndPage(ctx)
    finally:
        Quartz.CGPDFContextClose(ctx)
