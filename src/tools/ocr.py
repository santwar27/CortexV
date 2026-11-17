"""OCR for scanned PDFs, using the OCR macOS already ships (s83).

Three of Pedro's four PDFs are scans: an image of a page with no text layer,
where pypdf finds nothing to extract. Apple's Vision framework recognises
text on-device, and pyobjc is already a dependency here — so this costs no
new download, no torch, no cloud, and it keeps the sovereignty rule: the
page never leaves the Mac.

Measured on an M1 Pro: ~1s per page at 2x, accented Portuguese intact.
Each page is read at three scales and the lines merged (s85) — see _SCALES.

Fail-soft by design. Vision missing, a PDF that will not render, a page that
recognises nothing — all come back as "" and the caller says the file could
not be read, rather than raising into a voice turn.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Recognition runs per page and the reader truncates at 15k characters
# anyway, so a 200-page scan can't hold a turn hostage.
_MAX_PAGES = int(os.getenv("CORTEX_OCR_MAX_PAGES", "20"))
_MAX_CHARS = 15000
# 2x the PDF's own size: enough resolution for body text, still fast.
_SCALE = 2.0
# A page's bitmap is width × height × 4 bytes, sized by the PDF itself: a
# 20 000-point page asked for 6.4 GB at 2x, before any text limit (s88, Codex
# lote 3). 25 M pixels ≈ 100 MB. An A4 page at 2x is 2 M; an A0 poster (32 M
# at 2x) is drawn a little smaller instead.
MAX_PIXELS = 25_000_000
# Vision drops WHOLE LINES of a clean page, and which lines depends on the
# scale the page was drawn at (s85). the user's Functional Skills paper, 15
# questions: 8 question stems came back at 2x, 8 at 1x, 10 at 1.25x — each
# scale losing different ones ("5 What does each of these quotations…",
# perfectly legible, missing at 2x). Language, language correction, the
# recogniser revision and cropping the hatched margin changed nothing; the
# union of these three scales recovered all 15. Each extra scale is one more
# recognition pass per page.
def _parse_scales(raw: str) -> tuple:
    """CORTEX_OCR_SCALES → the scales to read at. A typo ("2,x") used to
    raise at import (s85 review), which turned OCR off in silence on the read
    path; bad or absurd values are skipped now, and nothing left means 2x."""
    out = []
    for part in (raw or "").split(","):
        try:
            value = float(part)
        except ValueError:
            continue
        if 0.25 <= value <= 6:
            out.append(value)
    return tuple(out) or (_SCALE,)


_SCALES = _parse_scales(os.getenv("CORTEX_OCR_SCALES", "2,1,1.25"))
# Two boxes are the same line when this much of the smaller one is covered.
_SAME_LINE_OVERLAP = 0.5
# The user reads and writes both; Vision takes the list in preference order.
_LANGUAGES = ["en-GB", "pt-BR"]


def available() -> bool:
    """True when this machine can OCR (macOS with the Vision bindings)."""
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except Exception:
        return False
    return True


def _render_page(Quartz, page, scale: float = _SCALE):
    """One PDF page → a CGImage, drawn on white (a transparent background
    recognises far worse)."""
    rect = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
    area = rect.size.width * rect.size.height
    if area > 0:
        scale = min(scale, (MAX_PIXELS / area) ** 0.5)
    width = int(rect.size.width * scale)
    height = int(rect.size.height * scale)
    if width <= 0 or height <= 0:
        return None
    colour_space = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(
        None, width, height, 8, 0, colour_space,
        Quartz.kCGImageAlphaNoneSkipLast,
    )
    if ctx is None:
        return None
    Quartz.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, width, height))
    Quartz.CGContextScaleCTM(ctx, scale, scale)
    Quartz.CGContextDrawPDFPage(ctx, page)
    return Quartz.CGBitmapContextCreateImage(ctx)


def _observations(Vision, image, languages: Sequence[str] = _LANGUAGES) -> list:
    """One image → [(text, (x, y, w, h))], the box normalised to the page
    (origin bottom-left), so boxes from different scales line up."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        image, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)          # 0 = accurate (1 = fast)
    request.setRecognitionLanguages_(list(languages))
    request.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        logger.warning("ocr: Vision request failed: %s", err)
        return []
    out = []
    for observation in (request.results() or []):
        candidates = observation.topCandidates_(1)
        if candidates:
            box = observation.boundingBox()
            out.append((str(candidates[0].string()),
                        (box.origin.x, box.origin.y,
                         box.size.width, box.size.height)))
    return out


def _recognise(Vision, image) -> str:
    """One image → the text Vision reads in it."""
    return "\n".join(text for text, _box in _observations(Vision, image))


def _covered(a: tuple, b: tuple) -> float:
    """How much of the smaller of two boxes the other one covers."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min(aw * ah, bw * bh)
    return (w * h) / smaller if smaller > 0 else 0.0


def dedupe_passes(passes: list) -> list:
    """Several recognition passes of ONE page → [(text, box)], each line
    once. A line from a later pass is kept only where no earlier pass
    already read something; the first pass wins ties, so the default
    scale's reading of a line is the one that survives."""
    kept: list = []
    for observations in passes:
        for text, box in observations:
            if not text.strip():
                continue
            if any(_covered(box, other) > _SAME_LINE_OVERLAP
                   for _t, other in kept):
                continue
            kept.append((text, box))
    return kept


def page_observations(page) -> list:
    """One CGPDFPage → [(text, (x, y, w, h))] read at every scale and
    deduplicated, boxes normalised to the page. What the answer writer
    needs: the words AND where they sit (s85)."""
    if not available():
        return []
    import Quartz
    import Vision

    passes = []
    for scale in _SCALES or (_SCALE,):
        image = _render_page(Quartz, page, scale)
        if image is not None:
            passes.append(_observations(Vision, image))
    return dedupe_passes(passes)


def merge_passes(passes: list) -> list:
    """Several recognition passes of ONE page → its lines, each once, in
    reading order (see dedupe_passes for which reading survives)."""
    kept = dedupe_passes(passes)
    # Top to bottom (Vision's y grows upwards), then left to right within a
    # row: two boxes share a row when their centres sit within half a line.
    kept.sort(key=lambda item: -(item[1][1] + item[1][3] / 2))
    rows: list = []
    for item in kept:
        centre = item[1][1] + item[1][3] / 2
        if rows:
            row = rows[-1]
            row_centre = row[0][1][1] + row[0][1][3] / 2
            if abs(centre - row_centre) < min(item[1][3], row[0][1][3]) / 2:
                row.append(item)
                continue
        rows.append([item])
    lines = []
    for row in rows:
        row.sort(key=lambda item: item[1][0])
        lines.append(" ".join(text for text, _box in row))
    return lines


def ocr_image(path: str,
              languages: Sequence[str] = ("pt-BR", "en-US")) -> List[str]:
    """Text lines read from an image file — a nutrition-label photo for
    scripts/read_label.py (Fitness slice 3.5, s76). Portuguese first: the
    labels are Brazilian. [] on any failure.

    The PDF work of s83 rewrote this module and dropped this function, so the
    script failed on import for two weeks without a single test noticing
    (s85); tests/test_nutrition_label.py now imports the script."""
    if not available():
        return []
    import Quartz
    import Vision
    from Foundation import NSURL

    try:
        source = Quartz.CGImageSourceCreateWithURL(
            NSURL.fileURLWithPath_(str(path)), None)
        if source is None:
            return []
        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if image is None:
            return []
        return [text for text, _box in _observations(Vision, image, languages)]
    except Exception:            # noqa: BLE001 — fail soft, like the PDF path
        logger.exception("ocr: failed on %s", os.path.basename(str(path)))
        return []


def ocr_pdf(path: str, max_pages: Optional[int] = None) -> str:
    """Read a scanned PDF page by page. "" when nothing can be read."""
    if not available():
        return ""
    import Quartz
    import Vision
    from Foundation import NSURL

    try:
        document = Quartz.CGPDFDocumentCreateWithURL(
            NSURL.fileURLWithPath_(path))
        if document is None:
            return ""
        pages = Quartz.CGPDFDocumentGetNumberOfPages(document)
        limit = min(pages, max_pages or _MAX_PAGES)
        out = []
        chars = 0
        read = 0
        for number in range(1, limit + 1):
            read = number
            page = Quartz.CGPDFDocumentGetPage(document, number)
            if page is None:
                continue
            passes = []
            for scale in _SCALES or (_SCALE,):
                image = _render_page(Quartz, page, scale)
                if image is not None:
                    passes.append(_observations(Vision, image))
            text = "\n".join(merge_passes(passes))
            if text:
                out.append(text)
                chars += len(text)
            if chars >= _MAX_CHARS:
                break
        text = "\n".join(out).strip()
        if text and read < pages:
            # Said, not silent: page 25 of a 30-page scan was simply absent,
            # and the answer read as if from the whole document (s88, Codex
            # lote 3). First, so the reader's 15 000-char cut can't drop it.
            text = (f"[SYSTEM NOTICE: OCR read pages 1–{read} of {pages}; "
                    f"pages {read + 1}–{pages} were NOT read.]\n\n" + text)
        return text
    except Exception:            # noqa: BLE001 — never raise into a turn
        logger.exception("ocr: failed on %s", os.path.basename(path))
        return ""
