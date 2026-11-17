"""The OCR page cap is said, not silent (s88, Codex lote 3).

ocr_pdf stops at CORTEX_OCR_MAX_PAGES (20). A 30-page scan with ~100
characters a page came back as 2 000 characters from pages 1–20 and nothing
else: the reader's notice only fires past 15 000 characters, so the answer
read as if from the whole document. Vision is stubbed: one line per page.
"""
import pytest

from src.tools import ocr
from src.tools.local_reader import LocalFileReader


def _blank_pdf(path, n):
    import Quartz
    from Foundation import NSURL
    box = Quartz.CGRectMake(0, 0, 595, 842)
    ctx = Quartz.CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(str(path)), box, None)
    for _ in range(n):
        Quartz.CGContextBeginPage(ctx, box)
        Quartz.CGContextEndPage(ctx)
    Quartz.CGPDFContextClose(ctx)


@pytest.fixture
def one_line_a_page(monkeypatch):
    import Quartz
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "_render_page",
                        lambda Q, page, scale=2.0: Quartz.CGPDFPageGetPageNumber(page))
    monkeypatch.setattr(ocr, "_observations", lambda V, n, languages=None: [
        (f"PAGINA {n} " + "x" * 90, (0.1, 0.5, 0.8, 0.05))])


def test_pages_past_the_cap_are_named(tmp_path, one_line_a_page):
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf, 30)
    text = LocalFileReader._ocr_labelled(ocr.ocr_pdf(str(pdf)))
    assert "PAGINA 20 " in text and "PAGINA 21 " not in text
    assert "OCR read pages 1–20 of 30; pages 21–30 were NOT read." in text


def test_the_notice_survives_the_readers_cut(tmp_path, one_line_a_page, monkeypatch):
    monkeypatch.setattr(ocr, "_observations", lambda V, n, languages=None: [
        (f"PAGINA {n} " + "x" * 2000, (0.1, 0.5, 0.8, 0.05))])
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf, 30)
    text = LocalFileReader._ocr_labelled(ocr.ocr_pdf(str(pdf)))
    assert "were NOT read." in text
    assert "Document truncated" in text


def test_a_whole_scan_has_no_notice(tmp_path, one_line_a_page):
    pdf = tmp_path / "scan.pdf"
    _blank_pdf(pdf, 3)
    assert "NOT read" not in ocr.ocr_pdf(str(pdf))


class TestThePixelCap:
    """The bitmap is sized by the PDF itself: a 20 000-point page asked for
    40 000 × 40 000 px (6.4 GB) at 2x, before any text limit (s88, Codex
    lote 3). Nothing is allocated here — the request is recorded."""

    def _page(self, tmp_path, side):
        import Quartz
        from Foundation import NSURL
        pdf = tmp_path / f"p{side}.pdf"
        box = Quartz.CGRectMake(0, 0, side, side)
        ctx = Quartz.CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(str(pdf)), box, None)
        Quartz.CGContextBeginPage(ctx, box)
        Quartz.CGContextEndPage(ctx)
        Quartz.CGPDFContextClose(ctx)
        doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(pdf)))
        return doc, Quartz.CGPDFDocumentGetPage(doc, 1)

    def _asked(self, page, scale):
        import Quartz
        asked = []

        class RecordingQuartz:
            def __getattr__(self, name):
                return getattr(Quartz, name)

            def CGBitmapContextCreate(self, _data, w, h, *_a):
                asked.append((w, h))
                return None
        ocr._render_page(RecordingQuartz(), page, scale)
        return asked[0]

    def test_a_giant_page_is_drawn_smaller(self, tmp_path):
        _doc, page = self._page(tmp_path, 20000)
        w, h = self._asked(page, 2.0)
        assert w * h <= ocr.MAX_PIXELS

    def test_an_a4_page_keeps_its_scale(self, tmp_path):
        _doc, page = self._page(tmp_path, 600)
        assert self._asked(page, 2.0) == (1200, 1200)

    def test_the_writer_refuses_a_giant_page(self, tmp_path):
        import Quartz
        from src.tools import pdf_answer_writer as w
        _doc, page = self._page(tmp_path, 20000)
        with pytest.raises(ValueError, match="too large"):
            w._gray(Quartz, page)
