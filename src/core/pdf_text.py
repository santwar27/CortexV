"""Fail-soft, per-file text choice through the contained Poppler runner."""
import logging
import os
import re
import unicodedata

from src.core import pdf_runner
from src.core.audit import audit
from src.core.pdf_info import MAX_PAGES_READ


log = logging.getLogger(__name__)
_WORD = re.compile(r"[^\W\d_]{3,}")
_NOTICE = "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"


def clean_page(text: str) -> str:
    """Task-62 answer dots and margin cleanup, shared by both readers."""
    return re.sub(r"\.(?:\s*\.){3,}", " … ", text).replace("DO NOT WRITE IN THIS AREA", "")


def _counts(raw_pages: list[str], chunks: list[str]) -> tuple[int, int, int]:
    total = lost = glued = 0
    for raw, chunk in zip(raw_pages, chunks):
        raw = unicodedata.normalize("NFKC", raw).lower()
        chunk = unicodedata.normalize("NFKC", chunk).lower()
        words = set(_WORD.findall(chunk))
        missing = words - set(_WORD.findall(raw))
        fingerprint = re.sub(r"\W", "", raw)
        page_glued = sum(word in fingerprint for word in missing)
        total += len(words)
        glued += page_glued
        lost += len(missing) - page_glued
    return total, lost, glued


def _layout_content(chunks: list[str]) -> str:
    content = ""
    for chunk in chunks:
        content += re.sub(r"[ \t]{3,}", "   ", clean_page(chunk)) + "\n"
        if len(content) > 15000:
            return content[:15000] + _NOTICE
    return content


def choose_text(path: str, raw_pages: list[str], content: str) -> str:
    """Compare only after OCR declined; any doubt preserves the pypdf result."""
    n = len(raw_pages)
    if (os.getenv("CORTEX_PDF_POPPLER", "1").strip().lower() in ("0", "false", "off")
            or not 1 <= n <= MAX_PAGES_READ):
        return content

    payload = {"reader": "pypdf", "poppler": "failed", "pages": n,
               "words": 0, "lost": 0, "glued": 0}
    result = content
    try:
        with pdf_runner.PdfJob() as job:
            staged = str(job.stage([path])[0])
            run = pdf_runner.run_tool(job, "pdftotext", [
                "-layout", "-enc", "UTF-8", "-eol", "unix", "-f", "1", "-l", str(n),
                staged, "-"])
        if run.status == "unavailable":
            payload["poppler"] = "unavailable"
        elif run.status == "exited" and run.exit_code == 0:
            chunks = run.stdout.decode("utf-8").split("\f")
            if len(chunks) == n + 1 and chunks[-1] == "":
                chunks = chunks[:-1]
                total, lost, glued = _counts(raw_pages, chunks)
                broken = total >= 50 and (lost * 100 >= 5 * total or glued * 100 >= 10 * total)
                if broken:
                    result = _layout_content(chunks)
                payload.update(reader="poppler" if broken else "pypdf", poppler="ok",
                               words=total, lost=lost, glued=glued)
    except Exception as error:
        log.warning("PDF text comparison failed: %r", error)
        result = content

    try:
        audit.log("–", "pdf", "read_choice", payload=payload)
    except Exception:
        pass  # Audit availability never changes the reading.
    return result
