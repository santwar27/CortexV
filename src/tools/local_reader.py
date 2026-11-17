import logging
import os
import re
from pypdf import PdfReader
from src.core import pdf_text

# Returned when a PDF carries no text layer at all. Callers match on this
# exact constant rather than re-guessing what "empty" meant.
PDF_NO_TEXT = (
    "[SYSTEM ERROR] This PDF has no text layer and on-device OCR read nothing "
    "in it either — a blank scan, a photo with no legible text, or a page the "
    "recogniser could not make out. Tell the user that; do NOT answer from "
    "guesswork."
)
import docx  # NOVA IMPORTAÇÃO

# A page with less text than this is an image with a stray text layer, not a
# page of text. Pearson exam scans carry only their barcode ("*P64979A0112*",
# 13 characters) — non-empty, so the empty-only OCR trigger never fired and
# the model was handed twelve barcodes (s85). Measured on the 40 PDFs on
# The user's Mac: the text ones average 75 to 2,900 characters a page.
_THIN_PAGE_CHARS = 40

OCR_LABEL_PREFIX = "[SYSTEM NOTICE: This PDF is a scan with no text layer."

# Whitelist for user-supplied filenames. Catches null bytes, slashes,
# backslashes, control chars, and other oddities that os.path normalisation
# wouldn't reject on its own.
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


log = logging.getLogger(__name__)


def read_by_ocr(text: str) -> bool:
    """Whether the reader labelled this document as an OCR reading."""
    return text.startswith(OCR_LABEL_PREFIX)


class LocalFileReader:
    """
    Tool to safely read local text and PDF files.
    Implements path traversal protection and memory limits.
    """

    def __init__(self, base_dir: str = "data/documents"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def read_document(self, filename: str) -> str:
        if not filename or not _FILENAME_RE.match(filename):
            return "[SYSTEM ERROR] Invalid filename. Use only letters, digits, dot, underscore, hyphen."

        # Find the real filename in the vault to handle case-insensitive matches
        files_in_vault = os.listdir(self.base_dir)
        actual_filename = next((f for f in files_in_vault if f.lower() == filename.lower()), None)

        if not actual_filename:
            return f"[SYSTEM ERROR] Document '{filename}' not found in the vault."

        # realpath (not abspath) resolves symlinks, so a planted symlink
        # inside the vault that points outside is caught by the prefix check.
        base_real = os.path.realpath(self.base_dir)
        safe_path = os.path.realpath(os.path.join(self.base_dir, actual_filename))
        if not safe_path.startswith(base_real + os.sep) and safe_path != base_real:
            return "[SYSTEM ERROR] Security breach attempt blocked. Path traversal denied."

        if not os.path.exists(safe_path):
            return f"[SYSTEM ERROR] Document '{filename}' not found in the vault."

        return self.read_path(safe_path)

    def read_path(self, path: str) -> str:
        """Extract text from a file at an ARBITRARY absolute path, dispatching
        on extension (.txt/.md/.pdf/.docx). Unlike `read_document`, this does
        NOT enforce the vault — the caller is responsible for the path being
        one the user is entitled to read.

        This exists for the screen-context "doc aberto" path (inovação #3):
        the document open in the front app (Preview/TextEdit) lives wherever
        the user saved it, not in the vault, so the vault filename guard that
        protects the LLM `read_document` tool would (correctly) reject it.
        The trust model there is the same as reading the active browser page
        or the selection — it's what's literally in front of the user, read
        once and transiently, never an LLM-chosen filename."""
        try:
            low = path.lower()
            if low.endswith('.pdf'):
                return self._read_pdf(path)
            # .md is read as plain text (M4-A12): the PROJECT_MAP advertises
            # txt/pdf/docx/md, and the whole knowledge vault is markdown — so
            # the most likely format must not bounce off "unsupported".
            elif low.endswith(('.txt', '.md')):
                return self._read_txt(path)
            elif low.endswith('.docx'):
                return self._read_docx(path)
            else:
                return "[SYSTEM ERROR] Unsupported file format. Use .txt, .md, .pdf, or .docx."

        except Exception as e:
            # The exception goes to the log, never into the result: the follow-up is
            # told to repeat a tool error verbatim, and the 8B read it out 6 times in
            # 6 ("O erro foi: EOF marker not found" — s88, Codex lote 8).
            log.warning("reading %r failed: %r", os.path.basename(path), e)
            return ("[SYSTEM ERROR] Failed to parse document: the file may be "
                    "damaged or in a format that can't be read.")

    def read_code(self, path: str) -> str:
        """Plain text with the document cap; callers must check the code gate."""
        try:
            return self._read_txt(path)
        except Exception as e:
            log.warning("reading %r failed: %r", os.path.basename(path), e)
            return ("[SYSTEM ERROR] Failed to parse document: the file may be "
                    "damaged or in a format that can't be read.")

    def _read_docx(self, path: str) -> str:
        doc = docx.Document(path)
        content = ""
        for para in doc.paragraphs:
            if para.text.strip():
                content += para.text + "\n"
            
            # Memory limit (~15000 characters)
            if len(content) > 15000:
                content = content[:15000] + "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"
                break
        return content

    def _read_txt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            # Read one char past the cap so a file of EXACTLY 15000 chars
            # isn't falsely flagged as truncated (M4-A14c).
            content = f.read(15001)
            if len(content) > 15000:
                content = content[:15000] + "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"
            return content

    def _read_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        content = ""
        raw_content = ""
        raw_pages = []
        notice = "\n\n[SYSTEM NOTICE: Document truncated due to local memory constraints.]"
        pages = thin = 0
        for page in reader.pages:
            text = page.extract_text()
            raw_pages.append(text or "")
            # Preserve OCR's raw sample, including where the old cap stopped it.
            if len(raw_content) <= 15000:
                pages += 1
                if len((text or "").strip()) < _THIN_PAGE_CHARS:
                    thin += 1
                if text:
                    raw_content += text + "\n"
                if len(raw_content) > 15000:
                    raw_content = raw_content[:15000] + notice
            if text:
                # Answer dots/margins took a real 15-question paper from 6,590
                # to 30,367 chars: keep its words before applying the same cap.
                content += pdf_text.clean_page(text) + "\n"
            
            # Memory constraint (roughly 15000 characters)
            if len(content) > 15000:
                content = content[:15000] + notice
                break
        # A scanned PDF is an IMAGE: pypdf finds no text layer and returns "".
        # Empty is not silence — three of the user's four PDFs came back with zero
        # characters (s83), and an empty read reaching the model invites it to
        # answer from nothing. Try the OCR macOS already ships (on-device, no
        # new dependency, the page never leaves the Mac); say what happened
        # when even that reads nothing.
        if not raw_content.strip():
            return self._ocr_fallback(path)
        # Mostly near-empty pages: a scan whose only text is a barcode (s85).
        # A CV with a blank second page is half thin, not mostly, and keeps
        # its text layer. OCR wins only when it actually read more.
        if thin * 2 > pages:
            recognised = self._ocr_text(path)
            if len(recognised.strip()) > 2 * len(raw_content.strip()):
                return self._ocr_labelled(recognised)
        return pdf_text.choose_text(path, raw_pages, content)

    @staticmethod
    def _ocr_text(path: str) -> str:
        """What Apple's Vision OCR reads in the PDF, or "" (switched off,
        unavailable, or nothing legible)."""
        if os.getenv("CORTEX_OCR", "1").strip().lower() in ("0", "false", "off"):
            return ""
        try:
            from src.tools.ocr import ocr_pdf
            return ocr_pdf(path)
        except Exception:                    # noqa: BLE001 — fail soft
            return ""

    def _ocr_fallback(self, path: str) -> str:
        """Last resort for a PDF with no text layer: Apple's Vision OCR."""
        recognised = self._ocr_text(path)
        if not recognised.strip():
            return PDF_NO_TEXT
        return self._ocr_labelled(recognised)

    @staticmethod
    def _ocr_labelled(recognised: str) -> str:
        """The notice rides along with the text because OCR is a reading, not
        a transcript — the model should hedge on an odd character rather than
        state it as written."""
        if len(recognised) > 15000:
            recognised = recognised[:15000] + (
                "\n\n[SYSTEM NOTICE: Document truncated due to local memory "
                "constraints.]")
        return (
            OCR_LABEL_PREFIX + " The text "
            "below was read by on-device OCR and may contain recognition "
            "errors — quote it with care.]\n\n" + recognised
        )
