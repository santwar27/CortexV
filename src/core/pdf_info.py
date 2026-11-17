"""Dormant PDF facts and bounded text coverage through the contained runner.

The caller owns read permission. No PDF text or diagnostics leave this module.
"""
from dataclasses import dataclass, replace
import re
from typing import Literal

from src.core import pdf_runner

MAX_PAGES_READ = 20
MAX_DECLARED_PAGES = 2000


@dataclass(frozen=True)
class PdfInfo:
    status: Literal["ok", "protected", "broken", "too_big", "timeout", "stopped",
                    "unavailable", "error"]
    pages: int | None
    encrypted: bool | None
    restrictions: tuple[str, ...]
    page_text: tuple[Literal["text", "thin", "none"], ...]
    pages_read: int
    detail: str


_DETAILS = {
    "protected": "PDF requires a password.",
    "broken": "PDF is broken.",
    "too_big": "PDF inspection limit exceeded.",
    "timeout": "PDF inspection timed out.",
    "stopped": "PDF inspection stopped.",
    "unavailable": "Contained PDF tools unavailable.",
    "error": "PDF inspection failed.",
}


def _failure(run):
    # Runner limits/cancellation take precedence over partial parser diagnostics.
    if run.status != "exited":
        return run.status if run.status in _DETAILS else "error"
    if "Incorrect password" in run.stderr_tail:
        return "protected"
    if run.exit_code == 1 and ("Syntax Error" in run.stderr_tail
                               or "trailer dictionary" in run.stderr_tail):
        return "broken"
    return None if run.exit_code == 0 else "error"


def _field(stdout, name):
    lines = [line[len(name):].strip() for line in stdout.splitlines()
             if line.startswith(name)]
    # Duplicate fields are ambiguous, even when one happens to be well formed.
    return lines[0] if len(lines) == 1 else b""


def inspect(path, *, cancelled=None) -> PdfInfo:
    """Inspect an authorised path without raising or retaining extracted content.

    A text check must return exactly one terminated chunk per requested page.
    Failed/incomplete checks keep metadata but never label unknown pages blank.
    """
    facts = PdfInfo("error", None, None, (), (), 0, _DETAILS["error"])
    try:
        with pdf_runner.PdfJob() as job:
            try:
                staged = str(job.stage([path])[0])
            except ValueError as error:
                if str(error) == "PDF input bytes exceed the job limit":
                    return replace(facts, status="too_big", detail="PDF input size limit exceeded.")
                detail = ("PDF input must be a regular file."
                          if str(error) == "PDF input must be a regular file"
                          else "PDF input staging failed.")
                return replace(facts, detail=detail)

            run = pdf_runner.run_tool(job, "pdfinfo", [staged], cancelled=cancelled)
            failure = _failure(run)
            if failure:
                return replace(facts, status=failure, detail=_DETAILS[failure])

            value = _field(run.stdout, b"Encrypted:")
            encrypted, restrictions = None, ()
            if value == b"no":
                encrypted = False
            elif re.fullmatch(rb"yes(?: \([^()\r\n]*\))?", value):
                encrypted = True
                restrictions = tuple(sorted(name.decode("ascii") for name in
                    re.findall(rb"(?:[ (])(print|copy|change|addNotes):no(?=[ )])", value)))
            facts = replace(facts, encrypted=encrypted, restrictions=restrictions)

            value = _field(run.stdout, b"Pages:")
            if not re.fullmatch(rb"[0-9]+", value):
                return replace(facts, detail="PDF page count unavailable.")
            try:
                pages = int(value.lstrip(b"0") or b"0")
            except ValueError:  # Python's integer digit limit: necessarily above our cap.
                return replace(facts, status="too_big", detail="PDF declared page limit exceeded.")
            if pages < 1:
                return replace(facts, detail="PDF page count unavailable.")
            facts = replace(facts, pages=pages)
            if pages > MAX_DECLARED_PAGES:
                return replace(facts, status="too_big", detail="PDF declared page limit exceeded.")

            count = min(pages, MAX_PAGES_READ)
            run = pdf_runner.run_tool(job, "pdftotext", [
                "-enc", "UTF-8", "-eol", "unix", "-f", "1", "-l", str(count), staged, "-"
            ], cancelled=cancelled)
            failure = _failure(run)
            if failure:
                suffix = (" " + _DETAILS[failure]
                          if run.status != "exited" or failure != "error" else "")
                return replace(facts, status=failure, detail="PDF text check failed." + suffix)
            try:
                chunks = run.stdout.decode("utf-8").split("\f")
            except UnicodeDecodeError:
                return replace(facts, detail="PDF text check failed.")
            if len(chunks) != count + 1 or chunks[-1] != "":
                return replace(facts, detail="PDF text check failed.")
            classes = []
            for chunk in chunks[:-1]:
                visible = sum(c.isprintable() and not c.isspace() for c in chunk)
                classes.append("text" if visible >= 40 else "thin" if visible else "none")
            return replace(facts, status="ok", page_text=tuple(classes), pages_read=count,
                           detail="PDF inspected.")
    except Exception:
        # Context setup/cleanup and unexpected runner errors also remain fail-soft.
        detail = "PDF text check failed." if facts.pages is not None else _DETAILS["error"]
        return replace(facts, status="error", detail=detail)


def scanned_pages(info: PdfInfo) -> tuple[int, ...]:
    """Physical page numbers needing OCR, within the pages actually read."""
    return tuple(number for number, kind in enumerate(info.page_text, 1)
                 if kind in ("none", "thin"))


def has_text_layer(info: PdfInfo) -> bool:
    """Whether every page read has substantial text; says nothing about unread pages."""
    return bool(info.page_text) and all(kind == "text" for kind in info.page_text)
