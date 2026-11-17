"""Pedro's explicit teaching, kept literally without asking a model to write it.

The orchestrator calls prepare only after existing dispatch has declined the
turn. The request scope carries the ORIGINAL text through routing normalisation
and lets both public entry points put the receipt after every other paragraph.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.core.audit import audit
from src.core.i18n import t
from src.core.memory import speaker_unverified
from src.core.redaction import redact_text

FILENAME = "pedro_ensinou.md"
TITLE = "# Pedro — o que ele ensinou ao Cortex"
FRAMING = (
    '(Palavras do Pedro para o Cortex: “você”, “seu”, “sua”, “teu”, “tua”, '
    '“you”, “your” = o Cortex; “eu”, “meu”, “minha”, “I”, “my” = o Pedro.)'
)
REPLY_NOTE = (
    "Cortex itself already handled saving the user's sentence; a fixed receipt "
    "will be added after your reply. Do not call any tool for it. Do not say "
    "you saved, stored, noted, or could not save anything. In the user's "
    "sentence, você/seu/sua/teu/tua/you/your means you, Cortex; eu/meu/minha/I/my "
    "means Pedro. Just reply briefly and naturally to what the user said."
)
MAX_SENTENCE_CHARS = 1000

# One shared PT/EN pattern. Fold accents on a disposable matching copy only.
TRIGGER_RE = re.compile(
    r"\b(?:aprend[ae]\s+(?:com\s+)?isso|guard[ae]\s+isso|"
    r"memoriz[ae]\s+isso|lembre-se\s+disso|lembra\s+disso|"
    r"nao\s+esquec[ae]\s+disso|remember\s+this|learn\s+this|"
    r"keep\s+this\s+in\s+mind)\b", re.IGNORECASE,
)
_WAKE_RE = re.compile(r"^c[óo]rtex\b[\s,!.?;:：\-–—]+", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:nao|nunca|nem|not|never|cannot|can['’]t|don['’]?t)\b")
_FILLER_RE = re.compile(
    r"\b(?:deixa\s+eu\s+te\s+dizer\s+algo|"
    r"let\s+me\s+tell\s+you\s+something|por\s+favor|please)\b"
)
_ENTRY_RE = re.compile(
    r"\n(?P<heading>## \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00)\n"
    r"<!-- cortex:taught-chars=(?P<chars>\d+) -->\n\n"
    + re.escape(FRAMING) + r"\n\n",
)
# Serialise read/duplicate-check/replace/index as one operation, also across
# separate KnowledgeBase instances in the same service process.
_WRITE_LOCK = threading.RLock()


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


def _sentence(text: str) -> str | None:
    sentence = _WAKE_RE.sub("", text.strip(), count=1).strip()
    if sentence.endswith("?"):
        return None
    folded = _fold(sentence)
    matches = list(TRIGGER_RE.finditer(folded))
    if not matches:
        return None
    # "não esqueça disso" is itself a positive imperative; a negation in
    # the preceding clause ("não precisa dizer aprenda isso") cancels it.
    for match in matches:
        prefix = re.split(r"[.,:;!?\n]", folded[:match.start()])[-1]
        if _NEGATION_RE.search(prefix):
            return None
    return sentence


def teaching_is_empty(text: str) -> bool:
    """Routing probe only: skip even the 8B classifier for an empty teach."""
    sentence = _sentence(text)
    return sentence is not None and not re.search(
        r"\w", _FILLER_RE.sub("", TRIGGER_RE.sub("", _fold(sentence))))


def taught_sections(content: str) -> list[tuple[str, str]]:
    """Read length-delimited entries; even a literal date H2 stays user text.

    The invisible Markdown comment describes the following sentence without
    escaping or changing any of its characters, including CRLF and Markdown.
    """
    if not content.startswith(TITLE + "\n"):
        raise ValueError("Unexpected teaching file title")
    sections = []
    cursor = len(TITLE) + 1
    while cursor < len(content):
        match = _ENTRY_RE.match(content, cursor)
        if match is None:
            raise ValueError("Malformed teaching entry")
        end = match.end() + int(match["chars"])
        if content[end:end + 1] != "\n":
            raise ValueError("Incomplete teaching entry")
        sections.append((match["heading"], content[match.end():end]))
        cursor = end + 1
    return sections


def taught_chunks(content: str) -> list[str]:
    return [f"{TITLE}\n\n{heading}\n\n{FRAMING}\n\n{sentence}"
            for heading, sentence in taught_sections(content)]


@dataclass(frozen=True)
class KeepResult:
    status: str
    sentence: str
    lang: str

    @property
    def receipt(self) -> str:
        replies = {
            "kept": (f"Guardei o que você me ensinou: “{self.sentence}”",
                     f"I kept what you taught me: “{self.sentence}”"),
            "already": ("Isso eu já tinha guardado.", "I had already kept that."),
            "empty": ("O que eu devo guardar? Diga, por exemplo: aprenda com isso: <o fato>.",
                      "What should I keep? Say, for example: learn this: <the fact>."),
            "too_long": ("É grande demais para eu guardar de uma vez — mande em partes menores.",
                         "That's too long to keep at once — send it in smaller parts."),
            "sensitive": ("Isso parece ter um dado sensível, então não guardei.",
                          "That looks like it has sensitive data, so I didn't keep it."),
            "write_failed": ("Não consegui guardar — nada mudou.",
                             "I couldn't keep it — nothing changed."),
            "index_failed": ("Guardei no arquivo, mas não consegui indexar agora.",
                             "I saved it to the file, but couldn't index it right now."),
        }
        return t(self.lang, *replies[self.status])


def keep_taught_fact(text: str, knowledge, lang: str) -> KeepResult | None:
    """Called only at the LLM boundary. Never stores an unverified speaker."""
    if speaker_unverified():
        return None
    sentence = _sentence(text)
    if sentence is None:
        return None

    def result(status):
        return KeepResult(status, sentence, lang)

    if len(sentence) > MAX_SENTENCE_CHARS:
        return result("too_long")
    if redact_text(sentence) != sentence:
        return result("sensitive")
    if teaching_is_empty(sentence):
        return result("empty")

    with _WRITE_LOCK:
        temporary = None
        try:
            # Lazy on the orchestrator path: ordinary turns and unverified
            # speakers never even need to access a memory/knowledge instance.
            knowledge = knowledge() if callable(knowledge) else knowledge
            path = Path(knowledge.approved_dir) / FILENAME
            if path.is_symlink():
                return result("write_failed")
            # newline="" prevents universal-newline conversion of the user's text.
            if path.exists():
                with path.open(encoding="utf-8", newline="") as source:
                    content = source.read()
                if not content.startswith(TITLE + "\n"):
                    return result("write_failed")
            else:
                content = TITLE + "\n"
            if any(old == sentence for _, old in taught_sections(content)):
                return result("already")
            stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            content += (f"\n## {stamp}\n"
                        f"<!-- cortex:taught-chars={len(sentence)} -->\n\n"
                        f"{FRAMING}\n\n{sentence}\n")
            fd, temporary = tempfile.mkstemp(prefix=".pedro_ensinou-", suffix=".tmp",
                                             dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
            temporary = None
        except Exception:
            return result("write_failed")
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        try:
            indexed = knowledge.index_file(str(path)) is True
        except Exception:
            indexed = False
        status = "kept" if indexed else "index_failed"
        audit.log("–", "memory", "taught_fact_write",
                  payload={"filename": FILENAME, "status": status})
        return result(status)


@dataclass
class TaughtTurn:
    text: str
    result: KeepResult | None = None

    def append_receipt(self, reply: str) -> str:
        if self.result is None:
            return reply
        return reply + ("\n\n" if reply else "") + self.result.receipt


_TURN: ContextVar[TaughtTurn | None] = ContextVar("taught_turn", default=None)


@contextmanager
def taught_turn(text: str, *, reuse: bool = False):
    # A streaming worker opens an outer scope to read the receipt produced
    # by process(). Neither scope crosses a generator yield/thread boundary.
    current = _TURN.get()
    if reuse and current is not None and current.text == (text or ""):
        yield current
        return
    turn = TaughtTurn(text or "")
    token = _TURN.set(turn)
    try:
        yield turn
    finally:
        _TURN.reset(token)


def prepare_taught_fact(knowledge, lang: str) -> KeepResult | None:
    turn = _TURN.get()
    if turn is None:
        return None
    turn.result = keep_taught_fact(turn.text, knowledge, lang)
    return turn.result
