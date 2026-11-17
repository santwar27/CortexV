"""
Redaction at the edge of every plaintext sink (M6-A6, consolidating
M3-A6 + M5-A9).

The Palace is one place secrets can land, but the review mapped a whole
SECOND trail outside it: `~/.cortex/audit.jsonl` (incl. voice
transcriptions via telemetry), `~/.cortex/gate_queue.jsonl` (S3 args),
`data/action_ledger.jsonl` — all plaintext, all append-only, none
covered by a future "Palace encriptado" (#14). A "minha senha é X"
dictated by voice or stated in chat would otherwise be materialised in
every one of them.

This module is the SINGLE point of decision: one skiplist of sensitive
attribute names + two redactors, applied AT THE BORDER of each sink
(the sink redacts just before writing; producers stay untouched).

  - `is_sensitive_attr(name)`  — is this key/predicate sensitive?
  - `redact_payload(obj)`      — deep-copy dicts/lists, masking values
                                 under sensitive keys and scrubbing
                                 free-text strings.
  - `redact_text(text)`        — scrub "senha ... é X"-style statements
                                 inside free text (the voice-transcript
                                 case, where there is no structured key).

Design bias mirrors fact_extractor's, but INVERTED: there false
positives are the enemy; here OVER-redaction is the safe direction. A
log line that masks too much costs a little debuggability; one that
masks too little leaks a secret into an append-only file. So attr
matching is by token (so `api_key` and `senha_do_banco` hit, but
`monkey`/`happiness` don't), and free-text redaction masks to the end
of the clause.

Pure module (no I/O), like fact_extractor — unit-tests trivially and
imports from anywhere without dragging the core package.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# The mask written in place of a sensitive value. Grep-able on purpose:
# its presence in a log line proves the redaction path ran.
REDACTED = "[REDACTED]"

# Sensitive attribute tokens (pt + en). Matched as WHOLE tokens after
# slugifying the attr name, never as substrings — see _attr_tokens.
_SENSITIVE_TOKENS = frozenset({
    "senha", "password", "pin", "token", "chave", "key",
    "secret", "credencial",
})

# Free-text statement of a secret: a sensitive word, then (within the
# same clause) a copula/separator, then the value. The value is masked
# to the END of the clause — over-redaction is the safe direction.
# The value's first token is taken WHOLE (s88, Codex lote 10): it stopped at
# the first ".", so "minha senha é abc.DEF987" logged "[REDACTED].DEF987" —
# punctuation inside a password is the password; a comma or a period
# followed by a space still ends the clause.
_TEXT_RE = re.compile(
    r"(\b(?:senha|password|pin|token|chave|key|secret|credencial|"
    r"c[óo]digo\s+de\s+(?:recupera[çc][ãa]o|backup|seguran[çc]a)|"
    r"(?:recovery|backup)\s+codes?)\b"
    r"[^.,;:!?\n]*?"
    r"(?:[:=]|\b(?:é|eh|is|são|sao|are)\b)\s*)"
    r"([^\s.,;!?]+(?:[.,;:!?][^\s.,;!?]+)*[^.,;!?\n]*)",
    re.IGNORECASE,
)

# Masked wherever they appear, label or not: a formatted CPF, and API keys
# whose prefix says what they are — pasted bare, "usa a chave sk-…" had no
# copula for _TEXT_RE to find.
_ANYWHERE_RE = re.compile(
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AIza[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|xox[abpr]-[A-Za-z0-9-]{10,}|"
    r"tvly-[A-Za-z0-9_-]{16,})"
)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _attr_tokens(name: str) -> list[str]:
    """Split an attribute name into lowercase ASCII tokens:
    'api_key' → ['api','key'], 'senha do banco' → ['senha','do','banco'].
    Token matching (not substring) keeps 'monkey'/'happiness' clean."""
    ascii_name = _strip_accents(str(name).lower())
    return [t for t in re.split(r"[^a-z0-9]+", ascii_name) if t]


def is_sensitive_attr(name: Any) -> bool:
    """True when an attribute/key/predicate name denotes a secret."""
    if not name:
        return False
    return any(t in _SENSITIVE_TOKENS for t in _attr_tokens(name))


def redact_text(text: str) -> str:
    """Mask secret VALUES stated in free text ("minha senha do banco é
    1234" → "minha senha do banco é [REDACTED]"). Non-str passes through
    unchanged. Never raises — a redaction bug must not break a sink."""
    if not isinstance(text, str) or not text:
        return text
    try:
        text = _TEXT_RE.sub(lambda m: m.group(1) + REDACTED, text)
        return _ANYWHERE_RE.sub(REDACTED, text)
    except Exception:  # noqa: BLE001 — fail to the safe side available
        return REDACTED


def redact_payload(obj: Any) -> Any:
    """Return a redacted deep copy of a payload (dict/list/scalars).

    Values under a sensitive KEY are replaced wholesale with REDACTED
    (whatever their type); free-text string values are scrubbed with
    redact_text; containers recurse. The input is never mutated."""
    try:
        if isinstance(obj, dict):
            return {
                k: (REDACTED if is_sensitive_attr(k) else redact_payload(v))
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            seq = [redact_payload(v) for v in obj]
            return seq if isinstance(obj, list) else tuple(seq)
        if isinstance(obj, str):
            return redact_text(obj)
        return obj
    except Exception:  # noqa: BLE001 — same fail-safe contract as redact_text
        return REDACTED
