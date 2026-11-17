"""Input normalisation applied before regex dispatch.

The dispatch regexes anchor at `^` to keep them cheap and unambiguous,
but real-world inputs from voice and chat frequently lead with a
vocative ("olá cortex, ...", "ei jarvis ...") that mis-anchors the
match. Without this step a polite "olá cortex, pode tocar matue no
spotify?" missed `SPOTIFY_PLAY_RE` and fell through to the LLM, which
then hallucinated a fake track listing (sessão 10, bloco F — see
JOURNAL).

Stripping the vocative is semantically lossless for routing: the
assistant identity is established by the system prompt, not by the
user's salutation, and the regexes care about the verb that follows.

Order of operations in the orchestrator's entry points
(`process_stream` and `_process_core`):
  1. `.strip()` (already done)
  2. `strip_vocatives()` — this module
  3. routing copy: `strip_trailing_sentence_punctuation()`, courtesy, aliases
  4. regex dispatch / pure-chat check; LLM fallback keeps the original text
"""

from __future__ import annotations

import re

from src.core.dispatch.flow_control import APPROVAL_RE, DENIAL_RE
from src.core.dispatch.reminders import REMINDER_TRIGGER_RE
from src.core.dispatch.studier_cmds import ADD_STUDY_RE, APPROVE_STUDY_RE, STUDY_RE


_SENTENCE_END_RE = re.compile(r"[.!?…]+$")
_ABBREVIATION_END_RE = re.compile(r"\b(?:[a-zA-Z]\.)+[a-zA-Z]$")


def strip_trailing_sentence_punctuation(text: str) -> str:
    """Remove a transcript's sentence ending from a ROUTING copy only.

    Trim trailing whitespace, then one terminal run of . ! ? … . Never
    change interior characters or empty a punctuation-only utterance.
    Closing quotes stay literal: removing one can corrupt a dictated title
    or quoted payload. Preserve dotted abbreviations (e.g.) and the study
    queue's existing terminal !! force flag, which is command syntax. Keep
    a meaningful question cue when removing it would turn "lembra disso?"
    (recall/chat) into a reminder; timed reminder requests still normalise.
    All approval/authority checks must continue to inspect their original
    input, including its punctuation. Denials may use this routing copy.
    """
    out = text.rstrip()
    ending = _SENTENCE_END_RE.search(out)
    if ending is None:
        return out
    body = out[:ending.start()]
    if not body.strip():
        return text
    if ("?" in ending.group() and REMINDER_TRIGGER_RE.match(body)
            and not REMINDER_TRIGGER_RE.match(body + "?")):
        return body + "?"
    study_text = strip_courtesy(out)
    # The study selector is authority too: "approve 1?" must reach its
    # approval handler with exactly the same selector as before stripping.
    if APPROVE_STUDY_RE.match(study_text):
        return out
    if (STUDY_RE.match(study_text) or ADD_STUDY_RE.match(study_text)):
        if out.endswith("!!"):
            return out
        if ending.group().startswith("!!"):
            return body + "!!"
    # The first dot in e.g... belongs to the abbreviation; only the rest
    # is sentence punctuation. Numbers/versions/paths end in non-mark text.
    if ending.group().startswith(".") and _ABBREVIATION_END_RE.search(body):
        body += "."
    return body.rstrip()


# Greeting + optional assistant name. Each token covers PT and EN
# variants. The trailing class swallows any punctuation/whitespace
# that separates the vocative from the actual command.
#
# Examples that match (group 0 = the entire vocative prefix):
#   "olá cortex, "
#   "oi cortex! "
#   "ei jarvis - "
#   "bom dia "                       (greeting alone, no name)
#   "good morning cortex: "
_VOCATIVE_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:"
    r"ol[áa]|oi+e?|al[oô]+|hey+|hi+|hello+|"
    r"bom\s+dia|boa\s+tarde|boa\s+noite|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"opa|e[aiy]+|yo"
    r")"
    r"(?:[\s,!.;:\-–—]+(?:c[óo]rtex|jarvis))?"
    # The user's (s87) — only WITH the name: "fala mais alto" is a command,
    # and "e aí" alone is a question.
    r"|(?:fala|salve|e\s+a[íi])[\s,!.;:\-–—]+(?:c[óo]rtex|jarvis)"
    r")"
    r"[\s,!.;:\-–—]+",
    re.IGNORECASE,
)

# A greeting that IS the whole message: "ola cortex", "Olá Cortex!", "e aí
# cortex". _VOCATIVE_RE needs a separator AFTER the name (it was built for
# "olá cortex, abre o spotify"), so on "ola cortex" it backtracked to "ola "
# and left the bare name: the reflex — which answers "oi", not "cortex" —
# missed, and the 8B got one word ("I'm thinking of a colour... Blue.", s87;
# 3 of 3 such messages in the audit). The greeting comes back alone, for the
# reflex. the user's greetings (16/10) included; "good night" is a goodbye.
_GREETING_ONLY_RE = re.compile(
    r"^\s*(?P<greeting>"
    r"ol[áa]|oi+e?|ei|hey+|hi+|hello+|"
    r"bom\s+dia|boa\s+tarde|boa\s+noite|"
    r"good\s+(?:morning|afternoon|evening)|"
    r"e\s*a[íi]|eae|fala|salve|opa"
    r")"
    r"[\s,!.;:\-–—]+(?:c[óo]rtex|jarvis)[\s,!.?;:]*$",
    re.IGNORECASE,
)

# Bare name without greeting: "cortex, toca matue" / "jarvis: pause".
# Matched independently from _VOCATIVE_RE because the assistant name
# alone is also a valid sentence starter.
_BARE_NAME_RE = re.compile(
    r"^\s*(?:c[óo]rtex|jarvis)[\s,!.;:\-–—]+",
    re.IGNORECASE,
)


# Acknowledgement prefixes (sessão 13 bloco F bug 2). Real chat
# logs show users prefixing commands with "ok.", "okay,", "beleza,"
# — fillers that desanchor the `^`-anchored dispatch regexes the
# same way vocatives do. These carry no routing intent (they
# acknowledge the previous turn) so it's safe to strip.
#
# Conservative scope: only prefixes that MUST be followed by
# punctuation or whitespace, so we don't accidentally chop the
# leading word of a real command. e.g. "ok faz X" — "ok" only
# strips when followed by ".", ",", "!", "?" or "\s+", not when
# attached to the next token without a separator.
_ACK_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"ok|okay|okey|kk|kkk|"
    r"belezas?|blz|beleza\s+ent[ãa]o|"
    r"show\s+de\s+bola|"
    r"valeu|vlw|vlws|"
    r"perfeito|"
    r"got\s+it|alright|sounds\s+good"
    r")"
    r"[\s,!.;:?]+",
    re.IGNORECASE,
)

# "show" sozinho como ack PT-BR ("show!" = "nice!") só vale quando
# vem pontuação a seguir — espaço puro NÃO conta, senão "show me my
# last 3 emails" vira "me my last 3 emails" e perde o LIST_EMAILS_RE
# (sessão 20 fix). Mantém "show, próxima música" e "show!" como ack
# legítimo, mas deixa "show + verbo" intacto pro dispatch real.
_ACK_SHOW_RE = re.compile(
    r"^\s*show[,!.;:?]+\s*",
    re.IGNORECASE,
)


# A leading run of emoji, dashes or bullets set off by a space (s88, Codex
# lote 9): "🙂 toca música" and "— fecha o Safari" missed their verb-first
# routes and went to the model. Only symbols, and only before a space, so a
# word or a number ("-5 graus") is never touched. Not ">": in markdown it
# quotes someone else, and a quoted order is not the user's.
_GRAPHIC_LEAD_RE = re.compile(
    "^\\s*[\u2010-\u2015\\-\u2022\u00b7*~\u2190-\u21ff\u2300-\u27bf"
    "\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f\u200d]+\\s+"
)


def strip_vocatives(text: str) -> str:
    """Remove a leading greeting / assistant-name / acknowledgement
    prefix from `text`.

    Strips up to two layers in order: (1) ack prefix ("ok.",
    "beleza,"), (2) vocative ("olá cortex,") or bare name ("cortex,").
    The ack pass runs first so chains like "ok. cortex, toca X"
    collapse cleanly to "toca X". Idempotent and safe on empty input.

    Returns a `.strip()`'d result so downstream regex `^` anchors
    land on the first real character. Returns the input unchanged
    if nothing matched — never raises, never returns None when
    given a non-empty string.
    """
    if not text:
        return text
    out = _GRAPHIC_LEAD_RE.sub("", text, count=1)
    out = _ACK_PREFIX_RE.sub("", out, count=1)
    out = _ACK_SHOW_RE.sub("", out, count=1)
    greeting_only = _GREETING_ONLY_RE.match(out)
    if greeting_only:
        return greeting_only.group("greeting")
    out2 = _VOCATIVE_RE.sub("", out, count=1)
    if out2 == out:
        # No greeting matched — try the bare-name pattern. Doing them
        # in series instead of one alternation keeps each regex
        # readable and lets us tune greetings independently of names.
        out2 = _BARE_NAME_RE.sub("", out, count=1)
    result = out2.strip()
    if not result:
        # Stripping emptied a NON-empty message: the whole input WAS the
        # ack/vocative ("ok.", "beleza!", bare "cortex?"). That's a real
        # message — most importantly a legitimate reply to a confirmation
        # prompt (APPROVAL_RE itself expects "ok" with [.!]?$) — and both
        # entry points would answer "[SYSTEM ERROR] Empty input." with the
        # action still staged (M1-A4). Return the original so the approval
        # check (or plain chat) can claim it downstream.
        return text.strip()
    return result


# Courtesy wrappers (sessão 75, Onda 8 A — smoke finding 1). The
# dispatch regexes are verb-first ("open|abre ..."), so a polite
# envelope defeats them: "Can you open Spotify please?" fell to the
# OFFLINE chat path while "Open the Spotify" routed to MAC_OPEN_APP.
# The wrapper carries no routing intent — strip it the same way
# strip_vocatives drops "olá cortex,".
#
# IMPORTANT: this strip is for ROUTING ONLY. "can you hear me?" must
# reach the LLM intact ("hear me?" is a different sentence), so the
# orchestrator applies strip_courtesy to a routing-side variable and
# keeps the original text for the chat/tool fallback.
_COURTESY_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"please\s+|"
    r"por\s+favor[\s,]+|"
    r"(?:voc[êe]|vc|tu)\s+(?:pode(?:ria)?|consegue)\s+|"
    r"pode(?:ria)?\s+|"
    r"consegue\s+|"
    r"d[áa]\s+pra\s+(?:voc[êe]\s+)?"
    r")",
    re.IGNORECASE,
)

# Trailing politeness. Requires a separator before the token so song
# titles ending in the word stay whole ("play Please Please Me" ends
# in "me", not in a courtesy token). "pra mim" included: OPEN_APP's
# lazy capture + `$` anchor would otherwise swallow it into the app
# name ("abre o spotify pra mim" → app "spotify pra mim").
_COURTESY_SUFFIX_RE = re.compile(
    r"[\s,]+(?:please|por\s+favor|por\s+gentileza|pfv?|"
    r"p(?:a)?ra\s+mim)[\s.!?]*$",
    re.IGNORECASE,
)


def strip_courtesy(text: str) -> str:
    """Remove a leading/trailing courtesy wrapper for dispatch routing.

    Guards: an input that IS an approval/denial reply ("pode crer",
    "pode.", "não") returns unchanged — APPROVAL_RE owns "pode" as a
    yes, and chopping it would break the staged-action sim/não flow. A strip that empties a non-empty
    message also returns the original (same contract as
    strip_vocatives). Idempotent.
    """
    if not text:
        return text
    if APPROVAL_RE.match(text) or DENIAL_RE.match(text):
        return text
    out = text
    # Up to two prefix layers: "please can you open X" / "por favor,
    # você pode abrir X".
    for _ in range(2):
        new = _COURTESY_PREFIX_RE.sub("", out, count=1)
        if new == out:
            break
        out = new
    # After a ":" comes what the user dictates — a body, a subject, a note — and
    # its last words are his: "corpo: Compre pão para mim" staged "Compre pão"
    # (s88, Codex lote 9). The trailing courtesy is only cut from a command.
    if ":" not in out:
        out = _COURTESY_SUFFIX_RE.sub("", out, count=1)
    result = out.strip()
    if not result:
        return text.strip()
    return result


def removed_portuguese_courtesy_suffix(text: str) -> str | None:
    """Return only a trailing pra/para mim actually removed for routing.

    Keep strip_courtesy's contract intact, including dictated colon bodies
    and inputs that would become empty. Other courtesy tokens are ineligible.
    """
    match = _COURTESY_SUFFIX_RE.search(text or "")
    if not match or ":" in text or strip_courtesy(text) == text.strip():
        return None
    portuguese = re.fullmatch(r"[\s,]+(p(?:a)?ra\s+mim)[\s.!?]*",
                              match.group(), re.IGNORECASE)
    return portuguese.group(1) if portuguese else None


# STT mishearings (sessão 75 cont.). Whisper small + BR accent on
# English product names produces stable, observed-in-audit corruptions
# of words the dispatch regexes (and the Spotify capture groups) need
# verbatim. Same family as the "Meck/Meque"→Mac aliases that live
# inside the macos regexes — but these hit CAPTURED text ("open spot 5"
# captures app "spot 5"), so they're fixed by rewriting the routing
# text, not by widening a verb regex.
#
# Conservative scope: only corruptions actually observed in the audit
# log, with word boundaries so genuine phrases survive. Routing-only,
# like strip_courtesy — the LLM still sees what Whisper heard.
#   "Can you open spot 5 please?" (audit 2026-06-12) → spotify
_STT_ALIASES = (
    # Whisper renders the user's "spotify" as "spot 5" / "spot five".
    (re.compile(r"\bspot\s*-?\s*(?:5|five)\b", re.IGNORECASE), "spotify"),
)


# Spans the aliases never touch; a capturing group, so split() keeps them.
_LITERAL_RE = re.compile(r"(\S+@\S+|https?://\S+|(?:~|/)\S+)")


def normalise_stt_aliases(text: str) -> str:
    """Rewrite known STT mishearings to the word the user actually said.

    Applied to the ROUTING text only (same contract as strip_courtesy):
    dispatch regexes and their capture groups see the corrected word;
    the chat path keeps the raw transcript. Idempotent — the
    replacements never produce text another alias matches.
    """
    if not text:
        return text
    # An address, a URL or a path is literal: "salva o email do João como
    # spot-5@example.com" saved spotify@example.com, and the structured email
    # staged it as the recipient (s88, Codex lote 9). Only the words between
    # them are rewritten.
    parts = _LITERAL_RE.split(text)
    for i in range(0, len(parts), 2):
        for rx, replacement in _STT_ALIASES:
            parts[i] = rx.sub(replacement, parts[i])
    return "".join(parts)
