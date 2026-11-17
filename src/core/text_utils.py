"""
Text processing utilities shared across Cortex modules.

Currently provides `strip_markdown_for_speech`, used by:
- src/voice/listener.py  (before passing reply to macOS `say`)
- src/api.py /ask_voice    (before returning plain text to iOS Shortcuts)
"""
import os
import re


# Whole tokens only: do not turn cursos, lollipop or embedded Kkk into laughs.
LAUGH_RE = re.compile(
    r"(?<!\w)(?:k{3,}|(?:rs){2,}|(?:ha){2,}a*|he(?:he)+e*|"
    r"(?:hua){2,}|lol|lmao)(?!\w)", re.IGNORECASE,
)
# Include keycaps as a unit (preserve ordinary digits), flags, skin tones,
# joined sequences, variation selectors and subdivision-flag tag characters.
EMOJI_RE = re.compile(
    r"[0-9#*]\ufe0f?\u20e3|["
    r"\U0001f000-\U0001faff\u2600-\u27bf"
    r"\u00a9\u00ae\u203c\u2049\u2122\u2139\u2194-\u2199\u21a9-\u21aa"
    r"\u231a-\u231b\u2328\u23cf\u23e9-\u23f3\u23f8-\u23fa\u24c2"
    r"\u25aa-\u25ab\u25b6\u25c0\u25fb-\u25fe\u2934-\u2935"
    r"\u2b05-\u2b07\u2b1b-\u2b1c\u2b50\u2b55\u3030\u303d\u3297\u3299"
    r"\u200d\ufe0e\ufe0f\u20e3\U000e0020-\U000e007f]"
)


def strip_markdown_for_speech(text: str, language: str | None = None) -> str:
    """Clean only the speech copy; preserve the original reply for display.

    The listener supplies its selected voice language. Other callers use the
    existing text-language heuristic, with PT for a laugh-only utterance.
    Empty CORTEX_SPOKEN_LAUGH_PT/EN values deliberately silence laughs.
    """
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+?)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^\*]+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^[\-\*\+]\s+", "", text, flags=re.MULTILINE)
    if language is None:
        from src.core.i18n import _detect_lang

        words = LAUGH_RE.sub("", EMOJI_RE.sub("", text))
        language = _detect_lang(words) if re.search(r"[^\W\d_]", words) else "pt"
    laugh = os.getenv(
        "CORTEX_SPOKEN_LAUGH_EN" if language == "en" else "CORTEX_SPOKEN_LAUGH_PT",
        "Ha, ha!" if language == "en" else "Há, há!",
    )
    text = LAUGH_RE.sub(lambda match: laugh, text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"\n\s*\n", ". ", text)
    text = re.sub(r"(?<=[.!?…:;])[ \t]*\n", " ", text)
    text = re.sub(r"\n+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
