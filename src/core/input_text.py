"""Transport cleanup shared by terminal and orchestrator entry points."""


def strip_paste_markers(text: str) -> str:
    """Remove only bracketed-paste delimiters, including mid-line repeats."""
    return text.replace("\x1b[200~", "").replace("\x1b[201~", "")
