"""
Dream-review dispatch (Fase 2 — DORMANT wiring).

Lets the user review the latest pending Dream by voice/text: show it, keep it,
or discard it. Gated by CORTEX_DREAM_LOOP (default off) in the handler — when
off it falls through. Dreams are only produced once the Dream Loop is active
(needs a strong model), so until then there's nothing to review; the wiring is
here so Mac Studio day is just flipping the flag.

Three actions via named groups: show / keep / drop.
"""
from __future__ import annotations

import re

DREAM_REVIEW_RE = re.compile(
    r"^(?:"
    # show the latest dream
    r"(?P<show>(?:me\s+)?(?:l[êe]|ler|mostra(?:r)?|conta(?:r)?|qual\s+(?:foi\s+)?o)\s+"
    r"(?:o\s+|meu\s+|[úu]ltimo\s+)*sonho"
    r"|show\s+(?:me\s+)?(?:my\s+|the\s+|the\s+last\s+)?dream)"
    r"|"
    # keep / approve
    r"(?P<keep>(?:guarda(?:r)?|aprova(?:r)?|salva(?:r)?|mant[ée]m)\s+"
    r"(?:o\s+|esse\s+|este\s+|esse\s+[úu]ltimo\s+)?sonho"
    r"|keep\s+(?:the\s+|that\s+)?dream)"
    r"|"
    # discard / reject
    r"(?P<drop>(?:descarta(?:r)?|rejeita(?:r)?|apaga(?:r)?|esquece(?:r)?)\s+"
    r"(?:o\s+|esse\s+|este\s+)?sonho"
    r"|discard\s+(?:the\s+|that\s+)?dream)"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
