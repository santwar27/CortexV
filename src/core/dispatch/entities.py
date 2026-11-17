"""
Entity-brain dispatch (Onda 5, M1-now slice).

Two commands, both answered deterministically from structured sources:
  • LAST_CONTACT_RE  — "quando falei com X pela última vez?" (captures X).
  • RECENT_PEOPLE_RE — "com quem andei falando?" / "pessoas recentes".

Anchored + narrow so they don't eat normal conversation.
"""
from __future__ import annotations

import re

# "quando (foi a última vez que) (eu) falei/conversei com X (pela última vez)"
# "when did I last talk to/speak with/email X"
LAST_CONTACT_RE = re.compile(
    r"^(?:"
    r"quando\s+(?:foi\s+(?:a\s+)?[úu]ltima\s+vez\s+que\s+)?(?:eu\s+)?"
    r"(?:falei|conversei|troquei\s+mensagens?|me\s+falei)\s+com\s+"
    r"|[úu]ltima\s+vez\s+que\s+(?:eu\s+)?(?:falei|conversei)\s+com\s+"
    r"|when\s+did\s+i\s+last\s+(?:talk\s+to|speak\s+(?:with|to)|email|message)\s+"
    r")(?P<name>.+?)(?:\s+pela\s+[úu]ltima\s+vez)?[\s.!?]*$",
    re.IGNORECASE,
)

# "com quem (eu) (andei) falei/falando", "pessoas recentes", "minhas pessoas"
RECENT_PEOPLE_RE = re.compile(
    r"^(?:"
    r"com\s+quem\s+(?:eu\s+)?(?:andei\s+)?(?:falei|falando|conversei|conversando|interagi)"
    r"|(?:minhas\s+)?pessoas\s+recentes|pessoas\s+(?:com\s+quem|que)\s+.+"
    r"|recent\s+(?:people|contacts)|who\s+have\s+i\s+(?:talked|spoken)\s+to"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
