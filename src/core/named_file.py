"""A file asked for by NAME, without a path: "lê meu currículo" (s87, 4.3).

Measured before (real 8B, both entry points): 1 useful reply in 28. "lê meu
currículo" never searched ("Não consegui ler o seu currículo. Você pode ler
ele para mim?"); "cadê meu currículo?" once named Cortex_comandos.txt as the
CV; "abre meu perfil britânico" tried to open an APP; "lê o perfil
britânico" invented a document. Two causes: when the 8B searched, it searched
by CONTENT every time (10 of 10), and "punctuation" inside code files buried
the one PDF called that; and nothing knew that currículo = CV = resume — by
name, "currículo" finds 0 files, "cv" 37.

So the words are read here, by rule, and the search is by NAME with Pedro's
synonyms (16/10). The model sees none of it. This module is pure apart from
`find()`, which calls Spotlight through file_finder (paths only, never
contents, the same exclusions).
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

READ, LOCATE, OPEN = "read", "locate", "open"

# The user's groups (16/10): a word in a group searches for every word in it.
SYNONYM_GROUPS = (
    ("curriculo", "cv", "resume", "curriculum"),
    ("prova", "exam", "test", "paper"),
    ("trabalho", "homework", "assignment"),
    ("perfil", "profile"),
)

# Words that name a TYPE of file, not the file: "o pdf de punctuation".
_TYPE_WORDS: Dict[str, Optional[str]] = {
    "pdf": ".pdf", "docx": ".docx", "txt": ".txt", "md": ".md",
    "arquivo": None, "arquivos": None, "documento": None, "documentos": None,
    "doc": None, "file": None, "document": None,
}
_STOPWORDS = frozenset(
    "de do da dos das o a os as um uma meu minha meus minhas seu sua "
    "em no na nos nas pra para por sobre chamado chamada que com "
    "my the a an of on about called your".split())

# Nouns that belong to other routes: "lê meu email" is the email tools'
# (it never matched the email regex — it reached them through the 8B), "lê a
# tela" the screen's, "minhas notas" the Notes'. Not a file unless the user says
# so with a type word: "lê o pdf do email do banco".
_OTHER_ROUTES_WORDS = frozenset(
    "email emails mail mails inbox caixa mensagem mensagens message messages "
    "nota notas note notes lembrete lembretes reminder reminders agenda "
    "calendario calendar evento eventos event events contato contatos contact "
    "contacts noticia noticias news tela screen musica musicas music playlist "
    "spotify clima tempo weather".split())

_DET = (r"(?:(?:o|a|os|as)\s+)?(?:meu|minha|meus|minhas)"
        r"|o|a|os|as|my|the")
_POSSESSIVE = frozenset({"meu", "minha", "meus", "minhas", "my"})
_VERBS = (
    (READ, r"l[eê]|leia|ler|read"),
    (LOCATE, r"cad[êe]|onde\s+(?:est[áa]|t[áa])|ach[ae]|procur[ae]|encontr[ae]|"
             r"where\s+(?:is|are)|find|locate"),
    (OPEN, r"abr[ea]|abrir|open"),
)
_REQUEST_RE = re.compile(
    r"^\s*(?:pode\s+|me\s+)?(?P<verb>" + "|".join(f"(?P<{name}>{rx})" for name, rx in _VERBS) + r")"
    r"(?:\s+(?:pra|para)\s+mim)?\s+(?P<det>" + _DET + r")\s+(?P<target>[^?!,;\n]+)"
    r"(?P<rest>[?!,;\n].*)?$",
    re.IGNORECASE | re.DOTALL,
)
# The target ends where the rest of the request starts: "o pdf de
# punctuation e responde a questão 5".
_TARGET_END_RE = re.compile(r"\s+(?:e|and)\s+", re.IGNORECASE)
_READ_IN_REST_RE = re.compile(
    r"\b(?:l[eê]|leia|ler|read|respond[ae]|responder|answer|resum[ae]|summari[sz]e)\b",
    re.IGNORECASE)


@dataclass(frozen=True)
class Request:
    verb: str                      # READ, LOCATE or OPEN
    target: str                    # what Pedro called it: "pdf de punctuation"
    terms: Tuple[str, ...]         # folded words that must be in the name
    suffix: Optional[str]          # ".pdf" when he said "o pdf de …"
    strong: bool                   # "meu"/"my" or a type word: surely a file


def fold(text: str) -> str:
    """Lower case, no accents: "Britânico" → "britanico"."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^0-9a-z]+", fold(text)) if t]


def parse_request(text: str) -> Optional[Request]:
    """The request in "lê meu currículo", "cadê o pdf de punctuation? lê e
    responde a questão 5", "abre meu perfil britânico" — None when the
    message is not one. "abre o spotify" is an app: OPEN needs "meu" or a
    type word ("abre o pdf de …")."""
    m = _REQUEST_RE.match(text or "")
    if not m:
        return None
    verb = next(name for name, _ in _VERBS if m.group(name))
    target_raw = _TARGET_END_RE.split(m.group("target"), maxsplit=1)
    target = target_raw[0].strip(" .")
    rest = (target_raw[1] if len(target_raw) > 1 else "") + (m.group("rest") or "")
    suffix, typed, terms = None, False, []
    for tok in _tokens(target):
        if tok in _TYPE_WORDS:
            typed = True
            suffix = suffix or _TYPE_WORDS[tok]
        elif tok not in _STOPWORDS:
            terms.append(tok)
    if not terms:
        return None                 # "lê o arquivo" — the other routes' job
    if not typed and any(t in _OTHER_ROUTES_WORDS for t in terms):
        return None                 # "lê meu email" — another route's words
    possessive = any(t in _POSSESSIVE for t in fold(m.group("det")).split())
    if verb == OPEN and not (possessive or typed):
        return None                 # "abre o spotify"
    if verb == LOCATE and _READ_IN_REST_RE.search(rest):
        verb = READ                 # "cadê o pdf de X? lê e responde…"
    return Request(verb=verb, target=target, terms=tuple(terms), suffix=suffix,
                   strong=possessive or typed)


def expand(term: str) -> Tuple[str, ...]:
    """A term and its synonyms (Pedro's groups)."""
    for group in SYNONYM_GROUPS:
        if term in group:
            return group
    return (term,)


def name_matches(term: str, filename: str) -> bool:
    """A term (or one of its synonyms) is a WORD of the file name — "cv"
    matches "Pedro_CV_one_page.docx", not "opencv_notes.pdf"; a longer
    term may also start a word ("britanic" → "britanico")."""
    tokens = _tokens(os.path.splitext(filename)[0])
    for word in expand(term):
        for tok in tokens:
            if tok == word or (len(word) > 3 and tok.startswith(word)):
                return True
    return False


@dataclass(frozen=True)
class Found:
    path: str
    name: str
    modified: Optional[str]


def rank(request: Request, hits: List[Found]) -> List[Found]:
    """The files whose name has EVERY term (synonyms counted), newest first.
    "perfil britânico" keeps perfil_britanico.md and drops perfil_dev.md."""
    seen, out = set(), []
    for hit in hits:
        if hit.path in seen:
            continue
        seen.add(hit.path)
        if request.suffix and not hit.name.lower().endswith(request.suffix):
            continue
        if all(name_matches(term, hit.name) for term in request.terms):
            out.append(hit)
    return sorted(out, key=lambda h: h.modified or "", reverse=True)


def find(request: Request, readable_only: bool) -> Tuple[List[Found], Optional[str]]:
    """Spotlight by NAME for each term and synonym, ranked. `readable_only`
    keeps what the reader can open (pdf, txt, md, docx); otherwise any
    regular file."""
    from src.tools.file_finder import finder, is_readable_path

    pool: List[Found] = []
    error = None
    for term in request.terms:
        for word in expand(term):
            hits, err = finder.search(word, limit=50)
            error = error or err
            pool += [Found(h.path, h.name, h.modified) for h in hits]
    if readable_only:
        pool = [h for h in pool if is_readable_path(h.path)[0]]
    else:
        # Files only: a folder or an app (a .app IS a folder) is not what
        # "abre meu perfil britânico" means.
        pool = [h for h in pool if os.path.isfile(h.path)]
    return rank(request, pool), error


_ORDINALS = {
    "1": 0, "um": 0, "primeiro": 0, "primeira": 0, "first": 0,
    "2": 1, "dois": 1, "segundo": 1, "segunda": 1, "second": 1,
    "3": 2, "tres": 2, "terceiro": 2, "terceira": 2, "third": 2,
    "4": 3, "quatro": 3, "quarto": 3, "quarta": 3, "fourth": 3,
    "5": 4, "cinco": 4, "quinto": 4, "quinta": 4, "fifth": 4,
}
_PICK_RE = re.compile(
    r"^\s*(?:(?P<verb>l[eê]|leia|ler|read|abr[ea]|abrir|open)\s+)?"
    r"(?:o|a|the)?\s*(?:n[uú]mero\s+|number\s+|#)?(?P<which>[0-9a-zçêé]+)"
    r"(?:\s+(?:deles|delas|one))?\s*[.!]?\s*$",
    re.IGNORECASE)


def parse_pick(text: str) -> Optional[Tuple[int, Optional[str]]]:
    """(index, verb) for "2", "o segundo", "abre o primeiro", "read the
    third one" — the answer to a numbered list. None otherwise."""
    m = _PICK_RE.match(text or "")
    if not m:
        return None
    index = _ORDINALS.get(fold(m.group("which")))
    if index is None:
        return None
    verb = m.group("verb")
    if verb:
        verb = OPEN if fold(verb).startswith(("abr", "open")) else READ
    return index, verb
