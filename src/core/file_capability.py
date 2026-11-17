"""Correct false file-capability denials, only when no tool ran (task 121)."""
import re


FILE_CAPABILITY_REPLY = {
    "pt": 'Eu consigo salvar arquivos. Para gravar o código da minha última resposta, '
          'diga "salva esse código". Para guardar um texto, diga "escreve no <nome> <texto>".',
    "en": 'I can save files. To save the code from my last reply, say "save this code". '
          'To keep a text, say "store <name> <text>".',
}

_PT_ABILITY = r"(?:a\s+)?(?:capacidade|permiss[aã]o)\s+(?:de|para)|como(?:\s+de)?"
_PT_DENIAL = (
    r"(?:\bn[aã]o\s+(?:posso|consigo|tenho\s+(?:" + _PT_ABILITY + r"))|"
    r"\b(?:o\s+)?cortex\b[^.!?\n]{0,160}?\bn[aã]o\s+"
    r"(?:pode|consegue|tem\s+(?:" + _PT_ABILITY + r")))"
    r"\s+(?:diretamente\s+)?(?:criar|salvar|gravar|escrever)\s+"
    r"(?:diretamente\s+)?(?:(?:um|uma|o|a|os|as|uns|umas)\s+)?"
    r"(?:arquivos?|documentos?|docs?)\b"
)
_EN_DENIAL = (
    r"\b(?:i\s+(?:can(?:not|'t)|am\s+unable\s+to|"
    r"(?:do\s+not|don't)\s+have\s+(?:the\s+)?(?:ability|permission|capability)\s+to)|"
    r"cortex\s+(?:can(?:not|'t)|is\s+unable\s+to|"
    r"(?:does\s+not|doesn't)\s+have\s+(?:the\s+)?(?:ability|permission|capability)\s+to))"
    r"\s+(?:directly\s+)?(?:create|save|write)\s+(?:directly\s+)?"
    r"(?:(?:a|an|the|your|any)\s+)?(?:files?|documents?|docs?)\b"
)
_DENIAL_RE = re.compile(_PT_DENIAL + "|" + _EN_DENIAL, re.IGNORECASE)
_QUOTES_RE = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»|(?<!\w)\'[^\'\n]*\'(?!\w)|`[^`]*`')
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ERROR_RE = re.compile(
    r"^\s*(?:\[(?:SYSTEM\s+)?ERROR\]|tool\s+(?:error|returned)\b|"
    r"(?:a\s+)?ferramenta\s+(?:respondeu|retornou)\s*:)", re.IGNORECASE,
)


def denies_file_capability(reply: str) -> bool:
    """Look for capability assertions in prose, excluding code/quoted errors.

    Unlike the action guard, this checks every sentence: the false denial
    often follows an explanation. Read/open errors and past failures do not
    deny Cortex's ability to create/save/write and deliberately do not match.
    """
    prose = []
    fence = None
    for line in (reply or "").splitlines():
        marker = _FENCE_RE.match(line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence or line.startswith(("    ", "\t")) or line.lstrip().startswith(">") or _ERROR_RE.match(line):
            # Keep a boundary: excluded material must not join two assertions.
            prose.append(".")
            continue
        prose.append(line)
    text = _QUOTES_RE.sub(" ", "\n".join(prose))
    text = text.replace("’", "'").replace("**", "").replace("__", "")
    return bool(_DENIAL_RE.search(text))
