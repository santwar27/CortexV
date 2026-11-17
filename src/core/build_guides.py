"""Small cached guides, selected once at the common orchestrator boundary."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata


def normalise(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(c))


_ACTION = re.compile(
    r'\b(?:cria|crie|criar|faz|faca|fazer|escreve|escreva|escrever|muda|mude|mudar|'
    r'altera|altere|alterar|corrige|corrija|corrigir|implementa|implemente|implementar|'
    r'desenvolve|desenvolva|desenvolver|monta|monte|montar|'
    r'create|build|write|make|change|edit|fix|implement|develop|redesign|restyle)\b')
# At most four closed filler phrases between a verb and its direct object.
# Languages qualify a code noun; they are not standalone build objects.
_FILLER = (r'(?:pra\s+mim|para\s+mim|for\s+me|me|um|uma|o|a|os|as|the|an|'
           r'novo|nova|simples|pequeno|pequena|rapido|rapida|one-page|single-page|'
           r'simple|new|small|quick|basic|meu|minha|meus|minhas|my|our|nosso|nossa)')
_OBJECT = re.compile(r'\s+(?:' + _FILLER + r'\s+){0,4}')
_WEB = re.compile(r'(?:site|website|web[ -]?page|pagina\s+(?:web|html|de\s+internet)|'
                  r'landing\s+page|homepage|home\s+page|html|css|front[ -]?end|'
                  r'layout|estilo\s+do\s+site|website\s+style)\b')
_CODE = re.compile(r'(?:(?:python|javascript|js)\s+)?'
                   r'(?:script|funcao|function|programa|program|api|back[ -]?end|codigo|code)\b')
_QUESTION = re.compile(r'^(?:como|how|o que|what|por que|why|explica|explique|explain)\b')


def detect_guide(text: str) -> str | None:
    text = normalise(text or '')
    if _QUESTION.search(text.strip()):
        return None
    kinds = set()
    for action in _ACTION.finditer(text):
        fillers = _OBJECT.match(text, action.end())
        if fillers:
            if _WEB.match(text, fillers.end()):
                kinds.add('web')
            elif _CODE.match(text, fillers.end()):
                kinds.add('code')
    # A combined code + web build uses the web publication guide.
    return 'web' if 'web' in kinds else 'code' if 'code' in kinds else None


@lru_cache(maxsize=2)
def guide_text(kind: str) -> str:
    if kind not in ('web', 'code'):
        raise ValueError('unknown build guide')
    return Path(__file__).with_name('guides').joinpath(kind + '.md').read_text(encoding='utf-8')
