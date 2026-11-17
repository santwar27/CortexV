"""Loss guidance is ephemeral context, on sync, stream and deferred turns."""
from unittest.mock import MagicMock

import pytest

from src.core.engine import OllamaEngine
from src.core.prompts import SYSTEM_PROMPT_V6
from src.core.taught_facts import REPLY_NOTE
from tests.test_taught_facts import orch as taught_orch  # noqa: F401


NEWS = "NEWS:\n" + SYSTEM_PROMPT_V6.split("NEWS:\n", 1)[1].split("\n\n", 1)[0]
PATHS = ["sync", "stream", "deferred"]


@pytest.fixture
def orch(taught_orch):
    """Use real thread-local context/history bindings, with model calls faked."""
    o = taught_orch
    o.seen_contexts = []
    o.spotify.play_anything.return_value = "Spotify route."
    for name in ("engine_analytical", "engine_creative"):
        old = getattr(o, name)
        engine = OllamaEngine.__new__(OllamaEngine)
        engine.memory = old.memory
        engine.history = []

        def reply(*args, _engine=engine, **kwargs):
            context = getattr(getattr(_engine, "_turn_context", None), "value", "")
            o.seen_contexts.append(context or "")
            return "Model answer."

        engine.chat = MagicMock(side_effect=reply)
        engine.chat_stream = MagicMock(side_effect=lambda *a, _reply=reply, **kw: iter([_reply()]))
        engine.chat_with_tools = MagicMock(side_effect=lambda *a, _reply=reply, **kw: (_reply(), None))
        setattr(o, name, engine)
    return o


def answer(o, text, path, **kwargs):
    if path == "sync":
        return o.process(text, **kwargs)[0]
    if path == "deferred":
        o._is_pure_chat = lambda text: False
    frames = list(o.process_stream(text, **kwargs))
    assert frames[-1][1]
    return "".join(f[0] for f in frames)


# The documented allowlist: death/bereavement, grave illness, and personal loss.
LOSS_MESSAGES = [
    "Morreu meu pai", "meus avós morreram", "Faleceu meu avô",
    "recebi a notícia do falecimento da Ana", "houve um óbito na família",
    "houve um obito na familia", "Luto pelo meu pai", "estou de luto", "estou em luto",
    "fui ao velório da Ana",
    "fui ao velorio da Ana", "Enterro do meu tio hoje", "hoje foi o funeral da Ana",
    "Câncer: minha mãe recebeu o diagnóstico", "minha mae tem cancer",
    "minha mãe está internada", "meu pai esta internado",
    "perdi meu pai", "ontem perdi minha irmã", "perdi meu cachorro",
    "My father died", "Died today: my father", "my mother passed away",
    "Funeral for my father today", "my mother has cancer", "I lost my father",
    "yesterday I lost my sister", "I lost my dog",
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("text", LOSS_MESSAGES)
def test_loss_note_reaches_engine(orch, path, text):
    assert answer(orch, text.upper(), path) == "Model answer."
    assert orch.seen_contexts == [NEWS]
    for engine in (orch.engine_analytical, orch.engine_creative):
        assert not getattr(engine._turn_context, "value", None)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("text", [
    "no sábado vou jogar um torneio de tênis", "meu treino de tênis é na quinta às 19h",
    "perdi a chave", "perdi meu tempo", "I lost my keys", "o jogo morreu no fim",
    "the battery died", "eu luto pelos meus direitos", "qual a capital da França?",
    "me lembra disso amanhã",
])
def test_note_absent_on_ordinary_turns(orch, path, text):
    orch._handle_reminder_regex = MagicMock(return_value=("Reminder route.", "REMINDER"))
    answer(orch, text, path)
    assert all("NEWS:" not in context for context in orch.seen_contexts)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("text,expected", [
    ("meu avô morreu ontem, não consigo acreditar", True),
    ("ele morreu e eu não sei o que fazer", True),
    ("she passed away, I can't believe it", True),
    ("my father died, I did not expect it", True),
    ("meu avô morreu, nunca vou esquecer", True),
    ("não consigo acreditar, meu avô morreu ontem", True),
    ("I never thought my father died", True),
    ("I can't believe she died", True),
    ("não acredito que morreu", True),
    ("não morreu e morreu", True),
    ("meu avô não morreu", False),
    ("meu avô ainda não morreu", False),
    ("meu avô nunca morreu", False),
    ("meus avós não morreram", False),
    ("meu avô não morreu ontem, morreu hoje", True),
    ("my father didn't die", False),
    ("my father didn’t die", False),
    ("my father has not died", False),
    ("my father has not yet died", False),
    ("my father has not actually yet died", False),
    ("my father never died", False),
    ("my father hasn't died", False),
    ("my father hasn’t really died", False),
    ("she has not passed away", False),
    ("meu avô quase morreu", False),
    ("my father almost died", False),
    ("meu avô não sabe. Meu pai morreu", True),
    ("meu avô não sabe; meu pai morreu", True),
    ("meu avô não sabe\nmeu pai morreu", True),
    ("meu avô não sabe, mas meu pai morreu", True),
    ("meu avô não morreu e meu pai morreu", True),
    ("o jogo morreu no fim, não acredito", False),
    ("the battery died, I can't believe it", False),
])
def test_death_verb_negation_is_local(orch, path, text, expected):
    assert answer(orch, text.upper(), path) == "Model answer."
    assert orch.seen_contexts == ([NEWS] if expected else [""])


@pytest.mark.parametrize("path", PATHS)
def test_loss_note_composes_with_taught_and_project_notes(orch, path):
    orch._project_context = lambda text: "Project context."
    text = "aprenda com isso: minha mãe morreu hoje"
    assert answer(orch, text, path).startswith("Model answer.\n\nGuardei")
    assert orch.seen_contexts == ["Project context.\n\n" + REPLY_NOTE + "\n\n" + NEWS]
    answer(orch, "qual a capital da França?", path)
    assert orch.seen_contexts[-1] == "Project context."


@pytest.mark.parametrize("path", PATHS)
def test_realtime_path_keeps_existing_context(orch, path):
    orch._needs_realtime.return_value = True
    orch.gatekeeper.is_lease_active.return_value = True
    def web(engine, text, lang):
        assert not getattr(engine._turn_context, "value", None)
        return "Web answer.", "ONLINE_RAG"
    orch._do_web_search = web
    assert answer(orch, "minha mãe morreu hoje", path) == "Web answer."
    assert orch.seen_contexts == []
