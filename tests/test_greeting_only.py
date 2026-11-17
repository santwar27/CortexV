"""A greeting that IS the whole message: "ola cortex" (s87).

Pedro typed "ola cortex", "Olá Cortex" and "oi cortex" in the terminal and
got "I'm not sure what you mean by 'cortex'", a self-description, and "I'm
thinking of a colour... Blue." The audit showed why: strip_vocatives turned
each into the bare word "cortex" — its greeting pattern needs a separator
AFTER the name (it was built for "olá cortex, abre o spotify"), so it
backtracked to "ola " and left the name. The reflex answers "oi", not
"cortex", so the 8B got one word. All 3 such messages in the audit, 3 of 3.
"""
import pytest

from src.core.dispatch.normalisation import strip_vocatives
from src.core.reflex import reflex_layer

# The user's greetings (16/10), with the name, as typed.
GREETINGS = [
    ("ola cortex", "ola"),
    ("Olá Cortex", "Olá"),
    ("oi cortex", "oi"),
    ("oi cortex!", "oi"),
    ("Oi, Cortex.", "Oi"),
    ("bom dia cortex", "bom dia"),
    ("boa tarde cortex", "boa tarde"),
    ("boa noite, cortex", "boa noite"),
    ("e aí cortex", "e aí"),
    ("e ai cortex", "e ai"),
    ("fala cortex", "fala"),
    ("salve cortex!", "salve"),
    ("ei cortex", "ei"),
    ("hey cortex", "hey"),
    ("hi cortex", "hi"),
    ("hello jarvis?", "hello"),
]


class TestTheGreetingIsKept:

    @pytest.mark.parametrize("text, greeting", GREETINGS)
    def test_the_name_goes_the_greeting_stays(self, text, greeting):
        assert strip_vocatives(text) == greeting

    @pytest.mark.parametrize("text, _g", GREETINGS)
    def test_the_reflex_answers_it(self, text, _g):
        assert reflex_layer.try_respond(strip_vocatives(text)) is not None

    @pytest.mark.parametrize("text, expected", [
        ("olá cortex, abre o spotify", "abre o spotify"),
        ("oi cortex! pause", "pause"),
        ("hi cortex pause", "pause"),
        ("fala cortex, que horas são", "que horas são"),
        ("salve cortex, toca matue", "toca matue"),
        ("e aí cortex, meus emails", "meus emails"),
        ("cortex?", "cortex?"),                     # the bare name: unchanged
        ("fala mais alto", "fala mais alto"),       # a command, not a greeting
        ("salve isso nas notas", "salve isso nas notas"),
    ])
    def test_a_command_after_it_survives(self, text, expected):
        assert strip_vocatives(text) == expected


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    return o


class TestBothEntryPoints:

    @pytest.mark.parametrize("text", ["ola cortex", "Olá Cortex", "oi cortex", "fala cortex"])
    def test_process(self, orch, text):
        reply, mode = orch._process_core(text)
        assert mode == "REFLEX" and reply.startswith("Oi!")
        orch.engine_analytical.chat.assert_not_called()

    @pytest.mark.parametrize("text", ["ola cortex", "Olá Cortex", "oi cortex", "hi cortex"])
    def test_stream(self, orch, text):
        parts, mode = [], None
        for chunk, final, m, _lang in orch.process_stream(text):
            parts.append(chunk)
            if final:
                mode = m
        assert mode == "REFLEX"
        orch.engine_analytical.chat_stream.assert_not_called()
