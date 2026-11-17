"""
Tests for src/core/reflex.py — ReflexLayer zero-LLM pre-filter.

Checks:
- Every declared pattern fires on representative inputs (hits)
- Inputs that should NOT be reflexed return None (misses)
- Time/date callables return sensible strings
- is_hit() mirrors try_respond()
"""
import re
from unittest.mock import patch

import pytest

from src.core.reflex import ReflexLayer, reflex_layer


@pytest.fixture
def rl():
    return ReflexLayer()


class TestHits:
    """Inputs that the reflex layer MUST handle (no LLM needed)."""

    @pytest.mark.parametrize("text", [
        "oi", "Oi", "OI", "olá", "ola", "hey", "eai", "e aí",
        "bom dia", "boa tarde", "boa noite", "salve",
    ])
    def test_greeting_pt(self, rl, text):
        assert rl.try_respond(text) is not None

    @pytest.mark.parametrize("text", [
        "tchau", "xau", "até mais", "ate mais", "falou", "vlw", "valeu", "flw",
    ])
    def test_goodbye_pt(self, rl, text):
        assert rl.try_respond(text) is not None

    @pytest.mark.parametrize("text", [
        "obrigado", "obrigada", "obg", "brigado", "brigada",
        "thanks", "thank you",
    ])
    def test_thanks(self, rl, text):
        assert rl.try_respond(text) is not None

    @pytest.mark.parametrize("text", [
        "que horas são", "que hora é", "qual a hora", "qual hora",
        "me diz a hora", "me fala a hora", "hora certa",
        # Broadened forms (voice fillers + BR colloquialisms) that the
        # original strict pattern missed → fell through to the slow LLM.
        "que horas são agora", "Que horas são agora?", "que horas que são",
        "são que horas", "me diz as horas", "qual a hora agora",
        "que horas tem", "qual é a hora", "me informa as horas",
    ])
    def test_time_query_pt(self, rl, text):
        reply = rl.try_respond(text)
        assert reply is not None
        assert re.search(r"\d{2}:\d{2}", reply), "time reply must contain HH:MM"

    @pytest.mark.parametrize("text", [
        "que dia é hoje", "qual a data", "qual data de hoje",
        "que data é hoje", "data de hoje",
    ])
    def test_date_query_pt(self, rl, text):
        reply = rl.try_respond(text)
        assert reply is not None
        # Should mention a day number
        assert re.search(r"\d+", reply)

    @pytest.mark.parametrize("text", [
        "hi", "hello", "Hey", "good morning", "good afternoon", "good evening",
    ])
    def test_greeting_en(self, rl, text):
        assert rl.try_respond(text) is not None

    @pytest.mark.parametrize("text", [
        "what's the time", "what is the time", "what time is it",
    ])
    def test_time_query_en(self, rl, text):
        reply = rl.try_respond(text)
        assert reply is not None
        assert re.search(r"\d{2}:\d{2}", reply)
        # The reply MUST read as English — the listener picks the TTS voice
        # from the reply language. A PT reply ("São HH:MM.") to an English
        # question made the PT voice answer in English (bug, session 28).
        assert "São" not in reply and "It's" in reply

    @pytest.mark.parametrize("text", [
        "what's the date", "what is today's date", "what day is it",
        "what day is today",
    ])
    def test_date_query_en(self, rl, text):
        from datetime import datetime as dt
        with patch("src.core.reflex._now_local", return_value=dt(2026, 5, 22)):
            reply = rl.try_respond(text)
        # English reply → English TTS voice (see test_time_query_en).
        assert reply == "Today is Friday, 22 May 2026."

    @pytest.mark.parametrize("text", [
        "ping", "status", "online", "alive",
    ])
    def test_status_ping(self, rl, text):
        assert rl.try_respond(text) is not None


class TestMisses:
    """Inputs that must NOT be reflexed — they need LLM or tool routing."""

    @pytest.mark.parametrize("text", [
        # Greetings embedded mid-sentence
        "oi, como funciona o RAG?",
        "olá, me explica transformers",
        # Time embedded in a question about something else
        "qual a hora que o evento começa?",
        "que horas são os meus lembretes?",
        # Real queries that happen to start with reflex-ish words
        "obrigado pelo resumo, mas me explica melhor",
        "bom dia, quero estudar TensorFlow",
        # Studier / tool commands
        "estude attention mechanism",
        "que horas são no japão",  # needs internet
        # Broadened time pattern must still reject longer real questions
        "que horas devo acordar amanhã",
        "que horas o mercado abre",
        "quantas horas faltam pro almoço",
        "a que horas começa o jogo",
        # Empty
        "",
        # Just punctuation
        "!",
    ])
    def test_non_reflex_inputs(self, rl, text):
        assert rl.try_respond(text) is None

    def test_is_hit_mirrors_try_respond_true(self, rl):
        assert rl.is_hit("oi") is True

    def test_is_hit_mirrors_try_respond_false(self, rl):
        assert rl.is_hit("me explica transformers") is False


class TestTimeReplies:
    """Time/date responses are dynamic — mock datetime to pin the output."""

    def test_time_reply_format(self, rl):
        with patch("src.core.reflex._now_local") as mock_now:
            from datetime import datetime as dt
            mock_now.return_value = dt(2026, 5, 22, 9, 5)
            reply = rl.try_respond("que horas são")
        assert reply == "São 09:05."

    def test_date_reply_pt(self, rl):
        with patch("src.core.reflex._now_local") as mock_now:
            from datetime import datetime as dt
            mock_now.return_value = dt(2026, 5, 22, 9, 0)  # sexta-feira
            reply = rl.try_respond("que dia é hoje")
        assert "sexta-feira" in reply
        assert "22" in reply
        assert "maio" in reply
        assert "2026" in reply

    def test_combined_datetime_reply(self, rl):
        with patch("src.core.reflex._now_local") as mock_now:
            from datetime import datetime as dt
            mock_now.return_value = dt(2026, 5, 22, 14, 30)
            reply = rl.try_respond("hora e data")
        assert re.search(r"\d{2}:\d{2}", reply)
        assert "22" in reply

    def test_combined_reply_reads_the_clock_once(self, rl):
        """Across midnight, two readings said "São 23:59. Hoje é domingo, 25"
        — a time and a date from different days (s88, Codex lote 8)."""
        from datetime import datetime as dt
        with patch("src.core.reflex._now_local",
                   side_effect=[dt(2026, 10, 24, 23, 59, 59),
                                dt(2026, 10, 25, 0, 0, 0)]):
            reply = rl.try_respond("hora e data")
        assert reply == "São 23:59. Hoje é sábado, 24 de outubro de 2026."


class TestModuleSingleton:
    """The module-level singleton is the same class, import-safe."""

    def test_singleton_is_reflex_layer(self):
        assert isinstance(reflex_layer, ReflexLayer)

    def test_singleton_responds(self):
        assert reflex_layer.try_respond("oi") is not None
