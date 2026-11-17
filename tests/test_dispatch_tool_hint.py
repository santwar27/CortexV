"""Tests for the TOOL_HINT_RE coverage in src/core/dispatch/heuristics.py.

TOOL_HINT_RE is the "should I route this through chat_with_tools
instead of free chat?" signal. When it fires the orchestrator gives
the LLM the tool list; when it doesn't, the orchestrator calls
plain `engine.chat()` which can't perform actions.

Sessão 11 bloco H (fix D minimal) extended the pattern beyond the
original document/email/calendar/reminder set to include music,
playback verbs, system control words, and app-open verbs. The goal:
when the dispatch regexes miss a natural-language phrasing, the
LLM at least gets a chance to call the right tool — and if it
ignores the tools, the anti-hallucination detector downstream
swaps the lie for a truthful "couldn't do it" message.

Tests are split into:
  - LEGACY positive — pre-existing matches must still fire
  - NEW positive — phrasings the sessão 11 extension was meant to catch
  - NEGATIVE — generic chat that must NOT trigger tool routing

False positives in TOOL_HINT_RE just cost ~500ms latency on
knowledge queries that mention these words; false negatives cost
the user a hallucinated reply. We err on the side of more matches.
"""
import pytest

from src.core.dispatch.heuristics import TOOL_HINT_RE


# =====================================================================
# Legacy coverage — must still match
# =====================================================================

class TestLegacyToolHint:

    @pytest.mark.parametrize("text", [
        # Document
        "lê o documento agentes",
        "abre minhas notas de hoje",
        "show me the file",
        # Email
        "manda email pro João",
        "envia mensagem pra Maria",
        "responde aquele email",
        "list my inbox",
        # Calendar
        "agenda uma reunião pra sexta",
        "marca o evento na agenda",
        "schedule a meeting tomorrow",
        # Reminder
        "me lembre de comprar leite",
        "criar lembrete pras 18h",
        "remind me to call John",
    ])
    def test_legacy_intents_still_match(self, text):
        assert TOOL_HINT_RE.search(text), text


# =====================================================================
# New coverage (sessão 11 bloco H)
# =====================================================================

class TestExtendedToolHintMusic:

    @pytest.mark.parametrize("text", [
        # Spotify / music keyword
        "será que dá pra tocar matue no spotify?",
        "quero ouvir uma playlist de pagode",
        "que música é essa que tá tocando?",
        "manda uma música boa aí",
        "põe alguma music pra tocar",
        # Playback verbs without exact dispatch match
        "vamos ouvir queen",
        "põe a próxima",
        "listen to drake",
        "play something chill",
        "put on coldplay",
        "escutar matuê",
    ])
    def test_music_intents_match(self, text):
        assert TOOL_HINT_RE.search(text), f"music intent missed: {text}"


class TestExtendedToolHintSystem:

    @pytest.mark.parametrize("text", [
        # Volume / brightness
        "tu pode aumentar o brilho?",
        "podia abaixar o brilho um pouco",
        "brightness up please",
        # Dark / light mode
        "ativa modo escuro",
        "muda pro tema escuro",
        "dark mode please",
        "modo claro agora",
        # Lock
        "tu pode travar a tela?",
        "podia bloquear o mac?",
        "lock the screen",
        # Screenshot
        "tira um print da tela",
        "captura a tela inteira",
        "take a screenshot",
        # App open
        "abre o Notion pra mim",
        "podia abrir o Spotify?",
        "launch VS Code",
        "open Calendar",
    ])
    def test_system_intents_match(self, text):
        assert TOOL_HINT_RE.search(text), f"system intent missed: {text}"


# =====================================================================
# Negative — generic chat must NOT trigger
# =====================================================================

class TestToolHintNegative:
    """Knowledge queries and small talk that don't need any tool.
    False positives here cost latency, not correctness — but the
    less noise we add, the better."""

    @pytest.mark.parametrize("text", [
        # Pure small talk
        "oi",
        "tudo bem?",
        "como vai?",
        "obrigado",
        # Time / date trivia (handled by reflex, not tools)
        "que horas são",
        "que dia é hoje",
        # Generic knowledge that doesn't reference our keywords
        "explica RAG",
        "qual a diferença entre TCP e UDP",
        "como funciona o GIL do python",
        "history of brazil",
        "quem inventou a relatividade",
    ])
    def test_chat_only_does_not_match(self, text):
        assert not TOOL_HINT_RE.search(text), f"false positive: {text}"


# =====================================================================
# REAL_TOOL_HINT_RE — subset that maps to a tool that EXISTS in
# TOOL_SPECS (s38). Music/macOS hints must NOT fire here: they have no
# tool, so the orchestrator offers an empty tool list for them, which
# stops the 8B fabricating an unrelated tool (the user's create_reminder /
# read_document misfire from "amo música brasileira").
# =====================================================================

class TestRealToolHint:
    from src.core.dispatch.heuristics import REAL_TOOL_HINT_RE as _RE

    @pytest.mark.parametrize("text", [
        "preciso lembrar de comprar pão",
        "criar um lembrete pra amanhã",
        "manda um email pro joão",
        "o que tem na minha caixa de entrada",
        "marca uma reunião sexta às 15h",
        "quais meus eventos da semana",
        "lê o documento agentes.txt",
        "salva isso nas minhas notas",
    ])
    def test_real_tool_intents_match(self, text):
        assert self._RE.search(text), f"real-tool intent missed: {text}"

    @pytest.mark.parametrize("text", [
        # The user's exact misfire inputs — only "música" matched TOOL_HINT_RE,
        # which has NO real tool → must NOT match REAL_TOOL_HINT_RE.
        "eu amo a musica brasileira, MPB, funk, trap, bossa nova",
        "que música é essa",          # handled by spotify regex upstream, not tools
        "amo música brasileira mais que internacional",
        # macOS-only words (no macos tool in TOOL_SPECS)
        "ativa o modo escuro",
        "tira um screenshot",
        # pure small talk
        "oi tudo bem",
    ])
    def test_music_and_macos_do_not_match(self, text):
        assert not self._RE.search(text), f"false real-tool positive: {text}"

    @pytest.mark.parametrize("text", [
        "manda um email pro joão",
        "marca uma reunião sexta",
        "cria um lembrete pra amanhã",
        "lê o documento agentes.txt",
        "salva nas minhas notas",
        "quais meus eventos da semana",
        "encaminhar a mensagem",
        "agendar uma consulta",
    ])
    def test_real_is_subset_of_tool_hint(self, text):
        """Drift guard: every phrase REAL_TOOL_HINT_RE matches MUST also match
        TOOL_HINT_RE. The two regexes duplicate the doc/email/calendar/reminder
        tokens; if someone edits one and not the other they silently desync and
        a real tool stops being offered. This catches that."""
        assert self._RE.search(text), f"REAL should match: {text}"
        assert TOOL_HINT_RE.search(text), f"TOOL drifted from REAL: {text}"
