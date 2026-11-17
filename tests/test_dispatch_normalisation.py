"""Tests for src/core/dispatch/normalisation.py — vocative stripping
applied at the orchestrator entry before regex dispatch.

Coverage:
  - greeting + name combos (PT + EN)
  - greeting alone, name alone
  - punctuation variants between greeting and command
  - inputs that should NOT be stripped (no vocative at front)
  - idempotency
  - integration: stripped output flowing into SPOTIFY_PLAY_RE matches
"""
import pytest

from src.core.dispatch.normalisation import strip_vocatives
from src.core.dispatch.spotify import SPOTIFY_PLAY_RE


class TestStripVocativesIdentity:

    def test_empty_returns_empty(self):
        assert strip_vocatives("") == ""

    def test_none_returns_none(self):
        # Documented contract: pass-through on falsy.
        assert strip_vocatives(None) is None  # type: ignore[arg-type]

    def test_command_without_vocative_unchanged(self):
        assert strip_vocatives("toca matue") == "toca matue"

    def test_no_vocative_only_strips_whitespace(self):
        assert strip_vocatives("   pause   ") == "pause"

    def test_word_starting_with_oi_not_stripped(self):
        # "ouve" doesn't start with a greeting; must remain whole.
        assert strip_vocatives("ouve essa música") == "ouve essa música"

    def test_word_starting_with_hi_not_stripped(self):
        # "history" should NOT lose "hi" — vocative regex requires
        # punctuation/whitespace after the greeting token.
        assert strip_vocatives("history of math") == "history of math"


class TestStripVocativesGreetingPlusName:

    @pytest.mark.parametrize("text, expected", [
        ("olá cortex, toca matue", "toca matue"),
        ("oi cortex! pause", "pause"),
        ("ei cortex, próxima", "próxima"),
        ("hey jarvis, play matue", "play matue"),
        ("hi cortex pause", "pause"),
        ("hello cortex - what is RAG?", "what is RAG?"),
        ("bom dia cortex, meus emails", "meus emails"),
        ("boa noite cortex, estude rede neural", "estude rede neural"),
        ("good morning cortex, list emails", "list emails"),
        ("good evening jarvis: pause", "pause"),
    ])
    def test_strips_greeting_plus_name(self, text, expected):
        assert strip_vocatives(text) == expected


class TestStripVocativesGreetingOnly:

    @pytest.mark.parametrize("text, expected", [
        ("olá, pause", "pause"),
        ("oi! toca matue", "toca matue"),
        ("hey play matue", "play matue"),
        ("bom dia, meus emails", "meus emails"),
    ])
    def test_strips_bare_greeting(self, text, expected):
        assert strip_vocatives(text) == expected


class TestStripVocativesBareName:

    @pytest.mark.parametrize("text, expected", [
        ("cortex, toca matue", "toca matue"),
        ("cortex: pause", "pause"),
        ("cortex - next", "next"),
        ("jarvis, play matue", "play matue"),
        ("córtex, pause", "pause"),  # accented variant
        ("CORTEX pause", "pause"),    # case insensitive
    ])
    def test_strips_bare_name(self, text, expected):
        assert strip_vocatives(text) == expected


class TestStripVocativesNonLeadingNoMatch:

    @pytest.mark.parametrize("text", [
        "say hi to cortex",       # name in middle, not vocative
        "search hello world",     # "hello" but not at start
        "fala bom dia pro João",  # "bom dia" mid-sentence
    ])
    def test_non_leading_vocative_unchanged(self, text):
        assert strip_vocatives(text) == text


class TestStripVocativesIdempotency:

    @pytest.mark.parametrize("text", [
        "olá cortex, toca matue",
        "toca matue",
        "pause",
        "",
        "cortex, pause",
    ])
    def test_double_strip_equals_single(self, text):
        once = strip_vocatives(text)
        twice = strip_vocatives(once)
        assert once == twice


class TestStripVocativesIntegrationWithSpotify:
    """The original bug: 'olá cortex, pode tocar matue no spotify?'
    falling through to the LLM. With strip_vocatives at the
    orchestrator entry AND polite verbs in SPOTIFY_PLAY_RE, this now
    routes deterministically."""

    @pytest.mark.parametrize("raw, expected_query", [
        ("olá cortex, toca matue no spotify",       "matue"),
        ("olá cortex, toca matue no spotify?",      "matue"),
        ("oi cortex, pode tocar matue",             "matue"),
        ("cortex, pode tocar matue no spotify?",    "matue"),
        ("ei cortex, quero ouvir matue",            "matue"),
        ("hey jarvis, could you play matue",        "matue"),
        ("bom dia cortex, pode colocar bohemian rhapsody no spotify?", "bohemian rhapsody"),
    ])
    def test_full_dispatch_path(self, raw, expected_query):
        routing = strip_vocatives(raw)
        m = SPOTIFY_PLAY_RE.match(routing)
        assert m is not None, f"Failed to match: {routing!r}"
        assert m.group(1).strip() == expected_query


class TestStripVocativesLoneAckSurvives:
    """M1-A4: a message that IS entirely an ack/vocative must come back
    whole, not empty — "ok." is a legitimate reply to a confirmation
    prompt (APPROVAL_RE itself allows "ok" + [.!]), and an empty result
    made both entry points answer "[SYSTEM ERROR] Empty input." with the
    action still staged."""

    @pytest.mark.parametrize("text", [
        "ok.", "ok!", "okay,", "beleza!", "blz.", "valeu!", "vlw!",
        "perfeito.", "show!", "got it.",
    ])
    def test_lone_ack_returned_whole(self, text):
        assert strip_vocatives(text) == text.strip()

    def test_lone_vocative_returned_whole(self):
        # Bare wake-word ("cortex?") is a message too — better a chat
        # greeting than a SYSTEM ERROR.
        assert strip_vocatives("cortex?") == "cortex?"

    def test_ack_plus_command_still_stripped(self):
        # The fix must not regress the original purpose: ack PREFIXES
        # still strip so dispatch regexes stay anchored.
        assert strip_vocatives("ok. toca matue") == "toca matue"

    def test_truly_empty_still_empty(self):
        assert strip_vocatives("") == ""
        assert strip_vocatives("   ") == ""


class TestLoneAckApprovesStagedAction:
    """The full M1-A4 seam, orchestrator-level: with an action staged,
    'ok.' must reach _execute_pending_action — not die as INPUT_ERROR."""

    def _orch(self):
        from unittest.mock import MagicMock

        from src.core.orchestrator import CortexOrchestrator
        from datetime import datetime, timezone

        o = CortexOrchestrator.__new__(CortexOrchestrator)
        # staged_at: a FRESH action (M1-A9 TTL) — without it the slot is
        # fail-safe discarded as unknown-age before the ack is matched.
        o._pending_action = {
            "type": "send_email", "args": {},
            "staged_at": datetime.now(timezone.utc).isoformat(),
        }
        o._pending_query = None
        o._reply_lang = "pt"
        o._shadow_classify = lambda text: None
        o._execute_pending_action = MagicMock(
            return_value=("Enviado.", "AGENT_EXECUTED"))
        o._cancel_pending_action = MagicMock(
            return_value=("Cancelado.", "AGENT_CANCELLED"))
        return o

    @pytest.mark.parametrize("ack", [
        "ok.", "ok!", "beleza!", "okay,",
        # M1-A4 second half: punctuated acks the APPROVAL_RE now covers —
        # without them, the anything-else branch DROPPED the staged action.
        "valeu!", "show!", "perfeito.", "blz", "vlw!", "got it.",
    ])
    def test_lone_ack_executes_pending(self, ack):
        o = self._orch()
        reply, mode = o._process_core(ack)
        assert mode == "AGENT_EXECUTED", f"{ack!r} died before approval: {mode}"
        o._execute_pending_action.assert_called_once()

    def test_lone_nao_still_cancels(self):
        o = self._orch()
        reply, mode = o._process_core("não.")
        assert mode == "AGENT_CANCELLED"


class TestPorIsAPrepositionFirst:
    """"Por que azul?" was answered by putting on a pagode album (s83): bare
    "por" was accepted as the verb "pôr", so every "por que …?" question
    played music. The verb has to stay — the courtesy stripper turns the
    legitimate "pode por matuê" into "por matuê" — so the fixed phrases the
    PREPOSITION lives in are what gets excluded."""

    @pytest.mark.parametrize("text", [
        "por que azul?",
        "por que você escolheu azul?",
        "por que isso?",
        "por que não?",
        "por quê?",
        "por favor me explica",
        "por acaso tu sabe?",
        "por enquanto não",
        "por isso que eu falei",
        "por exemplo, o que é RAG?",
        "por volta das 3",
    ])
    def test_a_question_is_not_a_song(self, text):
        assert not SPOTIFY_PLAY_RE.match(text), text

    @pytest.mark.parametrize("text, expected", [
        ("por Djavan", "Djavan"),
        ("por matuê no spotify", "matuê"),
        ("põe Matuê", "Matuê"),
        ("pôr Michael Jackson", "Michael Jackson"),
        ("poe matue", "matue"),
    ])
    def test_the_verb_still_plays(self, text, expected):
        m = SPOTIFY_PLAY_RE.match(text)
        assert m and m.group(1).strip() == expected, text

    def test_the_polite_form_survives_the_courtesy_stripper(self):
        """"pode por matuê" reaches the regex as "por matuê" — the case that
        stops this from being fixed by simply demanding the accent."""
        from src.core.dispatch.normalisation import strip_courtesy
        stripped = strip_courtesy("pode por matuê")
        m = SPOTIFY_PLAY_RE.match(stripped)
        assert m and m.group(1).strip() == "matuê"


class TestAGraphicLeadIsNotPartOfTheCommand:
    """"🙂 toca música" and "— fecha o Safari" missed their verb-first routes
    and went to the model (s88, Codex lote 9)."""

    @pytest.mark.parametrize("text,plain", [
        ("🙂 toca música", "toca música"),
        ("— fecha o Safari", "fecha o Safari"),
        ("- abre o spotify", "abre o spotify"),
        ("👍🏽 ok", "ok"),
    ])
    def test_it_is_stripped(self, text, plain):
        from src.core.dispatch.normalisation import strip_vocatives
        assert strip_vocatives(text) == plain

    @pytest.mark.parametrize("text", ["-5 graus", "🙂", "> apaga as refeições de hoje",
                                      "toca música"])
    def test_nothing_else_is(self, text):
        from src.core.dispatch.normalisation import strip_vocatives
        assert strip_vocatives(text) == text

    @pytest.mark.parametrize("text,plain", [("🙂 toca música", "toca música"),
                                            ("— fecha o Safari", "fecha o Safari")])
    def test_it_routes_like_the_plain_command(self, tmp_path, monkeypatch, text, plain):
        monkeypatch.setenv("HOME", str(tmp_path))
        from tests.test_dispatch_parity import _build_real_orch
        o = _build_real_orch(tmp_path)
        o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
        o.background = False
        assert o._process_core(text)[1] == o._process_core(plain)[1] != "OFFLINE"
