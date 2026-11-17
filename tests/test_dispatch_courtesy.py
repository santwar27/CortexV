"""Tests for strip_courtesy (s75, Onda 8 A — smoke finding 1, s74).

The dispatch regexes are verb-first; a courtesy wrapper defeated them:
"Can you open Spotify please?" fell to OFFLINE chat while "Open the
Spotify" routed to MAC_OPEN_APP. strip_courtesy peels the envelope for
the ROUTING side only — the orchestrator keeps the original text for
the LLM fallback ("can you hear me?" must stay intact).

Coverage:
  - EN/PT prefixes, trailing politeness, both combined, stacked prefixes
  - approval/denial guard ("pode crer" is a YES, not a courtesy prefix)
  - empty-result guard (whole message IS the courtesy)
  - idempotency, non-leading tokens untouched, song-title false positive
  - integration with MAC_OPEN_APP_RE (the exact s74 smoke phrase)
  - the streaming pure-chat veto sees through the wrapper
"""
import pytest

from src.core.dispatch.macos import MAC_OPEN_APP_RE
from src.core.dispatch.normalisation import strip_courtesy


class TestStripCourtesyPrefixes:

    @pytest.mark.parametrize("text, expected", [
        ("can you open spotify", "open spotify"),
        ("Could you open Spotify", "open Spotify"),
        ("would you play matue", "play matue"),
        ("can you please open spotify", "open spotify"),
        ("please open spotify", "open spotify"),
        ("você pode abrir o spotify", "abrir o spotify"),
        ("vc pode abrir o spotify", "abrir o spotify"),
        ("pode abrir o spotify", "abrir o spotify"),
        ("poderia abrir o spotify", "abrir o spotify"),
        ("consegue abrir o spotify", "abrir o spotify"),
        ("dá pra abrir o spotify", "abrir o spotify"),
        ("por favor, abre o spotify", "abre o spotify"),
    ])
    def test_strips_leading_courtesy(self, text, expected):
        assert strip_courtesy(text) == expected

    @pytest.mark.parametrize("text, expected", [
        # Stacked: up to two prefix layers.
        ("please can you open spotify", "open spotify"),
        ("por favor, você pode abrir o spotify", "abrir o spotify"),
    ])
    def test_strips_stacked_prefixes(self, text, expected):
        assert strip_courtesy(text) == expected


class TestStripCourtesySuffixes:

    @pytest.mark.parametrize("text, expected", [
        ("open spotify please", "open spotify"),
        ("open spotify, please?", "open spotify"),
        ("abre o spotify por favor", "abre o spotify"),
        ("abre o spotify por gentileza", "abre o spotify"),
        ("abre o spotify pfv", "abre o spotify"),
        ("abre o spotify pra mim", "abre o spotify"),
        ("abre o spotify para mim", "abre o spotify"),
    ])
    def test_strips_trailing_courtesy(self, text, expected):
        assert strip_courtesy(text) == expected

    def test_both_ends(self):
        assert strip_courtesy("Can you open Spotify please?") == "open Spotify"


class TestStripCourtesyGuards:

    @pytest.mark.parametrize("text", [
        # APPROVAL_RE owns these — chopping "pode " would turn a staged-
        # action YES into "crer"/"mandar" headed for chat.
        "pode", "pode.", "pode crer", "pode mandar",
        "sim", "ok.", "claro",
        # Denials equally untouched.
        "não", "não.", "nope",
    ])
    def test_approval_denial_returned_whole(self, text):
        assert strip_courtesy(text) == text

    @pytest.mark.parametrize("text", [
        # The whole message IS courtesy → return original, never empty
        # (same contract as strip_vocatives' lone-ack guard).
        "can you", "você pode", "pode ser",  # "pode ser" is approval-ish chat
    ])
    def test_never_returns_empty(self, text):
        assert strip_courtesy(text).strip() != "" or text == ""

    def test_falsy_passthrough(self):
        assert strip_courtesy("") == ""
        assert strip_courtesy(None) is None  # type: ignore[arg-type]


class TestStripCourtesyNoFalsePositives:

    @pytest.mark.parametrize("text", [
        "play Please Please Me",        # title ends in "me", not courtesy
        "show me my last 3 emails",     # no courtesy anywhere
        "toca matue no spotify",
        "what can you do",              # "can you" not leading
        "ele pode abrir o spotify",     # "pode" not leading
    ])
    def test_unchanged(self, text):
        assert strip_courtesy(text) == text


class TestStripCourtesyIdempotency:

    @pytest.mark.parametrize("text", [
        "Can you open Spotify please?",
        "pode abrir o spotify por favor",
        "open spotify",
        "pode crer",
        "",
    ])
    def test_double_strip_equals_single(self, text):
        once = strip_courtesy(text)
        assert strip_courtesy(once) == once


class TestCourtesyIntegrationOpenApp:
    """The s74 smoke finding, end to end at the regex seam: the polite
    phrase must hit MAC_OPEN_APP_RE with a clean app capture."""

    @pytest.mark.parametrize("raw, app", [
        ("Can you open Spotify please?", "Spotify"),
        ("could you launch Notion", "Notion"),
        ("você pode abrir o Spotify por favor?", "Spotify"),
        ("pode abrir o Spotify?", "Spotify"),
        ("abre o Spotify pra mim", "Spotify"),
    ])
    def test_courtesy_phrase_routes_to_open_app(self, raw, app):
        routing = strip_courtesy(raw)
        m = MAC_OPEN_APP_RE.match(routing)
        assert m is not None, f"Failed to match: {routing!r}"
        assert m.group(1).strip() == app


class TestDictatedTextKeepsItsLastWords:
    """"corpo: Compre pão para mim" staged the body "Compre pão": the trailing
    courtesy strip ran over the dictated field (s88, Codex lote 9)."""

    @pytest.mark.parametrize("text", [
        "envia email para joao@example.com; assunto: compra; corpo: Compre pão para mim",
        "coloca na nota Mercado: arroz pra mim",
        "assunto: um favor por favor",
    ])
    def test_a_field_after_a_colon_is_left_whole(self, text):
        assert strip_courtesy(text) == text

    def test_a_command_still_loses_it(self):
        assert strip_courtesy("abre o spotify pra mim") == "abre o spotify"

    def test_the_staged_body_is_whole(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from tests.test_dispatch_parity import _build_real_orch
        o = _build_real_orch(tmp_path)
        o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
        o.background = False
        monkeypatch.setattr("src.integrations.contacts.contact_book.list_all", lambda: [])
        o._process_core("envia email para joao@example.com; assunto: compra; "
                        "corpo: Compre pão para mim")
        assert o._pending_action["args"]["body"] == "Compre pão para mim"


class TestSttAliases:
    """normalise_stt_aliases (s75 cont.) — Whisper small + BR accent
    renders Pedro's "spotify" as "spot 5" (observed in the audit:
    'Can you open spot 5 please?'). Routing-only rewrite, same contract
    as strip_courtesy."""

    @pytest.mark.parametrize("text, expected", [
        ("open spot 5", "open spotify"),
        ("open Spot 5", "open spotify"),
        ("open spot five", "open spotify"),
        ("open spot-5", "open spotify"),
        ("toca matue no spot 5", "toca matue no spotify"),
    ])
    def test_rewrites_mishearing(self, text, expected):
        from src.core.dispatch.normalisation import normalise_stt_aliases
        assert normalise_stt_aliases(text) == expected

    @pytest.mark.parametrize("text", [
        "spotted 5 birds",      # word boundary: 'spotted' stays whole
        "spot 50 on the map",   # '5' must be the whole token
        "sunspot 5",            # no boundary before 'spot'
        "open spotify",         # already correct — idempotent
    ])
    def test_no_false_positives(self, text):
        from src.core.dispatch.normalisation import normalise_stt_aliases
        assert normalise_stt_aliases(text) == text

    @pytest.mark.parametrize("text", [
        "salva o email do João como spot-5@example.com",
        "envia email para spot5@x.com; assunto: oi; corpo: teste",
        "abre https://spot-5.example.com/x",
        "lê /Users/pedro/spot 5/notas.txt",
    ])
    def test_an_address_url_or_path_is_literal(self, text):
        """"spot-5@example.com" was saved, and staged as the recipient, as
        spotify@example.com (s88, Codex lote 9)."""
        from src.core.dispatch.normalisation import normalise_stt_aliases
        assert normalise_stt_aliases(text) == text

    def test_the_words_around_an_address_still_rewrite(self):
        from src.core.dispatch.normalisation import normalise_stt_aliases
        assert (normalise_stt_aliases("manda spot-5@x.com o link do spot 5")
                == "manda spot-5@x.com o link do spotify")

    def test_full_chain_audit_phrase(self):
        # The exact transcript from the audit log (2026-06-12 16:05):
        # courtesy strip + alias rewrite must land on MAC_OPEN_APP.
        from src.core.dispatch.normalisation import normalise_stt_aliases
        routing = normalise_stt_aliases(
            strip_courtesy("Can you open spot 5 please?"))
        m = MAC_OPEN_APP_RE.match(routing)
        assert m is not None, f"Failed to match: {routing!r}"
        assert m.group(1).strip() == "spotify"


class TestCourtesyPureChatVeto:
    """Two-entry-points lesson: process_stream's pure-chat probe runs on
    the courtesy-stripped text, so the wrapped command can't stream chat
    past the handler. Pinned at the seam the probe uses (the blocking
    regex list) rather than a full orchestrator instance."""

    def test_wrapped_open_app_blocks_pure_chat(self):
        from src.core.orchestrator import CortexOrchestrator

        routing = strip_courtesy("Can you open Spotify please?")
        rx_names = CortexOrchestrator._PURE_CHAT_BLOCKING_RES
        assert "_MAC_OPEN_APP_RE" in rx_names
        assert MAC_OPEN_APP_RE.match(routing) is not None
