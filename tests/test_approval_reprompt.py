"""
Garbled reply to a pending internet-approval prompt (s76 smoke).

Live bug: "How's the weather like in London?" → AWAITING_APPROVAL →
Pedro said "yes" → Whisper transcribed it as "Ah!" → the anything-else
branch silently dropped the pending query and routed "Ah!" to the LLM,
which hallucinated an answer from PAST CONTEXT. APPROVAL_RE was never
the problem — it covers "yes" — the input it received was noise.

The fix: a SHORT non-matching reply (≤ 12 chars, ≤ 2 words) re-prompts
ONCE ("Não entendi — sim ou não?") keeping the pending query, under
mode AWAITING_APPROVAL so the voice listener reopens its follow-up
window. Fails closed: nothing is approved without an APPROVAL_RE match.
A second miss, or any longer reply, flows exactly as before (pending
dropped, input routed as a new command).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import src.core.orchestrator as orch_mod
from src.core.gatekeeper import LogicGate
from src.core.orchestrator import CortexOrchestrator


def _orch():
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o.engine_analytical = MagicMock()
    o.engine_analytical.chat.return_value = "resposta sintetizada"
    o.gatekeeper = LogicGate()  # state file isolated by conftest
    o.lease_hours = 4
    o.web_searcher = MagicMock()
    o.web_searcher.search.return_value = "Source 1 (X):\n15 graus\n[url]"
    o._pending_action = None
    o._pending_query = "how's the weather in london"
    o._pending_engine = o.engine_analytical
    o._pending_query_reprompted = False
    o._shadow_classify = lambda text: None
    return o


def _stub_registry(monkeypatch):
    """Intercept routing after the pending branch: the drop path falls
    through to the dispatch registry; tests only need to see THAT it got
    there, not exercise the whole table."""
    routed = {}

    def run(self, text):
        routed["text"] = text
        return ("routed", "ROUTED_STUB")

    monkeypatch.setattr(orch_mod, "dispatch_registry", SimpleNamespace(run=run))
    return routed


class TestGarbledReplyReprompts:
    def test_short_garbled_reply_reprompts_and_keeps_pending(self):
        o = _orch()
        reply, mode = o._process_core("Ah!", language="en")
        assert mode == "AWAITING_APPROVAL"
        assert "yes or no" in reply
        assert o._pending_query is not None, "pending must survive the re-prompt"
        assert not o.gatekeeper.is_lease_active(), "fails closed: no lease"

    def test_reprompt_is_localised_pt(self):
        o = _orch()
        reply, mode = o._process_core("O quê?", language="pt")
        assert mode == "AWAITING_APPROVAL"
        assert "sim ou não" in reply

    def test_yes_after_reprompt_still_approves(self):
        o = _orch()
        o._process_core("Ah!", language="en")
        reply, mode = o._process_core("yes", language="en")
        assert mode == "AUTO_SEARCH"
        assert o.gatekeeper.is_lease_active()

    def test_no_after_reprompt_still_denies(self):
        o = _orch()
        o._process_core("Ah!", language="en")
        reply, mode = o._process_core("no", language="en")
        assert mode == "APPROVAL_DENIED"
        assert o._pending_query is None
        assert not o.gatekeeper.is_lease_active()

    def test_second_garbled_reply_drops_pending(self, monkeypatch):
        routed = _stub_registry(monkeypatch)
        o = _orch()
        o._process_core("Ah!", language="en")
        reply, mode = o._process_core("Hmm.", language="en")
        assert mode == "ROUTED_STUB", "second miss routes as a new command"
        assert o._pending_query is None
        assert routed, "the input itself must reach the dispatch registry"

    def test_long_reply_is_a_new_command_no_reprompt(self, monkeypatch):
        _stub_registry(monkeypatch)
        o = _orch()
        reply, mode = o._process_core("toca matuê no spotify")
        assert mode == "ROUTED_STUB", "a real sentence never re-prompts"
        assert o._pending_query is None

    def test_restaging_resets_the_one_shot(self):
        # After a re-prompt was consumed, a NEW pending query staged later
        # must get its own re-prompt budget.
        o = _orch()
        o._process_core("Ah!", language="en")
        assert o._pending_query_reprompted is True
        # simulate the orchestrator staging a fresh pending query
        o._pending_query = None
        o._pending_query_reprompted = False  # what the staging sites do
        o._pending_query = "weather in tokyo"
        reply, mode = o._process_core("Eh?", language="en")
        assert mode == "AWAITING_APPROVAL"
