"""
Guard: dispatch registry ↔ _is_pure_chat consistency (Onda 7 A1, s74).

The orchestrator has TWO entry points (the s24 lesson): process() walks
the dispatch registry, but process_stream() first probes _is_pure_chat —
and if that says "pure chat", the LLM reply is streamed WITHOUT ever
reaching the registry. So every regex the registry can fire must also
veto the pure-chat shortcut, or the spoken path streams chat right past
the handler. This nearly shipped twice: s24 (time reflex) and s73 (the
Notes v1.1 routes were in the registry but not in the veto list).

This test makes the invariant automatic: a Route whose regex is bound as
a class attribute on CortexOrchestrator MUST have (at least one of) its
attribute names in _PURE_CHAT_BLOCKING_RES. Legitimate exceptions go in
ROUTE_EXCEPTIONS with a written reason — documented, not forced (the
test_dispatch_parity style).

Found on first run (and fixed in the same commit): SILENCE_RE was a real
gap — "silêncio"/"cala" contain no TOOL_HINT_RE word, so the spoken
command would have streamed a chat reply over the shush. REMINDER and
CALENDAR triggers were only saved by TOOL_HINT_RE matching words like
"lembre"/"calendário" — incidental coverage, now explicit.
"""

import re

import pytest

from src.core.dispatch.registry import DISPATCH_ROUTES, POST_PLANNER_ROUTES
from src.core.orchestrator import CortexOrchestrator

# Route.name -> reason it is allowed to bypass the pure-chat veto.
# Empty today. An entry here must explain WHY streaming chat over the
# route is acceptable (e.g. the route is dormant and gated off).
ROUTE_EXCEPTIONS: dict = {}


def _attr_names_by_pattern_id():
    """Map id(compiled pattern) -> [class attribute names] on the
    orchestrator. Identity (not name) is the join key: the registry
    exports NOTE_CREATE_RE, the orchestrator binds _NOTE_CREATE_RE —
    same object, different names."""
    out = {}
    for name, val in vars(CortexOrchestrator).items():
        if isinstance(val, re.Pattern):
            out.setdefault(id(val), []).append(name)
    return out


class TestRegistryPureChatConsistency:
    def test_every_route_regex_vetoes_pure_chat(self):
        attr_names = _attr_names_by_pattern_id()
        blocking = set(CortexOrchestrator._PURE_CHAT_BLOCKING_RES)
        missing = []
        for route in DISPATCH_ROUTES + POST_PLANNER_ROUTES:
            if route.name in ROUTE_EXCEPTIONS:
                continue
            names = attr_names.get(id(route.regex), [])
            assert names, (
                f"Route {route.name!r}: regex is not bound as a class "
                "attribute on CortexOrchestrator at all — _is_pure_chat "
                "cannot see it, so the voice path would stream an LLM "
                "reply over this handler. Bind it and add it to "
                "_PURE_CHAT_BLOCKING_RES (or document an exception here)."
            )
            if not any(n in blocking for n in names):
                missing.append(f"{route.name} (bound as {names})")
        assert not missing, (
            "Routes in the dispatch registry whose regex does NOT veto "
            "the streaming pure-chat shortcut — spoken input matching "
            "them would stream an LLM chat reply instead of dispatching "
            "(the s24/s73 two-entry-points bug):\n  " + "\n  ".join(missing)
        )

    def test_blocking_list_names_exist(self):
        # A typo'd name in the veto list only explodes when getattr is
        # reached at runtime, on the voice path. Catch it here instead.
        for name in CortexOrchestrator._PURE_CHAT_BLOCKING_RES:
            val = getattr(CortexOrchestrator, name, None)
            assert isinstance(val, re.Pattern), (
                f"_PURE_CHAT_BLOCKING_RES entry {name!r} is not a compiled "
                "regex class attribute on CortexOrchestrator"
            )


class TestSilenceGapFixed:
    """The concrete bug the guard caught: silence must not be pure chat."""

    def _bare_orchestrator(self):
        # The veto loop runs before any collaborator (gatekeeper, engine,
        # classifier) is touched, so a __new__ shell with the two pending
        # fields is enough — if the veto DOESN'T fire, the AttributeError
        # on gatekeeper fails the test loudly, which is what we want.
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        o._pending_query = None
        return o

    def test_silence_command_is_not_pure_chat(self):
        o = self._bare_orchestrator()
        for phrase in ("silêncio", "cala", "para de falar", "shut up"):
            assert o._is_pure_chat(phrase) is False, (
                f"{phrase!r} must veto the streaming shortcut — it has to "
                "reach the SILENCE_RE route, not an LLM chat stream"
            )

    def test_reminder_and_calendar_veto_explicitly(self):
        o = self._bare_orchestrator()
        assert o._is_pure_chat("me lembre de pagar a conta amanhã às 9h") is False
        assert o._is_pure_chat(
            "adicione ao calendário: dentista sexta às 15h") is False


class TestThePlannerTriggerVetoes:
    """The planner route is inline in _process_core, between the two
    registries, so the guard above can't see it. Without a tool word
    ("email", "calendário") the stream answered "planeja: …" with chat and
    never decomposed — 6 of 8 phrasings (s88, Codex lote 1 review)."""

    @pytest.mark.parametrize("phrase", [
        "planeja: organizar minha semana de estudos",
        "monte um plano para estudar pra prova de sexta",
        "cria um plano para arrumar a mudança",
        "plan: prepare for friday's interview",
        "decompõe: preparar a apresentação de segunda",
    ])
    def test_a_plan_request_is_not_pure_chat(self, phrase):
        o = TestSilenceGapFixed()._bare_orchestrator()
        assert o._is_pure_chat(phrase) is False
