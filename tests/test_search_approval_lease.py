"""
Approved search vs. the lease (M1-A10, review Fable 5).

On "sim" to a search proposal, the lease was granted BEFORE the search —
correct, the search needs it — but a failed search (Tavily down, no
results) was masked as success: the SEARCH_ERROR mode was discarded
(`reply, _ =`), the user got "[Internet liberada até …]" + the error
text under mode AUTO_SEARCH, and 4 hours of internet stayed open for a
search that never happened. Now the real mode propagates and the lease
is rolled back when the very search that justified it failed.
"""
from unittest.mock import MagicMock

import pytest

from src.core.gatekeeper import LogicGate
from src.core.orchestrator import CortexOrchestrator


def _orch():
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o.engine_analytical = MagicMock()
    o.engine_analytical.chat.return_value = "resposta sintetizada"
    o.gatekeeper = LogicGate()  # state file isolated by conftest
    o.lease_hours = 4
    o.web_searcher = MagicMock()
    o._pending_action = None
    o._pending_query = "resultado do jogo do flamengo"
    o._pending_engine = o.engine_analytical
    o._shadow_classify = lambda text: None
    return o


class TestApprovedSearchFails:
    def test_failed_search_rolls_back_lease_and_reports_error(self):
        o = _orch()
        o.web_searcher.search.return_value = (
            "[SYSTEM NOTICE: No relevant information found on the web "
            "for this query.]"
        )
        reply, mode = o._process_core("sim")
        assert mode == "SEARCH_ERROR"
        assert "Internet liberada" not in reply
        assert not o.gatekeeper.is_lease_active(), (
            "a search that never happened must not leave 4h of lease open"
        )

    def test_successful_search_keeps_lease_and_prefix(self):
        o = _orch()
        o.web_searcher.search.return_value = "Source 1 (X):\nflamengo 2x0\n[url]"
        reply, mode = o._process_core("sim")
        assert mode == "AUTO_SEARCH"
        assert "Internet liberada" in reply
        assert o.gatekeeper.is_lease_active()
