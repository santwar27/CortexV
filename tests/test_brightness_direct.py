"""Brightness through both public entry points; never call an engine or the OS."""
from unittest.mock import MagicMock, call

import pytest


@pytest.mark.parametrize("stream", [False, True], ids=["process", "process_stream"])
def test_brightness_zero_bypasses_engines(stream, monkeypatch):
    from src.core.orchestrator import CortexOrchestrator

    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch.engine_analytical = MagicMock()
    orch.engine_creative = MagicMock()
    for engine in (orch.engine_analytical, orch.engine_creative):
        for method in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, method).side_effect = AssertionError("brightness must bypass the 8B")
    orch.macos = MagicMock()
    orch.macos.set_brightness.return_value = "Brilho em 0%."
    orch._pending_action = None
    orch._pending_query = None
    orch._shadow_classify = MagicMock()
    orch._maybe_announce_pending = lambda reply, mode: reply
    # A miss must stop locally, before any planner or model can run.
    monkeypatch.setattr(orch, "_handle_user_paths", MagicMock(side_effect=AssertionError("registry missed")))

    text = "Diminua brilho no mac para 0"
    if stream:
        assert list(orch.process_stream(text)) == [("Brilho em 0%.", True, "MAC_BRIGHTNESS", "pt")]
    else:
        assert orch.process(text) == ("Brilho em 0%.", "MAC_BRIGHTNESS")
    assert orch.macos.method_calls == [call.set_brightness(0, lang="pt")]
    for engine in (orch.engine_analytical, orch.engine_creative):
        # Project context can be pushed without asking a model for anything.
        for method in ("chat", "chat_stream", "chat_with_tools", "generate"):
            getattr(engine, method).assert_not_called()
