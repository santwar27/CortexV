"""
E2E of the background governance tower (C5/M6-B3, review Fable 5):

    REAL producer → gate_queue.enqueue → gate_policy → gate_review → execution

The whole tower existed (queue, policy, review CLI — each unit-tested)
but had NEVER run end-to-end with a real producer: the only live
background producer (Studier) doesn't stage S3 actions, so the path
"background job picks a destructive tool" was theory. This drives the
REAL producer boundary — `staging_mixin._execute_tool` with
`background=True`, the exact code `chat_with_tools` hands its
tool_executor — through every decision branch, with fakes only at the
network edge. Prerequisite for flipping ANY autonomous scaffold
(DeepStudier, Dream Loop) that could produce S3 actions.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from src import gate_review
from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.gate_queue import gate_queue


def _background_orch():
    """A real orchestrator shaped like a background producer (the way
    nightly/scheduled jobs would hold it): background=True routes every
    destructive tool into the gate queue instead of inline staging."""
    from src.core.orchestrator import CortexOrchestrator

    orch = CortexOrchestrator.__new__(CortexOrchestrator)
    orch.background = True
    orch.background_origin = "studier-nightly"
    orch._pending_action = None
    orch._intent_cache = OrderedDict()
    orch.gatekeeper = MagicMock()
    orch.reader = MagicMock()
    orch.writer = MagicMock()
    orch.web_searcher = MagicMock()
    return orch


def _queued_ids():
    return [e["id"] for e in gate_queue.pending()]


class TestProducerToReviewToExecution:
    def test_s3_action_is_queued_not_executed(self):
        orch = _background_orch()
        result = orch._execute_tool(
            "send_email",
            {"to": "lara@example.org", "subject": "oi", "body": "corpo"},
        )
        # The producer gets the skip-followup brief, nothing ran.
        assert result.startswith(SKIP_FOLLOWUP_MARKER)
        assert "enfileirada" in result
        pend = gate_queue.pending()
        assert len(pend) == 1
        assert pend[0]["type"] == "send_email"
        assert pend[0]["origin"] == "studier-nightly"
        # Fail-closed policy reason is recorded on the entry.
        assert "fail-closed" in (pend[0].get("reason") or "")

    def test_pending_action_slot_untouched_by_background(self):
        # The whole point of the queue: a background S3 must NOT clobber
        # what an interactive user has staged.
        orch = _background_orch()
        orch._pending_action = {"type": "create_event", "args": {}}
        orch._execute_tool("send_email", {"to": "a@b", "subject": "s",
                                          "body": "b"})
        assert orch._pending_action == {"type": "create_event", "args": {}}

    def test_review_list_shows_the_queued_action(self):
        orch = _background_orch()
        orch._execute_tool("send_email", {"to": "lara@example.org",
                                          "subject": "oi", "body": "x"})
        listing = gate_review.render_list()
        assert "send_email" in listing
        assert "studier-nightly" in listing
        assert "lara@example.org" in listing

    def test_approve_executes_via_runner_and_marks_executed(self):
        orch = _background_orch()
        orch._execute_tool("send_email", {"to": "a@b", "subject": "s",
                                          "body": "corpo"})
        gid = _queued_ids()[0]

        ran = []

        def runner(action_type, args):
            ran.append((action_type, args))
            return ("Email enviado.", "AGENT_ACTION_DONE")

        out = gate_review.approve(gid, runner=runner)
        assert "[executado]" in out
        assert ran == [("send_email", {"to": "a@b", "subject": "s",
                                       "body": "corpo"})]
        assert gate_queue.status(gid) == "executed"
        assert gate_queue.pending() == []

    def test_reject_drops_without_running(self):
        orch = _background_orch()
        orch._execute_tool("send_email", {"to": "a@b", "subject": "s",
                                          "body": "x"})
        gid = _queued_ids()[0]
        out = gate_review.reject(gid)
        assert "rejeitado" in out
        assert gate_queue.status(gid) == "rejected"
        assert gate_queue.pending() == []

    def test_decided_action_cannot_be_double_executed(self):
        orch = _background_orch()
        orch._execute_tool("send_email", {"to": "a@b", "subject": "s",
                                          "body": "x"})
        gid = _queued_ids()[0]
        gate_review.reject(gid)
        ran = []
        out = gate_review.approve(
            gid, runner=lambda t, a: ran.append(t) or ("x", "AGENT_ACTION_DONE")
        )
        assert "já foi decidido" in out
        assert ran == []

    def test_failed_execution_is_recorded_as_failed(self):
        orch = _background_orch()
        orch._execute_tool("send_email", {"to": "a@b", "subject": "s",
                                          "body": "x"})
        gid = _queued_ids()[0]
        out = gate_review.approve(
            gid, runner=lambda t, a: ("Gmail fora", "AGENT_ACTION_ERROR")
        )
        assert "FALHOU" in out
        assert gate_queue.status(gid) == "failed"


class TestPolicyAutoReject:
    def test_background_system_control_is_auto_rejected(self):
        # design-reference ch. 13: a 3am job must never sleep the Mac. The policy
        # decides AUTO_REJECT, the queue records the decision trail, and
        # nothing waits for the human.
        orch = _background_orch()
        result = orch._execute_tool("mac_sleep", {})
        assert "recusada por política" in result
        assert gate_queue.pending() == []
        # The decision trail exists (queued then rejected).
        hist = gate_queue.events() if hasattr(gate_queue, "events") else None
        if hist is not None:
            kinds = [e.get("event") for e in hist]
            assert "queued" in kinds and "rejected" in kinds

    def test_zero_auto_approve_rules_ship(self):
        # Soberania red line (M6-B2): auto-approve is opt-in forever.
        from src.core.gate_policy import _DEFAULT_APPROVE_RULES
        assert _DEFAULT_APPROVE_RULES == []
