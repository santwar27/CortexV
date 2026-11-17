"""
gate_review CLI — list / approve / reject, plus the full producer→queue→
review→execute path end to end.

The CLI takes injectable `queue` + `runner` seams so the approval flow is
exercised without a live orchestrator or real Gmail/Calendar calls: the
fake runner stands in for `orchestrator._run_action`.
"""
import pytest

from src import gate_review
from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.gate_queue import GateQueue
from src.core.orchestrator import CortexOrchestrator


@pytest.fixture
def queue(tmp_path):
    return GateQueue(str(tmp_path / "gate_queue.jsonl"))


class TestRenderList:
    def test_empty(self, queue):
        out = gate_review.render_list(queue)
        assert "Nenhuma ação" in out

    def test_lists_pending_with_args(self, queue):
        queue.enqueue("send_email", {"to": "a@b", "subject": "Oi"}, origin="studier")
        out = gate_review.render_list(queue)
        assert "send_email" in out
        assert "to=a@b" in out
        assert "origin=studier" in out
        assert "approve" in out  # shows the next-step hint

    def test_shows_long_arg_values_whole(self, queue):
        queue.enqueue("send_email", {"body": "x" * 200})
        out = gate_review.render_list(queue)
        assert "x" * 200 in out
        assert "…" not in out


class TestApprove:
    def test_approve_runs_and_marks_executed(self, queue):
        gid = queue.enqueue("create_reminder", {"title": "Pay bill"})
        ran = []

        def fake_runner(t, a):
            ran.append((t, a))
            return ("Lembrete criado.", "AGENT_ACTION_DONE")

        out = gate_review.approve(gid, queue=queue, runner=fake_runner)
        assert "executado" in out
        assert ran == [("create_reminder", {"title": "Pay bill"})]
        assert queue.pending() == []
        assert queue.status(gid) == "executed"

    def test_approve_failure_marks_failed(self, queue):
        gid = queue.enqueue("send_email", {"to": "a@b"})

        def failing_runner(t, a):
            return ("Falha ao enviar email: nope", "AGENT_ACTION_ERROR")

        out = gate_review.approve(gid, queue=queue, runner=failing_runner)
        assert "FALHOU" in out
        assert queue.status(gid) == "failed"

    def test_approve_unknown_id(self, queue):
        out = gate_review.approve("nope", queue=queue, runner=lambda t, a: ("", "X"))
        assert "não encontrado" in out

    def test_approve_already_decided_does_not_rerun(self, queue):
        gid = queue.enqueue("send_email", {})
        calls = []
        runner = lambda t, a: (calls.append(1), ("ok", "AGENT_ACTION_DONE"))[1]
        gate_review.approve(gid, queue=queue, runner=runner)
        # second approve must be a no-op (no second execution)
        out = gate_review.approve(gid, queue=queue, runner=runner)
        assert "já foi decidido" in out
        assert len(calls) == 1


class TestReject:
    def test_reject_drops_without_running(self, queue):
        gid = queue.enqueue("send_email", {})
        out = gate_review.reject(gid, queue=queue)
        assert "rejeitado" in out
        assert queue.pending() == []
        assert queue.status(gid) == "rejected"

    def test_reject_unknown_id(self, queue):
        assert "não encontrado" in gate_review.reject("nope", queue=queue)

    def test_cannot_reject_after_approve(self, queue):
        gid = queue.enqueue("send_email", {})
        gate_review.approve(gid, queue=queue, runner=lambda t, a: ("ok", "AGENT_ACTION_DONE"))
        out = gate_review.reject(gid, queue=queue)
        assert "já foi decidido" in out


class TestEndToEnd:
    def test_producer_to_approval(self, tmp_path, monkeypatch):
        """Background orchestrator enqueues an S3 action → CLI lists it →
        approve runs it via a fake runner → executed event recorded."""
        q = GateQueue(str(tmp_path / "q.jsonl"))
        monkeypatch.setattr("src.core.staging_mixin.gate_queue", q)

        # Producer: a background orchestrator dispatches a destructive tool.
        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        o._PENDING_ACTION_FILE = str(tmp_path / "pa.json")
        o.background = True
        o.background_origin = "scheduled:test"
        reply = o._execute_tool("create_reminder", {"title": "Buy milk"})
        assert reply.startswith(SKIP_FOLLOWUP_MARKER)

        # Queue now has exactly one pending action.
        pend = q.pending()
        assert len(pend) == 1
        gid = pend[0]["id"]

        # CLI sees it.
        assert "create_reminder" in gate_review.render_list(q)

        # Approve → runs via the (faked) per-tool dispatch.
        executed = {}

        def fake_runner(t, a):
            executed["call"] = (t, a)
            return (f"Lembrete '{a.get('title')}' criado.", "AGENT_ACTION_DONE")

        out = gate_review.approve(gid, queue=q, runner=fake_runner)
        assert "executado" in out
        assert executed["call"] == ("create_reminder", {"title": "Buy milk"})
        assert q.status(gid) == "executed"
        assert q.pending() == []

    def test_main_list_smoke(self, tmp_path, monkeypatch, capsys):
        # `python -m src.gate_review` (no args) → list path, exit 0.
        monkeypatch.setenv("CORTEX_GATE_QUEUE", str(tmp_path / "empty.jsonl"))
        # render_list uses the module singleton; rebind it to honour the env.
        monkeypatch.setattr(gate_review, "gate_queue", GateQueue())
        rc = gate_review.main([])
        assert rc == 0
        assert "Nenhuma ação" in capsys.readouterr().out
