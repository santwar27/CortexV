"""
M8-A1 — the promoter↔registry gate plumbing on the orchestrator side.

The registry half (tests/test_skill_registry.py) proves the tool_sequence
contract against fake callbacks. This file proves the REAL callbacks the
orchestrator injects:

  - `_skill_macro_run_step` executes S1/S2 via the per-tool dispatch and
    REFUSES S3 even if called with one (defence in depth — the registry
    already diverts S3 to the stager, so two independent layers must be
    wrong for a macro to bypass the gate);
  - `_skill_macro_stage_step` stages an S3 step in `_pending_action`
    exactly like a direct S3 tool call (same confirmation prompt, same
    sim/não), and in background context routes to the batch gate queue
    instead (never clobbering the interactive slot);
  - end-to-end: a registered macro runs its safe prefix and ends staged,
    waiting for confirmation.

Builds the orchestrator via __new__ (no __init__) — the same trick
tests/test_gate_background_staging.py uses.
"""
from types import SimpleNamespace

import pytest

from src.core.gate_queue import GateQueue
from src.core.orchestrator import CortexOrchestrator
from src.core.skill_registry import SkillRegistry
from src.memory.palace import Palace


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._pending_action = None
    o._PENDING_ACTION_FILE = str(tmp_path / "pending_action.json")
    o._reply_lang = "pt"
    o.background = False
    o.background_origin = "background"
    q = GateQueue(str(tmp_path / "gate_queue.jsonl"))
    monkeypatch.setattr("src.core.staging_mixin.gate_queue", q)
    return o, q


class TestRunStep:
    def test_s1_step_runs_through_tool_dispatch(self, orch):
        o, _ = orch
        o.reader = SimpleNamespace(
            read_document=lambda filename: "conteúdo do relatório")
        out = o._skill_macro_run_step("read_document", {"filename": "r.txt"})
        assert out == "conteúdo do relatório"

    def test_s3_step_is_refused_even_here(self, orch):
        """Defence in depth: the registry never sends S3 to run_step, but
        if a bug (or a future caller) does, the executor itself refuses."""
        o, q = orch
        out = o._skill_macro_run_step(
            "send_email", {"to": "a@b", "subject": "s", "body": "x"})
        assert out.startswith("[SYSTEM ERROR")
        assert "gate" in out
        # And nothing was staged or queued.
        assert o._pending_action is None
        assert q.pending() == []


class TestStageStep:
    def test_interactive_stages_like_direct_s3(self, orch):
        o, q = orch
        prompt = o._skill_macro_stage_step(
            "send_email", {"to": "a@b", "subject": "s", "body": "x"})
        # Same pending slot, same deterministic human prompt as a direct
        # S3 tool call — the existing sim/não flow takes it from here.
        staged = dict(o._pending_action)
        staged.pop("staged_at", None)  # TTL stamp (M1-A9), time-dependent
        staged.pop("id", None)  # the action's identity (s88), random
        assert staged == {
            "type": "send_email",
            "args": {"to": "a@b", "subject": "s", "body": "x"},
            "origin": None,
        }
        assert "Confirmar envio do email?" in prompt
        assert q.pending() == []

    def test_background_enqueues_never_stages(self, orch):
        o, q = orch
        o.background = True
        o.background_origin = "scheduled:test"
        msg = o._skill_macro_stage_step("create_reminder", {"title": "x"})
        assert "enfileirada" in msg
        assert o._pending_action is None
        pend = q.pending()
        assert len(pend) == 1
        assert pend[0]["type"] == "create_reminder"
        assert pend[0]["origin"] == "scheduled:test"


class TestEndToEnd:
    def test_macro_runs_safe_prefix_then_stages_s3(self, orch, tmp_path):
        o, q = orch
        o.reader = SimpleNamespace(
            read_document=lambda filename: "conteúdo do relatório")
        palace = Palace(db_path=str(tmp_path / "palace.db"))
        try:
            registry = SkillRegistry(palace)
            registry.register(
                "manda-relatorio", r"\bmanda o relatório\b", "tool_sequence",
                {"steps": [
                    {"tool": "read_document", "args": {"filename": "r.txt"}},
                    {"tool": "send_email",
                     "args": {"to": "a@b", "subject": "Relatório",
                              "body": "segue"}},
                ]},
            )
            skill = registry.match("manda o relatório")
            assert skill is not None
            reply = registry.execute(
                skill, "manda o relatório",
                run_step=o._skill_macro_run_step,
                stage_step=o._skill_macro_stage_step,
            )
            # The S1 step ran, the S3 step is STAGED (not sent) and the
            # reply carries both the output and the confirmation ask.
            assert "conteúdo do relatório" in reply
            assert "Confirmar envio do email?" in reply
            staged = dict(o._pending_action)
            staged.pop("staged_at", None)  # TTL stamp (M1-A9)
            staged.pop("id", None)  # the action's identity (s88), random
            assert staged == {
                "type": "send_email",
                "args": {"to": "a@b", "subject": "Relatório",
                         "body": "segue"},
                "origin": None,
            }
            assert q.pending() == []
            # Macro did its job → counts as an invocation.
            rows = registry.list_all()
            assert rows[0]["invocation_count"] == 1
        finally:
            palace.close()
