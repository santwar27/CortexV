"""
Self-test of the autouse state isolation (M9-A2, review Fable 5).

conftest._isolate_state_sinks must redirect every mutable state sink to
tmp_path so the suite can never pollute production again (gate_queue,
action_ledger, pending_action, gate_state — the audit was already
covered). These tests pin the fixture's behaviour; if someone removes or
weakens it, they fail before the pollution reaches Pedro's machine.
"""
import os


def _under_pytest_tmp(path: str) -> bool:
    # tmp_path lives under the pytest basetemp; the robust property to pin
    # is "not the production location", checked per sink below, plus
    # "absolute path into a temp dir" here.
    return os.path.isabs(path) and "pytest" in path


class TestStateSinkIsolation:
    def test_projects_redirected(self):
        from src.core.projects import ProjectStore, store
        assert store.db_path != ProjectStore.DEFAULT_PATH
        assert _under_pytest_tmp(store.db_path)
        assert os.environ["CORTEX_PROJECTS_DB"] == store.db_path
        assert ProjectStore().db_path == store.db_path

    def test_gate_queue_redirected(self):
        from src.core.gate_queue import gate_queue
        assert gate_queue.path != gate_queue.DEFAULT_PATH
        assert _under_pytest_tmp(gate_queue.path)

    def test_action_ledger_redirected(self):
        from src.core.action_ledger import ledger
        assert ledger.path != ledger.DEFAULT_PATH
        assert _under_pytest_tmp(ledger.path)

    def test_pending_action_file_redirected(self):
        from src.core.orchestrator import CortexOrchestrator
        assert CortexOrchestrator._PENDING_ACTION_FILE != \
            "data/pending_action.json"
        assert _under_pytest_tmp(CortexOrchestrator._PENDING_ACTION_FILE)

    def test_bare_logicgate_default_redirected(self):
        # LogicGate() with no args must NOT touch data/gate_state.json —
        # the function default is patched per test.
        from src.core.gatekeeper import LogicGate
        gate = LogicGate()
        assert gate.state_file != "data/gate_state.json"
        assert _under_pytest_tmp(gate.state_file)

    def test_explicit_paths_still_win(self, tmp_path):
        from src.core.gatekeeper import LogicGate
        explicit = str(tmp_path / "my_gate.json")
        assert LogicGate(state_file=explicit).state_file == explicit

    def test_writes_land_in_tmp_not_production(self):
        from src.core.gate_queue import gate_queue
        gate_queue.enqueue(
            "send_email", {"to": "x@y"}, origin="pytest-isolation",
        )
        # The write landed in the per-test tmp file…
        assert os.path.exists(gate_queue.path)
        with open(gate_queue.path) as f:
            assert "pytest-isolation" in f.read()
        # …and NOT in the production queue.
        prod = os.path.expanduser("~/.cortex/gate_queue.jsonl")
        if os.path.exists(prod):
            with open(prod) as f:
                assert "pytest-isolation" not in f.read()


class TestTheLastThreeDatabases:
    """fitness, jobs and costs (s88, Codex lote 5). The parity sweep runs
    "limpa as refeições de hoje" for real — a DELETE on today's meals in
    Pedro's ~/.cortex/fitness.db on every pre-commit, until this."""

    def test_fitness_redirected(self):
        from src.integrations import fitness
        log = fitness.get_log()
        assert log.db_path != fitness.FitnessLog.DEFAULT_PATH
        assert _under_pytest_tmp(log.db_path)

    def test_jobs_redirected(self):
        from src.integrations import jobs
        log = jobs.get_log()
        assert log.db_path != jobs.JobLog.DEFAULT_PATH
        assert _under_pytest_tmp(log.db_path)

    def test_costs_redirected(self):
        from src.core.cost_tracker import CostTracker, cost_tracker
        assert cost_tracker.db_path != CostTracker.DEFAULT_PATH
        assert _under_pytest_tmp(cost_tracker.db_path)
        assert _under_pytest_tmp(CostTracker().db_path)     # a fresh one too

    def test_clearing_today_clears_the_tmp_db(self):
        from src.integrations import fitness
        log = fitness.get_log()
        log.clear_today()
        assert os.path.exists(log.db_path)


class TestThePalace:
    """data/palace.db is Pedro's memory. ~480 openings of it per run came
    from default paths — the KnowledgeBase, every OllamaEngine() a test
    builds, and the Studier's retention pass, which archives stale
    documents (s88, Codex lote 5)."""

    def test_every_default_points_at_tmp(self):
        from src.core import system_status
        from src.core.knowledge_base import KnowledgeBase
        from src.core.memory import CortexMemory
        from src.core.studier import NightlyStudier
        from src.memory.palace import Palace
        for path in (Palace.__init__.__defaults__[0], CortexMemory.DEFAULT_PALACE_DB,
                     KnowledgeBase.DEFAULT_PALACE_DB, NightlyStudier.PALACE_DB_PATH,
                     str(system_status._gather_palace_docs.__defaults__[0])):
            assert _under_pytest_tmp(path), path
            assert not path.endswith("data/palace.db")

    def test_an_engine_built_bare_opens_the_tmp_palace(self):
        from src.core.engine import OllamaEngine
        e = OllamaEngine(model_name="cortex-lite")
        assert _under_pytest_tmp(e.memory._palace.db_path)
