"""
Shared pytest fixtures for the CORTEX test suite.

Goal: cover the pure logic (regex, sanitisers, queue mutations, intake
filter, dup detection) without touching ChromaDB, Ollama, Spotify, or
the live filesystem outside a tmp directory.
"""
import os
from pathlib import Path
import sys
from uuid import uuid4

# Ensure the project root is on sys.path so 'src.*' imports resolve when
# pytest is invoked from anywhere. CWD-independent.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


def _palace_defaults_to(path: str) -> None:
    """Point every DEFAULT Palace path at `path` (see _isolate_state_sinks)."""
    from src.core import system_status
    from src.core.knowledge_base import KnowledgeBase
    from src.core.memory import CortexMemory
    from src.core.studier import NightlyStudier
    from src.memory.palace import Palace
    Palace.__init__.__defaults__ = (path,) + Palace.__init__.__defaults__[1:]
    CortexMemory.DEFAULT_PALACE_DB = path
    KnowledgeBase.DEFAULT_PALACE_DB = path
    NightlyStudier.PALACE_DB_PATH = path
    system_status._gather_palace_docs.__defaults__ = (Path(path),)


def _studies_defaults_to(root: Path, set_attr=setattr) -> None:
    """Redirect captured study defaults; explicit test paths still take priority."""
    from src.core import digest
    from src.core.knowledge_base import KnowledgeBase
    from src.core.studier import NightlyStudier
    from src.core.studies_review import StudiesReview
    from src.ui import aggregator

    pending, approved = str(root / "pending"), str(root / "approved")
    set_attr(StudiesReview.__init__, "__defaults__",
             (pending, approved) + StudiesReview.__init__.__defaults__[2:])
    defaults = KnowledgeBase.__init__.__defaults__
    set_attr(KnowledgeBase.__init__, "__defaults__",
             defaults[:1] + (approved,) + defaults[2:])
    set_attr(NightlyStudier.__init__, "__defaults__",
             NightlyStudier.__init__.__defaults__[:-1] + (pending,))
    set_attr(digest, "_STUDIER_DIR", pending)
    set_attr(aggregator.studier_snapshot, "__defaults__",
             aggregator.studier_snapshot.__defaults__[:-1] + (str(root),))

    from src.core.orchestrator import CortexOrchestrator
    set_attr(CortexOrchestrator, "_LAST_ANNOUNCE_PATH",
             str(root.parent / "last_announce.txt"))


# Before any test module is imported: one that imports src.api builds the
# global orchestrator at import time — its engines open the default Palace
# before a single fixture runs (s88: 3 openings of the real data/palace.db
# per run, after the per-test isolation below). Each test still gets its own.
import atexit as _atexit  # noqa: E402
import shutil as _shutil  # noqa: E402
import tempfile as _tempfile  # noqa: E402
_SESSION_DIR = _tempfile.mkdtemp(prefix="cortex-tests-")
_atexit.register(_shutil.rmtree, _SESSION_DIR, True)
# src.api installs a repository-relative auth.log handler during collection.
# Prebind that logger so HTTP regressions leave no test entries in a real log.
import logging as _logging  # noqa: E402
_auth_handler = _logging.FileHandler(os.path.join(_SESSION_DIR, "auth.log"))
_auth_handler.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
_logging.getLogger("cortex.auth").addHandler(_auth_handler)
_atexit.register(_auth_handler.close)
# src.api builds its orchestrator during collection, before fixtures can
# redirect the effect singleton opened by reconcile_abandoned().
os.environ["CORTEX_EFFECTS_DB"] = os.path.join(_SESSION_DIR, "effects.db")
os.environ["CORTEX_PDF_POPPLER"] = "0"
_palace_defaults_to(os.path.join(_SESSION_DIR, "palace.db"))
# src.api constructs StudiesReview during collection too. Function defaults
# capture their paths at definition time; patching module constants is too late.
_studies_defaults_to(Path(_SESSION_DIR) / "knowledge")


@pytest.fixture(autouse=True)
def _block_real_ollama(monkeypatch):
    # On 20/10 pre-commit hung connecting to real Ollama: 7 tests leaked a
    # model call through a fail-soft path and pass with Ollama switched off.
    import errno
    import socket

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def is_ollama(sock, address):
        return (
            sock.family in (socket.AF_INET, socket.AF_INET6)
            and isinstance(address, tuple)
            and len(address) >= 2
            and address[1] == 11434
        )

    def connect(sock, address):
        if is_ollama(sock, address):
            raise ConnectionRefusedError(
                errno.ECONNREFUSED, "real Ollama blocked in tests"
            )
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if is_ollama(sock, address):
            return errno.ECONNREFUSED
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path, monkeypatch):
    """Redirect the audit singleton to a per-test temp file so the suite never
    pollutes the real ~/.cortex/audit.jsonl.

    The audit log feeds the evidence miners (policy_detector, skill_promoter);
    before this, any test that drove the orchestrator (e.g. the planner
    end-to-end round-trip) appended its synthetic "planeja: a.txt -> sim"
    events to the production log and skewed those reports. The singleton
    resolves its path once at import, so setting the env var alone is too
    late — we patch `audit.path` directly. We also export CORTEX_AUDIT_LOG so
    any AuditLogger built mid-test (or a subprocess) lands in the temp path
    too. monkeypatch restores both after each test."""
    from src.core.audit import audit
    audit_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit, "path", audit_path)
    monkeypatch.setenv("CORTEX_AUDIT_LOG", audit_path)
    yield


@pytest.fixture(autouse=True)
def _isolate_state_sinks(tmp_path, monkeypatch):
    """Same isolation contract as the audit fixture, for the OTHER mutable
    state singletons (M9-A2, review Fable 5). Before this, only the audit
    was redirected: a test driving the destructive S3 path appended to the
    production ~/.cortex/gate_queue.jsonl and data/action_ledger.jsonl, and
    any test that forgot to patch _PENDING_ACTION_FILE wrote a REAL
    data/pending_action.json — which a later "sim" from Pedro could
    execute (pending has no TTL — M1-A9).

    - gate_queue / ledger: singletons resolve their path at import, so the
      env var alone is too late — patch `.path` directly (env exported too
      for instances built mid-test).
    - pending action: class attr on CortexOrchestrator; tests that patch it
      themselves just override this default with their own tmp file.
    - gatekeeper: LogicGate is NOT a singleton — its default state_file
      ("data/gate_state.json") is a function default, patched via
      __defaults__ so a bare LogicGate() in a test can't touch the real
      lease state. Explicit-path instantiation is unaffected."""
    # Empty studies and an independent daily announcement marker for EVERY test.
    # Tests wanting briefings can supply their own paths or populate these.
    _studies_defaults_to(tmp_path / "knowledge", monkeypatch.setattr)
    gq_path = str(tmp_path / "gate_queue.jsonl")
    ledger_path = str(tmp_path / "action_ledger.jsonl")
    pending_path = str(tmp_path / "pending_action.json")
    gate_state_path = str(tmp_path / "gate_state.json")

    from src.core.gate_queue import gate_queue
    monkeypatch.setattr(gate_queue, "path", gq_path)
    monkeypatch.setenv("CORTEX_GATE_QUEUE", gq_path)

    from src.core.action_ledger import ledger
    monkeypatch.setattr(ledger, "path", ledger_path)
    monkeypatch.setenv("CORTEX_ACTION_LEDGER", ledger_path)

    from src.core.orchestrator import CortexOrchestrator
    monkeypatch.setattr(
        CortexOrchestrator, "_PENDING_ACTION_FILE", pending_path
    )

    from src.core.gatekeeper import LogicGate
    monkeypatch.setattr(
        LogicGate.__init__, "__defaults__", (gate_state_path,)
    )

    # deep_queue (M8-A2, s66): resolved from the env at every deep_queue()
    # call, so the env var alone is enough. Added after s67 found two
    # development-probe topics sitting in the PRODUCTION
    # data/deep_queue.jsonl, waiting for the nightly to burn real searches
    # on them. Tests that set their own CORTEX_DEEP_QUEUE just override.
    monkeypatch.setenv("CORTEX_DEEP_QUEUE", str(tmp_path / "deep_queue.jsonl"))

    # contact_book (s67 observation, closed s69): the singleton resolves
    # ~/.cortex/contacts.db at import, so without this an unpatched lookup
    # reads (and add/update WRITES) the real contact DB. Same patch-the-
    # singleton pattern as audit/gate_queue; the lazy connection is reset
    # so a previous test's handle can't keep pointing at the old path.
    from src.integrations.contacts import contact_book
    monkeypatch.setattr(contact_book, "db_path", str(tmp_path / "contacts.db"))
    monkeypatch.setattr(contact_book, "_conn", None)
    monkeypatch.setenv("CORTEX_CONTACTS_DB", str(tmp_path / "contacts.db"))

    # PS5 pairing credentials are resolved lazily, never from the real home.
    monkeypatch.setenv("CORTEX_PS5_CONFIG", str(tmp_path / "ps5.json"))

    # chat_store (s82): the /ui chat history — ~/.cortex/chat.db, resolved by
    # the singleton at import, same shape as contact_book above. Missing here
    # until now because when the store was written (s81) only the UI route
    # touched it, and those tests bring their own. Then s82 wired the VOICE
    # mirror into it (_persist_voice_turn), and every TestClient call to
    # /ask_voice or /ask_stream started appending to the user's REAL chat: he
    # opened the app and found a "🎙️ Voz · Tue 08/09" conversation holding six
    # turns he never spoke ("Resposta falada.", "Oi the user"), written by this
    # suite in 17ms. Nothing went red — the tests assert on the in-memory
    # buffer, so the pollution was silent.
    from src.ui.chat_store import store as chat_store
    monkeypatch.setattr(chat_store, "db_path", str(tmp_path / "chat.db"))
    monkeypatch.setattr(chat_store, "_conn", None)
    monkeypatch.setenv("CORTEX_CHAT_DB", str(tmp_path / "chat.db"))

    from src.ui.session_store import store as session_store
    monkeypatch.setattr(session_store, "db_path", str(tmp_path / "ui_sessions.db"))
    monkeypatch.setattr(session_store, "_conn", None)
    monkeypatch.setenv("CORTEX_UI_SESSIONS_DB", str(tmp_path / "ui_sessions.db"))

    # UI turns also read the project singleton, even without project commands.
    from src.core.projects import store as projects_store
    monkeypatch.setattr(projects_store, "db_path", str(tmp_path / "projects.db"))
    monkeypatch.setattr(projects_store, "_conn", None)
    monkeypatch.setenv("CORTEX_PROJECTS_DB", str(tmp_path / "projects.db"))

    # focus store (s86): ~/.cortex/focus.db holds the user's real focus sessions
    # — the same singleton-at-import shape as chat_store, isolated from the
    # day it was written so a route test can never open or close his session.
    from src.core.focus import store as focus_store
    monkeypatch.setattr(focus_store, "db_path", str(tmp_path / "focus.db"))
    monkeypatch.setattr(focus_store, "_conn", None)
    monkeypatch.setenv("CORTEX_FOCUS_DB", str(tmp_path / "focus.db"))
    # ...and its reminders (s86 bloco 2b): every focus start queues a REAL
    # Reminders.app call on a background worker — in the suite that is a
    # reminder in the user's "Cortex Foco" list per test. Queued jobs are
    # dropped here; the tests that exercise them run them inline against a
    # stubbed client.
    from src.core import focus_mixin
    monkeypatch.setattr(focus_mixin, "_in_background", lambda job, *args: None)

    # Approved S3 effects must never enter the user's real durable record.
    from src.core.effect_store import store as effect_store
    monkeypatch.setattr(effect_store, "db_path", str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "_conn", None)
    monkeypatch.setattr(effect_store, "_owner_token", uuid4().hex)
    monkeypatch.setattr(effect_store, "_owner_fd", None)
    monkeypatch.setenv("CORTEX_EFFECTS_DB", str(tmp_path / "effects.db"))

    # fitness / jobs / costs (s88, Codex lote 5): the last three ~/.cortex
    # databases a test could reach. The parity sweep runs "limpa as refeições
    # de hoje" and "limpa os treinos de hoje" for real: DELETE FROM meals/
    # workouts for TODAY in the user's fitness.db, on every pre-commit. Nothing
    # of his was lost — he logged nothing on those days — but the id counter
    # shows 408 meals and 306 workouts written there and deleted since 03/10.
    # fitness and jobs are get_log() singletons built on first use (reset so
    # a new one reads the env); the cost tracker is built at import (patched).
    monkeypatch.setenv("CORTEX_FITNESS_DB", str(tmp_path / "fitness.db"))
    from src.integrations import fitness
    monkeypatch.setattr(fitness, "_log_singleton", None)
    monkeypatch.setenv("CORTEX_JOBS_DB", str(tmp_path / "jobs.db"))
    from src.integrations import jobs
    monkeypatch.setattr(jobs, "_log_singleton", None)
    monkeypatch.setenv("CORTEX_COST_DB", str(tmp_path / "costs.db"))
    from src.core.cost_tracker import cost_tracker
    monkeypatch.setattr(cost_tracker, "db_path", str(tmp_path / "costs.db"))
    monkeypatch.setattr(cost_tracker, "_conn", None)

    # the Palace (s88, Codex lote 5): data/palace.db is the user's memory. A spy
    # on Palace() found ~480 openings of it per run, from four defaults: the
    # KnowledgeBase (372, on the turn path), CortexMemory (102, every
    # OllamaEngine() a test builds), system_status (read-only) — and the
    # Studier's retention pass, which ARCHIVES stale documents in the real
    # Palace from the suite (7). Nothing was written in the measured runs; the
    # path pointed there anyway. Every default now resolves to this test's tmp.
    palace_path = str(tmp_path / "palace.db")
    from src.memory.palace import Palace
    monkeypatch.setattr(Palace.__init__, "__defaults__",
                        (palace_path,) + Palace.__init__.__defaults__[1:])
    from src.core.memory import CortexMemory
    monkeypatch.setattr(CortexMemory, "DEFAULT_PALACE_DB", palace_path)
    from src.core.knowledge_base import KnowledgeBase
    monkeypatch.setattr(KnowledgeBase, "DEFAULT_PALACE_DB", palace_path)
    from src.core.studier import NightlyStudier
    monkeypatch.setattr(NightlyStudier, "PALACE_DB_PATH", palace_path)
    from src.core import system_status
    monkeypatch.setattr(system_status._gather_palace_docs, "__defaults__",
                        (Path(palace_path),))

    # entities graph cache (M7-A5, s70): module-level TTL cache — a graph
    # built by one test (from patched gather()) must not leak into the next.
    from src.core import entities
    entities.invalidate_cache()

    # screen-context kill-switch flag (#3): the "para de ver minha tela" /
    # "volta a ver minha tela" voice handlers WRITE/REMOVE this flag file, and
    # the parity sweep runs them for real. Point it at a per-test tmp path so a
    # test can't raise (or clear) the user's real /tmp/cortex_screen_off — which
    # would silently turn his screen reading off (or back on) mid-suite. Tests
    # that drive the flag directly just set their own value over this default.
    monkeypatch.setenv(
        "CORTEX_SCREEN_OFF_FLAG", str(tmp_path / "screen_off_flag"))
    yield
    if effect_store._owner_fd is not None:
        os.close(effect_store._owner_fd)


@pytest.fixture
def tmp_study_queue(tmp_path):
    """Fresh StudyQueue backed by a tmp JSONL file."""
    from src.core.study_queue import StudyQueue
    return StudyQueue(path=str(tmp_path / "study_queue.jsonl"))
