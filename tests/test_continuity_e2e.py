"""UI -> fresh app/stores -> voice, using the production API and SQLite.

Only model calls, external integrations and time are replaced. Each boot
executes src/api.py anew, without replacing its routes or orchestrator.
TestClient buffers SSE, so cancellation is coordinated at the fake LLM
while a second request presses the real /ui/chat/cancel endpoint.
"""
import copy
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core import engine, focus, focus_mixin, projects, projects_mixin
from src.core import orchestrator as orchestrator_module
from src.core.gatekeeper import LogicGate
from src.core.knowledge_base import KnowledgeBase
from src.core.memory import CortexMemory
from src.memory.embeddings import OllamaEmbedder
from src.memory.palace import Palace
from src.ui import chat_store, routes
from src.ui.conversation import ConversationBuffer
from tests.test_backup_script import (
    BACKUP_SCRIPT, RESTORE_SCRIPT, _latest_snapshot, _run_backup,
)


ROOT = Path(__file__).resolve().parents[1]
AUTH = {"X-Cortex-Key": "teste-fake"}
QUESTION = "onde a gente parou?"
NEXT = "mandar o PDF até sexta"
EXPECTED = (
    'Projeto “Currículo” — objetivo: candidatura da vaga X.\n'
    'Último feito: revisar a experiência (ontem às 15:15).\n'
    'Falta: pedir a carta.\n'
    'Próximo passo: mandar o PDF até sexta (hoje às 15:15).\n'
    'Bora 25 min nisso? (sim)'
)


class Model:
    """Fail even when production catches Exception; retain actual payloads."""

    def __init__(self):
        self.allowed = False
        self.calls = []
        self.entered = Event()
        self.release = Event()
        self.interrupt = False

    def record(self, kind, kwargs):
        if not self.allowed:
            pytest.fail(f"LLM {kind} chamado em comando/retomada: {kwargs}")
        self.calls.append((kind, copy.deepcopy(kwargs)))

    def generate(self, **kwargs):
        self.record("generate", kwargs)
        return {"response": "NO"}  # Real realtime classifier, fake inference.

    def chat(self, **kwargs):
        self.record("chat", kwargs)
        reply = {"message": {"role": "assistant", "content": "Resposta fictícia."}}
        if not kwargs.get("stream"):
            return reply

        def tokens():
            yield {"message": {"content": "Sugestão parcial. "}}
            if self.interrupt:
                self.entered.set()
                assert self.release.wait(10), "cancelamento não liberou o LLM"
            # Model output must never become explicit project progress.
            yield {"message": {"content": "feito: conclusão inventada pelo modelo."}}
        return tokens()

    def system(self):
        calls = [kw for kind, kw in self.calls if kind == "chat"]
        assert calls, "o turno não chegou ao engine real"
        return calls[-1]["messages"][0]["content"]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=datetime(2026, 10, 19, 15, 15,
                                         tzinfo=ZoneInfo("Europe/London")))
    model = Model()
    monkeypatch.setenv("CORTEX_API_KEY", "teste-fake")
    monkeypatch.setenv("CORTEX_IP_ALLOWLIST", "")
    monkeypatch.setenv("CORTEX_RATE_LIMIT", "10000/minute")
    monkeypatch.setenv("CORTEX_ARBITER_SHADOW", "0")
    monkeypatch.setattr(focus_mixin, "local_now", lambda: clock.now)
    monkeypatch.setattr(projects_mixin, "local_now", lambda: clock.now)
    monkeypatch.setattr(projects, "_now", lambda: clock.now.astimezone(timezone.utc).isoformat())
    monkeypatch.setattr(engine.ollama, "chat", model.chat)
    monkeypatch.setattr(engine.ollama, "generate", model.generate)
    monkeypatch.setattr(OllamaEmbedder, "embed",
                        lambda self, text: np.array([1., 0., 0.], dtype=np.float32))

    # External clients only; readers, writers, memory, dispatch and gates stay real.
    for name in ("SpotifyPlayer", "MacOSController", "VisionProcessor", "WebSearcher"):
        monkeypatch.setattr(orchestrator_module, name, Mock(return_value=Mock()))
    from src.integrations import reminders_client, gmail_client, calendar_client
    for module, names in ((gmail_client, ("send_email", "list_recent")),
                          (calendar_client, ("list_events", "create_event"))):
        for name in names:
            monkeypatch.setattr(module, name, Mock(side_effect=AssertionError(
                f"integração inesperada: {name}")))
    monkeypatch.setattr(reminders_client, "ensure_list", Mock())
    monkeypatch.setattr(reminders_client, "create_reminder",
                        Mock(return_value={"id": "reminder-ficticio"}))
    monkeypatch.setattr(reminders_client, "complete_reminder", Mock(return_value=True))
    # conftest drops reminder jobs; run the real job against the fake client.
    monkeypatch.setattr(focus_mixin, "_in_background", lambda job, *args: job(*args))

    class Runtime:
        def __init__(self):
            self.project = tmp_path / "cortex_project"
            self.home = tmp_path / "home"
            self.client = None
            self.clock = clock
            self.model = model

        def close(self):
            if self.client is None:
                return
            self.client.close()
            for store in (self.projects, self.focus, self.chat):
                if store._conn is not None:
                    store._conn.close()
                    store._conn = None
            for eng in (self.orch.engine_analytical, self.orch.engine_creative):
                eng.memory._palace.close()
            self.orch.studies_review.knowledge_base._palace.close()
            self.client = None

        def boot(self, project=None, home=None):
            self.close()
            if project is not None:
                self.project, self.home = project, home
            (self.project / "data").mkdir(parents=True, exist_ok=True)
            self.home.mkdir(parents=True, exist_ok=True)
            monkeypatch.chdir(self.project)
            monkeypatch.setenv("HOME", str(self.home))
            sidecar = self.home / ".cortex"
            for key, filename in (("CORTEX_PROJECTS_DB", "projects.db"),
                                  ("CORTEX_FOCUS_DB", "focus.db"),
                                  ("CORTEX_CHAT_DB", "chat.db")):
                monkeypatch.setenv(key, str(sidecar / filename))
            self.projects = projects.ProjectStore()
            self.focus = focus.FocusStore()
            self.chat = chat_store.ChatStore()
            monkeypatch.setattr(projects, "store", self.projects)
            monkeypatch.setattr(focus, "store", self.focus)
            monkeypatch.setattr(chat_store, "store", self.chat)
            monkeypatch.setattr(routes, "chat_store", self.chat)
            monkeypatch.setattr(routes, "conversation_buffer", ConversationBuffer())
            palace_path = str(self.project / "data/palace.db")
            monkeypatch.setattr(Palace.__init__, "__defaults__",
                                (palace_path,) + Palace.__init__.__defaults__[1:])
            monkeypatch.setattr(CortexMemory, "DEFAULT_PALACE_DB", palace_path)
            monkeypatch.setattr(KnowledgeBase, "DEFAULT_PALACE_DB", palace_path)
            monkeypatch.setattr(LogicGate.__init__, "__defaults__",
                                (str(self.project / "data/gate_state.json"),))
            monkeypatch.setattr(orchestrator_module.CortexOrchestrator,
                                "_PENDING_ACTION_FILE", str(self.project / "data/pending_action.json"))
            # Execute the actual API source in a fresh namespace, leaving any
            # src.api imported by other tests untouched. No substitute app/router.
            spec = importlib.util.spec_from_file_location("src.api", ROOT / "src/api.py")
            self.api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.api)
            self.api.conversation_buffer = ConversationBuffer()
            self.orch = self.api.orchestrator
            self.client = TestClient(self.api.app)
            return self

        def conversation(self, title):
            response = self.client.post("/ui/conversations", headers=AUTH, json={"title": title})
            assert response.status_code == 200, response.text
            return response.json()["id"]

        def ui_response(self, cid, text):
            response = self.client.post("/ui/chat/stream", headers=AUTH,
                                        json={"text": text, "conversation_id": cid})
            assert response.status_code == 200, response.text
            return response

        def ui(self, cid, text):
            frames = sse(self.ui_response(cid, text).text)
            assert not any(f.get("type") == "error" for f in frames), frames
            assert frames[-1]["type"] == "final", frames
            return frames[-1]["turn"]

        def voice(self, text, endpoint="/ask_stream", mode=None):
            response = self.client.post(endpoint, headers=AUTH, json={"user_input": text})
            assert response.status_code == 200, response.text
            # Also verify the synchronous endpoint's mode and durable mirror.
            saved = self.chat._conn.execute(
                "SELECT user, source, status_mode FROM turns ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            assert tuple(saved[:2]) == (text, "voice")
            if mode:
                assert saved[2] == mode
            if endpoint == "/ask_voice":
                return response.text
            frames = sse(response.text)
            assert not any("error" in f for f in frames), frames
            assert frames[-1]["done"] is True
            if mode:
                assert frames[-1]["status_mode"] == mode
            return "".join(f.get("chunk", "") for f in frames)

    rt = Runtime().boot()
    try:
        yield rt
    finally:
        model.release.set()
        rt.close()


def sse(text):
    return [json.loads(line[5:].strip()) for line in text.splitlines() if line.startswith("data:")]


def curriculum(rt):
    """All project progress enters through authenticated production UI requests."""
    cid = rt.conversation("Conversa A fictícia")
    today = rt.clock.now
    rt.clock.now = today - timedelta(days=2)
    commands = (
        ("projeto: Currículo", "PROJECT_ACTIVE"),
        ("objetivo: candidatura da vaga X", "PROJECT_NOTED"),
        ("feito: rascunho em uma página", "PROJECT_NOTED"),
        ("falta: revisar a experiência", "PROJECT_NOTED"),
        ("falta: pedir a carta", "PROJECT_NOTED"),
        ("feito: revisar a experiência", "PROJECT_NOTED"),
        ("próximo: " + NEXT, "PROJECT_NOTED"),
    )
    for index, (text, mode) in enumerate(commands):
        if index == 5:
            rt.clock.now = today - timedelta(days=1)
        elif index == 6:
            rt.clock.now = today
        assert rt.ui(cid, text)["status_mode"] == mode
    assert len(rt.chat.get_turns(cid)) == len(commands)
    assert [e.text for e in rt.projects.entries(rt.projects.active().id, "falta", True)] == [
        "pedir a carta"]
    return cid


@pytest.mark.parametrize("endpoint", ["/ask_stream", "/ask_voice"])
def test_ui_restart_voice_resume_and_yes_starts_focus(runtime, endpoint):
    rt = runtime
    cid = curriculum(rt)
    old_app, old_orch = rt.api.app, rt.orch
    old_stores = rt.projects, rt.focus, rt.chat
    rt.boot()
    assert rt.api.app is not old_app and rt.orch is not old_orch
    assert all(new is not old for new, old in zip((rt.projects, rt.focus, rt.chat), old_stores))
    assert rt.orch.engine_analytical._global_history == []
    assert len(rt.chat.get_turns(cid)) == 7
    assert rt.projects.project_for_conversation(cid).name == "Currículo"
    # These lines end in sentence punctuation; speech joins them with spaces.
    expected = EXPECTED.replace("\n", " ") if endpoint == "/ask_voice" else EXPECTED
    assert rt.voice(QUESTION, endpoint, "PROJECT_WHERE") == expected
    reply = rt.voice("sim", endpoint, "FOCUS_STARTED")
    assert "Começa por: " + NEXT in reply
    session = rt.focus.current()
    assert (session.task, session.first_step, session.ends_at - session.started_at) == (
        "Currículo", NEXT, timedelta(minutes=25))
    assert rt.model.calls == []


def test_two_ui_conversations_and_voice_follow_last_ui_project(runtime):
    rt = runtime
    a = curriculum(rt)
    b = rt.conversation("Conversa B fictícia")
    for text in ("projeto: Casa", "feito: medir a parede", "próximo: escolher a tinta"):
        rt.ui(b, text)
    rt.boot()
    for cid, name, done, step, other in (
        (a, "Currículo", "revisar a experiência", NEXT, "Casa"),
        (b, "Casa", "medir a parede", "escolher a tinta", "Currículo"),
        (a, "Currículo", "revisar a experiência", NEXT, "Casa"),
    ):
        turn = rt.ui(cid, QUESTION)
        assert turn["status_mode"] == "PROJECT_WHERE"
        assert all(text in turn["reply"] for text in (name, done, step))
        assert other not in turn["reply"]
        assert rt.projects.active().name == name
        assert rt.voice(QUESTION, mode="PROJECT_WHERE") == turn["reply"]


@pytest.mark.parametrize("endpoint", ["/ask_stream", "/ask_voice"])
def test_voice_records_progress_with_voice_provenance(runtime, endpoint):
    rt = runtime
    curriculum(rt)
    rt.boot()
    assert rt.voice("feito: carta pedida", endpoint, "PROJECT_NOTED") == (
        'Anotado em “Currículo”: feito — carta pedida.')
    entry = rt.projects.entries(rt.projects.active().id, "feito")[0]
    assert (entry.text, entry.channel, entry.conversation_id, entry.source) == (
        "carta pedida", "voice", None, "user_said")
    rt.boot()
    assert 'Último feito: carta pedida (hoje às 15:15, pela voz).' in rt.voice(
        QUESTION, endpoint, "PROJECT_WHERE")


def test_cancelled_ui_turn_and_stale_operational_state_do_not_change_resume(runtime):
    rt = runtime
    cid = curriculum(rt)
    before = list(rt.projects._conn.iterdump())
    rt.model.allowed = rt.model.interrupt = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(rt.ui_response, cid, "Explique como escrever com clareza.")
        try:
            assert rt.model.entered.wait(10), "o LLM não chegou ao meio do stream"
            with routes._active_turns_lock:
                turn_id, = routes._active_turns
            cancelled = rt.client.post("/ui/chat/cancel", headers=AUTH, json={"id": turn_id})
            assert cancelled.status_code == 200
            assert cancelled.json() == {"cancelled": True}
        finally:
            rt.model.release.set()
        frames = sse(pending.result(timeout=10).text)
        assert not any(f.get("type") == "error" for f in frames), frames
        assert any(f.get("type") == "token" for f in frames), frames
    assert not routes._active_turns
    rt.model.allowed = False
    assert list(rt.projects._conn.iterdump()) == before
    # Persist an expired action on disk, then make boot exercise its real TTL loader.
    pending_file = rt.project / "data/pending_action.json"
    pending_file.write_text(json.dumps({
        "id": "acao-ficticia", "type": "send_email", "args": {"to": "x@example.test"},
        "staged_at": "2000-01-01T00:00:00+00:00", "origin": {"channel": "ui", "conversation_id": cid},
    }))
    rt.focus.start("Organizar livros fictícios", "separar caixas", 25,
                   rt.clock.now - timedelta(days=2))
    rt.boot()
    assert rt.orch._pending_action is None and not pending_file.exists()
    assert rt.voice(QUESTION, mode="PROJECT_WHERE") == EXPECTED


def test_without_project_voice_uses_today_focus(runtime):
    rt = runtime
    assert rt.voice(QUESTION, mode="FOCUS_WHERE") == (
        'Ainda não tem nenhum foco guardado. Pra começar: foco: <tarefa>.')
    rt.voice("foco: organizar livros fictícios", mode="FOCUS_ASK_STEP")
    rt.voice("separar caixas", mode="FOCUS_STARTED")
    rt.boot()
    assert rt.projects.list() == []
    assert rt.voice(QUESTION, mode="FOCUS_WHERE") == (
        'Você está em organizar livros fictícios até 15:40. O passo era: separar caixas.')


def test_current_project_reaches_real_llm_payload_only_on_relevant_voice_turn(runtime):
    rt = runtime
    cid = curriculum(rt)
    rt.boot()
    assert rt.projects.available()
    before = list(rt.projects._conn.iterdump())
    rt.model.allowed = True
    rt.ui(cid, "qual a capital da Austrália?")
    system = rt.model.system()
    assert "CURRENT PROJECT — notes Pedro wrote himself." in system
    assert "- Project: Currículo (goal: candidatura da vaga X)" in system
    assert "Next step (chosen by Pedro — this is THE next step): " + NEXT in system
    assert "- Last done: revisar a experiência" in system
    assert "- Still to do (not the next step): pedir a carta" in system
    assert "END OF CURRENT PROJECT" in system
    rt.voice("qual a capital da Austrália?")
    assert "CURRENT PROJECT" not in rt.model.system()
    rt.voice("me ajuda com o próximo passo")
    assert "CURRENT PROJECT — notes Pedro wrote himself." in rt.model.system()
    assert "- Project: Currículo" in rt.model.system()
    assert len([call for call in rt.model.calls if call[0] == "chat"]) == 3
    assert list(rt.projects._conn.iterdump()) == before
    for eng in (rt.orch.engine_analytical, rt.orch.engine_creative):
        assert "CURRENT PROJECT" not in eng.system_prompt["content"]


def test_real_backup_restore_preserves_voice_resume(runtime, tmp_path):
    import subprocess

    rt = runtime
    cid = curriculum(rt)
    expected = rt.voice(QUESTION, mode="PROJECT_WHERE")
    # A real Palace row and focus session also have to survive the scripts.
    rt.orch.engine_analytical.memory._palace.add_document("conversation", "Registro fictício")
    rt.focus.start("Livros fictícios", "caixas", 25, rt.clock.now - timedelta(days=2))
    backup = tmp_path / "backups"
    assert BACKUP_SCRIPT == ROOT / "scripts/backup.sh"
    result = _run_backup(rt.project, backup, home=rt.home)
    assert result.returncode == 0, result.stdout + result.stderr
    restored, restored_home = tmp_path / "restored", tmp_path / "restored_home"
    rt.close()
    result = subprocess.run(["bash", str(RESTORE_SCRIPT), str(_latest_snapshot(backup))],
                            env={**os.environ, "CORTEX_SRC_DIR": str(restored),
                                 "HOME": str(restored_home), "CORTEX_RESTORE_SKIP_RUNNING_CHECK": "1"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rt.boot(restored, restored_home)
    assert len(rt.chat.get_turns(cid)) == 7
    assert rt.projects.project_for_conversation(cid).name == "Currículo"
    assert rt.focus.last().task == "Livros fictícios"
    palace = rt.orch.engine_analytical.memory._palace
    assert palace._conn.execute("SELECT texto FROM documentos").fetchall()[0][0] == "Registro fictício"
    assert rt.voice(QUESTION, mode="PROJECT_WHERE") == expected == EXPECTED
