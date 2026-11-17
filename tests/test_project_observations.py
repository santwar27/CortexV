"""Tool observations are durable context, never progress or permission."""
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core import planner, projects
from src.core.action_ledger import ledger
from tests.test_project_routes import store, turn
from tests.test_project_where import setup
from tests.test_backup_script import (fake_cortex, backup_dir, fake_home,
                                     TestTheRestoreGoesBackWhereTheAppReads as _BackupRoundtrip)


@pytest.fixture
def env(setup, store, tmp_path, monkeypatch):
    make, clock = setup
    o = make()
    p = store.create_or_get("Currículo")
    store.link_conversation("A", p.id)
    store.set_active(p.id)
    path = str(tmp_path / "cv.txt")
    monkeypatch.setattr("src.tools.file_finder.is_readable_path",
                        lambda raw: (raw == path, path))
    o.reader.read_path.return_value = "conteúdo fictício"
    o._synthesise = Mock(return_value=("resumo", "LOCAL_RAG"))
    monkeypatch.setattr("src.integrations.gmail_client.send_email", Mock(
        return_value={"id": "fake", "to": "x@example.test"}))
    o.set_turn_origin("ui", "A")
    o._remember_seen_path(path)
    o._remember_read_approval(path)
    o.remember_emails_in("x@example.test")
    return o, p, path, make, clock


def email(o):
    o._pending_action = {"type": "send_email", "args": {
        "to": "x@example.test", "subject": "teste", "body": "fictício"}}
    o._persist_pending_action()
    return o._execute_pending_action()


@pytest.mark.parametrize("entry", ["process", "stream", "worker"])
def test_success_through_entrypoints(env, store, monkeypatch, entry):
    o, p, path, _, _ = env
    threads = []

    def execute(*a, **kw):
        threads.append(threading.get_ident())
        assert o._turn_project().id == p.id
        assert o._execute_tool("read_file_at", {"path": path}) == "resumo"
        o._execute_tool("write_document", {"filename": "cv", "content": "teste"})
        return email(o)

    monkeypatch.setattr(o, "_process_core", execute)
    if entry == "stream":
        def stream(*a, **kw):
            reply, mode = execute()
            yield reply, True, mode, "pt"
        monkeypatch.setattr(o, "_process_stream_impl", stream)
    elif entry == "worker":
        monkeypatch.setattr(o, "_is_pure_chat", lambda *_: False)
    turn(o, "sim", stream=entry != "process")
    rows = store.observations(p.id)
    assert [(r.kind, r.tool, r.reference, r.channel, r.conversation_id)
            for r in reversed(rows)] == [
        ("read", "read_file_at", path, "ui", "A"),
        ("written", "write_document", "cv.txt", "ui", "A"),
        ("action", "send_email", "email para x@example.test", "ui", "A")]
    assert all(r.created_at.utcoffset() == timedelta(0) for r in rows)
    assert store.entries(p.id) == []
    assert o._turn_project() is None
    assert (threads[0] != threading.get_ident()) == (entry == "worker")


@pytest.mark.parametrize("channel,conversation,text,expected", [
    ("ui", "B", "projeto Currículo", 0), ("voice", None, "sim", 0),
    ("voice", None, "no Currículo", 1), ("terminal", None, "próximo passo", 1)])
def test_turn_selection(env, store, channel, conversation, text, expected):
    o, p, _, _, _ = env
    o.set_turn_origin(channel, conversation)
    with o._project_scope(text):
        o._execute_tool("write_document", {"filename": "cv", "content": "x"})
    rows = store.observations(p.id)
    assert len(rows) == expected
    if rows:
        assert rows[0].channel == channel and rows[0].conversation_id is None


def test_failures_and_pending(env, store, monkeypatch):
    o, p, path, _, _ = env
    with o._project_scope("sim"):
        o.reader.read_path.return_value = "[SYSTEM ERROR] erro de leitura"
        assert o._run_read_file_at({"path": path})[1] == "FILE_ERROR"
        o._remember_seen_path(path + "missing")
        assert o._run_read_file_at({"path": path + "missing"})[1] == "AGENT_ACTION_ERROR"
        o._approved_read_paths.clear()
        assert "CONFIRMATION REQUIRED" in o._execute_tool("read_file_at", {"path": path})
        o.reader.read_document.return_value = "[SYSTEM ERROR] not found"
        o.writer.write_document.return_value = "[SYSTEM ERROR] refused"
        for run in (o._execute_tool, o._execute_tool_unstaged):
            run("read_document", {"filename": "missing.txt"})
            run("write_document", {"filename": "cv", "content": "x"})
        monkeypatch.setattr("src.integrations.gmail_client.send_email",
                            Mock(side_effect=OSError("failed")))
        assert email(o)[1] == "AGENT_ACTION_ERROR"
        ledger.record("send_email", {"to": "failed"}, status="error")
    assert store.observations(p.id) == []


def test_documents_multiread_pdf_and_plan(env, store, monkeypatch):
    o, p, path, _, _ = env
    monkeypatch.setattr("src.tools.file_finder.folder_gate", lambda _: (True, ""))
    monkeypatch.setattr("src.tools.file_finder.is_readable_path", lambda raw: (True, raw))
    monkeypatch.setattr("src.tools.pdf_answer_writer.write_answers", Mock(
        return_value=SimpleNamespace(path=path + ".pdf", written=[1], skipped=[], crosses=0)))
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "plano feito")
    with o._project_scope("sim"):
        o._run_read_files({"paths": [path]})
        o._remember_seen_path(path + ".paper.pdf")
        o._run_write_answers_pdf({"paper": path + ".paper.pdf", "out": path + ".pdf",
                                  "answers": {"1": "teste"}})
        o._execute_tool("read_document", {"filename": "doc.txt"})
        o._pending_action = {"type": "execute_plan", "args": {"passos": [
            {"id": "p1", "tool": "read_document", "args": {"filename": "doc.txt"}},
            {"id": "p2", "tool": "write_document", "args": {"filename": "cv", "content": "x"}},
            {"id": "p3", "tool": "send_email", "args": {"to": "x@example.test"}}]}}
        o._persist_pending_action()
        assert o._execute_pending_action()[1] == "AGENT_ACTION_DONE"
        o._handle_note_create("nota fictícia")
        o._handle_note_append("nota fictícia", "mais")
    assert [r.tool for r in reversed(store.observations(p.id))] == [
        "read_files", "write_answers_pdf", "read_document", "read_document",
        "write_document", "send_email", "create_note", "append_to_note"]


@pytest.mark.parametrize("en", [False, True])
def test_where_and_unchanged_context(env, store, en):
    o, p, _, _, clock = env
    question = "where was I?" if en else "onde a gente parou?"
    before = turn(o, question)
    with o._project_scope("sim"):
        context = o._project_context("sim")
    for n in range(5):
        store.observe(p.id, "read", "read_document", f"{n}.txt", "ui", "A")
        store.observe(p.id, "action", "create_reminder", f"lembrete {n}", "ui", "A")
    store.observe(p.id, "written", "write_document", "4.txt", "ui", "A")
    store._conn.execute("UPDATE observations SET created_at = ? WHERE kind = 'action'",
                        ((clock() - timedelta(days=1)).astimezone(timezone.utc).isoformat(),))
    store._conn.commit()
    reply, mode = turn(o, question)
    assert mode == before[1] == "PROJECT_WHERE"
    assert reply.splitlines()[0] == before[0]
    assert reply.splitlines()[1] == ("Files: 4.txt (written), 3.txt (read), 2.txt (read)." if en else
                                     "Arquivos: 4.txt (gravado), 3.txt (lido), 2.txt (lido).")
    actions = reply.splitlines()[2]
    assert actions.startswith("Cortex did: " if en else "O Cortex fez: ")
    assert actions.count("yesterday at" if en else "ontem às") == 3
    assert ("reminder 4" if en else "lembrete 4") in actions
    assert "lembrete 0" not in actions and "lembrete 1" not in actions
    with o._project_scope("sim"):
        assert o._project_context("sim") == context
    assert store.entries(p.id) == []


def test_observation_is_not_permission_and_broken_store(env, store, monkeypatch):
    o, p, path, make, _ = env
    store.observe(p.id, "read", "read_file_at", path, "ui", "A")
    fresh = make()
    assert not fresh._path_was_seen(path) and not fresh._read_file_approved(path)
    fresh._remember_seen_path(path)
    assert "CONFIRMATION REQUIRED" in fresh._execute_tool("read_file_at", {"path": path})
    fresh.remember_emails_in("x@example.test")
    assert "CONFIRMATION REQUIRED" in fresh._execute_tool("send_email", {"to": "x@example.test"})
    monkeypatch.setattr(store, "observe", Mock(side_effect=OSError("broken")))
    with o._project_scope("sim"):
        assert o._run_read_file_at({"path": path}) == ("resumo", "LOCAL_RAG")
        assert email(o)[1] == "AGENT_ACTION_DONE"
    assert len(store.observations(p.id)) == 1
    baseline = turn(o, "onde a gente parou?")[0].splitlines()[0]
    monkeypatch.setattr(store, "observations", Mock(side_effect=OSError("broken")))
    assert turn(o, "onde a gente parou?") == (baseline, "PROJECT_WHERE")


def test_store_validation_limit_and_failure(env, store, tmp_path):
    _, p, _, _, _ = env
    assert store.observe(p.id, "read", "read_document", "x" * 200).reference == "x" * 150
    assert store.observe(p.id, "other", "tool", "x") is None
    assert store.observe(p.id, "read", "tool", " ") is None
    assert store.observations(p.id, 0) == []
    broken = projects.ProjectStore(str(tmp_path))
    assert broken.observe(p.id, "read", "tool", "x") is None
    assert broken.observations(p.id) == []


def test_thread_isolation_and_ledger_failure(env, store, monkeypatch, tmp_path):
    o, p, _, make, _ = env
    other = store.create_or_get("Casa")
    store.link_conversation("B", other.id)
    barrier = threading.Barrier(2)
    monkeypatch.setattr(ledger, "path", str(tmp_path))  # Ledger I/O fails, action succeeded.

    def run(conversation):
        o.set_turn_origin("ui", conversation)
        with o._project_scope("sim"):
            barrier.wait(timeout=5)
            ledger.record("create_note", {"title": conversation})
        assert o._turn_project() is None
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, ["A", "B"]))
    assert store.observations(p.id)[0].reference == "nota A"
    assert store.observations(other.id)[0].reference == "nota B"
    ledger.record("create_note", {"title": "fora do turno"})
    assert len(store.observations(p.id)) == 1


def test_existing_backup_roundtrip_includes_observations(
        fake_cortex, backup_dir, fake_home, tmp_path, monkeypatch):
    add = projects.ProjectStore.add_entry

    def observed(s, projeto, *a, **kw):
        s.observe(projeto, "read", "read_document", "horta.txt", "ui", "c1")
        return add(s, projeto, *a, **kw)
    monkeypatch.setattr(projects.ProjectStore, "add_entry", observed)
    _BackupRoundtrip().test_projects_survive_a_backup_and_restore(
        fake_cortex, backup_dir, fake_home, tmp_path)
    restored = projects.ProjectStore(str(tmp_path / "restored_home/.cortex/projects.db"))
    assert restored.observations(restored.active().id)[0].reference == "horta.txt"
