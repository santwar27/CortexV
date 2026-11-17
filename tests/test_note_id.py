"""Notes keep their id; undo never chooses a namesake or deletes from trash."""
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core import action_ledger as al
from src.integrations import notes_client as nc

NOTE_ID = "x-coredata://933F6320-BA85-4FD1-930A-598B23DF1AB2/ICNote/p598"


@pytest.fixture(autouse=True)
def run_calls(monkeypatch):
    calls = {"scripts": [], "stdout": ""}

    def fake_run(args, capture_output, text, timeout):
        calls["scripts"].append(args[2])
        return subprocess.CompletedProcess(args, 0, stdout=calls["stdout"], stderr="")

    monkeypatch.setattr("src.integrations.applescript.subprocess.run", fake_run)
    return calls


@pytest.fixture
def context(tmp_path, monkeypatch):
    from src.core.orchestrator import CortexOrchestrator
    lg = al.ActionLedger(path=str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr("src.core.notes_mixin._action_ledger", lg)
    monkeypatch.setattr(al, "ledger", lg)
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o._reply_lang = "pt"
    o._pending_action = None
    o.notes = SimpleNamespace(create_note=Mock(return_value={"id": NOTE_ID}))
    return o, lg


@pytest.mark.parametrize("folder", [None, "Trabalho"])
@pytest.mark.parametrize("stdout, expected", [
    (NOTE_ID, NOTE_ID), (f'note id "{NOTE_ID}" of application "Notes"', NOTE_ID),
    ("garbage", None), ("", None),
])
def test_create_returns_id(run_calls, folder, stdout, expected):
    run_calls["stdout"] = stdout
    assert nc.create_note("Work", folder=folder)["id"] == expected
    script = run_calls["scripts"][0]
    target = 'at folder "Trabalho" ' if folder else ""
    assert f"return id of (make new note {target}with properties" in script


def test_create_route_records_id(context):
    o, lg = context
    assert o._handle_note_create("Work\nreview")[1] == "NOTE_CREATE"
    [entry] = lg.load_entries()
    assert entry["args"] == {"title": "Work", "note_id": NOTE_ID}
    assert entry["reversible"] is True
    assert entry["reverse"] == {"tool": "delete_note", "args": {
        "note_id": NOTE_ID, "title": "Work"}}


@pytest.mark.parametrize("result", [Mock(), {}, {"id": Mock()},
                                        {"id": "garbage"}, {"id": NOTE_ID + "x"}])
def test_create_route_legacy_result(context, result):
    o, lg = context
    o.notes.create_note.return_value = result
    assert o._handle_note_create("Work")[1] == "NOTE_CREATE"
    assert lg.load_entries()[0]["args"] == {"title": "Work"}


@pytest.mark.parametrize("note_id", [None, NOTE_ID])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_delete_only_single_match_outside_trash(run_calls, note_id, count):
    run_calls["stdout"] = str(count)
    result = nc.delete_note("Work", note_id=note_id) if note_id else nc.delete_note("Work")
    assert result == count
    script = run_calls["scripts"][0]
    which = f'id is "{NOTE_ID}"' if note_id else 'name is "Work"'
    assert "repeat with f in folders" in script
    guard = ('if (name of f) is not "Recently Deleted" and '
             '(name of f) is not "Apagados Recentemente" then')
    assert script.index(guard) < script.index("count of (notes of f whose " + which)
    assert "if n is not 1 then return (n as text)" in script
    assert script.index("if n is not 1") < script.index("delete (")
    assert f'delete (notes of targetFolder whose {which})' in script
    assert "last item" not in script
    if note_id:
        assert "Work" not in script
    else:
        assert "name contains" not in script


@pytest.mark.parametrize("note_id", ['x" & do shell script "y', "", NOTE_ID + "x"])
def test_delete_rejects_malformed_id_without_script(run_calls, note_id):
    with pytest.raises(nc.NotesUnavailable):
        nc.delete_note("Work", note_id=note_id)
    assert run_calls["scripts"] == []


def test_undo_just_created_note_uses_id(context, monkeypatch):
    o, lg = context
    delete = Mock(return_value=1)
    monkeypatch.setattr(nc, "delete_note", delete)
    o._handle_note_create("Work")
    assert o._handle_undo()[1] == "UNDO_DONE"
    delete.assert_called_once_with("Work", note_id=NOTE_ID)
    assert lg.load_entries()[-1]["action"] == "undo"


@pytest.mark.parametrize("lang, reply", [
    ("pt", "Tem 2 notas 'Work' e não sei qual foi a minha — não apaguei nenhuma."),
    ("en", "There are 2 notes called 'Work' and I can't tell which one was mine — I deleted none."),
])
def test_legacy_undo_refuses_namesakes(context, monkeypatch, lang, reply):
    o, lg = context
    o._reply_lang = lang
    delete = Mock(return_value=2)
    monkeypatch.setattr(nc, "delete_note", delete)
    lg.record("create_note", {"title": "Work"})
    assert o._handle_undo() == (reply, "AGENT_ACTION_ERROR")
    delete.assert_called_once_with("Work")
    assert len(lg.load_entries()) == 1


def test_legacy_undo_keeps_one_positional_argument(context, monkeypatch):
    o, lg = context
    deleted = []
    monkeypatch.setattr(nc, "delete_note", lambda title: deleted.append(title) or 1)
    lg.record("create_note", {"title": "Work"})
    assert o._handle_undo()[1] == "UNDO_DONE"
    assert deleted == ["Work"]


def test_id_undo_already_gone_keeps_error_reply(context, monkeypatch):
    o, lg = context
    monkeypatch.setattr(nc, "delete_note", Mock(return_value=0))
    lg.record("create_note", {"title": "Work", "note_id": NOTE_ID})
    assert o._handle_undo() == (
        "Não achei a nota pra remover (talvez já tenha saído).", "AGENT_ACTION_ERROR")
    assert len(lg.load_entries()) == 1
