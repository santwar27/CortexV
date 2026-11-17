"""Explicit multi-file reads keep one confirmation and Pedro's question."""
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.exam_reader import requested_numbers
from tests.test_dispatch_parity import _build_real_orch


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def orch(tmp_path, home):
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    yield o
    for engine in (o.engine_analytical, o.engine_creative):
        engine.chat.assert_not_called()
        engine.chat_with_tools.assert_not_called()
        engine.chat_stream.assert_not_called()
    o.reader.read_path.assert_not_called()


@pytest.fixture
def files(home):
    folder = home / "Downloads" / "college" / "test"
    folder.mkdir(parents=True)
    paths = [folder / "questions1.pdf", folder / "text1.pdf"]
    for path in paths:
        path.write_bytes(b"%PDF-")
    return paths


def _assert_question(args, paths):
    assert args["paths"] == [os.path.realpath(p) for p in paths]
    question = args["question"]
    assert requested_numbers(question) == [5]
    assert question == " ".join(question.split())
    for path in paths:
        assert path.name not in question
        assert str(path) not in question


def _assert_staged(orch, paths):
    assert orch._pending_action["type"] == "read_files"
    _assert_question(orch._pending_action["args"], paths)
    assert all(orch._path_was_seen(str(p)) for p in paths)


@pytest.mark.parametrize("message", [
    "Read {p1} && {p2} and answer question 5",
    "Read {p1}\n&&\n{p2}\n\nAnd answer question 5",
    "lê {p1} e {p2} e responde a questão 5",
])
def test_one_confirmation_keeps_the_question(orch, files, message):
    reply, mode = orch._process_core(message.format(p1=files[0], p2=files[1]))
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    _assert_staged(orch, files)
    assert all(p.name in reply for p in files)


def test_no_read_intent_keeps_the_pick(orch, files):
    reply, mode = orch._process_core(f"{files[0]} e {files[1]}", language="pt")
    assert mode == "FILE_ERROR" and '"os dois"' in reply
    assert orch._pending_action is None
    assert not Path(orch._PENDING_ACTION_FILE).exists()
    _reply, mode = orch._process_core("os dois")
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action["type"] == "read_files"
    assert orch._pending_action["args"]["paths"] == [str(p) for p in files]


def test_four_files_still_ask_which(orch, files):
    for name in ("questions2.pdf", "text2.pdf"):
        path = files[0].parent / name
        path.write_bytes(b"%PDF-")
        files.append(path)
    reply, mode = orch._process_core("lê " + " e ".join(map(str, files)))
    assert mode == "FILE_ERROR" and "qual deles?" in reply
    assert all(p.name in reply for p in files)
    assert orch._pending_action is None
    assert not Path(orch._PENDING_ACTION_FILE).exists()


def test_unreadable_second_file_stages_nothing(orch, files):
    unreadable = files[0].parent / "notes.py"
    unreadable.write_text("# not a readable document")
    reply, mode = orch._process_core(f"lê {files[0]} e {unreadable}")
    assert mode == "FILE_ERROR"
    assert "não posso ler notes.py:" in reply
    assert orch._pending_action is None
    assert not Path(orch._PENDING_ACTION_FILE).exists()


def test_already_approved_runs_once_with_the_question(orch, files, monkeypatch):
    for path in files:
        orch._remember_read_approval(os.path.realpath(path))
    run = Mock(return_value=("answered question 5", "LOCAL_RAG"))
    monkeypatch.setattr(orch, "_run_read_files", run)
    reply, mode = orch._process_core(
        f"Read {files[0]} && {files[1]} and answer question 5")
    assert (reply, mode) == ("answered question 5", "LOCAL_RAG")
    run.assert_called_once()
    _assert_question(run.call_args.args[0], files)
    assert all(orch._path_was_seen(str(p)) for p in files)
    assert orch._pending_action is None
    assert not Path(orch._PENDING_ACTION_FILE).exists()


def test_stream_stages_without_the_pure_chat_shortcut(orch, files):
    text = f"Read {files[0]} && {files[1]} and answer question 5"
    *_, (reply, _final, mode, _lang) = orch.process_stream(text, None)
    assert mode == "AGENT_AWAITING_CONFIRMATION"
    _assert_staged(orch, files)
    assert all(p.name in reply for p in files)
