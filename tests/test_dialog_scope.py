"""Questions and permissions stay with their conversation, including stream workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.orchestrator import CortexOrchestrator
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch
from tests.test_email_recipient import ASK, INVENTED
from tests.test_pending_identity import REMINDER_ASK, clock


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    o = _build_real_orch(tmp_path)
    o.background = False
    monkeypatch.setattr(o, "_needs_realtime", lambda *_: False)
    monkeypatch.setattr(o, "_semantic_route_live", lambda *_: None)
    monkeypatch.setattr("src.integrations.contacts.contact_book.list_all", lambda: [])
    o.engine_analytical.chat_stream.return_value = ["resposta"]
    o._synthesise = Mock(return_value=("resumo", "LOCAL_RAG"))
    o.reader.read_path.return_value = "conteúdo"
    return o


def turn(o, text, stream, channel="ui", conversation="A"):
    o.set_turn_origin(channel, conversation)
    return _turn(o, text, stream)


@pytest.mark.parametrize("stream", [False, True])
def test_file_question_survives_other_conversation(orch, tmp_path, monkeypatch, stream):
    paths = [str(tmp_path / name) for name in ("cv.txt", "cv-old.txt")]
    for path in paths:
        Path(path).write_text("cv")
    hits = [SimpleNamespace(path=p, name=p.rsplit("/", 1)[-1], modified="")
            for p in paths]
    monkeypatch.setattr("src.core.named_file.find", lambda *a, **kw: (hits, None))
    assert turn(orch, "cadê meu currículo", stream)[1] == "FILE_LIST"
    assert turn(orch, "o primeiro", stream, conversation="B")[1] != "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action is None
    assert turn(orch, "o primeiro", stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert orch._pending_action["args"]["path"] == paths[0]


@pytest.mark.parametrize("stream", [False, True])
def test_search_is_not_confirmed_by_another_channel(orch, monkeypatch, stream):
    search = Mock(return_value=("resultado", "AUTO_SEARCH"))
    monkeypatch.setattr(orch, "_do_web_search", search)
    assert turn(orch, "pesquise preço do ouro", stream)[1] == "AWAITING_APPROVAL"
    assert turn(orch, "Ah!", stream)[1] == "AWAITING_APPROVAL"
    assert orch._pending_query_reprompted is True
    for channel, conversation in (("voice", None), ("network", None), ("ui", "B")):
        assert turn(orch, "sim", stream, channel, conversation)[1] == "NOTHING_PENDING"
        search.assert_not_called()
    assert turn(orch, "sim", stream)[1] == "AUTO_SEARCH"
    search.assert_called_once_with(orch.engine_analytical, "pesquise preço do ouro", "pt")


@pytest.mark.parametrize("stream", [False, True])
def test_seen_paths_and_read_permission_are_local(orch, tmp_path, stream):
    path = tmp_path / "cv.txt"
    path.write_text("cv")
    assert turn(orch, f"lê {path}", stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert turn(orch, "sim", stream)[1] == "LOCAL_RAG"
    assert orch._read_file_approved(str(path))
    turn(orch, "olá", stream, conversation="B")
    assert not orch._path_was_seen(str(path))
    assert orch.readable_candidates() == []
    assert not orch._read_file_approved(str(path))
    assert "não tenho esse caminho" in orch._execute_tool("read_file_at", {"path": str(path)})
    assert orch._pending_action is None
    assert turn(orch, f"lê {path}", stream, conversation="B")[1] == "AGENT_AWAITING_CONFIRMATION"
    assert turn(orch, "não", stream, conversation="B")[1] == "AGENT_ACTION_CANCELLED"
    assert turn(orch, f"lê {path}", stream)[1] == "LOCAL_RAG"


@pytest.mark.parametrize("stream", [False, True])
def test_seen_email_does_not_authorize_another_conversation(orch, stream):
    turn(orch, "o endereço é joao@example.com", stream)

    def draft(text, specs, execute, **kw):
        return execute("send_email", INVENTED), "send_email"

    orch.engine_analytical.chat_with_tools.side_effect = draft
    reply, _ = turn(orch, ASK, stream, conversation="B")
    assert orch._pending_action is None
    assert "qual é?" in reply
    assert orch._offer["kind"] == "email_address"
    assert not orch._recipient_known(INVENTED["to"])
    orch.set_turn_origin("ui", "A")
    assert orch._recipient_known(INVENTED["to"])


@pytest.mark.parametrize("stream", [False, True])
def test_ui_preview_can_still_be_confirmed_by_voice(orch, monkeypatch, clock, stream):
    run = Mock(return_value=("feito", "AGENT_ACTION_DONE"))
    monkeypatch.setattr(orch, "_run_action", run)
    assert turn(orch, REMINDER_ASK, stream)[1] == "AGENT_AWAITING_CONFIRMATION"
    assert turn(orch, "sim", stream, "voice", None) == ("feito", "AGENT_ACTION_DONE")
    run.assert_called_once()


@pytest.mark.parametrize("stream", [False, True])
def test_no_origin_keeps_one_shared_scope(orch, monkeypatch, stream):
    orch.remember_emails_in("joao@example.com")
    orch._remember_seen_path("/example.txt")
    orch._remember_read_approval("/example.txt")
    search = Mock(return_value=("resultado", "AUTO_SEARCH"))
    monkeypatch.setattr(orch, "_do_web_search", search)
    assert turn(orch, "pesquise preço do ouro", stream, None)[1] == "AWAITING_APPROVAL"
    assert turn(orch, "sim", stream, None)[1] == "AUTO_SEARCH"
    assert orch._recipient_known("joao@example.com")
    assert orch._path_was_seen("/example.txt") and orch._read_file_approved("/example.txt")
    assert orch._pending_query is None


DEFAULTS = {
    "_offer": None, "_pending_query": None, "_pending_engine": None,
    "_pending_query_reprompted": False, "_seen_paths": set(), "_seen_order": [],
    "_approved_read_paths": set(), "_seen_emails": set(), "_last_user_text": "",
    "_last_listed": None, "_last_exam": None,
}


def test_new_defaults_mutation_and_lru():
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    for name, default in DEFAULTS.items():
        assert getattr(o, name) == default
    o._offer = {"kind": "default"}
    o.set_turn_origin("ui", "oldest")
    for name in DEFAULTS:
        value = ({"saved"} if isinstance(DEFAULTS[name], set)
                 else ["saved"] if isinstance(DEFAULTS[name], list) else "saved")
        setattr(o, name, value)
        assert getattr(o, name) == value
    o.set_turn_origin("ui", "recent")
    for name, default in DEFAULTS.items():
        assert getattr(o, name) == default
    o._seen_paths.add("recent")
    o._seen_order.append("recent")
    o._offer = {"kind": "recent"}
    for i in range(63):
        o.set_turn_origin("ui", str(i))
        assert o._offer is None
        if i == 62:
            o.set_turn_origin("ui", "recent")  # Reads refresh recency too.
            assert o._offer == {"kind": "recent"}
    o.set_turn_origin("ui", "oldest")
    for name, default in DEFAULTS.items():
        assert getattr(o, name) == default
    o.set_turn_origin("ui", "recent")
    assert o._seen_paths == {"recent"} and o._seen_order == ["recent"]
    o.set_turn_origin(None)
    assert o._offer == {"kind": "default"}
    assert len(o._dialog_scopes) <= 65


def test_concurrent_scope_creation_on_new_instance():
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o.set_turn_origin(None)

    def remember(i):
        o.set_turn_origin("ui", str(i))
        o._seen_paths.add(str(i))
        o._seen_order.append(str(i))
        return o._seen_paths, o._seen_order

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(remember, range(32))) == [({str(i)}, [str(i)]) for i in range(32)]
