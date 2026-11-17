"""121b: live plain chat, authoritative final text, real HTTP/storage paths."""
import json
import sqlite3

import pytest

from tests.test_file_capability_guard import REPLACEMENT, rig  # noqa: F401
from tests.test_engine_mac_hallucination import rig as mac_rig  # noqa: F401
from tests.test_taught_facts import orch  # noqa: F401
from tests.test_ui_routes import client, ui_app  # noqa: F401


QUESTION = "can you create files?"
DENIAL = "Here is the explanation. However, I cannot create files directly."
SAFE = "Here is the explanation. You can save the code using the command."
AUTH = {"X-Cortex-Key": "test-key-xyz"}


def parse_sse(body):
    return [json.loads(line[6:]) for line in body.splitlines()
            if line.startswith("data: ")]


def install_model(rig, reply):
    state = {"finished": False}

    def tokens(**kwargs):
        assert kwargs["stream"] is True
        for part in (reply[:24], reply[24:43], reply[43:]):
            yield {"message": {"content": part}}
        state["finished"] = True

    rig[1].side_effect = tokens
    return state


@pytest.mark.parametrize("reply", [SAFE, DENIAL])
def test_plain_chat_first_chunk_precedes_model_completion(rig, reply):
    state = install_model(rig, reply)
    stream = rig[0].chat_stream(QUESTION, "en")
    try:
        assert next(stream) == reply[:24]
        assert not state["finished"]
        list(stream)
        assert state["finished"]
    finally:
        stream.close()


def test_corrected_stream_delivery_outcome_matches_final(rig, orch):
    from src.core.staging_mixin import Delivery

    install_model(rig, DENIAL)
    orch.engine_analytical = orch.engine_creative = rig[0]
    delivery = Delivery()
    orch.set_turn_delivery(delivery)
    try:
        frames = list(orch.process_stream(QUESTION, language="en"))
    finally:
        orch.set_turn_delivery(None)
    assert frames[-1][0] == REPLACEMENT["en"]
    assert frames[-1][1] is True
    assert delivery.finished.is_set()
    assert delivery.outcome == (REPLACEMENT["en"], "OFFLINE")


def test_correction_preserves_spoken_teaching_receipt(rig, orch, monkeypatch):
    from src.core.taught_facts import KeepResult

    taught = KeepResult("kept", "Cortex can save files", "en")
    monkeypatch.setattr("src.core.orchestrator.keep_taught_fact", lambda *a: taught)
    install_model(rig, DENIAL)
    orch.engine_analytical = orch.engine_creative = rig[0]
    frames = list(orch.process_stream(QUESTION, language="en"))
    suffix = "\n\n" + taught.receipt
    assert "".join(f[0] for f in frames[:-1]) == DENIAL + suffix
    assert frames[-1] == (REPLACEMENT["en"] + suffix, True, "OFFLINE", "en")


@pytest.mark.parametrize("reply,expected", [(DENIAL, REPLACEMENT["en"]), (SAFE, SAFE)])
def test_ui_live_tokens_and_authoritative_final_in_chat_db(
        rig, orch, ui_app, client, reply, expected):
    from src.ui import routes

    install_model(rig, reply)
    orch.engine_analytical = orch.engine_creative = rig[0]
    ui_app.state.orchestrator = orch
    cid = client.post("/ui/conversations", headers=AUTH, json={}).json()["id"]
    response = client.post("/ui/chat/stream", headers=AUTH, json={
        "text": QUESTION, "language": "en", "conversation_id": cid,
    })
    assert response.status_code == 200
    frames = parse_sse(response.text)
    final = frames[-1]
    assert final["type"] == "final"
    assert final["turn"]["reply"] == expected
    assert final["turn"]["status_mode"] == "OFFLINE"
    # Read the actual durable row, not just the HTTP echo or an in-memory stub.
    with sqlite3.connect(routes.chat_store.db_path) as db:
        assert db.execute("SELECT reply FROM turns WHERE conversation_id = ?",
                          (cid,)).fetchall() == [(expected,)]
    assert "".join(f["text"] for f in frames if f["type"] == "token") == reply
    rig[0].memory.store_interaction.assert_called_once_with(QUESTION, expected)


@pytest.mark.parametrize("reply,expected", [(DENIAL, REPLACEMENT["en"]), (SAFE, SAFE)])
def test_voice_live_tokens_and_corrected_final_text(rig, orch, monkeypatch, reply, expected):
    from fastapi.testclient import TestClient
    import src.api as api
    from src.ui.conversation import buffer

    install_model(rig, reply)
    orch.engine_analytical = orch.engine_creative = rig[0]
    monkeypatch.setattr(api, "orchestrator", orch)
    monkeypatch.setattr(api, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api, "_ALLOWLIST", [])
    response = TestClient(api.app).post("/ask_stream", headers=AUTH, json={
        "user_input": QUESTION, "language": "en",
    })
    assert response.status_code == 200
    frames = parse_sse(response.text)
    assert frames[-1]["done"] is True
    if reply == DENIAL:
        assert frames[-1]["reply"] == expected
    assert "".join(f["chunk"] for f in frames if "chunk" in f) == reply
    assert buffer.snapshot()[-1]["reply"] == expected
