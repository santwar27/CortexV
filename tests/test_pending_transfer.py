"""An explicit transfer moves a voice preview, never its approval or clock."""
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import planner
from src.core.audit import audit
from src.core.staging_mixin import Delivery
from tests.test_pending_contract import AUTH, BODY, EMAIL, contract, disk
from tests.test_ui_routes import client, ui_app


def voice(o, text=EMAIL, delivery=None):
    o.set_turn_origin("voice")
    o.set_turn_delivery(delivery)
    try:
        return o.process(text, language="pt")
    finally:
        o.set_turn_origin(None)
        o.set_turn_delivery(None)


def transfer(client, action, cid, headers=AUTH):
    return client.post("/ui/pending/transfer", headers=headers, json={
        "pending_id": action["id"], "conversation_id": cid})


def pending(client, cid):
    return client.get("/ui/pending", headers=AUTH,
                      params={"conversation_id": cid}).json()["pending"]


@pytest.mark.parametrize("decision", ["sim", "não"])
def test_voice_preview_transfer_and_bound_decision(contract, client, decision):
    o, send, ids, ask = contract
    delivery = Delivery()
    delivery.taken.set()
    assert voice(o, delivery=delivery)[1] == "AGENT_AWAITING_CONFIRMATION"
    action = disk(o)
    before = pending(client, ids[0])
    assert before["here"] is False
    assert ask("sim", pending_id=action["id"])[-1]["turn"]["status_mode"] == "AGENT_ACTION_ELSEWHERE"
    result = transfer(client, action, ids[0])
    assert result.status_code == 200
    assert result.json() == {"transferred": True, "reason": "transferred",
                             "pending": pending(client, ids[0])}
    moved = disk(o)
    assert {k: moved[k] for k in ("id", "type", "args", "staged_at")} == {
        k: action[k] for k in ("id", "type", "args", "staged_at")}
    assert moved["origin"] == {"channel": "ui", "conversation": ids[0]}
    assert moved["transferred_from"] == action["origin"]
    assert datetime.fromisoformat(moved["transferred_at"]).utcoffset() == timedelta(0)
    assert "by" not in moved and o._pending_action == moved
    after = pending(client, ids[0])
    assert after["here"] is True and after["expires_at"] == before["expires_at"]
    assert after["preview"] == before["preview"] and BODY in after["preview"]
    assert o.withdraw_pending(delivery.id) is False and disk(o) == moved
    assert ask("sim", cid=ids[1], pending_id=action["id"])[-1]["turn"]["status_mode"] == "AGENT_ACTION_ELSEWHERE"
    send.assert_not_called()
    events = [json.loads(line) for line in Path(audit.path).read_text().splitlines()]
    event = next(e for e in events if e["event"] == "transferred")
    assert event["stage"] == "pending"
    assert event["payload"] == {"type": "send_email", "from": "voice", "to": "ui"}
    ask(decision, pending_id=action["id"])
    assert not Path(o._PENDING_ACTION_FILE).exists()
    if decision == "sim":
        send.assert_called_once()
        assert send.call_args.kwargs["body"] == BODY
    else:
        send.assert_not_called()
    assert transfer(client, action, ids[0]).json() == {
        "transferred": False, "reason": "stale", "pending": None}


@pytest.mark.parametrize("case,reason,status", [
    ("wrong_id", "stale", 200), ("expired", "expired", 200),
    ("ui", "not_voice", 200), ("terminal", "not_voice", 200),
    ("network", "not_voice", 200), (None, "not_voice", 200),
    ("already_here", "already_here", 200), ("missing_conversation", None, 404),
    ("no_key", None, 401),
])
def test_refusals_leave_disk_unchanged(contract, client, case, reason, status):
    o, send, ids, _ask = contract
    voice(o)
    action = disk(o)
    if case == "expired":
        action["staged_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    elif case in ("ui", "terminal", "network", None):
        action["origin"] = {"channel": case, "conversation": ids[1] if case == "ui" else None} if case else None
    path = Path(o._PENDING_ACTION_FILE)
    path.write_text(json.dumps(action))
    if case == "already_here":
        assert transfer(client, action, ids[0]).json()["reason"] == "transferred"
    before = path.read_bytes()
    result = transfer(client, {**action, "id": "wrong"} if case == "wrong_id" else action,
                      "missing" if case == "missing_conversation" else ids[0],
                      headers={} if case == "no_key" else AUTH)
    assert result.status_code == status
    if reason:
        assert result.json() == {"transferred": False, "reason": reason,
                                 "pending": pending(client, ids[0])}
    assert path.read_bytes() == before
    send.assert_not_called()


@pytest.mark.parametrize("field", ["pending_id", "conversation_id"])
@pytest.mark.parametrize("value", [None, 123, "x" * 65])
def test_transfer_payload_validation(contract, client, field, value):
    o, send, ids, _ask = contract
    voice(o)
    path = Path(o._PENDING_ACTION_FILE)
    before = path.read_bytes()
    payload = {"pending_id": disk(o)["id"], "conversation_id": ids[0], field: value}
    assert client.post("/ui/pending/transfer", headers=AUTH, json=payload).status_code == 422
    assert path.read_bytes() == before
    send.assert_not_called()


@pytest.mark.parametrize("failure", ["dump", "replace"])
def test_failed_write_preserves_record(contract, client, monkeypatch, failure):
    o, send, ids, _ask = contract
    voice(o)
    path = Path(o._PENDING_ACTION_FILE)
    before, memory = path.read_bytes(), o._pending_action.copy()
    with monkeypatch.context() as patch:
        patch.setattr("src.core.staging_mixin." + ("json.dump" if failure == "dump" else "os.replace"),
                      Mock(side_effect=OSError("fake write failure")))
        result = transfer(client, disk(o), ids[0])
    assert result.json() == {"transferred": False, "reason": "error",
                             "pending": pending(client, ids[0])}
    assert path.read_bytes() == before and o._pending_action == memory
    assert not Path(str(path) + ".tmp").exists()
    send.assert_not_called()


@pytest.mark.parametrize("race", ["transfer", "screen"])
@pytest.mark.parametrize("first", ["voice", "other"])
def test_transfer_and_approval_races(contract, client, monkeypatch, race, first):
    o, send, ids, _ask = contract
    for _ in range(3):
        voice(o)
        action = disk(o)
        if race == "screen":
            assert transfer(client, action, ids[0]).json()["reason"] == "transferred"
        barrier, held, attempted = threading.Barrier(2), threading.Event(), threading.Event()
        local = threading.local()
        original_lock, original_claim = o._pending_locked, o._claim_pending

        @contextmanager
        def ordered_lock():
            if getattr(local, "race", False):
                if threading.current_thread().name != first:
                    assert held.wait(5)
                    attempted.set()
                with original_lock():
                    if threading.current_thread().name == first:
                        held.set()
                        assert attempted.wait(5)
                    yield
            else:
                with original_lock():
                    yield

        def claim():
            barrier.wait(timeout=5)
            local.race = True
            return original_claim()

        def other():
            if race == "transfer":
                barrier.wait(timeout=5)
                local.race = True
                return o.transfer_pending_to_ui(action["id"], ids[0])[0]
            o.set_turn_origin("ui", ids[0])
            o.set_turn_pending_id(action["id"])
            return o.process("sim", language="pt")[1]

        results, errors = {}, []

        def run(name, fn):
            try:
                results[name] = fn()
            except BaseException as exc:
                errors.append(exc)

        with monkeypatch.context() as patch:
            patch.setattr(o, "_pending_locked", ordered_lock)
            patch.setattr(o, "_claim_pending", claim)
            threads = [threading.Thread(name=name, target=run, args=(name, fn)) for name, fn in (
                ("voice", lambda: voice(o, "sim")[1]), ("other", other))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(15)
                assert not thread.is_alive()
        assert not errors, errors
        assert send.call_count == 1
        assert not Path(o._PENDING_ACTION_FILE).exists()
        if race == "transfer":
            assert results["other"] == ("stale" if first == "voice" else "transferred")
        send.reset_mock()


def test_plan_keeps_whole_preview(contract, client, monkeypatch):
    o, send, ids, _ask = contract
    plan = planner.Plan(objetivo="Send complete email", passos=[planner.Step(
        id="p1", tool="send_email", args={"body": BODY}, depende_de=[], descricao="Send")])
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: plan)
    voice(o, "planeja: send the email")
    before, action = pending(client, ids[0]), disk(o)
    assert before["type"] == "execute_plan" and BODY in before["preview"]
    assert transfer(client, action, ids[0]).json()["reason"] == "transferred"
    assert pending(client, ids[0])["preview"] == before["preview"]
    send.assert_not_called()
