"""Effects link the executing reply and its preview without changing approval origins."""
import json
from pathlib import Path

import pytest

from src.core import effect_store, planner
from src.core.staging_mixin import Delivery
from src.ui.chat_store import ChatStore
from tests.test_claim_remove_failure import _turn
from tests.test_pending_contract import AUTH, contract
from tests.test_ui_routes import client, ui_app


@pytest.fixture
def linked(contract, tmp_path, monkeypatch):
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "Done")
    monkeypatch.setattr("ollama.chat", lambda *a, **kw: pytest.fail("Unexpected LLM"))
    yield (*contract, store)
    if store._conn is not None:
        store._conn.close()


def read(client, **params):
    response = client.get("/ui/effects", headers=AUTH, params=params)
    assert response.status_code == 200
    return response.json()


def seed(store, origin):
    effect = store.create("send_email", {}, origem=origin)
    assert store.transition(effect.id, "preparando", "executando")
    assert store.transition(effect.id, "executando", "desconhecido")
    return effect.id


@pytest.mark.parametrize("lost", [False, True])
def test_saved_reply_and_preview(linked, client, lost):
    o, send, (a, b), ask, store = linked
    preview = ask()
    action = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
    if lost:
        send.side_effect = RuntimeError("lost")
    approved = ask("sim", pending_id=action["id"])
    saved = client.get(f"/ui/conversations/{a}", headers=AUTH).json()["turns"]
    assert [t["id"] for t in saved] == [preview[0]["id"], approved[0]["id"]]
    assert saved == [preview[-1]["turn"], approved[-1]["turn"]]
    effect = store.get(action["id"])
    assert effect.state == ("desconhecido" if lost else "feito")
    assert effect.origin == {"channel": "ui", "conversation": a,
                             "turn": approved[0]["id"],
                             "asked": {"conversation": a, "turn": preview[0]["id"]}}
    response = read(client, conversation_id=a)
    item = next(e for e in response["effects"] if e["id"] == effect.id)
    assert item["turn"] == {"conversation_id": a, "turn_id": saved[1]["id"]}
    assert item["asked_turn"] == {"conversation_id": a, "turn_id": saved[0]["id"]}
    assert response["conversation"] == ([item] if lost else [])
    if lost:
        assert response["notices"] == [item]
        assert client.post("/ui/effects/seen", headers=AUTH,
                           json={"ids": [effect.id]}).json() == {"seen": [effect.id]}
        response = read(client, conversation_id=a)
        assert response["notices"] == []
        assert [e["id"] for e in response["conversation"]] == [effect.id]
    assert read(client, conversation_id=b)["conversation"] == []
    assert "conversation" not in read(client)
    assert client.get("/ui/effects", headers=AUTH,
                      params={"conversation_id": "x" * 65}).status_code == 422
    old = seed(store, {"channel": "ui", "conversation": a})
    old_item = next(e for e in read(client)["effects"] if e["id"] == old)
    assert old_item["turn"] is None and old_item["asked_turn"] is None
    assert old not in [e["id"] for e in read(client, conversation_id=a)["conversation"]]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("channel", ["ui", "terminal", None])
def test_note_origin(linked, stream, channel):
    o, _send, (a, _b), _ask, store = linked
    delivery = Delivery()
    o.set_turn_delivery(delivery)
    o.set_turn_origin(channel, a)
    o.notes.create_note.return_value = {"title": "Work"}
    assert _turn(o, "anota na pasta Office: Work\nreview", stream)[1] == "NOTE_CREATE"
    [effect] = store.in_states(["feito"])
    expected = None if channel is None else {"channel": channel, "conversation": None}
    if channel == "ui":
        expected = {"channel": "ui", "conversation": a, "turn": delivery.id}
    assert effect.origin == expected


@pytest.mark.parametrize("stream", [False, True])
def test_voice_approval_and_no_leak(linked, client, stream, monkeypatch):
    o, send, (a, _b), ask, store = linked
    preview = ask()
    action = json.loads(Path(o._PENDING_ACTION_FILE).read_text())
    o.set_turn_origin("voice")
    o.set_turn_delivery(Delivery())
    send.side_effect = RuntimeError("lost")
    assert _turn(o, "sim", stream)[1] == "AGENT_ACTION_ERROR"
    assert store.get(action["id"]).origin == {
        "channel": "voice", "conversation": None,
        "asked": {"conversation": a, "turn": preview[0]["id"]}}
    [item] = read(client, conversation_id=a)["conversation"]
    assert item["turn"] is None
    assert item["asked_turn"] == {"conversation_id": a, "turn_id": preview[0]["id"]}
    # A raised runner must also clear the claimed preview on this same thread.
    o._pending_action = action
    o._persist_pending_action()
    with monkeypatch.context() as patch:
        patch.setattr(o, "_run_action", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            o._execute_pending_action()
    o.set_turn_origin("terminal")
    o.set_turn_delivery(None)
    o.notes.create_note.return_value = {"title": "After"}
    assert _turn(o, "anota na pasta Office: After", stream)[1] == "NOTE_CREATE"
    assert store.in_states(["feito"])[0].origin == {"channel": "terminal", "conversation": None}


@pytest.mark.parametrize("stream", [False, True])
def test_plan_steps_share_links(linked, stream, monkeypatch):
    o, send, (a, _b), ask, store = linked
    args = {"to": "joao@example.com", "subject": "Meeting", "body": "Hello"}
    plan = planner.Plan(objetivo="Send two emails", passos=[planner.Step(
        id=f"p{i}", tool="send_email", args=args, depende_de=[], descricao="Send")
        for i in (1, 2)])
    monkeypatch.setattr(planner, "decompose", lambda *a, **kw: plan)
    monkeypatch.setattr(o, "_recipient_known", lambda _to: True)
    preview = ask("planeja: send two emails")
    delivery = Delivery()
    o.set_turn_origin("ui", a)
    o.set_turn_delivery(delivery)
    _turn(o, "sim", stream)
    effects = store.in_states(["feito"])
    assert len(effects) == send.call_count == 2
    assert all(e.origin == {"channel": "ui", "conversation": a, "turn": delivery.id,
                           "asked": {"conversation": a, "turn": preview[0]["id"]}}
               for e in effects)


def test_conversation_read_order_and_unavailable(linked, client, tmp_path, monkeypatch):
    _o, _send, (a, _b), _ask, store = linked
    first = seed(store, {"channel": "ui", "conversation": a, "turn": "first"})
    second = seed(store, {"asked": {"conversation": a, "turn": "preview"}})
    for origin in ({"channel": "ui", "conversation": a, "turn": 5},
                   {"asked": {"conversation": a, "turn": ""}},
                   {"channel": "voice", "conversation": a, "turn": "voice"}):
        seed(store, origin)
    # Equal timestamps exercise rowid ordering; old seen effects survive a recent-page limit.
    with store._conn:
        store._conn.execute("UPDATE effects SET created_at = '2000-01-01T00:00:00+00:00'")
    store.mark_seen([first, second])
    before = store._conn.total_changes
    assert [e["id"] for e in read(client, conversation_id=a)["conversation"]] == [first, second]
    assert [e["id"] for e in read(client, conversation_id=a, limit=1)["conversation"]] == [first]
    assert store._conn.total_changes == before
    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", lambda: None)
        assert store.unknown_for_conversation(a) == []
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    unavailable = effect_store.EffectStore(str(blocker / "effects.db"))
    monkeypatch.setattr(effect_store, "store", unavailable)
    assert unavailable.unknown_for_conversation(a) == []
    assert read(client, conversation_id=a) == {
        "available": False, "effects": [], "notices": [], "conversation": []}


@pytest.mark.parametrize("turn_id", ["abc123", "x", "x" * 64, None, "", "x" * 65, 5])
def test_append_turn_id(tmp_path, turn_id):
    store = ChatStore(str(tmp_path / "chat.db"))
    cid = store.create_conversation()["id"]
    turn = store.append_turn(cid, "user", "reply", turn_id=turn_id)
    if isinstance(turn_id, str) and 1 <= len(turn_id) <= 64:
        assert turn["id"] == turn_id
    else:
        assert len(turn["id"]) == 12
    assert store.get_turns(cid) == [turn]
