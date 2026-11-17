"""A lost Calendar reply is checked by approval identity, never inserted twice."""
import base64
import hashlib
import re
from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError
from httplib2 import Response

from src.core import action_ledger, effect_store, planner
from src.integrations import calendar_client as cal
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch

ARGS = {"summary": "Reunião", "start_iso": "2026-10-20T10:00:00.123+01:00",
        "end_iso": "2026-10-20T11:00:00+01:00"}


def http_error(status):
    return HttpError(Response({"status": str(status)}), b'{}')


class Calendar:
    def __init__(self, case="created"):
        self.case, self.inserts, self.gets = case, [], []
        self.lost = TimeoutError("lost reply")

    def events(self):
        return self

    def insert(self, **kwargs):
        self.inserts.append(kwargs)
        error = (self.lost if self.case.startswith("timeout") else
                 http_error(409) if self.case.startswith("conflict") else None)
        return Mock(execute=Mock(side_effect=error, return_value=self.event()))

    def event(self):
        return {"id": self.inserts[-1]["body"].get("id", "server-id"),
                "summary": "Other" if self.case.endswith("different") else ARGS["summary"],
                "start": {"dateTime": "2026-10-20T10:00:00+01:00"},
                "end": {"dateTime": "2026-10-20T11:00:00+01:00"}}

    def get(self, **kwargs):
        self.gets.append(kwargs)
        error = (http_error(404) if self.case.endswith("missing") else
                 http_error(500) if self.case.endswith("http_error") else
                 ConnectionError("check failed") if self.case.endswith("check_error") else None)
        return Mock(execute=Mock(side_effect=error, return_value=self.event()))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    monkeypatch.setattr(cal, "_local_iana_timezone", lambda: "Europe/London")
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.set_turn_origin("ui", "calendar")
    service = Calendar()
    monkeypatch.setattr(cal, "_service", lambda: service)
    yield o, store, service
    if store._conn is not None:
        store._conn.close()


def test_event_id_for():
    eid = cal.event_id_for("approval")
    expected = base64.b32hexencode(hashlib.sha256(b"approval").digest()).decode().lower().rstrip("=")
    assert eid == expected == cal.event_id_for("approval")
    assert len(eid) == 52 and re.fullmatch(r"[0-9a-v]{5,1024}", eid)
    assert eid != cal.event_id_for("another approval")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["created", "timeout_same", "timeout_missing",
                                 "timeout_different", "timeout_check_error",
                                 "conflict_same", "conflict_different", "conflict_missing",
                                 "conflict_check_error", "conflict_http_error"])
def test_inline_identity_and_outcome(rig, stream, lang, case):
    o, store, service = rig
    service.case = case
    o._pending_action = {"type": "create_event", "args": dict(ARGS)}
    o._persist_pending_action()
    eid = o._pending_action["id"]
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    success = case in {"created", "timeout_same", "conflict_same"}
    state = "feito" if success else "desconhecido"
    assert store.get(eid).state == state
    assert [h.to_state for h in store.history(eid)] == ["preparando", "executando", state]
    assert len(service.inserts) == 1
    body = service.inserts[0]["body"]
    assert body["id"] == cal.event_id_for(eid)
    assert body["start"] == {"dateTime": "2026-10-20T10:00:00.123", "timeZone": "Europe/London"}
    assert service.gets == ([] if case == "created" else [
        {"calendarId": "primary", "eventId": body["id"]}])
    entries = action_ledger.ledger.load_entries()
    if success:
        assert mode == "AGENT_ACTION_DONE"
        assert reply == ("Evento 'Reunião' criado." if lang == "pt" else "Event 'Reunião' created.")
        assert [(e["result_id"], e["source"]) for e in entries] == [(body["id"], "gate")]
    else:
        assert mode == "AGENT_ACTION_ERROR" and not entries
        if case.startswith("timeout"):
            assert reply.startswith("Não sei se deu certo (lost reply)" if lang == "pt" else
                                    "I don't know whether it went through (lost reply)")
        else:
            assert reply.startswith("Não sei se deu certo (" if lang == "pt" else
                                    "I don't know whether it went through (")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["created", "conflict_same", "conflict_different",
                                 "conflict_missing", "conflict_check_error", "conflict_http_error"])
def test_plan_step_identity(rig, monkeypatch, stream, lang, case):
    o, store, service = rig
    service.case = case
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "plan done")
    o._pending_action = {"type": "execute_plan", "args": {
        "objetivo": "Calendar", "lang": lang, "passos": [
            {"id": "p1", "tool": "create_event", "args": dict(ARGS),
             "depende_de": [], "descricao": "Calendar"}]}}
    o._persist_pending_action()
    eid = o._pending_action["id"] + ":p1"
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    success = case in {"created", "conflict_same"}
    state = "feito" if success else "desconhecido"
    assert store.get(eid).state == state
    assert [h.to_state for h in store.history(eid)] == ["preparando", "executando", state]
    assert len(service.inserts) == 1
    assert service.inserts[0]["body"]["id"] == cal.event_id_for(eid)
    assert service.gets == ([] if case == "created" else [
        {"calendarId": "primary", "eventId": cal.event_id_for(eid)}])
    if not success:
        assert mode == "AGENT_ACTION_ERROR"
        assert ("Não sei se o passo p1" if lang == "pt" else "I don't know whether step p1") in reply
        assert not action_ledger.ledger.load_entries()
        return
    assert mode == "AGENT_ACTION_DONE"
    [entry] = action_ledger.ledger.load_entries()
    assert (entry["result_id"], entry["source"]) == (cal.event_id_for(eid), "plan")


@pytest.mark.parametrize("plan", [False, True])
def test_generated_effect_identity(rig, monkeypatch, plan):
    o, store, service = rig
    o._reply_lang = "pt"
    insert = service.insert
    def check_intent(**kwargs):
        [effect] = store.in_states(["executando"])
        assert kwargs["body"]["id"] == cal.event_id_for(effect.id)
        return insert(**kwargs)
    monkeypatch.setattr(service, "insert", check_intent)
    call = o._execute_tool_unstaged if plan else o._run_action
    call("create_event", dict(ARGS))
    [entry] = action_ledger.ledger.load_entries()
    [effect] = store.in_states(["feito"])
    assert entry["result_id"] == cal.event_id_for(effect.id)


@pytest.mark.parametrize("plan", [False, True])
@pytest.mark.parametrize("effect_id", [None, ""])
def test_direct_call_without_effect_id(rig, plan, effect_id):
    o, _, service = rig
    o._reply_lang = "pt"
    call = o._execute_tool_unstaged_call if plan else o._run_action_call
    call("create_event", dict(ARGS), effect_id)
    assert len(service.inserts) == 1 and "id" not in service.inserts[0]["body"]
    assert not service.gets
    [entry] = action_ledger.ledger.load_entries()
    assert entry["result_id"] == "server-id"


@pytest.mark.parametrize("case", ["created", "timeout_same", "conflict_same"])
def test_recovered_result(rig, case):
    _, _, service = rig
    service.case = case
    result = cal.create_event(**ARGS, event_id="chosen")
    assert result["id"] == "chosen" and result["summary"] == ARGS["summary"]
    assert result["start"] == "2026-10-20T10:00:00+01:00"
    assert result.get("existing", False) == (case != "created")


@pytest.mark.parametrize("case", ["timeout_missing", "timeout_different", "timeout_check_error"])
def test_check_preserves_original_exception(rig, case):
    _, _, service = rig
    service.case = case
    with pytest.raises(TimeoutError) as caught:
        cal.create_event(**ARGS, event_id="chosen")
    assert caught.value is service.lost
    assert len(service.inserts) == len(service.gets) == 1


@pytest.mark.parametrize("field", ["summary", "start", "end"])
def test_conflict_compares_each_field(rig, monkeypatch, field):
    _, _, service = rig
    service.case = "conflict_same"
    event = service.event
    def different():
        result = event()
        result[field] = "Other" if field == "summary" else {"dateTime": "2026-10-20T12:00:00Z"}
        return result
    monkeypatch.setattr(service, "event", different)
    with pytest.raises(cal.CalendarUnavailable) as caught:
        cal.create_event(**ARGS, event_id="chosen")
    assert caught.value.uncertain is True
    assert len(service.inserts) == len(service.gets) == 1


@pytest.mark.parametrize("case", ["conflict_different", "conflict_missing",
                                 "conflict_check_error", "conflict_http_error"])
def test_unresolved_conflict_is_uncertain(rig, case):
    _, _, service = rig
    service.case = case
    with pytest.raises(cal.CalendarUnavailable) as caught:
        cal.create_event(**ARGS, event_id="chosen")
    assert caught.value.uncertain is True
    assert len(service.inserts) == len(service.gets) == 1


@pytest.mark.parametrize("status", [404, 410, 403, 500])
def test_find_event_errors(monkeypatch, status):
    service = Mock()
    service.events.return_value.get.return_value.execute.side_effect = http_error(status)
    monkeypatch.setattr(cal, "_service", lambda: service)
    if status in {404, 410}:
        assert cal.find_event("chosen") is None
    else:
        with pytest.raises(cal.CalendarUnavailable):
            cal.find_event("chosen")
    service.events.return_value.get.assert_called_once_with(calendarId="primary", eventId="chosen")


@pytest.mark.parametrize("case", ["created", "timeout_same", "conflict_same"])
def test_without_id_keeps_old_behavior(rig, case):
    _, _, service = rig
    service.case = case
    if case == "created":
        assert cal.create_event(**ARGS)["id"] == "server-id"
    else:
        with pytest.raises(TimeoutError if case.startswith("timeout") else cal.CalendarUnavailable):
            cal.create_event(**ARGS)
    assert len(service.inserts) == 1 and "id" not in service.inserts[0]["body"]
    assert not service.gets
