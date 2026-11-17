"""A lost Gmail reply permits one check by Message-ID, never another send."""
import base64
import hashlib
import re
from email import message_from_bytes
from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError
from httplib2 import Response

from src.core import action_ledger, effect_store, planner
from src.integrations import gmail_client as gmail
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch

ARGS = {"to": "ana@example.com", "subject": "Olá", "body": "Mensagem"}
CASES = ["sent", "timeout_found", "timeout_missing", "timeout_check_error",
         "timeout_http_error", "503_found", "503_missing", "503_check_error",
         "503_http_error", "400"]


def http_error(status):
    return HttpError(Response({"status": str(status)}), b'{}')


class Gmail:
    def __init__(self):
        self.case, self.sends, self.queries = "sent", [], []
        self.lost = TimeoutError("lost reply")

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, **kwargs):
        self.sends.append(kwargs)
        error = (self.lost if self.case.startswith("timeout") else
                 http_error(503) if self.case.startswith("503") else
                 http_error(400) if self.case == "400" else None)
        return Mock(execute=Mock(side_effect=error, return_value={"id": "sent-id"}))

    def list(self, **kwargs):
        self.queries.append(kwargs)
        error = (ConnectionError("check failed") if self.case.endswith("check_error") else
                 http_error(500) if self.case.endswith("http_error") else None)
        result = {"messages": [{"id": "sent-id", "threadId": "thread"}]} if self.case.endswith("found") else {}
        return Mock(execute=Mock(side_effect=error, return_value=result))

    def mime(self):
        [request] = self.sends
        assert request["userId"] == "me"
        return message_from_bytes(base64.urlsafe_b64decode(request["body"]["raw"]))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(effect_store, "store", store)
    o = _build_real_orch(tmp_path)
    o.background, o._offer = False, None
    o.set_turn_origin("ui", "gmail")
    o.remember_emails_in(ARGS["to"])
    service = Gmail()
    monkeypatch.setattr(gmail, "_service", lambda: service)
    yield o, store, service
    if store._conn is not None:
        store._conn.close()


def test_message_id_for():
    mid = gmail.message_id_for("approval")
    assert mid == gmail.message_id_for("approval")
    assert mid == f"<cortex.{hashlib.sha256(b'approval').hexdigest()[:32]}@cortex.local>"
    assert re.fullmatch(r"<[^<>\s@]+@[^<>\s@]+>", mid)
    assert mid != gmail.message_id_for("another approval")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("plan", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_approved_identity_and_outcome(rig, monkeypatch, stream, lang, plan, case):
    o, store, service = rig
    service.case = case
    monkeypatch.setattr(planner, "summarise", lambda *a, **kw: "plan done")
    o._pending_action = {"type": "execute_plan", "args": {
        "objetivo": "Email", "lang": lang, "passos": [
            {"id": "p1", "tool": "send_email", "args": dict(ARGS),
             "depende_de": [], "descricao": "Email"}]}} if plan else {
        "type": "send_email", "args": dict(ARGS)}
    o._persist_pending_action()
    eid = o._pending_action["id"] + (":p1" if plan else "")
    reply, mode = _turn(o, "sim" if lang == "pt" else "yes", stream, lang)
    success = case in {"sent", "timeout_found", "503_found"}
    state = "feito" if success else "falhou" if case == "400" else "desconhecido"
    assert store.get(eid).state == state
    assert [h.to_state for h in store.history(eid)] == ["preparando", "executando", state]
    mid = gmail.message_id_for(eid)
    mime = service.mime()  # Also asserts exactly one send.
    assert mime["Message-ID"] == mid
    assert mime["to"] == ARGS["to"]
    assert mime.get_payload(decode=True).decode("utf-8") == ARGS["body"]
    assert service.queries == ([] if case in {"sent", "400"} else [
        {"userId": "me", "q": f"rfc822msgid:{mid}", "maxResults": 1}])
    entries = action_ledger.ledger.load_entries()
    assert mode == ("AGENT_ACTION_DONE" if success else "AGENT_ACTION_ERROR")
    if success:
        [entry] = entries
        assert (entry["action"], entry["result_id"], entry["source"]) == (
            "send_email", "sent-id", "plan" if plan else "gate")
        if not plan:
            assert reply == ("Email enviado para ana@example.com (id sent-id)." if lang == "pt" else
                             "Email sent to ana@example.com (id sent-id).")
    else:
        assert not entries
        if state == "desconhecido":
            text = (("Não sei se o passo p1" if lang == "pt" else "I don't know whether step p1")
                    if plan else ("Não sei se deu certo (" if lang == "pt" else
                                  "I don't know whether it went through ("))
            assert text in reply
        elif not plan:
            assert reply.startswith("Falha ao enviar email:" if lang == "pt" else "Failed to send email:")


@pytest.mark.parametrize("case", ["sent", "timeout_found", "503_found"])
def test_send_result(rig, case):
    _, _, service = rig
    service.case = case
    result = gmail.send_email(**ARGS, message_id="<chosen@cortex.local>")
    expected = {"id": "sent-id", "to": ARGS["to"], "subject": ARGS["subject"]}
    if case != "sent":
        expected["confirmed"] = True
    assert result == expected
    assert service.mime()["Message-ID"] == "<chosen@cortex.local>"


@pytest.mark.parametrize("case", ["timeout_missing", "timeout_check_error", "timeout_http_error"])
def test_check_preserves_original_exception(rig, case):
    _, _, service = rig
    service.case = case
    with pytest.raises(TimeoutError) as caught:
        gmail.send_email(**ARGS, message_id="<chosen@cortex.local>")
    assert caught.value is service.lost
    assert len(service.sends) == len(service.queries) == 1


@pytest.mark.parametrize("case", ["503_missing", "503_check_error", "503_http_error"])
def test_unresolved_5xx_is_uncertain(rig, case):
    _, _, service = rig
    service.case = case
    with pytest.raises(gmail.GmailUnavailable) as caught:
        gmail.send_email(**ARGS, message_id="<chosen@cortex.local>")
    assert caught.value.uncertain is True
    assert isinstance(caught.value.__cause__, HttpError)
    assert len(service.sends) == len(service.queries) == 1


@pytest.mark.parametrize("case", ["timeout_found", "timeout_missing", "timeout_http_error", "timeout_check_error"])
def test_find_sent(rig, case):
    _, _, service = rig
    service.case = case
    if case.endswith("error"):
        with pytest.raises(gmail.GmailUnavailable if case.endswith("http_error") else ConnectionError):
            gmail.find_sent("<chosen@cortex.local>")
    else:
        assert gmail.find_sent("<chosen@cortex.local>") == (
            {"id": "sent-id"} if case.endswith("found") else None)
    assert service.queries == [{"userId": "me", "q": "rfc822msgid:<chosen@cortex.local>", "maxResults": 1}]
    assert not service.sends


@pytest.mark.parametrize("case", ["sent", "timeout_found", "503_found", "400"])
def test_without_id_keeps_old_behavior(rig, case):
    _, _, service = rig
    service.case = case
    if case == "sent":
        assert gmail.send_email(**ARGS) == {"id": "sent-id", "to": ARGS["to"], "subject": ARGS["subject"]}
    else:
        with pytest.raises(TimeoutError if case.startswith("timeout") else gmail.GmailUnavailable) as caught:
            gmail.send_email(**ARGS)
        assert not getattr(caught.value, "uncertain", False)
    assert service.mime()["Message-ID"] is None
    assert not service.queries
