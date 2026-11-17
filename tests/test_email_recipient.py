"""A recipient the 8B made up never reaches "sim" (s85, fila 4.1).

Measured: "manda um email pro João dizendo que vou atrasar" was staged to
joao@example.com 6 times in 6 — there is no João in the contacts. Same rule
as paths: the address must have been written by Pedro or be in the
contacts. Otherwise Cortex asks, keeps the draft, and the reply that brings
the address is staged through the same gate.
"""
import pytest

from src.core.engine import SKIP_FOLLOWUP_MARKER
from src.core.orchestrator import CortexOrchestrator

ASK = "manda um email pro João dizendo que vou atrasar"
INVENTED = {"to": "joao@example.com", "subject": "Atraso", "body": "Vou atrasar."}


@pytest.fixture
def orch(tmp_path, monkeypatch):
    from tests.test_dispatch_parity import _build_real_orch
    o = _build_real_orch(tmp_path)
    o._PENDING_ACTION_FILE = str(tmp_path / "pending.json")
    o.background = False
    o._reply_lang = "pt"
    monkeypatch.setattr("src.integrations.contacts.contact_book.list_all",
                        lambda: [])
    return o


class TestTheGate:

    def test_an_invented_recipient_is_asked_for_not_staged(self, orch):
        orch.remember_emails_in(ASK)
        reply = orch._execute_tool("send_email", INVENTED)
        assert orch._pending_action is None
        assert reply == SKIP_FOLLOWUP_MARKER + (
            "Não tenho o email do João nos seus contatos — qual é?")
        assert orch._offer == {"kind": "email_address",
                               "args": {"subject": "Atraso", "body": "Vou atrasar."}}

    def test_an_address_pedro_wrote_is_staged(self, orch):
        orch.remember_emails_in("manda email pra joao.silva@gmail.com dizendo oi")
        orch._execute_tool("send_email", {**INVENTED, "to": "joao.silva@gmail.com"})
        assert orch._pending_action["type"] == "send_email"

    def test_an_address_in_the_contacts_is_staged(self, orch, monkeypatch):
        contact = type("C", (), {"email": "Joao@Firma.com"})()
        monkeypatch.setattr("src.integrations.contacts.contact_book.list_all",
                            lambda: [contact])
        orch.remember_emails_in(ASK)
        orch._execute_tool("send_email", {**INVENTED, "to": "joao@firma.com"})
        assert orch._pending_action["type"] == "send_email"

    def test_without_a_name_the_question_is_generic(self, orch):
        orch.remember_emails_in("manda aquele email de ontem de novo")
        reply = orch._execute_tool("send_email", INVENTED)
        assert "me passa o endereço" in reply

    def test_inside_an_approved_plan_nothing_is_mailed(self, orch, monkeypatch):
        sent = []
        monkeypatch.setattr("src.integrations.gmail_client.send_email",
                            lambda **kw: sent.append(kw) or {"id": "x"})
        out = orch._execute_tool_unstaged("send_email", INVENTED)
        assert out.startswith("[SYSTEM ERROR]") and sent == []


class TestTheAnswer:

    def test_the_address_given_next_stages_the_kept_draft(self, orch):
        orch.remember_emails_in(ASK)
        orch._execute_tool("send_email", INVENTED)
        reply, mode = orch._process_core("é joao.silva@gmail.com")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert orch._pending_action["args"] == {
            "to": "joao.silva@gmail.com", "subject": "Atraso", "body": "Vou atrasar."}
        assert "joao.silva@gmail.com" in reply

    def test_the_question_lasts_one_turn(self, orch):
        orch.remember_emails_in(ASK)
        orch._execute_tool("send_email", INVENTED)
        orch._process_core("quem foi einstein")
        _reply, mode = orch._process_core("joao.silva@gmail.com")
        assert mode != "AGENT_AWAITING_CONFIRMATION"

    def test_the_stream_does_not_chat_over_the_answer(self, orch):
        orch._offer = {"kind": "email_address", "args": {}}
        assert orch._is_pure_chat("joao.silva@gmail.com") is False


class TestBothEntryPoints:

    def test_addresses_are_harvested_in_both(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "orchestrator.py").read_text()
        # stream + _process_core, each on the user's own words (s85 review)
        assert source.count("self.remember_emails_in(own)") == 2
