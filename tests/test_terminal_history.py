"""Terminal turns land in the chat history too (s85).

Pedro talked to Cortex in the terminal (`cortex`, `cx`), closed it, and the
conversation was gone: only the voice and the chat page wrote to the history.
Asked to "bring it here", Cortex described an unrelated old conversation.
Measured before: 0 terminal turns in the history (daily 1, ui 2, voice 46).
"""
import pytest
from fastapi.testclient import TestClient

import src.api as api_mod
from src.ui import chat_store as cs


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = cs.ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cs, "store", s)
    return s


def _convos(store):
    return {c["title"]: c["id"] for c in store.list_conversations()}


class TestOneConversationPerDay:

    def test_the_days_turns_share_one_conversation(self, store):
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "oi", "Olá, Pedro.")
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "que horas são", "São 10:00.")
        title = cs.daily_title(cs.TERMINAL_LABEL)
        assert list(_convos(store)) == [title]
        turns = store.get_turns(_convos(store)[title])
        assert [t["user"] for t in turns] == ["oi", "que horas são"]
        assert {t["source"] for t in turns} == {"terminal"}

    def test_terminal_and_voice_are_separate_rows(self, store):
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "oi", "Olá.")
        cs.append_daily("voice", cs.VOICE_LABEL, "oi", "Olá.")
        titles = set(_convos(store))
        assert titles == {cs.daily_title(cs.TERMINAL_LABEL),
                          cs.daily_title(cs.VOICE_LABEL)}

    def test_empty_turns_are_not_stored(self, store):
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "", "x")
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "x", "  ")
        assert _convos(store) == {}

    def test_a_storage_failure_is_swallowed(self, monkeypatch):
        class Boom:
            def conversation_titled(self, title):
                raise RuntimeError("db locked")

        monkeypatch.setattr(cs, "store", Boom())
        cs.append_daily("terminal", cs.TERMINAL_LABEL, "oi", "Olá.")   # no raise


class _Stub:
    def process(self, user_input, image_b64=None, language=None, conversation_history=None):
        return ("Resposta.", "OFFLINE")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_mod, "SECRET_API_KEY", "test-key-xyz")
    monkeypatch.setattr(api_mod, "orchestrator", _Stub())
    monkeypatch.setattr(api_mod, "_ALLOWLIST", [])
    return TestClient(api_mod.app)


class TestAsk:
    """`cx` posts to /ask. From this Mac it is the terminal; over the network
    (the iPhone via Tailscale) it is labelled as such."""

    def test_a_network_call_is_not_called_terminal(self, client, store):
        r = client.post("/ask", headers={"X-Cortex-Key": "test-key-xyz"},
                        json={"user_input": "meus emails"})
        assert r.status_code == 200 and r.json()["response"] == "Resposta."
        assert list(_convos(store)) == [cs.daily_title(cs.NETWORK_LABEL)]

    @pytest.mark.parametrize("host,label", [
        ("127.0.0.1", cs.TERMINAL_LABEL), ("::1", cs.TERMINAL_LABEL),
        ("100.64.0.42", cs.NETWORK_LABEL),
    ])
    def test_labelled_by_where_it_came_from(self, store, host, label):
        request = type("R", (), {"client": type("C", (), {"host": host})()})()
        api_mod._record_ask_turn(request, "oi", "Olá.", "REFLEX")
        assert list(_convos(store)) == [cs.daily_title(label)]

    def test_the_reply_is_unchanged(self, client, store):
        """The iPhone Shortcut reads `response`: the contract must not move."""
        r = client.post("/ask", headers={"X-Cortex-Key": "test-key-xyz"},
                        json={"user_input": "oi"})
        assert set(r.json()) == {"response", "status_mode"}


class TestInteractiveCli:
    """`cortex` runs its own orchestrator inside the terminal — nothing of it
    ever reached the API."""

    def test_every_turn_is_kept(self, store, monkeypatch):
        import src.main as cli

        class FakeOrch:
            engine_analytical = engine_creative = type("E", (), {"model_name": "m"})()

            def process(self, text, image_b64=None, conversation_history=None):
                return (f"eco: {text}", "OFFLINE")

        typed = iter(["lê os dois", "sim", "sair"])
        monkeypatch.setattr(cli, "CortexOrchestrator", FakeOrch)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(typed))
        with pytest.raises(SystemExit):
            cli.run()
        turns = store.get_turns(_convos(store)[cs.daily_title(cs.TERMINAL_LABEL)])
        assert [(t["user"], t["reply"]) for t in turns] == [
            ("lê os dois", "eco: lê os dois"), ("sim", "eco: sim")]
