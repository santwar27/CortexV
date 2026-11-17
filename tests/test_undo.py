"""
Tests for the "Cortex, desfaz isso" undo path (Onda 3b).

Covers:
  - UNDO_RE dispatch regex (matches the command, not normal talk)
  - action_ledger.last_action / undone_ts_set (finding the target, no double-undo)
  - _handle_undo end-to-end on a bare orchestrator, with the delete_* clients
    mocked (no real Calendar/Reminders hit)
"""
import pytest

from src.core import action_ledger as al
from src.core.dispatch.undo import UNDO_RE


# --- dispatch regex --------------------------------------------------------

@pytest.mark.parametrize("text", [
    "desfaz", "desfaz isso", "desfazer", "desfaça", "desfaz aí",
    "desfaz a última", "desfaz a última ação", "desfaz essa",
    "cancela a última ação", "cancelar a última",
    "undo", "undo that", "undo it", "undo the last action",
    "Desfaz isso.", "UNDO",
])
def test_undo_re_matches_commands(text):
    assert UNDO_RE.match(text)


@pytest.mark.parametrize("text", [
    "não", "cancela", "cancela isso",            # bare cancel = denial path
    "como faço pra desfazer um nó?",             # not a command
    "desfaço o nó da gravata",                   # different verb form mid-talk
    "me explica o que é undo em git",            # topic, not command
    "",
])
def test_undo_re_ignores_non_commands(text):
    assert not UNDO_RE.match(text)


# --- ledger helpers --------------------------------------------------------

def _entry(action, ts, *, reversible=True, reverse=None, undone_ts=None):
    e = {"ts": ts, "action": action, "reversible": reversible, "reverse": reverse}
    if undone_ts:
        e["undone_ts"] = undone_ts
    return e


def test_last_action_returns_most_recent():
    entries = [
        _entry("create_event", "2026-06-05T10:00:00+00:00"),
        _entry("create_reminder", "2026-06-05T11:00:00+00:00"),
    ]
    assert al.last_action(entries)["action"] == "create_reminder"


def test_last_action_skips_undo_records():
    entries = [
        _entry("create_event", "2026-06-05T10:00:00+00:00"),
        {"ts": "2026-06-05T10:05:00+00:00", "action": "undo",
         "undone_ts": "2026-06-05T10:00:00+00:00"},
    ]
    # The only real action was undone → nothing left to undo.
    assert al.last_action(entries) is None


def test_last_action_no_double_undo():
    # event then reminder; reminder already undone → next target is the event.
    entries = [
        _entry("create_event", "2026-06-05T10:00:00+00:00"),
        _entry("create_reminder", "2026-06-05T11:00:00+00:00"),
        {"ts": "2026-06-05T11:05:00+00:00", "action": "undo",
         "undone_ts": "2026-06-05T11:00:00+00:00"},
    ]
    assert al.last_action(entries)["action"] == "create_event"


def test_last_action_empty():
    assert al.last_action([]) is None


# --- _handle_undo integration ---------------------------------------------

def _bare_orch():
    from src.core.orchestrator import CortexOrchestrator
    return CortexOrchestrator.__new__(CortexOrchestrator)


def _seed(tmp_path, monkeypatch):
    lg = al.ActionLedger(path=str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr("src.core.action_ledger.ledger", lg)
    return lg


def test_undo_reverses_last_event(tmp_path, monkeypatch):
    import src.integrations.calendar_client as cal
    deleted = {}
    monkeypatch.setattr(cal, "delete_event", lambda event_id: deleted.update(id=event_id))

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_event", {"summary": "Reunião"}, result_id="evt42", source="gate")

    reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_DONE"
    assert deleted == {"id": "evt42"}              # the right event was deleted
    # An undo record was appended pointing at the original.
    entries = lg.load_entries()
    assert entries[-1]["action"] == "undo"
    assert entries[-1]["undone_ts"] == entries[0]["ts"]


def test_undo_reminder_by_title(tmp_path, monkeypatch):
    """A ledger line from before the id (s86): by title."""
    import src.integrations.reminders_client as rem
    monkeypatch.setattr(rem, "delete_reminder",
                        lambda title, list_name=None, reminder_id=None: 1)

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_reminder", {"title": "comprar leite"}, source="gate")

    reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_DONE"


def test_undo_reminder_by_its_id(tmp_path, monkeypatch):
    """By title, an older "comprar leite" still open was the one deleted
    when Pedro had completed Cortex's (s88, Codex lote 2)."""
    import src.integrations.reminders_client as rem
    got = {}
    monkeypatch.setattr(rem, "delete_reminder",
                        lambda title, list_name=None, reminder_id=None:
                        got.update(id=reminder_id) or 1)

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_reminder", {"title": "comprar leite"},
              result_id="x-apple-reminder://ABCD-1234", source="gate")

    _reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_DONE"
    assert got["id"] == "x-apple-reminder://ABCD-1234"


def test_undo_reminder_refuses_an_ambiguous_title(tmp_path, monkeypatch):
    import src.integrations.reminders_client as rem
    monkeypatch.setattr(rem, "delete_reminder",
                        lambda title, list_name=None, reminder_id=None: 2)

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_reminder", {"title": "comprar leite"}, source="gate")

    o = _bare_orch(); o._reply_lang = "pt"
    reply, mode = o._handle_undo()
    assert mode != "UNDO_DONE"
    assert "não apaguei nenhum" in reply


class TestTheRemindersScript:
    """What delete_reminder asks Reminders.app — never run here."""

    def _script(self, monkeypatch, **kw):
        import src.integrations.reminders_client as rem
        seen = []
        monkeypatch.setattr(rem, "_run", lambda s: seen.append(s) or "1")
        rem.delete_reminder("comprar leite", **kw)
        return seen[0]

    def test_by_id_ignores_the_title(self, monkeypatch):
        s = self._script(monkeypatch, reminder_id="x-apple-reminder://ABCD-1234")
        assert 'whose id is "x-apple-reminder://ABCD-1234"' in s
        assert "comprar leite" not in s

    def test_by_title_deletes_only_a_single_match(self, monkeypatch):
        s = self._script(monkeypatch)
        assert 'whose name is "comprar leite" and completed is false' in s
        assert "if n is not 1 then return (n as text)" in s
        assert "last item" not in s

    @pytest.mark.parametrize("kw", [{}, {"reminder_id": "x-apple-reminder://ABCD-1234"}])
    def test_through_the_reference_never_a_variable(self, monkeypatch, kw):
        """`set matches to (reminders whose …)` took 24 s in the real app;
        counting and acting on the `whose` itself, ~1 s (s88)."""
        s = self._script(monkeypatch, **kw)
        assert "set matches" not in s and "set n to count of (reminders whose" in s

    def test_a_malformed_id_is_refused(self, monkeypatch):
        import src.integrations.reminders_client as rem
        with pytest.raises(rem.RemindersUnavailable):
            rem.delete_reminder("x", reminder_id='x" & do shell script "y')


def test_undo_note_by_title(tmp_path, monkeypatch):
    import src.integrations.notes_client as notes
    deleted = {}
    monkeypatch.setattr(notes, "delete_note", lambda title: (deleted.setdefault("t", title), 1)[1])

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_note", {"title": "ideias de projeto"}, source="dispatch")

    reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_DONE"
    assert deleted["t"] == "ideias de projeto"


def test_undo_irreversible_email_is_honest(tmp_path, monkeypatch):
    lg = _seed(tmp_path, monkeypatch)
    lg.record("send_email", {"to": "x@y.com"}, result_id="m1", source="gate")

    reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_IRREVERSIBLE"
    assert "e-mail" in reply and "não dá pra desfazer" in reply


def test_undo_nothing_to_undo(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    reply, mode = _bare_orch()._handle_undo()
    assert mode == "UNDO_NONE"


def test_double_undo_moves_to_previous(tmp_path, monkeypatch):
    import src.integrations.calendar_client as cal
    import src.integrations.reminders_client as rem
    monkeypatch.setattr(cal, "delete_event", lambda event_id: None)
    monkeypatch.setattr(rem, "delete_reminder",
                        lambda title, list_name=None, reminder_id=None: 1)

    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_event", {"summary": "x"}, result_id="evt1", source="gate")
    lg.record("create_reminder", {"title": "t"}, source="gate")

    o = _bare_orch()
    _, m1 = o._handle_undo()   # undoes the reminder (most recent)
    _, m2 = o._handle_undo()   # should now undo the event
    assert m1 == "UNDO_DONE" and m2 == "UNDO_DONE"
    # third undo: nothing left
    _, m3 = o._handle_undo()
    assert m3 == "UNDO_NONE"


def test_undo_reverse_failure_surfaces_error(tmp_path, monkeypatch):
    import src.integrations.calendar_client as cal

    def _boom(event_id):
        from src.integrations.calendar_client import CalendarUnavailable
        raise CalendarUnavailable("api down")

    monkeypatch.setattr(cal, "delete_event", _boom)
    lg = _seed(tmp_path, monkeypatch)
    lg.record("create_event", {"summary": "x"}, result_id="evt1", source="gate")

    reply, mode = _bare_orch()._handle_undo()
    assert mode == "AGENT_ACTION_ERROR"
    # The action was NOT marked undone, so a retry is still possible.
    assert lg.load_entries()[-1]["action"] == "create_event"


def test_undo_after_a_plan_reverses_the_plans_action(tmp_path, monkeypatch):
    """A plan created event B without a ledger line, so "desfaz isso" deleted
    the event A made before it (s88, Codex lote 2)."""
    import src.integrations.calendar_client as cal
    from tests.test_dispatch_parity import _build_real_orch
    ids = iter(["evento-A", "evento-B"])
    monkeypatch.setattr(cal, "create_event",
                        lambda **kw: {"id": next(ids), "summary": kw["summary"]})
    deleted = []
    monkeypatch.setattr(cal, "delete_event", lambda event_id: deleted.append(event_id))
    o = _build_real_orch(tmp_path)
    o.background = False
    o._reply_lang = "pt"
    o._run_action("create_event", {"summary": "A", "start_iso": "2026-10-20T10:00:00",
                                   "end_iso": "2026-10-20T11:00:00"})
    o._pending_action = {"type": "execute_plan", "args": {
        "objetivo": "B", "lang": "pt", "passos": [{
            "id": "p1", "tool": "create_event", "depende_de": [], "descricao": "B",
            "args": {"summary": "B", "start_iso": "2026-10-21T10:00:00",
                     "end_iso": "2026-10-21T11:00:00"}}]}}
    o._execute_staged_plan()
    entry = al.ledger.load_entries()[-1]
    assert (entry["action"], entry["result_id"], entry["source"]) == (
        "create_event", "evento-B", "plan")
    _reply, mode = o._handle_undo()
    assert mode == "UNDO_DONE" and deleted == ["evento-B"]
