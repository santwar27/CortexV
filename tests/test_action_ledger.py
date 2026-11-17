"""
Tests for src/core/action_ledger.py — the side-effect journal.

`record` is exercised against a temp path (append + fail-soft); `reverse_spec`
is pure; `render`/`load_entries` mirror the policy_detector test style. The
call sites that PRODUCE records live in staging_mixin._run_action.
"""
import json

from src.core import action_ledger as al


# --- reverse_spec (pure) ---------------------------------------------------

def test_code_change_reverses_only_with_id_and_path():
    for result, args, pid in [("result", {"id": "arg", "path": "/code.py"}, "result"),
                              (None, {"id": "arg", "path": "/code.py"}, "arg")]:
        assert al.reverse_spec("apply_code_change", args, result) == (
            True, {"tool": "restore_code_change", "args": {"id": pid, "path": "/code.py"}})
    for result, args in [(None, {"path": "/code.py"}), ("id", {}), (None, {})]:
        assert al.reverse_spec("apply_code_change", args, result) == (False, None)


def test_send_email_is_irreversible():
    reversible, reverse = al.reverse_spec("send_email", {"to": "a@b.com"}, "msg123")
    assert reversible is False and reverse is None


def test_create_event_reversible_with_id():
    reversible, reverse = al.reverse_spec("create_event", {"summary": "x"}, "evt99")
    assert reversible is True
    assert reverse == {"tool": "delete_event", "args": {"event_id": "evt99"}}


def test_create_event_not_reversible_without_id():
    reversible, reverse = al.reverse_spec("create_event", {"summary": "x"}, None)
    assert reversible is False and reverse is None


def test_create_reminder_reverses_by_title():
    reversible, reverse = al.reverse_spec("create_reminder", {"title": "comprar leite"}, None)
    assert reversible is True
    assert reverse == {"tool": "delete_reminder", "args": {"title": "comprar leite"}}


def test_reverse_spec_reminder_by_its_id():
    """create_reminder answers with the reminder's id (s86) — the undo
    targets that one, not any namesake (s88, Codex lote 2)."""
    reversible, reverse = al.reverse_spec(
        "create_reminder", {"title": "comprar leite"}, "x-apple-reminder://ABCD-1234")
    assert reversible is True
    assert reverse == {"tool": "delete_reminder", "args": {
        "reminder_id": "x-apple-reminder://ABCD-1234", "title": "comprar leite"}}


def test_create_note_reverses_by_title():
    reversible, reverse = al.reverse_spec("create_note", {"title": "ideias"}, None)
    assert reversible is True
    assert reverse == {"tool": "delete_note", "args": {"title": "ideias"}}


def test_create_note_reverses_by_id():
    note_id = "x-coredata://933F6320-BA85-4FD1-930A-598B23DF1AB2/ICNote/p598"
    assert al.reverse_spec("create_note", {"title": "Work", "note_id": note_id}, None) == (
        True, {"tool": "delete_note", "args": {"note_id": note_id, "title": "Work"}})


def test_mac_quit_reverses_by_relaunch():
    reversible, reverse = al.reverse_spec(
        "mac_quit_app", {"app_name": "Safari", "was_running": True}, None)
    assert reversible is True
    assert reverse == {"tool": "mac_open_app", "args": {"app_name": "Safari"}}


def test_mac_quit_only_reopens_an_app_cortex_closed():
    for args in [{}, {"was_running": False}, {"was_running": 1},
                 {"was_running": True, "uncertain": True}, {"uncertain": True}]:
        assert al.reverse_spec("mac_quit_app", {"app_name": "Safari", **args}, None) == (False, None)


def test_unknown_action_not_reversible():
    assert al.reverse_spec("mac_sleep", {}, None) == (False, None)


# --- record + load (fail-soft I/O) -----------------------------------------

def _ledger(tmp_path):
    return al.ActionLedger(path=str(tmp_path / "ledger.jsonl"))


def test_record_appends_one_line(tmp_path):
    lg = _ledger(tmp_path)
    lg.record("create_event", {"summary": "Reunião"}, result_id="evt1", source="gate")
    entries = lg.load_entries()
    assert len(entries) == 1
    e = entries[0]
    assert e["action"] == "create_event"
    assert e["result_id"] == "evt1"
    assert e["risk"] == "S3"            # auto-derived from tool_risk
    assert e["reversible"] is True
    assert e["reverse"]["tool"] == "delete_event"
    assert e["source"] == "gate"
    assert "ts" in e and e["status"] == "done"


def test_record_appends_multiple(tmp_path):
    lg = _ledger(tmp_path)
    lg.record("send_email", {"to": "x@y.com"}, result_id="m1")
    lg.record("create_reminder", {"title": "t"})
    assert len(lg.load_entries()) == 2


def test_record_truncates_long_args(tmp_path):
    lg = _ledger(tmp_path)
    lg.record("send_email", {"body": "x" * 500}, result_id="m1")
    body = lg.load_entries()[0]["args"]["body"]
    assert len(body) <= al._ARG_VALUE_CAP
    assert body.endswith("…")


def test_record_redacts_sensitive_args(tmp_path):
    # M6-A6: redaction at the border, before truncation.
    lg = _ledger(tmp_path)
    lg.record("send_email", {
        "to": "x@y.com",
        "body": "o token é abc123",
        "api_key": "sk-1",
    })
    args = lg.load_entries()[0]["args"]
    assert args["to"] == "x@y.com"
    assert args["api_key"] == "[REDACTED]"
    assert "abc123" not in args["body"]


def test_send_email_recorded_irreversible(tmp_path):
    lg = _ledger(tmp_path)
    lg.record("send_email", {"to": "x@y.com"}, result_id="m1")
    assert lg.load_entries()[0]["reversible"] is False


def test_load_missing_file_is_empty(tmp_path):
    assert _ledger(tmp_path).load_entries() == []


def test_load_skips_malformed(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(
        '{"action":"send_email"}\n'
        "garbage\n"
        '{"action":"create_event"}\n'
    )
    assert len(al.ActionLedger(path=str(p)).load_entries()) == 2


def test_record_is_fail_soft_on_bad_path(tmp_path):
    # A path whose parent can't be created should swallow the error, not raise.
    lg = al.ActionLedger(path="/this/does/not/exist/\x00/ledger.jsonl")
    lg.record("send_email", {"to": "x"})  # must not raise


# --- render ----------------------------------------------------------------

def test_render_empty():
    assert "Nada ainda" in al.render([])


def test_render_lists_actions_no_tables(tmp_path):
    lg = _ledger(tmp_path)
    lg.record("create_event", {"summary": "x"}, result_id="evt1", source="gate")
    text = al.render(lg.load_entries())
    assert "create_event" in text
    assert "reversível" in text
    assert "|" not in text          # bullets, never markdown tables
    assert "desfaz" in text         # points at the undo command


def test_render_newest_first_and_truncates(tmp_path):
    lg = _ledger(tmp_path)
    for i in range(40):
        lg.record("create_reminder", {"title": f"t{i}"})
    text = al.render(lg.load_entries(), limit=10)
    assert "+30 mais" in text


def test_main_runs_clean(tmp_path, capsys):
    p = tmp_path / "ledger.jsonl"
    al.ActionLedger(path=str(p)).record("send_email", {"to": "x@y.com"}, result_id="m1")
    rc = al.main(["--path", str(p)])
    assert rc == 0
    assert "send_email" in capsys.readouterr().out


# --- integration: _run_action records into the ledger ----------------------

def test_run_action_records_into_ledger(tmp_path, monkeypatch):
    """A successful _run_action (the single chokepoint for S3 actions) must
    append a ledger entry — without changing the action's (reply, mode)."""
    import src.integrations.gmail_client as gmail
    from src.core.orchestrator import CortexOrchestrator

    monkeypatch.setattr(
        gmail, "send_email",
        lambda to, subject, body, message_id=None: {"to": to, "id": "msg_test", "subject": subject},
    )
    # Redirect the shared ledger singleton at a temp file for this test.
    monkeypatch.setattr("src.core.staging_mixin._action_ledger",
                        al.ActionLedger(path=str(tmp_path / "ledger.jsonl")))

    o = CortexOrchestrator.__new__(CortexOrchestrator)
    reply, mode = o._run_action("send_email", {"to": "x@y.com", "subject": "oi", "body": "corpo"})

    assert mode == "AGENT_ACTION_DONE"
    entries = al.ActionLedger(path=str(tmp_path / "ledger.jsonl")).load_entries()
    assert len(entries) == 1
    assert entries[0]["action"] == "send_email"
    assert entries[0]["result_id"] == "msg_test"
    assert entries[0]["source"] == "gate"


def test_handle_note_create_records_into_ledger(tmp_path, monkeypatch):
    """Creating a note via dispatch must log it (so undo can delete it)."""
    from src.core.orchestrator import CortexOrchestrator

    # _handle_note_create lives in NotesMixin (s59) — patch its module's ledger.
    monkeypatch.setattr("src.core.notes_mixin._action_ledger",
                        al.ActionLedger(path=str(tmp_path / "ledger.jsonl")))
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    o.notes = type("N", (), {"create_note": staticmethod(
        lambda title, body, folder=None: {"title": title})})()

    reply, mode = o._handle_note_create("Ideias\nlinha dois")
    assert mode == "NOTE_CREATE"
    entries = al.ActionLedger(path=str(tmp_path / "ledger.jsonl")).load_entries()
    assert len(entries) == 1
    assert entries[0]["action"] == "create_note"
    assert entries[0]["args"]["title"] == "Ideias"
    assert entries[0]["reversible"] is True   # reverse_spec → delete_note


def test_run_action_error_does_not_record(tmp_path, monkeypatch):
    """A failed action must NOT leave a 'done' ledger entry."""
    import src.integrations.gmail_client as gmail
    from src.core.orchestrator import CortexOrchestrator

    def _boom(to, subject, body, message_id=None):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(gmail, "send_email", _boom)
    monkeypatch.setattr("src.core.staging_mixin._action_ledger",
                        al.ActionLedger(path=str(tmp_path / "ledger.jsonl")))

    o = CortexOrchestrator.__new__(CortexOrchestrator)
    reply, mode = o._run_action("send_email", {"to": "x@y.com", "subject": "s", "body": "b"})

    assert mode == "AGENT_ACTION_ERROR"
    assert al.ActionLedger(path=str(tmp_path / "ledger.jsonl")).load_entries() == []


def test_module_doc_does_not_claim_undo_is_unwired():
    """M6-B4 (review Fable 5): the docstring said 'nothing executes it yet'
    long after the undo was wired (dispatch/undo.py → staging_mixin). Pin
    the doc so the vaporware claim can't drift back and mislead a future
    contributor into re-implementing undo."""
    doc = al.__doc__ or ""
    for stale in ("nothing executes it yet", "no client has the delete_*"):
        assert stale not in doc
    assert "dispatch/undo.py" in doc


class TestTheUndoSpecKeepsNoSecret:
    """"args" was redacted and "reverse" wrote the same title whole (s88,
    Codex lote 10)."""

    def test_with_an_id_the_undo_still_works(self, tmp_path):
        import json
        from src.core.action_ledger import ActionLedger
        path = tmp_path / "l.jsonl"
        ActionLedger(str(path)).record("create_reminder", {"title": "senha é Sentinela123"},
                                       result_id="x-apple-reminder://ABC")
        assert "Sentinela123" not in path.read_text()
        rec = json.loads(path.read_text())
        assert rec["reversible"] is True
        assert rec["reverse"]["args"]["reminder_id"] == "x-apple-reminder://ABC"

    def test_by_name_only_it_is_given_up(self, tmp_path):
        import json
        from src.core.action_ledger import ActionLedger
        path = tmp_path / "l.jsonl"
        ActionLedger(str(path)).record("create_note", {"title": "senha é Sentinela456"})
        assert "Sentinela456" not in path.read_text()
        rec = json.loads(path.read_text())
        assert rec["reversible"] is False and rec["reverse"] is None

    def test_an_ordinary_title_is_untouched(self, tmp_path):
        import json
        from src.core.action_ledger import ActionLedger
        path = tmp_path / "l.jsonl"
        ActionLedger(str(path)).record("create_note", {"title": "Mercado"})
        rec = json.loads(path.read_text())
        assert rec["reverse"] == {"tool": "delete_note", "args": {"title": "Mercado"}}
