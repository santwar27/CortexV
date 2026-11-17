"""
i18n regression shield — scheduling (staged reminder/calendar confirmation) +
features/undo (s61 cont, Claude-A). Completes scope-C item (2): the two "heavy"
families the contacts/email shield (test_contacts_email_i18n) didn't cover.

Pattern mirrors test_macos_i18n: host composites of the REAL mixins + I18nMixin,
`_reply_lang` forced, deps borrowed (the @staticmethod helpers live on the
orchestrator) or monkeypatched. Asserts EN replies when lang='en' and PT
unchanged by default. 100% test — no hot-path change → no restart.
"""
from datetime import datetime, timezone

import pytest

from src.core.tool_risk import build_preview
from src.core.scheduling_mixin import SchedulingMixin
from src.core.staging_mixin import StagingMixin
from src.core.features_mixin import FeaturesMixin
from src.core.i18n import I18nMixin
from src.core.orchestrator import CortexOrchestrator as _O

NOW = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


# ====================================================================
# A) build_preview — the source of the spoken confirmation text
#    (reached by the staged reminder/calendar/email handlers via
#    StagingMixin._human_confirmation_prompt). The "llm" face stays
#    English on purpose (internal brief the model reframes).
# ====================================================================
class TestBuildPreviewEN:
    def test_reminder_en(self):
        p = build_preview("create_reminder",
                          {"title": "leite", "due_iso": "2026-06-10T09:00:00+00:00"},
                          lang="en")
        assert p["human"].startswith("Confirm creating the reminder?")
        assert "syncs to your iPhone" in p["human"]
        assert "Reply 'yes' to create or 'no' to cancel." in p["human"]
        assert p["cancel_label"] == "creating the reminder"

    def test_reminder_pt_default(self):
        p = build_preview("create_reminder", {"title": "leite"})
        assert p["human"].startswith("Confirmar criação do lembrete?")
        assert "sincroniza com seu iPhone" in p["human"]
        assert p["cancel_label"] == "a criação do lembrete"

    def test_event_en(self):
        p = build_preview("create_event",
                          {"summary": "R", "start_iso": "x", "end_iso": "y"}, lang="en")
        assert p["human"].startswith("Confirm creating the event?")
        assert p["cancel_label"] == "creating the event"

    def test_event_pt_default(self):
        p = build_preview("create_event", {"summary": "R", "start_iso": "x", "end_iso": "y"})
        assert p["human"].startswith("Confirmar criação do evento?")

    def test_send_email_en(self):
        p = build_preview("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}, lang="en")
        assert p["human"].startswith("Confirm sending the email?")
        assert p["cancel_label"] == "sending the email"

    def test_llm_face_stays_english_even_in_pt(self):
        # Deliberate: the LLM brief is internal (the model reframes it in the
        # user's language), so it is always English regardless of `lang`.
        assert build_preview("create_reminder", {"title": "x"}, lang="pt")["llm"] \
            .startswith("CONFIRMATION REQUIRED")


# ====================================================================
# B) Scheduling handlers — staged reminder/calendar confirmation prompt
# ====================================================================
class _SchedHost(SchedulingMixin, StagingMixin, I18nMixin):
    """Real scheduling + staging surface; the parse/format helpers are
    @staticmethod on the orchestrator, so we borrow them verbatim."""
    _REMINDER_BODY_RE = _O._REMINDER_BODY_RE
    _VAGUE_CALENDAR_RE = _O._VAGUE_CALENDAR_RE
    _parse_reminder_due = staticmethod(_O._parse_reminder_due)
    _calendar_end_iso = staticmethod(_O._calendar_end_iso)
    _format_iso_for_display = staticmethod(_O._format_iso_for_display)

    def __init__(self, pending_file: str):
        self._pending_action = None
        self._PENDING_ACTION_FILE = pending_file


@pytest.fixture
def sched(tmp_path):
    return _SchedHost(str(tmp_path / "pending.json"))


class TestSchedulingConfirmationEN:
    def test_reminder_confirmation_en(self, sched):
        sched._reply_lang = "en"
        reply, mode = sched._handle_reminder_regex("comprar leite amanhã às 9h", NOW)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert reply.startswith("Confirm creating the reminder?")

    def test_reminder_confirmation_pt_default(self, sched):
        reply, mode = sched._handle_reminder_regex("comprar leite amanhã às 9h", NOW)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert reply.startswith("Confirmar criação do lembrete?")

    def test_calendar_confirmation_en(self, sched):
        sched._reply_lang = "en"
        reply, mode = sched._handle_calendar_regex("reunião amanhã às 15h", NOW)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert reply.startswith("Confirm creating the event?")

    def test_calendar_confirmation_pt_default(self, sched):
        reply, mode = sched._handle_calendar_regex("reunião amanhã às 15h", NOW)
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert reply.startswith("Confirmar criação do evento?")


# ====================================================================
# C) features/undo — the deterministic last-action reverse
# ====================================================================
class _UndoHost(FeaturesMixin, StagingMixin, I18nMixin):
    _UNDO_ACTION_LABELS = _O._UNDO_ACTION_LABELS
    _UNDO_ACTION_LABELS_EN = _O._UNDO_ACTION_LABELS_EN


class TestUndoEN:
    def test_no_target_en(self, monkeypatch):
        import src.core.action_ledger as al
        monkeypatch.setattr(al, "last_action", lambda entries: None)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [])
        reply, mode = _make_undo("en")._handle_undo()
        assert mode == "UNDO_NONE"
        assert reply == "There's nothing recent to undo."

    def test_no_target_pt_default(self, monkeypatch):
        import src.core.action_ledger as al
        monkeypatch.setattr(al, "last_action", lambda entries: None)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [])
        reply, mode = _make_undo("pt")._handle_undo()
        assert mode == "UNDO_NONE"
        assert reply == "Não há nada recente pra desfazer."

    def test_irreversible_en(self, monkeypatch):
        import src.core.action_ledger as al
        target = {"action": "send_email", "reversible": False, "reverse": None,
                  "ts": datetime.now(timezone.utc).isoformat()}
        monkeypatch.setattr(al, "last_action", lambda entries: target)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [target])
        reply, mode = _make_undo("en")._handle_undo()
        assert mode == "UNDO_IRREVERSIBLE"
        assert reply == "The last thing I did was send an email, and that can't be undone."

    def test_irreversible_pt_default(self, monkeypatch):
        import src.core.action_ledger as al
        target = {"action": "send_email", "reversible": False, "reverse": None,
                  "ts": datetime.now(timezone.utc).isoformat()}
        monkeypatch.setattr(al, "last_action", lambda entries: target)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [target])
        reply, mode = _make_undo("pt")._handle_undo()
        assert mode == "UNDO_IRREVERSIBLE"
        assert "não dá pra desfazer" in reply
        assert "enviar um e-mail" in reply

    def test_reversible_done_en(self, monkeypatch):
        import src.core.action_ledger as al
        target = {"action": "create_event", "reversible": True,
                  "reverse": {"tool": "delete_event", "args": {"event_id": "x"}},
                  "ts": datetime.now(timezone.utc).isoformat()}
        monkeypatch.setattr(al, "last_action", lambda entries: target)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [target])
        monkeypatch.setattr(al.ledger, "record", lambda *a, **k: None)
        host = _make_undo("en")
        # Stub the actual reverse execution — its localisation is covered in
        # staging tests; here we only check undo's EN wrapper.
        host._run_reverse = lambda reverse: ("Event removed from the calendar.", "AGENT_ACTION_DONE")
        reply, mode = host._handle_undo()
        assert mode == "UNDO_DONE"
        assert reply == "Done, undid: Event removed from the calendar."

    def test_reversible_done_pt_default(self, monkeypatch):
        import src.core.action_ledger as al
        target = {"action": "create_event", "reversible": True,
                  "reverse": {"tool": "delete_event", "args": {"event_id": "x"}},
                  "ts": datetime.now(timezone.utc).isoformat()}
        monkeypatch.setattr(al, "last_action", lambda entries: target)
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [target])
        monkeypatch.setattr(al.ledger, "record", lambda *a, **k: None)
        host = _make_undo("pt")
        host._run_reverse = lambda reverse: ("Evento removido do calendário.", "AGENT_ACTION_DONE")
        reply, mode = host._handle_undo()
        assert mode == "UNDO_DONE"
        assert reply == "Pronto, desfiz: Evento removido do calendário."


def _make_undo(lang: str) -> "_UndoHost":
    h = _UndoHost.__new__(_UndoHost)
    h._reply_lang = lang
    return h
