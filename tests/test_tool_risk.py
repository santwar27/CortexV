"""
Tests for the Tool Risk Levels module (Tier 3.4 + 3.3).

Three layers:
  - Classification: every registered tool maps to the right level;
    unknown tools fall back conservatively.
  - is_destructive: replacement for the inline _DESTRUCTIVE_TOOLS
    frozenset that used to live in the orchestrator.
  - build_preview: per-tool surfaces (human/llm/cancel_label) match
    what _human_confirmation_prompt and _format_confirmation_brief
    used to produce before the refactor — keeps the user-facing
    behaviour byte-identical while moving the per-tool logic into
    one place.

These tests intentionally check exact strings on the preview output
(rather than just "contains") because the LLM-facing brief and the
human prompt are part of the orchestrator contract — a regression
that changed the wording could quietly confuse the model or the user.
"""
from __future__ import annotations

import pytest

from src.core.tool_risk import (
    DEFAULT_RISK,
    S1, S2, S3,
    TOOL_RISK,
    build_preview,
    is_destructive,
    risk_level,
)


# ----------------------------------------------------------------------
# classification — every registered tool
# ----------------------------------------------------------------------


class TestRiskClassification:
    @pytest.mark.parametrize("name,expected", [
        ("read_document",   S1),
        ("list_emails",     S1),
        ("list_events",     S1),
        ("write_document",  S2),
        ("web_search",      S2),
        ("send_email",      S3),
        ("create_event",    S3),
        ("create_reminder", S3),
        ("mac_quit_app",    S3),
    ])
    def test_known_tool_levels(self, name, expected):
        assert risk_level(name) == expected
        assert TOOL_RISK[name] == expected

    def test_unknown_tool_falls_back_to_default(self):
        assert risk_level("totally_made_up_tool") == DEFAULT_RISK
        # M6-A5: fail-closed — an UNCLASSIFIED tool gets the human gate.
        # The old S2 default meant a new external/irreversible tool added
        # without a TOOL_RISK entry executed immediately with just an
        # audit tag. Friction until classified is the deliberate cost.
        assert DEFAULT_RISK == S3

    def test_empty_name_falls_back_to_default(self):
        assert risk_level("") == DEFAULT_RISK

    def test_every_tool_in_specs_is_classified(self):
        # The companion of the S3 default (M6-A5): the live tool surface
        # never RELIES on the default — adding a tool to TOOL_SPECS
        # without classifying it breaks CI here, not in production.
        from src.core.tool_specs import TOOL_SPECS
        spec_names = {s["function"]["name"] for s in TOOL_SPECS}
        unclassified = spec_names - set(TOOL_RISK)
        assert not unclassified, (
            f"tools in TOOL_SPECS without a TOOL_RISK entry: {unclassified}"
        )


# ----------------------------------------------------------------------
# is_destructive — the gate that decides _pending_action staging
# ----------------------------------------------------------------------


class TestIsDestructive:
    @pytest.mark.parametrize(
        "name",
        ["send_email", "create_event", "create_reminder", "mac_quit_app"],
    )
    def test_s3_tools_are_destructive(self, name):
        assert is_destructive(name) is True

    @pytest.mark.parametrize("name", [
        "read_document", "list_emails", "list_events",   # S1
        "write_document", "web_search",                  # S2
    ])
    def test_non_s3_tools_not_destructive(self, name):
        assert is_destructive(name) is False

    def test_unknown_tool_is_destructive(self):
        # M6-A5: default S3 → unknown tool stages for the human gate.
        assert is_destructive("totally_unknown_tool") is True


# ----------------------------------------------------------------------
# build_preview — per-tool surfaces
# ----------------------------------------------------------------------


class TestSendEmailPreview:
    def test_human_layout(self):
        p = build_preview("send_email", {
            "to": "alice@example.com",
            "subject": "lunch?",
            "body": "free at 1pm?",
        })
        assert "Confirmar envio do email?" in p["human"]
        assert "Para: alice@example.com" in p["human"]
        assert "Assunto: lunch?" in p["human"]
        assert "Corpo: free at 1pm?" in p["human"]
        assert "'sim'" in p["human"] and "'não'" in p["human"]

    def test_human_body_is_shown_whole(self):
        """"sim" sends all of it: the old cut at 200 hid a PS after it (s88,
        Codex lote 2)."""
        long_body = "x" * 250 + " PS: o que vem depois"
        p = build_preview("send_email", {"to": "a@b", "subject": "s", "body": long_body})
        assert long_body in p["human"]
        assert "…" not in p["human"]

    def test_llm_brief_is_english_and_not_executed(self):
        p = build_preview("send_email", {"to": "a@b", "subject": "s", "body": "hi"})
        assert "CONFIRMATION REQUIRED" in p["llm"]
        assert "has NOT been sent" in p["llm"]
        # Must instruct the LLM not to claim success
        assert "Do not" in p["llm"] and "claim it was sent" in p["llm"]

    def test_llm_body_truncates_at_300_with_ellipsis(self):
        long_body = "y" * 400
        p = build_preview("send_email", {"to": "a@b", "subject": "s", "body": long_body})
        assert "y" * 300 + "…" in p["llm"]

    def test_cancel_label(self):
        p = build_preview("send_email", {"to": "a@b", "subject": "s", "body": "x"})
        assert p["cancel_label"] == "o envio do email"


class TestCreateEventPreview:
    def test_human_layout_minimal(self):
        p = build_preview("create_event", {
            "summary": "Lunch",
            "start_iso": "2026-05-22T13:00:00",
            "end_iso": "2026-05-22T14:00:00",
        })
        assert "Confirmar criação do evento?" in p["human"]
        assert "Título: Lunch" in p["human"]
        # Location/attendees are conditional — absent here
        assert "Local:" not in p["human"]
        assert "Participantes:" not in p["human"]

    def test_human_layout_with_location_and_attendees(self):
        p = build_preview("create_event", {
            "summary": "Standup",
            "start_iso": "x", "end_iso": "y",
            "location": "Office",
            "attendees": ["alice@x", "bob@y"],
        })
        assert "Local:  Office" in p["human"]
        assert "Participantes: alice@x, bob@y" in p["human"]

    def test_uses_injected_fmt_iso(self):
        called = []
        def fmt(iso):
            called.append(iso)
            return f"<<{iso}>>"
        p = build_preview("create_event", {
            "summary": "X", "start_iso": "S", "end_iso": "E",
        }, fmt_iso=fmt)
        assert "Início: <<S>>" in p["human"]
        assert "Fim:    <<E>>" in p["human"]
        assert called == ["S", "E"]

    def test_llm_brief_uses_raw_iso(self):
        # The LLM-facing brief intentionally shows the raw ISO so the
        # model can reason about timezone offsets if needed.
        p = build_preview("create_event", {
            "summary": "X",
            "start_iso": "2026-05-22T13:00:00+01:00",
            "end_iso": "2026-05-22T14:00:00+01:00",
        })
        assert "2026-05-22T13:00:00+01:00" in p["llm"]
        assert "has NOT been created" in p["llm"]

    def test_cancel_label(self):
        p = build_preview("create_event", {"summary": "x", "start_iso": "a", "end_iso": "b"})
        assert p["cancel_label"] == "a criação do evento"


class TestCreateReminderPreview:
    def test_human_layout_with_due(self):
        p = build_preview("create_reminder", {
            "title": "Buy milk",
            "due_iso": "2026-05-22T17:00:00",
        }, fmt_iso=lambda iso: f"<<{iso}>>")
        assert "Confirmar criação do lembrete?" in p["human"]
        assert "Título: Buy milk" in p["human"]
        assert "Quando: <<2026-05-22T17:00:00>>" in p["human"]
        assert "sincroniza com seu iPhone" in p["human"]

    def test_human_layout_without_due(self):
        p = build_preview("create_reminder", {"title": "Open thing"})
        assert "Quando: sem data definida" in p["human"]

    def test_llm_brief(self):
        p = build_preview("create_reminder", {"title": "x", "due_iso": "y"})
        assert "CONFIRMATION REQUIRED" in p["llm"]
        assert "has NOT been created" in p["llm"]

    def test_cancel_label_was_missing_before_refactor(self):
        # Before this refactor the cancel dict had only send_email +
        # create_event, so a cancelled reminder fell through to "ação".
        # The single-source-of-truth refactor fixes that gap.
        p = build_preview("create_reminder", {"title": "x"})
        assert p["cancel_label"] == "a criação do lembrete"


class TestMacQuitAppPreview:
    def test_human_layout_warns_unsaved(self):
        p = build_preview("mac_quit_app", {"app_name": "Notion"})
        assert "Fechar o Notion?" in p["human"]
        assert "não salvo" in p["human"]
        assert "'sim'" in p["human"] and "'não'" in p["human"]

    def test_llm_brief(self):
        p = build_preview("mac_quit_app", {"app_name": "Notion"})
        assert "CONFIRMATION REQUIRED" in p["llm"]
        assert "has NOT been quit" in p["llm"]
        assert "Do not" in p["llm"]

    def test_cancel_label(self):
        p = build_preview("mac_quit_app", {"app_name": "Notion"})
        assert p["cancel_label"] == "o fechamento do Notion"


class TestPreviewUnknownTool:
    def test_generic_fallback(self):
        p = build_preview("never_heard_of_it", {"some": "arg"})
        assert p["human"] == "Confirmar a ação? Responda 'sim' ou 'não'."
        assert "CONFIRMATION REQUIRED" in p["llm"]
        assert p["cancel_label"] == "a ação"

    def test_empty_name(self):
        p = build_preview("", {})
        assert p["cancel_label"] == "a ação"

    def test_none_args_treated_as_empty(self):
        # Builder must not blow up when args is None — sometimes the
        # orchestrator passes args=None for a fresh staging.
        p = build_preview("send_email", None)  # type: ignore[arg-type]
        assert "Confirmar envio do email?" in p["human"]
        # Empty/missing fields render as empty string, not KeyError.
        assert "Para: " in p["human"]


# ----------------------------------------------------------------------
# orchestrator integration — refactor preserves the user-visible output
# ----------------------------------------------------------------------


class TestOrchestratorRefactorParity:
    """The refactor moved per-tool branches out of the orchestrator and
    into build_preview. The orchestrator's `_human_confirmation_prompt`
    and `_format_confirmation_brief` are now thin delegates. These
    tests verify the orchestrator still produces the same output the
    LLM and user used to see."""

    def _orch(self):
        from src.core.orchestrator import CortexOrchestrator
        return CortexOrchestrator.__new__(CortexOrchestrator)

    def test_human_prompt_send_email(self):
        orch = self._orch()
        out = orch._human_confirmation_prompt({
            "type": "send_email",
            "args": {"to": "a@b", "subject": "s", "body": "hello"},
        })
        assert "Confirmar envio do email?" in out
        assert "Para: a@b" in out

    def test_human_prompt_create_event_uses_orch_fmt_iso(self):
        # The human prompt for create_event passes args through the
        # orchestrator's _format_iso_for_display, which renders BST/GMT
        # etc. Verify the function is wired by checking that the output
        # is NOT the raw ISO string.
        orch = self._orch()
        out = orch._human_confirmation_prompt({
            "type": "create_event",
            "args": {
                "summary": "Lunch",
                "start_iso": "2026-05-22T13:00:00",
                "end_iso": "2026-05-22T14:00:00",
            },
        })
        assert "Lunch" in out
        # _format_iso_for_display would either render the friendly form
        # ("sex 22/05 13:00 (BST)") or pass through, but never insert
        # the literal "2026-05-22T13:00:00" because that's the raw ISO.
        assert "2026-05-22T13:00:00" not in out

    def test_llm_brief_send_email(self):
        orch = self._orch()
        out = orch._format_confirmation_brief("send_email", {
            "to": "a@b", "subject": "s", "body": "x",
        })
        assert "CONFIRMATION REQUIRED" in out
        assert "has NOT been sent" in out

    def test_cancel_message_includes_reminder_label(self):
        """The pre-refactor cancel dict had no 'create_reminder' entry,
        so a cancelled reminder said 'Cancelado. Não vou prosseguir
        com o ação.' This refactor fixes that gap."""
        orch = self._orch()
        orch._pending_action = {
            "type": "create_reminder",
            "args": {"title": "x"},
        }
        # _persist_pending_action writes to disk — stub it.
        orch._persist_pending_action = lambda: None
        reply, mode = orch._cancel_pending_action()
        assert mode == "AGENT_ACTION_CANCELLED"
        assert "a criação do lembrete" in reply
        # Sanity: not the generic fallback
        assert "Não vou prosseguir com o ação" not in reply
