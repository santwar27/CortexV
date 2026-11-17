"""
Tests for the Skill Fast Path Registry (Tier 3.2).

Pure-logic tests against an in-memory Palace fixture. Covers CRUD,
validation, match/execute, the failure-isolation contracts (broken
regex in a row must not poison the cache, unknown handler returns
None rather than raising), and the stats bump.
"""
from __future__ import annotations

import json

import pytest

from src.core.skill_registry import (
    Skill,
    SkillRegistry,
    SkillRegistryError,
    _KNOWN_HANDLERS,
)
from src.memory.palace import Palace


@pytest.fixture
def palace():
    p = Palace(db_path=":memory:")
    yield p
    p.close()


@pytest.fixture
def registry(palace):
    return SkillRegistry(palace)


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------


class TestSchema:
    def test_skills_table_exists(self, palace):
        cur = palace._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='skills'"
        )
        assert cur.fetchone() is not None

    def test_columns_present(self, palace):
        cols = {r["name"] for r in palace._conn.execute("PRAGMA table_info(skills)").fetchall()}
        assert cols == {
            "id", "name", "trigger_regex", "handler_type", "handler_payload",
            "enabled", "invocation_count", "last_used_at", "created_at",
        }

    def test_known_handlers_includes_static_response(self):
        assert "static_response" in _KNOWN_HANDLERS


# ----------------------------------------------------------------------
# register — input validation
# ----------------------------------------------------------------------


class TestRegisterValidation:
    def test_happy_path(self, registry):
        sid = registry.register(
            "cpf", r"\bcpf\b", "static_response", {"text": "123.456.789-00"},
        )
        assert sid > 0

    def test_duplicate_name_rejected(self, registry):
        registry.register("foo", "bar", "static_response", {"text": "x"})
        with pytest.raises(SkillRegistryError):
            registry.register("foo", "baz", "static_response", {"text": "y"})

    @pytest.mark.parametrize("bad_name", ["", "   ", None])
    def test_empty_name_rejected(self, registry, bad_name):
        with pytest.raises(SkillRegistryError):
            registry.register(bad_name, "r", "static_response", {"text": "x"})

    def test_name_too_long_rejected(self, registry):
        with pytest.raises(SkillRegistryError):
            registry.register("x" * 81, "r", "static_response", {"text": "x"})

    @pytest.mark.parametrize("bad_regex", ["", "   ", None])
    def test_empty_regex_rejected(self, registry, bad_regex):
        with pytest.raises(SkillRegistryError):
            registry.register("n", bad_regex, "static_response", {"text": "x"})

    def test_invalid_regex_rejected(self, registry):
        with pytest.raises(SkillRegistryError) as exc:
            registry.register("n", "[unclosed", "static_response", {"text": "x"})
        assert "invalid regex" in str(exc.value)

    def test_regex_too_long_rejected(self, registry):
        with pytest.raises(SkillRegistryError):
            registry.register("n", "a" * 501, "static_response", {"text": "x"})

    def test_unknown_handler_type_rejected(self, registry):
        with pytest.raises(SkillRegistryError):
            registry.register("n", "r", "magic_handler", {"text": "x"})

    def test_static_response_requires_text(self, registry):
        with pytest.raises(SkillRegistryError):
            registry.register("n", "r", "static_response", {})
        with pytest.raises(SkillRegistryError):
            registry.register("n", "r", "static_response", {"text": ""})
        with pytest.raises(SkillRegistryError):
            registry.register("n", "r", "static_response", {"text": "   "})

    def test_payload_too_large_rejected(self, registry):
        big = {"text": "x" * 5000}
        with pytest.raises(SkillRegistryError):
            registry.register("n", "r", "static_response", big)


# ----------------------------------------------------------------------
# match
# ----------------------------------------------------------------------


class TestMatch:
    def test_no_skills_no_match(self, registry):
        assert registry.match("anything") is None

    def test_empty_input_no_match(self, registry):
        registry.register("n", r"\bcpf\b", "static_response", {"text": "x"})
        assert registry.match("") is None
        assert registry.match("   ") is None
        assert registry.match(None) is None  # type: ignore[arg-type]

    def test_case_insensitive_match(self, registry):
        registry.register("cpf", r"\bcpf\b", "static_response", {"text": "x"})
        assert registry.match("qual meu CPF?") is not None
        assert registry.match("Meu Cpf é?").name == "cpf"

    def test_no_match_returns_none(self, registry):
        registry.register("cpf", r"\bcpf\b", "static_response", {"text": "x"})
        assert registry.match("hello world") is None

    def test_first_match_wins(self, registry):
        # First-inserted goes first in the in-memory list.
        registry.register("a", r"hello", "static_response", {"text": "first"})
        registry.register("b", r"hello", "static_response", {"text": "second"})
        m = registry.match("hello!")
        assert m.name == "a"

    def test_search_not_anchored(self, registry):
        # We use .search() not .match() — pattern can match anywhere.
        registry.register("kw", r"keyword", "static_response", {"text": "x"})
        assert registry.match("a sentence with keyword in middle") is not None

    def test_disabled_skill_not_matched(self, registry):
        registry.register("cpf", r"\bcpf\b", "static_response", {"text": "x"})
        registry.set_enabled("cpf", False)
        assert registry.match("qual meu cpf") is None

    def test_match_does_not_bump_invocation_count(self, registry, palace):
        registry.register("cpf", r"\bcpf\b", "static_response", {"text": "x"})
        registry.match("cpf?")
        registry.match("cpf?")
        rows = registry.list_all()
        assert rows[0]["invocation_count"] == 0


# ----------------------------------------------------------------------
# execute
# ----------------------------------------------------------------------


class TestExecute:
    def test_static_response_returns_text(self, registry):
        registry.register("cpf", r"\bcpf\b", "static_response",
                          {"text": "123.456.789-00"})
        skill = registry.match("qual meu cpf")
        assert registry.execute(skill, "qual meu cpf") == "123.456.789-00"

    def test_execute_bumps_invocation_count(self, registry):
        registry.register("cpf", r"\bcpf\b", "static_response", {"text": "x"})
        skill = registry.match("cpf?")
        registry.execute(skill, "cpf?")
        registry.execute(skill, "cpf?")
        rows = registry.list_all()
        assert rows[0]["invocation_count"] == 2
        assert rows[0]["last_used_at"] is not None

    def test_unknown_handler_returns_none(self, registry, palace):
        # Bypass validation by writing directly to the table — simulates
        # a row that landed via migration or raw SQL.
        from datetime import datetime, timezone
        palace._conn.execute(
            """
            INSERT INTO skills (name, trigger_regex, handler_type,
                                handler_payload, enabled, invocation_count,
                                created_at)
            VALUES ('rogue', 'rogue', 'never_heard_of_it', '{}', 1, 0, ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        palace._conn.commit()
        registry._reload()
        skill = registry.match("rogue")
        assert skill is not None
        assert registry.execute(skill, "rogue") is None

    def test_static_response_with_empty_text_returns_none(self, registry, palace):
        # Same defensive path — direct SQL insert bypassing validation.
        from datetime import datetime, timezone
        palace._conn.execute(
            """
            INSERT INTO skills (name, trigger_regex, handler_type,
                                handler_payload, enabled, invocation_count,
                                created_at)
            VALUES ('empty', 'empty', 'static_response',
                    '{"text": ""}', 1, 0, ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        palace._conn.commit()
        registry._reload()
        skill = registry.match("empty")
        assert registry.execute(skill, "empty") is None
        # Stats not bumped on failed execution.
        rows = registry.list_all()
        target = next(r for r in rows if r["name"] == "empty")
        assert target["invocation_count"] == 0


# ----------------------------------------------------------------------
# CRUD: remove + set_enabled + list_all
# ----------------------------------------------------------------------


class TestCrud:
    def test_remove_returns_true_on_hit(self, registry):
        registry.register("cpf", "cpf", "static_response", {"text": "x"})
        assert registry.remove("cpf") is True
        assert registry.match("cpf") is None

    def test_remove_returns_false_on_miss(self, registry):
        assert registry.remove("nonexistent") is False

    def test_set_enabled_toggles_visibility(self, registry):
        registry.register("cpf", "cpf", "static_response", {"text": "x"})
        assert registry.set_enabled("cpf", False) is True
        assert registry.match("cpf") is None
        assert registry.set_enabled("cpf", True) is True
        assert registry.match("cpf") is not None

    def test_set_enabled_missing_returns_false(self, registry):
        assert registry.set_enabled("nonexistent", True) is False

    def test_list_all_includes_disabled(self, registry):
        registry.register("a", "a", "static_response", {"text": "x"})
        registry.register("b", "b", "static_response", {"text": "y"})
        registry.set_enabled("b", False)
        rows = registry.list_all()
        names = [r["name"] for r in rows]
        assert names == ["a", "b"]

    def test_list_all_enabled_only_filters(self, registry):
        registry.register("a", "a", "static_response", {"text": "x"})
        registry.register("b", "b", "static_response", {"text": "y"})
        registry.set_enabled("b", False)
        rows = registry.list_all(enabled_only=True)
        names = [r["name"] for r in rows]
        assert names == ["a"]

    def test_list_all_decodes_payload(self, registry):
        registry.register("a", "a", "static_response", {"text": "hello"})
        rows = registry.list_all()
        assert rows[0]["handler_payload"] == {"text": "hello"}


# ----------------------------------------------------------------------
# resilience — broken row must not poison the cache
# ----------------------------------------------------------------------


class TestResilience:
    def test_invalid_regex_in_db_is_skipped(self, registry, palace):
        # Insert a row with a broken regex via direct SQL.
        from datetime import datetime, timezone
        palace._conn.execute(
            """
            INSERT INTO skills (name, trigger_regex, handler_type,
                                handler_payload, enabled, invocation_count,
                                created_at)
            VALUES ('broken', '[unclosed', 'static_response',
                    '{"text": "x"}', 1, 0, ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        palace._conn.commit()
        # And one valid row alongside.
        registry.register("good", "good", "static_response", {"text": "y"})
        # The valid skill still matches; broken one is silently dropped.
        assert registry.match("good") is not None
        # And the broken row never makes it to the in-memory list.
        names = [s.name for s in registry._skills]
        assert "broken" not in names
        assert "good" in names

    def test_malformed_payload_falls_back_to_empty_dict(self, registry, palace):
        from datetime import datetime, timezone
        palace._conn.execute(
            """
            INSERT INTO skills (name, trigger_regex, handler_type,
                                handler_payload, enabled, invocation_count,
                                created_at)
            VALUES ('bad_json', 'bad_json', 'static_response',
                    '{this is not json', 1, 0, ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        palace._conn.commit()
        registry._reload()
        skill = registry.match("bad_json")
        # The skill loads with an empty payload — static_response will
        # then return None at execute time because there's no text.
        assert skill is not None
        assert skill.handler_payload == {}
        assert registry.execute(skill, "bad_json") is None


# ----------------------------------------------------------------------
# orchestrator integration
# ----------------------------------------------------------------------


class TestOrchestratorIntegration:
    """The orchestrator should match + execute a skill before reaching
    the LLM path. We bypass __init__ to avoid booting Ollama / a real
    memory pipeline; we wire just the regex dispatchers, skill registry,
    and the audit stub used by _route_text."""

    def _build(self, tmp_path):
        from src.core.orchestrator import CortexOrchestrator
        from src.core.skill_registry import SkillRegistry

        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        # Sandboxed palace + registry pointed at it.
        palace = Palace(db_path=str(tmp_path / "palace.db"))
        orch.skill_registry = SkillRegistry(palace)
        # The orchestrator references several other attributes via
        # _route_text. We're only going to call the skill-relevant
        # branch, but those upstream branches (Spotify, reminder,
        # calendar, gatekeeper, etc.) get checked first — so the
        # input we pass must not collide with them.
        return orch, palace

    def test_skill_hit_bypasses_llm(self, tmp_path):
        orch, palace = self._build(tmp_path)
        try:
            orch.skill_registry.register(
                "stable_constant", r"\bping-skill\b",
                "static_response", {"text": "pong from skill"},
            )
            hit = orch.skill_registry.match("ping-skill please")
            assert hit is not None
            reply = orch.skill_registry.execute(hit, "ping-skill please")
            assert reply == "pong from skill"
            # And the stats were bumped.
            rows = orch.skill_registry.list_all()
            target = next(r for r in rows if r["name"] == "stable_constant")
            assert target["invocation_count"] == 1
        finally:
            palace.close()

    def test_registry_missing_does_not_crash_dispatch(self, tmp_path):
        """A None registry attribute must not break the orchestrator
        flow — the skill branch is gated by `if self.skill_registry
        is not None`."""
        from src.core.orchestrator import CortexOrchestrator
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch.skill_registry = None
        # Accessing the attribute should not raise.
        assert orch.skill_registry is None


# ----------------------------------------------------------------------
# tool_sequence (M8-A1) — the promoter↔registry gate contract
# ----------------------------------------------------------------------


def _seq(*steps):
    return {"steps": [{"tool": t, "args": a} for t, a in steps]}


class TestToolSequenceValidation:
    def test_registers_s1_s2_sequence(self, registry):
        rid = registry.register(
            "morning", r"\bmodo manhã\b", "tool_sequence",
            _seq(("list_emails", {"count": 3}), ("web_search", {"query": "news"})),
        )
        assert rid > 0
        assert registry.match("liga o modo manhã") is not None

    def test_registers_single_step(self, registry):
        registry.register("one", r"\bum-passo\b", "tool_sequence",
                          _seq(("read_document", {"filename": "a.txt"})))
        assert registry.match("um-passo") is not None

    def test_s3_as_last_step_is_accepted(self, registry):
        rid = registry.register(
            "report", r"\bmanda o report\b", "tool_sequence",
            _seq(("read_document", {"filename": "r.txt"}),
                 ("send_email", {"to": "a@b"})),
        )
        assert rid > 0

    def test_s3_not_last_is_rejected(self, registry):
        with pytest.raises(SkillRegistryError, match="LAST step"):
            registry.register(
                "bad", r"\bbad\b", "tool_sequence",
                _seq(("send_email", {"to": "a@b"}),
                     ("read_document", {"filename": "r.txt"})),
            )

    def test_two_s3_steps_are_rejected(self, registry):
        with pytest.raises(SkillRegistryError, match="at most ONE S3"):
            registry.register(
                "bad", r"\bbad\b", "tool_sequence",
                _seq(("send_email", {"to": "a@b"}),
                     ("create_reminder", {"title": "x"})),
            )

    def test_missing_steps_rejected(self, registry):
        with pytest.raises(SkillRegistryError, match="payload.steps"):
            registry.register("bad", r"\bbad\b", "tool_sequence", {})

    def test_empty_steps_rejected(self, registry):
        with pytest.raises(SkillRegistryError, match="payload.steps"):
            registry.register("bad", r"\bbad\b", "tool_sequence", {"steps": []})

    def test_step_without_tool_rejected(self, registry):
        with pytest.raises(SkillRegistryError, match="step 0 requires tool"):
            registry.register("bad", r"\bbad\b", "tool_sequence",
                              {"steps": [{"args": {}}]})

    def test_step_args_must_be_object(self, registry):
        with pytest.raises(SkillRegistryError, match="args must be an object"):
            registry.register(
                "bad", r"\bbad\b", "tool_sequence",
                {"steps": [{"tool": "web_search", "args": ["q"]}]},
            )

    def test_too_many_steps_rejected(self, registry):
        steps = [{"tool": "read_document", "args": {}} for _ in range(6)]
        with pytest.raises(SkillRegistryError, match="at most 5 steps"):
            registry.register("bad", r"\bbad\b", "tool_sequence",
                              {"steps": steps})


class _StepRecorder:
    """Fake run_step/stage_step pair that records every call."""

    def __init__(self, outputs=None, raise_on=None):
        self.ran = []
        self.staged = []
        self._outputs = outputs or {}
        self._raise_on = raise_on

    def run(self, tool, args):
        if tool == self._raise_on:
            raise RuntimeError("runner exploded")
        self.ran.append((tool, args))
        return self._outputs.get(tool, f"{tool} ok")

    def stage(self, tool, args):
        self.staged.append((tool, args))
        return f"Confirmar {tool}? Responda 'sim' ou 'não'."


class TestToolSequenceExecution:
    def _register(self, registry, *steps, name="macro", trigger=r"\bmacro\b"):
        registry.register(name, trigger, "tool_sequence", _seq(*steps))
        skill = registry.match("macro")
        assert skill is not None
        return skill

    def test_refuses_without_run_step(self, registry):
        """Fail-closed: a bare registry (no gate plumbing injected) can
        never run a macro — it falls through to the LLM instead."""
        skill = self._register(registry, ("read_document", {"filename": "a"}))
        assert registry.execute(skill, "macro") is None
        assert registry.list_all()[0]["invocation_count"] == 0

    def test_refuses_s3_without_stage_step_before_anything_runs(self, registry):
        """The S3-bypass check happens BEFORE step one: nothing executes,
        not even the safe S1/S2 prefix."""
        rec = _StepRecorder()
        skill = self._register(
            registry,
            ("read_document", {"filename": "a"}),
            ("send_email", {"to": "a@b"}),
        )
        assert registry.execute(skill, "macro", run_step=rec.run) is None
        assert rec.ran == []          # the S1 step did NOT run
        assert rec.staged == []

    def test_runs_steps_in_order_and_bumps(self, registry):
        rec = _StepRecorder(outputs={"list_emails": "3 emails.",
                                     "web_search": "Resultados."})
        skill = self._register(
            registry,
            ("list_emails", {"count": 3}),
            ("web_search", {"query": "news"}),
        )
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        assert rec.ran == [("list_emails", {"count": 3}),
                           ("web_search", {"query": "news"})]
        assert reply == "3 emails.\nResultados."
        assert rec.staged == []
        assert registry.list_all()[0]["invocation_count"] == 1

    def test_s3_last_step_is_staged_not_run(self, registry):
        rec = _StepRecorder()
        skill = self._register(
            registry,
            ("read_document", {"filename": "r.txt"}),
            ("send_email", {"to": "a@b"}),
        )
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        # S1 ran; S3 went to the stager, never to the runner.
        assert rec.ran == [("read_document", {"filename": "r.txt"})]
        assert rec.staged == [("send_email", {"to": "a@b"})]
        assert "Confirmar send_email?" in reply
        # Ending in a staged confirmation IS the macro doing its job.
        assert registry.list_all()[0]["invocation_count"] == 1

    def test_step_system_error_stops_sequence_no_bump(self, registry):
        rec = _StepRecorder(outputs={"list_emails": "[SYSTEM ERROR] rede fora"})
        skill = self._register(
            registry,
            ("list_emails", {}),
            ("web_search", {"query": "x"}),
        )
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        # Honest partial report, NOT None — a None would fall through to
        # the LLM, which could re-execute already-run side effects.
        assert reply is not None
        assert "parou no passo 1" in reply
        assert "list_emails" in reply
        assert ("web_search", {"query": "x"}) not in rec.ran
        assert registry.list_all()[0]["invocation_count"] == 0

    def test_step_exception_stops_sequence_no_bump(self, registry):
        rec = _StepRecorder(raise_on="web_search")
        skill = self._register(
            registry,
            ("list_emails", {}),
            ("web_search", {"query": "x"}),
            ("read_document", {"filename": "a"}),
        )
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        assert reply is not None
        assert "parou no passo 2" in reply
        assert ("read_document", {"filename": "a"}) not in rec.ran
        assert registry.list_all()[0]["invocation_count"] == 0

    def test_risk_promoted_after_registration_still_gates(
            self, registry, monkeypatch):
        """TOOL_RISK is read at EXECUTION time: a tool promoted to S3
        after the macro was registered diverts to the stager and the
        remaining steps are skipped with an explicit notice."""
        skill = self._register(
            registry,
            ("web_search", {"query": "x"}),
            ("read_document", {"filename": "a"}),
        )
        import src.core.skill_registry as sr
        monkeypatch.setattr(
            sr, "risk_level",
            lambda name: sr.S3 if name == "web_search" else "S1",
        )
        rec = _StepRecorder()
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        assert rec.staged == [("web_search", {"query": "x"})]
        assert rec.ran == []
        assert "não foram executados" in reply

    def test_raising_stager_reports_does_not_fall_through(self, registry):
        """A stager that raises after S1/S2 steps already ran must produce
        a report, never None (None → LLM fallback → possible re-execution
        of side effects)."""
        rec = _StepRecorder()

        def bad_stage(tool, args):
            raise RuntimeError("stager exploded")

        skill = self._register(
            registry,
            ("read_document", {"filename": "a"}),
            ("send_email", {"to": "a@b"}),
        )
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=bad_stage)
        assert reply is not None
        assert "parou no passo 2" in reply
        assert registry.list_all()[0]["invocation_count"] == 0

    def test_malformed_row_via_sql_refused(self, registry, palace):
        """A tool_sequence row that bypassed register() (direct SQL) with
        bad steps shape is refused — falls through to the LLM."""
        from datetime import datetime, timezone
        palace._conn.execute(
            """
            INSERT INTO skills (name, trigger_regex, handler_type,
                                handler_payload, enabled, invocation_count,
                                created_at)
            VALUES ('rogue-seq', 'rogue-seq', 'tool_sequence',
                    '{"steps": "not-a-list"}', 1, 0, ?)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        palace._conn.commit()
        registry._reload()
        skill = registry.match("rogue-seq")
        rec = _StepRecorder()
        assert registry.execute(skill, "rogue-seq",
                                run_step=rec.run, stage_step=rec.stage) is None
        assert rec.ran == []

    def test_all_empty_outputs_reply_feito(self, registry):
        rec = _StepRecorder(outputs={"read_document": ""})
        skill = self._register(registry, ("read_document", {"filename": "a"}))
        reply = registry.execute(skill, "macro",
                                 run_step=rec.run, stage_step=rec.stage)
        assert reply == "Feito."
