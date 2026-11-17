"""
Tests for the Plan-Execute module (Tier 3.1).

Layered:
  - Data model + validation (pure).
  - Topological order + cycle detection.
  - Ref resolution.
  - decompose() with mock LLM.
  - execute() with mock tool_executor.
  - summarise() with mock LLM + deterministic fallback.
  - Orchestrator integration (regex trigger, staging, dispatch).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.planner import (
    Plan,
    PlanResult,
    PlannerError,
    Step,
    StepResult,
    _DECOMPOSE_SYSTEM,
    _parse_plan_json,
    _resolve_refs,
    _topological_order,
    _validate_plan,
    decompose,
    execute,
    render_preview,
    summarise,
)


def _llm_returning(content: str):
    def fn(**kwargs):
        return {"message": {"content": content}}
    return fn


# ----------------------------------------------------------------------
# data model + validation
# ----------------------------------------------------------------------


class TestValidatePlan:
    def test_happy_path(self):
        plan = Plan(
            objetivo="x",
            passos=[
                Step("p1", "read_document", {"filename": "a.txt"}),
                Step("p2", "write_document",
                     {"filename": "b.txt", "content": "{p1.text}"},
                     depende_de=["p1"]),
            ],
        )
        _validate_plan(plan, allowed_tools=["read_document", "write_document"])

    def test_empty_plan_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[]))

    def test_duplicate_step_id_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[
                Step("p1", "read_document", {"filename": "a"}),
                Step("p1", "write_document", {"filename": "b", "content": "x"}),
            ]))

    def test_invalid_id_shape_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[
                Step("step1", "read_document", {"filename": "a"}),
            ]))

    def test_unknown_tool_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(
                Plan(objetivo="x", passos=[
                    Step("p1", "magic_tool", {"x": 1}),
                ]),
                allowed_tools=["read_document"],
            )

    def test_unknown_dependency_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[
                Step("p1", "read_document", {"filename": "a"},
                     depende_de=["p_unknown"]),
            ]))

    def test_ref_to_unknown_step_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[
                Step("p1", "write_document",
                     {"filename": "a", "content": "{p99.text}"}),
            ]))

    def test_self_ref_rejected(self):
        with pytest.raises(PlannerError):
            _validate_plan(Plan(objetivo="x", passos=[
                Step("p1", "read_document",
                     {"filename": "{p1.text}"}),
            ]))


# ----------------------------------------------------------------------
# topological order
# ----------------------------------------------------------------------


class TestTopologicalOrder:
    def test_linear_chain(self):
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}),
            Step("p2", "write_document", {"filename": "b", "content": "x"},
                 depende_de=["p1"]),
            Step("p3", "create_reminder", {"title": "x"}, depende_de=["p2"]),
        ])
        ordered = _topological_order(plan)
        assert [s.id for s in ordered] == ["p1", "p2", "p3"]

    def test_diamond(self):
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}),
            Step("p2", "read_document", {"filename": "b"}, depende_de=["p1"]),
            Step("p3", "read_document", {"filename": "c"}, depende_de=["p1"]),
            Step("p4", "write_document", {"filename": "d", "content": "x"},
                 depende_de=["p2", "p3"]),
        ])
        ordered = _topological_order(plan)
        ids = [s.id for s in ordered]
        assert ids[0] == "p1"
        assert ids[-1] == "p4"
        assert set(ids[1:3]) == {"p2", "p3"}

    def test_independent_steps_keep_declared_order(self):
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}),
            Step("p2", "read_document", {"filename": "b"}),
            Step("p3", "read_document", {"filename": "c"}),
        ])
        ordered = _topological_order(plan)
        assert [s.id for s in ordered] == ["p1", "p2", "p3"]

    def test_cycle_detected(self):
        # Manually construct a cycle (validation would normally catch this,
        # but topological_order is the canonical detector).
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}, depende_de=["p2"]),
            Step("p2", "read_document", {"filename": "b"}, depende_de=["p1"]),
        ])
        with pytest.raises(PlannerError) as exc:
            _topological_order(plan)
        assert "cycle" in str(exc.value).lower()


# ----------------------------------------------------------------------
# ref resolution
# ----------------------------------------------------------------------


class TestResolveRefs:
    def test_no_refs_pass_through(self):
        out = _resolve_refs({"a": "hello", "b": 42}, prior={})
        assert out == {"a": "hello", "b": 42}

    def test_string_output_substituted(self):
        prior = {
            "p1": StepResult("p1", success=True, output="document text"),
        }
        out = _resolve_refs({"content": "FROM: {p1.text}"}, prior=prior)
        assert out == {"content": "FROM: document text"}

    def test_dict_output_field_substituted(self):
        prior = {
            "p1": StepResult("p1", success=True,
                             output={"to": "alice@x", "subject": "hi"}),
        }
        out = _resolve_refs(
            {"recipient": "{p1.to}", "header": "Re: {p1.subject}"},
            prior=prior,
        )
        assert out == {"recipient": "alice@x", "header": "Re: hi"}

    def test_unresolved_ref_becomes_empty_string(self):
        out = _resolve_refs({"x": "{p1.field}"}, prior={})
        assert out == {"x": ""}

    def test_upstream_failed_ref_becomes_empty(self):
        prior = {"p1": StepResult("p1", success=False, output="error", error="x")}
        out = _resolve_refs({"x": "{p1}"}, prior=prior)
        assert out == {"x": ""}

    def test_bare_ref_to_string_output(self):
        prior = {"p1": StepResult("p1", success=True, output="bare value")}
        out = _resolve_refs({"x": "{p1}"}, prior=prior)
        assert out == {"x": "bare value"}

    def test_non_string_arg_passthrough(self):
        out = _resolve_refs({"count": 5, "flag": True, "list": [1, 2]}, prior={})
        assert out == {"count": 5, "flag": True, "list": [1, 2]}

    def test_field_on_string_output_falls_back_to_whole_string(self, caplog):
        """Defensive tolerance: the decompose prompt now tells the LLM to
        use whole-output {p1} refs only, so {p1.title} shouldn't appear.
        If it regresses and emits one anyway, we substitute the whole
        upstream string (don't drop data) and log a warning breadcrumb."""
        import logging
        prior = {
            "p1": StepResult("p1", success=True,
                             output="Brasil x França — 26/05 17h; Brasil x ..."),
        }
        with caplog.at_level(logging.WARNING, logger="src.core.planner"):
            out = _resolve_refs({"summary": "{p1.title}"}, prior=prior)
        # Field ignored → whole string substituted (current contract).
        assert out["summary"].startswith("Brasil x França")
        # ...and the mismatch was logged.
        assert any("requested a field" in r.message for r in caplog.records)

    def test_decompose_prompt_steers_to_whole_output_refs(self):
        """The decompose prompt must instruct {pN} whole-output refs and
        explicitly forbid .field accessors — this is the steering that
        keeps the LLM from emitting the {p1.title} footgun in the first
        place (tool outputs are plain text, not structured records)."""
        prompt = _DECOMPOSE_SYSTEM
        assert "{p<id>}" in prompt
        assert "plain TEXT" in prompt or "plain text" in prompt
        assert ".field" in prompt  # the explicit prohibition
        # The old template advertised "{p<id>.<field>}" as a valid form —
        # that must be gone so the model stops reaching for it.
        assert "{p<id>.<field>}" not in prompt

    def test_text_field_on_string_is_silent(self, caplog):
        """'.text' is the convention for 'the whole string output' — it
        must NOT warn (only non-'text' fields on a string do)."""
        import logging
        prior = {"p1": StepResult("p1", success=True, output="hello world")}
        with caplog.at_level(logging.WARNING, logger="src.core.planner"):
            out = _resolve_refs({"content": "{p1.text}"}, prior=prior)
        assert out == {"content": "hello world"}
        assert not any("requested a field" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# decompose — LLM-based plan generation
# ----------------------------------------------------------------------


class TestDecompose:
    def test_empty_objective_short_circuits(self):
        sentinel = MagicMock(side_effect=AssertionError("must not call"))
        assert decompose("", sentinel, "m", ["read_document"]) is None
        assert decompose("   ", sentinel, "m", ["read_document"]) is None

    def test_empty_allowed_tools_returns_none(self):
        sentinel = MagicMock(side_effect=AssertionError("must not call"))
        assert decompose("x", sentinel, "m", []) is None

    def test_happy_path(self):
        llm = _llm_returning(
            '{"objetivo":"x","passos":['
            '{"id":"p1","tool":"read_document","args":{"filename":"a.txt"},'
            '"depende_de":[],"descricao":"read a"}'
            ']}'
        )
        plan = decompose("read a.txt", llm, "m", ["read_document"])
        assert plan is not None
        assert plan.objetivo == "x"
        assert len(plan.passos) == 1
        assert plan.passos[0].tool == "read_document"

    def test_multi_step_with_ref(self):
        llm = _llm_returning(
            '{"objetivo":"copy","passos":['
            '{"id":"p1","tool":"read_document","args":{"filename":"a.txt"},'
            '"depende_de":[],"descricao":""},'
            '{"id":"p2","tool":"write_document",'
            '"args":{"filename":"b.txt","content":"{p1.text}"},'
            '"depende_de":["p1"],"descricao":""}'
            ']}'
        )
        plan = decompose("copy a to b", llm, "m",
                         ["read_document", "write_document"])
        assert plan is not None
        assert len(plan.passos) == 2
        assert plan.passos[1].args["content"] == "{p1.text}"

    def test_malformed_json_returns_none(self):
        llm = _llm_returning('{this is not json')
        assert decompose("x", llm, "m", ["read_document"]) is None

    def test_no_json_returns_none(self):
        llm = _llm_returning("I'm sorry, I cannot decompose this.")
        assert decompose("x", llm, "m", ["read_document"]) is None

    def test_disallowed_tool_returns_none(self):
        # The plan uses a tool not in allowed_tools — validation rejects it.
        llm = _llm_returning(
            '{"objetivo":"x","passos":['
            '{"id":"p1","tool":"hack_the_planet","args":{},"depende_de":[]}'
            ']}'
        )
        assert decompose("x", llm, "m", ["read_document"]) is None

    def test_llm_exception_returns_none(self):
        def boom(**kw):
            raise RuntimeError("ollama down")
        assert decompose("x", boom, "m", ["read_document"]) is None

    def test_temperature_zero_passed(self):
        captured = {}
        def fn(**kw):
            captured.update(kw)
            return {"message": {"content": '{"objetivo":"x","passos":[]}'}}
        decompose("x", fn, "m", ["read_document"])
        assert captured["options"]["temperature"] == 0.0


class TestParsePlanJson:
    @pytest.mark.parametrize("body", ["{p1}", "Result: {p1.text} / {p1}"])
    @pytest.mark.parametrize("dependencies", [[], ["p1"]])
    def test_arg_refs_supply_dependencies(self, body, dependencies):
        content = json.dumps({"objetivo": "x", "passos": [
            {"id": "p2", "tool": "send_email",
             "args": {"to": "a@example.com", "subject": "s", "body": body},
             "depende_de": dependencies},
            {"id": "p1", "tool": "web_search", "args": {"query": "q"},
             "depende_de": []},
        ]})
        plan = _parse_plan_json(content, ["send_email", "web_search"])
        assert plan is not None
        assert plan.passos[0].depende_de == ["p1"]
        assert [s.id for s in _topological_order(plan)] == ["p1", "p2"]

    def test_extracts_outermost_object(self):
        content = (
            "Here's the plan:\n"
            '{"objetivo":"x","passos":['
            '{"id":"p1","tool":"read_document","args":{"filename":"a"},"depende_de":[]}'
            ']}\n'
            "Hope this helps."
        )
        plan = _parse_plan_json(content, ["read_document"])
        assert plan is not None
        assert plan.passos[0].id == "p1"

    def test_returns_none_for_array(self):
        # Top-level array, not object — invalid for plans.
        assert _parse_plan_json("[]", ["read_document"]) is None

    def test_empty_passos_validates_to_none(self):
        # Empty plan fails _validate_plan → returns None.
        assert _parse_plan_json('{"objetivo":"x","passos":[]}',
                                ["read_document"]) is None


# ----------------------------------------------------------------------
# execute — sequential runner
# ----------------------------------------------------------------------


class TestExecute:
    def _make_plan(self, *steps):
        return Plan(objetivo="x", passos=list(steps))

    def test_single_step_success(self):
        plan = self._make_plan(
            Step("p1", "read_document", {"filename": "a.txt"}),
        )
        result = execute(plan, tool_executor=lambda n, a: "file contents")
        assert result.success
        assert len(result.results) == 1
        assert result.results[0].success
        assert result.results[0].output == "file contents"

    def test_multi_step_with_ref_resolved(self):
        plan = self._make_plan(
            Step("p1", "read_document", {"filename": "a.txt"}),
            Step("p2", "write_document",
                 {"filename": "b.txt", "content": "PREFIX: {p1.text}"},
                 depende_de=["p1"]),
        )
        captured_args = []
        def tool_exec(name, args):
            captured_args.append(args)
            return f"ran {name}"
        result = execute(plan, tool_executor=tool_exec)
        assert result.success
        # The write_document call got the resolved content
        assert captured_args[1]["content"] == "PREFIX: ran read_document"

    @pytest.mark.parametrize("ref", ["{p1}", "{p1.text}"])
    def test_unresolved_ref_fails_before_calling_tool(self, ref):
        plan = self._make_plan(
            Step("p2", "send_email",
                 {"to": "a@example.com", "subject": "s", "body": ref}),
            Step("p1", "web_search", {"query": "q"}),
        )
        calls = []
        seen = []
        def tool_exec(name, args):
            calls.append((name, args))
            return "ok"
        result = execute(plan, tool_executor=tool_exec,
                         on_step=lambda step, res: seen.append(res))
        assert calls == []
        assert not result.success
        assert not result.results[0].success
        assert not result.results[0].skipped
        assert result.results[0].error == f"unresolved reference {ref}"
        assert result.error == result.results[0].error
        assert result.results[1].skipped
        assert seen == result.results

    def test_declared_dependency_sends_resolved_body(self):
        plan = self._make_plan(
            Step("p2", "send_email", {"body": "{p1}"}, depende_de=["p1"]),
            Step("p1", "web_search", {"query": "q"}),
        )
        calls = []
        def tool_exec(name, args):
            calls.append((name, args))
            return "search result" if name == "web_search" else "sent"
        result = execute(plan, tool_executor=tool_exec)
        assert result.success
        assert all(res.success for res in result.results)
        assert calls == [("web_search", {"query": "q"}),
                         ("send_email", {"body": "search result"})]

    def test_fail_stop_skips_remaining(self):
        plan = self._make_plan(
            Step("p1", "read_document", {"filename": "a"}),
            Step("p2", "write_document", {"filename": "b", "content": "x"},
                 depende_de=["p1"]),
            Step("p3", "create_reminder", {"title": "x"}, depende_de=["p2"]),
        )
        def tool_exec(name, args):
            if name == "write_document":
                return "[SYSTEM ERROR] disk full"
            return "ok"
        result = execute(plan, tool_executor=tool_exec)
        assert not result.success
        assert result.results[0].success           # p1 ran
        assert not result.results[1].success       # p2 failed
        assert "disk full" in result.results[1].error
        assert result.results[2].skipped            # p3 skipped

    def test_tool_executor_exception_caught(self):
        plan = self._make_plan(
            Step("p1", "read_document", {"filename": "a"}),
        )
        def tool_exec(name, args):
            raise RuntimeError("simulated crash")
        result = execute(plan, tool_executor=tool_exec)
        assert not result.success
        assert "simulated crash" in result.results[0].error

    def test_on_step_callback_called(self):
        plan = self._make_plan(
            Step("p1", "read_document", {"filename": "a"}),
            Step("p2", "create_reminder", {"title": "x"}, depende_de=["p1"]),
        )
        seen = []
        def on_step(step, result):
            seen.append((step.id, result.success))
        execute(plan, tool_executor=lambda n, a: "ok", on_step=on_step)
        assert seen == [("p1", True), ("p2", True)]

    def test_cyclic_plan_returns_failure_without_executing(self):
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}, depende_de=["p2"]),
            Step("p2", "read_document", {"filename": "b"}, depende_de=["p1"]),
        ])
        called = []
        def tool_exec(name, args):
            called.append(name)
            return "ok"
        result = execute(plan, tool_executor=tool_exec)
        assert not result.success
        assert called == []  # no tool ever ran


# ----------------------------------------------------------------------
# summarise
# ----------------------------------------------------------------------


class TestSummarise:
    def _make_result(self, success=True, error="", n_skipped=0):
        plan = Plan(objetivo="x", passos=[
            Step("p1", "read_document", {"filename": "a"}, descricao="read"),
            Step("p2", "write_document", {"filename": "b", "content": "x"},
                 depende_de=["p1"], descricao="write"),
        ])
        results = [
            StepResult("p1", success=True, output="hello"),
            StepResult("p2", success=success,
                       output="" if not success else "ok",
                       error=error, skipped=False),
        ]
        return PlanResult(plan=plan, results=results,
                          success=success, error=error)

    def test_llm_used_when_provided(self):
        pr = self._make_result()
        llm = _llm_returning("Plano concluído. Li o arquivo e escrevi a cópia.")
        out = summarise(pr, llm_chat_fn=llm, model="m")
        assert "concluído" in out.lower()

    def test_deterministic_fallback_when_no_llm(self):
        pr = self._make_result()
        out = summarise(pr, llm_chat_fn=None, model=None)
        # Each step appears with its descricao
        assert "read" in out
        assert "write" in out

    def test_llm_exception_falls_back(self):
        pr = self._make_result()
        def boom(**kw):
            raise RuntimeError("ollama down")
        out = summarise(pr, llm_chat_fn=boom, model="m")
        assert "read" in out and "write" in out

    def test_failed_step_visible_in_fallback(self):
        pr = self._make_result(success=False, error="disk full")
        out = summarise(pr, llm_chat_fn=None, model=None)
        assert "disk full" in out


# ----------------------------------------------------------------------
# render_preview
# ----------------------------------------------------------------------


class TestRenderPreview:
    def test_preview_includes_steps_and_args(self):
        plan = Plan(objetivo="copiar a.txt pra b.txt", passos=[
            Step("p1", "read_document", {"filename": "a.txt"}, descricao="ler"),
            Step("p2", "write_document",
                 {"filename": "b.txt", "content": "{p1.text}"},
                 depende_de=["p1"], descricao="escrever"),
        ])
        out = render_preview(plan)
        assert "copiar a.txt" in out
        assert "[read_document]" in out and "[write_document]" in out
        assert "depende de p1" in out
        assert "{p1.text}" in out
        assert "'sim'" in out and "'não'" in out

    def test_a_long_argument_is_shown_whole(self):
        """The plan-level "sim" sends every byte of it: the cut at 80 chars
        hid the end of an email body (s88, Codex lote 2)."""
        body = "segue o resumo da reunião de ontem. " * 4 + "PS: o que vem depois"
        plan = Plan(objetivo="avisar a Ana", passos=[
            Step("p1", "send_email", {"to": "ana@x.com", "subject": "R", "body": body},
                 descricao="enviar"),
        ])
        assert f"body: {body}\n" in render_preview(plan)

    def test_empty_plan_renders_safe(self):
        plan = Plan(objetivo="nothing", passos=[])
        out = render_preview(plan)
        assert "vazio" in out.lower()
        # No "sim/não" prompt since there's nothing to execute
        assert "'sim'" not in out


# ----------------------------------------------------------------------
# i18n (M1-A8) — the planner was PT-only end to end; lang now threads
# through preview/summarise and rides with the staged plan.
# ----------------------------------------------------------------------


class TestPlannerI18n:
    def _plan(self):
        return Plan(objetivo="copy a.txt to b.txt", passos=[
            Step("p1", "read_document", {"filename": "a.txt"}, descricao="read"),
            Step("p2", "write_document",
                 {"filename": "b.txt", "content": "{p1.text}"},
                 depende_de=["p1"], descricao="write"),
        ])

    def _result(self):
        plan = self._plan()
        return PlanResult(plan=plan, results=[
            StepResult("p1", success=True, output="hello"),
            StepResult("p2", success=False, error="disk full"),
        ], success=False, error="disk full")

    def test_preview_en(self):
        out = render_preview(self._plan(), lang="en")
        assert "Plan for: copy a.txt" in out
        assert "(depends on p1)" in out
        assert "Reply 'yes' to run the plan or 'no' to cancel." in out
        assert "Responda" not in out and "depende" not in out

    def test_preview_default_stays_pt(self):
        out = render_preview(self._plan())
        assert "Plano para:" in out and "'sim'" in out

    def test_empty_plan_en(self):
        out = render_preview(Plan(objetivo="nothing", passos=[]), lang="en")
        assert "Empty plan" in out and "Nothing to execute" in out

    def test_deterministic_summary_en(self):
        out = summarise(self._result(), llm_chat_fn=None, model=None, lang="en")
        assert out.startswith("Plan: copy a.txt")
        assert "disk full" in out
        assert "Plano" not in out

    def test_summarise_en_uses_en_system_prompt(self):
        from src.core.planner import _SUMMARISE_SYSTEM_EN
        seen = {}
        def llm(**kw):
            seen["system"] = kw["messages"][0]["content"]
            seen["user"] = kw["messages"][1]["content"]
            return {"message": {"content": "Done. Read the file and wrote the copy."}}
        out = summarise(self._result(), llm_chat_fn=llm, model="m", lang="en")
        assert seen["system"] == _SUMMARISE_SYSTEM_EN
        # The LLM-facing body is monolingual EN too.
        assert "Objective:" in seen["user"] and "Steps:" in seen["user"]
        assert ": FAILED — disk full" in seen["user"]
        assert "Done." in out

    def test_summarise_pt_keeps_pt_prompt(self):
        from src.core.planner import _SUMMARISE_SYSTEM
        seen = {}
        def llm(**kw):
            seen["system"] = kw["messages"][0]["content"]
            return {"message": {"content": "Feito."}}
        summarise(self._result(), llm_chat_fn=llm, model="m")
        assert seen["system"] == _SUMMARISE_SYSTEM


# ----------------------------------------------------------------------
# orchestrator integration
# ----------------------------------------------------------------------


class TestOrchestratorIntegration:

    def _build(self, tmp_path, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator
        from src.memory.palace import Palace

        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._pending_action = None
        # Minimal stubs the planner path touches:
        orch.engine_analytical = MagicMock()
        orch.engine_analytical.model_name = "cortex-lite"
        # In-memory palace for any incidental Palace use.
        orch._palace = Palace(db_path=":memory:")
        return orch

    def test_trigger_regex_matches_planeja_prefix(self):
        from src.core.orchestrator import CortexOrchestrator
        m = CortexOrchestrator._PLANNER_TRIGGER_RE.match(
            "planeja: agendar reunião amanhã às 14h e me lembrar 30 min antes"
        )
        assert m is not None
        assert "agendar reunião" in m.group("objective")

    @pytest.mark.parametrize("text,expected", [
        ("planeja: copiar a.txt pra b.txt",  "copiar a.txt pra b.txt"),
        ("planejar copiar arquivos",          "copiar arquivos"),
        ("decompõe: criar evento e lembrete", "criar evento e lembrete"),
        ("plan: book a meeting",              "book a meeting"),
        ("make a plan for the report",        "the report"),
        ("monte um plano para a semana",      "a semana"),
    ])
    def test_trigger_variants(self, text, expected):
        from src.core.orchestrator import CortexOrchestrator
        m = CortexOrchestrator._PLANNER_TRIGGER_RE.match(text)
        assert m is not None
        assert m.group("objective").strip() == expected

    @pytest.mark.parametrize("text", [
        "qual o clima hoje",
        "me lembre da reunião",
        "toca matuê",
        "qual seu plano de vida",  # 'plano' without trigger verb
    ])
    def test_trigger_does_not_overmatch(self, text):
        from src.core.orchestrator import CortexOrchestrator
        assert CortexOrchestrator._PLANNER_TRIGGER_RE.match(text) is None

    def test_handle_stages_plan_in_pending_action(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch)

        # Mock ollama.chat so decompose() gets a valid plan
        plan_json = (
            '{"objetivo":"copiar e lembrar","passos":['
            '{"id":"p1","tool":"read_document","args":{"filename":"a.txt"},'
            '"depende_de":[],"descricao":"ler a.txt"},'
            '{"id":"p2","tool":"create_reminder",'
            '"args":{"title":"revisar a.txt"},'
            '"depende_de":["p1"],"descricao":"lembrete"}'
            ']}'
        )
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": plan_json}}
            result = orch._handle_planner_request("copiar e lembrar")

        assert result is not None
        reply, mode = result
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "Plano para" in reply
        assert orch._pending_action is not None
        assert orch._pending_action["type"] == "execute_plan"
        passos = orch._pending_action["args"]["passos"]
        assert len(passos) == 2
        assert passos[0]["tool"] == "read_document"

    def test_handle_returns_none_on_decompose_failure(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch)
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": "not json"}}
            result = orch._handle_planner_request("something")
        assert result is None
        assert orch._pending_action is None

    def test_handle_returns_none_on_empty_objective(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch)
        assert orch._handle_planner_request("") is None
        assert orch._handle_planner_request("   ") is None

    def test_execute_staged_plan_runs_all_steps(self, tmp_path, monkeypatch):
        """End-to-end: stage a plan, then 'sim' fires _execute_staged_plan
        which runs each step via _execute_tool_unstaged."""
        orch = self._build(tmp_path, monkeypatch)
        # Stub the writer so write_document doesn't touch the disk.
        orch.writer = MagicMock()
        orch.writer.write_document.return_value = "Information successfully saved to 'b.txt'."
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "file contents from a"

        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "copy a to b",
                "passos": [
                    {"id": "p1", "tool": "read_document",
                     "args": {"filename": "a.txt"},
                     "depende_de": [], "descricao": "ler"},
                    {"id": "p2", "tool": "write_document",
                     "args": {"filename": "b.txt", "content": "{p1.text}"},
                     "depende_de": ["p1"], "descricao": "escrever"},
                ],
            },
        }

        # Stub the summariser LLM to return a known string
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": "Plano executado com sucesso."}}
            reply, mode = orch._execute_staged_plan()

        assert mode == "AGENT_ACTION_DONE"
        # The writer received the RESOLVED content from p1's output
        orch.writer.write_document.assert_called_once()
        called_filename, called_content = orch.writer.write_document.call_args[0]
        assert called_filename == "b.txt"
        assert called_content == "file contents from a"

    def test_execute_staged_plan_summary_fallback(self, tmp_path, monkeypatch):
        """If the summariser LLM fails, the user still gets a
        deterministic summary."""
        orch = self._build(tmp_path, monkeypatch)
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "ok"

        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "ler a",
                "passos": [
                    {"id": "p1", "tool": "read_document",
                     "args": {"filename": "a.txt"},
                     "depende_de": [], "descricao": "ler arquivo a"},
                ],
            },
        }
        # `_execute_staged_plan` lives on StagingMixin (phase 3B), so
        # the summariser `ollama.chat` resolves through
        # `src.core.staging_mixin.ollama` now, not the
        # orchestrator module.
        with patch("src.core.staging_mixin.ollama") as ollama_mock:
            ollama_mock.chat.side_effect = RuntimeError("ollama down")
            reply, mode = orch._execute_staged_plan()
        assert mode == "AGENT_ACTION_DONE"
        # Deterministic fallback includes the step description
        assert "ler arquivo a" in reply

    def test_execute_staged_plan_handles_step_failure(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch)
        orch.writer = MagicMock()
        orch.writer.write_document.return_value = "[SYSTEM ERROR] Refused"

        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "fail",
                "passos": [
                    {"id": "p1", "tool": "write_document",
                     "args": {"filename": "x", "content": "y"},
                     "depende_de": [], "descricao": "x"},
                ],
            },
        }
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": "Falhou."}}
            reply, mode = orch._execute_staged_plan()
        assert mode == "AGENT_ACTION_ERROR"

    def test_execute_pending_action_routes_execute_plan(self, tmp_path, monkeypatch):
        """Calling _execute_pending_action with type='execute_plan'
        delegates to _execute_staged_plan."""
        orch = self._build(tmp_path, monkeypatch)
        orch._persist_pending_action = lambda: None  # silence file IO
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "content"

        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "ler",
                "passos": [
                    {"id": "p1", "tool": "read_document",
                     "args": {"filename": "a"},
                     "depende_de": [], "descricao": ""},
                ],
            },
        }
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": "done."}}
            reply, mode = orch._execute_pending_action()
        assert mode == "AGENT_ACTION_DONE"
        # And the pending action got cleared
        assert orch._pending_action is None


# ----------------------------------------------------------------------
# Full E2E via process() — turn 1 stages plan, turn 2 "sim" executes
# ----------------------------------------------------------------------


class TestPlannerEndToEnd:
    """Drives the orchestrator's PUBLIC `process()` method just like the
    HTTP API would, simulating the two-turn conversation:

      turn 1: user says "planeja: ...", server returns preview
              + AGENT_AWAITING_CONFIRMATION; _pending_action is set.
      turn 2: user says "sim", server runs the plan via
              _execute_tool_unstaged and returns the LLM-aggregated
              summary + AGENT_ACTION_DONE.

    Closes the smoke-E2E item from the resume point. Pedro asked for a
    test that hits the same path the iOS Shortcut would, not just the
    handler methods in isolation.
    """

    def _orch(self, tmp_path, monkeypatch):
        """Build a fully-functional orchestrator without booting Ollama
        or any of the integrations the planner path touches. Returns
        the orchestrator with reader/writer/spotify stubbed so the
        plan can execute end-to-end."""
        from src.core.orchestrator import CortexOrchestrator
        from src.memory.palace import Palace

        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)

        # Engines (only model_name is read on the planner path)
        for eng_attr in ("engine_analytical", "engine_creative"):
            eng = MagicMock()
            eng.model_name = "cortex-lite"
            setattr(orch, eng_attr, eng)

        # Memory + queues
        from src.core.studies_review import StudiesReview
        from src.core.study_queue import StudyQueue
        from src.tools.spotify_player import SpotifyPlayer
        orch._palace = Palace(db_path=":memory:")
        orch.study_queue = StudyQueue(path=str(tmp_path / "study_queue.json"))
        orch.studies_review = StudiesReview()
        orch.spotify = SpotifyPlayer()
        orch.skill_registry = None  # bypass

        # Pending state
        orch._pending_action = None
        orch._pending_query = None
        orch._pending_engine = None

        # Reader / writer for read_document / write_document tools
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "conteúdo de a.txt"
        orch.writer = MagicMock()
        orch.writer.write_document.return_value = "Information successfully saved to 'b.txt'."

        # Vision + searcher: not used by the planner path
        orch.vision = MagicMock()
        orch.web_searcher = MagicMock()

        # Gatekeeper + caches + lease
        from src.core.gatekeeper import LogicGate
        orch.gatekeeper = LogicGate()
        orch.lease_hours = 4
        from collections import OrderedDict
        orch._intent_cache = OrderedDict()
        return orch

    def test_full_round_trip_via_process(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)

        # The plan the LLM will "decompose" — a realistic 2-step request.
        plan_json = (
            '{"objetivo":"copiar a.txt para b.txt","passos":['
            '{"id":"p1","tool":"read_document",'
            '"args":{"filename":"a.txt"},'
            '"depende_de":[],"descricao":"ler a.txt"},'
            '{"id":"p2","tool":"write_document",'
            '"args":{"filename":"b.txt","content":"{p1}"},'
            '"depende_de":["p1"],"descricao":"escrever b.txt"}'
            ']}'
        )

        # Two patches because turn 1 (decompose via _handle_planner_request)
        # uses ollama from the orchestrator module while turn 2 (summarise
        # via _execute_staged_plan) uses ollama from staging_mixin
        # after phase 3B. The mocks need independent side_effects: each
        # mock consumes its own list, so duplicating the joined sequence
        # would feed the summariser the plan_json instead of the reply.
        with patch("src.core.orchestrator.ollama") as ollama_mock, \
             patch("src.core.staging_mixin.ollama") as staging_ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": plan_json}}
            staging_ollama_mock.chat.return_value = {
                "message": {"content": "Pronto. Copiei a.txt para b.txt."}
            }

            # ===== TURN 1: user says "planeja: ..." =====
            reply1, mode1 = orch.process(
                "planeja: copiar o conteúdo de a.txt para b.txt"
            )
            assert mode1 == "AGENT_AWAITING_CONFIRMATION", reply1
            assert "Plano para" in reply1
            assert "[read_document]" in reply1
            assert "[write_document]" in reply1
            assert "'sim'" in reply1 and "'não'" in reply1
            # Pending action is staged and persisted to the tmp file
            assert orch._pending_action is not None
            assert orch._pending_action["type"] == "execute_plan"

            # ===== TURN 2: user says "sim" =====
            reply2, mode2 = orch.process("sim")
            assert mode2 == "AGENT_ACTION_DONE", reply2
            assert "Copiei" in reply2 or "Pronto" in reply2
            # Pending action cleared
            assert orch._pending_action is None

            # Verify the tools actually fired with resolved refs.
            orch.reader.read_document.assert_called_once_with("a.txt")
            orch.writer.write_document.assert_called_once()
            wd_args = orch.writer.write_document.call_args[0]
            assert wd_args[0] == "b.txt"
            # {p1} ref was resolved to p1's output ("conteúdo de a.txt")
            assert wd_args[1] == "conteúdo de a.txt"

    def test_turn_2_denial_cancels_without_executing(self, tmp_path, monkeypatch):
        """Same plan staging, but user replies 'não' on turn 2 — no
        tool should fire, the pending action clears, and the user
        sees a cancel message."""
        orch = self._orch(tmp_path, monkeypatch)

        plan_json = (
            '{"objetivo":"ler","passos":['
            '{"id":"p1","tool":"read_document",'
            '"args":{"filename":"a.txt"},'
            '"depende_de":[],"descricao":"ler"}'
            ']}'
        )
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": plan_json}}
            reply1, mode1 = orch.process("planeja: ler a.txt")
            assert mode1 == "AGENT_AWAITING_CONFIRMATION"

        # Now deny. No more ollama calls.
        reply2, mode2 = orch.process("não")
        assert mode2 == "AGENT_ACTION_CANCELLED"
        assert orch._pending_action is None
        orch.reader.read_document.assert_not_called()


# ----------------------------------------------------------------------
# web_search inside a plan (M1-A1, review Fable 5)
# ----------------------------------------------------------------------
# The _execute_tool_unstaged web_search branch was MISSING from s15 until
# the Fable 5 review: every approved plan containing a web_search step —
# including the auto-multi-intent motivator "busca os jogos e adiciona ao
# calendário" — died as "[SYSTEM ERROR] Tool desconhecido no plano",
# lease or no lease. These tests pin the branch: lease active → the
# search runs; lease inactive → the speakable lease refusal (which
# planner._ERROR_MARKERS recognises, so the plan fail-stops cleanly).


class TestPlanWebSearch:

    def _orch(self, tmp_path, monkeypatch, lease_active):
        helper = TestPlannerEndToEnd()
        orch = helper._orch(tmp_path, monkeypatch)
        orch.gatekeeper = MagicMock()
        orch.gatekeeper.is_lease_active.return_value = lease_active
        orch.web_searcher = MagicMock()
        orch.web_searcher.search.return_value = (
            "Próximo jogo da seleção: sábado 16h."
        )
        return orch

    # -- unit: the branch itself --

    def test_unstaged_lease_active_runs_search(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch, lease_active=True)
        out = orch._execute_tool_unstaged(
            "web_search", {"query": "jogos da seleção"}
        )
        orch.web_searcher.search.assert_called_once_with("jogos da seleção")
        assert "seleção" in out

    def test_unstaged_lease_inactive_refuses_without_searching(
        self, tmp_path, monkeypatch
    ):
        orch = self._orch(tmp_path, monkeypatch, lease_active=False)
        out = orch._execute_tool_unstaged(
            "web_search", {"query": "jogos da seleção"}
        )
        orch.web_searcher.search.assert_not_called()
        # The exact substring planner._ERROR_MARKERS keys on:
        assert "Web search is unavailable" in out
        # The pre-fix symptom must never come back:
        assert "Tool desconhecido" not in out

    def test_unstaged_empty_query_is_error(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch, lease_active=True)
        out = orch._execute_tool_unstaged("web_search", {})
        orch.web_searcher.search.assert_not_called()
        assert "query is required" in out

    # -- E2E via process(): stage on turn 1, "sim" executes on turn 2 --

    _PLAN_JSON = (
        '{"objetivo":"buscar jogos e anotar","passos":['
        '{"id":"p1","tool":"web_search",'
        '"args":{"query":"jogos da seleção"},'
        '"depende_de":[],"descricao":"buscar jogos"},'
        '{"id":"p2","tool":"write_document",'
        '"args":{"filename":"jogos.txt","content":"{p1}"},'
        '"depende_de":["p1"],"descricao":"anotar"}'
        ']}'
    )

    def _stage_and_approve(self, orch):
        with patch("src.core.orchestrator.ollama") as ollama_mock, \
             patch("src.core.staging_mixin.ollama") as staging_ollama_mock:
            ollama_mock.chat.return_value = {
                "message": {"content": self._PLAN_JSON}
            }
            staging_ollama_mock.chat.return_value = {
                "message": {"content": "Busquei e anotei em jogos.txt."}
            }
            reply1, mode1 = orch.process("planeja: busca os jogos e anota")
            assert mode1 == "AGENT_AWAITING_CONFIRMATION", reply1
            assert "[web_search]" in reply1
            return orch.process("sim")

    def test_e2e_plan_with_lease_executes_search(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch, lease_active=True)
        reply2, mode2 = self._stage_and_approve(orch)
        assert mode2 == "AGENT_ACTION_DONE", reply2
        orch.web_searcher.search.assert_called_once_with("jogos da seleção")
        # {p1} ref resolved: the search output flowed into the document.
        wd_args = orch.writer.write_document.call_args[0]
        assert wd_args[1] == "Próximo jogo da seleção: sábado 16h."

    def test_e2e_plan_without_lease_fail_stops_cleanly(
        self, tmp_path, monkeypatch
    ):
        orch = self._orch(tmp_path, monkeypatch, lease_active=False)
        reply2, mode2 = self._stage_and_approve(orch)
        assert mode2 == "AGENT_ACTION_ERROR", reply2
        orch.web_searcher.search.assert_not_called()
        # Fail-stop: the dependent step never ran on a failed search.
        orch.writer.write_document.assert_not_called()
        assert orch._pending_action is None


# ----------------------------------------------------------------------
# M1-B2/M4-B1 (s66) — extraction flag (dormant) + {"text","fields"} convention
# ----------------------------------------------------------------------


class TestRunnerConventionRefs:
    """The {"text": str, "fields": dict} runner convention (M4-B1):
    surfaces read text, refs extract typed fields. _resolve_refs must
    understand it TODAY so converting runners on swap day needs zero
    planner changes."""

    _OUT = {
        "text": "Brasil x Chile, 13/06 às 19h, Maracanã",
        "fields": {"title": "Brasil x Chile",
                   "start_iso": "2026-06-13T19:00:00-03:00"},
    }

    def _prior(self):
        return {"p1": StepResult("p1", success=True, output=dict(self._OUT))}

    def test_whole_ref_yields_text_not_json_dump(self):
        out = _resolve_refs({"body": "{p1}"}, prior=self._prior())
        assert out == {"body": self._OUT["text"]}

    def test_field_ref_extracts_from_fields(self):
        out = _resolve_refs({"start_iso": "{p1.start_iso}",
                             "summary": "{p1.title}"}, prior=self._prior())
        assert out == {"start_iso": "2026-06-13T19:00:00-03:00",
                       "summary": "Brasil x Chile"}

    def test_field_ref_falls_back_to_top_level(self):
        # A non-convention dict (no "fields") keeps the old behaviour.
        prior = {"p1": StepResult("p1", success=True,
                                  output={"to": "alice@x"})}
        out = _resolve_refs({"r": "{p1.to}"}, prior=prior)
        assert out == {"r": "alice@x"}

    def test_missing_field_is_empty_string(self):
        out = _resolve_refs({"x": "{p1.nope}"}, prior=self._prior())
        assert out == {"x": ""}


class TestExtractionFlagDormant:
    def _system_prompt(self, monkeypatch, flag: str | None):
        """Run decompose with a capturing fake LLM; return the system
        prompt it was given."""
        from src.core.planner import decompose
        if flag is None:
            monkeypatch.delenv("CORTEX_PLANNER_EXTRACTION", raising=False)
        else:
            monkeypatch.setenv("CORTEX_PLANNER_EXTRACTION", flag)
        captured = {}

        def fake_chat(model, messages, **kw):
            captured["system"] = messages[0]["content"]
            return {"message": {"content":
                    '{"objetivo":"x","passos":[]}'}}

        decompose("busca e agenda", fake_chat, "m",
                  allowed_tools=["web_search", "create_event"])
        return captured["system"]

    def test_default_off_keeps_extraction_limit(self, monkeypatch):
        system = self._system_prompt(monkeypatch, None)
        assert "CRITICAL EXTRACTION LIMIT" in system
        # Example 3 is still the honesty-refusal one (it cites the
        # forbidden ref only to forbid it).
        assert 'NEVER write start_iso="{p1.start_iso}"' in system
        assert '"start_iso":"{p1.start_iso}"' not in system

    def test_flag_on_allows_field_refs_and_chains_example3(self, monkeypatch):
        system = self._system_prompt(monkeypatch, "1")
        assert "CRITICAL EXTRACTION LIMIT" not in system
        assert "{p<id>.<field>}" in system
        # Example 3 flips from honesty-refusal to the chained plan.
        assert '"start_iso":"{p1.start_iso}"' in system
        assert "never fabricate a step" in system.lower()

    def test_variant_built_without_drift(self):
        """The import-time asserts guarantee every replace fired; this
        pins the two prompts actually differing (a vacuous variant would
        make the flag a no-op)."""
        from src.core.planner import (
            _DECOMPOSE_SYSTEM,
            _DECOMPOSE_SYSTEM_EXTRACTION,
        )
        assert _DECOMPOSE_SYSTEM != _DECOMPOSE_SYSTEM_EXTRACTION
        # And the shared skeleton survives (JSON shape, ids, tools).
        for anchor in ('"passos"', '"depende_de"', "{AVAILABLE_TOOLS}",
                       "{TEMPORAL_CONTEXT}"):
            assert anchor in _DECOMPOSE_SYSTEM_EXTRACTION


# ----------------------------------------------------------------------
# i18n at the orchestrator boundary (M1-A8): the staged plan carries the
# turn's language, and the morning announce follows the reply language.
# ----------------------------------------------------------------------


class TestPlannerI18nOrchestrator:
    def _build(self, tmp_path, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator
        from src.memory.palace import Palace

        monkeypatch.setattr(
            CortexOrchestrator, "_PENDING_ACTION_FILE",
            str(tmp_path / "pending_action.json"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch._pending_action = None
        orch.engine_analytical = MagicMock()
        orch.engine_analytical.model_name = "cortex-lite"
        orch._palace = Palace(db_path=":memory:")
        return orch

    _PLAN_JSON = (
        '{"objetivo":"copy and remind","passos":['
        '{"id":"p1","tool":"read_document","args":{"filename":"a.txt"},'
        '"depende_de":[],"descricao":"read a.txt"}'
        ']}'
    )

    def test_en_turn_stages_en_preview_and_lang(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch)
        orch._reply_lang = "en"
        with patch("src.core.orchestrator.ollama") as ollama_mock:
            ollama_mock.chat.return_value = {"message": {"content": self._PLAN_JSON}}
            reply, mode = orch._handle_planner_request("copy and remind")
        assert mode == "AGENT_AWAITING_CONFIRMATION"
        assert "Plan for:" in reply
        assert "Reply 'yes' to run the plan or 'no' to cancel." in reply
        assert "Plano" not in reply
        assert orch._pending_action["args"]["lang"] == "en"

    def test_staged_lang_wins_over_sim_turn_lang(self, tmp_path, monkeypatch):
        # The plan was staged in EN; the approving "sim" turn resolves as
        # PT (approvals always do). The summary must come back in EN.
        from src.core.planner import _SUMMARISE_SYSTEM_EN
        orch = self._build(tmp_path, monkeypatch)
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "contents"
        orch._reply_lang = "pt"  # the "sim" turn
        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "read a",
                "lang": "en",
                "passos": [
                    {"id": "p1", "tool": "read_document",
                     "args": {"filename": "a.txt"},
                     "depende_de": [], "descricao": "read"},
                ],
            },
        }
        seen = {}
        with patch("src.core.staging_mixin.ollama") as ollama_mock:
            def chat(**kw):
                seen["system"] = kw["messages"][0]["content"]
                return {"message": {"content": "Done."}}
            ollama_mock.chat.side_effect = chat
            reply, mode = orch._execute_staged_plan()
        assert mode == "AGENT_ACTION_DONE"
        assert seen["system"] == _SUMMARISE_SYSTEM_EN

    def test_pre_m1a8_staged_plan_defaults_pt(self, tmp_path, monkeypatch):
        # A plan persisted before M1-A8 has no lang key — summary stays PT.
        from src.core.planner import _SUMMARISE_SYSTEM
        orch = self._build(tmp_path, monkeypatch)
        orch.reader = MagicMock()
        orch.reader.read_document.return_value = "contents"
        orch._pending_action = {
            "type": "execute_plan",
            "args": {
                "objetivo": "ler a",
                "passos": [
                    {"id": "p1", "tool": "read_document",
                     "args": {"filename": "a.txt"},
                     "depende_de": [], "descricao": "ler"},
                ],
            },
        }
        seen = {}
        with patch("src.core.staging_mixin.ollama") as ollama_mock:
            def chat(**kw):
                seen["system"] = kw["messages"][0]["content"]
                return {"message": {"content": "Feito."}}
            ollama_mock.chat.side_effect = chat
            reply, mode = orch._execute_staged_plan()
        assert mode == "AGENT_ACTION_DONE"
        assert seen["system"] == _SUMMARISE_SYSTEM


class TestAnnounceI18n:
    def _build(self, tmp_path, monkeypatch, topics):
        from src.core.orchestrator import CortexOrchestrator
        monkeypatch.setattr(
            CortexOrchestrator, "_LAST_ANNOUNCE_PATH",
            str(tmp_path / "last_announce.txt"),
        )
        orch = CortexOrchestrator.__new__(CortexOrchestrator)
        orch.studies_review = MagicMock()
        orch.studies_review.list_pending.return_value = [
            {"topic": t} for t in topics
        ]
        return orch

    def test_announce_en(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch, ["LLM news"])
        orch._reply_lang = "en"
        out = orch._maybe_announce_pending("Sure thing.", "OFFLINE")
        assert "By the way" in out
        assert "1 study ready for review: LLM news" in out
        assert "Say 'review studies'" in out
        assert "Por sinal" not in out and "estudo" not in out

    def test_announce_default_pt(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch, ["LLM news"])
        out = orch._maybe_announce_pending("Claro.", "OFFLINE")
        assert "Por sinal" in out
        assert "1 estudo pronto pra revisar: LLM news" in out
        assert "Diga 'revisar'" in out

    def test_announce_en_gate_queue(self, tmp_path, monkeypatch):
        from src.core import orchestrator as orch_mod
        orch = self._build(tmp_path, monkeypatch, [])
        orch._reply_lang = "en"
        monkeypatch.setattr(
            orch_mod.gate_queue, "pending",
            lambda: [{"type": "send_email"}],
        )
        out = orch._maybe_announce_pending("Sure.", "OFFLINE")
        assert "1 action awaiting your approval: send_email" in out
        assert "gate review" in out
        assert "aguardando" not in out

    def test_announce_once_per_day(self, tmp_path, monkeypatch):
        orch = self._build(tmp_path, monkeypatch, ["X"])
        first = orch._maybe_announce_pending("Claro.", "OFFLINE")
        second = orch._maybe_announce_pending("Claro.", "OFFLINE")
        assert "Por sinal" in first
        assert second == "Claro."


class TestRunnerFieldsConvention:
    """M4-B1 (dormant, default OFF): planner runners may return
    {"text","fields"}; _resolve_refs extracts a field, surfaces/summary read
    'text'. The consumer was wired in s66; this pins the producer + the
    last text-only consumer (_format_results_for_llm)."""

    # ---- consumer side (planner) ----
    def test_step_text_reads_text_from_dict(self):
        from src.core.planner import _step_text
        assert _step_text({"text": "Evento criado.", "fields": {"id": "e1"}}) == "Evento criado."
        assert _step_text("plain") == "plain"
        assert _step_text(None) == ""

    def test_resolve_refs_extracts_from_fields(self):
        prior = {"p1": StepResult(
            "p1", success=True,
            output={"text": "Evento criado.", "fields": {"htmlLink": "http://x", "id": "e1"}})}
        out = _resolve_refs({"body": "Link: {p1.htmlLink}", "ref": "{p1.id}"}, prior=prior)
        assert out == {"body": "Link: http://x", "ref": "e1"}

    def test_resolve_bare_ref_uses_text(self):
        prior = {"p1": StepResult(
            "p1", success=True,
            output={"text": "Email enviado.", "fields": {"id": "m1"}})}
        assert _resolve_refs({"x": "{p1}"}, prior=prior) == {"x": "Email enviado."}

    def test_format_results_uses_text_not_raw_dict(self):
        from src.core.planner import Plan, Step, _format_results_for_llm
        plan = Plan(objetivo="o", passos=[
            Step(id="p1", tool="create_event", args={}, depende_de=[], descricao="cria")])
        results = [StepResult(
            "p1", success=True,
            output={"text": "Evento criado.", "fields": {"id": "e1"}})]
        body = _format_results_for_llm(plan, results, lang="pt")
        assert "Evento criado." in body
        assert "fields" not in body and "'id'" not in body  # dict never leaked raw

    # ---- producer side (staging_mixin) ----
    def test_runner_result_off_returns_bare_text(self, monkeypatch):
        from src.core import staging_mixin
        monkeypatch.delenv("CORTEX_RUNNER_FIELDS", raising=False)
        assert staging_mixin._runner_result("hi", id="x") == "hi"

    def test_runner_result_on_wraps_and_drops_none(self, monkeypatch):
        from src.core import staging_mixin
        monkeypatch.setenv("CORTEX_RUNNER_FIELDS", "1")
        out = staging_mixin._runner_result(
            "Evento criado.", id="e1", htmlLink=None, start="2026-06-15T10:00")
        assert out == {"text": "Evento criado.", "fields": {"id": "e1", "start": "2026-06-15T10:00"}}
