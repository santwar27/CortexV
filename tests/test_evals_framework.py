"""Self-tests for the eval framework itself.

`evals/` is a behavioural suite that we expect to evolve, but the
PLUMBING (loader, runner, dispatch simulator, kind handlers) must
be airtight — if those break, every eval result is suspect. These
tests pin the contract.

Coverage:
  - core.EvalCase / EvalKind / EvalResult dataclasses construct cleanly
  - core.load_all_cases discovers every cases/*.py module
  - dispatch.simulate_dispatch produces stable outcomes for canonical
    inputs AND stays in sync with the orchestrator's regex ordering
  - runner._run_one routes to the right handler per kind
  - report.format_terminal builds a sane string (smoke)
"""
from __future__ import annotations

import pytest

from evals.core import (
    EvalCase,
    EvalKind,
    EvalResult,
    EvalStatus,
    load_all_cases,
)
from evals.dispatch import DispatchOutcome, simulate_dispatch
from evals.runner import run, _run_one
from evals.report import format_terminal


# =====================================================================
# core dataclasses
# =====================================================================

class TestEvalCaseConstruction:

    def test_minimum_dispatch_case(self):
        c = EvalCase(
            name="x",
            kind=EvalKind.DISPATCH,
            category="spotify",
            input="toca matue",
            expected_mode="SPOTIFY_PLAY",
        )
        assert c.kind == EvalKind.DISPATCH
        assert c.tags == []  # default factory

    def test_minimum_detector_case(self):
        c = EvalCase(
            name="y",
            kind=EvalKind.DETECTOR,
            category="anti_hallucination",
            detector_input="Tocando X.",
            expected_flagged=True,
        )
        assert c.kind == EvalKind.DETECTOR

    def test_minimum_regex_case(self):
        c = EvalCase(
            name="z",
            kind=EvalKind.REGEX,
            category="conflict",
            regex_dotted="src.core.dispatch.spotify.SPOTIFY_PLAY_RE",
            regex_input="toca matue",
            expected_regex_match=True,
        )
        assert c.regex_use_search is False  # default

    def test_eval_kind_values_are_strings(self):
        # The CLI accepts kind by string name, so the enum must be
        # str-valued (avoids json.dumps coercion issues too).
        assert EvalKind.DISPATCH.value == "dispatch"
        assert EvalKind.DETECTOR.value == "detector"


# =====================================================================
# load_all_cases — discovery
# =====================================================================

class TestLoader:

    def test_loads_some_cases(self):
        cases = load_all_cases()
        assert len(cases) > 30, "expected at least 30 cases in the first batch"

    def test_every_case_has_required_fields(self):
        for case in load_all_cases():
            assert case.name, f"case missing name in category {case.category}"
            assert case.category, f"case {case.name} missing category"
            assert isinstance(case.kind, EvalKind)

    def test_all_categories_represented(self):
        cats = {c.category for c in load_all_cases()}
        # First-batch categories per docs/JOURNAL bloco I plan.
        for required in ("spotify", "macos", "anti_hallucination",
                         "regression", "studier", "conflict", "adversarial",
                         "recall", "humility", "digest", "entities",
                         "contacts", "dormant"):
            assert required in cats, f"category {required} missing"

    def test_case_names_unique_within_category(self):
        """Two cases with the same name in the same category make
        failures ambiguous to triage. Force uniqueness."""
        seen = set()
        for c in load_all_cases():
            key = (c.category, c.name)
            assert key not in seen, f"duplicate case: {key}"
            seen.add(key)


# =====================================================================
# simulate_dispatch — orchestrator parity
# =====================================================================

class TestSimulateDispatch:

    def test_empty_input(self):
        o = simulate_dispatch("")
        assert o.mode == "INPUT_ERROR"
        assert o.matcher == "empty_input"

    def test_whitespace_only(self):
        o = simulate_dispatch("   \n\t ")
        assert o.mode == "INPUT_ERROR"

    def test_vocative_stripped_before_match(self):
        # The exact regression-test input from bloco F.
        o = simulate_dispatch("ola cortex, pode tocar matue no spotify?")
        assert o.mode == "SPOTIFY_PLAY"
        assert o.capture == "matue"

    def test_pausa_routes_to_spotify_pause(self):
        o = simulate_dispatch("pausa")
        assert o.mode == "SPOTIFY_PAUSE"
        assert o.matcher == "SPOTIFY_PAUSE_RE"

    def test_volume_50_bare_is_mac_and_named_spotify_is_spotify(self):
        # Critical UX guarantee, also tested via eval CASES.
        o = simulate_dispatch("volume 50")
        assert o.mode == "MAC_VOLUME"
        assert o.matcher == "MAC_VOLUME_SET_RE"
        assert o.capture == "50"

        o = simulate_dispatch("volume do spotify 50")
        assert o.mode == "SPOTIFY_VOLUME"
        assert o.matcher == "SPOTIFY_VOLUME_RE"
        assert o.capture == "50"

    def test_volume_sistema_50_is_mac(self):
        o = simulate_dispatch("volume sistema 50")
        assert o.mode == "MAC_VOLUME"
        assert o.matcher == "MAC_VOLUME_SET_RE"

    def test_fallthrough_reports_tool_hint(self):
        # A general question that mentions music — TOOL_HINT_RE fires.
        o = simulate_dispatch("será que dá pra tocar matue no spotify?")
        assert o.mode == "OFFLINE"
        assert o.matcher == "fallthrough"
        assert o.would_use_tools is True

    def test_pure_chat_no_tool_hint(self):
        # "tudo bem" matches APPROVAL_RE, so avoid it. A definitional
        # query has none of the dispatch triggers AND none of the
        # TOOL_HINT_RE keywords, so it's a clean fallthrough.
        o = simulate_dispatch("qual é a capital da França")
        assert o.mode == "OFFLINE"
        assert o.matcher == "fallthrough"
        assert o.would_use_tools is False

    def test_close_and_quit_beat_open(self):
        # All three are broad "<verb> <app>" shapes. "fecha" closes the
        # window (red button), "encerra"/"quit" quits — both dispatch
        # before the generic open, and neither is stolen by it.
        assert simulate_dispatch("fecha o Notion").mode == "MAC_CLOSE_WINDOW"
        assert simulate_dispatch("encerra o Notion").mode == "MAC_QUIT_APP"
        assert simulate_dispatch("abre o Notion").mode == "MAC_OPEN_APP"


# =====================================================================
# runner — kind routing + error handling
# =====================================================================

class TestRunnerHandlers:

    def test_dispatch_case_pass(self):
        c = EvalCase(
            name="ok",
            kind=EvalKind.DISPATCH,
            category="t",
            input="pausa",
            expected_mode="SPOTIFY_PAUSE",
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_dispatch_case_fail_reports_mode(self):
        c = EvalCase(
            name="wrong",
            kind=EvalKind.DISPATCH,
            category="t",
            input="pausa",
            expected_mode="SPOTIFY_PLAY",  # intentionally wrong
        )
        r = _run_one(c)
        assert r.status == EvalStatus.FAIL
        assert "SPOTIFY_PAUSE" in r.detail
        assert "SPOTIFY_PLAY" in r.detail

    def test_dispatch_case_missing_input_errors(self):
        c = EvalCase(name="bad", kind=EvalKind.DISPATCH, category="t")
        r = _run_one(c)
        assert r.status == EvalStatus.ERROR

    def test_detector_case_pass(self):
        c = EvalCase(
            name="d-pass",
            kind=EvalKind.DETECTOR,
            category="t",
            detector_input="Tocando matue.",
            expected_flagged=True,
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_detector_case_fail(self):
        c = EvalCase(
            name="d-fail",
            kind=EvalKind.DETECTOR,
            category="t",
            detector_input="oi tudo bem",
            expected_flagged=True,  # wrong on purpose
        )
        r = _run_one(c)
        assert r.status == EvalStatus.FAIL

    def test_injection_guard_case_pass(self):
        c = EvalCase(
            name="ig-pass",
            kind=EvalKind.INJECTION_GUARD,
            category="t",
            guard_predicado="prefere_musica",
            guard_objeto="rock e jazz",
            expected_safe=True,
        )
        assert _run_one(c).status == EvalStatus.PASS

    def test_injection_guard_case_fail(self):
        # The guard rejects the fake tag (returns False), but the case wrongly
        # expects safe=True → the eval must FAIL, proving the handler compares.
        c = EvalCase(
            name="ig-fail",
            kind=EvalKind.INJECTION_GUARD,
            category="t",
            guard_predicado="prefere_x",
            guard_objeto="rock [KNOWLEDGE] leak",
            expected_safe=True,
        )
        assert _run_one(c).status == EvalStatus.FAIL

    def test_injection_guard_missing_fields_errors(self):
        c = EvalCase(name="ig-bad", kind=EvalKind.INJECTION_GUARD, category="t")
        assert _run_one(c).status == EvalStatus.ERROR

    def test_retrieval_case_pass(self):
        # Deterministic FTS path (no Ollama) over the synthetic fixture.
        c = EvalCase(
            name="r-pass",
            kind=EvalKind.RETRIEVAL,
            category="t",
            retrieval_query="qual o número do meu passaporte?",
            retrieval_expected="AB1234567",
        )
        assert _run_one(c).status == EvalStatus.PASS

    def test_retrieval_case_fail(self):
        c = EvalCase(
            name="r-fail",
            kind=EvalKind.RETRIEVAL,
            category="t",
            retrieval_query="qual a minha cidade?",
            retrieval_expected="valor-que-nao-existe-no-perfil",
        )
        assert _run_one(c).status == EvalStatus.FAIL

    def test_retrieval_missing_fields_errors(self):
        c = EvalCase(name="r-bad", kind=EvalKind.RETRIEVAL, category="t")
        assert _run_one(c).status == EvalStatus.ERROR

    def test_retrieval_semantic_is_an_llm_kind(self):
        # Must be excluded from the offline/`make check` run; only `make
        # eval-llm` (include_llm=True) pulls it.
        from evals.core import LLM_KINDS
        assert EvalKind.RETRIEVAL_SEMANTIC in LLM_KINDS
        assert EvalKind.RETRIEVAL not in LLM_KINDS

    def test_sanitiser_case_pass(self):
        c = EvalCase(
            name="s-pass",
            kind=EvalKind.SANITISER,
            category="t",
            sanitiser_input='{"kw": "x"}\nReal answer.',
            expected_contains="Real answer.",
            expected_not_contains='{"kw"',
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_regex_case_match(self):
        c = EvalCase(
            name="r-match",
            kind=EvalKind.REGEX,
            category="t",
            regex_dotted="src.core.dispatch.spotify.SPOTIFY_PAUSE_RE",
            regex_input="pausa",
            expected_regex_match=True,
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_regex_case_no_match(self):
        c = EvalCase(
            name="r-nomatch",
            kind=EvalKind.REGEX,
            category="t",
            regex_dotted="src.core.dispatch.spotify.SPOTIFY_PAUSE_RE",
            regex_input="hello world",
            expected_regex_match=False,
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_regex_case_with_group_assertion(self):
        c = EvalCase(
            name="r-group",
            kind=EvalKind.REGEX,
            category="t",
            regex_dotted="src.core.dispatch.spotify.SPOTIFY_PLAY_RE",
            regex_input="toca matue",
            expected_regex_match=True,
            expected_regex_group="matue",
        )
        r = _run_one(c)
        assert r.status == EvalStatus.PASS

    def test_regex_case_bad_dotted_errors(self):
        c = EvalCase(
            name="r-bad",
            kind=EvalKind.REGEX,
            category="t",
            regex_dotted="this.does.not.exist",
            regex_input="x",
            expected_regex_match=True,
        )
        r = _run_one(c)
        assert r.status == EvalStatus.ERROR


# =====================================================================
# runner — top-level filters
# =====================================================================

class TestRunnerFilters:

    def test_no_filter_runs_all(self):
        cases = load_all_cases()
        results = run(cases)
        assert len(results) == len(cases)

    def test_category_filter(self):
        cases = load_all_cases()
        results = run(cases, category="spotify")
        assert results
        assert all(r.case.category == "spotify" for r in results)

    def test_unknown_category_returns_empty(self):
        cases = load_all_cases()
        results = run(cases, category="nope")
        assert results == []

    def test_kind_filter(self):
        cases = load_all_cases()
        results = run(cases, kind=EvalKind.DETECTOR)
        assert results
        assert all(r.case.kind == EvalKind.DETECTOR for r in results)

    def test_tag_filter(self):
        cases = load_all_cases()
        results = run(cases, tag="regression")
        assert results
        assert all("regression" in r.case.tags for r in results)


# =====================================================================
# report — smoke
# =====================================================================

class TestReporter:

    def test_format_terminal_with_results(self):
        # Build 1 pass + 1 fail manually so we don't need to actually
        # run cases (faster + isolates the report formatter).
        pass_case = EvalCase(name="p", kind=EvalKind.DISPATCH, category="c")
        fail_case = EvalCase(
            name="f", kind=EvalKind.DISPATCH, category="c",
            description="this one was meant to fail",
        )
        results = [
            EvalResult(case=pass_case, status=EvalStatus.PASS, summary="ok"),
            EvalResult(
                case=fail_case, status=EvalStatus.FAIL,
                summary="mismatched mode",
                detail="mode: expected X, got Y",
            ),
        ]
        out = format_terminal(results)
        # The summary line names the count.
        assert "2 cases" in out
        assert "1 pass" in out
        assert "1 fail" in out
        # Failure detail must appear so the dev knows what broke.
        assert "mismatched mode" in out
        assert "mode: expected X, got Y" in out

    def test_format_terminal_empty(self):
        out = format_terminal([])
        assert "no results" in out

    def test_format_terminal_all_green_says_so(self):
        case = EvalCase(name="p", kind=EvalKind.DISPATCH, category="c")
        results = [
            EvalResult(case=case, status=EvalStatus.PASS, summary="ok"),
        ]
        out = format_terminal(results)
        assert "ALL GREEN" in out


# =====================================================================
# end-to-end — every shipped case must pass on the current codebase
# =====================================================================

class TestEveryShippedCasePasses:
    """If a case fails here, the eval suite drifted from production
    behaviour. Either fix the regression in the codebase or update
    the case to reflect the new expected behaviour — but never silently."""

    def test_all_cases_pass(self):
        results = run(load_all_cases())
        bad = [r for r in results if r.status != EvalStatus.PASS]
        if bad:
            lines = ["The following eval cases failed:"]
            for r in bad:
                lines.append(
                    f"  - {r.case.category}/{r.case.name}: {r.summary}"
                )
                if r.detail:
                    lines.append(f"    {r.detail.splitlines()[0]}")
            pytest.fail("\n".join(lines))

    def test_default_load_excludes_llm_behavior(self):
        """The deterministic suite (this test, make check) must NEVER
        pull LLM cases — they hit Ollama. load_all_cases() default
        excludes them; the runner therefore never calls the model here."""
        assert all(
            c.kind != EvalKind.LLM_BEHAVIOR for c in load_all_cases()
        )


# =====================================================================
# LLM_BEHAVIOR — handler logic (engine mocked, NO Ollama)
# =====================================================================

class _FakeEngine:
    """Stand-in for OllamaEngine in handler tests. `replies` is a list
    popped per chat() call so we can exercise the empty-retry path."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.model_name = "fake-model"
        self.history = []
        self.calls = 0

    def chat(self, text, response_language=None, options=None):
        self.calls += 1
        return self._replies.pop(0) if self._replies else ""


def _llm_case(**kw):
    base = dict(
        name="x", kind=EvalKind.LLM_BEHAVIOR, category="llm_behavior",
        input="ping",
    )
    base.update(kw)
    return EvalCase(**base)


class TestLLMBehaviorHandler:

    def _patch_engine(self, monkeypatch, replies):
        eng = _FakeEngine(replies)
        monkeypatch.setattr("evals.runner._get_llm_engine", lambda: eng)
        return eng

    def test_pass_when_substring_present(self, monkeypatch):
        self._patch_engine(monkeypatch, ["A capital é Paris."])
        r = _run_one(_llm_case(expected_contains_all=["paris"]))
        assert r.status == EvalStatus.PASS

    def test_drift_when_tolerant_and_missing(self, monkeypatch):
        self._patch_engine(monkeypatch, ["Não faço ideia."])
        r = _run_one(_llm_case(expected_contains_all=["paris"], drift_tolerant=True))
        assert r.status == EvalStatus.DRIFT  # visible, non-fatal

    def test_fail_when_pinned_and_missing(self, monkeypatch):
        self._patch_engine(monkeypatch, ["Não faço ideia."])
        r = _run_one(_llm_case(expected_contains_all=["paris"], drift_tolerant=False))
        assert r.status == EvalStatus.FAIL

    def test_contains_any_matches_one(self, monkeypatch):
        self._patch_engine(monkeypatch, ["Oi, tudo bem?"])
        r = _run_one(_llm_case(expected_contains_any=["olá", "oi", "bom dia"]))
        assert r.status == EvalStatus.PASS

    def test_case_insensitive(self, monkeypatch):
        self._patch_engine(monkeypatch, ["PARIS é a capital."])
        r = _run_one(_llm_case(expected_contains_all=["paris"]))
        assert r.status == EvalStatus.PASS

    def test_empty_reply_retries_once(self, monkeypatch):
        eng = self._patch_engine(monkeypatch, ["", "Paris."])
        r = _run_one(_llm_case(expected_contains_all=["paris"]))
        assert r.status == EvalStatus.PASS
        assert eng.calls == 2  # retried after the empty first reply

    def test_missing_expectations_is_error(self, monkeypatch):
        self._patch_engine(monkeypatch, ["whatever"])
        r = _run_one(_llm_case())  # no expected_contains_*
        assert r.status == EvalStatus.ERROR

    def test_engine_exception_is_error(self, monkeypatch):
        class _Boom:
            model_name = "boom"
            history = []
            def chat(self, *a, **k):
                raise RuntimeError("ollama down")
        monkeypatch.setattr("evals.runner._get_llm_engine", lambda: _Boom())
        r = _run_one(_llm_case(expected_contains_all=["x"]))
        assert r.status == EvalStatus.ERROR


class TestLLMCaseRegistry:

    def test_llm_cases_present_only_with_opt_in(self):
        default = load_all_cases()
        with_llm = load_all_cases(include_llm=True)
        assert len(with_llm) > len(default)
        assert any(c.kind == EvalKind.LLM_BEHAVIOR for c in with_llm)
        assert not any(c.kind == EvalKind.LLM_BEHAVIOR for c in default)

    def test_llm_cases_are_model_agnostic(self):
        """No case may hardcode a model name — the runner picks the
        model from CORTEX_MODEL so the same cases grade any model."""
        llm = [c for c in load_all_cases(include_llm=True)
               if c.kind == EvalKind.LLM_BEHAVIOR]
        assert llm, "expected some LLM_BEHAVIOR cases"
        for c in llm:
            blob = (c.input or "") + " ".join(
                c.expected_contains_any + c.expected_contains_all
            )
            low = blob.lower()
            for model_token in ("cortex-lite", "llama", "gemma", "cortex-gemma"):
                assert model_token not in low, (
                    f"case {c.name} references model {model_token!r}"
                )


# =====================================================================
# LLM_TOOL_CALL — handler logic (engine mocked, NO Ollama)
# =====================================================================

class _FakeToolEngine:
    """Stand-in for OllamaEngine in tool-call handler tests. Runs the
    executor (so arg capture is exercised) then returns a fixed
    (reply, tool_used) pair — mirroring chat_with_tools' contract."""
    def __init__(self, tool_used, args=None):
        self._tool_used = tool_used
        self._args = args or {}
        self.model_name = "fake-model"
        self.prompt_version = 1
        self.history = []

    def chat_with_tools(self, user_input, tools, tool_executor,
                        response_language=None):
        if self._tool_used is not None:
            tool_executor(self._tool_used, self._args)
        return ("final reply", self._tool_used)


def _tool_case(**kw):
    base = dict(name="t", kind=EvalKind.LLM_TOOL_CALL,
                category="llm_tool_call", input="do a thing")
    base.update(kw)
    return EvalCase(**base)


class TestLLMToolCallHandler:

    def _patch(self, monkeypatch, tool_used, args=None):
        eng = _FakeToolEngine(tool_used, args)
        monkeypatch.setattr("evals.runner._get_llm_engine", lambda: eng)
        return eng

    def test_pass_on_expected_tool(self, monkeypatch):
        self._patch(monkeypatch, "create_event")
        r = _run_one(_tool_case(expected_tool="create_event"))
        assert r.status == EvalStatus.PASS

    def test_drift_on_wrong_tool(self, monkeypatch):
        self._patch(monkeypatch, "web_search")
        r = _run_one(_tool_case(expected_tool="create_event"))
        assert r.status == EvalStatus.DRIFT  # tolerant by default

    def test_fail_when_pinned_and_wrong(self, monkeypatch):
        self._patch(monkeypatch, "web_search")
        r = _run_one(_tool_case(expected_tool="create_event", drift_tolerant=False))
        assert r.status == EvalStatus.FAIL

    def test_expect_no_tool_passes_when_none(self, monkeypatch):
        self._patch(monkeypatch, None)
        r = _run_one(_tool_case(expect_no_tool=True))
        assert r.status == EvalStatus.PASS

    def test_expect_no_tool_drifts_when_tool_fired(self, monkeypatch):
        self._patch(monkeypatch, "web_search")
        r = _run_one(_tool_case(expect_no_tool=True))
        assert r.status == EvalStatus.DRIFT

    def test_arg_substring_pass(self, monkeypatch):
        self._patch(monkeypatch, "send_email", {"to": "maria@empresa.com"})
        r = _run_one(_tool_case(expected_tool="send_email",
                                expected_arg_contains={"to": "maria"}))
        assert r.status == EvalStatus.PASS

    def test_arg_substring_drift_on_mismatch(self, monkeypatch):
        self._patch(monkeypatch, "send_email", {"to": "outro@x.com"})
        r = _run_one(_tool_case(expected_tool="send_email",
                                expected_arg_contains={"to": "maria"}))
        assert r.status == EvalStatus.DRIFT

    def test_missing_expectation_is_error(self, monkeypatch):
        self._patch(monkeypatch, "x")
        r = _run_one(_tool_case())  # neither expected_tool nor expect_no_tool
        assert r.status == EvalStatus.ERROR

    def test_payload_has_no_reply_key(self, monkeypatch):
        # Tool-call results must NOT carry 'reply' — that keeps them out
        # of the baseline (cosine on a tool name is meaningless).
        self._patch(monkeypatch, "create_event")
        r = _run_one(_tool_case(expected_tool="create_event"))
        assert "reply" not in r.payload
        assert r.payload["tool_used"] == "create_event"


class TestLLMKindsRegistry:

    def test_both_llm_kinds_excluded_by_default(self):
        from evals.core import LLM_KINDS
        default = load_all_cases()
        assert not any(c.kind in LLM_KINDS for c in default)

    def test_tool_call_cases_present_with_opt_in(self):
        with_llm = load_all_cases(include_llm=True)
        assert any(c.kind == EvalKind.LLM_TOOL_CALL for c in with_llm)


# =====================================================================
# DRIFT reporting — drift is visible but not fatal
# =====================================================================

class TestDriftReporting:

    def test_drift_does_not_say_failed(self):
        case = EvalCase(name="d", kind=EvalKind.LLM_BEHAVIOR, category="llm_behavior")
        results = [
            EvalResult(case=EvalCase(name="p", kind=EvalKind.LLM_BEHAVIOR,
                                     category="llm_behavior"),
                       status=EvalStatus.PASS, summary="ok"),
            EvalResult(case=case, status=EvalStatus.DRIFT, summary="drifted"),
        ]
        out = format_terminal(results)
        assert "drift" in out.lower()
        assert "FAILED" not in out
        assert "ALL PASS" in out

    def test_fail_still_says_failed_even_with_drift(self):
        results = [
            EvalResult(case=EvalCase(name="d", kind=EvalKind.LLM_BEHAVIOR,
                                     category="llm_behavior"),
                       status=EvalStatus.DRIFT, summary="drifted"),
            EvalResult(case=EvalCase(name="f", kind=EvalKind.DISPATCH,
                                     category="c"),
                       status=EvalStatus.FAIL, summary="broke"),
        ]
        out = format_terminal(results)
        assert "FAILED" in out
