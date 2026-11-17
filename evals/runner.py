"""Eval runner — dispatches each EvalCase to the right harness and
collects results.

Single entry point: `run(cases, filters)` → list[EvalResult]. The
CLI (`evals/__main__.py`) wires this up to argparse + the reporter.

Each `EvalKind` is handled by a tiny pure function that takes the
case and returns an EvalResult. They live below as `_run_*`. Adding
a new kind is one function + one dispatch entry in `_run_one`.

No I/O happens here beyond importlib (loading case modules); no
network, no LLM, no Palace. Speed target: 1000 cases in <2 seconds.
"""
from __future__ import annotations

import importlib
import os
import re
import traceback
from contextlib import contextmanager
from typing import Iterable, List, Optional

from evals.core import EvalCase, EvalKind, EvalResult, EvalStatus
from evals.dispatch import simulate_dispatch


# ---------------------------------------------------------------------
# Per-kind handlers
# ---------------------------------------------------------------------

def _run_dispatch(case: EvalCase) -> EvalResult:
    if case.input is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="DISPATCH case missing `input`",
        )
    outcome = simulate_dispatch(case.input)

    fails: list[str] = []
    if case.expected_mode is not None and outcome.mode != case.expected_mode:
        fails.append(
            f"mode: expected {case.expected_mode!r}, got {outcome.mode!r}"
        )
    if case.expected_matcher is not None and outcome.matcher != case.expected_matcher:
        fails.append(
            f"matcher: expected {case.expected_matcher!r}, got {outcome.matcher!r}"
        )
    if case.expected_capture is not None and outcome.capture != case.expected_capture:
        fails.append(
            f"capture: expected {case.expected_capture!r}, got {outcome.capture!r}"
        )

    payload = {
        "input": case.input,
        "mode": outcome.mode,
        "matcher": outcome.matcher,
        "capture": outcome.capture,
        "would_use_tools": outcome.would_use_tools,
    }

    if fails:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary=f"dispatch routed to {outcome.matcher} ({outcome.mode})",
            detail="\n".join(fails),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary=f"{outcome.matcher} → {outcome.mode}",
        payload=payload,
    )


def _run_dispatch_reply(case: EvalCase) -> EvalResult:
    """Execute the matched dispatch handler against the fake orchestrator with
    `case.language` forced, then assert the REPLY text. Guards the s60
    spoken-language localisation. Deterministic (stubs only)."""
    if case.input is None:
        return EvalResult(case=case, status=EvalStatus.ERROR,
                          summary="DISPATCH_REPLY case missing `input`")
    if not case.expected_contains and not case.expected_not_contains:
        return EvalResult(case=case, status=EvalStatus.ERROR,
                          summary="DISPATCH_REPLY needs expected_contains / _not_contains")

    from evals.fake_orch import run_reply

    lang = case.language or "pt"
    result = run_reply(case.input, lang)
    if result is None:
        return EvalResult(case=case, status=EvalStatus.FAIL,
                          summary=f"no route claimed {case.input!r}",
                          detail="dispatch_registry.run returned None")
    reply, mode = result

    fails: list[str] = []
    if case.expected_contains is not None and case.expected_contains not in reply:
        fails.append(f"expected_contains {case.expected_contains!r} not in reply")
    if case.expected_not_contains is not None and case.expected_not_contains in reply:
        fails.append(f"expected_not_contains {case.expected_not_contains!r} present")

    payload = {"input": case.input, "lang": lang, "mode": mode, "reply": reply}
    if fails:
        return EvalResult(case=case, status=EvalStatus.FAIL,
                          summary=f"reply ({lang}) didn't match",
                          detail="reply: " + repr(reply[:200]) + "\n" + "\n".join(fails),
                          payload=payload)
    return EvalResult(case=case, status=EvalStatus.PASS,
                      summary=f"{mode} reply OK ({lang})", payload=payload)


def _run_detector(case: EvalCase) -> EvalResult:
    if case.detector_input is None or case.expected_flagged is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="DETECTOR case missing detector_input or expected_flagged",
        )
    from src.core.hallucination import detect_hallucinated_action

    actual = detect_hallucinated_action(case.detector_input)
    payload = {
        "detector_input": case.detector_input[:200],
        "expected_flagged": case.expected_flagged,
        "actual_flagged": actual,
    }
    if actual != case.expected_flagged:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary=f"detector returned {actual}, expected {case.expected_flagged}",
            detail=(
                f"input: {case.detector_input!r}\n"
                f"expected_flagged: {case.expected_flagged}\n"
                f"actual_flagged:   {actual}"
            ),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary=f"detector → {actual}",
        payload=payload,
    )


def _run_retrieval(case: EvalCase) -> EvalResult:
    if case.retrieval_query is None or case.retrieval_expected is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="RETRIEVAL case missing retrieval_query or retrieval_expected",
        )
    from evals.recall_fixture import build_memory

    semantic = case.kind == EvalKind.RETRIEVAL_SEMANTIC
    memory = build_memory(semantic=semantic, polluted=case.retrieval_polluted)
    context = memory.retrieve_relevant_context(case.retrieval_query) or ""
    found = case.retrieval_expected.lower() in context.lower()
    payload = {
        "query": case.retrieval_query,
        "expected": case.retrieval_expected,
        "found": found,
        "context": context[:300],
    }
    if found:
        return EvalResult(
            case=case,
            status=EvalStatus.PASS,
            summary=f"recall ✓ ({case.retrieval_expected!r} in context)",
            payload=payload,
        )
    # Semantic retrieval varies with the embedder/model → soft DRIFT (visible,
    # non-fatal), matching the LLM-case contract. Lexical recall is exact →
    # a miss is a real FAIL (a regression in the deterministic retrieval path).
    status = EvalStatus.DRIFT if semantic else EvalStatus.FAIL
    return EvalResult(
        case=case,
        status=status,
        summary=f"recall miss: {case.retrieval_expected!r} not in context",
        detail=(
            f"query:    {case.retrieval_query!r}\n"
            f"expected: {case.retrieval_expected!r}\n"
            f"context:  {context[:300]!r}"
        ),
        payload=payload,
    )


def _run_injection_guard(case: EvalCase) -> EvalResult:
    if (case.guard_predicado is None or case.guard_objeto is None
            or case.expected_safe is None):
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="INJECTION_GUARD case missing guard_predicado/guard_objeto/expected_safe",
        )
    from src.memory.preferences import _safe_to_inject

    actual = _safe_to_inject(case.guard_predicado, case.guard_objeto)
    payload = {
        "guard_predicado": case.guard_predicado,
        "guard_objeto": case.guard_objeto[:200],
        "expected_safe": case.expected_safe,
        "actual_safe": actual,
    }
    if actual != case.expected_safe:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary=f"_safe_to_inject returned {actual}, expected {case.expected_safe}",
            detail=(
                f"predicado: {case.guard_predicado!r}\n"
                f"objeto:    {case.guard_objeto!r}\n"
                f"expected_safe: {case.expected_safe}\n"
                f"actual_safe:   {actual}"
            ),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary=f"_safe_to_inject → {actual}",
        payload=payload,
    )


def _run_sanitiser(case: EvalCase) -> EvalResult:
    if case.sanitiser_input is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="SANITISER case missing sanitiser_input",
        )
    from src.core.engine import OllamaEngine

    cleaned = OllamaEngine._sanitise_response(case.sanitiser_input)
    fails: list[str] = []
    if case.expected_contains is not None and case.expected_contains not in cleaned:
        fails.append(
            f"expected_contains {case.expected_contains!r} not in output"
        )
    if case.expected_not_contains is not None and case.expected_not_contains in cleaned:
        fails.append(
            f"expected_not_contains {case.expected_not_contains!r} appeared"
        )

    payload = {
        "input": case.sanitiser_input,
        "output": cleaned,
    }
    if fails:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary="sanitiser output mismatch",
            detail=f"output: {cleaned!r}\n" + "\n".join(fails),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary="sanitiser output matches",
        payload=payload,
    )


def _resolve_dotted(dotted: str):
    """Resolve 'pkg.mod.NAME' → live object. Used by REGEX kind so
    cases can name regexes by dotted path string."""
    parts = dotted.split(".")
    if len(parts) < 2:
        raise ValueError(f"regex_dotted must be 'pkg.mod.NAME', got {dotted!r}")
    mod_path = ".".join(parts[:-1])
    attr = parts[-1]
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _run_regex(case: EvalCase) -> EvalResult:
    if case.regex_dotted is None or case.regex_input is None \
       or case.expected_regex_match is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="REGEX case missing regex_dotted/regex_input/expected_regex_match",
        )
    try:
        rx = _resolve_dotted(case.regex_dotted)
    except Exception as exc:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary=f"could not resolve regex {case.regex_dotted}: {exc}",
        )

    m = rx.search(case.regex_input) if case.regex_use_search \
        else rx.match(case.regex_input)
    matched = m is not None

    fails: list[str] = []
    if matched != case.expected_regex_match:
        fails.append(
            f"match: expected {case.expected_regex_match}, got {matched}"
        )
    if matched and case.expected_regex_group is not None:
        try:
            captured = m.group(1)
        except IndexError:
            captured = None
        if captured != case.expected_regex_group:
            fails.append(
                f"group(1): expected {case.expected_regex_group!r}, got {captured!r}"
            )

    payload = {
        "regex": case.regex_dotted,
        "input": case.regex_input,
        "use_search": case.regex_use_search,
        "matched": matched,
    }
    if fails:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary="regex mismatch",
            detail="\n".join(fails),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary=f"regex {'match' if matched else 'no-match'}",
        payload=payload,
    )


# ---------------------------------------------------------------------
# LLM_BEHAVIOR — real engine.chat() call (opt-in, hits Ollama)
# ---------------------------------------------------------------------

# Deterministic options for every LLM eval call. temperature 0 + a fixed
# seed gets us as close to reproducible as the platform allows — note
# Metal/MPS reduction-order means it's NOT bit-perfect even so, which is
# exactly why these cases are drift_tolerant.
_LLM_OPTIONS = {"temperature": 0, "seed": 42}

# Cached engine — booting OllamaEngine (Palace, embeddings) costs a few
# seconds, so we pay it once per process and reuse across cases. Reset
# of per-case history happens in the handler, not here.
_LLM_ENGINE = None


class _EvalStubMemory:
    """Replaces the engine's Palace-backed memory during evals: no
    retrieval (so the reply reflects the prompt alone — reproducible)
    and no writes (so eval Q&A never pollutes the real long-term store)."""

    def retrieve_relevant_context(self, query, n_results=3, include_knowledge=True):
        return ""

    def store_interaction(self, user_input, reply):
        pass

    def get_user_profile(self):
        return ""


def _get_llm_engine():
    """Construct (once) a real OllamaEngine for LLM behaviour evals.

    Model is read from CORTEX_MODEL, falling back to the analytical
    model env var, then 'cortex-lite'. NEVER hardcodes a model —
    `CORTEX_MODEL=cortex-gemma make eval-llm` just works. Monkeypatched
    in unit tests so the framework self-tests don't hit Ollama.
    """
    global _LLM_ENGINE
    if _LLM_ENGINE is None:
        from src.core.engine import OllamaEngine

        model = (
            os.getenv("CORTEX_MODEL")
            or os.getenv("CORTEX_MODEL_ANALYTICAL")
            or "cortex-lite"
        )
        engine = OllamaEngine(model_name=model)
        # Strip memory: context-free + side-effect-free eval runs.
        engine.memory = _EvalStubMemory()
        _LLM_ENGINE = engine
    return _LLM_ENGINE


def _run_rag_prompt(case: EvalCase) -> EvalResult:
    """Deterministic: build the ONLINE_RAG prompt from frozen inputs and
    assert substrings on the prompt itself. No Ollama, no network."""
    if case.input is None or case.rag_web_context is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="RAG_PROMPT case missing `input` (query) or `rag_web_context`",
        )
    from src.core.orchestrator import build_rag_prompt
    prompt = build_rag_prompt(
        case.rag_web_context, case.input, today_line=case.rag_today,
    )

    low = prompt.lower()
    fails: list[str] = []
    for sub in case.expected_contains_all:
        if sub.lower() not in low:
            fails.append(f"expected_contains_all missing: {sub!r}")
    if case.expected_not_contains and case.expected_not_contains.lower() in low:
        fails.append(f"expected_not_contains present: {case.expected_not_contains!r}")

    if fails:
        return EvalResult(
            case=case,
            status=EvalStatus.FAIL,
            summary="built RAG prompt didn't match",
            detail="\n".join(fails),
        )
    return EvalResult(case=case, status=EvalStatus.PASS, summary="RAG prompt matched")


@contextmanager
def _case_turn_notes(case, engine):
    if not case.apply_turn_notes:
        yield
        return
    from src.core.turn_notes import reply_notes

    engine.push_turn_context(reply_notes(case.input or ""))
    try:
        yield
    finally:
        engine.pop_turn_context()


def _new_reply_checks(case, reply):
    reply = reply or ""
    fails = []
    if case.expected_reply_regex is not None and not re.search(case.expected_reply_regex, reply):
        fails.append(f"expected_reply_regex missing: {case.expected_reply_regex!r}")
    if case.forbidden_reply_regex is not None and re.search(case.forbidden_reply_regex, reply):
        fails.append(f"forbidden_reply_regex present: {case.forbidden_reply_regex!r}")
    if case.expected_first_sentence_regex is not None:
        # Decimal points inside numbers do not end a sentence.
        first = re.split(r"(?<!\d)\.|\.(?!\d)|[!?…\n]", reply, maxsplit=1)[0]
        if not re.search(case.expected_first_sentence_regex, first):
            fails.append(f"first sentence did not match: {case.expected_first_sentence_regex!r}")
    return fails


def _run_llm_behavior(case: EvalCase) -> EvalResult:
    if case.input is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="LLM_BEHAVIOR case missing `input`",
        )
    if (not case.expected_contains_any and not case.expected_contains_all
            and case.expected_reply_regex is None and case.forbidden_reply_regex is None
            and case.expected_first_sentence_regex is None):
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary="LLM_BEHAVIOR case has no expected_contains_any/all",
        )

    engine = _get_llm_engine()
    engine.history = []  # per-case independence (engine is reused)

    def _ask() -> str:
        with _case_turn_notes(case, engine):
            return engine.chat(
                case.input,
                response_language=case.language or "pt",
                options=_LLM_OPTIONS,
            )

    try:
        reply = _ask()
        # Retry ONCE only on an empty reply — that signals a real
        # transport hiccup, not the ordinary token-level drift.
        if not (reply or "").strip():
            engine.history = []
            reply = _ask()
    except Exception as exc:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary=f"engine.chat raised: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(limit=4),
        )

    low = (reply or "").lower()
    fails: list[str] = []
    for sub in case.expected_contains_all:
        if sub.lower() not in low:
            fails.append(f"expected_contains_all missing: {sub!r}")
    if case.expected_contains_any:
        if not any(sub.lower() in low for sub in case.expected_contains_any):
            fails.append(
                "expected_contains_any: none of "
                f"{case.expected_contains_any!r} present"
            )
    # s73 (RAG anti-hallucination): some cases must pin what the reply
    # may NOT say — e.g. the WRONG fixture picked from correct snippets.
    if case.expected_not_contains and case.expected_not_contains.lower() in low:
        fails.append(f"expected_not_contains present: {case.expected_not_contains!r}")

    fails.extend(_new_reply_checks(case, reply))

    payload = {
        "input": case.input,
        "model": engine.model_name,
        "prompt_version": getattr(engine, "prompt_version", None),
        "reply": reply,
        "drift_tolerant": case.drift_tolerant,
    }
    if fails:
        status = EvalStatus.DRIFT if case.drift_tolerant else EvalStatus.FAIL
        return EvalResult(
            case=case,
            status=status,
            summary=f"reply didn't match ({'drift' if case.drift_tolerant else 'pinned'})",
            detail="reply: " + repr(reply[:300]) + "\n" + "\n".join(fails),
            payload=payload,
        )
    return EvalResult(
        case=case,
        status=EvalStatus.PASS,
        summary="reply matched",
        payload=payload,
    )


# ---------------------------------------------------------------------
# LLM_TOOL_CALL — real engine.chat_with_tools() call (opt-in, hits Ollama)
# ---------------------------------------------------------------------

def _run_llm_tool_call(case: EvalCase) -> EvalResult:
    """Check which TOOL the model picks for a prompt, faithfully to
    production: the real TOOL_SPECS are exposed and the real
    chat_with_tools loop runs — but the executor is a STUB that records
    (name, args) and returns the skip-followup marker, so no real action
    fires and the slow second LLM round is skipped.

    The returned tool name (chat_with_tools' second tuple element) is the
    authoritative signal. We additionally capture args for substring
    checks. Drift semantics mirror LLM_BEHAVIOR.
    """
    if case.input is None:
        return EvalResult(case=case, status=EvalStatus.ERROR,
                          summary="LLM_TOOL_CALL case missing `input`")
    if case.expected_tool is None and not case.expect_no_tool:
        return EvalResult(case=case, status=EvalStatus.ERROR,
                          summary="LLM_TOOL_CALL needs expected_tool or expect_no_tool")

    from src.core.engine import SKIP_FOLLOWUP_MARKER
    from src.core.tool_specs import TOOL_SPECS

    engine = _get_llm_engine()
    engine.history = []  # per-case independence

    captured: dict = {}

    def _stub_executor(name: str, args: dict) -> str:
        # Record the FIRST call; short-circuit round 2 so nothing real
        # runs and we don't pay a second inference.
        captured.setdefault("name", name)
        captured.setdefault("args", args or {})
        return SKIP_FOLLOWUP_MARKER + "(eval stub: tool not executed)"

    try:
        with _case_turn_notes(case, engine):
            _reply, tool_used = engine.chat_with_tools(
                case.input, TOOL_SPECS, _stub_executor,
                response_language=case.language or "pt",
            )
    except Exception as exc:
        return EvalResult(
            case=case, status=EvalStatus.ERROR,
            summary=f"chat_with_tools raised: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(limit=4),
        )

    args = captured.get("args", {})
    fails: list[str] = []

    if case.expect_no_tool:
        if tool_used is not None:
            fails.append(f"expected NO tool, but called {tool_used!r}")
        if case.check_no_tool_reply:
            low = (_reply or "").lower()
            for sub in case.expected_contains_all:
                if sub.lower() not in low:
                    fails.append(f"expected_contains_all missing: {sub!r}")
            if case.expected_contains_any and not any(
                    sub.lower() in low for sub in case.expected_contains_any):
                fails.append(f"expected_contains_any missing: {case.expected_contains_any!r}")
            if case.expected_not_contains and case.expected_not_contains.lower() in low:
                fails.append(f"expected_not_contains present: {case.expected_not_contains!r}")
        fails.extend(_new_reply_checks(case, _reply))
    else:
        if tool_used != case.expected_tool:
            fails.append(f"expected tool {case.expected_tool!r}, got {tool_used!r}")
        else:
            for arg_name, sub in case.expected_arg_contains.items():
                got = str(args.get(arg_name, ""))
                if sub.lower() not in got.lower():
                    fails.append(
                        f"arg {arg_name!r}: {sub!r} not in {got!r}"
                    )
            for arg_name, expected in case.expected_arg_equals.items():
                if args.get(arg_name) != expected:
                    fails.append(f"arg {arg_name!r}: expected exactly {expected!r}, got {args.get(arg_name)!r}")
            for arg_name, pattern in case.expected_arg_regex.items():
                if not re.search(pattern, str(args.get(arg_name, ""))):
                    fails.append(f"arg {arg_name!r}: required regex missing: {pattern!r}")
            for arg_name, pattern in case.forbidden_arg_regex.items():
                if re.search(pattern, str(args.get(arg_name, ""))):
                    fails.append(f"arg {arg_name!r}: forbidden regex present: {pattern!r}")

    payload = {
        "input": case.input,
        "model": engine.model_name,
        "prompt_version": getattr(engine, "prompt_version", None),
        "tool_used": tool_used,
        "tool_args": args,
        "drift_tolerant": case.drift_tolerant,
    }
    if case.expect_no_tool and (case.check_no_tool_reply or case.expected_reply_regex is not None
                               or case.forbidden_reply_regex is not None
                               or case.expected_first_sentence_regex is not None):
        payload["reply"] = _reply
    if fails:
        status = EvalStatus.DRIFT if case.drift_tolerant else EvalStatus.FAIL
        return EvalResult(
            case=case, status=status,
            summary=f"tool selection didn't match ({'drift' if case.drift_tolerant else 'pinned'})",
            detail="\n".join(fails),
            payload=payload,
        )
    return EvalResult(
        case=case, status=EvalStatus.PASS,
        summary=f"tool → {tool_used}" if tool_used else "no tool (as expected)",
        payload=payload,
    )


# Dispatcher table.
_HANDLERS = {
    EvalKind.DISPATCH:      _run_dispatch,
    EvalKind.DISPATCH_REPLY: _run_dispatch_reply,
    EvalKind.DETECTOR:      _run_detector,
    EvalKind.SANITISER:     _run_sanitiser,
    EvalKind.REGEX:         _run_regex,
    EvalKind.RAG_PROMPT:    _run_rag_prompt,
    EvalKind.INJECTION_GUARD: _run_injection_guard,
    EvalKind.RETRIEVAL:       _run_retrieval,
    EvalKind.RETRIEVAL_SEMANTIC: _run_retrieval,
    EvalKind.LLM_BEHAVIOR:  _run_llm_behavior,
    EvalKind.LLM_TOOL_CALL: _run_llm_tool_call,
}


def _run_one(case: EvalCase) -> EvalResult:
    handler = _HANDLERS.get(case.kind)
    if handler is None:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary=f"no handler for EvalKind {case.kind!r}",
        )
    try:
        return handler(case)
    except Exception as exc:
        return EvalResult(
            case=case,
            status=EvalStatus.ERROR,
            summary=f"handler raised: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(limit=4),
        )


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def run(
    cases: Iterable[EvalCase],
    category: Optional[str] = None,
    kind: Optional[EvalKind] = None,
    tag: Optional[str] = None,
) -> List[EvalResult]:
    """Execute the given cases, optionally filtered by category /
    kind / tag. Returns one EvalResult per executed case, in input
    order so the reporter can show them grouped."""
    results: List[EvalResult] = []
    for case in cases:
        if category is not None and case.category != category:
            continue
        if kind is not None and case.kind != kind:
            continue
        if tag is not None and tag not in case.tags:
            continue
        results.append(_run_one(case))
    return results
