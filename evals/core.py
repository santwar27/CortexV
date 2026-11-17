"""Core dataclasses for the eval framework.

An `EvalCase` is the unit of behaviour we check. Each case declares:
  - what kind of check it is (DISPATCH / DETECTOR / SANITISER / REGEX)
  - what to feed in
  - what to expect out
  - a description so failure output is self-explanatory

The runner inspects `kind` and dispatches to the right harness.
Adding a new check type is a 2-step:
  1. extend the `EvalKind` enum
  2. add a handler branch in `runner._run_one`

Why dataclasses instead of YAML/JSON: type checker + IDE catches
typos in field names at edit time, no parser stage, refactors with
`mypy` and `grep`. Cost of adding a case is one Python line vs
multi-line YAML. The Cortex codebase is 100% Python — keeping the
eval surface in the same language is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalKind(str, Enum):
    """What kind of check this case performs."""

    # Input text → expected mode + (optional) which dispatcher matched.
    # Replicates the orchestrator's _process_core regex ordering.
    DISPATCH = "dispatch"

    # Input text + `language` → EXECUTE the matched dispatch handler against a
    # light fake orchestrator (evals/fake_orch.py — stubs, no Palace/Ollama)
    # and assert the REPLY text (expected_contains / expected_not_contains).
    # Guards the s60 spoken-language localisation of the deterministic command
    # path against regression. DETERMINISTIC → runs inside `make check`.
    # Scope: mixin-localised families (notes, studier); controller families
    # (spotify/macos) + assemblers are covered by pytest unit tests.
    DISPATCH_REPLY = "dispatch_reply"

    # `detect_hallucinated_action(reply)` → expected bool.
    DETECTOR = "detector"

    # `_sanitise_response(raw)` → expected output substring/regex.
    SANITISER = "sanitiser"

    # A given compiled regex .match/.search → expected match boolean
    # and optional captured group equality.
    REGEX = "regex"

    # `preferences._safe_to_inject(predicado, objeto)` → expected bool.
    # Locks the deterministic prompt-injection guard: a (predicado, objeto)
    # carrying a fake [KNOWLEDGE] tag, a newline, an oversized value, or a
    # malformed predicado must be REJECTED (expected_safe=False). Pure +
    # offline, so it runs inside `make check` (unlike the opt-in LLM set).
    INJECTION_GUARD = "injection_guard"

    # Golden recall — seed a throwaway Palace with a synthetic profile, run a
    # query through CortexMemory.retrieve_relevant_context, assert the right
    # section/value enters the retrieved context. Guards the s39 retrieval fix
    # against silent regression (alpha / n_results / chunking).
    #   RETRIEVAL          — lexical-only (embed_fn=None) → DETERMINISTIC,
    #                        runs inside `make check`.
    #   RETRIEVAL_SEMANTIC — real embeddings (paraphrase queries) → needs
    #                        Ollama, OPT-IN via `make eval-llm` (in LLM_KINDS).
    RETRIEVAL = "retrieval"
    RETRIEVAL_SEMANTIC = "retrieval_semantic"

    # Build the ONLINE_RAG synthesis prompt (orchestrator.build_rag_prompt)
    # from frozen snippets/query/date and assert substrings on the PROMPT
    # itself — no LLM, pure string assembly → DETERMINISTIC, runs inside
    # `make check`. Guards the s73 anti-hallucination anchors (TODAY'S DATE
    # adjacent to the data + temporal-selection instruction) against drift.
    RAG_PROMPT = "rag_prompt"

    # Real LLM call via engine.chat() → expected substring(s) in the
    # reply. Hits Ollama: slow, non-deterministic (Metal/MPS isn't
    # bit-perfect even with a seed), so these run OPT-IN via
    # `make eval-llm`, NEVER inside `make check`. See runner.
    LLM_BEHAVIOR = "llm_behavior"

    # Real engine.chat_with_tools() call → expected which TOOL the model
    # picked (and optionally arg substrings), via a side-effect-free stub
    # executor. Same opt-in / drift-tolerant contract as LLM_BEHAVIOR.
    LLM_TOOL_CALL = "llm_tool_call"


# The kinds that hit Ollama. Excluded from the default/offline run and
# from `make check`; only the opt-in `make eval-llm` path includes them.
# RETRIEVAL_SEMANTIC needs the embedder, so it lives here too (its lexical
# sibling RETRIEVAL stays deterministic and in `make check`).
LLM_KINDS = frozenset({
    EvalKind.LLM_BEHAVIOR, EvalKind.LLM_TOOL_CALL, EvalKind.RETRIEVAL_SEMANTIC,
})


class EvalStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    # Soft fail for drift-tolerant cases (LLM behaviour): the check
    # didn't match but we don't block the pipeline — the model output
    # legitimately varies run to run. Visible, not fatal.
    DRIFT = "drift"


@dataclass
class EvalCase:
    """One behavioural check.

    Mandatory: `name`, `kind`, `category`, and the input/expected
    fields for that kind. Description is shown in the report when
    the case fails, so write it as a sentence a human can act on.
    """
    name: str
    kind: EvalKind
    category: str
    description: str = ""

    # ----- DISPATCH fields -----
    # The input text fed to the dispatch simulator.
    input: Optional[str] = None
    # If set, only run when the dispatch simulator resolves to this
    # mode. e.g. "SPOTIFY_PLAY", "MAC_DARK", "OFFLINE".
    expected_mode: Optional[str] = None
    # If set, the dispatch simulator must report that this matcher
    # name fired (e.g. "SPOTIFY_PLAY_RE", "MAC_DARK_TOGGLE_RE",
    # "fallthrough"). Useful for catching the case where two regexes
    # happen to give the same mode but routing changed.
    expected_matcher: Optional[str] = None
    # Captured group from the matching regex (when applicable),
    # e.g. for SPOTIFY_PLAY_RE → "matue".
    expected_capture: Optional[str] = None
    # Language hint for dispatch (rarely used today — strip_vocatives
    # runs before, but kept for future prompt-version evals).
    language: Optional[str] = None

    # ----- DETECTOR fields -----
    # Reply text passed to `detect_hallucinated_action`.
    detector_input: Optional[str] = None
    # True if the reply MUST be flagged as a hallucination, False if
    # it MUST NOT be flagged.
    expected_flagged: Optional[bool] = None

    # ----- INJECTION_GUARD fields -----
    # The (predicado, objeto) pair fed to preferences._safe_to_inject.
    guard_predicado: Optional[str] = None
    guard_objeto: Optional[str] = None
    # True if the pair MUST be deemed safe to splice into the prompt,
    # False if the guard MUST reject it (the adversarial case).
    expected_safe: Optional[bool] = None

    # ----- RETRIEVAL / RETRIEVAL_SEMANTIC fields -----
    # The query run through CortexMemory.retrieve_relevant_context on the
    # synthetic-profile fixture (evals/recall_fixture.py).
    retrieval_query: Optional[str] = None
    # Substring that MUST appear in the retrieved context (the value/section
    # that proves recall worked, e.g. the synthetic passport number).
    retrieval_expected: Optional[str] = None
    # M3-A2: run against the POLLUTED fixture (decoy section + repeated past
    # copies of the question stored as conversation turns) — recall must
    # survive the echo/crowding dynamic, not just a clean corpus.
    retrieval_polluted: bool = False

    # ----- SANITISER fields -----
    # Raw LLM output handed to OllamaEngine._sanitise_response.
    sanitiser_input: Optional[str] = None
    # Substring that MUST appear in the cleaned output.
    expected_contains: Optional[str] = None
    # Substring that MUST NOT appear in the cleaned output.
    expected_not_contains: Optional[str] = None

    # ----- REGEX fields -----
    # Fully-qualified dotted name like
    # "src.core.dispatch.spotify.SPOTIFY_PLAY_RE".
    regex_dotted: Optional[str] = None
    # Text matched against the regex.
    regex_input: Optional[str] = None
    # True if the regex MUST match (via .match or .search per
    # `regex_use_search`); False if it MUST NOT.
    expected_regex_match: Optional[bool] = None
    # Use re.search instead of re.match.
    regex_use_search: bool = False
    # If set, the matched group(1) MUST equal this string (only when
    # expected_regex_match is True).
    expected_regex_group: Optional[str] = None

    # ----- RAG_PROMPT fields -----
    # (input is reused as the USER QUERY; expected_contains_all /
    # expected_not_contains are reused on the BUILT prompt.)
    # Frozen snippet block fed as the LIVE WEB DATA.
    rag_web_context: Optional[str] = None
    # Frozen "today" line, e.g. "2026-06-11 (Thursday)" — None uses the
    # real clock (avoid in cases; freeze it so the case can't rot).
    rag_today: Optional[str] = None

    # ----- LLM_BEHAVIOR fields -----
    # (input is reused as the chat prompt; language above sets the
    # response_language directive.) Matching is case-insensitive
    # substring — deliberately loose, since the reply wording varies.
    # At least one of any/all should be set.
    expected_contains_any: List[str] = field(default_factory=list)
    expected_contains_all: List[str] = field(default_factory=list)
    # When True (default), a non-match reports DRIFT (visible, non-fatal)
    # instead of FAIL. Set False only for a reply that MUST be pinned.
    # Shared with LLM_TOOL_CALL.
    drift_tolerant: bool = True

    # ----- LLM_TOOL_CALL fields -----
    # (input is reused as the user prompt; language sets response_language.)
    # The tool the model is expected to CALL, e.g. "create_event". A miss
    # reports DRIFT (drift_tolerant) — tool selection on an 8B varies.
    expected_tool: Optional[str] = None
    # When True, the model must NOT call any tool (a direct chat answer is
    # correct). Mutually exclusive with expected_tool in practice.
    expect_no_tool: bool = False
    # Optional per-argument substring checks: {arg_name: substring}. Each
    # must appear (case-insensitive) in str(args[arg_name]) of the call.
    expected_arg_contains: Dict[str, str] = field(default_factory=dict)

    # ----- shared -----
    # Free-form tags for filtering / reporting.
    tags: List[str] = field(default_factory=list)

    # Optional task-169 assertions; defaults preserve every existing case.
    # Regexes use re.search; set inline flags (e.g. (?i)) explicitly.
    expected_reply_regex: Optional[str] = None
    forbidden_reply_regex: Optional[str] = None
    expected_first_sentence_regex: Optional[str] = None
    # Apply the existing contains_any/all/not_contains fields to a no-tool
    # reply too. New reply regex fields also apply on that path.
    check_no_tool_reply: bool = False
    expected_arg_equals: Dict[str, str] = field(default_factory=dict)
    expected_arg_regex: Dict[str, str] = field(default_factory=dict)
    forbidden_arg_regex: Dict[str, str] = field(default_factory=dict)
    # Opt in to production's ephemeral turn notes (not persisted in history).
    apply_turn_notes: bool = False


@dataclass
class EvalResult:
    """Outcome of running one EvalCase."""
    case: EvalCase
    status: EvalStatus
    # Short human-readable summary; used by the reporter.
    summary: str = ""
    # Detailed diff (multi-line allowed) when status is FAIL/ERROR.
    detail: str = ""
    # Free-form payload for the JSON reporter.
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Case registry. Each cases/<file>.py exposes a top-level `CASES: list[EvalCase]`.
# The runner walks `evals.cases.__path__`, imports each submodule, and
# collects them. Submodules don't need to register themselves explicitly.
# ---------------------------------------------------------------------------

def load_all_cases(include_llm: bool = False) -> List[EvalCase]:
    """Walk evals.cases.* and concatenate every CASES list found.

    Empty / missing CASES on a submodule is fine — it just contributes
    nothing. An ImportError aborts the run with a clear pointer (we
    want surfacing broken case modules immediately, not silently).

    `include_llm` defaults to False so the deterministic consumers (the
    default CLI run, the `test_all_cases_pass` self-test, `make check`)
    never pull LLM cases (LLM_KINDS) — those hit Ollama and would make
    the offline/fast suite slow and flaky. The opt-in `make eval-llm`
    path passes include_llm=True.
    """
    import importlib
    import pkgutil

    from evals import cases as cases_pkg

    out: List[EvalCase] = []
    for modinfo in pkgutil.iter_modules(cases_pkg.__path__):
        fqname = f"evals.cases.{modinfo.name}"
        module = importlib.import_module(fqname)
        chunk = getattr(module, "CASES", None)
        if chunk is None:
            continue
        if not isinstance(chunk, list):
            raise TypeError(
                f"{fqname}.CASES must be a list of EvalCase, got "
                f"{type(chunk).__name__}"
            )
        out.extend(chunk)
    if not include_llm:
        out = [c for c in out if c.kind not in LLM_KINDS]
    return out
