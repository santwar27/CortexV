# Cortex evals

Behavioural regression suite. Sister of `tests/` with a different
contract:

- **`tests/`** — unit/integration. Every test MUST pass on every
  commit. Failure blocks merge.
- **`evals/`** — behavioural. "When the user says X, the system
  produces Y." A failure means behaviour drifted. Sometimes that's
  intentional (e.g. prompt v2 changed answer style); sometimes
  it's a regression. The runner reports drift; the human decides.

## Run

```bash
venv/bin/python -m evals                  # all cases
venv/bin/python -m evals --category spotify
venv/bin/python -m evals --kind dispatch
venv/bin/python -m evals --tag regression
venv/bin/python -m evals --verbose        # show every case, not just failures
venv/bin/python -m evals --json out.json  # also write JSON
venv/bin/python -m evals --json -         # JSON to stdout
```

Exit code: `0` all green, `1` any failure/error, `2` no cases
matched the filter (catches typo'd category).

## Adding a case

Pick the file in `evals/cases/` matching the area, or create a new
one. Each file exposes a top-level `CASES: list[EvalCase]`. The
loader walks `evals.cases.*` automatically — no registration needed.

```python
# evals/cases/spotify.py
from evals.core import EvalCase, EvalKind

CASES: list[EvalCase] = [
    EvalCase(
        name="play_natural_phrasing",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="bota uma música do djonga",
        expected_mode="OFFLINE",       # falls through, but...
        # ...TOOL_HINT_RE should mark this for chat_with_tools:
        description="Natural phrasing routes via LLM tools.",
    ),
    # ... existing cases ...
]
```

## Eval kinds

- **DISPATCH** — input text → `simulate_dispatch()` → expected
  `mode` / `matcher` / `capture`. Replicates the orchestrator's
  `_process_core` regex ordering without instantiating the full
  pipeline (no Palace, no Ollama, no audit log).
- **DETECTOR** — reply text → `detect_hallucinated_action()` →
  expected `bool`. Tests the bloco-H anti-hallucination guard.
- **SANITISER** — raw LLM output → `OllamaEngine._sanitise_response()`
  → expected `contains` / `not_contains`. Tests both legacy JSON
  artefact stripping AND the bloco-12B brace-wrapped pseudo-tool-call
  remover.
- **REGEX** — direct regex match/search against a named pattern,
  with optional `group(1)` assertion. Good for edge cases that
  don't naturally fit DISPATCH (e.g. testing `MAC_BRIGHTNESS_SET_RE`
  in isolation).
- **LLM_BEHAVIOR** — real `engine.chat()` call → expected
  `contains_any` / `contains_all` (case-insensitive substring). See
  below.
- **LLM_TOOL_CALL** — real `engine.chat_with_tools()` call → expected
  which TOOL the model picked (`expected_tool`) or that it picked none
  (`expect_no_tool`), plus optional `expected_arg_contains`. Uses a
  side-effect-free stub executor: nothing is sent/scheduled/read. See
  below.

## LLM behaviour cases (the only kind that hits Ollama)

The four kinds above are deterministic, offline, and run in <1s — they
ride `make check` (pre-commit + pre-push). `LLM_BEHAVIOR` is different:
it calls the real model through `engine.chat()`.

```
make eval-llm                         # default model (CORTEX_MODEL or cortex-lite)
CORTEX_MODEL=cortex-gemma make eval-llm   # grade a different model, same cases
venv/bin/python -m evals --llm-only       # equivalent
```

Rules that make this safe to live alongside the fast suite:

- **Opt-in only.** Excluded from `make check`, the default `python -m
  evals` run, and the `test_all_cases_pass` self-test (via
  `load_all_cases(include_llm=False)`). It needs Ollama running.
- **Drift-tolerant.** Output varies run to run — Metal/MPS reduction
  order isn't bit-perfect even with `temperature=0, seed=42`. A miss
  reports **DRIFT** (yellow `↯`), which is *visible but non-fatal*:
  exit code stays 0, the run doesn't go red. Only `FAIL`/`ERROR` (a
  `drift_tolerant=False` case, or a transport crash) breaks the run.
- **Model-agnostic.** Cases are pure input + expected substrings — no
  model name anywhere. The runner reads `CORTEX_MODEL` so the same
  cases grade Llama today and Gemma later (A/B is a future session).
- **No side effects.** The eval engine's memory is stubbed: no Palace
  retrieval (reply reflects the prompt alone) and no writes (eval Q&A
  never pollutes the long-term store). History resets per case.

Write a case with the `_llm(...)` helper in `cases/llm_behavior.py`.
Use `expected_contains_any` for "any of these words is fine" and
`expected_contains_all` when several must all appear. Keep matchers
loose — you're checking the model *got the gist*, not exact wording.

### Baseline + cosine drift

Substring matching answers "did the reply contain the right words". It
can't tell you the reply *shifted shape* since last time — a model swap
or prompt-version bump can keep the substrings while changing the whole
answer. That's what the baseline catches.

```
make eval-llm-rebaseline                  # record replies as the baseline
make eval-llm                             # run + compare against the baseline
```

- A baseline is one JSON file per `(model, prompt_version)`:
  `evals/baselines/<model>__v<N>.json` (e.g. `cortex-lite__v1.json`).
  Swapping model or bumping the prompt writes a **new** file and leaves
  the old one — `git diff` on a rebaseline is the record of how answers
  moved, and the kept files are the raw material for A/B comparison.
- Re-recorded **only** by the explicit `eval-llm-rebaseline` target — a
  baseline is a deliberate "these answers are correct now", never a
  side effect of a normal run.
- On a plain `make eval-llm`, each reply is embedded (`nomic-embed-text`
  via Ollama, same model the Palace uses) and scored by **cosine
  similarity** against the recorded reply. Below **0.92** is reported as
  baseline drift. The threshold is `< 1.0` on purpose: Metal/MPS isn't
  bit-perfect even at `temperature=0, seed=42`.
- **This is a second, independent axis.** A case can PASS its substring
  check yet drift from baseline, or DRIFT on substring yet stay cosine-
  identical to baseline (a *consistently* wrong answer — e.g.
  `explain_rag`, where cortex-lite reliably mis-defines "RAG"). Cosine
  drift is informational: it shows in the report, never changes the
  exit code.
- The baseline files ARE committed. Commit the diff when you rebaseline.

### Tool-call cases (`chat_with_tools`)

`LLM_BEHAVIOR` exercises `chat()` — pure conversation. But an ACTION
("envia um email", "agenda uma reunião") routes through
`chat_with_tools`, where the real question is *did the model pick the
right tool?* `LLM_TOOL_CALL` cases check exactly that.

```python
# evals/cases/llm_tool_call.py
_tool("send_email_pt",
      "envia um email para maria@empresa.com avisando que vou atrasar",
      tool="send_email", args={"to": "maria"})
_tool("stable_knowledge_no_tool_pt",
      "o que é fotossíntese?", no_tool=True)   # must NOT web_search
```

- The runner exposes the **real `TOOL_SPECS`** and runs the **real**
  tool-calling loop — faithful to production. Only the executor is
  swapped for a stub that records `(name, args)` and returns the
  skip-followup marker, so **nothing fires** and the slow second LLM
  round is skipped.
- `expected_tool` asserts which tool was called; `expect_no_tool`
  asserts none was (over-tooling guard — e.g. stable knowledge should
  be answered directly, not web-searched).
- `expected_arg_contains={"to": "maria"}` checks an arg substring
  (case-insensitive). Keep these light — an 8B won't reproduce exact
  phrasing, so pin only gross correctness.
- Same drift policy as `LLM_BEHAVIOR`: a miss is **DRIFT**, not failure.
- These run in `make eval-llm` alongside the chat cases. They are NOT
  baselined (cosine on a tool name is meaningless — the tool name IS
  the discrete check).

Not yet here (later sessions): A/B model comparison framework, CI
integration.

## Why a parallel suite (vs more `tests/`)?

`tests/` enforces unit-level correctness on a granular surface.
Evals are about **observable system behaviour** — "the original
Matuê hallucination input must keep routing to SPOTIFY_PLAY".
Splitting them lets:

- `evals/` be **looser** (drift is informational, not blocking)
- `tests/` stay **strict** (any failure blocks merge)
- Future LLM-quality evals (sessão futura) live somewhere they're
  not expected to be 100% deterministic

The `tests/test_evals_framework.py` self-test ensures the
plumbing stays airtight, including a final assertion that every
shipped case currently passes — so a behaviour regression DOES
fail the test suite. The "evals are looser" stance is reserved
for cases we mark as drift-tolerant (none yet; will come with
LLM-quality evals).

## Files

```
evals/
├── __init__.py           # package docstring + run hints
├── __main__.py           # CLI: python -m evals
├── core.py               # EvalCase, EvalKind, EvalResult, loader
├── dispatch.py           # simulate_dispatch — replicates _process_core order
├── runner.py             # per-kind handlers + filters
├── report.py             # ANSI terminal output (auto-disabled for non-TTY)
├── similarity.py         # cosine + nomic-embed-text wrapper (injectable)
├── llm_baseline.py       # baseline path / save / load (one per model+version)
├── baselines/            # recorded LLM replies, committed (<model>__v<N>.json)
├── README.md             # this file
└── cases/
    ├── __init__.py
    ├── spotify.py        # SPOTIFY_* family + polite forms + EN
    ├── macos.py          # mac volume/brightness/dark/lock/screenshot/open/quit
    ├── anti_hallucination.py  # detector positive + negative + edge
    ├── regression.py     # specific historical bugs (tagged jou-sNN-blocoX)
    ├── studier.py        # nightly Studier verbs
    ├── conflict.py       # priority guarantees between competing dispatchers
    ├── multi_intent.py   # compound "X and Y" → planner autoroute
    ├── llm_behavior.py   # real engine.chat() calls (opt-in, make eval-llm)
    └── llm_tool_call.py  # real engine.chat_with_tools() — tool selection
```

## Conventions

- Case `name` is snake_case, unique within `category`.
- `description` is a sentence a human can act on when the case
  fails — avoid restating the input.
- Tag `regression` + `jou-sNN-blocoX` for bugs traced to a specific
  JOURNAL entry; tag `conflict` for priority-between-dispatchers
  cases.
- Never silently downgrade an expectation. If a case is failing
  because the behaviour DID change deliberately, document the
  change in JOURNAL and update the case in the same commit.
