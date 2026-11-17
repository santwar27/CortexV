"""
Plan-Execute Multi-step — Tier 3.1.

Decomposes a multi-step objective into a topologically-ordered sequence
of tool calls, executes them sequentially with output references
({p1.field}) flowing between steps, and aggregates the results into a
single user-facing reply.

Design reference: Cap 9 `planner/decompose.py` — Plan-Execute as default,
Tree of Thought only when caminhos qualitativamente diferentes + custo
alto de erro + função de avaliação articulável (~2% dos casos). ToT is
explicitly out of scope here.

Pipeline:

  1. decompose(objective, llm_chat_fn, model, tool_names) →
       LLM emits JSON describing a Plan: objective + ordered Steps with
       dependencies and arg references. We validate the shape and the
       dependency graph BEFORE persisting anything — a malformed plan
       returns None so the orchestrator falls back to single-step
       dispatch.

  2. _topological_order(plan) →
       Kahn's algorithm produces a deterministic linear order. Cycle
       detected → raises PlannerError. Multiple valid topo orders →
       we pick the one matching the plan's declared step order, so
       previews stay stable.

  3. execute(plan, tool_executor) →
       Walks the topo order. For each Step, resolves any {p<id>.<field>}
       references in args against prior StepResults, then calls
       tool_executor(name, resolved_args). On any failure, fail-stop:
       remaining steps are marked `skipped`. The user's plan-level
       approval (granted via the orchestrator's _pending_action flow
       before execute() runs) covers all steps — we do NOT re-prompt
       per S3 step, because the preview already showed every action.

  4. summarise(plan, results, llm_chat_fn, model) →
       LLM aggregates the StepResults into a coherent pt-BR reply.
       Falls back to a deterministic bullet list when the LLM call
       fails — the user always sees an outcome.

Design constraints:

  - Sequential execution. Parallelising independent steps would be
    valid topologically but creates races on _pending_action and on
    audit ordering. Defer until measured worth.

  - All LLM calls go through an injected llm_chat_fn so the module is
    fully unit-testable without an Ollama process. Same convention as
    preferences.py and consolidator.py.

  - The plan is FROZEN before execution. No mid-flight re-planning in
    v1 — if a step fails, the user sees the partial result and can
    issue a new objective. Closed-loop replanning is Tier 4 work.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.audit import audit

from src.core.i18n import t

log = logging.getLogger(__name__)


# =====================================================================
# data model
# =====================================================================


@dataclass(frozen=True)
class Step:
    """One node in the plan graph.

    id          short string like 'p1', 'p2'. Used as the key for ref
                substitution: a later step can reference this step's
                output via '{p1.field}' inside its args.
    tool        name of the tool to invoke. Must be in the set the
                orchestrator passed to decompose().
    args        dict of arg names → values. Values may be strings
                containing '{p<id>.<field>}' refs that resolve to
                prior steps' outputs at execution time.
    depende_de  list of step ids this step needs to run after. A step
                that uses '{p1.field}' gets 'p1' added here during JSON
                parsing so the topological sort is correct.
    descricao   short pt-BR human-readable summary, for the preview.
    """
    id: str
    tool: str
    args: Dict[str, Any]
    depende_de: List[str] = field(default_factory=list)
    descricao: str = ""


@dataclass(frozen=True)
class Plan:
    """A decomposed multi-step objective ready for execution."""
    objetivo: str
    passos: List[Step]
    contexto_inicial: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Outcome of executing one Step.

    success   True iff the tool returned without error AND the output
              was non-empty / non-error-shaped.
    output    Raw return value of tool_executor. Usually a string;
              ref substitution can grab .text for strings or named
              fields if the tool returned a dict.
    error     Short error message when success=False. Empty when ok.
    skipped   True when a prior failure aborted the plan before this
              step ran. Distinguishes "didn't run" from "ran and failed".
    """
    step_id: str
    success: bool
    output: Any = None
    error: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class PlanResult:
    """Aggregate outcome of execute()."""
    plan: Plan
    results: List[StepResult]
    success: bool       # all non-skipped steps succeeded
    error: str = ""     # short error from the first failed step, if any
    cancelled: bool = False
    paused_at: Optional[str] = None


class PlannerError(Exception):
    """Raised for malformed plans the validator can detect statically:
    duplicate step ids, missing dependencies, cycles, etc. The
    orchestrator should treat this as 'fall back to single-step LLM
    dispatch' rather than surface the exception to the user."""


# =====================================================================
# validation + topological order
# =====================================================================


_STEP_ID_RE = re.compile(r"^p[0-9]{1,3}$")  # p1, p2, ..., p999
_REF_RE = re.compile(r"\{(p[0-9]{1,3})(?:\.([a-zA-Z_][a-zA-Z0-9_]*))?\}")


def _validate_plan(plan: Plan, allowed_tools: Optional[List[str]] = None) -> None:
    """Static checks on a Plan. Raises PlannerError on any issue.

    Checks:
      - At least one step.
      - Step ids match the p<N> shape and are unique.
      - Every step.tool is in allowed_tools (when provided).
      - Every depende_de id points to a known step.
      - Every {p<id>.<field>} ref in args points to a known step.
      - No cycles in the dependency graph.
    """
    if not plan.passos:
        raise PlannerError("plan has no steps")

    seen_ids: set = set()
    for step in plan.passos:
        if not _STEP_ID_RE.match(step.id):
            raise PlannerError(f"invalid step id {step.id!r} (expected p1..p999)")
        if step.id in seen_ids:
            raise PlannerError(f"duplicate step id {step.id!r}")
        seen_ids.add(step.id)
        if not step.tool:
            raise PlannerError(f"step {step.id!r} has no tool")
        if allowed_tools is not None and step.tool not in allowed_tools:
            raise PlannerError(
                f"step {step.id!r} uses unknown tool {step.tool!r} "
                f"(allowed: {sorted(allowed_tools)})"
            )

    # Dependencies and refs must point to known step ids.
    for step in plan.passos:
        for dep in step.depende_de:
            if dep not in seen_ids:
                raise PlannerError(
                    f"step {step.id!r} depends on unknown step {dep!r}"
                )
        # Refs inside arg values
        for arg_value in step.args.values():
            if not isinstance(arg_value, str):
                continue
            for m in _REF_RE.finditer(arg_value):
                ref_id = m.group(1)
                if ref_id not in seen_ids:
                    raise PlannerError(
                        f"step {step.id!r} references unknown step {ref_id!r}"
                    )
                if ref_id == step.id:
                    raise PlannerError(
                        f"step {step.id!r} references itself"
                    )

    # Cycle detection via Kahn's. If we can't drain every node, there's
    # a cycle.
    _topological_order(plan)


def _topological_order(plan: Plan) -> List[Step]:
    """Kahn's algorithm. Returns Steps in a deterministic linear order
    that respects depende_de. Among equally-eligible steps at any
    point, we pick the one that came first in plan.passos — keeps
    previews stable across runs."""
    by_id = {s.id: s for s in plan.passos}
    in_degree = {s.id: len(s.depende_de) for s in plan.passos}
    # Pre-sort by original plan order so the queue picks the
    # earliest-declared eligible step.
    plan_order = {s.id: i for i, s in enumerate(plan.passos)}
    ordered: List[Step] = []
    ready = sorted(
        [sid for sid, d in in_degree.items() if d == 0],
        key=lambda sid: plan_order[sid],
    )
    while ready:
        sid = ready.pop(0)
        ordered.append(by_id[sid])
        for s in plan.passos:
            if sid in s.depende_de:
                in_degree[s.id] -= 1
                if in_degree[s.id] == 0:
                    # Insert in plan order
                    idx = next(
                        (i for i, x in enumerate(ready)
                         if plan_order[x] > plan_order[s.id]),
                        len(ready),
                    )
                    ready.insert(idx, s.id)
    if len(ordered) != len(plan.passos):
        missing = [s.id for s in plan.passos if s.id not in {o.id for o in ordered}]
        raise PlannerError(f"cycle detected — could not order {missing}")
    return ordered


# =====================================================================
# ref resolution
# =====================================================================


def _resolve_refs(args: Dict[str, Any], prior: Dict[str, StepResult], *,
                  keep_unresolved: bool = False) -> Dict[str, Any]:
    """Substitute {p<id>.<field>} occurrences inside string arg values
    against `prior` step outputs.

    `keep_unresolved` preserves future references when staging a remainder.

    Lookup rules for each ref:
      - No field, or field='text' → str(prior[id].output)
      - prior[id].output is a dict → prior[id].output.get(field, '')
      - Otherwise → str(prior[id].output)

    Unknown refs (step missing from prior) are left literal — the
    caller already validated graph shape, so this only fires when
    upstream skipped/failed. Empty string is the safe fallback rather
    than raising mid-execute.

    REF CONTRACT: tool outputs are plain TEXT, so the supported ref is
    `{pN}` — the whole upstream output. The decompose prompt instructs
    the LLM to use exactly that and to NEVER emit `.field` accessors
    (there's nothing structured to index into). The `.field` branch
    below is kept as defensive tolerance: if the LLM regresses and emits
    `{p1.title}` anyway, we substitute the whole string and log a
    warning rather than dropping the data. The dict path (genuine field
    extraction) stays wired so a future tool that DOES return a dict
    works for free — no tool returns one today."""
    resolved: Dict[str, Any] = {}
    for key, value in args.items():
        if not isinstance(value, str):
            resolved[key] = value
            continue

        def _sub(match: re.Match) -> str:
            step_id = match.group(1)
            field_name = match.group(2)
            res = prior.get(step_id)
            if res is None or not res.success:
                if keep_unresolved:
                    return match.group(0)
                return ""  # upstream missing or failed
            out = res.output
            if isinstance(out, dict):
                # Structured output. The runner convention (M4-B1) is
                # {"text": str, "fields": dict}: surfaces show "text",
                # refs extract from "fields". A plain dict (no
                # convention) keeps the original top-level behaviour.
                if field_name:
                    fields = out.get("fields")
                    if isinstance(fields, dict) and field_name in fields:
                        return str(fields[field_name])
                    return str(out.get(field_name, ""))
                if isinstance(out.get("text"), str):
                    return out["text"]
                return json.dumps(out, ensure_ascii=False)
            # Non-dict (usually a plain string) output. 'text' is the
            # convention for "the whole output", so it's expected; any
            # OTHER field name can't be honoured — warn and fall back to
            # the whole string (see KNOWN LIMITATION above).
            if field_name and field_name != "text" and out is not None:
                log.warning(
                    "planner: ref {%s.%s} requested a field but step %s "
                    "returned a %s, not a dict — substituting the whole "
                    "output. Tool must return a dict for field extraction.",
                    step_id, field_name, step_id, type(out).__name__,
                )
            return str(out) if out is not None else ""

        resolved[key] = _REF_RE.sub(_sub, value)
    return resolved


# =====================================================================
# decompose — LLM-based plan generation
# =====================================================================


_DECOMPOSE_SYSTEM = """You decompose a user objective into a sequence of tool calls. Output ONLY a JSON object. No prose, no markdown fences, nothing outside the object.

TEMPORAL CONTEXT (use this for relative dates like 'hoje', 'amanhã', 'sexta'):
  {TEMPORAL_CONTEXT}

When the objective mentions a relative date, COMPUTE the absolute ISO 8601 timestamp yourself using the temporal context above. Include the timezone offset shown there. NEVER emit placeholder syntax like "{date: +1}" or "{tomorrow}" or "TODAY+1" — those are not real refs and the tools will fail on them. The ONLY ref form the runtime understands is "{p<id>}" — the WHOLE text output of a PRIOR STEP. Tool outputs are plain TEXT, not structured records, so you CANNOT pull a sub-field out of them: never write "{p1.title}", "{p1.date}", "{p1.to}" or any ".field" accessor. If a later step needs information from an earlier step, pass the entire output with "{p<id>}" and let the tool read what it needs.

The object MUST have this exact shape:
{
  "objetivo": "<concise restatement of the user's goal in pt-BR>",
  "passos": [
    {
      "id": "p1",
      "tool": "<tool_name>",
      "args": { "<arg_name>": "<value or {p<id>} ref>", ... },
      "depende_de": [],
      "descricao": "<one-line pt-BR summary of what this step does>"
    },
    ...
  ]
}

Rules:
  - Step ids are "p1", "p2", ... in declaration order.
  - depende_de lists the ids of steps that MUST run before this one.
  - To reference an earlier step's output, put "{p<id>}" inside an arg value as a string. The runtime substitutes the prior step's WHOLE text output at execution time. THIS IS THE ONLY REF FORM — never use ".field" accessors (e.g. "{p1.title}"), and never use "{date: ...}", "{tomorrow}", "TODAY+1" or similar.
  - CRITICAL EXTRACTION LIMIT: tool outputs come back as PLAIN TEXT prose. You CANNOT reliably extract a specific date, time, name, ID, or any other structured value out of that prose and feed it as a typed argument to a downstream step. If the user's objective would require that (e.g. "search for X and then schedule it"), return ONLY the steps you CAN execute deterministically (typically just the search), and let the user read the result and follow up. NEVER fabricate a downstream step whose args depend on extraction you can't perform — that is worse than no plan.
  - All dates and times MUST be concrete ISO 8601 strings with timezone offset (e.g. "2026-05-24T14:00:00-03:00"), computed from the temporal context above. NEVER use a `{p<id>}` ref where an ISO timestamp is expected — refs return prose, not timestamps.
  - Use ONLY the tools listed below — never invent a tool name. `list_events` reads the USER'S calendar; it does NOT parse a web_search result.
  - Keep the plan minimal: don't add steps that aren't needed for the objective.
  - Return at most 5 steps.

If the objective doesn't need decomposition (one tool is enough, or no tool is needed), still return a single-step plan with just that one tool — OR return {"objetivo": "...", "passos": []} if nothing tool-shaped applies.

Available tools (only these): {AVAILABLE_TOOLS}

Example 1 (assuming today is 2026-05-23 sex) — no ref needed:

Objective: "marca uma reunião com bruno amanhã às 14h e me lembre disso 30 min antes"
Output:
{"objetivo":"agendar reunião com Bruno e criar lembrete prévio","passos":[{"id":"p1","tool":"create_event","args":{"summary":"Reunião com Bruno","start_iso":"2026-05-24T14:00:00-03:00","end_iso":"2026-05-24T15:00:00-03:00"},"depende_de":[],"descricao":"agenda a reunião"},{"id":"p2","tool":"create_reminder","args":{"title":"Reunião com Bruno em 30min","due_iso":"2026-05-24T13:30:00-03:00"},"depende_de":["p1"],"descricao":"lembrete 30 min antes"}]}

Example 2 — passing a prior step's WHOLE output forward with {p1}:

Objective: "pesquisa o placar do jogo do Flamengo e me manda por email"
Output:
{"objetivo":"buscar o placar e enviar por email","passos":[{"id":"p1","tool":"web_search","args":{"query":"placar do jogo do Flamengo hoje"},"depende_de":[],"descricao":"busca o placar"},{"id":"p2","tool":"send_email","args":{"subject":"Placar do Flamengo","body":"{p1}"},"depende_de":["p1"],"descricao":"envia o resultado da busca"}]}

Example 3 — EXTRACTION-LIMIT honesty (do NOT fabricate a step you can't execute):

Objective: "procura os próximos dois jogos da Seleção Brasileira e agenda no meu calendário"
Output:
{"objetivo":"buscar os próximos jogos da Seleção Brasileira","passos":[{"id":"p1","tool":"web_search","args":{"query":"próximos jogos da seleção brasileira de futebol"},"depende_de":[],"descricao":"busca os próximos jogos"}]}

Notes on Example 3: the calendar step is NOT added. It would require extracting specific dates from p1's free-text output and feeding them as start_iso/end_iso — extraction the runtime cannot perform on text. The user reads p1's result and asks next: "agenda jogo do Brasil dia 13/06 às 19h" (a single-step plan with concrete dates). NEVER write start_iso="{p1.start_iso}" — that ref is invalid (`.field` accessors are forbidden) AND would inject prose where a timestamp is expected.
"""


def _extraction_enabled() -> bool:
    """M1-B2 (s66): DORMANT flag for planner field-extraction. The base
    prompt's EXTRACTION LIMIT exists because the 8B can't pull a date out
    of prose — so "busca os jogos E agenda" never schedules, by design.
    A frontier-class brain does that extraction trivially; this flag swaps
    the decompose prompt for the variant that allows `{pN.field}` refs.
    Flip on brain-swap day, AFTER the runners that feed chained plans
    return the {"text", "fields"} convention (M4-B1) — _resolve_refs
    already understands both."""
    return os.environ.get("CORTEX_PLANNER_EXTRACTION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# The extraction variant (M1-B2), built from the base prompt by targeted
# replaces so the two can never drift on the parts they share. Each
# replace is asserted below — a silent no-op would ship the base prompt
# under the extraction flag (the V5 lesson from prompts.py).
_BASE_REF_PARAGRAPH = (
    'The ONLY ref form the runtime understands is "{p<id>}" — the WHOLE text '
    "output of a PRIOR STEP. Tool outputs are plain TEXT, not structured "
    'records, so you CANNOT pull a sub-field out of them: never write '
    '"{p1.title}", "{p1.date}", "{p1.to}" or any ".field" accessor. If a '
    "later step needs information from an earlier step, pass the entire "
    'output with "{p<id>}" and let the tool read what it needs.'
)
_EXTRACTION_REF_PARAGRAPH = (
    'Ref forms the runtime understands: "{p<id>}" — the WHOLE text output '
    'of a PRIOR STEP — and "{p<id>.<field>}" — a STRUCTURED FIELD of that '
    'step\'s output (e.g. "{p1.start_iso}"). Use a .field ref only when the '
    "downstream arg needs a specific typed value; pass the whole output "
    'with "{p<id>}" otherwise.'
)

_BASE_REF_RULE = (
    '  - To reference an earlier step\'s output, put "{p<id>}" inside an arg '
    "value as a string. The runtime substitutes the prior step's WHOLE text "
    'output at execution time. THIS IS THE ONLY REF FORM — never use '
    '".field" accessors (e.g. "{p1.title}"), and never use "{date: ...}", '
    '"{tomorrow}", "TODAY+1" or similar.'
)
_EXTRACTION_REF_RULE = (
    '  - To reference an earlier step\'s output, put "{p<id>}" (whole text) '
    'or "{p<id>.<field>}" (structured field, e.g. "{p1.start_iso}") inside '
    'an arg value as a string. Never use "{date: ...}", "{tomorrow}", '
    '"TODAY+1" or similar.'
)

_BASE_EXTRACTION_LIMIT_RULE = (
    "  - CRITICAL EXTRACTION LIMIT: tool outputs come back as PLAIN TEXT "
    "prose. You CANNOT reliably extract a specific date, time, name, ID, or "
    "any other structured value out of that prose and feed it as a typed "
    "argument to a downstream step. If the user's objective would require "
    'that (e.g. "search for X and then schedule it"), return ONLY the steps '
    "you CAN execute deterministically (typically just the search), and let "
    "the user read the result and follow up. NEVER fabricate a downstream "
    "step whose args depend on extraction you can't perform — that is worse "
    "than no plan."
)
_EXTRACTION_LIMIT_RULE = (
    "  - EXTRACTION: when a downstream step needs a specific typed value "
    "(date, time, name, ID) produced by an earlier step, reference it as "
    '"{p<id>.<field>}" — the runtime extracts it from that step\'s '
    "structured fields. If the value the objective hinges on cannot be "
    "obtained this way, return only the steps you CAN execute and let the "
    "user follow up — never fabricate a step."
)

_BASE_ISO_RULE = (
    "NEVER use a `{p<id>}` ref where an ISO timestamp is expected — refs "
    "return prose, not timestamps."
)
_EXTRACTION_ISO_RULE = (
    "A whole-output `{p<id>}` ref is prose and NEVER goes where an ISO "
    "timestamp is expected — use a `{p<id>.<field>}` ref to a field that "
    "carries a timestamp (e.g. start_iso) instead."
)

_BASE_EXAMPLE_3 = (
    "Example 3 — EXTRACTION-LIMIT honesty (do NOT fabricate a step you "
    "can't execute):\n\nObjective: \"procura os próximos dois jogos da "
    "Seleção Brasileira e agenda no meu calendário\"\nOutput:\n"
    '{"objetivo":"buscar os próximos jogos da Seleção Brasileira","passos":'
    '[{"id":"p1","tool":"web_search","args":{"query":"próximos jogos da '
    'seleção brasileira de futebol"},"depende_de":[],"descricao":"busca os '
    'próximos jogos"}]}\n\nNotes on Example 3: the calendar step is NOT '
    "added. It would require extracting specific dates from p1's free-text "
    "output and feeding them as start_iso/end_iso — extraction the runtime "
    "cannot perform on text. The user reads p1's result and asks next: "
    '"agenda jogo do Brasil dia 13/06 às 19h" (a single-step plan with '
    'concrete dates). NEVER write start_iso="{p1.start_iso}" — that ref is '
    "invalid (`.field` accessors are forbidden) AND would inject prose "
    "where a timestamp is expected."
)
_EXTRACTION_EXAMPLE_3 = (
    "Example 3 — chaining with field refs (search feeds the calendar):\n\n"
    "Objective: \"procura o próximo jogo da Seleção Brasileira e agenda no "
    "meu calendário\"\nOutput:\n"
    '{"objetivo":"buscar o próximo jogo da Seleção e agendar","passos":'
    '[{"id":"p1","tool":"web_search","args":{"query":"próximo jogo da '
    'seleção brasileira de futebol data e horário"},"depende_de":[],'
    '"descricao":"busca o próximo jogo"},{"id":"p2","tool":"create_event",'
    '"args":{"summary":"{p1.title}","start_iso":"{p1.start_iso}",'
    '"end_iso":"{p1.end_iso}"},"depende_de":["p1"],"descricao":"agenda o '
    'jogo encontrado"}]}\n\nNotes on Example 3: the .field refs are only '
    "valid because the runtime extracts structured fields from p1's "
    "output. If a needed field may not exist, prefer the honest single-"
    "step plan."
)

_DECOMPOSE_SYSTEM_EXTRACTION = (
    _DECOMPOSE_SYSTEM
    .replace(_BASE_REF_PARAGRAPH, _EXTRACTION_REF_PARAGRAPH)
    .replace(_BASE_REF_RULE, _EXTRACTION_REF_RULE)
    .replace(_BASE_EXTRACTION_LIMIT_RULE, _EXTRACTION_LIMIT_RULE)
    .replace(_BASE_ISO_RULE, _EXTRACTION_ISO_RULE)
    .replace(_BASE_EXAMPLE_3, _EXTRACTION_EXAMPLE_3)
)
for _marker in (_EXTRACTION_REF_PARAGRAPH, _EXTRACTION_REF_RULE,
                _EXTRACTION_LIMIT_RULE, _EXTRACTION_ISO_RULE,
                _EXTRACTION_EXAMPLE_3):
    assert _marker in _DECOMPOSE_SYSTEM_EXTRACTION, (
        "planner: an extraction-prompt replace did not fire — the base "
        "prompt text drifted; update the _BASE_* constant to match."
    )
assert "CRITICAL EXTRACTION LIMIT" not in _DECOMPOSE_SYSTEM_EXTRACTION
del _marker


# Regex matches a JSON object (outermost braces).
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_plan_json(content: str, allowed_tools: Optional[List[str]] = None) -> Optional[Plan]:
    """Extract and validate a Plan from raw LLM output. Returns None on
    any parse / validation failure (caller falls back to single-step
    dispatch)."""
    match = _JSON_OBJECT_RE.search(content)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    objetivo = str(raw.get("objetivo", "")).strip()
    raw_passos = raw.get("passos") or []
    if not isinstance(raw_passos, list):
        return None

    steps: List[Step] = []
    for raw_p in raw_passos:
        if not isinstance(raw_p, dict):
            return None
        sid = str(raw_p.get("id", "")).strip()
        tool = str(raw_p.get("tool", "")).strip()
        args = raw_p.get("args") or {}
        depende_de = raw_p.get("depende_de") or []
        descricao = str(raw_p.get("descricao", "")).strip()
        if not isinstance(args, dict) or not isinstance(depende_de, list):
            return None
        # The model may omit dependencies, but refs still require their
        # producers to run first. Derive those edges before validation.
        dependencies = [str(d) for d in depende_de]
        for value in args.values():
            if isinstance(value, str):
                for ref in _REF_RE.finditer(value):
                    if ref.group(1) not in dependencies:
                        dependencies.append(ref.group(1))
        steps.append(Step(
            id=sid,
            tool=tool,
            args=dict(args),
            depende_de=dependencies,
            descricao=descricao,
        ))

    plan = Plan(objetivo=objetivo, passos=steps)
    try:
        _validate_plan(plan, allowed_tools=allowed_tools)
    except PlannerError as e:
        log.warning("planner: plan validation failed: %s", e)
        return None
    return plan


def _build_temporal_context() -> str:
    """Render the current date + local time + tz for the planner prompt.
    Mirrors engine._build_system_prompt so absolute-date computations
    stay consistent between the LLM main path and the planner."""
    from datetime import datetime
    now_local = datetime.now().astimezone()
    return (
        f"- Current date: {now_local.strftime('%Y-%m-%d (%A)')}\n"
        f"  - Current local time: {now_local.strftime('%H:%M')} "
        f"({now_local.tzname() or now_local.strftime('%z')})\n"
        f"  - When generating ISO 8601 timestamps, include the timezone "
        f"offset shown above ({now_local.strftime('%z')})"
    )


def decompose(
    objective: str,
    llm_chat_fn: Callable[..., Any],
    model: str,
    allowed_tools: List[str],
    max_input_chars: int = 4000,
) -> Optional[Plan]:
    """Call the LLM with the objective and the list of available tools,
    parse the returned JSON, validate, return a Plan or None.

    `allowed_tools` is the set the orchestrator exposes (tool spec
    names, e.g. read_document, send_email, etc). The prompt is built
    from it dynamically so the planner only ever produces plans the
    orchestrator can execute.

    The temporal context (current date/time/timezone) is injected at
    call time so the LLM can compute absolute ISO 8601 timestamps for
    relative phrases like 'amanhã'. Without this the local 8B model
    invents placeholder syntax like `{date: +1}` that would later
    blow up when handed to Calendar/Reminders.
    """
    if not isinstance(objective, str) or not objective.strip():
        return None
    if not allowed_tools:
        return None

    base = (
        _DECOMPOSE_SYSTEM_EXTRACTION if _extraction_enabled()
        else _DECOMPOSE_SYSTEM
    )
    system = base.replace(
        "{TEMPORAL_CONTEXT}", _build_temporal_context(),
    ).replace(
        "{AVAILABLE_TOOLS}", ", ".join(sorted(allowed_tools)),
    )
    try:
        response = llm_chat_fn(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": objective[:max_input_chars]},
            ],
            # Temperature 0 keeps the plan reproducible on the same
            # input — critical for the user's preview to match the
            # actual execution.
            options={"temperature": 0.0},
        )
    except Exception as e:
        log.warning("planner.decompose: LLM call failed: %s", e)
        return None
    msg = (response or {}).get("message") or {}
    content = msg.get("content") or ""
    return _parse_plan_json(content, allowed_tools=allowed_tools)


# =====================================================================
# execute
# =====================================================================


# Conventional error markers we treat as failure even when the tool
# "returned without raising" (they're the orchestrator's way to surface
# tool-side errors via the message-passing protocol).
#
# "Web search is unavailable" added 2026-05-28 sessão 15 — when
# web_search is included in a plan but no internet lease is open,
# staging_mixin's _execute_tool_unstaged returns "[SYSTEM] Web search
# is unavailable: ..." (note the `[SYSTEM]` prefix, NOT `[SYSTEM ERROR]`,
# so the LLM tool-call path keeps its friendlier user-facing message).
# The substring match here lets the planner still fail-stop cleanly
# without mutating the lease-denied wording elsewhere.
_ERROR_MARKERS = (
    "[SYSTEM ERROR]",
    "[OFFLINE]",
    "Insufficient data",
    "Web search is unavailable",
)


def _output_looks_failed(output: Any) -> bool:
    """The orchestrator's tool dispatch sometimes returns error strings
    rather than raising. Detect them so the planner can fail-stop on
    a Spotify 404, a missing document, etc., instead of continuing."""
    if not isinstance(output, str):
        return False
    if output in ('Não consegui salvar o código — nada foi gravado.',
                  "I couldn't save the code — nothing was written."):
        return True
    return any(marker in output for marker in _ERROR_MARKERS)


def _step_text(output: Any) -> str:
    """User-facing text of a step output. Honours the runner convention
    `{"text": str, "fields": dict}` (M4-B1) — surfaces and the summariser
    read `text`; a plain value stringifies as before. Mirrors the `text`
    branch of `_resolve_refs` so a structured output never leaks raw."""
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        return output["text"]
    return str(output) if output is not None else ""


def execute(
    plan: Plan,
    tool_executor: Callable[[str, Dict[str, Any]], Any],
    on_step: Optional[Callable[[Step, StepResult], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    pause_before: Optional[Callable[[Step], bool]] = None,
    *,
    step_executor: Optional[Callable[[Step, Dict[str, Any]], Any]] = None,
) -> PlanResult:
    """Run a validated Plan top-to-bottom in topological order.

    `tool_executor(name, args) → output` is the orchestrator's
    `_execute_tool_unstaged` (or equivalent) — must NOT stage destructive
    tools, because plan-level approval already covered them.

    `on_step(step, result)` is called after each step completes (or is
    skipped). Used by the orchestrator to emit audit events tagged with
    the step id and plan trace.

    Returns a PlanResult with one StepResult per step. Fails fast: as
    soon as one step fails, remaining steps are marked skipped (no
    side effects).

    `pause_before(step)` can require a new preview for referenced content
    the original preview could not show. Literal arguments run as usual.
    `step_executor(step, args)` overrides dispatch when step identity is needed.
    """
    try:
        ordered = _topological_order(plan)
    except PlannerError as e:
        return PlanResult(plan=plan, results=[], success=False, error=str(e))

    prior: Dict[str, StepResult] = {}
    fail_error = ""
    cancelled = False
    paused_at = None

    for step in ordered:
        if cancelled or (should_stop is not None and should_stop()):
            cancelled = True
            audit.log("–", "turn", "cancelled", payload={
                "at": "planner.execute", "tool": step.tool})
            res = StepResult(step_id=step.id, success=False,
                             error="cancelled", skipped=True)
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            continue
        if fail_error:
            res = StepResult(step_id=step.id, success=False,
                             error="skipped after prior failure", skipped=True)
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            continue
        if paused_at is not None:
            res = StepResult(step_id=step.id, success=False,
                             error="awaiting confirmation", skipped=True)
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            continue
        # Manually built plans can omit edges too. Never let a missing
        # successful producer turn an approved action's argument empty.
        unresolved_ref = next((
            match.group(0)
            for value in step.args.values() if isinstance(value, str)
            for match in _REF_RE.finditer(value)
            if match.group(1) not in prior or not prior[match.group(1)].success
        ), None)
        if unresolved_ref is not None:
            res = StepResult(step_id=step.id, success=False,
                             error=f"unresolved reference {unresolved_ref}")
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            fail_error = res.error
            continue
        if (pause_before is not None
                and any(_REF_RE.search(v) for v in step.args.values()
                        if isinstance(v, str)) and pause_before(step)):
            paused_at = step.id
            audit.log("–", "planner", "paused",
                      payload={"step_id": step.id, "tool": step.tool})
            res = StepResult(step_id=step.id, success=False,
                             error="awaiting confirmation", skipped=True)
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            continue
        resolved_args = _resolve_refs(step.args, prior)
        try:
            output = (step_executor(step, resolved_args) if step_executor is not None
                      else tool_executor(step.tool, resolved_args))
        except Exception as e:
            res = StepResult(step_id=step.id, success=False,
                             error=f"tool raised: {e}")
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            fail_error = res.error
            continue
        if _output_looks_failed(output):
            res = StepResult(step_id=step.id, success=False,
                             output=output, error=str(output)[:200])
            prior[step.id] = res
            if on_step:
                on_step(step, res)
            fail_error = res.error
            continue
        res = StepResult(step_id=step.id, success=True, output=output)
        prior[step.id] = res
        if on_step:
            on_step(step, res)

    results = [prior[s.id] for s in plan.passos]
    return PlanResult(
        plan=plan,
        results=results,
        success=not fail_error and not cancelled and paused_at is None,
        error=fail_error,
        cancelled=cancelled,
        paused_at=paused_at,
    )


# =====================================================================
# summarise — LLM aggregation of step results
# =====================================================================


_SUMMARISE_SYSTEM = """Você produz uma resposta única em português brasileiro a partir do resultado de um plano multi-step que acabou de executar.

Receba:
  - o objetivo original do usuário
  - a lista de passos executados, cada um com seu output ou erro

Produza uma resposta CURTA E DIRETA (~2-4 frases) que:
  - diga ao usuário o que foi feito
  - se algo falhou, diga claramente qual passo e por quê
  - NÃO inclua tabelas, marcadores, JSON, ou repetições verbose
  - NÃO peça aprovação retroativa nem ofereça "posso ajudar com mais alguma coisa"

Retorne APENAS a resposta. Sem prefixo, sem cabeçalho.
"""

# EN twin of _SUMMARISE_SYSTEM (M1-A8). Two fixed templates instead of a
# dynamic "reply in {lang}" directive — the 8B follows a monolingual
# system prompt far more reliably than an in-prompt language switch
# (same reason the s60 handlers use t(lang, pt, en) pairs).
_SUMMARISE_SYSTEM_EN = """You produce a single English reply from the result of a multi-step plan that just executed.

You receive:
  - the user's original objective
  - the list of executed steps, each with its output or error

Produce a SHORT AND DIRECT reply (~2-4 sentences) that:
  - tells the user what was done
  - if something failed, says clearly which step and why
  - does NOT include tables, bullet points, JSON, or verbose repetition
  - does NOT ask for retroactive approval nor offer "can I help with anything else"

Return ONLY the reply. No prefix, no header.
"""


def _format_results_for_llm(plan: Plan, results: List[StepResult],
                            lang: str = "pt") -> str:
    """Compact text representation of the executed plan for the
    summariser's user message. Bounded so we don't blow context on a
    long Spotify response. Labels follow `lang` so the whole message
    the 8B sees is monolingual (mixed-language input degrades its
    instruction-following)."""
    lines = [
        t(lang, f"Objetivo: {plan.objetivo}", f"Objective: {plan.objetivo}"),
        t(lang, "Passos:", "Steps:"),
    ]
    by_id = {s.id: s for s in plan.passos}
    for r in results:
        step = by_id.get(r.step_id)
        if step is None:
            continue
        descricao = step.descricao or step.tool
        head = f"  - {r.step_id} [{step.tool}] {descricao}"
        if r.skipped:
            lines.append(head + t(lang, ": PULADO (falha anterior)",
                                  ": SKIPPED (earlier failure)"))
        elif r.success:
            out_preview = _step_text(r.output)[:300]
            lines.append(head + f": OK — {out_preview}")
        else:
            lines.append(head + t(lang, ": FALHOU — ", ": FAILED — ") + str(r.error))
    return "\n".join(lines)


def _deterministic_summary(plan: Plan, results: List[StepResult],
                           lang: str = "pt") -> str:
    """Fallback used when the LLM summariser is unavailable or returns
    empty. Just lists what happened — never invents."""
    lines = [t(lang, f"Plano: {plan.objetivo}", f"Plan: {plan.objetivo}")]
    by_id = {s.id: s for s in plan.passos}
    for r in results:
        step = by_id.get(r.step_id)
        if step is None:
            continue
        descricao = step.descricao or step.tool
        if r.skipped:
            lines.append(f"  ✗ {descricao}: " + t(lang, "pulado", "skipped"))
        elif r.success:
            lines.append(f"  ✓ {descricao}")
        else:
            lines.append(f"  ✗ {descricao}: {r.error}")
    return "\n".join(lines)


def summarise(
    plan_result: PlanResult,
    llm_chat_fn: Optional[Callable[..., Any]] = None,
    model: Optional[str] = None,
    lang: str = "pt",
) -> str:
    """Build a user-facing reply for a PlanResult.

    With an LLM available, ask it to write a concise summary in `lang`
    (pt-BR default — M1-A8 threaded the turn language through here).
    Without one — or on any LLM failure — fall back to a deterministic
    bullet list. We NEVER raise; the user always sees something.
    """
    if llm_chat_fn is None or model is None:
        return _deterministic_summary(plan_result.plan, plan_result.results, lang)
    body = _format_results_for_llm(plan_result.plan, plan_result.results, lang)
    try:
        response = llm_chat_fn(
            model=model,
            messages=[
                {"role": "system",
                 "content": t(lang, _SUMMARISE_SYSTEM, _SUMMARISE_SYSTEM_EN)},
                {"role": "user", "content": body[:4000]},
            ],
            options={"temperature": 0.0},
        )
    except Exception as e:
        log.warning("planner.summarise: LLM call failed: %s", e)
        return _deterministic_summary(plan_result.plan, plan_result.results, lang)
    msg = (response or {}).get("message") or {}
    content = (msg.get("content") or "").strip()
    return content or _deterministic_summary(plan_result.plan, plan_result.results, lang)


# =====================================================================
# preview — deterministic plan rendering for user approval
# =====================================================================


def render_preview(plan: Plan, lang: str = "pt") -> str:
    """Produce the deterministic preview a user sees before approving
    plan execution (pt-BR default; `lang` follows the turn — M1-A8).
    Lists every step (including S3 actions that would normally need
    per-step confirmation) so the approval is informed. The approval
    sentence mirrors the tool_risk dry-run previews ("Reply 'yes'…")
    so the two staged surfaces speak the same dialect."""
    if not plan.passos:
        return t(lang,
                 f"Plano vazio: {plan.objetivo}\nNada a executar.",
                 f"Empty plan: {plan.objetivo}\nNothing to execute.")
    lines = [t(lang, f"Plano para: {plan.objetivo}",
               f"Plan for: {plan.objetivo}")]
    by_id = {s.id: s for s in plan.passos}
    for i, step in enumerate(plan.passos, 1):
        deps = (
            t(lang, " (depende de ", " (depends on ")
            + ", ".join(step.depende_de) + ")"
            if step.depende_de else ""
        )
        descricao = step.descricao or step.tool
        lines.append(f"  {i}. [{step.tool}] {descricao}{deps}")
        # Show args inline, whole: the plan-level "sim" runs every one of
        # them, so a cut at 80 chars hid part of an email body it would send
        # (s88, Codex lote 2). Refs stay as {p<id>} so the user understands
        # the dataflow.
        for k, v in step.args.items():
            v_str = str(v) if not isinstance(v, str) else v
            lines.append(f"       {k}: {v_str}")
    lines.append(t(
        lang,
        "Responda 'sim' para executar o plano ou 'não' para cancelar.",
        "Reply 'yes' to run the plan or 'no' to cancel.",
    ))
    return "\n".join(lines)
