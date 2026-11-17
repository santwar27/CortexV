"""
Gate policy — narrow, fail-closed rules for background S3 actions.

When Cortex acts in the background and the LLM picks a destructive (S3)
tool, the gate batch queue (`gate_queue.py`) holds it for human review.
That's safe but everything waits for you. This layer lets a small set of
**narrow, conjunctive** rules pre-decide an action automatically — the
design-reference ch. 13 policy idea ("delete IF path under /tmp AND <7 days AND
<10MB", never "auto-approve everything from tool X").

The cardinal rule is FAIL-CLOSED: if no rule matches, the decision is
`REVIEW` (a human decides). Three possible decisions:

  - ``AUTO_REJECT`` — drop it without running. Safe to automate: the worst
    case is a legitimate action gets dropped, which is the safe direction.
  - ``REVIEW`` — the default; queue it for the human (nothing runs).
  - ``AUTO_APPROVE`` — run it unattended. RISKY (it executes something
    irreversible with no human in the loop), so per the reference we **ship with
    ZERO auto-approve rules**; the mechanism exists for you to add narrow,
    well-understood rules later, opt-in.

Rules are tried in order, first match wins; AUTO_REJECT rules are checked
before AUTO_APPROVE so the safe direction always pre-empts. The module is
pure (no I/O, no queue) so it unit-tests trivially; the caller
(`staging_mixin._execute_tool`) owns the queue side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Decision tokens — plain strings so they serialise into audit/queue JSON
# without ceremony.
REVIEW = "review"
AUTO_REJECT = "auto_reject"
AUTO_APPROVE = "auto_approve"


@dataclass(frozen=True)
class PolicyDecision:
    """What the policy decided for one action, plus a human-readable why."""
    action: str  # REVIEW | AUTO_REJECT | AUTO_APPROVE
    reason: str


# A rule inspects (action_type, args, origin) and returns a PolicyDecision
# to claim the action, or None to pass it to the next rule.
Rule = Callable[[str, Dict, str], Optional[PolicyDecision]]


# --- shipped AUTO_REJECT rules (the only automatic decisions we ship) ----

# System-control tools should never run unattended from a background
# producer: a nightly task putting the Mac to sleep or quitting an app
# would clobber the user's live session / unsaved work. If a background
# LLM ever picks these, reject — don't even leave them pending (a human
# wouldn't approve "sleep my Mac" that a 3am job dreamt up anyway).
_SYSTEM_CONTROL_TOOLS = frozenset({"mac_sleep", "mac_quit_app", "ps5_standby"})


def _reject_background_system_control(
    action_type: str, args: Dict, origin: str
) -> Optional[PolicyDecision]:
    if action_type in _SYSTEM_CONTROL_TOOLS:
        return PolicyDecision(
            AUTO_REJECT,
            f"'{action_type}' em background é recusado por política "
            "(controle de sistema não deve rodar sem a sessão ativa).",
        )
    return None


# Default rule sets. AUTO_REJECT ships with the system-control guard;
# AUTO_APPROVE ships EMPTY on purpose (the reference: start without auto-approval).
_DEFAULT_REJECT_RULES: List[Rule] = [_reject_background_system_control]
_DEFAULT_APPROVE_RULES: List[Rule] = []


def evaluate(
    action_type: str,
    args: Optional[Dict] = None,
    origin: str = "background",
    *,
    reject_rules: Optional[List[Rule]] = None,
    approve_rules: Optional[List[Rule]] = None,
) -> PolicyDecision:
    """Decide what to do with a background S3 action. Fail-closed: with no
    matching rule the result is REVIEW (human decides).

    `reject_rules`/`approve_rules` default to the module's shipped sets but
    are injectable for tests. AUTO_REJECT is evaluated first so the safe
    direction always wins over any auto-approve rule on the same action."""
    args = args or {}
    rej = _DEFAULT_REJECT_RULES if reject_rules is None else reject_rules
    app = _DEFAULT_APPROVE_RULES if approve_rules is None else approve_rules

    for rule in rej:
        decision = rule(action_type, args, origin)
        if decision is not None:
            return decision
    for rule in app:
        decision = rule(action_type, args, origin)
        if decision is not None:
            return decision
    return PolicyDecision(
        REVIEW,
        "sem regra correspondente — revisão humana (fail-closed).",
    )
