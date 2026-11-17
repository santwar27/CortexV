"""
gate_policy — narrow, fail-closed pre-decisions for background S3 actions.

Pure function over (type, args, origin) + injectable rule lists, so it tests
without the queue or any I/O. The load-bearing guarantees: REVIEW is the
default (fail-closed), AUTO_REJECT ships for system-control tools, and
AUTO_APPROVE ships with ZERO rules (nothing irreversible auto-runs).
"""
from src.core import gate_policy
from src.core.gate_policy import (
    AUTO_APPROVE,
    AUTO_REJECT,
    REVIEW,
    PolicyDecision,
    evaluate,
)


class TestDefaultFailClosed:
    def test_unmatched_action_is_review(self):
        for tool in ("send_email", "create_event", "create_reminder"):
            d = evaluate(tool, {}, "background")
            assert d.action == REVIEW
            assert d.reason  # always carries a why

    def test_unknown_tool_is_review(self):
        assert evaluate("some_future_tool", {}, "scheduled:x").action == REVIEW


class TestShippedAutoReject:
    def test_mac_sleep_rejected(self):
        d = evaluate("mac_sleep", {}, "background")
        assert d.action == AUTO_REJECT
        assert "mac_sleep" in d.reason

    def test_mac_quit_app_rejected(self):
        assert evaluate("mac_quit_app", {"app_name": "Safari"}, "background").action == AUTO_REJECT


class TestNoAutoApproveByDefault:
    def test_nothing_auto_approves_out_of_the_box(self):
        # The whole point: with shipped rules, no action is ever AUTO_APPROVE.
        for tool in ("send_email", "create_event", "create_reminder",
                     "mac_sleep", "mac_quit_app"):
            assert evaluate(tool, {}, "background").action != AUTO_APPROVE

    def test_default_approve_rule_set_is_empty(self):
        assert gate_policy._DEFAULT_APPROVE_RULES == []


class TestInjectableRules:
    def test_custom_approve_rule_can_fire(self):
        def approve_short_reminder(t, args, origin):
            if t == "create_reminder" and len(args.get("title", "")) < 60:
                return PolicyDecision(AUTO_APPROVE, "test rule")
            return None

        d = evaluate("create_reminder", {"title": "Buy milk"}, "background",
                     approve_rules=[approve_short_reminder])
        assert d.action == AUTO_APPROVE
        assert d.reason == "test rule"

    def test_reject_precedes_approve_on_same_action(self):
        # Even if an approve rule would fire, an auto-reject wins (safe
        # direction always pre-empts).
        always_reject = lambda t, a, o: PolicyDecision(AUTO_REJECT, "nope")
        always_approve = lambda t, a, o: PolicyDecision(AUTO_APPROVE, "yep")
        d = evaluate("send_email", {}, "background",
                     reject_rules=[always_reject], approve_rules=[always_approve])
        assert d.action == AUTO_REJECT
        assert d.reason == "nope"

    def test_empty_rule_sets_fall_through_to_review(self):
        d = evaluate("mac_sleep", {}, "background", reject_rules=[], approve_rules=[])
        assert d.action == REVIEW

    def test_first_matching_rule_wins(self):
        r1 = lambda t, a, o: None
        r2 = lambda t, a, o: PolicyDecision(AUTO_REJECT, "r2")
        r3 = lambda t, a, o: PolicyDecision(AUTO_REJECT, "r3")
        d = evaluate("x", {}, "bg", reject_rules=[r1, r2, r3])
        assert d.reason == "r2"


class TestArgsRobustness:
    def test_none_args_ok(self):
        assert evaluate("send_email", None, "background").action == REVIEW
