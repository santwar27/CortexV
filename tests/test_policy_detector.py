"""
Tests for src/core/policy_detector.py — the design-reference ch. 13 policy detector.

`analyze` is pure, so the window/threshold/approval-rate logic is tested
with synthetic audit records. `load_records` is exercised fail-soft. The
inline-gate instrumentation that PRODUCES these records is tested in
test_gate_inline_audit.py.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.core import policy_detector as pd


NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)


def _rec(atype, event, *, days_ago=0):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return {"ts": ts, "stage": "gate", "event": event, "payload": {"type": atype}}


def _approvals(atype, n, *, days_ago=0):
    return [_rec(atype, "approved", days_ago=days_ago) for _ in range(n)]


# --- analyze: core thresholds ---------------------------------------------

def test_suggests_when_streak_clears_threshold():
    recs = _approvals("create_reminder", 12)
    out = pd.analyze(recs, min_occurrences=10, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert s.action_type == "create_reminder"
    assert s.approvals == 12 and s.rejections == 0
    assert s.approval_rate == 1.0
    assert not s.high_harm


def test_below_min_occurrences_no_suggestion():
    out = pd.analyze(_approvals("create_reminder", 9), min_occurrences=10, now=NOW)
    assert out == []


def test_single_rejection_disqualifies():
    recs = _approvals("create_reminder", 20) + [_rec("create_reminder", "rejected")]
    out = pd.analyze(recs, min_occurrences=10, now=NOW)
    assert out == []  # 20/21 ≠ 100%


def test_window_excludes_old_decisions():
    recs = _approvals("create_reminder", 12, days_ago=45)  # outside 30d
    assert pd.analyze(recs, min_occurrences=10, window_days=30, now=NOW) == []
    # …and counts them with a wider window
    assert len(pd.analyze(recs, min_occurrences=10, window_days=60, now=NOW)) == 1


# --- high-harm handling ----------------------------------------------------

def test_high_harm_flagged_with_caveat():
    out = pd.analyze(_approvals("send_email", 15), min_occurrences=10, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert s.high_harm is True
    assert "NÃO auto-aprove" in s.recommendation


def test_low_harm_not_flagged():
    out = pd.analyze(_approvals("create_event", 11), min_occurrences=10, now=NOW)
    assert out[0].high_harm is False


# --- grouping + ordering ---------------------------------------------------

def test_groups_by_type_and_sorts_by_approvals():
    recs = _approvals("create_event", 11) + _approvals("create_reminder", 25)
    out = pd.analyze(recs, min_occurrences=10, now=NOW)
    assert [s.action_type for s in out] == ["create_reminder", "create_event"]


def test_ignores_non_gate_events():
    recs = [
        {"ts": NOW.isoformat(), "stage": "tool", "event": "approved",
         "payload": {"type": "create_reminder"}},  # wrong stage
    ] * 15
    assert pd.analyze(recs, min_occurrences=10, now=NOW) == []


def test_missing_type_ignored():
    recs = [{"ts": NOW.isoformat(), "stage": "gate", "event": "approved",
             "payload": {}}] * 15
    assert pd.analyze(recs, min_occurrences=10, now=NOW) == []


# --- render ---------------------------------------------------------------

def test_render_empty():
    out = pd.render([], window_days=30, min_occurrences=10)
    assert "Nada a sugerir" in out


def test_render_lists_suggestion_no_tables():
    sugg = pd.analyze(_approvals("create_reminder", 12), min_occurrences=10, now=NOW)
    text = pd.render(sugg, window_days=30, min_occurrences=10)
    assert "create_reminder" in text
    assert "|" not in text  # bullets, never markdown tables
    assert "gate_policy.py" in text  # points the human at where to act


# --- load_records fail-soft -----------------------------------------------

def test_load_records_missing_file(tmp_path):
    assert pd.load_records(str(tmp_path / "nope.jsonl")) == []


def test_load_records_skips_malformed(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text(
        '{"stage":"gate","event":"approved","payload":{"type":"x"}}\n'
        "not json at all\n"
        '{"stage":"gate","event":"rejected","payload":{"type":"x"}}\n'
    )
    recs = pd.load_records(str(p))
    assert len(recs) == 2


def test_main_runs_clean(tmp_path, capsys):
    p = tmp_path / "audit.jsonl"
    p.write_text("\n".join(
        '{"ts":"%s","stage":"gate","event":"approved","payload":{"type":"create_reminder"}}'
        % NOW.isoformat() for _ in range(12)
    ))
    rc = pd.main(["--path", str(p), "--min", "10", "--days", "3650"])
    assert rc == 0
    assert "create_reminder" in capsys.readouterr().out
