"""
Tests for src/core/arbiter_report.py — the shadow-classifier reader.

`analyze` is pure, so the window/distribution/low-confidence logic is tested
with synthetic `arbiter`/`classify` records. `load_records` is exercised
fail-soft. The shadow instrumentation that PRODUCES these records lives in the
orchestrator (`_maybe_arbiter_shadow`) and is covered by its own tests.
"""
from datetime import datetime, timedelta, timezone

from src.core import arbiter_report as ar


NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _rec(route, conf, *, layer="cascade", reason="marker", text="x", days_ago=0):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return {
        "ts": ts, "stage": "arbiter", "event": "classify",
        "payload": {"route": route, "confidence": conf, "layer": layer,
                    "reason": reason, "input": text},
    }


# --- analyze: distribution -------------------------------------------------

def test_counts_total_and_by_route():
    recs = [_rec("REFLEX", 1.0)] * 3 + [_rec("LOOKUP", 0.9)] * 2
    out = ar.analyze(recs, now=NOW)
    assert out.total == 5
    assert out.by_route == {"REFLEX": 3, "LOOKUP": 2}


def test_counts_by_layer_and_reason():
    recs = [
        _rec("SIMPLES", 0.5, layer="fallback", reason="no_marker"),
        _rec("SIMPLES", 0.5, layer="fallback", reason="no_marker"),
        _rec("REFLEX", 1.0, layer="cascade", reason="reflex_hit"),
    ]
    out = ar.analyze(recs, now=NOW)
    assert out.by_layer == {"fallback": 2, "cascade": 1}
    assert out.by_reason == {"no_marker": 2, "reflex_hit": 1}


# --- analyze: low-confidence collection ------------------------------------

def test_low_confidence_collected_and_sorted():
    recs = [
        _rec("SIMPLES", 0.5, text="medium"),
        _rec("LOOKUP", 0.9, text="confident"),   # >= threshold, excluded
        _rec("SIMPLES", 0.0, text="empty"),
    ]
    out = ar.analyze(recs, now=NOW)
    assert [c.confidence for c in out.low_confidence] == [0.0, 0.5]
    assert [c.text for c in out.low_confidence] == ["empty", "medium"]


def test_threshold_boundary_is_exclusive():
    # exactly at the threshold is NOT low-confidence (mirrors arbiter wake rule)
    out = ar.analyze([_rec("MEMORY", 0.8)], now=NOW)
    assert out.low_confidence == []


def test_custom_threshold():
    out = ar.analyze([_rec("LOOKUP", 0.9)], low_confidence_threshold=0.95, now=NOW)
    assert len(out.low_confidence) == 1


# --- analyze: window + filtering -------------------------------------------

def test_window_excludes_old_records():
    recs = [_rec("REFLEX", 1.0, days_ago=45)]
    assert ar.analyze(recs, window_days=30, now=NOW).total == 0
    assert ar.analyze(recs, window_days=60, now=NOW).total == 1


def test_ignores_non_arbiter_events():
    recs = [
        {"ts": NOW.isoformat(), "stage": "gate", "event": "approved", "payload": {}},
        {"ts": NOW.isoformat(), "stage": "arbiter", "event": "miss", "payload": {}},
    ]
    assert ar.analyze(recs, now=NOW).total == 0


def test_malformed_confidence_treated_as_zero():
    recs = [_rec("SIMPLES", "not-a-number")]
    out = ar.analyze(recs, now=NOW)
    assert out.total == 1
    assert out.low_confidence[0].confidence == 0.0


# --- render ----------------------------------------------------------------

def test_render_empty_hints_at_shadow_flag():
    out = ar.render(ar.analyze([], now=NOW), window_days=30)
    assert "CORTEX_ARBITER_SHADOW" in out


def test_render_lists_routes_no_tables():
    recs = [_rec("REFLEX", 1.0)] * 2 + [_rec("SIMPLES", 0.5, text="o que é x")]
    text = ar.render(ar.analyze(recs, now=NOW), window_days=30)
    assert "REFLEX" in text and "SIMPLES" in text
    assert "o que é x" in text          # the unsure case is surfaced
    assert "|" not in text              # bullets, never markdown tables
    assert "arbiter.py" in text         # points the human at where to act


def test_render_truncates_low_confidence_list():
    recs = [_rec("SIMPLES", 0.5, text=f"q{i}") for i in range(40)]
    text = ar.render(ar.analyze(recs, now=NOW), window_days=30, max_low=10)
    assert "+30 mais" in text


# --- load_records fail-soft ------------------------------------------------

def test_load_records_missing_file(tmp_path):
    assert ar.load_records(str(tmp_path / "nope.jsonl")) == []


def test_load_records_skips_malformed(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text(
        '{"stage":"arbiter","event":"classify","payload":{"route":"REFLEX"}}\n'
        "garbage line\n"
        '{"stage":"arbiter","event":"classify","payload":{"route":"LOOKUP"}}\n'
    )
    assert len(ar.load_records(str(p))) == 2


def test_main_runs_clean(tmp_path, capsys):
    p = tmp_path / "audit.jsonl"
    p.write_text("\n".join(
        '{"ts":"%s","stage":"arbiter","event":"classify",'
        '"payload":{"route":"REFLEX","confidence":1.0,"layer":"cascade","reason":"reflex_hit","input":"oi"}}'
        % NOW.isoformat() for _ in range(5)
    ))
    rc = ar.main(["--path", str(p), "--days", "3650"])
    assert rc == 0
    assert "REFLEX" in capsys.readouterr().out
