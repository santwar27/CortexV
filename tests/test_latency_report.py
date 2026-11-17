"""
Tests for src/core/latency_report.py — per-route turn-latency reader.

`analyze` is pure, so percentiles/grouping/window are tested with synthetic
`orchestrator`/`response` records. `load_records` is exercised fail-soft. The
instrumentation that PRODUCES these records lives in orchestrator.process.
"""
from datetime import datetime, timedelta, timezone

from src.core import latency_report as lr


NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _rec(mode, lat, *, days_ago=0):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return {
        "ts": ts, "stage": "orchestrator", "event": "response",
        "latency_ms": lat, "payload": {"mode": mode},
    }


# --- percentile helper -----------------------------------------------------

def test_percentile_basics():
    vals = [float(i) for i in range(1, 101)]  # 1..100 sorted
    assert lr._percentile(vals, 50) == 50.0
    assert lr._percentile(vals, 95) == 95.0
    assert lr._percentile(vals, 100) == 100.0


def test_percentile_single_value():
    assert lr._percentile([42.0], 95) == 42.0


def test_percentile_empty():
    assert lr._percentile([], 95) == 0.0


# --- analyze: grouping + stats ---------------------------------------------

def test_groups_by_mode_and_computes_stats():
    recs = [_rec("OFFLINE", 100), _rec("OFFLINE", 200), _rec("OFFLINE", 300)]
    out = lr.analyze(recs, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert s.bucket == "OFFLINE"
    assert s.count == 3
    assert s.p50 == 200.0
    assert s.max == 300.0
    assert s.mean == 200.0


def test_sorted_by_p95_descending():
    recs = [_rec("FAST", 10)] * 5 + [_rec("SLOW", 5000)] * 5
    out = lr.analyze(recs, now=NOW)
    assert [s.bucket for s in out] == ["SLOW", "FAST"]


def test_window_excludes_old():
    recs = [_rec("OFFLINE", 100, days_ago=45)]
    assert lr.analyze(recs, window_days=30, now=NOW) == []
    assert len(lr.analyze(recs, window_days=60, now=NOW)) == 1


def test_ignores_records_without_latency():
    recs = [
        {"ts": NOW.isoformat(), "stage": "orchestrator", "event": "response",
         "payload": {"mode": "OFFLINE"}},  # no latency_ms
        _rec("OFFLINE", 100),
    ]
    out = lr.analyze(recs, now=NOW)
    assert out[0].count == 1


def test_ignores_other_stages():
    recs = [
        {"ts": NOW.isoformat(), "stage": "voice", "event": "replied",
         "latency_ms": 999, "payload": {"mode": "X"}},
        _rec("OFFLINE", 100),
    ]
    out = lr.analyze(recs, now=NOW)
    assert len(out) == 1 and out[0].bucket == "OFFLINE"


def test_missing_mode_bucketed_as_unknown():
    recs = [{"ts": NOW.isoformat(), "stage": "orchestrator", "event": "response",
             "latency_ms": 50, "payload": {}}]
    assert lr.analyze(recs, now=NOW)[0].bucket == "?"


def test_custom_stage_event_bucket():
    recs = [{"ts": NOW.isoformat(), "stage": "voice", "event": "replied",
             "latency_ms": 1200, "payload": {"phase": "tts"}}]
    out = lr.analyze(recs, stage="voice", event="replied", bucket_key="phase", now=NOW)
    assert out[0].bucket == "tts" and out[0].count == 1


# --- render ----------------------------------------------------------------

def test_render_empty():
    assert "Nada registrado" in lr.render([], window_days=30)


def test_render_lists_buckets_no_tables():
    recs = [_rec("OFFLINE", 1500)] * 3 + [_rec("SPOTIFY_PLAY", 200)] * 2
    text = lr.render(lr.analyze(recs, now=NOW), window_days=30)
    assert "OFFLINE" in text and "SPOTIFY_PLAY" in text
    assert "1.5s" in text          # ms→s formatting for slow path
    assert "|" not in text         # bullets, never markdown tables
    assert "Total: 5 turnos" in text


# --- load_records fail-soft ------------------------------------------------

def test_load_records_missing_file(tmp_path):
    assert lr.load_records(str(tmp_path / "nope.jsonl")) == []


def test_load_records_skips_malformed(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text(
        '{"stage":"orchestrator","event":"response","latency_ms":1}\n'
        "not json\n"
        '{"stage":"orchestrator","event":"response","latency_ms":2}\n'
    )
    assert len(lr.load_records(str(p))) == 2


def test_main_runs_clean(tmp_path, capsys):
    p = tmp_path / "audit.jsonl"
    p.write_text("\n".join(
        '{"ts":"%s","stage":"orchestrator","event":"response",'
        '"latency_ms":%d,"payload":{"mode":"OFFLINE"}}' % (NOW.isoformat(), 100 + i)
        for i in range(5)
    ))
    rc = lr.main(["--path", str(p), "--days", "3650"])
    assert rc == 0
    assert "OFFLINE" in capsys.readouterr().out



def test_the_report_reads_the_rotated_archive(tmp_path, capsys):
    """load_records opened only the active file: after a rotation the 30-day
    report counted 2 turns over 102 (s88, Codex lote 13)."""
    import json
    from datetime import datetime, timedelta, timezone
    from src.core import latency_report as lr
    now = datetime.now(timezone.utc)

    def rec(i, ts):
        return json.dumps({"ts": ts.isoformat(), "stage": "orchestrator",
                           "event": "response", "latency_ms": 100 + i,
                           "payload": {"mode": "OFFLINE"}})
    (tmp_path / "archive").mkdir()
    rotated = (now - timedelta(days=2)).strftime("%Y%m%d-%H%M%S")
    (tmp_path / "archive" / f"audit-{rotated}-ab12cd.jsonl").write_text(
        "\n".join(rec(i, now - timedelta(days=3, minutes=i)) for i in range(100)) + "\n")
    (tmp_path / "audit.jsonl").write_text(
        "\n".join(rec(i, now - timedelta(hours=1, minutes=i)) for i in range(2)) + "\n")
    lr.main(["--path", str(tmp_path / "audit.jsonl")])
    assert "Total: 102 turnos" in capsys.readouterr().out



def test_a_line_cut_inside_a_character_is_skipped(tmp_path, capsys):
    """A write cut inside "ã" ended the report (and every audit reader) in
    UnicodeDecodeError (s88, Codex lote 13 — invariant 7)."""
    import json
    from datetime import datetime, timezone
    from src.core import latency_report as lr
    from src.core.audit import read_records
    now = datetime.now(timezone.utc).isoformat()
    ok = json.dumps({"ts": now, "stage": "orchestrator", "event": "response",
                     "latency_ms": 120, "payload": {"mode": "OFFLINE", "input": "previsão"}},
                    ensure_ascii=False)
    p = tmp_path / "audit.jsonl"
    p.write_bytes(((ok + "\n") * 3 + '{"ts": "' + now + '", "payload": {"input": "previs')
                  .encode("utf-8") + "ã".encode("utf-8")[:1])
    assert len(read_records(str(p))) == 3
    lr.main(["--path", str(p)])
    assert "Total: 3 turnos" in capsys.readouterr().out
