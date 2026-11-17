"""Black-box tests for scripts/audit_queries/*.sh.

The queries are bash + jq, so we test them as black boxes: write a
synthetic audit JSONL to a tmp file, point each script at it via
CORTEX_AUDIT_LOG, run it, and assert on stdout. Covers the happy path,
the half-written-line tolerance (fromjson?), the missing-file branch,
and the empty-day branch.

These never touch the real ~/.cortex/audit.jsonl.
"""
import subprocess
from pathlib import Path

import pytest


QUERIES_DIR = Path(__file__).resolve().parent.parent / "scripts" / "audit_queries"

# One valid day of events + a cross-day record + a deliberately broken
# trailing line that fromjson? must skip without aborting the query.
SAMPLE_LOG = (
    '{"ts":"2026-05-27T09:00:00+00:00","trace_id":"aaaa1111","stage":"orchestrator","event":"request","payload":{"input":"oi cortex"}}\n'
    '{"ts":"2026-05-27T09:00:00+00:00","trace_id":"aaaa1111","stage":"reflex","event":"hit","payload":{"input":"oi cortex"}}\n'
    '{"ts":"2026-05-27T09:00:00+00:00","trace_id":"aaaa1111","stage":"orchestrator","event":"response","latency_ms":12.5}\n'
    '{"ts":"2026-05-27T10:30:00+00:00","trace_id":"bbbb2222","stage":"reflex","event":"miss","payload":{"input":"toca matue"}}\n'
    '{"ts":"2026-05-27T10:30:01+00:00","trace_id":"bbbb2222","stage":"tool","event":"dispatch","payload":{"name":"write_document","risk":"S2","arg_keys":["content"]}}\n'
    '{"ts":"2026-05-27T10:30:05+00:00","trace_id":"bbbb2222","stage":"orchestrator","event":"response","latency_ms":4200.0}\n'
    '{"ts":"2026-05-27T11:00:00+00:00","trace_id":"cccc3333","stage":"tool","event":"dispatch","payload":{"name":"mac_quit_app","risk":"S3","arg_keys":["app_name"]}}\n'
    '{"ts":"2026-05-27T11:00:02+00:00","trace_id":"cccc3333","stage":"tool","event":"dispatch_in_plan","payload":{"name":"write_document","risk":"S2","arg_keys":["content"]}}\n'
    '{"ts":"2026-05-28T08:00:00+00:00","trace_id":"dddd4444","stage":"reflex","event":"hit","payload":{"input":"que horas"}}\n'
    'this-is-a-half-written-broken-line{"ts":\n'
)


@pytest.fixture
def audit_log(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text(SAMPLE_LOG, encoding="utf-8")
    return p


def _run(script, audit_path, *args):
    result = subprocess.run(
        [str(QUERIES_DIR / script), *args],
        capture_output=True,
        text=True,
        env={"CORTEX_AUDIT_LOG": str(audit_path), "PATH": __import__("os").environ["PATH"]},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestDay:
    def test_counts_only_target_day_events(self, audit_log):
        out = _run("day.sh", audit_log, "2026-05-27")
        # 8 valid records on the 27th; the 28th record and the broken
        # line are both excluded.
        assert "8 eventos" in out
        assert "mac_quit_app" in out
        assert "que horas" not in out  # belongs to the 28th

    def test_broken_line_does_not_abort(self, audit_log):
        # If fromjson? weren't tolerant, set -e + jq error would make
        # this non-zero and _run would assert. Reaching here = tolerated.
        out = _run("day.sh", audit_log, "2026-05-27")
        assert "orchestrator/request" in out

    def test_empty_day(self, audit_log):
        out = _run("day.sh", audit_log, "2020-01-01")
        assert "0 eventos" in out


class TestTools:
    def test_tally_groups_and_risk(self, audit_log):
        out = _run("tools.sh", audit_log, "2026-05-27")
        assert "2x  write_document  [S2]  (direct 1, in-plan 1)" in out
        assert "1x  mac_quit_app  [S3]  (direct 1, in-plan 0)" in out

    def test_empty_day(self, audit_log):
        out = _run("tools.sh", audit_log, "2020-01-01")
        assert "nenhum tool dispatch" in out


class TestReflex:
    def test_hit_miss_rate(self, audit_log):
        out = _run("reflex.sh", audit_log, "2026-05-27")
        assert "hits: 1" in out
        assert "miss: 1" in out
        assert "hit rate: 50%" in out

    def test_no_division_by_zero_on_empty_day(self, audit_log):
        out = _run("reflex.sh", audit_log, "2020-01-01")
        assert "total: 0" in out
        assert "hit rate" not in out  # guarded when total == 0


class TestLatency:
    def test_per_stage_stats(self, audit_log):
        out = _run("latency.sh", audit_log, "2026-05-27")
        # Two orchestrator/response records: 12.5 and 4200 → avg 2106.
        assert "orchestrator/response" in out
        assert "n=2" in out
        assert "min=12ms" in out
        assert "max=4200ms" in out
        assert "avg=2106ms" in out

    def test_empty_day(self, audit_log):
        out = _run("latency.sh", audit_log, "2020-01-01")
        assert "nenhum evento com latência" in out


class TestMissingFile:
    @pytest.mark.parametrize("script", ["day.sh", "tools.sh", "reflex.sh", "latency.sh"])
    def test_missing_audit_log_is_graceful(self, tmp_path, script):
        missing = tmp_path / "does_not_exist.jsonl"
        out = _run(script, missing, "2026-05-27")
        assert "sem audit log" in out
