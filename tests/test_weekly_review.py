"""
weekly_review — pure aggregation + render over synthetic audit records.

The interesting logic (folding audit events into counters, rendering the
digest) is pure, so it tests without touching disk or the Palace. load_records
I/O is covered with a tmp file. Mirrors the event names the live pipeline
emits (see src/core/audit.py usage across the codebase).
"""
from datetime import datetime, timedelta, timezone

from src import weekly_review as wr


def _rec(stage, event, ts, **payload):
    r = {"stage": stage, "event": event, "ts": ts.isoformat()}
    if payload:
        r["payload"] = payload
    return r


NOW = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)


def _sample_records():
    return [
        _rec("orchestrator", "request", NOW),
        _rec("orchestrator", "request", NOW),
        _rec("orchestrator", "response", NOW, mode="OFFLINE"),
        _rec("orchestrator", "response", NOW, mode="AGENT_ACTION_DONE"),
        _rec("orchestrator", "response", NOW, mode="AGENT_ACTION_ERROR"),
        _rec("reflex", "hit", NOW),
        _rec("reflex", "hit", NOW),
        _rec("tool", "dispatch", NOW, name="send_email"),
        _rec("tool", "dispatch", NOW, name="web_search"),
        _rec("tool", "dispatch_in_plan", NOW, name="send_email"),
        _rec("gate", "queued", NOW),
        _rec("gate", "queued", NOW),
        _rec("gate", "approved", NOW),
        _rec("gate", "failed", NOW),
        _rec("planner", "execute_start", NOW),
        _rec("engine", "hallucination_blocked", NOW),
        _rec("voice", "done", NOW),
    ]


class TestAggregate:
    def test_counts(self):
        s = wr.aggregate(_sample_records())
        assert s.total_records == 17
        assert s.requests == 2
        assert s.reflex_hits == 2
        assert s.plans_run == 1
        assert s.hallucinations_blocked == 1
        assert s.voice_turns == 1

    def test_modes_and_action_errors(self):
        s = wr.aggregate(_sample_records())
        assert s.modes["OFFLINE"] == 1
        assert s.modes["AGENT_ACTION_DONE"] == 1
        assert s.modes["AGENT_ACTION_ERROR"] == 1
        # any *ERROR mode counts as an action error
        assert s.action_errors == 1

    def test_tools_merge_plan_and_direct(self):
        s = wr.aggregate(_sample_records())
        assert s.tools["send_email"] == 2  # 1 dispatch + 1 dispatch_in_plan
        assert s.tools["web_search"] == 1

    def test_gate_events(self):
        s = wr.aggregate(_sample_records())
        assert s.gate["queued"] == 2
        assert s.gate["approved"] == 1
        assert s.gate["failed"] == 1
        assert s.gate["rejected"] == 0

    def test_empty(self):
        s = wr.aggregate([])
        assert s.total_records == 0
        assert s.requests == 0
        assert s.modes == {}

    def test_ignores_unknown_and_malformed_payload(self):
        recs = [
            _rec("orchestrator", "response", NOW),         # no mode
            {"stage": "tool", "event": "dispatch", "ts": NOW.isoformat(),
             "payload": "not-a-dict"},                     # bad payload type
            _rec("mystery", "stuff", NOW),
        ]
        s = wr.aggregate(recs)
        assert s.total_records == 3
        assert s.modes == {}
        assert s.tools["?"] == 1  # payload not a dict → name falls back to '?'


class TestRender:
    def test_render_has_sections_and_no_tables(self):
        s = wr.aggregate(_sample_records())
        out = wr.render_markdown(
            s, since=NOW - timedelta(days=7), until=NOW,
            week_label="2026-W23", pending_studies=3,
        )
        assert "# Cortex — revisão semanal 2026-W23" in out
        assert "## Atividade" in out
        assert "## Ferramentas" in out
        assert "## Gate de ações" in out
        assert "## Saúde" in out
        assert "Estudos pendentes de revisão: 3" in out
        assert "send_email: 2" in out
        # bullets, never a markdown table
        assert "|" not in out

    def test_render_empty_is_graceful(self):
        s = wr.aggregate([])
        out = wr.render_markdown(
            s, since=NOW - timedelta(days=7), until=NOW, week_label="2026-W23",
        )
        assert "nenhuma ferramenta despachada" in out
        assert "sem atividade no gate" in out
        assert "Pedidos atendidos: 0" in out


class TestLoadRecords:
    def test_window_filter_and_malformed(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        old = NOW - timedelta(days=30)
        lines = [
            _rec("orchestrator", "request", NOW),
            _rec("orchestrator", "request", old),   # outside window
            {"garbage": True},                        # no ts → skipped
        ]
        with open(p, "w", encoding="utf-8") as f:
            for r in lines:
                f.write(__import__("json").dumps(r) + "\n")
            f.write("this is not json\n")              # malformed → skipped
        recs = wr.load_records(str(p), since=NOW - timedelta(days=7))
        assert len(recs) == 1
        assert recs[0]["stage"] == "orchestrator"

    def test_missing_file_is_empty(self, tmp_path):
        assert wr.load_records(str(tmp_path / "nope.jsonl"), since=NOW) == []


def test_main_reads_the_project_env_before_the_audit_path(monkeypatch, tmp_path, capsys):
    """Under launchd the job had PATH only and never read .env, so an audit
    moved there by CORTEX_AUDIT_LOG was not the one reviewed (s88, Codex
    lote 10). The .env is read first, and its value decides the default."""
    import os
    import dotenv
    from src import weekly_review as wr
    audit = tmp_path / "moved_audit.jsonl"
    audit.write_text("")
    seen = []

    def fake_load(path, override=False):
        seen.append(path)
        os.environ.setdefault("CORTEX_AUDIT_LOG", str(audit))
        return True
    monkeypatch.setattr(dotenv, "load_dotenv", fake_load)
    monkeypatch.delenv("CORTEX_AUDIT_LOG", raising=False)
    read = []
    monkeypatch.setattr(wr, "load_records", lambda p, since: read.append(p) or [])
    assert wr.main(["--stdout"]) == 0
    assert seen and os.path.basename(seen[0]) == ".env"
    assert read == [str(audit)]
