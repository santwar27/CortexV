"""
SystemLease — the P-2 decision (Pedro, 2026-06-10) as code.

Background jobs get a SCOPED, BOUNDED, AUDITED, KILLABLE network window
instead of the pre-review state (Studier touching the web with no lease,
no cap, no per-query audit — M6-A2). These tests pin each property and
the Studier wiring.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.core.system_lease import (
    GovernedWebSearcher,
    SystemLease,
    SystemLeaseDenied,
)


def _audit_events(stage="system_lease"):
    from src.core.audit import audit
    events = []
    try:
        with open(audit.path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("stage") == stage:
                    events.append(rec)
    except FileNotFoundError:
        pass
    return events


class TestLeaseLifecycle:
    def test_acquire_is_audited(self):
        lease = SystemLease.acquire("web_search", "studier-nightly")
        assert lease.is_active()
        events = _audit_events()
        assert events and events[0]["event"] == "acquire"
        assert events[0]["payload"]["origin"] == "studier-nightly"

    def test_release_audits_usage_summary(self):
        lease = SystemLease.acquire("web_search", "studier-nightly")
        lease.authorize("web_search", detail="q1")
        lease.release()
        rel = [e for e in _audit_events() if e["event"] == "release"]
        assert len(rel) == 1
        assert rel[0]["payload"]["calls_made"] == 1
        assert not lease.is_active()

    def test_release_is_idempotent(self):
        lease = SystemLease.acquire("web_search", "x")
        lease.release()
        lease.release()
        assert len([e for e in _audit_events() if e["event"] == "release"]) == 1

    def test_context_manager_releases(self):
        with SystemLease.acquire("web_search", "x") as lease:
            assert lease.is_active()
        assert not lease.is_active()


class TestLeaseEnforcement:
    def test_every_call_is_audited_with_detail(self):
        lease = SystemLease.acquire("web_search", "studier-nightly")
        lease.authorize("web_search", detail="jogos da seleção")
        calls = [e for e in _audit_events() if e["event"] == "call"]
        assert len(calls) == 1
        assert calls[0]["payload"]["detail"] == "jogos da seleção"

    def test_max_calls_cap_denies_and_audits(self):
        lease = SystemLease.acquire("web_search", "x", max_calls=2)
        lease.authorize("web_search")
        lease.authorize("web_search")
        with pytest.raises(SystemLeaseDenied, match="cap"):
            lease.authorize("web_search")
        denies = [e for e in _audit_events() if e["event"] == "deny"]
        assert len(denies) == 1 and "cap" in denies[0]["payload"]["reason"]

    def test_expired_window_denies(self):
        lease = SystemLease.acquire("web_search", "x", minutes=90)
        lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert not lease.is_active()
        with pytest.raises(SystemLeaseDenied, match="expirada"):
            lease.authorize("web_search")

    def test_scope_mismatch_denies(self):
        lease = SystemLease.acquire("web_search", "x")
        with pytest.raises(SystemLeaseDenied, match="scope"):
            lease.authorize("send_email")
        # The mismatch consumed no call budget.
        assert lease.calls_made == 0

    def test_released_lease_denies(self):
        lease = SystemLease.acquire("web_search", "x")
        lease.release()
        with pytest.raises(SystemLeaseDenied):
            lease.authorize("web_search")

    def test_kill_switch_acquires_inactive(self, monkeypatch):
        monkeypatch.setenv("CORTEX_SYSTEM_LEASE", "0")
        lease = SystemLease.acquire("web_search", "x")
        assert not lease.is_active()
        with pytest.raises(SystemLeaseDenied, match="inativa"):
            lease.authorize("web_search")
        # Acquire is still audited (visible that the job ran ungranted).
        acq = [e for e in _audit_events() if e["event"] == "acquire"]
        assert acq and acq[0]["payload"]["enabled"] is False


class TestGovernedWebSearcher:
    def test_authorized_search_passes_through(self):
        inner = MagicMock()
        inner.search.return_value = "resultado"
        lease = SystemLease.acquire("web_search", "x")
        gov = GovernedWebSearcher(inner, lease)
        assert gov.search("query", max_results=5) == "resultado"
        inner.search.assert_called_once_with("query", max_results=5)

    def test_denied_search_never_reaches_inner(self):
        inner = MagicMock()
        lease = SystemLease.acquire("web_search", "x", max_calls=0)
        gov = GovernedWebSearcher(inner, lease)
        with pytest.raises(SystemLeaseDenied):
            gov.search("query")
        inner.search.assert_not_called()


class TestStudierUnderSystemLease:
    """The wiring: a nightly run searches ONLY through the lease, every
    query audited, the raw searcher restored afterwards."""

    def _studier(self, tmp_path, monkeypatch, n_topics=1):
        from src.core.studier import NightlyStudier
        from src.core.study_queue import StudyQueue

        queue = StudyQueue(path=str(tmp_path / "queue.jsonl"))
        # Distinct topics — the queue's dedup rejects similar titles.
        topics = ["fotossíntese em plantas", "buracos negros supermassivos"]
        for topic in topics[:n_topics]:
            queue.add(topic)

        searcher = MagicMock()
        searcher.search.return_value = (
            "Source 1 (Doc):\nconteúdo relevante\n[https://example.org/a]"
        )
        engine = MagicMock()
        engine.chat.return_value = "1. consulta um\n2. consulta dois"

        s = NightlyStudier(
            engine=engine, queue=queue, web_searcher=searcher,
            output_dir=str(tmp_path / "pending"),
        )
        # Keep the run hermetic: no Palace snapshot, no consolidation.
        monkeypatch.setattr(s, "_backup_palace", lambda: None)
        monkeypatch.setattr(s, "_consolidate_memory", lambda: None)
        monkeypatch.setattr(s, "_consolidate_preferences", lambda: None)
        return s, searcher

    def test_run_audits_acquire_calls_release(self, tmp_path, monkeypatch):
        s, searcher = self._studier(tmp_path, monkeypatch)
        s.run()
        events = [e["event"] for e in _audit_events()]
        assert "acquire" in events and "release" in events
        n_calls = len([e for e in events if e == "call"])
        assert n_calls == searcher.search.call_count > 0

    def test_raw_searcher_restored_after_run(self, tmp_path, monkeypatch):
        s, searcher = self._studier(tmp_path, monkeypatch)
        s.run()
        assert s.web_searcher is searcher

    def test_exhausted_lease_degrades_not_crashes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_SYSTEM_LEASE_MAX_CALLS", "1")
        s, searcher = self._studier(tmp_path, monkeypatch, n_topics=2)
        s.run()  # must not raise — denied queries are skipped per-query
        assert searcher.search.call_count == 1
        denies = [e for e in _audit_events() if e["event"] == "deny"]
        assert denies, "the over-cap searches must be audited as denials"

    def test_kill_switch_runs_offline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_SYSTEM_LEASE", "0")
        s, searcher = self._studier(tmp_path, monkeypatch)
        s.run()  # no crash
        searcher.search.assert_not_called()
