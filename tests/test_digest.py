"""
Tests for src/core/digest.py — the "o que mudou" delta digest (Onda 4a).

`assemble` is pure (synthetic DeltaSources). `gather` is exercised fail-soft
with mocked sources. `get/set_last_seen` round-trip + fail-soft. The dispatch
regex is locked in evals/cases/digest.py; here a sanity check + the handler.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.core import digest as dg
from src.core.dispatch.digest import WHATS_CHANGED_RE


NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=5)


# --- assemble (pure) -------------------------------------------------------

def test_assemble_empty():
    out = dg.assemble(dg.DeltaSources(), SINCE, NOW)
    assert "Nada novo" in out
    assert "O que mudou" in out


def test_assemble_emails_section():
    src = dg.DeltaSources(emails=[
        {"from": "João <joao@x.com>", "subject": "Reunião amanhã"},
        {"from": "ACME", "subject": "Fatura"},
    ])
    out = dg.assemble(src, SINCE, NOW)
    assert "E-mails novos (2)" in out
    assert "João" in out and "Reunião amanhã" in out
    assert "joao@x.com" not in out      # the <addr> is stripped for display


def test_assemble_actions_uses_labels():
    src = dg.DeltaSources(actions=[{"action": "send_email"}, {"action": "undo"}])
    out = dg.assemble(src, SINCE, NOW)
    assert "Ações que tomei (2)" in out
    assert "enviei um e-mail" in out
    assert "desfiz uma ação" in out


def test_assemble_studies_and_events():
    src = dg.DeltaSources(
        studies=["transformers_2026.md"],
        events=[{"summary": "Dentista", "start": "2026-06-05T15:00:00+00:00"}],
    )
    out = dg.assemble(src, SINCE, NOW)
    assert "Estudos novos do Studier (1)" in out
    assert "transformers_2026.md" in out
    assert "Próximos eventos" in out and "Dentista" in out


def test_assemble_no_tables():
    src = dg.DeltaSources(emails=[{"from": "x", "subject": "y"}])
    assert "|" not in dg.assemble(src, SINCE, NOW)


def test_assemble_truncates_long_lists():
    src = dg.DeltaSources(emails=[{"from": f"p{i}", "subject": "s"} for i in range(10)])
    out = dg.assemble(src, SINCE, NOW)
    assert "E-mails novos (10)" in out         # count is honest
    assert out.count("  • ") == 5              # but only 5 shown


def test_fmt_event_start():
    assert dg._fmt_event_start("2026-06-05T15:00:00+00:00")  # non-empty, parseable
    assert dg._fmt_event_start("not-a-date") == ""
    assert dg._fmt_event_start(None) == ""


@pytest.mark.parametrize("delta,expected", [
    (timedelta(minutes=30), "na última hora"),
    (timedelta(hours=10), "desde ontem"),
    (timedelta(days=3), "nos últimos 3 dias"),
])
def test_fmt_since(delta, expected):
    assert dg._fmt_since(NOW - delta, NOW) == expected


# --- last_seen round-trip + fail-soft --------------------------------------

def test_last_seen_roundtrip(tmp_path):
    p = str(tmp_path / "last_seen.json")
    assert dg.get_last_seen(p) is None
    dg.set_last_seen(NOW, p)
    got = dg.get_last_seen(p)
    assert got == NOW


def test_get_last_seen_missing_is_none(tmp_path):
    assert dg.get_last_seen(str(tmp_path / "nope.json")) is None


def test_get_last_seen_malformed_is_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert dg.get_last_seen(str(p)) is None


# --- gather (fail-soft + filtering) ----------------------------------------

def test_gather_filters_emails_by_since(monkeypatch):
    import src.integrations.gmail_client as gmail
    monkeypatch.setattr(gmail, "list_recent", lambda count=15, query=None: [
        {"from": "new", "subject": "depois", "date_iso": (NOW - timedelta(hours=1)).isoformat()},
        {"from": "old", "subject": "antes", "date_iso": (NOW - timedelta(hours=10)).isoformat()},
    ])
    # isolate other sources
    monkeypatch.setattr(dg, "_STUDIER_DIR", str(NOW))  # nonexistent dir → no studies
    import src.integrations.calendar_client as cal
    monkeypatch.setattr(cal, "list_events", lambda max_results=10: [])
    import src.core.action_ledger as al
    monkeypatch.setattr(al.ledger, "load_entries", lambda: [])

    out = dg.gather(SINCE, NOW)
    assert [m["subject"] for m in out.emails] == ["depois"]


def test_gather_filters_actions_by_since(monkeypatch):
    import src.core.action_ledger as al
    monkeypatch.setattr(al.ledger, "load_entries", lambda: [
        {"ts": (NOW - timedelta(hours=1)).isoformat(), "action": "create_event"},
        {"ts": (NOW - timedelta(hours=9)).isoformat(), "action": "send_email"},
    ])
    import src.integrations.gmail_client as gmail
    monkeypatch.setattr(gmail, "list_recent", lambda count=15, query=None: [])
    import src.integrations.calendar_client as cal
    monkeypatch.setattr(cal, "list_events", lambda max_results=10: [])
    monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")

    out = dg.gather(SINCE, NOW)
    assert [a["action"] for a in out.actions] == ["create_event"]


def test_gather_is_fail_soft(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")
    import src.integrations.gmail_client as gmail
    import src.integrations.calendar_client as cal
    import src.core.action_ledger as al
    monkeypatch.setattr(gmail, "list_recent", _boom)
    monkeypatch.setattr(cal, "list_events", _boom)
    monkeypatch.setattr(al.ledger, "load_entries", _boom)
    monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")

    out = dg.gather(SINCE, NOW)   # must not raise
    assert out.emails == [] and out.actions == [] and out.events == []


def test_gather_studies_by_mtime(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(dg, "_STUDIER_DIR", str(tmp_path))
    fresh = tmp_path / "fresh.md"; fresh.write_text("x")
    stale = tmp_path / "stale.md"; stale.write_text("y")
    os.utime(fresh, (NOW.timestamp() - 60, NOW.timestamp() - 60))
    os.utime(stale, (NOW.timestamp() - 99999, NOW.timestamp() - 99999))
    # isolate other sources
    import src.integrations.gmail_client as gmail
    import src.integrations.calendar_client as cal
    import src.core.action_ledger as al
    monkeypatch.setattr(gmail, "list_recent", lambda count=15, query=None: [])
    monkeypatch.setattr(cal, "list_events", lambda max_results=10: [])
    monkeypatch.setattr(al.ledger, "load_entries", lambda: [])

    out = dg.gather(SINCE, NOW)
    assert out.studies == ["fresh.md"]


# --- new_studies_since: the shared definition (M7-A6) -----------------------

def test_new_studies_since_is_shared_by_digest_and_daily(monkeypatch, tmp_path):
    # The headline of M7-A6: same folder, same answer. Before, the digest
    # filtered by the moving marker and the daily by a fixed 24h window with
    # its OWN copy of the glob — two definitions of "new" that could
    # disagree. Now both delegate to new_studies_since; here we pin that the
    # daily's gather really gets its list from the shared helper.
    import os
    monkeypatch.setattr(dg, "_STUDIER_DIR", str(tmp_path))
    fresh = tmp_path / "fresh.md"; fresh.write_text("x")
    old = tmp_path / "old.md"; old.write_text("y")     # 25h — daily must skip
    os.utime(fresh, (NOW.timestamp() - 3600, NOW.timestamp() - 3600))
    os.utime(old, (NOW.timestamp() - 90000, NOW.timestamp() - 90000))

    from datetime import timedelta
    assert dg.new_studies_since(NOW - timedelta(hours=24)) == ["fresh.md"]
    assert sorted(dg.new_studies_since(NOW - timedelta(hours=48))) == [
        "fresh.md", "old.md"]

    import src.daily_driver as dd
    import src.integrations.gmail_client as gmail
    import src.integrations.calendar_client as cal
    import src.core.gate_queue as gq
    monkeypatch.setattr(gmail, "list_recent", lambda count=10, query=None: [])
    monkeypatch.setattr(cal, "list_events", lambda max_results=15: [])
    monkeypatch.setattr(gq.gate_queue, "pending", lambda: [])
    out = dd.gather(NOW)
    assert out.studies_yesterday == ["fresh.md"]


def test_new_studies_since_unreadable_dir_is_empty(monkeypatch):
    monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")
    from datetime import timedelta
    assert dg.new_studies_since(NOW - timedelta(hours=24)) == []


# --- whats_changed advances the marker -------------------------------------

def test_whats_changed_advances_marker_with_overlap(monkeypatch, tmp_path):
    p = str(tmp_path / "last_seen.json")
    monkeypatch.setattr(dg, "_LAST_SEEN_PATH", p)
    monkeypatch.setattr(dg, "gather", lambda since, now: dg.DeltaSources())
    assert dg.get_last_seen(p) is None
    text = dg.whats_changed(now=NOW)
    assert "O que mudou" in text
    # M7-A7: marker parks BEHIND now (default 10min) — a digest lost on
    # the way out (SSE drop / TTS cut) is recoverable by re-asking soon.
    assert dg.get_last_seen(p) == NOW - timedelta(minutes=10)


def test_whats_changed_overlap_env_override(monkeypatch, tmp_path):
    p = str(tmp_path / "last_seen.json")
    monkeypatch.setattr(dg, "_LAST_SEEN_PATH", p)
    monkeypatch.setattr(dg, "gather", lambda since, now: dg.DeltaSources())
    monkeypatch.setenv("CORTEX_DIGEST_OVERLAP_MIN", "0")
    dg.whats_changed(now=NOW)
    assert dg.get_last_seen(p) == NOW   # overlap 0 → legacy behaviour


def test_whats_changed_marker_never_moves_backwards(monkeypatch, tmp_path):
    # Two asks 1min apart: the second's now-overlap is BEFORE the stored
    # marker; keeping the max stops repeated asks from re-widening the
    # window indefinitely.
    p = str(tmp_path / "last_seen.json")
    monkeypatch.setattr(dg, "_LAST_SEEN_PATH", p)
    monkeypatch.setattr(dg, "gather", lambda since, now: dg.DeltaSources())
    dg.whats_changed(now=NOW)
    first_marker = dg.get_last_seen(p)
    dg.whats_changed(now=NOW + timedelta(minutes=1))
    assert dg.get_last_seen(p) >= first_marker


# --- regex sanity (full lock in evals) -------------------------------------

def test_regex_matches_and_ignores():
    assert WHATS_CHANGED_RE.match("o que mudou")
    assert WHATS_CHANGED_RE.match("what did I miss")
    assert not WHATS_CHANGED_RE.match("como eu mudo a senha do wifi")


# --- handler integration ---------------------------------------------------

def test_handle_whats_changed(monkeypatch):
    from src.core.orchestrator import CortexOrchestrator
    monkeypatch.setattr(dg, "whats_changed", lambda now=None, lang="pt": "O que mudou: nada.")
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    reply, mode = o._handle_whats_changed()
    assert mode == "WHATS_CHANGED"
    assert "O que mudou" in reply



# --- a source that failed is not "nothing new" (s88, Codex lote 13) ---------

class TestAFailedSourceHoldsTheMarker:
    """Failures became empty lists and whats_changed advanced the marker
    anyway: Gmail down at 12:00 → "Nada novo — tudo tranquilo", marker at
    11:50, and once Gmail was back the 10:00 email was outside the window
    for good (s88, Codex lote 13 — invariant 7)."""

    T = staticmethod(lambda h, m=0: datetime(2026, 10, 19, h, m, tzinfo=timezone.utc))

    @pytest.fixture
    def world(self, monkeypatch, tmp_path):
        import src.integrations.gmail_client as gmail
        import src.integrations.calendar_client as cal
        import src.core.action_ledger as al
        p = str(tmp_path / "last_seen.json")
        monkeypatch.setattr(dg, "_LAST_SEEN_PATH", p)
        monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")
        state = {"down": True}
        mail = [{"from": "Ana <ana@x.com>", "subject": "Proposta",
                 "date_iso": self.T(10).isoformat()}]

        def list_recent(count=15, query=None):
            if state["down"]:
                raise RuntimeError("HTTP 503")
            return mail
        monkeypatch.setattr(gmail, "list_recent", list_recent)
        monkeypatch.setattr(cal, "list_events", lambda max_results=10: [])
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [])
        dg.set_last_seen(self.T(9), p)
        return state, p

    def test_the_email_is_shown_once_gmail_is_back(self, world):
        state, p = world
        first = dg.whats_changed(now=self.T(12))
        assert "tudo tranquilo" not in first
        assert "Não consegui ler agora: e-mails" in first
        assert dg.get_last_seen(p) == self.T(9)        # marker held
        state["down"] = False
        second = dg.whats_changed(now=self.T(12, 5))
        assert "Proposta" in second
        assert dg.get_last_seen(p) == self.T(12, 5) - timedelta(minutes=10)

    def test_a_calendar_failure_is_said_but_holds_nothing(self, world, monkeypatch):
        import src.integrations.calendar_client as cal
        state, p = world
        state["down"] = False

        def boom(max_results=10):
            raise RuntimeError("down")
        monkeypatch.setattr(cal, "list_events", boom)
        text = dg.whats_changed(now=self.T(12))
        assert "Proposta" in text and "agenda" in text
        assert dg.get_last_seen(p) == self.T(12) - timedelta(minutes=10)

    def test_gather_names_what_failed(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("down")
        import src.integrations.gmail_client as gmail
        import src.integrations.calendar_client as cal
        import src.core.action_ledger as al
        monkeypatch.setattr(gmail, "list_recent", _boom)
        monkeypatch.setattr(cal, "list_events", _boom)
        monkeypatch.setattr(al.ledger, "load_entries", _boom)
        monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")
        assert dg.gather(SINCE, NOW).unavailable == ["emails", "actions", "events"]

    def test_english(self):
        out = dg.assemble(dg.DeltaSources(unavailable=["emails"]), SINCE, NOW, lang="en")
        assert "Couldn't read just now: emails" in out
        assert "all quiet" not in out



class TestEveryNewEmailIsCounted:
    """Only the newest 15 of the inbox were read: 20 new emails were
    announced as "E-mails novos (15)" and the other five never surfaced
    (s88, Codex lote 13)."""

    def _gather(self, monkeypatch, n_new, total):
        import src.integrations.gmail_client as gmail
        import src.integrations.calendar_client as cal
        import src.core.action_ledger as al
        asked = []

        def list_recent(count=15, query=None):
            asked.append(query)
            return [{"from": f"p{i}", "subject": "s",
                     "date_iso": (NOW - timedelta(minutes=i + 1)).isoformat()}
                    for i in range(min(count, n_new))]
        monkeypatch.setattr(gmail, "list_recent", list_recent)
        monkeypatch.setattr(gmail, "count_matching", lambda q, cap=500: total)
        monkeypatch.setattr(cal, "list_events", lambda max_results=10: [])
        monkeypatch.setattr(al.ledger, "load_entries", lambda: [])
        monkeypatch.setattr(dg, "_STUDIER_DIR", "/nonexistent-dir-xyz")
        return dg.gather(SINCE, NOW), asked

    def test_twenty_are_twenty(self, monkeypatch):
        src, asked = self._gather(monkeypatch, n_new=20, total=20)
        assert asked == [f"after:{int(SINCE.timestamp())}"]   # the window itself
        assert "E-mails novos (20)" in dg.assemble(src, SINCE, NOW)

    def test_a_partial_page_needs_no_count(self, monkeypatch):
        src, _ = self._gather(monkeypatch, n_new=3, total=999)
        assert src.email_total is None
        assert "E-mails novos (3)" in dg.assemble(src, SINCE, NOW)


def test_count_matching_pages_through_ids(monkeypatch):
    import src.integrations.gmail_client as gmail
    pages = [{"messages": [{"id": str(i)} for i in range(500)], "nextPageToken": "t2"},
             {"messages": [{"id": "x"}] * 20}]
    seen = []

    class _List:
        def __init__(self, **kw):
            seen.append(kw)

        def execute(self):
            return pages[len(seen) - 1]

    class _Svc:
        def users(self):
            return self

        def messages(self):
            return self

        def list(self, **kw):
            return _List(**kw)
    monkeypatch.setattr(gmail, "_service", lambda: _Svc())
    assert gmail.count_matching("after:1", cap=1000) == 520
    assert seen[1]["pageToken"] == "t2" and seen[0]["q"] == "after:1"
