"""
Tests for src/daily_driver.py — the spoken morning briefing (Onda 4c).

`assemble`/`headline` are pure (synthetic DailySources). `gather` is exercised
fail-soft with mocked sources. `speak` and the notification are monkeypatched —
no real audio/banners in tests.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src import daily_driver as dd


# 08:00 local-ish; use a fixed aware UTC instant.
NOW = datetime(2026, 6, 5, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_real_refresh(monkeypatch):
    """Slice 5 v2 wired a real FAA fetch into main() on --speak/--notify. NO test
    may hit the network or the real ledger through it — neutralise refresh_jobs
    by default; the refresh-specific tests install their own spy on top."""
    import src.integrations.job_hunter as jh
    monkeypatch.setattr(jh, "refresh_jobs", lambda **k: dict(jh._ZERO))


def _ev(summary, start):
    return {"summary": summary, "start": start}


# --- assemble (pure) -------------------------------------------------------

def test_assemble_empty_is_calm():
    out = dd.assemble(dd.DailySources(), NOW)
    assert "Bom dia" in out
    assert "Nada urgente" in out
    assert "|" not in out


def test_assemble_events():
    today = NOW.astimezone().strftime("%Y-%m-%dT15:00:00%z")
    src = dd.DailySources(events_today=[_ev("Dentista", today)])
    out = dd.assemble(src, NOW)
    assert "1 evento" in out and "Dentista" in out


def test_assemble_unread_with_sender():
    src = dd.DailySources(unread=[{"from": "João <j@x.com>"}, {"from": "ACME"}])
    out = dd.assemble(src, NOW)
    assert "2 e-mails não lidos" in out
    assert "João" in out and "j@x.com" not in out


def test_assemble_studies_and_pending():
    src = dd.DailySources(
        studies_yesterday=["transformers.md"],
        pending=[{"id": "1", "type": "send_email"}],
    )
    out = dd.assemble(src, NOW)
    assert "1 estudo novo" in out
    assert "1 ação aguardando" in out and "make gate" in out


def test_assemble_singular_plural():
    one = dd.assemble(dd.DailySources(unread=[{"from": "x"}]), NOW)
    assert "1 e-mail não lido" in one
    two = dd.assemble(dd.DailySources(unread=[{"from": "x"}, {"from": "y"}]), NOW)
    assert "2 e-mails não lidos" in two


def test_greeting_by_hour():
    morning = datetime(2026, 6, 5, 6, 0, tzinfo=timezone.utc)  # may shift by tz
    assert dd._greeting(morning) in ("Bom dia", "Boa tarde", "Boa noite")


# --- headline --------------------------------------------------------------

def test_headline_summarises():
    src = dd.DailySources(
        events_today=[_ev("a", None)],
        unread=[{"from": "x"}],
        pending=[{"id": "1"}],
    )
    h = dd.headline(src)
    assert "1 evento(s)" in h and "1 não lidos" in h and "1 aprovação(ões)" in h


def test_headline_empty():
    assert dd.headline(dd.DailySources()) == "nada urgente"


# --- gather (fail-soft + filtering) ----------------------------------------

def test_gather_filters_events_to_today(monkeypatch):
    today = NOW.astimezone().strftime("%Y-%m-%dT15:00:00%z")
    tomorrow = (NOW + timedelta(days=1)).astimezone().strftime("%Y-%m-%dT15:00:00%z")
    import src.integrations.calendar_client as cal
    monkeypatch.setattr(cal, "list_events",
                        lambda max_results=15: [_ev("hoje", today), _ev("amanhã", tomorrow)])
    import src.integrations.gmail_client as gmail
    monkeypatch.setattr(gmail, "list_recent", lambda count=10, query=None: [])
    import src.core.gate_queue as gq
    monkeypatch.setattr(gq.gate_queue, "pending", lambda: [])
    monkeypatch.setattr("src.core.digest._STUDIER_DIR", "/nonexistent-xyz")

    out = dd.gather(NOW)
    assert [e["summary"] for e in out.events_today] == ["hoje"]


def test_gather_is_fail_soft(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")
    import src.integrations.calendar_client as cal
    import src.integrations.gmail_client as gmail
    import src.core.gate_queue as gq
    monkeypatch.setattr(cal, "list_events", _boom)
    monkeypatch.setattr(gmail, "list_recent", _boom)
    monkeypatch.setattr(gq.gate_queue, "pending", _boom)
    monkeypatch.setattr("src.core.digest._STUDIER_DIR", "/nonexistent-xyz")

    out = dd.gather(NOW)  # must not raise
    assert out.events_today == [] and out.unread == [] and out.pending == []


def test_gather_uses_unread_query(monkeypatch):
    seen = {}
    import src.integrations.gmail_client as gmail
    monkeypatch.setattr(gmail, "list_recent",
                        lambda count=10, query=None: seen.setdefault("q", query) or [])
    import src.integrations.calendar_client as cal
    import src.core.gate_queue as gq
    monkeypatch.setattr(cal, "list_events", lambda max_results=15: [])
    monkeypatch.setattr(gq.gate_queue, "pending", lambda: [])
    monkeypatch.setattr("src.core.digest._STUDIER_DIR", "/nonexistent-xyz")
    dd.gather(NOW)
    assert seen["q"] == "is:unread"


# --- speak (no real audio) -------------------------------------------------

def test_speak_invokes_say(monkeypatch):
    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return dd.subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(dd.subprocess, "run",
                        fake_run)
    assert dd.speak("olá") is True
    assert captured["cmd"][0] == "say"
    assert "olá" in captured["cmd"]


def test_speak_fail_soft(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no say")
    monkeypatch.setattr(dd.subprocess, "run", _boom)
    assert dd.speak("x") is False


# --- main ------------------------------------------------------------------

def test_main_stdout_default(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    spoke = {}
    monkeypatch.setattr(dd, "speak", lambda t, lang="pt": spoke.setdefault("s", True))
    rc = dd.main([])
    assert rc == 0
    assert "Bom dia" in capsys.readouterr().out
    assert "s" not in spoke   # no --speak → didn't speak


def test_main_speak_and_notify(monkeypatch, tmp_path):
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources(unread=[{"from": "x"}]))
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    spoke, notified = {}, {}
    monkeypatch.setattr(dd, "speak", lambda t, lang="pt": spoke.setdefault("s", t))
    import src.core.heartbeat as hb
    monkeypatch.setattr(hb, "notify", lambda title, msg: notified.setdefault("m", msg))
    rc = dd.main(["--speak", "--notify"])
    assert rc == 0
    assert "s" in spoke
    assert "não lidos" in notified["m"]


def test_main_writes_briefing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    dd.main(["--stdout"])
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1 and "Bom dia" in files[0].read_text()


# --- Job Hunter surface (slice 5) ------------------------------------------

def _job(jid, title, location="Reading, RG1 1AA", score=0.83, status="new",
         closing_at=None):
    return {"id": jid, "title": title, "location": location,
            "score": score, "status": status, "closing_at": closing_at}


class _FakeJobLog:
    """Minimal JobLog stand-in: list() honours the status filter, set_status()
    records calls. Rows are plain dicts (the code subscripts r['id'] etc.)."""
    def __init__(self, rows):
        self._rows = rows
        self.seen = []

    def list(self, *, limit=20, status=None, source=None, order="distance"):
        rows = [r for r in self._rows if status is None or r.get("status") == status]
        return rows[:limit]

    def list_open(self, *, limit=20, status=None, source=None, order="score"):
        """Mirrors the real over-fetch-then-filter: drop closed vacancies and
        THEN trim, so `limit` still yields `limit` live ones (s82). Uses the
        production predicate so this fake can't quietly disagree with it."""
        from src.integrations.ats import is_closed
        rows = self.list(limit=10_000, status=status, source=source, order=order)
        return [r for r in rows if not is_closed(r.get("closing_at"))][:limit]

    def set_status(self, job_id, status):
        self.seen.append((job_id, status))
        return True


def test_assemble_jobs_strips_postcode_and_points_to_voice():
    src = dd.DailySources(jobs=[
        _job(1, "Developer/Software Engineer Apprentice", "Farnborough, GU14 7LP"),
        _job(2, "Data & AI Consultant Apprenticeship", "London, NW6 1HU"),
    ])
    out = dd.assemble(src, NOW)
    assert "2 vagas novas alinhadas" in out
    assert "em Farnborough" in out and "GU14 7LP" not in out   # city only
    assert "tem vaga nova" in out                              # ties to the voice intent


def test_assemble_jobs_singular():
    src = dd.DailySources(jobs=[_job(1, "Software Tester Apprentice")])
    out = dd.assemble(src, NOW)
    assert "1 vaga nova alinhada" in out


def test_assemble_jobs_alone_is_not_calm():
    # Jobs present, everything else empty → the "nada urgente" line must NOT fire.
    out = dd.assemble(dd.DailySources(jobs=[_job(1, "X")]), NOW)
    assert "Nada urgente" not in out


def test_headline_includes_jobs():
    out = dd.headline(dd.DailySources(jobs=[_job(1, "X"), _job(2, "Y")]))
    assert "2 vaga(s)" in out


def _stub_other_sources(monkeypatch):
    """Silence calendar/gmail/gate/studier so a gather test isolates jobs."""
    import src.integrations.calendar_client as cal
    import src.integrations.gmail_client as gmail
    import src.core.gate_queue as gq
    monkeypatch.setattr(cal, "list_events", lambda max_results=15: [])
    monkeypatch.setattr(gmail, "list_recent", lambda count=10, query=None: [])
    monkeypatch.setattr(gq.gate_queue, "pending", lambda: [])
    monkeypatch.setattr("src.core.digest._STUDIER_DIR", "/nonexistent-xyz")


def test_gather_keeps_only_ranked_strong_matches(monkeypatch):
    _stub_other_sources(monkeypatch)
    fake = _FakeJobLog([
        _job(1, "Strong", score=0.83),
        _job(2, "Weak", score=0.50),      # below the 0.75 floor → dropped
        _job(3, "Unranked", score=None),  # never ranked → dropped
    ])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: fake)
    out = dd.gather(NOW)
    assert [j["title"] for j in out.jobs] == ["Strong"]


def test_gather_jobs_fail_soft(monkeypatch):
    _stub_other_sources(monkeypatch)
    def _boom():
        raise RuntimeError("ledger down")
    monkeypatch.setattr("src.integrations.jobs.get_log", _boom)
    out = dd.gather(NOW)          # must not raise
    assert out.jobs == []


def test_main_marks_jobs_seen_on_delivery(monkeypatch, tmp_path):
    fake = _FakeJobLog([])
    monkeypatch.setattr(dd, "gather",
                        lambda now: dd.DailySources(jobs=[_job(7, "Dev Apprentice")]))
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    monkeypatch.setattr(dd, "speak", lambda t, lang="pt": True)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: fake)
    dd.main(["--speak"])
    assert fake.seen == [(7, "seen")]     # surfaced once, won't repeat tomorrow


def test_main_stdout_does_not_consume_jobs(monkeypatch, tmp_path):
    fake = _FakeJobLog([])
    monkeypatch.setattr(dd, "gather",
                        lambda now: dd.DailySources(jobs=[_job(7, "Dev Apprentice")]))
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: fake)
    dd.main(["--stdout"])                 # preview only
    assert fake.seen == []                # nothing marked seen


# --- daily auto-refresh (slice 5 v2) ---------------------------------------

def _spy_refresh(monkeypatch, calls):
    import src.integrations.job_hunter as jh
    monkeypatch.setattr(jh, "refresh_jobs",
                        lambda **k: calls.append(k) or dict(jh._ZERO))


def test_main_refreshes_jobs_on_delivery(monkeypatch, tmp_path):
    calls = []
    _spy_refresh(monkeypatch, calls)
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    monkeypatch.setattr(dd, "speak", lambda t, lang="pt": True)
    monkeypatch.delenv("CORTEX_JOBS_DAILY_REFRESH", raising=False)
    dd.main(["--speak"])
    assert len(calls) == 1                # fetched fresh before assembling


def test_main_no_refresh_on_stdout_preview(monkeypatch, tmp_path):
    calls = []
    _spy_refresh(monkeypatch, calls)
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    dd.main(["--stdout"])                 # preview stays offline
    assert calls == []


def test_main_refresh_disabled_by_env(monkeypatch, tmp_path):
    calls = []
    _spy_refresh(monkeypatch, calls)
    monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
    monkeypatch.setattr(dd, "_BRIEFINGS_DIR", str(tmp_path))
    monkeypatch.setattr(dd, "speak", lambda t, lang="pt": True)
    monkeypatch.setenv("CORTEX_JOBS_DAILY_REFRESH", "0")
    dd.main(["--speak"])
    assert calls == []                    # opted out


def test_assemble_jobs_include_deadline():
    # Far-future date → stable "fecha em N dias" the assertion can pin.
    src = dd.DailySources(jobs=[
        {"id": 1, "title": "Software Apprentice", "location": "Reading, RG1 1AB",
         "score": 0.83, "closing_at": "2099-09-01T23:59:59Z"},
    ])
    out = dd.assemble(src, NOW)
    assert "Software Apprentice" in out and "em Reading" in out   # city only, no postcode
    assert "fecha em" in out                                     # deadline spoken


def test_assemble_jobs_without_deadline_is_clean():
    src = dd.DailySources(jobs=[
        {"id": 1, "title": "Software Apprentice", "location": "Reading",
         "score": 0.83, "closing_at": None},
    ])
    out = dd.assemble(src, NOW)
    assert "Software Apprentice" in out and "fecha" not in out


class TestPostToChat:
    """The morning briefing lands in the chat page as a fresh conversation
    (s81): opening /ui/chat shows the day, phrased as a Q&A Pedro can continue."""

    def _store(self, tmp_path, monkeypatch):
        from src.ui.chat_store import ChatStore
        store = ChatStore(db_path=str(tmp_path / "chat.db"))
        # daily_driver imports `store` from the module at call time
        monkeypatch.setattr("src.ui.chat_store.store", store)
        return store

    def test_creates_a_conversation_with_the_briefing(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        cid = dd.post_to_chat("Bom dia, Pedro. 3 emails não lidos.", NOW)
        assert cid is not None
        convos = store.list_conversations()
        assert len(convos) == 1
        assert "Bom dia" in convos[0]["title"]
        turns = store.get_turns(cid)
        assert turns[0]["user"] == "o que eu tenho pra hoje?"
        assert "3 emails" in turns[0]["reply"]
        assert turns[0]["source"] == "daily"

    def test_is_fail_soft(self, monkeypatch):
        # A broken chat store must never break the (already-delivered) briefing.
        import src.ui.chat_store as cs
        class _Boom:
            def create_conversation(self, *a, **k): raise RuntimeError("db down")
            def conversation_titled_on(self, *a, **k): raise RuntimeError("db down")
        monkeypatch.setattr(cs, "store", _Boom())
        assert dd.post_to_chat("qualquer texto", NOW) is None   # no exception

    def test_to_chat_flag_triggers_post(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
        monkeypatch.setattr(dd, "speak", lambda *a, **k: True)
        monkeypatch.setattr(dd, "_write_briefing", lambda *a, **k: None)
        dd.main(["--to-chat"])
        assert len(store.list_conversations()) == 1

    def test_no_flag_no_chat(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        monkeypatch.setattr(dd, "gather", lambda now: dd.DailySources())
        monkeypatch.setattr(dd, "_write_briefing", lambda *a, **k: None)
        dd.main(["--stdout"])
        assert store.list_conversations() == []   # a preview must not post


# --- the briefing never pushes a closed vacancy (s82) --------------------
# Same rule as the conversation handler, shared via ats.is_closed. Wired at
# BOTH surfaces on purpose: filtering only one is the recurring cross-cutting
# bug in this project (a live match pushed off the list by a dead one).

def _iso_in_days(days):
    from datetime import datetime, timedelta
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()}T12:00:00Z"


def test_gather_drops_closed_vacancies(monkeypatch):
    _stub_other_sources(monkeypatch)
    fake = _FakeJobLog([
        _job(1, "Closed Top", score=0.95, closing_at=_iso_in_days(-1)),
        _job(2, "Open Match", score=0.80, closing_at=_iso_in_days(6)),
    ])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: fake)
    out = dd.gather(NOW)
    # the dead one scored higher and would otherwise have led the briefing
    assert [j["title"] for j in out.jobs] == ["Open Match"]


def test_gather_keeps_vacancies_without_a_deadline(monkeypatch):
    _stub_other_sources(monkeypatch)
    fake = _FakeJobLog([_job(1, "No Deadline", score=0.83, closing_at=None)])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: fake)
    out = dd.gather(NOW)
    assert [j["title"] for j in out.jobs] == ["No Deadline"]


# --- a source that failed is not an empty source (s88, Codex lote 10) -------

def _broken_google(monkeypatch):
    from src.integrations import calendar_client as cal, gmail_client as gmail
    from src.core import gate_queue as gq

    def boom(**_k):
        raise cal.CalendarUnavailable("token expired")
    monkeypatch.setattr(cal, "list_events", boom)
    monkeypatch.setattr(gmail, "list_recent", boom)
    monkeypatch.setattr(gq.gate_queue, "pending", lambda: [])
    monkeypatch.setattr("src.core.digest._STUDIER_DIR", "/nonexistent-xyz")


def test_a_failed_calendar_is_not_an_empty_day(monkeypatch):
    """With the Google token broken the briefing said "Você não tem eventos
    na agenda de hoje… Nada urgente por aqui" — a meeting could be there."""
    _broken_google(monkeypatch)
    out = dd.gather(NOW)
    text = dd.assemble(out, NOW)
    assert "não tem eventos" not in text and "Nada urgente" not in text
    assert "Não consegui consultar a sua agenda" in text
    assert "Também não consegui ver os seus e-mails" in text
    assert dd.headline(out).startswith("agenda indisponível")


def test_a_quiet_day_is_still_calm():
    out = dd.DailySources()
    assert "Nada urgente" in dd.assemble(out, NOW)
    assert dd.headline(out) == "nada urgente"


def test_only_the_email_failed():
    out = dd.DailySources(events_today=[_ev("Reunião", "2026-06-05T10:00:00+00:00")],
                          failed=["email"])
    text = dd.assemble(out, NOW)
    assert "Hoje você tem 1 evento" in text and "Não consegui ver os seus e-mails." in text


def test_a_failed_job_refresh_is_said_and_keeps_the_day_from_calm():
    out = dd.DailySources(failed=["jobs_refresh"])
    text = dd.assemble(out, NOW)
    assert "Não consegui buscar vagas novas hoje." in text and "Nada urgente" not in text
