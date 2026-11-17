"""
Job Hunter surface (slice 4, s76): regex routing + read-only handler.

Offline — the handler reads an in-memory JobLog (no network, no LLM, no
embedding). Mirrors test_fitness_mixin's shape.
"""
import re

from src.core.dispatch.jobs import JOBS_TOP_RE
from src.core.dispatch.registry import simulate
from src.core.jobs_mixin import JobsMixin
from src.core.text_utils import strip_markdown_for_speech
from src.integrations.ats import JobPosting
from src.integrations.jobs import JobLog


# --- regex --------------------------------------------------------------

def test_jobs_top_re_matches_pt_and_en():
    for s in ("tem vaga nova?", "tem vagas novas", "vagas pra mim",
              "vagas",   # bare plural — the courtesy-stripped "vagas pra mim"
              "quais as minhas melhores vagas", "minhas vagas",
              "vagas de emprego", "vagas de trabalho",
              "any new jobs", "any jobs for me", "my best jobs",
              "top jobs", "new jobs", "job matches"):
        assert JOBS_TOP_RE.match(s), f"should match: {s!r}"


def test_jobs_top_re_does_not_hijack():
    # "vaga" as a parking space (no job-sense qualifier) and mid-sentence forms.
    for s in ("tem vaga no estacionamento?", "minhas vagas estão ruins",
              "vaga", "jobs", "tem uma vaga ali"):
        assert not JOBS_TOP_RE.match(s), f"should NOT match: {s!r}"


# --- handler ------------------------------------------------------------

def _posting(url, title, *, location="Reading", distance=None):
    return JobPosting(source="findapprenticeship", board="uk", company="X",
                      ext_id="1", title=title, location=location, url=url,
                      description="d", posted_at=None, distance=distance)


def _ledger_with(scored):
    """In-memory JobLog populated with (title, distance, score) rows."""
    db = JobLog(db_path=":memory:")
    db.upsert([_posting(f"u{i}", t, distance=d)
               for i, (t, d, _) in enumerate(scored)])
    for row in db.list(limit=100):
        # match the score back by title (urls are u0,u1,… in insert order)
        for t, _d, sc in scored:
            if row["title"] == t:
                db.set_score(row["id"], sc)
    return db


def test_handler_empty_ledger(monkeypatch):
    db = JobLog(db_path=":memory:")
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_EMPTY"
    assert "fetch_jobs" in reply               # honest next-step hint


def test_handler_stored_but_unranked(monkeypatch):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("u1", "Software Developer Apprentice")])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_UNRANKED"
    assert "rank_jobs" in reply and "1" in reply   # mentions the stored count


def test_handler_ranked_orders_by_score_pt(monkeypatch):
    db = _ledger_with([
        ("Developer/Software Engineer Apprentice", 17.0, 0.83),
        ("NEC Software Developer Apprentice", 9.6, 0.79),
        ("Data Analyst Apprentice", 25.0, 0.80),
    ])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="minhas melhores vagas")
    assert mode == "JOBS_TOP"
    lines = reply.splitlines()
    assert lines[0] == "Suas melhores vagas agora:"   # PT header
    # best score first, then 80%, then 79%
    # title is a markdown link (s82) — see TestJobLink for the why
    assert lines[1].startswith("1. [Developer/Software Engineer Apprentice](")
    assert "83%" in lines[1] and "17 milhas" in lines[1]
    assert "80%" in lines[2] and "79%" in lines[3]


def test_handler_language_from_trigger_en(monkeypatch):
    db = _ledger_with([("Software Engineer Apprentice", None, 0.81)])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="my best jobs")
    assert mode == "JOBS_TOP"
    assert reply.splitlines()[0] == "Your best matches right now:"   # EN header
    assert "81%" in reply


def test_handler_caps_at_top_n(monkeypatch):
    db = _ledger_with([(f"Job {i}", float(i), 0.9 - i * 0.01) for i in range(6)])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="vagas pra mim")
    # header + exactly 3 job lines (the small spoken headline)
    assert len(reply.splitlines()) == 1 + 3


# --- routing through the shared registry (text AND voice) ---------------

def test_simulate_routes_to_jobs_top():
    # "vagas" = what "vagas pra mim" becomes after the dispatch courtesy-strip.
    for s in ("tem vaga nova?", "vagas pra mim", "vagas", "any new jobs"):
        match = simulate(s)
        assert match is not None and match.mode == "JOBS_TOP"
        assert match.matcher == "JOBS_TOP_RE"


def _scored_ledger_with_deadline(closing_at):
    """One ranked job carrying a given closing_at (or None)."""
    db = JobLog(db_path=":memory:")
    db.upsert([JobPosting(source="findapprenticeship", board="uk", company="X",
                          ext_id="1", title="Software Apprentice", location="Reading",
                          url="u1", description="d", posted_at=None,
                          closing_at=closing_at)])
    db.set_score(db.list()[0]["id"], 0.83)
    return db


def test_handler_speaks_deadline_when_present(monkeypatch):
    # Far-future date → a stable "fecha em N dias" phrase the assertion can pin.
    db = _scored_ledger_with_deadline("2099-09-01T23:59:59Z")
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert "fecha em" in reply                  # deadline surfaced (PT)


def test_handler_speaks_deadline_en(monkeypatch):
    db = _scored_ledger_with_deadline("2099-09-01T23:59:59Z")
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="my best jobs")
    assert "closes in" in reply                 # deadline surfaced (EN)


# --- the job link (s82) --------------------------------------------------
# The user's ask: "o Cortex podia me passar o link da vaga". The handler emits a
# MARKDOWN link, which the UI renders clickable and the speech stripper (used
# at both voice entry points) reduces to the bare title. That split is the
# contract these tests pin — the handler stays channel-agnostic.

_URL = "https://findapprenticeship.service.gov.uk/apprenticeship/123"


def _ledger_with_url(url, title="Software Developer Apprentice"):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting(url, title)])
    db.set_score(db.list()[0]["id"], 0.83)
    return db


def test_reply_carries_the_job_url_as_markdown(monkeypatch):
    db = _ledger_with_url(_URL)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_TOP"
    assert f"[Software Developer Apprentice]({_URL})" in reply


def test_spoken_reply_drops_the_url_but_keeps_the_title(monkeypatch):
    # The whole reason a markdown link is safe for a voice-first assistant.
    db = _ledger_with_url(_URL)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    spoken = strip_markdown_for_speech(reply)
    assert "http" not in spoken and "findapprenticeship" not in spoken
    assert "Software Developer Apprentice" in spoken
    assert "83%" in spoken                       # the rest survives intact


def test_no_bare_urls_reach_the_speech_layer(monkeypatch):
    # Guard for the corollary in the module contract: a BARE url would be read
    # aloud. Every url in the reply must be inside a markdown link.
    db = _ledger_with_url(_URL)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert re.sub(r"\[[^\]]+\]\([^)]+\)", "", reply).find("http") == -1


def test_urlless_row_degrades_to_plain_title(monkeypatch):
    """A row with no url must render a plain title, never "[title]()".

    The ledger enforces a non-empty url on insert, so this is unreachable
    through upsert — hence a hand-built row, the same defensiveness as
    _row_get: a hand-edited/legacy DB must not produce a dead link."""
    class _FakeLog:
        _ROW = {"title": "Software Developer Apprentice", "url": "",
                "location": "Reading", "distance": None, "score": 0.83,
                "closing_at": None}

        def list_open(self, **kw):
            return [self._ROW]

        def list(self, **kw):
            return [self._ROW]

        def count(self):
            return 1

    monkeypatch.setattr("src.integrations.jobs.get_log", _FakeLog)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_TOP"
    assert "1. Software Developer Apprentice" in reply
    assert "[" not in reply and "]()" not in reply


def test_handler_no_deadline_tail_when_absent(monkeypatch):
    db = _scored_ledger_with_deadline(None)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, _ = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert "fecha" not in reply and "closes" not in reply   # nothing appended


# --- re-rank by voice ----------------------------------------------------

from src.core.dispatch.jobs import JOBS_RERANK_RE  # noqa: E402


def test_rerank_re_matches_and_is_disjoint_from_top():
    for s in ("re-rankeia minhas vagas", "rankeia as vagas", "re-rankear vagas",
              "recalcula as vagas", "atualiza o ranking das vagas",
              "re-rank my jobs", "rank my jobs again", "re-score jobs"):
        assert JOBS_RERANK_RE.match(s), f"should match: {s!r}"
    for s in ("minhas vagas", "tem vaga nova", "vagas", "my best jobs",
              "rankeia minha playlist"):
        assert not JOBS_RERANK_RE.match(s), f"should NOT match: {s!r}"


def _ledger_ready_to_rank():
    db = JobLog(db_path=":memory:")
    db.set_resume("software engineer python", name="cv.txt")
    db.upsert([_posting("u1", "Software Apprentice")])
    return db


def test_rerank_no_resume_is_honest(monkeypatch):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("u1", "X")])               # jobs but no résumé
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_rerank(raw_text="re-rankeia minhas vagas")
    assert mode == "JOBS_RERANK_NO_RESUME" and "rank_jobs.py" in reply


def test_rerank_no_jobs_is_honest(monkeypatch):
    db = JobLog(db_path=":memory:")
    db.set_resume("cv text")
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_rerank(raw_text="rank my jobs")
    assert mode == "JOBS_RERANK_EMPTY"


def test_rerank_happy_acks_and_spawns(monkeypatch):
    db = _ledger_ready_to_rank()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    spawned = []
    monkeypatch.setattr(JobsMixin, "_spawn_rerank", lambda self: spawned.append(True))
    orch = JobsMixin()
    reply, mode = orch._handle_jobs_rerank(raw_text="re-rankeia minhas vagas")
    assert mode == "JOBS_RERANK" and "segundo plano" in reply
    assert spawned == [True]                       # background work kicked off
    assert orch._rerank_running is True            # flag set


def test_rerank_busy_does_not_double_spawn(monkeypatch):
    db = _ledger_ready_to_rank()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    spawned = []
    monkeypatch.setattr(JobsMixin, "_spawn_rerank", lambda self: spawned.append(True))
    orch = JobsMixin()
    orch._rerank_running = True                     # a run is already in flight
    reply, mode = orch._handle_jobs_rerank(raw_text="rank my jobs")
    assert mode == "JOBS_RERANK_BUSY" and spawned == []


def test_rerank_en_language_from_jobs_word(monkeypatch):
    db = _ledger_ready_to_rank()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    monkeypatch.setattr(JobsMixin, "_spawn_rerank", lambda self: None)
    reply, _ = JobsMixin()._handle_jobs_rerank(raw_text="re-rank my jobs")
    assert "background" in reply                    # EN ack (says "jobs")


def test_rerank_worker_runs_ranker_and_clears_flag(monkeypatch):
    calls = []
    monkeypatch.setattr("src.integrations.job_ranker.rank_jobs",
                        lambda log: calls.append(log) or 7)
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: "LOG")
    orch = JobsMixin()
    orch._rerank_running = True
    orch._rerank_worker()                           # synchronous call (no thread)
    assert calls == ["LOG"]                         # ranker invoked
    assert orch._rerank_running is False            # flag cleared even on success


# --- closed vacancies never reach the reply (s82) ------------------------
# The real ledger was 95/182 closed: without this, "tem vaga nova?" happily
# recommended dead pages — and now with a clickable link on each.

from datetime import datetime, timedelta  # noqa: E402


def _iso_in(days: int) -> str:
    d = datetime.now().astimezone().date() + timedelta(days=days)
    return f"{d.isoformat()}T12:00:00Z"


def _ledger_open_and_closed():
    """Closed jobs scored HIGHER than the open one — so an unfiltered handler
    would put the dead ones on top."""
    db = JobLog(db_path=":memory:")
    db.upsert([
        JobPosting(source="findapprenticeship", board="uk", company="X",
                   ext_id="1", title="Dead Job", location="Reading",
                   url="https://x/1", description="d", posted_at=None,
                   closing_at=_iso_in(-3)),
        JobPosting(source="findapprenticeship", board="uk", company="X",
                   ext_id="2", title="Live Job", location="Reading",
                   url="https://x/2", description="d", posted_at=None,
                   closing_at=_iso_in(7)),
    ])
    for row in db.list(limit=10):
        db.set_score(row["id"], 0.99 if row["title"] == "Dead Job" else 0.60)
    return db


def test_closed_job_is_not_recommended(monkeypatch):
    db = _ledger_open_and_closed()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_TOP"
    assert "Live Job" in reply
    assert "Dead Job" not in reply          # despite scoring 0.99
    assert "encerrada" not in reply


def _all_closed_ledger():
    db = JobLog(db_path=":memory:")
    db.upsert([JobPosting(source="findapprenticeship", board="uk", company="X",
                          ext_id=str(i), title=f"Old Job {i}", location="Reading",
                          url=f"https://x/{i}", description="d", posted_at=None,
                          closing_at=_iso_in(-5)) for i in range(3)])
    for row in db.list(limit=10):
        db.set_score(row["id"], 0.8)
    return db


def test_all_closed_says_so_and_points_at_a_refetch(monkeypatch):
    # A stale ledger is the common case, and the honest next step is a
    # re-FETCH — telling him to re-rank would waste his time.
    db = _all_closed_ledger()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_ALL_CLOSED"
    assert "encerraram" in reply and "fetch_jobs" in reply
    assert "3" in reply                     # how many had been ranked
    assert "rank_jobs" not in reply         # not the useful advice here


def test_all_closed_en(monkeypatch):
    db = _all_closed_ledger()
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    reply, mode = JobsMixin()._handle_jobs_top(raw_text="my best jobs")
    assert mode == "JOBS_ALL_CLOSED"
    assert "closed" in reply and "fetch_jobs" in reply


def test_empty_ledger_still_reports_empty_not_all_closed(monkeypatch):
    # Regression guard: the new branch must not swallow the older ones.
    db = JobLog(db_path=":memory:")
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    _, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_EMPTY"


def test_unranked_ledger_still_reports_unranked(monkeypatch):
    db = JobLog(db_path=":memory:")
    db.upsert([_posting("https://x/9", "Unranked Job")])
    monkeypatch.setattr("src.integrations.jobs.get_log", lambda: db)
    _, mode = JobsMixin()._handle_jobs_top(raw_text="tem vaga nova?")
    assert mode == "JOBS_UNRANKED"
