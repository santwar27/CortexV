"""
Tests for src/core/entities.py — the deterministic people graph (Onda 5).

`build`/`lookup` are pure (synthetic sources). `gather` is fail-soft (mocked).
The dispatch regexes are sanity-checked here; full lock is in evals.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.core import entities as en
from src.core.dispatch.entities import LAST_CONTACT_RE, RECENT_PEOPLE_RE


NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


class _C:
    def __init__(self, name, email):
        self.name, self.email = name, email


def _email(frm, when):
    return {"from": frm, "date_iso": when.isoformat()}


# --- build (pure) ----------------------------------------------------------

def test_build_from_email_from_header():
    ents = en.build([_email("João Silva <joao@x.com>", NOW)], [], [])
    e = ents["joao@x.com"]
    assert e.name == "João Silva" and e.email == "joao@x.com"
    assert e.count == 1 and "e-mail" in e.sources


def test_build_aggregates_last_seen_and_count():
    ents = en.build([
        _email("João <joao@x.com>", NOW - timedelta(days=2)),
        _email("João <joao@x.com>", NOW),
    ], [], [])
    e = ents["joao@x.com"]
    assert e.count == 2
    assert e.last_seen == NOW   # most recent wins


def test_build_merges_email_and_calendar():
    emails = [_email("Ana <ana@x.com>", NOW - timedelta(days=1))]
    events = [{"attendees": ["ana@x.com"], "start": NOW.isoformat()}]
    ents = en.build(emails, events, [])
    e = ents["ana@x.com"]
    assert e.count == 2
    assert e.sources == {"e-mail", "agenda"}
    assert e.last_seen == NOW   # the event is more recent


def test_build_names_calendar_attendee_from_contacts():
    events = [{"attendees": ["bob@x.com"], "start": NOW.isoformat()}]
    ents = en.build([], events, [_C("Bob Marley", "bob@x.com")])
    assert ents["bob@x.com"].name == "Bob Marley"


def test_build_upgrades_placeholder_name():
    # email with no display name, then a contact gives the name via calendar
    emails = [_email("zoe@x.com", NOW - timedelta(days=1))]
    events = [{"attendees": ["zoe@x.com"], "start": NOW.isoformat()}]
    ents = en.build(emails, events, [_C("Zoe Q", "zoe@x.com")])
    assert ents["zoe@x.com"].name == "Zoe Q"


# --- non-person filter -----------------------------------------------------

def test_build_skips_automated_senders():
    ents = en.build([
        _email("No Reply <noreply@x.com>", NOW),
        _email("LinkedIn Job Alerts <jobs-listings@linkedin.com>", NOW),
        _email("João <joao@x.com>", NOW),
    ], [], [])
    keys = set(ents)
    assert "joao@x.com" in keys
    assert "noreply@x.com" not in keys
    assert "jobs-listings@linkedin.com" not in keys


@pytest.mark.parametrize("addr,is_person", [
    ("joao@x.com", True),
    ("maria.silva@gmail.com", True),
    ("noreply@x.com", False),
    ("no-reply@x.com", False),
    ("notifications@github.com", False),
    ("jobs-listings@linkedin.com", False),
    ("alerts@bank.com", False),
    # M7-A8: ambiguous local-parts are PEOPLE until proven otherwise —
    # team@startup.com / news@jornalista.com are real humans at small
    # domains; the old list silenced them.
    ("team@startup.com", True),
    ("news@jornalista.com", True),
    ("info@pousada.com.br", True),
    ("hello@founder.io", True),
    ("support@consultor.dev", True),
    ("contact@artista.com", True),
])
def test_is_person_addr(addr, is_person):
    assert en._is_person_addr(addr) is is_person


def test_saved_contact_outranks_nonperson_filter():
    # M7-A8 lado 2: o usuário salvou notifications@empresa.com como contato
    # → é pessoa por decreto do usuário, o filtro de local-part não derruba.
    ents = en.build(
        [_email("Empresa <notifications@empresa.com>", NOW)],
        [],
        [_C("Rê da Empresa", "notifications@empresa.com")],
    )
    assert "notifications@empresa.com" in ents
    # (nome de display do header ganha do contact book — precedência
    # existente, fora do escopo do M7-A8; o ponto aqui é NÃO filtrar)
    assert ents["notifications@empresa.com"].name == "Empresa"


def test_unsaved_nonperson_still_filtered():
    ents = en.build(
        [_email("GitHub <notifications@github.com>", NOW)], [], [])
    assert ents == {}


# --- lookup ----------------------------------------------------------------

def test_lookup_by_name_substring():
    ents = en.build([_email("João Silva <joao@x.com>", NOW)], [], [])
    assert en.lookup("joão", ents).email == "joao@x.com"
    assert en.lookup("silva", ents).email == "joao@x.com"


def test_lookup_by_email():
    ents = en.build([_email("X <a@b.com>", NOW)], [], [])
    assert en.lookup("a@b.com", ents) is not None


def test_lookup_miss_is_none():
    ents = en.build([_email("X <a@b.com>", NOW)], [], [])
    assert en.lookup("ninguém", ents) is None


def test_lookup_prefers_most_recent_match():
    ents = en.build([
        _email("João Antigo <old@x.com>", NOW - timedelta(days=10)),
        _email("João Novo <new@x.com>", NOW),
    ], [], [])
    assert en.lookup("joão", ents).email == "new@x.com"


# --- replies (gather mocked) -----------------------------------------------

def test_last_contact_reply(monkeypatch):
    monkeypatch.setattr(en, "gather",
                        lambda: ([_email("João <joao@x.com>", NOW)], [], []))
    out = en.last_contact_reply("joão")
    assert "João" in out and "última interação" in out.lower()


def test_last_contact_reply_strips_article(monkeypatch):
    monkeypatch.setattr(en, "gather",
                        lambda: ([_email("João <joao@x.com>", NOW)], [], []))
    # "o João" must still match the João entity.
    assert "João" in en.last_contact_reply("o João")


def test_last_contact_reply_miss(monkeypatch):
    monkeypatch.setattr(en, "gather", lambda: ([], [], []))
    assert "Não achei" in en.last_contact_reply("Fulano")


def test_recent_people_reply(monkeypatch):
    monkeypatch.setattr(en, "gather", lambda: (
        [_email("Ana <ana@x.com>", NOW), _email("Bob <bob@x.com>", NOW - timedelta(days=1))],
        [], [],
    ))
    out = en.recent_people_reply()
    assert "Ana" in out and "Bob" in out
    assert out.index("Ana") < out.index("Bob")   # most recent first
    assert "|" not in out


def test_recent_people_empty(monkeypatch):
    monkeypatch.setattr(en, "gather", lambda: ([], [], []))
    assert "Não tenho interações" in en.recent_people_reply()


# --- graph cache (M7-A5) ----------------------------------------------------

def test_replies_share_one_gather_within_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_gather():
        calls["n"] += 1
        return ([_email("Ana <ana@x.com>", NOW)], [], [])

    monkeypatch.setattr(en, "gather", counting_gather)
    en.last_contact_reply("ana")
    en.recent_people_reply()
    en.last_contact_reply("ana")
    assert calls["n"] == 1, "graph must be reused within the TTL"


def test_cache_expires_after_ttl(monkeypatch):
    monkeypatch.setenv("CORTEX_ENTITIES_TTL_S", "0")
    calls = {"n": 0}

    def counting_gather():
        calls["n"] += 1
        return ([_email("Ana <ana@x.com>", NOW)], [], [])

    monkeypatch.setattr(en, "gather", counting_gather)
    en.recent_people_reply()
    en.recent_people_reply()
    assert calls["n"] == 2, "TTL 0 → every call refetches"


def test_empty_graph_is_not_cached(monkeypatch):
    # gather is fail-soft: an empty result usually means the sources
    # failed transiently — pinning it would turn a blip into a 5-min lie.
    calls = {"n": 0}
    results = [([], [], []), ([_email("Ana <ana@x.com>", NOW)], [], [])]

    def flaky_gather():
        calls["n"] += 1
        return results[min(calls["n"] - 1, 1)]

    monkeypatch.setattr(en, "gather", flaky_gather)
    assert "Não tenho interações" in en.recent_people_reply()
    assert "Ana" in en.recent_people_reply()   # retried, not pinned empty
    assert calls["n"] == 2


def test_invalidate_cache_forces_refetch(monkeypatch):
    calls = {"n": 0}

    def counting_gather():
        calls["n"] += 1
        return ([_email("Ana <ana@x.com>", NOW)], [], [])

    monkeypatch.setattr(en, "gather", counting_gather)
    en.recent_people_reply()
    en.invalidate_cache()
    en.recent_people_reply()
    assert calls["n"] == 2


def test_gather_is_fail_soft(monkeypatch):
    import src.integrations.gmail_client as gmail
    monkeypatch.setattr(gmail, "list_recent", lambda count=50: (_ for _ in ()).throw(RuntimeError("x")))
    emails, events, contacts = en.gather()   # must not raise
    assert emails == []


# --- dispatch regexes ------------------------------------------------------

@pytest.mark.parametrize("text,name", [
    ("quando falei com João pela última vez", "João"),
    ("quando foi a última vez que falei com Maria", "Maria"),
    ("última vez que conversei com o Pedro", "o Pedro"),
    ("when did I last talk to Sarah", "Sarah"),
])
def test_last_contact_re_captures(text, name):
    m = LAST_CONTACT_RE.match(text)
    assert m and m.group("name") == name


@pytest.mark.parametrize("text", [
    "com quem andei falando",
    "com quem eu falei",
    "pessoas recentes",
    "recent contacts",
])
def test_recent_people_re_matches(text):
    assert RECENT_PEOPLE_RE.match(text)


@pytest.mark.parametrize("text", [
    "fala com o João pra mim",        # imperative, not a query
    "quero falar com a Maria amanhã",
    "oi cortex",
])
def test_entity_res_ignore_non_queries(text):
    assert not LAST_CONTACT_RE.match(text)
    assert not RECENT_PEOPLE_RE.match(text)


# --- handler integration ---------------------------------------------------

def test_handlers(monkeypatch):
    from src.core.orchestrator import CortexOrchestrator
    monkeypatch.setattr(en, "last_contact_reply", lambda q, lang="pt": f"last:{q}")
    monkeypatch.setattr(en, "recent_people_reply", lambda lang="pt": "recent")
    o = CortexOrchestrator.__new__(CortexOrchestrator)
    assert o._handle_last_contact("João") == ("last:João", "ENTITY_LAST_CONTACT")
    assert o._handle_recent_people() == ("recent", "ENTITY_RECENT_PEOPLE")



# --- an upcoming meeting is not a past interaction (s88, Codex lote 13) -----

def test_a_future_meeting_is_not_the_last_interaction():
    """gather read the calendar from now to now+7d and build folded those
    starts into last_seen: "sua última interação com Ana foi em 22/10" on
    18/10, for next week's meeting."""
    now = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)
    emails = [_email("Ana <ana@x.com>", now - timedelta(days=3))]
    events = [{"attendees": ["ana@x.com"], "start": (now + timedelta(days=4)).isoformat()}]
    e = en.build(emails, events, [], now=now)["ana@x.com"]
    assert e.last_seen == now - timedelta(days=3)
    assert e.count == 1 and e.sources == {"e-mail"}


def test_a_meeting_already_held_still_counts():
    now = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)
    events = [{"attendees": ["ana@x.com"], "start": (now - timedelta(days=1)).isoformat()}]
    e = en.build([], events, [], now=now)["ana@x.com"]
    assert e.last_seen == now - timedelta(days=1) and e.sources == {"agenda"}


def test_gather_reads_the_calendar_backwards(monkeypatch):
    import src.integrations.calendar_client as cal
    import src.integrations.gmail_client as gmail
    asked = {}

    def list_events(time_min=None, time_max=None, max_results=10):
        asked.update(time_min=time_min, time_max=time_max)
        return []
    monkeypatch.setattr(cal, "list_events", list_events)
    monkeypatch.setattr(gmail, "list_recent", lambda count=50, query=None: [])
    before = datetime.now(timezone.utc)
    en.gather()
    tmin = datetime.fromisoformat(asked["time_min"])
    tmax = datetime.fromisoformat(asked["time_max"])
    assert tmax <= datetime.now(timezone.utc) and tmax >= before
    assert timedelta(days=13) < tmax - tmin <= timedelta(days=14, seconds=1)



@pytest.mark.parametrize("addr,is_person", [
    ("job@empresa.com", True),               # a first name (s88, Codex lote 13)
    ("jobs@empresa.com", False),
    ("jobs-listings@linkedin.com", False),
    ("jobalerts-noreply@linkedin.com", False),
    ("job-alerts@indeed.com", False),
    ("job.notifications@x.com", False),
])
def test_job_is_a_name_jobs_is_a_board(addr, is_person):
    assert en._is_person_addr(addr) is is_person


def test_a_person_called_job_is_found():
    g = en.build([_email("Job <job@empresa.com>", NOW)], [], [])
    assert en.lookup("Job", g).email == "job@empresa.com"



class TestASourceThatFailedIsSaid:
    """Source failures were swallowed: Gmail down made "não achei interações
    com a Ana", and the partial graph was cached for five minutes, so the
    same answer came back after Gmail recovered (s88, Codex lote 13 —
    invariant 7)."""

    @pytest.fixture
    def world(self, monkeypatch):
        import src.integrations.gmail_client as gmail
        import src.integrations.calendar_client as cal
        import src.integrations.contacts as contacts
        now = datetime.now(timezone.utc)
        state = {"down": True}

        def list_recent(count=50, query=None):
            if state["down"]:
                raise RuntimeError("HTTP 503")
            return [_email("Ana <ana@x.com>", now - timedelta(days=1))]
        monkeypatch.setattr(gmail, "list_recent", list_recent)
        monkeypatch.setattr(cal, "list_events", lambda **k: [
            {"start": (now - timedelta(days=2)).isoformat(), "attendees": ["bob@x.com"]}])
        monkeypatch.setattr(contacts.ContactBook, "list_all", lambda self: [])
        en.invalidate_cache()
        yield state
        en.invalidate_cache()

    def test_a_miss_says_what_was_not_read(self, world):
        out = en.last_contact_reply("Ana")
        assert "não consegui ler o e-mail agora" in out
        assert "no e-mail ou na agenda" not in out

    def test_the_partial_graph_is_not_cached(self, world):
        en.last_contact_reply("Ana")
        world["down"] = False
        out = en.last_contact_reply("Ana")
        assert "Ana" in out and "última interação" in out
        assert "não consegui" not in out

    def test_a_hit_and_the_list_carry_the_note(self, world):
        assert "não consegui ler o e-mail agora" in en.last_contact_reply("bob")
        assert "Lista incompleta" in en.recent_people_reply()

    def test_gather_names_what_failed(self, world):
        assert en.gather().unavailable == ["e-mail"]
