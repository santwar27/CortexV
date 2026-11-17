"""
EN localisation of the deterministic digest/status/entities assemblers (s60).

These pure assemble functions gained a `lang` kwarg (PT default). Tests assert
EN output when lang='en' and PT unchanged by default.
"""
from datetime import datetime, timedelta, timezone

import src.core.digest as dg
import src.core.system_status as ss
import src.core.entities as en


NOW = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


class TestDigestEN:
    def test_assemble_en(self):
        since = NOW - timedelta(hours=2)
        sources = dg.DeltaSources(
            emails=[{"from": "Ana <ana@x.com>", "subject": "Oi", "date_iso": NOW.isoformat()}],
            actions=[{"action": "send_email", "ts": NOW.isoformat()}],
        )
        out = dg.assemble(sources, since, NOW, lang="en")
        assert out.startswith("What changed")
        assert "New emails" in out
        assert "sent an email" in out

    def test_assemble_pt_default(self):
        since = NOW - timedelta(hours=2)
        sources = dg.DeltaSources(
            actions=[{"action": "send_email", "ts": NOW.isoformat()}])
        out = dg.assemble(sources, since, NOW)
        assert out.startswith("O que mudou")
        assert "enviei um e-mail" in out

    def test_empty_en(self):
        since = NOW - timedelta(hours=2)
        out = dg.assemble(dg.DeltaSources(), since, NOW, lang="en")
        assert "Nothing new" in out


class TestSystemStatusEN:
    def test_assemble_en(self):
        signals = {"jobs": {"total": 2, "stale": ["studier"]},
                   "palace_docs": 3, "gate_pending": 1}
        out = ss.assemble(signals, lang="en")
        assert "Heads up:" in out
        assert "Memory has 3 documents." in out
        assert "You have 1 action awaiting approval." in out

    def test_assemble_pt_default(self):
        signals = {"jobs": {"total": 2, "stale": []}, "palace_docs": 1, "gate_pending": 0}
        out = ss.assemble(signals)
        assert "Tudo certo por aqui" in out
        assert "1 documento." in out
        assert "Nenhuma ação pendente" in out

    def test_all_good_en(self):
        out = ss.assemble({"jobs": {"total": 1, "stale": []}, "palace_docs": None,
                           "gate_pending": 0}, lang="en")
        assert "All good here" in out


class TestEntitiesEN:
    def _email(self, frm, when):
        return {"from": frm, "date_iso": when.isoformat()}

    def test_last_contact_en(self, monkeypatch):
        monkeypatch.setattr(en, "gather",
                            lambda: ([self._email("João <joao@x.com>", NOW)], [], []))
        out = en.last_contact_reply("joão", lang="en")
        assert out.startswith("Your last interaction with João")
        assert "via email" in out

    def test_last_contact_miss_en(self, monkeypatch):
        monkeypatch.setattr(en, "gather", lambda: ([], [], []))
        out = en.last_contact_reply("ninguem", lang="en")
        assert "no recent interactions" in out

    def test_recent_people_en(self, monkeypatch):
        monkeypatch.setattr(en, "gather",
                            lambda: ([self._email("Ana <ana@x.com>", NOW)], [], []))
        out = en.recent_people_reply(lang="en")
        assert out.startswith("People you've interacted with recently:")

    def test_recent_people_pt_default(self, monkeypatch):
        monkeypatch.setattr(en, "gather", lambda: ([], [], []))
        out = en.recent_people_reply()
        assert "Não tenho interações recentes" in out
