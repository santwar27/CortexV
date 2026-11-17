"""The project record keeps only explicit words, across conversations."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest

from src.core.projects import ProjectStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CORTEX_PROJECTS_DB", str(tmp_path / ".cortex/projects.db"))
    result = ProjectStore()
    assert result._conn is None
    return result


def test_names_objective_and_order(store):
    p = store.create_or_get("  Currículo  novo ", "Enviar versão fictícia")
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert p.name == "  Currículo  novo "
    assert p.name_key == "curriculo novo"
    assert store.find(" CURRICULO\n novo ") == p == store.get(p.id)
    assert store.create_or_get("curriculo novo", "Outro objetivo") == p
    assert store.create_or_get(" \n ") is None
    q = store.create_or_get("Horta")
    assert store.list() == [q, p]
    assert store.set_objective(p.id, "Rever exemplos")
    assert store.list()[0].id == p.id
    assert store.get(p.id).objective == "Rever exemplos"
    assert store.set_objective(p.id, None)
    assert store.get(p.id).objective is None
    assert store.find("ausente") is None and store.get(-1) is None
    with pytest.raises(FrozenInstanceError):
        p.name = "changed"


def test_conversations_move_without_affecting_the_others(store):
    p, q = store.create_or_get("Horta"), store.create_or_get("Pintura")
    for cid in ("c1", "c2", "c3"):
        assert store.link_conversation(cid, p.id)
    assert set(store.conversations(p.id)) == {"c1", "c2", "c3"}
    assert store.link_conversation("c2", q.id)
    assert store.link_conversation("c2", q.id)
    assert set(store.conversations(p.id)) == {"c1", "c3"}
    assert store.conversations(q.id) == ["c2"]
    assert store.project_for_conversation("c2").id == q.id
    assert store.project_for_conversation("missing") is None
    assert not store.link_conversation("c2", -1)
    assert store.project_for_conversation("c2").id == q.id


def test_entries_are_literal_bounded_filtered_and_closed(store):
    p = store.create_or_get("Horta")
    rows = [store.add_entry(p.id, kind, "  Texto\n literal  ", channel, "c1")
            for kind, channel in zip(("feito", "falta", "proximo", "decisao"),
                                     ("ui", "voice", "terminal", "network"))]
    assert all(r.text == "  Texto\n literal  " and r.source == "user_said"
               and r.conversation_id == "c1" for r in rows)
    assert [r.channel for r in rows] == ["ui", "voice", "terminal", "network"]
    assert store.entries(p.id) == rows[::-1]
    assert store.entries(p.id, tipo="falta") == [rows[1]]
    assert store.close_entry(rows[1].id)
    assert store.entries(p.id, tipo="falta", abertas_apenas=True) == []
    assert store.entries(p.id, tipo="falta")[0].closed_at is not None
    assert not store.close_entry(rows[1].id)
    for kwargs in ({"tipo": "inventado"}, {"texto": " \n "},
                   {"canal": "model"}, {"fonte": "model"}, {"fonte": None}):
        args = dict(projeto=p.id, tipo="feito", texto="Exemplo")
        assert store.add_entry(**(args | kwargs)) is None
    clipped = store.add_entry(p.id, "feito", "á" * 501)
    assert clipped.text == "á" * 500 and clipped.channel is None
    assert store.entries(p.id, limite=1) == [clipped]
    assert store.entries(p.id, limite=0) == []
    assert len(store.entries(p.id, abertas_apenas=True)) == 4
    with pytest.raises(FrozenInstanceError):
        clipped.text = "changed"


def test_active_and_changes_are_visible_to_another_connection(store):
    other = ProjectStore(store.db_path)
    assert other.active() is None
    p = store.create_or_get("Horta")
    assert store.set_active(p.id)
    assert other.active() == store.get(p.id)
    assert ProjectStore(store.db_path).active() == store.get(p.id)
    q = other.create_or_get("Pintura")
    assert other.set_active(q.id)
    assert store.active().id == q.id
    assert not store.set_active(-1) and store.active().id == q.id
    assert other.link_conversation("c1", p.id)
    assert store.project_for_conversation("c1").id == p.id
    assert other.add_entry(p.id, "feito", "Semeei")
    assert len(store.entries(p.id)) == 1
    assert store.clear_active() and other.active() is None
    assert not store.clear_active()


def test_every_write_touches_the_affected_projects_and_uses_utc(store):
    p, q = store.create_or_get("Horta"), store.create_or_get("Pintura")
    def check(action, *ids):
        with store._conn:
            store._conn.execute("UPDATE projects SET updated_at = '2000-01-01T00:00:00+00:00'")
        assert action()
        for pid in ids:
            assert store.get(pid).updated_at.year > 2000
    check(lambda: store.set_objective(p.id, "Exemplo"), p.id)
    check(lambda: store.link_conversation("c1", p.id), p.id)
    check(lambda: store.link_conversation("c1", q.id), p.id, q.id)
    check(lambda: store.add_entry(p.id, "falta", "Regar"), p.id)
    check(lambda: store.close_entry(store.entries(p.id)[0].id), p.id)
    check(lambda: store.set_active(p.id), p.id)
    check(lambda: store.set_active(q.id), p.id, q.id)
    check(store.clear_active, q.id)
    assert store.set_active(p.id)
    for table, columns in (("projects", "created_at, updated_at"),
                           ("entries", "created_at, closed_at"),
                           ("active_project", "selected_at")):
        for row in store._conn.execute(f"SELECT {columns} FROM {table}"):
            assert all(datetime.fromisoformat(t).utcoffset() == timedelta(0)
                       for t in row if t is not None)


def test_unopenable_database_is_fail_soft(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    store = ProjectStore(str(blocker / "projects.db"))
    for call in (lambda: store.create_or_get("Horta"), lambda: store.get(1),
                 lambda: store.find("Horta"), store.active,
                 lambda: store.project_for_conversation("c1"),
                 lambda: store.add_entry(1, "feito", "Semeei")):
        assert call() is None
    assert store.list() == store.entries(1) == store.conversations(1) == []
    assert not store.set_objective(1, "Exemplo")
    assert not store.link_conversation("c1", 1)
    assert not store.close_entry(1) and not store.set_active(1)
    assert not store.clear_active()


def test_a_failure_after_inserting_rolls_back_and_releases_the_lock(store):
    p = store.create_or_get("Horta")
    with store._conn:
        store._conn.execute("""CREATE TRIGGER fail_touch BEFORE UPDATE ON projects
                              BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    assert store.add_entry(p.id, "feito", "Semeei") is None
    assert store.entries(p.id) == []
    assert not store._conn.in_transaction
    assert not store.link_conversation("c1", p.id)
    assert store.conversations(p.id) == []
    other = ProjectStore(store.db_path)
    assert other.create_or_get("Pintura") is not None
    assert store.add_entry(-1, "feito", "Órfã") is None
    assert not store.close_entry(-1) and not store.set_objective(-1, "Nada")
