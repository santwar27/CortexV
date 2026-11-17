"""Read models use recorded facts and never run the chat resumption flow."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core import focus, projects
from src.core.audit import audit
from tests.test_ui_routes import client, ui_app

KEY = {"X-Cortex-Key": "test-key-xyz"}
STAMP = datetime(2031, 2, 3, 12, tzinfo=timezone(timedelta(hours=2)))


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    p = projects.ProjectStore(str(tmp_path / "projects.db"))
    f = focus.FocusStore(str(tmp_path / "focus.db"))
    monkeypatch.setattr(projects, "store", p)
    monkeypatch.setattr(focus, "store", f)
    monkeypatch.setattr(projects, "_now", lambda: STAMP.isoformat())
    yield p, f
    for store in (p, f):
        if store._conn is not None:
            store._conn.close()


def get(client, url):
    response = client.get(url, headers=KEY)
    assert response.status_code == 200
    return response.json()


def record(obj, fields):
    result = {field: getattr(obj, field) for field in fields.split()}
    return {k: v.astimezone(timezone.utc).isoformat() if isinstance(v, datetime) else v
            for k, v in result.items()}


def project(p, active=False, conversations=None):
    return dict(record(p, "id name objective created_at updated_at"),
                active=active, conversations=conversations or [])


def test_list_and_empty(client, stores, monkeypatch):
    p, _ = stores
    assert get(client, "/ui/projects") == {"available": True, "active_id": None, "projects": []}
    a = p.create_or_get("Fictional Aurora", "Map imaginary stars")
    b = p.create_or_get("Fictional Comet")
    p.link_conversation("a2", a.id)
    p.link_conversation("a1", a.id)
    p.link_conversation("b1", b.id)
    monkeypatch.setattr(projects, "_now", lambda: (STAMP + timedelta(days=1)).isoformat())
    p.set_active(a.id)
    a = p.get(a.id)
    assert get(client, "/ui/projects") == {
        "available": True, "active_id": a.id,
        "projects": [project(a, True, ["a1", "a2"]), project(b, conversations=["b1"])]}


def test_detail_rule_and_order(client, stores):
    p, _ = stores
    a = p.create_or_get("Fictional Aurora")
    rows = [p.add_entry(a.id, kind, str(i), "voice", "a1") for i, kind in enumerate(
        ["feito", "feito", "falta", "falta", "falta", "decisao", "decisao", "proximo", "proximo"])]
    for i in (1, 2, 6, 8):
        p.close_entry(rows[i].id)
    observations = [p.observe(a.id, kind, "fictional_tool", str(i), "ui", "a1")
                    for i, kind in enumerate(["read", "written", "action"] * 8)]
    entries = [record(e, "id kind text channel conversation_id created_at closed_at")
               for e in p.where_entries(a.id)]
    result = get(client, f"/ui/projects/{a.id}")
    assert result == {"available": True, "project": project(a), "entries": entries,
                      "progress": {"last_done": entries[7], "todos": [entries[4], entries[5]],
                                   "last_decision": entries[2], "next_step": entries[1]},
                      "observations": [record(o, "id kind tool reference channel conversation_id created_at")
                                       for o in reversed(observations[-20:])]}
    assert [e["id"] for e in entries] == [r.id for r in reversed(rows)]
    assert client.get("/ui/projects/9999", headers=KEY).status_code == 404


def test_resume_precedence_and_honest_empty(client, stores):
    p, _ = stores
    assert get(client, "/ui/resume")["focus"] == {"available": True, "current": None, "last": None}
    a, b = [p.create_or_get(name) for name in ("Fictional Aurora", "Fictional Comet")]
    p.add_entry(a.id, "proximo", "Draw the next star")
    p.link_conversation("linked", a.id)
    p.set_active(b.id)
    for cid, expected, via in [("linked", a, "conversation"), ("unlinked", b, "active")]:
        result = get(client, f"/ui/resume?conversation_id={cid}")
        assert result["available"] is True
        assert result["project"] == project(expected, expected == b, ["linked"] if expected == a else [])
        assert result["project"]["id"] == p.progress_project(cid).id
        assert result["via"] == via
        assert result["progress"] == get(client, f"/ui/projects/{expected.id}")["progress"]
    p.clear_active()
    result = get(client, "/ui/resume?conversation_id=unlinked")
    assert p.progress_project("unlinked") is None
    assert result["available"] is True and result["project"] is None and result["via"] is None
    assert result["progress"] == {"last_done": None, "todos": [], "last_decision": None, "next_step": None}


@pytest.mark.parametrize("reminder,status", [("fictional-id", "created"), (focus.REMINDER_FAILED, "failed"), (None, "none"), (focus.REMINDER_UNKNOWN, "unknown")])
@pytest.mark.parametrize("expired", [False, True])
def test_focus(client, stores, reminder, status, expired):
    _, f = stores
    assert f.available() is True
    now = datetime.now(timezone(timedelta(hours=2)))
    session = f.start("Fictional task", "Open imaginary map", 25,
                      now - timedelta(hours=1) if expired else now)
    f.set_reminders(session.id, reminder, None)
    result = get(client, "/ui/resume")["focus"]
    expected = dict(record(session, "id task first_step started_at ends_at closed_at next_step"),
                    open=True, time_is_up=expired, reminder=status)
    assert result == {"available": True, "current": expected, "last": expected}
    f.stop("Draw the next star", now)
    closed = get(client, "/ui/resume")["focus"]
    assert closed["current"] is None
    assert closed["last"] == dict(expected, open=False, closed_at=now.astimezone(timezone.utc).isoformat(),
                                  next_step="Draw the next star")


def test_unavailable(client, stores, tmp_path, monkeypatch):
    p, f = stores
    monkeypatch.setattr(projects, "store", projects.ProjectStore(str(tmp_path)))
    assert get(client, "/ui/projects") == {"available": False, "active_id": None, "projects": []}
    detail = get(client, "/ui/projects/1")
    assert detail["available"] is False and detail["project"] is None
    assert detail["entries"] == detail["observations"] == []
    result = get(client, "/ui/resume")
    assert result["available"] is False and result["focus"]["available"] is True
    monkeypatch.setattr(projects, "store", p)
    monkeypatch.setattr(focus, "store", focus.FocusStore(str(tmp_path)))
    assert focus.store.available() is False
    result = get(client, "/ui/resume")
    assert result["available"] is True
    assert result["focus"] == {"available": False, "current": None, "last": None}
    assert get(client, "/ui/projects")["available"] is True
    f._connect().close()
    assert f.available() is False  # Opening a handle alone is not a successful SELECT 1.


def snapshot(store):
    conn = store._connect()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {t: [tuple(r) for r in conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid')] for t in tables}


def test_reads_never_touch_state_or_orchestrator(client, ui_app, stores):
    p, f = stores
    a = p.create_or_get("Fictional Aurora")
    p.link_conversation("linked", a.id)
    p.set_active(a.id)
    p.add_entry(a.id, "proximo", "Draw the next star")
    p.observe(a.id, "read", "fictional_tool", "imaginary.txt")
    f.start("Fictional task", "Open map", 25, datetime.now(timezone.utc))
    before = [snapshot(s) for s in stores]
    audit_path = Path(audit.path)
    audit_before = audit_path.read_bytes() if audit_path.exists() else None

    class Untouchable:
        def __getattr__(self, name):
            pytest.fail(f"Orchestrator/engine accessed: {name}")

    del ui_app.state.orchestrator
    for orchestrator in (None, Untouchable()):
        if orchestrator is not None:
            ui_app.state.orchestrator = orchestrator
        for _ in range(2):
            for url in ("/ui/projects", f"/ui/projects/{a.id}", "/ui/resume?conversation_id=linked"):
                get(client, url)
    assert [snapshot(s) for s in stores] == before
    assert (audit_path.read_bytes() if audit_path.exists() else None) == audit_before


@pytest.mark.parametrize("url", ["/ui/projects", "/ui/projects/1", "/ui/resume?conversation_id=linked"])
def test_auth(client, url):
    assert client.get(url).status_code == 401
