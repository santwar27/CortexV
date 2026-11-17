"""A dead owner leaves uncertainty, never permission to retry."""
import fcntl
import json
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.core import effect_store, orchestrator
from src.integrations import calendar_client, gmail_client, reminders_client
from src.tools import macos_controller as mac
from tests.test_claim_remove_failure import _turn
from tests.test_dispatch_parity import _build_real_orch

ITEMS = [("send_email", {"to": "ana@example.com", "subject": "Oi"}),
         ("create_reminder", {"title": "Comprar pão"}),
         ("mac_quit_app", {"app_name": "Notion"}), ("mac_sleep", {}),
         ("create_event", {"summary": "Reunião", "start_iso": "2026-10-20T10:00:00+01:00",
                           "end_iso": "2026-10-20T11:00:00+01:00"})]


@pytest.fixture(autouse=True)
def no_integrations(monkeypatch):
    calls = []
    for module, name in [(gmail_client, "send_email"), (calendar_client, "create_event"),
                         (reminders_client, "create_reminder")]:
        call = Mock(side_effect=AssertionError("must never execute"))
        monkeypatch.setattr(module, name, call)
        calls.append(call)
    call = Mock(side_effect=AssertionError("must never execute"))
    monkeypatch.setattr(mac, "_run", call)
    calls.append(call)
    yield
    for call in calls:
        call.assert_not_called()


def child(db, items=ITEMS[:1], state="executando"):
    code = """
import json, sys
from src.core.effect_store import EffectStore
s = EffectStore(sys.argv[1])
for action, args in json.loads(sys.argv[2]):
    e = s.create(action, args)
    assert e is not None
    if sys.argv[3] == 'executando':
        assert s.transition(e.id, 'preparando', 'executando')
print('ready', flush=True)
sys.stdin.readline()
"""
    return subprocess.Popen([sys.executable, "-c", code, str(db), json.dumps(items), state],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def crash(db, items=ITEMS[:1], state="executando"):
    p = child(db, items, state)
    try:
        assert p.stdout.readline().strip() == "ready"
        p.kill()  # No interpreter cleanup can release the owner's lock for us.
        p.communicate(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()


@pytest.mark.parametrize("state,target,detail", [
    ("executando", "desconhecido", "processo encerrado no meio da execução"),
    ("preparando", "cancelado", "processo encerrado antes de executar")])
def test_dead_process(tmp_path, state, target, detail):
    db = tmp_path / "effects.db"
    crash(db, state=state)
    store = effect_store.EffectStore(str(db))
    [before] = store.in_states([state])
    lock = tmp_path / "effect_owners" / f"{before.owner}.lock"
    assert lock.stat().st_mode & 0o777 == 0o600
    found = store.reconcile_abandoned()
    after = store.get(before.id)
    assert found == ([after] if target == "desconhecido" else [])
    assert (after.state, after.detail, after.version) == (target, detail, before.version + 1)
    assert store.history(after.id)[-1].detail == detail
    assert after.updated_at.utcoffset().total_seconds() == 0
    assert not lock.exists()
    assert store.reconcile_abandoned() == []


def test_live_owner_and_own_effects(tmp_path):
    db = tmp_path / "effects.db"
    p = child(db)
    try:
        assert p.stdout.readline().strip() == "ready"
        store = effect_store.EffectStore(str(db))
        [live] = store.in_states(["executando"])
        own = store.create("mac_sleep", {})
        assert own.owner != live.owner
        assert store.transition(own.id, "preparando", "executando")
        assert store.reconcile_abandoned() == []
        assert store.get(live.id) == live
        # Another instance in this same process must also respect the lock.
        assert effect_store.EffectStore(str(db)).reconcile_abandoned() == []
        p.communicate("exit\n", timeout=10)
        assert p.returncode == 0
        assert [e.id for e in store.reconcile_abandoned()] == [live.id]
        assert store.get(own.id).state == "executando"
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()


def test_legacy_and_restored_database(tmp_path):
    db = tmp_path / "effects.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE effects (id TEXT PRIMARY KEY NOT NULL, action TEXT NOT NULL, "
                     "args TEXT NOT NULL, origin TEXT, state TEXT NOT NULL, version INTEGER NOT NULL, "
                     "detail TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("INSERT INTO effects VALUES ('old', 'mac_sleep', '{}', NULL, 'executando', "
                     "2, NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')")
    store = effect_store.EffectStore(str(db))
    assert store.get("old").owner is None
    assert [e.id for e in store.reconcile_abandoned()] == ["old"]
    crash(db)
    [restored] = store.in_states(["executando"])
    (tmp_path / "effect_owners" / f"{restored.owner}.lock").unlink()
    assert [e.id for e in store.reconcile_abandoned()] == [restored.id]


def test_lock_failure_refuses_create(tmp_path, monkeypatch):
    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    monkeypatch.setattr(fcntl, "flock", Mock(side_effect=BlockingIOError()))
    assert store.create("mac_sleep", {}) is None
    assert store.in_states(["preparando"]) == []


def boot(tmp_path, monkeypatch, store):
    o = _build_real_orch(tmp_path)
    for cls, attr in [("LogicGate", "gatekeeper"), ("OllamaEngine", "engine_analytical"),
                      ("LocalFileReader", "reader"), ("LocalFileWriter", "writer"),
                      ("WebSearcher", "web_searcher"), ("VisionProcessor", "vision"),
                      ("StudyQueue", "study_queue"), ("StudiesReview", "studies_review"),
                      ("SpotifyPlayer", "spotify"), ("MacOSController", "macos")]:
        monkeypatch.setattr(orchestrator, cls, Mock(return_value=getattr(o, attr)))
    monkeypatch.setattr(effect_store, "store", store)
    monkeypatch.setattr(o, "_load_skill_registry", lambda: None)
    o.__init__()
    o._offer = None
    return o


def warning(lang, subset):
    labels = (["enviar um e-mail para ana@example.com \"Oi\"", 'criar um lembrete "Comprar pão"',
               "fechar um app Notion", "colocar o Mac pra dormir", 'criar um evento "Reunião"']
              if lang == "pt" else
              ['send an email to ana@example.com "Oi"', 'create a reminder "Comprar pão"',
               "quit an app Notion", "put the Mac to sleep", 'create an event "Reunião"'])
    items = "; ".join(labels[i] for i in subset[:3])
    if len(subset) > 3:
        items += f" (+{len(subset) - 3})"
    return (f"\n\n⚠️ Antes de reiniciar, eu estava no meio de: {items}. Não sei se foi feito — "
            "confere antes de pedir de novo; não vou repetir sozinho." if lang == "pt" else
            f"\n\n⚠️ Before restarting, I was in the middle of: {items}. I don't know whether it "
            "went through — please check before asking again; I won't retry on my own.")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("subset", [[0, 1, 2, 3], [3, 4]])
@pytest.mark.parametrize("preview_first", [False, True])
def test_notice_once_after_preview(tmp_path, monkeypatch, stream, lang, subset, preview_first):
    db = tmp_path / "effects.db"
    crash(db, [ITEMS[i] for i in subset])
    o = boot(tmp_path, monkeypatch, effect_store.EffectStore(str(db)))
    if preview_first:
        preview, mode = _turn(o, "manda email pra bob@example.com, assunto: Teste, corpo: Olá",
                             stream, lang)
        assert mode == "AGENT_AWAITING_CONFIRMATION" and "⚠️" not in preview
    # A normal turn discards the unrelated preview and receives the warning.
    reply, _ = _turn(o, "me explica RAG em detalhes", stream, lang)
    assert reply.endswith(warning(lang, subset))
    assert "⚠️" not in _turn(o, "me explica RAG em detalhes", stream, lang)[0]
    assert not any("⚠️" in str(call) for call in o.engine_analytical.mock_calls)


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("action,key,args,pt,en", [
    ("send_email", "subject", {"to": "ana@example.com"},
     "enviar um e-mail para ana@example.com", "send an email to ana@example.com"),
    ("send_email", "subject", {}, "enviar um e-mail", "send an email"),
    ("send_email", "to", {"subject": "Oi"}, 'enviar um e-mail "Oi"', 'send an email "Oi"'),
    ("create_event", "summary", {}, "criar um evento", "create an event"),
    ("create_reminder", "title", {}, "criar um lembrete", "create a reminder"),
    ("mac_quit_app", "app_name", {}, "fechar um app", "quit an app"),
])
def test_notice_without_detail(monkeypatch, lang, empty, action, key, args, pt, en):
    o = orchestrator.CortexOrchestrator.__new__(orchestrator.CortexOrchestrator)
    o._reply_lang = lang
    o._abandoned_effects_lock = threading.Lock()
    o._abandoned_effects = [Mock(id="notice", action=action, args={**args, **({key: ""} if empty else {})})]
    monkeypatch.setattr(effect_store.store, "claim_notices", lambda ids: ids)
    item = pt if lang == "pt" else en
    expected = (
        f"ok\n\n⚠️ Antes de reiniciar, eu estava no meio de: {item}. Não sei se foi feito — "
        "confere antes de pedir de novo; não vou repetir sozinho." if lang == "pt" else
        f"ok\n\n⚠️ Before restarting, I was in the middle of: {item}. I don't know whether it "
        "went through — please check before asking again; I won't retry on my own.")
    assert o._maybe_announce_abandoned("ok", "OFFLINE") == expected


def test_notice_concurrent_silent_and_missing_init(tmp_path, monkeypatch):
    db = tmp_path / "effects.db"
    crash(db)
    o = boot(tmp_path, monkeypatch, effect_store.EffectStore(str(db)))
    for mode in [*o._SILENT_ANNOUNCE_MODES, "AGENT_STAGE_ERROR", "AGENT_STAGE_NEW"]:
        assert o._maybe_announce_abandoned("ok", mode) == "ok"
    barrier = threading.Barrier(2)
    def turn(_):
        barrier.wait(timeout=10)
        return _turn(o, "me explica RAG em detalhes", False)[0]
    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(turn, range(2)))
    assert sum("⚠️" in reply for reply in replies) == 1
    bare = orchestrator.CortexOrchestrator.__new__(orchestrator.CortexOrchestrator)
    assert bare._maybe_announce_abandoned("ok", "OFFLINE") == "ok"


@pytest.mark.parametrize("raises", [False, True])
def test_unavailable_at_boot(tmp_path, monkeypatch, raises):
    blocked = tmp_path / "blocked"
    blocked.write_text("x")
    store = effect_store.EffectStore(str(blocked / "effects.db"))
    if raises:
        monkeypatch.setattr(store, "reconcile_abandoned", Mock(side_effect=OSError("unavailable")))
    o = boot(tmp_path, monkeypatch, store)
    assert "⚠️" not in _turn(o, "me explica RAG em detalhes", False)[0]
