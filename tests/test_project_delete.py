"""Project erasure uses temporary stores and the running engine's Palace only."""
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from src.core import projects
from src.core.audit import audit
from src.core.file_capability import FILE_CAPABILITY_REPLY
from src.core.memory import CortexMemory
from src.core.prompt_safety import OPEN, CLOSE
from src.memory.palace import Palace
from src.ui import routes
from src.ui.chat_store import ChatStore
from tests.test_ui_routes import client, ui_app  # noqa: F401
from tests.ui_parity import Document

ROOT = Path(__file__).resolve().parents[1]
KEY = {"X-Cortex-Key": "test-key-xyz"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = projects.ProjectStore(str(tmp_path / "projects.db"))
    monkeypatch.setattr(projects, "store", s)
    yield s
    if s._conn is not None:
        s._conn.close()


def populate(s):
    a, b = [s.create_or_get(name) for name in ("Finished", "Other")]
    for project, cid in ((a, "a"), (a, "b"), (b, "c")):
        s.link_conversation(cid, project.id)
        s.add_entry(project.id, "falta", cid)
        s.observe(project.id, "read", "read_file", cid)
    return a, b


@pytest.mark.parametrize("active_target", [True, False])
def test_store_removes_all_rows_and_only_its_active_slot(store, active_target):
    a, b = populate(store)
    store.set_active(a.id if active_target else b.id)
    b = store.get(b.id)
    assert store.delete_project(a.id) == ["a", "b"]
    assert store.get(a.id) is None
    for table in ("entries", "observations", "conversation_projects", "active_project"):
        assert not store._conn.execute(
            f"SELECT * FROM {table} WHERE project_id = ?", (a.id,)).fetchall()
    assert store.get(b.id) == b
    assert len(store.entries(b.id)) == len(store.observations(b.id)) == 1
    assert store.conversations(b.id) == ["c"]
    assert store.active() == (None if active_target else b)
    assert store.create_or_get("Finished").id != b.id


def test_store_unknown_unavailable_and_empty_project(store, tmp_path):
    assert store.delete_project(999) is None
    empty = store.create_or_get("Empty")
    assert store.delete_project(empty.id) == []
    assert projects.ProjectStore(str(tmp_path)).delete_project(1) is None


def test_store_failure_rolls_back_every_table(store):
    a, _ = populate(store)
    store.set_active(a.id)
    tables = ("projects", "entries", "observations", "conversation_projects", "active_project")
    def snapshot():
        return {t: [tuple(r) for r in store._conn.execute(f"SELECT * FROM {t}")] for t in tables}
    before = snapshot()
    store._conn.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON projects "
                        "BEGIN SELECT RAISE(ABORT, 'injected'); END")
    assert store.delete_project(a.id) is None
    assert snapshot() == before


@pytest.fixture
def memory(tmp_path):
    p = Palace(db_path=str(tmp_path / "palace.db"), embed_fn=None)
    yield CortexMemory(palace=p)
    p.close()


def add(memory, text, tipo="conversation"):
    return memory._palace.add_document(tipo=tipo, texto=text)


def test_memory_matching_is_exact_typed_deduplicated_and_applies_own_words(memory):
    user = f"{OPEN}\nexternal words\n{CLOSE}\nPERGUNTA: my question"
    turns = [{"user": user, "reply": "Ok."}, {"user": "old", "reply": "L" * 80}]
    exact = add(memory, "User: my question\nCortex: Ok.")
    fallback = add(memory, "User: different user\nCortex: " + "L" * 80)
    keep = [add(memory, "User: someone else\nCortex: Ok."),
            add(memory, "User: different\nCortex: " + "S" * 79),
            add(memory, "User: different\nCortex: " + "L" * 80 + "\n"),
            add(memory, "User: my question\nCortex: Ok. ")]
    for tipo in ("daily_summary", "knowledge", "study", "skill"):
        keep.append(add(memory, "User: my question\nCortex: Ok.", tipo))
        keep.append(add(memory, "User: different\nCortex: " + "L" * 80, tipo))
    # No default list_documents(limit=100) truncation may hide older matches.
    for i in range(105):
        add(memory, f"unrelated {i}")
    turns += [{"user": "old", "reply": "S" * 79}, *turns]
    assert set(memory.conversation_memory_ids(turns)) == {exact, fallback}
    assert memory.conversation_memory_ids([]) == []
    for doc_id in memory.conversation_memory_ids(turns):
        assert memory._palace.delete_document(doc_id)
    assert all(memory._palace.get_document(i) for i in keep)
    assert memory._palace.execute_read(
        "SELECT rowid FROM documentos_fts WHERE documentos_fts MATCH 'question'")
    assert memory._palace.execute_read(
        "SELECT rowid FROM documentos_fts WHERE documentos_fts MATCH 'question' "
        "AND rowid = ?", (exact,)) == []


def test_memory_unique_long_reply_matches_different_user_words(memory):
    reply = "A unique answer about this project's next steps. " * 3
    fallback = add(memory, f"User: stored words\nCortex: {reply}")
    # Other document types do not make a conversation reply ambiguous.
    add(memory, f"User: other words\nCortex: {reply}", "knowledge")
    assert memory.conversation_memory_ids([
        {"user": "original words", "reply": reply},
    ]) == [fallback]


@pytest.mark.parametrize("has_exact", [False, True])
def test_memory_repeated_long_reply_never_matches_other_user_words(memory, has_exact):
    reply = FILE_CAPABILITY_REPLY["pt"]
    assert len(reply) >= 80
    exact = add(memory, f"User: own words\nCortex: {reply}") if has_exact else None
    add(memory, f"User: other conversation\nCortex: {reply}")
    if not has_exact:
        add(memory, f"User: another conversation\nCortex: {reply}")
    assert memory.conversation_memory_ids([
        {"user": "own words", "reply": reply},
    ]) == ([exact] if has_exact else [])


def test_memory_turn_with_exact_match_does_not_use_reply_fallback(memory):
    # A delimiter inside the user's words must not obscure the exact match.
    user = "question\nCortex: quoted text"
    reply = "A detailed answer that occurs only once as a parsed reply. " * 2
    exact = add(memory, f"User: {user}\nCortex: {reply}")
    add(memory, f"User: different words\nCortex: {reply}")
    assert memory.conversation_memory_ids([{"user": user, "reply": reply}]) == [exact]


def test_project_delete_preserves_other_conversations_canned_reply(client, store, memory, ui_app):
    ui_app.state.orchestrator.engine_analytical = SimpleNamespace(memory=memory)
    reply = FILE_CAPABILITY_REPLY["pt"]
    p = memory._palace
    cids, mids = [], []
    projects_to_delete = [store.create_or_get(name) for name in ("Finished", "Other")]
    for project, user in zip(projects_to_delete, ("save my code", "can you save files?")):
        cid = routes.chat_store.create_conversation()["id"]
        routes.chat_store.append_turn(cid, user, reply)
        store.link_conversation(cid, project.id)
        cids.append(cid)
        mids.append(add(memory, f"User: {user}\nCortex: {reply}"))

    url = f"/ui/projects/{projects_to_delete[0].id}"
    assert client.get(url + "/delete-preview", headers=KEY).json() == {
        "name": "Finished", "conversations": 1, "memories": 1}
    assert all(p.get_document(mid) for mid in mids)
    assert client.post(url + "/delete", headers=KEY).json() == {
        "ok": True, "conversations": 1, "memories": 1}
    assert p.get_document(mids[0]) is None
    assert p.get_document(mids[1]) is not None
    assert routes.chat_store.get_conversation(cids[0]) is None
    assert routes.chat_store.get_conversation(cids[1]) is not None
    assert routes.chat_store.get_turns(cids[1])[0]["reply"] == reply
    assert store.get(projects_to_delete[1].id) is not None


@pytest.fixture
def deletion(ui_app, store, memory):
    ui_app.state.orchestrator.engine_analytical = SimpleNamespace(memory=memory)
    a, b = [store.create_or_get(name) for name in ("Finished", "Other")]
    cids, mids = [], []
    for i in range(3):
        cid = routes.chat_store.create_conversation()["id"]
        user, reply = f"question {i}", f"reply {i}"
        routes.chat_store.append_turn(cid, user, reply)
        mid = add(memory, f"User: {user}\nCortex: {reply}")
        store.link_conversation(cid, a.id if i < 2 else b.id)
        cids.append(cid)
        mids.append(mid)
    store.link_conversation("already-deleted", a.id)
    store.add_entry(a.id, "feito", "Finished")
    store.observe(a.id, "written", "write_document", "saved.txt")
    store.set_active(a.id)
    return a, b, cids, mids


def test_preview_and_delete_counts_and_preservation(client, deletion, store, memory, tmp_path):
    a, b, cids, mids = deletion
    saved = tmp_path / "saved.txt"
    saved.write_text("Keep saved files")
    audit_path = Path(audit.path)
    audit_path.write_text('existing audit log\n')
    p = memory._palace
    p.add_fact("Pedro", "likes", "subject")
    p.log_event("test", "keep")
    before = {t: [tuple(r) for r in p.execute_read(f"SELECT * FROM {t}")]
              for t in ("triplas", "eventos")}
    assert client.get(f"/ui/projects/{a.id}/delete-preview", headers=KEY).json() == {
        "name": "Finished", "conversations": 2, "memories": 2}
    assert all(p.get_document(mid) for mid in mids)
    response = client.post(f"/ui/projects/{a.id}/delete", headers=KEY)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "conversations": 2, "memories": 2}
    assert store.get(a.id) is store.active() is None
    assert store.get(b.id) is not None
    assert all(routes.chat_store.get_conversation(cid) is None for cid in cids[:2])
    assert all(routes.chat_store.get_turns(cid) == [] for cid in cids[:2])
    assert routes.chat_store.get_conversation(cids[2]) is not None
    assert [p.get_document(mid) is None for mid in mids] == [True, True, False]
    assert before == {t: [tuple(r) for r in p.execute_read(f"SELECT * FROM {t}")] for t in before}
    assert saved.read_text() == "Keep saved files"
    assert audit_path.read_text() == 'existing audit log\n'


@pytest.mark.parametrize("method,suffix", [("get", "delete-preview"), ("post", "delete")])
def test_auth_and_unknown(client, method, suffix):
    url = f"/ui/projects/999/{suffix}"
    assert getattr(client, method)(url).status_code == 401
    assert getattr(client, method)(url, headers=KEY).status_code == 404


@pytest.mark.parametrize("stage,removed_memories,removed_conversations", [
    ("memory", 1, 0), ("chat", 2, 1), ("project", 2, 2), ("collect", 0, 0),
])
def test_midway_failure_reports_only_completed_deletes_and_allows_retry(
        client, deletion, store, memory, monkeypatch, stage, removed_memories, removed_conversations):
    a, _, cids, mids = deletion
    target, method = {"memory": (memory._palace, "delete_document"),
                      "chat": (routes.chat_store, "delete_conversation"),
                      "project": (store, "delete_project"),
                      "collect": (routes.chat_store, "get_turns")}[stage]
    original = getattr(target, method)
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if stage in ("project", "collect") or calls == 2:
            if stage == "collect":
                raise sqlite3.OperationalError("injected read failure")
            return None if stage == "project" else False
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail)
        result = client.post(f"/ui/projects/{a.id}/delete", headers=KEY).json()
    assert result == {"ok": False, "memories": removed_memories, "conversations": removed_conversations}
    assert store.get(a.id) is not None
    assert sum(memory._palace.get_document(mid) is None for mid in mids) == removed_memories
    assert sum(routes.chat_store.get_conversation(cid) is None for cid in cids) == removed_conversations
    assert client.post(f"/ui/projects/{a.id}/delete", headers=KEY).json()["ok"] is True


@pytest.mark.parametrize("method", ["get_conversation", "get_turns"])
def test_strict_chat_reads_never_hide_database_failure(tmp_path, method):
    s = ChatStore(str(tmp_path))
    with pytest.raises(sqlite3.Error):
        getattr(s, method)("a", strict=True)
    assert getattr(s, method)("a") in (None, [])


@pytest.mark.parametrize("failure", ["links", "chat", "memory"])
def test_preview_failure_and_delete_fail_closed(client, deletion, store, ui_app, monkeypatch, failure):
    a, _, cids, mids = deletion
    if failure == "links":
        monkeypatch.setattr(store, "conversations", lambda *a, **k: None)
    elif failure == "chat":
        routes.chat_store._connect().close()
    else:
        del ui_app.state.orchestrator.engine_analytical.memory
    assert client.get(f"/ui/projects/{a.id}/delete-preview", headers=KEY).status_code == 503
    assert client.post(f"/ui/projects/{a.id}/delete", headers=KEY).json() == {
        "ok": False, "conversations": 0, "memories": 0}
    assert store.get(a.id) is not None


def test_strict_project_links_distinguish_read_failure(tmp_path):
    s = projects.ProjectStore(str(tmp_path))
    assert s.conversations(1) == []
    assert s.conversations(1, strict=True) is None


def test_project_delete_button_and_wire(client):
    doc = Document(client.get("/ui/novo", headers=KEY).text)
    button = doc.select("#delete-project")
    assert len(button) == 1 and button[0].text() == "Excluir projeto"
    assert button[0] in doc.select("#resume-card")[0].descendants()
    source = (ROOT / "src/ui/static/novo/projetos.js").read_text()
    assert "$('#delete-project').addEventListener('click', removeProject);" in source
    assert "request('/ui/projects'), request('/ui/conversations')" in source
    css = (ROOT / "src/ui/static/novo/secundario.css").read_text()
    assert '.resume-card { flex-wrap: wrap; }' in css
    assert re.search(r'#delete-project \{[^}]*min-height: 44px;', css)


@pytest.mark.parametrize("n,m,active,ok,confirmed,failure", [
    (1, 1, "linked", True, True, None), (2, 3, "other", True, True, None),
    (0, 0, "linked", False, True, None), (1, 2, "linked", True, False, None),
    (2, 3, "linked", False, True, "partial"), (2, 3, "linked", False, True, "preview"),
    (2, 3, "linked", False, True, "post"),
])
def test_project_delete_ui_behavior(n, m, active, ok, confirmed, failure):
    source = (ROOT / "src/ui/static/novo/projetos.js").read_text()
    match = re.search(r"async function removeProject\(\) \{\n.*?\n\}", source, re.S)
    assert match, "Missing project-delete handler"
    script = """const assert = require('node:assert/strict');
const events = [], button = {};
const $ = () => button;
const chat = {activeId: ACTIVE, inFlight: false, newConversation() { events.push('new'); }};
const cardProject = {id: 7, conversations: ['linked']};
const toast = text => events.push(text);
let deletingProject = false;
const refresh = async () => events.push('refresh');
const request = async (url, body) => {
  events.push(url);
  if (FAILURE === 'preview' && url.endsWith('delete-preview') || FAILURE === 'post' && body)
    throw new Error('HTTP 503');
  if (url === '/ui/conversations/linked') throw {status: 404};
  return body ? {ok: OK, conversations: FAILURE === 'partial' ? 1 : 0} : {name: 'Example', conversations: N, memories: M};
};
const confirm = message => {
  assert.equal(message, EXPECTED);
  events.push('confirm'); return CONFIRMED;
};
HANDLER
(async () => {
  await removeProject();
  assert.equal(events[0], '/ui/projects/7/delete-preview');
  if (!CONFIRMED) { assert.ok(!events.includes('/ui/projects/7/delete')); return; }
  if (FAILURE !== 'preview') assert.ok(events.indexOf('confirm') < events.indexOf('/ui/projects/7/delete'));
  assert.ok(events.includes(OK ? 'Projeto excluído.' : 'Não consegui excluir o projeto.'));
  assert.ok(events.includes('refresh'));
  assert.equal(events.includes('new'), (OK || FAILURE === 'partial') && ACTIVE === 'linked');
  assert.equal(button.disabled, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    expected = (f'Excluir o projeto “Example”? Isso apaga {n} '
                f'{"conversa" if n == 1 else "conversas"}, as anotações e {m} '
                f'{"memória" if m == 1 else "memórias"} dessas conversas. '
                'Os arquivos salvos ficam. Não tem volta.')
    values = dict(ACTIVE=active, OK=ok, N=n, M=m, CONFIRMED=confirmed, EXPECTED=expected, FAILURE=failure)
    script = re.sub(r"\b(ACTIVE|OK|N|M|CONFIRMED|EXPECTED|FAILURE)\b", lambda x: json.dumps(values[x[0]]), script)
    script = script.replace("HANDLER", match[0])
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def project_ui():
    """Run the shipped module with a small DOM and controllable HTTP responses."""
    source = (ROOT / "src/ui/static/novo/projetos.js").read_text()
    source = source.replace("window.cortexProjetos =", """
window.projectState = () => ({selected, cardProject});
window.cortexProjetos =""")
    harness = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.listeners = {};
    this.dataset = new Proxy({}, {set(target, key, value) { target[key] = String(value); return true; }});
    this.textContent = ''; this.className = ''; this.hidden = false;
    this.classList = {
      toggle: (name, enabled) => {
        const classes = new Set(this.className.split(' ').filter(Boolean));
        if (enabled) classes.add(name); else classes.delete(name);
        this.className = [...classes].join(' ');
      },
      add: name => this.classList.toggle(name, true),
      remove: name => this.classList.toggle(name, false),
      contains: name => this.className.split(' ').includes(name)
    };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute() {}
  addEventListener(type, callback) { this.listeners[type] = callback; }
  click() { return this.listeners.click(); }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [child, ...child.querySelectorAll(selector)])
      .filter(child => selector.startsWith('.') ? child.classList.contains(selector.slice(1)) : child.tag === selector);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const hosts = Object.fromEntries([
  '.nav-section:not(.recent)', '.nav-section.recent', '#projects-panel', '#resume-card',
  '.resume-card .tag', '.resume-card h2', '.resume-copy p', '#breadcrumb', '#toast', '#delete-project'
].map(selector => [selector, new Element('div')]));
for (const selector of ['.nav-section:not(.recent)', '.nav-section.recent', '#projects-panel'])
  hosts[selector].append(new Element('h2'));
const listeners = {};
const document = {
  hidden: false,
  querySelector: selector => hosts[selector],
  createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag),
  createTextNode: text => Object.assign(new Element('#text'), {textContent: text}),
  addEventListener: (type, callback) => { listeners[type] = callback; },
  dispatchEvent: event => listeners[event.type]?.(event)
};
const chat = {
  activeId: null, inFlight: false,
  open(id) { this.activeId = id; document.dispatchEvent({type: 'cortex-conversation'}); return true; },
  newConversation() { this.activeId = null; document.dispatchEvent({type: 'cortex-conversation'}); }
};
const window = {cortexConversa: chat};
const setTimeout = () => 1, clearTimeout = () => {};
const projects = [
  {id: 7, name: 'Tese', conversations: ['thesis'], active: false},
  {id: 8, name: 'Viagem', conversations: ['trip'], active: true}
];
let activeProject = projects[1], detailStatus = 200, previewGate, resumeGate;
const requests = [], confirms = [];
const fetch = async (url, options) => {
  requests.push({url, method: options.method});
  let data, status = 200;
  if (url === '/ui/projects') data = {projects, active_id: activeProject.id};
  else if (url === '/ui/conversations') data = {conversations: []};
  else if (url.startsWith('/ui/resume')) {
    if (resumeGate) await resumeGate;
    data = {project: activeProject};
  }
  else if (url.endsWith('/delete-preview')) {
    if (previewGate) await previewGate;
    data = {name: 'Tese (preview)', conversations: 1, memories: 2};
  } else if (url.endsWith('/delete')) {
    data = {ok: true, conversations: 1}; detailStatus = 404;
  } else {
    const project = projects.find(project => url === '/ui/projects/' + project.id);
    assert.ok(project, 'Unexpected request: ' + url);
    status = project.id === 7 ? detailStatus : 200;
    data = {project};
  }
  return {ok: status === 200, status, json: async () => data};
};
const confirm = message => { confirms.push(message); return true; };
const settle = async () => { for (let i = 0; i < 3; i++) await new Promise(setImmediate); };
const visibility = async () => {
  document.hidden = true; document.dispatchEvent({type: 'visibilitychange'});
  document.hidden = false; document.dispatchEvent({type: 'visibilitychange'});
  await settle();
};
const choose = async (id = 7, host = '.nav-section:not(.recent)') => {
  const selector = host === '#projects-panel' ? '.project-tile' : '.project';
  hosts[host].querySelectorAll(selector)[projects.findIndex(project => project.id === id)].click();
  await settle();
};
const assertCard = (id, selected) => {
  assert.equal(window.projectState().selected, selected);
  assert.equal(window.projectState().cardProject.id, id);
  assert.equal(hosts['#resume-card'].hidden, false);
  assert.equal(hosts['.resume-card h2'].textContent, 'Vamos continuar “' + projects.find(p => p.id === id).name + '”?');
  assert.deepEqual(hosts['.nav-section:not(.recent)'].querySelectorAll('.project')
    .filter(row => row.classList.contains('selected')).map(row => row.dataset.id), [String(id)]);
};
"""

    def run(steps):
        script = harness + source + "\n(async () => {\n" + """
window.cortexProjetos.init(() => {});
await settle();
""" + steps + "\n})().catch(error => { console.error(error); process.exitCode = 1; });"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
    return run


@pytest.mark.parametrize("host", [".nav-section:not(.recent)", "#projects-panel"])
def test_selected_project_survives_visibility_refresh(project_ui, host):
    project_ui("await choose(7, " + json.dumps(host) + ");" + """
assertCard(7, 7);
const start = requests.length;
await visibility();
assertCard(7, 7);
assert.ok(requests.slice(start).some(r => r.url === '/ui/projects/7'));
""")


def test_selected_project_delete_uses_choice_after_refresh_during_preview(project_ui):
    project_ui("""
await choose();
await visibility();
let release;
previewGate = new Promise(resolve => { release = resolve; });
const deleting = hosts['#delete-project'].click();
await visibility();
assertCard(7, 7);
release();
await deleting;
assert.equal(confirms.length, 1);
assert.ok(confirms[0].startsWith('Excluir o projeto “Tese (preview)”?'));
assert.deepEqual(requests.filter(r => r.method === 'POST').map(r => r.url), ['/ui/projects/7/delete']);
assertCard(8, null);
""")


@pytest.mark.parametrize("change", [
    "choose(8)", "chat.newConversation()", "(detailStatus = 503, visibility())",
])
def test_project_delete_aborts_if_card_changes_during_preview(project_ui, change):
    project_ui("""
await choose();
let release;
previewGate = new Promise(resolve => { release = resolve; });
const deleting = hosts['#delete-project'].click();
await """ + change + """;
await settle();
release();
await deleting;
assert.deepEqual(confirms, []);
assert.deepEqual(requests.filter(r => r.method === 'POST'), []);
assert.equal(hosts['#toast'].textContent, 'O projeto mudou — tente de novo.');
assert.equal(hosts['#delete-project'].disabled, false);
""")


def test_selected_project_missing_clears_choice_and_resumes(project_ui):
    project_ui("""
await choose();
detailStatus = 404;
const start = requests.length;
await visibility();
assertCard(8, null);
assert.ok(requests.slice(start).some(r => r.url === '/ui/projects/7'));
assert.ok(requests.slice(start).some(r => r.url === '/ui/resume'));
""")


def test_choosing_project_deleted_since_sidebar_read_resumes(project_ui):
    project_ui("""
detailStatus = 404;
await choose();
assertCard(8, null);
""")


@pytest.mark.parametrize("navigation", ["chat.open('trip')", "chat.newConversation()"])
def test_conversation_navigation_ends_project_choice(project_ui, navigation):
    project_ui("""
await choose();
""" + navigation + """;
await settle();
assertCard(8, null);
assert.ok(requests.some(r => r.url === '/ui/resume' +
  (chat.activeId ? '?conversation_id=' + chat.activeId : '')));
""")


def test_unselected_refresh_follows_current_resume(project_ui):
    project_ui("""
activeProject = projects[0];
await visibility();
assertCard(7, null);
activeProject = projects[1];
await visibility();
assertCard(8, null);
""")


def test_selected_project_read_failure_keeps_choice_for_retry(project_ui):
    project_ui("""
await choose();
detailStatus = 503;
const start = requests.length;
await visibility();
assert.equal(window.projectState().selected, 7);
assert.equal(hosts['#resume-card'].hidden, true);
assert.ok(!requests.slice(start).some(r => r.url.startsWith('/ui/resume')));
detailStatus = 200;
projects[0].name = 'Tese revisada';
await visibility();
assertCard(7, 7);
""")


def test_missing_project_fallback_cannot_replace_newer_choice(project_ui):
    project_ui("""
await choose();
let release;
resumeGate = new Promise(resolve => { release = resolve; });
detailStatus = 404;
await visibility();
assert.equal(window.projectState().selected, null);
await choose(8);
release();
await settle();
assertCard(8, 8);
""")
