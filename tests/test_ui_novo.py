"""The new shell preserves the approved design without inventing live data."""
from datetime import datetime
import hashlib
from pathlib import Path
import re

import pytest

from tests.test_ui_routes import client, ui_app  # noqa: F401
from tests.test_pending_contract import contract, disk  # noqa: F401
from tests.ui_parity import Document, METRICS, REFERENCE, check, demo_leftovers

KEY = "test-key-xyz"
HEADERS = {"X-Cortex-Key": KEY}
ASSETS = ["/ui/static/novo/style.css", "/ui/static/novo/shell.js"]
CORE = "/ui/static/novo/core.js"
CONVERSA = "/ui/static/novo/conversa.js"
PENDENTE = "/ui/static/novo/pendente.js"
MICROFONE = "/ui/static/novo/microfone.js"
ANEXOS = "/ui/static/novo/anexos.js"
CONTATOS = "/ui/static/novo/contatos.js"
QUARTO = "/ui/static/novo/quarto.js"
QUARTO_CSS = "/ui/static/novo/quarto.css"


def test_logout_header_order(page):
    children = [n for n in Document(page).select(".top-right")[0].children if not isinstance(n, str)]
    assert [(n.tag, n.attrs.get("class"), n.attrs.get("id")) for n in children] == [
        ("span", "prototype-badge", None), ("span", "privacy", None),
        ("a", "logout", None), ("button", "icon-button", "settings")]
    assert children[2].attrs["href"] == "/ui/logout" and children[2].text() == "Sair"


def test_contacts_central_placement_and_copy(page):
    doc = Document(page)
    options = [n for n in doc.select("#system-grid")[0].children if not isinstance(n, str)]
    assert [n.attrs["id"] for n in options] == ["option-" + key for key in METRICS] + ["option-contacts"]
    option = options[-1]
    assert option.tag == "button" and option.attrs == {
        "class": "system-option", "id": "option-contacts", "aria-label": "Contatos"}
    assert [(n.tag, n.attrs, n.text()) for n in option.descendants()[1:]] == [
        ("span", {"class": "option-heading"}, "Contatos↗"),
        ("i", {"class": "health-dot", "style": "background:var(--muted)"}, ""),
        ("span", {}, "Contatos"),
        ("span", {"class": "option-arrow", "aria-hidden": "true"}, "↗"),
        ("span", {"class": "option-summary"}, "Agenda local"),
        ("span", {"class": "option-description"}, "Nomes, emails e telefones salvos no Cortex")]
    assert check(page) == demo_leftovers(page) == []


def test_contacts_script_contract(client):
    response = client.get(CONTATOS)
    assert response.status_code == 200
    source = response.text
    assert source.startswith('"use strict";\n(() => {\n') and source.endswith("})();\n")
    assert re.findall(r"fetch\(([^,\n]+)", source) == ["'/ui/contacts'", "'/ui/contacts'", "'/ui/contacts/delete'"]
    assert re.findall(r"method:\s*'([^']+)'", source) == ["GET", "POST", "POST"]
    assert source.index("confirm('Apagar ' + name + '?')") < source.index("fetch('/ui/contacts/delete'")
    read = source.split("async function read(message = '') {", 1)[1].split("\nfunction render", 1)[0]
    guard = "if (version !== sequence || !dialog.open) return;"
    assert read.count(guard) == 2
    assert read.index("area.replaceChildren();") < read.index("fetch(") < read.index(guard) < read.index("render(")
    assert "area.replaceChildren(node('p', 'Não consegui carregar os contatos.'" in read
    assert source.count("await read(") == 2
    assert "if (button.id === 'option-contacts') read();" in source
    assert "const actions = node('div', null, 'contacts-actions');" in source
    assert "actions.append(edit, remove);" in source
    assert ("row.append(node('strong', contact.name), node('span', contact.email || '—'),\n"
            "      node('span', contact.phone || '—'), actions);") in source
    busy = source.split("function busy() {", 1)[1].split("\nasync function read", 1)[0]
    assert "area.querySelectorAll('.contacts-form button, .contacts-row button').forEach(button => {" in busy
    assert "button.disabled = writing;" in busy
    for fragment in (
        "document.addEventListener('click'", "closest('.system-option, #metric-return')",
        "++sequence", "button.id === 'option-contacts'", "const version = ++sequence",
        "version !== sequence || !dialog.open", "dialog.addEventListener('close'",
        "credentials: 'same-origin'", "cache: 'no-store'", "AbortSignal.timeout(10000)",
        "headers: {'Content-Type': 'application/json'}", "textContent", "replaceChildren",
        "Agenda local do Cortex: nome, email, telefone e notas.",
        "['name', 200, 'Nome *', 'Nome']", "['email', 320, 'email', 'Email']",
        "['phone', 60, 'telefone', 'Telefone']", "['notes', 2000, 'notas', 'Notas']",
        "input.required = key === 'name'", "setAttribute('aria-label', label)",
        "Salvar", "aria-live", "polite", "editar", "apagar", "'Editar ' + contact.name",
        "'Apagar ' + contact.name", "contact.email || '—'", "contact.phone || '—'",
        "Nenhum contato ainda — adicione um acima.", "Não consegui carregar os contatos.",
        ".trim()", "Nome é obrigatório.", "phone: value('phone') || null",
        "notes: value('notes') || null", "action === 'updated'", " atualizado.", " adicionado.",
        "form.reset()", "Não consegui salvar.", "Editando ", " — salve para atualizar.",
        "fields.name.focus()", "JSON.stringify({name})", " apagado.", "Não consegui apagar ",
        "if (writing) return;", "button.disabled = writing",
    ):
        assert fragment in source
    assert not re.search(r"innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|new\s+Function|"
                         r"document\.write|https?://|setInterval|visibilitychange|DOMContentLoaded|window\.\w+\s*=", source)


def test_contacts_and_logout_css(client):
    css = client.get(SECONDARY).text
    assert re.search(r"#option-contacts\s*\{\s*grid-column:\s*1 / -1;\s*\}", css)
    assert re.search(r"\.logout:hover\s*\{[^}]*color: var\(--text\)", css)
    for fragment in (".contacts-form", ".contacts-row", "flex-wrap: wrap", "min-width: 0", "overflow-wrap: anywhere"):
        assert fragment in css
    actions = re.search(r"\.contacts-actions\s*\{([^}]*)\}", css)
    assert actions
    for declaration in ("display: flex;", "gap: 8px;", "flex: 0 0 auto;", "margin-left: auto;"):
        assert declaration in actions.group(1)


@pytest.mark.parametrize("selector,font_size", [
    (".logout", "10px"), (".recent-select", "10px"),
    (".select-bar button", None), (".regen-button", "11px"),
])
def test_secondary_controls_touch_targets(client, selector, font_size):
    css = client.get(SECONDARY).text
    rules = re.findall(r"^" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    assert rules and any("min-height: 24px;" in rule for rule in rules)
    assert re.findall(r"font-size:\s*([^;]+);", " ".join(rules)) == ([font_size] if font_size else [])
    if selector in (".logout", ".recent-select"):
        assert any("min-width: 24px;" in rule for rule in rules)
        # Keep the original 12px line's footprint in the header and Recentes title.
        assert any("margin: -6px 0;" in rule for rule in rules)
    if selector == ".select-bar button":
        assert "font-size: 10px;" in re.search(r"^\.select-bar\s*\{([^}]*)\}", css, re.M).group(1)


def test_contacts_focus_after_successful_reread(client):
    source = client.get(CONTATOS).text
    render = source.split("function render(contacts, message) {", 1)[1].split("\n// Bubble", 1)[0]
    assert re.search(r"busy\(\);\s*//[^\n]*\n\s*"
                     r"if \(message && !dialog.contains\(document.activeElement\)\) \{\s*"
                     r"fields.name.focus\(\);\s*\}\s*\}$", render)


def test_contacts_routes_round_trip(client, tmp_path, monkeypatch):
    from src.integrations.contacts import ContactBook

    monkeypatch.setattr("src.integrations.contacts.contact_book", ContactBook(db_path=str(tmp_path / "contacts.db")))
    contact = {"name": "Ana", "email": "ana@example.com", "phone": "123", "notes": "Amiga"}
    added = client.post("/ui/contacts", headers=HEADERS, json=contact)
    assert added.status_code == 200 and added.json()["action"] == "added"
    assert client.get("/ui/contacts", headers=HEADERS).json() == {"contacts": [contact]}
    contact["email"] = "new@example.com"
    updated = client.post("/ui/contacts", headers=HEADERS, json=contact)
    assert updated.status_code == 200 and updated.json()["action"] == "updated"
    assert client.get("/ui/contacts", headers=HEADERS).json() == {"contacts": [contact]}
    deleted = client.post("/ui/contacts/delete", headers=HEADERS, json={"name": "Ana"})
    assert deleted.status_code == 200 and deleted.json()["ok"] is True
    assert client.get("/ui/contacts", headers=HEADERS).json() == {"contacts": []}
    assert client.post("/ui/contacts/delete", headers=HEADERS, json={"name": "Ana"}).status_code == 404


def attachment_source(client):
    response = client.get(ANEXOS)
    assert response.status_code == 200
    return response.text


def test_attachments_upload_and_safe_rendering(client):
    source = attachment_source(client)
    assert source.startswith('"use strict";\n(() => {\n') and source.endswith("})();\n")
    assert re.findall(r"window\.\w+\s*=", source) == ["window.cortexAnexos ="]
    assert re.findall(r"fetch\(([^,\n]+)", source) == ["'/ui/upload'"]
    for fragment in ("new FileReader()", "reader.readAsDataURL(file)",
                     "method: 'POST'", "credentials: 'same-origin'",
                     "headers: {'Content-Type': 'application/json'}",
                     "JSON.stringify({name: file.name, data_b64})",
                     "reader.result", "reader.addEventListener('error'",
                     "if (!response.ok)", "data.detail", "'Anexo: '",
                     "'não consegui ler esse arquivo'", "remove(item)",
                     "document.createElement('span')", "'attachment-chip'",
                     "'carregando ' + file.name + '…'", "' caracteres'",
                     "data.truncated ? ' · cortado' : ''", "data.chars",
                     "document.createElement('button')", "removeButton.type = 'button'",
                     "removeButton.textContent = '×'",
                     "removeButton.setAttribute('aria-label', 'Remover ' + file.name)"):
        assert fragment in source
    assert not re.search(r"\.innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|setInterval|"
                         r"https?://|XMLHttpRequest|WebSocket|sendBeacon|new\s+Function", source)


def test_attachments_limit_and_lifecycle(client):
    source = attachment_source(client)
    for fragment in ("fileInput.accept = '.txt,.md,.pdf,.docx'", "fileInput.click()",
                     "fileInput.addEventListener('change'", "2 - items.length",
                     "files.slice(0, slots)", "files.length > slots",
                     "No máximo 2 anexos por mensagem.", "Espere a resposta terminar.",
                     "item.ready = true", "items.some(item => !item.ready)",
                     "items.filter(item => item.ready).map", "({name, text}) => ({name, text})",
                     "items = [];", "list.replaceChildren()", "fileInput.value = ''"):
        assert fragment in source
    assert source.count("if (window.cortexConversa.inFlight)") == 2  # Picker and selection.
    assert source.count("if (!items.includes(item)) return;") >= 3  # No late resurrection/error.
    take = source.split("function take() {", 1)[1].split("\n}", 1)[0]
    assert take.index("items.filter") < take.index("clear();") < take.index("return ready;")
    shell = client.get(ASSETS[1]).text
    assert "window.cortexAnexos.clear();" in shell
    assert "Attachment controls arrive in a later task" not in shell


def test_attachments_normal_send_contract(conversa):
    submit = conversa.split("function submit(text = input.value.trim(), pendingId) {", 1)[1].split("\nfunction stop", 1)[0]
    normal = submit.split("if (!pendingId) {", 1)[1].split("\n  }", 1)[0]
    assert submit.count("window.cortexAnexos.take()") == normal.count("window.cortexAnexos.take()") == 1
    guard = submit.split("if (!pendingId) {", 1)[0]
    assert "window.cortexAnexos.uploading" in guard  # Even a card cannot send while loading.
    assert "Espere o anexo terminar de carregar." in guard and "return;" in guard
    for fragment in ("attachments = window.cortexAnexos.take()",
                     "'o que tem nesse arquivo?'", "'o que tem nesses arquivos?'",
                     "attachments.length === 1"):
        assert fragment in normal
    assert submit.index("window.cortexAnexos.uploading") < submit.index("input.value = ''")
    for fragment in ('"\\n\\n📎 "', ".join(', ')", "question", "attachments",
                     "if (!text) return;", "streamTurn(text, run)"):
        assert fragment in submit
    body = conversa.split("const body = run.pendingId", 1)[1].split("const response = await fetch('/ui/chat/stream'", 1)[0]
    pending, ordinary = body.split("pending_id: run.pendingId})", 1)
    assert "attachments" not in pending
    assert "attachments: run.attachments" in ordinary
    assert "run.attachments?.length" in ordinary  # Omit the field when empty.
    assert "addMessage('user', run.question)" in conversa
    assert "attachments.filter(item => item.text.trim()).map(item => item.name.trim() || 'arquivo')" in submit
    assert 'turn.user.includes("\\n\\n📎 ")' in conversa


@pytest.fixture
def page(client):
    response = client.get("/ui/novo", headers=HEADERS)
    assert response.status_code == 200
    assert response.template.name == "novo.html"
    assert response.headers["content-type"].startswith("text/html")
    assert KEY not in response.text
    return response.text


def microphone_source(client):
    response = client.get(MICROFONE)
    assert response.status_code == 200
    return response.text


def test_microphone_local_only(client):
    source = microphone_source(client)
    assert source.startswith('"use strict";\n(() => {\n') and source.endswith("})();\n")
    assert re.findall(r"window\.\w+\s*=", source) == ["window.cortexMicrofone ="]
    assert re.findall(r"fetch\(([^,\n]+)", source) == ["'/ui/voice/transcribe'"]
    assert source.count("method: 'POST'") == 1
    for fragment in ("credentials: 'same-origin'", "'Content-Type': blob.type || 'audio/webm'",
                     "body: blob", "AbortSignal.timeout(120000)", "window.isSecureContext",
                     "navigator.mediaDevices?.getUserMedia", "window.MediaRecorder",
                     "getUserMedia({audio: true})", "window.cortexConversa.inFlight"):
        assert fragment in source
    assert not re.search(r"SpeechRecognition|webkitSpeechRecognition|https?://|XMLHttpRequest|"
                         r"WebSocket|sendBeacon|submit|requestSubmit|\.innerHTML|insertAdjacentHTML|"
                         r"outerHTML|\beval\s*\(|setInterval", source)


def test_microphone_capture_lifecycle(client):
    source = microphone_source(client)
    start = source.split("recorder.addEventListener('start', () => {", 1)[1].split("\n    });", 1)[0]
    assert source.count("setLive('listening')") == start.count("setLive('listening')") == 1
    for fragment in ("recorder.start()", "new MediaRecorder(stream)", "setTimeout(stop, 60000)",
                     "'Parar gravação'", "'Gravar voz'", "mic.addEventListener('click'",
                     "if (active) { stop(); return; }", "recorder.addEventListener('error',",
                     "recorder.addEventListener('stop',", "recorder.addEventListener('dataavailable',",
                     "event.data.size", "if (!blob.size)", "mic.disabled = true",
                     "mic.disabled = false", "clearTimeout(run.timer)", "window.addEventListener('pagehide'"):
        assert fragment in source
    release = source.split("function release(run) {", 1)[1].split("\nfunction ", 1)[0]
    assert "getTracks().forEach(track => track.stop())" in release
    assert "setLive('idle')" in release and "'Gravar voz'" in release
    failure = source.split("function fail(run, message) {", 1)[1].split("\nfunction ", 1)[0]
    assert "release(run)" in failure and "run.failed = true" in failure
    stop = source.split("function stop() {", 1)[1].split("\nasync function ", 1)[0]
    assert "recorder.stop()" in stop and "release(run)" in stop and "fail(run," in stop
    assert "catch { fail(run," in source
    stopped = source.split("recorder.addEventListener('stop', () => {", 1)[1].split("\n    });", 1)[0]
    assert stopped.index("release(run)") < stopped.index("transcribe(new Blob(")
    assert "if (run.failed) { mic.disabled = false; return; }" in stopped
    assert source.index("if (!blob.size)") < source.index("await fetch(")


def test_microphone_transcript_and_messages(client):
    source = microphone_source(client)
    for fragment in ("document.querySelector('#message-input')", "const placeholder = input.placeholder",
                     "input.placeholder = 'Transcrevendo…'", "input.placeholder = placeholder",
                     "input.value += (input.value ? ' ' : '') + text",
                     "input.dispatchEvent(new Event('input', {bubbles: true}))", "input.focus()",
                     "response.status === 413", "data.error === 'too_long'", "data.error === 'too_large'",
                     "response.status === 400", "response.status === 503", "window.cortexProjetos.toast",
                     "Espere a resposta terminar para gravar.",
                     "O microfone só funciona por HTTPS ou no próprio Mac.",
                     "Sem permissão para usar o microfone.", "Não entendi nada na gravação.",
                     "A gravação passou de 1 minuto.", "A gravação ficou grande demais.",
                     "Não consegui ler o áudio.", "A transcrição local não está disponível agora.",
                     "Não consegui transcrever."):
        assert fragment in source


def test_pending_stale_sim_never_executes(contract):
    o, send, _ids, ask = contract
    ask()
    displayed = disk(o)["id"]
    ask()
    current = disk(o)
    assert current["id"] != displayed
    turn = ask("sim", pending_id=displayed, language="pt")[-1]["turn"]
    assert turn["status_mode"] == "AGENT_ACTION_STALE"
    assert disk(o) == current
    send.assert_not_called()


@pytest.fixture
def pendente(client):
    response = client.get(PENDENTE)
    assert response.status_code == 200
    return response.text


def test_pending_reads_transfer_and_refresh(pendente, conversa):
    assert re.findall(r"fetch\(([^,\n]+)", pendente) == ["'/ui/pending/transfer'", "url"]
    assert re.findall(r"read\(([^\n]+)", pendente) == [
        "url) {", "'/ui/pending' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : '')),",
        "'/ui/effects' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))",
    ]
    assert pendente.count("method: 'POST'") == 1
    for fragment in ("JSON.stringify({pending_id: pending.id, conversation_id: cid})",
                     "credentials: 'same-origin'", "cache: 'no-store'", "AbortSignal.timeout(10000)",
                     "++sequence", "version !== sequence || cid !== chat.activeId",
                     "card.hidden = notice.hidden = true", "Promise.allSettled",
                     "if (!response.ok) throw", "status === 'fulfilled'",
                     "document.addEventListener('cortex-conversation', refresh)",
                     "visibilitychange", "if (!document.hidden) refresh()",
                     "$('#metric-detail').addEventListener('close', refresh)",
                     "window.cortexPendente = {refresh};\nrefresh();"):
        assert fragment in pendente
    for fragment in ("function submit(text = input.value.trim(), pendingId)",
                     "if (flight) return;", "pendingId, tokens:",
                     "pending_id: run.pendingId", "if (!pendingId) {",
                     "if (addUser) addMessage('user', text);", "streamTurn(text, run)"):
        assert fragment in conversa


def test_pending_whole_preview_and_actions(pendente, client, page):
    for fragment in ("$('#attachment-list').before(card, notice)", "node('details')",
                     "node('summary', 'Ver a prévia inteira')", "node('pre', pending.preview",
                     "element.textContent = text", "if (pending.expired) return;",
                     "if (pending.here)", "action(card, 'Sim', () => chat.submit('sim', pending.id))",
                     "action(card, 'Não', () => chat.submit('não', pending.id))",
                     "else if (pending.origin?.channel === 'voice')", "if (chat.activeId)",
                     "Trazer para esta conversa", "Abra ou comece uma conversa para trazer esta confirmação.",
                     "pending.origin?.channel === 'ui' && pending.origin.conversation",
                     "chat.open(pending.origin.conversation)", "Abrir a conversa",
                     "if (chat.inFlight) return;", "Esperando sua confirmação",
                     "pedido nesta conversa", "pedido pela voz", "pedido em outra conversa",
                     "vale até ", "venceu às ", "getHours()", "getMinutes()",
                     "Essa confirmação já foi resolvida.", "Essa confirmação venceu.",
                     "Não consegui trazer a confirmação.", "window.cortexProjetos.toast",
                     "['transferred', 'already_here']", "finally { refresh(); }"):
        assert fragment in pendente
    assert pendente.index("if (pending.expired) return;") < pendente.index("action(card, 'Sim'")
    assert not re.search(r"slice|substring|\.innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|"
                         r"new\s+Function|document\.write|https?://|setInterval|/ui/effects/seen", pendente)
    assert pendente.startswith('"use strict";\n(() => {\n') and pendente.endswith("})();\n")
    assert re.findall(r"window\.\w+\s*=", pendente) == ["window.cortexPendente ="]
    assert Document(page).select(".pending-card") == []
    css = client.get(SECONDARY).text
    for fragment in (".pending-card", ".effect-notice", "white-space: pre-wrap",
                     "overflow-wrap: anywhere", "max-width: 100%", "flex-wrap: wrap",
                     "var(--surface)", "var(--line)", "var(--muted)"):
        assert fragment in css


def test_pending_title_reflects_expiry(pendente):
    assert "node('h2', pending.expired ? 'Confirmação vencida' : 'Esperando sua confirmação')" in pendente


def test_pending_unknown_notice_is_read_only(pendente):
    for fragment in ("data?.available ? data.notices.length : 0", "notice.hidden = count === 0",
                     "O Cortex não sabe se ", " ação(ões) aconteceram.", "action(notice, 'Ver'",
                     "$('#settings').click()", "$('#option-audit').click()"):
        assert fragment in pendente
    assert "seen" not in pendente and "Já conferi" not in pendente


def test_auth(client):
    assert client.get("/ui/novo").status_code == 401
    assert client.get("/ui/novo", headers={"X-Cortex-Key": "wrong"}).status_code == 401
    client.cookies.set("cortex_ui", KEY)
    assert client.get("/ui/novo").status_code == 401
    client.cookies.clear()
    response = client.post("/ui/login", data={"key": KEY}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/ui/novo"
    assert client.get("/ui/novo").status_code == 200
    response = client.get("/ui/chat", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/ui/novo"


@pytest.mark.parametrize("path", ["/", "/ui", "/ui/chat"])
def test_login_and_old_addresses_open_novo(client, path):
    response = client.post("/ui/login", data={"key": KEY}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/ui/novo"
    landing = client.get(response.headers["location"])
    assert landing.status_code == 200
    assert landing.template.name == "novo.html"
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/ui/novo"
    client.cookies.clear()
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/ui/login"
    response = client.get(path, headers=HEADERS, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/ui/novo"
    for asset in ("/ui/static/dashboard.js", "/ui/static/chat.js"):
        assert client.get(asset).status_code == 404


def test_no_old_ui_references_in_source():
    removed = (b"static/dashboard.js", b"static/orb.js", b"static/chat.js",
               b"static/chat.css", b"/ui/static/style.css", b"dashboard.html", b"chat.html")
    source = Path(__file__).resolve().parents[1] / "src"
    mentions = []
    for path in sorted(source.rglob("*")):
        # Generated bytecode is not source; include every source asset and comment.
        if path.is_file() and path.suffix != ".pyc":
            content = path.read_bytes()
            mentions.extend(f"{path.relative_to(source)}: {name.decode()}"
                            for name in removed if name in content)
    assert mentions == []


def test_parity_and_local_assets(client, page):
    css, js = [client.get(path) for path in ASSETS]
    assert css.status_code == js.status_code == 200
    assert css.content == (REFERENCE / "style.css").read_bytes()
    assert check(page, js.text) == []
    assert demo_leftovers(page) == []
    urls = [value for node in Document(page).nodes for attr, value in node.attrs.items()
            if attr in {"href", "src"} and not value.startswith("#")]
    assert urls == [ASSETS[0], SECONDARY, "/ui/static/novo/avatar.css", QUARTO_CSS, "/ui/logout", *["/ui/static/novo/avatar/duckcortex.png"] * 5, "/ui/static/novo/avatar.js", CORE, CONVERSA, QUARTO, PROJETOS, CENTRAL, PENDENTE, MICROFONE, ANEXOS, CONTATOS, ASSETS[1]]
    assert not re.search(r"https?://", page.replace("http://www.w3.org/2000/svg", ""))
    assert all(word not in js.text for word in (
        "fetch(", "XMLHttpRequest", "WebSocket", "EventSource", ".innerHTML",
        "setTimeout", "setInterval"))


def test_empty_slots(page):
    doc = Document(page)
    assert all(text not in page for text in (
        "Planejar minha semana", "Design do CORTEX", "Ideias e referências",
        "Design da nova experiência", "Ideias, planos e referências",
        "Vamos continuar o CORTEX?", "Próxima etapa", "Por onde vamos começar?",
        "586", "4 / 5", "11 no mês"))
    for selector in ("[data-project]", "[data-title]", ".sample-notice"):
        assert doc.select(selector) == []
    for selector in ("#breadcrumb", "#messages", "#state-label", ".privacy",
                     ".sidebar-bottom", ".resume-copy", "#metric-title", "#metric-content"):
        assert doc.select(selector)[0].text() == ""
    assert doc.select("#messages")[0].children == []
    assert "hidden" in doc.select("#resume-card")[0].attrs
    assert [node.text() for node in doc.select(".nav-section")] == ["Projetos", "Recentes"]
    assert doc.select("#projects-panel")[0].text() == "Seus projetos"
    for key, title in METRICS.items():
        option = doc.select("#option-" + key)[0]
        assert option.tag == "button" and option.attrs["class"] == "system-option"
        children = [node for node in option.descendants() if node is not option]
        assert [(node.tag, node.attrs.get("class"), node.text()) for node in children] == [
            ("span", "option-heading", title + "↗"), ("i", "health-dot", ""),
            ("span", None, title), ("span", "option-arrow", "↗"),
            ("span", "option-summary", ""), ("span", "option-description", "")]
        assert children[1].attrs["style"] == "background:var(--muted)"


@pytest.mark.parametrize("hour,greeting", [
    (0, "Boa noite"), (4, "Boa noite"), (5, "Bom dia"), (9, "Bom dia"),
    (11, "Bom dia"), (12, "Boa tarde"), (15, "Boa tarde"),
    (17, "Boa tarde"), (18, "Boa noite"), (23, "Boa noite"),
])
def test_greeting(client, monkeypatch, hour, greeting):
    monkeypatch.delenv("CORTEX_USER_NAME", raising=False)
    monkeypatch.setattr("src.ui.routes._local_now", lambda: datetime(2026, 10, 20, hour))
    page = client.get("/ui/novo", headers=HEADERS).text
    assert f"<h1>{greeting}, Pedro<span>.</span></h1>" in page


@pytest.mark.parametrize("name,expected", [
    ("", "Bom dia"), ("Ana", "Bom dia, Ana"), ("<b>x</b>", "Bom dia, &lt;b&gt;x&lt;/b&gt;"),
])
def test_greeting_name(client, monkeypatch, name, expected):
    monkeypatch.setenv("CORTEX_USER_NAME", name)
    monkeypatch.setattr("src.ui.routes._local_now", lambda: datetime(2026, 10, 20, 9))
    page = client.get("/ui/novo", headers=HEADERS).text
    assert f"<h1>{expected}<span>.</span></h1>" in page


def test_core_load_order_and_parity(client, page):
    scripts = [node.attrs["src"] for node in Document(page).nodes if node.tag == "script"]
    assert scripts == ["/ui/static/novo/avatar.js", CORE, CONVERSA, QUARTO, PROJETOS, CENTRAL, PENDENTE, MICROFONE, ANEXOS, CONTATOS, ASSETS[1]]
    core, conversa, projetos, shell = [client.get(path) for path in scripts if path not in ("/ui/static/novo/avatar.js", QUARTO, CENTRAL, PENDENTE, MICROFONE, ANEXOS, CONTATOS)]
    assert core.status_code == conversa.status_code == projetos.status_code == shell.status_code == 200
    assert check(page, shell.text) == []
    assert demo_leftovers(page) == []


def test_core_reference_drawing(client):
    response = client.get(CORE)
    assert response.status_code == 200
    core = response.text
    reference = (REFERENCE / "app.js").read_text()
    # Task 156 (the user chose "núcleo C" in the UI lab): the sphere is drawn in
    # depth, as a gyroscope inside the shell. The 165-point geometry, its web
    # and the canvas sizing stay byte-exact with the prototype; the drawing
    # itself is exercised by executing the module in test_ui_nucleo_activity.
    assert "const {intensity, reduced} = inputs;" in core
    assert "const {reduced} = inputs;" in core
    assert core.count("const state = inputs.preview ?? inputs.live;") == 2
    for start, end in (
        ("const points = [];", "function resizeCanvas()"),
        ("function resizeCanvas()", "function draw()"),
    ):
        block = reference.split(start, 1)[1].split(end, 1)[0]
        assert start + block in core
    for constant in ("count = 165", "Math.PI * (3 - Math.sqrt(5))", ".slice(0,4)",
                     "Math.min(width,height)*.365", "D/(D-v.z)",
                     "(reduced?180:32)", "Math.min(devicePixelRatio||1,2)"):
        assert constant in core
    # No library: the depth comes from the 2D canvas alone.
    assert all(word not in core for word in ("import", "require(", "webgl", "THREE"))


def test_core_honest_states(client):
    response = client.get(CORE)
    assert response.status_code == 200
    core = response.text
    assert "live: 'idle', preview: null" in core
    for state, label in (("idle", ""), ("thinking", "Pensando"),
                         ("searching", "Buscando na web"), ("reading", "Lendo"),
                         ("writing", "Escrevendo"), ("listening", "Ouvindo"),
                         ("speaking", "Respondendo")):
        assert f"{state}: '{label}'" in core
    for label in ("Em repouso", "Pensando", "Buscando na web", "Lendo", "Escrevendo",
                  "Ouvindo", "Respondendo"):
        assert f"'Prévia · {label}'" in core
    assert all(word not in core.lower() for word in ("simulação", "demonstração", "demo"))
    live = core.split("function setLive(next, avatarState = next) {", 1)[1].split("function preview(next)", 1)[0]
    preview = core.split("function preview(next) {", 1)[1].split("function setIntensity", 1)[0]
    assert "inputs.live = next;" in live
    assert "next === 'listening'" in live
    assert "classList.toggle('mic-active'," in live and "setAttribute('aria-pressed'," in live
    assert core.count("$('#mic')") == live.count("$('#mic')") == 2
    assert "inputs.preview = next;" in preview and "inputs.live =" not in preview
    assert "return inputs.live;" in preview
    assert "inputs.preview === null ? liveLabels[inputs.live] : previewLabels[inputs.preview]" in core


def test_core_private_and_local(client):
    response = client.get(CORE)
    assert response.status_code == 200
    core, shell = response.text, client.get(ASSETS[1]).text
    for source in (core, shell):
        assert all(word not in source for word in (
            "fetch(", "XMLHttpRequest", "WebSocket", "EventSource", ".innerHTML",
            "setTimeout", "setInterval"))
    assert core.startswith('"use strict";\n(() => {\n') and core.endswith("})();\n")
    assert re.findall(r"window\.\w+", core) == ["window.cortexAvatar", "window.cortexAvatar", "window.cortexCore"]
    assert "window.cortexCore = {setLive, preview, setIntensity, setReduced, pulseToken, caption, redraw: draw};" in core


def test_appearance_core_wiring(client):
    shell = client.get(ASSETS[1]).text
    assert "window.cortexCore.setReduced(value);" in shell
    assert "window.cortexCore.redraw();" in shell
    assert "preference('intensity')" in shell and "remember('intensity'," in shell
    assert "window.cortexCore.setIntensity(" in shell
    assert "if ($('#controls').open) window.cortexCore.preview(e.target.value);" in shell
    close = shell.split("$('#controls').addEventListener('close', () => {", 1)[1].split("});", 1)[0]
    assert "$('#state-select').value = window.cortexCore.preview(null);" in close
    compare = shell.split("$('#preview-chat').addEventListener('click', () => {", 1)[1].split("});", 1)[0]
    assert "$('#controls').close();" in compare
    assert "navigate(document.body.classList.contains('chat-view') ? 'home' : 'chat');" in compare
    assert "toast" not in compare


def test_appearance_preview_requires_selection(client):
    shell = client.get(ASSETS[1]).text

    def handler(selector, event):
        start = f"$('{selector}').addEventListener('{event}', "
        bodies = re.findall(r"^" + re.escape(start) + r"[^\n]*\{\n(.*?)^\}\);",
                            shell, re.MULTILINE | re.DOTALL)
        assert len(bodies) == 1
        return bodies[0]

    opening = handler("#open-appearance", "click")
    change = handler("#state-select", "change")
    calls = r"\bpreview\(([^;\n]*)\)"
    assert re.findall(calls, opening) == ["null"]
    assert "$('#state-select').value = window.cortexCore.preview(null);" in opening
    assert "closeHubTo($('#controls'));" in opening
    assert re.findall(calls, change) == ["e.target.value"]
    assert "if ($('#controls').open) window.cortexCore.preview(e.target.value);" in change
    outside_change = shell.replace(change, "", 1)
    assert all(argument == "null" for argument in re.findall(calls, outside_change))


@pytest.fixture
def conversa(client):
    response = client.get(CONVERSA)
    assert response.status_code == 200
    return response.text


def test_conversa_wire_contract(conversa):
    for fragment in (
        "fetch('/ui/chat/stream',", "JSON.stringify({text, conversation_id: run.cid})",
        "fetch('/ui/chat/cancel',", "JSON.stringify({id: run.id})",
        "fetch('/ui/conversations',", "method: 'POST'", "credentials: 'same-origin'",
        "buf += decoder.decode(value, {stream: true})", 'buf.indexOf("\\n\\n")',
        "buf = buf.slice(sep + 2)", "JSON.parse(dataLine.slice(5).trim())",
        "run.id = evt.id", "run.tokens += evt.text || ''", "saved(run.live, evt.turn)",
        "evt.detail || 'erro'", "new AbortController()", "run.controller.abort()",
    ):
        assert fragment in conversa
    assert set(re.findall(r"evt.type === '(\w+)'", conversa)) == {
        "turn", "status", "token", "final", "error"}
    assert conversa.count("method: 'POST'") == 4


def test_conversa_safe_markdown(conversa):
    # Verified byte-exact against the old renderer before removal; changes update this pin.
    start = "  function escapeHtml(s)"
    markdown = start + conversa.split(start, 1)[1].split("\nfunction badges(usage)", 1)[0]
    assert hashlib.sha256(markdown.encode("utf-8")).hexdigest() == (
        "e92de4695f329bd7adef246affdb66dea1301b7aef1206993d9c9eaac45d4476")
    writes = re.findall(r"\.innerHTML\s*([+]?=)\s*([^\n]+)", conversa)
    assert writes and all(op == "=" and rhs.startswith("renderMarkdown(") for op, rhs in writes)
    assert not re.search(r"\beval\s*\(|new\s+Function\b|document\.write|\bon\w+\s*=", conversa)
    assert not re.search(r"https?://", conversa.replace("http://www.w3.org/2000/svg", ""))
    for fragment in ("'message ' + role", "'message-content'", "'avatar'", "'author'",
                     "'CORTEX' : 'VOCÊ'", "$('.breadcrumb svg').cloneNode(true)",
                     "live.prepend(author)"):
        assert fragment in conversa


def test_conversa_truth_and_lifecycle(client, conversa):
    badges = conversa.split("function badges(usage) {", 1)[1].split("\n}", 1)[0]
    assert "privacy.textContent = model.textContent = '';" in badges
    assert "if (!Array.isArray(usage?.online) || !Array.isArray(usage?.models)) return;" in badges
    assert "privacy.textContent = usage.online.length ? 'Online' : 'Local';" in badges
    assert "model.textContent = usage.models.join(' · ');" in badges
    assert conversa.count("privacy.textContent") == badges.count("privacy.textContent") == 2
    for fragment in ("localStorage.getItem(", "localStorage.setItem(", "localStorage.removeItem(",
                     "await ready", "if (flight) return", "badges(turns.at(-1)?.usage)",
                     "badges(turn.usage)", "classList.toggle('is-stop',",
                     "'Parar resposta'", "setLive('thinking')", "setLive('speaking')",
                     "window.cortexCore.pulseToken?.()",
                     "setLive('idle')", "textContent = '— parado —'"):
        assert fragment in conversa
    shell = client.get(ASSETS[1]).text
    for call in ("init(navigate)", "submit()", "stop()", "newConversation()", "enter()"):
        assert "window.cortexConversa." + call in shell


def test_conversa_local_bounded_reconciliation(conversa):
    assert re.findall(r"fetch\(([^,\n]+)", conversa) == [
        "'/ui/conversations/' + encodeURIComponent(cid)",
        "'/ui/conversations'", "'/ui/chat/stream'",
        "'/ui/conversations/' + encodeURIComponent(cid) + '/pop_last'", "'/ui/chat/cancel'"]
    assert "setInterval" not in conversa
    for fragment in ("Date.now() + 63000", "attempt < 22", "setTimeout(resolve, 3000)",
                     "AbortSignal.timeout(", "deadline - Date.now()", "saved(run.live, matches[0])",
                     "!run.known.has(turn.id)", "turn.user === run.question",
                     "turn.source === 'ui'", "matches.length === 1", "run.epoch !== epoch",
                     "Date.parse(turn.asked_ts) >= run.started", "Date.parse(turn.asked_ts) <= run.stoppedAt"):
        assert fragment in conversa
    reconcile = conversa.split("async function reconcile(run)", 1)[1].split("\nasync function", 1)[0]
    assert "addMessage(" not in reconcile


PROJETOS = "/ui/static/novo/projetos.js"
SECONDARY = "/ui/static/novo/secundario.css"


def test_projects_assets_and_empty_hosts(client, page):
    assets = [ASSETS[0], SECONDARY, "/ui/static/novo/avatar.css", QUARTO_CSS, "/ui/static/novo/avatar.js", CORE, CONVERSA, QUARTO, PROJETOS, CENTRAL, PENDENTE, MICROFONE, ANEXOS, CONTATOS, ASSETS[1]]
    urls = [n.attrs.get("href", n.attrs.get("src")) for n in Document(page).nodes
            if n.tag in {"link", "script"}]
    assert urls == assets
    assert all(client.get(path).status_code == 200 for path in assets)
    assert client.get(ASSETS[0]).content == (REFERENCE / "style.css").read_bytes()
    assert check(page, client.get(ASSETS[1]).text) == demo_leftovers(page) == []
    assert [n.text() for n in Document(page).select(".nav-section")] == ["Projetos", "Recentes"]
    assert "hidden" in Document(page).select("#resume-card")[0].attrs


@pytest.fixture
def projetos(client):
    response = client.get(PROJETOS)
    assert response.status_code == 200
    return response.text


def test_projects_wire_and_refresh(projetos, conversa):
    assert re.findall(r"fetch\(([^,\n]+)", projetos) == ["url"]
    assert set(re.findall(r"request\(([^\n]+)", projetos)) == {
        "url, body) {", "'/ui/projects'), request('/ui/conversations'),",
        "'/ui/conversations/delete', {ids});",
        "'/ui/resume' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))",
        "id != null ? '/ui/projects/' + encodeURIComponent(id) :",
        "'/ui/projects/' + encodeURIComponent(id));",
        "'/ui/projects/' + encodeURIComponent(id) + '/delete-preview');",
        "'/ui/projects/' + encodeURIComponent(id) + '/delete', {});",
        "'/ui/conversations/' + encodeURIComponent(chat.activeId)); }",
        "'/ui/conversations/' + encodeURIComponent(id) + '/rename', {title}); }",
        "'/ui/conversations/' + encodeURIComponent(id) + '/delete', {}); }"}
    for fragment in ("credentials: 'same-origin'", "method: body ? 'POST' : 'GET'",
                     "JSON.stringify(body)", "visibilitychange", "document.hidden",
                     "++sequence", "version !== sequence", "Promise.allSettled",
                     "document.addEventListener('cortex-conversation', refresh)"):
        assert fragment in projetos
    assert "setInterval" not in projetos
    assert "new CustomEvent('cortex-conversation')" in conversa
    assert "await reconcile(run)" in conversa
    assert "flight = null; busy(); input.focus(); changed();" in conversa


def test_projects_safe_rendering(client, projetos):
    assert not re.search(r"\.innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|new\s+Function|document\.write|\bon\w+\s*=", projetos)
    assert not re.search(r"https?://", projetos.replace("http://www.w3.org/2000/svg", ""))
    assert "textContent" in projetos and "createElementNS" in projetos
    assert "'use'" in projetos and "'#i-folder'" in projetos and "'#i-file'" in projetos
    css = client.get(SECONDARY)
    assert css.status_code == 200
    assert not re.search(r"url\(|@import|https?://", css.text)
    for fragment in ("overflow-y: auto", ":focus-within", "(hover: none)", "var(--muted)"):
        assert fragment in css.text


def test_projects_honesty(projetos):
    for fragment in ("data.available === false || data.project == null", "card.hidden = true",
                     "data.progress?.next_step?.text", "Próximo passo: ",
                     "Nenhum próximo passo anotado. Dá pra anotar com “próximo:”.",
                     "tag.hidden = !tag.textContent", "Desta conversa", "Projeto ativo",
                     "Nenhum projeto ainda. Diga “projeto: nome” numa conversa para criar um.",
                     "Projetos indisponíveis agora.", "project.objective || ''",
                     "project.id === data.active_id", "project.conversations.includes(cid)"):
        assert fragment in projetos
    assert not re.search(r"last_done|todos|last_decision|\bCORTEX\b|\bPessoal\b", projetos)


def test_projects_actions_as_today(projetos, conversa):
    for fragment in ('prompt("Renomear conversa:", current || "")',
                     'confirm("Apagar esta conversa? (a memória de longo prazo do Cortex fica intacta)")',
                     "'dblclick'", "'Apagar conversa ' + title.textContent", "Não consegui renomear a conversa.",
                     "Não consegui apagar a conversa.", "error.status !== 404",
                     "if (id === chat.activeId) chat.newConversation()",
                     "conversations.find(c => project.conversations.includes(c.id))",
                     "await chat.open(recent.id)"):
        assert fragment in projetos
    for name in ("rename(id, current)", "remove(id)", "resume()"):
        assert f"async function {name} {{\n  if (chat.inFlight) return;" in projetos
    assert "function open(id) {\n  if (flight) return;" in conversa
    assert "Essa conversa não existe mais." in conversa


def test_recent_delete_names_its_conversation(projetos):
    assert "node('button', '×', 'recent-delete')" in projetos
    assert "del.setAttribute('aria-label', 'Apagar conversa ' + title.textContent);" in projetos


def test_regenerate_allowlist_and_latest(conversa):
    modes = re.findall(r"const REGENERATE_MODES = new Set\(\[([^]]+)\]\);", conversa)
    assert len(modes) == 1
    assert re.findall(r"'([^']+)'", modes[0]) == [
        "OFFLINE", "REFLEX", "ONLINE_RAG", "AUTO_SEARCH", "ONLINE_REQUEST", "LOCAL_RAG"]
    refresh = conversa.split("function refreshRegen() {", 1)[1].split("\nfunction ", 1)[0]
    for fragment in ("querySelectorAll('.regen-button')", "button.remove()",
                     "querySelectorAll('.assistant .message-content')", ".at(-1)",
                     "savedTurns.get(live)", "!turn?.id", "!turn.user?.trim()",
                     'turn.user.includes("\\n\\n📎 ")', "!REGENERATE_MODES.has(turn.status_mode)",
                     "button.textContent = 'regenerar'", "live.append(button)"):
        assert fragment in refresh
    assert conversa.count("button.textContent = 'regenerar'") == 1
    assert "button.title = 'descarta esta resposta e pergunta de novo'" in refresh
    saved = conversa.split("function saved(live, turn) {", 1)[1].split("\nasync function", 1)[0]
    assert "if (flight?.live !== live || !flight.stopped) savedTurns.set(live, turn);" in saved
    assert "refreshRegen();" in saved
    add = conversa.split("function addMessage(role, text) {", 1)[1].split("\nfunction saved", 1)[0]
    assert "refreshRegen();" in add
    assert "if (evt.type === 'final') saved(run.live, evt.turn);" in conversa


def test_regenerate_wire_and_stop(conversa):
    regen = conversa.split("async function regenerate(live, turn) {", 1)[1].split("\nfunction submit", 1)[0]
    for fragment in ("if (flight || !activeId) return;", "const cid = activeId;",
                     "question: turn.user", "flight = run; busy();",
                     "fetch('/ui/conversations/' + encodeURIComponent(cid) + '/pop_last',",
                     "method: 'POST'", "JSON.stringify({turn_id: turn.id})",
                     "if (run.epoch !== epoch) return;", "if (response.status === 409)",
                     "await load(cid)", "if (!response.ok) throw",
                     "live.closest('.assistant').remove();", "await streamTurn(turn.user, run, false);",
                     "Não consegui regenerar a resposta."):
        assert fragment in regen
    assert regen.index("flight = run; busy();") < regen.index("await fetch(")
    assert regen.index("if (!response.ok)") < regen.index(".remove();")
    assert "addMessage(" not in regen and regen.count(".remove();") == 1
    assert "if (addUser) addMessage('user', text);" in conversa
    assert "run.cid = run.cid || await ensureConversation(run);" in conversa
    assert "if (run.stopped && run.sent) await reconcile(run);" in conversa


def test_bulk_selection_and_wire(projetos):
    bulk = projetos.split("async function deleteSelected() {", 1)[1].split("\nasync function", 1)[0]
    for fragment in ("if (chat.inFlight || deleting) return;", "const ids = [...selectedIds];",
                     "if (!ids.length) return;", "if (!confirm(message)) return;",
                     "await request('/ui/conversations/delete', {ids});",
                     "if (ids.includes(chat.activeId)) chat.newConversation();",
                     "setSelecting(false);", "refresh();", "Não consegui apagar as conversas."):
        assert fragment in bulk
    assert bulk.count("confirm(") == bulk.count("request(") == 1
    assert "Apagar 1 conversa? (a memória de longo prazo do Cortex fica intacta)" in bulk
    assert "'Apagar ' + ids.length + ' conversas? (a memória de longo prazo do Cortex fica intacta)'" in bulk
    for fragment in ("'Selecionar conversas para apagar'", "selecting ? 'Concluir' : 'Selecionar'",
                     "n === 1 ? '1 selecionada' : n + ' selecionadas'", "n === 0 || deleting",
                     "'Cancelar'", "'Apagar'", "box.type = 'checkbox'",
                     "box.setAttribute('aria-label', 'Selecionar ' + title.textContent)",
                     "selecting ? toggleSelected(conversation.id) : chat.open(conversation.id)",
                     "del.hidden = selecting", "if (!selecting) rename(",
                     "selectedIds.clear()", "conversations.some(c => c.id === id)",
                     "e.key === 'Escape' && selecting", "setSelecting(false)"):
        assert fragment in projetos
    catch = bulk.split("catch", 1)[1]
    assert "selectedIds.clear()" not in catch and "setSelecting(false)" not in catch


def test_new_controls_safe_and_quiet(client, conversa, projetos):
    assert ".innerHTML" not in projetos
    writes = re.findall(r"\.innerHTML\s*([+]?=)\s*([^\n]+)", conversa)
    assert writes and all(op == "=" and rhs.startswith("renderMarkdown(") for op, rhs in writes)
    for source in (conversa, projetos):
        assert not re.search(r"\beval\s*\(|new\s+Function\b|document\.write|\bon\w+\s*=|setInterval", source)
        assert not re.search(r"https?://", source.replace("http://www.w3.org/2000/svg", ""))
    css = client.get(SECONDARY).text
    assert not re.search(r"url\(|@import|https?://", css)
    for fragment in (".regen-button", ".assistant:hover", ".assistant:focus-within",
                     "(hover: none)", ".recent-select", ".select-bar", "var(--muted)"):
        assert fragment in css


@pytest.mark.parametrize("kind", ["projeto", "nota", "lembrete", "ideia", "continuar"])
def test_mode_commands_match_fixed_rules(kind):
    from src.core import focus, projects
    from src.core.dispatch.notes import NOTE_CREATE_RE, NOTE_CREATE_FOLDER_RE
    from src.core.dispatch.reminders import REMINDER_TRIGGER_RE, REMINDER_BODY_RE

    shell = (Path(__file__).resolve().parents[1] / "src/ui/static/novo/shell.js").read_text()
    blocks = re.findall(r"const MODE_COMMANDS = \{\n(.*?)\n\};", shell, re.S)
    assert len(blocks) == 1
    commands = dict(re.findall(r"^  (\w+): '([^']*)',?$", blocks[0], re.M))
    assert commands == {
        "projeto": "projeto: ", "nota": "cria uma nota: ",
        "lembrete": "cria um lembrete de ", "ideia": "cria uma nota na pasta Ideias: ",
        "continuar": "onde a gente parou?",
    }
    sample = "pagar a conta amanhã às 9"
    command = commands[kind] + (sample if kind != "continuar" else "")
    if kind == "projeto":
        assert projects.parse_command(command) == ("active", sample, "pt")
    elif kind == "nota":
        match = NOTE_CREATE_RE.match(command)
        assert match and match.group("body") == sample
    elif kind == "ideia":
        match = NOTE_CREATE_FOLDER_RE.match(command)
        assert match and match.groupdict() == {"folder": "Ideias", "body": sample}
    elif kind == "lembrete":
        match = REMINDER_TRIGGER_RE.match(command)
        assert match and match.group("body") == sample
        body = REMINDER_BODY_RE.match(match.group("body"))
        assert body and body.groupdict() == {"title": "pagar a conta", "time_expr": "amanhã às 9"}
    else:
        assert focus.asks_where_was_i(command)


def test_mode_chooser_and_placeholders(client, page):
    shell = client.get(ASSETS[1]).text
    options = re.findall(r"const CREATE_OPTIONS = \{\n(.*?)\n\};", shell, re.S)
    assert len(options) == 1
    assert re.findall(r"(\w+): \['([^']+)', '([^']+)'\]", options[0]) == [
        ("projeto", "Projeto", "Nome do projeto"),
        ("nota", "Nota", "O que anotar?"),
        ("lembrete", "Lembrete", "Do que lembrar e quando? (ex.: pagar a conta amanhã às 9)"),
        ("ideia", "Ideia", "Qual é a ideia?"),
    ]
    for fragment in ("let currentMode = 'conversar', createChoice = null;",
                     "chooser.setAttribute('role', 'group');",
                     "chooser.setAttribute('aria-label', 'O que criar');",
                     "chooser.hidden = true;", "$('.modes').after(chooser);",
                     "Object.entries(CREATE_OPTIONS)", "button.type = 'button';",
                     "button.textContent = label;", "createChoice = kind;",
                     "input.placeholder = placeholder;",
                     "option.setAttribute('aria-pressed', String(option === button))"):
        assert fragment in shell
    mode = shell.split("function setMode(mode) {", 1)[1].split("\nfunction ", 1)[0]
    for fragment in ("currentMode = mode;", "createChoice = null;",
                     "chooser.hidden = mode !== 'criar';",
                     "button.setAttribute('aria-pressed', 'false')",
                     "'O que você quer criar?'", "'Envie para ver onde a gente parou'",
                     "'Escreva ou fale com o CORTEX...'"):
        assert fragment in mode
    assert check(page, shell) == demo_leftovers(page) == []
    css = client.get(SECONDARY).text
    assert ".create-chooser[hidden] { display: none; }" in css
    assert '.create-chooser button[aria-pressed="true"]' in css
    assert ".create-chooser button:focus-visible" in css
    assert not re.search(r"url\(|@import|https?://", css, re.I)


def test_mode_submit_and_resume_wiring(client, conversa):
    shell = client.get(ASSETS[1]).text
    submit = shell.split("$('#composer').addEventListener('submit', e => {", 1)[1].split("\n});", 1)[0]
    ordered = [
        "e.preventDefault();", "if (window.cortexConversa.inFlight) return;",
        "const text = input.value.trim();", "if (currentMode === 'criar') {",
        "if (!createChoice) {", "window.cortexProjetos.toast(",
        "'Escolha o que criar: projeto, nota, lembrete ou ideia.'", "return;",
        "if (!text) return;", "input.value = MODE_COMMANDS[createChoice] + text;",
        "} else if (currentMode === 'continuar' && !text) {",
        "input.value = MODE_COMMANDS.continuar;", "window.cortexConversa.submit();",
        "if (window.cortexConversa.inFlight && currentMode !== 'conversar') setMode('conversar');",
    ]
    remaining = submit
    for fragment in ordered:
        assert fragment in remaining
        remaining = remaining.split(fragment, 1)[1]
    assert submit.count("input.value =") == 2  # Ordinary text uses the existing path.
    assert shell.count("window.cortexConversa.submit()") == 1
    assert "flight = run; busy();" in conversa
    assert "if (addUser) addMessage('user', text);" in conversa
    assert "JSON.stringify({text, conversation_id: run.cid})" in conversa
    assert "if (b.dataset.mode === 'continuar' && !reduced && !card.hidden && card.getClientRects().length)" in shell
    resume = shell.split("$('#resume').addEventListener('click', () => {", 1)[1].split("\n});", 1)[0]
    assert "setMode('continuar');" in resume and "window.cortexProjetos.resume()" in resume


CENTRAL = "/ui/static/novo/central.js"


def test_central_no_lease_never_probes(client, monkeypatch):
    from src.ui import aggregator

    def forbidden_fetch(*args, **kwargs):
        pytest.fail("GET /ui/data without a lease must generate zero probe traffic")

    monkeypatch.setattr(aggregator, "lease_snapshot", lambda: {"active": False})
    monkeypatch.setattr(aggregator, "_netspeed_fetch", forbidden_fetch)
    response = client.get("/ui/data", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["net"]["status"] == "offline"


@pytest.fixture
def central(client):
    return client.get(CENTRAL).text


def test_central_open_only_and_freshness(client, central):
    assert re.findall(r"fetch\(([^,\n]+)", central) == [
        "'/ui/effects/seen'", "'/ui/gate/decide'", "'/ui/data'", "'/ui/effects'", "'/ui/gate'"]
    assert not re.search(r"setInterval|/ui/lease|visibilitychange|DOMContentLoaded", central)
    for fragment in ("credentials: 'same-origin'", "cache: 'no-store'",
                     "AbortSignal.timeout(10000)", "++sequence", "version !== sequence",
                     "if (!response.ok) throw", "await response.json()",
                     "window.cortexCentral = {openHub, openDetail}"):
        assert fragment in central
    # Only the exported opening paths may start a read, never script evaluation.
    assert re.findall(r"\bread\(([^)]*)\)", central) == ["id", "", "id"]
    assert re.findall(r"window\.\w+", central) == ["window.cortexCentral"]
    shell = client.get(ASSETS[1]).text
    hub = shell.split("function openHub(option) {", 1)[1].split("\n}", 1)[0]
    detail = shell.split("$$('.system-option').forEach", 1)[1].split("\n}));", 1)[0]
    assert "window.cortexCentral.openHub();" in hub
    assert "window.cortexCentral.openDetail(button.id.slice(7));" in detail
    assert "['internet', 'network', 'services', 'studier', 'audit', 'costs', 'voice', 'memory']" in central
    assert "version === sequence" in central  # Failed older reads cannot erase newer data.


def test_central_real_states_and_detail(central):
    for fragment in (
        "Permissão temporária de acesso", "Aberto · ", " min", "Fechado",
        "Acesso à internet", "Expira em ", "Expira às", "Tempo restante",
        "Sem permissão ativa", "Permissão ativa", "Nenhuma",
        "O Cortex só sai para a internet com esta permissão. Ela é dada na conversa, quando você diz sim a uma busca; abrir este painel não muda nada.",
        "Velocidade medida pelo Cortex", " Mbps", " ms", "Velocidade", "Latência",
        "Latência (TTFB)", "Não medida", "Medido às", "Erro na medição",
        "Sonda leve (~100 KB), no máximo uma por minuto e só com a permissão de internet aberta. É um piso honesto, não um speedtest.",
        "A permissão está aberta, mas a sonda falhou às ", " — a internet pode estar fora.",
        "Sem permissão de internet o Cortex não gera tráfego, então nada é medido. Isso não quer dizer que o Mac está sem internet.",
        "Serviços do Cortex no launchd", " carregados", "Serviços carregados",
        "Todos carregados", " não carregado(s)", "pid ", "carregado, ocioso",
        " · saída ", "não carregado", "desligado por escolha",
        "Serviços agendados ficam carregados sem processo entre uma execução e outra — isso é normal. Os desligados por escolha não contam.",
        "Fila de estudo e revisão", " na fila · ", " em revisão", "20+", "20 ou mais",
        "Itens na fila", "Na fila", "Aguardando revisão", "Aprovados", "Próximos estudos",
        "A revisão é feita na conversa; abrir este painel não aprova nada.",
        "Lido do Cortex às ", ". Fechar este painel preserva sua conversa; “Retomar central” volta à lista.",
        "Indisponível", "Não consegui ler o estado agora.",
        "metric-hero", "metric-hero-label", "metric-value", "metric-state", "health-dot",
        "metric-description", "metric-facts", "metric-footnote", "availability-bar",
        "rows.filter(row => !row.deferred)", "required.filter(row => !row.missing)",
        "loaded / required.length * 100", "var(--green)", "row.exit_code != null",
        "r >= 20", ".slice(0, 10)", "knowledge_approved_count", "host.generated_at",
        "getHours()", "getMinutes()", "textContent", "replaceChildren()",
    ):
        assert fragment in central


def test_central_omits_empty_facts(central):
    detail = central.split("function detail(item, readAt) {", 1)[1].split("\nfunction option", 1)[0]
    assert "if (item.facts.length) area.append(facts(item.facts));" in detail
    assert detail.count("facts(item.facts)") == 1


def test_central_focus_after_trust_reread(central):
    ack = central.split("button.addEventListener('click', async () => {", 1)[1].split("\n    area.append(button)", 1)[0]
    success = ack.split("} catch {", 1)[0]
    decision = central.split("async function decide(row, decision, button) {", 1)[1].split("\n    // Pending", 1)[0]
    final = decision.split("} finally {", 1)[1]
    for path, title in ((success, "Últimas ações"), (final, "Fila de fundo")):
        target = f"if (version === sequence) focusReturn = '{title}';"
        assert path.index("if (!$('#metric-detail').open) return;") < path.index(target)
        assert path.index(target) < path.index("if (version === sequence) openDetail('audit');")
    detail = central.split("function detail(item, readAt) {", 1)[1].split("\nfunction option", 1)[0]
    assert re.search(r"if \(focusReturn && \$\('#metric-detail'\).open &&\s*"
                     r"!\$\('#metric-detail'\).contains\(document.activeElement\)\) \{\s*"
                     r"const heading = \[\.\.\.area.querySelectorAll\('h3'\)\]"
                     r".find\(item => item.textContent === focusReturn\);\s*"
                     r"if \(heading\) \{\s*heading.tabIndex = -1; heading.focus\(\);\s*\}\s*"
                     r"\}\s*focusReturn = '';\s*\}$", detail)


def test_central_safe_rendering(client, central):
    assert central.startswith('"use strict";\n(() => {\n') and central.endswith("})();\n")
    assert not re.search(r"\.innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|new\s+Function|document\.write|\bon\w+\s*=|https?://", central)
    assert not re.search(r"url\(|@import|https?://", client.get(SECONDARY).text, re.I)


@pytest.mark.parametrize("fragments", [
    (
        "Histórico de atividade do Cortex", "Sem registro ainda", "Ainda não há arquivo de registro.",
        " hoje", "Registros hoje", "Dia em UTC, contado nas últimas linhas do registro",
        "Prévia curta de cada registro — não é a prévia de uma aprovação nem o recibo completo de uma ação.",
    ),
    (
        "Buscas na web por serviço medido", "Nenhum evento", " no mês", "Eventos no mês",
        "Contagem de chamadas, não dinheiro", "Valor no mês", "Sem preço registrado",
        "Desde o início", " eventos", "US$ ",
        "Conta as buscas na web feitas por um serviço medido. O modelo local não tem custo por uso e não entra aqui.",
    ),
    (
        "Conversas pela voz", "Sem registro ainda", " hoje", "Sessões de voz hoje",
        "Dia em UTC, contado nas últimas linhas do registro", "Tempo médio · ", " ms",
        " · incompleta", "(sem transcrição)",
        "Sessões que já terminaram ou pararam no meio. Isto não mostra se o microfone está ouvindo agora.",
    ),
    (
        "Memória de longo prazo", " documentos", "Documentos", "Só contagens", "Com embedding",
        "Fatos ativos", " de ", "Eventos", "Tamanho do banco", " MB",
        "Só contagens. Nenhum documento, fato ou lembrança é lido para mostrar este painel.",
    ),
], ids=["audit", "costs", "voice", "memory"])
def test_central_second_half_copy(central, fragments):
    for fragment in fragments:
        assert fragment in central


def test_central_audit_bounded_history(central):
    audit = central.split("if (id === 'audit') {", 1)[1].split("if (id === 'costs') {", 1)[0]
    assert "if (!audit.available)" in audit and "status: ''" in audit
    assert "status: 'ok'" in audit
    assert "summary: number(audit.count_today) + ' hoje'" in audit
    assert "value: number(audit.count_today)" in audit
    assert "state: 'Dia em UTC, contado nas últimas linhas do registro'" in audit
    # The aggregator already returns newest first; keep that order and its short preview.
    assert "(audit.recent || []).slice(0, 10).map" in audit and ".reverse(" not in audit
    assert "clock(row.ts) + ' · ' + row.stage + ' · ' + row.event" in audit
    assert "row.payload_preview || '—'" in audit


def test_central_costs_count_and_nullable_prices(central):
    costs = central.split("if (id === 'costs') {", 1)[1].split("if (id === 'voice') {", 1)[0]
    assert "if (!cost.available)" in costs and "summary: 'Nenhum evento'" in costs
    assert "status: ''" in costs and "status: 'ok'" in costs
    assert "summary: number(mtd.total_events) + ' no mês'" in costs
    assert "value: number(mtd.total_events)" in costs
    assert "number(allTime.total_events) + ' eventos'" in costs
    assert "(mtd.by_provider || []).map" in costs and "row.provider" in costs
    assert "number(row.events) + ' eventos · ' + number(row.units) + ' ' + row.unit_kind" in costs
    # Every currency prefix is guarded, including a real zero price; never invent USD.
    assert re.findall(r"[^\n]*US\$[^\n]*", central) == [
        "      facts: [['Valor no mês', mtd.total_cost_usd != null ? 'US$ ' + mtd.total_cost_usd : 'Sem preço registrado'],",
        "          (row.cost_usd != null ? ' · US$ ' + row.cost_usd : '')])],",
    ]


def test_central_voice_sessions_and_non_null_averages(central):
    voice = central.split("if (id === 'voice') {", 1)[1].split("if (id === 'memory') {", 1)[0]
    assert "if (!voice.available)" in voice and "summary: 'Sem registro ainda'" in voice
    assert "status: ''" in voice and "status: 'ok'" in voice
    assert "summary: number(voice.count_today) + ' hoje'" in voice
    assert "value: number(voice.count_today)" in voice
    assert "state: 'Dia em UTC, contado nas últimas linhas do registro'" in voice
    assert "Object.entries(voice.averages || {}).filter(([, ms]) => ms != null)" in voice
    assert ".map(([phase, ms]) => ['Tempo médio · ' + phase, number(ms) + ' ms'])" in voice
    assert "(voice.recent || []).slice(0, 10).map" in voice
    assert "row.ts_local" in voice and "row.complete === false ? ' · incompleta' : ''" in voice
    assert "row.text || '(sem transcrição)'" in voice


def test_central_memory_counts_and_missing_numbers(central):
    memory = central.split("if (id === 'memory') {", 1)[1].split("const studier", 1)[0]
    assert "if (!memory.available) return unavailable();" in memory
    assert "status: 'ok', summary: number(counts.documentos) + ' documentos'" in memory
    assert "value: number(counts.documentos)" in memory
    for field in ("documentos_with_embedding", "triplas_active", "triplas", "eventos"):
        assert "number(counts." + field + ")" in memory
    assert "number(counts.triplas_active) + ' de ' + number(counts.triplas)" in memory
    assert "Number.isFinite(memory.db_bytes) ? (memory.db_bytes / 1048576).toFixed(1) + ' MB' : '—'" in memory
    assert "return Number.isFinite(value) ? value : '—';" in central


def test_central_trust_reads_and_acknowledgement(central):
    assert re.findall(r"fetch\(([^,\n]+)", central) == [
        "'/ui/effects/seen'", "'/ui/gate/decide'", "'/ui/data'", "'/ui/effects'", "'/ui/gate'"]
    assert re.findall(r"method:\s*'([^']+)'", central) == ["POST", "POST"]
    assert "body: JSON.stringify({ids})" in central
    assert "headers: {'Content-Type': 'application/json'}" in central
    assert "Promise.all(" in central and "catch { return null; }" in central
    assert "const version = ++sequence" in central and "version !== sequence" in central
    assert "credentials: 'same-origin'" in central and "cache: 'no-store'" in central
    assert "AbortSignal.timeout(10000)" in central
    assert "const ids = notices.map(row => row.id)" in central
    assert "button.disabled = true" in central
    assert "if (version === sequence) openDetail('audit')" in central
    assert not re.search(r"setInterval|\.innerHTML|insertAdjacentHTML|outerHTML|\beval\s*\(|"
                         r"new\s+Function|document\.write|https?://", central)
    # The CLI is explanatory text only, never a call or a queue write.
    assert central.count("src.gate_review") == 1
    assert not re.search(r"exec\s*\(|spawn\s*\(|XMLHttpRequest|sendBeacon", central)


def test_central_trust_states_copy_and_whole_preview(client, central):
    for name, expected in (
        ("effectStates", {"preparando": "preparando", "esperando": "esperando",
                          "executando": "em andamento", "feito": "feito",
                          "cancelado": "cancelado", "falhou": "falhou", "desconhecido": "não sei"}),
        ("queueStatuses", {"queued": "na fila", "approved": "aprovado", "rejected": "rejeitado",
                           "executed": "executado", "failed": "falhou", "unknown": "não sei"}),
    ):
        block = central.split(f"const {name} = {{", 1)[1].split("};", 1)[0]
        assert dict(re.findall(r"(\w+): '([^']+)'", block)) == expected
    for text in (
        "Não sei", "Últimas ações", "Fila de fundo", "Já conferi",
        "O Cortex não sabe se estas ações aconteceram — a resposta se perdeu ou o processo parou. Ele não repete sozinho: confira no app (Mail, Calendário, Lembretes, Notas).",
        "Ações pedidas em segundo plano esperam sua aprovação. Leia a prévia inteira e aprove ou rejeite cada uma aqui; o terminal (python -m src.gate_review) continua valendo.",
        "Não consegui marcar como visto.", "Nenhuma ação registrada.", "Nada na fila de fundo.",
        "Fila de fundo indisponível agora.", "Não consegui ler as ações agora.",
        "Não consegui ler a fila de fundo agora.", "Não consegui ler o histórico de atividade agora.",
        " · no reinício", " · processo parado", " · na fila desde ", " não sei", " na fila de fundo",
    ):
        assert text in central
    for fragment in ("notices.length || pending.length", "item.status = 'warning'",
                     "row.owner_alive === false", "row.effect_owner_alive === false",
                     "row.restart", "row.detail || '—'", "row.reason || '—'",
                     "effects.effects.slice(0, 10)", "gate.recent.filter(row => row.status !== 'queued').slice(0, 10)",
                     "for (const row of gate.pending)", "node('pre', row.preview ?? '—', 'metric-preview')"):
        assert fragment in central
    assert not re.search(r"preview[^;\n]*(?:slice|substring)|(?:slice|substring)[^;\n]*preview", central)
    css = client.get(SECONDARY).text
    assert ".metric-ack" in css and ".metric-preview" in css
    assert "white-space: pre-wrap" in css and "overflow-wrap: anywhere" in css


def test_central_gate_buttons_belong_to_whole_pending_preview(client, central):
    pending = central.split("for (const row of gate.pending) {", 1)[1].split("\n    const recent", 1)[0]
    preview = "node('pre', row.preview ?? '—', 'metric-preview')"
    for name, label in (("approve", "Aprovar"), ("reject", "Rejeitar")):
        creation = f"const {name} = node('button', '{label}', 'metric-ack')"
        assert central.count(creation) == 1 and pending.index(preview) < pending.index(creation)
        assert f"{name}.type = 'button'" in pending
        assert f"{name}.setAttribute('aria-label', '{label} ' + row.label + ', na fila desde ' + clock(row.queued_at))" in pending
        assert f"{name}.addEventListener('click', () => decide(row, '{name}', {name}))" in pending
        assert f"{name}.disabled = deciding" in pending
    assert "actions.append(approve, reject);" in pending and "area.append(actions);" in pending
    css = client.get(SECONDARY).text
    row_style = css.split(".metric-decisions {", 1)[1].split("}", 1)[0]
    assert "display: flex" in row_style and "flex-wrap: wrap" in row_style


def test_central_gate_decision_confirmed_once_and_reread(central):
    decision = central.split("async function decide(row, decision, button) {", 1)[1].split("\n    // Pending", 1)[0]
    assert decision.index("if (!confirm(") < decision.index("fetch('/ui/gate/decide'")
    assert central.count("fetch('/ui/gate/decide'") == decision.count("fetch(") == 1
    for fragment in ("if (deciding || button.disabled) return;", "const version = sequence",
                     "method: 'POST'", "credentials: 'same-origin'", "cache: 'no-store'",
                     "headers: {'Content-Type': 'application/json'}", "AbortSignal.timeout(60000)",
                     "body: JSON.stringify({id: row.id, decision, lang: 'pt'})",
                     "deciding = true", "item.disabled = true", "'Aprovando…'", "'Rejeitando…'",
                     "response.status === 200", "response.status === 404",
                     "response.status === 409", "response.status === 503"):
        assert fragment in decision
    assert not re.search(r"\b(?:while|for)\s*\(|setTimeout\(|\bread\(", decision)
    catch, final = decision.split("} catch {", 1)[1].split("} finally {", 1)
    assert "fetch(" not in catch and "toast(" in catch
    for fragment in ("deciding = false", "item.disabled = false", "button.textContent = label",
                     "if (!$('#metric-detail').open) return;",
                     "if (version === sequence) openDetail('audit');"):
        assert fragment in final
    # Query the current detail again on completion, even if it was reopened during the POST.
    assert decision.count("area.querySelectorAll('.metric-decisions button')") == 2


def test_central_gate_decision_exact_copy(central):
    for text in (
        r"'Aprovar e executar agora: ' + row.label + '?\n\nO Cortex faz exatamente o que está na prévia, uma vez só.'",
        r"'Rejeitar ' + row.label + '?\n\nEla sai da fila e não será executada.'",
        "executed: 'Aprovado e executado.'", "failed: 'Aprovado, mas a ação falhou.'",
        "unknown: 'Aprovado, mas o Cortex não sabe se a ação aconteceu — confira no app.'",
        "rejected: 'Rejeitado — não vai rodar.'",
        "Essa ação não está mais na fila.", "Essa ação já tinha sido decidida.",
        "Não consegui registrar a decisão — nada foi executado.",
        "Não sei se a decisão chegou — relendo a fila. Não vou reenviar.",
        "Ações pedidas em segundo plano esperam sua aprovação. Leia a prévia inteira e aprove ou rejeite cada uma aqui; o terminal (python -m src.gate_review) continua valendo.",
    ):
        assert text in central


def test_central_gate_reject_only_displayed_id(client, ui_app, tmp_path, monkeypatch):
    from unittest.mock import Mock

    from src import gate_review
    from src.core.gate_queue import GateQueue

    queue = GateQueue(str(tmp_path / "gate.jsonl"))
    monkeypatch.setattr("src.core.gate_queue.gate_queue", queue)
    runner = Mock(side_effect=AssertionError("reject must execute nothing"))
    monkeypatch.setattr(ui_app.state.orchestrator, "_run_action", runner, raising=False)
    monkeypatch.setattr(gate_review, "_default_runner", runner)
    first = queue.enqueue("mac_sleep", {})
    second = queue.enqueue("send_email", {"to": "ana@example.com"})
    shown = client.get("/ui/gate", headers=HEADERS).json()
    assert shown["available"] and {row["id"] for row in shown["pending"]} == {first, second}
    displayed_id = next(row["id"] for row in shown["pending"] if row["id"] == first)
    response = client.post("/ui/gate/decide", headers=HEADERS,
                           json={"id": displayed_id, "decision": "reject", "lang": "pt"})
    assert response.status_code == 200 and response.json()["result"] == "rejected"
    after = client.get("/ui/gate", headers=HEADERS).json()
    assert [row["id"] for row in after["pending"]] == [second]
    assert next(row for row in after["recent"] if row["id"] == displayed_id)["status"] == "rejected"
    assert queue.status(second) == "queued"
    runner.assert_not_called()


def test_central_seen_preserves_unknown_and_background_queue(client, tmp_path, monkeypatch):
    from dataclasses import replace

    from src.core import effect_store
    from src.core.gate_queue import GateQueue

    store = effect_store.EffectStore(str(tmp_path / "effects.db"))
    queue = GateQueue(str(tmp_path / "gate.jsonl"))
    monkeypatch.setattr(effect_store, "store", store)
    monkeypatch.setattr("src.core.gate_queue.gate_queue", queue)
    gid = queue.enqueue("send_email", {"to": "ana@example.com"})
    effect = store.create("send_email", {"to": "ana@example.com"}, effect_id=gid)
    assert store.transition(effect.id, "preparando", "executando")
    assert store.transition(effect.id, "executando", "desconhecido", "resposta perdida")
    assert queue.resolve(gid, "approved") and queue.mark_unknown(gid)
    pending = queue.enqueue("mac_sleep", {})
    before_queue = Path(queue.path).read_bytes()
    before_gate = client.get("/ui/gate", headers=HEADERS).json()
    before_effect = store.get(gid)
    history = store.history(gid)
    assert [row["id"] for row in client.get("/ui/effects", headers=HEADERS).json()["notices"]] == [gid]
    response = client.post("/ui/effects/seen", headers=HEADERS, json={"ids": [gid]})
    assert response.status_code == 200 and response.json() == {"seen": [gid]}
    result = client.get("/ui/effects", headers=HEADERS).json()
    assert result["notices"] == [] and result["effects"][0]["state"] == "desconhecido"
    assert store.get(gid).seen_at is not None
    assert replace(store.get(gid), seen_at=None) == before_effect
    assert store.history(gid) == history
    assert Path(queue.path).read_bytes() == before_queue
    assert client.get("/ui/gate", headers=HEADERS).json() == before_gate
    assert queue.status(gid) == "unknown" and queue.status(pending) == "queued"
