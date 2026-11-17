"""Reply indicators follow the actual chat module's SSE lifecycle."""
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOVO = ROOT / "src/ui/static/novo"


@pytest.fixture
def chat_ui():
    source = (NOVO / "conversa.js").read_text()
    harness = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.parentNode = null;
    this.className = ''; this.attributes = {}; this.dataset = {}; this.style = {};
    this.value = ''; this._text = ''; this._html = '';
    this.classList = {
      contains: name => this.className.split(' ').includes(name),
      toggle: (name, enabled) => {
        const classes = new Set(this.className.split(' ').filter(Boolean));
        if (enabled) classes.add(name); else classes.delete(name);
        this.className = [...classes].join(' ');
      }
    };
  }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children.at(-1) || null; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(text) {
    this.replaceChildren(); this._html = '';
    if (this.tag === '#text') this._text = String(text);
    else if (text !== '') this.append(Object.assign(new Element('#text'), {_text: String(text)}));
  }
  get innerHTML() { return this._html; }
  set innerHTML(html) { this.replaceChildren(); this._html = html; }
  append(...children) {
    for (const child of children) {
      child.remove(); child.parentNode = this; this.children.push(child);
    }
  }
  prepend(child) { child.remove(); child.parentNode = this; this.children.unshift(child); }
  replaceChildren(...children) {
    this.children.forEach(child => { child.parentNode = null; });
    this.children = []; this._text = ''; this._html = ''; this.append(...children);
  }
  remove() {
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this);
    this.parentNode = null;
  }
  matches(selector) { return selector[0] === '.' ? this.classList.contains(selector.slice(1)) : this.tag === selector; }
  closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector); }
  querySelectorAll(selector) {
    const descendants = this.children.flatMap(c => [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return descendants;
    return descendants.filter(child => selector.split(',').some(part => {
      const pieces = part.trim().split(/\s+/), last = pieces.pop();
      return child.matches(last) && (!pieces.length || child.parentNode?.closest(pieces[0]));
    }));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name]; }
  cloneNode() { return new Element(this.tag); }
  addEventListener() {}
  focus() {}
}
const hosts = Object.fromEntries([
  '#message-input', '.sidebar-bottom small', '.privacy', '#send', '#content-scroll',
  '#messages', '.breadcrumb svg'
].map(selector => [selector, new Element('div')]));
const document = {
  querySelector: selector => hosts[selector],
  createElement: tag => new Element(tag),
  createTextNode: text => Object.assign(new Element('#text'), {_text: text}),
  dispatchEvent() {}
};
const localStorage = {getItem() { return null; }, setItem() {}, removeItem() {}};
const CustomEvent = class {};
const window = {
  cortexCore: {setLive() {}}, cortexProjetos: {toast() {}},
  cortexAnexos: {uploading: false, take: () => []}
};
let pendingRead, historyTurns = [];
const reader = {read: () => new Promise((resolve, reject) => { pendingRead = {resolve, reject}; })};
const fetch = async url => {
  if (url === '/ui/chat/stream') return {ok: true, body: {getReader: () => reader}};
  assert.ok(url.startsWith('/ui/conversations/'), 'Unexpected URL: ' + url);
  return {ok: true, json: async () => ({turns: historyTurns})};
};
const settle = async () => { for (let i = 0; i < 3; i++) await new Promise(setImmediate); };
const emit = async (...events) => {
  assert.ok(pendingRead, 'SSE reader should be waiting');
  const read = pendingRead; pendingRead = null;
  read.resolve({done: false, value: new TextEncoder().encode(
    events.map(event => 'data: ' + JSON.stringify(event) + '\n\n').join(''))});
  await settle();
};
const close = async error => {
  assert.ok(pendingRead);
  const read = pendingRead; pendingRead = null;
  if (error) read.reject(new Error(error)); else read.resolve({done: true});
  await settle();
};
const indicators = live => live.querySelectorAll('.reply-dots, .reply-cursor');
const assertIndicator = (live, kind) => {
  const nodes = indicators(live);
  assert.equal(nodes.length, 1, 'One active reply indicator');
  assert.equal(nodes[0].className, 'reply-' + kind);
  assert.equal(nodes[0].getAttribute('aria-hidden'), 'true');
  return nodes[0];
};
const turn = {id: 'saved', user: 'hello', reply: '**Finished**', status_mode: 'OFFLINE'};
"""

    def run(steps):
        script = harness + source + "\n(async () => {\n" + r"""
const chat = window.cortexConversa;
chat.init(() => {});
await chat.open('test');
chat.submit('hello');
await settle();
const live = hosts['#messages'].querySelectorAll('.assistant .message-content').at(-1);
assert.ok(live);
assert.equal(indicators(live).length, 0);
""" + steps + "\n})().catch(error => { console.error(error); process.exitCode = 1; });"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr

    return run


def test_status_dots_persist_then_become_streaming_cursor(chat_ui):
    chat_ui(r"""
await emit({type: 'status', label: '🧠 pensando…'});
const dots = assertIndicator(live, 'dots');
assert.equal(dots.children.length, 3);
assert.equal(live.querySelector('p').textContent, 'pensando…');
for (const label of ['🔍 buscando: <query>', '📖 lendo X…', '✍️ escrevendo X…', '🧠 pensando…']) {
  await emit({type: 'status', label});
  assert.equal(assertIndicator(live, 'dots'), dots, 'Keep the same animation across status updates');
  assert.equal(live.querySelector('p').textContent, label.replace(/^(?:🧠|🔍|📖|✍)\uFE0F?\s*/u, ''));
}
await emit({type: 'token', text: 'Hello <Pedro>'});
const cursor = assertIndicator(live, 'cursor');
assert.equal(cursor.textContent, '');
assert.equal(live.querySelector('p').textContent, 'Hello <Pedro>');
await emit({type: 'status', label: 'ignored'}, {type: 'token', text: '\nnext line'});
assert.equal(assertIndicator(live, 'cursor'), cursor, 'Keep the blink across tokens');
assert.equal(live.querySelector('p').textContent, 'Hello <Pedro>\nnext line');
await emit({type: 'final', turn});
await close();
assert.equal(indicators(live).length, 0);
""")


@pytest.mark.parametrize("phase", ["status", "token"])
@pytest.mark.parametrize("ending", ["final", "error", "stop", "throw", "disconnect", "switch"])
def test_reply_indicators_clear_on_every_exit(chat_ui, phase, ending):
    chat_ui(f"const phase = '{phase}', ending = '{ending}';\n" + r"""
await emit(phase === 'status' ? {type: 'status', label: '🧠 pensando…'} : {type: 'token', text: 'Partial'});
const active = assertIndicator(live, phase === 'status' ? 'dots' : 'cursor');
if (ending === 'final' || ending === 'error') {
  await emit(ending === 'final' ? {type: 'final', turn} : {type: 'error', detail: 'Failure'});
  assert.equal(indicators(live).length, 0, 'Clear before the connection closes');
  if (ending === 'final') assert.equal(live.innerHTML, '<p><strong>Finished</strong></p>');
  else assert.equal(live.querySelector('p').textContent, 'Failure');
  await emit({type: 'status', label: 'late'}, {type: 'token', text: 'late'});
  assert.equal(indicators(live).length, 0, 'Late frames cannot revive indicators');
  await close();
} else if (ending === 'stop' || ending === 'switch') {
  if (ending === 'stop') {
    chat.stop();
    assert.equal(live.querySelector('p').textContent, '— parado —');
    assert.equal(indicators(live).length, 0, 'Stop clears immediately');
  }
  chat.newConversation();
  assert.equal(indicators(live).length, 0, 'Detached messages must also be clear');
  await emit({type: 'status', label: 'late'}, {type: 'token', text: 'late'});
  assert.equal(hosts['#messages'].children.length, 0);
} else {
  await close(ending === 'throw' ? 'Reader failed' : null);
  assert.equal(live.querySelector('p').textContent,
    ending === 'throw' ? 'Reader failed' : 'conexão encerrada antes da resposta');
}
assert.equal(indicators(live).length, 0);
assert.equal(active.parentNode, null);
historyTurns = [turn];
await chat.open('history');
assert.equal(indicators(hosts['#messages']).length, 0, 'History has no live indicators');
assert.equal(hosts['#messages'].querySelector('.assistant .message-content').innerHTML,
  '<p><strong>Finished</strong></p>');
""")


@pytest.mark.parametrize("preference", ["system", "app"])
def test_reply_indicators_reduced_motion_is_steady(preference):
    css = (NOVO / "secundario.css").read_text()
    if preference == "system":
        rule = re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{"
                         r"\s*\.reply-dots i,\s*\.reply-cursor::after\s*\{([^}]+)\}", css)
    else:
        rule = re.search(r"body\.reduced-motion \.reply-dots i,\s*"
                         r"body\.reduced-motion \.reply-cursor::after\s*\{([^}]+)\}", css)
    assert rule, f"Missing {preference} reduced-motion rule for both indicators"
    assert "animation: none;" in rule[1]
    assert "opacity: 1;" in rule[1]


def test_reply_indicators_css_is_small_staggered_and_uses_theme_accent():
    css = (NOVO / "secundario.css").read_text()
    host = re.search(r"\.reply-dots,\s*\.reply-cursor\s*\{([^}]+)\}", css)
    assert host, "Indicators must not change the paragraph's footprint"
    for declaration in ("position: relative;", "display: inline-block;", "width: 0;", "height: 0;"):
        assert declaration in host[1]
    for selector in (r"\.reply-dots i", r"\.reply-cursor::after"):
        rule = re.search(selector + r"\s*\{([^}]+)\}", css)
        assert rule and "position: absolute;" in rule[1]
        assert "background: var(--accent);" in rule[1]
        assert "animation:" in rule[1]
    for index, delay in ((2, ".16s"), (3, ".32s")):
        rule = re.search(r"\.reply-dots i:nth-child\(" + str(index) + r"\)\s*\{([^}]+)\}", css)
        assert rule and f"animation-delay: {delay};" in rule[1]
