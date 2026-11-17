"use strict";
(() => {
const $ = (s) => document.querySelector(s);
const input = $('#message-input'), model = $('.sidebar-bottom small');
const privacy = document.createTextNode('');
$('.privacy').append(privacy);
let activeId = null, epoch = 0, flight = null, ready = Promise.resolve(), navigate;
const savedTurns = new WeakMap();
const STAGE_STATES = {
  thinking: 'thinking', web_search: 'searching', reading: 'reading',
  writing: 'writing', remembering: 'thinking'
};
// Re-asking an action can repeat it; unknown modes must fail closed.
const REGENERATE_MODES = new Set([
  'OFFLINE', 'REFLEX', 'ONLINE_RAG', 'AUTO_SEARCH', 'ONLINE_REQUEST', 'LOCAL_RAG'
]);
function changed() { document.dispatchEvent(new CustomEvent('cortex-conversation')); }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function sanitizeUrl(url) {
    var u = String(url).trim();
    if (/^(https?:|mailto:)/i.test(u)) return u;
    if (/^[/#]/.test(u)) return u;
    return null;
  }
  function inlineMd(s) {
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (m, txt, url) {
      var safe = sanitizeUrl(url);
      if (!safe) return txt;
      return '<a href="' + safe + '" target="_blank" rel="noopener noreferrer">' + txt + "</a>";
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_\w])_([^_\n]+)_/g, "$1<em>$2</em>");
    return s;
  }
  // Sentinel built at runtime so the source stays plain ASCII (a NUL never
  // appears in real model output, so it can't collide with prose).
  var NUL = String.fromCharCode(0);
  function renderMarkdown(src) {
    var text = String(src == null ? "" : src).replace(/\r\n?/g, "\n");
    var codeBlocks = [];
    text = text.replace(/```[^\n`]*\n([\s\S]*?)```/g, function (m, body) {
      codeBlocks.push(escapeHtml(body.replace(/\n$/, "")));
      return NUL + "B" + (codeBlocks.length - 1) + NUL;
    });
    text = escapeHtml(text);
    var inlineCodes = [];
    text = text.replace(/`([^`\n]+)`/g, function (m, c) {
      inlineCodes.push(c);
      return NUL + "I" + (inlineCodes.length - 1) + NUL;
    });
    var lines = text.split("\n");
    var out = [];
    var para = [], listType = null, listItems = [], quote = [];
    function flushPara() { if (para.length) { out.push("<p>" + inlineMd(para.join("<br>")) + "</p>"); para = []; } }
    function flushList() {
      if (listType) {
        out.push("<" + listType + ">" +
          listItems.map(function (li) { return "<li>" + inlineMd(li) + "</li>"; }).join("") +
          "</" + listType + ">");
        listType = null; listItems = [];
      }
    }
    function flushQuote() { if (quote.length) { out.push("<blockquote>" + inlineMd(quote.join("<br>")) + "</blockquote>"); quote = []; } }
    function flushAll() { flushPara(); flushList(); flushQuote(); }
    var cbRe = new RegExp("^" + NUL + "B(\\d+)" + NUL + "$");
    for (var k = 0; k < lines.length; k++) {
      var line = lines[k], m;
      if ((m = line.match(cbRe))) { flushAll(); out.push("<pre><code>" + codeBlocks[+m[1]] + "</code></pre>"); continue; }
      if (/^\s*$/.test(line)) { flushAll(); continue; }
      if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flushAll(); out.push("<h" + m[1].length + ">" + inlineMd(m[2]) + "</h" + m[1].length + ">"); continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushAll(); out.push("<hr>"); continue; }
      if ((m = line.match(/^\s*&gt;\s?(.*)$/))) { flushPara(); flushList(); quote.push(m[1]); continue; }
      flushQuote();
      if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) { flushPara(); if (listType !== "ul") { flushList(); listType = "ul"; } listItems.push(m[1]); continue; }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flushPara(); if (listType !== "ol") { flushList(); listType = "ol"; } listItems.push(m[1]); continue; }
      flushList();
      para.push(line);
    }
    flushAll();
    var html = out.join("\n");
    var icRe = new RegExp(NUL + "I(\\d+)" + NUL, "g");
    html = html.replace(icRe, function (m, i) { return "<code>" + inlineCodes[+i] + "</code>"; });
    // Safety net: any code-block placeholder not consumed as its own line
    // (e.g. a fence that landed mid-line) still renders as a block instead of
    // leaking the bare "B0" marker.
    var cbLeft = new RegExp(NUL + "B(\\d+)" + NUL, "g");
    html = html.replace(cbLeft, function (m, i) { return "<pre><code>" + codeBlocks[+i] + "</code></pre>"; });
    return html;
  }

function badges(usage) {
  privacy.textContent = model.textContent = '';
  if (!Array.isArray(usage?.online) || !Array.isArray(usage?.models)) return;
  privacy.textContent = usage.online.length ? 'Online' : 'Local';
  model.textContent = usage.models.join(' · ');
}
function remember(id) {
  activeId = id;
  try {
    if (id) localStorage.setItem('cortex-novo-conversation', id);
    else localStorage.removeItem('cortex-novo-conversation');
  } catch {}
  changed();
}
function busy() {
  $('#send').classList.toggle('is-stop', Boolean(flight && !flight.stopped));
  $('#send').setAttribute('aria-label', flight && !flight.stopped ? 'Parar resposta' : 'Enviar mensagem');
  $('#send').disabled = Boolean(flight?.stopped);
}
function scroll() { $('#content-scroll').scrollTop = $('#content-scroll').scrollHeight; }
function refreshRegen() {
  $('#messages').querySelectorAll('.regen-button').forEach(button => button.remove());
  const live = [...$('#messages').querySelectorAll('.assistant .message-content')].at(-1);
  const turn = savedTurns.get(live);
  if (!turn?.id || !turn.user?.trim() || turn.user.includes("\n\n📎 ") ||
      !REGENERATE_MODES.has(turn.status_mode)) return;
  const button = document.createElement('button');
  button.className = 'regen-button'; button.textContent = 'regenerar';
  button.title = 'descarta esta resposta e pergunta de novo';
  button.addEventListener('click', () => regenerate(live, turn));
  live.append(button);
}
function addMessage(role, text) {
  const row = document.createElement('div'), content = document.createElement('div');
  row.className = 'message ' + role;
  if (role === 'assistant') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.append($('.breadcrumb svg').cloneNode(true));
    row.append(avatar);
  }
  content.className = 'message-content';
  const author = document.createElement('span'), paragraph = document.createElement('p');
  author.className = 'author';
  author.textContent = role === 'assistant' ? 'CORTEX' : 'VOCÊ';
  paragraph.textContent = text;
  content.append(author, paragraph); row.append(content); $('#messages').append(row);
  refreshRegen(); scroll();
  return content;
}
function saved(live, turn) {
  const author = live.firstChild;
  live.innerHTML = renderMarkdown(turn.reply);
  live.prepend(author);
  if (typeof turn.id === 'string') live.closest('.message').dataset.turnId = turn.id;
  // Reconciliation may recover truth after Stop, but must not offer a retry.
  if (flight?.live !== live || !flight.stopped) savedTurns.set(live, turn);
  refreshRegen();
  badges(turn.usage); scroll();
  document.dispatchEvent(new CustomEvent('cortex-turn-saved'));
}
function clearProgress(run) {
  window.cortexAvatar?.removeHead(run.head);
  run.head = null;
  run.indicator?.remove();
  run.indicator = null;
  run.live?.lastChild?.classList.toggle('avatar-progress', false);
}
function replyProgress(run, text, kind, state = 'thinking') {
  if (!run.replyText) {
    run.replyText = document.createTextNode('');
    run.live.lastChild.replaceChildren(run.replyText);
  }
  run.live.lastChild.classList.toggle('avatar-progress', true);
  if (kind === 'reply-dots') {
    text = text.replace(/^\s*(?:🧠|🔍|📖|✍)\uFE0F?\s*/u, '');
    if (!run.head && window.cortexAvatar) {
      run.head = window.cortexAvatar.createHead(state);
      run.live.lastChild.prepend(run.head);
    } else if (run.head) window.cortexAvatar.updateHead(run.head, state);
  } else if (run.head) {
    window.cortexAvatar.removeHead(run.head); run.head = null;
  }
  run.replyText.textContent = text;
  // Keep the node so status updates and tokens do not restart its animation.
  if (run.indicator?.className === kind) return;
  run.indicator?.remove();
  const indicator = document.createElement('span');
  indicator.className = kind;
  indicator.setAttribute('aria-hidden', 'true');
  if (kind === 'reply-dots') {
    for (let i = 0; i < 3; i++) indicator.append(document.createElement('i'));
  }
  run.live.lastChild.append(indicator);
  run.indicator = indicator;
}
async function history(cid, timeout = 10000) {
  const response = await fetch('/ui/conversations/' + encodeURIComponent(cid), {
    credentials: 'same-origin', signal: AbortSignal.timeout(timeout), cache: 'no-store'
  });
  if (!response.ok) throw new Error('HTTP ' + response.status);
  return (await response.json()).turns || [];
}
async function load(id, enter = false) {
  const version = epoch;
  remember(id); $('#messages').replaceChildren(); badges(null);
  try {
    const turns = await history(id);
    if (version !== epoch) return;
    for (const turn of turns) {
      addMessage('user', turn.user); saved(addMessage('assistant', ''), turn);
    }
    badges(turns.at(-1)?.usage);
    if (enter) { navigate('chat'); input.focus(); }
    return true;
  } catch (error) {
    if (version !== epoch) return;
    if (error.message === 'HTTP 404') {
      remember(null); window.cortexProjetos.toast('Essa conversa não existe mais.'); return false;
    }
    window.cortexProjetos.toast('Não consegui abrir a conversa.');
  }
}
async function restore() {
  let id;
  try { id = localStorage.getItem('cortex-novo-conversation'); } catch {}
  if (id) return load(id);
}
function open(id) {
  if (flight) return;
  epoch++; window.cortexAvatar?.cancelGestures();
  ready = load(id, true); return ready;
}
async function ensureConversation(run) {
  if (activeId) return activeId;
  const response = await fetch('/ui/conversations', {
    method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
    body: '{}', signal: run.controller.signal
  });
  if (!response.ok) throw new Error('HTTP ' + response.status);
  const conversation = await response.json();
  if (run.epoch === epoch) remember(conversation.id);
  return conversation.id;
}
async function reconcile(run) {
  // Include a final read after the server's 60-second outcome wait.
  const deadline = Date.now() + 63000;
  for (let attempt = 0; attempt < 22 && Date.now() < deadline; attempt++) {
    if (run.epoch !== epoch) return;
    try {
      const turns = await history(run.cid, Math.max(1, Math.min(3000, deadline - Date.now())));
      // Never substitute an older or voice reply.
      const matches = turns.filter(turn => !run.known.has(turn.id) &&
        turn.source === 'ui' && turn.user === run.question &&
        Date.parse(turn.asked_ts) >= run.started && Date.parse(turn.asked_ts) <= run.stoppedAt);
      if (run.epoch !== epoch) return;
      if (matches.length === 1) { saved(run.live, matches[0]); return; }
    } catch {}
    if (deadline - Date.now() <= 3000) return;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
}
async function streamTurn(text, run, addUser = true) {
  try {
    await ready;
    if (run.epoch !== epoch) return;
    if (addUser && run.attachments?.length) addMessage('user', run.question);
    else if (addUser) addMessage('user', text);
    run.live = addMessage('assistant', '');
    badges(null);
    if (run.stopped) { run.live.lastChild.textContent = '— parado —'; return; }
    run.cid = run.cid || await ensureConversation(run);
    run.known = new Set((await history(run.cid)).map(turn => turn.id));
    if (run.stopped || run.epoch !== epoch) return;
    run.started = Date.now(); run.sent = true;
    const body = run.pendingId
      ? JSON.stringify({text, conversation_id: run.cid, pending_id: run.pendingId})
      : run.attachments?.length
        ? JSON.stringify({text, conversation_id: run.cid, attachments: run.attachments})
        : JSON.stringify({text, conversation_id: run.cid});
    const response = await fetch('/ui/chat/stream', {
      method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
      body, signal: run.controller.signal
    });
    if (!response.ok || !response.body) throw new Error('HTTP ' + response.status + ': ' + (await response.text()).slice(0, 160));
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (run.stopped || run.epoch !== epoch) break;
      if (done) { if (!run.finalized) throw new Error('conexão encerrada antes da resposta'); break; }
      buf += decoder.decode(value, {stream: true});
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, sep); buf = buf.slice(sep + 2);
        const dataLine = frame.split("\n").find(line => line.startsWith('data:'));
        if (!dataLine || run.finalized) continue;
        let evt;
        try { evt = JSON.parse(dataLine.slice(5).trim()); } catch { continue; }
        if (evt.type === 'turn') run.id = evt.id;
        else if (evt.type === 'status') {
          if (!run.streaming) {
            const state = window.cortexAvatar?.stateForStage(evt.stage) || 'thinking';
            replyProgress(run, evt.label || '', 'reply-dots', state);
            window.cortexCore.setLive(Object.hasOwn(STAGE_STATES, evt.stage) ? STAGE_STATES[evt.stage] : 'thinking', state);
          }
        } else if (evt.type === 'token') {
          run.streaming = true;
          run.tokens += evt.text || ''; replyProgress(run, run.tokens, 'reply-cursor');
          window.cortexCore.setLive('speaking');
          if (evt.text) window.cortexCore.pulseToken?.();
          scroll();
        } else if (evt.type === 'final' || evt.type === 'error') {
          clearProgress(run);
          if (evt.type === 'final') saved(run.live, evt.turn);
          else run.live.lastChild.textContent = evt.detail || 'erro';
          run.finalized = true; window.cortexCore.setLive('idle');
        }
      }
    }
  } catch (error) {
    if (!run.stopped && !run.finalized && run.epoch === epoch && run.live) run.live.lastChild.textContent = error.message;
  } finally {
    clearProgress(run);
    if (run.epoch === epoch) window.cortexCore.setLive('idle');
    if (run.stopped && run.sent) await reconcile(run);
    if (flight === run) { flight = null; busy(); input.focus(); changed(); }
  }
}
async function regenerate(live, turn) {
  if (flight || !activeId) return;
  const cid = activeId;
  const run = {epoch, cid, question: turn.user, tokens: '', controller: new AbortController()};
  flight = run; busy(); window.cortexCore.setLive('thinking');
  try {
    // Stop cancels the retry, but we still need to learn whether the pop succeeded.
    const response = await fetch('/ui/conversations/' + encodeURIComponent(cid) + '/pop_last', {
      method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({turn_id: turn.id}), signal: AbortSignal.timeout(10000)
    });
    if (run.epoch !== epoch) return;
    if (response.status === 409) { await load(cid); return; }
    if (!response.ok) throw new Error('HTTP ' + response.status);
    live.closest('.assistant').remove();
    await streamTurn(turn.user, run, false);
  } catch {
    if (run.epoch === epoch) window.cortexProjetos.toast('Não consegui regenerar a resposta.');
  } finally {
    if (flight === run) {
      flight = null; busy(); window.cortexCore.setLive('idle'); input.focus(); changed();
    }
  }
}
function submit(text = input.value.trim(), pendingId) {
  if (flight) return;
  if (window.cortexAnexos.uploading) {
    window.cortexProjetos.toast('Espere o anexo terminar de carregar.'); return;
  }
  let attachments = [];
  if (!pendingId) {
    attachments = window.cortexAnexos.take();
    if (!text && attachments.length) text = attachments.length === 1
      ? 'o que tem nesse arquivo?' : 'o que tem nesses arquivos?';
  }
  if (!text) return;
  // Match stored_user, including the server's empty-text and name normalization.
  const names = attachments.filter(item => item.text.trim()).map(item => item.name.trim() || 'arquivo');
  const question = text + (names.length ? "\n\n📎 " + names.join(', ') : '');
  const run = {epoch, question, attachments, pendingId, tokens: '', controller: new AbortController()};
  flight = run; busy(); navigate('chat'); window.cortexCore.setLive('thinking');
  if (!pendingId) window.cortexAvatar?.onSent(text);
  // Card decisions carry their displayed id and preserve any composer draft.
  if (!pendingId) { input.value = ''; input.style.height = 'auto'; }
  streamTurn(text, run);
}
function stop() {
  const run = flight;
  if (!run || run.stopped || run.finalized) return;
  run.stopped = true; run.stoppedAt = Date.now();
  if (run.id) fetch('/ui/chat/cancel', {
    method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: run.id}), signal: AbortSignal.timeout(5000)
  }).catch(() => {});
  run.controller.abort();
  clearProgress(run);
  if (run.live) run.live.lastChild.textContent = '— parado —';
  window.cortexCore.setLive('idle'); badges(null); busy();
}
function newConversation() {
  stop(); epoch++; flight = null; ready = Promise.resolve();
  window.cortexAvatar?.cancelGestures();
  remember(null); $('#messages').replaceChildren(); badges(null); busy();
  window.cortexCore.setLive('idle');
}
window.cortexConversa = {
  init(go) { navigate = go; ready = restore(); },
  open, get activeId() { return activeId; }, get inFlight() { return Boolean(flight); },
  submit, stop, newConversation, enter() { navigate('chat'); input.focus(); }
};
})();
