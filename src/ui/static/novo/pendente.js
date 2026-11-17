"use strict";
(() => {
const $ = s => document.querySelector(s), chat = window.cortexConversa;
const card = node('section', null, 'pending-card'), notice = node('div', null, 'effect-notice');
let sequence = 0;
let effectRead = null;
card.hidden = notice.hidden = true;
$('#attachment-list').before(card, notice);
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function action(parent, label, callback) {
  const button = node('button', label);
  button.type = 'button';
  button.addEventListener('click', () => callback(button));
  parent.append(button);
}
function clock(value) {
  const date = new Date(value);
  if (!value || !Number.isFinite(date.getTime())) return '—';
  return String(date.getHours()).padStart(2, '0') + ':' + String(date.getMinutes()).padStart(2, '0');
}
async function transfer(pending, button) {
  if (chat.inFlight) return;
  const cid = chat.activeId, version = sequence;
  if (!cid || button.disabled) return;
  button.disabled = true;
  try {
    const response = await fetch('/ui/pending/transfer', {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'Content-Type': 'application/json'}, signal: AbortSignal.timeout(10000),
      body: JSON.stringify({pending_id: pending.id, conversation_id: cid})
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    if (version !== sequence || cid !== chat.activeId) return;
    if (!['transferred', 'already_here'].includes(data.reason)) window.cortexProjetos.toast(
      data.reason === 'stale' ? 'Essa confirmação já foi resolvida.' :
      data.reason === 'expired' ? 'Essa confirmação venceu.' : 'Não consegui trazer a confirmação.');
  } catch {
    if (version === sequence && cid === chat.activeId) window.cortexProjetos.toast('Não consegui trazer a confirmação.');
  } finally { refresh(); }
}
function renderPending(pending) {
  card.replaceChildren();
  if (!pending) return;
  const origin = pending.here ? 'pedido nesta conversa' :
    pending.origin?.channel === 'voice' ? 'pedido pela voz' : 'pedido em outra conversa';
  const details = node('details');
  details.append(node('summary', 'Ver a prévia inteira'), node('pre', pending.preview, 'pending-preview'));
  card.append(node('h2', pending.expired ? 'Confirmação vencida' : 'Esperando sua confirmação'),
    node('p', origin + ' · ' + (pending.expired ? 'venceu às ' : 'vale até ') + clock(pending.expires_at)), details);
  card.hidden = false;
  // Expiry is the route's truth; a read never clears or renews the preview.
  if (pending.expired) return;
  if (pending.here) {
    action(card, 'Sim', () => chat.submit('sim', pending.id));
    action(card, 'Não', () => chat.submit('não', pending.id));
  } else if (pending.origin?.channel === 'voice') {
    if (chat.activeId) action(card, 'Trazer para esta conversa', button => transfer(pending, button));
    else card.append(node('p', 'Abra ou comece uma conversa para trazer esta confirmação.'));
  } else if (pending.origin?.channel === 'ui' && pending.origin.conversation) {
    action(card, 'Abrir a conversa', () => chat.open(pending.origin.conversation));
  }
}
function renderNotices(data) {
  const count = data?.available ? data.notices.length : 0;
  notice.replaceChildren(); notice.hidden = count === 0;
  if (!count) return;
  notice.append(node('span', 'O Cortex não sabe se ' + count + ' ação(ões) aconteceram.'));
  // Reuse the shell's dialog navigation, title and return focus handling.
  action(notice, 'Ver', () => { $('#settings').click(); $('#option-audit').click(); });
}
function mark() {
  $('#messages').querySelectorAll('.effect-mark').forEach(element => element.remove());
  const cid = chat.activeId, data = effectRead?.cid === cid ? effectRead.data : null;
  if (!data?.available || !Array.isArray(data.conversation)) return;
  const rows = [...$('#messages').querySelectorAll('.message.assistant')], groups = new Map();
  for (const effect of data.conversation) {
    let row = effect.turn?.conversation_id === cid ?
      rows.find(row => row.dataset.turnId === effect.turn.turn_id) : null;
    const fallback = !row;
    if (!row && effect.asked_turn?.conversation_id === cid) {
      row = rows.find(row => row.dataset.turnId === effect.asked_turn.turn_id);
    }
    if (!row) continue;
    if (!groups.has(row)) groups.set(row, []);
    groups.get(row).push({label: effect.label, fallback});
  }
  for (const [row, effects] of groups) {
    const labels = effects.map(effect => effect.label), marker = node('div', null, 'effect-mark');
    let text = effects.length === 1 ? 'Não sei se isto aconteceu: ' + labels[0] :
      'Não sei se estas ' + effects.length + ' ações aconteceram: ' + labels.join(', ');
    if (effects.every(effect => effect.fallback)) text += ' (a resposta do “sim” se perdeu)';
    marker.append(node('span', text + '.'));
    action(marker, 'Ver', () => { $('#settings').click(); $('#option-audit').click(); });
    row.querySelector('.message-content').append(marker);
  }
}
async function read(url) {
  const response = await fetch(url, {
    credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(10000)
  });
  if (!response.ok) throw new Error('HTTP ' + response.status);
  return response.json();
}
async function refresh() {
  const version = ++sequence, cid = chat.activeId;
  // Hide old state immediately, including when one of the new reads fails.
  card.hidden = notice.hidden = true;
  $('#messages').querySelectorAll('.effect-mark').forEach(element => element.remove());
  // A save event must not restore old marks while a read is pending or failed.
  effectRead = null;
  const [pending, effects] = await Promise.allSettled([
    read('/ui/pending' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : '')),
    read('/ui/effects' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))
  ]);
  if (version !== sequence || cid !== chat.activeId) return;
  if (pending.status === 'fulfilled') renderPending(pending.value.pending);
  if (effects.status === 'fulfilled') {
    effectRead = {cid, data: effects.value};
    renderNotices(effects.value);
    mark();
  }
}
document.addEventListener('cortex-turn-saved', mark);
document.addEventListener('cortex-conversation', refresh);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
// Audit may acknowledge notices; read again when returning to the conversation.
$('#metric-detail').addEventListener('close', refresh);
window.cortexPendente = {refresh};
refresh();
})();
