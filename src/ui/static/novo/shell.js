"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const input = $('#message-input');
const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
function preference(key) {
  try { return localStorage.getItem('cortex-novo-' + key); } catch { return null; }
}
function remember(key, value) {
  try { localStorage.setItem('cortex-novo-' + key, value); } catch {}
}
let motionChoice = preference('reduced-motion');
let reduced = motionChoice === null ? motionQuery.matches : motionChoice === 'true';
document.body.classList.toggle('light', preference('theme') === 'light');
function setReduced(value) {
  reduced = value;
  $('#reduce-motion').checked = value;
  document.body.classList.toggle('reduced-motion', value);
  window.cortexCore.setReduced(value);
}
setReduced(reduced);
$('#reduce-motion').addEventListener('change', e => {
  motionChoice = String(e.target.checked);
  remember('reduced-motion', motionChoice);
  setReduced(e.target.checked);
});
motionQuery.addEventListener('change', e => {
  // Follow the OS until the user explicitly chooses a preference.
  if (motionChoice === null) setReduced(e.matches);
});
$('#theme').addEventListener('click', () => {
  const light = document.body.classList.toggle('light');
  remember('theme', light ? 'light' : 'dark');
  window.cortexCore.redraw();
});
const savedIntensity = preference('intensity');
if (savedIntensity !== null && Number.isFinite(Number(savedIntensity))) {
  $('#intensity').value = Math.min(100, Math.max(20, Number(savedIntensity)));
}
window.cortexCore.setIntensity(Number($('#intensity').value) / 100);
$('#intensity').addEventListener('input', e => {
  window.cortexCore.setIntensity(Number(e.target.value) / 100);
  remember('intensity', e.target.value);
});
$('#state-select').addEventListener('change', e => {
  if ($('#controls').open) window.cortexCore.preview(e.target.value);
});
$('#gesture-select').addEventListener('change', e => {
  const controls = $('#controls'), gesture = e.target.value;
  e.target.value = '';
  if (!controls.open) return;
  // Run after the existing close listener has cleared the state preview.
  controls.addEventListener('close', () => window.cortexAvatar.play(gesture), {once: true});
  controls.close();
});
$('#controls').addEventListener('close', () => {
  $('#state-select').value = window.cortexCore.preview(null);
});
$('#preview-chat').addEventListener('click', () => {
  $('#controls').close();
  navigate(document.body.classList.contains('chat-view') ? 'home' : 'chat');
});
function setMenuOpen(open) {
  $('.sidebar').classList.toggle('open', open);
  $('.mobile-menu').setAttribute('aria-expanded', String(open));
}
setMenuOpen($('.sidebar').classList.contains('open'));
function navigate(next) {
  document.body.classList.toggle('chat-view', next === 'chat');
  document.body.classList.toggle('projects-view', next === 'projects');
  $('#projects-panel').hidden = next !== 'projects';
  $('#welcome').hidden = next !== 'home';
  $$('[data-page]').forEach(b => b.classList.toggle('active', b.dataset.page === next));
  setMenuOpen(false);
}
// These are ordinary messages, shared by both backend dispatch entry points.
const MODE_COMMANDS = {
  projeto: 'projeto: ',
  nota: 'cria uma nota: ',
  lembrete: 'cria um lembrete de ',
  ideia: 'cria uma nota na pasta Ideias: ',
  continuar: 'onde a gente parou?'
};
const CREATE_OPTIONS = {
  projeto: ['Projeto', 'Nome do projeto'],
  nota: ['Nota', 'O que anotar?'],
  lembrete: ['Lembrete', 'Do que lembrar e quando? (ex.: pagar a conta amanhã às 9)'],
  ideia: ['Ideia', 'Qual é a ideia?']
};
let currentMode = 'conversar', createChoice = null;
const chooser = document.createElement('div');
chooser.className = 'create-chooser';
chooser.setAttribute('role', 'group');
chooser.setAttribute('aria-label', 'O que criar');
chooser.hidden = true;
for (const [kind, [label, placeholder]] of Object.entries(CREATE_OPTIONS)) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', () => {
    createChoice = kind;
    input.placeholder = placeholder;
    [...chooser.children].forEach(option =>
      option.setAttribute('aria-pressed', String(option === button)));
    input.focus();
  });
  chooser.append(button);
}
$('.modes').after(chooser);
function setMode(mode) {
  currentMode = mode;
  createChoice = null;
  chooser.hidden = mode !== 'criar';
  [...chooser.children].forEach(button => button.setAttribute('aria-pressed', 'false'));
  $$('.mode').forEach(b => {
    const active = b.dataset.mode === mode;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  input.placeholder = mode === 'criar' ? 'O que você quer criar?' :
    mode === 'continuar' ? 'Envie para ver onde a gente parou' : 'Escreva ou fale com o CORTEX...';
}
function resetConversation() {
  window.cortexConversa.newConversation();
  $('#breadcrumb').textContent = '';
  window.cortexAnexos.clear();
  input.value = '';
  input.style.height = 'auto';
  setMode('conversar');
}
$$('[data-page]').forEach(b => b.addEventListener('click', () => navigate(b.dataset.page)));
$('.brand').addEventListener('click', e => { e.preventDefault(); navigate('home'); });
$('#new-chat').addEventListener('click', () => {
  resetConversation(); navigate('chat'); input.focus();
});
$$('.mode').forEach(b => b.addEventListener('click', () => {
  setMode(b.dataset.mode);
  const card = $('#resume-card');
  if (b.dataset.mode === 'continuar' && !reduced && !card.hidden && card.getClientRects().length) {
    card.animate([{borderColor:'var(--accent)'}, {borderColor:'var(--line)'}], {duration:800});
  }
  input.focus();
}));
$('#resume').addEventListener('click', () => {
  setMode('continuar');
  window.cortexProjetos.resume().then(entered => { if (entered) window.cortexConversa.enter(); });
});
window.cortexProjetos.init(navigate);
window.cortexConversa.init(navigate);
$('#composer').addEventListener('submit', e => {
  e.preventDefault();
  if (window.cortexConversa.inFlight) return;
  const text = input.value.trim();
  if (currentMode === 'criar') {
    if (!createChoice) {
      window.cortexProjetos.toast('Escolha o que criar: projeto, nota, lembrete ou ideia.');
      return;
    }
    if (!text) return;
    input.value = MODE_COMMANDS[createChoice] + text;
  } else if (currentMode === 'continuar' && !text) {
    input.value = MODE_COMMANDS.continuar;
  }
  window.cortexConversa.submit();
  // submit accepts synchronously; the next "sim" must stay an ordinary reply.
  if (window.cortexConversa.inFlight && currentMode !== 'conversar') setMode('conversar');
});
$('#send').addEventListener('click', e => {
  if ($('#send').classList.contains('is-stop')) {
    e.preventDefault(); window.cortexConversa.stop();
  }
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault(); $('#composer').requestSubmit();
  }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});
$('.mobile-menu').addEventListener('click', () =>
  setMenuOpen(!$('.sidebar').classList.contains('open')));
function consumeMenuTap(e) {
  if (!matchMedia('(max-width: 800px)').matches || !$('.sidebar').classList.contains('open')) return;
  e.preventDefault();
  e.stopPropagation();
  // Block focus on pointerdown, but keep the shield until the click is consumed.
  if (e.type === 'click') setMenuOpen(false);
}
$('.workspace').addEventListener('pointerdown', consumeMenuTap, true);
$('.workspace').addEventListener('click', consumeMenuTap, true);
let menuSwipe = null, menuSwipeClick = false;
function hasHorizontalMenuScroll(target) {
  for (let node = target; node; node = node.parentElement) {
    const overflowX = getComputedStyle(node).overflowX;
    if (node.scrollWidth > node.clientWidth && (overflowX === 'auto' || overflowX === 'scroll')) return true;
    if (node.matches('.workspace')) break;
  }
  return false;
}
function startMenuSwipe(e) {
  menuSwipe = null;
  menuSwipeClick = false;
  if (!matchMedia('(max-width: 800px)').matches || e.touches.length !== 1 ||
      document.querySelector('dialog[open]') || !(e.target instanceof Element)) return;
  if (e.target.closest('input, textarea, select, [contenteditable]') ||
      hasHorizontalMenuScroll(e.target)) return;
  const touch = e.touches[0];
  const open = $('.sidebar').classList.contains('open');
  if (!open && touch.clientX >= window.innerWidth / 2) return;
  // Safari owns swipes starting at its back-gesture edge; start a little inside.
  menuSwipe = { id: touch.identifier, x: touch.clientX, y: touch.clientY, open };
}
function moveMenuSwipe(e) {
  if (!menuSwipe) return;
  if (e.touches.length !== 1) {
    menuSwipe = null;
    return;
  }
  const touch = [...e.touches].find(touch => touch.identifier === menuSwipe.id);
  if (!touch) {
    menuSwipe = null;
    return;
  }
  const dx = touch.clientX - menuSwipe.x, dy = touch.clientY - menuSwipe.y;
  // Once vertical intent is clear, later sideways drift must not open the menu.
  // Ignore tiny initial jitter; leave all native scrolling and pinch zoom alone.
  if (Math.abs(dy) >= 10 && Math.abs(dx) <= 1.5 * Math.abs(dy)) menuSwipe = null;
}
function endMenuSwipe(e) {
  const swipe = menuSwipe;
  menuSwipe = null;
  if (!swipe || e.touches.length || !matchMedia('(max-width: 800px)').matches ||
      document.querySelector('dialog[open]') ||
      $('.sidebar').classList.contains('open') !== swipe.open) return;
  const touch = [...e.changedTouches].find(touch => touch.identifier === swipe.id);
  if (!touch) return;
  const dx = touch.clientX - swipe.x, dy = touch.clientY - swipe.y;
  if (Math.abs(dx) <= 1.5 * Math.abs(dy)) return;
  if (!swipe.open && dx >= 60) {
    setMenuOpen(true);
    menuSwipeClick = true;
  } else if (swipe.open && dx <= -60) {
    setMenuOpen(false);
    menuSwipeClick = true;
  }
}
function cancelMenuSwipe() {
  menuSwipe = null;
}
function resetMenuSwipeClick() {
  menuSwipeClick = false;
}
function consumeMenuSwipeClick(e) {
  if (!menuSwipeClick || !matchMedia('(max-width: 800px)').matches ||
      (e.detail === 0 && e.pointerType !== 'touch')) return;
  menuSwipeClick = false;
  e.preventDefault();
  e.stopPropagation();
}
// Touch Events keep tracking through native panning without changing touch-action.
document.addEventListener('touchstart', startMenuSwipe, { capture: true, passive: true });
document.addEventListener('touchmove', moveMenuSwipe, { capture: true, passive: true });
document.addEventListener('touchend', endMenuSwipe, { capture: true, passive: true });
document.addEventListener('touchcancel', cancelMenuSwipe, { capture: true, passive: true });
// Consume a swipe's compatibility click even after its overlay has disappeared.
// Reset on the next interaction so an absent compatibility click cannot eat a tap.
document.addEventListener('pointerdown', resetMenuSwipeClick, { capture: true, passive: true });
document.addEventListener('keydown', resetMenuSwipeClick, { capture: true, passive: true });
document.addEventListener('click', consumeMenuSwipeClick, true);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') setMenuOpen(false);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault(); navigate('chat'); input.focus();
  }
});
let lastOption = null;
function openHub(option) {
  for (const id of ['controls', 'metric-detail']) if ($('#' + id).open) $('#' + id).close();
  if (!$('#system-hub').open) $('#system-hub').showModal();
  window.cortexCentral.openHub();
  option?.focus();
}
function closeHubTo(dialog) {
  if ($('#system-hub').open) $('#system-hub').close();
  dialog.showModal();
}
$('#settings').addEventListener('click', () => openHub());
$$('.system-option').forEach(button => button.addEventListener('click', () => {
  lastOption = button;
  $('#metric-title').textContent = button.querySelector('.option-heading > span').textContent;
  $('#metric-content').replaceChildren();
  closeHubTo($('#metric-detail'));
  window.cortexCentral.openDetail(button.id.slice(7));
  $('#metric-return').focus();
}));
$('#metric-return').addEventListener('click', () => openHub(lastOption));
$('[data-return-hub]').addEventListener('click', () => openHub($('#open-appearance')));
$('#open-appearance').addEventListener('click', () => {
  // Opening Appearance keeps the live label; a preview requires a selection.
  $('#state-select').value = window.cortexCore.preview(null);
  closeHubTo($('#controls'));
});
$$('[data-close-dialog]').forEach(button =>
  button.addEventListener('click', () => $('#' + button.dataset.closeDialog).close()));
for (const id of ['system-hub', 'metric-detail', 'controls']) {
  $('#' + id).addEventListener('close', () => {
    if (!document.querySelector('dialog[open]')) $('#settings').focus();
  });
}
