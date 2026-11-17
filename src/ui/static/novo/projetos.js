"use strict";
(() => {
const $ = s => document.querySelector(s), chat = window.cortexConversa;
const sidebar = $('.nav-section:not(.recent)'), recents = $('.nav-section.recent');
const panel = $('#projects-panel'), card = $('#resume-card');
let sequence = 0, projects = [], conversations = [], cardProject = null, selected = null;
let navigate, toastTimer, recentSnapshot;
let selecting = false, deleting = false, selectedIds = new Set();
let deletingProject = false;
const selectMode = node('button', 'Selecionar', 'recent-select');
const selectBar = node('div', null, 'select-bar'), selectCount = node('span');
const selectDelete = node('button', 'Apagar'), selectCancel = node('button', 'Cancelar');
selectMode.title = 'Selecionar conversas para apagar';
recents.querySelector('h2').append(selectMode);
selectBar.append(selectCount, selectDelete, selectCancel); selectBar.hidden = true;
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function icon(symbol) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  use.setAttribute('href', symbol); svg.append(use); return svg;
}
function toast(text) {
  clearTimeout(toastTimer); $('#toast').textContent = text;
  $('#toast').classList.add('visible');
  toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 3000);
}
async function request(url, body) {
  const response = await fetch(url, {
    credentials: 'same-origin', method: body ? 'POST' : 'GET', cache: 'no-store',
    headers: body ? {'Content-Type': 'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(10000)
  });
  if (!response.ok) throw Object.assign(new Error('HTTP ' + response.status), {status: response.status});
  return response.json();
}
function selection() {
  sidebar.querySelectorAll('.project').forEach(row =>
    row.classList.toggle('selected', row.dataset.id === String(selected ?? cardProject?.id)));
  recents.querySelectorAll('.recent-chat').forEach(row =>
    row.classList.toggle('selected', selecting ? selectedIds.has(row.dataset.id) : row.dataset.id === chat.activeId));
  const cid = chat.activeId;
  $('#breadcrumb').textContent = cid ? projects.find(project => project.conversations.includes(cid))?.name || '' : '';
}
function renderCard(data) {
  cardProject = null; card.hidden = true;
  if (data.available === false || data.project == null) return;
  const project = data.project, tag = $('.resume-card .tag');
  cardProject = project;
  $('.resume-card h2').textContent = 'Vamos continuar “' + project.name + '”?';
  tag.textContent = project.conversations.includes(chat.activeId) ? 'Desta conversa' :
    project.active ? 'Projeto ativo' : '';
  tag.hidden = !tag.textContent;
  const step = data.progress?.next_step?.text;
  $('.resume-copy p').textContent = step != null ? 'Próximo passo: ' + step :
    'Nenhum próximo passo anotado. Dá pra anotar com “próximo:”.';
  card.hidden = false;
}
function unavailable() {
  if (!panel.querySelector('.projects-empty')) panel.append(node('p', '', 'projects-empty'));
  panel.querySelector('.projects-empty').textContent = 'Projetos indisponíveis agora.';
}
function renderProjects(data) {
  projects = data.available === false ? [] : data.projects;
  sidebar.replaceChildren(sidebar.querySelector('h2')); panel.replaceChildren(panel.querySelector('h2'));
  for (const project of projects) {
    const row = node('button', null, 'nav-item project'); row.dataset.id = project.id;
    row.append(icon('#i-folder'), document.createTextNode(project.name));
    if (project.id === data.active_id) row.append(node('span', null, 'tiny-dot'));
    const tile = node('button', null, 'project-tile'), copy = node('span');
    copy.append(node('strong', project.name), node('small', project.objective || ''));
    tile.append(icon('#i-folder'), copy, icon('#i-arrow'));
    for (const button of [row, tile]) button.addEventListener('click', () => selectProject(project.id));
    sidebar.append(row); panel.append(tile);
  }
  if (data.available === false) unavailable();
  else if (!projects.length) panel.append(node('p',
    'Nenhum projeto ainda. Diga “projeto: nome” numa conversa para criar um.', 'projects-empty'));
  // The duck's bookcase on the home stage shows this same list.
  window.cortexQuarto?.render(projects, selectProject, () => navigate('projects'));
}
function renderRecents(data) {
  conversations = data.conversations;
  selectedIds.forEach(id => { if (!conversations.some(c => c.id === id)) selectedIds.delete(id); });
  refreshSelectBar();
  // Keep title nodes alive across an open so the second click can rename.
  const snapshot = JSON.stringify(conversations);
  if (snapshot === recentSnapshot) return;
  recentSnapshot = snapshot; recents.replaceChildren(recents.querySelector('h2'));
  recents.append(selectBar);
  for (const conversation of conversations) {
    const wrapper = node('div', null, 'recent-row'), row = node('button', null, 'nav-item recent-chat');
    const title = node('span', conversation.title || 'Nova conversa'); row.dataset.id = conversation.id;
    row.append(icon('#i-file'), title);
    row.addEventListener('click', () => selecting ? toggleSelected(conversation.id) : chat.open(conversation.id));
    title.addEventListener('dblclick', () => { if (!selecting) rename(conversation.id, conversation.title); });
    if (selecting) {
      const box = node('input', null, 'recent-check'); box.type = 'checkbox';
      box.setAttribute('aria-label', 'Selecionar ' + title.textContent);
      box.checked = selectedIds.has(conversation.id); box.disabled = deleting;
      box.addEventListener('change', () => toggleSelected(conversation.id));
      wrapper.append(box);
    }
    const del = node('button', '×', 'recent-delete'); del.setAttribute('aria-label', 'Apagar conversa ' + title.textContent);
    del.addEventListener('click', () => remove(conversation.id));
    del.hidden = selecting;
    wrapper.append(row, del); recents.append(wrapper);
  }
  selection();
}
function refreshSelectBar() {
  const n = selectedIds.size;
  selectCount.textContent = n === 1 ? '1 selecionada' : n + ' selecionadas';
  selectDelete.disabled = n === 0 || deleting;
}
function toggleSelected(id) {
  if (deleting) return;
  if (selectedIds.has(id)) selectedIds.delete(id);
  else selectedIds.add(id);
  recents.querySelectorAll('.recent-row').forEach(wrapper => {
    const box = wrapper.querySelector('.recent-check');
    if (box) box.checked = selectedIds.has(wrapper.querySelector('.recent-chat').dataset.id);
  });
  refreshSelectBar(); selection();
}
function setSelecting(value) {
  selecting = value; selectedIds.clear();
  selectMode.textContent = selecting ? 'Concluir' : 'Selecionar';
  selectBar.hidden = !selecting; recents.classList.toggle('selecting', selecting);
  recentSnapshot = null; renderRecents({conversations});
}
async function deleteSelected() {
  if (chat.inFlight || deleting) return;
  const ids = [...selectedIds];
  if (!ids.length) return;
  const message = ids.length === 1
    ? 'Apagar 1 conversa? (a memória de longo prazo do Cortex fica intacta)'
    : 'Apagar ' + ids.length + ' conversas? (a memória de longo prazo do Cortex fica intacta)';
  if (!confirm(message)) return;
  deleting = true; refreshSelectBar();
  recents.querySelectorAll('.recent-check').forEach(box => { box.disabled = true; });
  try {
    await request('/ui/conversations/delete', {ids});
    if (ids.includes(chat.activeId)) chat.newConversation();
    setSelecting(false); refresh();
  } catch {
    // Keep the confirmed batch available to retry, even if Esc closed the mode.
    setSelecting(true); selectedIds = new Set(ids);
    toast('Não consegui apagar as conversas.');
  } finally {
    deleting = false; recentSnapshot = null; renderRecents({conversations});
  }
}
async function refresh(event) {
  // Only an explicit project choice is pinned; conversation navigation ends it.
  if (event?.type === 'cortex-conversation') selected = null;
  const version = ++sequence, cid = chat.activeId, id = selected;
  selection();
  let [projectList, recentList, resumeCard] = await Promise.allSettled([
    request('/ui/projects'), request('/ui/conversations'),
    request(id != null ? '/ui/projects/' + encodeURIComponent(id) :
      '/ui/resume' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))
  ]);
  if (version !== sequence) return;
  if (id != null && resumeCard.status === 'rejected' && resumeCard.reason.status === 404) {
    selected = null;
    [resumeCard] = await Promise.allSettled([
      request('/ui/resume' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))
    ]);
    if (version !== sequence) return;
  }
  if (projectList.status === 'fulfilled') renderProjects(projectList.value);
  else unavailable(); // Keep existing rows when the list request fails.
  if (recentList.status === 'fulfilled') renderRecents(recentList.value);
  renderCard(resumeCard.status === 'fulfilled' ? resumeCard.value : {available: false});
  selection();
}
async function selectProject(id) {
  const version = ++sequence; selected = id; selection(); navigate('home');
  renderCard({available: false});
  try {
    const data = await request('/ui/projects/' + encodeURIComponent(id));
    if (version !== sequence) return;
    renderCard(data);
  } catch (error) {
    if (version !== sequence) return;
    if (error.status === 404) { selected = null; await refresh(); }
    else renderCard({available: false});
  }
}
async function rename(id, current) {
  if (chat.inFlight) return;
  const answer = prompt("Renomear conversa:", current || ""), title = answer?.trim();
  if (!title) return;
  try { await request('/ui/conversations/' + encodeURIComponent(id) + '/rename', {title}); }
  catch { toast('Não consegui renomear a conversa.'); }
  refresh();
}
async function remove(id) {
  if (chat.inFlight) return;
  if (!confirm("Apagar esta conversa? (a memória de longo prazo do Cortex fica intacta)")) return;
  try { await request('/ui/conversations/' + encodeURIComponent(id) + '/delete', {}); }
  catch (error) {
    if (error.status !== 404) { toast('Não consegui apagar a conversa.'); return; }
  }
  if (id === chat.activeId) chat.newConversation();
  else refresh();
}
async function removeProject() {
  if (chat.inFlight || deletingProject || !cardProject) return;
  const project = cardProject, id = project.id, button = $('#delete-project');
  deletingProject = true; button.disabled = true;
  try {
    const preview = await request('/ui/projects/' + encodeURIComponent(id) + '/delete-preview');
    if (cardProject?.id !== id) { toast('O projeto mudou — tente de novo.'); return; }
    const n = preview.conversations, m = preview.memories;
    if (preview.ok === false) throw new Error('Preview failed');
    if (!confirm('Excluir o projeto “' + preview.name + '”? Isso apaga ' + n +
      (n === 1 ? ' conversa' : ' conversas') + ', as anotações e ' + m +
      (m === 1 ? ' memória' : ' memórias') + ' dessas conversas. Os arquivos salvos ficam. Não tem volta.')) return;
    const result = await request('/ui/projects/' + encodeURIComponent(id) + '/delete', {});
    if (project.conversations.includes(chat.activeId)) {
      if (result.ok) chat.newConversation();
      else if (result.conversations > 0) {
        // A partial failure can still have erased the conversation on screen.
        try { await request('/ui/conversations/' + encodeURIComponent(chat.activeId)); }
        catch (error) { if (error.status === 404) chat.newConversation(); }
      }
    }
    if (!result.ok) throw new Error('Deletion failed');
    toast('Projeto excluído.');
  } catch { toast('Não consegui excluir o projeto.'); }
  finally {
    deletingProject = false; button.disabled = false;
    await refresh();
  }
}
async function resume() {
  if (chat.inFlight) return;
  const project = cardProject;
  if (!project) return;
  if (project.conversations.includes(chat.activeId)) return true;
  let recent = conversations.find(c => project.conversations.includes(c.id));
  while (recent) {
    const opened = await chat.open(recent.id);
    if (opened !== false) return opened;
    // A linked conversation may have been deleted since the list read.
    conversations = conversations.filter(c => c.id !== recent.id);
    recent = conversations.find(c => project.conversations.includes(c.id));
  }
  chat.newConversation(); return true;
}
window.cortexProjetos = {toast, resume, init(go) {
  navigate = go;
  selectMode.addEventListener('click', () => setSelecting(!selecting));
  selectCancel.addEventListener('click', () => setSelecting(false));
  selectDelete.addEventListener('click', deleteSelected);
  $('#delete-project').addEventListener('click', removeProject);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && selecting) { e.preventDefault(); setSelecting(false); }
  });
  document.addEventListener('cortex-conversation', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
}};
})();
