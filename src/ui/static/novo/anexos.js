"use strict";
(() => {
const fileInput = document.querySelector('#file'), list = document.querySelector('#attachment-list');
const toast = message => window.cortexProjetos.toast(message);
let items = [];
fileInput.accept = '.txt,.md,.pdf,.docx';
function remove(item) {
  items = items.filter(entry => entry !== item);
  item.chip.remove();
}
function clear() {
  items = []; list.replaceChildren(); fileInput.value = '';
}
function take() {
  const ready = items.filter(item => item.ready).map(({name, text}) => ({name, text}));
  clear();
  return ready;
}
async function upload(file) {
  const chip = document.createElement('span'), label = document.createTextNode('carregando ' + file.name + '…');
  chip.className = 'attachment-chip';
  const removeButton = document.createElement('button');
  removeButton.type = 'button'; removeButton.textContent = '×';
  removeButton.setAttribute('aria-label', 'Remover ' + file.name);
  const item = {chip, ready: false};
  items.push(item); chip.append(label, removeButton); list.append(chip);
  removeButton.addEventListener('click', () => remove(item));
  try {
    const reader = new FileReader();
    const data_b64 = await new Promise((resolve, reject) => {
      reader.addEventListener('load', () => resolve(reader.result));
      reader.addEventListener('error', reject);
      reader.readAsDataURL(file);
    });
    // Removed/reset chips must never return when their asynchronous work finishes.
    if (!items.includes(item)) return;
    const response = await fetch('/ui/upload', {
      method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: file.name, data_b64}), signal: AbortSignal.timeout(120000)
    });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error(), {detail: data.detail});
    if (!items.includes(item)) return;
    item.name = data.name; item.text = data.text; item.ready = true;
    label.textContent = data.name + ' · ' + data.chars + ' caracteres' + (data.truncated ? ' · cortado' : '');
  } catch (error) {
    if (!items.includes(item)) return;
    remove(item);
    toast('Anexo: ' + (error.detail || 'não consegui ler esse arquivo'));
  }
}
document.querySelector('#attach').addEventListener('click', () => {
  if (window.cortexConversa.inFlight) { toast('Espere a resposta terminar.'); return; }
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  const files = [...fileInput.files]; fileInput.value = '';
  if (window.cortexConversa.inFlight) { toast('Espere a resposta terminar.'); return; }
  const slots = 2 - items.length;
  if (files.length > slots) toast('No máximo 2 anexos por mensagem.');
  files.slice(0, slots).forEach(upload);
});
window.cortexAnexos = {get uploading() { return items.some(item => !item.ready); }, take, clear};
})();
