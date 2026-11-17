"use strict";
(() => {
const dialog = document.querySelector('#metric-detail');
const area = document.querySelector('#metric-content');
let sequence = 0, writing = false;
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function options() {
  return {credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(10000)};
}
function busy() {
  area.querySelectorAll('.contacts-form button, .contacts-row button').forEach(button => {
    button.disabled = writing;
  });
}
async function read(message = '') {
  const version = ++sequence;
  area.replaceChildren();
  try {
    const response = await fetch('/ui/contacts', {...options(), method: 'GET'});
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (version !== sequence || !dialog.open) return;
    render(data.contacts, message);
  } catch {
    if (version !== sequence || !dialog.open) return;
    area.replaceChildren(node('p', 'Não consegui carregar os contatos.', 'metric-description'));
  }
}
function render(contacts, message) {
  const version = sequence, current = () => version === sequence && dialog.open;
  const form = node('form', null, 'contacts-form'), fields = {};
  form.autocomplete = 'off';
  form.noValidate = true; // Report blank names in the same live message as other errors.
  for (const [key, limit, placeholder, label] of [
    ['name', 200, 'Nome *', 'Nome'], ['email', 320, 'email', 'Email'],
    ['phone', 60, 'telefone', 'Telefone'], ['notes', 2000, 'notas', 'Notas']
  ]) {
    const input = node('input');
    input.name = key; input.type = 'text'; input.maxLength = limit;
    input.placeholder = placeholder; input.required = key === 'name';
    input.setAttribute('aria-label', label);
    fields[key] = input; form.append(input);
  }
  const save = node('button', 'Salvar', 'metric-ack');
  save.type = 'submit'; form.append(save);
  const status = node('p', message, 'contacts-message');
  status.setAttribute('aria-live', 'polite');
  const list = node('div', null, 'contacts-list');
  area.replaceChildren(node('p', 'Agenda local do Cortex: nome, email, telefone e notas.', 'metric-description'),
    form, status, list);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (writing) return;
    const value = key => fields[key].value.trim(), name = value('name');
    if (!name) { status.textContent = 'Nome é obrigatório.'; return; }
    writing = true; busy();
    try {
      const response = await fetch('/ui/contacts', {...options(), method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, email: value('email'), phone: value('phone') || null,
          notes: value('notes') || null})});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error();
      if (!current()) return;
      form.reset();
      await read(name + (data.action === 'updated' ? ' atualizado.' : ' adicionado.'));
    } catch {
      if (current()) status.textContent = 'Não consegui salvar.';
    } finally { writing = false; busy(); }
  });
  if (!contacts.length) list.append(node('p', 'Nenhum contato ainda — adicione um acima.'));
  for (const contact of contacts) {
    const row = node('div', null, 'contacts-row');
    const actions = node('div', null, 'contacts-actions');
    const edit = node('button', 'editar', 'metric-ack'), remove = node('button', 'apagar', 'metric-ack');
    edit.type = remove.type = 'button';
    edit.setAttribute('aria-label', 'Editar ' + contact.name);
    remove.setAttribute('aria-label', 'Apagar ' + contact.name);
    edit.addEventListener('click', () => {
      for (const key of Object.keys(fields)) fields[key].value = contact[key] || '';
      fields.name.focus();
      status.textContent = 'Editando ' + contact.name + ' — salve para atualizar.';
    });
    remove.addEventListener('click', async () => {
      if (writing) return;
      const name = contact.name;
      if (!confirm('Apagar ' + name + '?')) return;
      writing = true; busy();
      try {
        const response = await fetch('/ui/contacts/delete', {...options(), method: 'POST',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name})});
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error();
        if (current()) await read(name + ' apagado.');
      } catch {
        if (current()) status.textContent = 'Não consegui apagar ' + name + '.';
      } finally { writing = false; busy(); }
    });
    actions.append(edit, remove);
    row.append(node('strong', contact.name), node('span', contact.email || '—'),
      node('span', contact.phone || '—'), actions);
    list.append(row);
  }
  busy();
  // Successful writes rebuild the form; recover focus only if it fell outside.
  if (message && !dialog.contains(document.activeElement)) {
    fields.name.focus();
  }
}
// Bubble after the shell has cleared and opened the selected detail.
document.addEventListener('click', event => {
  const button = event.target.closest('.system-option, #metric-return');
  if (!button) return;
  ++sequence;
  if (button.id === 'option-contacts') read();
});
dialog.addEventListener('close', () => { ++sequence; });
})();
