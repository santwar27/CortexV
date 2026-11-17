"use strict";
(() => {
const $ = s => document.querySelector(s);
const ids = ['internet', 'network', 'services', 'studier', 'audit', 'costs', 'voice', 'memory'];
const descriptions = ['Permissão temporária de acesso', 'Velocidade medida pelo Cortex',
  'Serviços do Cortex no launchd', 'Fila de estudo e revisão', 'Histórico de atividade do Cortex',
  'Buscas na web por serviço medido', 'Conversas pela voz', 'Memória de longo prazo'];
let sequence = 0;
let deciding = false;
let focusReturn = '';
const effectStates = {preparando: 'preparando', esperando: 'esperando', executando: 'em andamento',
  feito: 'feito', cancelado: 'cancelado', falhou: 'falhou', desconhecido: 'não sei'};
const queueStatuses = {queued: 'na fila', approved: 'aprovado', rejected: 'rejeitado',
  executed: 'executado', failed: 'falhou', unknown: 'não sei'};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function clock(value) {
  // Net already supplies local HH:MM:SS; the other timestamps are ISO UTC.
  if (/^\d{2}:\d{2}:\d{2}$/.test(value)) return value.slice(0, 5);
  const date = new Date(value || NaN);
  return Number.isNaN(date.getTime()) ? '—' :
    String(date.getHours()).padStart(2, '0') + ':' + String(date.getMinutes()).padStart(2, '0');
}
function studyDate(value) {
  const date = new Date(value || NaN);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleDateString('pt-BR') + ' · ' + clock(value);
}
function dot(element, status) {
  element.className = 'health-dot' + (status ? ' ' + status : '');
  element.style.background = status ? '' : 'var(--muted)';
}
function unavailable() {
  return {summary: 'Indisponível', value: 'Indisponível', unit: '', state: '', status: '',
    text: 'Não consegui ler o estado agora.', facts: []};
}
function number(value) {
  // An absent measurement is not a measured zero.
  return Number.isFinite(value) ? value : '—';
}
function metric(data, id) {
  if (id === 'internet') {
    const lease = data.lease, minutes = lease.remaining_minutes + ' min';
    return {status: lease.active ? 'ok' : '', unit: 'Acesso à internet',
      summary: lease.active ? 'Aberto · ' + minutes : 'Fechado',
      value: lease.active ? 'Aberto' : 'Fechado',
      state: lease.active ? 'Expira em ' + minutes : 'Sem permissão ativa',
      facts: lease.active ? [['Expira às', clock(lease.expires_at)], ['Tempo restante', minutes]] :
        [['Permissão ativa', 'Nenhuma']],
      text: 'O Cortex só sai para a internet com esta permissão. Ela é dada na conversa, quando você diz sim a uma busca; abrir este painel não muda nada.'};
  }
  if (id === 'network') {
    const net = data.net, measured = clock(net.measured_at);
    if (net.status === 'ok') {
      const speed = net.mbps == null ? null : net.mbps + ' Mbps', latency = net.latency_ms + ' ms';
      return {status: 'ok', summary: speed ? speed + ' · ' + latency : latency,
        value: speed || latency, unit: speed ? 'Velocidade' : 'Latência', state: 'Medido às ' + measured,
        facts: [['Latência (TTFB)', latency], ['Velocidade', speed || 'Não medida'], ['Medido às', measured]],
        text: 'Sonda leve (~100 KB), no máximo uma por minuto e só com a permissão de internet aberta. É um piso honesto, não um speedtest.'};
    }
    if (net.status === 'error') return {status: 'warning', summary: 'Erro na medição',
      value: 'Erro na medição', unit: 'Velocidade', state: 'Medido às ' + measured, facts: [],
      text: 'A permissão está aberta, mas a sonda falhou às ' + measured + ' — a internet pode estar fora.'};
    if (net.status === 'offline') return {status: '', summary: 'Não medida', value: 'Não medida',
      unit: 'Velocidade', state: 'Sem permissão ativa', facts: [],
      text: 'Sem permissão de internet o Cortex não gera tráfego, então nada é medido. Isso não quer dizer que o Mac está sem internet.'};
    return unavailable();
  }
  if (id === 'services') {
    const rows = data.services, required = rows.filter(row => !row.deferred);
    if (!required.length) return unavailable();
    const loaded = required.filter(row => !row.missing).length, all = loaded === required.length;
    return {status: all ? 'ok' : loaded ? 'warning' : 'offline',
      summary: loaded + ' / ' + required.length + ' carregados', value: loaded + ' / ' + required.length,
      unit: 'Serviços carregados', state: all ? 'Todos carregados' : required.length - loaded + ' não carregado(s)',
      percent: loaded / required.length * 100,
      facts: rows.map(row => [row.label, row.missing ?
        (row.deferred ? 'desligado por escolha' : 'não carregado') : row.pid != null ?
        'pid ' + row.pid : 'carregado, ocioso' + (row.exit_code != null ? ' · saída ' + row.exit_code : '')]),
      text: 'Serviços agendados ficam carregados sem processo entre uma execução e outra — isso é normal. Os desligados por escolha não contam.'};
  }
  if (id === 'audit') {
    const audit = data.audit || {};
    if (!audit.available) return {status: '', summary: 'Sem registro ainda', value: 'Sem registro ainda',
      unit: 'Registros hoje', state: '', facts: [], text: 'Ainda não há arquivo de registro.'};
    return {status: 'ok', summary: number(audit.count_today) + ' hoje', value: number(audit.count_today),
      unit: 'Registros hoje', state: 'Dia em UTC, contado nas últimas linhas do registro',
      // The snapshot already puts the newest records first and shortens their payloads.
      facts: (audit.recent || []).slice(0, 10).map(row =>
        [clock(row.ts) + ' · ' + row.stage + ' · ' + row.event, row.payload_preview || '—']),
      text: 'Prévia curta de cada registro — não é a prévia de uma aprovação nem o recibo completo de uma ação.'};
  }
  if (id === 'costs') {
    const cost = data.cost || {}, mtd = cost.mtd || {}, allTime = cost.all_time || {};
    const text = 'Conta as buscas na web feitas por um serviço medido. O modelo local não tem custo por uso e não entra aqui.';
    if (!cost.available) return {status: '', summary: 'Nenhum evento', value: 'Nenhum evento',
      unit: 'Eventos no mês', state: 'Contagem de chamadas, não dinheiro', facts: [], text};
    return {status: 'ok', summary: number(mtd.total_events) + ' no mês', value: number(mtd.total_events),
      unit: 'Eventos no mês', state: 'Contagem de chamadas, não dinheiro',
      facts: [['Valor no mês', mtd.total_cost_usd != null ? 'US$ ' + mtd.total_cost_usd : 'Sem preço registrado'],
        ['Desde o início', number(allTime.total_events) + ' eventos'],
        ...(mtd.by_provider || []).map(row => [row.provider,
          number(row.events) + ' eventos · ' + number(row.units) + ' ' + row.unit_kind +
          (row.cost_usd != null ? ' · US$ ' + row.cost_usd : '')])],
      text};
  }
  if (id === 'voice') {
    const voice = data.voice || {};
    const text = 'Sessões que já terminaram ou pararam no meio. Isto não mostra se o microfone está ouvindo agora.';
    if (!voice.available) return {status: '', summary: 'Sem registro ainda', value: 'Sem registro ainda',
      unit: 'Sessões de voz hoje', state: '', facts: [], text};
    return {status: 'ok', summary: number(voice.count_today) + ' hoje', value: number(voice.count_today),
      unit: 'Sessões de voz hoje', state: 'Dia em UTC, contado nas últimas linhas do registro',
      facts: [...Object.entries(voice.averages || {}).filter(([, ms]) => ms != null)
        .map(([phase, ms]) => ['Tempo médio · ' + phase, number(ms) + ' ms']),
        ...(voice.recent || []).slice(0, 10).map(row =>
          [row.ts_local + (row.complete === false ? ' · incompleta' : ''), row.text || '(sem transcrição)'])],
      text};
  }
  if (id === 'memory') {
    const memory = data.memory || {}, counts = memory.counts || {};
    if (!memory.available) return unavailable();
    return {status: 'ok', summary: number(counts.documentos) + ' documentos', value: number(counts.documentos),
      unit: 'Documentos', state: 'Só contagens',
      facts: [['Com embedding', number(counts.documentos_with_embedding)],
        ['Fatos ativos', number(counts.triplas_active) + ' de ' + number(counts.triplas)],
        ['Eventos', number(counts.eventos)],
        ['Tamanho do banco', Number.isFinite(memory.db_bytes) ? (memory.db_bytes / 1048576).toFixed(1) + ' MB' : '—']],
      text: 'Só contagens. Nenhum documento, fato ou lembrança é lido para mostrar este painel.'};
  }
  const studier = data.studier, q = studier.queue_pending.length, r = studier.knowledge_pending.length;
  return {status: r > 0 ? 'warning' : 'ok', summary: q + ' na fila · ' + (r >= 20 ? '20+' : r) + ' em revisão',
    value: q, unit: 'Itens na fila', state: '',
    facts: [['Na fila', q], ['Aguardando revisão', r >= 20 ? '20 ou mais' : r],
      ['Aprovados', studier.knowledge_approved_count]],
    lists: [['Próximos estudos', studier.queue_pending.slice(0, 10).map(item => [item.topic, studyDate(item.added_at)])],
      ['Aguardando revisão', studier.knowledge_pending.slice(0, 10).map(item => [item.name, studyDate(item.mtime)])]],
    text: 'A revisão é feita na conversa; abrir este painel não aprova nada.'};
}
function facts(rows) {
  const list = node('dl', null, 'metric-facts');
  for (const [label, value] of rows) {
    const row = node('div'); row.append(node('dt', label), node('dd', value)); list.append(row);
  }
  return list;
}
function toast(text) {
  const element = $('#toast');
  element.textContent = text; element.classList.add('visible');
  setTimeout(() => element.classList.remove('visible'), 3000);
}
function trustDetail(area, effects, gate) {
  const notices = effects?.available ? effects.notices : [];
  if (notices.length) {
    area.append(node('h3', 'Não sei', 'metric-list-title'),
      node('p', 'O Cortex não sabe se estas ações aconteceram — a resposta se perdeu ou o processo parou. Ele não repete sozinho: confira no app (Mail, Calendário, Lembretes, Notas).', 'metric-description'),
      facts(notices.map(row => [row.label + ' · ' + clock(row.updated_at) +
        (row.restart ? ' · no reinício' : ''), row.detail || '—'])));
    const button = node('button', 'Já conferi', 'metric-ack');
    const ids = notices.map(row => row.id), version = sequence;
    button.type = 'button';
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        const response = await fetch('/ui/effects/seen', {
          method: 'POST', credentials: 'same-origin', cache: 'no-store',
          signal: AbortSignal.timeout(10000), headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ids})
        });
        if (!response.ok) throw new Error('HTTP ' + response.status);
        // A late acknowledgement must not reopen an older detail.
        if (!$('#metric-detail').open) return;
        if (version === sequence) focusReturn = 'Últimas ações';
        if (version === sequence) openDetail('audit');
      } catch {
        toast('Não consegui marcar como visto.');
      } finally { button.disabled = false; }
    });
    area.append(button);
  }
  area.append(node('h3', 'Últimas ações', 'metric-list-title'));
  if (!effects?.available) area.append(node('p', 'Não consegui ler as ações agora.', 'metric-description'));
  else if (!effects.effects.length) area.append(node('p', 'Nenhuma ação registrada.', 'metric-description'));
  else area.append(facts(effects.effects.slice(0, 10).map(row =>
    [row.label + ' · ' + clock(row.updated_at), (effectStates[row.state] || '—') +
      (row.owner_alive === false ? ' · processo parado' : '')])));
  area.append(node('h3', 'Fila de fundo', 'metric-list-title'),
    node('p', 'Ações pedidas em segundo plano esperam sua aprovação. Leia a prévia inteira e aprove ou rejeite cada uma aqui; o terminal (python -m src.gate_review) continua valendo.', 'metric-description'));
  if (!gate) area.append(node('p', 'Não consegui ler a fila de fundo agora.', 'metric-description'));
  else if (!gate.available) area.append(node('p', 'Fila de fundo indisponível agora.', 'metric-description'));
  else {
    async function decide(row, decision, button) {
      if (deciding || button.disabled) return;
      if (!confirm(decision === 'approve' ?
        'Aprovar e executar agora: ' + row.label + '?\n\nO Cortex faz exatamente o que está na prévia, uma vez só.' :
        'Rejeitar ' + row.label + '?\n\nEla sai da fila e não será executada.')) return;
      const version = sequence, label = button.textContent;
      deciding = true;
      area.querySelectorAll('.metric-decisions button').forEach(item => { item.disabled = true; });
      button.textContent = decision === 'approve' ? 'Aprovando…' : 'Rejeitando…';
      try {
        const response = await fetch('/ui/gate/decide', {
          method: 'POST', credentials: 'same-origin', cache: 'no-store',
          headers: {'Content-Type': 'application/json'}, signal: AbortSignal.timeout(60000),
          body: JSON.stringify({id: row.id, decision, lang: 'pt'})
        });
        let message;
        if (response.status === 200) {
          const data = await response.json();
          const outcomes = {executed: 'Aprovado e executado.', failed: 'Aprovado, mas a ação falhou.',
            unknown: 'Aprovado, mas o Cortex não sabe se a ação aconteceu — confira no app.',
            rejected: 'Rejeitado — não vai rodar.'};
          if (!Object.hasOwn(outcomes, data.result)) throw new Error('Unknown result');
          message = outcomes[data.result];
        } else if (response.status === 404) message = 'Essa ação não está mais na fila.';
        else if (response.status === 409) message = 'Essa ação já tinha sido decidida.';
        else if (response.status === 503) message = 'Não consegui registrar a decisão — nada foi executado.';
        else throw new Error('HTTP ' + response.status);
        toast(message);
      } catch {
        // The server may have decided already; only reread, never resend.
        toast('Não sei se a decisão chegou — relendo a fila. Não vou reenviar.');
      } finally {
        deciding = false;
        // A reopened detail may have new buttons waiting for this same request.
        area.querySelectorAll('.metric-decisions button').forEach(item => { item.disabled = false; });
        button.textContent = label;
        if (!$('#metric-detail').open) return;
        if (version === sequence) focusReturn = 'Fila de fundo';
        if (version === sequence) openDetail('audit');
      }
    }
    // Pending is the whole history, already ordered oldest first by the route.
    for (const row of gate.pending) {
      area.append(facts([[row.label + ' · na fila desde ' + clock(row.queued_at), row.reason || '—']]),
        node('pre', row.preview ?? '—', 'metric-preview'));
      const actions = node('div', null, 'metric-decisions');
      const approve = node('button', 'Aprovar', 'metric-ack');
      const reject = node('button', 'Rejeitar', 'metric-ack');
      approve.type = 'button'; reject.type = 'button';
      approve.disabled = deciding; reject.disabled = deciding;
      approve.setAttribute('aria-label', 'Aprovar ' + row.label + ', na fila desde ' + clock(row.queued_at));
      reject.setAttribute('aria-label', 'Rejeitar ' + row.label + ', na fila desde ' + clock(row.queued_at));
      approve.addEventListener('click', () => decide(row, 'approve', approve));
      reject.addEventListener('click', () => decide(row, 'reject', reject));
      actions.append(approve, reject); area.append(actions);
    }
    const recent = gate.recent.filter(row => row.status !== 'queued').slice(0, 10);
    if (recent.length) area.append(facts(recent.map(row =>
      [row.label + ' · ' + clock(row.queued_at), (queueStatuses[row.status] || '—') +
        (row.effect_owner_alive === false ? ' · processo parado' : '')])));
    if (!gate.pending.length && !recent.length) area.append(node('p', 'Nada na fila de fundo.', 'metric-description'));
  }
}
function detail(item, readAt) {
  const area = $('#metric-content'), hero = node('div', null, 'metric-hero');
  const state = node('div', null, 'metric-state'), indicator = node('i');
  dot(indicator, item.status); state.append(indicator, node('span', item.state));
  hero.append(node('div', item.unit, 'metric-hero-label'), node('div', item.value, 'metric-value'), state);
  if (item.percent != null) {
    const bar = node('div', null, 'availability-bar'), fill = node('span');
    fill.style.width = item.percent + '%';
    if (item.status === 'ok') fill.style.background = 'var(--green)';
    bar.setAttribute('aria-label', item.summary); bar.append(fill); hero.append(bar);
  }
  area.replaceChildren();
  area.append(hero, node('p', item.text, 'metric-description'));
  if (item.facts.length) area.append(facts(item.facts));
  for (const [title, rows] of item.lists || []) {
    if (rows.length) area.append(node('h3', title, 'metric-list-title'), facts(rows));
  }
  if (item.trust) trustDetail(area, item.trust.effects, item.trust.gate);
  // A failed read has no timestamp to claim.
  if (readAt) area.append(node('p', 'Lido do Cortex às ' + readAt +
    '. Fechar este painel preserva sua conversa; “Retomar central” volta à lista.', 'metric-footnote'));
  // Replacing the content can drop focus; preserve any focus still in the dialog.
  if (focusReturn && $('#metric-detail').open && !$('#metric-detail').contains(document.activeElement)) {
    const heading = [...area.querySelectorAll('h3')].find(item => item.textContent === focusReturn);
    if (heading) { heading.tabIndex = -1; heading.focus(); }
  }
  focusReturn = '';
}
function option(id, item) {
  const button = $('#option-' + id);
  button.querySelector('.option-summary').textContent = item.summary;
  button.querySelector('.option-description').textContent = descriptions[ids.indexOf(id)];
  dot(button.querySelector('.health-dot'), item.status);
}
function publish(data, id, effects, gate) {
  const items = ids.map(key => data ? metric(data, key) : unavailable());
  const item = items[ids.indexOf('audit')];
  const notices = effects?.available ? effects.notices : [], pending = gate?.available ? gate.pending : [];
  if (!data) item.text = 'Não consegui ler o histórico de atividade agora.';
  if (notices.length) item.summary += ' · ' + notices.length + ' não sei';
  if (pending.length) item.summary += ' · ' + pending.length + ' na fila de fundo';
  if (notices.length || pending.length) item.status = 'warning';
  item.trust = {effects, gate};
  ids.forEach((key, index) => option(key, items[index]));
  if (id && $('#metric-detail').open) detail(items[ids.indexOf(id)], data ? clock(data.host.generated_at) : null);
}
async function read(id) {
  const version = ++sequence;
  // Never leave a previous number visible while a fresh read is pending.
  ids.forEach(key => option(key, {summary: '', status: ''}));
  if (id) $('#metric-content').replaceChildren();
  const options = {credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(10000)};
  // Catch each read independently: one missing source must not erase the other two.
  const [data, effects, gate] = await Promise.all([
    fetch('/ui/data', options), fetch('/ui/effects', options), fetch('/ui/gate', options)
  ].map(async request => {
    try {
      const response = await request;
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return await response.json();
    } catch { return null; }
  }));
  if (version !== sequence) return;
  publish(data, id, effects, gate);
}
function openHub() { read(); }
function openDetail(id) {
  // Other options stay empty and invalidate any older detail still in flight.
  if (!ids.includes(id)) { ++sequence; return; }
  read(id);
}
window.cortexCentral = {openHub, openDetail};
})();
