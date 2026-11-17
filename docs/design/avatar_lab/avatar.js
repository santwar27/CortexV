/* Isolated lab: no application scripts, storage or network.
   CSS animates the duck; searching alone adds a shared 20fps canvas clock. */
'use strict';
(() => {
  // Measured on the unmodified 1254-square PNG at native resolution: visual
  // bounds of each feature, checked against decoded RGBA pixels. [x, y, w, h].
  // The irregular AI pixel grid is deliberately not snapped or reconstructed.
  const REGIONS = {
    "head": [250, 24, 680, 682],
    "wing": [778, 594, 256, 310],
    "leftEye": [510, 390, 132, 125],
    "rightEye": [732, 358, 100, 104],
    "antenna": [436, 46, 52, 62],
    "ear": [270, 418, 143, 192]
  };
  // Samples read directly from the PNG's decoded RGBA bytes; never theme-tinted.
  const SAMPLES = {
    yellow: {at: [460, 75], rgba: [253, 239, 87, 253]},
    visor: {at: [660, 390], rgba: [6, 14, 56, 253]},
    red: {at: [330, 445], rgba: [224, 30, 74, 253]},
    cyan: {at: [580, 420], rgba: [129, 249, 251, 253]},
    outline: {at: [445, 35], rgba: [5, 12, 49, 253]},
    lime: {at: [700, 820], rgba: [196, 239, 7, 253]},
    blue: {at: [620, 210], rgba: [165, 197, 251, 253]}
  };
  // Head contour excludes the raised wing inside the lower-right of its bounds.
  const HEAD = [[250, 24], [930, 24], [930, 590], [870, 590], [870, 640],
    [815, 640], [815, 685], [750, 685], [750, 706], [250, 706]];
  // Wing contour refines its bounding rectangle, avoiding the beak and torso.
  // Points follow the outer silhouette and the shoulder seam, in file pixels.
  const WING = [[778, 724], [820, 683], [944, 594], [1008, 594],
    [1034, 612], [1034, 680], [991, 721], [991, 791], [930, 856],
    [864, 904], [819, 904], [819, 812], [778, 812]];
  const LABELS = {
    idle: 'Em repouso', thinking: 'Pensando', searching: 'Buscando na web',
    reading: 'Lendo', writing: 'Escrevendo', remembering: 'Lembrando',
    listening: 'Ouvindo', speaking: 'Respondendo', processing: 'Processando'
  };
  const REPLY_LABELS = Object.freeze({
    thinking: 'pensando…', searching: 'buscando: clima em Londres…',
    reading: 'lendo notas.pdf…', writing: 'escrevendo resumo.md…',
    remembering: 'lembrando…', processing: 'processando…'
  });
  const REPLY_STATES = Object.keys(REPLY_LABELS);
  const REPLY_TIMINGS = Object.freeze({status: 3800, character: 45, rest: 4000});
  const REPLY_TEXT = 'Pronto. Li as notas e escrevi o resumo. Podemos continuar por aqui.';
  // Racket art: 15 × 30 big pixels. Each non-dot cell becomes one rect
  // on the 1254/96 grid. Palette values name the documented PNG samples above.
  const RACKET = {
    cell: 1254 / 96, origin: [66, 21],
    palette: {N: 'outline', R: 'red', Y: 'yellow', G: 'lime', B: 'blue'},
    rows: [
      '.....NNNNN.....',
      '...NNRRRRRNN...',
      '..NRRYYYYYRRN..',
      '.NRYYNNNNNYYRN.',
      '.NRYNBNBNBNYRN.',
      'NRYNBBBNBBBNYRN',
      'NRYNBNBNBNBNYRN',
      'NRYNNNNNNNNNYRN',
      'NRYNBNBNBNBNYRN',
      'NRYNBBBNBBBNYRN',
      'NRYNNNNNNNNNYRN',
      'NRYNBNBNBNBNYRN',
      'NRYNBBBNBBBNYRN',
      '.NRYNBNBNBNYRN.',
      '.NRYYNNNNNYYRN.',
      '..NRRYYYYYRRN..',
      '...NNRRRRRNN...',
      '.....NRRN......',
      '.....NRRN......',
      '......NNN......',
      '......NGN......',
      '......NYN......',
      '......NGN......',
      '......NYN......',
      '......NGN......',
      '......NYN......',
      '......NGN......',
      '......NYN......',
      '.....NGGGN.....',
      '......NNN......'
    ]
  };
  // Fold accents and case before matching; all language/emoji lists live here.
  const SIGNALS = {
    greetings: ['oi', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'e ai', 'eai', 'hey', 'hi', 'hello'],
    happy: ['haha', 'hahaha', 'kkk', 'rs', 'otimo', 'perfeito', 'show', 'massa', 'top',
      'adorei', 'amei', 'incrivel', 'demais', 'uhul', 'valeu', 'great', 'awesome', 'love it', 'yay'],
    emoji: ['😄', '😁', '😂', '🥳', '🎉', '👏', '🎾', '❤️'],
    exclamations: ['boa'],
    negations: ['nao', 'nem', 'nada de', 'not', 'never', 'no'],
    negativeRank: ['piores', 'worst']
  };
  const fold = text => String(text).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const words = text => text.match(/[\p{L}\p{N}]+/gu) || [];
  // Conservatively reject a positive within four words of a preceding
  // negation, within this sentence only (handles "não está/ficou ótimo").
  function isNegated(tokens, index) {
    return SIGNALS.negations.some(term => {
      const phrase = term.split(' ');
      for (let i = Math.max(0, index - 4); i <= index - phrase.length; i++) {
        if (phrase.every((word, j) => tokens[i + j] === word)) return true;
      }
      return false;
    });
  }
  function detectMessage(text) {
    const normalized = fold(text);
    const start = words(normalized).join(' ');
    const greeting = SIGNALS.greetings.some(term => start === term || start.startsWith(`${term} `));
    const happy = normalized.split(/[.!?;\n]+/u).some(sentence => {
      const tokens = words(sentence);
      return tokens.some((token, i) => {
        const phrase = tokens.slice(i, i + 2).join(' ');
        const positive = SIGNALS.happy.includes(token) || SIGNALS.happy.includes(phrase)
          || /^(?:ha){2,}$|^k{3,}$/.test(token);
        const badRank = token === 'top' && SIGNALS.negativeRank.some(word => tokens.slice(i + 1, i + 4).includes(word));
        return positive && !badRank && !isNegated(tokens, i);
      }) || SIGNALS.emoji.some(emoji => {
        const index = sentence.indexOf(emoji);
        return index >= 0 && !isNegated(tokens, words(sentence.slice(0, index)).length);
      });
    }) || SIGNALS.exclamations.some(term => {
      const pattern = new RegExp(`(?:^|[^\\p{L}\\p{N}])${term}\\s*!`, 'gu');
      return [...normalized.matchAll(pattern)].some(match => {
        const prefix = normalized.slice(0, match.index).split(/[.!?;\n]/u).at(-1);
        const tokens = words(prefix);
        return !isNegated(tokens, tokens.length);
      });
    });
    return {greeting, happy};
  }

  const $ = selector => document.querySelector(selector);
  const duck = $('#duck');
  const replyHead = $('#reply-head');
  const replyProgress = $('#reply-progress');
  const replyText = document.createTextNode('');
  // Keep task 131's text node and three-dot indicator through status changes.
  const replyDots = document.createElement('span');
  replyDots.className = 'reply-dots'; replyDots.setAttribute('aria-hidden', 'true');
  for (let i = 0; i < 3; i++) replyDots.append(document.createElement('i'));
  const replyCursor = document.createElement('span');
  replyCursor.className = 'reply-cursor'; replyCursor.setAttribute('aria-hidden', 'true');
  const ducks = [duck];
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  const DURATIONS = {wave: 1800, racket: 2600, look: 1400, hop: 900, stretch: 1600};
  const RANDOM_GESTURES = ['look', 'hop', 'wave', 'stretch'];
  let state = 'idle', reduced = false, comparing = false;
  let eyeTimer = 0, gestureTimer = 0, randomTimer = 0, step = 0;
  let gesture = 'none', queue = [], previousRandom = '', seed = Date.now() >>> 0;
  let replyMode = $('#reply-demo-mode').value === 'auto' ? 'auto' : 'off';
  let replyPhase = replyMode === 'auto' ? 'welcome' : 'status';
  let replyTimer = 0, replyIndex = 0, replyLength = 0, replyKind = '';
  const percent = n => `${n / 1254 * 100}%`;
  const points = list => list.map(([x, y]) => `${percent(x)} ${percent(y)}`).join(', ');
  const polygon = list => `polygon(${points(list)})`;
  const colour = sample => `rgb(${sample.rgba.slice(0, 3).join(' ')})`;
  function outside(list) {
    return `polygon(evenodd, ${points([[0, 0], [1254, 0], [1254, 1254], [0, 1254], [0, 0], ...list, list[0]])})`;
  }
  function configureArt(node) {
    Object.entries(REGIONS).forEach(([name, [x, y, w, h]]) => {
      const rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
      node.style.setProperty(`--${name}-clip`, polygon(rect));
      node.style.setProperty(`--${name}-outside`, outside(rect));
      for (const [key, value] of Object.entries({x, y, w, h})) node.style.setProperty(`--${name}-${key}`, percent(value));
    });
    node.style.setProperty('--head-clip', polygon(HEAD));
    node.style.setProperty('--head-outside', outside(HEAD));
    node.style.setProperty('--wing-clip', polygon(WING));
    node.style.setProperty('--wing-outside', outside(WING));
    for (const [name, sample] of Object.entries(SAMPLES)) node.style.setProperty(`--duck-${name}`, colour(sample));
  }
  configureArt(duck);
  configureArt(replyHead);
  // Translate the full file behind a 28px head, retaining its measured contour.
  const [headX, headY, headWidth, headHeight] = REGIONS.head;
  for (const [name, value] of Object.entries({size: 1254 / headHeight,
    left: -headX / headHeight, top: -headY / headHeight, width: headWidth / headHeight})) {
    replyHead.style.setProperty(`--reply-file-${name}`, value);
  }
  for (const [name, duration] of Object.entries(DURATIONS)) duck.style.setProperty(`--duration-${name}`, `${duration}ms`);
  const racket = duck.querySelector('.duck-racket');
  RACKET.rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === '.') return;
    const rect = document.createElementNS(racket.namespaceURI, 'rect');
    rect.setAttribute('x', (RACKET.origin[0] + x) * RACKET.cell);
    rect.setAttribute('y', (RACKET.origin[1] + y) * RACKET.cell);
    rect.setAttribute('width', RACKET.cell);
    rect.setAttribute('height', RACKET.cell);
    rect.setAttribute('fill', colour(SAMPLES[RACKET.palette[cell]]));
    racket.append(rect);
  }));
  for (const size of [96, 128, 144]) {
    const figure = document.createElement('figure');
    const copy = duck.cloneNode(true);
    copy.removeAttribute('id');
    copy.style.setProperty('--duck-size', `${size}px`);
    const caption = document.createElement('figcaption');
    caption.textContent = `${size} px`;
    figure.append(copy, caption);
    $('#size-previews').append(figure);
    ducks.push(copy);
  }
  // 3 × 5 binary rows: 0, 1, 2, 3, 7, <, >, /, =. Rects only, no font or
  // source-image drawing. Duck cells retain the 1254/96 grid in a larger halo;
  // the head badge uses 1 CSS-px cells so its seven columns stay legible.
  const RAIN_GLYPHS = [
    [7, 5, 5, 5, 7], [2, 6, 2, 2, 7], [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7], [7, 1, 2, 2, 2], [1, 2, 4, 2, 1],
    [4, 2, 1, 2, 4], [1, 1, 2, 4, 4], [0, 7, 0, 7, 0]
  ];
  const RAIN_INTERVAL = 50;
  const rainFigures = [...ducks, replyHead].map(node => ({
    node, tiny: node === replyHead, width: node === replyHead ? 40 : 144,
    height: node === replyHead ? 40 : 120, canvas: null, context: null, active: false
  }));
  let rainTimer = 0, rainTime = 0, rainColour = 'green', rainPaletteKey = '', rainPalette = [], rainBadgePalette = [];
  $('#rain-colour').value = rainColour;
  function rainStill() {
    return reduced || media.matches || document.body.classList.contains('reduced-motion');
  }
  function configureRainPalette() {
    const light = document.body.classList.contains('light');
    const key = `${rainColour}:${light}`;
    if (key === rainPaletteKey) return;
    rainPaletteKey = key;
    function palette(onLight) {
      const rgb = rainColour === 'cyan' ? (onLight ? [24, 116, 124] : SAMPLES.cyan.rgba.slice(0, 3))
        : (onLight ? [22, 119, 55] : [57, 255, 112]);
      // Brighter heads on dark surfaces; inkier heads on the light chat ground.
      const head = rgb.map(value => Math.round(onLight ? value * .75 : value + (255 - value) * .65));
      return [head, rgb, ...[.8, .65, .5, .4, .3, .22].map(factor =>
        rgb.map(value => Math.round(value * factor)))].map(channels => `rgb(${channels.join(' ')})`);
    }
    rainPalette = palette(light);
    rainBadgePalette = light ? palette(false) : rainPalette;
  }
  function paintRain(figure, still) {
    const {context, width, height, tiny} = figure;
    if (!context) return;
    context.clearRect(0, 0, width, height);
    const columns = tiny ? 7 : 24, trail = tiny ? 5 : 8;
    const palette = tiny ? rainBadgePalette : rainPalette;
    const time = still ? 0 : rainTime;
    for (let column = 0; column < columns; column++) {
      const speed = (tiny ? 6 : 12) + column % 3 * (tiny ? 2 : 4);
      const head = (Math.floor(time * speed) + 12 + column * 17) % (height + trail * 6);
      for (let tail = 0; tail < trail; tail++) {
        const y = head - tail * 6;
        const index = ((Math.floor(time * 2) + column * 3 - tail) % RAIN_GLYPHS.length + RAIN_GLYPHS.length) % RAIN_GLYPHS.length;
        const glyph = RAIN_GLYPHS[index];
        context.fillStyle = palette[tail];
        // Alpha fades as well as RGB, so darker trails recede on light surfaces.
        context.globalAlpha = (still ? .26 : .95) * (1 - tail / trail);
        for (let row = 0; row < 5; row++) {
          if (y + row < 0 || y + row >= height) continue;
          for (let cell = 0; cell < 3; cell++) {
            if (glyph[row] & (4 >> cell)) context.fillRect((tiny ? 3 : 1) + column * (tiny ? 5 : 6) + cell, y + row, 1, 1);
          }
        }
      }
    }
  }
  function rainFrame() {
    rainTimer = 0;
    if (document.hidden) return;
    const still = rainStill();
    if (!still) rainTime += RAIN_INTERVAL / 1000;
    for (const figure of rainFigures) if (figure.active) paintRain(figure, still);
    // Fixed storage and glyph tables; no arrays, paths or nodes allocated per frame.
    // A delayed callback advances one step only; never catch up after a hidden tab.
    if (!still) rainTimer = setTimeout(rainFrame, RAIN_INTERVAL);
  }
  function synchronizeRain() {
    clearTimeout(rainTimer); rainTimer = 0;
    configureRainPalette();
    let any = false;
    for (const figure of rainFigures) {
      figure.active = !comparing && state === 'searching' && (figure.node === replyHead
        ? replyMode !== 'off' && replyPhase === 'status'
        : figure.node === duck || !$('#size-comparison').hidden);
      if (figure.active) {
        any = true;
        if (!figure.canvas) {
          const canvas = document.createElement('canvas');
          canvas.className = 'code-rain'; canvas.setAttribute('aria-hidden', 'true');
          canvas.width = figure.width; canvas.height = figure.height;
          figure.node.prepend(canvas); figure.canvas = canvas;
          figure.context = canvas.getContext('2d');
        }
        figure.canvas.hidden = false;
        if (!document.hidden) paintRain(figure, rainStill());
      } else if (figure.canvas) {
        figure.context?.clearRect(0, 0, figure.width, figure.height);
        figure.canvas.hidden = true;
      }
    }
    if (!any) rainTime = 0;
    if (any && !document.hidden && !rainStill()) rainTimer = setTimeout(rainFrame, RAIN_INTERVAL);
  }
  // Also honour a body-class change made outside the lab checkbox. Palette/class
  // observation does not touch state, text, gestures or the existing CSS clocks.
  if (typeof MutationObserver !== 'undefined') {
    new MutationObserver(synchronizeRain).observe(document.body, {attributes: true, attributeFilter: ['class']});
  }
  const data = (key, value) => ducks.forEach(node => { node.dataset[key] = value; });
  function synchronizeAnimations() {
    // No frame loop: align CSS timelines after updates and when revealing copies.
    // getAnimations flushes style, including a formerly display:none comparison.
    if (!duck.getAnimations) return;
    const times = new Map(duck.getAnimations({subtree: true}).map(animation => [animation.animationName, animation.currentTime]));
    for (const copy of ducks.slice(1)) {
      for (const animation of copy.getAnimations({subtree: true})) {
        if (times.has(animation.animationName)) animation.currentTime = times.get(animation.animationName);
      }
    }
  }
  function stopped() { return reduced || media.matches || comparing || document.hidden; }
  function scheduleReply() {
    clearTimeout(replyTimer); replyTimer = 0;
    if (stopped() || replyMode === 'off') return;
    if (replyPhase === 'welcome') {
      replyTimer = setTimeout(() => setReplyDemo('auto'), DURATIONS.wave);
    } else if (replyPhase === 'status' && replyMode === 'auto') {
      replyTimer = setTimeout(() => {
        if (++replyIndex === REPLY_STATES.length) arriveReply();
        else { state = REPLY_STATES[replyIndex]; render(); }
      }, REPLY_TIMINGS.status);
    } else if (replyPhase === 'stream') {
      replyTimer = setTimeout(() => {
        replyLength++;
        if (replyLength >= REPLY_TEXT.length) replyPhase = 'final';
        renderReply();
      }, REPLY_TIMINGS.character);
    } else if (replyPhase === 'final' && replyMode === 'auto') {
      replyTimer = setTimeout(() => setReplyDemo('auto'), REPLY_TIMINGS.rest);
    }
  }
  function renderReply() {
    $('#reply-demo-mode').value = replyMode;
    $('#reply-demo').hidden = replyMode === 'off' || replyPhase === 'welcome' || comparing;
    $('#reply-arrive').disabled = replyMode === 'off' || replyPhase !== 'status' || comparing;
    replyHead.dataset.state = state;
    replyHead.dataset.paused = String(document.hidden);
    if (replyMode !== 'off' && replyPhase === 'status') {
      replyText.textContent = REPLY_LABELS[state];
      if (replyKind !== 'status') replyProgress.replaceChildren(replyHead, replyText, replyDots);
      replyKind = 'status';
    } else if (replyPhase === 'stream' || replyPhase === 'final') {
      // The first token replaces the entire status: no head or dots survive.
      if ((reduced || media.matches) && replyPhase === 'stream') {
        replyLength = REPLY_TEXT.length; replyPhase = 'final';
      }
      replyText.textContent = REPLY_TEXT.slice(0, replyLength);
      if (replyKind !== replyPhase) {
        replyProgress.replaceChildren(replyText, ...(replyPhase === 'stream' ? [replyCursor] : []));
      }
      replyKind = replyPhase;
    } else {
      if (replyKind !== 'empty') replyProgress.replaceChildren();
      replyKind = 'empty';
    }
    scheduleReply(); synchronizeRain();
  }
  function setReplyDemo(mode) {
    if (mode !== 'auto' && mode !== 'off' && !Object.hasOwn(REPLY_LABELS, mode)) return;
    replyMode = mode; replyPhase = 'status'; replyIndex = 0; replyLength = 0;
    if (mode !== 'off') state = mode === 'auto' ? REPLY_STATES[0] : mode;
    render();
  }
  function arriveReply() {
    if (replyMode === 'off' || replyPhase !== 'status' || comparing) return;
    replyPhase = 'stream'; replyLength = 1; state = 'idle'; render();
  }
  function clearEyes() {
    clearTimeout(eyeTimer); eyeTimer = 0;
    data('eyes', 'open'); step = 0;
  }
  function clearTransient() {
    clearEyes();
    clearTimeout(gestureTimer); clearTimeout(randomTimer);
    gestureTimer = randomTimer = 0;
    gesture = 'none'; queue = [];
    data('gesture', 'none'); data('part', 'none');
  }
  function random() {
    // Mulberry32: deterministic uint32 seed, no reliance on Math.random.
    seed = (seed + 0x6D2B79F5) >>> 0;
    let value = seed;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  }
  function scheduleRandom() {
    clearTimeout(randomTimer); randomTimer = 0;
    if (stopped() || state !== 'idle' || gesture !== 'none' || queue.length) return;
    randomTimer = setTimeout(() => { randomTimer = 0; randomGestureNow(); }, 25000 + Math.floor(random() * 35001));
  }
  function setSeed(value) {
    seed = Number(value) >>> 0; previousRandom = '';
    scheduleRandom();
  }
  function scheduleEyes(delay = 4600) {
    if (stopped() || gesture !== 'none') return;
    eyeTimer = setTimeout(() => {
      eyeTimer = 0;
      if (stopped() || gesture !== 'none') return;
      data('eyes', state === 'reading' ? (step++ % 2 ? 'down-two' : 'down-one') : 'closed');
      eyeTimer = setTimeout(() => {
        eyeTimer = 0; data('eyes', 'open');
        scheduleEyes(state === 'reading' ? 1300 : state === 'remembering' ? 3300 : 4600);
      }, state === 'remembering' ? 180 : 160);
    }, delay);
  }
  function playGesture(name) {
    if (!Object.hasOwn(DURATIONS, name) || stopped() || state !== 'idle' || gesture !== 'none') return false;
    clearEyes(); clearTimeout(randomTimer); randomTimer = 0;
    gesture = name;
    data('gesture', name);
    data('part', ['wave', 'racket'].includes(name) ? 'wing' : name === 'look' ? 'head' : 'none');
    synchronizeAnimations();
    gestureTimer = setTimeout(() => {
      gestureTimer = 0; gesture = 'none';
      data('gesture', 'none'); data('part', 'none'); data('eyes', 'open');
      // Flush the rest pose before a queued repeat of the same CSS animation.
      synchronizeAnimations();
      if (queue.length && !stopped()) playGesture(queue.shift());
      else { scheduleEyes(); scheduleRandom(); }
    }, DURATIONS[name]);
    return true;
  }
  function randomGestureNow() {
    if (stopped() || state !== 'idle' || gesture !== 'none' || queue.length) return false;
    const choices = RANDOM_GESTURES.filter(name => name !== previousRandom);
    const name = choices[Math.floor(random() * choices.length)];
    if (!playGesture(name)) return false;
    previousRandom = name;
    return true;
  }
  function requestGestures(names) {
    if (stopped() || !names.length) return;
    // An explicit interaction settles a simulated busy state before reacting.
    // Automatic/random requests never change the selected state.
    if (state !== 'idle' || replyMode !== 'off') {
      replyMode = 'off'; replyPhase = 'status'; state = 'idle'; render();
    }
    // Keep explicit gestures serial, bounded even if a button is clicked rapidly.
    queue.push(...names.slice(0, Math.max(0, 8 - queue.length)));
    if (gesture === 'none' && queue.length) playGesture(queue.shift());
  }
  function render() {
    clearTransient();
    // With no welcome motion available, show a still thinking state immediately.
    if (replyPhase === 'welcome' && stopped()) { replyPhase = 'status'; state = 'thinking'; }
    const still = reduced || media.matches;
    const shown = comparing ? 'idle' : state;
    data('state', shown); data('reduced', String(still));
    data('frozen', String(comparing)); data('paused', String(document.hidden));
    if (!stopped()) data('part', shown === 'writing' ? 'wing' : shown !== 'idle' ? 'head' : 'none');
    ducks.forEach(node => node.setAttribute('aria-label', `Pato do Cortex · ${LABELS[shown]}`));
    $('#duck-state').textContent = LABELS[shown];
    $('#state-select').value = state;
    $('#motion-note').textContent = still ? 'Movimento reduzido · estados estáticos; selecione um estado ou faça o texto chegar.'
      : 'Gestos em repouso · saudação e felicidade reagem à mensagem · pausa ao sair desta aba.';
    scheduleEyes(state === 'reading' ? 900 : state === 'remembering' ? 1400 : 4600);
    scheduleRandom(); synchronizeAnimations(); renderReply();
  }
  function setState(next) {
    if (!Object.hasOwn(LABELS, next)) return;
    if (replyMode !== 'off') {
      replyMode = Object.hasOwn(REPLY_LABELS, next) ? next : 'off'; replyPhase = 'status';
    }
    state = next; render();
  }
  function setReduced(value) {
    reduced = Boolean(value);
    document.body.classList.toggle('reduced-motion', reduced);
    $('#reduce-motion').checked = reduced; render();
  }
  function setComparison(value) {
    comparing = Boolean(value);
    $('#comparison').hidden = !comparing;
    $('#compare-original').setAttribute('aria-pressed', String(comparing)); render();
  }
  function setSizeComparison(value) {
    $('#size-comparison').hidden = !value;
    $('#compare-sizes').setAttribute('aria-pressed', String(Boolean(value)));
    synchronizeAnimations(); synchronizeRain();
  }
  function submitMessage(text) {
    const message = String(text).trim();
    if (!message) return false;
    const bubble = document.createElement('article'); bubble.className = 'message user';
    const content = document.createElement('div'); content.className = 'message-content';
    const paragraph = document.createElement('p'); paragraph.textContent = message;
    content.append(paragraph); bubble.append(content); $('.messages').append(bubble);
    bubble.scrollIntoView({block: 'nearest'});
    const signals = detectMessage(message);
    requestGestures([...(signals.greeting ? ['wave'] : []), ...(signals.happy ? ['racket'] : [])]);
    return true;
  }
  function sendComposer() {
    const input = $('#mock-message');
    if (submitMessage(input.value)) input.value = '';
  }
  $('#mock-message').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendComposer(); }
  });
  $('.send').addEventListener('click', sendComposer);
  $('#wave-hello').addEventListener('click', () => requestGestures(['wave']));
  $('#happy-pedro').addEventListener('click', () => requestGestures(['racket']));
  $('#random-gesture').addEventListener('click', randomGestureNow);
  $('#state-select').addEventListener('change', event => setState(event.target.value));
  $('#reply-demo-mode').addEventListener('change', event => setReplyDemo(event.target.value));
  $('#reply-arrive').addEventListener('click', arriveReply);
  $('#size-select').addEventListener('change', event => {
    const size = Number(event.target.value);
    if ([96, 128, 144].includes(size)) document.body.style.setProperty('--duck-size', `${size}px`);
  });
  $('#reduce-motion').addEventListener('change', event => setReduced(event.target.checked));
  $('#rain-colour').addEventListener('change', event => {
    if (!['green', 'cyan'].includes(event.target.value)) return;
    rainColour = event.target.value; synchronizeRain();
  });
  $('#light-theme').addEventListener('change', event => {
    document.body.classList.toggle('light', event.target.checked); synchronizeRain();
  });
  $('#compare-original').addEventListener('click', () => setComparison(!comparing));
  $('#compare-sizes').addEventListener('click', () => setSizeComparison($('#size-comparison').hidden));
  media.addEventListener('change', render);
  document.addEventListener('visibilitychange', render);
  window.cortexAvatarLab = Object.freeze({setState, setReduced, setComparison, setSizeComparison,
    detectMessage, submitMessage, playGesture, randomGestureNow, setSeed, setReplyDemo, arriveReply,
    replyLabels: REPLY_LABELS, replyTimings: REPLY_TIMINGS, replyText: REPLY_TEXT,
    regions: REGIONS, samples: SAMPLES, racket: RACKET, signals: SIGNALS, durations: DURATIONS});
  render();
  playGesture('wave');
})();
