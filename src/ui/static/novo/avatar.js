/* Pedro's untouched PNG and measured clips from the approved avatar lab.
   Decorative local animation; one clock paints Matrix rain only when searching. */
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
    "ear": [270, 418, 143, 192],
    "blueWing": [208, 732, 228, 158],
    "leftFoot": [318, 1106, 212, 132],
    "rightFoot": [606, 1116, 220, 110]
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
  // Native-pixel samples along the blue shoulder seam and narrow yellow ankles.
  // Feet include their dark outlines; the red/blue joints stay with the torso.
  const BLUE_WING = [[436, 780], [372, 780], [352, 762], [322, 760],
    [308, 742], [232, 732], [214, 752], [208, 826], [222, 844],
    [222, 858], [240, 872], [244, 890], [264, 890], [272, 864],
    [290, 850], [304, 834], [338, 828], [360, 814], [396, 816], [436, 804]];
  const LEFT_FOOT = [[410, 1106], [472, 1106], [472, 1146], [492, 1146],
    [492, 1158], [516, 1158], [530, 1180], [530, 1218], [500, 1230],
    [400, 1238], [372, 1226], [338, 1226], [318, 1208], [318, 1176],
    [344, 1158], [372, 1146], [390, 1128], [410, 1128]];
  const RIGHT_FOOT = [[638, 1116], [696, 1116], [696, 1130], [730, 1130],
    [730, 1148], [768, 1148], [798, 1164], [826, 1172], [826, 1202],
    [798, 1216], [752, 1216], [752, 1226], [680, 1226], [680, 1216],
    [626, 1216], [606, 1198], [606, 1162], [624, 1146], [638, 1146]];
  const CONTOURS = {wing: WING, blueWing: BLUE_WING, leftFoot: LEFT_FOOT, rightFoot: RIGHT_FOOT};
  const LABELS = {
    idle: 'Em repouso', thinking: 'Pensando', searching: 'Buscando na web',
    reading: 'Lendo', writing: 'Escrevendo', remembering: 'Lembrando',
    listening: 'Ouvindo', speaking: 'Respondendo', processing: 'Processando'
  };
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

  const STAGE_STATES = Object.freeze({thinking: 'thinking', web_search: 'searching',
    reading: 'reading', writing: 'writing', remembering: 'remembering'});
  const stateForStage = stage => Object.hasOwn(STAGE_STATES, stage) ? STAGE_STATES[stage] : 'thinking';
  const duck = document.querySelector('#cortex-duck');
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  const DURATIONS = {wave: 1800, racket: 2600, look: 1400, hop: 900, stretch: 1600,
    turn: 760, walk: 3000, pirouette: 1440, flap: 1200, stumble: 1400, wake: 600, cross: 6000,
    dance: 2400, peek: 2800, cheer: 1600, no: 1300, startle: 700, poop: 1700, knock: 1500};
  const WEIGHTS = Object.freeze({look: 6, walk: 6, hop: 2, wave: 2, stretch: 2,
    turn: 2, pirouette: 1, flap: 2, stumble: 1, dance: 2, peek: 2, cheer: 2, no: 1});
  // Only a click can bring the racket or startle the duck.
  const CLICK_WEIGHTS = Object.freeze({racket: 8, startle: 4});
  // Rare pranks: only at rest, on the stage, at most one every five minutes (Pedro's choice).
  const PRANK_WEIGHTS = Object.freeze({poop: 2, knock: 2}), PRANK_GAP_MS = 300000;
  let state = 'idle', liveState = 'idle', previewState = null;
  let visible = false, reduced = false, welcomed = false;
  let eyeTimer = 0, gestureTimer = 0, randomTimer = 0, step = 0;
  let gesture = 'none', queue = [], reacting = false, previousRandom = '', seed = Date.now() >>> 0;
  const dock = document.querySelector('#cortex-duck-dock');
  let homeY = 0, homeRoute = [], homeGoal = null, homeMap = null;
  let chatView = document.body.classList.contains('chat-view');
  let x = 0, walkTimer = 0, facingTimers = [], continuation = null, orienting = false;
  let destination = 0, walkingLimit = 0, returning = false;
  duck.dataset.facing = 'right'; duck.dataset.x = '0'; duck.dataset.y = '0';
  // Build new same-source layers at runtime: the approved dock markup stays exact.
  const pose = duck.querySelector('.duck-pose');
  const motion = document.createElement('div'); motion.className = 'duck-motion';
  for (const child of [...pose.children]) motion.append(child);
  pose.append(motion);
  for (const name of ['blueWing', 'leftFoot', 'rightFoot']) {
    const image = document.createElement('img'); image.className = `duck-part duck-${name}`;
    image.setAttribute('src', '/ui/static/novo/avatar/duckcortex.png');
    image.setAttribute('width', '1254'); image.setAttribute('height', '1254');
    image.setAttribute('alt', ''); image.setAttribute('draggable', 'false'); motion.append(image);
  }
  const percent = n => `${n / 1254 * 100}%`;
  const points = list => list.map(([x, y]) => `${percent(x)} ${percent(y)}`).join(', ');
  const polygon = list => `polygon(${points(list)})`;
  const colour = sample => `rgb(${sample.rgba.slice(0, 3).join(' ')})`;
  function outside(...lists) {
    // Retraced bridges join disjoint holes without adding a filled triangle.
    const path = [[0, 0], [1254, 0], [1254, 1254], [0, 1254], [0, 0]];
    for (const list of lists) path.push(...list, list[0], [0, 0]);
    return `polygon(evenodd, ${points(path)})`;
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
    for (const [name, contour] of Object.entries(CONTOURS)) {
      node.style.setProperty(`--${name}-clip`, polygon(contour));
      node.style.setProperty(`--${name}-outside`, outside(contour));
    }
    node.style.setProperty('--limbs-outside', outside(...Object.values(CONTOURS)));
    for (const [name, sample] of Object.entries(SAMPLES)) node.style.setProperty(`--duck-${name}`, colour(sample));
  }
  configureArt(duck);
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
  // 3 × 5 binary rows: 0, 1, 2, 3, 7, <, >, /, =. Rects only, no font or
  // source-image drawing. Duck cells retain the 1254/96 grid in a larger halo;
  // the head badge uses 1 CSS-px cells so its seven columns stay legible.
  const RAIN_GLYPHS = [
    [7, 5, 5, 5, 7], [2, 6, 2, 2, 7], [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7], [7, 1, 2, 2, 2], [1, 2, 4, 2, 1],
    [4, 2, 1, 2, 4], [1, 1, 2, 4, 4], [0, 7, 0, 7, 0]
  ];
  const RAIN_INTERVAL = 50;
  const rainFigures = [{node: duck, tiny: false, width: 144, height: 120,
    canvas: null, context: null, active: false}];
  let rainTimer = 0, rainTime = 0, rainPaletteKey = '', rainPalette = [], rainBadgePalette = [];
  function rainStill() { return reduced || media.matches || document.body.classList.contains('reduced-motion'); }
  function configureRainPalette() {
    const light = document.body.classList.contains('light');
    const key = String(light);
    if (key === rainPaletteKey) return;
    rainPaletteKey = key;
    function palette(onLight) {
      const rgb = onLight ? [22, 119, 55] : [57, 255, 112];
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
      figure.active = figure.node === duck ? visible && habitat === 'stage' && state === 'searching'
        : figure.node.dataset.state === 'searching';
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
  // Heads belong to one pending reply each. Disposal clears detached canvases
  // immediately and releases their records; history never calls this factory.
  function createHead(next) {
    const head = document.createElement('span');
    head.className = 'reply-head'; head.setAttribute('aria-hidden', 'true');
    const motion = document.createElement('span'), file = document.createElement('span');
    motion.className = 'reply-head-motion'; file.className = 'reply-head-file';
    const image = document.createElement('img');
    image.setAttribute('src', '/ui/static/novo/avatar/duckcortex.png');
    image.setAttribute('width', '1254'); image.setAttribute('height', '1254');
    image.setAttribute('alt', ''); image.setAttribute('draggable', 'false');
    file.append(image);
    for (const name of ['antenna-glow', 'ear-glow', 'eye-patch left-eye', 'eye-patch right-eye']) {
      const effect = document.createElement('span'); effect.className = name; file.append(effect);
    }
    motion.append(file); head.append(motion); configureArt(head);
    // Translate the original file behind a 28px head, retaining its contour.
    const [x, y, width, height] = REGIONS.head;
    for (const [name, value] of Object.entries({size: 1254 / height,
      left: -x / height, top: -y / height, width: width / height})) {
      head.style.setProperty(`--reply-file-${name}`, value);
    }
    rainFigures.push({node: head, tiny: true, width: 40, height: 40,
      canvas: null, context: null, active: false});
    updateHead(head, next); return head;
  }
  function updateHead(head, next) {
    head.dataset.state = next;
    head.dataset.paused = String(document.hidden);
    synchronizeRain();
  }
  function removeHead(head) {
    if (!head) return;
    const index = rainFigures.findIndex(figure => figure.node === head);
    if (index >= 0) {
      const figure = rainFigures[index];
      figure.context?.clearRect(0, 0, figure.width, figure.height);
      if (figure.canvas) figure.canvas.hidden = true;
      rainFigures.splice(index, 1);
    }
    head.remove(); synchronizeRain();
  }
  const data = (key, value) => { duck.dataset[key] = value; };
  // Attention uses cached viewport geometry. Events only record intent; at most
  // one animation frame paints the two existing same-PNG eye copies.
  const SLEEP_AFTER_MS = 180000;
  const composer = document.querySelector('#message-input');
  let box = null, composerBox = null, messageBox = null, bubbleBox = null;
  let gazeFrame = 0, pointerTimer = 0, typingTimer = 0, replyTimer = 0, turnTimer = 0, centeringTimer = 0;
  let pointerPoint = null, touchPoint = null, typing = false, replyAttention = false, replyPending = false;
  let behindSide = '', behindSince = 0, lastAttentionTurn = -Infinity;
  let asleep = false, waking = false, suppressDuckClick = false;
  let sleepTimer = 0, lastActivity = Date.now();
  // Replaceable Z z z cell map; cyan is sampled from the original visor.
  // Native-grid bounds stay within the walking band's existing 25% top halo.
  const ZZZ = {cell: 1254 / 96, origin: [60, -18], palette: {C: 'cyan'}, rows: [
    'CCCCC..............', '...C...CCC...CCC...', '..C.....C.....C....',
    '.C.....C.....C.....', 'CCCCC..CCC...CCC...'
  ]};
  const zzz = document.createElementNS(racket.namespaceURI, 'svg');
  zzz.setAttribute('class', 'duck-zzz');
  zzz.setAttribute('viewBox', '0 -313.5 1254 1567.5');
  zzz.setAttribute('shape-rendering', 'crispEdges'); zzz.setAttribute('aria-hidden', 'true');
  zzz.hidden = true; duck.append(zzz);
  ZZZ.rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === '.') return;
    const rect = document.createElementNS(racket.namespaceURI, 'rect');
    rect.setAttribute('x', (ZZZ.origin[0] + x) * ZZZ.cell);
    rect.setAttribute('y', (ZZZ.origin[1] + y) * ZZZ.cell);
    rect.setAttribute('width', ZZZ.cell); rect.setAttribute('height', ZZZ.cell);
    rect.setAttribute('fill', colour(SAMPLES[ZZZ.palette[cell]])); zzz.append(rect);
  }));
  function refreshAttentionRects() {
    box = duck.getBoundingClientRect(); composerBox = composer?.getBoundingClientRect();
    const messages = document.querySelector('#messages');
    messageBox = messages?.getBoundingClientRect();
    const bubbles = messages?.querySelectorAll('.assistant');
    bubbleBox = bubbles?.length ? bubbles[bubbles.length - 1].getBoundingClientRect() : null;
  }
  const centre = rect => rect && ({x: rect.left + rect.width / 2, y: rect.top + rect.height / 2});
  function attentionTarget() {
    if (replyAttention) return {kind: 'reply', point: centre(bubbleBox?.width ? bubbleBox : messageBox)};
    if (typing && ['idle', 'listening'].includes(state)) return {kind: 'typing', point: centre(composerBox)};
    if (['idle', 'listening'].includes(state)) {
      if (touchPoint) return {kind: 'touch', point: touchPoint};
      if (pointerPoint) return {kind: 'pointer', point: pointerPoint};
    }
    return null;
  }
  function sideOf(point) { return point && box && point.x < box.left + box.width / 2 ? 'left' : 'right'; }
  function paintGaze() {
    gazeFrame = 0;
    const target = !asleep && !waking && visible && !document.hidden && previewState === null
      && !rainStill() && gesture === 'none' ? attentionTarget() : null;
    const point = target?.point;
    if (!point && duck.dataset.attention !== 'none' && gesture === 'none' && !asleep && !rainStill()) {
      data('centering', 'true'); clearTimeout(centeringTimer);
      centeringTimer = setTimeout(() => { centeringTimer = 0; data('centering', 'false'); }, 240);
    } else if (point) {
      clearTimeout(centeringTimer); centeringTimer = 0; data('centering', 'false');
    }
    data('attention', point ? target.kind : 'none');
    // Two small steps, matching the existing .8/1.5px reading glances.
    // Mirroring the pose reverses local X so screen-space tracking stays correct.
    const quantize = delta => Math.abs(delta) < 24 ? 0 : Math.sign(delta) * (Math.abs(delta) < 160 ? .8 : 1.5);
    const cx = box ? box.left + box.width * (duck.dataset.facing === 'left' ? .46 : .54) : 0;
    const cy = box ? box.top + box.height * .345 : 0;
    const dx = point ? quantize(point.x - cx) * (duck.dataset.facing === 'left' ? -1 : 1) : 0;
    const dy = point ? quantize(point.y - cy) : 0;
    duck.style.setProperty('--gaze-x', `${dx}px`); duck.style.setProperty('--gaze-y', `${dy}px`);
    considerAttentionTurn();
  }
  function requestGaze() { if (!gazeFrame) gazeFrame = requestAnimationFrame(paintGaze); }
  function clearAttention() {
    clearTimeout(pointerTimer); clearTimeout(typingTimer); clearTimeout(replyTimer); clearTimeout(turnTimer);
    clearTimeout(centeringTimer); centeringTimer = 0; data('centering', 'false');
    pointerTimer = typingTimer = replyTimer = turnTimer = 0;
    pointerPoint = touchPoint = null; typing = replyAttention = replyPending = false; behindSide = '';
    cancelAnimationFrame(gazeFrame); gazeFrame = 0;
    data('attention', 'none'); duck.style.setProperty('--gaze-x', '0px'); duck.style.setProperty('--gaze-y', '0px');
  }
  function considerAttentionTurn() {
    clearTimeout(turnTimer); turnTimer = 0;
    const target = attentionTarget(), side = target?.point && sideOf(target.point);
    if (!side || target.kind === 'touch' || side === duck.dataset.facing || stopped() || asleep || waking || previewState !== null) {
      behindSide = ''; return;
    }
    if (behindSide !== side) { behindSide = side; behindSince = Date.now(); }
    if (gesture !== 'none' || reacting || orienting) return;
    const dwell = target.kind === 'pointer' ? 1500 : 0;
    const delay = Math.max(0, behindSince + dwell - Date.now(), lastAttentionTurn + 4000 - Date.now());
    if (delay) { turnTimer = setTimeout(considerAttentionTurn, delay); return; }
    lastAttentionTurn = Date.now(); behindSide = '';
    beginGesture('turn', () => { render(); });
  }
  function startReplyAttention() {
    if (!replyPending || state !== 'speaking' || gesture !== 'none') return;
    replyPending = false;
    if (rainStill()) return;
    replyAttention = true; refreshAttentionRects();
    clearTimeout(replyTimer);
    replyTimer = setTimeout(() => { replyTimer = 0; replyAttention = false; requestGaze(); }, 2000);
    requestGaze();
  }
  function sleepEligible() {
    return visible && !document.hidden && previewState === null && liveState === 'idle' && state === 'idle' && !waking;
  }
  function synchronizeSleep() {
    if (asleep || !sleepEligible()) { clearTimeout(sleepTimer); sleepTimer = 0; return; }
    if (!sleepTimer) sleepTimer = setTimeout(checkSleep, Math.max(0, SLEEP_AFTER_MS - (Date.now() - lastActivity)));
  }
  function checkSleep() {
    sleepTimer = 0;
    if (!sleepEligible()) return;
    if (Date.now() - lastActivity >= SLEEP_AFTER_MS) fallAsleep(); else synchronizeSleep();
  }
  function fallAsleep() {
    if (!visible || document.hidden || liveState !== 'idle' || previewState !== null) return false;
    waking = false; clearTransient(); clearAttention(); clearTimeout(sleepTimer); sleepTimer = 0;
    asleep = true; data('asleep', 'true'); data('eyes', 'closed'); zzz.hidden = false;
    return true;
  }
  function activity() {
    lastActivity = Date.now();
    if (asleep) {
      asleep = false; data('asleep', 'false'); zzz.hidden = true; clearEyes();
      if (!stopped()) {
        waking = true; state = 'idle'; data('state', 'idle');
        beginGesture('wake', () => {
          waking = false;
          if (liveState !== 'idle') data('facing', liveFacing(liveState));
          if (queue.length) playGesture(queue.shift()); else finishReaction();
        });
      }
    }
    synchronizeSleep();
  }
  function pointerMove(event) {
    if (event.pointerType === 'touch') return;
    activity(); pointerPoint = {x: event.clientX, y: event.clientY}; touchPoint = null;
    clearTimeout(pointerTimer);
    pointerTimer = setTimeout(() => { pointerTimer = 0; pointerPoint = null; requestGaze(); }, 8000);
    requestGaze();
  }
  function pointerLeave() {
    pointerPoint = touchPoint = null; clearTimeout(pointerTimer); pointerTimer = 0;
    requestGaze();
  }
  function onDuck(target) { return target === duck || !!target?.closest?.('.duck'); }
  document.addEventListener('pointermove', pointerMove, {passive: true});
  document.addEventListener('pointerleave', pointerLeave);
  document.addEventListener('pointerout', event => { if (!event.relatedTarget) pointerLeave(); });
  window.addEventListener?.('blur', pointerLeave);
  document.addEventListener('keydown', activity, true);
  document.addEventListener('click', activity);
  document.addEventListener('pointerdown', event => {
    suppressDuckClick = asleep && onDuck(event.target);
    activity();
    if (event.pointerType !== 'touch' || onDuck(event.target)) return;
    touchPoint = {x: event.clientX, y: event.clientY}; pointerPoint = null;
    clearTimeout(pointerTimer);
    pointerTimer = setTimeout(() => { pointerTimer = 0; touchPoint = null; requestGaze(); }, 1500);
    requestGaze();
  }, {passive: true});
  document.addEventListener('pointercancel', () => { suppressDuckClick = false; });
  composer?.addEventListener('focus', activity);
  composer?.addEventListener('input', () => {
    activity();
    if (document.activeElement !== composer) return;
    typing = true; clearTimeout(typingTimer);
    typingTimer = setTimeout(() => { typingTimer = 0; typing = false; requestGaze(); }, 2000);
    requestGaze();
  });
  composer?.addEventListener('blur', () => {
    typing = false; clearTimeout(typingTimer); typingTimer = 0; requestGaze();
  });

  function synchronizeAnimations() {
    // Flush the settled pose before a queued repeat of the same CSS animation.
    duck.getAnimations?.({subtree: true});
  }
  function stopped() { return reduced || media.matches || !visible || document.hidden; }
  function clearEyes() {
    clearTimeout(eyeTimer); eyeTimer = 0;
    data('eyes', asleep ? 'closed' : 'open'); step = 0;
  }
  function clearTransient() {
    homeRoute = [];
    cancelCrossing();
    clearEyes();
    clearTimeout(gestureTimer); clearTimeout(randomTimer); clearTimeout(walkTimer); walkTimer = 0;
    facingTimers.forEach(clearTimeout); facingTimers = []; continuation = null; orienting = false;
    data('next', '');
    gestureTimer = randomTimer = 0;
    gesture = 'none'; queue = []; reacting = false;
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
    if (asleep || waking || stopped() || travel || previewState !== null || state !== 'idle' || gesture !== 'none' || queue.length) return;
    randomTimer = setTimeout(() => { randomTimer = 0; randomGestureNow(); }, 8000 + Math.floor(random() * 12001));
  }
  function scheduleEyes(delay = 4600) {
    if (asleep || waking || stopped() || gesture !== 'none') return;
    eyeTimer = setTimeout(() => {
      eyeTimer = 0;
      if (asleep || waking || stopped() || gesture !== 'none') return;
      data('eyes', state === 'reading' ? (step++ % 2 ? 'down-two' : 'down-one') : 'closed');
      eyeTimer = setTimeout(() => {
        eyeTimer = 0; data('eyes', 'open');
        scheduleEyes(state === 'reading' ? 1300 : state === 'remembering' ? 3300 : 4600);
      }, state === 'remembering' ? 180 : 160);
    }, delay);
  }
  // The dock remains the CSS stage/pinned anchor. Only its existing duck moves
  // into this body-level viewport; no clone, new artwork or template change.
  const scrollArea = document.querySelector('#content-scroll');
  const stripLayer = document.createElement('div');
  stripLayer.className = 'duck-strip-layer'; stripLayer.setAttribute('aria-hidden', 'true');
  document.body.append(stripLayer);
  let habitat = 'stage', stripBox = null, travel = null, travelTimer = 0;
  let stripWalkTarget = 0;
  data('habitat', habitat); data('travel', 'none');
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  function stripGeometry() {
    const scroll = scrollArea?.getBoundingClientRect();
    const column = document.querySelector('#messages')?.getBoundingClientRect();
    const result = {scroll, column, left: null, right: null};
    if (!scroll?.width || !scroll.height || !column?.width) return result;
    function band(left, right) {
      const width = right - left, size = width >= 168 ? 144 : width >= 144 ? 120 : 0;
      const minX = left + 12, maxX = right - 12 - size;
      const minY = Math.ceil(scroll.top + 12), maxY = Math.floor(scroll.bottom - 12 - size);
      if (!size || maxX < minX || maxY < minY) return null;
      return {left, right, size, minX, maxX, minY, maxY};
    }
    result.left = band(scroll.left, Math.min(scroll.right, column.left));
    result.right = band(Math.max(scroll.left, column.right), scroll.right);
    return result;
  }
  function paintStrip(point) {
    stripBox = {...point};
    const scroll = stripGeometry().scroll;
    duck.style.left = `${point.left - scroll.left}px`;
    duck.style.top = `${point.top - scroll.top}px`;
    duck.style.setProperty('--duck-size', `${point.size}px`);
    refreshAttentionRects(); requestGaze();
  }
  function clipStrips(geometry) {
    const {scroll, column} = geometry;
    if (!scroll?.width || !column) return;
    for (const key of ['left', 'top', 'width', 'height']) stripLayer.style[key] = `${scroll[key]}px`;
    // A U-shaped viewport excludes the entire message column (including C and
    // user bubbles). Above the first message, the stage's existing free row is
    // available for a short entry/return. The outer overflow clips top crossings.
    const left = clamp(column.left - scroll.left - 12, 0, scroll.width);
    const right = clamp(column.right - scroll.left + 12, 0, scroll.width);
    const top = clamp(column.top - scroll.top - 12, 0, scroll.height);
    stripLayer.style.clipPath = `polygon(0 0, 100% 0, 100% 100%, ${right}px 100%, ${right}px ${top}px, ${left}px ${top}px, ${left}px 100%, 0 100%)`;
  }
  function stripPoint(band, point = stripBox) {
    // Preserve a fractional layout edge at an exact 168/144px threshold rather
    // than rounding away the clearance. Travel steps themselves use whole px.
    return {left: clamp(Math.round(point.left), band.minX, band.maxX),
      top: Math.round(clamp(point.top, band.minY, band.maxY)), size: band.size};
  }
  function anchorPoint() {
    const rect = dock.getBoundingClientRect();
    return {left: rect.left, top: rect.top, size: rect.width};
  }
  function stopTravel() {
    clearTimeout(travelTimer); travelTimer = 0; travel = null; data('travel', 'none');
  }
  function restoreStage() {
    stopTravel(); dock.append(duck); habitat = 'stage'; stripBox = null;
    data('habitat', habitat);
    for (const key of ['left', 'top', '--duck-size']) duck.style.removeProperty(key);
    synchronizeRain(); refreshAttentionRects(); requestGaze();
  }
  function finishTravel() {
    const done = travel.done; stopTravel(); done?.();
    alignLiveFacing(); scheduleRandom(); requestGaze();
  }
  function alignLiveFacing() {
    if (!visible || state === 'idle' || reacting || waking || orienting) return;
    if (stopped()) data('facing', liveFacing(state));
    else setState(state);
  }
  function travelStep() {
    travelTimer = 0;
    if (!travel || document.hidden) return;
    const leg = travel.legs[travel.index], from = travel.from;
    // Scroll can keep moving the empty anchor during the final hop. Land on its
    // current position, so reparenting cannot add a one-frame correction.
    if (travel.kind === 'return' && travel.index === travel.legs.length - 1) Object.assign(leg, anchorPoint());
    const fraction = Math.min(1, (Date.now() - travel.started) / leg.ms);
    const t = travel.kind === 'cross' ? fraction : fraction * fraction * (3 - 2 * fraction);
    const mix = key => fraction === 1 ? leg[key] : from[key] === leg[key] ? from[key]
      : Math.round(from[key] + (leg[key] - from[key]) * t);
    const hop = ['enter', 'return'].includes(travel.kind) ? Math.round(6 * Math.sin(Math.PI * t)) : 0;
    paintStrip({left: mix('left'), top: mix('top') - hop, size: mix('size')});
    if (fraction === 1) {
      if (++travel.index === travel.legs.length) { finishTravel(); return; }
      travel.from = {...stripBox}; travel.started = Date.now();
    }
    travelTimer = setTimeout(travelStep, 20);
  }
  function startTravel(kind, legs, done = null) {
    stopTravel(); hideSay();
    if (stopped()) { paintStrip(legs.at(-1)); done?.(); alignLiveFacing(); return; }
    clearTimeout(randomTimer); randomTimer = 0;
    travel = {kind, legs, done, index: 0, from: {...stripBox}, started: Date.now(), layout: stripLayout()};
    data('travel', kind); travelTimer = setTimeout(travelStep, 20);
  }
  function attachStrip(side, point, geometry) {
    habitat = side; data('habitat', side); clipStrips(geometry);
    stripLayer.append(duck); paintStrip(point); synchronizeRain();
  }
  function topRoute(from, target, geometry) {
    const top = Math.floor(geometry.scroll.top - Math.max(from.size, target.size) - 12);
    return [{left: from.left, top, size: from.size, ms: 600},
      {left: target.left, top, size: target.size, ms: 400}, {...target, ms: 600}];
  }
  function stripLayout() {
    const {scroll, column} = stripGeometry();
    return [scroll.left, scroll.top, scroll.width, scroll.height, column.left, column.right].join(',');
  }
  function returnToStage(geometry) {
    if (gesture === 'cross') { stopTravel(); releaseCrossingPose(); }
    setPosition(Math.min(x, walkingBand().max));
    const target = anchorPoint(), from = stripBox;
    // Only the left strip connects to the stage walking band. A right-side
    // return takes the clipped top route even with the stage visible: its orb
    // occupies the middle of that row.
    const direct = from.left + from.size <= geometry.column.left - 12
      && (target.left + target.size <= geometry.column.left - 12
        || Math.max(from.top + from.size, target.top + target.size) <= geometry.column.top - 12);
    const legs = direct ? [{...target, ms: 600}] : topRoute(from, target, geometry);
    startTravel('return', legs, restoreStage);
  }
  function updateHabitat(immediate = false) {
    if (!chatView) { updateHome(); return; }
    if (!visible) { if (habitat !== 'stage') restoreStage(); return; }
    const geometry = stripGeometry(), {scroll} = geometry;
    if (!scroll?.width) return;
    clipStrips(geometry);
    if (stripBox) paintStrip(stripBox);
    if (stopped() && travel) {
      const last = travel.legs.at(-1), done = travel.done;
      if (travel.kind === 'return') Object.assign(last, anchorPoint());
      stopTravel(); paintStrip(last); done?.(); alignLiveFacing();
    }
    if (travel?.kind === 'cross' && travel.layout !== stripLayout()) cancelCrossing();
    // The anchor stays measurable even while its child is in the fixed layer.
    const home = anchorPoint();
    const pinned = window.innerWidth >= 1658;
    const homeVisible = home.top >= scroll.top && home.top + home.size <= scroll.bottom;
    const side = habitat !== 'stage' ? (geometry[habitat] ? habitat : null)
      : geometry.left ? 'left' : geometry.right ? 'right' : null;
    const wanted = side && (pinned || !homeVisible);
    if (immediate) {
      stopTravel();
      if (wanted) attachStrip(side, stripPoint(geometry[side], home), geometry);
      else if (habitat !== 'stage') restoreStage();
      return;
    }
    if (habitat === 'stage') {
      if (!wanted) return;
      const rect = duck.getBoundingClientRect();
      const from = {left: rect.left, top: rect.top, size: rect.width};
      attachStrip(side, from, geometry);
      // Horizontal stage travel must not continue after taking the strip.
      clearTimeout(walkTimer); walkTimer = 0;
      const target = stripPoint(geometry[side], from);
      startTravel('enter', side === 'right' ? topRoute(from, target, geometry) : [{...target, ms: 600}]);
    } else if (!wanted) {
      if (travel?.kind !== 'return') returnToStage(geometry);
    } else {
      if (travel?.kind === 'return') stopTravel();
      const target = stripPoint(geometry[side]);
      const last = travel?.legs.at(-1);
      if (travel && last && side === habitat && last.size === target.size
          && last.left >= geometry[side].minX && last.left <= geometry[side].maxX
          && last.top >= geometry[side].minY && last.top <= geometry[side].maxY) return;
      if (travel?.kind === 'cross' && geometry.left && geometry.right) return;
      if (side !== habitat || target.size !== stripBox.size || target.left !== stripBox.left || target.top !== stripBox.top) {
        habitat = side; data('habitat', side);
        startTravel('resize', [{...target, ms: 600}]);
      }
    }
  }
  function releaseCrossingPose() {
    if (gesture !== 'cross') return;
    clearTimeout(gestureTimer); gestureTimer = 0; gesture = 'none'; continuation = null;
    data('gesture', 'none'); data('part', 'none'); scheduleEyes();
  }
  function cancelCrossing() {
    if (travel?.kind !== 'cross') return;
    const geometry = stripGeometry();
    const side = stripBox.left + stripBox.size / 2 < (geometry.column.left + geometry.column.right) / 2 ? 'left' : 'right';
    stopTravel(); releaseCrossingPose();
    if (!geometry[side]) { updateHabitat(); return; }
    habitat = side; data('habitat', side);
    const target = stripPoint(geometry[side]);
    // A mid-crossing horizontal relocation is entirely above the clip.
    if (stripBox.top + stripBox.size <= geometry.scroll.top) paintStrip({...stripBox, left: target.left});
    startTravel('settle', [{...target, ms: 600}]);
  }
  function startCrossing() {
    const geometry = stripGeometry(), side = habitat === 'left' ? 'right' : 'left';
    if (habitat === 'stage' || travel || !geometry[side]) return false;
    const target = stripPoint(geometry[side]);
    const top = Math.floor(geometry.scroll.top - Math.max(stripBox.size, target.size) - 12);
    const legs = [{...stripBox, top}, {left: target.left, top, size: target.size}, target];
    let from = stripBox;
    for (const leg of legs) {
      leg.ms = Math.max(300, Math.ceil(Math.max(Math.abs(leg.top - from.top), Math.abs(leg.left - from.left)) * 4 / 20) * 20);
      from = leg;
    }
    DURATIONS.cross = legs.reduce((sum, leg) => sum + leg.ms, 0) + 20;
    beginGesture('cross');
    startTravel('cross', legs, () => { habitat = side; data('habitat', side); });
    return true;
  }
  function stripWalkStep() {
    walkTimer = 0;
    if (gesture !== 'walk' || stopped() || travel || habitat === 'stage') return;
    const band = stripGeometry()[habitat];
    if (!band) { updateHabitat(); return; }
    stripWalkTarget = clamp(stripWalkTarget, band.minY, band.maxY);
    const delta = Math.sign(stripWalkTarget - stripBox.top) * Math.min(2, Math.abs(stripWalkTarget - stripBox.top));
    paintStrip({...stripBox, top: Math.round(clamp(stripBox.top + delta, band.minY, band.maxY))});
    if (stripBox.top !== stripWalkTarget) walkTimer = setTimeout(stripWalkStep, 50);
  }
  function startStripWalk() {
    const band = stripGeometry()[habitat];
    if (!band || travel) return false;
    DURATIONS.walk = 1500 + Math.floor(random() * 3501);
    duck.style.setProperty('--duration-walk', `${DURATIONS.walk}ms`);
    const reach = Math.floor(DURATIONS.walk / 50) * 2;
    stripWalkTarget = Math.round(clamp(stripBox.top + (random() < .5 ? -reach : reach), band.minY, band.maxY));
    // Vertical movement keeps the current facing. Attention may turn afterwards.
    beginGesture('walk'); walkTimer = setTimeout(stripWalkStep, 50); return true;
  }
  // Home uses configuration-space rectangles: each point is the duck box's
  // top-left, already inset for its motion/ZZZ halo and 12px of clearance.
  // The orb's enclosing square is deliberately conservative at its corners.
  function homeGeometry() {
    const rect = selector => document.querySelector(selector)?.getBoundingClientRect();
    const stage = rect('.stage'), scroll = scrollArea?.getBoundingClientRect();
    const anchor = dock.getBoundingClientRect(), size = anchor.width;
    const haloX = state === 'searching' ? size / 4 : 0, haloY = size / 4;
    const result = {size, haloX, haloY, zones: [], obstacles: []};
    if (!stage?.width || !scroll?.width || !size) return result;
    const area = rect('.composer-area'), bar = rect('.topbar');
    const bounds = {minX: Math.ceil(Math.max(stage.left, scroll.left) + haloX + 12),
      maxX: Math.floor(Math.min(stage.right, scroll.right) - size - haloX - 12),
      minY: Math.ceil(Math.max(stage.top, scroll.top, bar?.bottom || 0) + haloY + 12),
      maxY: Math.floor(Math.min(scroll.bottom, area?.top ?? scroll.bottom) - size - 12)};
    result.bounds = bounds;
    const text = [];
    for (const element of document.querySelector('.welcome')?.querySelectorAll('p, h1') || []) {
      const range = document.createRange(); range.selectNodeContents(element);
      for (const line of range.getClientRects()) if (line.width && line.height) text.push(line);
      range.detach();
    }
    const orb = rect('#orb'), label = rect('.orb-status'), modes = rect('.modes');
    const obstacles = [orb, label, ...text, modes];
    // These may appear after choosing a mode/project. Hidden/empty hosts add no
    // obstacle; a new visible control cuts holes in the same free-space graph.
    for (const selector of ['#resume-card', '#projects-panel', '.create-chooser', '#messages', '.q-bookcase']) {
      const element = document.querySelector(selector);
      if (element && !element.hidden) obstacles.push(element.getBoundingClientRect());
    }
    result.obstacles = obstacles.filter(o => o?.width && o.height).map(o => ({
      minX: Math.floor(o.left - size - haloX - 12), maxX: Math.ceil(o.right + haloX + 12),
      minY: Math.floor(o.top - size - 12), maxY: Math.ceil(o.bottom + haloY + 12)}));
    const contains = (r, p) => p.left > r.minX && p.left < r.maxX && p.top > r.minY && p.top < r.maxY;
    function add(name, candidate) {
      let pieces = [{...bounds, ...candidate}];
      for (const o of result.obstacles) {
        const next = [];
        for (const r of pieces) {
          const a = Math.max(r.minX, o.minX), b = Math.min(r.maxX, o.maxX);
          const c = Math.max(r.minY, o.minY), d = Math.min(r.maxY, o.maxY);
          if (a >= b || c >= d) { next.push(r); continue; }
          next.push({...r, maxX: a}, {...r, minX: b},
            {...r, minX: a, maxX: b, maxY: c}, {...r, minX: a, maxX: b, minY: d});
        }
        pieces = next.filter(r => r.maxX >= r.minX && r.maxY >= r.minY);
      }
      for (const r of pieces) if (r.maxX >= r.minX && r.maxY >= r.minY
          && !result.obstacles.some(o => contains(o, {left: (r.minX + r.maxX) / 2, top: (r.minY + r.maxY) / 2})))
        result.zones.push({name, ...r});
    }
    if (!orb?.width || bounds.maxX < bounds.minX || bounds.maxY < bounds.minY) return result;
    const rowEnd = Math.min(bounds.maxY, Math.floor(stage.bottom - size - 12));
    if (orb.top - size - 12 >= bounds.minY) {
      // Compact layouts put the orb below the duck's own row, like chat.
      add('stage-left', {maxY: Math.min(rowEnd, Math.floor(orb.top - size - 12))});
    } else {
      add('stage-left', {maxX: Math.min(bounds.maxX, Math.floor(orb.left - size - haloX - 12)), maxY: rowEnd});
      add('stage-right', {minX: Math.max(bounds.minX, Math.ceil(orb.right + haloX + 12)), maxY: rowEnd});
    }
    const centreObstacles = [orb, label, ...text, modes].filter(o => o?.width && o.height);
    add('left-column', {maxX: Math.min(bounds.maxX, ...centreObstacles.map(o => Math.floor(o.left - size - haloX - 12)))});
    add('right-column', {minX: Math.max(bounds.minX, ...centreObstacles.map(o => Math.ceil(o.right + haloX + 12)))});
    const below = [modes, ...text].filter(o => o?.width && o.height);
    if (below.length) add('bottom', {minY: Math.max(bounds.minY, ...below.map(o => Math.ceil(o.bottom + haloY + 12)))});
    return result;
  }
  const inZone = (point, zone) => point.left >= zone.minX && point.left <= zone.maxX
    && point.top >= zone.minY && point.top <= zone.maxY;
  function homePath(from, to, geometry = homeGeometry()) {
    const zones = geometry.zones, previous = new Map(), todo = [];
    zones.forEach((z, i) => { if (inZone(from, z)) { previous.set(i, -1); todo.push(i); } });
    let end = -1;
    for (let cursor = 0; cursor < todo.length; cursor++) {
      const i = todo[cursor], a = zones[i];
      if (inZone(to, a)) { end = i; break; }
      zones.forEach((b, j) => {
        if (!previous.has(j) && Math.max(a.minX, b.minX) <= Math.min(a.maxX, b.maxX)
            && Math.max(a.minY, b.minY) <= Math.min(a.maxY, b.maxY)) {
          previous.set(j, i); todo.push(j);
        }
      });
    }
    if (end < 0) return [];
    const chain = [];
    for (let i = end; i !== -1; i = previous.get(i)) chain.unshift(zones[i]);
    let point = {left: from.left, top: from.top};
    const path = [];
    function leg(target) {
      if (point.left !== target.left) { point = {...point, left: target.left}; path.push(point); }
      if (point.top !== target.top) { point = {...point, top: target.top}; path.push(point); }
    }
    for (let i = 1; i < chain.length; i++) {
      const a = chain[i - 1], b = chain[i];
      leg({left: clamp(point.left, Math.max(a.minX, b.minX), Math.min(a.maxX, b.maxX)),
        top: clamp(point.top, Math.max(a.minY, b.minY), Math.min(a.maxY, b.maxY))});
    }
    leg(to); return path.length ? path : [{left: to.left, top: to.top}];
  }
  function paintHome(point) {
    const anchor = dock.getBoundingClientRect();
    homeY = Math.round(point.top - (anchor.top - homeY));
    data('y', String(homeY)); dock.style.setProperty('--duck-y', `${homeY}px`);
    setPosition(point.left - (anchor.left - x)); requestGaze();
  }
  function updateHome() {
    if (habitat !== 'stage') restoreStage();
    const geometry = homeGeometry(), changed = JSON.stringify(geometry) !== JSON.stringify(homeMap);
    homeMap = geometry;
    if (!visible || !changed) return;
    homeGoal = null;
    if (homeRoute.length || gesture === 'walk' || duck.dataset.next === 'walk') { clearTransient(); scheduleEyes(); scheduleRandom(); }
    const point = dock.getBoundingClientRect();
    if (!geometry.zones.some(z => inZone(point, z)) && geometry.zones.length) {
      const nearest = geometry.zones.map(z => ({left: clamp(Math.round(point.left), z.minX, z.maxX),
        top: clamp(Math.round(point.top), z.minY, z.maxY)})).sort((a, b) =>
        Math.abs(a.left - point.left) + Math.abs(a.top - point.top) - Math.abs(b.left - point.left) - Math.abs(b.top - point.top))[0];
      // Layout changes and preview halos can invalidate a resting point. Place
      // immediately rather than animate through a newly occupied surface.
      paintHome(nearest);
    }
  }
  function homeWalkStep() {
    walkTimer = 0;
    if (chatView || stopped() || gesture !== 'walk' || !homeRoute.length) return;
    const from = dock.getBoundingClientRect(), target = homeRoute[0];
    const direction = target.left < from.left ? 'left' : target.left > from.left ? 'right' : null;
    if (direction && direction !== duck.dataset.facing) {
      clearTimeout(gestureTimer); gestureTimer = 0;
      beginGesture('turn', () => { beginGesture('walk'); walkTimer = setTimeout(homeWalkStep, 20); });
      return;
    }
    const step = key => from[key] + Math.sign(target[key] - from[key]) * Math.min(4, Math.abs(target[key] - from[key]));
    const point = {left: step('left'), top: step('top')}; paintHome(point); footprint();
    if (point.left === target.left && point.top === target.top) homeRoute.shift();
    if (!homeRoute.length) homeGoal = null;
    else walkTimer = setTimeout(homeWalkStep, 20);
  }
  function startHomeWalk() {
    updateHome();
    const from = dock.getBoundingClientRect(), geometry = homeMap;
    const current = geometry.zones.filter(z => inZone(from, z));
    if (!current.length) return false;
    DURATIONS.walk = 1500 + Math.floor(random() * 3501);
    duck.style.setProperty('--duration-walk', `${DURATIONS.walk}ms`);
    if (!homeGoal || !homePath(from, homeGoal, geometry).length) {
      const others = geometry.zones.filter(z => !inZone(from, z) && homePath(from,
        {left: Math.round((z.minX + z.maxX) / 2), top: Math.round((z.minY + z.maxY) / 2)}, geometry).length);
      const change = others.length && random() < .45;
      const choices = change ? others : current;
      const zone = choices[Math.floor(random() * choices.length)];
      homeGoal = {left: Math.round(zone.minX + random() * (zone.maxX - zone.minX)),
        top: change ? Math.round((zone.minY + zone.maxY) / 2) : from.top};
    }
    const route = homePath(from, homeGoal, geometry);
    if (!route.length) { homeGoal = null; return false; }
    homeRoute = route; beginGesture('walk'); walkTimer = setTimeout(homeWalkStep, 20); return true;
  }
  // Task 160: things the duck leaves behind. New art only (rect cell maps, never
  // the PNG), drawn inside the box the duck occupies at that moment, on the stage
  // it is standing on. Any layout change wipes them, so they can never end up
  // over text or the sphere.
  const PROP_ART = {
    print: {palette: {O: 'yellow'}, rows: ['O.O.O', 'O.O.O', '.OOO.', '..O..']},
    poop: {palette: {B: 'rgb(122 74 34)', L: 'rgb(176 122 60)', W: 'rgb(244 247 251)', K: 'outline'}, rows: [
      '....B....', '...BBB...', '...BLB...', '..BBBBB..', '..BLBBB..', '.BWKBWKB.', '.BLBBBBB.', 'BBBBBBBBB']},
    crack: {palette: {W: 'rgb(230 237 247)', G: 'rgb(154 170 192)'}, rows: [
      '......W.........', '......W....G....', '.G....WW..GG....', '..GG...W.G......', '....WW.WWW......',
      '......WWW.......', '....WWW.WWWW....', '..WW...W....WW..', '.W.....W......G.', '.......WG.......', '.......W.G......']}
  };
  const stageHost = document.querySelector('.stage');
  let props = [], strideCount = 0;
  function propsAllowed() {
    return !!stageHost && habitat === 'stage' && duck.parentNode === dock && !stopped()
      && !(chatView && window.innerWidth >= 1658);
  }
  function removeProp(entry) {
    clearTimeout(entry.timer); entry.node.remove(); props = props.filter(item => item !== entry);
  }
  function clearProps() { for (const entry of [...props]) removeProp(entry); }
  function dropProp(name, fx, fy, life, front = false) {
    if (!propsAllowed()) return null;
    const {palette, rows} = PROP_ART[name], box = duck.getBoundingClientRect(), host = stageHost.getBoundingClientRect();
    const unit = Math.max(2, Math.round(box.width / 48)), width = rows[0].length * unit, height = rows.length * unit;
    const node = document.createElementNS(racket.namespaceURI, 'svg');
    node.setAttribute('class', `duck-prop duck-prop-${name}`); node.setAttribute('aria-hidden', 'true');
    node.setAttribute('shape-rendering', 'crispEdges'); node.setAttribute('viewBox', `0 0 ${rows[0].length} ${rows.length}`);
    rows.forEach((row, y) => [...row].forEach((cell, x) => {
      if (cell === '.') return;
      const rect = document.createElementNS(racket.namespaceURI, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', y); rect.setAttribute('width', 1); rect.setAttribute('height', 1);
      rect.setAttribute('fill', Object.hasOwn(SAMPLES, palette[cell]) ? colour(SAMPLES[palette[cell]]) : palette[cell]);
      node.append(rect);
    }));
    node.style.width = `${width}px`; node.style.height = `${height}px`;
    node.style.left = `${Math.round(box.left - host.left + box.width * fx - width / 2)}px`;
    node.style.top = `${Math.round(box.top - host.top + box.height * fy - height)}px`;
    if (front) stageHost.append(node); else stageHost.prepend(node);
    const entry = {node, name, timer: 0};
    entry.timer = setTimeout(() => {
      node.dataset.gone = 'true'; entry.timer = setTimeout(() => removeProp(entry), 900);
    }, life);
    props.push(entry);
    while (props.length > 24) removeProp(props[0]);
    return entry;
  }
  let lastPrank = Date.now();
  function prankReady() { return Date.now() - lastPrank >= PRANK_GAP_MS && propsAllowed(); }
  function leavePoop() {
    // Behind the duck, at its feet, inside its own box; a click cleans it up.
    // (24 s: outside the 8-20 s window of the rest timer.)
    const entry = dropProp('poop', duck.dataset.facing === 'left' ? .8 : .2, .95, 24000, true);
    if (!entry) return;
    entry.node.addEventListener('click', () => {
      if (!props.includes(entry)) return;
      removeProp(entry);
      if (asleep || waking || stopped()) return;
      if (state === 'idle' && gesture === 'none' && !travel) beginGesture('startle');
      say('Tá bom, limpei.', 2400);
    });
    say('Ops.', 2400);
  }
  function leaveCrack() {
    // Task 164: the screen "cracks" where the beak hit it — in front of the duck,
    // inside its own box, drawn over it like glass. It mends by itself in 5 s.
    if (!dropProp('crack', duck.dataset.facing === 'left' ? .2 : .8, .62, 5000, true)) return;
    say('Foi sem querer.', 2400);
  }
  function footprint() {
    // One print every sixth 50ms step, alternating feet, under the foot itself.
    if (strideCount++ % 6) return;
    const foot = Math.floor(strideCount / 6) % 2 ? .35 : .53;
    dropProp('print', duck.dataset.facing === 'left' ? 1 - foot : foot, .94, 5000);
  }
  function setPosition(value) {
    x = Math.round(value); data('x', String(x)); dock.style.setProperty('--duck-x', `${x}px`); refreshAttentionRects();
  }
  function walkingBand() {
    const rect = selector => document.querySelector(selector)?.getBoundingClientRect();
    const stage = rect('.stage'), box = dock.getBoundingClientRect();
    if (!stage || !box.width) return {min: 0, max: 0};
    const size = box.width, home = dock.getBoundingClientRect().left - x;
    const top = box.top - size * .25, bottom = box.top + size;
    // Reserve the entire rain rectangle, including its transparent mask edges.
    let max = stage.right - 12 - home - size * 1.25;
    const pinned = window.innerWidth >= 1658;
    const obstacles = ['#orb', '.orb-status', '#messages'];
    // The scrolling dock is clipped before it can paint over these surfaces.
    if (pinned) obstacles.push('.composer-area', '.topbar');
    for (const selector of obstacles) {
      const obstacle = rect(selector);
      if (!obstacle || !obstacle.width || (!obstacle.height && !(pinned && selector === '#messages'))) continue;
      let left = obstacle.left;
      if (!(pinned && selector === '#messages')) {
        // Fractional layout must not turn an edge-touch into an overlap.
        if (Math.round(bottom + 12) <= Math.round(obstacle.top) ||
            Math.round(top - 12) >= Math.round(obstacle.bottom)) continue;
        if (selector === '#orb') {
          // Project the real circle into the duck's row, expanded by clearance.
          const radius = Math.min(obstacle.width, obstacle.height) / 2;
          const cy = obstacle.top + obstacle.height / 2;
          const dy = Math.max(top - 12 - cy, cy - bottom - 12, 0);
          if (dy >= radius) continue;
          left = obstacle.left + obstacle.width / 2 - Math.sqrt(radius * radius - dy * dy);
        }
      }
      // Only obstacles ahead of home bound this connected horizontal interval.
      // A newly overlapping obstacle collapses it, prompting a gradual return.
      if (obstacle.right > home - size * .25 - 12)
        max = Math.min(max, left - 12 - home - size * 1.25);
    }
    return {min: 0, max: Math.max(0, Math.floor(max))};
  }
  function flip() { data('facing', duck.dataset.facing === 'right' ? 'left' : 'right'); requestGaze(); }
  function walkStep() {
    walkTimer = 0;
    if (gesture !== 'walk' || habitat !== 'stage' || stopped()) return;
    const stride = Math.max(1, Math.round(2 * duck.getBoundingClientRect().width / 144));
    const delta = Math.sign(destination - x) * Math.min(stride, Math.abs(destination - x));
    if (returning) setPosition(x + delta); // resize recovery: no teleport across the band
    else setPosition(Math.min(walkingLimit, Math.max(0, x + delta)));
    footprint();
    if (x !== destination) walkTimer = setTimeout(walkStep, 50);
  }
  function startWalk(target = null, recovery = false) {
    if (!chatView) return startHomeWalk();
    if (habitat !== 'stage') return startStripWalk();
    const band = walkingBand(); walkingLimit = band.max;
    recovery = recovery || x > band.max; returning = recovery;
    if (recovery && target === null) target = band.max;
    const size = duck.getBoundingClientRect().width;
    const stride = Math.max(1, Math.round(2 * size / 144));
    DURATIONS.walk = 1500 + Math.floor(random() * 3501);
    duck.style.setProperty('--duration-walk', `${DURATIONS.walk}ms`);
    // Sometimes choose home; otherwise travel outwards. End before the deadline.
    const homeward = x > 0 && random() < .35;
    destination = target ?? (homeward ? 0 : band.max);
    const reach = Math.floor(DURATIONS.walk / 50) * stride;
    destination = Math.round(Math.max(x - reach, Math.min(x + reach, destination)));
    if (!recovery && band.max < Math.max(4, Math.round(8 * size / 144))) destination = x;
    const direction = destination < x ? 'left' : destination > x ? 'right' : duck.dataset.facing;
    const begin = () => {
      // The preceding turn can outlast a scroll-triggered drop. Re-evaluate
      // the habitat before starting its walk; the empty dock must stay put.
      if (habitat !== 'stage') {
        if (!startStripWalk()) { scheduleEyes(); scheduleRandom(); }
        return;
      }
      walkingLimit = walkingBand().max;
      if (!returning) destination = Math.min(destination, walkingLimit);
      beginGesture('walk');
      if (destination !== x) walkTimer = setTimeout(walkStep, 50);
    };
    if (direction !== duck.dataset.facing) {
      data('next', 'walk'); beginGesture('turn', begin);
    } else begin();
    return true;
  }
  // Task 161: a short compliment in a speech bubble. Fixed phrases chosen by
  // Pedro (tennis + cheeky), in rotation: nothing here comes from the model.
  const PHRASES = Object.freeze(['Saque perfeito.', 'Até que enfim, hein?', 'Game, set e match.',
    'Eu faria mais rápido. Mentira.', 'Essa foi na linha.', 'Olha ele trabalhando!', 'Ace!',
    'Tá bonito hoje. A tela, digo.']);
  const THANKS = ['obrigado', 'obrigada', 'brigado', 'brigada', 'valeu', 'vlw', 'thanks', 'thank you', 'tks'];
  const COMPLIMENT_EVERY = 6, COMPLIMENT_GAP_MS = 270000;
  const SAY_OBSTACLES = ['#orb', '.orb-status', '#messages', '.composer-area', '.topbar', '.modes',
    '.welcome', '#resume-card', '#projects-panel', '.create-chooser', '.q-bookcase'];
  let bubble = null, bubbleTimer = 0, pendingSay = null;
  let phraseIndex = 0, repliesDone = 0, lastCompliment = -Infinity;
  function detectThanks(text) {
    return fold(text).split(/[.!?;\n]+/u).some(sentence => {
      const tokens = words(sentence);
      return tokens.some((token, i) => (THANKS.includes(token) || THANKS.includes(tokens.slice(i, i + 2).join(' ')))
        && !isNegated(tokens, i));
    });
  }
  function hideSay() { clearTimeout(bubbleTimer); bubbleTimer = 0; bubble?.remove(); bubble = null; }
  function sayFits(spot) {
    // Inside the content area and 8px clear of the sphere, its label and every text block.
    const area = scrollArea?.getBoundingClientRect();
    if (area?.width && (spot.left < area.left + 8 || spot.right > area.right - 8
      || spot.top < area.top + 8 || spot.bottom > area.bottom - 8)) return false;
    return SAY_OBSTACLES.every(selector => {
      const o = document.querySelector(selector)?.getBoundingClientRect();
      return !o?.width || !o.height || spot.right + 8 <= o.left || spot.left - 8 >= o.right
        || spot.bottom + 8 <= o.top || spot.top - 8 >= o.bottom;
    });
  }
  function say(text, life = 3200) {
    hideSay();
    if (stopped() || asleep || waking) return false;
    const node = document.createElement('div');
    node.className = 'duck-say'; node.setAttribute('aria-hidden', 'true'); node.textContent = text;
    duck.append(node);
    const box = duck.getBoundingClientRect(), own = node.getBoundingClientRect();
    // Beside the head on the right, else on the left, else above; with no free spot, no bubble.
    for (const [side, dx, dy] of [['right', box.width + 10, box.height * .04],
      ['left', -own.width - 10, box.height * .04], ['above', (box.width - own.width) / 2, -own.height - 8]]) {
      const left = Math.round(box.left + dx), top = Math.round(box.top + dy);
      if (!sayFits({left, top, right: left + own.width, bottom: top + own.height})) continue;
      node.dataset.side = side;
      node.style.left = `${left - box.left}px`; node.style.top = `${top - box.top}px`;
      bubble = node; bubbleTimer = setTimeout(hideSay, life);
      return true;
    }
    node.remove(); return false;
  }
  function checkSay() {
    // After a layout change the bubble stays only if its place is still free.
    if (!bubble) return;
    const r = bubble.getBoundingClientRect();
    if (!sayFits({left: r.left, top: r.top, right: r.right, bottom: r.bottom})) hideSay();
  }
  // Task 165: when the bubble cannot be shown (no free spot beside the duck, or
  // reduced motion, where the duck does not cheer), the same phrase goes to the
  // quiet label under the sphere instead of being dropped.
  // The label is free only at rest: a phrase earned while a turn runs (a thank-you
  // starts one) waits, for two minutes at most, until the label is empty again.
  const CAPTION_MS = 3600;
  let captionLater = null, captionTimer = 0;
  function hideCaption() {
    if (!captionTimer) return;
    clearTimeout(captionTimer); captionTimer = 0; window.cortexCore?.caption?.('');
  }
  function caption(text, wait = false) {
    captionLater = null; hideCaption();
    if (!wait && window.cortexCore?.caption?.('Pato · ' + text) === true) {
      captionTimer = setTimeout(hideCaption, CAPTION_MS); return true;
    }
    captionLater = {text, until: Date.now() + 120000};
    return false;
  }
  function captionDue() {
    const due = captionLater; captionLater = null;
    if (due && Date.now() <= due.until && previewState === null && !document.hidden) caption(due.text);
  }
  function stillOnly() { return (reduced || media.matches) && visible && !document.hidden && !asleep && !waking; }
  function captionNow(wait = false) {
    lastCompliment = Date.now(); caption(PHRASES[phraseIndex++ % PHRASES.length], wait);
  }
  function promiseSay(gestureName) {
    lastCompliment = Date.now();
    pendingSay = {text: PHRASES[phraseIndex++ % PHRASES.length], on: gestureName, until: Date.now() + 15000};
  }
  function compliment() {
    if (asleep || waking || stopped() || state !== 'idle' || gesture !== 'none') return false;
    promiseSay('cheer'); beginGesture('cheer'); return true;
  }
  function beginGesture(name, after = null) {
    clearEyes(); clearTimeout(randomTimer); randomTimer = 0;
    clearTimeout(turnTimer); turnTimer = 0; data('attention', 'none');
    clearTimeout(centeringTimer); centeringTimer = 0; data('centering', 'false');
    gesture = name; continuation = after;
    data('gesture', name);
    // The bubble is measured for one spot: it leaves when the duck starts to move.
    if (['walk', 'cross'].includes(name)) hideSay();
    if (pendingSay && name === pendingSay.on) {
      if (Date.now() <= pendingSay.until && !say(pendingSay.text)) caption(pendingSay.text);
      pendingSay = null;
    }
    data('part', ['wave', 'racket'].includes(name) ? 'wing' : ['look', 'no'].includes(name) ? 'head'
      : ['walk', 'cross', 'pirouette', 'flap', 'stumble', 'dance', 'cheer', 'startle', 'poop'].includes(name) ? 'limbs' : 'none');
    if (name === 'turn') facingTimers.push(setTimeout(flip, DURATIONS.turn / 2));
    if (name === 'poop') { lastPrank = Date.now(); facingTimers.push(setTimeout(leavePoop, 1150)); }
    if (name === 'knock') {
      lastPrank = Date.now();
      facingTimers.push(setTimeout(() => say('toc', 420), 250), setTimeout(() => say('toc', 420), 760),
        setTimeout(leaveCrack, 1150));
    }
    if (name === 'pirouette') {
      facingTimers.push(setTimeout(flip, DURATIONS.pirouette * .25), setTimeout(flip, DURATIONS.pirouette * .75));
    }
    synchronizeAnimations();
    gestureTimer = setTimeout(() => {
      gestureTimer = 0; gesture = 'none'; clearTimeout(walkTimer); walkTimer = 0;
      facingTimers.forEach(clearTimeout); facingTimers = [];
      data('gesture', 'none'); data('part', 'none'); data('eyes', 'open'); data('next', '');
      synchronizeAnimations();
      const next = continuation; continuation = null;
      if (next && !stopped()) next();
      else if (queue.length && !stopped()) playGesture(queue.shift());
      else if (reacting) finishReaction();
      else if (chatView && name === 'walk' && habitat === 'stage' && x > walkingBand().max && !stopped()) resizeBand();
      else { scheduleEyes(); scheduleRandom(); startReplyAttention(); requestGaze(); }
    }, DURATIONS[name]);
  }
  function playGesture(name) {
    if (name === 'sleep') return fallAsleep();
    if (name === 'compliment') return compliment();
    if (['poop', 'knock'].includes(name) && !propsAllowed()) return false;
    if (name === 'wake' || asleep || waking || !Object.hasOwn(DURATIONS, name) || stopped() || state !== 'idle' || gesture !== 'none') return false;
    if (name === 'cross') return startCrossing();
    if (name === 'walk') return startWalk();
    beginGesture(name); return true;
  }
  function randomGestureNow(includeRacket = false) {
    if (asleep || waking || stopped() || travel || previewState !== null || state !== 'idle' || gesture !== 'none' || queue.length) return false;
    // Clicks share the rest generator and exclusion history, plus the click-only moves.
    const pool = {...WEIGHTS, ...(includeRacket ? CLICK_WEIGHTS : prankReady() ? PRANK_WEIGHTS : {}),
      ...(chatView && habitat !== 'stage' && stripGeometry().left && stripGeometry().right ? {cross: 1} : {})};
    const choices = Object.entries(pool).filter(([name]) => name !== previousRandom);
    let ticket = random() * choices.reduce((total, [, weight]) => total + weight, 0);
    const name = choices.find(([, weight]) => (ticket -= weight) < 0)[0];
    if (!playGesture(name)) return false;
    previousRandom = name;
    return true;
  }
  function setSeed(value) { seed = Number(value) >>> 0; previousRandom = ''; scheduleRandom(); }
  function resizeBand() {
    clearProps(); checkSay();
    if (!chatView) { updateHome(); refreshAttentionRects(); requestGaze(); return; }
    updateHabitat();
    if (habitat !== 'stage') return;
    refreshAttentionRects(); requestGaze();
    if (!visible) return;
    const max = walkingBand().max;
    if (x > max && !asleep && !waking && state === 'idle' && !reacting && !stopped()) {
      clearTransient(); startWalk(max, true);
    } else if (x > max && (reduced || media.matches)) setPosition(max);
    else if (gesture === 'walk' && !returning) {
      walkingLimit = max; destination = Math.min(destination, max);
    }
  }
  function requestGestures(names) {
    if (stopped() || previewState !== null || !names.length) return;
    // Sent-text and welcome reactions own the rest pose until their bounded
    // queue finishes. liveState keeps advancing independently (latest wins).
    if (!reacting && !waking) { state = 'idle'; render(); }
    reacting = true;
    queue.push(...names.slice(0, Math.max(0, 8 - queue.length)));
    if (!waking && gesture === 'none' && queue.length) playGesture(queue.shift());
  }
  function render() {
    clearTransient();
    if (state !== 'idle' && (reduced || media.matches)) data('facing', liveFacing(state));
    data('state', state); data('reduced', String(reduced || media.matches));
    data('paused', String(document.hidden || !visible));
    if (!stopped()) data('part', state === 'writing' ? 'wing' : state !== 'idle' ? 'head' : 'none');
    scheduleEyes(state === 'reading' ? 900 : state === 'remembering' ? 1400 : 4600);
    scheduleRandom(); synchronizeRain();
    if (!chatView) updateHome();
    data('asleep', String(asleep)); if (asleep) data('eyes', 'closed');
    startReplyAttention(); requestGaze(); synchronizeSleep();
  }
  function liveFacing(next) {
    return habitat !== 'stage' ? (habitat === 'right' ? 'left' : 'right')
      : next === 'speaking' ? sideOf(centre(messageBox)) : 'right';
  }
  function setState(next) {
    if (next !== 'idle') cancelCrossing();
    if (next === state && !(next !== 'idle' && duck.dataset.facing !== liveFacing(next) && !stopped())) return;
    if (next !== 'idle' && duck.dataset.facing !== liveFacing(next) && !stopped()) {
      clearTransient(); state = 'idle'; data('state', 'idle'); orienting = true;
      beginGesture('turn', () => { orienting = false; state = previewState ?? liveState; render(); });
    } else { state = next; render(); if (next === 'idle') resizeBand(); }
  }
  function setLive(next) {
    if (!Object.hasOwn(LABELS, next)) return;
    const firstReply = next === 'speaking' && liveState !== 'speaking';
    // Now and then a finished reply earns one too: every sixth, at most one per four and a half minutes (Pedro's choice).
    const earned = liveState === 'speaking' && next === 'idle' && ++repliesDone % COMPLIMENT_EVERY === 0
      && Date.now() - lastCompliment >= COMPLIMENT_GAP_MS;
    liveState = next; activity();
    if (next === 'idle') captionDue();
    if (firstReply) { replyPending = true; refreshAttentionRects(); }
    if (next !== 'speaking') { replyPending = replyAttention = false; clearTimeout(replyTimer); replyTimer = 0; }
    if (waking) return;
    if ((reacting || orienting) && !stopped() && previewState === null) return;
    setState(previewState ?? liveState);
    if (earned && !stopped() && previewState === null) { promiseSay('cheer'); requestGestures(['cheer']); }
    else if (earned && stillOnly() && previewState === null) captionNow();
  }
  function finishReaction() {
    const next = previewState ?? liveState;
    clearTransient();
    // Queue completion applies the latest live state with the same facing rule.
    if (!stopped() && next !== 'idle' && duck.dataset.facing !== liveFacing(next)) {
      state = 'idle'; setState(next);
    } else { state = next; render(); if (next === 'idle') resizeBand(); }
  }
  function cancelGestures() {
    if (waking) waking = false;
    state = previewState ?? liveState; render();
  }
  function preview(next) {
    if (next !== null && !Object.hasOwn(LABELS, next)) return;
    previewState = next; activity(); clearAttention(); cancelGestures();
  }
  function environment() {
    const nextChat = document.body.classList.contains('chat-view');
    const switched = nextChat !== chatView;
    const facing = duck.dataset.facing;
    const nextVisible = nextChat || !document.body.classList.contains('projects-view');
    const nextReduced = media.matches || document.body.classList.contains('reduced-motion');
    const becameVisible = !visible && nextVisible;
    const changed = switched || visible !== nextVisible || reduced !== nextReduced;
    chatView = nextChat;
    if (switched) {
      clearTransient(); restoreStage(); homeGoal = null; homeMap = null;
      homeY = 0; data('y', '0'); dock.style.setProperty('--duck-y', '0px'); setPosition(0);
    }
    visible = nextVisible; reduced = nextReduced;
    if (changed) {
      clearProps(); hideSay(); pendingSay = null; captionLater = null; hideCaption(); clearAttention(); refreshAttentionRects();
      cancelGestures();
      if (becameVisible) activity();
    } else synchronizeRain();
    updateHabitat(switched);
    if (switched) data('facing', facing);
    synchronizeSleep();
    if (visible && !welcomed) {
      welcomed = true; requestGestures(['wave']);
    }
  }
  function onSent(text) {
    environment(); activity();
    const signals = detectMessage(text), thanks = detectThanks(text);
    const names = [...(signals.greeting ? ['wave'] : []), ...(signals.happy ? ['racket'] : [])];
    // A thank-you earns a compliment on the celebration it already causes, or on a cheer.
    if (thanks && !signals.happy) names.push('cheer');
    if (thanks && !stopped() && previewState === null) promiseSay(names.at(-1));
    else if (thanks && stillOnly() && previewState === null) captionNow(true);
    requestGestures(names);
  }
  duck.addEventListener('click', () => {
    if (asleep || suppressDuckClick) { suppressDuckClick = false; activity(); return; }
    if (liveState === 'idle') randomGestureNow(true);
  });
  new MutationObserver(environment).observe(document.body, {attributes: true, attributeFilter: ['class']});
  media.addEventListener('change', environment);
  document.addEventListener('visibilitychange', () => {
    clearAttention();
    if (document.hidden) { clearProps(); hideSay(); pendingSay = null; captionLater = null; hideCaption(); }
    cancelGestures(); updateHabitat();
    if (!document.hidden) { refreshAttentionRects(); activity(); }
    synchronizeSleep();
    for (const figure of rainFigures) figure.node.dataset.paused = String(document.hidden);
  });
  const bandObserver = new ResizeObserver(resizeBand);
  for (const selector of ['.stage', '#orb', '.orb-status', '#messages', '#content-scroll', '.sidebar', '.composer-area', '.topbar', '.welcome', '.modes', '#resume-card', '#projects-panel', '.create-chooser']) {
    const element = document.querySelector(selector); if (element) bandObserver.observe(element);
  }
  // Position-only changes (e.g. a sidebar class or column margin) need not fire
  // ResizeObserver. Watch layout hosts, never the duck's own animated styles.
  const layoutObserver = new MutationObserver(resizeBand);
  for (const selector of ['.sidebar', '.workspace', '.stage', '#content-scroll', '#messages']) {
    const element = document.querySelector(selector);
    if (element) layoutObserver.observe(element, {attributes: true, attributeFilter: ['class', 'style']});
  }
  const observedHome = new Set();
  const homeObserver = new MutationObserver(() => { observeHome(); resizeBand(); });
  function observeHome() {
    for (const selector of ['.welcome', '.modes', '#resume-card', '#projects-panel', '.create-chooser']) {
      const element = document.querySelector(selector);
      if (!element || observedHome.has(element)) continue;
      observedHome.add(element); bandObserver.observe(element);
      homeObserver.observe(element, {attributes: true, childList: true, characterData: true, subtree: true});
    }
  }
  // The shell inserts its create chooser after this module initializes.
  if (scrollArea) homeObserver.observe(scrollArea, {childList: true});
  observeHome();
  window.addEventListener?.('resize', resizeBand);
  window.addEventListener?.('scroll', () => { updateHabitat(); refreshAttentionRects(); requestGaze(); }, {capture: true, passive: true});
  window.cortexAvatar = Object.freeze({setLive, preview, onSent, cancelGestures, createHead, updateHead, removeHead,
    stateForStage, detectMessage, play: playGesture, setSeed, randomGestureNow, walkingBand, stripGeometry, homeGeometry, homePath,
    SLEEP_AFTER_MS, regions: REGIONS, contours: CONTOURS, weights: WEIGHTS, clickWeights: CLICK_WEIGHTS, durations: DURATIONS,
    detectThanks, phrases: PHRASES, COMPLIMENT_GAP_MS, prankWeights: PRANK_WEIGHTS, PRANK_GAP_MS});
  refreshAttentionRects(); render(); environment();
})();
