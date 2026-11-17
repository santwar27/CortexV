"use strict";
// Task 163 — the home stage is the duck's room (Pedro's choice B): a floor at
// its feet, a rug, a window, a racket, a lamp, a bed and one low bookcase whose
// books are the real projects. New art only (rect cell maps); the duck PNG is
// never read or changed. Project names reach the page as text, never as markup.
(() => {
const stage = document.querySelector('.stage'), dock = document.querySelector('#cortex-duck-dock');
const orb = document.querySelector('#orb');
if (!stage || !dock || !orb) return;
const COLORS = ['#d9415b', '#3b6fd9', '#d99a1e', '#2f9e77', '#7a5cd6'];
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};
// A small seeded generator, restarted on every draw: the same shelf each time.
let seed = 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

// ---- pixel art: fixed cell maps turned into one crisp SVG ----
function sprite(width, height, paint, unit, className) {
  const cells = [];
  paint((x, y, w, h, color) => cells.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${color}"/>`));
  const holder = el('span', 'q-sprite ' + className);
  holder.setAttribute('aria-hidden', 'true');
  holder.innerHTML = `<svg viewBox="0 0 ${width} ${height}" shape-rendering="crispEdges" ` +
    `style="stroke:none;width:${width * unit}px;height:${height * unit}px">${cells.join('')}</svg>`;
  return holder;
}
const plant = unit => sprite(9, 11, r => {
  r(4, 0, 1, 2, '#3fbf8a'); r(1, 1, 2, 2, '#2f9e77'); r(6, 1, 2, 2, '#2f9e77'); r(2, 3, 5, 2, '#3fbf8a');
  r(0, 3, 2, 1, '#2f9e77'); r(7, 3, 2, 1, '#2f9e77'); r(3, 5, 3, 1, '#2f9e77'); r(4, 6, 1, 1, '#256b55');
  r(1, 7, 7, 1, '#c96f3b'); r(2, 8, 5, 3, '#b25f31'); r(2, 8, 1, 3, '#d8844a');
}, unit, 'q-plant');
const trophy = unit => sprite(9, 10, r => {
  r(2, 0, 5, 1, '#ffd65c'); r(2, 1, 5, 3, '#f2b632'); r(0, 1, 2, 1, '#f2b632'); r(7, 1, 2, 1, '#f2b632');
  r(0, 2, 1, 2, '#f2b632'); r(8, 2, 1, 2, '#f2b632'); r(1, 3, 1, 1, '#f2b632'); r(7, 3, 1, 1, '#f2b632');
  r(3, 4, 3, 1, '#d99a1e'); r(4, 5, 1, 2, '#d99a1e'); r(3, 7, 3, 1, '#f2b632'); r(2, 8, 5, 2, '#5b4636'); r(3, 1, 1, 2, '#fff1b8');
}, unit, 'q-trophy');
const ball = unit => sprite(6, 6, r => {
  r(1, 0, 4, 6, '#cfe94a'); r(0, 1, 6, 4, '#cfe94a'); r(1, 1, 1, 1, '#f4ffc0'); r(3, 0, 1, 2, '#ffffff');
  r(4, 2, 1, 1, '#ffffff'); r(1, 4, 1, 1, '#ffffff'); r(2, 5, 1, 1, '#ffffff'); r(4, 4, 1, 1, '#a9c232');
}, unit, 'q-ball');
const windowArt = unit => sprite(18, 18, r => {
  r(0, 0, 18, 18, '#34465f'); r(1, 1, 16, 16, '#0b1730'); r(1, 1, 16, 5, '#0e1d3d');
  r(11, 3, 4, 4, '#f6ecc4'); r(10, 4, 1, 2, '#f6ecc4'); r(15, 4, 1, 2, '#f6ecc4'); r(13, 4, 2, 2, '#e3d6a2');
  r(3, 3, 1, 1, '#ffffff'); r(6, 7, 1, 1, '#9db4e8'); r(4, 11, 1, 1, '#ffffff'); r(13, 12, 1, 1, '#9db4e8'); r(7, 13, 1, 1, '#ffffff');
  r(8, 1, 2, 16, '#34465f'); r(1, 8, 16, 2, '#34465f'); r(0, 17, 18, 1, '#4a5f7d');
}, unit, 'q-window');
const lamp = unit => sprite(9, 24, r => {
  r(2, 0, 5, 1, '#ffe08a'); r(1, 1, 7, 3, '#f2b632'); r(0, 4, 9, 2, '#d99a1e'); r(2, 1, 1, 3, '#ffe9a8');
  r(4, 6, 1, 15, '#4a5f7d'); r(2, 21, 5, 1, '#4a5f7d'); r(1, 22, 7, 2, '#34465f');
}, unit, 'q-lamp');
const bed = unit => sprite(30, 13, r => {
  r(0, 0, 2, 13, '#5b4636'); r(28, 4, 2, 9, '#5b4636'); r(2, 7, 26, 4, '#6c5340'); r(2, 5, 26, 2, '#e8eef8');
  r(3, 3, 7, 3, '#ffffff'); r(3, 5, 7, 1, '#c9d4e6'); r(11, 4, 17, 4, '#3b6fd9'); r(11, 4, 17, 1, '#5d8bea');
  r(14, 6, 2, 1, '#cfe94a'); r(20, 5, 2, 1, '#cfe94a'); r(24, 7, 2, 1, '#cfe94a'); r(2, 11, 2, 2, '#4a392c'); r(26, 11, 2, 2, '#4a392c');
}, unit, 'q-bed');
const racket = unit => sprite(9, 20, r => {
  r(2, 0, 5, 1, '#d9415b'); r(1, 1, 1, 1, '#d9415b'); r(7, 1, 1, 1, '#d9415b'); r(0, 2, 1, 6, '#d9415b'); r(8, 2, 1, 6, '#d9415b');
  r(1, 8, 1, 1, '#d9415b'); r(7, 8, 1, 1, '#d9415b'); r(2, 9, 5, 1, '#d9415b'); r(3, 10, 3, 2, '#d9415b');
  for (let y = 2; y < 8; y += 2) r(1, y, 7, 1, '#9aaac0'); for (let x = 2; x < 8; x += 2) r(x, 1, 1, 8, '#9aaac0');
  r(4, 12, 1, 4, '#e8eef8'); r(3, 16, 3, 4, '#34465f'); r(3, 17, 3, 1, '#4a5f7d');
}, unit, 'q-racket');

// ---- what fits: pure arithmetic on the measured stage ----
// Wide screens: the floor passes the sphere, the furniture stands on both sides
// of it. Otherwise the duck has its own row and only the bookcase and the lamp
// stand beside it ("tight").
function plan(m) {
  const tight = m.floor <= m.orbTop - 8, start = m.duckLeft + m.duckSize + 20;
  const wall = tight ? m.width - start - 4 : m.width - m.orbRight - 52;
  const fit = bed => Math.max(0, tight ? Math.floor((wall - 28 - 14 + 1) / 23)
    : Math.floor((wall - 52 - (bed ? 106 : 0) - 24 + 2) / 28));
  // The bed gives its place to the books when the projects need the room.
  const bedFits = !tight && wall >= 350 && m.count <= fit(true);
  const capacity = fit(bedFits);
  const over = m.count > capacity;
  const books = over ? Math.max(0, capacity - 1) : m.count;
  const hidden = m.count - books;
  const used = books + (over && capacity ? 1 : 0);
  const fillers = m.count ? (tight ? 0 : Math.min(2, capacity - used)) : Math.min(6, capacity);
  return {tight, start, bed: bedFits, capacity, books, hidden: over && capacity ? hidden : 0, fillers};
}

// ---- the room ----
const room = el('div', 'q-room');
const floorLeft = el('b', 'q-floor'), floorRight = el('b', 'q-floor'), rug = el('b', 'q-rug');
const left = el('div', 'q-wall q-wall-left'), right = el('div', 'q-wall q-wall-right');
const bookcase = el('div', 'q-bookcase');
bookcase.setAttribute('role', 'group'); bookcase.setAttribute('aria-label', 'Estante do pato: seus projetos');
for (const node of [floorLeft, floorRight, rug, left]) node.setAttribute('aria-hidden', 'true');
room.append(floorLeft, floorRight, rug, left, right);
stage.prepend(room);
let projects = [], open = () => {}, more = () => {}, drawn = null;

function colorOf(id) {
  let sum = 0;
  for (const ch of String(id)) sum = (sum * 31 + ch.codePointAt(0)) >>> 0;
  return COLORS[sum % COLORS.length];
}
function projectBook(project, index) {
  const book = el('button', 'q-book q-project');
  book.type = 'button'; book.title = 'Abrir o projeto “' + project.name + '”';
  book.setAttribute('aria-label', book.title);
  book.style.setProperty('--c', colorOf(project.id));
  book.style.setProperty('--h', (90 + ((index * 37) % 5) * 2.5) + '%');
  book.append(el('span', '', project.name));
  book.addEventListener('click', () => open(project.id));
  return book;
}
function moreBook(count) {
  const book = el('button', 'q-book q-project q-more');
  book.type = 'button'; book.title = 'Ver todos os projetos (mais ' + count + ')';
  book.setAttribute('aria-label', book.title);
  book.append(el('span', '', '+' + count));
  book.addEventListener('click', () => more());
  return book;
}
function filler() {
  const book = el('i', 'q-book q-filler');
  book.setAttribute('aria-hidden', 'true');
  book.style.setProperty('--w', (8 + Math.floor(rnd() * 4) * 3) + 'px');
  book.style.setProperty('--h', (58 + Math.floor(rnd() * 8) * 5) + '%');
  book.dataset.tone = String(Math.floor(rnd() * 4));
  return book;
}
function furnish(p) {
  seed = 20261111;
  const unit = p.tight ? 2 : 4, shelf = el('div', 'q-row'), top = el('div', 'q-top');
  top.setAttribute('aria-hidden', 'true'); top.append(plant(3), trophy(3), ball(3));
  shelf.append(...projects.slice(0, p.books).map(projectBook));
  if (p.hidden) shelf.append(moreBook(p.hidden));
  for (let i = 0; i < p.fillers; i++) shelf.append(filler());
  shelf.append(el('b', 'q-plank'));
  bookcase.replaceChildren(top, shelf);
  left.replaceChildren(...(p.tight ? [] : [windowArt(4), racket(4)]));
  right.replaceChildren(...(p.capacity ? [bookcase] : []), lamp(unit), ...(p.bed ? [bed(3)] : []));
}
function draw() {
  const box = stage.getBoundingClientRect(), size = dock.offsetWidth, disc = orb.getBoundingClientRect();
  // Hidden views measure as zero: keep the last drawing until home is back.
  if (!box.width || !size || !disc.width) return;
  // The feet end at 98.7% of the duck's box (measured from the art). The rug
  // starts 3px above that line and the floor 2px below it, so the duck stands
  // on the rug instead of hanging over its lower edge.
  const floor = Math.round(dock.offsetTop + size * .987) + 2;
  const orbLeft = disc.left - box.left, orbRight = disc.right - box.left;
  const p = plan({width: box.width, orbRight, orbTop: disc.top - box.top, duckLeft: dock.offsetLeft,
    duckSize: size, floor, count: projects.length});
  const key = JSON.stringify([p, projects.map(project => [project.id, project.name])]);
  if (key !== drawn) { drawn = key; furnish(p); }
  room.classList.toggle('q-tight', p.tight);
  room.style.setProperty('--floor', floor + 'px');
  floorLeft.style.cssText = `left:0;width:${p.tight ? box.width : Math.max(0, orbLeft - 28)}px`;
  floorRight.style.cssText = p.tight ? 'display:none' : `left:${orbRight + 28}px;right:0`;
  rug.style.cssText = `left:${dock.offsetLeft - size * .12}px;width:${size * 1.24}px`;
  left.style.cssText = p.tight ? 'display:none' : `left:${p.start}px;width:${Math.max(0, orbLeft - 40 - p.start)}px`;
  right.style.cssText = p.tight ? `left:${p.start}px;right:4px` : `left:${orbRight + 44}px;right:8px`;
}
// The list comes from projetos.js (the same one the sidebar shows). Only the
// fields the shelf uses are kept; a failed or empty list leaves filler books.
function render(list, onOpen, onMore) {
  projects = (Array.isArray(list) ? list : []).filter(project => project && project.id != null)
    .map(project => ({id: project.id, name: String(project.name ?? '')}));
  if (typeof onOpen === 'function') open = onOpen;
  if (typeof onMore === 'function') more = onMore;
  draw();
}
addEventListener('resize', draw);
if (typeof ResizeObserver === 'function') {
  const watcher = new ResizeObserver(draw); watcher.observe(stage); watcher.observe(orb);
}
if (typeof MutationObserver === 'function')
  new MutationObserver(draw).observe(document.body, {attributes: true, attributeFilter: ['class']});
setTimeout(draw, 400);
draw();
window.cortexQuarto = {render, plan, draw};
})();
