"""The avatar lab must preserve Pedro's source art and stay entirely offline."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from tests.ui_parity import Document

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "docs/design/avatar_lab"
PNG_URL = "/src/ui/static/novo/avatar/duckcortex.png"
SHA256 = "a848bd285577cdb4eaf901aa15bc8798f4a76af6e46f3869d47a0ad5d650925e"
STATES = {
    "idle": "Em repouso", "thinking": "Pensando", "searching": "Buscando na web",
    "reading": "Lendo", "writing": "Escrevendo", "remembering": "Lembrando",
    "listening": "Ouvindo", "speaking": "Respondendo",
    "processing": "Processando",
}
REPLY_STATES = ("thinking", "searching", "reading", "writing", "remembering", "processing")


def test_source_png_is_untouched():
    assert hashlib.sha256((ROOT / PNG_URL.lstrip("/")).read_bytes()).hexdigest() == SHA256


def test_lab_assets_are_local_and_only_use_the_original_duck():
    html = (LAB / "index.html").read_text()
    assert 'href="/src/ui/static/novo/style.css"' in html
    assert 'href="/src/ui/static/novo/secundario.css"' in html
    assert PNG_URL in html
    doc = Document(html)
    for img in doc.select("img"):
        assert img.attrs["src"] == PNG_URL
    for path in LAB.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix in {".html", ".js", ".css", ".json", ".md"}, path
        source = path.read_text()
        assert not re.search(r"https?://|fetch\s*\(|XMLHttpRequest|WebSocket|EventSource|data:|@import|@font-face", source), path
        rasters = re.findall(r"[^\s\"'`()<>]+\.(?:png|webp|jpe?g|gif|avif)", source, re.I)
        assert all(url == PNG_URL or (path.name == "README.md" and
                    url == "/ui/static/novo/avatar/duckcortex.png") for url in rasters), (path, rasters)
    # Every declared asset resolves with the repository as the web root.
    for tag in [*doc.select("link"), *doc.select("script"), *doc.select("img")]:
        url = tag.attrs.get("src", tag.attrs.get("href"))
        assert (ROOT / url.lstrip("/") if url.startswith("/") else LAB / url).is_file(), url


def test_state_size_and_accessibility_controls():
    doc = Document((LAB / "index.html").read_text())
    options = [n for n in doc.select("#state-select")[0].descendants() if n.tag == "option"]
    assert [(n.attrs["value"], n.text()) for n in options] == list(STATES.items())
    sizes = [n.attrs["value"] for n in doc.select("#size-select")[0].descendants() if n.tag == "option"]
    assert sizes == ["96", "128", "144"]
    for id_ in ("reduce-motion", "light-theme", "compare-original"):
        assert doc.select("#" + id_)


def test_motion_has_both_reduced_modes_and_hidden_tab_pause():
    css = (LAB / "avatar.css").read_text()
    assert re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)", css)
    assert "body.reduced-motion" in css
    assert "animation: none" in css
    js = (LAB / "avatar.js").read_text()
    assert "visibilitychange" in js and "document.hidden" in js


def test_region_map_is_bounded_and_colours_have_sample_coordinates():
    js = (LAB / "avatar.js").read_text()
    match = re.search(r"const REGIONS = (\{.*?\});", js, re.S)
    assert match, "Keep the measured region map as a JSON-compatible constant"
    regions = json.loads(match[1])
    assert {"head", "wing", "leftEye", "rightEye", "antenna", "ear"} <= regions.keys()
    for name, rect in regions.items():
        assert len(rect) == 4, name
        x, y, width, height = rect
        assert all(isinstance(n, int) for n in rect)
        assert x >= 0 and y >= 0 and width > 0 and height > 0
        assert x + width <= 1254 and y + height <= 1254, name
    assert "sample" in js.lower() and "measured" in js.lower()


def test_reply_head_is_a_fixed_size_clip_of_the_original_url():
    doc = Document((LAB / "index.html").read_text())
    head = doc.select("#reply-head")
    assert head, "The status needs the original duck's clipped head"
    images = [node for node in head[0].descendants() if node.tag == "img"]
    assert len(images) == 1 and images[0].attrs["src"] == PNG_URL
    assert head[0].attrs["aria-hidden"] == "true"
    css = (LAB / "avatar.css").read_text()
    assert re.search(r"\.reply-head\s*\{[^}]*width:\s*32px;[^}]*height:\s*32px", css)
    assert "--reply-head-height: 28px" in css and "vertical-align: middle" in css
    assert re.search(r"\.reply-head img\s*\{[^}]*clip-path:\s*var\(--head-clip\)", css)
    js = (LAB / "avatar.js").read_text()
    assert "REGIONS.head" in js and "configureArt(replyHead)" in js
    # Existing asset test rejects every other raster URL and any derived file.


@pytest.mark.parametrize("state", REPLY_STATES)
def test_reply_state_has_animation_and_static_cue_in_both_reduced_modes(state):
    css = (LAB / "avatar.css").read_text()
    assert re.search(rf'\.reply-head\[data-state="{state}"\] \.reply-head-motion\s*'
                     rf'\{{[^}}]*animation:\s*reply-{state}\b', css)
    assert f"@keyframes reply-{state}" in css
    cue = re.search(rf'\.reply-head\[data-state="{state}"\]\s*\{{([^}}]+)', css)
    assert cue and "--reply-still-" in cue[1]
    # Both paths consume the same state-specific pose/light/eye cues, without JS.
    for selector, block in (("body.reduced-motion ", css.split("/* Reply head: lab reduced toggle */")[1].split("/* Reply head: OS reduced motion */")[0]),
                            ("", css.split("/* Reply head: OS reduced motion */")[1])):
        assert re.search(re.escape(selector) + r"\.reply-head \*\s*\{[^}]*animation: none !important", block)
        if not selector:
            assert "@media (prefers-reduced-motion: reduce)" in block
        for prop in ("pose", "antenna", "eyes", "ear"):
            assert f"var(--reply-still-{prop}" in block
    cues = [re.search(rf'\.reply-head\[data-state="{s}"\]\s*\{{([^}}]+)', css)[1]
            for s in REPLY_STATES]
    assert len(set(cues)) == 6


def test_reply_demo_labels_controls_and_task_131_markup():
    doc = Document((LAB / "index.html").read_text())
    assert doc.select("#reply-demo-mode") and doc.select("#reply-arrive")
    options = [node.attrs["value"] for node in doc.select("#reply-demo-mode")[0].descendants()
               if node.tag == "option"]
    assert options == ["auto", *REPLY_STATES, "off"]
    assert doc.select("#reply-progress")[0].text() == "pensando…"
    dots = doc.select(".reply-dots")
    assert len(dots) == 1 and dots[0].attrs["aria-hidden"] == "true"
    assert len([node for node in dots[0].descendants() if node.tag == "i"]) == 3


def test_reply_demo_cycles_in_sync_then_removes_icon_and_dots_for_stream(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, progress=document.querySelector('#reply-progress');
const duck=document.querySelector('#duck'), head=document.querySelector('#reply-head');
const replaceChildren=progress.replaceChildren.bind(progress);
let replacements=0;
progress.replaceChildren=(...nodes)=>{replacements++;replaceChildren(...nodes);};
const expected={thinking:'pensando…',searching:'buscando: clima em Londres…',
 reading:'lendo notas.pdf…',writing:'escrevendo resumo.md…',remembering:'lembrando…',processing:'processando…'};
assert.deepEqual(lab.replyLabels,expected);
function indicator(name) {return progress.children.find(node=>node.className===name);}
for(const [state,label] of Object.entries(expected)) {
  lab.setReplyDemo(state);
  assert.equal(duck.dataset.state,state); assert.equal(head.dataset.state,state);
  assert.equal(progress.children[0],head); assert.equal(progress.children[1].textContent,label);
  assert.equal(indicator('reply-dots').children.length,3);
  assert.equal(indicator('reply-dots').attrs['aria-hidden'],'true');
  advance(15000); assert.equal(head.dataset.state,state); // held, animation may continue
}
lab.setReplyDemo('auto');
let dots=indicator('reply-dots'), text=progress.children[1];
const statusReplacements=replacements;
for(const state of Object.keys(expected)) {
  assert.equal(replacements,statusReplacements); // no detach/reinsert restarting dots
  assert.equal(duck.dataset.state,state); assert.equal(head.dataset.state,state);
  assert.equal(indicator('reply-dots'),dots); assert.equal(progress.children[1],text);
  advance(lab.replyTimings.status);
}
assert.equal(progress.children.includes(head),false);
assert.equal(indicator('reply-dots'),undefined);
assert.ok(indicator('reply-cursor')); assert.ok(progress.children[0].textContent.length>0);
const streamReplacements=replacements, cursor=indicator('reply-cursor');
advance(lab.replyTimings.character);
assert.equal(replacements,streamReplacements); assert.equal(indicator('reply-cursor'),cursor);
assert.ok(progress.children[0].textContent.length>1);
advance(lab.replyTimings.character*lab.replyText.length);
assert.equal(progress.children[0].textContent,lab.replyText);
assert.equal(progress.children.length,1); // final removes cursor too
advance(lab.replyTimings.rest); assert.equal(head.dataset.state,'thinking');
lab.arriveReply(); assert.equal(indicator('reply-dots'),undefined);
assert.equal(progress.children.includes(head),false); assert.ok(indicator('reply-cursor'));
lab.setReplyDemo('reading'); // interrupts streaming; no late token or stale cursor
advance(lab.replyTimings.character*lab.replyText.length);
assert.equal(progress.children[1].textContent,expected.reading);
assert.equal(indicator('reply-cursor'),undefined);
lab.setState('writing'); assert.equal(head.dataset.state,'writing');
assert.equal(document.querySelector('#reply-demo-mode').value,'writing');
lab.setState('listening'); assert.equal(document.querySelector('#reply-demo').hidden,true);
""")


def test_reply_demo_pauses_for_reduction_visibility_and_original_comparison(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, head=document.querySelector('#reply-head');
lab.setReplyDemo('auto'); lab.setReduced(true);
assert.equal(head.dataset.state,'thinking'); assert.equal(pending.size,0);
advance(60000); assert.equal(head.dataset.state,'thinking');
lab.setReplyDemo('remembering'); assert.equal(head.dataset.state,'remembering');
assert.equal(pending.size,0);
lab.setReduced(false); media.matches=true; media.change(); assert.equal(pending.size,0);
lab.setReplyDemo('auto'); assert.equal(head.dataset.state,'thinking');
lab.arriveReply(); assert.equal(document.querySelector('#reply-progress').children.length,1);
assert.equal(pending.size,0); // reduced motion delivers the full text at once
media.matches=false; media.change(); lab.setReplyDemo('auto');
document.hidden=true; document.listeners.visibilitychange();
assert.equal(head.dataset.paused,'true'); assert.equal(pending.size,0);
document.hidden=false; document.listeners.visibilitychange();
assert.ok(pending.size>0); assert.equal(head.dataset.state,'thinking');
lab.setComparison(true); assert.equal(pending.size,0);
assert.equal(document.querySelector('#reply-demo').hidden,true);
lab.setComparison(false); assert.equal(head.dataset.state,document.querySelector('#duck').dataset.state);
assert.equal(document.querySelector('#reply-demo').hidden,false);
""")
    avatar_ui(r"""
assert.equal(document.querySelector('#duck').dataset.gesture,'wave');
advance(window.cortexAvatarLab.durations.wave);
assert.equal(document.querySelector('#reply-head').dataset.state,'thinking');
assert.equal(document.querySelector('#duck').dataset.state,'thinking');
advance(window.cortexAvatarLab.replyTimings.status);
assert.equal(document.querySelector('#reply-head').dataset.state,'searching');
""", setup="document.querySelector('#reply-demo-mode').value='auto';")


@pytest.fixture
def avatar_ui():
    source = (LAB / "avatar.js").read_text()
    harness = r"""
const assert = require('node:assert/strict');
const observers = [], createdCanvases = [];
class MutationObserver {
  constructor(callback) {this.callback=callback;}
  observe(target) {observers.push({target,callback:this.callback});}
}
class Element {
  constructor() {
    this.dataset = {}; this.style = {setProperty(k,v) {this[k]=v;}};
    this.attrs = {}; this.listeners = {}; this.checked = false; this.hidden = false;
    this.value = ''; this.textContent = ''; this.children = []; this.parts = new Map();
    const classes = new Set();
    const changeClass = (c,on) => {
      const before=classes.has(c);
      if(on) classes.add(c); else classes.delete(c);
      if(before!==classes.has(c)) for(const observer of observers)
        if(observer.target===this) observer.callback();
    };
    this.classList = {add: c => changeClass(c,true), remove: c => changeClass(c,false),
      contains: c => classes.has(c), toggle: changeClass};
  }
  setAttribute(k,v) {this.attrs[k] = v;}
  addEventListener(k,fn) {this.listeners[k] = fn;}
  querySelectorAll() {return [];}
  querySelector(s) {if(!this.parts.has(s)) this.parts.set(s,new Element()); return this.parts.get(s);}
  append(...nodes) {this.children.push(...nodes);}
  prepend(...nodes) {this.children.unshift(...nodes);}
  replaceChildren(...nodes) {this.children = [...nodes];}
  cloneNode() {const node=new Element(); node.dataset={...this.dataset}; return node;}
  removeAttribute(k) {delete this.attrs[k];}
  scrollIntoView() {}
  getAnimations() {return this.animations || [];}
  getContext() {
    if (!this.context) this.context = {
      marks: [], clears: 0, globalAlpha: 1, fillStyle: '',
      clearRect(...bounds) {this.marks = []; this.clears++; this.clearBounds=bounds;},
      fillRect(x,y,w,h) {this.marks.push([x,y,w,h,this.fillStyle,this.globalAlpha]);}
    };
    return this.context;
  }
}
const elements = new Map();
const document = {hidden: false, body: new Element(), listeners: {},
  querySelector(s) {if (!elements.has(s)) elements.set(s,new Element()); return elements.get(s);},
  createElement(tag) {const node=new Element(); if(tag==='canvas') createdCanvases.push(node); return node;},
  createElementNS() {return new Element();},
  createTextNode(text) {const node=new Element(); node.textContent=text; return node;},
  addEventListener(k,fn) {this.listeners[k]=fn;}};
const media = {matches: false, addEventListener(k,fn) {this.change=fn;}};
const matchMedia = () => media;
const window = {};
// Existing corner/gesture tests opt out of the independent automatic reply demo.
document.querySelector('#reply-demo-mode').value = 'off';
document.querySelector('#size-comparison').hidden = true;
let sequence = 0, now = 0;
const pending = new Map();
const setTimeout = (fn,delay) => {pending.set(++sequence,{fn,delay,at:now+delay}); return sequence;};
const clearTimeout = id => pending.delete(id);
function advance(ms) {
  const end=now+ms;
  for(let guard=0;guard<10000;guard++) {
    const entry=[...pending.entries()].sort((a,b)=>a[1].at-b[1].at)[0];
    if(!entry || entry[1].at>end) {now=end; return;}
    now=entry[1].at; pending.delete(entry[0]); entry[1].fn();
  }
  throw Error('runaway timers');
}
function tick() {if(pending.size) advance(Math.min(...[...pending.values()].map(job=>job.at))-now);}
"""

    def run(steps, setup=""):
        result = subprocess.run(["node", "-e", harness + setup + source + "\n" + steps],
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return result.stdout
    return run


def test_every_state_can_return_to_an_untouched_rest_frame(avatar_ui):
    avatar_ui("const states = " + json.dumps(list(STATES)) + r""";
const avatar = document.querySelector('#duck');
const lab = window.cortexAvatarLab;
for (const from of states) {
  for (const to of states) {
    lab.setState(from); tick(); lab.setState(to); tick(); lab.setState('idle');
    assert.equal(avatar.dataset.state, 'idle');
    assert.equal(avatar.dataset.eyes, 'open');
    assert.equal(avatar.dataset.part, 'none');
    lab.setComparison(true);
    assert.equal(avatar.dataset.frozen, 'true');
    assert.equal(pending.size, 0);
    assert.equal(avatar.dataset.eyes, 'open');
    lab.setComparison(false);
  }
}
lab.setState('constructor'); assert.equal(avatar.dataset.state, 'idle');
lab.setState('__proto__'); assert.equal(avatar.dataset.state, 'idle');
""")


def test_reduce_visibility_and_comparison_cancel_all_transient_patches(avatar_ui):
    avatar_ui(r"""
const lab = window.cortexAvatarLab, avatar = document.querySelector('#duck');
lab.setState('reading'); tick();
lab.setReduced(true);
assert.equal(avatar.dataset.eyes, 'open');
assert.equal(avatar.dataset.part, 'none');
assert.equal(avatar.dataset.reduced, 'true');
assert.equal(pending.size, 0);
lab.setReduced(false);
media.matches=true; media.change();
assert.equal(avatar.dataset.reduced, 'true'); assert.equal(pending.size, 0);
media.matches=false; media.change();
assert.equal(avatar.dataset.reduced, 'false');
assert.ok(pending.size > 0);
document.hidden=true; document.listeners.visibilitychange();
assert.equal(pending.size, 0); assert.equal(avatar.dataset.eyes, 'open');
assert.equal(avatar.dataset.paused, 'true');
lab.setState('remembering'); assert.equal(pending.size, 0);
document.hidden=false; document.listeners.visibilitychange();
assert.ok(pending.size > 0);
lab.setComparison(true);
assert.equal(avatar.dataset.state, 'idle'); assert.equal(pending.size, 0);
assert.equal(document.querySelector('#comparison').hidden, false);
lab.setComparison(false);
assert.equal(avatar.dataset.state, 'remembering');
assert.ok(pending.size > 0);
""")


def test_left_dock_centered_sphere_and_size_comparison():
    css = (LAB / "avatar.css").read_text()
    dock = re.search(r"\.duck-dock\s*\{([^}]+)", css)[1]
    assert "left:" in dock and "right:" not in dock
    assert "align-items: center" in css
    assert "overflow-x: auto" in css  # phone comparison retains true sizes
    assert "height: auto; min-height: 100dvh" in css  # controls never squeeze the chat on phones
    assert "overflow: visible" in css
    doc = Document((LAB / "index.html").read_text())
    assert doc.select("#compare-sizes")[0].text() == "Comparar tamanhos"
    assert doc.select("#size-comparison")
    for id_ in ("wave-hello", "happy-pedro", "random-gesture"):
        assert doc.select("#" + id_)
    assert "readonly" not in doc.select("#mock-message")[0].attrs
    assert "disabled" not in doc.select(".send")[0].attrs


def test_message_detection_is_deterministic_with_word_boundaries_and_negation(avatar_ui):
    avatar_ui(r"""
const detect=window.cortexAvatarLab.detectMessage;
for(const text of ['oi','OI, Pedro','Olá!','ola…','Bom dia.','boa tarde','BOA NOITE!',
  'e aí?','eai','Hey!','hi there','hello, perfeito!','... Oi!']) assert.equal(detect(text).greeting,true,text);
for(const text of ['foi ótimo','oito','hiato','heyday','helloWorld','diga oi','boa!','bom trabalho',''])
  assert.equal(detect(text).greeting,false,text);
for(const text of ['haha','hahaha','kkkk','rs','ótimo','otimo','perfeito','show','massa','top',
 'adorei','amei','incrível','demais','uhul','valeu','boa!','great','awesome','love it','yay',
 '😄','😁','😂','🥳','🎉','👏','🎾','❤️','Oi, adorei!', 'Não gostei. Agora está ótimo!', 'de novo, boa!'])
  assert.equal(detect(text).happy,true,text);
for(const text of ['boa noite','boa tarde','boa','não ficou ótimo','não está ótimo','nem perfeito',
 'nada de perfeito','not great','not really awesome','não amei','não 🎉','top 10 piores',
 'showcase','topologia','não ficou ótimo, nem perfeito','não boa!','não está boa!','']) assert.equal(detect(text).happy,false,text);
assert.deepEqual(detect('Oi, perfeito!'),{greeting:true,happy:true});
""")


def test_gesture_queue_and_all_gestures_return_to_original(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, duck=document.querySelector('#duck');
assert.equal(duck.dataset.gesture,'wave'); // one automatic welcome
advance(lab.durations.wave);
function rest() {
  assert.equal(duck.dataset.gesture,'none'); assert.equal(duck.dataset.part,'none');
  assert.equal(duck.dataset.eyes,'open'); assert.equal(duck.dataset.state,'idle');
  assert.equal(duck.style.transform,undefined); assert.equal(duck.style.clipPath,undefined);
}
rest();
for(const name of Object.keys(lab.durations)) {
  assert.equal(lab.playGesture(name),true,name);
  assert.equal(duck.dataset.gesture,name);
  advance(lab.durations[name]); rest();
}
lab.submitMessage('Olá, amei!');
assert.equal(duck.dataset.gesture,'wave');
advance(lab.durations.wave); assert.equal(duck.dataset.gesture,'racket');
advance(lab.durations.racket); rest();
lab.playGesture('racket'); document.hidden=true; document.listeners.visibilitychange();
rest(); assert.equal(pending.size,0);
document.hidden=false; document.listeners.visibilitychange();
lab.playGesture('wave'); lab.setReduced(true); rest(); assert.equal(pending.size,0);
lab.setReduced(false); lab.playGesture('racket'); lab.setComparison(true); rest();
assert.equal(pending.size,0);
""")


def test_seeded_random_intervals_exclusion_and_no_adjacent_repeats(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, duck=document.querySelector('#duck');
lab.setState('idle'); lab.setSeed(133);
function sequenceOfGestures() {
  const result=[];
  for(let i=0;i<100;i++) {
    const wait=[...pending.values()].find(job=>job.delay>=25000);
    assert.ok(wait && wait.delay<=60000);
    assert.equal(lab.randomGestureNow(),true);
    const name=duck.dataset.gesture;
    assert.ok(['look','hop','wave','stretch'].includes(name));
    assert.notEqual(name,result.at(-1)); result.push(name);
    assert.equal(lab.randomGestureNow(),false); // no overlap
    advance(lab.durations[name]);
    assert.equal(duck.dataset.gesture,'none');
  }
  return result;
}
const first=sequenceOfGestures(); lab.setSeed(133);
assert.deepEqual(sequenceOfGestures(),first);
lab.setSeed(42);
const wait=[...pending.values()].find(job=>job.delay>=25000);
advance(wait.at-now); assert.notEqual(duck.dataset.gesture,'none'); // actual timer fires
for(const state of ['thinking','searching','reading','writing','remembering','listening','speaking']) {
  lab.setState(state); assert.equal(lab.randomGestureNow(),false);
  advance(120000); assert.equal(duck.dataset.gesture,'none');
}
lab.setState('idle');
for(const mode of ['checkbox','os','hidden','comparison']) {
  if(mode==='checkbox') lab.setReduced(true);
  if(mode==='os') {media.matches=true;media.change();}
  if(mode==='hidden') {document.hidden=true;document.listeners.visibilitychange();}
  if(mode==='comparison') lab.setComparison(true);
  assert.equal(lab.randomGestureNow(),false); assert.equal(lab.playGesture('racket'),false);
  lab.submitMessage('oi, ótimo!'); advance(120000);
  assert.equal(duck.dataset.gesture,'none'); assert.equal(pending.size,0);
  lab.setReduced(false); media.matches=false; media.change();
  document.hidden=false; document.listeners.visibilitychange(); lab.setComparison(false);
}
""")


def test_size_copies_share_motion_and_composer_is_local_text(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab;
lab.setState('idle');lab.setSizeComparison(true);
const figures=document.querySelector('#size-previews').children;
assert.equal(figures.length,3);
const copies=figures.map(figure=>figure.children[0]);
assert.deepEqual(copies.map(copy=>copy.style['--duck-size']),['96px','128px','144px']);
assert.deepEqual(figures.map(figure=>figure.children[1].textContent),['96 px','128 px','144 px']);
document.querySelector('#duck').animations=[{animationName:'duck-float',currentTime:1234}];
for(const copy of copies) copy.animations=[{animationName:'duck-float',currentTime:0}];
lab.setSizeComparison(false); lab.setSizeComparison(true);
for(const copy of copies) assert.equal(copy.animations[0].currentTime,1234);
for(const name of Object.keys(lab.durations)) {
  lab.playGesture(name);
  for(const copy of copies) assert.equal(copy.dataset.gesture,name);
  advance(lab.durations[name]);
  for(const copy of copies) {assert.equal(copy.dataset.gesture,'none');assert.equal(copy.dataset.part,'none');}
}
lab.setState('reading'); tick();
for(const copy of copies) assert.equal(copy.dataset.eyes,document.querySelector('#duck').dataset.eyes);
lab.setState('idle');
const input=document.querySelector('#mock-message'), messages=document.querySelector('.messages');
input.value='<img onerror=alert(1)> oi';
input.listeners.keydown({key:'Enter',shiftKey:false,isComposing:false,preventDefault(){}});
assert.equal(messages.children.at(-1).children[0].children[0].textContent,'<img onerror=alert(1)> oi');
assert.equal(input.value,'');
const count=messages.children.length;
input.value='hello'; input.listeners.keydown({key:'Enter',shiftKey:true});
input.listeners.keydown({key:'Enter',shiftKey:false,isComposing:true});
assert.equal(messages.children.length,count);
document.querySelector('.send').listeners.click();
assert.equal(messages.children.length,count+1);
assert.equal(document.querySelector('#duck').dataset.gesture,'wave');
lab.setState('thinking'); lab.submitMessage('Hello, awesome!');
assert.equal(document.querySelector('#duck').dataset.state,'idle');
assert.equal(document.querySelector('#duck').dataset.gesture,'wave');
advance(lab.durations.wave); assert.equal(document.querySelector('#duck').dataset.gesture,'racket');
""")


def test_startup_respects_os_motion_and_hidden_tab(avatar_ui):
    for setup in ('media.matches=true;', 'document.hidden=true;'):
        avatar_ui(r"""
const lab=window.cortexAvatarLab, duck=document.querySelector('#duck');
assert.equal(duck.dataset.gesture,'none'); assert.equal(pending.size,0);
lab.submitMessage('oi, ótimo!'); advance(120000);
assert.equal(duck.dataset.gesture,'none'); assert.equal(pending.size,0);
""", setup=setup)


def test_racket_is_sampled_grid_art_and_only_visible_during_its_gesture(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, art=lab.racket;
assert.equal(art.cell,1254/96);
assert.ok(art.rows.length>=20 && art.rows.length<=36);
assert.ok(art.rows.every(row=>row.length===art.rows[0].length));
assert.ok(Object.keys(art.palette).length>=5);
for(const sample of Object.values(art.palette)) assert.ok(Object.hasOwn(lab.samples,sample));
const svg=document.querySelector('#duck').querySelector('.duck-racket');
assert.ok(svg.children.length>100);
for(const rect of svg.children) {
  for(const attr of ['x','y','width','height']) {
    const cells=Number(rect.attrs[attr])/art.cell;
    assert.ok(Math.abs(cells-Math.round(cells))<1e-9,attr);
  }
  assert.ok(Object.values(lab.samples).some(sample=>
    rect.attrs.fill===`rgb(${sample.rgba.slice(0,3).join(' ')})`));
}
""")
    html = (LAB / "index.html").read_text()
    assert 'viewBox="0 0 1254 1254"' in html and 'shape-rendering="crispEdges"' in html
    css = (LAB / "avatar.css").read_text()
    assert re.search(r'\.duck-racket\s*\{[^}]*display:\s*none', css)
    assert '[data-gesture="racket"] .duck-racket' in css
    assert 'z-index: 0' in css and 'z-index: 2' in css  # behind body, then under wing
    assert '[data-gesture="wave"] .duck-wing' in css
    assert 'head-nod' in css


def test_racket_timeline_never_overlaps_the_head_bounds(avatar_ui):
    """Use rendered cells and CSS interpolation, even behind the source or faded out."""
    css = (LAB / "avatar.css").read_text()
    source = (LAB / "avatar.js").read_text()
    head = json.loads(re.search(r"const HEAD = (\[.*?\]);", source, re.S)[1])
    avatar_ui("const css = " + json.dumps(css) + ";\nconst head = " + json.dumps(head) + r""";
const svg=document.querySelector('#duck').querySelector('.duck-racket');
const corners=svg.children.flatMap(rect=> {
  const [x,y,w,h]=['x','y','width','height'].map(key=>Number(rect.attrs[key]));
  return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
});
const headBounds=[Math.min(...head.map(p=>p[0])),Math.min(...head.map(p=>p[1])),
  Math.max(...head.map(p=>p[0])),Math.max(...head.map(p=>p[1]))];
function origin(selector) {
  const rule=css.match(new RegExp('\\.'+selector+'\\s*\\{([^}]+)'))[1];
  const match=rule.match(/transform-origin:\s*([\d.]+)%\s+([\d.]+)%/);
  assert.ok(match, 'model the actual percentage transform origin');
  return match.slice(1).map(value=>Number(value)*1254/100);
}
function track(name) {
  const block=css.match(new RegExp('@keyframes '+name+'\\s*\\{([\\s\\S]*?)\\n\\}'))[1];
  return [...block.matchAll(/([\d.%,\s]+)\{([^}]+)\}/g)].flatMap(([,stops,body])=> {
    // CSS interpolates each property independently: skip z-index-only frames.
    const match=body.match(/transform:\s*([^;]+);/);
    if(!match) return [];
    const transform=match[1].match(/^(?:translate\((-?[\d.]+%?),\s*(-?[\d.]+%?)\)\s+)?rotate\((-?[\d.]+)(?:deg)?\)$/);
    assert.ok(transform, 'extend the collision model for new transform functions');
    const translation=value=> {
      if(value===undefined || value==='0') return 0;
      assert.ok(value.endsWith('%'), 'translations must scale with the duck');
      return parseFloat(value)*1254/100;
    };
    const values=[translation(transform[1]),translation(transform[2]),Number(transform[3])];
    return stops.split(',').map(stop=>({at:parseFloat(stop)/100,values}));
  }).sort((a,b)=>a.at-b.at);
}
// CSS ease-in-out = cubic-bezier(.42,0,.58,1), applied between property stops.
function ease(progress) {
  const bezier=(t,a,b)=>3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3;
  let low=0,high=1;
  for(let i=0;i<40;i++) {
    const t=(low+high)/2;
    if(bezier(t,.42,.58)<progress) low=t; else high=t;
  }
  return bezier((low+high)/2,0,1);
}
function pose(frames,time) {
  const right=frames.findIndex(frame=>frame.at>=time);
  if(right===0) return frames[0].values;
  const a=frames[right-1],b=frames[right],amount=ease((time-a.at)/(b.at-a.at));
  return a.values.map((value,i)=>value+(b.values[i]-value)*amount);
}
function transform([x,y],[tx,ty,degrees],[ox,oy]) {
  const angle=degrees*Math.PI/180;
  // translate() rotate(): rotation about the origin, then parent-axis translation.
  return [ox+tx+(x-ox)*Math.cos(angle)-(y-oy)*Math.sin(angle),
    oy+ty+(x-ox)*Math.sin(angle)+(y-oy)*Math.cos(angle)];
}
assert.match(css,/racket-celebrate var\(--duration-racket\) ease-in-out both/);
assert.match(css,/wing-forehand var\(--duration-racket\) ease-in-out both/);
const frames=track('racket-celebrate'),wingFrames=track('wing-forehand');
const pivot=origin('duck-racket'),wingPivot=origin('duck-wing');
// Every 1% of every transform segment, plus exact reviewer frames and endpoints.
const samples=new Set([.45,.20,.65,0,1]);
for(let i=1;i<frames.length;i++) for(let step=0;step<=100;step++)
  samples.add(frames[i-1].at+(frames[i].at-frames[i-1].at)*step/100);
for(const time of samples) {
  const matrix=pose(frames,time),points=corners.map(point=>transform(point,matrix,pivot));
  const bounds=[Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),
    Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))];
  assert.ok(bounds[2]<=headBounds[0] || bounds[0]>=headBounds[2] ||
    bounds[3]<=headBounds[1] || bounds[1]>=headBounds[3],
    `racket overlaps HEAD bounds at ${(time*100).toFixed(2)}%: ${bounds}`);
  // Once the grip is in front of the body, the wing and racket share a matrix.
  if(time>=.31 && time<=.91) {
    assert.deepEqual(pivot,wingPivot);
    pose(wingFrames,time).forEach((value,i)=>assert.ok(Math.abs(value-matrix[i])<1e-8,
      `wing loses the racket grip at ${time*100}%`));
  }
}
""")


def test_every_colour_sample_matches_original_rgba():
    """Decode PNG scanlines with stdlib: no re-encoding or raster copies."""
    import struct
    import zlib
    png = (ROOT / PNG_URL.lstrip('/')).read_bytes()
    offset, compressed = 8, bytearray()
    while offset < len(png):
        size = struct.unpack('>I', png[offset:offset + 4])[0]
        kind, chunk = png[offset + 4:offset + 8], png[offset + 8:offset + 8 + size]
        if kind == b'IHDR':
            width, height, depth, colour, *_ = struct.unpack('>IIBBBBB', chunk)
            assert depth == 8 and colour == 6
        if kind == b'IDAT':
            compressed.extend(chunk)
        offset += size + 12
    decoded = zlib.decompress(compressed)
    previous, rows, offset = bytearray(width * 4), [], 0
    for _ in range(height):
        filter_ = decoded[offset]
        row = bytearray(decoded[offset + 1:offset + 1 + width * 4])
        offset += 1 + width * 4
        for i in range(len(row)):
            a, b, c = row[i - 4] if i >= 4 else 0, previous[i], previous[i - 4] if i >= 4 else 0
            p = a + b - c
            distances = [abs(p - a), abs(p - b), abs(p - c)]
            predictor = [0, a, b, (a + b) // 2, [a, b, c][distances.index(min(distances))]][filter_]
            row[i] = (row[i] + predictor) % 256
        rows.append(row)
        previous = row
    source = (LAB / 'avatar.js').read_text()
    samples = re.findall(r'at:\s*\[(\d+),\s*(\d+)\], rgba:\s*\[([\d, ]+)\]', source)
    assert len(samples) >= 7
    for x, y, rgba in samples:
        assert list(rows[int(y)][int(x) * 4:int(x) * 4 + 4]) == [int(n) for n in rgba.split(',')]


def test_search_rain_control_layers_and_pixel_only_drawing():
    doc = Document((LAB / 'index.html').read_text())
    controls = doc.select('#rain-colour')
    assert controls, 'Add the Cor da chuva control'
    assert 'Cor da chuva' in (LAB / 'index.html').read_text()
    assert [(n.attrs['value'], n.text()) for n in controls[0].descendants() if n.tag == 'option'] == [
        ('green', 'Verde Matrix'), ('cyan', 'Ciano do visor')]
    css = (LAB / 'avatar.css').read_text()
    rain = re.search(r'\.code-rain\s*\{([^}]+)', css)
    assert rain and 'z-index: 0' in rain[1] and 'pointer-events: none' in rain[1]
    assert 'radial-gradient' in rain[1] and 'mask-image:' in rain[1]
    assert 'image-rendering: pixelated' in rain[1]
    for selector in ('.duck-body', '.reply-head-motion'):
        assert re.search(re.escape(selector) + r'\s*\{[^}]*z-index: 1', css)
    assert re.search(r'\.duck, \.reply-head\s*\{[^}]*isolation: isolate', css)
    js = (LAB / 'avatar.js').read_text()
    assert not re.search(r'fillText|strokeText|drawImage|toDataURL|new Image\b|FontFace', js)
    # Existing asset and SHA tests also reject added images, fonts and embedded URLs.


@pytest.mark.parametrize('kind', ['duck', 'head'])
def test_search_rain_reaches_visible_outer_ring(avatar_ui, kind):
    """Record real fillRect calls for 2s; discard the opaque box and faint mask cells.

    At least 25% of the intended outer ring must light up over 40 capped frames,
    with at least 5% present in every animated frame, including first entry.
    The entire duck/head rectangle is treated as opaque: stricter than its PNG.
    CSS placement and mask stops are read from the shipped stylesheet.
    """
    css = (LAB / 'avatar.css').read_text()
    avatar_ui('const css=' + json.dumps(css) + '; const kind=' + json.dumps(kind) + r""";
function rule(selector) {
  const escaped=selector.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  return Object.fromEntries((css.match(new RegExp(escaped+'\\s*\\{([^}]+)'))?.[1] || '')
    .split(';').filter(s=>s.includes(':')).map(s=>s.split(':').map(v=>v.trim())));
}
const common=rule('.code-rain'), own=rule(kind==='duck'?'.duck .code-rain':'.reply-head .code-rain');
const style={...common,...own}, lab=window.cortexAvatarLab;
const node=document.querySelector(kind==='duck'?'#duck':'#reply-head');
const px=(value,base)=>value?.endsWith('%')?parseFloat(value)*base/100:parseFloat(value || '0');
for(const size of (kind==='duck'?[96,128,144]:[32])) {
  const scale=kind==='duck'?size/96:1;
  const target=kind==='duck'?[-24*scale,-24*scale,144*scale,120*scale]:[-4,-4,40,40];
  const opaque=kind==='duck'?[0,0,size,size]:[2,2,28,28];
  const [left,top,width,height]=[px(style.left,size),px(style.top,size),
    px(style.width,size),px(style.height,size)];
  const outside=(x,y)=>x<opaque[0] || x>=opaque[0]+opaque[2] || y<opaque[1] || y>=opaque[1]+opaque[3];
  const ring=new Set();
  // Unit logical cells, sampling pixel centres after scaling into CSS geometry.
  for(let y=0;y<target[3]/scale;y++) for(let x=0;x<target[2]/scale;x++) {
    const cx=target[0]+(x+.5)*scale, cy=target[1]+(y+.5)*scale;
    const r=Math.hypot((cx-target[0]-target[2]/2)/(target[2]/2),
      (cy-target[1]-target[3]/2)/(target[3]/2));
    if(r<=.9 && outside(cx,cy)) ring.add(`${x},${y}`);
  }
  const stops=style['mask-image'].match(/#000\s+([\d.]+)%,\s*transparent\s+([\d.]+)%/);
  assert.ok(stops,'model the actual mask stops');
  const [solid,end]=stops.slice(1).map(n=>Number(n)/100);
  const radius=style['mask-image'].includes('closest-side')?1:Math.SQRT2;
  for(const light of [false,true]) for(const colour of ['green','cyan']) {
    lab.setState('idle');
    document.querySelector('#light-theme').listeners.change({target:{checked:light}});
    document.querySelector('#rain-colour').listeners.change({target:{value:colour}});
    lab.setReplyDemo('searching');
    const canvas=node.children.find(child=>child.className==='code-rain');
    const union=new Set(); let minimum=1;
    for(let frame=0;frame<=40;frame++) {
      const lit=new Set();
      for(const [x,y,w,h,rgb,alpha] of canvas.context.marks) {
        const cx=left+(x+.5)*width/canvas.width, cy=top+(y+.5)*height/canvas.height;
        const r=Math.hypot((cx-left-width/2)/(width/2),(cy-top-height/2)/(height/2))/radius;
        const mask=Math.max(0,Math.min(1,(end-r)/(end-solid)));
        if(alpha*mask<.15 || !outside(cx,cy)) continue;
        const key=`${Math.floor((cx-target[0])/scale)},${Math.floor((cy-target[1])/scale)}`;
        if(ring.has(key)) {lit.add(key);union.add(key);}
      }
      minimum=Math.min(minimum,lit.size/ring.size);
      if(frame<40) advance(50);
    }
    const description=`${kind} ${size}px ${colour} light=${light}`;
    console.log(`${description}: coverage=${(union.size/ring.size*100).toFixed(1)}%, minimum=${(minimum*100).toFixed(1)}%`);
    assert.ok(union.size/ring.size>=.25,`${description}: outer-ring coverage ${(union.size/ring.size*100).toFixed(1)}% < 25%`);
    assert.ok(minimum>=.05,`${description}: sparse frame ${(minimum*100).toFixed(1)}% < 5%`);
  }
}
""")


def test_search_rain_bounds_clear_labels_and_text():
    """Conservative rectangular envelopes, including fully transparent mask edges.

    Read shipped CSS instead of inventing a separate layout configuration. The
    dock's stage and preview gutters contain the halo. The inline badge may use
    the paragraph's top margin and line leading, but never its horizontal text
    area, author row, or the start of a wrapped line.
    """
    css = (LAB / 'avatar.css').read_text()

    def rules(selector):
        return [dict(re.findall(r'([\w-]+):\s*([^;]+)', block)) for block in
                re.findall(re.escape(selector) + r'\s*\{([^}]+)', css)]

    def rule(selector):
        result = {}
        for declarations in rules(selector):
            result.update(declarations)
        return result

    def px(value, base=0):
        return float(value[:-1]) * base / 100 if value.endswith('%') else float(value.removesuffix('px'))

    halo, dock = rule('.duck .code-rain'), rule('.duck-dock')
    gutters = rule('.size-previews figure')['padding'].split()
    assert len(gutters) == 3  # top, both sides, bottom
    stage_height = min(px(r['min-height']) for r in rules('.chat-view .stage'))
    for size in (96, 128, 144):
        left, top = px(halo['left'], size), px(halo['top'], size)
        right = left + px(halo['width'], size)
        bottom = top + px(halo['height'], size)
        # Every responsive dock declaration must retain the reserved gutter.
        for declaration in rules('.duck-dock'):
            placement = {**dock, **declaration}
            assert px(placement['left']) + left >= 0
            assert px(placement['top']) + top >= 0
            assert px(placement['top']) + bottom <= stage_height
            assert bottom < size + px(placement['gap'])  # status-label top
        assert px(gutters[0]) + top >= 0  # no rain above preview's scrollport
        assert px(gutters[1]) + left >= 0
        assert right <= size + px(gutters[1])  # no spill into neighbouring copy
        caption_top = size + px(rule('.comparison figcaption')['margin-top'])
        assert bottom < caption_top
    badge, slot = rule('.reply-head .code-rain'), rule('.reply-head')
    left, top = px(badge['left']), px(badge['top'])
    right, bottom = left + px(badge['width']), top + px(badge['height'])
    text_start = px(slot['width']) + px(slot['margin-right'])
    assert right < text_start  # the entire badge clears status label by 4px
    progress = rule('#reply-progress')
    assert px(slot['height']) == 32
    assert px(progress['line-height']) == px(progress['min-height']) == 36
    assert bottom <= px(progress['line-height'])  # following wrapped text row
    base_css = (ROOT / 'src/ui/static/novo/style.css').read_text()
    paragraph_margin = re.search(r'\.message p\{[^}]*margin:([\d.]+)px', base_css)
    assert paragraph_margin
    assert top > -float(paragraph_margin[1])  # author above the paragraph
    message_gap = re.search(r'\.message\{[^}]*gap:([\d.]+)px', base_css)
    assert message_gap
    assert left > -float(message_gap[1])  # separate avatar column
    assert 'background' in badge and badge['border-radius'] == '50%'


def test_search_rain_lifecycle_and_shared_size_clock(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab, duck=document.querySelector('#duck'), head=document.querySelector('#reply-head');
const canvas=node=>node.children.find(child=>child.className==='code-rain');
assert.equal(canvas(duck),undefined); assert.equal(canvas(head),undefined); // lazy
assert.equal(createdCanvases.length,0);
lab.setReplyDemo('searching');
const corner=canvas(duck), icon=canvas(head);
assert.ok(corner && icon); assert.equal(duck.children[0],corner); assert.equal(head.children[0],icon);
assert.equal(corner.attrs['aria-hidden'],'true');
assert.deepEqual([corner.width,corner.height,icon.width,icon.height],[144,120,40,40]);
assert.equal(corner.hidden,false); assert.equal(icon.hidden,false);
assert.ok(corner.context.marks.length>0 && icon.context.marks.length>0);
for(const c of [corner,icon]) assert.deepEqual(c.context.clearBounds,[0,0,c.width,c.height]);
assert.equal(createdCanvases.length,2); // hidden previews are still lazy
for(const c of [corner,icon]) for(const [x,y,w,h] of c.context.marks) {
  assert.ok(Number.isInteger(x) && Number.isInteger(y));
  assert.ok(x>=0 && x<c.width && y>=0 && y<c.height); assert.equal(w,1); assert.equal(h,1);
}
const firstColumn=corner.context.marks.filter(mark=>mark[0]>=1 && mark[0]<4 && mark[1]>=12);
assert.ok(firstColumn.length>0);
advance(250); // same glyph, three cells down at 12 cells/sec
for(const mark of firstColumn) assert.ok(corner.context.marks.some(next=>next[0]===mark[0] && next[1]===mark[1]+3 && next[4]===mark[4]));
const initial=JSON.stringify(corner.context.marks);
advance(1000); assert.notEqual(JSON.stringify(corner.context.marks),initial);
assert.ok(corner.context.marks.some(mark=>mark[4]==='rgb(57 255 112)'));
const headAlpha=corner.context.marks.find(mark=>mark[4]==='rgb(57 255 112)')[5];
const trailAlpha=corner.context.marks.find(mark=>mark[4]==='rgb(37 166 73)')[5];
assert.ok(trailAlpha<headAlpha);
const brightest=corner.context.marks.find(mark=>mark[5]===.95);
assert.equal(brightest[4],'rgb(186 255 205)');
assert.ok(brightest[5]>headAlpha);
lab.setSizeComparison(true);
const copies=document.querySelector('#size-previews').children.map(figure=>figure.children[0]);
for (const copy of copies) assert.deepEqual(canvas(copy).context.marks,corner.context.marks);
advance(500);
for (const copy of copies) assert.deepEqual(canvas(copy).context.marks,corner.context.marks);
// No more than 20 paint ticks per second, regardless of the number of figures.
const clears=corner.context.clears; advance(1000);
assert.ok(corner.context.clears-clears<=20);
const all=[corner,icon,...copies.map(canvas)];
assert.equal(createdCanvases.length,5);
lab.setSizeComparison(false);
for(const copy of copies) {assert.equal(canvas(copy).hidden,true); assert.deepEqual(canvas(copy).context.marks,[]);}
lab.setSizeComparison(true);
for(const copy of copies) assert.deepEqual(canvas(copy).context.marks,corner.context.marks);
document.hidden=true; document.listeners.visibilitychange();
const frozen=all.map(c=>JSON.stringify(c.context.marks));
assert.equal(pending.size,0); advance(60000);
assert.deepEqual(all.map(c=>JSON.stringify(c.context.marks)),frozen);
document.hidden=false; document.listeners.visibilitychange();
advance(500); assert.notEqual(JSON.stringify(corner.context.marks),frozen[0]);
lab.setComparison(true);
for(const c of all) {assert.equal(c.hidden,true); assert.deepEqual(c.context.marks,[]);}
assert.equal(pending.size,0); lab.setComparison(false);
assert.equal(canvas(duck),corner); assert.ok(corner.context.marks.length);
lab.arriveReply();
for(const c of all) {
  assert.equal(c.hidden,true); assert.deepEqual(c.context.marks,[]);
  assert.deepEqual(c.context.clearBounds,[0,0,c.width,c.height]);
}
lab.setReplyDemo('searching'); lab.setReplyDemo('off');
assert.equal(icon.hidden,true); assert.deepEqual(icon.context.marks,[]);
assert.equal(corner.hidden,false); // the independently selected corner is still searching
for(const state of ['thinking','reading','writing','remembering','processing','listening','speaking','idle']) {
  lab.setReplyDemo('searching'); lab.setState(state);
  for(const c of all) {assert.equal(c.hidden,true); assert.deepEqual(c.context.marks,[]);}
}
lab.setState('searching'); assert.equal(canvas(duck),corner); // reuse the single canvas
assert.equal(duck.children.filter(child=>child.className==='code-rain').length,1);
assert.equal(createdCanvases.length,5);
lab.setReduced(true); lab.setState('idle'); assert.equal(pending.size,0);
""")


@pytest.mark.parametrize('mode', ['toggle', 'media', 'body'])
def test_search_rain_is_static_in_each_reduced_mode(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
const lab=window.cortexAvatarLab;
lab.setReplyDemo('searching');
if(mode==='toggle') lab.setReduced(true);
if(mode==='media') {media.matches=true; media.change();}
if(mode==='body') document.body.classList.add('reduced-motion');
const nodes=[document.querySelector('#duck'),document.querySelector('#reply-head')];
const rain=nodes.map(node=>node.children.find(child=>child.className==='code-rain'));
for(const c of rain) {
  assert.ok(c && !c.hidden && c.context.marks.length>0);
  assert.ok(c.context.marks.every(mark=>mark[5]<=.26));
}
const still=rain.map(c=>JSON.stringify(c.context.marks));
advance(10000); assert.deepEqual(rain.map(c=>JSON.stringify(c.context.marks)),still);
if(mode==='toggle') lab.setReduced(false);
if(mode==='media') {media.matches=false; media.change();}
if(mode==='body') document.body.classList.remove('reduced-motion');
advance(1000); assert.notEqual(JSON.stringify(rain[0].context.marks),still[0]);
lab.setState('idle');
for(const c of rain) {assert.equal(c.hidden,true); assert.deepEqual(c.context.marks,[]);}
""")


def test_search_rain_palette_and_theme_changes_repaint_in_place(avatar_ui):
    avatar_ui(r"""
const lab=window.cortexAvatarLab;
lab.setReplyDemo('searching');
const nodes=[document.querySelector('#duck'),document.querySelector('#reply-head')];
const canvases=nodes.map(node=>node.children[0]);
const select=document.querySelector('#rain-colour');
assert.equal(select.value,'green');
select.listeners.change({target:{value:'cyan'}});
for(const c of canvases) assert.ok(c.context.marks.some(mark=>mark[4]===`rgb(${lab.samples.cyan.rgba.slice(0,3).join(' ')})`));
document.querySelector('#light-theme').listeners.change({target:{checked:true}});
assert.ok(canvases[0].context.marks.some(mark=>mark[4]==='rgb(24 116 124)'));
assert.ok(!canvases[0].context.marks.some(mark=>mark[4]==='rgb(129 249 251)'));
// The dark badge retains neon colours in the light theme for legible contrast.
assert.ok(canvases[1].context.marks.some(mark=>mark[4]==='rgb(129 249 251)'));
select.listeners.change({target:{value:'green'}});
assert.ok(canvases[0].context.marks.some(mark=>mark[4]==='rgb(22 119 55)'));
assert.ok(canvases[1].context.marks.some(mark=>mark[4]==='rgb(57 255 112)'));
document.querySelector('#light-theme').listeners.change({target:{checked:false}});
for(const c of canvases) assert.ok(c.context.marks.some(mark=>mark[4]==='rgb(57 255 112)'));
assert.deepEqual(nodes.map(node=>node.children[0]),canvases);
""")
