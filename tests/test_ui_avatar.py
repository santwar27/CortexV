"""Production avatar: untouched art, real modules and SSE lifecycle under Node."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from tests.ui_parity import DUCK_MARKUP, Document, check

ROOT = Path(__file__).resolve().parents[1]
NOVO = ROOT / 'src/ui/static/novo'
PNG = NOVO / 'avatar/duckcortex.png'
HASH = 'a848bd285577cdb4eaf901aa15bc8798f4a76af6e46f3869d47a0ad5d650925e'


def test_unique_original_and_all_urls():
    assert PNG.is_file()
    assert hashlib.sha256(PNG.read_bytes()).hexdigest() == HASH
    matches = []
    for path in ROOT.rglob('*'):
        if '.git' in path.parts or not path.is_file():
            continue
        if path.name == 'duckcortex.png':
            matches.append(path)
        elif path.stat().st_size == PNG.stat().st_size:
            # Integrity scan only; never open the excluded review documents.
            assert not any(p in path.parts for p in ('review_codex', 'review_fable5'))
            if hashlib.sha256(path.read_bytes()).hexdigest() == HASH:
                matches.append(path)
    assert matches == [PNG]
    for name in ('index.html', 'README.md', 'avatar.js', 'avatar.css'):
        source = (ROOT / 'docs/design/avatar_lab' / name).read_text()
        assert '/docs/design/duckcortex.png' not in source
    lab = Document((ROOT / 'docs/design/avatar_lab/index.html').read_text())
    assert all(n.attrs['src'] == '/src/ui/static/novo/avatar/duckcortex.png' for n in lab.select('img'))


def test_markup_parity_is_exact_and_assets_load_before_core():
    html = (ROOT / 'src/ui/templates/novo.html').read_text()
    doc = Document(html)
    dock = doc.select('#cortex-duck-dock')
    assert len(dock) == 1
    assert dock[0].attrs['aria-hidden'] == 'true'
    assert dock[0] in doc.select('.stage')[0].children
    assert all(n.attrs['src'] == '/ui/static/novo/avatar/duckcortex.png' for n in dock[0].descendants() if n.tag == 'img')
    assert not [n for n in dock[0].descendants() if n.tag in ('button', 'input', 'select') or 'tabindex' in n.attrs]
    assert '/ui/static/novo/avatar.css' in html
    assert html.index('/ui/static/novo/avatar.js') < html.index('/ui/static/novo/core.js')
    assert not check(html)
    start = html.index('<div id="cortex-duck-dock"')
    # Change a nested art layer, the URL, or remove the entire addition.
    for changed in (html.replace(DUCK_MARKUP, ''),
                    html.replace('class="duck-source"', 'class="duck-source-broken"', 1),
                    html.replace('/ui/static/novo/avatar/duckcortex.png', '/wrong.png', 1),
                    html[:start] + html[start:].replace('id="cortex-duck-dock"', 'id="missing-duck"', 1)):
        assert check(changed), 'Parity must reject a missing or modified duck'


def test_production_css_size_motion_and_offline_scope():
    css = (NOVO / 'avatar.css').read_text()
    assert '--duck-size: 144px' in css
    assert re.search(r'@media\s*\(max-width:\s*800px\)\s*\{[^}]*--duck-size:\s*96px', css)
    assert 'pointer-events: none' in css
    assert 'body.reduced-motion' in css and '@media (prefers-reduced-motion: reduce)' in css
    for name in ('avatar.js', 'avatar.css'):
        text = (NOVO / name).read_text()
        assert not re.search(r'https?://|fetch\s*\(|XMLHttpRequest|WebSocket|EventSource|@import|@font-face|toDataURL|drawImage', text)
    assert 'rain-colour' not in (NOVO / 'avatar.js').read_text()


@pytest.fixture
def avatar_ui():
    # Reuse task 131's real tree/text/SSE fake, then extend it with canvas and clocks.
    import ast
    tree = ast.parse((ROOT / 'tests/test_ui_bolha_viva.py').read_text())
    fixture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'chat_ui')
    harness = next(ast.literal_eval(n.value) for n in fixture.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'harness' for t in n.targets))
    harness = harness.replace("if (url === '/ui/chat/stream')", "if (url === '/ui/conversations') return {ok:true,json:async()=>({id:'created'})};\n  if (url === '/ui/chat/stream')")
    extra = r"""
const observers=[];
const MutationObserver=class {constructor(callback){this.callback=callback;} observe(target){observers.push({target,callback:this.callback});}};
const resizers=[];
const ResizeObserver=class {constructor(callback){resizers.push(callback);} observe(){}};
const resize=()=>resizers.forEach(fn=>fn());
let now=0, sequence=0; const pending=new Map();
// Sleep and rest use independent timers; RAF is flushed explicitly per event batch.
const setTimeout=(fn,delay)=>{pending.set(++sequence,{fn,delay,at:now+delay});return sequence;};
const clearTimeout=id=>pending.delete(id);
Date.now=()=>now;
const frames=new Map();let frameId=0;
const requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId;};
const cancelAnimationFrame=id=>frames.delete(id), devicePixelRatio=1;
const flushFrames=()=>{const batch=[...frames.values()];frames.clear();batch.forEach(fn=>fn(now));};
function advance(ms){const end=now+ms;for(let guard=0;guard<10000;guard++){
 const entry=[...pending.entries()].sort((a,b)=>a[1].at-b[1].at)[0];
 if(!entry||entry[1].at>end){now=end;return;}now=entry[1].at;pending.delete(entry[0]);entry[1].fn();
}throw Error('runaway timers');}
const media={matches:false,addEventListener(k,fn){this.change=fn;}};
const matchMedia=()=>media;
Element.prototype.getAnimations=function(){return [];};
Element.prototype.addEventListener=function(name,fn){(this.eventListeners ||= {})[name] ||= [];this.eventListeners[name].push(fn);};
Element.prototype.dispatchEvent=function(event){for(const fn of this.eventListeners?.[event.type]||[])fn(event);};
const clickDuck=()=>{duck.dispatchEvent({type:'click',target:duck});page('click',{target:duck});};
Element.prototype.getContext=function(){return this.context ||= {marks:[],setTransform(){},clearRect(){this.marks=[];},fillRect(...rect){this.marks.push([...rect,this.fillStyle,this.globalAlpha]);}};};
Element.prototype.removeAttribute=function(name){delete this.attributes[name];};
const setAttribute=Element.prototype.setAttribute;
Element.prototype.setAttribute=function(name,value){setAttribute.call(this,name,value);if(name==='class')this.className=value;};
const originalCreate=document.createElement;
document.createElement=tag=>{const node=originalCreate(tag);node.style.setProperty=function(k,v){this[k]=v;};node.style.removeProperty=function(k){delete this[k];};return node;};
document.createElementNS=(ns,tag)=>document.createElement(tag);
const make=(tag,classes='')=>{const n=document.createElement(tag);n.className=classes;return n;};
document.body=make('body');document.hidden=false;
const originalToggle=document.body.classList.toggle;
document.body.classList.toggle=(name,on)=>{originalToggle(name,on);for(const o of observers)if(o.target===document.body)o.callback();};
document.body.classList.add=name=>document.body.classList.toggle(name,true);
document.body.classList.remove=name=>document.body.classList.toggle(name,false);
const listeners={};document.addEventListener=(name,fn)=>{(listeners[name] ||= []).push(fn);};
const page=(type,props={})=>{for(const fn of listeners[type]||[])fn({type,target:document.body,...props});};
const pointer=(x,y)=>{page('pointermove',{clientX:x,clientY:y,pointerType:'mouse'});flushFrames();};
const input=hosts['#message-input'];
const focusInput=()=>{document.activeElement=input;input.dispatchEvent({type:'focus'});};
const typeKey=()=>{page('keydown',{target:input,key:'a'});input.dispatchEvent({type:'input'});flushFrames();};
const blurInput=()=>{document.activeElement=null;input.dispatchEvent({type:'blur'});flushFrames();};
const visibility=hidden=>{document.hidden=hidden;for(const fn of listeners.visibilitychange||[])fn();};
const duck=make('div','duck'), dock=make('div','duck-dock'), racket=make('svg','duck-racket');
duck.append(racket);dock.append(duck);hosts['#cortex-duck']=duck;hosts['#cortex-duck-dock']=dock;
hosts['#orb']=make('canvas');hosts['#state-label']=make('span');hosts['#mic']=make('button');
const pose=make('div','duck-pose');duck.append(pose);pose.append(racket);
const source=make('img','duck-source');source.setAttribute('src','/ui/static/novo/avatar/duckcortex.png');pose.append(source);
const rect=(left,top,width,height)=>({left,top,width,height,right:left+width,bottom:top+height});
const geometry={stage:rect(0,64,1108,228),duck:rect(36,112,144,144),
 orb:rect(467,69,174,174),label:rect(467,239,174,20),messages:rect(119,320,870,450),
 composer:rect(0,780,1108,120),topbar:rect(0,0,1108,64)};
for(const [selector,key] of Object.entries({'.stage':'stage','#orb':'orb','.orb-status':'label','#messages':'messages','.composer-area':'composer','.topbar':'topbar'})) {
 hosts[selector] ||= make('div');hosts[selector].getBoundingClientRect=()=>geometry[key];
}
input.getBoundingClientRect=()=>geometry.composer;
window.innerWidth=1440;
const windowListeners={};window.addEventListener=(name,fn)=>{(windowListeners[name] ||= []).push(fn);};
Element.prototype.getBoundingClientRect=function(){return rect(0,0,0,0);};
dock.getBoundingClientRect=()=>rect(geometry.duck.left+Number(duck.dataset.x||0),geometry.duck.top+Number(duck.dataset.y||0),geometry.duck.width,geometry.duck.height);
duck.getBoundingClientRect=()=>{
 if(duck.parentNode===dock)return dock.getBoundingClientRect();
 const layer=duck.parentNode,s=duck.style;
 return rect(parseFloat(layer.style.left)+parseFloat(s.left),parseFloat(layer.style.top)+parseFloat(s.top),
  parseFloat(s['--duck-size']),parseFloat(s['--duck-size']));
};
const scrollEvent=()=>{for(const fn of windowListeners.scroll||[])fn();flushFrames();};
function measured(width=1440,scroll=0,left=null,right=null) {
 window.innerWidth=width;
 geometry.scroll=rect(252,64,width-252,721);
 const gutter=(width-252-870)/2;
 geometry.messages=rect(left??252+gutter,374-scroll,(right??width-gutter)-(left??252+gutter),900);
 geometry.stage=rect(292,64-scroll,width-332,228);
 geometry.duck=rect(328,width>=1658?112:112-scroll,144,144);
 geometry.orb=rect((width+252)/2-87,67-scroll,174,174);
 geometry.label=rect((width+252)/2-7,237-scroll,14,20);
 geometry.topbar=rect(252,0,width-252,64);geometry.composer=rect(376,799,width-500,63);
 hosts['#content-scroll'].getBoundingClientRect=()=>geometry.scroll;
 hosts['#content-scroll'].scrollTop=scroll;
}
const settledStrip=()=>{measured(1440,100);scrollEvent();advance(2000);};
function assertStrip(side) {
 const b=duck.getBoundingClientRect(),s=geometry.scroll,m=geometry.messages;
 assert.equal(duck.dataset.habitat,side);assert.notEqual(duck.parentNode,dock);
 assert.ok(b.top>=s.top+12 && b.bottom<=s.bottom-12,JSON.stringify(b));
 assert.ok(side==='left'?b.left>=s.left+12 && b.right<=m.left-12:
  b.left>=m.right+12 && b.right<=s.right-12,JSON.stringify(b));
 for(const value of [b.left,b.top,b.width])assert.equal(value,Math.round(value));
 for(const obstacle of [m,rect(m.left,m.top,32,32),rect(m.right-300,m.top+100,300,60),geometry.composer,geometry.topbar])
  assert.ok(b.right<=obstacle.left||b.left>=obstacle.right||b.bottom<=obstacle.top||b.top>=obstacle.bottom);
}

// Home fixtures: 1440 is Pedro's measured layout; the other widths are explicit
// responsive fixtures, not browser measurements. Text boxes deliberately fill
// the row, while Range returns the actual ink/line bounds (including nested span).
let rangeReads=0;
for(const selector of ['.welcome','.modes','#resume-card'])hosts[selector]=make('div');
const homeText=['.welcome .eyebrow','.welcome h1','.welcome > p:last-child'].map(selector=>{
 const node=make('p');node.textRects=[];hosts[selector]=node;return node;
});
hosts['.welcome'].querySelectorAll=()=>homeText;
hosts['#resume-card'].hidden=true;
document.createRange=()=>({selectNodeContents(node){this.node=node;},getClientRects(){rangeReads++;return this.node.textRects;},detach(){}});
function measuredHome(width=1440) {
 window.innerWidth=width;const phone=width<=800, left=phone?0:252, stageLeft=phone?18:292;
 geometry.scroll=rect(left,phone?58:64,width-left,phone?643:721);
 geometry.stage=rect(stageLeft,geometry.scroll.top,width-stageLeft-(phone?18:40),phone?432:278);
 const center=(left+width)/2,size=phone?96:144;
 geometry.duck=rect(stageLeft+(phone?24:36),geometry.stage.top+(phone?36:48),size,size);
 geometry.orb=rect(center-(phone?130:127.5),phone?214:71,phone?260:255,phone?260:255);
 geometry.label=rect(center-7,phone?470:322,14,20);
 geometry.topbar=rect(left,0,width-left,geometry.scroll.top);
 geometry.composer=rect(left,geometry.scroll.bottom,width-left,115);
 geometry.messages=rect(0,0,0,0);
 geometry.modes=rect(phone?18:center-280,phone?609:475,phone?width-36:560,42);
 const lines=phone?[rect(center-100,507,200,9),rect(center-137,526,274,36),rect(center-110,571,220,18)]:
  [rect(center-105,359,210,9),rect(center-174,378,348,49),rect(center-116,436,232,19)];
 homeText.forEach((node,i)=>{node.textRects=[lines[i]];node.getBoundingClientRect=()=>rect(stageLeft,lines[i].top,geometry.stage.width,lines[i].height);});
 hosts['.modes'].getBoundingClientRect=()=>geometry.modes;
 hosts['#content-scroll'].getBoundingClientRect=()=>geometry.scroll;
 hosts['#content-scroll'].scrollTop=0;
}
function homeSafe() {
 const b=duck.getBoundingClientRect(), h=duck.dataset.state==='searching'?b.width*.25:0;
 const halo=rect(b.left-h,b.top-b.width*.25,b.width+2*h,b.height+b.width*.25),a=geometry.scroll;
 assert.ok(halo.left>=a.left+12&&halo.right<=a.right-12&&halo.top>=a.top+12&&halo.bottom<=geometry.composer.top-12,JSON.stringify(halo));
 for(const o of [geometry.orb,geometry.label,geometry.modes,...homeText.flatMap(n=>n.textRects)])
  assert.ok(halo.right+12<=o.left||halo.left-12>=o.right||halo.bottom+12<=o.top||halo.top-12>=o.bottom,JSON.stringify({halo,o}));
 for(const n of [b.left,b.top])assert.equal(n,Math.round(n));
}
const canvasOf=node=>node.querySelector('.code-rain');
"""
    modules = '\n'.join((NOVO / name).read_text() for name in ('avatar.js', 'core.js', 'conversa.js') if (NOVO / name).exists())

    def run(steps, chat=False, home=False, width=1440):
        setup = "const avatar=window.cortexAvatar, core=window.cortexCore; assert.ok(avatar, 'Production avatar API');\n"
        if chat:
            setup += "const chat=window.cortexConversa; chat.init(()=>document.body.classList.add('chat-view')); await chat.open('test');\n"
            # Ordinary turn tests start after the one-time welcome has completed.
            setup += "advance(avatar.durations.wave);\n"
        initial = f'measuredHome({width});' if home else "document.body.classList.add('chat-view');"
        result = subprocess.run(['node', '-e', harness + extra + initial + modules + '\n(async()=>{\n' + setup + steps + '\n})().catch(e=>{console.error(e);process.exitCode=1;});'], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return result.stdout
    return run


@pytest.mark.parametrize('message,gestures', [('oi', ['wave']), ('oi! ótimo', ['wave', 'racket'])])
def test_sent_gestures_complete_while_real_status_and_head_advance(avatar_ui, message, gestures):
    avatar_ui('const message=' + json.dumps(message) + ';const gestures=' + json.dumps(gestures) + r""";
chat.submit(message);await settle();const live=hosts['#messages'].querySelectorAll('.assistant .message-content').at(-1);
advance(100);await emit({type:'status',stage:'thinking',label:'Pensando'});
assert.equal(duck.dataset.gesture,'wave');
assert.equal(live.querySelector('.reply-head').dataset.state,'thinking');
assert.equal(hosts['#state-label'].textContent,'Pensando');
advance(200);await emit({type:'status',stage:'web_search',label:'Buscando'});
const head=live.querySelector('.reply-head');assert.equal(head.dataset.state,'searching');
assert.ok(canvasOf(head) && !canvasOf(head).hidden);
assert.equal(hosts['#state-label'].textContent,'Buscando na web');
// Sample through both full durations, including just before each boundary.
let elapsed=300;
for(const name of gestures) {
 const duration=avatar.durations[name];
 while(elapsed<duration-1) {
  assert.equal(duck.dataset.state,'idle');assert.equal(duck.dataset.gesture,name);
  const delta=Math.min(100,duration-1-elapsed);advance(delta);elapsed+=delta;
 }
 assert.equal(duck.dataset.gesture,name);advance(1);elapsed=0;
}
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'searching');
assert.ok(canvasOf(duck) && !canvasOf(duck).hidden);
await emit({type:'final',turn});await close();
""", chat=True)


@pytest.mark.parametrize('ending', ['final', 'epoch', 'open', 'leave', 'hidden', 'body', 'media', 'preview'])
def test_sent_queue_releases_latest_state_or_cancels_without_replay(avatar_ui, ending):
    avatar_ui('const ending=' + json.dumps(ending) + r""";
chat.submit('oi! ótimo');await settle();advance(100);
await emit({type:'status',stage:'reading',label:'Lendo'});
if(ending==='final'||ending==='open') {
 await emit({type:'final',turn});await close();
 // A completed turn still lets both reactions finish, unless its epoch changes.
 assert.equal(duck.dataset.gesture,'wave');
 if(ending==='open')await chat.open('other');
 else {
  advance(avatar.durations.wave-101);assert.equal(duck.dataset.gesture,'wave');advance(1);
  assert.equal(duck.dataset.gesture,'racket');advance(avatar.durations.racket-1);
  assert.equal(duck.dataset.gesture,'racket');advance(1);
 }
 assert.equal(duck.dataset.state,'idle');
} else if(ending==='epoch') {
 chat.newConversation();assert.equal(duck.dataset.state,'idle');await close();
} else {
 if(ending==='leave')document.body.classList.remove('chat-view');
 if(ending==='hidden')visibility(true);
 if(ending==='body')document.body.classList.add('reduced-motion');
 if(ending==='media'){media.matches=true;media.change();}
 if(ending==='preview')core.preview('idle'); // Same displayed state must cancel too.
 assert.equal(duck.dataset.state,ending==='preview'?'idle':'reading');
 core.setLive('writing');assert.equal(duck.dataset.state,ending==='preview'?'idle':'writing');
 avatar.onSent('oi! ótimo');assert.equal(duck.dataset.gesture,ending==='leave'?'wave':'none');
 if(ending==='leave')document.body.classList.add('chat-view');
 if(ending==='hidden')visibility(false);
 if(ending==='body')document.body.classList.remove('reduced-motion');
 if(ending==='media'){media.matches=false;media.change();}
 if(ending==='preview')core.preview(null);
 assert.equal(duck.dataset.state,'writing');
 await emit({type:'final',turn});await close();
}
assert.equal(duck.dataset.gesture,'none');advance(5000);assert.equal(duck.dataset.gesture,'none');
assert.equal(duck.dataset.state,'idle');
""", chat=True)


def test_welcome_finishes_with_latest_state_and_random_still_yields(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(100);core.setLive('thinking');
advance(200);core.setLive('reading');advance(1499);
assert.equal(duck.dataset.gesture,'wave');advance(1);
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'reading');
core.setLive('idle');
const wait=[...pending.values()].find(job=>job.delay>=8000 && job.delay<=20000);advance(wait.at-now);
assert.notEqual(duck.dataset.gesture,'none');advance(100);core.setLive('thinking');
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'thinking');
advance(3000);assert.equal(duck.dataset.gesture,'none');
""")


def test_reaction_queue_is_bounded_and_latest_state_wins_across_both_gestures(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(avatar.durations.wave);
avatar.onSent('oi! ótimo');
// Eight waiting gestures maximum, in addition to the one already playing.
for(let i=0;i<20;i++)avatar.onSent('oi! ótimo');
core.setLive('thinking');
const expected=['wave','racket','wave','racket','wave','racket','wave','racket','wave'];
for(const [index,name] of expected.entries()) {
 assert.equal(duck.dataset.gesture,name);advance(avatar.durations[name]-1);
 core.setLive(index%2?'reading':'writing');assert.equal(duck.dataset.gesture,name);
 advance(1);
}
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'writing');
""")


def test_listening_and_preview_retain_the_distinct_live_memory(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view'); advance(1800);
core.setLive('listening');assert.equal(duck.dataset.state,'listening');
core.setLive('thinking','remembering');assert.equal(duck.dataset.state,'remembering');
for(const state of ['idle','thinking','searching','reading','writing','listening','speaking']) {
 core.preview(state);assert.equal(duck.dataset.state,state);
 core.setLive('thinking','remembering');assert.equal(duck.dataset.state,state);
}
core.preview(null);assert.equal(duck.dataset.state,'remembering');
""")


def test_welcome_once_random_waits_gestures_and_cancellation(avatar_ui):
    avatar_ui(r"""
assert.equal(duck.dataset.gesture,'wave');assert.ok(pending.size);
document.body.classList.add('chat-view');assert.equal(duck.dataset.gesture,'wave');advance(1800);
document.body.classList.remove('chat-view');assert.ok(pending.size,'home keeps its rest life');
document.body.classList.add('chat-view');assert.equal(duck.dataset.gesture,'none');
// This scheduler test keeps Pedro present; inactivity itself has separate tests below.
for(let i=0,previous='';i<20;i++) {
 page('keydown');
 const wait=[...pending.values()].find(job=>job.delay>=8000 && job.delay<=20000);assert.ok(wait && wait.delay<=20000);
 advance(wait.at-now); if(duck.dataset.next==='walk') advance(avatar.durations.turn); const name=duck.dataset.gesture;
 assert.ok(Object.keys(avatar.weights).includes(name)||Object.hasOwn(avatar.prankWeights,name));assert.notEqual(name,previous);previous=name;
 advance(avatar.durations[name]);assert.equal(duck.dataset.part,'none');assert.equal(duck.dataset.eyes,'open');
}
avatar.onSent('Oi, perfeito!');assert.equal(duck.dataset.gesture,'wave');advance(1800);
assert.equal(duck.dataset.gesture,'racket');advance(2600);assert.equal(duck.dataset.gesture,'none');
core.setLive('reading');if(duck.dataset.gesture==='turn')advance(avatar.durations.turn);advance(900);assert.equal(duck.dataset.eyes,'down-one');
visibility(true);assert.equal(pending.size,0);assert.equal(duck.dataset.eyes,'open');
visibility(false);assert.ok(pending.size);core.setLive('idle');
""")


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_search_rain_clears_and_reduced_motion_is_static(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);core.setLive('searching');
const rain=canvasOf(duck);assert.ok(rain && !rain.hidden);const initial=JSON.stringify(rain.context.marks);
advance(1000);assert.notEqual(JSON.stringify(rain.context.marks),initial);
if(mode==='body')document.body.classList.add('reduced-motion');else{media.matches=true;media.change();}
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.part,'none');assert.equal(duck.dataset.reduced,'true');
assert.equal(pending.size,0);assert.ok(rain.context.marks.every(m=>m[5]<=.26));
const still=JSON.stringify(rain.context.marks);advance(60000);assert.equal(JSON.stringify(rain.context.marks),still);
avatar.onSent('Oi, perfeito!');assert.equal(duck.dataset.gesture,'none');
document.body.classList.add('light');assert.ok(rain.context.marks.some(m=>m[4]==='rgb(22 119 55)'));
core.setLive('reading');assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
core.setLive('searching');assert.equal(canvasOf(duck),rain);
visibility(true);assert.equal(pending.size,0);visibility(false);
for(const state of ['idle','thinking','reading','writing','remembering','listening','speaking','processing']) {
 avatar.setLive('searching');avatar.setLive(state);assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
}
""")


def test_real_status_stages_head_order_label_stripping_and_first_token(avatar_ui):
    avatar_ui(r"""
chat.submit('ordinary question');await settle();const live=hosts['#messages'].querySelectorAll('.assistant .message-content').at(-1);
const expected={thinking:'thinking',web_search:'searching',reading:'reading',writing:'writing',remembering:'remembering',unknown:'thinking'};
let head,dots,text;
for(const [stage,state] of Object.entries(expected)) {
 await emit({type:'status',stage,label:'🔍 buscando: 📖 query 🧠.txt'});
 const p=live.querySelector('p');
 if(!head){head=p.children[0];text=p.children[1];dots=p.children[2];}
 assert.equal(head.className,'reply-head');assert.equal(head.getAttribute('aria-hidden'),'true');
 assert.equal(head.dataset.state,state);assert.equal(duck.dataset.state,state);
 assert.equal(p.children[0],head);assert.equal(p.children[1],text);assert.equal(p.children[2],dots);
 assert.equal(dots.className,'reply-dots');assert.equal(dots.children.length,3);
 assert.equal(text.textContent,'buscando: 📖 query 🧠.txt');
}
for(const [label,clean] of [['  ✍️  writing','writing'],['🧠thinking','thinking'],['📖 file','file'],['plain 🔍 query','plain 🔍 query'],['🎾 keep','🎾 keep']]) {
 await emit({type:'status',stage:'writing',label});assert.equal(text.textContent,clean);
}
await emit({type:'status',stage:'web_search',label:'🔍 query'});const rain=canvasOf(head);assert.ok(rain && !rain.hidden);
await emit({type:'token',text:'First'});assert.equal(duck.dataset.state,'speaking');
assert.equal(head.parentNode,null);assert.equal(dots.parentNode,null);assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
assertIndicator(live,'cursor');await emit({type:'status',stage:'reading',label:'late'});assert.equal(head.parentNode,null);
await emit({type:'final',turn});await close();assert.equal(duck.dataset.state,'idle');assert.equal(indicators(live).length,0);
""", chat=True)


@pytest.mark.parametrize('ending', ['final', 'error', 'stop', 'epoch', 'throw', 'closed'])
def test_head_and_rain_clear_on_all_turn_exits_and_history(avatar_ui, ending):
    avatar_ui('const ending=' + json.dumps(ending) + r""";
chat.submit('question');await settle();const live=hosts['#messages'].querySelectorAll('.assistant .message-content').at(-1);
await emit({type:'status',stage:'web_search',label:'🔍 query'});
const head=live.querySelector('.reply-head'), rain=canvasOf(head);
if(ending==='final'||ending==='error') {await emit(ending==='final'?{type:'final',turn}:{type:'error',detail:'Failure'});}
else if(ending==='stop'||ending==='epoch') {if(ending==='stop')chat.stop();chat.newConversation();}
else await close(ending==='throw'?'Failed':null);
assert.equal(head.parentNode,null);assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
assert.equal(indicators(live).length,0);assert.equal(duck.dataset.state,'idle');
assert.equal(live.querySelectorAll('.avatar-progress').length,0);
if(pendingRead) await close();
historyTurns=[{...turn,user:'Oi, perfeito!',reply:'Oi, perfeito!'}];await chat.open('history');
assert.equal(hosts['#messages'].querySelectorAll('.reply-head').length,0);
assert.equal(duck.dataset.gesture,'none');
""", chat=True)


def test_detection_matches_lab_negatives_and_runs_only_on_accepted_sends(avatar_ui):
    # Keep the approved lab's full deterministic signal table as an independent oracle.
    import ast
    tree = ast.parse((ROOT / 'tests/test_avatar_lab.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_message_detection_is_deterministic_with_word_boundaries_and_negation')
    steps = ast.literal_eval(fn.body[0].value.args[0]).replace('window.cortexAvatarLab', 'window.cortexAvatar')
    avatar_ui(steps)
    avatar_ui(r"""
let sent=[];const original=avatar.onSent;window.cortexAvatar={...avatar,onSent:text=>{sent.push(text);original(text);}};
historyTurns=[{...turn,user:'Oi, perfeito!',reply:'Oi, perfeito!'}];await chat.open('history');assert.deepEqual(sent,[]);
chat.submit('Oi, perfeito!');await settle();assert.deepEqual(sent,['Oi, perfeito!']);
chat.submit('ignored while busy');assert.equal(sent.length,1);
await emit({type:'token',text:'Oi, perfeito!'}, {type:'final',turn});await close();assert.equal(sent.length,1);
chat.submit('sim','pending-card');await settle();assert.equal(sent.length,1);chat.newConversation();await close();
""", chat=True)


def test_approved_art_signals_and_animation_tracks_stay_in_sync():
    lab = (ROOT / 'docs/design/avatar_lab/avatar.js').read_text()
    production = (NOVO / 'avatar.js').read_text()
    for name in ('SAMPLES', 'HEAD', 'WING', 'RACKET', 'SIGNALS', 'RAIN_GLYPHS'):
        pattern = rf'const {name} = ([\s\S]*?);'
        assert re.search(pattern, production)[1] == re.search(pattern, lab)[1], name
    old_regions = json.loads(re.search(r'const REGIONS = (\{.*?\});', lab, re.S)[1])
    new_regions = json.loads(re.search(r'const REGIONS = (\{.*?\});', production, re.S)[1])
    assert all(new_regions[name] == rect for name, rect in old_regions.items())
    lab_css = (ROOT / 'docs/design/avatar_lab/avatar.css').read_text()
    css = (NOVO / 'avatar.css').read_text()
    # All measured movement tracks, including racket/wing registration, are exact.
    tracks = lambda s: re.findall(r'@keyframes [\w-]+\s*\{(?:[^{}]|\{[^{}]*\})*\}', s)
    assert all(track in tracks(css) for track in tracks(lab_css))
    for state in ('thinking', 'searching', 'reading', 'writing', 'remembering', 'processing'):
        assert f'.reply-head[data-state="{state}"] .reply-head-motion' in css
        assert re.search(rf'\.reply-head\[data-state="{state}"\]\s*\{{[^}}]*--reply-still-pose:', css)
    for block in (css.split('/* Reply head: app reduced toggle */')[1].split('/* Reply head: OS reduced motion */')[0],
                  css.split('/* Reply head: OS reduced motion */')[1]):
        assert 'animation: none !important' in block
        for cue in ('pose', 'eyes', 'ear', 'antenna'):
            assert f'var(--reply-still-{cue}' in block


def css_rules(source, width, height):
    """Small cascade reader for this project's flat rules and width/height media."""
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    result = []
    pos = 0
    while pos < len(source):
        start = source.find('{', pos)
        if start < 0:
            break
        heading = source[pos:start].strip()
        depth, end = 1, start + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[start + 1:end - 1]
        if heading.startswith('@media'):
            bounds = re.findall(r'(min|max)-(width|height):\s*(\d+)px', heading)
            if bounds and all((width if axis == 'width' else height) >= int(n) if limit == 'min'
                              else (width if axis == 'width' else height) <= int(n) for limit, axis, n in bounds):
                result.extend(css_rules(body, width, height))
        elif not heading.startswith('@'):
            declarations = dict(re.findall(r'([\w-]+)\s*:\s*([^;]+)', body))
            result.extend((selector.strip(), declarations) for selector in heading.split(','))
        pos = end
    return result


@pytest.mark.parametrize('width', [375, 480, 512, 513, 600, 800, 801, 1000, 1100, 1101,
                                 1280, 1440, 1657, 1658, 1728, 1920])
@pytest.mark.parametrize('height', [667, 900, 1000])
def test_actual_css_envelopes_clear_sphere_label_topbar_and_messages(width, height):
    base = (NOVO / 'style.css').read_text()
    css = (NOVO / 'avatar.css').read_text()
    rules = css_rules(base + '\n' + css, width, height)

    def rule(selector, source_rules=rules):
        merged = {}
        for name, declarations in source_rules:
            if name == selector:
                merged.update(declarations)
        return merged

    def px(s):
        return float(s.removesuffix('px'))

    # Frozen shell gutters, with the sidebar removed from flow on mobile.
    sidebar = px(rule('.sidebar')['width']) if width > 800 else 0
    scroll = rule('.content-scroll')
    padding = scroll['padding'].split()
    gutter = px(padding[1])
    if width > 800:
        gutter = px(scroll.get('padding-left', padding[1]))
    stage_width = width - sidebar - 2 * gutter
    stage = {**rule('.stage'), **rule('.chat-view .stage')}
    dock = {**rule('.duck-dock'), **rule('.chat-view .duck-dock')}
    # Read the same dock's scrolling placement independently of its fixed rule.
    scrolling_rules = [(name, declarations) for name, declarations in rules
                       if declarations.get('position') != 'fixed']
    scrolling_dock = rule('.chat-view .duck-dock', scrolling_rules)
    flow_left, flow_top = px(scrolling_dock['left']), px(scrolling_dock['top'])
    size = px(rule('.chat-view')['--duck-size'])
    orb = px(rule('.chat-view')['--orb-size'])
    assert size == (96 if width <= 800 else 144)
    assert orb == (164 if width <= 800 else 190 if height >= 950 else 154 if height <= 790 else 174)
    left, top = px(dock['left']), px(dock['top'])
    topbar = px(rule('.topbar')['height'])
    rain = rule('.duck .code-rain')
    percent = lambda value: float(value.removesuffix('%')) / 100
    rain_left, rain_top = percent(rain['left']), percent(rain['top'])
    rain_right = rain_left + percent(rain['width'])
    rain_bottom = rain_top + percent(rain['height'])
    clearance = 12
    messages = rule('.messages')
    assert messages['margin'] == '0 auto'
    message_max = px(messages['max-width'])
    message_left = (stage_width - min(stage_width, message_max)) / 2
    required_gutter = flow_left + size * max(1, rain_right) + clearance
    fits = width > 800 and message_left >= required_gutter
    assert dock['position'] == ('fixed' if fits else 'absolute')
    assert dock['pointer-events'] == 'none'
    assert (flow_left, flow_top) == ((36, 48) if width > 800 else (24, 32))
    # The breakpoint must follow the real shell, centered message column, duck
    # and full rain canvas, including transparent mask edges. At 1657px the
    # gutter is 0.5px short; at 1658px it clears by exactly 12px.
    if width > 1100:
        threshold = sidebar + 2 * gutter + message_max + 2 * required_gutter
        assert threshold == 1658
        assert fits == (width >= threshold)
    if width > 800:
        viewport_left = left if fits else sidebar + gutter + left
        viewport_top = top if fits else topbar + top
        # Fixed from first paint: its position equals the scrolling placement
        # at scrollTop=0, so starting to scroll cannot cause a jump.
        assert (viewport_left, viewport_top) == (sidebar + gutter + flow_left, topbar + flow_top)
        assert viewport_top + size * rain_top >= topbar
        composer_budget = px(rule('.composer textarea')['max-height']) + 140
        assert viewport_top + size < max(height, px(rule('.app-shell')['min-height'])) - composer_budget
    if fits:
        assert left + size <= sidebar + gutter + message_left - clearance
        assert left + size * rain_right <= sidebar + gutter + message_left - clearance
        assert int(dock['z-index']) > 0
        for ancestor in ('.app-shell', '.workspace', '.content-scroll', '.stage', '.chat-view .stage'):
            style = rule(ancestor)
            for prop in ('transform', 'filter', 'perspective'):
                assert style.get(prop, 'none') == 'none'  # fixed remains viewport-relative
        left -= sidebar + gutter
        top -= topbar
    elif width <= 800:
        assert (left, top) == (24, 32)
    # Full rectangular canvas, including its transparent mask edges.
    halo = (left + size * rain_left, top + size * rain_top,
            left + size * rain_right, top + size * rain_bottom)
    orb_left, orb_top = (stage_width - orb) / 2, px(stage['padding-top'])
    orb_bounds = (orb_left, orb_top, orb_left + orb, orb_top + orb)
    separate = lambda a, b: a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    assert separate(halo, orb_bounds), (width, height, halo, orb_bounds)
    label = (orb_left, orb_top + orb - 4, orb_left + orb, orb_top + orb + 16)
    assert separate(halo, label)
    assert halo[0] >= 0 and halo[1] >= 0  # stage starts below topbar
    stage_height = max(px(stage['min-height']), orb_top + orb + 16)
    assert halo[2] <= stage_width and halo[3] < stage_height
    # Even full-image idle float, hop and stretch remain in the reserved row.
    assert top - size * .05 > 0
    assert '.content-scroll {' not in css
    assert '.composer-area' not in css and '.welcome' not in css


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_head_rain_static_repaint_pause_and_disposal(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
const head=avatar.createHead('thinking');assert.equal(head.querySelector('img').getAttribute('src'),'/ui/static/novo/avatar/duckcortex.png');
assert.equal(head.style['--reply-file-size'],1254/682);
assert.equal(canvasOf(head),null);avatar.updateHead(head,'searching');const rain=canvasOf(head);
assert.deepEqual([rain.width,rain.height],[40,40]);assert.equal(rain.getAttribute('aria-hidden'),'true');
if(mode==='body')document.body.classList.add('reduced-motion');else{media.matches=true;media.change();}
assert.ok(rain.context.marks.length && rain.context.marks.every(m=>m[5]<=.26));
const still=JSON.stringify(rain.context.marks);advance(60000);assert.equal(JSON.stringify(rain.context.marks),still);
document.body.classList.add('light');assert.ok(rain.context.marks.some(m=>m[4]==='rgb(57 255 112)'));
visibility(true);assert.equal(head.dataset.paused,'true');assert.equal(pending.size,0);
visibility(false);assert.equal(head.dataset.paused,'false');
avatar.updateHead(head,'reading');assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
avatar.updateHead(head,'searching');assert.equal(canvasOf(head),rain);
avatar.removeHead(head);assert.equal(rain.hidden,true);assert.deepEqual(rain.context.marks,[]);
avatar.setLive('idle');assert.equal(pending.size,1,'only the inactivity clock remains');assert.ok([...pending.values()][0].delay>20000);
""")


@pytest.mark.parametrize('name', ['turn', 'walk', 'pirouette', 'flap', 'stumble'])
def test_new_gesture_duration_cleanup_and_facing(avatar_ui, name):
    avatar_ui('const name=' + json.dumps(name) + r""";
document.body.classList.add('chat-view');advance(avatar.durations.wave);
assert.equal(duck.dataset.facing,'right');assert.equal(avatar.play(name),true);
const duration=avatar.durations[name];
assert.ok(duration>=({turn:600,walk:1500,pirouette:1200,flap:1000,stumble:1200})[name]);
assert.ok(duration<=({turn:900,walk:5000,pirouette:1600,flap:1400,stumble:1600})[name]);
advance(duration-1);assert.equal(duck.dataset.gesture,name);advance(1);
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.part,'none');assert.equal(duck.dataset.eyes,'open');
assert.equal(duck.dataset.facing,name==='turn'?'left':'right');
for(const node of [duck,...duck.querySelectorAll('*')]) {
 assert.equal(node.style.transform,undefined);assert.equal(node.style.clipPath,undefined);
}
if(name==='turn') {assert.ok(avatar.play('turn'));advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'right');}
""")


def test_measured_parts_and_original_rest_partition(avatar_ui):
    clips = json.loads(avatar_ui(r"""
for(const name of ['leftFoot','rightFoot','blueWing']) {
 assert.ok(avatar.regions[name]);assert.ok(avatar.contours[name].length>=8);
 const image=duck.querySelector('.duck-'+name);
 assert.ok(image);assert.equal(image.getAttribute('src'),'/ui/static/novo/avatar/duckcortex.png');
 assert.ok(duck.style['--'+name+'-clip'].startsWith('polygon('));
}
console.log(JSON.stringify(duck.style['--limbs-outside']));
"""))
    # At native pixel centres the inverse and moving clips partition every source
    # byte exactly once (no alpha doubling or omitted pixels), also in combination.
    source = (NOVO / 'avatar.js').read_text()
    contours = [json.loads(re.search(rf'const {name} = (\[.*?\]);', source, re.S)[1])
                for name in ('WING', 'BLUE_WING', 'LEFT_FOOT', 'RIGHT_FOOT')]
    def spans(points, y):
        from math import ceil
        crossings = sorted(ax + (y - ay) * (bx - ax) / (by - ay)
                           for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1])
                           if (ay > y) != (by > y))
        return [(max(0, ceil(a - .5)), min(1254, ceil(b - .5)))
                for a, b in zip(crossings[::2], crossings[1::2])]
    # Independent scanline PNG decoder from task 133; no imaging dependency.
    import ast
    text = (ROOT / 'tests/test_avatar_lab.py').read_text()
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_every_colour_sample_matches_original_rgba')
    body = fn.body[:next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'source' for t in n.targets))]
    namespace = {'ROOT': ROOT, 'PNG_URL': '/src/ui/static/novo/avatar/duckcortex.png'}
    exec(compile(ast.Module(body=body, type_ignores=[]), '<png decoder>', 'exec'), namespace)
    inverse = [[round(float(x) * 1254 / 100, 9), round(float(y) * 1254 / 100, 9)]
               for x, y in re.findall(r'([\d.]+)% ([\d.]+)%', clips)]
    assert len(inverse) > sum(map(len, contours))
    original = b''.join(namespace['rows'])
    composite = bytearray(len(original))
    for y in range(1254):
        coverage = bytearray(1254)
        for contour in [inverse, *contours]:
            for left, right in spans(contour, y + .5):
                for x in range(left, right):
                    coverage[x] += 1
                start, end = (y * 1254 + left) * 4, (y * 1254 + right) * 4
                composite[start:end] = original[start:end]
        assert coverage == bytearray([1]) * 1254
    assert bytes(composite) == original
    css = (NOVO / 'avatar.css').read_text()
    assert 'clip-path: var(--limbs-outside)' in css
    assert '.duck[data-facing="left"] .duck-pose { transform: scaleX(-1); }' in css


def test_walk_band_integer_steps_reverse_resize_and_live_interrupt(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(10);
assert.ok(avatar.play('walk'));const band=avatar.walkingBand();assert.ok(band.max>20);
let last=0;
for(let i=0;i<20;i++){advance(100);const x=Number(duck.dataset.x);assert.equal(x,Math.round(x));assert.ok(x>=0 && x<=band.max);last=x;}
assert.ok(last>0);core.setLive('thinking');assert.equal(duck.dataset.x,String(last));
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'thinking');
core.setLive('idle');avatar.setSeed(2);assert.ok(avatar.play('walk'));
assert.equal(duck.dataset.gesture,'turn');const before=duck.dataset.x;advance(avatar.durations.turn);
assert.equal(duck.dataset.facing,'left');assert.equal(duck.dataset.gesture,'walk');assert.equal(duck.dataset.x,before);
advance(300);assert.ok(Number(duck.dataset.x)<last);
const interrupted=duck.dataset.x;core.setLive('searching');assert.equal(duck.dataset.x,interrupted);
assert.equal(duck.dataset.gesture,'turn');assert.equal(duck.dataset.part,'none');
core.setLive('reading');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.state,'reading');
core.setLive('idle');avatar.setSeed(10);avatar.play('walk');advance(1200);
const old=Number(duck.dataset.x);geometry.orb=rect(220,112,174,174);resize();
assert.equal(Number(duck.dataset.x),old,'resize must not teleport');
advance(avatar.durations.turn+5000);assert.ok(Number(duck.dataset.x)<=avatar.walkingBand().max);
""")


@pytest.mark.parametrize('obstacle', ['orb', 'label', 'messages', 'composer', 'topbar'])
def test_walk_band_uses_each_live_obstacle_and_full_halo(avatar_ui, obstacle):
    avatar_ui('const obstacle=' + json.dumps(obstacle) + r""";
document.body.classList.add('chat-view');advance(1800);
if(['composer','topbar'].includes(obstacle)) {
 window.innerWidth=1728;geometry.messages=rect(900,320,870,450);
}
geometry.orb=rect(900,112,100,100);geometry.label=rect(900,250,100,20);
geometry[obstacle]=rect(300,100,100,180);
assert.equal(avatar.walkingBand().max,72); // 300 - 12 - home36 - 144*1.25
geometry[obstacle]=rect(218,100,100,180);avatar.setSeed(10);avatar.play('walk');
assert.equal(duck.dataset.gesture,'walk');advance(avatar.durations.walk);
assert.equal(duck.dataset.x,'0','less than one step: in-place waddle');
""")


@pytest.mark.parametrize('top', [111.5, 112, 40])
def test_unpinned_measured_desktop_band_survives_scroll(avatar_ui, top):
    avatar_ui('const top=' + str(top) + r""";
window.innerWidth=1440;geometry.stage=rect(292,64,1108,228);
geometry.duck=rect(328,top,144,144);geometry.topbar=rect(252,0,1188,64);
geometry.orb=rect(759,67,174,174);geometry.label=rect(839,237,14,20);
geometry.messages=rect(411,374,870,374);geometry.composer=rect(252,784.5,1188,115.5);
hosts['#content-scroll'].scrollTop=112-top;
hosts['#content-scroll'].getBoundingClientRect=()=>rect(252,64,1188,720.5);
assert.equal(avatar.walkingBand().max,239,'sphere bounds the clipped walking row');
""")


@pytest.mark.parametrize('obstacle', ['topbar', 'composer'])
def test_unpinned_band_ignores_obstacles_outside_scroll_container(avatar_ui, obstacle):
    avatar_ui('const obstacle=' + json.dumps(obstacle) + r""";
geometry[obstacle]=rect(0,100,1440,180);
assert.equal(avatar.walkingBand().max,239);
""")


@pytest.mark.parametrize('top,expected', [(112, 380), (111.5, 380), (110.5, 0)])
def test_pinned_band_tolerates_subpixel_offset_but_not_real_overlap(avatar_ui, top, expected):
    avatar_ui('const top=' + str(top) + ';const expected=' + str(expected) + r""";
window.innerWidth=1728;geometry.stage=rect(292,64,1396,228);geometry.duck=rect(328,top,144,144);
geometry.orb=rect(900,67,174,174);geometry.label=rect(980,237,14,20);
geometry.messages=rect(900,374,870,374);geometry.topbar=rect(252,0,1476,64);
geometry.composer=rect(252,784.5,1476,115.5);
assert.equal(avatar.walkingBand().max,expected);
""")


@pytest.mark.parametrize('obstacle', ['orb', 'label', 'messages', 'composer', 'topbar'])
@pytest.mark.parametrize('edge', ['top', 'bottom'])
def test_all_band_obstacles_use_whole_pixel_vertical_clearance(avatar_ui, obstacle, edge):
    avatar_ui('const obstacle=' + json.dumps(obstacle) + ';const edge=' + json.dumps(edge) + r""";
// Messages reserve a column when pinned; exercise their row overlap when clipped.
window.innerWidth=obstacle==='messages'?1440:1728;
geometry.orb=rect(900,112,100,100);geometry.label=rect(900,239,100,20);
geometry.messages=rect(900,374,870,374);
geometry[obstacle]=rect(-300,edge==='top'?-36.5:267.5,1000,100);
assert.equal(avatar.walkingBand().max,672,'rounded edges leave the full clearance');
geometry[obstacle]=rect(-300,edge==='top'?-34.5:265.5,1000,100);
assert.equal(avatar.walkingBand().max,0,'a real overlap still bounds the row');
""")


def test_pinned_mobile_and_sent_wave_preserve_position(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);
window.innerWidth=1658;geometry.stage=rect(292,64,1326,228);geometry.duck=rect(328,112,144,144);
geometry.orb=rect(868,69,174,174);geometry.label=rect(868,239,174,20);
geometry.messages=rect(520,400,870,100);assert.equal(avatar.walkingBand().max,0);
geometry.messages=rect(555,400,870,100);assert.equal(avatar.walkingBand().max,35);
geometry.messages=rect(555,400,870,0);assert.equal(avatar.walkingBand().max,35,'reserve even an empty pinned message column');
window.innerWidth=375;geometry.stage=rect(18,58,339,340);geometry.duck=rect(42,90,96,96);
geometry.orb=rect(106,214,164,164);geometry.label=rect(181,374,14,20);
geometry.messages=rect(18,410,339,200);geometry.topbar=rect(0,0,375,58);geometry.composer=rect(0,700,375,100);
assert.equal(avatar.walkingBand().max,183);avatar.setSeed(10);avatar.play('walk');advance(800);
const x=duck.dataset.x;assert.ok(Number(x)>0);avatar.onSent('oi!');
assert.equal(duck.dataset.gesture,'wave');assert.equal(duck.dataset.x,x);
core.setLive('writing');advance(1800);assert.equal(duck.dataset.state,'writing');assert.equal(duck.dataset.x,x);
""")


def test_walk_turns_from_blocked_left_side_and_uses_phone_band(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);
window.innerWidth=375;geometry.stage=rect(18,58,339,340);geometry.duck=rect(42,90,96,96);
geometry.orb=rect(106,214,164,164);geometry.label=rect(181,374,14,20);
geometry.messages=rect(18,410,339,200);geometry.topbar=rect(0,0,375,58);
geometry.composer=rect(0,700,375,112);
avatar.play('turn');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'left');
avatar.setSeed(10);assert.ok(avatar.play('walk'));
assert.equal(duck.dataset.gesture,'turn');assert.equal(duck.dataset.next,'walk');assert.equal(duck.dataset.x,'0');
advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.gesture,'walk');
advance(50);assert.equal(duck.dataset.x,'1');
advance(avatar.durations.walk);assert.ok(Number(duck.dataset.x)>0);assert.ok(Number(duck.dataset.x)<=183);
""")


@pytest.mark.parametrize('mode', ['body', 'media', 'hidden', 'preview'])
def test_weighted_scheduler_seed_nonrepeat_and_rest_eligibility(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
assert.deepEqual(avatar.weights,{look:6,walk:6,hop:2,wave:2,stretch:2,turn:2,pirouette:1,flap:2,stumble:1,dance:2,peek:2,cheer:2,no:1});
function sequenceFor(seed) {
 avatar.setSeed(seed);let names=[];
 for(let i=0;i<80;i++) {
 page('keydown');
  const wait=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);assert.ok(wait && wait.delay<=20000);
  advance(wait.at-now);let name=duck.dataset.gesture;
  // A selected walk may begin by turning to face its destination.
  if(name==='turn' && duck.dataset.next==='walk'){advance(avatar.durations.turn);name=duck.dataset.gesture;}
  assert.ok(Object.hasOwn(avatar.weights,name)||Object.hasOwn(avatar.prankWeights,name));names.push(name);advance(avatar.durations[name]);
 }
 return names;
}
const names=sequenceFor(42);avatar.cancelGestures();duck.dataset.facing='right';
// No-repeat is on the selected movement (a prerequisite turn is not a selection).
assert.ok(names.every((n,i)=>!i||n!==names[i-1]));assert.ok(names.filter(n=>n==='look').length>names.filter(n=>n==='stumble').length);
if(mode==='body')document.body.classList.add('reduced-motion');
if(mode==='media'){media.matches=true;media.change();}
if(mode==='hidden')visibility(true);
if(mode==='preview')core.preview('idle');
assert.equal(avatar.randomGestureNow(),false);advance(120000);assert.equal(duck.dataset.gesture,'none');
if(mode!=='preview')assert.equal(avatar.play('flap'),false);
else {assert.equal(avatar.play('flap'),true);advance(avatar.durations.flap);core.preview('thinking');assert.equal(avatar.play('flap'),false);}
""")


def test_demo_select_exact_parity_and_real_listener(avatar_ui):
    from tests.ui_parity import GESTURE_SELECT
    html = (ROOT / 'src/ui/templates/novo.html').read_text()
    assert GESTURE_SELECT in html
    assert not check(html)
    for changed in (html.replace(GESTURE_SELECT, ''), html.replace('id="gesture-select"', 'id="wrong-select"'),
                    html.replace('>Tropeço</option>', '>Wrong</option>'), html.replace('<option value="flap">Bater as asas</option>', '')):
        assert check(changed)
    shell = (NOVO / 'shell.js').read_text()
    start = shell.index("$('#gesture-select').addEventListener")
    end = shell.index("$('#preview-chat').addEventListener", start)
    # Run both production listeners; dialog.close() queues its close event.
    avatar_ui(r"""
const controls=new EventTarget();controls.open=true;let closeEvents=0;
controls.close=()=>{controls.open=false;setTimeout(()=>{controls.dispatchEvent(new Event('close'));closeEvents++;},0);};
const select=new EventTarget(),stateSelect={value:'idle'};
const $=selector=>({'#controls':controls,'#gesture-select':select,'#state-select':stateSelect})[selector];
""" + shell[start:end] + r"""
const choose=value=>{select.value=value;select.dispatchEvent(new Event('change'));};
document.body.classList.add('chat-view');advance(1800);core.preview('idle');
choose('flap');assert.equal(select.value,'');assert.equal(controls.open,false);
assert.equal(closeEvents,0);assert.equal(duck.dataset.gesture,'none','wait for the asynchronous close event');
advance(0);assert.equal(closeEvents,1);assert.equal(stateSelect.value,'idle');assert.equal(duck.dataset.gesture,'flap');
advance(avatar.durations.flap-1);assert.equal(duck.dataset.gesture,'flap');advance(1);
assert.equal(duck.dataset.gesture,'none');
controls.open=true;core.preview('thinking');choose('turn');assert.equal(select.value,'');assert.equal(controls.open,false);
advance(0);assert.equal(closeEvents,2);assert.equal(stateSelect.value,'idle');assert.equal(duck.dataset.gesture,'turn');
advance(avatar.durations.turn);choose('wave');assert.equal(select.value,'');
advance(0);assert.equal(closeEvents,2);assert.equal(duck.dataset.gesture,'none','closed dialog cannot request a demo');
""")


@pytest.mark.parametrize('name,elapsed', [('walk', 800), ('turn', 500), ('pirouette', 600), ('flap', 600), ('stumble', 600)])
def test_live_interrupts_every_new_move_and_cancels_all_part_clocks(avatar_ui, name, elapsed):
    avatar_ui('const name=' + json.dumps(name) + ';const elapsed=' + str(elapsed) + r""";
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(10);
avatar.play(name);advance(elapsed);const x=duck.dataset.x;const left=duck.dataset.facing==='left';
core.setLive('listening');assert.equal(duck.dataset.x,x);assert.equal(duck.dataset.part,left?'none':'head');
if(left){assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);}
assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.state,'listening');assert.equal(duck.dataset.gesture,'none');
advance(5000);assert.equal(duck.dataset.x,x);assert.equal(duck.dataset.gesture,'none');
""")


def test_epoch_cancellation_never_starts_another_gesture(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.play('turn');advance(avatar.durations.turn);
avatar.onSent('oi!');core.setLive('thinking');const x=duck.dataset.x;
avatar.cancelGestures();assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.x,x);
advance(5000);assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.x,x);
""")


def test_seed_reproducibility_and_hidden_wait_is_fresh(avatar_ui):
    script = r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(31415);
const result=[];
for(let i=0;i<80;i++) {
 page('keydown');
 const wait=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);advance(wait.at-now);
 const name=duck.dataset.next==='walk'?'walk':duck.dataset.gesture;
 result.push([wait.delay,name]);if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 advance(avatar.durations[name]);
}
console.log(JSON.stringify(result));
visibility(true);assert.equal(pending.size,0);advance(120000);visibility(false);
assert.equal(duck.dataset.gesture,'none');const wait=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);
assert.ok(wait.delay<=20000);advance(wait.delay-1);assert.equal(duck.dataset.gesture,'none');
"""
    assert avatar_ui(script) == avatar_ui(script)


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_reduced_interrupt_faces_right_without_a_turn_and_preserves_x(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(10);avatar.play('walk');advance(1000);
const x=duck.dataset.x;avatar.cancelGestures();avatar.play('turn');advance(avatar.durations.turn);
if(mode==='body')document.body.classList.add('reduced-motion');else {media.matches=true;media.change();}
core.setLive('speaking');assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.gesture,'none');
assert.equal(duck.dataset.x,x);assert.equal(pending.size,0);assert.equal(avatar.play('turn'),false);
""")



def test_resize_recovery_can_take_multiple_legs_and_never_jumps(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(10);
for(const key of ['orb','label','messages'])geometry[key]=rect(1800,400,100,100);
geometry.stage=rect(0,64,2200,228);
for(let i=0;i<4;i++) {avatar.play('walk');advance(avatar.durations.walk);}
assert.ok(Number(duck.dataset.x)>200);const start=Number(duck.dataset.x);
geometry.stage=rect(0,64,220,228);avatar.setSeed(0);resize();assert.equal(Number(duck.dataset.x),start);
let last=start;
for(let t=0;t<35000;t+=50) {
 advance(50);const x=Number(duck.dataset.x);
 assert.ok(x<=last && last-x<=2,'recovery stays at its calm walking pace');last=x;
 if(x===0)break;
}
assert.equal(duck.dataset.x,'0');
""")


def test_sent_reaction_from_left_finishes_before_one_rightward_turn(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.play('turn');advance(avatar.durations.turn);
avatar.onSent('oi! ótimo');core.setLive('thinking');
assert.equal(duck.dataset.gesture,'wave');assert.equal(duck.dataset.facing,'left');advance(1800);
assert.equal(duck.dataset.gesture,'racket');core.setLive('reading');advance(2600);
assert.equal(duck.dataset.gesture,'turn');core.setLive('writing');advance(avatar.durations.turn);
assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'writing');
""")


@pytest.mark.parametrize('width', [375, 800, 1440, 1728])
def test_click_hit_box_only_and_calm_touch_css(width):
    rules = css_rules((NOVO / 'avatar.css').read_text(), width, 900)

    def rule(selector):
        return {key: value for name, declarations in rules if name == selector
                for key, value in declarations.items()}

    hit = rule('.duck')
    for key, value in {'pointer-events': 'auto', 'cursor': 'pointer',
                       'touch-action': 'manipulation', '-webkit-tap-highlight-color': 'transparent',
                       'user-select': 'none', '-webkit-user-select': 'none', 'outline': 'none'}.items():
        assert hit[key] == value
    assert hit['width'] == hit['height'] == 'var(--duck-size)'
    # All nested artwork ignores hits, even where transforms spill out of the box.
    assert rule('.duck *')['pointer-events'] == 'none'
    assert rule('.duck img')['-webkit-user-drag'] == 'none'
    for selector in ('.duck-dock', '.chat-view .duck-dock', '.code-rain',
                     '.duck-racket', '.duck-part', '.duck-effect'):
        assert rule(selector)['pointer-events'] == 'none'
        assert rule(selector).get('cursor') != 'pointer'
    # Task 162: besides the figure, only the pile it leaves takes a click (to clean it up).
    assert [name for name, declarations in rules if declarations.get('pointer-events') == 'auto'] == ['.duck', '.duck-prop-poop']


def test_click_uses_same_seed_and_rest_weights_plus_racket(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);
assert.equal(duck.eventListeners?.click?.length,1,'bind only the figure');
assert.equal(dock.eventListeners?.click,undefined);
function seeded(seed) {
 return () => {
  seed=(seed+0x6D2B79F5)>>>0;let v=seed;
  v=Math.imul(v^v>>>15,v|1);v^=v+Math.imul(v^v>>>7,v|61);
  return ((v^v>>>14)>>>0)/4294967296;
 };
}
const seen=new Set();
// 160 seeds: with the task-157 moves in the pool, the rarest one first appears at seed 143.
for(let seed=0;seed<160;seed++) {
 avatar.cancelGestures();avatar.setSeed(seed);const rng=seeded(seed);rng(); // scheduled rest wait
 assert.deepEqual(avatar.clickWeights,{racket:8,startle:4});
 const choices=Object.entries({...avatar.weights,...avatar.clickWeights});
 let ticket=rng()*choices.reduce((n,[,weight])=>n+weight,0);
 const expected=choices.find(([,weight])=>(ticket-=weight)<0)[0];
 clickDuck();const name=duck.dataset.next==='walk'?'walk':duck.dataset.gesture;
 assert.equal(name,expected,'same seeded generator and weights');seen.add(name);
 const jobs=[...pending.entries()];clickDuck();assert.deepEqual([...pending.entries()],jobs,'no extra timers or restart');
 if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 advance(avatar.durations[name]);assert.equal(duck.dataset.gesture,'none','one gesture only');
 assert.equal(duck.dataset.part,'none');
}
assert.deepEqual([...seen].sort(),[...Object.keys(avatar.weights),'racket','startle'].sort());
""")


def test_click_and_rest_share_nonrepeat_history_and_are_reproducible(avatar_ui):
    script = r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(31415);
const names=[];
for(let i=0;i<100;i++) {
 page('keydown');
 if(i%3===1) {
  const wait=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);advance(wait.at-now);
 } else clickDuck();
 const name=duck.dataset.next==='walk'?'walk':duck.dataset.gesture;
 assert.ok(Object.hasOwn(avatar.weights,name)||Object.hasOwn(avatar.clickWeights,name)||Object.hasOwn(avatar.prankWeights,name));
 assert.notEqual(name,names.at(-1),'clicks and rest exclude the previous selection');names.push(name);
 const jobs=[...pending.entries()];clickDuck();assert.deepEqual([...pending.entries()],jobs);
 if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 advance(avatar.durations[name]);assert.equal(duck.dataset.gesture,'none');
}
console.log(JSON.stringify(names));
"""
    assert avatar_ui(script) == avatar_ui(script)


@pytest.mark.parametrize('live', ['thinking', 'searching', 'reading', 'writing', 'remembering',
                                  'listening', 'speaking', 'processing'])
def test_click_ignored_in_every_live_state(avatar_ui, live):
    avatar_ui('const live=' + json.dumps(live) + r""";
document.body.classList.add('chat-view');advance(1800);avatar.setLive(live);
const before={...duck.dataset},jobs=[...pending.entries()];clickDuck();
assert.deepEqual({...duck.dataset},before);assert.deepEqual([...pending.entries()],jobs);
avatar.setLive('idle');advance(7999);assert.equal(duck.dataset.gesture,'none','no deferred click');
""")


@pytest.mark.parametrize('mode', ['body', 'media', 'hidden', 'preview', 'projects'])
def test_click_ignored_when_motion_unavailable_or_previewing(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='body')document.body.classList.add('reduced-motion');
if(mode==='media'){media.matches=true;media.change();}
if(mode==='hidden')visibility(true);
if(mode==='preview')core.preview('idle');
if(mode==='projects'){document.body.classList.remove('chat-view');document.body.classList.add('projects-view');}
const before={...duck.dataset},jobs=[...pending.entries()];clickDuck();
assert.deepEqual({...duck.dataset},before);assert.deepEqual([...pending.entries()],jobs);
advance(120000);assert.equal(duck.dataset.gesture,'none');
if(mode==='preview') {
 assert.ok(avatar.play('racket'),'lab preview gesture remains available');
 const playing=[...pending.entries()];clickDuck();assert.deepEqual([...pending.entries()],playing);
 advance(avatar.durations.racket);assert.equal(duck.dataset.gesture,'none');
}
""")


@pytest.mark.parametrize('kind', ['welcome', 'sent', 'demo'])
def test_click_ignored_during_existing_gesture_without_queue(avatar_ui, kind):
    avatar_ui('const kind=' + json.dumps(kind) + r""";
document.body.classList.add('chat-view');
if(kind!=='welcome') {
 advance(1800);
 if(kind==='sent')avatar.onSent('oi! ótimo');else avatar.play('flap');
}
advance(100);const before={...duck.dataset},jobs=[...pending.entries()];clickDuck();
assert.deepEqual({...duck.dataset},before);assert.deepEqual([...pending.entries()],jobs);
advance(avatar.durations[before.gesture]-100);
if(kind==='sent'){assert.equal(duck.dataset.gesture,'racket');advance(avatar.durations.racket);}
assert.equal(duck.dataset.gesture,'none');advance(7999);assert.equal(duck.dataset.gesture,'none');
""")


def test_click_restarts_rest_wait_only_after_gesture_finishes(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(42);
const old=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);advance(old.at-now-1);clickDuck();
const name=duck.dataset.next==='walk'?'walk':duck.dataset.gesture;assert.notEqual(name,'none');
assert.ok(![...pending.values()].some(j=>j.delay>=8000 && j.delay<=20000),'old rest timer cleared');
if(duck.dataset.next==='walk')advance(avatar.durations.turn);
advance(avatar.durations[name]-1);assert.equal(duck.dataset.gesture,name);
assert.ok(![...pending.values()].some(j=>j.delay>=8000 && j.delay<=20000));advance(1);
assert.equal(duck.dataset.gesture,'none');const next=[...pending.values()].filter(j=>j.delay>=8000 && j.delay<=20000);
assert.equal(next.length,1);assert.ok(next[0].delay<=20000);assert.equal(next[0].at,now+next[0].delay);
advance(next[0].delay-1);assert.equal(duck.dataset.gesture,'none');advance(1);
assert.notEqual(duck.dataset.gesture,'none');
""")


@pytest.mark.parametrize('mode', ['desktop', 'phone', 'pinned', 'narrow'])
def test_click_walk_keeps_band_and_turns_before_moving(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='phone') {
 window.innerWidth=375;geometry.stage=rect(18,58,339,340);geometry.duck=rect(42,90,96,96);
 geometry.orb=rect(106,214,164,164);geometry.label=rect(181,374,14,20);geometry.messages=rect(18,410,339,200);
}
if(mode==='pinned') {
 window.innerWidth=1728;geometry.stage=rect(292,64,1396,228);geometry.duck=rect(328,111.5,144,144);
 geometry.orb=rect(900,67,174,174);geometry.label=rect(980,237,14,20);geometry.messages=rect(555,374,870,374);
}
if(mode==='narrow')geometry.orb=rect(218,100,100,180);
avatar.play('turn');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'left');
// Seed 16: after scheduling a rest wait, the next weighted ticket selects walk.
avatar.setSeed(16);clickDuck();const band=avatar.walkingBand();
if(mode!=='narrow') {
 assert.equal(duck.dataset.gesture,'turn');assert.equal(duck.dataset.next,'walk');
 advance(avatar.durations.turn-1);assert.equal(duck.dataset.x,'0');advance(1);
 assert.equal(duck.dataset.facing,'right');
}
assert.equal(duck.dataset.gesture,'walk');let last=0;
const duration=avatar.durations.walk;
for(let elapsed=0;elapsed<duration;elapsed+=50) {
 advance(Math.min(50,duration-elapsed));const x=Number(duck.dataset.x);
 assert.equal(x,Math.round(x));assert.ok(x>=last && x<=band.max);assert.ok(x-last<=2);last=x;
}
assert.equal(duck.dataset.gesture,'none');assert.equal(mode==='narrow'?last===0:last>0,true);
""")


def test_attention_pointer_steps_expiry_cache_and_listening(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);
let reads=0;const box=duck.getBoundingClientRect;duck.getBoundingClientRect=()=>{reads++;return box();};resize();reads=0;
const eye=axis=>Number.parseFloat(duck.style['--gaze-'+axis]);
for(const [px,py,sx,sy] of [[900,180,1,0],[-900,180,-1,0],[108,-900,0,-1],[108,900,0,1]]) {
 pointer(px,py);assert.equal(Math.sign(eye('x')),sx);assert.equal(Math.sign(eye('y')),sy);
 assert.ok(Math.abs(eye('x'))<=1.5 && Math.abs(eye('y'))<=1.5);
}
assert.equal(reads,0,'pointer events and frames use cached geometry');
advance(7999);assert.equal(duck.dataset.attention,'pointer');advance(1);flushFrames();
assert.equal(eye('x'),0);assert.equal(eye('y'),0);assert.equal(duck.dataset.attention,'none');
assert.equal(duck.dataset.centering,'true');advance(240);assert.equal(duck.dataset.centering,'false');
core.setLive('listening');pointer(900,900);assert.equal(duck.dataset.attention,'pointer');assert.ok(eye('y')>0);
page('pointerout',{relatedTarget:null});flushFrames();assert.equal(eye('y'),0);
""")


def test_attention_turn_dwell_rate_limit_and_busy_exclusion(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);pointer(-900,180);
advance(1499);assert.equal(duck.dataset.gesture,'none');advance(1);assert.equal(duck.dataset.gesture,'turn');
advance(avatar.durations.turn);flushFrames();assert.equal(duck.dataset.facing,'left');
assert.ok(parseFloat(duck.style['--gaze-x'])>0,'local X reverses with the mirrored pose');pointer(900,180);
advance(1500);assert.equal(duck.dataset.gesture,'none');advance(4000-1500-avatar.durations.turn);
assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'right');
page('pointerleave');flushFrames();avatar.play('walk');pointer(-900,180);advance(1500);
assert.equal(duck.dataset.gesture,'walk');avatar.cancelGestures();avatar.play('racket');pointer(-800,180);
advance(1500);assert.equal(duck.dataset.gesture,'racket');
""")


def test_attention_touch_and_typing(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);
page('pointermove',{pointerType:'touch',clientX:900,clientY:900});flushFrames();assert.equal(duck.dataset.attention,'none');
page('pointerdown',{pointerType:'touch',clientX:900,clientY:900});flushFrames();assert.equal(duck.dataset.attention,'touch');
advance(1499);assert.equal(duck.dataset.attention,'touch');advance(1);flushFrames();assert.equal(duck.dataset.attention,'none');
page('pointerdown',{pointerType:'touch',target:duck,clientX:100,clientY:150});clickDuck();assert.notEqual(duck.dataset.gesture,'none');
avatar.cancelGestures();avatar.play('turn');advance(avatar.durations.turn);
focusInput();typeKey();assert.equal(duck.dataset.gesture,'turn');
advance(avatar.durations.turn);flushFrames();assert.equal(duck.dataset.attention,'typing');assert.equal(duck.dataset.facing,'right');assert.ok(parseFloat(duck.style['--gaze-y'])>0);
advance(1239);assert.equal(duck.dataset.attention,'typing');advance(1);flushFrames();assert.equal(duck.dataset.attention,'none');
typeKey();advance(1000);typeKey();advance(1999);assert.equal(duck.dataset.attention,'typing');
blurInput();assert.equal(duck.dataset.attention,'none');
""")


def test_attention_first_token_geometry_and_sent_queue(avatar_ui):
    avatar_ui(r"""
avatar.play('turn');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'left');
chat.submit('oi! ótimo');await settle();await emit({type:'token',text:'A'});
assert.equal(duck.dataset.gesture,'wave');advance(1800);assert.equal(duck.dataset.gesture,'racket');advance(2600);
assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);flushFrames();
assert.equal(duck.dataset.facing,'right');assert.equal(duck.dataset.attention,'reply');assert.ok(parseFloat(duck.style['--gaze-y'])>0);
advance(1999);assert.equal(duck.dataset.attention,'reply');advance(1);flushFrames();assert.equal(duck.dataset.attention,'none');
await emit({type:'token',text:'B'});flushFrames();assert.equal(duck.dataset.attention,'none','later tokens do not restart glance');
await emit({type:'final',turn});await close();
// Real rects can also put the column on the left (no hard-coded right side).
geometry.messages=rect(-1000,400,500,300);resize();core.setLive('speaking');
assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);assert.equal(duck.dataset.facing,'left');
""", chat=True)


@pytest.mark.parametrize('source', ['pointer', 'key', 'click', 'click-outside', 'touch', 'touch-outside', 'focus', 'sent', 'live', 'reply'])
def test_sleep_deadline_and_every_wake_source(avatar_ui, source):
    avatar_ui('const source=' + json.dumps(source) + r""";
document.body.classList.add('chat-view');advance(1800);
assert.equal(avatar.SLEEP_AFTER_MS,180000);assert.equal(Object.getOwnPropertyDescriptor(avatar,'SLEEP_AFTER_MS').writable,false);avatar.SLEEP_AFTER_MS=1;assert.equal(avatar.SLEEP_AFTER_MS,180000);
advance(avatar.SLEEP_AFTER_MS-1800-1);assert.equal(duck.dataset.asleep,'false');advance(1);
assert.equal(duck.dataset.asleep,'true');assert.equal(duck.dataset.eyes,'closed');assert.equal(duck.dataset.gesture,'none');
assert.equal(duck.querySelector('.duck-zzz').hidden,false);assert.equal(pending.size,0,'sleep cancels rest and eye clocks');
if(source==='pointer')pointer(900,900);
if(source==='key')page('keydown',{key:'x'});
if(source==='click')clickDuck();
if(source==='click-outside')page('click');
if(source==='touch-outside')page('pointerdown',{pointerType:'touch',clientX:900,clientY:900});
if(source==='touch'){page('pointerdown',{pointerType:'touch',target:duck});clickDuck();}
if(source==='focus')focusInput();
if(source==='sent')avatar.onSent('ordinary message');
if(source==='live')core.setLive('reading');
if(source==='reply')core.setLive('speaking');
assert.equal(duck.dataset.asleep,'false');assert.equal(duck.dataset.eyes,'open');assert.equal(duck.dataset.gesture,'wake');
assert.equal(duck.querySelector('.duck-zzz').hidden,true);advance(avatar.durations.wake-1);assert.equal(duck.dataset.gesture,'wake');advance(1);
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,source==='live'?'reading':source==='reply'?'speaking':'idle');
""")


@pytest.mark.parametrize('source', ['pointer', 'key', 'click', 'touch', 'focus', 'sent', 'stream'])
def test_sleep_activity_resets_clock_and_hidden_time_does_not_count(avatar_ui, source):
    avatar_ui('const source=' + json.dumps(source) + r""";
document.body.classList.add('chat-view');advance(100000);
if(source==='pointer')pointer(900,900);
if(source==='key')page('keydown',{key:'x'});
if(source==='click')page('click');
if(source==='touch')page('pointerdown',{pointerType:'touch',clientX:900,clientY:900});
if(source==='focus')focusInput();
if(source==='sent')avatar.onSent('ordinary message');
if(source==='stream'){core.setLive('speaking');advance(300000);assert.equal(duck.dataset.asleep,'false');core.setLive('idle');}
advance(179999);assert.equal(duck.dataset.asleep,'false');advance(1);assert.equal(duck.dataset.asleep,'true');
page('keydown');advance(900);visibility(true);advance(600000);assert.equal(pending.size,0);
visibility(false);advance(179999);assert.equal(duck.dataset.asleep,'false');advance(1);assert.equal(duck.dataset.asleep,'true');
""")


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_sleep_reduced_motion_and_preview_guard(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='body')document.body.classList.add('reduced-motion');else {media.matches=true;media.change();}
advance(178200);assert.equal(duck.dataset.asleep,'true');assert.equal(duck.dataset.eyes,'closed');
core.setLive('reading');assert.equal(duck.dataset.asleep,'false');assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'reading');
core.setLive('idle');core.preview('idle');advance(200000);assert.equal(duck.dataset.asleep,'false');
core.preview(null);document.body.classList.remove('chat-view');advance(200000);assert.equal(duck.dataset.asleep,'true','home also sleeps');
""")


def test_sleep_demo_parity_and_only_new_art_is_rect_cell_map(avatar_ui):
    from tests.ui_parity import GESTURE_SELECT
    option = '<option value="sleep">Dormir</option>'
    html = (ROOT / 'src/ui/templates/novo.html').read_text()
    assert option in GESTURE_SELECT and GESTURE_SELECT in html
    assert check(html.replace(option, ''))
    shell = (NOVO / 'shell.js').read_text()
    start = shell.index("$('#gesture-select').addEventListener")
    end = shell.index("$('#preview-chat').addEventListener", start)
    avatar_ui(r"""
const controls=new EventTarget();controls.open=true;
controls.close=()=>{controls.open=false;setTimeout(()=>controls.dispatchEvent(new Event('close')),0);};
const select=new EventTarget(),stateSelect={value:'idle'};
const $=s=>({'#controls':controls,'#gesture-select':select,'#state-select':stateSelect})[s];
""" + shell[start:end] + r"""
document.body.classList.add('chat-view');advance(1800);core.preview('thinking');
select.value='sleep';select.dispatchEvent(new Event('change'));assert.notEqual(duck.dataset.asleep,'true');
advance(0);assert.equal(duck.dataset.asleep,'true');assert.equal(controls.open,false);
const z=duck.querySelector('.duck-zzz');assert.equal(z.getAttribute('aria-hidden'),'true');
assert.equal(z.getAttribute('shape-rendering'),'crispEdges');assert.ok(z.children.length>20);
for(const cell of z.children) {
 assert.equal(cell.tag,'rect');assert.equal(cell.getAttribute('fill'),'rgb(129 249 251)');
 const x=Number(cell.getAttribute('x')),y=Number(cell.getAttribute('y')),size=Number(cell.getAttribute('width'));
 assert.ok(x>=0 && x+size<=1254);assert.ok(y>=-1254*.2 && y+size<24,'above head, inside reserved halo');
}
clickDuck();assert.equal(duck.dataset.gesture,'wake');
""")
    css = (NOVO / 'avatar.css').read_text()
    assert 'duck-sleep-breathe' in css and 'duck-zzz-float' in css
    assert 'transition: transform 240ms' in css
    assert '.duck-zzz' in css and 'pointer-events: none' in css
    assert '.duck[data-asleep="true"] .eye-patch' in css


@pytest.mark.parametrize('surface', ['tab', 'chat'])
def test_sleep_return_wakes_with_hop_and_live_latest_wins(avatar_ui, surface):
    avatar_ui('const surface=' + json.dumps(surface) + r""";
document.body.classList.add('chat-view');advance(1800);avatar.play('sleep');
if(surface==='tab')visibility(true);else document.body.classList.remove('chat-view');
advance(600000);assert.equal(duck.dataset.asleep,'true');assert.equal(pending.size,0);
if(surface==='tab')visibility(false);else {document.body.classList.add('chat-view');page('click');}
assert.equal(duck.dataset.gesture,'wake');core.setLive('reading');advance(200);core.setLive('writing');
assert.equal(duck.dataset.state,'idle');advance(399);assert.equal(duck.dataset.gesture,'wake');advance(1);
assert.equal(duck.dataset.state,'writing');assert.equal(duck.dataset.gesture,'none');
""")


def test_wake_then_sent_queue_then_first_reply_is_serial(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);avatar.play('sleep');avatar.onSent('oi! ótimo');
core.setLive('speaking');assert.equal(duck.dataset.gesture,'wake');advance(600);
assert.equal(duck.dataset.gesture,'wave');advance(1800);assert.equal(duck.dataset.gesture,'racket');advance(2600);
assert.equal(duck.dataset.state,'speaking');assert.equal(duck.dataset.gesture,'none');flushFrames();
assert.equal(duck.dataset.attention,'reply');advance(2000);flushFrames();assert.equal(duck.dataset.attention,'none');
""")


def test_attention_coalesces_pointer_events_and_updates_cached_rects(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);flushFrames();
let reads=0;const original=duck.getBoundingClientRect;duck.getBoundingClientRect=()=>{reads++;return original();};
const before=frames.size;
for(let i=0;i<100;i++)page('pointermove',{pointerType:'mouse',clientX:900,clientY:900});
assert.equal(reads,0);assert.equal(frames.size,before+1);flushFrames();assert.equal(reads,0);
// A scroll invalidates all cached targets, including the pinned gutter's column.
geometry.duck=rect(1000,112,144,144);
for(const callback of windowListeners.scroll||[])callback();flushFrames();
assert.ok(reads>0);assert.ok(parseFloat(duck.style['--gaze-x'])<0);
page('pointerleave');flushFrames();advance(240);avatar.play('walk');advance(50);
assert.ok(reads>1,'walking refreshes its cached box at each step');
""")


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_sleep_css_static_closed_eyes_and_z_halo_geometry(mode, avatar_ui):
    # Check the real CSS hooks and measured geometry for static sampled eyes/Z.
    css = (NOVO / 'avatar.css').read_text()
    assert '.duck[data-centering="true"][data-eyes="open"] .eye-copy' in css
    assert 'clip-path: var(--leftEye-clip)' in css and 'clip-path: var(--rightEye-clip)' in css
    assert 'translate(var(--gaze-x, 0px), var(--gaze-y, 0px))' in css
    avatar_ui('const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='body')document.body.classList.add('reduced-motion');else {media.matches=true;media.change();}
assert.ok(avatar.play('sleep'));clickDuck();assert.equal(duck.dataset.asleep,'false');assert.equal(duck.dataset.eyes,'open');
assert.equal(duck.dataset.gesture,'none');assert.ok(avatar.play('sleep'));
const z=duck.querySelector('.duck-zzz');
for(const size of [96,128,144])for(const cell of z.children) {
 const x=Number(cell.getAttribute('x'))*size/1254,y=Number(cell.getAttribute('y'))*size/1254;
 const w=Number(cell.getAttribute('width'))*size/1254;
 // The float's -5% is relative to the 125%-high SVG, not the duck.
 assert.ok(y-size*1.25*.05>=-size*.25);assert.ok(x>=0 && x+w<=size);assert.ok(y+w<0);
}
""")


def test_reply_gaze_uses_newest_bubble_and_does_not_restart_per_token(avatar_ui):
    avatar_ui(r"""
chat.submit('ordinary message');await settle();
const bubble=hosts['#messages'].querySelectorAll('.assistant').at(-1);
bubble.getBoundingClientRect=()=>rect(100,220,20,20);
await emit({type:'token',text:'A'});flushFrames();
assert.equal(duck.dataset.attention,'reply');assert.equal(parseFloat(duck.style['--gaze-x']),0);
assert.equal(parseFloat(duck.style['--gaze-y']),.8,'aim at the bubble rather than the column centre');
advance(1500);await emit({type:'token',text:'B'});advance(500);flushFrames();
assert.equal(duck.dataset.attention,'none');await emit({type:'final',turn});await close();
""", chat=True)


def test_typing_turn_uses_pointer_turn_cooldown(avatar_ui):
    avatar_ui(r"""
document.body.classList.add('chat-view');advance(1800);pointer(-900,180);advance(1500);
const turnedAt=now;advance(avatar.durations.turn);page('pointerleave');flushFrames();focusInput();typeKey();
assert.equal(duck.dataset.gesture,'none');advance(1500);typeKey();
advance(4000-(now-turnedAt)-1);typeKey();assert.equal(duck.dataset.gesture,'none');advance(1);
assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);flushFrames();assert.equal(duck.dataset.facing,'right');
assert.equal(duck.dataset.attention,'typing');blurInput();advance(4000);assert.equal(duck.dataset.gesture,'none');
""")


@pytest.mark.parametrize('width,size', [(1409, None), (1410, 120), (1440, 120), (1458, 144), (1728, 144), (1920, 144)])
def test_strip_live_geometry_clearance_and_size(avatar_ui, width, size):
    avatar_ui(f'const width={width},size={json.dumps(size)};' + r"""
measured(width,400);document.body.classList.add('chat-view');advance(2000);
const bands=avatar.stripGeometry();
if(size===null){assert.equal(bands.left,null);assert.equal(bands.right,null);assert.equal(duck.parentNode,dock);}
else {
 assert.equal(bands.left.size,size);assert.equal(bands.right.size,size);
 assertStrip('left');assert.equal(duck.getBoundingClientRect().width,size);
 assert.equal(bands.left.right-bands.left.left,(width-252-870)/2);
}
""")


def test_strip_scroll_drop_return_and_stage_walk_resume(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);
assert.equal(duck.parentNode,dock);measured(1440,47);scrollEvent();assert.equal(duck.parentNode,dock);
measured(1440,49);const before=duck.getBoundingClientRect();scrollEvent();
assert.equal(duck.dataset.travel,'enter');assert.deepEqual(duck.getBoundingClientRect(),before,'reparent without a jump');
advance(200);const middle=duck.getBoundingClientRect();assert.ok(middle.width>120&&middle.width<144);
advance(1800);assertStrip('left');assert.equal(duck.getBoundingClientRect().width,120);
const layer=duck.parentNode;assert.equal(layer.style.left,'252px');assert.equal(layer.style.top,'64px');
assert.equal(layer.style.width,'1188px');assert.equal(layer.style.height,'721px');
const fixed=duck.getBoundingClientRect();measured(1440,450);scrollEvent();assert.deepEqual(duck.getBoundingClientRect(),fixed);
measured(1440,0);scrollEvent();assert.equal(duck.dataset.travel,'return');
assert.deepEqual(duck.getBoundingClientRect(),fixed);advance(200);assert.ok(duck.getBoundingClientRect().width>120);
advance(6000);assert.equal(duck.parentNode,dock);assert.equal(duck.dataset.habitat,'stage');
assert.equal(duck.getBoundingClientRect().width,144);avatar.cancelGestures();avatar.setSeed(10);avatar.play('walk');
advance(1000);assert.ok(Number(duck.dataset.x)>0);
""")


def test_strip_right_only_vertical_walk_and_top_crossing(avatar_ui):
    avatar_ui(r"""
measured(1440,400,390,1281);document.body.classList.add('chat-view');advance(2000);assertStrip('right');
measured(1440,400);resize();advance(2000);assertStrip('right');
const facing=duck.dataset.facing;let ys=new Set();
for(let i=0;i<12;i++) {
 avatar.cancelGestures();avatar.setSeed(i);avatar.play('walk');
 for(let t=0;t<6000;t+=50){advance(50);assertStrip('right');ys.add(duck.getBoundingClientRect().top);}
 assert.equal(duck.dataset.facing,facing,'vertical travel preserves facing');page('keydown');
}
assert.ok(ys.size>20);avatar.cancelGestures();assert.ok(avatar.play('cross'));
let last=duck.getBoundingClientRect(),hiddenHorizontal=false;
for(let t=0;t<12000&&duck.dataset.travel==='cross';t+=20) {
 advance(20);const b=duck.getBoundingClientRect();
 if(b.left!==last.left){assert.ok(b.bottom<=geometry.scroll.top);hiddenHorizontal=true;}
 if(b.bottom>geometry.scroll.top)assert.ok(b.right<=geometry.messages.left-12||b.left>=geometry.messages.right+12);
 last=b;
}
assert.ok(hiddenHorizontal);assertStrip('left');
""")


def test_strip_search_queue_click_attention_sleep_and_preview(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
avatar.setLive('searching');advance(800);const head=avatar.createHead('searching');
assert.equal(duck.dataset.state,'searching');assert.ok(!canvasOf(duck)||canvasOf(duck).hidden);
assert.ok(canvasOf(head)&&!canvasOf(head).hidden);assert.ok(canvasOf(head).context.marks.length);
avatar.onSent('oi! ótimo');avatar.setLive('reading');assert.equal(duck.dataset.gesture,'wave');
advance(1800);assert.equal(duck.dataset.gesture,'racket');advance(3400);assert.equal(duck.dataset.state,'reading');assertStrip('left');
avatar.setLive('idle');avatar.setSeed(0);clickDuck();assert.notEqual(duck.dataset.gesture,'none');avatar.cancelGestures();
focusInput();typeKey();flushFrames();assert.equal(duck.dataset.attention,'typing');blurInput();
avatar.setLive('speaking');advance(800);flushFrames();assert.equal(duck.dataset.attention,'reply');
avatar.setLive('idle');avatar.play('sleep');assert.equal(duck.dataset.asleep,'true');assert.ok(!duck.querySelector('.duck-zzz').hidden);assertStrip('left');
clickDuck();advance(600);assert.equal(duck.dataset.asleep,'false');
avatar.preview('searching');assert.equal(duck.dataset.state,'searching');assert.ok(!canvasOf(duck)||canvasOf(duck).hidden);
assert.equal(avatar.randomGestureNow(),false);avatar.preview(null);assert.equal(duck.dataset.state,'idle');
""")


@pytest.mark.parametrize('mode', ['body', 'media', 'hidden'])
def test_strip_motion_preferences_and_hidden_cancel_travel(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
avatar.play('cross');advance(100);
if(mode==='hidden')visibility(true);
else if(mode==='body')document.body.classList.add('reduced-motion');
else {media.matches=true;media.change();}
assert.equal(duck.dataset.travel,'none');assert.equal(duck.dataset.gesture,'none');
const still=duck.getBoundingClientRect();advance(10000);assert.deepEqual(duck.getBoundingClientRect(),still);
if(mode==='hidden'){assert.equal(pending.size,0);visibility(false);}
else {
 measured(1440,0);scrollEvent();assert.equal(duck.parentNode,dock);assert.equal(duck.dataset.travel,'none');
 measured(1440,100);scrollEvent();assertStrip('left');assert.equal(duck.getBoundingClientRect().width,120);
 assert.equal(duck.dataset.travel,'none');assert.equal(avatar.play('walk'),false);
}
""")


def test_strip_resize_column_changes_and_shrink_recovery(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
measured(1458,100);const before=duck.getBoundingClientRect();resize();assert.deepEqual(duck.getBoundingClientRect(),before);
advance(200);assert.ok(duck.getBoundingClientRect().width>120&&duck.getBoundingClientRect().width<144);
advance(2000);assertStrip('left');assert.equal(duck.getBoundingClientRect().width,144);
measured(1409,100);const old=duck.getBoundingClientRect();resize();assert.equal(duck.dataset.travel,'return');
assert.deepEqual(duck.getBoundingClientRect(),old,'resize recovery has no teleport');advance(6000);assert.equal(duck.parentNode,dock);
measured(1440,100,400,1270);resize();advance(2000);assertStrip('left');
assert.equal(duck.getBoundingClientRect().width,120);
// Sidebar movement and a message-column mutation use real rects, not viewport thresholds.
geometry.scroll=rect(0,64,1440,721);geometry.messages=rect(285,0,870,900);resize();advance(2000);
assertStrip('left');assert.equal(duck.getBoundingClientRect().width,144);
document.body.classList.remove('chat-view');assert.equal(duck.parentNode,dock);
""")


def test_strip_layer_and_zzz_css_safety():
    css = (NOVO / 'avatar.css').read_text()
    rules = css_rules(css, 1440, 900)
    layer = {k: v for name, props in rules if name == '.duck-strip-layer' for k, v in props.items()}
    assert layer['position'] == 'fixed'
    assert layer['overflow'] == 'hidden'
    assert layer['pointer-events'] == 'none'
    assert '.duck[data-habitat="left"] .duck-zzz' in css
    assert '.duck[data-habitat="right"] .duck-zzz' in css


@pytest.mark.parametrize('kind', ['enter', 'return', 'resize'])
@pytest.mark.parametrize('mode', ['body', 'media', 'hidden'])
def test_strip_interrupt_geometry_animation_is_immediate(avatar_ui, kind, mode):
    avatar_ui('const kind=' + json.dumps(kind) + ',mode=' + json.dumps(mode) + r""";
measured();document.body.classList.add('chat-view');advance(1800);
if(kind==='enter'){measured(1440,100);scrollEvent();}
else {
 settledStrip();
 if(kind==='return'){measured(1440,0);scrollEvent();}
 else {measured(1458,100);resize();}
}
advance(100);assert.equal(duck.dataset.travel,kind);
if(mode==='hidden')visibility(true);
else if(mode==='body')document.body.classList.add('reduced-motion');
else {media.matches=true;media.change();}
assert.equal(duck.dataset.travel,'none');
const still=duck.getBoundingClientRect();advance(3000);assert.deepEqual(duck.getBoundingClientRect(),still);
if(mode==='hidden')assert.equal(pending.size,0);
if(kind==='return')assert.equal(duck.parentNode,dock);else assertStrip('left');
""")


def test_strip_return_follows_moving_anchor_without_last_frame_jump(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
assertStrip('left');
measured(1440,40);scrollEvent();advance(300);
measured(1440,0);scrollEvent();advance(280);const before=duck.getBoundingClientRect();advance(20);
assert.equal(duck.parentNode,dock);const after=duck.getBoundingClientRect();
assert.ok(Math.abs(after.top-before.top)<=4,JSON.stringify({before,after}));
""")


def test_strip_right_only_entry_and_lost_side_do_not_cross_visible_column(avatar_ui):
    avatar_ui(r"""
measured(1440,0,390,1281);document.body.classList.add('chat-view');advance(1800);
measured(1440,49,390,1281);scrollEvent();let last=duck.getBoundingClientRect(),crossed=false;
for(let t=0;t<3000;t+=20) {
 advance(20);const b=duck.getBoundingClientRect();
 if(b.left>last.left&&b.left<1281&&b.right>geometry.orb.left){assert.ok(b.bottom<=geometry.scroll.top);crossed=true;}
 last=b;
}
assert.ok(crossed);assertStrip('right');
// Losing the occupied strip returns to the dock, even if the other side fits.
measured(1440,100,411,1310);resize();assert.equal(duck.dataset.travel,'return');
advance(3000);assert.equal(duck.parentNode,dock);
""")


@pytest.mark.parametrize('phase', [100, 1000, 4000, 5000])
def test_strip_crossing_interrupted_by_live_or_resized_geometry(avatar_ui, phase):
    avatar_ui('const phase=' + str(phase) + r""";
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
avatar.play('cross');advance(phase);avatar.setLive('searching');advance(2000);
const side=duck.dataset.habitat;assertStrip(side);
assert.equal(duck.dataset.state,'searching');assert.equal(duck.dataset.facing,side==='left'?'right':'left');
assert.ok(!canvasOf(duck)||canvasOf(duck).hidden);
avatar.setLive('idle');advance(1000);avatar.play('cross');advance(phase);
measured(1409,400);resize();assert.equal(duck.dataset.travel,'return');advance(6000);assert.equal(duck.parentNode,dock);
""")


def test_strip_random_crossing_is_rare_and_vertical_walks_go_both_ways(avatar_ui):
    avatar_ui(r"""
measured(1728,400);document.body.classList.add('chat-view');advance(1800);
const counts={},directions=new Set();
for(let seed=0;seed<160;seed++) {
 avatar.cancelGestures();advance(1000);page('keydown');avatar.setSeed(seed);avatar.randomGestureNow();
 const name=duck.dataset.gesture;counts[name]=(counts[name]||0)+1;
 const before=duck.getBoundingClientRect().top;advance(avatar.durations[name]+1000);
 if(name==='walk')directions.add(Math.sign(duck.getBoundingClientRect().top-before));
 assertStrip(duck.dataset.habitat);
}
assert.ok(counts.cross>0&&counts.cross<counts.walk/2,JSON.stringify(counts));
assert.ok(directions.has(-1)&&directions.has(1));
""")


def test_strip_exact_fractional_threshold_still_fits_144(avatar_ui):
    avatar_ui(r"""
measured(1458,400);geometry.scroll=rect(252.5,64,1206,721);geometry.messages=rect(420.5,-26,870,900);
document.body.classList.add('chat-view');advance(2000);
assert.equal(avatar.stripGeometry().left.size,144);const b=duck.getBoundingClientRect();
assert.equal(b.width,144);assert.ok(b.left>=264.5&&b.right<=408.5);
""")


@pytest.mark.parametrize('mode', ['normal', 'reduced'])
def test_strip_existing_live_state_faces_inner_side_after_right_entry(avatar_ui, mode):
    avatar_ui('const mode=' + json.dumps(mode) + r""";
measured(1440,0,390,1281);document.body.classList.add('chat-view');advance(1800);
if(mode==='reduced')document.body.classList.add('reduced-motion');
avatar.setLive('reading');measured(1440,49,390,1281);scrollEvent();advance(3000);
assertStrip('right');assert.equal(duck.dataset.state,'reading');assert.equal(duck.dataset.facing,'left');
measured(1440,0,390,1281);scrollEvent();advance(3000);
assert.equal(duck.parentNode,dock);assert.equal(duck.dataset.state,'reading');assert.equal(duck.dataset.facing,'right');
""")


def test_strip_crossing_remeasures_both_sides_on_resize(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
avatar.play('cross');advance(1300);
measured(1728,400);resize();advance(10000);assertStrip(duck.dataset.habitat);
""")


def test_strip_observes_column_and_sidebar_attribute_changes(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
geometry.messages=rect(430,274,830,900);
const changes=observers.filter(o=>o.target===hosts['#messages']);assert.ok(changes.length);
changes.forEach(o=>o.callback());advance(2000);assertStrip('left');assert.equal(duck.getBoundingClientRect().width,144);
""")


def test_strip_right_return_uses_clipped_top_instead_of_crossing_the_orb(avatar_ui):
    avatar_ui(r"""
measured(1440,100,390,1281);document.body.classList.add('chat-view');advance(2000);assertStrip('right');
measured(1440,0,390,1281);scrollEvent();let moved=false,last=duck.getBoundingClientRect();
for(let t=0;t<2000;t+=20) {
 advance(20);const b=duck.getBoundingClientRect();
 if(b.left!==last.left){assert.ok(b.bottom<=geometry.scroll.top,'horizontal return is fully clipped');moved=true;}
 last=b;
}
assert.ok(moved);assert.equal(duck.parentNode,dock);
""")


def test_strip_entry_during_stage_walk_turn_never_moves_empty_dock(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);
avatar.play('turn');advance(avatar.durations.turn);avatar.setSeed(10);avatar.play('walk');
assert.equal(duck.dataset.next,'walk');advance(200);measured(1440,100);scrollEvent();advance(6000);
assertStrip('left');assert.equal(duck.dataset.x,'0','a stage walk continuation cannot move the empty dock');
assert.ok([...pending.values()].some(j=>j.delay>=8000&&j.delay<=20000),'rest scheduler resumes');
""")


def test_strip_return_cancels_crossing_pose_and_resumes_stage_life(avatar_ui):
    avatar_ui(r"""
measured();document.body.classList.add('chat-view');advance(1800);settledStrip();
assertStrip('left');
avatar.play('cross');advance(100);measured(1440,0);scrollEvent();advance(2000);
assert.equal(duck.parentNode,dock);assert.equal(duck.dataset.gesture,'none');
assert.ok([...pending.values()].some(j=>j.delay>=8000&&j.delay<=20000));
""")


@pytest.mark.parametrize('width,size', [(1440, 144), (1728, 144), (375, 96)])
def test_home_first_shown_welcome_and_zones(avatar_ui, width, size):
    avatar_ui(r"""
assert.equal(duck.dataset.paused,'false');assert.equal(duck.dataset.gesture,'wave');
assert.equal(duck.getBoundingClientRect().width,SIZE);homeSafe();advance(1800);
const g=avatar.homeGeometry();assert.ok(rangeReads>0,'measure text with Range');
assert.ok(g.zones.some(z=>z.name==='stage-left'));assert.equal(g.size,SIZE);
if(SIZE===144) {
 assert.deepEqual([...new Set(g.zones.map(z=>z.name))].sort(),['bottom','left-column','right-column','stage-left','stage-right']);
 assert.ok(g.zones.find(z=>z.name==='left-column').maxX>geometry.stage.left+100);
} else assert.deepEqual(g.zones.map(z=>z.name),['stage-left']);
const facing=duck.dataset.facing;
measured(WIDTH);document.body.classList.add('chat-view');
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.travel,'none');assert.equal(duck.dataset.facing,facing);
measuredHome(WIDTH);document.body.classList.remove('chat-view');
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');homeSafe();
""".replace('SIZE', str(size)).replace('WIDTH', str(width)), home=True, width=width)


def test_home_css_visible_and_only_duck_accepts_hits():
    css = (NOVO / 'avatar.css').read_text()
    rules = css_rules(css, 1440, 900)
    dock = {k: v for name, props in rules if name == '.duck-dock' for k, v in props.items()}
    assert dock['display'] == 'block'
    assert dock['pointer-events'] == 'none'
    assert 'pointer-events: auto' in css


@pytest.mark.parametrize('width', [1440, 1728, 375])
def test_home_walk_and_routes_never_cross_obstacles(avatar_ui, width):
    avatar_ui(r"""
advance(1800);const g=avatar.homeGeometry(),start=duck.getBoundingClientRect();
for(const zone of g.zones) {
 const target={left:Math.round((zone.minX+zone.maxX)/2),top:Math.round((zone.minY+zone.maxY)/2)};
 const path=avatar.homePath(start,target,g);assert.ok(path.length,zone.name);
 let from=start;
 for(const to of path) {
  assert.ok(from.left===to.left||from.top===to.top,'only rectilinear legs');
  for(let t=0;t<=1;t+=.05) {
   const p={left:from.left+(to.left-from.left)*t,top:from.top+(to.top-from.top)*t};
   const halo=rect(p.left,p.top-g.haloY,g.size,g.size+g.haloY);
   for(const o of [geometry.orb,geometry.label,geometry.modes,...homeText.flatMap(n=>n.textRects)])
    assert.ok(halo.right+12<=o.left+.001||halo.left-12>=o.right-.001||halo.bottom+12<=o.top+.001||halo.top-12>=o.bottom-.001);
  }from=to;
 }
}
let moved=false,vertical=false,seen=new Set();
for(let seed=1;seed<=65;seed++) {
 avatar.cancelGestures();page('keydown');avatar.setSeed(seed);assert.ok(avatar.play('walk'));
 for(let i=0;i<300&&duck.dataset.gesture!=='none';i++) {
  const b=duck.getBoundingClientRect();advance(20);homeSafe();
  const a=duck.getBoundingClientRect();moved ||=a.left!==b.left;vertical ||=a.top!==b.top;
  for(const z of avatar.homeGeometry().zones)if(a.left>=z.minX&&a.left<=z.maxX&&a.top>=z.minY&&a.top<=z.maxY)seen.add(z.name);
 }
}
assert.ok(moved);if(WIDTH>800){assert.ok(vertical);assert.ok(seen.has('bottom'));assert.ok(seen.has('stage-right'));}
""".replace('WIDTH', str(width)), home=True, width=width)


def test_home_disconnected_zones_and_live_resize_cancel_route(avatar_ui):
    avatar_ui(r"""
advance(1800);
// The modes become a wall: both bottom and stage zones still exist, but there
// is no duck-width column connecting them. Never jump through that wall.
geometry.modes=rect(geometry.stage.left,475,geometry.stage.width,42);resize();
let g=avatar.homeGeometry(),bottom=g.zones.find(z=>z.name==='bottom');assert.ok(bottom);
assert.deepEqual(avatar.homePath(duck.getBoundingClientRect(),{left:bottom.minX,top:bottom.minY},g),[]);
const initial=duck.getBoundingClientRect();
for(let seed=1;seed<20;seed++){avatar.cancelGestures();avatar.setSeed(seed);avatar.play('walk');advance(6000);homeSafe();assert.equal(duck.getBoundingClientRect().top,initial.top);page('keydown');}
measuredHome();resize();avatar.cancelGestures();avatar.setSeed(3);avatar.play('walk');advance(200);
geometry.modes=rect(geometry.stage.left,260,geometry.stage.width,42);resize();
for(let i=0;i<400;i++){advance(20);homeSafe();}
""", home=True)


def test_home_attention_click_sleep_preview_and_hidden(avatar_ui):
    avatar_ui(r"""
advance(1800);pointer(1100,130);assert.equal(duck.dataset.attention,'pointer');
pointer(270,130);advance(1500);assert.equal(duck.dataset.gesture,'turn');advance(avatar.durations.turn);
assert.equal(duck.dataset.facing,'left');focusInput();typeKey();advance(4000);flushFrames();
avatar.cancelGestures();typeKey();advance(avatar.durations.turn);typeKey();flushFrames();assert.equal(duck.dataset.attention,'typing');
blurInput();page('pointerleave');flushFrames();avatar.cancelGestures();clickDuck();assert.notEqual(duck.dataset.gesture,'none');
avatar.cancelGestures();core.preview('searching');homeSafe();assert.equal(duck.dataset.state,'searching');
assert.equal(canvasOf(duck).hidden,false);assert.equal(avatar.randomGestureNow(),false);
core.preview(null);page('keydown');advance(180000);assert.equal(duck.dataset.asleep,'true');homeSafe();
assert.equal(duck.querySelector('.duck-zzz').hidden,false);clickDuck();assert.equal(duck.dataset.gesture,'wake');advance(600);
avatar.play('walk');advance(200);visibility(true);const b=duck.getBoundingClientRect();advance(300000);
assert.deepEqual(duck.getBoundingClientRect(),b);assert.equal(pending.size,0);assert.equal(duck.dataset.paused,'true');
visibility(false);assert.equal(duck.dataset.paused,'false');assert.equal(duck.dataset.gesture,'none');homeSafe();
""", home=True)


@pytest.mark.parametrize('mode', ['body', 'media'])
def test_home_reduced_motion_and_switches_cancel_all_travel(avatar_ui, mode):
    avatar_ui('const mode='+json.dumps(mode)+r""";
advance(1800);avatar.play('turn');advance(avatar.durations.turn);const facing=duck.dataset.facing;
avatar.setSeed(3);avatar.play('walk');advance(200);
if(mode==='body')document.body.classList.add('reduced-motion');else {media.matches=true;media.change();}
const b=duck.getBoundingClientRect();assert.equal(avatar.play('walk'),false);advance(10000);assert.deepEqual(duck.getBoundingClientRect(),b);
measured(1728);document.body.classList.add('chat-view');assertStrip('left');
assert.equal(duck.dataset.facing,facing);assert.equal(duck.dataset.travel,'none');assert.equal(duck.dataset.gesture,'none');
measuredHome();document.body.classList.remove('chat-view');assert.equal(duck.parentNode,dock);
assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');assert.equal(duck.dataset.facing,facing);homeSafe();
""", home=True)


def test_home_to_chat_during_walk_places_immediately_and_keeps_facing(avatar_ui):
    avatar_ui(r"""
advance(1800);avatar.play('turn');advance(avatar.durations.turn);avatar.setSeed(2);avatar.play('walk');advance(100);
const facing=duck.dataset.facing;measured(1728);document.body.classList.add('chat-view');
assertStrip('left');assert.equal(duck.dataset.travel,'none');assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.facing,facing);
advance(7000);assertStrip('left');
measuredHome();document.body.classList.remove('chat-view');assert.equal(duck.parentNode,dock);
assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');homeSafe();
""", home=True)


@pytest.mark.parametrize('width', [375, 513, 560, 600, 800, 801, 1100, 1101, 1440, 1728])
@pytest.mark.parametrize('height', [667, 900, 1000])
def test_home_css_initial_row_clears_larger_sphere(width, height):
    rules = css_rules((NOVO / 'style.css').read_text() + '\n' + (NOVO / 'avatar.css').read_text(), width, height)
    def rule(selector):
        return {k: v for name, props in rules if name == selector for k, v in props.items()}
    px = lambda value: float(value.removesuffix('px'))
    size = px(rule('body')['--duck-size'])
    dock = rule('.duck-dock')
    stage_width = width - (252 if width > 1100 else 220 if width > 800 else 0) - (80 if width > 1100 else 50 if width > 800 else 36)
    orb_size = 260 if width <= 800 else 350 if height >= 950 else 230 if height <= 790 else max(255, min(height * .28, 330))
    orb_left = (stage_width - orb_size) / 2
    orb_top = px(rule('.stage')['padding-top'])
    left, top = px(dock['left']), px(dock['top'])
    assert size == (96 if width <= 800 else 144)
    assert top - size / 4 >= 12
    # Searching expands horizontally; homeGeometry can move to another free
    # point for that preview. The first idle/welcome pose must already fit.
    assert left + size + 12 <= orb_left or top + size + 12 <= orb_top


def test_home_text_mutations_controls_and_turn_route_invalidation(avatar_ui):
    avatar_ui(r"""
advance(1800);
// A changed text line may be far wider than the title; Range, not the full-width
// paragraph element, determines whether the left connection still exists.
homeText[2].textRects=[rect(304,436,1084,19)];
for(const o of observers)if(o.target===hosts['.welcome'])o.callback();
let g=avatar.homeGeometry();assert.ok(!g.zones.some(z=>z.name==='left-column'));
const bottom=g.zones.find(z=>z.name==='bottom');assert.ok(bottom);
assert.deepEqual(avatar.homePath(duck.getBoundingClientRect(),{left:bottom.minX,top:bottom.minY},g),[]);
measuredHome();resize();
// Shell creates this chooser after avatar.js has loaded. Discover it through
// the scroll host's child-list observer, then observe its visibility/size.
const chooser=make('div','create-chooser');hosts['.create-chooser']=chooser;
chooser.getBoundingClientRect=()=>rect(292,529,1108,55);
for(const o of [...observers])if(o.target===hosts['#content-scroll'])o.callback();
g=avatar.homeGeometry();
const target={left:328,top:600};
assert.deepEqual(avatar.homePath(duck.getBoundingClientRect(),target,g),[]);
chooser.hidden=true;
for(const o of [...observers])if(o.target===chooser)o.callback();
assert.ok(avatar.homePath(duck.getBoundingClientRect(),target).length);
// Invalidate a pending turn as well as an active walking step.
avatar.play('turn');advance(avatar.durations.turn);avatar.setSeed(10);avatar.play('walk');advance(20);
assert.equal(duck.dataset.gesture,'turn');
geometry.modes=rect(292,260,1108,42);resize();
assert.equal(duck.dataset.gesture,'none');advance(6000);homeSafe();
""", home=True)


def test_home_real_first_submit_and_open_cancel_old_position(avatar_ui):
    avatar_ui(r"""
const chat=window.cortexConversa;
chat.init(()=>{measured();document.body.classList.add('chat-view');});
advance(1800);avatar.setSeed(10);avatar.play('walk');advance(200);
const facing=duck.dataset.facing;chat.submit('ordinary message');await settle();
assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');assert.equal(duck.dataset.travel,'none');
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.facing,facing);
await emit({type:'final',turn});await close();
measuredHome();document.body.classList.remove('chat-view');homeSafe();
assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');
await chat.open('other');assert.equal(duck.dataset.travel,'none');assert.equal(duck.dataset.gesture,'none');
assert.equal(duck.dataset.x,'0');assert.equal(duck.dataset.y,'0');
""", home=True)


NEW_MOVES = {'dance': 'limbs', 'peek': 'none', 'cheer': 'limbs', 'no': 'head', 'startle': 'limbs'}


@pytest.mark.parametrize('name,part', NEW_MOVES.items())
def test_cheeky_moves_play_in_place_and_end_clean(avatar_ui, name, part):
    """Task 157: each new move is one bounded gesture that never walks, turns or leaves a part cut out."""
    avatar_ui('const name=' + json.dumps(name) + ', part=' + json.dumps(part) + r""";
document.body.classList.add('chat-view');advance(avatar.durations.wave);
const duration=avatar.durations[name];
assert.ok(duration>=600&&duration<=3000,'a short, bounded move');
const before={x:duck.dataset.x,y:duck.dataset.y,facing:duck.dataset.facing};
assert.equal(avatar.play(name),true);
assert.equal(duck.dataset.gesture,name);assert.equal(duck.dataset.part,part);
assert.equal(avatar.play(name),false,'no restart while it plays');
advance(duration-1);assert.equal(duck.dataset.gesture,name);
assert.deepEqual({x:duck.dataset.x,y:duck.dataset.y,facing:duck.dataset.facing},before,'stays in place');
advance(1);
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.part,'none');assert.equal(duck.dataset.eyes,'open');
// A turn interrupts it at once and nothing of it survives.
assert.equal(avatar.play(name),true);core.setLive('thinking');
if(duck.dataset.gesture==='turn')advance(avatar.durations.turn);
assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.state,'thinking');
core.setLive('idle');
// Reduced motion and sleep refuse it.
document.body.classList.add('reduced-motion');
assert.equal(avatar.play(name),false);assert.equal(duck.dataset.gesture,'none');
""")


def test_cheeky_moves_are_css_only_and_listed_for_preview():
    css = (NOVO / 'avatar.css').read_text()
    script = (NOVO / 'avatar.js').read_text()
    html = (ROOT / 'src/ui/templates/novo.html').read_text()
    tracks = set(re.findall(r'@keyframes ([\w-]+)', css))
    for name, track in (('dance', 'gesture-dance'), ('peek', 'gesture-peek'), ('cheer', 'gesture-cheer'),
                        ('no', 'head-no'), ('startle', 'gesture-startle')):
        assert track in tracks
        assert re.search(rf'\.duck\[data-gesture="{name}"\] \.duck-(motion|face) \{{ animation: {track} var\(--duration-{name}\)', css)
        assert f'<option value="{name}">' in html
    # The hiding ledge is a clip on the pose: the PNG layers and the rain are untouched.
    assert '.duck[data-gesture="peek"] .duck-pose { clip-path: inset(-40% -40% 7% -40%); }' in css
    # The moves sit before the reduced-motion rules, which therefore still win.
    assert css.index('@keyframes gesture-startle') < css.index('.duck[data-reduced="true"] *, body.reduced-motion .duck *')
    # The startle only ever comes from a click; the rest pool does not have it.
    assert "const CLICK_WEIGHTS = Object.freeze({racket: 8, startle: 4});" in script
    assert 'startle' not in re.search(r'const WEIGHTS = Object\.freeze\(\{(.*?)\}\);', script, re.S)[1]
    # No new artwork: still one PNG, referenced by the same URL only.
    assert set(re.findall(r"[\w/.-]+\.(?:png|webp|jpe?g|gif|svg)", css + script)) == {'/ui/static/novo/avatar/duckcortex.png'}


def test_rest_pose_breathes_and_yields_to_gestures_sleep_and_reduced_motion():
    """Task 159: breathing is the rest animation of the body only; every older override still wins."""
    css = (NOVO / 'avatar.css').read_text()
    rest = '.duck-body { transform-origin: 50% 91%; animation: duck-breathe 3.6s ease-in-out infinite; }'
    assert css.count(rest) == 1 and 'animation: duck-float' not in css
    track = re.search(r'@keyframes duck-breathe \{(.*)\}', css)[1]
    # A gentle change: under 3% of the figure, and it returns to the exact rest pose.
    assert '0%, 100% { transform: scale(1, 1); }' in track
    assert all(abs(float(n) - 1) <= .03 for n in re.findall(r'scale\(([\d.]+), ([\d.]+)\)', track)[1])
    for later in ('.duck:not([data-gesture="none"]) .duck-body { animation: none; }',
                  '.duck[data-asleep="true"] .duck-body { animation: duck-sleep-breathe',
                  '.duck[data-reduced="true"] *, body.reduced-motion .duck * { animation: none !important;'):
        assert css.index(rest) < css.index(later)


FOOTPRINTS = r"""
const stage=hosts['.stage'];
const prints=()=>stage.children.filter(n=>n.className==='duck-prop duck-prop-print');
const known=new Set();
function walkAndWatch() {
 assert.equal(avatar.play('walk'),true);
 if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 const before=known.size;let most=0;
 for(let t=0;t<avatar.durations.walk;t+=50) {
  advance(50);const b=duck.getBoundingClientRect(),s=geometry.stage;
  for(const n of prints()) {
   if(known.has(n))continue;known.add(n);
   // New art only: a tiny rect map in the duck's own sampled yellow, never an image.
   assert.equal(n.tag,'svg');assert.equal(n.getAttribute('aria-hidden'),'true');
   assert.equal(n.getAttribute('shape-rendering'),'crispEdges');assert.ok(n.children.length>=6);
   for(const cell of n.children){assert.equal(cell.tag,'rect');assert.equal(cell.getAttribute('fill'),'rgb(253 239 87)');}
   // Dropped inside the box the duck stands in at that moment.
   const left=s.left+parseFloat(n.style.left),top=s.top+parseFloat(n.style.top);
   const w=parseFloat(n.style.width),h=parseFloat(n.style.height);
   assert.ok(left>=b.left&&left+w<=b.right&&top>=b.top&&top+h<=b.bottom,JSON.stringify({left,top,w,h,b}));
   for(const v of [left,top])assert.equal(v,Math.round(v));
  }
  most=Math.max(most,prints().length);
 }
 return {made:known.size-before,most};
}
"""


def test_walking_leaves_fading_footprints_inside_the_ducks_own_box(avatar_ui):
    """Task 160: prints appear while walking, fade by themselves and vanish on any layout change."""
    avatar_ui(FOOTPRINTS + r"""
advance(1800);assert.equal(prints().length,0);
// Home strolls are short: three of them together leave a visible trail.
let made=0,most=0;
for(let i=0;i<3;i++){avatar.cancelGestures();const walk=walkAndWatch();made+=walk.made;most=Math.max(most,walk.most);}
assert.ok(made>=3,'a visible trail: '+made);assert.ok(most<=24,'bounded');
assert.ok(prints().length>0);
advance(5000);assert.ok(prints().every(n=>n.dataset.gone==='true'),'all fading after 5 s');
advance(900);assert.equal(prints().length,0,'and removed');
// A layout change wipes the trail at once.
avatar.cancelGestures();walkAndWatch();assert.ok(prints().length>0);
resize();assert.equal(prints().length,0);
// So does leaving the screen they were drawn on.
avatar.cancelGestures();walkAndWatch();assert.ok(prints().length>0);
document.body.classList.add('projects-view');assert.equal(prints().length,0);
""", home=True)


@pytest.mark.parametrize('mode', ['reduced', 'pinned', 'strip'])
def test_no_footprints_without_motion_or_off_the_stage(avatar_ui, mode):
    avatar_ui(FOOTPRINTS + 'const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='reduced'){document.body.classList.add('reduced-motion');assert.equal(avatar.play('walk'),false);}
if(mode==='pinned'){measured(1728);resize();avatar.play('walk');advance(avatar.durations.turn+avatar.durations.walk);}
if(mode==='strip'){settledStrip();assert.notEqual(duck.dataset.habitat,'stage');avatar.play('walk');advance(avatar.durations.walk);}
assert.equal(prints().length,0);
""")


SAY = r"""
// Give the bubble a real size, so that placement is measured like in a browser.
const createPlain=document.createElement;
document.createElement=tag=>{const node=createPlain(tag);
 if(tag==='div')node.getBoundingClientRect=function(){
  if(this.className!=='duck-say')return rect(0,0,0,0);
  const b=duck.getBoundingClientRect();
  return rect(b.left+parseFloat(this.style.left||0),b.top+parseFloat(this.style.top||0),130,32);};
 return node;};
const bubble=()=>duck.children.find(n=>n.className==='duck-say');
const clear=(r,o)=>r.right+8<=o.left||r.left-8>=o.right||r.bottom+8<=o.top||r.top-8>=o.bottom;
"""


def test_thanks_detection(avatar_ui):
    avatar_ui(r"""
for(const text of ['obrigado','Muito obrigada!','valeu, ficou bom','vlw','brigado pela ajuda','thanks','Thank you so much','tks'])
 assert.equal(avatar.detectThanks(text),true,text);
for(const text of ['','oi','nada de obrigado','não precisa de obrigado','sou obrigadoa','isso é obrigatório','not thanks'])
 assert.equal(avatar.detectThanks(text),false,text);
// The approved sent-text signals are unchanged: thanks is a separate reading.
assert.deepEqual(avatar.detectMessage('obrigado'),{greeting:false,happy:false});
assert.deepEqual(avatar.phrases,['Saque perfeito.','Até que enfim, hein?','Game, set e match.',
 'Eu faria mais rápido. Mentira.','Essa foi na linha.','Olha ele trabalhando!','Ace!','Tá bonito hoje. A tela, digo.']);
""")


def test_thanks_earns_a_bubble_on_the_celebration_or_a_cheer(avatar_ui):
    avatar_ui(SAY + r"""
measured(1440);advance(avatar.durations.wave);assert.equal(bubble(),undefined);
// "valeu" already celebrates with the racket: the phrase rides on it.
avatar.onSent('valeu!');
assert.equal(duck.dataset.gesture,'racket');
let b=bubble();assert.ok(b,'bubble shown');assert.equal(b.textContent,avatar.phrases[0]);
assert.equal(b.getAttribute('aria-hidden'),'true');assert.equal(b.dataset.side,'right');
const r=b.getBoundingClientRect();
for(const o of [geometry.orb,geometry.label,geometry.messages,geometry.composer,geometry.topbar])assert.ok(clear(r,o),JSON.stringify({r,o}));
assert.ok(r.left>=geometry.scroll.left+8&&r.right<=geometry.scroll.right-8);
advance(3199);assert.ok(bubble());advance(1);assert.equal(bubble(),undefined,'leaves after 3.2 s');
advance(avatar.durations.racket);
// A plain thank-you has no celebration of its own: a cheer carries the next phrase.
avatar.onSent('obrigado');assert.equal(duck.dataset.gesture,'cheer');
assert.equal(bubble().textContent,avatar.phrases[1]);
advance(avatar.durations.cheer);assert.equal(duck.dataset.gesture,'none');assert.ok(bubble(),'outlives the short cheer');
advance(3200);assert.equal(bubble(),undefined);
// No thanks, no bubble; a greeting still only waves.
avatar.onSent('oi');assert.equal(duck.dataset.gesture,'wave');assert.equal(bubble(),undefined);
advance(avatar.durations.wave);
avatar.onSent('nada de obrigado');assert.equal(duck.dataset.gesture,'none');assert.equal(bubble(),undefined);
""", chat=True)


@pytest.mark.parametrize('blocked,side', [('right', 'left'), ('right+left', 'above'), ('all', None)])  # noqa: E501
def test_bubble_takes_the_first_free_spot_or_stays_silent(avatar_ui, blocked, side):
    avatar_ui(SAY + 'const blocked=' + json.dumps(blocked) + ', side=' + json.dumps(side) + r""";
measured(1440);advance(avatar.durations.wave);
// Mid-stage, so that the left side is inside the content area too.
geometry.duck=rect(700,112,144,144);geometry.messages=rect(371,374,870,900);
const d=geometry.duck;
// Put the sphere where the bubble would go.
geometry.orb=rect(d.right+4,d.top,160,60);
if(blocked!=='right')geometry.label=rect(d.left-170,d.top,166,60);
if(blocked==='all')geometry.topbar=rect(geometry.scroll.left,0,geometry.scroll.width,d.top-4);
avatar.onSent('obrigado');assert.equal(duck.dataset.gesture,'cheer','the cheer plays either way');
if(side===null)assert.equal(bubble(),undefined,'no free spot: no words');
else {
 const b=bubble();assert.equal(b.dataset.side,side);const r=b.getBoundingClientRect();
 for(const o of [geometry.orb,geometry.label,geometry.messages,geometry.composer,geometry.topbar])assert.ok(clear(r,o),JSON.stringify({r,o}));
 // A later layout change that takes the spot removes the bubble at once.
 geometry.label=rect(r.left-2,r.top-2,r.width+4,r.height+4);resize();assert.equal(bubble(),undefined);
}
""", chat=True)


def test_every_sixth_finished_reply_compliments_with_a_minimum_gap(avatar_ui):
    avatar_ui(SAY + r"""
measured(1440);advance(avatar.durations.wave);assert.equal(avatar.COMPLIMENT_GAP_MS,270000);
// Real turns leave time for the duck to face the reply before it ends.
const reply=()=>{core.setLive('thinking');advance(avatar.durations.turn);core.setLive('speaking');advance(avatar.durations.turn);core.setLive('idle');};
for(let i=1;i<=5;i++){reply();assert.equal(bubble(),undefined,'reply '+i);assert.notEqual(duck.dataset.gesture,'cheer');}
reply();assert.equal(duck.dataset.gesture,'cheer');assert.equal(bubble().textContent,avatar.phrases[0]);
advance(avatar.durations.cheer+3200);assert.equal(bubble(),undefined);
// Twelve more inside the four and a half minutes: silence.
for(let i=0;i<12;i++){reply();assert.equal(bubble(),undefined);}
advance(avatar.COMPLIMENT_GAP_MS);
for(let i=0;i<5;i++){reply();assert.equal(bubble(),undefined);}
reply();assert.equal(bubble().textContent,avatar.phrases[1],'the next phrase in rotation');
// A cancelled or failed turn that never spoke does not count.
advance(avatar.durations.cheer+3200+600000);
for(let i=0;i<12;i++){core.setLive('thinking');core.setLive('idle');}
assert.equal(bubble(),undefined);
""", chat=True)


@pytest.mark.parametrize('mode', ['reduced', 'preview', 'walk', 'view'])
def test_bubble_respects_reduced_motion_preview_movement_and_view_changes(avatar_ui, mode):
    avatar_ui(SAY + 'const mode=' + json.dumps(mode) + r""";
measured(1440);advance(avatar.durations.wave);
if(mode==='reduced'){document.body.classList.add('reduced-motion');avatar.onSent('obrigado');assert.equal(bubble(),undefined);
 document.body.classList.remove('reduced-motion');advance(20000);assert.equal(bubble(),undefined,'nothing stale later');}
if(mode==='preview'){core.preview('thinking');avatar.onSent('obrigado');assert.equal(bubble(),undefined);}
if(mode==='walk'){avatar.onSent('obrigado');advance(avatar.durations.cheer);assert.ok(bubble());
 avatar.play('walk');if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 assert.equal(duck.dataset.gesture,'walk');assert.equal(bubble(),undefined,'measured for one spot only');}
if(mode==='view'){avatar.onSent('obrigado');assert.ok(bubble());document.body.classList.remove('chat-view');assert.equal(bubble(),undefined,'left the conversation');}
""", chat=True)


def test_compliment_is_css_placed_text_and_listed_for_preview(avatar_ui):
    css = (NOVO / 'avatar.css').read_text()
    assert re.search(r'\.duck-say \{ position: absolute; z-index: 5; width: max-content; max-width: 150px;', css)
    # A child of the figure, outside the mirrored pose: the words never flip.
    script = (NOVO / 'avatar.js').read_text()
    assert "node.className = 'duck-say'" in script and 'duck.append(node);' in script
    assert '<option value="compliment">Elogio</option>' in (ROOT / 'src/ui/templates/novo.html').read_text()
    avatar_ui(SAY + r"""
measured(1440);advance(avatar.durations.wave);
assert.equal(avatar.play('compliment'),true);assert.equal(duck.dataset.gesture,'cheer');assert.ok(bubble());
assert.equal(avatar.play('compliment'),false,'not while a gesture plays');
""", chat=True)


POOP = r"""
const stage=hosts['.stage'];
const piles=()=>stage.children.filter(n=>n.className==='duck-prop duck-prop-poop');
"""


def test_poop_prank_squats_leaves_a_pile_in_its_own_box_and_a_click_cleans_it(avatar_ui):
    avatar_ui(SAY + POOP + r"""
advance(1800);
assert.equal(avatar.play('poop'),true);assert.equal(duck.dataset.gesture,'poop');assert.equal(duck.dataset.part,'limbs');
const box=duck.getBoundingClientRect(),s=geometry.stage;
advance(1149);assert.equal(piles().length,0);advance(1);
assert.equal(piles().length,1);const pile=piles()[0];
assert.equal(stage.children.at(-1),pile,'in front, so it can be clicked');
assert.equal(pile.tag,'svg');assert.equal(pile.getAttribute('aria-hidden'),'true');assert.ok(pile.children.length>30);
for(const cell of pile.children)assert.equal(cell.tag,'rect');
const left=s.left+parseFloat(pile.style.left),top=s.top+parseFloat(pile.style.top),w=parseFloat(pile.style.width),h=parseFloat(pile.style.height);
assert.ok(left>=box.left&&left+w<=box.right&&top>=box.top&&top+h<=box.bottom,'inside the box the duck stands in');
assert.ok(left+w/2<box.left+box.width/2,'behind a duck that faces right');
assert.equal(bubble()?.textContent,'Ops.');
advance(avatar.durations.poop-1150);assert.equal(duck.dataset.gesture,'none');assert.equal(duck.dataset.part,'none');
assert.equal(piles().length,1,'stays after the duck gets up');
// A click cleans it: the duck startles and owns up.
pile.dispatchEvent({type:'click',target:pile});
assert.equal(piles().length,0);assert.equal(duck.dataset.gesture,'startle');assert.equal(bubble().textContent,'Tá bom, limpei.');
pile.dispatchEvent({type:'click',target:pile});assert.equal(duck.dataset.gesture,'startle','a second click does nothing');
advance(avatar.durations.startle+2400);assert.equal(bubble(),undefined);
// Left alone it fades after twenty-four seconds.
assert.equal(avatar.play('poop'),true);advance(avatar.durations.poop);assert.equal(piles().length,1);
advance(24000-(avatar.durations.poop-1150)-1);assert.notEqual(piles()[0].dataset.gone,'true');
advance(1);assert.equal(piles()[0].dataset.gone,'true');advance(900);assert.equal(piles().length,0);
""", home=True)


def test_poop_prank_is_rare_rest_only_and_never_from_a_click(avatar_ui):
    avatar_ui(POOP + r"""
document.body.classList.add('chat-view');advance(1800);avatar.setSeed(7);
assert.deepEqual(avatar.prankWeights,{poop:2,knock:2});assert.equal(avatar.PRANK_GAP_MS,300000);
for(const prank of ['poop','knock'])assert.ok(!Object.hasOwn(avatar.weights,prank)&&!Object.hasOwn(avatar.clickWeights,prank));
const seen=new Set();
const times=[];
// Two hours of rest: every gesture the duck picks by itself.
while(now<7200000) {
 page('keydown');
 const wait=[...pending.values()].find(j=>j.delay>=8000 && j.delay<=20000);advance(wait.at-now);
 if(duck.dataset.next==='walk')advance(avatar.durations.turn);
 const name=duck.dataset.gesture;if(Object.hasOwn(avatar.prankWeights,name)){times.push(now);seen.add(name);}
 advance(avatar.durations[name]);
}
assert.ok(times.length>=2,'it does happen: '+times.length);assert.ok(times.length<=24);
assert.deepEqual([...seen].sort(),['knock','poop'],'both pranks share the one five-minute gap');
assert.ok(times[0]>=300000,'never in the first five minutes');
for(let i=1;i<times.length;i++)assert.ok(times[i]-times[i-1]>=300000,'at least five minutes apart');
// Clicks never pick it, even when it is due.
advance(600000);
for(let i=0;i<60;i++){page('keydown');avatar.cancelGestures();clickDuck();assert.ok(!['poop','knock'].includes(duck.dataset.gesture));
 if(duck.dataset.next==='walk')advance(avatar.durations.turn);advance(avatar.durations[duck.dataset.gesture]||0);}
""")


@pytest.mark.parametrize('mode', ['reduced', 'pinned', 'strip', 'turn'])
def test_no_poop_without_motion_off_the_stage_or_once_a_turn_starts(avatar_ui, mode):
    avatar_ui(POOP + 'const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='reduced'){document.body.classList.add('reduced-motion');assert.equal(avatar.play('poop'),false);}
if(mode==='pinned'){measured(1728);resize();assert.equal(avatar.play('poop'),false);}
if(mode==='strip'){settledStrip();assert.equal(avatar.play('poop'),false);}
if(mode==='turn'){assert.equal(avatar.play('poop'),true);advance(600);core.setLive('thinking');advance(5000);
 assert.notEqual(duck.dataset.gesture,'poop');}
assert.equal(piles().length,0);
""")


def test_poop_is_new_rect_art_css_keyframes_and_listed_for_preview():
    css = (NOVO / 'avatar.css').read_text()
    script = (NOVO / 'avatar.js').read_text()
    assert '.duck[data-gesture="poop"] .duck-motion { transform-origin: 50% 91%; animation: gesture-poop var(--duration-poop) ease-in-out both; }' in css
    assert '@keyframes gesture-poop' in css and '.duck-prop-poop { pointer-events: auto; cursor: pointer; }' in css
    assert css.index('@keyframes gesture-poop') < css.index('.duck[data-reduced="true"] *, body.reduced-motion .duck *')
    assert '<option value="poop">Cocô</option>' in (ROOT / 'src/ui/templates/novo.html').read_text()
    # Still one raster in the whole avatar: the pile and the prints are rect cell maps.
    assert set(re.findall(r"[\w/.-]+\.(?:png|webp|jpe?g|gif|svg)", css + script)) == {'/ui/static/novo/avatar/duckcortex.png'}



# Task 164 — the knock-knock prank.
KNOCK = r"""
const stage=hosts['.stage'];
const cracks=()=>stage.children.filter(n=>n.className==='duck-prop duck-prop-crack');
"""


def test_knock_prank_says_toc_twice_and_cracks_the_screen_inside_its_own_box(avatar_ui):
    avatar_ui(SAY + KNOCK + r"""
advance(1800);
assert.equal(avatar.play('knock'),true);assert.equal(duck.dataset.gesture,'knock');assert.equal(duck.dataset.part,'none');
const box=duck.getBoundingClientRect(),s=geometry.stage;
advance(249);assert.equal(bubble(),undefined);advance(1);assert.equal(bubble().textContent,'toc');
advance(420);assert.equal(bubble(),undefined,'each toc is short');
advance(90);assert.equal(bubble().textContent,'toc');
advance(389);assert.equal(cracks().length,0);advance(1);
assert.equal(cracks().length,1);const crack=cracks()[0];
assert.equal(stage.children.at(-1),crack,'on the glass, in front of the duck');
assert.equal(crack.tag,'svg');assert.equal(crack.getAttribute('aria-hidden'),'true');assert.ok(crack.children.length>30);
for(const cell of crack.children)assert.equal(cell.tag,'rect');
assert.equal(crack.eventListeners?.click,undefined,'the crack takes no click');
const left=s.left+parseFloat(crack.style.left),top=s.top+parseFloat(crack.style.top),w=parseFloat(crack.style.width),h=parseFloat(crack.style.height);
assert.ok(left>=box.left&&left+w<=box.right&&top>=box.top&&top+h<=box.bottom,'inside the box the duck stands in');
assert.ok(left+w/2>box.left+box.width/2,'in front of a duck that faces right');
assert.equal(bubble().textContent,'Foi sem querer.');
advance(avatar.durations.knock-1150);assert.equal(duck.dataset.gesture,'none');assert.equal(cracks().length,1);
// The screen mends by itself after five seconds.
advance(5000-(avatar.durations.knock-1150)-1);assert.notEqual(crack.dataset.gone,'true');
advance(1);assert.equal(crack.dataset.gone,'true');advance(900);assert.equal(cracks().length,0);
""", home=True)


@pytest.mark.parametrize('mode', ['reduced', 'pinned', 'strip', 'turn'])
def test_no_knock_without_motion_off_the_stage_or_once_a_turn_starts(avatar_ui, mode):
    avatar_ui(SAY + KNOCK + 'const mode=' + json.dumps(mode) + r""";
document.body.classList.add('chat-view');advance(1800);
if(mode==='reduced'){document.body.classList.add('reduced-motion');assert.equal(avatar.play('knock'),false);}
if(mode==='pinned'){measured(1728);resize();assert.equal(avatar.play('knock'),false);}
if(mode==='strip'){settledStrip();assert.equal(avatar.play('knock'),false);}
if(mode==='turn'){assert.equal(avatar.play('knock'),true);advance(200);core.setLive('thinking');advance(5000);
 assert.notEqual(duck.dataset.gesture,'knock');}
assert.equal(cracks().length,0);
if(mode==='turn')assert.notEqual(bubble()?.textContent,'Foi sem querer.');
""")


def test_knock_is_new_rect_art_css_keyframes_and_listed_for_preview():
    css = (NOVO / 'avatar.css').read_text()
    script = (NOVO / 'avatar.js').read_text()
    assert '.duck[data-gesture="knock"] .duck-motion { transform-origin: 50% 91%; animation: gesture-knock var(--duration-knock) ease-in-out both; }' in css
    assert '@keyframes gesture-knock' in css and '.duck-prop-crack' not in css
    assert css.index('@keyframes gesture-knock') < css.index('.duck[data-reduced="true"] *, body.reduced-motion .duck *')
    assert '<option value="poop">Cocô</option><option value="knock">Toc-toc</option>' in (ROOT / 'src/ui/templates/novo.html').read_text()
    assert set(re.findall(r"[\w/.-]+\.(?:png|webp|jpe?g|gif|svg)", css + script)) == {'/ui/static/novo/avatar/duckcortex.png'}
    assert hashlib.sha256(PNG.read_bytes()).hexdigest() == HASH


# Task 165 — the compliment as a caption under the sphere, when the bubble cannot show.
CAPTION = r"""
const label=()=>hosts['#state-label'].textContent;
const turn=()=>{core.setLive('thinking');advance(avatar.durations.turn);core.setLive('speaking');advance(avatar.durations.turn);core.setLive('idle');};
"""


def test_caption_carries_the_phrase_when_the_bubble_has_no_free_spot(avatar_ui):
    avatar_ui(SAY + CAPTION + r"""
measured(1440);advance(avatar.durations.wave);
geometry.duck=rect(700,112,144,144);geometry.messages=rect(371,374,870,900);
const d=geometry.duck;geometry.orb=rect(d.right+4,d.top,160,60);geometry.label=rect(d.left-170,d.top,166,60);
geometry.topbar=rect(geometry.scroll.left,0,geometry.scroll.width,d.top-4);
// No room for the bubble: the cheer still plays and the label says the phrase.
avatar.onSent('obrigado');assert.equal(duck.dataset.gesture,'cheer','the cheer still plays');assert.equal(bubble(),undefined);
assert.equal(label(),'Pato · '+avatar.phrases[0]);
advance(3599);assert.equal(label(),'Pato · '+avatar.phrases[0]);advance(1);assert.equal(label(),'','leaves after 3.6 s');
// If a turn is already running, the phrase waits for it to end.
advance(avatar.durations.cheer);core.setLive('thinking');avatar.onSent('valeu');assert.equal(label(),'Pensando');
advance(avatar.durations.racket);core.setLive('idle');assert.equal(label(),'Pato · '+avatar.phrases[1]);
advance(3600);assert.equal(label(),'');
""", chat=True)


def test_caption_replaces_the_bubble_under_reduced_motion_and_waits_for_the_turn_to_end(avatar_ui):
    avatar_ui(SAY + CAPTION + r"""
measured(1440);advance(avatar.durations.wave);document.body.classList.add('reduced-motion');
// A thank-you starts a turn: the phrase waits until the label is empty again.
avatar.onSent('obrigado');assert.equal(label(),'');assert.equal(duck.dataset.gesture,'none','no cheer under reduced motion');
core.setLive('thinking');assert.equal(label(),'Pensando');core.setLive('speaking');assert.equal(label(),'Respondendo');
core.setLive('idle');assert.equal(bubble(),undefined);assert.equal(label(),'Pato · '+avatar.phrases[0]);
// A new turn takes the label back at once, and the old line never returns.
core.setLive('thinking');assert.equal(label(),'Pensando');core.setLive('idle');assert.equal(label(),'');
advance(10000);assert.equal(label(),'');
// A finished reply that earns one shows it at once (still at most one per four and a half minutes).
advance(avatar.COMPLIMENT_GAP_MS);
let shown=0;for(let i=0;i<12;i++){core.setLive('speaking');core.setLive('idle');if(label()){shown++;assert.equal(label(),'Pato · '+avatar.phrases[1]);advance(3600);assert.equal(label(),'');}}
assert.equal(shown,1);
""", chat=True)


@pytest.mark.parametrize('mode', ['stale', 'preview', 'view', 'hidden'])
def test_waiting_caption_is_dropped_when_stale_previewing_or_the_view_changes(avatar_ui, mode):
    avatar_ui(SAY + CAPTION + 'const mode=' + json.dumps(mode) + r""";
measured(1440);advance(avatar.durations.wave);document.body.classList.add('reduced-motion');
avatar.onSent('obrigado');core.setLive('thinking');
if(mode==='stale')advance(120001);
if(mode==='preview')core.preview('thinking');
if(mode==='view')document.body.classList.remove('chat-view');
if(mode==='hidden'){document.hidden=true;page('visibilitychange');document.hidden=false;page('visibilitychange');}
core.setLive('idle');
assert.notEqual(label().slice(0,4),'Pato');
if(mode==='preview'){core.preview(null);assert.equal(label(),'');}
""", chat=True)


def test_core_caption_only_fills_an_empty_label_and_keeps_no_clock():
    core = (NOVO / 'core.js').read_text()
    assert 'setTimeout' not in core and 'setInterval' not in core
    assert "(inputs.preview === null ? liveLabels[inputs.live] : previewLabels[inputs.preview]) || note;" in core
    assert "note = typeof text === 'string' && inputs.live === 'idle' && inputs.preview === null ? text : '';" in core
    assert "if (next !== 'idle') note = '';" in core
    script = (NOVO / 'avatar.js').read_text()
    # Fixed phrases only: the caption is fed from the same rotation as the bubble.
    assert script.count("window.cortexCore?.caption?.(") == 2
    assert "window.cortexCore?.caption?.('Pato · ' + text)" in script and "window.cortexCore?.caption?.('')" in script
