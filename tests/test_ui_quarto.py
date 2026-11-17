"""Task 163 — the home stage as the duck's room, with the projects as books."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOVO = ROOT / 'src/ui/static/novo'
JS = (NOVO / 'quarto.js').read_text()
CSS = (NOVO / 'quarto.css').read_text()
HASH = 'a848bd285577cdb4eaf901aa15bc8798f4a76af6e46f3869d47a0ad5d650925e'

# A very small page: only what quarto.js touches. Sizes come from `geometry`.
HARNESS = r"""
const assert=require('assert');
class Node{
 constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.dataset={};this.className='';
  this.vars={};this.style={cssText:'',setProperty:(k,v)=>{this.vars[k]=v;}};
  const self=this;this.classList={toggle(name,on){const set=new Set(self.className.split(' ').filter(Boolean));
   on?set.add(name):set.delete(name);self.className=[...set].join(' ');},contains:name=>self.className.split(' ').includes(name)};}
 setAttribute(k,v){this.attributes[k]=String(v);}
 append(...nodes){for(const n of nodes){n.parent=this;this.children.push(n);}}
 prepend(...nodes){for(const n of nodes)n.parent=this;this.children.unshift(...nodes);}
 replaceChildren(...nodes){this.children=[];this.append(...nodes);}
 addEventListener(name,fn){(this.listeners[name] ||= []).push(fn);}
 click(){for(const fn of this.listeners.click||[])fn({type:'click',target:this});}
 getBoundingClientRect(){return this.rect||{left:0,top:0,right:0,bottom:0,width:0,height:0};}
 all(test,out=[]){for(const c of this.children){if(test(c))out.push(c);c.all(test,out);}return out;}
 has(name){return this.className.split(' ').includes(name);}
}
const rect=(left,top,width,height)=>({left,top,width,height,right:left+width,bottom:top+height});
const stage=new Node('section'),dock=new Node('div'),orb=new Node('canvas'),body=new Node('body');
const hosts={'.stage':stage,'#cortex-duck-dock':dock,'#orb':orb};
const created=[];
const document={body,querySelector:s=>hosts[s]||null,createElement:tag=>{const n=new Node(tag);created.push(n);return n;}};
const window={};const resizers=[];let observed=0;
const addEventListener=(name,fn)=>{if(name==='resize')resizers.push(fn);};
const setTimeout=()=>0;
class ResizeObserver{constructor(fn){this.fn=fn;}observe(){observed++;}}
class MutationObserver{constructor(fn){this.fn=fn;}observe(){observed++;}}
function measure(width){
 const phone=width<=800,stageLeft=phone?18:292,stageWidth=width-stageLeft-(phone?18:40);
 stage.rect=rect(stageLeft,phone?58:64,stageWidth,phone?432:278);
 const size=phone?96:144;dock.offsetWidth=size;dock.offsetLeft=phone?24:36;dock.offsetTop=phone?36:48;
 const center=stageLeft+stageWidth/2,disc=phone?260:255;
 orb.rect=rect(center-disc/2,phone?214:71,disc,disc);
}
const named=name=>stage.all(n=>n.has(name));
const books=()=>named('q-project').map(n=>n.children[0].textContent);
"""


def run(steps, width=1440):
    script = HARNESS + f'measure({width});\n' + JS + '\nconst quarto=window.cortexQuarto;\n' + steps
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_assets_are_local_new_art_and_the_duck_png_is_untouched():
    png = NOVO / 'avatar/duckcortex.png'
    assert hashlib.sha256(png.read_bytes()).hexdigest() == HASH
    html = (ROOT / 'src/ui/templates/novo.html').read_text()
    assert html.index('/ui/static/novo/avatar.css') < html.index('/ui/static/novo/quarto.css')
    # The room exists before the project list is drawn into it.
    assert html.index('/ui/static/novo/quarto.js') < html.index('/ui/static/novo/projetos.js')
    for source in (JS, CSS):
        assert 'duckcortex' not in source and '.png' not in source
        assert not re.search(r'https?://|url\(|@import', source)
    assert 'fetch(' not in JS and 'localStorage' not in JS and 'import(' not in JS


def test_project_names_never_become_markup():
    # The only innerHTML is the fixed cell-map sprite; names go through textContent.
    assert JS.count('innerHTML') == 1
    sprite = JS[JS.index('function sprite('):JS.index('const plant')]
    assert 'innerHTML' in sprite and 'project' not in sprite and 'name' not in sprite.replace('className', '')
    out = run(r"""
quarto.render([{id:1,name:'<img src=x onerror=alert(1)>'}],()=>{},()=>{});
const book=named('q-project')[0];
assert.equal(book.children[0].textContent,'<img src=x onerror=alert(1)>');
assert.equal(book.innerHTML,undefined);assert.equal(book.children[0].innerHTML,undefined);
assert.ok(!created.some(n=>n.tag==='img'||n.tag==='script'));
console.log('ok');
""")
    assert out.strip() == 'ok'


def test_desktop_room_has_every_piece_and_books_open_their_project():
    out = run(r"""
assert.equal(stage.children[0],named('q-room')[0],'the room is the first layer of the stage');
const opened=[];
quarto.render([{id:'a',name:'Site do clube',objective:'x'},{id:'b',name:'Receitas'},{id:7,name:'Viagem 2027'}],
 id=>opened.push(id),()=>opened.push('more'));
const room=named('q-room')[0];
assert.ok(!room.has('q-tight'));
for(const piece of ['q-floor','q-rug','q-window','q-racket','q-bookcase','q-lamp','q-bed','q-plant','q-trophy','q-ball'])
 assert.ok(named(piece).length>=1,piece);
assert.deepEqual(books(),['Site do clube','Receitas','Viagem 2027']);
assert.equal(named('q-more').length,0);assert.equal(named('q-filler').length,2);
const all=named('q-project');all[1].click();all[2].click();assert.deepEqual(opened,['b',7]);
for(const book of all){assert.equal(book.tag,'button');assert.equal(book.type,'button');
 assert.ok(book.attributes['aria-label'].startsWith('Abrir o projeto “'));}
// Decoration is hidden from assistive tech; the bookcase is a labelled group.
assert.equal(named('q-bookcase')[0].attributes.role,'group');
assert.equal(room.attributes['aria-hidden'],undefined);
for(const piece of [...named('q-sprite'),...named('q-filler'),...named('q-floor'),...named('q-rug')])
 assert.equal(piece.attributes['aria-hidden'],'true');
// The floor is at the duck's feet and stops short of the sphere on both sides.
// The feet end at 98.7% of the box (y=190): the rug (floor-5) starts 3px above, the floor 2px below.
assert.equal(room.vars['--floor'],'192px');
const [floorLeft,floorRight]=named('q-floor'),orbLeft=orb.rect.left-stage.rect.left,orbRight=orb.rect.right-stage.rect.left;
assert.equal(floorLeft.style.cssText,`left:0;width:${orbLeft-28}px`);
assert.equal(floorRight.style.cssText,`left:${orbRight+28}px;right:0`);
assert.equal(named('q-wall-right')[0].style.cssText,`left:${orbRight+44}px;right:8px`);
console.log(JSON.stringify(named('q-project').map(b=>b.vars['--c'])));
""")
    colours = json.loads(out)
    assert len(colours) == 3 and all(re.fullmatch(r'#[0-9a-f]{6}', c) for c in colours)


def test_same_project_keeps_its_colour_and_the_shelf_is_the_same_on_every_draw():
    out = run(r"""
const shelf=()=>JSON.stringify([named('q-project').map(b=>[b.children[0].textContent,b.vars['--c']]),
 named('q-filler').map(b=>[b.vars['--w'],b.vars['--h'],b.dataset.tone])]);
quarto.render([{id:'a',name:'Um'},{id:'b',name:'Dois'}]);const first=shelf();
quarto.render([{id:'b',name:'Dois'},{id:'a',name:'Um'}]);
const colour=name=>named('q-project').find(b=>b.children[0].textContent===name).vars['--c'];
assert.equal(colour('Um'),JSON.parse(first)[0][0][1]);assert.equal(colour('Dois'),JSON.parse(first)[0][1][1]);
quarto.render([{id:'a',name:'Um'},{id:'b',name:'Dois'}]);assert.equal(shelf(),first);
// An unchanged draw does not rebuild the books (keyboard focus survives a resize).
const before=named('q-project')[0];for(const fn of resizers)fn();assert.equal(named('q-project')[0],before);
console.log('ok');
""")
    assert out.strip() == 'ok'


@pytest.mark.parametrize('width, capacity', [(375, 6), (1440, 10)])
def test_more_projects_than_fit_end_in_one_book_that_opens_the_list(width, capacity):
    out = run(r"""
const opened=[],list=Array.from({length:14},(_,i)=>({id:i,name:'Projeto '+i}));
quarto.render(list,id=>opened.push(id),()=>opened.push('more'));
const plan=quarto.plan({width:stage.rect.width,orbRight:orb.rect.right-stage.rect.left,orbTop:orb.rect.top-stage.rect.top,
 duckLeft:dock.offsetLeft,duckSize:dock.offsetWidth,floor:Math.round(dock.offsetTop+dock.offsetWidth*.987)+2,count:14});
const shown=books();
assert.equal(shown.length,plan.capacity);assert.equal(plan.books,plan.capacity-1);
assert.deepEqual(shown.slice(0,-1),list.slice(0,plan.capacity-1).map(p=>p.name));
assert.equal(shown.at(-1),'+'+(14-plan.books));assert.equal(named('q-filler').length,0);
named('q-more')[0].click();assert.deepEqual(opened,['more']);
assert.equal(named('q-bed').length,0,'the bed gives its place to the books');
console.log(plan.capacity);
""", width)
    assert int(out) == capacity


def test_phone_keeps_only_floor_rug_bookcase_and_lamp_beside_the_duck():
    out = run(r"""
quarto.render([{id:1,name:'Site do clube'},{id:2,name:'Receitas'}]);
const room=named('q-room')[0];assert.ok(room.has('q-tight'));
for(const gone of ['q-window','q-racket','q-bed'])assert.equal(named(gone).length,0,gone);
for(const kept of ['q-rug','q-bookcase','q-lamp'])assert.equal(named(kept).length,1,kept);
assert.equal(room.vars['--floor'],(36+95+2)+'px');
const [floorLeft,floorRight]=named('q-floor');
assert.equal(floorLeft.style.cssText,`left:0;width:${stage.rect.width}px`);assert.equal(floorRight.style.cssText,'display:none');
assert.equal(named('q-wall-left')[0].style.cssText,'display:none');
// The furniture starts 20px to the right of the duck's box.
assert.equal(named('q-wall-right')[0].style.cssText,'left:140px;right:4px');
assert.deepEqual(books(),['Site do clube','Receitas']);
console.log('ok');
""", 375)
    assert out.strip() == 'ok'


def test_empty_failed_or_odd_lists_leave_a_shelf_of_plain_books():
    out = run(r"""
assert.equal(named('q-project').length,0);assert.equal(named('q-filler').length,6);
for(const list of [[],null,undefined,'x',[null,{name:'sem id'}]]){
 quarto.render(list);assert.equal(named('q-project').length,0);assert.equal(named('q-filler').length,6);}
quarto.render([{id:1}]);assert.deepEqual(books(),['']);
// A hidden view measures as zero: the last drawing stays.
quarto.render([{id:1,name:'Um'}]);stage.rect=rect(0,0,0,0);quarto.render([]);assert.deepEqual(books(),['Um']);
assert.ok(observed>=3&&resizers.length===1);
console.log('ok');
""")
    assert out.strip() == 'ok'


def test_plan_never_overflows_and_drops_the_bed_first():
    out = run(r"""
const base={orbTop:7,duckLeft:36,duckSize:144,floor:188};
for(let width=560;width<=1600;width+=20){
 const orbRight=width/2+127.5;
 for(const count of [0,1,5,30]){
  const p=quarto.plan({...base,width,orbRight,count});
  assert.ok(!p.tight);assert.ok(p.capacity>=0);assert.ok(p.books+(p.hidden?1:0)+(count?p.fillers:0)<=Math.max(p.capacity,0)||!p.capacity);
  assert.equal(p.books+p.hidden,p.capacity?count:p.books);
  const used=(p.capacity?12+12+p.capacity*28-2:0)+16+36+(p.bed?16+90:0);
  assert.ok(used<=width-orbRight-52||!p.capacity,width+' '+used);
  const withBed=quarto.plan({...base,width,orbRight,count:0});
  assert.equal(withBed.bed,width-orbRight-52>=350);
  assert.equal(p.bed,withBed.bed&&count<=withBed.capacity);
 }
}
console.log('ok');
""")
    assert out.strip() == 'ok'


def test_css_home_only_back_layer_clicks_and_reduced_motion():
    flat = re.sub(r'\s+', ' ', CSS)
    assert '.chat-view .q-room, .projects-view .q-room { display: none; }' in flat
    room = flat[flat.index('.q-room {'):flat.index('}', flat.index('.q-room {'))]
    assert 'pointer-events: none' in room and 'z-index: -1' in room and 'position: absolute' in room
    # The back layer of the home stage only: nothing else gets a z-index, so the duck's
    # props (the poop that takes a click) keep their place in front of the duck's box.
    assert 'body:not(.chat-view) .stage { isolation: isolate; }' in flat
    assert flat.count('z-index') == 1 and '.duck' not in flat and '.orb' not in flat
    # Only the bookcase takes a click.
    assert flat.count('pointer-events: auto') == 1
    assert re.search(r'\.q-bookcase \{[^}]*pointer-events: auto', flat)
    assert 'animation' not in flat and '@keyframes' not in flat
    assert flat.count('transition') == 3
    assert '@media (prefers-reduced-motion: reduce) { .q-project { transition: none; } .q-project:hover, .q-project:focus-visible { transform: none; } }' in flat
    assert 'body.reduced-motion .q-project { transition: none; }' in flat
    assert 'body.reduced-motion .q-project:hover, body.reduced-motion .q-project:focus-visible { transform: none; }' in flat
    assert 'body.light .q-room {' in flat


def test_bookcase_is_an_obstacle_for_the_duck_and_its_bubble_and_gets_the_real_list():
    avatar = (NOVO / 'avatar.js').read_text()
    walk = avatar[avatar.index('function homeGeometry()'):avatar.index('function homePath(')]
    assert "'#messages', '.q-bookcase']" in walk
    say = avatar[avatar.index('const SAY_OBSTACLES'):avatar.index('let bubble')]
    assert "'.q-bookcase'" in say
    projetos = (NOVO / 'projetos.js').read_text()
    render = projetos[projetos.index('function renderProjects('):projetos.index('function renderRecents(')]
    assert "window.cortexQuarto?.render(projects, selectProject, () => navigate('projects'));" in render
