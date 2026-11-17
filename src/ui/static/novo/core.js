"use strict";
(() => {
const $ = (s) => document.querySelector(s);
const canvas = $('#orb'), ctx = canvas.getContext('2d', {alpha:true});
const inputs = {
  live: 'idle', preview: null, intensity: .7,
  reduced: matchMedia('(prefers-reduced-motion: reduce)').matches
};
let frame = 0, time = 0, lastTime = 0, lastDraw = 0;
let replyKick = 0;
const liveLabels = {
  idle: '', thinking: 'Pensando', searching: 'Buscando na web',
  reading: 'Lendo', writing: 'Escrevendo', listening: 'Ouvindo', speaking: 'Respondendo'
};
const previewLabels = {
  idle: 'Prévia · Em repouso', thinking: 'Prévia · Pensando',
  searching: 'Prévia · Buscando na web', reading: 'Prévia · Lendo', writing: 'Prévia · Escrevendo',
  listening: 'Prévia · Ouvindo', speaking: 'Prévia · Respondendo'
};
// Depth: a gyroscope inside the same 165-point shell, drawn far side first.
// Canvas 2D only. A mouse makes it lean a little toward the pointer; touch never does.
const TURN = Math.PI * 2;
const LIGHT = [-.45, -.62, .64];
const par = {x: 0, y: 0, tx: 0, ty: 0};
const clamp1 = v => Math.max(-1, Math.min(1, v));
document.addEventListener('pointermove', e => {
  if (e.pointerType === 'touch') return;
  // Full lean two sphere widths away, measured from the sphere's own box.
  const b = canvas.getBoundingClientRect(); if (!b.width || !b.height) return;
  par.tx = clamp1((e.clientX - b.left - b.width / 2) / (b.width * 2));
  par.ty = clamp1((e.clientY - b.top - b.height / 2) / (b.height * 2));
}, {passive: true});
document.addEventListener('pointerleave', () => { par.tx = par.ty = 0; });
// The small solid at the centre: a regular icosahedron (12 corners, 30 edges).
const ico = [];
{ const g = (1 + Math.sqrt(5)) / 2, n = Math.hypot(1, g);
  for (const a of [-1, 1]) for (const b of [-g, g]) ico.push({x: 0, y: a / n, z: b / n}, {x: a / n, y: b / n, z: 0}, {x: b / n, y: 0, z: a / n}); }
const icoEdges = [];
ico.forEach((p, i) => ico.forEach((q, j) => { if (j > i && Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z) < 1.1) icoEdges.push([i, j]); }));
const rotX = (p, a) => { const c = Math.cos(a), s = Math.sin(a); return {x: p.x, y: p.y * c - p.z * s, z: p.y * s + p.z * c}; };
const rotY = (p, a) => { const c = Math.cos(a), s = Math.sin(a); return {x: p.x * c + p.z * s, y: p.y, z: -p.x * s + p.z * c}; };
const rotZ = (p, a) => { const c = Math.cos(a), s = Math.sin(a); return {x: p.x * c - p.y * s, y: p.x * s + p.y * c, z: p.z}; };
const points = [];
const edges = [];
// Deterministic sphere geometry; each point has at most four nearby connections.
const count = 165, golden = Math.PI * (3 - Math.sqrt(5));
for(let i=0;i<count;i++) {const y=1-(i/(count-1))*2,r=Math.sqrt(1-y*y),a=i*golden;points.push({x:Math.cos(a)*r,y,z:Math.sin(a)*r});}
points.forEach((p,i)=>{const near=points.map((q,j)=>({j,d:(p.x-q.x)**2+(p.y-q.y)**2+(p.z-q.z)**2})).filter(q=>q.j!==i).sort((a,b)=>a.d-b.d).slice(0,4);for(const q of near)if(q.j>i)edges.push([i,q.j]);});
function resizeCanvas(){const size=canvas.getBoundingClientRect();const ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(size.width*ratio);canvas.height=Math.round(size.height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw();}
new ResizeObserver(resizeCanvas).observe(canvas);
function draw(){
const {intensity, reduced} = inputs;
const state = inputs.preview ?? inputs.live;
const width=canvas.clientWidth,height=canvas.clientHeight;if(!width||!height)return;ctx.clearRect(0,0,width,height);
const light=document.body.classList.contains('light');
// D is the camera distance in sphere radii; the shell shrinks a little so the orbits fit.
const D=2.6,r=Math.min(width,height)*.365*.88;
const energy=state==='thinking'?1.16:state==='listening'?1.06:1;
// Only the laboratory supplies sample batches; live output follows actual tokens.
const kick=inputs.preview==='speaking'?Math.exp(-(time%.8)*7)*.45:replyKick;
const pulse=reduced?1:state==='speaking'?1+kick*.012*intensity:1+Math.sin(time*1.2)*.018*energy;
const effectTime=reduced?0:time;
const C=light
  ?{line:[81,91,114],lineBlue:[41,80,153],red:[183,37,70],blue:[43,88,173],white:[59,76,107]}
  :{line:[194,210,237],lineBlue:[92,144,236],red:[249,78,111],blue:[101,168,255],white:[211,225,249]};
const ink=(c,a)=>`rgba(${c[0]},${c[1]},${c[2]},${Math.max(0,Math.min(1,a))})`;
const boost=light?1.6:1;
const ay=time*.12+par.x*.45,ax=.28+Math.sin(time*.12)*.14-par.y*.3;
const view=p=>rotX(rotY(p,ay),ax);
const proj=v=>{const s=D/(D-v.z);return {x:v.x*r*s,y:v.y*r*s,z:v.z,s};};
// Orbits and rings keep their own axes; they only lean with the pointer.
const lean=p=>rotX(rotY(p,par.x*.3),-par.y*.22);
// One stroke for the near (side 1) or far (side -1) segments of a path.
function strokeSide(pts,closed,side,style,lineWidth){
  const n=pts.length,last=closed?n:n-1;
  ctx.beginPath();
  for(let i=0;i<last;i++){
    const a=pts[i],b=pts[(i+1)%n];
    if(((a.z+b.z)/2<0?-1:1)!==side)continue;
    ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
  }
  ctx.strokeStyle=style;ctx.lineWidth=lineWidth;ctx.stroke();
}
ctx.save();ctx.translate(width/2,height/2);ctx.scale(pulse,pulse);
const halo=ctx.createRadialGradient(0,0,r*.08,0,0,r*1.34);halo.addColorStop(0,light?'rgba(185,34,67,.045)':`rgba(221,37,76,${.17*intensity})`);halo.addColorStop(.56,light?'rgba(68,102,169,.025)':`rgba(59,91,174,${.07*intensity})`);halo.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=halo;ctx.fillRect(-width/2,-height/2,width,height);
const V=points.map(view),P=V.map(proj);
// Three satellites on tilted orbits, each with a short fading trail.
const orbits=[];
for(let i=0;i<3;i++){
  const at=th=>proj(lean(rotZ(rotX({x:Math.cos(th)*1.2,y:Math.sin(th)*1.2,z:0},[1.25,1.05,1.4][i]),i*1.1+.3)));
  const pts=[];for(let n=0;n<64;n++)pts.push(at(n*TURN/64));
  const head=effectTime*(.5+.13*i)+i*2,trail=[];
  for(let n=0;n<=12;n++)trail.push(at(head-(12-n)*.045));
  orbits.push({pts,trail});
}
function shell(side){
  for(const o of orbits){
    strokeSide(o.pts,true,side,ink(C.lineBlue,(side<0?.08:.24)*intensity*boost),side<0?.45:.8);
    for(let n=0;n<12;n++){
      const a=o.trail[n],b=o.trail[n+1];
      if((b.z<0?-1:1)!==side)continue;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
      ctx.strokeStyle=ink(C.blue,n/12*(side<0?.35:.9)*intensity);ctx.lineWidth=1.2*b.s;ctx.stroke();
    }
    const h=o.trail[12];
    if((h.z<0?-1:1)===side){
      ctx.beginPath();ctx.arc(h.x,h.y,1.7*h.s,0,TURN);ctx.fillStyle=ink(C.white,side<0?.45:1);
      if(!light&&side>0){ctx.shadowColor='#4b9aff';ctx.shadowBlur=8*intensity;}
      ctx.fill();ctx.shadowBlur=0;
    }
  }
  for(const [i,j] of edges){
    const a=P[i],b=P[j],depth=(a.z+b.z)/2;
    if((depth<0?-1:1)!==side)continue;
    const t=(depth+1)/2;
    ctx.strokeStyle=ink((i+j)%5===0?C.lineBlue:C.line,(.03+.4*t*t)*intensity*boost);
    ctx.lineWidth=.3+.75*t;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }
  for(let i=0;i<P.length;i++){
    const p=P[i];if((p.z<0?-1:1)!==side)continue;
    const v=V[i],t=(p.z+1)/2,red=i%9===0,blue=i%5===0;
    // Points facing the light are larger and brighter; far ones shrink and fade.
    const lam=Math.max(0,v.x*LIGHT[0]+v.y*LIGHT[1]+v.z*LIGHT[2]);
    ctx.beginPath();ctx.arc(p.x,p.y,(red||blue?1.5:.7)*(.4+1.2*t*t+.25*lam),0,TURN);
    ctx.fillStyle=ink(red?C.red:blue?C.blue:C.white,(.14+.6*t+.3*lam)*(light?1.3:1));
    if((red||blue)&&t>.7&&!light){ctx.shadowColor=red?'#ff345b':'#4b9aff';ctx.shadowBlur=(6+6*lam)*intensity;}
    ctx.fill();ctx.shadowBlur=0;
  }
}
shell(-1);
// Gyroscope: three rings on their own axes (0 = outer) around a small turning solid.
const rings=[.86,.7,.54].map((R,i)=>{
  const pts=[];
  for(let n=0;n<56;n++){
    const th=n*TURN/56;
    pts.push(proj(lean(rotY(rotX({x:Math.cos(th)*R,y:Math.sin(th)*R,z:0},time*(.42+i*.11)+i*1.3+.9),time*.31*(i%2?-1:1)+i*2.1))));
  }
  // Quiet static cues: all rings for thought, outer for listening, inner for reply.
  const cue=reduced&&(state==='thinking'||state==='listening'&&i===0||state==='speaking'&&i===2)?2:1;
  return {pts,colour:[C.red,C.white,C.blue][i],cue};
});
for(const g of rings)strokeSide(g.pts,true,-1,ink(g.colour,.3*intensity*boost),.8*g.cue);
const Q=ico.map(p=>proj(lean(rotX(rotY({x:p.x*.36,y:p.y*.36,z:p.z*.36},-time*.5),time*.33))));
for(const [i,j] of icoEdges){
  const a=Q[i],b=Q[j],t=((a.z+b.z)/2/.36+1)/2;
  ctx.strokeStyle=ink(C.red,(.15+.7*t)*intensity*boost);ctx.lineWidth=.5+.7*t;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
}
const coreRadius=r*.15*(1+kick*.25);const core=ctx.createRadialGradient(-coreRadius*.15,-coreRadius*.2,1,0,0,coreRadius*1.6);core.addColorStop(0,light?'rgba(255,206,211,.95)':`rgba(255,199,208,${.95*intensity})`);core.addColorStop(.18,light?'#bc2947':`rgba(255,57,100,${.95*intensity})`);core.addColorStop(.55,light?'rgba(176,38,66,.8)':`rgba(196,23,65,${.55*intensity})`);core.addColorStop(1,'rgba(196,23,65,0)');ctx.fillStyle=core;ctx.beginPath();ctx.arc(0,0,coreRadius*1.6,0,TURN);ctx.fill();
for(const g of rings){
  strokeSide(g.pts,true,1,ink(g.colour,.85*intensity*boost),1.5*g.cue);
  const bead=g.pts[Math.floor(((effectTime*.2)%1)*56)];
  ctx.beginPath();ctx.arc(bead.x,bead.y,1.6*bead.s,0,TURN);ctx.fillStyle=ink(C.white,bead.z<0?.4:1);ctx.fill();
}
shell(1);
// Activity accents follow the real surface: a meridian sweeps, a latitude descends.
if(state==='searching'){
  const phi=effectTime*1.6-Math.PI/4,line=[];
  for(let n=0;n<=28;n++){const th=n*Math.PI/28;line.push(proj(view({x:Math.sin(th)*Math.sin(phi)*1.04,y:Math.cos(th)*1.04,z:Math.sin(th)*Math.cos(phi)*1.04})));}
  strokeSide(line,false,-1,ink(C.blue,.3*intensity),1);
  strokeSide(line,false,1,ink(C.blue,.9*intensity),1.9);
  // Angular distance behind the sweep gives a fading radar trail.
  for(let i=0;i<P.length;i++){
    const glow=Math.max(0,1-((phi-Math.atan2(points[i].x,points[i].z))%TURN+TURN)%TURN/.9)*(.3+.7*(P[i].z+1)/2);
    if(!glow)continue;
    ctx.globalAlpha=glow*intensity;ctx.beginPath();ctx.arc(P[i].x,P[i].y,(1+glow*1.4)*P[i].s,0,TURN);
    ctx.fillStyle=i%9===0?(light?'#b72546':'#ff6b8b'):(light?'#305da8':'#9bc5ff');ctx.fill();
  }
  ctx.globalAlpha=1;
}else if(state==='reading'){
  const ys=reduced?-.2:(effectTime*.34%1)*2.1-1.05;
  // The band exists between the poles; the points start to glow as it arrives.
  if(Math.abs(ys)<1){
    const rho=Math.sqrt(1-ys*ys)*1.04,band=[];
    for(let n=0;n<48;n++){const th=n*TURN/48;band.push(proj(view({x:Math.cos(th)*rho,y:ys,z:Math.sin(th)*rho})));}
    strokeSide(band,true,-1,ink(C.blue,.2*intensity),r*.02);
    strokeSide(band,true,1,ink(C.blue,.5*intensity),r*.04);
  }
  for(let i=0;i<P.length;i++){
    const glow=Math.max(0,1-Math.abs(points[i].y-ys)/.18)*(.3+.7*(P[i].z+1)/2);
    if(!glow)continue;
    ctx.globalAlpha=glow*intensity;ctx.beginPath();ctx.arc(P[i].x,P[i].y,(1+glow*1.3)*P[i].s,0,TURN);
    ctx.fillStyle=i%9===0?(light?'#b72546':'#ff6b8b'):(light?'#305da8':'#9bc5ff');ctx.fill();
  }
  ctx.globalAlpha=1;
}
// A few bright travelling points, without changing the static network geometry.
for(let k=0;k<(state==='writing'?21:7);k++){const edge=edges[(k*41)%edges.length],a=P[edge[0]],b=P[edge[1]],v=(effectTime*(state==='writing'?.72:.18)+k*.143)%1,near=.45+.55*((a.z+b.z)/2+1)/2;ctx.globalAlpha=near;ctx.beginPath();ctx.arc(a.x+(b.x-a.x)*v,a.y+(b.y-a.y)*v,1.2,0,Math.PI*2);ctx.fillStyle=k%2?(light?'#305da8':'#9bc5ff'):(light?'#b72546':'#ff6b8b');ctx.fill();
if(state==='writing'){
  const tail=Math.max(0,v-.22);
  ctx.beginPath();ctx.moveTo(a.x+(b.x-a.x)*tail,a.y+(b.y-a.y)*tail);
  ctx.lineTo(a.x+(b.x-a.x)*v,a.y+(b.y-a.y)*v);
  ctx.strokeStyle=ctx.fillStyle;ctx.globalAlpha=.55*intensity*near;ctx.lineWidth=.8;ctx.stroke();
}
ctx.globalAlpha=1;}
ctx.restore();}
function animate(now){
const {reduced} = inputs;
const state = inputs.preview ?? inputs.live;
frame=requestAnimationFrame(animate);
if(document.hidden)return;
if(now-lastDraw<(reduced?180:32))return;
const dt=Math.min((now-(lastTime||now))/1000,.06);lastTime=now;lastDraw=now;
if(!reduced){
  time+=dt*(state==='thinking'?2.8:state==='listening'?1.5:state==='speaking'?1.8:1);
  replyKick*=Math.exp(-dt*7);
  par.x+=(par.tx-par.x)*Math.min(1,dt*5);par.y+=(par.ty-par.y)*Math.min(1,dt*5);
}else{par.x=par.y=0;}
draw();}
frame=requestAnimationFrame(animate);
document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;replyKick=0;}else{lastTime=0;if(!frame)frame=requestAnimationFrame(animate);}});
// Task 165: a short line from the duck may use the label under the sphere, only
// while the label is otherwise empty (at rest, no preview). Any real state wins.
// No clock here: the caller takes the line away again with caption('').
let note = '';
function updateLabel() {
  $('#state-label').textContent =
    (inputs.preview === null ? liveLabels[inputs.live] : previewLabels[inputs.preview]) || note;
}
function caption(text) {
  note = typeof text === 'string' && inputs.live === 'idle' && inputs.preview === null ? text : '';
  updateLabel();
  return note !== '';
}
function setLive(next, avatarState = next) {
  if (!Object.hasOwn(liveLabels, next)) return;
  if (inputs.live !== next) replyKick = 0;
  inputs.live = next;
  if (next !== 'idle') note = '';
  window.cortexAvatar?.setLive(avatarState);
  $('#mic').classList.toggle('mic-active', next === 'listening');
  $('#mic').setAttribute('aria-pressed', String(next === 'listening'));
  updateLabel();
  draw();
}
function preview(next) {
  if (next !== null && !Object.hasOwn(previewLabels, next)) return;
  if (inputs.preview !== next) replyKick = 0;
  inputs.preview = next;
  window.cortexAvatar?.preview(next);
  updateLabel();
  draw();
  // Let the shell reset its selector without exposing mutable live state.
  return inputs.live;
}
function setIntensity(value) {
  if (!Number.isFinite(value)) return;
  inputs.intensity = Math.min(1, Math.max(.2, value));
  draw();
}
function setReduced(value) {
  inputs.reduced = Boolean(value);
  replyKick = 0;
  if (inputs.reduced) par.x = par.y = 0;
  draw();
}
function pulseToken() {
  if (inputs.reduced || document.hidden || inputs.preview !== null || inputs.live !== 'speaking') return;
  replyKick = Math.min(1, replyKick + .35);
}
window.cortexCore = {setLive, preview, setIntensity, setReduced, pulseToken, caption, redraw: draw};
})();
