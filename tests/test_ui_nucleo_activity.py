"""Activity cues exercise the real canvas API and the chat's SSE reader in Node."""
import json
from pathlib import Path
import subprocess

import pytest

from tests.test_ui_bolha_viva import chat_ui  # noqa: F401
from tests.ui_parity import Document


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "idle": "", "thinking": "Pensando", "searching": "Buscando na web",
    "reading": "Lendo", "writing": "Escrevendo", "listening": "Ouvindo",
    "speaking": "Respondendo",
}


@pytest.fixture
def core_ui():
    source = (ROOT / "src/ui/static/novo/core.js").read_text()
    harness = r"""
const assert = require('node:assert/strict');
let ops = [], draws = 0, systemReduced = SYSTEM_REDUCED;
const ctx = new Proxy({}, {
  get(target, name) {
    if (Object.hasOwn(target, name)) return target[name];
    if (name === 'createRadialGradient') return (...args) => {
      ops.push([name, ...args]);
      return {addColorStop: (...args) => ops.push(['addColorStop', ...args])};
    };
    return (...args) => {
      if (name === 'clearRect') { ops = []; draws++; }
      ops.push([name, ...args]);
    };
  },
  set(target, name, value) {
    target[name] = value;
    ops.push([name, typeof value === 'object' ? 'gradient' : value]); return true;
  }
});
const canvas = {
  clientWidth: 320, clientHeight: 320, getContext: () => ctx,
  getBoundingClientRect: () => ({width: 320, height: 320})
};
const mic = {pressed: 'false', classList: {toggle() {}},
  setAttribute(name, value) { if (name === 'aria-pressed') this.pressed = value; }};
const label = {textContent: ''};
let light = false;
const listeners = {};
const document = {hidden: false, body: {classList: {contains: () => light}},
  querySelector: selector => ({'#orb': canvas, '#mic': mic, '#state-label': label})[selector],
  addEventListener: (name, fn) => { listeners[name] = fn; }};
const window = {}, devicePixelRatio = 1;
const matchMedia = () => ({matches: systemReduced});
const ResizeObserver = class { observe() {} };
let rafId = 0;
const frames = new Map();
const requestAnimationFrame = fn => { frames.set(++rafId, fn); return rafId; };
const cancelAnimationFrame = id => frames.delete(id);
const tick = now => {
  const current = [...frames]; frames.clear();
  for (const [, fn] of current) fn(now);
};
const snapshot = () => JSON.stringify(ops);
const scale = () => ops.find(op => op[0] === 'scale')[1];
const visible = value => { document.hidden = !value; listeners.visibilitychange(); };
"""

    def run(steps, *, system_reduced=False):
        script = harness.replace("SYSTEM_REDUCED", json.dumps(system_reduced))
        script += source + "\nconst core = window.cortexCore;\n" + steps
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr

    return run


def test_live_and_preview_labels_validation_and_isolation(core_ui):
    core_ui("const labels = " + json.dumps(LABELS) + r""";
for (const [state, text] of Object.entries(labels)) {
  core.setLive(state);
  assert.equal(label.textContent, text, state);
  assert.equal(mic.pressed, String(state === 'listening'));
  assert.equal(core.preview(state), state);
  assert.equal(label.textContent, 'Prévia · ' + (text || 'Em repouso'));
  core.setLive('thinking');
  assert.equal(label.textContent, 'Prévia · ' + (text || 'Em repouso'));
  assert.equal(core.preview(null), 'thinking');
  assert.equal(label.textContent, 'Pensando');
}
for (const invalid of ['junk', '__proto__', 'constructor', null, undefined]) {
  core.setLive('reading'); core.setLive(invalid);
  assert.equal(label.textContent, 'Lendo');
  core.preview('writing');
  if (invalid !== null) core.preview(invalid);
  assert.equal(label.textContent, 'Prévia · Escrevendo');
  core.preview(null);
}
""")


def test_activity_preview_options():
    doc = Document((ROOT / "src/ui/templates/novo.html").read_text())
    options = [node for node in doc.select("#state-select")[0].descendants() if node.tag == "option"]
    assert [(node.attrs["value"], node.text()) for node in options] == [
        (state, text or "Em repouso") for state, text in LABELS.items()]


def test_status_stages_and_tokens_drive_sphere(chat_ui):
    chat_ui(r"""
const states = [], kicks = [];
window.cortexCore = {setLive: state => states.push(state), pulseToken: () => kicks.push(true)};
for (const [stage, state] of [
  ['thinking', 'thinking'], ['web_search', 'searching'], ['reading', 'reading'],
  ['writing', 'writing'], ['remembering', 'thinking'], ['unknown', 'thinking'],
  ['constructor', 'thinking'], ['__proto__', 'thinking'], [undefined, 'thinking']
]) {
  await emit({type: 'status', stage, label: 'Detail stays in bubble'});
  assert.equal(states.at(-1), state, String(stage));
}
await emit({type: 'token', text: 'Hello'});
assert.equal(states.at(-1), 'speaking');
assert.equal(kicks.length, 1);
const count = states.length;
for (const stage of ['thinking', 'web_search', 'reading', 'writing', 'remembering', 'unknown']) {
  await emit({type: 'status', stage, label: 'late'});
  assert.equal(states.length, count, 'No state changes after first token');
}
await emit({type: 'token', text: ' world'}, {type: 'token', text: ''});
assert.equal(kicks.length, 2, 'Only actual text batches give kicks');
await emit({type: 'final', turn});
assert.equal(states.at(-1), 'idle');
await close();
assert.equal(states.at(-1), 'idle');
""")


@pytest.mark.parametrize("state", ["searching", "reading", "writing"])
@pytest.mark.parametrize("light", [False, True], ids=["dark", "light"])
def test_new_activity_effects_move_and_preview_matches_live(core_ui, state, light):
    core_ui(f"light = {json.dumps(light)}; const state = '{state}';\n" + r"""
core.setLive('idle');
const idle = snapshot();
core.setLive(state);
assert.notEqual(snapshot(), idle, 'Activity changes the drawing');
const live = snapshot();
core.preview(state);
assert.equal(snapshot(), live, 'Preview uses the same activity effect');
tick(1000); tick(1064);
assert.notEqual(snapshot(), live, 'Activity animates');
for (let now = 1128; now <= 1448; now += 64) tick(now);
// The sweep and the band are the near halves of real 3D paths (task 156):
// find each by its own stroke width and read the points of that path.
const path = lineWidth => {
  const end = ops.findIndex(op => op[0] === 'lineWidth' && Math.abs(op[1] - lineWidth) < 1e-9);
  if (end < 0) return null;
  const start = ops.slice(0, end).map(op => op[0]).lastIndexOf('beginPath');
  const moves = ops.slice(start, end).filter(op => op[0] === 'moveTo');
  return moves.length ? moves : null;
};
const meanY = moves => moves.reduce((sum, op) => sum + op[2], 0) / moves.length;
const cue = () => state === 'searching' ? path(1.9)
  : state === 'reading' ? path(320*.365*.88*.04)
    : ops.filter(op => op[0] === 'arc' && op[3] === 1.2);
const before = cue();
assert.ok(before, 'Activity has its own visible cue');
tick(1512);
const after = cue();
if (state === 'searching') assert.notDeepEqual(after, before, 'Meridian sweeps round the sphere');
else if (state === 'reading') assert.ok(meanY(after) > meanY(before), 'Scanner moves down');
else {
  assert.equal(after.length, 21, 'Writing triples the travelling points');
  assert.notDeepEqual(after, before, 'Writing points travel');
}
""")


@pytest.mark.parametrize("preference", ["system", "app"])
@pytest.mark.parametrize("light", [False, True], ids=["dark", "light"])
def test_reduced_activity_cues_are_distinct_and_do_not_advance(core_ui, preference, light):
    core_ui(f"light = {json.dumps(light)};\n" + r"""
core.setReduced(true);
const pictures = new Set();
let now = 1000;
for (const state of ['idle', 'thinking', 'searching', 'reading', 'writing', 'listening', 'speaking']) {
  core.setLive(state);
  const fixed = snapshot();
  assert.ok(!pictures.has(fixed), 'Distinct static cue: ' + state);
  pictures.add(fixed);
  const before = draws;
  tick(now += 200); tick(now += 200); tick(now += 200);
  assert.equal(draws, before + 3, 'Exercise actual reduced frames');
  assert.equal(snapshot(), fixed, 'Reduced motion freezes ' + state);
  if (state === 'speaking') {
    core.pulseToken();
    tick(now += 200);
    assert.equal(snapshot(), fixed, 'Reduced motion ignores token kicks');
  }
}
""".replace("core.setReduced(true);", "" if preference == "system" else "core.setReduced(true);"),
            system_reduced=preference == "system")


def test_token_pulse_is_bounded_decays_and_resets(core_ui):
    core_ui(r"""
core.setLive('speaking'); tick(1000); tick(1064);
assert.equal(scale(), 1, 'Live speaking has no synthetic sine pulse');
const before = draws;
core.pulseToken();
assert.equal(draws, before, 'Kicks wait for the existing frame loop');
tick(1096);
const kicked = scale();
assert.ok(kicked > 1 && kicked < 1.015, 'Small token kick');
tick(1160);
assert.ok(scale() > 1 && scale() < kicked, 'Kick decays without tokens');
for (let i = 0; i < 1000; i++) core.pulseToken();
tick(1224);
assert.ok(scale() <= 1.015, 'Bursts cannot accumulate unbounded energy');
core.setLive('idle'); core.setLive('speaking');
assert.equal(scale(), 1, 'New turn starts without leftover kick');
core.pulseToken(); core.setReduced(true); core.pulseToken();
core.setReduced(false); core.redraw();
assert.equal(scale(), 1, 'Reduced motion does not queue kicks');
core.preview('reading'); const preview = snapshot();
core.pulseToken(); core.redraw();
assert.equal(snapshot(), preview, 'Live tokens do not alter a preview');
core.preview(null);
assert.equal(scale(), 1);
core.pulseToken(); visible(false); core.pulseToken();
assert.equal(frames.size, 0, 'Hidden page cancels its frame');
visible(true); tick(2000);
assert.equal(scale(), 1, 'Hidden token batches do not queue kicks');
""")


def test_frame_caps_and_hidden_pause(core_ui):
    core_ui(r"""
core.setLive('searching'); tick(1000);
const first = draws;
tick(1016); assert.equal(draws, first);
tick(1032); assert.equal(draws, first + 1);
core.setReduced(true);
const reduced = draws;
tick(1100); assert.equal(draws, reduced);
tick(1212); assert.equal(draws, reduced + 1);
const fixed = snapshot();
visible(false); tick(1500);
assert.equal(draws, reduced + 1);
assert.equal(frames.size, 0);
visible(true); tick(2000);
assert.equal(snapshot(), fixed, 'Pause must not advance the frozen sweep');
""")
