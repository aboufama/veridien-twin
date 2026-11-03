// The one-view page's state and input mapping (docs/WEB_TWIN.md §18.2), in numbers. Run: node --test web/twin/view/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as S from '../state.js';
import { Controls } from '../../control/input.js';
import { defaultState, forward, port, spec, ensureLenses, projectD, posesFromState, scopeDov } from '../../control/kinematics.js';
import { Limits } from '../../control/limits.js';
import { viewBasis, scopeBasisFromPose, pinholeOffset } from '../../control/view_basis.js';
import { sceneDirs, loadSceneDir } from '../../tests/control_helpers.mjs';

const ALL = ['SCOPE', 'CAM1', 'CAM2', 'CAM3'];

test('default main view: SCOPE when the renderer has it, else CAM1; ?view= wins when available', () => {
  assert.equal(S.initialState({ available: ALL }).main, 'SCOPE');
  assert.equal(S.initialState({ available: ['CAM1', 'CAM2', 'CAM3'] }).main, 'CAM1');
  assert.equal(S.initialState({ available: ALL, requested: 'CAM2' }).main, 'CAM2');
  assert.equal(S.initialState({ available: ['CAM1', 'CAM2', 'CAM3'], requested: 'SCOPE' }).main, 'CAM1');
  const s = S.initialState({ available: ALL });
  assert.equal(s.grid, false); assert.equal(s.help, false); assert.equal(s.markers, false);
});

test('V toggles the grid, a tile click makes it main and closes the grid, Esc closes help first then the grid', () => {
  let s = S.initialState({ available: ALL });
  s = S.reduce(s, S.keyAction({ code: 'KeyV' }));
  assert.equal(s.grid, true);
  assert.deepEqual(S.visibleViews(s).map((v) => v.name), ALL);
  s = S.reduce(s, { type: 'pick', name: 'CAM3' });
  assert.equal(s.main, 'CAM3'); assert.equal(s.grid, false);
  assert.deepEqual(S.visibleViews(s), [{ slot: 'main', name: 'CAM3', drawable: true }]);
  s = S.reduce(S.reduce(s, { type: 'toggleGrid' }), { type: 'toggleHelp' });
  assert.equal(s.grid && s.help, true);
  s = S.reduce(s, S.keyAction({ code: 'Escape' }));
  assert.equal(s.help, false); assert.equal(s.grid, true);
  s = S.reduce(s, S.keyAction({ code: 'Escape' }));
  assert.equal(s.grid, false);
  const same = S.reduce(s, S.keyAction({ code: 'Escape' }));
  assert.equal(same, s, 'Esc with nothing open changes nothing');
});

test('a tile the renderer cannot draw cannot become main; it shows as not drawable in the grid', () => {
  let s = S.initialState({ available: ['CAM1', 'CAM2', 'CAM3'], grid: true });
  assert.deepEqual(S.visibleViews(s).map((v) => v.drawable), [false, true, true, true]);
  const t = S.reduce(s, { type: 'pick', name: 'SCOPE' });
  assert.equal(t, s);
});

test('M toggles markers only while a model is attached', () => {
  const s = S.initialState({ available: ALL });
  assert.equal(S.reduce(s, { ...S.keyAction({ code: 'KeyM' }), modelAttached: false }).markers, false);
  assert.equal(S.reduce(s, { ...S.keyAction({ code: 'KeyM' }), modelAttached: true }).markers, true);
});

test('page keys: ? and H open help, A autoplay, modifiers and repeats are ignored, tool keys are left to control/input.js', () => {
  assert.equal(S.keyAction({ code: 'Slash', shiftKey: true, key: '?' }).type, 'toggleHelp');
  assert.equal(S.keyAction({ code: 'KeyH' }).type, 'toggleHelp');
  assert.equal(S.keyAction({ code: 'KeyA' }).type, 'autoplay');
  assert.equal(S.keyAction({ code: 'KeyV', metaKey: true }), null);
  assert.equal(S.keyAction({ code: 'KeyV', repeat: true }), null);
  for (const code of ['Digit1', 'Digit2', 'Digit3', 'KeyW', 'KeyS', 'KeyQ', 'KeyE', 'KeyJ', 'KeyK', 'Space', 'ArrowLeft']) assert.equal(S.keyAction({ code }), null, code);
});

test('render sizes: main capped at 1280x720, tiles at 640x360, 16:9, even', () => {
  assert.deepEqual(S.renderSize(1420, 799, 1, S.MAIN_CAP), [1280, 720]);
  assert.deepEqual(S.renderSize(1420, 799, 2, S.MAIN_CAP), [1280, 720]);
  assert.deepEqual(S.renderSize(800, 450, 1, S.MAIN_CAP), [800, 450]);
  assert.deepEqual(S.renderSize(705, 397, 2, S.TILE_CAP), [640, 360]);
  assert.deepEqual(S.renderSize(400, 225, 1, S.TILE_CAP), [400, 224]);
  for (const [w, h] of [S.renderSize(1000, 700, 1, S.MAIN_CAP), S.renderSize(333, 600, 1, S.TILE_CAP)]) {
    assert.ok(Math.abs(w / h - 16 / 9) < 0.02, `${w}x${h}`); assert.equal(w % 2, 0); assert.equal(h % 2, 0);
  }
  const lb = S.letterbox(1420, 846);
  assert.equal(lb.w, 1420); assert.ok(Math.abs(lb.w / lb.h - 16 / 9) < 0.01); assert.equal(lb.y, Math.floor((846 - lb.h) / 2));
});

test('renderer plan: the asked one first, then lite candidates needing SCOPE, then trace, then stub; unknown names ignored', () => {
  assert.deepEqual(S.rendererPlan(null).map((x) => x.kind), ['lite_nodes', 'lite', 'lite_pt', 'lite_prt', 'lite_rt', 'trace', 'stub']);
  assert.ok(S.rendererPlan(null).slice(0, 5).every((x) => x.needScope));
  assert.ok(S.rendererPlan(null).slice(5).every((x) => !x.needScope));
  // how each draws: the tracer's present() takes a texture, so the page blits; lite_rt senses inside present()
  assert.equal(S.presentMode('trace', { present() {} }), 'blit');
  assert.equal(S.presentMode('lite_nodes', { present() {} }), 'sense+present');
  assert.equal(S.presentMode('lite_rt', { present() {} }), 'present');
  assert.equal(S.presentMode('lite', { present() {}, presentSenses: true }), 'present');
  assert.equal(S.presentMode('lite', {}), 'blit');
  const p = S.rendererPlan('lite_pt');
  assert.deepEqual(p[0], { kind: 'lite_pt', needScope: false });
  assert.equal(p.filter((x) => x.kind === 'lite_pt').length, 1);
  assert.deepEqual(S.rendererPlan('stub').map((x) => x.kind)[0], 'stub');
  assert.deepEqual(S.rendererPlan('bogus').map((x) => x.kind)[0], 'lite_nodes');
  assert.deepEqual(S.rendererCameras({ cameras: ['SCOPE', 'CAM1'] }), ['SCOPE', 'CAM1']);
  assert.deepEqual(S.rendererCameras({ cameras: { CAM1: {}, CAM2: {} } }), ['CAM1', 'CAM2']);
  assert.deepEqual(S.rendererCameras({ cams: { CAM3: {} } }), ['CAM3']);
});

test('display pacing: one frame in flight; full rate while moving; stillFps when still', () => {
  assert.equal(S.frameDue({ now: 100, lastAt: 0, busy: true, moving: true }), false);
  assert.equal(S.frameDue({ now: 100, lastAt: 99, busy: false, moving: true }), true);
  assert.equal(S.frameDue({ now: 150, lastAt: 100, busy: false, moving: false, stillFps: 10 }), false);
  assert.equal(S.frameDue({ now: 201, lastAt: 100, busy: false, moving: false, stillFps: 10 }), true);
  assert.equal(S.frameDue({ now: 101, lastAt: 100, busy: false, moving: false, dirty: true }), true);
});

test('status line: empty when nothing is wrong; the stub and a trace fallback are named', () => {
  assert.equal(S.statusText({ kind: 'lite_nodes' }), '');
  assert.match(S.statusText({ kind: 'stub' }), /Stub renderer/);
  assert.match(S.statusText({ kind: 'trace', fallback: true }), /Path tracer/);
  assert.equal(S.statusText({ kind: 'trace', fallback: false }), '');
  assert.match(S.statusText({ kind: 'lite', modelWanted: true, socket: 'connecting' }), /socket/);
});

// ---- input mapping through the page's wiring: the screen basis of the CURRENT main view

const dir = sceneDirs()[0];
const { scene } = loadSceneDir(dir, { meshes: false });
const limits = new Limits(scene, {});
const s0 = limits.clampState(defaultState(scene)).state;
const tipOf = (st, id) => {
  const ins = scene.instruments.find((i) => i.id === id);
  const p = port(scene, ins.port);
  return forward(p.pivot, p.R, spec(ins), st.instruments.find((q) => q.id === id)).tip;
};

function dragIn(viewName, tool, dx, dy) {
  let views = S.initialState({ available: ALL, requested: viewName });
  const c = new Controls({ surface: null, scene, keyTarget: new EventTarget(), orbit: () => {},
    screenBasis: () => viewBasis(scene, s0, views.main, { physics: null }) });
  c.select(tool);
  c.drag.dx = dx; c.drag.dy = dy;
  return limits.clampState(c.step(s0, 0.008)).state;
}

const angles = [];
for (const cam of ['CAM1', 'CAM2', 'CAM3']) {
  for (const tool of ['L', 'R']) {
    test(`drag right / up in ${cam} moves the ${tool} tip right / up in ${cam}'s image`, () => {
      const C = ensureLenses(scene).find((x) => x.name === cam);
      const u0 = projectD(C, tipOf(s0, tool));
      if (!u0) return;
      const uR = projectD(C, tipOf(dragIn(cam, tool, 40, 0), tool));
      const uU = projectD(C, tipOf(dragIn(cam, tool, 0, -40), tool));
      // the tip moves perpendicular to its shaft, so a foreshortened shaft adds cross-talk: the image motion must be
      // within 55 deg of the drag direction (cos >= 0.57), and its main component must have the drag's sign
      const cosTo = (d, [ex, ey]) => (d[0] * ex + d[1] * ey) / Math.hypot(d[0], d[1]);
      const dR = [uR[0] - u0[0], uR[1] - u0[1]], dU = [uU[0] - u0[0], uU[1] - u0[1]];
      assert.ok(cosTo(dR, [1, 0]) >= 0.57, `right: ${u0} -> ${uR} (cos ${cosTo(dR, [1, 0]).toFixed(2)})`);
      assert.ok(cosTo(dU, [0, -1]) >= 0.57, `up: ${u0} -> ${uU} (cos ${cosTo(dU, [0, -1]).toFixed(2)})`);
      angles.push(Math.acos(cosTo(dR, [1, 0])) * 180 / Math.PI, Math.acos(cosTo(dU, [0, -1])) * 180 / Math.PI);
    });
  }
}

test('report: the drag-vs-image angle in CAM1-3 (deg)', () => {
  if (angles.length) console.log(`# drag angle CAM1-3: median ${[...angles].sort((a, b) => a - b)[angles.length >> 1].toFixed(1)} deg, max ${Math.max(...angles).toFixed(1)} deg`);
});

test('drag right / up in the SCOPE view moves a grasper tip right / up in the scope image', () => {
  const T = posesFromState(scene, s0).bodies.scope;
  if (!T) return;
  const b = scopeBasisFromPose(T, scopeDov(scene, null));
  const o = [T[12], T[13], T[14]];
  for (const tool of ['L', 'R']) {
    const p0 = pinholeOffset(b, o, tipOf(s0, tool));
    if (!p0) continue;
    const pR = pinholeOffset(b, o, tipOf(dragIn('SCOPE', tool, 40, 0), tool));
    const pU = pinholeOffset(b, o, tipOf(dragIn('SCOPE', tool, 0, -40), tool));
    assert.ok(pR[0] > p0[0] && pR[0] - p0[0] > 2 * Math.abs(pR[1] - p0[1]), `${tool} right: ${p0} -> ${pR}`);
    assert.ok(pU[1] < p0[1] && p0[1] - pU[1] > 2 * Math.abs(pU[0] - p0[0]), `${tool} up: ${p0} -> ${pU}`);
  }
});

test('start state: the scope is re-aimed so both grasper tips are inside the scope image (was off the top edge)', async () => {
  const { aimScope } = await import('../aim.js');
  const { scene: sc, meshes } = loadSceneDir(dir, { meshes: true });
  const lim = new Limits(sc, meshes);
  const st0 = lim.clampState(defaultState(sc)).state;
  if (!st0.scope) return;
  const a = aimScope(sc, lim, st0);
  assert.ok(a.after < 1 * Math.PI / 180, `aim error ${(a.after * 180 / Math.PI).toFixed(2)} deg`);
  assert.deepEqual(lim.clampState(a.state).state.scope, a.state.scope, 'the aimed pose is within the limits');
  const T = posesFromState(sc, a.state).bodies.scope;
  const b = scopeBasisFromPose(T, scopeDov(sc, null));
  const f = 640 / Math.tan(37.5 * Math.PI / 180);            // 1280 px wide, 75 deg HFOV (physics.yaml, assumed)
  const { truthFromState } = await import('../../control/kinematics.js');
  const tr = truthFromState(sc, a.state);
  for (const ins of tr.instruments) {
    const p = pinholeOffset(b, tr.scope.lens_D, ins.tip_D, f);
    assert.ok(p && Math.hypot(p[0], p[1]) < 0.95 * 640 && Math.abs(p[1]) < 360, `${ins.id} tip at ${p}`);
  }
  console.log(`# scope aim: ${(a.before * 180 / Math.PI).toFixed(1)} deg -> ${(a.after * 180 / Math.PI).toFixed(3)} deg`);
});

test('?try= reorders the default lite search (trace and stub always last); GPU errors and notes reach the status line', () => {
  assert.deepEqual(S.rendererPlan(null, ['lite', 'lite_pt', 'stub', 'bogus']).map((x) => x.kind), ['lite', 'lite_pt', 'trace', 'stub']);
  assert.match(S.statusText({ kind: 'lite', gpuErrors: 5 }), /GPU errors \(5\)/);
  assert.equal(S.statusText({ kind: 'lite_pt', note: 'lite renderer had GPU errors; showing lite_pt.' }), 'lite renderer had GPU errors; showing lite_pt.');
});
