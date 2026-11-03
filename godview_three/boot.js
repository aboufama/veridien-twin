// Shared boot for the god's-eye pages: scene bundle, physics (JSON from the server, else the judge harness copy) and
// godview.yaml (re-read at every load, since the reference agent fills its gain and exposure).
import { parseYaml } from './yaml_lite.js';
import { loadSceneBundle } from './godview.js';

const Q = new URLSearchParams(location.search);

async function tryFetch(urls, kind) {
  for (const u of urls) {
    if (!u) continue;
    try {
      const r = await fetch(u, { cache: 'no-store' });
      if (!r.ok) continue;
      if (kind === 'json') return { url: u, data: await r.json() };
      const text = await r.text();
      if (u.endsWith('.json')) return { url: u, data: JSON.parse(text) };
      return { url: u, data: parseYaml(text) };
    } catch { /* next */ }
  }
  return null;
}

export async function bootConfigs() {
  if (window.__GODVIEW_CONFIGS__) return window.__GODVIEW_CONFIGS__;
  const here = new URL('.', location.href);
  const rel = (p) => new URL(p, here).href;
  const [bundle, phys, gv] = await Promise.all([
    loadSceneBundle(Q.get('scene') || rel('../assets/')),
    tryFetch([Q.get('physics'), '/veridien-twin/configs/twin/physics.json', rel('../../../runs/webtwin/harness/physics.json'),
      '/runs/webtwin/harness/physics.json', rel('physics.snapshot.json')], 'json'),
    tryFetch([Q.get('godview'), '/veridien-twin/configs/twin/godview.json', '/veridien-twin/configs/twin/godview.yaml', rel('../../../configs/twin/godview.yaml'),
      rel('godview.snapshot.json')], 'auto'),
  ]);
  if (!phys) throw new Error('physics: no physics.json reachable');
  if (!gv) throw new Error('godview: no godview.yaml reachable');
  return { bundle, physics: phys.data, godview: gv.data, sources: { physics: phys.url, godview: gv.url } };
}
