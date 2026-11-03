// Material parameters and the element math of docs/JELLY.md ("energy", "viscosity", "fracture"). Pure, f64.

/** Mirror of the nominal column of docs/JELLY.md "material" (configs/twin/jelly.yaml, pass of 2026-11-01), used only
 * when the config is not loaded; every field is overridable. */
export const DEFAULT_PARAMS = Object.freeze({
  E: 25e3,            // Pa, Young's modulus (assumed, Fromageau 2003 per-cycle trend)
  nu: 0.49,           // Poisson's ratio (undrained on grasp timescales)
  rho: 1024,          // kg/m^3
  eta: 48.6,          // Pa s, Kelvin-Voigt shear viscosity (G''/omega at 1 Hz from the Prony terms)
  Gc: 100,            // J/m^2, fracture energy
  W0: 28e3,           // J/m^3, damage onset energy density (Neo-Hookean W at lambda_b = 3); null -> kappa * Gc / l_e
  kappa: 0.8,         // used only when W0 is null
  mu_s: 0.3,          // static friction, gel on PLA / bench (tools carry their own: steel 0.15 / 0.08)
  mu_k: 0.2,          // kinetic friction
  gravity: [0, 0, -9.81],
  floor_z: null,      // null: no floor; else the bench z in D
  substeps: 20,       // per step(dt)
  fracture: true,
  viscosity: true,
  maxDamageRate: 1,   // not used by the model; reserved
});

export function resolveParams(p = {}) {
  const q = { ...DEFAULT_PARAMS, ...p };
  q.mu = q.E / (2 * (1 + q.nu));
  q.lam = q.E * q.nu / ((1 + q.nu) * (1 - 2 * q.nu));
  q.lamp = q.lam + q.mu;                   // lam' (docs/JELLY.md energy)
  // Volumetric split against locking (docs/JELLY.md "discretisation"): lam' = lamTet + lamNode. The per-tet pair keeps
  // lamTet (with -mu (J-1), so F = I and F = R stay exact fixed points); lamNode acts on the nodal-averaged J_v.
  // nodalVolume = false puts all of lam' on the tets (the plain Macklin 2021 element, which locks at nu = 0.49).
  if (q.nodalVolume === undefined) q.nodalVolume = true;
  q.lamTet = q.nodalVolume ? Math.min(q.lamp, (q.lamTetOverMu ?? 1) * q.mu) : q.lamp;
  q.lamNode = q.lamp - q.lamTet;
  q.alphaRest = 1 + q.mu / q.lamTet;       // per-tet C_H = J - alphaRest
  return q;
}

/** Map configs/twin/jelly.yaml (parsed) to solver params; tolerant of the field names the config agent may use. */
export function paramsFromConfig(cfg) {
  if (!cfg) return {};
  const m = cfg.material ?? cfg;
  const pick = (...keys) => { for (const k of keys) { const v = k.split('.').reduce((o, s) => (o == null ? o : o[s]), m); if (v != null) return typeof v === 'object' && 'value' in v ? v.value : v; } return undefined; };
  const out = {};
  const set = (k, v) => { if (v !== undefined && v !== null) out[k] = Number(v); };
  set('E', pick('E', 'E_Pa', 'youngs_modulus', 'young_modulus_Pa', 'E_pa'));
  set('nu', pick('nu', 'poisson', 'poisson_ratio'));
  set('rho', pick('rho', 'density', 'rho_kg_m3', 'density_kg_m3'));
  set('Gc', pick('Gc', 'G_c', 'fracture_energy', 'Gc_J_m2', 'G_c_J_m2'));
  set('eta', pick('eta', 'viscosity', 'eta_Pa_s', 'kelvin_voigt_eta', 'viscosity_Pa_s'));
  const mu = pick('mu_friction', 'friction', 'mu_f');
  if (mu !== undefined) { out.mu_s = Number(mu); out.mu_k = Number(mu); }
  set('mu_s', pick('mu_static', 'mu_s'));
  set('mu_k', pick('mu_kinetic', 'mu_k'));
  set('kappa', pick('kappa', 'damage_onset'));
  set('W0', pick('W0', 'W0_J_m3', 'onset_energy', 'damage_onset_energy'));
  return out;
}

// ------------------------------------------------------------------ 3x3 helpers (row-major arrays of 9)

export function det3(m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
export function inv3(m) {
  const d = det3(m);
  const r = [
    m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
    m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
    m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]];
  for (let i = 0; i < 9; i++) r[i] /= d;
  return r;
}
export function mul3(a, b) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return r;
}
/** cof(F) = dJ/dF, row-major; columns c0 = f1 x f2, c1 = f2 x f0, c2 = f0 x f1 (f_j = columns of F). */
export function cof3(F) {
  const f0 = [F[0], F[3], F[6]], f1 = [F[1], F[4], F[7]], f2 = [F[2], F[5], F[8]];
  const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const c0 = cr(f1, f2), c1 = cr(f2, f0), c2 = cr(f0, f1);
  return [c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]];
}

/** Psi(F), J/m^3 (docs/JELLY.md energy). */
export function psi(F, P) {
  let Ic = 0; for (let i = 0; i < 9; i++) Ic += F[i] * F[i];
  const J = det3(F);
  return 0.5 * P.mu * (Ic - 3) - P.mu * (J - 1) + 0.5 * P.lamp * (J - 1) * (J - 1);
}
/** First Piola-Kirchhoff stress dPsi/dF, row-major. */
export function piola(F, P) {
  const J = det3(F), c = cof3(F);
  const k = -P.mu + P.lamp * (J - 1);
  return F.map((f, i) => P.mu * f + k * c[i]);
}

/** Largest eigenvalue of a symmetric 3x3 (row-major), closed form (Smith 1961). */
export function maxEigSym3(A) {
  const a = A[0], b = A[4], c = A[8], d = A[1], e = A[5], f = A[2];
  const p1 = d * d + e * e + f * f;
  const q = (a + b + c) / 3;
  if (p1 < 1e-30) return Math.max(a, b, c);
  const p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const B = [(a - q) / p, d / p, f / p, d / p, (b - q) / p, e / p, f / p, e / p, (c - q) / p];
  let r = det3(B) / 2;
  r = Math.max(-1, Math.min(1, r));
  const phi = Math.acos(r) / 3;
  return q + 2 * p * Math.cos(phi);
}

/** F^T F, row-major. */
export function rightCG(F) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = F[i] * F[j] + F[3 + i] * F[3 + j] + F[6 + i] * F[6 + j];
  return C;
}

/** Crack band (docs/JELLY.md fracture): damage from the history H and the band width l_e. */
export function damageOf(H, l_e, P) {
  const psic = P.Gc / l_e;
  const psi0 = P.W0 != null ? Math.min(P.W0, psic) : P.kappa * psic;   // W0 > psic: l_e too large (snap-back), fail at psic
  const Hf = 2 * psic - psi0;                                           // integral of H dd from psi0 to Hf = psic
  if (H <= psi0) return 0;
  if (H >= Hf) return 1;
  return (H - psi0) / (Hf - psi0);
}
