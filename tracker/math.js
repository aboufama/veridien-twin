// Small numeric helpers for the in-browser tracker (web/twin/tracker/*.js). They mirror numpy's own arithmetic where the
// tracker's Python reference (src/veridien/multicam/tipnet.py, steady.py, shaftdir.py) depends on it: round-half-even
// (np.rint / np.round), pairwise summation (np.sum / np.nansum / np.mean, float64 and float32), np.linspace, np.arange,
// floor-mod (np.remainder) and a 3x3 LU solve with partial pivoting (np.linalg.solve). Pure; SI units throughout.

export const f32 = Math.fround;

/** numpy rint: round half to even. */
export function rint(x) {
  if (!Number.isFinite(x)) return x;
  const r = Math.round(x);
  return (r - x === 0.5 && r % 2 !== 0) ? r - 1 : r;   // Math.round rounds .5 up: pull odd ties back to even
}

/** np.round(x, n) for n >= 0: rint(x * 10^n) / 10^n. */
export function npRound(x, n) {
  const f = 10 ** n;
  return rint(x * f) / f;
}

/** Python's built-in round(x, n) on a float (correctly rounded decimal, ties to even): close enough via toFixed
 *  except for exact binary ties, which do not occur for the tracker's values. */
export function pyRound(x, n) {
  return x == null || !Number.isFinite(x) ? x : Number(x.toFixed(n));
}

function pairwise(a, lo, n, f) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f(res + a[lo + i]);
    return res;
  }
  if (n <= 128) {
    const r = [];
    for (let j = 0; j < 8; j++) r.push(a[lo + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = f(r[j] + a[lo + i + j]);
    let res = f(f(f(r[0] + r[1]) + f(r[2] + r[3])) + f(f(r[4] + r[5]) + f(r[6] + r[7])));
    for (; i < n; i++) res = f(res + a[lo + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f(pairwise(a, lo, n2, f) + pairwise(a, lo + n2, n - n2, f));
}
const id = (x) => x;

/** np.sum of a contiguous float64 sequence (numpy's pairwise summation). */
export const npSum = (a) => pairwise(a, 0, a.length, id);
/** The same in float32 (np.sum of a float32 array). */
export const npSum32 = (a) => pairwise(a, 0, a.length, f32);

/** np.linspace(start, stop, num) (endpoint: the last element is exactly stop). */
export function linspace(start, stop, num) {
  const out = new Array(num);
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) out[i] = i * step + start;
  if (num > 1) out[num - 1] = stop;
  return out;
}

/** np.arange(start, stop, step) for floats: start + i*step, length ceil((stop - start) / step). */
export function arange(start, stop, step) {
  const n = Math.max(0, Math.ceil((stop - start) / step));
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = start + i * step;
  return out;
}

/** np.remainder (Python floor-mod) on floats, as numpy's npy_divmod computes it. */
export function floorMod(a, b) {
  let mod = a % b;
  if (mod !== 0) {
    if ((b < 0) !== (mod < 0)) mod += b;
  } else mod = b < 0 ? -0 : 0;
  return mod;
}

/** np.radians. */
export const radians = (d) => d * (Math.PI / 180);

// ---- 3-vectors (plain arrays) ----
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a) => Math.sqrt(dot(a, a));
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const isFinite3 = (a) => Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]);
/** R (3x3) . x */
export const mulMv = (R, x) => [
  R[0][0] * x[0] + R[0][1] * x[1] + R[0][2] * x[2],
  R[1][0] * x[0] + R[1][1] * x[1] + R[1][2] * x[2],
  R[2][0] * x[0] + R[2][1] * x[1] + R[2][2] * x[2]];
/** R^T . x */
export const mulMtv = (R, x) => [
  R[0][0] * x[0] + R[1][0] * x[1] + R[2][0] * x[2],
  R[0][1] * x[0] + R[1][1] * x[1] + R[2][1] * x[2],
  R[0][2] * x[0] + R[1][2] * x[1] + R[2][2] * x[2]];

/** Solve the n x n system M x = b (LU with partial pivoting, like LAPACK dgesv). M is not modified. */
export function solveLinear(M, b) {
  const n = b.length;
  const A = M.map((r) => r.slice());
  const x = b.slice();
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(A[k][k]);
    for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > best) { best = Math.abs(A[i][k]); p = i; }
    if (p !== k) { [A[k], A[p]] = [A[p], A[k]]; [x[k], x[p]] = [x[p], x[k]]; }
    const piv = A[k][k];
    for (let i = k + 1; i < n; i++) {
      const l = A[i][k] / piv;
      if (l === 0) continue;
      for (let j = k; j < n; j++) A[i][j] -= l * A[k][j];
      x[i] -= l * x[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

/** Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi): {values (ascending), vectors (columns, unit)}. */
export function eigSym3(S) {
  const a = S.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-30 * (a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2 + 1e-300)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (a[p][q] === 0) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return { values: idx.map((i) => a[i][i]), vectors: idx.map((i) => [v[0][i], v[1][i], v[2][i]]) };
}
