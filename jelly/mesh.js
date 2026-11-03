// Tet mesher for the jelly block (docs/JELLY.md). Pure ES module (node + browser).
// A regular grid of nx x ny x nz cubes, each split into the 6 Kuhn (Freudenthal) tets that share the cube's main
// diagonal. The same split in every cube makes the mesh conforming; every tet is oriented det(Dm) > 0.
// l_e (the crack-band width) is the grid spacing: a crack through the grid removes one layer of cubes.

/** gridBlock({size: [Lx, Ly, Lz] m, n: [nx, ny, nz], origin: [x, y, z] (min corner)}) ->
 *  {X: Float64Array(3N) rest positions, tets: Uint32Array(4T), l_e: Float64Array(T), h: [hx, hy, hz], n, size, origin} */
export function gridBlock({ size, n, origin = [0, 0, 0] }) {
  const [nx, ny, nz] = n;
  const hx = size[0] / nx, hy = size[1] / ny, hz = size[2] / nz;
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1;
  const N = NX * NY * NZ;
  const X = new Float64Array(3 * N);
  const vid = (i, j, k) => (k * NY + j) * NX + i;
  for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const v = vid(i, j, k);
    X[3 * v] = origin[0] + i * hx; X[3 * v + 1] = origin[1] + j * hy; X[3 * v + 2] = origin[2] + k * hz;
  }
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const T = 6 * nx * ny * nz;
  const tets = new Uint32Array(4 * T);
  const l_e = new Float64Array(T);
  const le = Math.cbrt(hx * hy * hz);
  let t = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    for (const p of perms) {
      const c = [i, j, k];
      const v0 = vid(c[0], c[1], c[2]);
      c[p[0]]++; const v1 = vid(c[0], c[1], c[2]);
      c[p[1]]++; const v2 = vid(c[0], c[1], c[2]);
      c[p[2]]++; const v3 = vid(c[0], c[1], c[2]);
      const q = [v0, v1, v2, v3];
      if (signedVolume(X, q) < 0) { const s = q[2]; q[2] = q[3]; q[3] = s; }
      tets.set(q, 4 * t); l_e[t] = le; t++;
    }
  }
  return { X, tets, l_e, h: [hx, hy, hz], n: [nx, ny, nz], size: [...size], origin: [...origin] };
}

export function signedVolume(X, q) {
  const a = q[0] * 3, b = q[1] * 3, c = q[2] * 3, d = q[3] * 3;
  const e1 = [X[b] - X[a], X[b + 1] - X[a + 1], X[b + 2] - X[a + 2]];
  const e2 = [X[c] - X[a], X[c + 1] - X[a + 1], X[c + 2] - X[a + 2]];
  const e3 = [X[d] - X[a], X[d + 1] - X[a + 1], X[d + 2] - X[a + 2]];
  return (e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])) / 6;
}

/** Faces of a positively oriented tet, each opposite vertex f, wound outward. */
export const TET_FACES = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/** Per-tet face neighbours: nbr[4t + f] = the tet across face f, or -1 on the block's boundary. */
export function faceNeighbours(tets) {
  const T = tets.length / 4;
  const nbr = new Int32Array(4 * T).fill(-1);
  const map = new Map();
  for (let t = 0; t < T; t++) for (let f = 0; f < 4; f++) {
    const fv = TET_FACES[f].map((l) => tets[4 * t + l]).sort((a, b) => a - b);
    const key = fv[0] + ',' + fv[1] + ',' + fv[2];
    const o = map.get(key);
    if (o === undefined) map.set(key, 4 * t + f);
    else { nbr[4 * t + f] = o >> 2; nbr[o] = t; map.delete(key); }
  }
  return nbr;
}

/** Vertex -> incident tets, CSR: {start: Uint32Array(N+1), list: Uint32Array(4T) of (4 t + local)}. */
export function vertexTets(tets, N) {
  const T = tets.length / 4;
  const count = new Uint32Array(N + 1);
  for (let i = 0; i < 4 * T; i++) count[tets[i] + 1]++;
  for (let v = 0; v < N; v++) count[v + 1] += count[v];
  const fill = count.slice(0, N);
  const list = new Uint32Array(4 * T);
  for (let t = 0; t < T; t++) for (let l = 0; l < 4; l++) list[fill[tets[4 * t + l]]++] = 4 * t + l;
  return { start: count, list };
}

/** Greedy vertex-disjoint colouring (tets sharing a vertex get different colours), then the tets reordered so each
 * colour is one contiguous range. Returns {tets, l_e, order (old index of each new tet), colourStart: Uint32Array(C+1)}. */
export function colourTets(tets, l_e, N) {
  const T = tets.length / 4;
  const vt = vertexTets(tets, N);
  const colour = new Int32Array(T).fill(-1);
  const used = new Uint8Array(256);
  let C = 0;
  for (let t = 0; t < T; t++) {
    used.fill(0);
    for (let l = 0; l < 4; l++) {
      const v = tets[4 * t + l];
      for (let k = vt.start[v]; k < vt.start[v + 1]; k++) { const c = colour[vt.list[k] >> 2]; if (c >= 0) used[c] = 1; }
    }
    let c = 0; while (used[c]) c++;
    colour[t] = c; if (c + 1 > C) C = c + 1;
  }
  const colourStart = new Uint32Array(C + 1);
  for (let t = 0; t < T; t++) colourStart[colour[t] + 1]++;
  for (let c = 0; c < C; c++) colourStart[c + 1] += colourStart[c];
  const fill = colourStart.slice(0, C);
  const order = new Uint32Array(T);
  for (let t = 0; t < T; t++) order[fill[colour[t]]++] = t;
  const tets2 = new Uint32Array(4 * T), le2 = new Float64Array(T);
  for (let i = 0; i < T; i++) { const t = order[i]; tets2.set(tets.subarray(4 * t, 4 * t + 4), 4 * i); le2[i] = l_e[t]; }
  return { tets: tets2, l_e: le2, order, colourStart };
}

/** Rigidly place a block: rotate about z by yaw, then translate. Returns a new X. */
export function placeBlock(X, { yaw = 0, translate = [0, 0, 0] } = {}) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const Y = new Float64Array(X.length);
  for (let i = 0; i < X.length; i += 3) {
    Y[i] = c * X[i] - s * X[i + 1] + translate[0];
    Y[i + 1] = s * X[i] + c * X[i + 1] + translate[1];
    Y[i + 2] = X[i + 2] + translate[2];
  }
  return Y;
}

/** Zero-width cut (a notch): every vertex on the plane X[axis] = value that satisfies inCut(X) is duplicated, and the
 * tets on the + side of the plane (centroid > value) take the copy. Returns a new {X, tets, l_e, ...} (rest positions). */
export function cutPlane(g, axis, value, inCut, tol = 1e-12) {
  const N = g.X.length / 3, T = g.tets.length / 4;
  const dup = new Int32Array(N).fill(-1);
  const extra = [];
  for (let i = 0; i < N; i++) {
    const p = [g.X[3 * i], g.X[3 * i + 1], g.X[3 * i + 2]];
    if (Math.abs(p[axis] - value) <= tol && inCut(p)) { dup[i] = N + extra.length; extra.push(p); }
  }
  const X = new Float64Array(3 * (N + extra.length));
  X.set(g.X); extra.forEach((p, k) => X.set(p, 3 * (N + k)));
  const tets = Uint32Array.from(g.tets);
  for (let t = 0; t < T; t++) {
    let c = 0; for (let l = 0; l < 4; l++) c += g.X[3 * g.tets[4 * t + l] + axis] / 4;
    if (c <= value) continue;
    for (let l = 0; l < 4; l++) { const v = tets[4 * t + l]; if (dup[v] >= 0) tets[4 * t + l] = dup[v]; }
  }
  return { ...g, X, tets };
}
