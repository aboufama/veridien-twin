// Bounding-volume hierarchy for the ray tracer: a binned-SAH BVH2 over triangles, built once in JS (or a worker),
// flattened to the layout the WGSL traversal reads (wgsl/bvh.wgsl.js), plus a CPU traversal of that same layout
// (closest and any hit) used by visible(), the gather-table checks and the tests (brute force is the judge).
//
// Node layout: 16 x 32-bit words per node, the two children's boxes stored in the parent:
//   [c0.min.xyz, c0.ref] [c0.max.xyz, c0.count] [c1.min.xyz, c1.ref] [c1.max.xyz, c1.count]
//   count > 0: the child is a leaf of `count` triangles starting at triangle `ref`; count = 0: interior node `ref`.
// No child is ever empty: a root that would be a single leaf is stored as two half leaves.
// Triangles are reordered so every leaf is a contiguous range; `order[k]` is the input index of slot k.

export const NODE_WORDS = 16;
const BIG = 1e30;

/**
 * Build a binned-SAH BVH. pos: Float32Array|Float64Array xyz per vertex; idx: Uint32Array, 3 per triangle.
 * Returns { nodes: Float32Array (u32 refs bit-cast in place), nodeCount, order: Uint32Array, bmin, bmax, ms }.
 */
export function buildBVH(pos, idx, { leafSize = 4, bins = 16, travCost = 1.0, triCost = 1.0 } = {}) {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const n = idx.length / 3;
  const tbmin = new Float32Array(n * 3), tbmax = new Float32Array(n * 3), cen = new Float32Array(n * 3);
  for (let t = 0; t < n; t++) {
    const a = idx[3 * t] * 3, b = idx[3 * t + 1] * 3, c = idx[3 * t + 2] * 3;
    for (let k = 0; k < 3; k++) {
      const lo = Math.min(pos[a + k], pos[b + k], pos[c + k]), hi = Math.max(pos[a + k], pos[b + k], pos[c + k]);
      tbmin[3 * t + k] = lo; tbmax[3 * t + k] = hi; cen[3 * t + k] = 0.5 * (lo + hi);
    }
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;

  let cap = Math.max(4, Math.ceil(n / Math.max(1, leafSize)) * 2);
  let nodes = new Float32Array(cap * NODE_WORDS);
  let nodesU = new Uint32Array(nodes.buffer);
  let nodeCount = 0;
  const alloc = () => {
    if (nodeCount >= cap) {
      cap *= 2;
      const nn = new Float32Array(cap * NODE_WORDS); nn.set(nodes); nodes = nn; nodesU = new Uint32Array(nodes.buffer);
    }
    return nodeCount++;
  };

  // per-bin scratch
  const bCount = new Uint32Array(bins), bMin = new Float64Array(bins * 3), bMax = new Float64Array(bins * 3);
  const rArea = new Float64Array(bins), rCount = new Uint32Array(bins);

  const rangeBounds = (s, e) => {
    const lo = [BIG, BIG, BIG], hi = [-BIG, -BIG, -BIG], clo = [BIG, BIG, BIG], chi = [-BIG, -BIG, -BIG];
    for (let i = s; i < e; i++) {
      const t = order[i];
      for (let k = 0; k < 3; k++) {
        const a = tbmin[3 * t + k], b = tbmax[3 * t + k], c = cen[3 * t + k];
        if (a < lo[k]) lo[k] = a; if (b > hi[k]) hi[k] = b;
        if (c < clo[k]) clo[k] = c; if (c > chi[k]) chi[k] = c;
      }
    }
    return { lo, hi, clo, chi };
  };
  const area = (lo, hi) => {
    const dx = Math.max(0, hi[0] - lo[0]), dy = Math.max(0, hi[1] - lo[1]), dz = Math.max(0, hi[2] - lo[2]);
    return dx * dy + dy * dz + dz * dx;
  };

  // Find the best SAH split of [s, e); returns the partition point or -1 for a leaf.
  const split = (s, e, B) => {
    const cnt = e - s;
    if (cnt <= 1) return -1;
    let best = Infinity, bestAxis = -1, bestBin = -1;
    const pA = area(B.lo, B.hi);
    for (let ax = 0; ax < 3; ax++) {
      const lo = B.clo[ax], ext = B.chi[ax] - lo;
      if (!(ext > 0)) continue;
      bCount.fill(0); bMin.fill(BIG); bMax.fill(-BIG);
      const sc = bins / ext;
      for (let i = s; i < e; i++) {
        const t = order[i];
        let b = Math.floor((cen[3 * t + ax] - lo) * sc); if (b >= bins) b = bins - 1;
        bCount[b]++;
        for (let k = 0; k < 3; k++) {
          if (tbmin[3 * t + k] < bMin[3 * b + k]) bMin[3 * b + k] = tbmin[3 * t + k];
          if (tbmax[3 * t + k] > bMax[3 * b + k]) bMax[3 * b + k] = tbmax[3 * t + k];
        }
      }
      // sweep from the right
      let lo3 = [BIG, BIG, BIG], hi3 = [-BIG, -BIG, -BIG], c = 0;
      for (let b = bins - 1; b > 0; b--) {
        c += bCount[b];
        for (let k = 0; k < 3; k++) { lo3[k] = Math.min(lo3[k], bMin[3 * b + k]); hi3[k] = Math.max(hi3[k], bMax[3 * b + k]); }
        rCount[b] = c; rArea[b] = c ? area(lo3, hi3) : 0;
      }
      lo3 = [BIG, BIG, BIG]; hi3 = [-BIG, -BIG, -BIG]; c = 0;
      for (let b = 0; b < bins - 1; b++) {
        c += bCount[b];
        for (let k = 0; k < 3; k++) { lo3[k] = Math.min(lo3[k], bMin[3 * b + k]); hi3[k] = Math.max(hi3[k], bMax[3 * b + k]); }
        if (c === 0 || rCount[b + 1] === 0) continue;
        const cost = travCost + triCost * (c * area(lo3, hi3) + rCount[b + 1] * rArea[b + 1]) / pA;
        if (cost < best) { best = cost; bestAxis = ax; bestBin = b; }
      }
    }
    if (bestAxis < 0) {
      // all centroids coincide: split in the middle of the range if it is too big for a leaf
      return cnt > leafSize * 4 ? s + (cnt >> 1) : -1;
    }
    if (cnt <= leafSize && best >= triCost * cnt) return -1;
    const lo = B.clo[bestAxis], sc = bins / (B.chi[bestAxis] - lo);
    let i = s, j = e - 1;
    while (i <= j) {
      const t = order[i];
      let b = Math.floor((cen[3 * t + bestAxis] - lo) * sc); if (b >= bins) b = bins - 1;
      if (b <= bestBin) i++;
      else { order[i] = order[j]; order[j] = t; j--; }
    }
    if (i === s || i === e) return s + (cnt >> 1);
    return i;
  };

  const writeChild = (node, slot, B, ref, count) => {
    const o = node * NODE_WORDS + slot * 8;
    nodes[o] = B.lo[0]; nodes[o + 1] = B.lo[1]; nodes[o + 2] = B.lo[2]; nodesU[o + 3] = ref;
    nodes[o + 4] = B.hi[0]; nodes[o + 5] = B.hi[1]; nodes[o + 6] = B.hi[2]; nodesU[o + 7] = count;
  };
  const root = alloc();
  const R = rangeBounds(0, n);
  const stack = [];   // [node, s, e, Bounds]
  const m0 = n ? split(0, n, R) : -1;
  if (n === 0) {
    // no triangles: two leaves whose boxes are a point at 1e30, which no ray reaches (never an interior self-loop)
    const far = { lo: [BIG, BIG, BIG], hi: [BIG, BIG, BIG] };
    writeChild(root, 0, far, 0, 1); writeChild(root, 1, far, 0, 1);
  } else if (m0 < 0) {
    // a root that is one leaf: two half leaves (the same one for a single triangle), so no child is ever empty
    const h = n > 1 ? n >> 1 : n;
    writeChild(root, 0, rangeBounds(0, h), 0, h);
    if (n > 1) writeChild(root, 1, rangeBounds(h, n), h, n - h); else writeChild(root, 1, R, 0, n);
  } else stack.push([root, 0, n, m0]);
  let maxDepth = 1;
  const depth = [1];
  while (stack.length) {
    const [node, s, e, m] = stack.pop();
    const d = depth.pop();
    maxDepth = Math.max(maxDepth, d);
    const ranges = [[s, m], [m, e]];
    for (let slot = 0; slot < 2; slot++) {
      const [a, b] = ranges[slot];
      const B = rangeBounds(a, b);
      const mm = split(a, b, B);
      if (mm < 0) writeChild(node, slot, B, a, b - a);
      else {
        const child = alloc();
        writeChild(node, slot, B, child, 0);
        stack.push([child, a, b, mm]); depth.push(d + 1);
      }
    }
  }
  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    nodes: nodes.slice(0, nodeCount * NODE_WORDS), nodeCount, order, triCount: n,
    bmin: R.lo, bmax: R.hi, maxDepth, ms: t1 - t0,
  };
}

/** Shift a BVH's node and triangle references so it can be appended at (nodeOff, triOff) of a combined buffer. */
export function relocate(nodes, nodeOff, triOff) {
  const out = new Float32Array(nodes);
  const u = new Uint32Array(out.buffer);
  for (let o = 0; o < u.length; o += 8) {
    if (u[o + 7] === 0) u[o + 3] += nodeOff;
    else u[o + 3] += triOff;
  }
  return out;
}

/** Triangle records for the tracer: 12 floats per BVH slot: v0.xyz, -, e1.xyz, -, e2.xyz, - (w words left for attributes). */
export function packTriangles(pos, idx, order, out = null, offset = 0) {
  const n = order.length;
  const T = out || new Float32Array(n * 16);
  for (let k = 0; k < n; k++) {
    const t = order[k], a = idx[3 * t] * 3, b = idx[3 * t + 1] * 3, c = idx[3 * t + 2] * 3;
    const o = (offset + k) * 16;
    T[o] = pos[a]; T[o + 1] = pos[a + 1]; T[o + 2] = pos[a + 2];
    T[o + 4] = pos[b] - pos[a]; T[o + 5] = pos[b + 1] - pos[a + 1]; T[o + 6] = pos[b + 2] - pos[a + 2];
    T[o + 8] = pos[c] - pos[a]; T[o + 9] = pos[c + 1] - pos[a + 1]; T[o + 10] = pos[c + 2] - pos[a + 2];
  }
  return T;
}

/** Moller-Trumbore on a packed record (16 floats per triangle). Returns t or Infinity; writes barycentrics. */
export function intersectTri(T, k, o, d, tmin, tmax, bary) {
  const b = k * 16;
  const e1x = T[b + 4], e1y = T[b + 5], e1z = T[b + 6], e2x = T[b + 8], e2y = T[b + 9], e2z = T[b + 10];
  const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-30) return Infinity;
  const inv = 1 / det;
  const sx = o[0] - T[b], sy = o[1] - T[b + 1], sz = o[2] - T[b + 2];
  const u = (sx * px + sy * py + sz * pz) * inv;
  if (u < 0 || u > 1) return Infinity;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
  if (v < 0 || u + v > 1) return Infinity;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  if (t <= tmin || t >= tmax) return Infinity;
  if (bary) { bary[0] = u; bary[1] = v; }
  return t;
}

function slab(N, o, inv, x, d, tmin, tmax) {
  // box of child at word offset x; returns entry t or Infinity
  let t0 = tmin, t1 = tmax;
  for (let k = 0; k < 3; k++) {
    let a = (N[x + k] - o[k]) * inv[k], b = (N[x + 4 + k] - o[k]) * inv[k];
    if (a > b) { const s = a; a = b; b = s; }
    if (a > t0) t0 = a; if (b < t1) t1 = b;
    if (t0 > t1) return Infinity;
  }
  return t0;
}

/**
 * Traverse a flattened BVH from `root` (closest hit, or any hit when anyHit). nodes: Float32Array, T: packed
 * triangles. Returns {t, tri, u, v} (tri = -1 on a miss).
 */
export function traverse(nodes, T, o, d, { tmin = 0, tmax = Infinity, anyHit = false, root = 0, filter = null } = {}) {
  const U = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
  const inv = [1 / d[0], 1 / d[1], 1 / d[2]];
  const bary = [0, 0];
  let best = tmax, tri = -1, bu = 0, bv = 0;
  const stack = [root];
  while (stack.length) {
    const node = stack.pop();
    const base = node * NODE_WORDS;
    const tc = [slab(nodes, o, inv, base, d, tmin, best), slab(nodes, o, inv, base + 8, d, tmin, best)];
    const push = [];
    for (let c = 0; c < 2; c++) {
      if (!(tc[c] < best)) continue;
      const ref = U[base + c * 8 + 3], cnt = U[base + c * 8 + 7];
      if (cnt > 0) {
        for (let k = ref; k < ref + cnt; k++) {
          if (filter && !filter(k)) continue;
          const t = intersectTri(T, k, o, d, tmin, best, bary);
          if (t < best) { best = t; tri = k; bu = bary[0]; bv = bary[1]; if (anyHit) return { t, tri, u: bu, v: bv }; }
        }
      } else push.push([tc[c], ref]);
    }
    push.sort((a, b) => b[0] - a[0]);          // far first onto the stack, near popped next
    for (const [, ref] of push) stack.push(ref);
  }
  return { t: tri >= 0 ? best : Infinity, tri, u: bu, v: bv };
}

/** Reference: test every triangle. */
export function bruteForce(T, count, o, d, { tmin = 0, tmax = Infinity } = {}) {
  const bary = [0, 0];
  let best = tmax, tri = -1, bu = 0, bv = 0;
  for (let k = 0; k < count; k++) {
    const t = intersectTri(T, k, o, d, tmin, best, bary);
    if (t < best) { best = t; tri = k; bu = bary[0]; bv = bary[1]; }
  }
  return { t: tri >= 0 ? best : Infinity, tri, u: bu, v: bv };
}

/** SAH cost of a flattened BVH (for reports). */
export function sahCost(bvh, travCost = 1, triCost = 1) {
  const N = bvh.nodes, U = new Uint32Array(N.buffer, N.byteOffset, N.length);
  const area = (o) => {
    const dx = N[o + 4] - N[o], dy = N[o + 5] - N[o + 1], dz = N[o + 6] - N[o + 2];
    return dx < 0 ? 0 : dx * dy + dy * dz + dz * dx;
  };
  const rootA = area(0) + 0;
  const ext = [bvh.bmax[0] - bvh.bmin[0], bvh.bmax[1] - bvh.bmin[1], bvh.bmax[2] - bvh.bmin[2]];
  const A0 = ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0] || rootA || 1;
  let cost = 0;
  for (let i = 0; i < bvh.nodeCount; i++) for (let c = 0; c < 2; c++) {
    const o = i * NODE_WORDS + 8 * c, a = area(o) / A0, cnt = U[o + 7];
    cost += cnt > 0 ? a * cnt * triCost : a * travCost;
  }
  return cost + travCost;
}
