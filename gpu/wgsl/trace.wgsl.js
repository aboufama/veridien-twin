// WGSL ray queries: BVH2 traversal of the layout in gpu/bvh.js (both child boxes in the parent, ordered descent,
// a short stack with entry distances), the static scene plus every posed body BLAS (ray moved into the body frame
// at the ray's own time in the exposure: slerp + lerp), and the surface record at a hit.
// `override CUTAWAY` drops static hits beyond the overview's cut plane.
export const TRACE = /* wgsl */ `
override CUTAWAY: bool = false;

struct Hit { t: f32, tri: u32, u: f32, v: f32, inst: u32 }

// Traversal stacks live in workgroup memory (on-chip), STACK entries per invocation; every entry point that
// traces sets gBase = local_invocation_index * STACK. All tracing kernels use 64 invocations per workgroup.
const STACK: u32 = 32u;
const MAX_VISITS: u32 = 2048u;         // hard cap on node visits per traversal (a typical ray needs 50-400)
const MAX_LEAF: u32 = 64u;             // hard cap on triangles per leaf (the builder makes <= 4, rarely 16)
var<workgroup> wstackN: array<u32, 2048>;
var<workgroup> wstackT: array<f32, 2048>;
var<private> gBase: u32 = 0u;
var<private> gRoot: u32 = 0u;          // static BVH root: the interior-only BVH for camera feeds and lighting, 0 = full

fn slab(lo: vec3f, hi: vec3f, o: vec3f, inv: vec3f, tmin: f32, tmax: f32) -> f32 {
  let t0 = (lo - o) * inv;
  let t1 = (hi - o) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  let a = max(max(tn.x, tn.y), max(tn.z, tmin));
  let b = min(min(tf.x, tf.y), min(tf.z, tmax));
  return select(BIG, a, a <= b);
}

fn safeInv(d: vec3f) -> vec3f {
  let s = select(vec3f(-1.0), vec3f(1.0), d >= vec3f(0.0));
  let m = max(abs(d), vec3f(1e-20));
  return s / m;
}

// Moller-Trumbore; returns (t, u, v) or t = BIG
fn triHit(k: u32, o: vec3f, d: vec3f, tmin: f32, tmax: f32) -> vec3f {
  let v0 = tris[4u * k].xyz;
  let e1 = tris[4u * k + 1u].xyz;
  let e2 = tris[4u * k + 2u].xyz;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-30) { return vec3f(BIG, 0.0, 0.0); }
  let inv = 1.0 / det;
  let s = o - v0;
  let u = dot(s, p) * inv;
  if (u < 0.0 || u > 1.0) { return vec3f(BIG, 0.0, 0.0); }
  let q = cross(s, e1);
  let v = dot(d, q) * inv;
  if (v < 0.0 || u + v > 1.0) { return vec3f(BIG, 0.0, 0.0); }
  let t = dot(e2, q) * inv;
  if (t <= tmin || t >= tmax) { return vec3f(BIG, 0.0, 0.0); }
  return vec3f(t, u, v);
}

// Traverse one BVH from root. o, d in that BVH's frame. Updates best; returns true on any hit when anyHit.
fn traverse(root: u32, o: vec3f, d: vec3f, tmin: f32, best: ptr<function, Hit>, anyHit: bool, inst: u32, wo: vec3f, wd: vec3f) -> bool {
  let inv = safeInv(d);
  var sp: u32 = 0u;
  var node = root;
  var visits = 0u;
  loop {
    visits++;
    if (visits > MAX_VISITS) { atomicAdd(&dbg[0], 1u); break; }      // cap hit: flag it, never hang
    let b = node * 4u;
    let a0 = nodes[b];
    let a1 = nodes[b + 1u];
    let a2 = nodes[b + 2u];
    let a3 = nodes[b + 3u];
    var t0 = slab(a0.xyz, a1.xyz, o, inv, tmin, (*best).t);
    var t1 = slab(a2.xyz, a3.xyz, o, inv, tmin, (*best).t);
    let c0 = bitcast<u32>(a1.w);
    let c1 = bitcast<u32>(a3.w);
    let r0 = bitcast<u32>(a0.w);
    let r1 = bitcast<u32>(a2.w);
    for (var c = 0u; c < 2u; c++) {
      let tc = select(t1, t0, c == 0u);
      let cnt = select(c1, c0, c == 0u);
      let ref0 = select(r1, r0, c == 0u);
      if (tc < (*best).t && cnt > 0u) {
        if (cnt > MAX_LEAF) { atomicAdd(&dbg[1], 1u); }
        for (var k = ref0; k < ref0 + min(cnt, MAX_LEAF); k++) {
          let h = triHit(k, o, d, tmin, (*best).t);
          if (h.x < (*best).t) {
            if (CUTAWAY && inst == MISS) {
              let pw = wo + wd * h.x;
              if (dot(scn.cut.xyz, pw) > scn.cut.w) { continue; }
            }
            *best = Hit(h.x, k, h.y, h.z, inst);
            if (anyHit) { return true; }
          }
        }
        if (c == 0u) { t0 = BIG; } else { t1 = BIG; }
      }
    }
    t0 = select(BIG, t0, t0 < (*best).t);
    t1 = select(BIG, t1, t1 < (*best).t);
    // children are always stored after their parent (gpu/bvh.js); anything else is a corrupt tree: stop, flag
    if ((t0 < BIG && c0 == 0u && r0 <= node) || (t1 < BIG && c1 == 0u && r1 <= node)) { atomicAdd(&dbg[6], 1u); break; }
    let go0 = t0 < BIG;
    let go1 = t1 < BIG;
    if (go0 && go1) {
      if (t1 < t0) { wstackN[gBase + sp] = r0; wstackT[gBase + sp] = t0; node = r1; }
      else { wstackN[gBase + sp] = r1; wstackT[gBase + sp] = t1; node = r0; }
      sp = min(sp + 1u, STACK - 1u);
    } else if (go0) { node = r0; }
    else if (go1) { node = r1; }
    else {
      var found = false;
      for (var pops = 0u; pops <= STACK; pops++) {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        if (wstackT[gBase + sp] < (*best).t) { node = wstackN[gBase + sp]; found = true; break; }
      }
      if (!found) { break; }
    }
  }
  return false;
}

// Does the ray segment o + t d, t in [t0, t1], pass within r of the segment a-b? (closest points of two segments)
fn segNearCapsule(o: vec3f, d: vec3f, t0: f32, t1: f32, a: vec3f, b: vec3f, r: f32) -> bool {
  let p0 = o + d * t0;
  let u = d * (t1 - t0);
  let v = b - a;
  let w = p0 - a;
  let A = dot(u, u); let B = dot(u, v); let C = dot(v, v); let D = dot(u, w); let E = dot(v, w);
  let den = A * C - B * B;
  var s = 0.0;
  if (den > 1e-20 * A * C) { s = clamp((B * E - C * D) / den, 0.0, 1.0); }
  var t = select(0.0, (B * s + E) / C, C > 1e-30);
  if (t < 0.0) { t = 0.0; s = clamp(-D / max(A, 1e-30), 0.0, 1.0); }
  else if (t > 1.0) { t = 1.0; s = clamp((B - D) / max(A, 1e-30), 0.0, 1.0); }
  let q = (p0 + u * s) - (a + v * t);
  return dot(q, q) <= r * r;
}

// mask: 1 = static scene, 2 = posed bodies
fn traceScene(o: vec3f, d: vec3f, tmin: f32, tmax: f32, time: f32, anyHit: bool, mask: u32) -> Hit {
  return traceSceneSkip(o, d, tmin, tmax, time, anyHit, mask, 0u);
}

// skip: bit i set = ignore body i (a camera's own lens front, for rays from that camera's centre)
fn traceSceneSkip(o: vec3f, d: vec3f, tmin: f32, tmax: f32, time: f32, anyHit: bool, mask: u32, skip: u32) -> Hit {
  var best = Hit(tmax, MISS, 0.0, 0.0, MISS);
  if ((mask & 1u) != 0u) {
    if (traverse(gRoot, o, d, tmin, &best, anyHit, MISS, o, d)) { return best; }
  }
  if ((mask & 2u) != 0u) {
    let inv = safeInv(d);
    let n = min(scn.info.x, MAX_INST);
    for (var i = 0u; i < n; i++) {
      let I = scn.inst[i];
      if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((skip >> i) & 1u) != 0u) { continue; }
      if (slab(I.wmin.xyz, I.wmax.xyz, o, inv, tmin, best.t) >= best.t) { continue; }
      if (!segNearCapsule(o, d, tmin, min(best.t, 1e3), I.bcA.xyz, I.bcB.xyz, I.bcA.w)) { continue; }
      let P = instPose(i, time);
      let ol = (o - P.p) * P.r;           // R^T (o - p)
      let dl = d * P.r;
      if (traverse(bitcast<u32>(I.p0.w), ol, dl, tmin, &best, anyHit, i, o, d)) { return best; }
    }
  }
  return best;
}

struct Surf {
  p: vec3f, ng: vec3f, ns: vec3f,
  b: vec3f, vi: vec3u,
  mesh: u32, flags: u32, mat: u32, inst: u32,
}

fn surface(h: Hit, o: vec3f, d: vec3f, time: f32) -> Surf {
  let k = h.tri;
  let r0 = tris[4u * k];
  let r1 = tris[4u * k + 1u];
  let r2 = tris[4u * k + 2u];
  let at = tris[4u * k + 3u];
  let vi = vec3u(bitcast<u32>(r0.w), bitcast<u32>(r1.w), bitcast<u32>(r2.w));
  let mesh = bitcast<u32>(at.x);
  let flags = bitcast<u32>(at.y);
  let b = vec3f(1.0 - h.u - h.v, h.u, h.v);
  var ng = normalize(cross(r1.xyz, r2.xyz));
  var ns = normalize(b.x * verts[2u * vi.x + 1u].xyz + b.y * verts[2u * vi.y + 1u].xyz + b.z * verts[2u * vi.z + 1u].xyz);
  if (h.inst != MISS) {
    let P = instPose(h.inst, time);
    ng = P.r * ng;
    ns = P.r * ns;
  }
  // two-sided: face the incoming ray; keep the shading normal on the same side
  if (dot(ng, d) > 0.0) { ng = -ng; }
  if (dot(ns, ng) < 0.0) { ns = -ns; }
  if (dot(ns, -d) < 1e-4) { ns = ng; }
  let mt = params[MESH_OFF + 2u * mesh];
  return Surf(o + d * h.t, ng, ns, b, vi, mesh, flags, bitcast<u32>(mt.w), h.inst);
}

fn offsetRay(p: vec3f, ng: vec3f, dir: vec3f) -> vec3f {
  let s = select(-1.0, 1.0, dot(ng, dir) > 0.0);
  return p + ng * (s * EPS_OFF);
}
`;
