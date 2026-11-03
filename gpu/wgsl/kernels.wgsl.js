// WGSL entry points and their bindings, assembled into full shader modules:
//   camera   - one dispatch per tracking camera (exact KB4 rays, box-integrated pixel, motion blur) or the
//              overview (pinhole, cutaway via the CUTAWAY override, optional viewing-aid light)
//   radiosity - gather-table build (once), per-frame direct light at the nodes, warm-started Jacobi iterations
//   display  - linear radiance -> 8-bit sRGB preview (overview and debugging only, not the camera chain)
//   tests    - the WGSL lens, BSDF, capsule occlusion and traversal exposed for tests/gpu_test.html
import { COMMON, defines } from './common.wgsl.js';
import { LENS } from './lens.wgsl.js';
import { BSDF } from './bsdf.wgsl.js';
import { TRACE } from './trace.wgsl.js';
import { SHADE, SHADE_CAMERA } from './shade.wgsl.js';

const G0 = (nodeStateAccess, dbgBinding) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> tris: array<vec4f>;
@group(0) @binding(2) var<storage, read> verts: array<vec4f>;
@group(0) @binding(3) var<storage, read> params: array<vec4f>;
@group(0) @binding(4) var<storage, ${nodeStateAccess}> nodeState: array<vec4f>;
@group(0) @binding(${dbgBinding}) var<storage, read_write> dbg: array<atomic<u32>, 16>;   // loop-cap counters
`;

const CAMERA_MAIN = /* wgsl */ `
@group(0) @binding(6) var boardTex: texture_2d<f32>;      // the ChArUco print mask (1 = paper), bilinear on the mesh uvs
@group(0) @binding(7) var boardSmp: sampler;
@group(1) @binding(0) var<uniform> scn: SceneU;
@group(1) @binding(1) var<uniform> cam: CamU;
@group(1) @binding(2) var outTex: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3) var<storage, read_write> acc1: array<vec4f>;    // pass 1: radiance sum (rgb), key (w bits)
@group(1) @binding(4) var<storage, read_write> hitCache: array<vec4f>; // pass-1 static first hits (t, tri, u, v) per sample
@group(1) @binding(5) var<storage, read_write> shadowCache: array<u32>; // pass-1 static occlusion bits per sample

struct PixSeeds { s0: f32, s1: f32, s2: f32, s3: f32, s4: f32, s5: f32, s6: f32, s7: f32, s8: f32 }

fn pixSeeds(x: u32, y: u32) -> PixSeeds {
  let hs = hash3(x, y, cam.size.w * 64u + cam.flags.x);
  return PixSeeds(u01(pcg(hs)), u01(pcg(hs + 1u)), u01(pcg(hs + 2u)), u01(pcg(hs + 3u)), u01(pcg(hs + 4u)),
                  u01(pcg(hs + 5u)), u01(pcg(hs + 6u)), u01(pcg(hs + 7u)), u01(pcg(hs + 8u)));
}

// sub-sample i of pixel (x, y): its own position in the pixel box, time in the exposure, NL light samples and a
// BSDF sample (Kronecker sequences with per-pixel shifts: stratified within the pixel, deterministic per seed)
fn subSample(x: u32, y: u32, i: u32, S: PixSeeds) -> vec3f {
  let uv = vec2f(f32(x) + kr(i, R2A, S.s0) - 0.5, f32(y) + kr(i, R2B, S.s1) - 0.5);
  let tt = mix(cam.expo.x, cam.expo.y, kr(i, GOLD, S.s2));
  var dc: vec3f;
  if (cam.pos.w < 0.5) { dc = kb4Unproject(Lens(cam.lensA, cam.lensB, cam.lensC), uv); }
  else { dc = normalize(vec3f((uv.x - cam.lensA.z) / cam.lensA.x, (uv.y - cam.lensA.w) / cam.lensA.y, 1.0)); }
  let d = normalize(cam.rx.xyz * dc.x + cam.ry.xyz * dc.y + cam.rz.xyz * dc.z);
  let nl = max(1u, bitcast<u32>(cam.lensC.z));
  let ul = vec3f(kr(i * nl, KL0, S.s3), kr(i * nl, KL1, S.s4), kr(i * nl, KL2, S.s5));
  let ub = vec3f(kr(i, KB0, S.s6), kr(i, KB1, S.s7), kr(i, KB2, S.s8));
  return radiance(cam.pos.xyz, d, tt, ul, ub, cam.flags.y, cam.flags.w, i * nl, S);
}

// The cameras are fixed and the static scene is static, so with a fixed sample pattern a sub-sample's static first
// hit never changes: it is traced once and kept (t = 0 marks an empty entry; the renderer clears the cache whenever
// the pattern or the camera changes). Exact.
fn staticFirstHit(o: vec3f, d: vec3f, tt: f32, ci: u32) -> Hit {
  if ((cam.flags.y & O_CACHE_WRITE) != 0u) {
    let c = hitCache[ci];
    if (c.x > 0.0) { return Hit(c.x, bitcast<u32>(c.y), c.z, c.w, MISS); }
    let h = traceSceneSkip(o, d, 0.0, BIG, tt, false, 1u, 0u);
    hitCache[ci] = vec4f(h.t, bitcast<f32>(h.tri), h.u, h.v);
    return h;
  }
  return traceSceneSkip(o, d, 0.0, BIG, tt, false, 1u, 0u);
}

// camera ray of sub-sample i (as subSample) with its first hit through the caches; returns the shaded radiance
fn cachedSubSample(px: vec2u, i: u32, S: PixSeeds) -> vec3f {
  let W = cam.size.x; let H = cam.size.y;
  let uv = vec2f(f32(px.x) + kr(i, R2A, S.s0) - 0.5, f32(px.y) + kr(i, R2B, S.s1) - 0.5);
  let tt = mix(cam.expo.x, cam.expo.y, kr(i, GOLD, S.s2));
  var dc: vec3f;
  if (cam.pos.w < 0.5) { dc = kb4Unproject(Lens(cam.lensA, cam.lensB, cam.lensC), uv); }
  else { dc = normalize(vec3f((uv.x - cam.lensA.z) / cam.lensA.x, (uv.y - cam.lensA.w) / cam.lensA.y, 1.0)); }
  let d = normalize(cam.rx.xyz * dc.x + cam.ry.xyz * dc.y + cam.rz.xyz * dc.z);
  let nl = max(1u, bitcast<u32>(cam.lensC.z));
  let ul = vec3f(kr(i * nl, KL0, S.s3), kr(i * nl, KL1, S.s4), kr(i * nl, KL2, S.s5));
  let ub = vec3f(kr(i, KB0, S.s6), kr(i, KB1, S.s7), kr(i, KB2, S.s8));
  let ci = (i * H + px.y) * W + px.x;
  var h = staticFirstHit(cam.pos.xyz, d, tt, ci);
  let hd = traceSceneSkip(cam.pos.xyz, d, 0.0, h.t, tt, false, 2u, cam.flags.w);
  if (hd.tri != MISS) { h = hd; }
  gCacheIdx = ci;
  gShadowCache = (cam.flags.y & O_SHADOW_CACHE) != 0u;
  let L = shadeHit(h, cam.pos.xyz, d, tt, ul, ub, cam.flags.y, i * nl, S);
  gShadowCache = false;
  return L;
}

// ---------------------------------------------------------------- packet traversal of the static BVH
// The 64 primary rays of a workgroup (an 8x8 pixel block, highly coherent) walk the static BVH together: one
// shared stack in workgroup memory, each node fetched once for all, each ray testing the boxes against its own
// closest hit; a child is entered if any ray wants it, near-first for the majority. Every ray still finds its own
// closest hit (the union of the nodes the rays need is visited). Loop capped at MAX_PACKET; the cap flags dbg[7].
const MAX_PACKET: u32 = 8192u;
var<workgroup> pNode: u32;
var<workgroup> pSp: u32;
var<workgroup> pStack: array<u32, 64>;
var<workgroup> pVote: atomic<u32>;
var<workgroup> pNear1: atomic<u32>;

fn tracePacketStatic(o: vec3f, d: vec3f, live: bool, lid: u32) -> Hit {
  var best = Hit(BIG, MISS, 0.0, 0.0, MISS);
  let inv = safeInv(d);
  if (lid == 0u) { pNode = gRoot; pSp = 0u; atomicStore(&pVote, 0u); atomicStore(&pNear1, 0u); }
  for (var it = 0u; it < MAX_PACKET; it++) {
    let node = workgroupUniformLoad(&pNode);
    if (node == MISS) { break; }
    if (live) {
      let b = node * 4u;
      let a0 = nodes[b]; let a1 = nodes[b + 1u]; let a2 = nodes[b + 2u]; let a3 = nodes[b + 3u];
      let t0 = slab(a0.xyz, a1.xyz, o, inv, 0.0, best.t);
      let t1 = slab(a2.xyz, a3.xyz, o, inv, 0.0, best.t);
      let c0 = bitcast<u32>(a1.w); let c1 = bitcast<u32>(a3.w);
      let r0 = bitcast<u32>(a0.w); let r1 = bitcast<u32>(a2.w);
      var want = 0u;
      for (var c = 0u; c < 2u; c++) {
        let tc = select(t1, t0, c == 0u);
        let cnt = select(c1, c0, c == 0u);
        let ref0 = select(r1, r0, c == 0u);
        if (tc >= best.t) { continue; }
        if (cnt == 0u) { want = want | (1u << c); continue; }
        for (var k = ref0; k < ref0 + min(cnt, MAX_LEAF); k++) {
          let h = triHit(k, o, d, 0.0, best.t);
          if (h.x < best.t) {
            if (CUTAWAY) { if (dot(scn.cut.xyz, o + d * h.x) > scn.cut.w) { continue; } }
            best = Hit(h.x, k, h.y, h.z, MISS);
          }
        }
      }
      if (want != 0u) { atomicOr(&pVote, want); }
      if (want == 3u && t1 < t0) { atomicAdd(&pNear1, 1u); }
    }
    workgroupBarrier();
    if (lid == 0u) {
      let v = atomicLoad(&pVote);
      let b = node * 4u;
      let r0 = bitcast<u32>(nodes[b].w); let r1 = bitcast<u32>(nodes[b + 2u].w);
      var next = MISS;
      if (v == 3u) {
        let first1 = atomicLoad(&pNear1) * 2u > 64u;        // most rays that want both see child 1 nearer
        next = select(r0, r1, first1);
        if (pSp < 64u) { pStack[pSp] = select(r1, r0, first1); pSp = pSp + 1u; }
      } else if (v == 1u) { next = r0; }
      else if (v == 2u) { next = r1; }
      else if (pSp > 0u) { pSp = pSp - 1u; next = pStack[pSp]; }
      if (v != 0u && next <= node) { atomicAdd(&dbg[6], 1u); next = MISS; }   // a child always follows its parent
      pNode = next;
      atomicStore(&pVote, 0u); atomicStore(&pNear1, 0u);
      if (it + 1u == MAX_PACKET) { atomicAdd(&dbg[7], 1u); }
    }
  }
  return best;
}

// Pass 1: cam.size.z sub-samples per pixel. Writes the image and, per pixel, the sum and a key (mesh of the first
// hit, + 0x10000 if a posed body was hit, + 0x20000 if a shadow ray had to be traced) for the refinement pass.
// All 64 invocations stay in the loops (inactive ones only vote), as the packet traversal needs uniform control flow.
@compute @workgroup_size(8, 8)
fn camMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  gRoot = bitcast<u32>(cam.lensC.y);
  let W = cam.size.x;
  let H = cam.size.y;
  let px = gid.xy + cam.tile.xy;                        // this dispatch covers one tile (docs/WEB_TWIN.md §12)
  let valid = gid.x < cam.tile.z && gid.y < cam.tile.w && px.x < W && px.y < H;
  let spp = clamp(cam.size.z, 1u, 64u);
  let S = pixSeeds(px.x, px.y);
  let packet = (cam.flags.y & O_PACKET) != 0u;
  var acc = vec3f(0.0);
  var key = 0u;
  for (var i = 0u; i < spp; i++) {
    gKey = 0xffffu;
    // the sub-sample's ray (as subSample), traced by the packet (static) + per ray (posed bodies)
    let uv = vec2f(f32(px.x) + kr(i, R2A, S.s0) - 0.5, f32(px.y) + kr(i, R2B, S.s1) - 0.5);
    let tt = mix(cam.expo.x, cam.expo.y, kr(i, GOLD, S.s2));
    var dc: vec3f;
    if (cam.pos.w < 0.5) { dc = kb4Unproject(Lens(cam.lensA, cam.lensB, cam.lensC), uv); }
    else { dc = normalize(vec3f((uv.x - cam.lensA.z) / cam.lensA.x, (uv.y - cam.lensA.w) / cam.lensA.y, 1.0)); }
    let d = normalize(cam.rx.xyz * dc.x + cam.ry.xyz * dc.y + cam.rz.xyz * dc.z);
    let nl = max(1u, bitcast<u32>(cam.lensC.z));
    let ul = vec3f(kr(i * nl, KL0, S.s3), kr(i * nl, KL1, S.s4), kr(i * nl, KL2, S.s5));
    let ub = vec3f(kr(i, KB0, S.s6), kr(i, KB1, S.s7), kr(i, KB2, S.s8));
    var h: Hit;
    let ci = (i * H + px.y) * W + px.x;
    if (packet) {
      h = tracePacketStatic(cam.pos.xyz, d, valid, lid);
      if (valid) {
        let hd = traceSceneSkip(cam.pos.xyz, d, 0.0, h.t, tt, false, 2u, cam.flags.w);
        if (hd.tri != MISS) { h = hd; }
      }
    } else if (valid) {
      h = staticFirstHit(cam.pos.xyz, d, tt, ci);
      let hd = traceSceneSkip(cam.pos.xyz, d, 0.0, h.t, tt, false, 2u, cam.flags.w);
      if (hd.tri != MISS) { h = hd; }
    }
    if (valid) {
      gCacheIdx = ci;
      gShadowCache = (cam.flags.y & O_SHADOW_CACHE) != 0u;
      acc = acc + shadeHit(h, cam.pos.xyz, d, tt, ul, ub, cam.flags.y, i * nl, S);
      gShadowCache = false;
      if (i == 0u) { key = gKey; } else if ((gKey & 0xffffu) != (key & 0xffffu)) { key = key | 0x40000u; }
      key = key | (gKey & 0x30000u);
    }
  }
  if (valid) {
    acc1[px.y * W + px.x] = vec4f(acc, bitcast<f32>(key));
    // alpha 0 marks a pixel whose rays all missed (the overview's display paints it with its background)
    textureStore(outTex, vec2i(px), vec4f(acc * (cam.expo.z / f32(spp)), select(1.0, 0.0, key == 0xfffeu)));
  }
}

// Pass 2: pixels on an edge (a different first-hit mesh in the 3x3 neighbourhood or among their own sub-samples),
// on a posed body (glossy instruments), or whose shadow rays could not be culled (penumbrae, near the tools) get
// cam.expo.w more sub-samples, continuing the same sequences; the others keep their pass-1 value.
@compute @workgroup_size(8, 8)
fn camRefine(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  gRoot = bitcast<u32>(cam.lensC.y);
  let W = cam.size.x;
  let H = cam.size.y;
  let px = gid.xy + cam.tile.xy;
  if (gid.x >= cam.tile.z || gid.y >= cam.tile.w || px.x >= W || px.y >= H) { return; }
  let spp = clamp(cam.size.z, 1u, 64u);
  let extra = min(u32(cam.expo.w), 64u);
  let a = acc1[px.y * W + px.x];
  let k = bitcast<u32>(a.w);
  // refine: mesh edges among the sub-samples or in the 3x3 block, posed bodies, and (option) traced shadow rays
  var refine = (k & 0x50000u) != 0u || ((k & 0x20000u) != 0u && (cam.flags.y & O_REFINE_SHADOW) != 0u);
  if (!refine) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let xx = clamp(i32(px.x) + dx, 0, i32(W) - 1);
        let yy = clamp(i32(px.y) + dy, 0, i32(H) - 1);
        let kn = bitcast<u32>(acc1[u32(yy) * W + u32(xx)].w);
        if ((kn & 0xffffu) != (k & 0xffffu)) { refine = true; }
      }
    }
  }
  if (!refine || extra == 0u) { return; }
  // statistics of why pixels are refined (dbg[2..5]: total, posed body, shadow traced, sub-sample mesh change)
  atomicAdd(&dbg[2], 1u);
  if ((k & 0x10000u) != 0u) { atomicAdd(&dbg[3], 1u); }
  if ((k & 0x20000u) != 0u) { atomicAdd(&dbg[4], 1u); }
  if ((k & 0x40000u) != 0u) { atomicAdd(&dbg[5], 1u); }
  let S = pixSeeds(px.x, px.y);
  var acc = a.xyz;
  for (var i = spp; i < spp + extra; i++) { acc = acc + cachedSubSample(px, i, S); }
  textureStore(outTex, vec2i(px), vec4f(acc * (cam.expo.z / f32(spp + extra)), 1.0));
}
`;

const RADIOSITY_MAIN = /* wgsl */ `
struct RadU { nActive: u32, slotIn: u32, slotOut: u32, nStatic: u32, start: u32, count: u32, root: u32, moved: u32 }   // a chunk [start, start+count); moved: incremental mask
@group(0) @binding(5) var<storage, read_write> gtable: array<u32>;
@group(0) @binding(6) var<storage, read> activeNodes: array<u32>;
@group(0) @binding(7) var<storage, read_write> aux: array<vec4f>;
@group(0) @binding(8) var<storage, read> nbr: array<vec2u>;
@group(0) @binding(9) var<storage, read_write> dbgR: array<u32, 1>;   // (placeholder; dbg is declared with G0)      // per active node: NBR_K (active index, weight bits)
@group(1) @binding(0) var<uniform> scn: SceneU;
@group(1) @binding(1) var<uniform> rad: RadU;

fn nodeMat(i: u32) -> u32 { return bitcast<u32>(params[MESH_OFF + 2u * bitcast<u32>(verts[2u * i].w)].w); }

fn nodeMatAlbedo(m: u32, mu: f32) -> vec3f { return albedoSpec(m, mu) + albedoDiff(m, mu); }

// gather-table ray j of static node i (normal n): K cosine-distributed rays (Fibonacci spiral on the disk, rotated
// per node). Shared by gatherBuild, gatherRemap and jacobiGather so the three see the same direction.
fn gatherDir(i: u32, j: u32, n: vec3f) -> vec3f {
  let t = basisT(n);
  let b = basisB(n);
  let rot = u01(pcg(i * 2654435761u + 17u));
  let rr = sqrt((f32(j) + 0.5) / f32(GATHER_K));
  let ph = 2.0 * PI * fract(f32(j) * GOLD + rot);
  return normalize(t * (rr * cos(ph)) + b * (rr * sin(ph)) + n * sqrt(max(0.0, 1.0 - rr * rr)));
}

// a node is active (lit and iterated) when at most half its gather rays escape; aux.x holds the misses after gatherCount
fn nodeActiveByMisses(v: u32) -> bool { return aux[v].x <= 0.5 * f32(GATHER_K); }

// K cosine-distributed rays per node against the static BVH; each entry is the nearest node of the hit triangle,
// MISS for an escape or a back-face hit. One invocation per ray (node-major), so a small chunk still fills the GPU:
// rays [start, start + count) of the flat table.
@compute @workgroup_size(64)
fn gatherBuild(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let r = gid.x + rad.start;
  let i = r / GATHER_K;
  let j = r % GATHER_K;
  if (gid.x >= rad.count || i >= rad.nStatic) { return; }
  let p = verts[2u * i].xyz;
  let n = normalize(verts[2u * i + 1u].xyz);
  let dir = gatherDir(i, j, n);
  let h = traceScene(p + n * EPS_OFF, dir, 0.0, BIG, 0.5, false, 1u);
  var v = MISS;
  if (h.tri != MISS) {
    let bb = vec3f(1.0 - h.u - h.v, h.u, h.v);
    let vi = vec3u(bitcast<u32>(tris[4u * h.tri].w), bitcast<u32>(tris[4u * h.tri + 1u].w), bitcast<u32>(tris[4u * h.tri + 2u].w));
    let nv = bb.x * verts[2u * vi.x + 1u].xyz + bb.y * verts[2u * vi.y + 1u].xyz + bb.z * verts[2u * vi.z + 1u].xyz;
    if (dot(nv, dir) < 0.0) {
      if (bb.x >= bb.y && bb.x >= bb.z) { v = vi.x; } else if (bb.y >= bb.z) { v = vi.y; } else { v = vi.z; }
    }
  }
  gtable[i * GATHER_K + j] = v;
}

// After gatherCount: an entry whose nearest vertex is an inactive node (outer skin, the rim faces of seams and holes,
// bench under the wall) would read L = 0 although the ray hit a surface that reflects. Re-trace that ray (the same
// ray as gatherBuild) and point it at the hit triangle's highest-barycentric ACTIVE vertex; keep it if none is active.
@compute @workgroup_size(64)
fn gatherRemap(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let r = gid.x + rad.start;
  let i = r / GATHER_K;
  let j = r % GATHER_K;
  if (gid.x >= rad.count || i >= rad.nStatic) { return; }
  let v0 = gtable[r];
  if (v0 == MISS || nodeActiveByMisses(v0)) { return; }
  let p = verts[2u * i].xyz;
  let n = normalize(verts[2u * i + 1u].xyz);
  let dir = gatherDir(i, j, n);
  let h = traceScene(p + n * EPS_OFF, dir, 0.0, BIG, 0.5, false, 1u);
  if (h.tri == MISS) { return; }
  let bb = vec3f(1.0 - h.u - h.v, h.u, h.v);
  let vi = vec3u(bitcast<u32>(tris[4u * h.tri].w), bitcast<u32>(tris[4u * h.tri + 1u].w), bitcast<u32>(tris[4u * h.tri + 2u].w));
  var best = v0;
  var bw = -1.0;
  if (nodeActiveByMisses(vi.x) && bb.x > bw) { best = vi.x; bw = bb.x; }
  if (nodeActiveByMisses(vi.y) && bb.y > bw) { best = vi.y; bw = bb.y; }
  if (nodeActiveByMisses(vi.z) && bb.z > bw) { best = vi.z; bw = bb.z; }
  gtable[r] = best;
}

// escapes per node, from the finished table (aux.x), for the interior / exterior split
@compute @workgroup_size(64)
fn gatherCount(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + rad.start;
  if (gid.x >= rad.count || i >= rad.nStatic) { return; }
  var misses = 0u;
  for (var j = 0u; j < GATHER_K; j++) { if (gtable[i * GATHER_K + j] == MISS) { misses++; } }
  aux[i] = vec4f(f32(misses), 0.0, 0.0, 0.0);
}

// direct irradiance at each active node from the emitter (NODE_SAMPLES fixed samples, shadow rays against the
// whole scene at mid-exposure), its first reflection B_dir = sum E_s A(mu_s), and the capsule visibility
@compute @workgroup_size(64)
fn nodeDirect(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  gRoot = rad.root;
  let k = gid.x + rad.start;
  if (gid.x >= rad.count || k >= rad.nActive) { return; }
  let i = activeNodes[k];
  let p = verts[2u * i].xyz;
  let n = normalize(verts[2u * i + 1u].xyz);
  // incremental update (only posed bodies other than the scope moved): keep the node unless the segment to the
  // emitter passes near a moved body's bounding capsule, where it is now or where it was at the last update
  if (rad.moved != 0u) {
    let P = instPose(scn.info.y, 0.5);
    let dE = (P.r * scn.emitC.xyz + P.p) - p;
    var near = false;
    for (var b = 0u; b < min(scn.info.x, MAX_INST); b++) {
      if (((rad.moved >> b) & 1u) == 0u) { continue; }
      let I = scn.inst[b];
      let rr = scn.emitC.w + 1e-3;
      if (segNearCapsule(p, dE, 0.0, 1.0, I.bcA.xyz, I.bcB.xyz, I.bcA.w + rr) || segNearCapsule(p, dE, 0.0, 1.0, I.pcA.xyz, I.pcB.xyz, I.pcA.w + rr)) { near = true; break; }
    }
    if (!near) { return; }
  }
  let m = nodeMat(i);
  let hs = pcg(i * 2246822519u + 3u);
  let s0 = u01(pcg(hs)); let s1 = u01(pcg(hs + 1u)); let s2 = u01(pcg(hs + 2u));
  var E = vec3f(0.0);
  var B = vec3f(0.0);
  var nLit = 0u;
  var nVis = 0u;
  let o = p + n * EPS_OFF;
  for (var s = 0u; s < NODE_SAMPLES; s++) {
    let ls = sampleEmitter(vec3f(kr(s, KL0, s0), kr(s, KL1, s1), kr(s, KL2, s2)), 0.5);
    var wi = ls.p - p;
    let d2 = dot(wi, wi);
    let dist = sqrt(d2);
    wi = wi / dist;
    let cosE = dot(-wi, ls.n);
    let cosN = dot(n, wi);
    if (cosE <= 0.0 || cosN <= 0.0) { continue; }
    let Le = emitted(ls.n, ls.axis, -wi);
    if (max(Le.x, max(Le.y, Le.z)) <= 0.0) { continue; }
    nLit++;
    let sh = traceSceneSkip(o, wi, 0.0, dist - 2e-5, 0.5, true, 3u, bitcast<u32>(scn.axis.w));
    if (sh.tri != MISS) { continue; }
    nVis++;
    let e = Le * (cosE * cosN * scn.L0.w / d2);
    E = E + e;
    B = B + e * (albedoSpec(m, cosN) + albedoDiff(m, cosN));
  }
  E = E / f32(NODE_SAMPLES);
  B = B / f32(NODE_SAMPLES);
  // visibility code for the pixels' shadow-ray culling: 1 all lit samples visible, 0 none, 0.5 mixed, -1 no lit sample
  var vis = 0.5;
  if (nLit == 0u) { vis = -1.0; } else if (nVis == nLit) { vis = 1.0; } else if (nVis == 0u) { vis = 0.0; }
  nodeState[4u * i + 3u] = vec4f(B, vis);
  let old = nodeState[4u * i];
  nodeState[4u * i] = vec4f(old.xyz, capsuleVis(p, n));
}

// One Jacobi step in two passes, with the reciprocal separable form of the section-8 BSDF for node-to-node
// transport: f(l, v) ~ A(mu_l) A(mu_v) / (pi A_hh), A = albedoSpec + albedoDiff (the tables). It keeps every
// node's directional albedo exact (integral of f cos over l = A(mu_v)) and reciprocity; a uniform (Lambertian)
// re-emitter with reflectance A_hh would make the sender's exitance flat in angle and the receiver's reflectance
// independent of incidence, which loses indirect light in the high-gain dome (docs/DECISIONS.md, ADR web-twin r1).
// A node stores its separable source Ls (slot 1 or 2): it sends the radiance A_m(mu_out) Ls toward mu_out.
// jacobiGather: aux = (pi / K) sum_j A_i(mu_in_j) A_h(mu_out_j) Ls_h / A_hh,i, the pixel-equivalent irradiance
// (a pixel reflects E A(mu_v) / pi; for a uniform incident field aux = pi L_in exactly).
@compute @workgroup_size(64)
fn jacobiGather(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x + rad.start;
  if (gid.x >= rad.count || k >= rad.nActive) { return; }
  let i = activeNodes[k];
  let n = normalize(verts[2u * i + 1u].xyz);
  let mi = nodeMat(i);
  var acc = vec3f(0.0);
  for (var j = 0u; j < GATHER_K; j++) {
    let h = gtable[i * GATHER_K + j];
    if (h == MISS) { continue; }
    let dir = gatherDir(i, j, n);
    let muIn = sqrt(max(0.0, 1.0 - (f32(j) + 0.5) / f32(GATHER_K)));
    let muOut = clamp(-dot(dir, normalize(verts[2u * h + 1u].xyz)), 0.0, 1.0);
    acc = acc + nodeMatAlbedo(mi, muIn) * nodeMatAlbedo(nodeMat(h), muOut) * nodeState[4u * h + rad.slotIn].xyz;
  }
  let hh = max(params[MAT_OFF + MAT_STRIDE * mi + 2u].xyz, vec3f(1e-4));
  aux[k] = vec4f(acc * (PI / f32(GATHER_K)) / hh, 0.0);
}

// jacobiUpdate: E_free = Gaussian-weighted mean of aux over the node's neighbours within a few mm (same-facing
// surfaces only; it averages out the table's frozen Monte Carlo noise), E_ind = E_free V, and the separable source
// Ls = (B_dir + E_ind A_hh) / (pi A_hh): the node's exitance (B_dir = sum E_s A(mu_s) is exact for direct light)
// divided by pi A_hh, so the radiance it sends toward mu_out is A(mu_out) Ls.
@compute @workgroup_size(64)
fn jacobiUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x + rad.start;
  if (gid.x >= rad.count || k >= rad.nActive) { return; }
  let i = activeNodes[k];
  var Ef = vec3f(0.0);
  var ws = 0.0;
  for (var j = 0u; j < NBR_K; j++) {
    let e = nbr[k * NBR_K + j];
    let w = bitcast<f32>(e.y);
    if (w <= 0.0) { break; }
    Ef = Ef + aux[e.x].xyz * w;
    ws = ws + w;
  }
  if (ws > 0.0) { Ef = Ef / ws; } else { Ef = aux[k].xyz; }
  let st = nodeState[4u * i];
  let m = nodeMat(i);
  let hh = max(params[MAT_OFF + MAT_STRIDE * m + 2u].xyz, vec3f(1e-4));
  let L = (nodeState[4u * i + 3u].xyz + Ef * st.w * hh) * INV_PI / hh;
  nodeState[4u * i] = vec4f(Ef, st.w);
  nodeState[4u * i + rad.slotOut] = vec4f(L, 1.0);     // w = 1: an active (solved) node, for the pixels' weights
}
`;

const DISPLAY = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
struct DispU { scale: vec4f, bg: vec4f }
@group(0) @binding(2) var<uniform> du: DispU;     // display scale; background (linear sRGB, already encoded) for misses

fn srgb(x: f32) -> f32 {
  let c = clamp(x, 0.0, 1.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

@compute @workgroup_size(8, 8)
fn displayMain(@builtin(global_invocation_id) gid: vec3u) {
  let sz = textureDimensions(src);
  if (gid.x >= sz.x || gid.y >= sz.y) { return; }
  let t = textureLoad(src, vec2i(gid.xy), 0);
  let c = t.xyz * du.scale.x;
  var o = vec3f(srgb(c.x), srgb(c.y), srgb(c.z));
  if (t.w < 0.5 && du.bg.w > 0.5) { o = du.bg.xyz; }
  textureStore(dst, vec2i(gid.xy), vec4f(o, 1.0));
}
`;

const TESTS = /* wgsl */ `
struct TestU { la: vec4f, lb: vec4f, lc: vec4f, n: u32, mode: u32, mat: u32, p: u32 }
@group(1) @binding(0) var<uniform> scn: SceneU;
@group(1) @binding(1) var<uniform> tu: TestU;
@group(1) @binding(2) var<storage, read> tin: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> tout: array<vec4f>;

@compute @workgroup_size(64)
fn lensTest(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= tu.n) { return; }
  let L = Lens(tu.la, tu.lb, tu.lc);
  let a = tin[i];
  // tin: (u, v, -, -) for unproject and (x, y, z, -) in the next slot for project
  let ray = kb4Unproject(L, a.xy);
  let uv = kb4Project(L, tin[tu.n + i].xyz);
  tout[2u * i] = vec4f(ray, 0.0);
  tout[2u * i + 1u] = vec4f(uv, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn bsdfTest(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= tu.n) { return; }
  // tin per case: n, l, v, (u0, u1, u2, material)
  let n = tin[4u * i].xyz;
  let l = tin[4u * i + 1u].xyz;
  let v = tin[4u * i + 2u].xyz;
  let u = tin[4u * i + 3u];
  let M = loadMat(u32(u.w));
  let f = evalBSDF(M, n, l, v);
  let pdf = pdfBSDF(M, n, v, l);
  let s = sampleBSDF(M, n, v, u.xyz);
  tout[3u * i] = vec4f(f, pdf);
  tout[3u * i + 1u] = s;
  tout[3u * i + 2u] = vec4f(albedoSpec(M.idx, dot(n, v)), layerNormal(n, l * 0.1, normalize(v + vec3f(0.3, 0.1, 0.2)), 2e-4, 2e-5).x);
}

@compute @workgroup_size(64)
fn capsuleTest(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= tu.n) { return; }
  // tin per case: p, n, a (w = r), b
  let a = tin[4u * i + 2u];
  tout[i] = vec4f(capsuleOcc(tin[4u * i].xyz, tin[4u * i + 1u].xyz, a.xyz, tin[4u * i + 3u].xyz, a.w), 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn traceTest(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  gBase = lid * STACK;
  let i = gid.x;
  if (i >= tu.n) { return; }
  let o = tin[2u * i];
  let d = tin[2u * i + 1u];
  let h = traceScene(o.xyz, d.xyz, 0.0, BIG, o.w, tu.mode == 1u, tu.mat);
  tout[i] = vec4f(h.t, bitcast<f32>(h.tri), bitcast<f32>(h.inst), h.u);
}

@compute @workgroup_size(64)
fn profileTest(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= tu.n) { return; }
  tout[i] = vec4f(profile(tin[i].x), 0.0, 0.0, 0.0);
}
`;

/** Full WGSL source for a kernel family, given the scene's constant block `c` (see common.wgsl.js defines). */
export function shaderSource(kind, c) {
  const head = defines(c) + COMMON;
  switch (kind) {
    case 'camera': return head + G0('read', 5) + CAMERA_MAIN + LENS + BSDF + TRACE + SHADE + SHADE_CAMERA;
    case 'radiosity': return head + G0('read_write', 9) + RADIOSITY_MAIN + BSDF + TRACE + SHADE;
    case 'tests': return head + G0('read', 5) + TESTS + LENS + BSDF + TRACE + SHADE;
    case 'display': return DISPLAY;
    default: throw new Error(`unknown shader ${kind}`);
  }
}
