// WGSL light transport for camera rays (docs/WEB_TWIN.md §8):
// * the scope light outlet is an area emitter with L_e = L0 p(theta), theta from the emission axis, sampled by area
//   (triangle CDF) with a shadow ray against the whole scene, posed with the scope at the ray's time;
// * MIS (power heuristic) between light and BSDF sampling on instruments and the scope, whose BSDF-sampled ray
//   also carries their glossy and diffuse indirect light by reading the radiosity of the static surface it hits;
// * static surfaces read their indirect irradiance from the three radiosity nodes of the hit triangle
//   (barycentric), times an analytic, noise-free instrument occlusion (capsules), times the directional albedo.
export const SHADE = /* wgsl */ `
override CULL_SHADOWS: bool = false;
var<private> gKey: u32 = 0xffffu;       // first-hit mesh | 0x10000 posed body | 0x20000 shadow ray traced

fn profile(th: f32) -> f32 {
  // piecewise linear p(theta), 0 past the last knot (physics.yaml light.profile)
  let last = params[PROF_OFF + PROF_N - 1u];
  if (th >= last.x) { return select(0.0, last.y, th == last.x); }
  var lo = 0u;
  var hi = PROF_N - 1u;
  for (var it = 0u; it < 40u; it++) {
    if (hi - lo <= 1u) { break; }
    let m = (lo + hi) / 2u;
    if (params[PROF_OFF + m].x <= th) { lo = m; } else { hi = m; }
  }
  let a = params[PROF_OFF + lo];
  let b = params[PROF_OFF + hi];
  let f = clamp((th - a.x) / max(b.x - a.x, 1e-12), 0.0, 1.0);
  return mix(a.y, b.y, f);
}

struct LightSample { p: vec3f, n: vec3f, axis: vec3f }

fn sampleEmitter(u: vec3f, time: f32) -> LightSample {
  var lo = 0u;
  var hi = EMIT_N - 1u;
  for (var it = 0u; it < 40u; it++) {
    if (lo >= hi) { break; }
    let m = (lo + hi) / 2u;
    if (params[EMIT_OFF + 4u * m].w < u.x) { lo = m + 1u; } else { hi = m; }
  }
  let v0 = params[EMIT_OFF + 4u * lo].xyz;
  let e1 = params[EMIT_OFF + 4u * lo + 1u].xyz;
  let e2 = params[EMIT_OFF + 4u * lo + 2u].xyz;
  let nl = params[EMIT_OFF + 4u * lo + 3u].xyz;
  let su = sqrt(u.y);
  let pl = v0 + e1 * (su * (1.0 - u.z)) + e2 * (su * u.z);
  let P = instPose(scn.info.y, time);
  return LightSample(P.r * pl + P.p, P.r * nl, P.r * scn.axis.xyz);
}

// radiance leaving the emitter at a point with normal n toward direction w
fn emitted(n: vec3f, axis: vec3f, w: vec3f) -> vec3f {
  if (dot(w, n) <= 0.0) { return vec3f(0.0); }
  return scn.L0.xyz * profile(acos(clamp(dot(w, axis), -1.0, 1.0)));
}

fn emitterAxisAt(time: f32) -> vec3f { return instPose(scn.info.y, time).r * scn.axis.xyz; }

// Cosine-weighted fraction of the hemisphere above (p, n) hidden by a cylinder of radius r along a-b, by the
// closed-form integral over the axis angle (thin-cylinder limit, clipped to the horizon, softened at contact).
fn capsuleOcc(p: vec3f, n: vec3f, a: vec3f, bb: vec3f, r: f32) -> f32 {
  let ab = bb - a;
  let L = length(ab);
  if (L < 1e-9 || r <= 0.0) { return 0.0; }
  let e = ab / L;
  let ap = p - a;
  let s0 = dot(ap, e);                 // foot of the perpendicular, along the axis from a
  let perp = ap - s0 * e;              // from the axis to p
  let h = sqrt(dot(perp, perp) + r * r);
  var ep = -perp / max(length(perp), 1e-12);    // direction from p toward the axis
  if (length(perp) < 1e-12) { ep = normalize(cross(e, basisT(e))); }
  let np = dot(n, ep);
  let na = dot(n, e);
  // phi = angle along the axis seen from p: point s on the axis at tan(phi) = (s - s0) / h
  var f1 = atan((0.0 - s0) / h);
  var f2 = atan((L - s0) / h);
  let f0 = atan2(na, np);              // n.omega = sqrt(np^2 + na^2) cos(phi - f0) > 0 on (f0 - pi/2, f0 + pi/2)
  f1 = max(f1, f0 - 0.5 * PI);
  f2 = min(f2, f0 + 0.5 * PI);
  if (f2 <= f1) { return 0.0; }
  // integral of cos(phi) (np cos(phi) + na sin(phi)) dphi
  let I = np * (0.5 * (f2 - f1) + 0.25 * (sin(2.0 * f2) - sin(2.0 * f1))) + na * 0.5 * (sin(f2) * sin(f2) - sin(f1) * sin(f1));
  return clamp(INV_PI * (2.0 * r / h) * I, 0.0, 1.0);
}

// 1 - (1 - rho_eff) O summed over the bodies' capsules: indirect visibility factor at a static point
fn capsuleVis(p: vec3f, n: vec3f) -> f32 {
  var v = 1.0;
  let cnt = min(scn.info.x, MAX_INST);
  for (var i = 0u; i < cnt; i++) {
    let I = scn.inst[i];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u) { continue; }
    let o = capsuleOcc(p, n, I.capA.xyz, I.capB.xyz, I.capA.w);
    v = v - o * (1.0 - I.capB.w);
  }
  return clamp(v, 0.0, 1.0);
}

fn meshRow1(mesh: u32) -> vec4f { return params[MESH_OFF + 2u * mesh + 1u]; }

// shading normal with optional FDM layer lines
fn shadingNormal(s: Surf, opts: u32) -> vec3f {
  if ((opts & O_LAYER) == 0u || (s.flags & F_LAYER) == 0u) { return s.ns; }
  let r0 = params[MESH_OFF + 2u * s.mesh];
  let r1 = meshRow1(s.mesh);
  var n = layerNormal(s.ns, s.p, r0.xyz, r1.y, r1.z);
  if (dot(n, s.ng) <= 0.0) { n = s.ns; }
  return n;
}

// Shadow-ray culling on static surfaces, decided once per sub-sample: when the three radiosity nodes of the hit
// triangle agree that every lit emitter sample is visible (1) or hidden (0), and no posed body (other than the
// scope, which cannot occlude its own outlet) comes within its bounding capsule + the emitter radius + 0.5 mm of
// the segment to the emitter centroid, every shadow ray of this point has that answer and none is traced.
// Returns 1 (lit), 0 (shadowed) or -1 (trace).
fn cachedVisibility(s: Surf, time: f32) -> f32 {
  if ((s.flags & F_STATIC) == 0u) { return -1.0; }
  let a = nodeState[4u * s.vi.x + 3u].w;
  let b = nodeState[4u * s.vi.y + 3u].w;
  let c = nodeState[4u * s.vi.z + 3u].w;
  if (!(a == b && b == c && (a == 1.0 || a == 0.0))) { return -1.0; }
  let P = instPose(scn.info.y, time);
  let d = (P.r * scn.emitC.xyz + P.p) - s.p;
  let cnt = min(scn.info.x, MAX_INST);
  let skip = bitcast<u32>(scn.axis.w);
  for (var i = 0u; i < cnt; i++) {
    let I = scn.inst[i];
    if ((bitcast<u32>(I.p1.w) & 2u) != 0u || ((skip >> i) & 1u) != 0u) { continue; }
    if (segNearCapsule(s.p, d, 0.0, 1.0, I.bcA.xyz, I.bcB.xyz, I.bcA.w + scn.emitC.w + scn.eAvg.w)) { return -1.0; }
  }
  return a;
}

// one emitter sample's unshadowed contribution and its shadow segment (the caller decides visibility)
struct LightEval { c: vec3f, o: vec3f, wi: vec3f, tmax: f32, ok: bool }

fn lightEval(s: Surf, n: vec3f, M: Mat, wo: vec3f, time: f32, u: vec3f, mis: bool, nL: f32) -> LightEval {
  var e = LightEval(vec3f(0.0), vec3f(0.0), vec3f(0.0, 0.0, 1.0), 0.0, false);
  let ls = sampleEmitter(u, time);
  var wi = ls.p - s.p;
  let d2 = dot(wi, wi);
  let dist = sqrt(d2);
  wi = wi / dist;
  let cosE = dot(-wi, ls.n);
  if (cosE <= 0.0) { return e; }
  let cosS = dot(n, wi);
  if (cosS <= 0.0 || dot(s.ng, wi) <= 0.0) { return e; }
  let Le = emitted(ls.n, ls.axis, -wi);
  if (max(Le.x, max(Le.y, Le.z)) <= 0.0) { return e; }
  let f = evalBSDF(M, n, wi, wo);
  if (max(f.x, max(f.y, f.z)) <= 0.0) { return e; }
  let pdfW = d2 / (cosE * scn.L0.w);
  var w = 1.0;
  if (mis) {
    let pb = pdfBSDF(M, n, wo, wi);
    w = nL * nL * pdfW * pdfW / (nL * nL * pdfW * pdfW + pb * pb);
  }
  return LightEval(f * Le * (cosS * w / pdfW), offsetRay(s.p, s.ng, wi), wi, dist - 2e-5, true);
}

// direct light by sampling the emitter (one sample), optionally MIS-weighted against BSDF sampling
fn directLight(s: Surf, n: vec3f, M: Mat, wo: vec3f, time: f32, u: vec3f, mis: bool, nL: f32, cached: f32) -> vec3f {
  let ls = sampleEmitter(u, time);
  var wi = ls.p - s.p;
  let d2 = dot(wi, wi);
  let dist = sqrt(d2);
  wi = wi / dist;
  let cosE = dot(-wi, ls.n);
  if (cosE <= 0.0) { return vec3f(0.0); }
  let cosS = dot(n, wi);
  if (cosS <= 0.0 || dot(s.ng, wi) <= 0.0) { return vec3f(0.0); }
  let Le = emitted(ls.n, ls.axis, -wi);              // before the shadow ray: p(theta) = 0 outside the cone
  if (max(Le.x, max(Le.y, Le.z)) <= 0.0) { return vec3f(0.0); }
  let f = evalBSDF(M, n, wi, wo);
  if (max(f.x, max(f.y, f.z)) <= 0.0) { return vec3f(0.0); }
  let vis = cached;
  if (vis == 0.0) { return vec3f(0.0); }
  if (vis < 0.0) {
    gKey = gKey | 0x20000u;
    let o = offsetRay(s.p, s.ng, wi);
    let sh = traceSceneSkip(o, wi, 0.0, dist - 2e-5, time, true, 3u, bitcast<u32>(scn.axis.w));
    if (sh.tri != MISS) { return vec3f(0.0); }
  }
  let pdfW = d2 / (cosE * scn.L0.w);
  var w = 1.0;
  if (mis) {
    let pb = pdfBSDF(M, n, wo, wi);
    w = nL * nL * pdfW * pdfW / (nL * nL * pdfW * pdfW + pb * pb);      // power heuristic, nL light samples : 1
  }
  return f * Le * (cosS * w / pdfW);
}

// Barycentric weights renormalised over the triangle's ACTIVE nodes (jacobiUpdate marks a node's L slot with w = 1;
// inactive nodes - outer skin, rim faces, bench under the wall - are never solved and hold 0, which would darken a
// pixel on a triangle that mixes both for no physical reason). No active vertex: the plain barycentrics (all zero).
fn nodeWeights(vi: vec3u, b: vec3f) -> vec3f {
  let slot = scn.info.z;
  let w = b * vec3f(nodeState[4u * vi.x + slot].w, nodeState[4u * vi.y + slot].w, nodeState[4u * vi.z + slot].w);
  let sw = w.x + w.y + w.z;
  if (sw <= 1e-6) { return b; }
  return w / sw;
}
fn nodeEfree(vi: vec3u, b0: vec3f) -> vec3f {
  let b = nodeWeights(vi, b0);
  return b.x * nodeState[4u * vi.x].xyz + b.y * nodeState[4u * vi.y].xyz + b.z * nodeState[4u * vi.z].xyz;
}
fn nodeLout(vi: vec3u, b0: vec3f) -> vec3f {
  let slot = scn.info.z;
  let b = nodeWeights(vi, b0);
  return b.x * nodeState[4u * vi.x + slot].xyz + b.y * nodeState[4u * vi.y + slot].xyz + b.z * nodeState[4u * vi.z + slot].xyz;
}
`;

// Camera-ray part (needs the camera uniform `cam`).
export const SHADE_CAMERA = /* wgsl */ `
var<private> gShadowCache: bool = false;   // pass-1 sub-sample of a tracking camera with a valid cache key
var<private> gCacheIdx: u32 = 0u;
// outward (B-rep) normal of a hit on an emitter triangle, posed
fn emitterNormal(h: Hit, s: Surf, time: f32) -> vec3f {
  let n = normalize(s.b.x * verts[2u * s.vi.x + 1u].xyz + s.b.y * verts[2u * s.vi.y + 1u].xyz + s.b.z * verts[2u * s.vi.z + 1u].xyz);
  return instPose(h.inst, time).r * n;
}

// radiance arriving along a secondary ray (BSDF-sampled from a posed body): the emitter (MIS-weighted against
// light sampling), the cached radiosity of a static surface, or a coarse estimate for another posed body
fn secondary(o: vec3f, l: vec3f, time: f32, pdfB: f32, opts: u32, nL: f32) -> vec3f {
  let h = traceScene(o, l, 0.0, BIG, time, false, 3u);
  if (h.tri == MISS) { return vec3f(0.0); }
  let s = surface(h, o, l, time);
  if ((s.flags & F_EMITTER) != 0u) {
    if ((opts & O_NO_MIS) != 0u) { return vec3f(0.0); }      // light sampling alone carries the direct light
    let nE = emitterNormal(h, s, time);
    let cosE = dot(-l, nE);
    if (cosE <= 0.0) { return vec3f(0.0); }
    let Le = emitted(nE, emitterAxisAt(time), -l);
    let pdfL = h.t * h.t / (cosE * scn.L0.w);
    return Le * (pdfB * pdfB / (pdfB * pdfB + nL * nL * pdfL * pdfL));
  }
  if ((opts & (O_DIRECT_ONLY | O_NO_INDIRECT)) != 0u) { return vec3f(0.0); }
  // a static node sends A(mu_out) Ls toward mu_out (the separable node transport, kernels.wgsl.js jacobiGather)
  if ((s.flags & F_STATIC) != 0u) { return (albedoSpec(s.mat, clamp(dot(s.ns, -l), 0.0, 1.0)) + albedoDiff(s.mat, clamp(dot(s.ns, -l), 0.0, 1.0))) * nodeLout(s.vi, s.b); }
  let M2 = loadMat(s.mat);
  return M2.hh * scn.eAvg.xyz * INV_PI;
}

// radiance along a camera ray; ul: light-sample numbers, ub: BSDF-sample numbers
fn radiance(o: vec3f, d: vec3f, time: f32, ul: vec3f, ub: vec3f, opts: u32, skip: u32, lightIndex: u32, S: PixSeeds) -> vec3f {
  return shadeHit(traceSceneSkip(o, d, 0.0, BIG, time, false, 3u, skip), o, d, time, ul, ub, opts, lightIndex, S);
}

// shading of a camera ray's first hit h (from the per-ray or the packet traversal)
fn shadeHit(h: Hit, o: vec3f, d: vec3f, time: f32, ul: vec3f, ub: vec3f, opts: u32, lightIndex: u32, S: PixSeeds) -> vec3f {
  if (h.tri == MISS) { gKey = 0xfffeu; return vec3f(0.0); }
  let s = surface(h, o, d, time);
  gKey = s.mesh | select(0x10000u, 0u, h.inst == MISS || (s.flags & F_STATIC) != 0u);
  let wo = -d;
  if ((opts & O_NORMALS) != 0u) { return s.ns * 0.5 + vec3f(0.5); }
  if ((s.flags & F_EMITTER) != 0u) { return emitted(emitterNormal(h, s, time), emitterAxisAt(time), wo); }
  var M = loadMat(s.mat);
  if ((s.flags & F_TEXTURED) != 0u) {
    // charuco_print: rho_d and roughness mixed between ink (M.x) and paper (M.c, M.alpha) by the printed mask
    let uv = s.b.x * vec2f(verts[2u * s.vi.x].w, verts[2u * s.vi.x + 1u].w) + s.b.y * vec2f(verts[2u * s.vi.y].w, verts[2u * s.vi.y + 1u].w)
           + s.b.z * vec2f(verts[2u * s.vi.z].w, verts[2u * s.vi.z + 1u].w);
    let m = textureSampleLevel(boardTex, boardSmp, uv, 0.0).r;
    M.c = mix(M.x.xyz, M.c, m);
    M.alpha = mix(M.x.w, M.alpha, m);
  }
  let n = shadingNormal(s, opts);
  let isStatic = (s.flags & F_STATIC) != 0u;
  let mis = !isStatic && (opts & O_NO_MIS) == 0u;
  // nL light samples
  let nl = clamp(bitcast<u32>(cam.lensC.z), 1u, 16u);
  var L = vec3f(0.0);
  if (gShadowCache && isStatic) {
    // Static-occlusion cache (fixed camera, still scope): bit j of the word = light sample j hidden by the static
    // scene, bit 31 = valid. Posed bodies are tested every frame (capsule-culled instance traversal). Exact.
    var word = shadowCache[gCacheIdx];
    let valid = (word & 0x80000000u) != 0u;
    var nw = 0x80000000u;
    var nvis = 0u; var nlit = 0u;
    var lmin = BIG; var lmax = 0.0;
    for (var j = 0u; j < nl; j++) {
      let uj = vec3f(kr(lightIndex + j, KL0, S.s3), kr(lightIndex + j, KL1, S.s4), kr(lightIndex + j, KL2, S.s5));
      let e = lightEval(s, n, M, wo, time, uj, mis, f32(nl));
      if (!e.ok) { continue; }
      nlit++;
      var occS = false;
      if (valid) { occS = ((word >> j) & 1u) != 0u; }
      else {
        occS = traceSceneSkip(e.o, e.wi, 0.0, e.tmax, time, true, 1u, 0u).tri != MISS;
        if (occS) { nw = nw | (1u << j); }
      }
      if (occS) { continue; }
      if (traceSceneSkip(e.o, e.wi, 0.0, e.tmax, time, true, 2u, bitcast<u32>(scn.axis.w)).tri != MISS) { gKey = gKey | 0x20000u; continue; }
      nvis++;
      L = L + e.c;
      let y = luminance(e.c);
      lmin = min(lmin, y); lmax = max(lmax, y);
    }
    if (!valid) { shadowCache[gCacheIdx] = nw; }
    if (nvis > 0u && nvis < nlit) { gKey = gKey | 0x20000u; }       // a penumbra
    // light samples that disagree by more than 20 % (the profile's roll-off, glossy lobes): worth refining
    if (nvis > 1u && lmax - lmin > 0.2 * (lmax + lmin) * 0.5) { gKey = gKey | 0x20000u; }
  } else {
    var cached = -1.0;
    if (CULL_SHADOWS) { cached = cachedVisibility(s, time); }
    L = directLight(s, n, M, wo, time, ul, mis, f32(nl), cached);
    for (var j = 1u; j < nl; j++) {
      let uj = vec3f(kr(lightIndex + j, KL0, S.s3), kr(lightIndex + j, KL1, S.s4), kr(lightIndex + j, KL2, S.s5));
      L = L + directLight(s, n, M, wo, time, uj, mis, f32(nl), cached);
    }
  }
  L = L / f32(nl);
  if (isStatic) {
    if ((opts & (O_NO_INDIRECT | O_DIRECT_ONLY)) == 0u) {
      var E = nodeEfree(s.vi, s.b);
      if ((opts & O_NO_CAPSULE) == 0u) { E = E * capsuleVis(s.p, n); }
      let mu = max(dot(n, wo), 0.0);
      L = L + E * INV_PI * (albedoSpec(s.mat, mu) + albedoDiff(s.mat, mu));
    }
  } else {
    let bs = sampleBSDF(M, n, wo, ub);
    if (bs.w > 0.0 && dot(s.ng, bs.xyz) > 0.0) {
      let l = bs.xyz;
      let thr = evalBSDF(M, n, l, wo) * (dot(n, l) / bs.w);
      L = L + thr * secondary(offsetRay(s.p, s.ng, l), l, time, bs.w, opts, f32(nl));
    }
  }
  if ((opts & O_AID) != 0u) {
    // overview only: a neutral headlight (Lambert on the diffuse albedo, F0 x 0.6 for metals), never in camera feeds
    var alb = M.c;
    if (M.metal > 0.5) { alb = M.c * 0.6; }
    L = L + alb * (max(dot(n, normalize(cam.aid.xyz)), 0.0) * cam.aid.w * INV_PI);
  }
  return L;
}
`;
