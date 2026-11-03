#!/usr/bin/env python3
"""Bake the graspers' rigid colliders for the jelly solver (docs/JELLY.md contact) from web/twin/assets/scene.json.

Each jaw mesh (in its own jaw frame at the hinge, docs/WEB_TWIN.md §4/§13) is cut into K overlapping slabs along the
jaw axis (z) and each slab is replaced by a k-DOP of its convex hull: the planes n.p = max_v n.v for NDIR fixed
directions (Fibonacci sphere + the 6 axes), which contains the hull; the bake reports how far the k-DOP's corners stand
outside the exact hull (Qhull), i.e. the collider's geometric error. A union of convex pieces follows
the Maryland's curve and the jaw taper; it fills the fenestration window and the gaps between teeth (reported as the
hull/mesh volume excess). The shaft + collar is one capsule in the shaft frame S (radius from the meshes).

    .venv/bin/python web/twin/jelly/tools/bake_colliders.py   ->  web/twin/jelly/colliders.json
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN = os.path.abspath(os.path.join(HERE, "..", ".."))
SCENE = os.path.join(TWIN, "assets", "scene.json")
OUT = os.path.join(HERE, "..", "colliders.json")
K = int(os.environ.get("JELLY_SLABS", "8"))
OVERLAP = 0.4e-3
NDIR = int(os.environ.get("JELLY_DIRS", "122"))


def directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    D = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    return np.concatenate([D, np.eye(3), -np.eye(3)])


def load_mesh(scene, name):
    m = next(m for m in scene["meshes"] if m["name"] == name)
    raw = open(os.path.join(TWIN, "assets", m["file"]), "rb").read()
    off, n = m["layout"]["positions"]
    P = np.frombuffer(raw, dtype="<f4", count=3 * n, offset=off).reshape(n, 3).astype(np.float64)
    ioff, ni = m["layout"]["indices"]
    I = np.frombuffer(raw, dtype="<u4", count=ni, offset=ioff).reshape(-1, 3)
    return P, I, hashlib.sha256(raw).hexdigest()[:16]


def mesh_volume(P, I):
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def merged_planes(h):
    eq = h.equations  # n.x + off <= 0 inside
    out = []
    for e in eq:
        if not any(np.linalg.norm(e[:3] - o[:3]) < 1e-7 and abs(e[3] - o[3]) < 1e-10 for o in out):
            out.append(e)
    return [[float(e[0]), float(e[1]), float(e[2]), float(-e[3])] for e in out]   # n.p - d <= 0 inside


def jaw_pieces(P):
    z0, z1 = P[:, 2].min(), P[:, 2].max()
    edges = np.linspace(z0, z1, K + 1)
    pieces = []
    for k in range(K):
        lo, hi = edges[k] - OVERLAP, edges[k + 1] + OVERLAP
        Q = P[(P[:, 2] >= lo) & (P[:, 2] <= hi)]
        if len(Q) < 4:
            continue
        h = ConvexHull(Q)
        V = h.points[h.vertices]
        D = directions(NDIR)
        d = (V @ D.T).max(axis=0)
        c = V.mean(axis=0)
        hs = HalfspaceIntersection(np.concatenate([D, -d[:, None]], 1), c)
        corners = hs.intersections
        err = float(np.max(corners @ h.equations[:, :3].T + h.equations[:, 3]))   # how far k-DOP corners stand outside
        dop = ConvexHull(corners)
        r = float(np.linalg.norm(corners - c, axis=1).max()) + 1e-4
        pieces.append(dict(planes=[float(v) for k_ in range(len(D)) for v in (*D[k_], d[k_])], center=[float(v) for v in c], radius=r,
                           z_range=[float(lo), float(hi)], hull_volume_m3=float(h.volume), dop_volume_m3=float(dop.volume),
                           dop_max_outside_hull_m=err))
    return pieces


def main():
    scene = json.load(open(SCENE))
    out = dict(source=dict(scene=hashlib.sha256(open(SCENE, "rb").read()).hexdigest()[:16], slabs=K, overlap_m=OVERLAP),
               note="planes: flat [nx, ny, nz, d] * k in the body frame, inside where n.p - d < 0 for every plane",
               instruments={})
    for ins in scene["instruments"]:
        e = dict()
        for part in ins["parts"]:
            if part["body"] not in ("jaw_a", "jaw_b"):
                continue
            P, I, sha = load_mesh(scene, part["mesh"])
            pieces = jaw_pieces(P)
            vm = mesh_volume(P, I)
            vh = sum(p["hull_volume_m3"] for p in pieces)
            e[part["body"]] = dict(mesh=part["mesh"], mesh_sha=sha, pieces=pieces, mesh_volume_m3=vm,
                                   hull_union_volume_upper_m3=vh, tip_in_jaw_frame=ins.get("tip_in_jaw_frame"))
            print(f"{ins['id']} {part['body']:6s} {part['mesh']:9s} {len(pieces)} pieces x {len(pieces[0]['planes']) // 4} planes, "
                  f"mesh {vm * 1e9:.1f} mm^3, hull sum {vh * 1e9:.1f} mm^3 (overlaps counted twice), "
                  f"k-DOP corners outside the hull <= {max(p['dop_max_outside_hull_m'] for p in pieces) * 1e3:.3f} mm")
        # shaft + collar: one capsule along S z; radius = the largest radial extent of shaft/collar meshes
        rmax = 0.0
        for part in ins["parts"]:
            if part["body"] == "shaft" and part["mesh"].endswith(("_shaft", "_collar")):
                P, _, _ = load_mesh(scene, part["mesh"])
                rmax = max(rmax, float(np.hypot(P[:, 0], P[:, 1]).max()))
        r = rmax or ins["spec"]["shaft_radius"]
        z_top = ins.get("collar", {}).get("to_S_z", 0.0)
        e["shaft"] = dict(kind="capsule", a=[0.0, 0.0, float(ins.get("shaft_from_S_z", -0.31))], b=[0.0, 0.0, float(z_top - r)], r=r,
                          note="flat-ended collar approximated by a capsule whose cap ends at the collar face")
        out["instruments"][ins["id"]] = e
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", os.path.relpath(OUT), os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    sys.exit(main())
