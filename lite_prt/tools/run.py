"""Driver for the lite PRT renderer (owner: lite_prt racer). Run on the GPU pod under gpu_guard:

    python3 web/twin/lite_prt/tools/run.py smoke [--prt 0|1]
    python3 web/twin/lite_prt/tools/run.py gen --n 600 --out DIR          # PRT training data (node solver, ?prt=0)
    python3 web/twin/lite_prt/tools/run.py fit --data DIR                  # PCA + RBF, held-out check, writes assets/
    python3 web/twin/lite_prt/tools/run.py judge --out DIR [--prt 0|1]     # capture all judge states x CAM1-3 + Mitsuba compare
    python3 web/twin/lite_prt/tools/run.py perf --out DIR
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LITE = HERE.parent
REPO = LITE.parents[2]
sys.path.insert(0, str(REPO / "src"))


def write_physics():
    import yaml
    src = REPO / "configs" / "twin" / "physics.yaml"
    (LITE / "assets").mkdir(exist_ok=True)
    (LITE / "assets" / "physics.json").write_text(json.dumps(yaml.safe_load(src.read_text(encoding="utf-8"))))


def inherit_lock():
    """Under `gpu_guard run` our parent holds the machine-wide lock for us; a second flock here would deadlock."""
    import os
    from veridien.webtwin import gpu_guard
    holder = gpu_guard.LOCK_PATH.read_text().split()[0] if gpu_guard.LOCK_PATH.exists() else ""
    if holder == str(os.getppid()):
        gpu_guard._depth = max(gpu_guard._depth, 1)


def page(prt: str = "1", extra: str = ""):
    from veridien.webtwin.headless import TwinPage
    inherit_lock()
    q = f"lite_prt/judge.html?prt={prt}" + (f"&{extra}" if extra else "")
    return TwinPage(path=q, harness=False, ready_timeout=600)


def dec(s, dtype=np.float32):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def render(p, state, cams, state1=None, seed=0):
    o = {"cameras": cams, "linear": True, "seed": seed}
    if state1:
        o["state1"] = state1
    r = p.page.evaluate("async ({s, o}) => await window.twinJudge.renderB64(s, o)", {"s": state, "o": o})
    out = {}
    for c in cams:
        w, h = r[c]["width"], r[c]["height"]
        out[c] = dec(r[c]["linear"]).reshape(h, w, 4)
    return out, r["_meta"]


def clamped_states():
    d = json.loads(Path("/workspace/refs/clamped_states.json").read_text())
    return d


def cmd_smoke(a):
    write_physics()
    with page(a.prt, a.extra) as p:
        print("info", json.dumps(p.page.evaluate("() => window.twinJudge.info()"), default=str)[:3000])
        st = clamped_states()["default"]
        t0 = time.time()
        imgs, meta = render(p, st["state"], a.cams.split(","))
        print("render s", time.time() - t0, "converge ms", meta.get("converge_ms"))
        for c, im in imgs.items():
            Y = im[..., :3].mean(-1)
            print(c, im.shape, "mean", float(Y.mean()), "p99", float(np.percentile(Y, 99)), "nan", int(np.isnan(im).sum()))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "smoke.npz", **{c: v for c, v in imgs.items()})
        tl = meta.get("timings", {})
        print("frames", json.dumps(tl.get("frames", [])[-4:])[:3000])
        print("maxSubmit", tl.get("maxSubmit"), "over", tl.get("over", [])[-3:])
        print("console", "\n".join(p.console[-30:]))


def sample_scopes(n, seed, base):
    rng = np.random.default_rng(seed)
    # stratified (scrambled Sobol) over yaw, pitch, roll, insertion around the default aim
    from scipy.stats import qmc
    u = qmc.Sobol(4, scramble=True, seed=seed).random(n)
    lo = np.array([-0.32, base["pitch"] - 0.28, -1.6, base["insertion"] - 0.0015])
    hi = np.array([0.32, base["pitch"] + 0.24, 1.6, base["insertion"] + 0.035])
    x = lo + u * (hi - lo)
    # greedy nearest-neighbour order so each solve warm-starts from a nearby pose
    z = (x - lo) / (hi - lo)
    order, left = [0], set(range(1, n))
    while left:
        L = np.fromiter(left, int)
        j = int(L[np.argmin(((z[L] - z[order[-1]]) ** 2).sum(1))])
        order.append(j); left.remove(j)
    x = x[order]
    del rng
    return [{"yaw": float(v[0]), "pitch": float(v[1]), "roll": float(v[2]), "insertion": float(v[3])} for v in x]


def cmd_gen(a):
    write_physics()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = clamped_states()
    base = st["default"]["state"]
    scopes = []
    # the judge states' scope poses are NOT in the training set (held out by construction); training = Sobol samples
    scopes += sample_scopes(a.n, a.seed, base["scope"])
    held = sample_scopes(a.nheld, a.seed + 1000, base["scope"])
    judge_sc = [v["state"]["scope"] for v in st.values()]
    with page("0") as p:
        info = p.page.evaluate("() => window.twinJudge.nodeInfo()")
        act = dec(p.page.evaluate("() => window.twinJudge.activeB64()"), np.uint32)
        np.save(out / "active.npy", act)
        (out / "info.json").write_text(json.dumps(info))
        print("nodes", info, flush=True)
        for tag, lst in (("judge", judge_sc), ("held", held), ("train", scopes)):
            feats, E, sc = [], [], []
            t0 = time.time()
            for i in range(0, len(lst), a.batch):
                r = p.page.evaluate("async ({s, o}) => await window.twinJudge.gen(s, o)", {"s": lst[i:i + a.batch], "o": {"base": base, "iters": a.iters}})
                for x in r:
                    feats.append(x["feat"]); sc.append(x["scope"])
                    E.append(dec(x["E"]).astype(np.float16).reshape(-1, 3))
                print(f"[gen] {tag} {i + len(r)}/{len(lst)} {time.time() - t0:.1f} s", flush=True)
            np.savez(out / f"{tag}.npz", feat=np.array(feats), E=np.stack(E), scope=json.dumps(sc))


def tps(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = r * r * np.log(r + 1e-12)
    return v


def rbf_fit(X, Y, smooth=1e-6):
    N, F = X.shape
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    A = tps(D) + smooth * np.eye(N)
    P = np.hstack([X, np.ones((N, 1))])
    M = np.zeros((N + F + 1, N + F + 1))
    M[:N, :N] = A; M[:N, N:] = P; M[N:, :N] = P.T
    rhs = np.zeros((N + F + 1, Y.shape[1])); rhs[:N] = Y
    sol = np.linalg.solve(M, rhs)
    return sol[:N], sol[N:]          # weights (N x K), linear (F+1 x K)


def rbf_eval(X, C, Wt, Ln):
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    return tps(D) @ Wt + np.hstack([X, np.ones((len(X), 1))]) @ Ln


LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def cmd_fit(a):
    d = Path(a.data)
    tr = np.load(d / "train.npz"); he = np.load(d / "held.npz"); ju = np.load(d / "judge.npz")
    act = np.load(d / "active.npy"); info = json.loads((d / "info.json").read_text())
    Xtr, Etr = tr["feat"], tr["E"].astype(np.float32)          # (N, n, 3)
    N, n, _ = Etr.shape
    Ytr = Etr @ LUM                                              # (N, n)
    chroma = (Etr.sum(0) / np.maximum(Ytr.sum(0), 1e-12)[:, None]).astype(np.float32)   # (n, 3) rgb / Y
    mean = Ytr.mean(0)
    Yc = Ytr - mean
    G = Yc @ Yc.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    res = {}
    for K in [int(k) for k in a.k.split(",")]:
        B = (Yc.T @ V[:, :K]) / np.sqrt(np.maximum(w[:K], 1e-30))   # (n, K) orthonormal columns
        coef = Yc @ B
        for sm in [float(x) for x in a.smooth.split(",")]:
            Wt, Ln = rbf_fit(Xtr, coef, sm)
            out = {}
            for tag, z in (("held", he), ("judge", ju)):
                Xh, Eh = z["feat"], z["E"].astype(np.float32)
                Lt = Eh @ LUM
                ch = rbf_eval(Xh, Xtr, Wt, Ln)
                Lp = mean + ch @ B.T
                Lj = mean + ((Lt - mean) @ B) @ B.T
                Ergb = Lp[..., None] * chroma[None]
                ok = Lt > 0.02 * np.percentile(Lt, 99, axis=1, keepdims=True)
                rel = np.abs(Lp - Lt)[ok] / Lt[ok]
                relp = np.abs(Lj - Lt)[ok] / Lt[ok]
                relc = (np.abs(Ergb - Eh).max(-1))[ok] / Lt[ok]
                lvl = (Lp * ok).sum(1) / (Lt * ok).sum(1)
                out[tag] = {"median_rel": float(np.median(rel)), "p95_rel": float(np.percentile(rel, 95)),
                            "pca_only_median": float(np.median(relp)), "pca_only_p95": float(np.percentile(relp, 95)),
                            "rgb_maxch_median": float(np.median(relc)), "level_min": float(lvl.min()), "level_max": float(lvl.max())}
            res[(K, sm)] = out
            print("K", K, "smooth", sm, "explained", float(w[:K].sum() / w.sum()), json.dumps(out), flush=True)
    K, best = min(res, key=lambda t: res[t]["held"]["median_rel"] + 0.25 * res[t]["held"]["p95_rel"])
    B = (Yc.T @ V[:, :K]) / np.sqrt(np.maximum(w[:K], 1e-30))
    coef = Yc @ B
    Wt, Ln = rbf_fit(Xtr, coef, best)
    basis = np.vstack([mean[None, :], B.T]).astype(np.float16).reshape(-1)    # (K+1) x n
    chf = chroma.astype(np.float16).reshape(-1)
    parts, meta, off = [], {}, 0
    for name, arr in (("active", act.astype(np.uint32)), ("centers", Xtr.astype(np.float32)), ("weights", Wt.astype(np.float32)),
                      ("lin", Ln.astype(np.float32)), ("basis", basis), ("chroma", chf)):
        b = arr.tobytes()
        meta[f"{name}_offset"] = off
        parts.append(b); off += len(b)
        pad = (-off) % 16
        parts.append(b"\0" * pad); off += pad
    meta.update({"static_verts": int(info["staticVerts"]), "n_active": int(n), "k": int(K), "n_centers": int(N), "n_feat": int(Xtr.shape[1]),
                 "rbf_kernel": "tps", "rbf_eps": 1.0, "rbf_smooth": best, "basis_file": "prt.bin",
                 "features": "emitter centroid - C pivot (/0.03 m), emission axis (D frame); renderer.js scopeFeatures",
                 "heldout": res[(K, best)], "source": str(d), "explained": float(w[:K].sum() / w.sum())})
    outd = Path(a.assets)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "prt.bin").write_bytes(b"".join(parts))
    (outd / "prt.json").write_text(json.dumps(meta, indent=1))
    print("wrote", outd, off / 1e6, "MB, K", K, "smooth", best, flush=True)


def cmd_judge(a):
    import veridien.webtwin.judges as JG
    from veridien.webtwin.judges import judge_mitsuba_offline, write_json
    try:
        import matplotlib  # noqa: F401
    except ImportError:          # the pod has no matplotlib: numbers only, no error figures
        JG.error_figure = lambda *x, **k: None
    inherit_lock()
    write_physics()
    out = Path(a.out)
    cap = out / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    st = clamped_states()
    names = a.states.split(",") if a.states else list(st)
    cams = ["CAM1", "CAM2", "CAM3"]
    meta = {"source": f"lite_prt/judge.html?prt={a.prt}&{a.extra}", "query": a.extra, "frames": 1, "warmup": 0, "states": {}}
    if a.skip_capture:
        names = []
    with page(a.prt, a.extra) if names else _null() as p:
        if p is not None:
            meta["info"] = p.page.evaluate("() => window.twinJudge.info()")
        for s in names:
            v = st[s]
            t0 = time.time()
            imgs, m = render(p, v["state"], cams + (["SCOPE"] if s == "default" else []), v.get("state1"), seed=1)
            np.savez_compressed(cap / f"{s}.npz", **{c: imgs[c][None, ..., :3].astype(np.float32) for c in cams})
            if s == "default":
                np.save(out / "scope_default.npy", imgs["SCOPE"][..., :3].astype(np.float32))
            meta["states"][s] = {"state": v["state"], "state1": v.get("state1"), "seconds": time.time() - t0}
            print(f"[capture] {s} {time.time() - t0:.1f} s", flush=True)
        if names:
            meta["timings"] = p.page.evaluate("() => window.twinJudge.timings()")
            meta["console"] = p.console[-50:]
    if names:
        write_json(cap / "capture.json", meta)
    res = judge_mitsuba_offline({"lite_prt": cap}, REPO / "web/twin/assets/scene.json", REPO / "configs/twin/physics.yaml", out,
                                spp=4096, seed=1, noise=False)
    write_json(out / "mitsuba.json", res)
    pr = res["pooled"]["lite_prt"]
    print("POOLED", json.dumps({k: pr[k] for k in ("live", "verdict")}, default=str)[:2500])
    rows = []
    for sname, sres in res["states"].items():
        for cam, cres in sres["cameras"].items():
            c = cres["lite_prt"]["live"]
            o = c["overall"]
            reg = {k: (v.get("median_rel") if isinstance(v, dict) else None) for k, v in (c.get("regions") or {}).items()}
            rows.append((sname, cam, o.get("level_ratio"), o.get("median_rel"), o.get("p95_rel"), reg, (c.get("edges") or {}).get("all", {}).get("median_abs_px")))
            print(f"{sname:16s} {cam} level {o.get('level_ratio'):.4f} med {o.get('median_rel'):.4f} p95 {o.get('p95_rel'):.4f} edges {rows[-1][-1]} regions {json.dumps(reg)}", flush=True)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *e):
        return False


def cmd_perf(a):
    write_physics()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = clamped_states()["default"]["state"]
    with page(a.prt, a.extra) as p:
        info = p.page.evaluate("() => window.twinJudge.info()")
        r = p.page.evaluate("async ({s, c, n}) => await window.twinJudge.perf(s, c, n, {move: true})", {"s": st, "c": ["SCOPE", "CAM1", "CAM2", "CAM3"], "n": a.n})
    summ = {"loadMs": info.get("loadMs"), "stats": info.get("stats")}
    for cam, v in r.items():
        fr = v["frames"][5:]
        keys = sorted({k for f in fr for k in f if k not in ("frame", "submits", "maxSubmit")})
        tot = [sum(f.get(k, 0) for k in keys if not k.startswith("radiosity") and "prt" not in k) for f in fr]
        summ[cam] = {"gpu_ms_median": float(np.median(tot)), "wall_ms_median": float(np.median(v["wall"][5:])),
                     "per_pass": {k: float(np.median([f.get(k, 0) for f in fr])) for k in keys}}
    (out / "perf.json").write_text(json.dumps({"summary": summ, "raw": r}, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str)[:4000])


def cmd_contact(a):
    """Mitsuba | candidate for CAM1-3 at the default state, plus the SCOPE view; one display transform for both."""
    from PIL import Image, ImageDraw
    from veridien.webtwin.judges import reference_cached, poses_for_state
    from veridien.webtwin.mitsuba_ref import Physics, Reference, TwinScene
    inherit_lock()
    d = Path(a.run)
    scene = TwinScene.load(REPO / "web/twin/assets/scene.json"); phys = Physics.load(REPO / "configs/twin/physics.yaml")
    ref = Reference(scene, phys, max_depth=1024, rr_depth=16)
    st = clamped_states()["default"]
    p0, _ = poses_for_state(scene, st["state"])
    z = np.load(d / "capture" / "default.npz")
    tiles = []
    def tm(x, s):
        y = np.clip(x / s, 0, 1)
        y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
        return (np.clip(y, 0, 1) * 255).astype(np.uint8)
    for cam in ("CAM1", "CAM2", "CAM3"):
        r = reference_cached(ref, cam, {"name": "default", **st["state"]}, p0, None, 4096, 64, 1, None, print)
        R = r.mean.astype(np.float32); C = z[cam][0]
        Y = R.mean(-1); s = float(np.percentile(Y[Y > 0], 99.5)) * 1.1
        tiles.append((f"{cam} Mitsuba 4096 spp", tm(R, s)))
        tiles.append((f"{cam} lite_prt (median |rel| in run json)", tm(C, s)))
    S = np.load(d / "scope_default.npy")
    Ys = S.mean(-1); tiles.append(("SCOPE lite_prt (75 deg pinhole, circle 0.95)", tm(S, float(np.percentile(Ys[Ys > 0], 99.5)) * 1.1)))
    w, h = 640, 360
    W = Image.new("RGB", (2 * w + 30, 4 * (h + 28) + 10), (250, 249, 246))
    dr = ImageDraw.Draw(W)
    for i, (lab, im) in enumerate(tiles):
        x = 10 + (i % 2) * (w + 10); y = 10 + (i // 2) * (h + 28)
        if i == 6: x = 10 + (w + 10) // 2
        W.paste(Image.fromarray(im).resize((w, h), Image.BILINEAR), (x, y + 18))
        dr.text((x, y + 2), lab, fill=(20, 20, 20))
    W.save(d / "contact.png")
    print("wrote", d / "contact.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    s = sp.add_parser("smoke"); s.add_argument("--prt", default="1"); s.add_argument("--extra", default=""); s.add_argument("--cams", default="CAM1,SCOPE"); s.add_argument("--out", default="/root/race_out/smoke")
    s = sp.add_parser("gen"); s.add_argument("--n", type=int, default=512); s.add_argument("--nheld", type=int, default=48); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--iters", type=int, default=64); s.add_argument("--batch", type=int, default=8); s.add_argument("--out", required=True)
    s = sp.add_parser("fit"); s.add_argument("--data", required=True); s.add_argument("--k", default="16,24,32"); s.add_argument("--smooth", default="1e-8,1e-6,1e-4,1e-3")
    s.add_argument("--assets", default=str(LITE / "assets"))
    s = sp.add_parser("judge"); s.add_argument("--out", required=True); s.add_argument("--prt", default="1"); s.add_argument("--extra", default=""); s.add_argument("--states", default=""); s.add_argument("--skip-capture", action="store_true")
    s = sp.add_parser("perf"); s.add_argument("--out", required=True); s.add_argument("--prt", default="1"); s.add_argument("--extra", default=""); s.add_argument("--n", type=int, default=40)
    s = sp.add_parser("contact"); s.add_argument("--run", required=True)
    a = ap.parse_args()
    {"smoke": cmd_smoke, "gen": cmd_gen, "fit": cmd_fit, "judge": cmd_judge, "perf": cmd_perf, "contact": cmd_contact}[a.cmd](a)
