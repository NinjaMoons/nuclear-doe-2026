"""Reusable CPU-only acceptance runner for candidate DOE checkpoints (NIH-MATVEY-LAPTOP-CANDIDATE-QA-011).

Independent of repo physics code: own numpy single-FFT Fresnel, own Pearson/global SSIM, own inter-z matrix.
Optional forward replay uses a USER-PROVIDED loader (module:function -> model or (model, payload));
the model is called as model(targets[1,4,1,H,W] float32 in [0,1]) -> phase [1,1,H,W]. Nothing is
retrofitted into stock V2. Never opens TEST: refuses any export dir whose name is not 'val'.

usage: python -I accept_cpu.py --run-root DIR [--sums SHA256SUMS.txt] [--loader mod:fn --ckpt ARM=PATH ...]
        [--replay-quartets 000,008] [--pythonpath REPO_WORKTREE] --out report.json
"""
import argparse, glob, hashlib, importlib, json, os, re, sys
import numpy as np
from PIL import Image


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_u8(p, mode="L"):
    im = Image.open(p); assert im.mode == mode, (p, im.mode); return np.array(im)


def fresnel_amp(phi, lam, dx, z):
    n = phi.shape[0]; x = (np.arange(n) - n // 2) * dx; X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.exp(1j * phi) * np.exp(1j * np.pi / (lam * z) * (X ** 2 + Y ** 2))
    a = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u))))
    return a / a.max()


def u8_half_up(a): return np.floor(a * 255 + 0.5).clip(0, 255).astype(np.uint8)


def pearson(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel(); a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else float("nan")


def ssim_g(a, b, L=1.0):
    a = a.astype(np.float64); b = b.astype(np.float64); c1, c2 = (.01 * L) ** 2, (.03 * L) ** 2
    ma, mb = a.mean(), b.mean(); va, vb = a.var(), b.var(); cv = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cv + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def check_quartet(q):
    man = json.load(open(os.path.join(q, "manifest.json"))); met = json.load(open(os.path.join(q, "metrics.json")))
    cfg = man["config"]; lam = cfg["wavelength_m"]; dx = cfg["input_pitch_m"][0]; zs = cfg["distances_m"]
    r = {"q": os.path.basename(q), "fail": []}
    src = man.get("input_sources", [{}])[0].get("path", "")
    m = re.search(r"corpus[\\/](\w+?)(\d+)[\\/](\d+)", src); r["key"] = f"{m.group(1)}{m.group(2)}:{m.group(3)}" if m else src
    r["group"] = m.group(1) if m else "?"
    if man.get("observation") != "amplitude": r["fail"].append("observation")
    doe = load_u8(os.path.join(q, "DOE.png"))
    n = doe.shape[0]
    if doe.shape != tuple(cfg["image_shape"]) or doe.dtype != np.uint8: r["fail"].append("doe_format")
    dsha = sha(os.path.join(q, "DOE.png"))
    phi = doe.astype(np.float64) / 255 * 2 * np.pi - np.pi
    recs, tgts, lsb = [], [], []
    for k, z in enumerate(zs):
        pit = man["output_pitches_m"][k][0]
        if abs(pit - lam * z / (n * dx)) > 1e-12: r["fail"].append(f"pitch{k}")
        pl = met["planes"][k]
        if pl["doe_sha256"] != dsha: r["fail"].append(f"doe_sha{k}")
        rec = load_u8(os.path.join(q, f"reconstructed_{k+1:02d}.png")); tgt = load_u8(os.path.join(q, "targets", f"target_{k+1:02d}.png"))
        if rec.shape != (n, n): r["fail"].append(f"rec_shape{k}")
        mine = u8_half_up(fresnel_amp(phi, lam, dx, z))
        lsb.append(int(np.abs(mine.astype(int) - rec.astype(int)).max()))
        rf, tf = rec / 255.0, tgt / 255.0; recs.append(rf); tgts.append(tf)
        p, s = pearson(rf, tf), ssim_g(rf, tf)
        if abs(p - pl["pearson"]) > 1e-6 or abs(s - pl["ssim"]) > 1e-6: r["fail"].append(f"metric{k}")
    r["max_lsb"] = max(lsb)
    if r["max_lsb"] > 1: r["fail"].append("fresnel_replay")
    C = np.array([[pearson(recs[i], tgts[j]) for j in range(4)] for i in range(4)])
    r["P"] = [float(C[i, i]) for i in range(4)]; r["S"] = [ssim_g(recs[k], tgts[k]) for k in range(4)]
    r["own_argmax"] = int(sum(int(np.argmax(C[i]) == i) for i in range(4)))
    off = lambda M: float(np.mean([M[i][j] for i in range(4) for j in range(4) if i != j]))
    r["interz_rec"] = off([[pearson(recs[i], recs[j]) for j in range(4)] for i in range(4)])
    r["interz_tgt"] = off([[pearson(tgts[i], tgts[j]) for j in range(4)] for i in range(4)])
    r["diag_minus_offdiag"] = float(np.mean(np.diag(C)) - off(C))
    r["targets"] = np.stack(tgts)
    return r


def summarize(rows):
    P = np.array([r["P"] for r in rows]); S = np.array([r["S"] for r in rows])
    out = {"n": len(rows), "pearson_mean": float(P.mean()), "ssim_mean": float(S.mean()),
           "pearson_plane": P.mean(0).tolist(), "ssim_plane": S.mean(0).tolist(),
           "worst_plane_pearson": float(P.mean(0).min()), "worst_pair_pearson": float(P.min()),
           "worst_case": min(((r["key"], k, r["P"][k]) for r in rows for k in range(4)), key=lambda t: t[2]),
           "max_lsb": max(r["max_lsb"] for r in rows), "fails": {r["q"]: r["fail"] for r in rows if r["fail"]}}
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        out[f"group_{g}"] = {"quartets": len(rs), "own_argmax": sum(r["own_argmax"] for r in rs), "planes": 4 * len(rs),
                             "interz_rec": float(np.mean([r["interz_rec"] for r in rs])), "interz_tgt": float(np.mean([r["interz_tgt"] for r in rs])),
                             "diag_minus_offdiag": float(np.mean([r["diag_minus_offdiag"] for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True); ap.add_argument("--sums"); ap.add_argument("--out", required=True)
    ap.add_argument("--loader"); ap.add_argument("--ckpt", action="append", default=[]); ap.add_argument("--replay-quartets", default="000,008")
    ap.add_argument("--pythonpath")
    a = ap.parse_args(); rep = {}
    if a.sums:
        base = os.path.dirname(a.sums); bad = 0; tot = 0
        for line in open(a.sums, encoding="utf-8"):
            if not line.strip(): continue
            h, f = line.split(None, 1); f = f.strip().lstrip("*"); tot += 1; bad += sha(os.path.join(base, f)) != h
        rep["sha256sums"] = {"total": tot, "bad": bad}
    arms = {}
    for vd in sorted(glob.glob(os.path.join(a.run_root, "**", "export", "*", "*"), recursive=True)):
        if not os.path.isdir(vd): continue
        assert os.path.basename(vd) == "val", f"refusing non-VAL export dir {vd}"
        arm = os.path.relpath(vd, a.run_root).split(os.sep)[0]
        rows = [check_quartet(q) for q in sorted(glob.glob(os.path.join(vd, "*"))) if os.path.isdir(q)]
        arms[arm] = (vd, rows); rep[arm] = summarize(rows)
    if a.loader:
        if a.pythonpath: sys.path.insert(0, a.pythonpath)
        import torch
        mod, fn = a.loader.split(":"); load = getattr(importlib.import_module(mod), fn)
        for spec in a.ckpt:
            arm, path = spec.split("=", 1); res = load(path); model = res[0] if isinstance(res, tuple) else res
            ck = torch.load(path, map_location="cpu", weights_only=True)
            rep.setdefault("replay", {})[arm] = {"ckpt_sha256": sha(path), "has_optimizer_state": any("optim" in k for k in ck)}
            model.eval(); vd, rows = arms[arm]
            for qn in a.replay_quartets.split(","):
                r = next(r for r in rows if r["q"] == qn)
                t = torch.from_numpy(r["targets"].astype(np.float32))[None, :, None]
                with torch.no_grad(): ph = model(t)[0, 0].double().numpy()
                v = u8_half_up((ph + np.pi) / (2 * np.pi))
                saved = load_u8(os.path.join(vd, qn, "DOE.png")).astype(int)
                d = np.abs(v.astype(int) - saved); d = np.minimum(d, 255 - d)  # phase wrap
                rep["replay"][arm][qn] = {"max_code_diff": int(d.max()), "identical_frac": float((d == 0).mean())}
    json.dump(rep, open(a.out, "w"), indent=1, default=str); print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
