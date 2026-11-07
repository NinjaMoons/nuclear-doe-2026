"""RESEARCH ONLY (R12 #39): paired VAL analysis of arm runs (reads saved exports only, VAL only).
usage: python -I analyze_r12.py <out.json> <label=run_dir> ...   (first label = reference for paired deltas)"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F

N = 128
LAM, DX, ZS = 532e-9, 10e-6, (0.0241, 0.0253, 0.0265, 0.0277)
PITCH = [LAM * z / (N * DX) for z in ZS]


def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def to_z1(img, k):
    n = np.arange(N) - N // 2; g = (n * PITCH[0] / PITCH[k] + N // 2) / (N - 1) * 2 - 1
    gy, gx = np.meshgrid(g, g, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1)[None])
    return F.grid_sample(torch.from_numpy(img)[None, None], grid, mode="bilinear", align_corners=True)[0, 0].numpy()


def band(img, lo, hi):
    f = np.fft.fftfreq(N) / 0.5; r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * ((r > lo) & (r <= hi))))


def mask_spec(doe):
    code = np.asarray(Image.open(doe), dtype=np.float64); phi = -np.pi + 2 * np.pi / 255 * code
    p = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phi)))) ** 2
    f = (np.arange(N) - N // 2) / (N / 2); r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    o = np.argsort(r.ravel()); cum = np.cumsum(p.ravel()[o]) / p.sum()
    w = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    rough = np.concatenate([np.abs(w(np.diff(phi, axis=0))).ravel(), np.abs(w(np.diff(phi, axis=1))).ravel()]).mean()
    return float(p[r > 0.5].sum() / p.sum()), float(r.ravel()[o][np.searchsorted(cum, 0.9)]), float(rough)


def quartet(q):
    man = json.loads((q / "manifest.json").read_text()); key = man["metadata"]["sample_key"]
    assert man["metadata"]["split"] == "val", key
    pl = sorted(json.loads((q / "metrics.json").read_text())["planes"], key=lambda p: p["index"])
    R = [np.asarray(Image.open(q / f"reconstructed_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
    T = [np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
    C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
    Rp = [to_z1(R[k], k) for k in range(4)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    iz = {lab: float(np.nanmean([pear(band(Rp[i], lo, hi) if hi else Rp[i], band(Rp[j], lo, hi) if hi else Rp[j]) for i, j in pairs]))
          for lab, lo, hi in (("full", 0, None), ("low", -1, 0.15), ("band", 0.15, 0.5))}
    amp = np.load(q / "raw.npz")["amplitude_quantized"][0, :, 0].astype(np.float64); pw = amp ** 2
    hf, r90, rough = mask_spec(q / "DOE.png")
    return {"key": key, "g": key[0], "P": [p["pearson"] for p in pl], "S": [p["ssim"] for p in pl], "C": C,
            "own": sum(int(np.nanargmax(C[k]) == k) for k in range(4)),
            "margin": float(np.mean([C[k][k] - max(C[k][j] for j in range(4) if j != k) for k in range(4)])),
            "interz": iz, "hf": hf, "r90": r90, "rough": rough,
            "hot": float(np.mean([(r >= 0.9).mean() for r in R])), "level": float(np.mean([r.mean() for r in R])),
            "top": float(np.mean([pw[k].max() / pw[k].sum() for k in range(4)])),
            "bright": float(np.mean([(amp[k] / amp[k].max() > 0.5).mean() for k in range(4)]))}


def select(run):
    E = json.loads((run / "eval_summary.json").read_text())
    rows = [c for c in E["checkpoints"] if c["name"].startswith("step_")]
    final = max(rows, key=lambda c: c["step"])
    top = max(c["val"]["pearson_mean"] for c in rows)
    near = [c for c in rows if c["val"]["pearson_mean"] >= top - 0.005]
    sel = sorted(near, key=lambda c: (-c["val"]["worst_plane_ssim"], c["step"]))[0]
    return {"final": final, "rule": sel}


def summarize(rows):
    out = {}
    for g in ("ALL", "A", "B"):
        s = [r for r in rows if g == "ALL" or r["g"] == g]
        P = np.array([r["P"] for r in s]); S = np.array([r["S"] for r in s])
        out[g] = {"n": len(s), "P": float(P.mean()), "SSIM": float(S.mean()), "P_z": P.mean(0).tolist(), "S_z": S.mean(0).tolist(),
                  "worst_plane_P": float(P.mean(0).min()), "worst_plane_S": float(S.mean(0).min()),
                  "own_argmax": int(sum(r["own"] for r in s)), "planes": 4 * len(s),
                  "margin": float(np.mean([r["margin"] for r in s])),
                  **{f"interz_{k}": float(np.mean([r["interz"][k] for r in s])) for k in ("full", "low", "band")},
                  **{k: float(np.mean([r[k] for r in s])) for k in ("hf", "r90", "rough", "hot", "level", "top", "bright")}}
    return out


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(np.mean(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    res, per = {}, {}
    for arg in sys.argv[2:]:
        label, run = arg.split("=", 1); run = Path(run)
        sel = select(run); res[label] = {"checkpoints": {}}
        for which, c in sel.items():
            d = run / "eval_val" / "export" / c["name"] / "val"
            rows = [quartet(q) for q in sorted(d.iterdir())]
            res[label]["checkpoints"][which] = {"name": c["name"], "step": c["step"], "sha256": c["checkpoint_sha256"],
                                                "summary": summarize(rows), "rows": rows}
            per[(label, which)] = {r["key"]: r for r in rows}
        tl = json.loads((run / "train_log.json").read_text())
        res[label]["train_elapsed_s"] = tl["elapsed_s"]; res[label]["s_per_step"] = float(np.mean([h["step_s"] for h in tl["history"]]))
    ref = sys.argv[2].split("=", 1)[0]
    deltas = {}
    for label in res:
        if label == ref:
            continue
        for which in ("rule", "final"):
            a, b = per[(ref, which)], per[(label, which)]
            keys = sorted(a)
            dP = [np.mean(b[k]["P"]) - np.mean(a[k]["P"]) for k in keys]
            dS = [np.mean(b[k]["S"]) - np.mean(a[k]["S"]) for k in keys]
            Bk = [k for k in keys if k.startswith("B")]
            dM = [b[k]["margin"] - a[k]["margin"] for k in Bk]
            dO = [b[k]["own"] - a[k]["own"] for k in Bk]
            deltas[f"{label}-{ref}/{which}"] = {"dP": boot(dP), "dSSIM": boot(dS), "dB_margin": boot(dM), "dB_own": boot(dO),
                                                "P_better_quartets": int(sum(x > 0 for x in dP)), "S_better_quartets": int(sum(x > 0 for x in dS))}
    res["paired_deltas_vs_" + ref] = deltas
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for label in [k for k in res if not k.startswith("paired")]:
        for which in ("rule", "final"):
            c = res[label]["checkpoints"][which]; s = c["summary"]
            print(f"{label:11s} {which:5s} step {c['step']:4d} | P {s['ALL']['P']:.4f} S {s['ALL']['SSIM']:.4f} | A P {s['A']['P']:.4f} S {s['A']['SSIM']:.4f} | "
                  f"B P {s['B']['P']:.4f} S {s['B']['SSIM']:.4f} own {s['B']['own_argmax']}/{s['B']['planes']} marg {s['B']['margin']:.3f} "
                  f"iz {s['B']['interz_full']:.3f}/{s['B']['interz_low']:.3f}/{s['B']['interz_band']:.3f} | worst P {s['ALL']['worst_plane_P']:.3f} S {s['ALL']['worst_plane_S']:.3f} | "
                  f"HF {s['ALL']['hf']:.3f} hot {s['ALL']['hot']:.4f} top {s['ALL']['top']:.4f} lvl {s['ALL']['level']:.3f}")
    for k, v in deltas.items():
        print(k, {a: [round(x, 4) for x in b] if isinstance(b, list) else b for a, b in v.items()})


if __name__ == "__main__":
    main()
