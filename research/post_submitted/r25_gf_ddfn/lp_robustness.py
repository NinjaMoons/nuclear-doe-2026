"""R25 LP robustness diagnostic (#51/6719481693 step 2). Analysis only: reuses the existing frozen-V2 official exports of the
16 pre-selected TRAIN B/independent quartets (train_subtypes.py, keys fixed before metrics). No model, no learning, no TEST.
Question: is "LP inter-z of V2 reconstructions > .9" robust, or a resampling / border / mean-field artifact?
Variants: sigma in {0,1,2,4} px; grid native (index-aligned) vs physical registration onto z1 (bilinear | bicubic);
LP padding reflect | zero; full frame vs central 96x96 ROI.
Controls: (i) SAME-quartet inter-z vs CROSS-quartet inter-z (plane k of quartet i vs plane l of quartet i+1): if cross is
as high as same, the shared band is a generic envelope (mean field), not quartet-specific ghosting; (ii) corr(R_k, own T_k)
vs corr(R_k, mean_j T_j) vs corr(R_k, mean of the 3 OTHER targets); (iii) per-plane least-squares gain g_k=<R,T>/<R,R> on
the native grid and brightness; (iv) constant / near-constant targets count.
usage: python -m r25_gf_ddfn.lp_robustness EXPORT_VAL_DIR OUT.json"""
import json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from r25_gf_ddfn.data_truth_audit import PITCH, pear

N = 128


def register(img, k, mode):
    if mode == "native" or k == 0:
        return img
    n = np.arange(N) - N // 2
    g = (n * PITCH[0] / PITCH[k] + N // 2) / (N - 1) * 2 - 1
    gy, gx = np.meshgrid(g, g, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1)[None]).double()
    return F.grid_sample(torch.from_numpy(img)[None, None], grid, mode=mode, align_corners=True)[0, 0].numpy()


def lp(img, s, pad):
    if s == 0:
        return img
    r = int(3 * s); x = np.arange(-r, r + 1); k = np.exp(-x ** 2 / (2 * s * s)); k /= k.sum()
    a = np.pad(img, r, mode="reflect" if pad == "reflect" else "constant")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)


def roi(img, kind):
    return img if kind == "full" else img[16:112, 16:112]


def main():
    d = Path(sys.argv[1]); qs = sorted(d.iterdir())
    R, T, keys, met = [], [], [], []
    for q in qs:
        keys.append(json.loads((q / "manifest.json").read_text(encoding="utf-8"))["metadata"]["sample_key"])
        R.append([np.asarray(Image.open(q / f"reconstructed_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)])
        T.append([np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)])
        met.append(json.loads((q / "metrics.json").read_text(encoding="utf-8"))["planes"])
    n = len(qs); rep = {"export_dir": str(d), "keys": keys, "n_quartets": n}
    rep["native_official"] = {"pearson_plane": np.mean([[p["pearson"] for p in m] for m in met], 0).tolist(),
                              "ssim_plane": np.mean([[p["ssim"] for p in m] for m in met], 0).tolist()}
    rep["gain_native"] = np.mean([[float((R[i][k] * T[i][k]).sum() / max((R[i][k] ** 2).sum(), 1e-12)) for k in range(4)] for i in range(n)], 0).tolist()
    rep["brightness_rec_tgt"] = [float(np.mean([[r.mean() for r in x] for x in R])), float(np.mean([[t.mean() for t in x] for x in T]))]
    rep["near_constant_targets"] = int(sum(t.std() < 1e-3 for x in T for t in x))
    rows = []
    for grid in ("native", "bilinear", "bicubic"):
        for pad in ("reflect", "zero"):
            for rk in ("full", "roi96"):
                for s in (0, 1, 2, 4):
                    if s == 0 and pad == "zero":
                        continue
                    f = lambda img, k: roi(lp(register(img, k, grid), s, pad), rk)
                    Rf = [[f(R[i][k], k) for k in range(4)] for i in range(n)]
                    Tf = [[f(T[i][k], k) for k in range(4)] for i in range(n)]
                    pairs = list(combinations(range(4), 2))
                    same_r = np.mean([pear(Rf[i][a], Rf[i][b]) for i in range(n) for a, b in pairs])
                    cross_r = np.mean([pear(Rf[i][a], Rf[(i + 1) % n][b]) for i in range(n) for a, b in pairs])
                    same_t = np.mean([pear(Tf[i][a], Tf[i][b]) for i in range(n) for a, b in pairs])
                    cross_t = np.mean([pear(Tf[i][a], Tf[(i + 1) % n][b]) for i in range(n) for a, b in pairs])
                    own = np.mean([pear(Rf[i][k], Tf[i][k]) for i in range(n) for k in range(4)])
                    mean4 = np.mean([pear(Rf[i][k], sum(Tf[i]) / 4) for i in range(n) for k in range(4)])
                    oth3 = np.mean([pear(Rf[i][k], sum(Tf[i][j] for j in range(4) if j != k) / 3) for i in range(n) for k in range(4)])
                    rows.append({"grid": grid, "lp_pad": pad, "roi": rk, "sigma": s, "interz_rec_same": float(same_r),
                                 "interz_rec_cross_quartet": float(cross_r), "interz_tgt_same": float(same_t),
                                 "interz_tgt_cross_quartet": float(cross_t), "corr_rec_own": float(own),
                                 "corr_rec_mean4": float(mean4), "corr_rec_other3": float(oth3)})
    rep["rows"] = rows
    Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "keys")}))
    print("grid pad roi sigma | rec_same rec_cross | tgt_same tgt_cross | own mean4 other3")
    for r in rows:
        print(r["grid"], r["lp_pad"], r["roi"], r["sigma"], "|", *(f"{r[k]:.3f}" for k in ("interz_rec_same", "interz_rec_cross_quartet",
              "interz_tgt_same", "interz_tgt_cross_quartet", "corr_rec_own", "corr_rec_mean4", "corr_rec_other3")))


if __name__ == "__main__":
    main()
