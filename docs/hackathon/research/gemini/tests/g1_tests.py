"""Read-only implementation of Gemini G1 Test 1 and Test 2 (#49). Saved export dirs only (DOE.png, recon PNGs, targets).
usage: PYTHONPATH=<runtime src> python g1_tests.py <out.json> <label=export_dir> ...
Test 1: band inter-z / own Pearson on recon PNG (index grid, nu = cycles/px; low nu<=0.08, high 0.08<nu<=0.5) and
        foreground target-plane phase roughness R_psi from the saved uint8 DOE via the official Fresnel.
Test 2: in dark (T_k <= q25) outer (r>20 px) pixels of plane k: Pearson(A_k, T_m) index-aligned vs physically scaled
        T_m (z_k/z_m about the centre index 64), all ordered pairs k!=m; delta = rho_idx - rho_phys."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.phase_codec import load_doe_png
CFG = CaseConfigV1(); N = 128; Z = [CFG.plane(k).distance_m for k in range(4)]
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b, m=None):
    a, b = (a[m], b[m]) if m is not None else (a.ravel(), b.ravel())
    a, b = a - a.mean(), b - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
nu = np.sqrt(np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2)
band = lambda x, lo, hi: np.real(np.fft.ifft2(np.fft.fft2(x) * ((nu > lo) & (nu <= hi))))
def scale_about_center(x, s):
    n = np.arange(N) - N // 2; g = (n * s + N // 2) / (N - 1) * 2 - 1      # sample T_m at index 64 + s*(p-64)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    return F.grid_sample(torch.from_numpy(x)[None, None], torch.from_numpy(np.stack([gx, gy], -1)[None]), mode="bilinear", align_corners=True, padding_mode="zeros")[0, 0].numpy()
yy, xx = np.meshgrid(np.arange(N) - 64, np.arange(N) - 64, indexing="ij"); outer = np.hypot(yy, xx) > 20
res = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1); d = Path(d)
    R = [img(d / f"reconstructed_{k:02d}.png") for k in range(1, 5)]
    T = [img(d / "targets" / f"target_{k:02d}.png") for k in range(1, 5)]
    out = {}
    for name, (lo, hi) in {"low": (-1, 0.08), "high": (0.08, 0.5)}.items():
        Rb = [band(r, lo, hi) for r in R]; Tb = [band(t, lo, hi) for t in T]
        out[f"interz_{name}"] = float(np.nanmean([pear(Rb[i], Rb[j]) for i in range(4) for j in range(i + 1, 4)]))
        out[f"own_{name}"] = float(np.nanmean([pear(Rb[k], Tb[k]) for k in range(4)]))
    phi = load_doe_png(d / "DOE.png", dtype=torch.float64)
    rough = []
    for k in range(4):
        U = propagate_fresnel(torch.polar(torch.ones_like(phi), phi), CFG.plane(k)).field[0, 0].numpy()
        psi, w = np.angle(U), np.abs(U) ** 2; q = np.quantile(T[k], 0.75); om = T[k] > q if (T[k] > q).any() else T[k] >= q
        wr = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        dpx = np.zeros_like(psi); dpx[:, :-1] = wr(np.diff(psi, axis=1)); dpy = np.zeros_like(psi); dpy[:-1] = wr(np.diff(psi, axis=0))
        rough.append(float((w[om] * (dpx[om] ** 2 + dpy[om] ** 2)).sum() / (2 * w[om].sum())))
    out["R_psi"] = rough; out["R_psi_mean"] = float(np.mean(rough))
    dl = []
    for k in range(4):
        m = (T[k] <= np.quantile(T[k], 0.25)) & outer
        for j in range(4):
            if j == k: continue
            ri, rp = pear(R[k], T[j], m), pear(R[k], scale_about_center(T[j], Z[k] / Z[j]), m)
            dl.append((ri, rp))
    dl = np.array(dl); out["rho_idx_mean"], out["rho_phys_mean"] = float(np.nanmean(dl[:, 0])), float(np.nanmean(dl[:, 1]))
    out["delta_shortcut_mean"] = float(np.nanmean(dl[:, 0] - dl[:, 1])); out["frac_pairs_idx_gt_phys"] = float(np.mean(dl[:, 0] > dl[:, 1]))
    res[label] = out
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k, v in res.items():
    print(f"{k:22s} interz low {v['interz_low']:+.2f} high {v['interz_high']:+.2f} | own low {v['own_low']:+.2f} high {v['own_high']:+.2f} | R_psi {v['R_psi_mean']:.2f} | rho_idx {v['rho_idx_mean']:+.3f} rho_phys {v['rho_phys_mean']:+.3f} d {v['delta_shortcut_mean']:+.3f} frac {v['frac_pairs_idx_gt_phys']:.2f}")
