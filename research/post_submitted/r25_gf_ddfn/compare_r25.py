"""R25 post-pilot comparison on saved official VAL16 exports (read-only; reloaded DOE.png / reconstructed_0k.png only).
usage: python -m r25_gf_ddfn.compare_r25 OUT_DIR v2=EXPORT_VAL_DIR control=EXPORT_VAL_DIR treatment=EXPORT_VAL_DIR
Per model: per-z Pearson/SSIM (from metrics.json), A/B, worst plane, own-target argmax over 4x4 C[k,j] (native and
physically registered on the z1 grid), inter-z reconstruction corr (physical), brightness, dark fraction (rec < .05 where
target > .25), speckle (std/mean of rec inside target > .5 mask, minus the same for the target), DOE phase roughness
(mean |wrapped neighbour difference| of the reloaded L8 DOE).  Paired treatment - control deltas with a 10000x quartet
bootstrap 95% CI per group.  Contact sheets: row TARGET (4 z), DOE.png, then rec rows per model, for B64:012, B64:016, A64:060.
"""
import json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from r25_gf_ddfn.data_truth_audit import pear, to_z1

SHEET_KEYS = ("B64:012", "B64:016", "A64:060")


def lp(img, sigma=2.0):  # separable Gaussian low-pass (reflect), used to show decorrelation is not just added noise
    r = int(3 * sigma); x = np.arange(-r, r + 1); k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    pad = np.pad(img, r, mode="reflect")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, pad)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)


def own(C, k):  # 1 if row k's best-matching target is k; undefined rows (all NaN, e.g. constant image) count as 0
    r = np.asarray(C[k], dtype=float)
    return 0 if np.all(np.isnan(r)) else int(np.nanargmax(r) == k)


def nm(x):  # nan-safe mean (NaN if nothing defined), no RuntimeWarning
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def wrapdiff(a):
    return np.abs((a + np.pi) % (2 * np.pi) - np.pi)


def load(d):
    rows = {}
    for q in sorted(Path(d).iterdir()):
        man = json.loads((q / "manifest.json").read_text(encoding="utf-8")); key = man["metadata"]["sample_key"]
        R = [np.asarray(Image.open(q / f"reconstructed_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
        T = [np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
        doe = np.asarray(Image.open(q / "DOE.png"), dtype=np.float64) * 2 * np.pi / 255 - np.pi
        met = json.loads((q / "metrics.json").read_text(encoding="utf-8"))["planes"]
        Cn = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
        Rp, Tp = [to_z1(R[k], k) for k in range(4)], [to_z1(T[k], k) for k in range(4)]
        Cp = [[pear(Rp[k], Tp[j]) for j in range(4)] for k in range(4)]
        RL, TL = [lp(x) for x in Rp], [lp(x) for x in Tp]
        CL = [[pear(RL[k], TL[j]) for j in range(4)] for k in range(4)]
        dark = [float(((R[k] < .05) & (T[k] > .25)).sum() / max((T[k] > .25).sum(), 1)) for k in range(4)]
        def spk(img, m):
            v = img[m]; return float(v.std() / v.mean()) if m.sum() > 10 and v.mean() > 0 else np.nan
        speck = [spk(R[k], T[k] > .5) - spk(T[k], T[k] > .5) for k in range(4)]
        rows[key] = {"g": key[0], "P": [p["pearson"] if p.get("pearson_defined", p["pearson"] is not None) else float("nan") for p in met], "S": [p["ssim"] for p in met],
                     "own_native": sum(own(Cn, k) for k in range(4)),
                     "own_phys": sum(own(Cp, k) for k in range(4)),
                     "diag_off_phys": (nm([Cp[k][k] for k in range(4)]) - nm([Cp[k][j] for k in range(4) for j in range(4) if j != k])),
                     "interz_rec": nm([pear(Rp[i], Rp[j]) for i, j in combinations(range(4), 2)]),
                     "interz_rec_lp": nm([pear(RL[i], RL[j]) for i, j in combinations(range(4), 2)]),
                     "own_phys_lp": sum(own(CL, k) for k in range(4)),
                     "diag_off_phys_lp": (nm([CL[k][k] for k in range(4)]) - nm([CL[k][j] for k in range(4) for j in range(4) if j != k])),
                     "interz_tgt_lp": nm([pear(TL[i], TL[j]) for i, j in combinations(range(4), 2)]),
                     "interz_tgt": nm([pear(Tp[i], Tp[j]) for i, j in combinations(range(4), 2)]),
                     "bright": float(np.mean([r.mean() for r in R])), "dark": float(np.mean(dark)),
                     "speckle_excess": nm(speck),
                     "doe_rough": float((wrapdiff(np.diff(doe, axis=0)).mean() + wrapdiff(np.diff(doe, axis=1)).mean()) / 2),
                     "_R": R, "_T": T, "_doe": doe}
    return rows


def summarize(rows):
    out = {}
    for G in ("A", "B", "all"):
        r = [v for v in rows.values() if G == "all" or v["g"] == G]
        P = np.array([v["P"] for v in r], dtype=float); S = np.array([v["S"] for v in r], dtype=float)
        out[G] = {"PRIMARY_official_native": True, "n_quartets": len(r), "undefined_pearson": int(np.isnan(P).sum()),
                  "pearson": float(np.nanmean(P)), "ssim": float(S.mean()), "pearson_plane": np.nanmean(P, 0).tolist(),
                  "worst_plane_pearson": float(np.nanmean(P, 0).min()), "own_phys": sum(v["own_phys"] for v in r),
                  "own_native": sum(v["own_native"] for v in r),
                  "own_phys_lp": sum(v["own_phys_lp"] for v in r), "planes": 4 * len(r)}
        for k in ("diag_off_phys", "diag_off_phys_lp", "interz_rec", "interz_rec_lp", "interz_tgt_lp", "interz_tgt", "bright", "dark", "speckle_excess", "doe_rough"):
            out[G][k] = nm([v[k] for v in r])  # DIAGNOSTIC (registration / interpolation / ROI dependent)
    return out


def boot(d, n=10000, seed=2026):  # quartet-level paired bootstrap; with n=8 per group this is NOT evidence of generalization
    rng = np.random.default_rng(seed); d = np.asarray(d, dtype=float); d = d[~np.isnan(d)]
    if d.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(m, .025)), float(np.quantile(m, .975))], "n": int(d.size)}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    models = dict(a.split("=", 1) for a in sys.argv[2:])
    data = {m: load(d) for m, d in models.items()}
    keys = sorted(next(iter(data.values())))
    assert all(sorted(v) == keys for v in data.values()), "different quartets"
    for m, v in data.items():  # same target pixels for every model (no target swap)
        for k in keys:
            assert all(np.array_equal(a, b) for a, b in zip(v[k]["_T"], data[next(iter(data))][k]["_T"])), (m, k)
    rep = {"note": "PRIMARY = official native-grid Pearson/SSIM from metrics.json; inter-z (physically registered, bilinear), "
                   "lp(sigma=2) and own-argmax are DIAGNOSTICS; CIs are quartet-level paired bootstrap (n=8 per group).",
           "models": models, "summary": {m: summarize(v) for m, v in data.items()}}
    if "control" in data and "treatment" in data:
        c, t = data["control"], data["treatment"]; rep["paired_treatment_minus_control"] = {}
        for G in ("A", "B", "all"):
            ks = [k for k in keys if G == "all" or k[0] == G]
            rep["paired_treatment_minus_control"][G] = {
                "PRIMARY_pearson": boot([nm(t[k]["P"]) - nm(c[k]["P"]) for k in ks]),
                "PRIMARY_ssim": boot([nm(t[k]["S"]) - nm(c[k]["S"]) for k in ks]),
                "PRIMARY_worst_plane_quartet": boot([np.nanmin(t[k]["P"]) - np.nanmin(c[k]["P"]) for k in ks]),
                "interz_rec": boot([t[k]["interz_rec"] - c[k]["interz_rec"] for k in ks]),
                "diag_off_phys": boot([t[k]["diag_off_phys"] - c[k]["diag_off_phys"] for k in ks]),
                "interz_rec_lp": boot([t[k]["interz_rec_lp"] - c[k]["interz_rec_lp"] for k in ks]),
                "diag_off_phys_lp": boot([t[k]["diag_off_phys_lp"] - c[k]["diag_off_phys_lp"] for k in ks]),
                "speckle_excess": boot([t[k]["speckle_excess"] - c[k]["speckle_excess"] for k in ks]),
                "own_phys": sum(t[k]["own_phys"] - c[k]["own_phys"] for k in ks)}
    rep["per_quartet"] = {m: {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in v.items()} for m, v in data.items()}
    for key in SHEET_KEYS:
        if key not in keys:
            continue
        pad = lambda a: np.pad((np.clip(a, 0, 1) * 255).round().astype(np.uint8), 2, constant_values=128)
        first = data[next(iter(data))][key]
        rowsimg = [np.concatenate([pad(x) for x in first["_T"]] + [pad((first["_doe"] + np.pi) / (2 * np.pi))], 1)]
        for m, v in data.items():
            rowsimg.append(np.concatenate([pad(x) for x in v[key]["_R"]] + [pad((v[key]["_doe"] + np.pi) / (2 * np.pi))], 1))
        Image.fromarray(np.concatenate(rowsimg, 0)).save(out / f"sheet_{key.replace(':', '_')}.png")
    rep["sheet_layout"] = "row0 = TARGET z1..z4 + DOE of first model; next rows = " + ", ".join(models) + " (rec z1..z4 + own DOE.png), raw unretouched PNG values"
    (out / "compare.json").write_text(json.dumps(rep, indent=1))
    for m, s in rep["summary"].items():
        print(m, {G: {k: round(v, 4) if isinstance(v, float) else v for k, v in s[G].items() if k != "pearson_plane"} for G in ("A", "B")})
    if "paired_treatment_minus_control" in rep:
        print(json.dumps(rep["paired_treatment_minus_control"], indent=1))


if __name__ == "__main__":
    main()
