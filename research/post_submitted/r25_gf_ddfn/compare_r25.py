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
        dark = [float(((R[k] < .05) & (T[k] > .25)).sum() / max((T[k] > .25).sum(), 1)) for k in range(4)]
        def spk(img, m):
            v = img[m]; return float(v.std() / v.mean()) if m.sum() > 10 and v.mean() > 0 else np.nan
        speck = [spk(R[k], T[k] > .5) - spk(T[k], T[k] > .5) for k in range(4)]
        rows[key] = {"g": key[0], "P": [p["pearson"] for p in met], "S": [p["ssim"] for p in met],
                     "own_native": sum(int(np.nanargmax(Cn[k]) == k) for k in range(4)),
                     "own_phys": sum(int(np.nanargmax(Cp[k]) == k) for k in range(4)),
                     "diag_off_phys": float(np.mean([Cp[k][k] for k in range(4)]) - np.nanmean([Cp[k][j] for k in range(4) for j in range(4) if j != k])),
                     "interz_rec": float(np.mean([pear(Rp[i], Rp[j]) for i, j in combinations(range(4), 2)])),
                     "interz_tgt": float(np.mean([pear(Tp[i], Tp[j]) for i, j in combinations(range(4), 2)])),
                     "bright": float(np.mean([r.mean() for r in R])), "dark": float(np.mean(dark)),
                     "speckle_excess": float(np.nanmean(speck)),
                     "doe_rough": float((wrapdiff(np.diff(doe, axis=0)).mean() + wrapdiff(np.diff(doe, axis=1)).mean()) / 2),
                     "_R": R, "_T": T, "_doe": doe}
    return rows


def summarize(rows):
    out = {}
    for G in ("A", "B", "all"):
        r = [v for v in rows.values() if G == "all" or v["g"] == G]
        P = np.array([v["P"] for v in r]); S = np.array([v["S"] for v in r])
        out[G] = {"pearson": float(P.mean()), "ssim": float(S.mean()), "pearson_plane": P.mean(0).tolist(),
                  "worst_plane_pearson": float(P.mean(0).min()), "own_phys": sum(v["own_phys"] for v in r),
                  "own_native": sum(v["own_native"] for v in r), "planes": 4 * len(r)}
        for k in ("diag_off_phys", "interz_rec", "interz_tgt", "bright", "dark", "speckle_excess", "doe_rough"):
            out[G][k] = float(np.mean([v[k] for v in r]))
    return out


def boot(d, n=10000, seed=2026):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(m, .025)), float(np.quantile(m, .975))]}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    models = dict(a.split("=", 1) for a in sys.argv[2:])
    data = {m: load(d) for m, d in models.items()}
    keys = sorted(next(iter(data.values())))
    assert all(sorted(v) == keys for v in data.values()), "different quartets"
    rep = {"models": models, "summary": {m: summarize(v) for m, v in data.items()}}
    if "control" in data and "treatment" in data:
        c, t = data["control"], data["treatment"]; rep["paired_treatment_minus_control"] = {}
        for G in ("A", "B", "all"):
            ks = [k for k in keys if G == "all" or k[0] == G]
            rep["paired_treatment_minus_control"][G] = {
                "pearson": boot([np.mean(t[k]["P"]) - np.mean(c[k]["P"]) for k in ks]),
                "ssim": boot([np.mean(t[k]["S"]) - np.mean(c[k]["S"]) for k in ks]),
                "worst_plane_quartet": boot([min(t[k]["P"]) - min(c[k]["P"]) for k in ks]),
                "interz_rec": boot([t[k]["interz_rec"] - c[k]["interz_rec"] for k in ks]),
                "diag_off_phys": boot([t[k]["diag_off_phys"] - c[k]["diag_off_phys"] for k in ks]),
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
