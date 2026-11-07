"""Analysis of the pre-registered #41 pilot (PLAN_WC_PILOT.md). Read-only on eval exports.

  python research/wc_stem_pilot/analyze.py EVAL_DIR OUT_DIR
Writes OUT_DIR/analysis.json, OUT_DIR/ANALYSIS.md and raw contact sheets (no contrast stretch).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SETS = ("val16", "wc_holdout64", "orgsets10")
REF = "stem-candidate-6785eca-s0step900__stem_seed0_step_000900.pt"
FINAL, EARLY = 4000, 1000


def load(eval_dir):
    rows = json.loads((eval_dir / "eval.json").read_text(encoding="utf-8"))["rows"]
    out = {}
    for r in rows:
        m = re.match(r"(A_native|B_wc)_s(\d)__step_(\d+)\.pt", r["name"])
        key = ("ref40", 0, 900) if r["name"] == REF else (m.group(1), int(m.group(2)), int(m.group(3)))
        out[key] = r
    return out


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "p_le_0": float((m <= 0).mean()), "n_quartets": len(d)}


def arm_stats(R, arm, step, s):
    rows = [R[(arm, sd, step)][s] for sd in (0, 1)]
    out = {"pearson": float(np.mean([r["pearson_mean"] for r in rows])), "ssim": float(np.mean([r["ssim_mean"] for r in rows])),
           "worst_plane_pearson": float(np.mean([r["worst_plane_pearson"] for r in rows])),
           "pearson_plane": np.mean([r["pearson_plane"] for r in rows], 0).tolist(),
           "ssim_plane": np.mean([r["ssim_plane"] for r in rows], 0).tolist(),
           "per_seed_pearson": [r["pearson_mean"] for r in rows], "per_seed_ssim": [r["ssim_mean"] for r in rows]}
    if s == "val16":
        for g in ("A", "B"):
            out[g] = {"pearson": float(np.mean([r[g]["pearson"] for r in rows])), "ssim": float(np.mean([r[g]["ssim"] for r in rows]))}
    return out


def sheet(eval_dir, s, names, idx_sel, path, labels):
    T, PAD, LAB = 128, 4, 14
    lines = []
    for i in idx_sel:
        base = eval_dir / s / "export" / names[0] / "val" / f"{i:03d}"
        key = json.loads((base / "manifest.json").read_text(encoding="utf-8"))["metadata"]["sample_key"]
        lines.append(("TARGET " + key, [base / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]))
        for lab, nm in zip(labels, names):
            q = eval_dir / s / "export" / nm / "val" / f"{i:03d}"
            assert json.loads((q / "manifest.json").read_text(encoding="utf-8"))["metadata"]["sample_key"] == key
            pl = json.loads((q / "metrics.json").read_text(encoding="utf-8"))["planes"]
            lines.append((f"{lab}  P " + " ".join(f"{p['pearson']:.2f}" for p in pl) + "  S " +
                          " ".join(f"{p['ssim']:.2f}" for p in pl), [q / f"reconstructed_{k:02d}.png" for k in range(1, 5)]))
    W, H = 4 * (T + PAD) + PAD, len(lines) * (T + LAB + PAD) + PAD
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for r, (lab, ps) in enumerate(lines):
        y = PAD + r * (T + LAB + PAD)
        dr.text((PAD, y), lab, fill=255)
        for k, p in enumerate(ps):
            img.paste(Image.open(p).convert("L"), (PAD + k * (T + PAD), y + LAB))
    img.save(path)


def main(eval_dir, out):
    eval_dir, out = Path(eval_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R = load(eval_dir)
    steps = sorted({k[2] for k in R if k[0] != "ref40"})
    res = {"steps": {}, "ref40": {s: {"pearson": R[("ref40", 0, 900)][s]["pearson_mean"],
                                      "ssim": R[("ref40", 0, 900)][s]["ssim_mean"],
                                      "worst_plane_pearson": R[("ref40", 0, 900)][s]["worst_plane_pearson"],
                                      **({g: R[("ref40", 0, 900)][s][g] for g in ("A", "B")} if s == "val16" else {})}
                                  for s in SETS}}
    for st in steps:
        res["steps"][st] = {s: {"A_native": arm_stats(R, "A_native", st, s), "B_wc": arm_stats(R, "B_wc", st, s)} for s in SETS}
        for s in SETS:
            e = res["steps"][st][s]
            e["delta"] = {k: e["B_wc"][k] - e["A_native"][k] for k in ("pearson", "ssim", "worst_plane_pearson")}
            if s == "val16":
                e["delta"].update({g: e["B_wc"][g]["pearson"] - e["A_native"][g]["pearson"] for g in ("A", "B")})
            dP = np.mean([np.subtract(R[("B_wc", sd, st)][s]["per_quartet_pearson"], R[("A_native", sd, st)][s]["per_quartet_pearson"])
                          for sd in (0, 1)], 0)
            dS = np.mean([np.subtract(R[("B_wc", sd, st)][s]["per_quartet_ssim"], R[("A_native", sd, st)][s]["per_quartet_ssim"])
                          for sd in (0, 1)], 0)
            e["bootstrap_pearson"], e["bootstrap_ssim"] = boot(dP), boot(dS)
    f = res["steps"][FINAL]["val16"]["delta"]
    if f["pearson"] >= 0.01 and f["ssim"] >= -0.005 and min(f["A"], f["B"]) >= -0.02 and f["worst_plane_pearson"] >= -0.01:
        verdict = "WC_HELPS"
    elif f["pearson"] <= -0.01:
        verdict = "WC_HURTS"
    else:
        verdict = "WC_NEUTRAL"
    res["verdict"] = verdict
    # best single checkpoint vs ref40 (proposal rule)
    cand = []
    for k, r in R.items():
        if k[0] == "ref40":
            continue
        ref = R[("ref40", 0, 900)]
        ok = (r["val16"]["pearson_mean"] > ref["val16"]["pearson_mean"] and r["val16"]["ssim_mean"] > ref["val16"]["ssim_mean"]
              and r["orgsets10"]["pearson_mean"] >= ref["orgsets10"]["pearson_mean"])
        cand.append({"run": f"{k[0]}_s{k[1]}", "step": k[2], "val16_P": r["val16"]["pearson_mean"], "val16_S": r["val16"]["ssim_mean"],
                     "org_P": r["orgsets10"]["pearson_mean"], "org_S": r["orgsets10"]["ssim_mean"],
                     "wc_P": r["wc_holdout64"]["pearson_mean"], "beats_ref40_numeric": ok, "sha256": r["sha256"]})
    res["checkpoints"] = sorted(cand, key=lambda c: (c["run"], c["step"]))
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # sheets: final step seed 0, raw PNGs
    names = [REF, f"A_native_s0__step_{FINAL:06d}.pt", f"B_wc_s0__step_{FINAL:06d}.pt"]
    labels = ["#40 ref", "A native", "B nat+WC"]
    ids = R[("ref40", 0, 900)]["val16"]["ids"]
    sheet(eval_dir, "val16", names, [i for i, k in enumerate(ids) if k.startswith("A")], out / "sheet_val16_A.png", labels)
    sheet(eval_dir, "val16", names, [i for i, k in enumerate(ids) if k.startswith("B")], out / "sheet_val16_B.png", labels)
    sheet(eval_dir, "orgsets10", names, range(10), out / "sheet_orgsets10.png", labels)
    sheet(eval_dir, "wc_holdout64", names, range(6), out / "sheet_wc_holdout_first6.png", labels)
    # markdown
    L = [f"# #41 pilot analysis — verdict **{verdict}** (rule fixed in PLAN_WC_PILOT.md)", "",
         "Seed mean (0,1). Official path DOE.png u8 -> reload -> Fresnel -> PNG -> reload.", "",
         "| step | set | A native P/SSIM | B nat+WC P/SSIM | dP [95% CI] | dSSIM [95% CI] |", "|---|---|---|---|---|---|"]
    for st in steps:
        for s in SETS:
            e = res["steps"][st][s]
            bp, bs = e["bootstrap_pearson"], e["bootstrap_ssim"]
            L.append(f"| {st} | {s} | {e['A_native']['pearson']:.4f} / {e['A_native']['ssim']:.4f} | "
                     f"{e['B_wc']['pearson']:.4f} / {e['B_wc']['ssim']:.4f} | {bp['mean']:+.4f} [{bp['ci95'][0]:+.3f},{bp['ci95'][1]:+.3f}] | "
                     f"{bs['mean']:+.4f} [{bs['ci95'][0]:+.3f},{bs['ci95'][1]:+.3f}] |")
    L += ["", "#40 reference (569e1700, s0 step900): " + ", ".join(
        f"{s} P {v['pearson']:.4f} / SSIM {v['ssim']:.4f}" for s, v in res["ref40"].items())]
    (out / "ANALYSIS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
