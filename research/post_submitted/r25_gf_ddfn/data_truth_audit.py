"""R25 PHASE 2 read-only audits (CPU; TRAIN + VAL only; TEST never decoded; no training; frozen V2 19b05dad).
P1 data truth: split/kind/relation counts from manifest_expanded (B64 pilot = per-plane independent draws, see
   doe_experiments.corpus._make_b); target-to-target 4x4 Pearson per quartet, (a) native pixel grid (index-aligned,
   NOT physically registered) and (b) physically registered onto the z1 output grid (bilinear, pitch_k -> pitch_1);
   VAL16 per quartet + TRAIN group summaries; VAL16 target contact sheet.
P2 V2 diagnostic: 8 TRAIN A, 8 TRAIN B independent (B208 relation=independent), 8 TRAIN B similar (B208 relation=similar),
   chosen by sorted key + torch.randperm(seed 2026) -- no cherry-picking -- and VAL16, through the OFFICIAL
   tm.evaluate_checkpoint export (DOE.png L8 save/readback, four amplitude PNG L8 save/readback; TRAIN subsets are
   passed in the `val` slot only because the evaluator accepts val/test names; metadata records the true split).
   Per-z Pearson / SSIM, 4x4 C[k,j] = corr(rec_k, T_j) on the native grid of plane k (own-target argmax), the same
   matrix with physical registration of rec_k and T_j onto the z1 grid, inter-z reconstruction correlation (physical).
usage: python -m r25_gf_ddfn.run_r25_notest is not used; python -m r25_gf_ddfn.data_truth_audit OUT_DIR V2_CKPT"""
import json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, r"C:\nuclear-doe-operator\b-sampler\research\wc_stem_pilot")
import pilot  # noqa: E402
from doe_core.contracts_v1 import CaseConfigV1  # noqa: E402
import doe_models.train_multiplane as tm  # noqa: E402
from r25_gf_ddfn.train_r25 import research_load  # noqa: E402

CORPUS = Path(r"C:\nuclear-doe-operator\desktop-41\corpus")
CFG = CaseConfigV1()
N = 128
PITCH = [CFG.plane(k).output_pitch_m(N, N)[1] for k in range(4)]


def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def to_z1(img, k):  # identical to back_projection_stem/separation.py to_z1_grid
    n = np.arange(N) - N // 2
    g = (n * PITCH[0] / PITCH[k] + N // 2) / (N - 1) * 2 - 1
    gy, gx = np.meshgrid(g, g, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1)[None]).float()
    return F.grid_sample(torch.from_numpy(img)[None, None].float(), grid, mode="bilinear", align_corners=True)[0, 0].numpy()


def tt_stats(T):  # T [4,H,W] float64
    off = list(combinations(range(4), 2))
    nat = [pear(T[i], T[j]) for i, j in off]
    Tp = [to_z1(T[k], k) for k in range(4)]
    phy = [pear(Tp[i], Tp[j]) for i, j in off]
    return {"native_offdiag": nat, "native_mean": float(np.mean(nat)), "phys_offdiag": phy, "phys_mean": float(np.mean(phy))}


def group_of(s):
    if s["kind"] == "A":
        return "A"
    rel = (s.get("procedural") or {}).get("relation")
    return "B/" + (rel or "pilot_independent")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False); v2path = sys.argv[2]
    man = json.loads((CORPUS / "manifest_expanded.json").read_bytes())
    samples = {s["key"]: s for s in man["samples"] if s["split"] != "test"}
    data, msha = pilot.native(CORPUS, splits=("train", "val"))
    rep = {"manifest_sha256": msha, "test_decoded": False, "pitch_m": PITCH, "counts": {}}
    for sp in ("train", "val"):
        c = {}
        for k in data[sp]["ids"]:
            g = group_of(samples[k]); c[g] = c.get(g, 0) + 1
        rep["counts"][sp] = c
    # ---- P1 target-to-target similarity
    tt = {}
    for sp in ("train", "val"):
        for i, k in enumerate(data[sp]["ids"]):
            tt[(sp, k)] = tt_stats(data[sp]["targets"][i, :, 0].double().numpy())
    rep["val_quartets"] = [{"key": k, "group": group_of(samples[k]), **tt[("val", k)]} for k in data["val"]["ids"]]
    summ = {}
    for (sp, k), v in tt.items():
        g = f"{sp}:{group_of(samples[k])}"
        summ.setdefault(g, {"native": [], "phys": []})
        summ[g]["native"].append(v["native_mean"]); summ[g]["phys"].append(v["phys_mean"])
    rep["target_similarity_by_group"] = {g: {"n": len(v["native"]), "native_mean": float(np.mean(v["native"])),
                                             "native_median": float(np.median(v["native"])), "phys_mean": float(np.mean(v["phys"])),
                                             "phys_q10_q90": [float(np.quantile(v["phys"], .1)), float(np.quantile(v["phys"], .9))]}
                                         for g, v in sorted(summ.items())}
    sheet = (data["val"]["targets"][:, :, 0].numpy() * 255).round().astype(np.uint8)  # [16,4,H,W]
    rows = [np.concatenate([np.pad(sheet[i, k], 2, constant_values=128) for k in range(4)], 1) for i in range(16)]
    Image.fromarray(np.concatenate(rows, 0)).save(out / "val16_targets_contact_sheet.png")
    # ---- P2 frozen V2 diagnostic
    tr = data["train"]; g = torch.Generator().manual_seed(2026)
    sel = {}
    for name, grp in (("train_A8", "A"), ("train_Bindep8", "B/independent"), ("train_Bsimilar8", "B/similar")):
        pool = sorted(i for i, k in enumerate(tr["ids"]) if group_of(samples[tr["ids"][i]]) == grp)
        pool = sorted(pool, key=lambda i: tr["ids"][i])
        sel[name] = sorted(pool[j] for j in torch.randperm(len(pool), generator=g)[:8].tolist())
    sets = {name: ("train", idx) for name, idx in sel.items()}
    sets["val16"] = ("val", list(range(16)))
    comps = tm.production_components(); tm.load_checkpoint = research_load
    rep["v2_diag"] = {}
    for name, (sp, idx) in sets.items():
        part = data[sp]
        sub = {"targets": part["targets"][idx], "ids": [part["ids"][i] for i in idx], "origins": [part["origins"][i] for i in idx],
               "paths": [part["paths"][i] for i in idx], "kinds": [part["kinds"][i] for i in idx]}
        d = {"splits": {"val": sub}, "manifest_sha256": msha, "synthetic": True}
        s = tm.evaluate_checkpoint(Path(v2path), d, comps, out / name, torch.device("cpu"), CFG, ("val",))
        exp = out / name / "export" / Path(v2path).name / "val"
        rows = []
        for q, key in zip(sorted(exp.iterdir()), sub["ids"]):
            R = [np.asarray(Image.open(q / f"reconstructed_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
            T = [np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
            met = json.loads((q / "metrics.json").read_text(encoding="utf-8"))["planes"]
            Cn = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
            Rp, Tp = [to_z1(R[k], k) for k in range(4)], [to_z1(T[k], k) for k in range(4)]
            Cp = [[pear(Rp[k], Tp[j]) for j in range(4)] for k in range(4)]
            rows.append({"key": key, "pearson": [p["pearson"] for p in met], "ssim": [p["ssim"] for p in met],
                         "C_native": Cn, "C_phys": Cp, "own_argmax_native": sum(int(np.nanargmax(Cn[k]) == k) for k in range(4)),
                         "own_argmax_phys": sum(int(np.nanargmax(Cp[k]) == k) for k in range(4)),
                         "diag_minus_offdiag_phys": float(np.mean([Cp[k][k] for k in range(4)]) - np.nanmean([Cp[k][j] for k in range(4) for j in range(4) if j != k])),
                         "interz_rec_phys": float(np.mean([pear(Rp[i], Rp[j]) for i, j in combinations(range(4), 2)])),
                         "interz_tgt_phys": tt[(sp, key)]["phys_mean"],
                         "rec_mean_brightness": float(np.mean([r.mean() for r in R])), "tgt_mean_brightness": float(np.mean([t.mean() for t in T]))})
        agg = lambda f: float(np.mean([f(r) for r in rows]))
        rep["v2_diag"][name] = {"split": sp, "keys": sub["ids"], "pearson_mean": s["val"]["pearson_mean"], "ssim_mean": s["val"]["ssim_mean"],
                                "worst_plane_pearson": s["val"]["worst_plane_pearson"],
                                "own_argmax_native": sum(r["own_argmax_native"] for r in rows), "own_argmax_phys": sum(r["own_argmax_phys"] for r in rows),
                                "planes": 4 * len(rows), "diag_minus_offdiag_phys": agg(lambda r: r["diag_minus_offdiag_phys"]),
                                "interz_rec_phys": agg(lambda r: r["interz_rec_phys"]), "interz_tgt_phys": agg(lambda r: r["interz_tgt_phys"]),
                                "rec_brightness": agg(lambda r: r["rec_mean_brightness"]), "tgt_brightness": agg(lambda r: r["tgt_mean_brightness"]),
                                "per_quartet_pearson": [float(np.mean(r["pearson"])) for r in rows], "rows": rows}
        v = rep["v2_diag"][name]
        print(name, {k: v[k] for k in ("pearson_mean", "ssim_mean", "worst_plane_pearson", "own_argmax_phys", "planes",
                                       "diag_minus_offdiag_phys", "interz_rec_phys", "interz_tgt_phys")}, flush=True)
    (out / "data_truth_audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"counts": rep["counts"], "similarity": rep["target_similarity_by_group"]}, indent=1))


if __name__ == "__main__":
    main()
