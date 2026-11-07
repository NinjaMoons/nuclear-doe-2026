"""R13 Stage 0 (#46): no-training field diagnostic of a target-plane phase carrier psi = pi*(p^2+q^2)/128.
python -m r13_phase.r13_stage0 <corpus_dir_with_manifest_expanded> <stem_checkpoint> <out_dir>
TRAIN quartets only (fixed list, PLAN_STAGE0.md). Phase-only comparators are classical DIAGNOSTICS, not contest methods."""
import hashlib
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from back_projection_stem.infer_neural_stem import research_load_checkpoint
from back_projection_stem.stem import inverse_fresnel
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.export import export_run, production_components
from doe_models.train_multiplane import infer_phase

CORPUS, CKPT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
CFG, N = CaseConfigV1(), 128
KEYS = ["B208:006", "A64:000", "A64:001", "A64:002", "A64:003", "A64:004", "A64:005", "A64:007", "A64:009",
        "B64:000", "B64:001", "B64:004", "B64:006", "B64:007", "B64:008", "B64:009", "B64:010"]
n = np.arange(N) - N // 2
P_, Q_ = np.meshgrid(n, n, indexing="ij")
PSI = np.pi * (P_ ** 2 + Q_ ** 2) / N
R = np.hypot(P_, Q_)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def pear(a, b):  # identical to back_projection_stem/separation.py
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def to_z1_grid(img, pitch, pitch1):  # identical to back_projection_stem/separation.py
    import torch.nn.functional as F
    nn_ = np.arange(N) - N // 2
    g = (nn_ * pitch1 / pitch + N // 2) / (N - 1) * 2 - 1
    gy, gx = np.meshgrid(g, g, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1)[None]).float()
    return F.grid_sample(torch.from_numpy(img)[None, None].float(), grid, mode="bilinear", align_corners=True)[0, 0].numpy()
man = json.loads((CORPUS / "manifest_expanded.json").read_text(encoding="utf-8"))
samples = {s["key"]: s for s in man["samples"]}
comps = production_components()
model, payload = research_load_checkpoint(CKPT)
assert payload["sha256"] == sha(CKPT)


def fields(T, carrier):
    out, err = [], []
    for k in range(4):
        v = T[k] * (np.exp(1j * PSI) if carrier else 1.0)
        V = torch.from_numpy(v.astype(np.complex128))[None, None]
        b = inverse_fresnel(V, CFG.plane(k))
        err.append(float((propagate_fresnel(b, CFG.plane(k)).field - V).abs().pow(2).sum().sqrt() / V.abs().pow(2).sum().sqrt()))
        out.append(b[0, 0].numpy())
    return out, max(err)


def aperture(b):
    e = np.abs(b) ** 2
    tot = e.sum()
    if tot <= 0:
        return {"inner_r<32": None, "annulus_32-64": None, "corners_r>=64": None}
    return {"inner_r<32": float(e[R < 32].sum() / tot), "annulus_32-64": float(e[(R >= 32) & (R < 64)].sum() / tot),
            "corners_r>=64": float(e[R >= 64].sum() / tot)}


def radial(b):
    e = np.abs(b) ** 2
    e = e / e.sum()
    return [float(e[(R >= r) & (R < r + 8)].sum()) for r in range(0, 96, 8)]


def phase_of_sum(bs):
    s = sum(bs)
    mag = np.abs(s)
    phi = np.arctan2(s.imag, s.real)
    phi[mag < 1e-12 * mag.max()] = 0.0
    return phi


def evaluate(phase, paths, tag):
    d = Path(tempfile.mkdtemp(dir=OUT)) / tag
    export_run([Path(p) for p in paths], torch.from_numpy(np.asarray(phase, dtype=np.float32))[None, None], d,
               components=comps, config=CFG, metadata={"method": "r13_stage0_diagnostic", "arm": tag}, synthetic_test=True)
    met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    pitches = [p[1] for p in json.loads((d / "manifest.json").read_text(encoding="utf-8"))["output_pitches_m"]]
    Rc = [np.asarray(Image.open(d / f"reconstructed_0{k}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
    Tg = [np.asarray(Image.open(d / "targets" / f"target_0{k}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
    C = [[pear(Rc[k], Tg[j]) for j in range(4)] for k in range(4)]
    Rp = [to_z1_grid(Rc[k], pitches[k], pitches[0]) for k in range(4)]
    bright = [float(Tg[k][Tg[k] > 0.25].size and Rc[k][Tg[k] > 0.25].mean()) for k in range(4)]
    spk = [float(Rc[k].std() / max(Rc[k].mean(), 1e-12)) for k in range(4)]
    return {"P": [p["pearson"] for p in met["planes"]], "S": [p["ssim"] for p in met["planes"]], "C": C,
            "own_argmax": sum(int(np.nanargmax(C[k]) == k) for k in range(4)),
            "interz_phys": float(np.nanmean([pear(Rp[i], Rp[j]) for i, j in combinations(range(4), 2)])),
            "display_mean": [float(r.mean()) for r in Rc], "bright_region_mean": bright, "speckle_contrast": spk,
            "dir": str(d), "doe_sha256": sha(d / "DOE.png")}


rows = []
for key in KEYS:
    s = samples[key]
    assert s["split"] == "train", key
    paths = [CORPUS / p for p in s["paths"]]
    assert [sha(p) for p in paths] == s["sha256"], key
    T = np.stack([np.asarray(Image.open(p), dtype=np.float64) / 255 for p in paths])
    B0, e0 = fields(T, False)
    Bc, ec = fields(T, True)
    t = torch.from_numpy(T.astype(np.float32))[None, :, None]
    stem_phase = infer_phase(model, t)[0, 0].numpy()
    row = {"key": key, "g": key[0], "roundtrip_rel_err": max(e0, ec),
           "aperture_zero": [aperture(b) for b in B0], "aperture_carrier": [aperture(b) for b in Bc],
           "radial_zero": [radial(b) for b in B0], "radial_carrier": [radial(b) for b in Bc],
           "aperture_sum_zero": aperture(sum(B0)), "aperture_sum_carrier": aperture(sum(Bc)),
           "phi_zero": evaluate(phase_of_sum(B0), paths, f"{key.replace(':', '-')}_phi0"),
           "phi_carrier": evaluate(phase_of_sum(Bc), paths, f"{key.replace(':', '-')}_phic"),
           "stem_s0_900": evaluate(stem_phase, paths, f"{key.replace(':', '-')}_stem")}
    rows.append(row)
    print(key, "rt", f"{row['roundtrip_rel_err']:.1e}", "| annulus+corners zero/carrier (mean z):",
          round(np.mean([a["annulus_32-64"] + a["corners_r>=64"] for a in row["aperture_zero"]]), 3),
          round(np.mean([a["annulus_32-64"] + a["corners_r>=64"] for a in row["aperture_carrier"]]), 3),
          "| P phi0/phic/stem:", *(round(float(np.mean(row[a]["P"])), 3) for a in ("phi_zero", "phi_carrier", "stem_s0_900")),
          "| own:", *(row[a]["own_argmax"] for a in ("phi_zero", "phi_carrier", "stem_s0_900")), flush=True)

# synthetic counterexamples (aperture only)
syn = {}
disk = (R < 20).astype(float)
sq = ((np.abs(P_ - 30) < 10) & (np.abs(Q_ + 25) < 10)).astype(float)
rng = np.random.default_rng(0)
for name, img in {"centred_disk": disk, "offcentre_square": sq, "uniform": np.ones((N, N)),
                  "random_binary": (rng.random((N, N)) > 0.5).astype(float)}.items():
    T = np.stack([img] * 4)
    syn[name] = {"zero": [aperture(b) for b in fields(T, False)[0]], "carrier": [aperture(b) for b in fields(T, True)[0]]}


def summ(arm, g=None):
    rs = [r for r in rows if g in (None, r["g"])]
    return {"P": float(np.mean([np.mean(r[arm]["P"]) for r in rs])), "S": float(np.mean([np.mean(r[arm]["S"]) for r in rs])),
            "own_argmax": int(sum(r[arm]["own_argmax"] for r in rs)), "planes": 4 * len(rs),
            "interz_phys": float(np.mean([r[arm]["interz_phys"] for r in rs])),
            "speckle_contrast": float(np.mean([np.mean(r[arm]["speckle_contrast"]) for r in rs])),
            "display_mean": float(np.mean([np.mean(r[arm]["display_mean"]) for r in rs]))}


def ap_summ(key, g=None):
    rs = [r for r in rows if g in (None, r["g"])]
    a = [x for r in rs for x in r[key]]
    return {k: float(np.mean([x[k] for x in a])) for k in a[0]}


summary = {"arms": {arm: {"ALL": summ(arm), "A": summ(arm, "A"), "B": summ(arm, "B")} for arm in ("phi_zero", "phi_carrier", "stem_s0_900")},
           "aperture_per_plane_field": {"zero": {"ALL": ap_summ("aperture_zero"), "A": ap_summ("aperture_zero", "A"), "B": ap_summ("aperture_zero", "B")},
                                        "carrier": {"ALL": ap_summ("aperture_carrier"), "A": ap_summ("aperture_carrier", "A"), "B": ap_summ("aperture_carrier", "B")}},
           "aperture_coherent_sum": {"zero": {k: float(np.mean([r["aperture_sum_zero"][k] for r in rows])) for k in rows[0]["aperture_sum_zero"]},
                                     "carrier": {k: float(np.mean([r["aperture_sum_carrier"][k] for r in rows])) for k in rows[0]["aperture_sum_carrier"]}},
           "max_roundtrip_rel_err": max(r["roundtrip_rel_err"] for r in rows), "synthetic": syn,
           "stem_checkpoint_sha256": payload["sha256"], "n_quartets": len(rows)}
z, c = summary["arms"]["phi_zero"], summary["arms"]["phi_carrier"]
dP, dS = c["ALL"]["P"] - z["ALL"]["P"], c["ALL"]["S"] - z["ALL"]["S"]
summary["decision"] = {"dP_carrier_minus_zero": dP, "dSSIM": dS, "B_own_zero": z["B"]["own_argmax"], "B_own_carrier": c["B"]["own_argmax"],
                       "label": "R13_STAGE0_DONE" if (dP >= 0.02 and c["B"]["own_argmax"] >= z["B"]["own_argmax"] and dS >= -0.01) else "R13_STAGE0_NO_GO"}
(OUT / "r13_stage0.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
