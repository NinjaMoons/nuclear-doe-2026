"""R25 zero-training diagnostic (TRAIN only, no NN, no VAL/TEST): does the GF-DDFN fixed target-phase anchor improve the
analytic multi-plane seed?  DOE = arg(sum_k w_k F_k^-1(T_k e^{i psi_k})), w_k = 1/rms, -> official L8 codec -> official
forward -> per-plane Pearson of |U_k| vs T_k, and own-target argmax (B ghosting).  psi variants: zero (V2 stem
convention), quad (Gemini 16pi r^2 carrier), randhf (Gemini uniform +-0.75pi per pixel, seed 2026), gemini (quad+randhf).
usage: python -m r25_gf_ddfn.run_r25_notest is NOT used; run: python -m r25_gf_ddfn.anchor_seed_diag OUT.json"""
import json, math, sys

import numpy as np
import torch

sys.path.insert(0, r"C:\nuclear-doe-operator\b-sampler\research\wc_stem_pilot")
import pilot  # noqa: E402
from pathlib import Path  # noqa: E402
from back_projection_stem.stem import inverse_fresnel  # noqa: E402
from doe_core.contracts_v1 import CaseConfigV1  # noqa: E402
from doe_core.fresnel import propagate_fresnel  # noqa: E402
from doe_experiments.phase_codec import decode_phase, encode_phase  # noqa: E402
from r25_gf_ddfn.gf_ddfn_fixed import raw  # noqa: E402
from r25_gf_ddfn.train_r25 import subset64  # noqa: E402

CFG = CaseConfigV1()


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else float("nan")


def main():
    data, msha = pilot.native(Path(r"C:\nuclear-doe-operator\desktop-41\corpus"), splits=("train",))
    tr = data["train"]; idx, grp = subset64(tr, 2026)
    T = tr["targets"][idx].double()  # [64,4,1,128,128]
    gem = raw.GF_DDFN()
    anchor = gem.psi_anchor.double()[0]  # [4,128,128]
    N = 128; c = (torch.arange(N, dtype=torch.float64) - N // 2) * 10e-6
    R2 = c[:, None] ** 2 + c[None, :] ** 2
    quad = (math.pi * R2 / R2.max() * 16.0).expand(4, N, N)
    variants = {"zero": torch.zeros(4, N, N, dtype=torch.float64), "quad": quad, "randhf": anchor - quad, "gemini": anchor}
    res = {"manifest_sha256": msha, "n": len(idx), "subset_seed": 2026}
    for name, psi in variants.items():
        rows = []
        for j in range(T.shape[0]):
            t = T[j:j + 1, :, 0]
            s = 0
            for k in range(4):
                b = inverse_fresnel(torch.polar(t[:, k:k + 1], psi[k][None, None]), CFG.plane(k))
                s = s + b / b.abs().pow(2).mean().sqrt()
            ph = torch.from_numpy(decode_phase(encode_phase(torch.angle(s))))
            u = torch.polar(torch.ones_like(ph), ph)
            amps = [propagate_fresnel(u, CFG.plane(k)).field.abs()[0, 0] for k in range(4)]
            C = np.array([[pear(amps[k], t[0, m]) for m in range(4)] for k in range(4)])
            rows.append({"g": grp[idx[j]], "P": float(np.mean(np.diag(C))), "worst": float(np.min(np.diag(C))),
                         "own": int(sum(int(np.argmax(C[k]) == k) for k in range(4)))})
        out = {}
        for G in ("A", "B", "all"):
            r = [x for x in rows if G == "all" or x["g"] == G]
            out[G] = {"P": float(np.mean([x["P"] for x in r])), "worst_plane_mean": float(np.mean([x["worst"] for x in r])),
                      "own_argmax": int(sum(x["own"] for x in r)), "planes": 4 * len(r)}
        res[name] = out; print(name, json.dumps(out), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
