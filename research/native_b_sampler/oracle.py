"""DIAGNOSTIC ONLY (#43 follow-up): per-input Adam upper bound on VAL16 for the same physics/codec/metrics.
Never a contest method (per-input Adam at inference is forbidden); used only to measure the headroom of a single
phase-only DOE vs the trained STEM, especially group-B ghosting. TEST is never decoded.

  python research/native_b_sampler/oracle.py --corpus C --init CKPT --out NEW --steps 2000
O1 = stock synthesize_multiplane (contract loss, random init, lr .05); O2 = Adam on the R3 objective started from the
STEM phase (lr .02). Both exported through the official export_run (DOE.png u8 -> reload -> Fresnel -> PNG -> reload).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "wc_stem_pilot"))
import pilot  # noqa: E402
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402
from doe_models.adam_multiplane import synthesize_multiplane  # noqa: E402


def r3_refine(phase0, targets, cfg, prop, steps, lr):
    phase = phase0.clone().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    best, best_phase = None, phase0.clone()
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(phase, cfg, prop), targets, tw.R3_NAME)
        v = float(loss)
        if best is None or v < best:
            best, best_phase = v, phase.detach().clone()
        if step == steps:
            break
        loss.backward()
        opt.step()
    return torch.remainder(best_phase + np.pi, 2 * np.pi) - np.pi, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    data, msha = native(Path(a.corpus), splits=("val",))
    val = data["val"]
    T = val["targets"].to(dev)
    cfg = CaseConfigV1()
    comps = tm.production_components()
    model, payload = tw.research_load_checkpoint(a.init)
    model = model.to(dev)
    stem_phase = tm.infer_phase(model, T)
    phases = {"STEM": stem_phase}
    torch.manual_seed(0)
    r = synthesize_multiplane(T, cfg, propagate_fn=comps.propagate, steps=a.steps, lr=0.05, seed=0)
    phases["O1_adam_contract_rand"] = r.phase_rad
    phases["O2_adam_r3_from_stem"], _ = r3_refine(stem_phase, T, cfg, comps.propagate, a.steps, 0.02)
    summary = {}
    for name, ph in phases.items():
        P, S = [], []
        for i, (key, files) in enumerate(zip(val["ids"], val["paths"])):
            run = out / name / f"{i:03d}"
            run.parent.mkdir(parents=True, exist_ok=True)
            comps.export_run(files, ph[i:i + 1].detach().cpu(), run, synthetic_test=True,
                             metadata={"sample_key": key, "split": "val", "diagnostic": name, "manifest_sha256": msha})
            pl = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["planes"]
            P.append([p["pearson"] for p in pl])
            S.append([p["ssim"] for p in pl])
        P, S = np.array(P), np.array(S)
        g = {k: [i for i, kk in enumerate(val["kinds"]) if kk == k] for k in "AB"}
        summary[name] = {"pearson": float(P.mean()), "ssim": float(S.mean()),
                         "A": {"pearson": float(P[g["A"]].mean()), "ssim": float(S[g["A"]].mean())},
                         "B": {"pearson": float(P[g["B"]].mean()), "ssim": float(S[g["B"]].mean())},
                         "pearson_plane": P.mean(0).tolist(), "per_quartet_pearson": P.mean(1).tolist()}
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[name].items()
                                if k in ("pearson", "ssim", "A", "B")}))
    (out / "oracle_summary.json").write_text(json.dumps({"steps": a.steps, "init": payload["sha256"], "rows": summary,
                                                         "ids": val["ids"], "note": "diagnostic upper bound only"},
                                                        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
