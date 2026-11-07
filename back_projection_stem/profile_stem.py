"""Measured round-trip errors and per-step cost, 4-ch production U-Net vs 12-ch stem (CPU fp32). No training run.
python -m back_projection_stem.profile_stem <out.json>"""
import json
import platform
import sys
import time

import torch

from doe_core.contracts_v1 import CaseConfigV1, FresnelGeometry
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from doe_models.unet_multiplane import build_model

from back_projection_stem.r3_loss import r3_loss
from back_projection_stem.stem import backprojection_features, build_stem_model, inverse_fresnel

torch.set_num_threads(8)
CFG = CaseConfigV1()
out = {"torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
       "platform": platform.platform(), "processor": platform.processor(), "vram": "PENDING (no CUDA device on this machine)"}


def rel(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


errs = {}
geoms = {f"official_z{k+1}": CFG.plane(k) for k in range(4)}
geoms["aniso_10x13um_z3mm"] = FresnelGeometry(wavelength_m=5.32e-7, input_pitch_m=(10e-6, 13e-6), distance_m=0.003)
for shape in ((2, 1, 4, 6), (3, 1, 5, 7), (2, 1, 128, 128)):
    for gname, g in geoms.items():
        for dt in (torch.complex64, torch.complex128):
            gen = torch.Generator().manual_seed(0)
            real = torch.float32 if dt == torch.complex64 else torch.float64
            u = torch.complex(torch.randn(shape, generator=gen, dtype=real), torch.randn(shape, generator=gen, dtype=real))
            a = rel(inverse_fresnel(propagate_fresnel(u, g).field, g), u)
            b = rel(propagate_fresnel(inverse_fresnel(u, g), g).field, u)
            errs[f"{shape[-2]}x{shape[-1]}/{gname}/{str(dt).split('.')[-1]}"] = {"inv_fwd": a, "fwd_inv": b}
out["roundtrip_rel_l2"] = errs
out["roundtrip_max"] = {d: max(max(v.values()) for k, v in errs.items() if k.endswith(d)) for d in ("complex64", "complex128")}

gen = torch.Generator().manual_seed(0)
targets = torch.floor(torch.rand((4, 4, 1, 128, 128), generator=gen) * 255 + 0.5) / 255


def time_steps(model, n_warm=3, n=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ts = []
    for i in range(n_warm + n):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = r3_loss(multiplane_amplitudes(model(targets), CFG, propagate_fresnel), targets)
        loss.backward()
        opt.step()
        if i >= n_warm:
            ts.append(time.perf_counter() - t0)
    return {"mean_s": sum(ts) / len(ts), "min_s": min(ts), "max_s": max(ts), "n": n,
            "params": sum(p.numel() for p in model.parameters())}


t0 = time.perf_counter()
for _ in range(10):
    backprojection_features(targets)
feat_s = (time.perf_counter() - t0) / 10
m4, m12 = build_model(seed=0), build_stem_model(seed=0)
rounds = {"unet4": [], "stem12": []}
for r in range(4):  # interleaved A/B/B/A rounds to cancel drift/thermal order effects
    order = (("unet4", m4), ("stem12", m12)) if r % 2 == 0 else (("stem12", m12), ("unet4", m4))
    for name, m in order:
        rounds[name].append(time_steps(m, n_warm=2, n=8))
def merge(rs):
    means = [x["mean_s"] for x in rs]
    return {"mean_s": sum(means) / len(means), "round_means_s": means, "min_s": min(x["min_s"] for x in rs),
            "n": sum(x["n"] for x in rs), "params": rs[0]["params"]}
step4, step12 = merge(rounds["unet4"]), merge(rounds["stem12"])
out["step_cost"] = {"batch": 4, "shape": [128, 128], "precision": "float32", "loss": "R3 beta0.3 (r3_loss)",
                    "scope": "zero_grad + model forward + 4-plane Fresnel + loss + backward + Adam step",
                    "unet4": step4, "stem12": step12, "stem12_over_unet4": step12["mean_s"] / step4["mean_s"],
                    "feature_build_s_per_batch": feat_s}
print(json.dumps({k: v for k, v in out.items() if k != "roundtrip_rel_l2"}, indent=1))
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(out, indent=1))
