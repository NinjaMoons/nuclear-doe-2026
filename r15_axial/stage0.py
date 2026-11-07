"""R15 Stage 0: math identity + anti-degeneracy counterexamples + gradient checks for R3 + 0.05*L_ax. No training.
python -m r15_axial.stage0 <corpus_dir> <out.json>   (uses TRAIN A64:000 and B64:010 only)"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import doe_models.train_multiplane as tm
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from r15_axial.l_ax import LAMBDA, axial_innovation, r3, total

CORPUS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((CORPUS / "manifest_expanded.json").read_text(encoding="utf-8"))
S = {s["key"]: s for s in man["samples"]}


def quartet(key):
    assert S[key]["split"] == "train"
    return torch.from_numpy(np.stack([np.asarray(Image.open(CORPUS / p), dtype=np.float64) / 255 for p in S[key]["paths"]]))[None, :, None]


rng = np.random.default_rng(0)
res = {"lambda": LAMBDA, "cases": {}, "checks": {}}
fails = []


def ev(name, A, T):
    A = torch.as_tensor(A, dtype=torch.float64)
    A = torch.clamp(A, min=0)
    lax = float(axial_innovation(A, T))
    base, info = r3(A, T)
    tot = float(base) + LAMBDA * lax
    res["cases"][name] = {"L_ax": lax, "R3": float(base), "total": tot, "contract_mse": info["contract_mse"]}
    if not all(math.isfinite(v) for v in (lax, float(base), tot)):
        fails.append(f"non-finite {name}")
    return tot


for qkey, label in (("A64:000", "A"), ("B64:010", "B")):
    T = quartet(qkey)
    Tn = T / T.amax(dim=(-2, -1), keepdim=True)          # targets rescaled to max 1 so that a perfect display exists
    tbar = Tn.mean(dim=1, keepdim=True).expand_as(Tn)
    noise = torch.from_numpy(rng.normal(size=Tn.shape))
    speck = torch.from_numpy(np.abs(rng.normal(size=Tn.shape) + 1j * rng.normal(size=Tn.shape)))
    dim = 1e-3 * torch.ones_like(Tn)
    dim[..., 3, 3] = 1.0
    swapped = Tn.clone()
    swapped[:, 0], swapped[:, 1] = Tn[:, 1].clone(), Tn[:, 0].clone()
    p = f"{label}_{qkey}"
    perfect = ev(f"{p}/perfect (A=3*T)", 3 * Tn, Tn)
    others = {
        "4 copies of target mean": ev(f"{p}/4 copies of target mean", tbar, Tn),
        "black/dim + one spike": ev(f"{p}/black/dim + one spike", dim, Tn),
        "noisy recon T+0.3N": ev(f"{p}/noisy recon T+0.3N", Tn + 0.3 * noise, Tn),
        "random bright speckle": ev(f"{p}/random bright speckle", speck, Tn),
        "swapped planes z1<->z2": ev(f"{p}/swapped planes z1<->z2", swapped, Tn),
        "near-zero amplitude 1e-9*T+1e-10": ev(f"{p}/near-zero amplitude 1e-9*T+1e-10", 1e-9 * Tn + 1e-10, Tn),
    }
    for k, v in others.items():
        if not v > perfect:
            fails.append(f"{p}: '{k}' total {v} not > perfect {perfect}")
    # identity L_ax = (HW*c - ||e_bar||^2)/den  on the noisy case
    A = torch.clamp(Tn + 0.3 * noise, min=0)
    x = A / A.amax(dim=(-2, -1), keepdim=True)
    e = x - Tn
    hw = Tn.shape[-1] * Tn.shape[-2]
    c = float(e.pow(2).mean())
    ebar2 = float(e.mean(dim=1).pow(2).sum())
    den = float(Tn.pow(2).sum(dim=(-2, -1)).mean()) + 1e-12 * hw
    ident = (hw * c - ebar2) / den
    res["checks"][f"{p}/identity_rel_err"] = abs(ident - float(axial_innovation(A, Tn))) / abs(ident)
    if res["checks"][f"{p}/identity_rel_err"] > 1e-10:
        fails.append(f"identity {p}")

# identical A goals: same image on all planes; any common output -> L_ax == 0 exactly
base_img = quartet("A64:000")[:, :1].expand(1, 4, 1, 128, 128).clone()
base_img = base_img / base_img.amax()
same_out = torch.clamp(base_img + 0.05 * torch.from_numpy(rng.normal(size=(1, 1, 1, 128, 128))).expand_as(base_img), min=1e-6)
res["checks"]["identical_goals_common_output_L_ax"] = float(axial_innovation(same_out, base_img))
if abs(res["checks"]["identical_goals_common_output_L_ax"]) > 1e-20:
    fails.append("identical goals common output not zero")
# correlated A: slightly shifted goals; output = common base -> small positive L_ax
shift = torch.stack([torch.roll(base_img[:, 0], shifts=s, dims=-1) for s in (0, 1, 2, 3)], dim=1)
res["checks"]["correlated_goals_common_output_L_ax"] = float(axial_innovation(base_img.clone(), shift))
res["checks"]["correlated_goals_perfect_L_ax"] = float(axial_innovation(3 * shift, shift))
# lambda=0 parity and gradients through the exact Fresnel
cfg = CaseConfigV1()
g = torch.Generator().manual_seed(0)
T = quartet("B64:010")
phi = (torch.rand((1, 1, 128, 128), generator=g, dtype=torch.float64) * 2 * math.pi).requires_grad_(True)
amp = multiplane_amplitudes(phi, cfg, propagate_fresnel)
t0, _ = total(amp, T, lam=0.0)
r, _ = r3(amp, T)
res["checks"]["lambda0_parity_abs"] = abs(float(t0) - float(r))
if res["checks"]["lambda0_parity_abs"] != 0.0:
    fails.append("lambda=0 parity")
tl, _ = total(amp, T)
(gr,) = torch.autograd.grad(tl, phi)
res["checks"]["grad_finite"] = bool(torch.isfinite(gr).all())
res["checks"]["grad_rms"] = float(gr.pow(2).mean().sqrt())
d = torch.randn(phi.shape, generator=g, dtype=torch.float64)
d = d / d.norm()
h = 1e-5
fp = float(total(multiplane_amplitudes((phi + h * d).detach(), cfg, propagate_fresnel), T)[0])
fm = float(total(multiplane_amplitudes((phi - h * d).detach(), cfg, propagate_fresnel), T)[0])
fd, ad = (fp - fm) / (2 * h), float((gr * d).sum())
res["checks"]["fd_vs_autograd_rel"] = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
if not res["checks"]["grad_finite"] or res["checks"]["fd_vs_autograd_rel"] > 1e-3:
    fails.append(f"grad check {res['checks']['fd_vs_autograd_rel']}")
# exact-zero plane differentiability (one plane all zero except a spike)
z = torch.full((1, 4, 1, 128, 128), 0.0, dtype=torch.float64, requires_grad=True)
zz = z + torch.where(torch.arange(128 * 128).reshape(128, 128) == 5, 1.0, 0.0).double()
lz = axial_innovation(zz, quartet("B64:010"))
(gz,) = torch.autograd.grad(lz, z)
res["checks"]["zero_plane_grad_finite"] = bool(torch.isfinite(gz).all())
res["fails"] = fails
res["verdict"] = "R15_MATH_PASS" if not fails else "R15_MATH_NO_GO"
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
