"""Read-only ceiling test (#49): expressivity of a complex back-projection mixing head.
phi = arg( c0(x)*exp(i*phi_v2) + sum_k c_k(x) * B_k_hat(x) ),  B_k_hat = inverse_fresnel(T_k+0j, z_k)/rms (STEM features),
c0, c_k complex maps on a 32x32 grid, bilinear-upsampled to 128 (smooth, like a U-Net output). Per-quartet Adam on R3
(TRAIN only) -> official export. Compares with warm-teacher (free phase) and v2. Variants: with/without phi_v2 term.
usage: PYTHONPATH=<runtime src> python bpmix_ceiling.py <out_dir> <label=v2_export_dir> ..."""
import json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.export import export_run
from doe_experiments.phase_codec import load_doe_png
from doe_experiments.data import load_png_quartet
from doe_models.adam_multiplane import multiplane_amplitudes
from back_projection_stem.r3_loss import r3_loss
from back_projection_stem.stem import inverse_fresnel
CFG = CaseConfigV1(); G = 32; STEPS = 1000
import os; VARIANTS = tuple(os.environ.get("BPMIX_VARIANTS", "v2+bp,bp_only").split(","))
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
up = lambda c: F.interpolate(c, size=(128, 128), mode="bilinear", align_corners=False)
out = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1); d = Path(d)
    targets = [d / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]
    T = load_png_quartet(targets)
    phi0 = load_doe_png(d / "DOE.png", dtype=torch.float64)[0, 0]
    B = []
    for k in range(4):
        b = inverse_fresnel(torch.complex(T[:, k].double(), torch.zeros_like(T[:, k].double())), CFG.plane(k))[0, 0]
        B.append(b / b.abs().pow(2).mean().sqrt())
    B = torch.stack(B)                                                          # [4,128,128] complex128
    for variant in VARIANTS:
        torch.manual_seed(0)
        cr = torch.zeros((1, 5, G, G), dtype=torch.float64); ci = torch.zeros_like(cr)
        if variant != "bp_only": cr[:, 0] = 1.0
        else: cr[:, 1:] = 1.0
        cr.requires_grad_(True); ci.requires_grad_(True)
        opt = torch.optim.Adam([cr, ci], lr=0.02); best = (math.inf, None)
        def phase():
            c = torch.complex(up(cr), up(ci))[0]                                # [5,128,128]
            if variant == "ctrl_smooth":      # control: same complex 32x32 maps, NO back-projection fields (c0 only)
                s = c[0] * torch.exp(1j * phi0) + (c[1:] * 0).sum(0)
            elif variant == "ctrl_smooth5":     # control: 5 complex smooth maps times FIXED smooth unit fields
                s = c[0] * torch.exp(1j * phi0) + c[1:].sum(0)
            else:
                s = (c[1:] * B).sum(0) + (c[0] * torch.exp(1j * phi0) if variant == "v2+bp" else 0)
            return torch.angle(s + 1e-12)[None, None].float()
        for step in range(STEPS + 1):
            ph = phase(); loss = r3_loss(multiplane_amplitudes(ph, CFG, propagate_fresnel), T); v = float(loss)
            if v < best[0]: best = (v, ph.detach().clone())
            if step == STEPS: break
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        od = out / f"{label}_{variant}"
        export_run(targets, best[1], od, metadata={"test": "bpmix_ceiling", "variant": variant}, synthetic_test=True)
        R = [img(od / f"reconstructed_{k:02d}.png") for k in range(1, 5)]; Tn = [img(t) for t in targets]
        C = [[pear(R[k], Tn[j]) for j in range(4)] for k in range(4)]
        p = best[1][0, 0].double().numpy(); f = np.fft.fftfreq(128) / 0.5; rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
        pw = np.abs(np.fft.fft2(np.exp(1j * p))) ** 2; hf = float(pw[rr > 0.5].sum() / pw.sum())
        res[f"{label}/{variant}"] = dict(R3=best[0], P=float(np.mean([C[k][k] for k in range(4)])), own=sum(int(np.argmax(C[k]) == k) for k in range(4)),
                                         iz=float(np.mean([pear(R[i], R[j]) for i in range(4) for j in range(i + 1, 4)])), HF=hf)
        print(label, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{label}/{variant}"].items()}, flush=True)
json.dump(res, open(out / "bpmix_result.json", "w"), indent=1)
