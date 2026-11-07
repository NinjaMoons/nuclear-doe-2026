"""RESEARCH ONLY (R21 pre-check, TRAIN-B): is the per-sample teacher's target-plane phase theta_k a learnable function of
the targets? Teachers = Adam on R3 (lr .05, 1000 steps) per quartet from (a) V2 phase + N(0,.1 rad) seed 0, (b) same seed 1,
(c) uniform random phase. Report P/SSIM/own (official export), |grad theta| (rad/px) on lit target pixels, and circular
agreement |mean_{lit} e^{i(theta_a - theta_b)}| between teachers (1 = same phase up to a constant, ~0 = unrelated)."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.phase_codec import load_doe_png
from doe_experiments.data import load_png_quartet
from doe_models.adam_multiplane import multiplane_amplitudes
from back_projection_stem.r3_loss import r3_loss
from xplane.precheck import evaluate
CFG = CaseConfigV1(); STEPS = 1000
def theta(ph):
    u = torch.complex(torch.cos(ph.double()), torch.sin(ph.double()))
    return [torch.angle(propagate_fresnel(u, CFG.plane(k)).field)[0, 0].numpy() for k in range(4)]
def teacher(phi0, T):
    p = phi0.clone().requires_grad_(True); opt = torch.optim.Adam([p], lr=0.05); best = (math.inf, None)
    for s in range(STEPS + 1):
        l = r3_loss(multiplane_amplitudes(p, CFG, propagate_fresnel), T); v = float(l.detach())
        if v < best[0]: best = (v, p.detach().clone())
        if s == STEPS: break
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    return best[1]
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
def grad_rough(th, lit):
    gx = wrap(np.diff(th, axis=1))[:-1, :]; gy = wrap(np.diff(th, axis=0))[:, :-1]
    return float(np.sqrt(gx ** 2 + gy ** 2)[lit[:-1, :-1]].mean())
out = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    lab, d = arg.split("=", 1); d = Path(d); tg = [d / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]; T = load_png_quartet(tg)
    lit = [T[0, k, 0].numpy() > 0.5 for k in range(4)]
    phi0 = load_doe_png(d / "DOE.png", dtype=torch.float64).float(); TH, row = {}, {}
    inits = {"v2": None, "t_n0": phi0 + 0.1 * torch.randn(phi0.shape, generator=torch.Generator().manual_seed(0)),
             "t_n1": phi0 + 0.1 * torch.randn(phi0.shape, generator=torch.Generator().manual_seed(1)),
             "t_rand": (torch.rand(phi0.shape, generator=torch.Generator().manual_seed(2)) * 2 - 1) * math.pi}
    for name, init in inits.items():
        ph = phi0 if init is None else teacher(init, T)
        m = evaluate(wrap(ph[0, 0].double().numpy()), tg, out / f"{lab}_{name}")
        TH[name] = theta(ph); m["rough_theta"] = float(np.mean([grad_rough(TH[name][k], lit[k]) for k in range(4)])); row[name] = m
    agree = lambda a, b: float(np.mean([abs(np.exp(1j * (TH[a][k] - TH[b][k]))[lit[k]].mean()) for k in range(4)]))
    row["agree"] = {"n0~n1": agree("t_n0", "t_n1"), "n0~rand": agree("t_n0", "t_rand"), "n0~v2": agree("t_n0", "v2"), "v2~rand": agree("v2", "t_rand")}
    res[lab] = row
    print(lab, json.dumps({n: ({k: round(v, 3) for k, v in m.items() if k in ("P", "S", "own", "rough_theta")} if n != "agree" else {k: round(v, 3) for k, v in m.items()}) for n, m in row.items()}), flush=True)
json.dump(res, open(out / "teacher_theta.json", "w"), indent=1)
