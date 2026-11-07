"""RESEARCH ONLY (R20 CPU pre-check, TRAIN quartets only): per-quartet Adam on phase from the v2 export,
arm R3 vs R3 + lam*L_con (lam: lam*L_con(init) = 0.25*R3(init)). Same lr/steps/seed. Official export of the best-R3
phase, then P/own/inter-z (full, low r<=.15, mid .15-.5) on reloaded PNG, mean brightness, hot-pixel fraction.
usage: python -m xplane.precheck <out_dir> <label=v2_export_dir> ..."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.export import export_run
from doe_experiments.phase_codec import load_doe_png
from doe_experiments.data import load_png_quartet
from doe_models.adam_multiplane import multiplane_amplitudes
from back_projection_stem.r3_loss import r3_loss
from xplane.loss import xplane_contrast
from doe_experiments.metrics import evaluate_multiplane
CFG = CaseConfigV1(); N = 128; STEPS = int(__import__("os").environ.get("STEPS", 300)); LR = 0.05
f = np.fft.fftfreq(N) / 0.5; r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
band = lambda x, m: np.fft.ifft2(np.fft.fft2(x) * m).real
def evaluate(phi, targets, od):
    export_run(targets, torch.from_numpy(phi)[None, None].float(), od, metadata={"test": "r20_precheck"}, synthetic_test=True)
    R = [img(od / f"reconstructed_{k:02d}.png") for k in range(1, 5)]; T = [img(t) for t in targets]
    C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    iz = lambda m: float(np.mean([pear(band(R[i], m), band(R[j], m)) for i, j in pairs]))
    S = float(np.mean([m.ssim for m in evaluate_multiplane(torch.from_numpy(np.stack(R))[None, :, None], torch.from_numpy(np.stack(T))[None, :, None])[0]]))
    return dict(P=float(np.mean([C[k][k] for k in range(4)])), S=S, own=sum(int(np.argmax(C[k]) == k) for k in range(4)),
                iz=iz(np.ones_like(r)), iz_low=iz(r <= .15), iz_mid=iz((r > .15) & (r <= .5)),
                bright=float(np.mean([x.mean() for x in R])), hot=float(np.mean([(x >= 0.999).mean() for x in R])))
out = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1); d = Path(d)
    targets = [d / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]; T = load_png_quartet(targets)
    phi0 = load_doe_png(d / "DOE.png", dtype=torch.float64).float()
    with torch.no_grad():
        A0 = multiplane_amplitudes(phi0, CFG, propagate_fresnel); r30 = float(r3_loss(A0, T)); lc0 = float(xplane_contrast(A0, T))
    lam = float(__import__("os").environ.get("FRAC", 0.25)) * r30 / max(lc0, 1e-8); res[label] = {"R3_init": r30, "Lcon_init": lc0, "lam": lam}
    res[label]["v2"] = evaluate(((phi0 + math.pi) % (2 * math.pi) - math.pi)[0, 0].double().numpy(), targets, out / f"{label}_v2")
    for arm, w in (("control", 0.0), ("r20", lam)):
        torch.manual_seed(0); phase = phi0.clone().requires_grad_(True); opt = torch.optim.Adam([phase], lr=LR); best = (math.inf, phi0.clone(), None)
        for step in range(STEPS + 1):
            A = multiplane_amplitudes(phase, CFG, propagate_fresnel); l3 = r3_loss(A, T); lc = xplane_contrast(A, T); loss = l3 + w * lc
            if float(loss) < best[0]: best = (float(loss), phase.detach().clone(), (float(l3), float(lc)))
            if step == STEPS: break
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        ph = ((best[1] + math.pi) % (2 * math.pi) - math.pi)[0, 0].double().numpy()
        m = evaluate(ph, targets, out / f"{label}_{arm}"); m.update(R3=best[2][0], Lcon=best[2][1]); res[label][arm] = m
    print(label, json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in res[label].items()}), flush=True)
json.dump(res, open(out / "r20_precheck.json", "w"), indent=1)
