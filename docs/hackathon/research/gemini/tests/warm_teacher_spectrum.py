"""Read-only diagnostic (#49): does a WARM teacher (Adam on R3 from the v2 phase) gain quality by adding high spatial
frequencies to the mask, or while staying smooth? TRAIN quartets only. For steps in CHECK: official export of the
best-so-far phase, P/own/inter-z, mask HF fraction (r>0.5 Nyq), R90, and robustness to low-pass rc=0.7.
usage: PYTHONPATH=<runtime src> python warm_teacher_spectrum.py <out_dir> <label=v2_export_dir> ..."""
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
CFG = CaseConfigV1(); N = 128; CHECK = (0, 50, 300, 1000, 2000)
f = np.fft.fftfreq(N) / 0.5; r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
def spec(phi):
    p = np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2; o = np.argsort(r.ravel()); c = np.cumsum(p.ravel()[o]) / p.sum()
    return float(p[r > 0.5].sum() / p.sum()), float(r.ravel()[o][np.searchsorted(c, 0.9)])
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
def evaluate(phi, targets, od):
    export_run(targets, torch.from_numpy(phi)[None, None].float(), od, metadata={"test": "warm_teacher"}, synthetic_test=True)
    R = [img(od / f"reconstructed_{k:02d}.png") for k in range(1, 5)]; T = [img(t) for t in targets]
    C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
    return float(np.mean([C[k][k] for k in range(4)])), sum(int(np.argmax(C[k]) == k) for k in range(4)), float(np.mean([pear(R[i], R[j]) for i in range(4) for j in range(i + 1, 4)]))
out = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1); d = Path(d)
    targets = [d / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]
    T = load_png_quartet(targets)
    phi0 = load_doe_png(d / "DOE.png", dtype=torch.float64).float()
    phase = phi0.clone().requires_grad_(True); opt = torch.optim.Adam([phase], lr=0.05)
    best = (math.inf, phi0.clone()); rows = {}
    for step in range(max(CHECK) + 1):
        loss = r3_loss(multiplane_amplitudes(phase, CFG, propagate_fresnel), T); v = float(loss.detach())
        if v < best[0]: best = (v, phase.detach().clone())
        if step in CHECK:
            ph = ((best[1] + math.pi) % (2 * math.pi) - math.pi)[0, 0].double().numpy()
            P, own, iz = evaluate(ph, targets, out / f"{label}_s{step}")
            u = np.fft.ifft2(np.fft.fft2(np.exp(1j * ph)) * (r <= 0.7)); Plp, ownlp, izlp = evaluate(np.angle(u), targets, out / f"{label}_s{step}_lp07")
            hf, r90 = spec(ph); rows[step] = dict(R3=best[0], P=P, own=own, iz=iz, HF=hf, R90=r90, P_lp07=Plp, iz_lp07=izlp)
            print(f"{label:10s} step {step:4d} R3 {best[0]:.4f} P {P:.3f} own {own}/4 iz {iz:.2f} | HF {hf:.3f} R90 {r90:.2f} | LPF0.7: P {Plp:.3f} iz {izlp:.2f}", flush=True)
        if step == max(CHECK): break
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    res[label] = rows
json.dump(res, open(out / "warm_teacher_result.json", "w"), indent=1)
