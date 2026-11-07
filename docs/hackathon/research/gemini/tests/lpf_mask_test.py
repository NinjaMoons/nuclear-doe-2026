"""Read-only (#49): does B-plane separation need high spatial frequencies in the DOE mask?
Low-pass exp(i*phi) of saved teacher masks (radial cutoff r_c in Nyquist units), phi' = arg(LPF), then the OFFICIAL
export (DOE.png uint8 -> reload -> 4 Fresnel -> 4 PNG -> metrics). Reports P, own-argmax, B inter-z (z1 physical grid).
usage: PYTHONPATH=<runtime src> python lpf_mask_test.py <out_dir> <label=export_dir> ..."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from doe_core.contracts_v1 import CaseConfigV1
from doe_experiments.export import export_run
from doe_experiments.phase_codec import load_doe_png
CFG = CaseConfigV1(); N = 128; P = [CFG.plane(k).output_pitch_m(N, N)[0] for k in range(4)]
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
def z1(im, k):
    n = np.arange(N) - 64; g = (n * P[0] / P[k] + 64) / 127 * 2 - 1; gy, gx = np.meshgrid(g, g, indexing="ij")
    return F.grid_sample(torch.from_numpy(im)[None, None], torch.from_numpy(np.stack([gx, gy], -1)[None]), mode="bilinear", align_corners=True)[0, 0].numpy()
f = np.fft.fftfreq(N) / 0.5; r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
out_root = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1); d = Path(d)
    phi = load_doe_png(d / "DOE.png", dtype=torch.float64)[0, 0].numpy()
    targets = [d / "targets" / f"target_{k:02d}.png" for k in range(1, 5)]
    T = [img(t) for t in targets]
    for rc in (9.0, 0.7, 0.5, 0.4, 0.3):
        u = np.fft.ifft2(np.fft.fft2(np.exp(1j * phi)) * (r <= rc))
        keep = float((np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2 * (r <= rc)).sum() / (np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2).sum())
        ph = torch.from_numpy(np.angle(u))[None, None].float()
        od = out_root / f"{label}_rc{rc}"
        if not od.exists():
            export_run(targets, ph, od, metadata={"test": "lpf_mask", "rc": rc}, synthetic_test=True)
        R = [img(od / f"reconstructed_{k:02d}.png") for k in range(1, 5)]
        C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
        Rp = [z1(R[k], k) for k in range(4)]
        res[f"{label}/rc{rc}"] = {"power_kept": keep, "P": [C[k][k] for k in range(4)], "own": sum(int(np.argmax(C[k]) == k) for k in range(4)),
                                  "interz_phys": float(np.mean([pear(Rp[i], Rp[j]) for i in range(4) for j in range(i + 1, 4)]))}
        v = res[f"{label}/rc{rc}"]; print(f"{label:22s} rc {rc:3.1f} kept {keep:.2f} P {np.mean(v['P']):.3f} {[round(x,2) for x in v['P']]} own {v['own']}/4 iz {v['interz_phys']:.2f}")
json.dump(res, open(out_root / "lpf_result.json", "w"), indent=1)
