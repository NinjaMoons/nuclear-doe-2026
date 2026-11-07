"""Speckle / grain readout on saved VAL exports (#49 review item 3). Read-only.
python -m teacher_student.speckle <out.json> <label=export_val_dir> ...
Per arm: HF fraction of |FFT exp(i*phi)|^2 beyond 0.5 Nyquist (phi from DOE.png codec), DOE roughness (mean |grad|
of wrapped phase), hot-pixel fraction of reconstructions (PNG >= 0.9), mean display level, inter-z (native) in the
0.15-0.3 Nyquist band."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from doe_experiments.phase_codec import decode_phase

N = 128
f = np.fft.fftfreq(N)
FR = np.hypot(*np.meshgrid(f, f, indexing="ij")) / 0.5           # radius in units of Nyquist


def band(img, lo, hi):
    F = np.fft.fft2(img - img.mean())
    F[(FR < lo) | (FR >= hi)] = 0
    return np.real(np.fft.ifft2(F))


def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


out = {}
for arg in sys.argv[2:]:
    lab, d = arg.split("=", 1)
    hf, rough, hot, mean, iz = [], [], [], [], []
    for q in sorted(Path(d).iterdir()):
        phi = decode_phase(np.asarray(Image.open(q / "DOE.png")))
        P = np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2
        hf.append(float(P[FR > 0.5].sum() / P.sum()))
        dphi = np.angle(np.exp(1j * np.diff(phi, axis=0)))
        rough.append(float(np.abs(dphi).mean()))
        R = [np.asarray(Image.open(q / f"reconstructed_0{k}.png"), dtype=float) / 255 for k in range(1, 5)]
        hot.append(float(np.mean([(r >= 0.9).mean() for r in R])))
        mean.append(float(np.mean([r.mean() for r in R])))
        B = [band(r, 0.15, 0.3) for r in R]
        iz.append(float(np.nanmean([pear(B[i], B[j]) for i, j in combinations(range(4), 2)])))
    out[lab] = {"hf_frac_gt_0.5nyq": float(np.mean(hf)), "doe_roughness": float(np.mean(rough)), "hot_px_ge_0.9": float(np.mean(hot)),
                "display_mean": float(np.mean(mean)), "interz_band_0.15_0.3": float(np.mean(iz))}
    print(lab, json.dumps({k: round(v, 5) for k, v in out[lab].items()}))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
