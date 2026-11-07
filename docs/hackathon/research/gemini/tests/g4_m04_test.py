"""Read-only (#49, G4 sec.3): coherent HF transplant vs isospectral scrambled HF, no training.
M0 V2 | M1 LPF(Adam) arg (unit modulus restored) | M1f full Adam | M2 LF(V2)+HF(Adam) | M3 LF(V2)+|HF(Adam)|*rand phase
| M4 LF(V2)+HF(one-step Fresnel residual of V2). Raised-cosine split 0.65-0.75 Nyquist (r in axis-Nyquist units).
Official export (DOE.png u8 -> reload -> 4 Fresnel -> 4 PNG). Extra: Pearson in centre |p|,|q|<45 vs periphery (FoV claim).
usage: PYTHONPATH=<runtime> python g4_m04_test.py <out_dir> <q=v2_dir,adam_dir,targets_dir> ..."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from doe_core.contracts_v1 import CaseConfigV1
from doe_experiments.export import export_run
from doe_experiments.phase_codec import load_doe_png
CFG = CaseConfigV1(); N = 128; LAM = 532e-9; DX = 10e-6; Z = [0.0241, 0.0253, 0.0265, 0.0277]
img = lambda p: np.asarray(Image.open(p), dtype=np.float64) / 255
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else np.nan
def dft(n):
    k = np.arange(n) - n // 2; return np.exp(-2j * np.pi * np.outer(k, k) / n)
D = dft(N); ax = (np.arange(N) - N // 2) * DX
def fwd(U, z):
    o = (np.arange(N) - N // 2) * LAM * z / (N * DX)
    cin = np.exp(1j * np.pi * (ax[:, None] ** 2 + ax[None] ** 2) / (LAM * z)); cout = np.exp(1j * np.pi * (o[:, None] ** 2 + o[None] ** 2) / (LAM * z))
    return cout * (D @ (U * cin) @ D.T) / N
def inv(V, z):  # exact inverse of fwd
    o = (np.arange(N) - N // 2) * LAM * z / (N * DX)
    cin = np.exp(1j * np.pi * (ax[:, None] ** 2 + ax[None] ** 2) / (LAM * z)); cout = np.exp(1j * np.pi * (o[:, None] ** 2 + o[None] ** 2) / (LAM * z))
    return np.conj(cin) * (D.conj().T @ (np.conj(cout) * V) @ D.conj()) / N
f = np.fft.fftfreq(N) / 0.5; r = np.sqrt(f[:, None] ** 2 + f[None] ** 2)
LF = np.clip((0.75 - r) / 0.10, 0, 1); LF = 0.5 - 0.5 * np.cos(np.pi * LF); HF = 1 - LF  # raised cosine 0.65..0.75
c = np.abs(np.arange(N) - 64) < 45; CEN = c[:, None] & c[None]
rng = np.random.default_rng(49)
out_root = Path(sys.argv[1]); res = {}
for arg in sys.argv[2:]:
    q, dirs = arg.split("=", 1); dv, da, dt = (Path(x) for x in dirs.split(","))
    targets = [dt / f"target_{k:02d}.png" for k in range(1, 5)]; T = [img(t) for t in targets]
    ps = load_doe_png(dv / "DOE.png", dtype=torch.float64)[0, 0].numpy(); pa = load_doe_png(da / "DOE.png", dtype=torch.float64)[0, 0].numpy()
    Us, Ua = np.exp(1j * ps), np.exp(1j * pa); Ua = Ua * np.exp(-1j * np.angle(np.sum(Ua * Us.conj())))
    Fs, Fa = np.fft.fft2(Us), np.fft.fft2(Ua)
    # M4: one-step residual of V2 (Gemini formula, alpha = lsq gain)
    res_f = 0
    for k in range(4):
        V = fwd(Us, Z[k]); a = np.abs(V); al = (a * T[k]).sum() / (a * a).sum()
        res_f = res_f + inv((T[k] - 0.5 * al * a) * np.exp(1j * np.angle(V)), Z[k])
    Fr = np.fft.fft2(res_f); Fr *= np.sqrt((np.abs(Fa) ** 2 * HF).sum() / max((np.abs(Fr) ** 2 * HF).sum(), 1e-30))  # same HF power as Adam
    masks = {"M0_v2": np.fft.fft2(Us), "M1_lpfAdam": LF * Fa, "M1f_adam": Fa, "M2_coherentHF": LF * Fs + HF * Fa,
             "M3_scrambledHF": LF * Fs + HF * np.abs(Fa) * np.exp(2j * np.pi * rng.random((N, N))), "M4_residHF": LF * Fs + HF * Fr}
    for name, F in masks.items():
        ph = torch.from_numpy(np.angle(np.fft.ifft2(F)))[None, None].float(); od = out_root / f"{q}_{name}"
        if not od.exists(): export_run(targets, ph, od, metadata={"test": "g4_m04", "mask": name}, synthetic_test=True)
        R = [img(od / f"reconstructed_{k:02d}.png") for k in range(4 + 1)[1:]]
        C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
        own = sum(int(np.argmax(C[k]) == k) for k in range(4))
        ghost = float(np.mean([max(0, C[k][j] - pear(T[k], T[j])) for k in range(4) for j in range(4) if j != k]))
        cen = float(np.mean([pear(R[k][CEN], T[k][CEN]) for k in range(4)])); per = float(np.mean([pear(R[k][~CEN], T[k][~CEN]) for k in range(4)]))
        fr = float((np.abs(R[0][~CEN]) ** 2).sum() / max((R[0] ** 2).sum(), 1e-30))
        v = res[f"{q}/{name}"] = {"P": [C[k][k] for k in range(4)], "P_mean": float(np.mean([C[k][k] for k in range(4)])), "own": own,
                                  "ghost_excess": ghost, "P_centre": cen, "P_periph": per, "periph_energy_z1": fr}
        print(f"{q:8s} {name:15s} P {v['P_mean']:.3f} own {own}/4 ghost {ghost:.3f} cen {cen:.3f} per {per:.3f} perE {fr:.2f}")
json.dump(res, open(out_root / "g4_m04_result.json", "w"), indent=1)
