"""Read-only separation diagnostics on saved VAL exports: own-target matrix C[k,j], argmax counts, inter-z Pearson on
the physical grid (z2..z4 resampled onto z1's centred grid). argv: <out.json> <label=export_val_dir> ..."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

N = 128


def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def to_z1_grid(img, pitch, pitch1):
    """Bilinear resample of a plane with pitch `pitch` onto plane-1 centred grid x=(n-64)*pitch1 (align_corners=True)."""
    n = np.arange(N) - N // 2
    xs = n * pitch1 / pitch                              # target positions in source pixel units (centred)
    g = (xs + N // 2) / (N - 1) * 2 - 1                  # -> normalised coords for align_corners=True
    gy, gx = np.meshgrid(g, g, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1)[None]).float()
    return F.grid_sample(torch.from_numpy(img)[None, None].float(), grid, mode="bilinear", align_corners=True)[0, 0].numpy()


out = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1)
    rows = []
    for q in sorted(Path(d).iterdir()):
        man = json.loads((q / "manifest.json").read_text(encoding="utf-8"))
        key, pitches = man["metadata"]["sample_key"], [p[1] for p in man["output_pitches_m"]]
        R = [np.asarray(Image.open(q / f"reconstructed_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
        T = [np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float64) / 255 for k in range(1, 5)]
        C = [[pear(R[k], T[j]) for j in range(4)] for k in range(4)]
        own = sum(int(np.nanargmax(C[k]) == k) for k in range(4))
        Rp = [to_z1_grid(R[k], pitches[k], pitches[0]) for k in range(4)]
        Tp = [to_z1_grid(T[k], pitches[k], pitches[0]) for k in range(4)]
        rows.append({"key": key, "g": key[0], "C": C, "own_argmax": own,
                     "interz_rec_phys": float(np.nanmean([pear(Rp[i], Rp[j]) for i, j in combinations(range(4), 2)])),
                     "interz_tgt_phys": float(np.nanmean([pear(Tp[i], Tp[j]) for i, j in combinations(range(4), 2)]))})
    summ = {}
    for g in ("A", "B"):
        s = [r for r in rows if r["g"] == g]
        summ[g] = {"own_argmax": sum(r["own_argmax"] for r in s), "planes": 4 * len(s),
                   "interz_rec_phys": float(np.mean([r["interz_rec_phys"] for r in s])),
                   "interz_tgt_phys": float(np.mean([r["interz_tgt_phys"] for r in s])),
                   "diag_minus_offdiag": float(np.mean([np.mean([r["C"][k][k] for k in range(4)]) -
                                                       np.nanmean([r["C"][k][j] for k in range(4) for j in range(4) if j != k]) for r in s]))}
    out[label] = {"summary": summ, "rows": rows}
    print(label, json.dumps(summ))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
