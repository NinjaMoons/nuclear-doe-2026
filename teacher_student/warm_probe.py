"""Cheap label-quality probe (TRAIN only, no student training): student v2 phase vs teacher = Adam warm-started from it.
python -m teacher_student.warm_probe <corpus> <v2_ckpt> <out.json>"""
import json, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np, torch
from PIL import Image
from back_projection_stem.infer_neural_stem import research_load_checkpoint
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.metrics import evaluate_multiplane
from doe_experiments.phase_codec import encode_phase, decode_phase
from doe_experiments.png_io import amplitude_to_u8
from doe_models.adam_multiplane import multiplane_amplitudes
from doe_models.train_multiplane import infer_phase
from teacher_student.teacher import teacher_phase
CORPUS, CKPT, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = CaseConfigV1(); man = json.loads((CORPUS / "manifest_expanded.json").read_text(encoding="utf-8")); S = {s["key"]: s for s in man["samples"]}
model, payload = research_load_checkpoint(CKPT)
def pear(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 0 else float("nan")
def ev(phase, t):
    q = torch.from_numpy(decode_phase(encode_phase(phase[0, 0].double().numpy())))[None, None].float()   # DOE uint8 roundtrip
    a = multiplane_amplitudes(q, cfg, propagate_fresnel)
    x = torch.from_numpy(amplitude_to_u8(a.detach()).astype(np.float64) / 255)
    m = evaluate_multiplane(x, t.double())[0]
    R = [x[0, k, 0].numpy() for k in range(4)]; T = [t[0, k, 0].double().numpy() for k in range(4)]
    own = sum(int(np.nanargmax([pear(R[k], T[j]) for j in range(4)]) == k) for k in range(4))
    return {"P": float(np.mean([v.pearson for v in m])), "S": float(np.mean([v.ssim for v in m])), "own": own,
            "interz_native": float(np.mean([pear(R[i], R[j]) for i, j in combinations(range(4), 2)]))}
rows = []
for key in ("B208:006", "B64:010", "A64:000", "B64:001"):
    s = S[key]; assert s["split"] == "train"
    t = torch.from_numpy(np.stack([np.asarray(Image.open(CORPUS / p), dtype=np.float32) / 255 for p in s["paths"]]))[None, :, None]
    sp = infer_phase(model, t)
    r = {"key": key, "student_v2": ev(sp, t)}
    for steps in (50, 150):
        ph, _, info = teacher_phase(t, steps=steps, init=sp.double())
        r[f"teacher_warm{steps}"] = ev(ph, t) | {"elapsed_s": info["elapsed_s"]}
    rows.append(r); print(json.dumps(r), flush=True)
OUT.write_text(json.dumps({"checkpoint": payload["sha256"], "rows": rows}, indent=1), encoding="utf-8")
