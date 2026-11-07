"""RESEARCH ONLY (R19 acceptance): independent CPU replay of GPU checkpoints on the saved VAL16 exports.
For each arm: forward on targets/*.png (one pass, optimizer/backward forbidden) -> official encode -> compare uint8 codes
with the exported DOE.png; for tphase also theta stats (mean|theta|, mean|grad theta|) and |g_k|.
usage: python -m tphase_head.audit.replay_r19 <out.json> <label=run_dir> ..."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from tphase_head.train_r19 import research_load
from doe_experiments.phase_codec import load_doe_png
def forbid(*a, **k): raise AssertionError("optimizer/backward called during inference")
torch.optim.Optimizer.step = forbid; torch.autograd.backward = forbid; torch.Tensor.backward = forbid
res = {}
for arg in sys.argv[2:]:
    label, run = arg.split("=", 1); run = Path(run)
    model, payload = research_load(run / "step_006000.pt"); model.eval()
    d = run / "eval_val" / "export" / "step_006000.pt" / "val"; rows = []
    for q in sorted(d.iterdir()):
        T = torch.stack([torch.from_numpy(np.asarray(Image.open(q / "targets" / f"target_{k:02d}.png"), dtype=np.float32) / 255) for k in range(1, 5)])[None, :, None]
        with torch.no_grad():
            ph = model(T)
        mine = load_doe_png(q / "DOE.png", dtype=torch.float64)[0, 0]
        dd = torch.remainder(ph[0, 0].double() - mine + np.pi, 2 * np.pi) - np.pi
        row = {"key": json.loads((q / "manifest.json").read_text())["metadata"].get("sample_key"), "max_abs_dphi_vs_exported": float(dd.abs().max()),
               "codes_off_by_gt1": int((dd.abs() > 1.5 * 2 * np.pi / 255).sum())}
        if hasattr(model, "target_phase"):
            with torch.no_grad():
                th = model.target_phase(T)[0]
            row["theta_abs"] = float(th.abs().mean()); row["theta_grad"] = float((th[:, 1:] - th[:, :-1]).abs().mean() / 2 + (th[:, :, 1:] - th[:, :, :-1]).abs().mean() / 2)
        rows.append(row)
    out = {"ckpt_sha256": payload["sha256"], "rows": rows, "max_dphi": max(r["max_abs_dphi_vs_exported"] for r in rows),
           "codes_off_by_gt1": sum(r["codes_off_by_gt1"] for r in rows)}
    if hasattr(model, "gain"):
        out["gain_abs"] = [float(torch.linalg.vector_norm(model.gain[k])) for k in range(4)]
        out["theta_abs_mean"] = float(np.mean([r["theta_abs"] for r in rows])); out["theta_grad_mean"] = float(np.mean([r["theta_grad"] for r in rows]))
    res[label] = out; print(label, {k: v for k, v in out.items() if k != "rows"})
json.dump(res, open(sys.argv[1], "w"), indent=1)
