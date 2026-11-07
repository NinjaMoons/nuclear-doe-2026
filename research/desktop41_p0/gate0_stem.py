"""Desktop #41 Gate0 for 12-ch STEM (research code 44f0378): fwd+bwd+Adam step timing on CUDA and CPU.

usage (cwd = research worktree at 44f0378):  python <this> <z1> <z2> <z3> <z4> <out.json>
Random init (seed 0), batch 4 (the synthetic quartet repeated), R3 research objective, fp32. Nothing is saved
except the JSON report; no corpus is read; this is a timing/health benchmark, not training.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch

import back_projection_stem.train_wrapper as tw  # patches the R3 objective name into the driver
import doe_models.train_multiplane as tm
from back_projection_stem.stem import build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from doe_models.adam_multiplane import multiplane_amplitudes


def bench(device, targets, n_warm=3, n=20):
    torch.manual_seed(0)
    model = build_stem_model(seed=0).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    comp = tm.production_components()
    cfg = CaseConfigV1()
    batch = targets.repeat(4, 1, 1, 1, 1).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, rows = [], []
    for i in range(n_warm + n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        amps = multiplane_amplitudes(model(batch), cfg, comp.propagate)
        loss, info = tw.research_objective(amps, batch, tw.R3_NAME)
        loss.backward()
        gnorm = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters() if p.grad is not None)))
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= n_warm:
            times.append(dt)
        rows.append({"loss": float(loss), "grad_norm": gnorm})
    finite = all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in rows)
    out = {"device": str(device), "step_median_s": float(np.median(times)), "step_p90_s": float(np.percentile(times, 90)),
           "n_timed": n, "n_warm": n_warm, "finite": finite, "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
           "params_finite": all(torch.isfinite(p).all().item() for p in model.parameters()),
           "in_channels": model.hparams.get("in_channels")}
    if device.type == "cuda":
        out["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    else:
        out["cpu_threads"] = torch.get_num_threads()
    return out


def main(argv):
    *paths, out = argv
    out = Path(out)
    if out.exists():
        raise SystemExit("refusing to overwrite")
    comp = tm.production_components()
    targets = comp.load_quartet([Path(p) for p in paths])
    rep = {"tool": "desktop41_gate0_stem", "torch": torch.__version__, "objective": tw.R3_NAME, "batch": 4, "dtype": "float32"}
    rep["cuda"] = bench(torch.device("cuda"), targets) if torch.cuda.is_available() else "unavailable"
    torch.set_num_threads(8)
    rep["cpu8"] = bench(torch.device("cpu"), targets, n_warm=2, n=5)
    if isinstance(rep["cuda"], dict):
        s = rep["cuda"]["step_median_s"]
        rep["estimate_cuda_train_only_s"] = {"100_updates": 100 * s, "500_updates": 500 * s, "1000_updates": 1000 * s,
                                             "note": "optimizer steps only; excludes data load, eval and checkpoint IO"}
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
