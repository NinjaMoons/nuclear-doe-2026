"""Desktop #41 P0/Gate0 GPU diagnostic. READ-ONLY w.r.t. runtime source and checkpoint.

usage (from the extracted runtime root, so `doe_*` imports resolve to the frozen runtime):
  python -I <this> <checkpoint.pt> <z1> <z2> <z3> <z4> <out.json>

1. Loads the checkpoint with the official load_checkpoint (CPU), runs official infer_phase on CPU and on a
   deep copy moved to CUDA; compares phases (wrapped) and the official uint8 DOE codes.
2. Gate0: one forward + backward + Adam step of the R3-loss path (physics_loss) on CUDA on a SEPARATE copy;
   time, peak VRAM, finiteness. Nothing is saved; the checkpoint is not modified.
3. Fresnel CUDA vs CPU: complex FFT forward agreement and autograd gradient agreement on a random phase.
"""
import copy, hashlib, json, math, platform, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch

from doe_core.contracts_v1 import CaseConfigV1
from doe_models.adam_multiplane import multiplane_amplitudes, multiplane_loss
from doe_models.train_multiplane import load_checkpoint, infer_phase, production_components
from doe_experiments.phase_codec import save_doe_png, load_doe_png


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv):
    ckpt, *targets, out = argv
    targets = [Path(t) for t in targets]
    out = Path(out)
    if out.exists():
        raise SystemExit("refusing to overwrite " + str(out))
    report = {"tool": "desktop41_gate0_gpu", "checkpoint": str(Path(ckpt).resolve()), "checkpoint_sha256": sha(ckpt),
              "targets_sha256": [sha(t) for t in targets], "python": platform.python_version(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        report["status"] = "BLOCKED_NO_CUDA"
        out.write_text(json.dumps(report, indent=2))
        return 2
    dev = torch.device("cuda")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["vram_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
    comp = production_components()
    config = CaseConfigV1()
    model, payload = load_checkpoint(ckpt)
    tq = comp.load_quartet(targets)
    p = next(model.parameters())
    tq = tq.to(dtype=p.dtype)

    t0 = time.perf_counter(); phase_cpu = infer_phase(model, tq); t_cpu = time.perf_counter() - t0
    gmodel = copy.deepcopy(model).to(dev)
    tg = tq.to(dev)
    sync(); t0 = time.perf_counter(); phase_gpu = infer_phase(gmodel, tg); sync(); t_gpu_first = time.perf_counter() - t0
    times = []
    for _ in range(10):
        sync(); t0 = time.perf_counter(); infer_phase(gmodel, tg); sync(); times.append(time.perf_counter() - t0)
    d = torch.remainder(phase_gpu.cpu().double() - phase_cpu.double() + math.pi, 2 * math.pi) - math.pi
    tmp = out.parent / (out.stem + "_doe")
    tmp.mkdir(exist_ok=False)
    save_doe_png(phase_cpu, tmp / "cpu_DOE.png")
    save_doe_png(phase_gpu.cpu(), tmp / "gpu_DOE.png")
    from PIL import Image
    a = np.asarray(Image.open(tmp / "cpu_DOE.png")).astype(int)
    b = np.asarray(Image.open(tmp / "gpu_DOE.png")).astype(int)
    code_diff = np.abs(a - b)
    code_diff = np.minimum(code_diff, 256 - code_diff)
    report["inference"] = {
        "cpu_infer_phase_s": t_cpu, "cuda_infer_phase_first_s": t_gpu_first,
        "cuda_infer_phase_warm_median_s": float(np.median(times)),
        "phase_max_abs_wrapped_diff_rad": float(d.abs().max()), "phase_mean_abs_wrapped_diff_rad": float(d.abs().mean()),
        "doe_u8_codes_differing": int((code_diff > 0).sum()), "doe_u8_max_circular_code_diff": int(code_diff.max()),
        "cpu_DOE_sha256": sha(tmp / "cpu_DOE.png"), "gpu_DOE_sha256": sha(tmp / "gpu_DOE.png"),
        "scope": "infer_phase only (no IO/export); cuda timings with synchronize",
    }

    # Gate0: one optimizer step on a separate copy (official physics_loss path = multiplane forward + loss)
    tmodel = copy.deepcopy(model).to(dev).train()
    opt = torch.optim.Adam(tmodel.parameters(), lr=1e-3)
    tb = tg.repeat(4, 1, 1, 1) if tg.dim() == 4 else tg.repeat(4, 1, 1, 1, 1)
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for i in range(6):
        sync(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        phase = tmodel(tb)
        amps = multiplane_amplitudes(phase, config, comp.propagate)
        loss, per_plane = multiplane_loss(amps, tb)
        loss.backward()
        gfinite = all(torch.isfinite(q.grad).all().item() for q in tmodel.parameters() if q.grad is not None)
        opt.step()
        sync(); steps.append({"s": time.perf_counter() - t0, "loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
                              "grads_finite": gfinite})
    pfinite = all(torch.isfinite(q).all().item() for q in tmodel.parameters())
    report["gate0_train_step"] = {"batch": int(tb.shape[0]), "optimizer": "Adam lr1e-3", "loss": "multiplane_loss (adam_multiplane)",
                                  "steps": steps, "warm_step_median_s": float(np.median([s["s"] for s in steps[1:]])),
                                  "params_finite_after": pfinite,
                                  "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
                                  "note": "diagnostic copy, not saved; 4ch FourInputUNet, not STEM"}

    # Fresnel CUDA vs CPU forward + autograd
    g = torch.Generator().manual_seed(0)
    ph = (torch.rand(1, 1, 128, 128, generator=g, dtype=torch.float64) * 2 * math.pi)
    res = {}
    grads = {}
    for name, device in (("cpu", "cpu"), ("cuda", "cuda")):
        x = ph.clone().to(device).requires_grad_(True)
        amps = multiplane_amplitudes(x, config, comp.propagate)
        tgt = tq.double().to(device)
        loss, _ = multiplane_loss(amps, tgt)
        loss.backward()
        res[name] = amps.detach().cpu()
        grads[name] = x.grad.detach().cpu()
    report["fresnel_cuda_vs_cpu_float64"] = {
        "amplitude_max_abs_diff": float((res["cpu"] - res["cuda"]).abs().max()),
        "amplitude_max": float(res["cpu"].abs().max()),
        "grad_max_abs_diff": float((grads["cpu"] - grads["cuda"]).abs().max()),
        "grad_max_abs": float(grads["cpu"].abs().max()),
        "grad_finite": bool(torch.isfinite(grads["cuda"]).all()),
    }
    report["status"] = "PASS"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
