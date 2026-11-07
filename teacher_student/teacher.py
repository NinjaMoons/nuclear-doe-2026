"""R16 teacher: per-quartet phase-only Adam on the R3 objective through the official Fresnel. TRAIN LABELS ONLY.
Never called at inference; never on VAL/TEST. Mirrors Grigory's DIRECT ADAM (#14/6707999572): uniform[-pi,pi) init from
Generator(seed), Adam lr 0.05, return the best state by objective (initial state included), wrapped to [-pi,pi)."""
import math
import time

import torch

from back_projection_stem.r3_loss import r3_loss
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes


def teacher_phase(targets: torch.Tensor, steps: int = 300, lr: float = 0.05, seed: int = 0, init=None):
    """targets [1,4,1,H,W] float -> (best_phase [1,1,H,W] float32 wrapped, best_amplitudes, info)."""
    cfg = CaseConfigV1()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if init is None:
        init = (torch.rand((1, 1) + tuple(targets.shape[-2:]), generator=gen, dtype=torch.float64) * 2 - 1) * math.pi
    phase = init.clone().float().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    best, best_phase, best_step, t0 = math.inf, None, -1, time.perf_counter()
    for step in range(steps + 1):
        amp = multiplane_amplitudes(phase, cfg, propagate_fresnel)
        loss = r3_loss(amp, targets)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite teacher objective at step {step}")
        if v < best:
            best, best_phase, best_step = v, phase.detach().clone(), step
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wrapped = torch.remainder(best_phase + math.pi, 2 * math.pi) - math.pi
    with torch.no_grad():
        amp = multiplane_amplitudes(wrapped, cfg, propagate_fresnel)
    return wrapped, amp, {"best_objective": best, "best_step": best_step, "steps": steps, "lr": lr, "seed": seed,
                          "elapsed_s": time.perf_counter() - t0}
