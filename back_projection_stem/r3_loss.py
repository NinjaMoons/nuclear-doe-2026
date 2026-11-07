"""R3 research loss (beta = 0.3), identical formula to r3_train_wrapper.py 9e8cd4a9 (#5/6706710241, #5/6707419097):
lsq_gain_nmse + BETA * mean_planes(1 - corr(A/max, T)) + 0.1 * contract_mse; planes with std(T) < 0.5/255 add 0
to the structure term. Uses the unchanged 49b96041 training_objective for the stock part."""
import torch

import doe_models.train_multiplane as tm

BETA, EPS, CONST_STD, DIMS = 0.3, 1e-12, 0.5 / 255, (-2, -1)


def r3_loss(amplitudes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    base, _ = tm.training_objective(amplitudes, targets, tm.LSQ_MIX_OBJECTIVE)
    a, t = amplitudes.double(), targets.double()
    x = a / a.amax(dim=DIMS, keepdim=True)
    dx, dt = x - x.mean(dim=DIMS, keepdim=True), t - t.mean(dim=DIMS, keepdim=True)
    vx, vt = (dx * dx).mean(dim=DIMS), (dt * dt).mean(dim=DIMS)
    corr = (dx * dt).mean(dim=DIMS) / torch.sqrt((vx + EPS) * (vt + EPS))
    term = torch.where(vt <= CONST_STD ** 2, torch.zeros_like(corr), 1 - corr).mean()
    return base + BETA * term
