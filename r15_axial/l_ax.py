"""R15 (#48): axial-innovation auxiliary loss. RESEARCH ONLY.

Representation (declared): the contract display x_k = A_k / max(A_k) of the raw amplitude |U| on each plane's NATIVE
output grid, compared index-by-index with T_k (PNG/255), exactly like the contract MSE and the official per-plane
metrics. This is LABEL-SPACE (pixel-index) differencing across planes whose physical pitch differs by up to 15 %
(z4/z1); it is NOT a physical-frame comparison and no physical-separation claim is made from it.

    e_k = x_k - T_k,  e_bar = mean_k e_k,  d_k = e_k - e_bar  (sum_k d_k = 0)
    L_ax = mean_k ||d_k||^2 / (mean_k ||T_k||^2 + eps)          (||.||^2 sums pixels; batch mean)

Identity: mean_k ||e_k||^2 = ||e_bar||^2 + mean_k ||d_k||^2, so with c = contract MSE = mean_k ||e_k||^2 / (H W):
    L_ax = (H W c - ||e_bar||^2) / (mean_k ||T_k||^2 + eps)
i.e. L_ax is the contract-domain error with its common-mode (plane-averaged) part removed. Adding lambda*L_ax to R3
re-weights the contract-domain error toward its plane-DIFFERENTIAL part; it brings no new optical information.
"""
import torch

import doe_models.train_multiplane as tm

LAMBDA = 0.05
EPS = 1e-12
DIMS = (-2, -1)


def axial_innovation(amplitudes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    a, t = amplitudes.double(), targets.double()
    x = a / a.amax(dim=DIMS, keepdim=True)
    e = x - t                                                       # [B,4,1,H,W]
    d = e - e.mean(dim=1, keepdim=True)
    num = d.pow(2).sum(dim=DIMS).mean(dim=1)                        # [B,1]
    den = t.pow(2).sum(dim=DIMS).mean(dim=1) + EPS * t.shape[-1] * t.shape[-2]
    return (num / den).mean()


def r3(amplitudes, targets, beta=0.3):
    """Exact R3 = lsq_gain_nmse + 0.3*mean(1-corr(A/max,T)) + 0.1*contract (identical to train_wrapper 9e8cd4a9)."""
    base, info = tm.training_objective(amplitudes, targets, tm.LSQ_MIX_OBJECTIVE)
    a, t = amplitudes.double(), targets.double()
    x = a / a.amax(dim=DIMS, keepdim=True)
    dx, dt = x - x.mean(dim=DIMS, keepdim=True), t - t.mean(dim=DIMS, keepdim=True)
    vx, vt = (dx * dx).mean(dim=DIMS), (dt * dt).mean(dim=DIMS)
    corr = (dx * dt).mean(dim=DIMS) / torch.sqrt((vx + 1e-12) * (vt + 1e-12))
    term = torch.where(vt <= (0.5 / 255) ** 2, torch.zeros_like(corr), 1 - corr).mean()
    return base + beta * term, info


def total(amplitudes, targets, lam=LAMBDA):
    base, info = r3(amplitudes, targets)
    return base + lam * axial_innovation(amplitudes, targets), info
