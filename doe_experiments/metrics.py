"""Global amplitude metrics under the case v1.1 fixed display domain."""
from __future__ import annotations

import math

import numpy as np
import torch

from doe_core.contracts_v1 import PairMetricsV1


def _values(value: torch.Tensor) -> np.ndarray:
    if (not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] < 1
            or tuple(value.shape[1:]) != (4, 1, 128, 128)
            or value.dtype not in (torch.float32, torch.float64)):
        raise ValueError("metric inputs must be float32/64 [B,4,1,128,128], B >= 1")
    values = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("metric inputs must be finite in [0,1]")
    return values


def evaluate_multiplane(prediction: torch.Tensor, target: torch.Tensor) -> list[list[PairMetricsV1]]:
    """Return [batch][plane] Pearson and global population SSIM, without rescaling."""
    predictions, targets = _values(prediction), _values(target)
    if predictions.shape != targets.shape:
        raise ValueError("metric inputs must have equal shapes")
    result = []
    for predicted_batch, target_batch in zip(predictions, targets):
        batch = []
        for predicted, expected in zip(predicted_batch, target_batch):
            x, y = predicted.ravel(), expected.ravel()
            # Remove a reference pixel before averaging so near-constant
            # inputs retain ULP-scale variation instead of losing it in DC.
            shifted_x, shifted_y = x - x[0], y - y[0]
            mean_x, mean_y = float(shifted_x.mean()), float(shifted_y.mean())
            mx, my = float(x[0]) + mean_x, float(y[0]) + mean_y
            dx, dy = shifted_x - mean_x, shifted_y - mean_y
            sx, sy = float(np.dot(dx, dx)), float(np.dot(dy, dy))
            cross = float(np.dot(dx, dy))
            scale_x, scale_y = float(np.max(np.abs(dx))), float(np.max(np.abs(dy)))
            defined = scale_x > 0 and scale_y > 0
            if defined:
                # Scale centered vectors to avoid underflow for tiny, valid
                # nonconstant images; this is the same centered Pearson sum.
                nx, ny = dx / scale_x, dy / scale_y
                pearson = float(np.dot(nx, ny) / math.sqrt(float(np.dot(nx, nx)) * float(np.dot(ny, ny))))
            else:
                pearson = None
            vx, vy, covariance = sx / x.size, sy / x.size, cross / x.size
            ssim = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)
                    / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)))
            batch.append(PairMetricsV1(pearson=pearson, ssim=float(ssim), pearson_defined=defined))
        result.append(batch)
    return result
