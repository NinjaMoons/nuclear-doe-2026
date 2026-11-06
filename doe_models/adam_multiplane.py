"""Multi-plane Adam synthesis of ONE shared DOE phase (contract v1, Issue #3).

One real phase [B,1,H,W] -> U = exp(i*phase) (A=1) -> injected propagator at the four
official distances -> raw intensities [B,4,1,H,W] -> per-plane max display -> MSE
against targets in [0,1]. The propagator is always injected; this module contains no
optics of its own.

The loss/normalisation policy is PROVISIONAL: it is the v1.0 proposal of PR #15
(intensity I=|U|^2, per-plane max). The official DOCX asks for reconstructed field
modulus |U|; until the updated contract fixes the compared quantity and its
normalisation, the whole policy lives only in `multiplane_loss` so it can be replaced
in one place. The optimiser, forward and best-state logic do not depend on it.
"""
from __future__ import annotations

import dataclasses
import math
import time
from numbers import Real

import torch

from doe_core.contracts_v1 import CaseConfigV1, FresnelField, FresnelPropagator, MultiPlaneResult

__all__ = ["LOSS_POLICY", "LOSS_POLICY_STATUS", "multiplane_intensities", "multiplane_loss", "synthesize_multiplane"]

LOSS_POLICY = "per_plane_max_display_mse_v1"
LOSS_POLICY_STATUS = "provisional: PR #15 v1.0 proposal on I=|U|^2; DOCX |U| policy pending contract update"
NUM_PLANES = 4
_MAX_EPS = 1e-12
_REAL_DTYPES = (torch.float32, torch.float64)


def _forward(phase: torch.Tensor, config: CaseConfigV1, propagate_fn: FresnelPropagator):
    field = torch.complex(torch.cos(phase), torch.sin(phase))
    planes, pitches = [], []
    for k in range(NUM_PLANES):
        geometry = config.plane(k)
        result = propagate_fn(field, geometry)
        out = getattr(result, "field", None)
        if not isinstance(result, FresnelField) or not isinstance(out, torch.Tensor) \
                or not out.is_complex() or out.shape != field.shape:
            raise ValueError(f"propagate_fn must return FresnelField with complex field of shape "
                             f"{tuple(field.shape)} (plane {k})")
        planes.append(out.real ** 2 + out.imag ** 2)
        pitches.append([float(p) for p in result.output_pitch_m])
    return torch.stack(planes, dim=1), pitches


def multiplane_intensities(phase: torch.Tensor, config: CaseConfigV1,
                           propagate_fn: FresnelPropagator) -> torch.Tensor:
    """Raw intensities [B,4,1,H,W] of one phase [B,1,H,W] at the four case distances."""
    if not isinstance(phase, torch.Tensor) or phase.dtype not in _REAL_DTYPES:
        raise ValueError("phase must be a float32/float64 tensor")
    if phase.ndim != 4 or phase.shape[1] != 1 or tuple(phase.shape[-2:]) != config.image_shape:
        raise ValueError(f"phase must have shape [B,1,{config.image_shape[0]},{config.image_shape[1]}]")
    return _forward(phase, config, propagate_fn)[0]


def multiplane_loss(intensities: torch.Tensor, targets: torch.Tensor):
    """Return (total, per_plane[4]) for mean((I/max(I) - T)^2), max per plane, not detached.

    PROVISIONAL policy (see module docstring). Planes are weighted equally (1/4).
    """
    if intensities.ndim != 5 or intensities.shape[1] != NUM_PLANES or intensities.shape != targets.shape:
        raise ValueError("intensities and targets must share shape [B,4,1,H,W]")
    peak = intensities.amax(dim=(-2, -1), keepdim=True)
    if bool((peak <= _MAX_EPS).any()):
        raise ValueError(f"predicted intensity max <= {_MAX_EPS} on at least one plane")
    sq = (intensities / peak - targets) ** 2
    return sq.mean(), sq.mean(dim=(0, 2, 3, 4))


def _check_int(value, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")


def _validate(targets, config, steps, lr, seed, propagate_fn) -> None:
    if not isinstance(config, CaseConfigV1):
        raise ValueError("config must be CaseConfigV1")
    _check_int(steps, "steps", 1)
    _check_int(seed, "seed", 0)
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and > 0")
    if not callable(propagate_fn):
        raise ValueError("propagate_fn must be callable")
    if not isinstance(targets, torch.Tensor) or targets.dtype not in _REAL_DTYPES:
        raise ValueError("targets must be a float32/float64 tensor")
    h, w = config.image_shape
    if targets.ndim != 5 or targets.shape[0] < 1 or tuple(targets.shape[1:]) != (NUM_PLANES, 1, h, w):
        raise ValueError(f"targets must have shape [B,4,1,{h},{w}]")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets contain NaN/Inf")
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        raise ValueError("targets must lie in [0,1]")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _wrap(phase: torch.Tensor) -> torch.Tensor:
    wrapped = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
    return torch.where(wrapped >= math.pi, wrapped - 2 * math.pi, wrapped)


def _json_config(config: CaseConfigV1) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(config).items()}


def synthesize_multiplane(targets: torch.Tensor, config: CaseConfigV1, *,
                          propagate_fn: FresnelPropagator, steps: int = 100,
                          lr: float = 0.05, seed: int = 0) -> MultiPlaneResult:
    """Optimise one phase per quartet with Adam; return the best state (initial included)."""
    _validate(targets, config, steps, lr, seed, propagate_fn)
    device, dtype = targets.device, targets.dtype
    _sync(device)
    start = time.perf_counter()

    batch = targets.shape[0]
    h, w = config.image_shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    initial = (torch.rand((batch, 1, h, w), generator=gen, dtype=torch.float64) * 2 - 1) * math.pi
    phase = initial.to(device=device, dtype=dtype).requires_grad_(True)
    optimizer = torch.optim.Adam([phase], lr=float(lr))

    history: list[float] = []
    best_loss, best_step, best_phase = math.inf, -1, None
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = multiplane_loss(_forward(phase, config, propagate_fn)[0], targets)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss at step {step}")
        history.append(value)
        if value < best_loss:
            best_loss, best_step, best_phase = value, step, phase.detach().clone()
        if step == steps:
            break
        loss.backward()
        optimizer.step()

    returned = _wrap(best_phase)
    with torch.no_grad():
        intensities, pitches = _forward(returned, config, propagate_fn)
        final_loss, plane_losses = multiplane_loss(intensities, targets)
    _sync(device)
    elapsed = time.perf_counter() - start

    metadata = {
        "method": "adam_multiplane",
        "seed": seed,
        "steps": steps,
        "lr": float(lr),
        "loss_policy": LOSS_POLICY,
        "loss_policy_status": LOSS_POLICY_STATUS,
        "plane_weights": [1.0 / NUM_PLANES] * NUM_PLANES,
        "illumination": "A=1, U=exp(i*phase)",
        "initial_phase": "uniform[-pi,pi) from torch.Generator(cpu).manual_seed(seed), float64, then cast",
        "phase_wrap": "remainder(phi+pi, 2pi)-pi -> [-pi,pi)",
        "config": _json_config(config),
        "distances_m": list(config.distances_m),
        "output_pitches_m": pitches,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "best_step": best_step,
        "best_loss": best_loss,
        "final_loss": float(final_loss),
        "plane_losses": [float(v) for v in plane_losses],
        "timing_scope": "validation+init+optimisation+final forward",
    }
    return MultiPlaneResult(phase_rad=returned.detach(), intensities=intensities.detach(),
                            loss_history=history, elapsed_s=elapsed, metadata=metadata)
