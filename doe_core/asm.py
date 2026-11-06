"""Differentiable scalar angular-spectrum propagation (contract v0, docs/INTERFACES.md).

Discrete operator for a field U[B,1,H,W] sampled with pitch (dy, dx):

    U_out = crop(IFFT2(H * FFT2(pad(U_in))))
    H(fy, fx) = exp(i * 2*pi * z * kz),  kz = sqrt(1/lambda^2 - fx^2 - fy^2), Im(kz) >= 0

with unshifted ``fftfreq`` grids and ``norm="ortho"``. ``pad_factor == 1`` is a periodic
domain; ``pad_factor > 1`` zero-pads symmetrically to factor*(H, W) (odd remainder at the
bottom/right) and crops the original central window afterwards. No band limit, no
normalisation, no detach: gradients flow through the field.

Sampling limits (aliasing of H) are documented in docs/modules/asm.md.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from doe_core.contracts import OpticsConfig

__all__ = ["propagate", "intensity"]

_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


def _validate_field(field: object, name: str = "field") -> None:
    if not isinstance(field, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(field).__name__}")
    if field.dtype not in _COMPLEX_DTYPES:
        raise ValueError(f"{name} must be complex64 or complex128, got {field.dtype}")
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W], got {tuple(field.shape)}")
    batch, _, height, width = field.shape
    if batch < 1 or height < 2 or width < 2:
        raise ValueError(f"{name} needs B>=1 and H,W>=2, got {tuple(field.shape)}")
    if not bool(torch.isfinite(field).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _transfer_cpu(
    height: int, width: int, wavelength_m: float, dy: float, dx: float, distance_m: float
) -> Tensor:
    """Transfer function on CPU in complex128, built fresh on every call.

    Computed in float64 because 2*pi*z/lambda reaches ~1e6 rad for typical
    geometries, where float32 phase error would be ~0.1 rad. Deliberately not
    cached: a tensor cached under torch.inference_mode() would later break
    autograd (review of PR #13).
    """
    fy = torch.fft.fftfreq(height, d=dy, dtype=torch.float64)
    fx = torch.fft.fftfreq(width, d=dx, dtype=torch.float64)
    arg = (1.0 / wavelength_m) ** 2 - fy[:, None] ** 2 - fx[None, :] ** 2
    root = torch.sqrt(arg.abs())
    propagating = arg >= 0
    two_pi_z = 2.0 * math.pi * distance_m
    # Propagating: |H| = 1, phase 2*pi*z*kz.  Evanescent: kz = i*root -> H = exp(-2*pi*z*root).
    phase = torch.where(propagating, two_pi_z * root, torch.zeros_like(root))
    magnitude = torch.where(propagating, torch.ones_like(root), torch.exp(-two_pi_z * root))
    return torch.polar(magnitude, phase)


def _pad_window(size: int, factor: int) -> tuple[int, int]:
    """(start, padded_size) of the original window inside the padded axis."""
    padded = size * factor
    return (padded - size) // 2, padded


def propagate(field: Tensor, config: OpticsConfig) -> Tensor:
    """Propagate a complex field [B,1,H,W] by ``config.distance_m`` (metres).

    Returns a tensor with the same shape, dtype and device. ``distance_m == 0``
    returns a copy of the input (identity, including high-frequency components).
    """
    _validate_field(field)
    if not isinstance(config, OpticsConfig):
        raise ValueError(f"config must be OpticsConfig, got {type(config).__name__}")

    if config.distance_m == 0:
        return field.clone()

    height, width = field.shape[-2:]
    factor = config.pad_factor
    top, padded_h = _pad_window(height, factor)
    left, padded_w = _pad_window(width, factor)

    if factor > 1:
        padded = field.new_zeros(field.shape[:-2] + (padded_h, padded_w))
        padded[..., top : top + height, left : left + width] = field
    else:
        padded = field

    dy, dx = config.pixel_pitch_m
    transfer = _transfer_cpu(
        padded_h, padded_w, float(config.wavelength_m), float(dy), float(dx),
        float(config.distance_m),
    ).to(device=field.device, dtype=field.dtype)

    spectrum = torch.fft.fft2(padded, dim=(-2, -1), norm="ortho")
    out = torch.fft.ifft2(spectrum * transfer, dim=(-2, -1), norm="ortho")

    if factor > 1:
        out = out[..., top : top + height, left : left + width]
    return out


def intensity(field: Tensor) -> Tensor:
    """Raw intensity |field|^2 (no normalisation); real dtype matching the field."""
    _validate_field(field)
    return field.real.square() + field.imag.square()
