"""Single-FFT Fresnel diffraction (official case, slide 2; contract v1, docs/INTERFACES_V1.md).

Continuous formula from the case presentation:

    K(x,y) = exp(ikz)/(i*lambda*z) * exp(i*pi*(x^2+y^2)/(lambda*z))
             * FFT{ H(u,v) * exp(i*pi*(u^2+v^2)/(lambda*z)) }

Discretisation (team convention, contract v1):

    u = (arange(W) - W//2)*dx,   v = (arange(H) - H//2)*dy          input plane
    x = (arange(W) - W//2)*dx_o, y = (arange(H) - H//2)*dy_o        output plane
    dx_o = lambda*z/(W*dx),      dy_o = lambda*z/(H*dy)
    F    = fftshift(fft2(ifftshift(Q), norm="ortho"))               negative Fourier sign
    Uout = exp(i*2*pi*z/lambda)/(i*lambda*z) * exp(i*pi*(x^2+y^2)/(lambda*z))
           * sqrt(H*W)*dy*dx * F                                     (= dx*dy * unnormalised DFT)

Exactly one spatial FFT per call; no padding, crop, resampling, band limit or
normalisation. All phase factors are built in float64 on CPU on every call (no cache)
and cast to the field's complex dtype/device. Gradients flow through ``field``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from doe_core.contracts_v1 import FresnelField, FresnelGeometry

__all__ = ["propagate_fresnel", "sampling_report"]

_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


def _validate_field(field: object) -> None:
    if not isinstance(field, Tensor):
        raise ValueError(f"field must be a torch.Tensor, got {type(field).__name__}")
    if field.dtype not in _COMPLEX_DTYPES:
        raise ValueError(f"field must be complex64 or complex128, got {field.dtype}")
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(f"field must have shape [B,1,H,W], got {tuple(field.shape)}")
    batch, _, height, width = field.shape
    if batch < 1 or height < 2 or width < 2:
        raise ValueError(f"field needs B>=1 and H,W>=2, got {tuple(field.shape)}")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("field contains NaN or Inf")


def _centered_axis(size: int, pitch: float) -> Tensor:
    """Centred integer grid (index - size//2) * pitch, float64 on CPU."""
    return (torch.arange(size, dtype=torch.float64) - size // 2) * pitch


def _chirp(y: Tensor, x: Tensor, wavelength: float, distance: float) -> Tensor:
    """exp(i*pi*(x^2+y^2)/(lambda*z)) on the (y, x) grid, complex128."""
    phase = math.pi * (y[:, None] ** 2 + x[None, :] ** 2) / (wavelength * distance)
    return torch.polar(torch.ones_like(phase), phase)


def propagate_fresnel(field: Tensor, geometry: FresnelGeometry) -> FresnelField:
    """Propagate a complex field [B,1,H,W] from the DOE plane to distance z > 0.

    Returns the complex output field (same shape, dtype, device) together with the
    output pitch (dy_out, dx_out) in metres and the distance.
    """
    _validate_field(field)
    if not isinstance(geometry, FresnelGeometry):
        raise ValueError(f"geometry must be FresnelGeometry, got {type(geometry).__name__}")

    height, width = field.shape[-2:]
    wavelength = float(geometry.wavelength_m)
    distance = float(geometry.distance_m)
    dy, dx = (float(p) for p in geometry.input_pitch_m)
    dy_out, dx_out = geometry.output_pitch_m(height, width)

    in_chirp = _chirp(_centered_axis(height, dy), _centered_axis(width, dx), wavelength, distance)
    carrier = complex(math.cos(2 * math.pi * distance / wavelength), math.sin(2 * math.pi * distance / wavelength))
    scale = carrier / (1j * wavelength * distance) * math.sqrt(height * width) * dy * dx
    out_factor = scale * _chirp(_centered_axis(height, dy_out), _centered_axis(width, dx_out), wavelength, distance)

    to_field = dict(device=field.device, dtype=field.dtype)
    shifted = torch.fft.ifftshift(field * in_chirp.to(**to_field), dim=(-2, -1))
    spectrum = torch.fft.fftshift(torch.fft.fft2(shifted, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    out = spectrum * out_factor.to(**to_field)
    return FresnelField(field=out, output_pitch_m=(dy_out, dx_out), distance_m=distance)


def sampling_report(height: int, width: int, geometry: FresnelGeometry) -> dict[str, float]:
    """Discretisation diagnostics for one plane (no tensors; JSON-compatible floats).

    - ``input_chirp_step_rad``: largest phase change of exp(i*pi*(u^2+v^2)/(lambda*z))
      between adjacent input pixels. Must be < pi, otherwise the FFT input is aliased.
    - ``output_chirp_step_rad``: same for the output chirp. It only changes the output
      phase, so it does not affect intensities, but matters for complex-field use.
    - ``paraxial_phase_bound_rad``: pi*r^4/(4*lambda*z^3) with r the largest distance
      between an input and an output pixel; a conservative upper bound of the phase
      neglected by the Fresnel (paraxial) approximation, not an error estimate.
    """
    if not isinstance(geometry, FresnelGeometry):
        raise ValueError(f"geometry must be FresnelGeometry, got {type(geometry).__name__}")
    dy_out, dx_out = geometry.output_pitch_m(height, width)
    dy, dx = geometry.input_pitch_m
    lz = geometry.wavelength_m * geometry.distance_m

    def max_step(size: int, pitch: float) -> float:
        # |phi(n+1) - phi(n)| is largest at the most negative edge: (2*(size//2) - 1) * pitch^2.
        return math.pi * (2 * (size // 2) - 1) * pitch**2 / lz

    reach_y = (height // 2) * (dy + dy_out)
    reach_x = (width // 2) * (dx + dx_out)
    r = math.hypot(reach_y, reach_x)
    return {
        "distance_m": float(geometry.distance_m),
        "output_pitch_y_m": dy_out,
        "output_pitch_x_m": dx_out,
        "input_chirp_step_rad": max(max_step(height, dy), max_step(width, dx)),
        "output_chirp_step_rad": max(max_step(height, dy_out), max_step(width, dx_out)),
        "paraxial_phase_bound_rad": math.pi * r**4 / (4 * geometry.wavelength_m * geometry.distance_m**3),
    }
