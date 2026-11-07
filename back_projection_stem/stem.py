"""RESEARCH ONLY (#5/6707710203, #5/6707752950): physics-informed 12-channel back-projection stem.

Not production: the official forward `doe_core.fresnel.propagate_fresnel`, `doe_models.unet_multiplane` and the
driver are imported read-only and never modified.

Exact inverse of the official single-FFT Fresnel forward (same grids, chirps, carrier and scale):
    F(U)    = o * fftshift(fft2(ifftshift(c_in * U), norm="ortho")),  o = s * c_out,
              s = exp(i 2 pi z / lambda) / (i lambda z) * sqrt(H W) * dy * dx
    F^-1(V) = fftshift(ifft2(ifftshift(V / o), norm="ortho")) / c_in
Features (heuristic, NOT a physical reconstruction): V_k = T_k + 0j on the native output grid of z_k (zero output phase
assumed), B_k = F_k^-1(V_k) / rms(B_k); channel order [T1..T4, Re1, Im1, Re2, Im2, Re3, Im3, Re4, Im4].
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from doe_core.contracts_v1 import CaseConfigV1, FresnelGeometry
from doe_core.fresnel import _centered_axis, _chirp  # read-only reuse of the official grid/chirp helpers
from doe_models.unet_multiplane import NUM_PLANES, FourInputUNet, _block

__all__ = ["inverse_fresnel", "backprojection_features", "BackProjectionStemUNet", "build_stem_model",
           "STEM_CHANNELS", "CHANNEL_ORDER"]

STEM_CHANNELS = 3 * NUM_PLANES
CHANNEL_ORDER = ["T1", "T2", "T3", "T4"] + [f"{p}{k}" for k in range(1, 5) for p in ("Re", "Im")]
_RMS_EPS = 1e-12


def _factors(height: int, width: int, geometry: FresnelGeometry):
    """(c_in, o) exactly as built inside propagate_fresnel, complex128 on CPU."""
    wavelength, distance = float(geometry.wavelength_m), float(geometry.distance_m)
    dy, dx = (float(p) for p in geometry.input_pitch_m)
    dy_out, dx_out = geometry.output_pitch_m(height, width)
    in_chirp = _chirp(_centered_axis(height, dy), _centered_axis(width, dx), wavelength, distance)
    carrier = complex(math.cos(2 * math.pi * distance / wavelength), math.sin(2 * math.pi * distance / wavelength))
    scale = carrier / (1j * wavelength * distance) * math.sqrt(height * width) * dy * dx
    out_factor = scale * _chirp(_centered_axis(height, dy_out), _centered_axis(width, dx_out), wavelength, distance)
    return in_chirp, out_factor


def inverse_fresnel(field: Tensor, geometry: FresnelGeometry) -> Tensor:
    """Exact inverse of propagate_fresnel for one distance: output-plane field [B,1,H,W] -> DOE-plane field."""
    if not isinstance(field, Tensor) or field.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("field must be a complex64/complex128 tensor")
    if field.ndim != 4 or field.shape[1] != 1 or min(field.shape[-2:]) < 2:
        raise ValueError("field must have shape [B,1,H,W], H,W >= 2")
    if not isinstance(geometry, FresnelGeometry):
        raise ValueError("geometry must be FresnelGeometry")
    h, w = field.shape[-2:]
    in_chirp, out_factor = _factors(h, w, geometry)
    to = dict(device=field.device, dtype=field.dtype)
    spectrum = torch.fft.ifftshift(field / out_factor.to(**to), dim=(-2, -1))
    shifted = torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(shifted, dim=(-2, -1)) / in_chirp.to(**to)


def backprojection_features(targets: Tensor, config: CaseConfigV1 | None = None) -> Tensor:
    """targets [B,4,1,H,W] (PNG/255 amplitudes) -> features [B,12,H,W] in the dtype of targets (computed in complex128)."""
    if not isinstance(targets, Tensor) or targets.dtype not in (torch.float32, torch.float64):
        raise ValueError("targets must be float32/float64")
    if targets.ndim != 5 or targets.shape[1] != NUM_PLANES or targets.shape[2] != 1:
        raise ValueError("targets must have shape [B,4,1,H,W]")
    config = config or CaseConfigV1()
    t = targets.detach().to(torch.float64)
    chans = [t[:, k, 0] * 2 - 1 for k in range(NUM_PLANES)]          # same affine as FourInputUNet
    for k in range(NUM_PLANES):
        v = torch.complex(t[:, k], torch.zeros_like(t[:, k]))        # V_k = T_k + 0j  (amplitude, no sqrt)
        b = inverse_fresnel(v, config.plane(k))[:, 0]               # [B,H,W] complex128
        rms = b.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        b = torch.where(rms > _RMS_EPS, b / torch.where(rms > _RMS_EPS, rms, torch.ones_like(rms)), torch.zeros_like(b))
        chans += [b.real, b.imag]
    return torch.stack(chans, dim=1).to(targets.dtype)


class BackProjectionStemUNet(FourInputUNet):
    """FourInputUNet whose first encoder takes the 12 stem channels; everything after it is unchanged."""

    def __init__(self, base_channels: int = 16, depth: int = 3, config: CaseConfigV1 | None = None) -> None:
        super().__init__(base_channels=base_channels, depth=depth)
        self.encoders[0] = _block(STEM_CHANNELS, self.encoders[0][0].out_channels)
        self._config = config or CaseConfigV1()

    @property
    def hparams(self) -> dict:
        hp = dict(super().hparams)
        hp.update({"in_channels": STEM_CHANNELS, "stem": "backprojection_v0", "channel_order": CHANNEL_ORDER})
        return hp

    def forward(self, targets: Tensor) -> Tensor:
        self._check(targets)
        x = backprojection_features(targets, self._config)
        return self._unet(x, targets.shape[-2:])

    def _unet(self, x: Tensor, hw) -> Tensor:
        # identical to FourInputUNet.forward after its input mapping (kept in sync by test_matches_parent_body)
        import torch.nn.functional as F
        from doe_models.unet_multiplane import _wrap
        h, w = hw
        m = 2 ** self.depth
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for red, dec in zip(self.reducers, self.decoders):
            skip = skips.pop()
            x = red(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            x = dec(torch.cat([x, skip], dim=1))
        return _wrap(self.head(x)[..., :h, :w])


def build_stem_model(*, seed: int, base_channels: int = 16, depth: int = 3) -> BackProjectionStemUNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BackProjectionStemUNet(base_channels=base_channels, depth=depth)
