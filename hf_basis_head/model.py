"""RESEARCH ONLY (R12, #39): residual basis phase head on top of the 12-channel back-projection STEM.

phi_final = wrap(phi_STEM + dphi),  dphi(x,y) = sum_{j=1..16} a_j(x,y) * R_j(x,y)
- phi_STEM: the unchanged scalar 1x1 head of BackProjectionStemUNet (same body, same init for the same seed).
- a_j: coefficient head = 1x1 conv on the last decoder feature map, average-pooled to a 32x32 grid, then bilinearly
  upsampled (align_corners=False) to HxW. Weight and bias start at zero -> dphi = 0 at step 0.
- R_j: fixed, non-trainable buffer [16,H,W] (saved in state_dict):
    basis="hf":    uniform[-pi,pi) from torch.Generator(cpu).manual_seed(2026), float64 -> float32 (white / high-frequency);
    basis="const": all ones (capacity control: identical coefficient path and parameter count, no HF content).
Interface unchanged: targets [B,4,1,128,128] -> phase [B,1,128,128] in [-pi,pi). Production code is not modified.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

from back_projection_stem.stem import BackProjectionStemUNet
from doe_models.unet_multiplane import _wrap

__all__ = ["HFBasisStemUNet", "build_hf_stem_model", "make_basis", "N_BASIS", "COEF_GRID", "BASIS_SEED", "PHASE_HEAD"]

N_BASIS, COEF_GRID, BASIS_SEED, SIZE = 16, 32, 2026, 128
PHASE_HEAD = "stem_residual_basis_v0"
BASIS_NAMES = {"hf": "hf_uniform_pm_pi_seed2026", "const": "constant_ones"}


def make_basis(kind: str, n: int = N_BASIS, size: int = SIZE) -> torch.Tensor:
    if kind == "hf":
        g = torch.Generator(device="cpu").manual_seed(BASIS_SEED)
        return ((torch.rand((n, size, size), generator=g, dtype=torch.float64) * 2 - 1) * math.pi).float()
    if kind == "const":
        return torch.ones((n, size, size), dtype=torch.float32)
    raise ValueError(f"basis must be one of {tuple(BASIS_NAMES)}")


def basis_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class HFBasisStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels: int = 16, depth: int = 3, config=None, basis: str = "hf") -> None:
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        width = self.head.in_channels
        self.coef_head = nn.Conv2d(width, N_BASIS, 1)
        with torch.no_grad():
            self.coef_head.weight.zero_()
            self.coef_head.bias.zero_()
        self.basis_kind = basis
        self.register_buffer("basis_maps", make_basis(basis))

    @property
    def hparams(self) -> dict:
        hp = dict(super().hparams)
        hp.update({"phase_head": PHASE_HEAD, "basis_kind": self.basis_kind, "basis": BASIS_NAMES[self.basis_kind],
                   "n_basis": N_BASIS, "coef_grid": COEF_GRID, "coef_upsample": "bilinear_align_corners_false",
                   "basis_sha256": basis_sha256(make_basis(self.basis_kind))})
        return hp

    def coefficients(self, feat: torch.Tensor, hw) -> torch.Tensor:
        """a_j on the 32x32 grid, [B,16,32,32] (pooling first, then the linear 1x1 head)."""
        h, w = hw
        return self.coef_head(F.adaptive_avg_pool2d(feat[..., :h, :w], (COEF_GRID, COEF_GRID)))

    def delta_phase(self, feat: torch.Tensor, hw) -> torch.Tensor:
        h, w = hw
        if (h, w) != tuple(self.basis_maps.shape[-2:]):
            raise ValueError(f"basis maps are {tuple(self.basis_maps.shape[-2:])}, input is {(h, w)}; no silent resize")
        a = F.interpolate(self.coefficients(feat, hw), size=(h, w), mode="bilinear", align_corners=False)
        return (a * self.basis_maps.to(dtype=a.dtype)[None]).sum(dim=1, keepdim=True)

    def _unet(self, x, hw):
        # identical to BackProjectionStemUNet._unet up to the head; the head output gets the residual basis term
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
        return _wrap(self.head(x)[..., :h, :w] + self.delta_phase(x, hw))


def build_hf_stem_model(*, seed: int, base_channels: int = 16, depth: int = 3, basis: str = "hf") -> HFBasisStemUNet:
    """Same seed-local construction as build_stem_model: body/stem init identical, extra head created last."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return HFBasisStemUNet(base_channels=base_channels, depth=depth, basis=basis)
