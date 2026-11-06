"""Compact four-input U-Net: four target amplitudes -> ONE DOE phase (contract v1.1, Issue #5).

Input targets [B,4,1,H,W] (PNG/255 amplitudes in [0,1]); the four planes become four
input channels, the batch axis is never mixed with planes (GroupNorm, no BatchNorm).
Output: one phase [B,1,H,W] in [-pi,pi). The head is a single unconstrained channel
followed by wrap remainder(phi+pi, 2pi)-pi: the physics loss depends only on
exp(i*phi), wrap has gradient 1 almost everywhere, and there is no atan2 singularity.
Arbitrary H,W >= 2 are supported by replicate padding to a multiple of 2**depth and
cropping back.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["FourInputUNet", "build_model"]

NUM_PLANES = 4
_GROUPS = 4


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(_GROUPS, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(_GROUPS, cout), nn.SiLU(),
    )


def _wrap(phase: torch.Tensor) -> torch.Tensor:
    wrapped = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
    return torch.where(wrapped >= math.pi, wrapped - 2 * math.pi, wrapped)


class FourInputUNet(nn.Module):
    """targets [B,4,1,H,W] -> phase [B,1,H,W] in radians, [-pi, pi)."""

    def __init__(self, base_channels: int = 16, depth: int = 3) -> None:
        super().__init__()
        if type(base_channels) is not int or base_channels < _GROUPS or base_channels % _GROUPS:
            raise ValueError(f"base_channels must be an int >= {_GROUPS} divisible by {_GROUPS}")
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be an int >= 1")
        self.base_channels, self.depth = base_channels, depth
        widths = [base_channels * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList([_block(NUM_PLANES, widths[0])] +
                                      [_block(widths[i], widths[i + 1]) for i in range(depth)])
        self.reducers = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 1) for i in reversed(range(depth))])
        self.decoders = nn.ModuleList([_block(2 * widths[i], widths[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(widths[0], 1, 1)

    @property
    def hparams(self) -> dict:
        return {"base_channels": self.base_channels, "depth": self.depth,
                "in_planes": NUM_PLANES, "head": "wrapped_linear", "norm": "groupnorm4"}

    @staticmethod
    def _check(targets) -> None:
        if not isinstance(targets, torch.Tensor) or targets.dtype not in (torch.float32, torch.float64):
            raise ValueError("targets must be a float32/float64 tensor")
        if targets.ndim != 5 or targets.shape[1] != NUM_PLANES or targets.shape[2] != 1 \
                or targets.shape[0] < 1 or min(targets.shape[-2:]) < 2:
            raise ValueError("targets must have shape [B,4,1,H,W] with B>=1, H,W>=2")
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("targets contain NaN/Inf")
        if bool((targets < 0).any()) or bool((targets > 1).any()):
            raise ValueError("targets are PNG/255 amplitudes and must lie in [0,1]; not clipped silently")

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        self._check(targets)
        h, w = targets.shape[-2:]
        x = targets[:, :, 0] * 2 - 1                        # [B,4,H,W], planes as channels
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
        phase = self.head(x)[..., :h, :w]
        return _wrap(phase)


def build_model(*, seed: int, base_channels: int = 16, depth: int = 3) -> FourInputUNet:
    """Seed-local construction: weights depend only on `seed`; global RNG is restored."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an int >= 0")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FourInputUNet(base_channels=base_channels, depth=depth)
