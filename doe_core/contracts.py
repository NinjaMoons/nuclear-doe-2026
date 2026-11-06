"""Shared interfaces only; this module implements no propagation or synthesis."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch import Tensor


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")


@dataclass(frozen=True)
class OpticsConfig:
    """Lengths in metres; pixel_pitch_m is (dy, dx) in the medium."""

    wavelength_m: float
    pixel_pitch_m: tuple[float, float]
    distance_m: float
    pad_factor: int = 1

    def __post_init__(self) -> None:
        _positive_finite(self.wavelength_m, "wavelength_m")
        if not isinstance(self.pixel_pitch_m, tuple) or len(self.pixel_pitch_m) != 2:
            raise ValueError("pixel_pitch_m must be a (dy, dx) tuple")
        for value in self.pixel_pitch_m:
            _positive_finite(value, "pixel_pitch_m")
        if isinstance(self.distance_m, bool) or not isinstance(self.distance_m, Real):
            raise ValueError("distance_m must be finite and nonnegative")
        if not math.isfinite(self.distance_m) or self.distance_m < 0:
            raise ValueError("distance_m must be finite and nonnegative")
        if type(self.pad_factor) is not int or self.pad_factor < 1:
            raise ValueError("pad_factor must be a positive integer")


@dataclass
class SynthesisResult:
    """Tensor/result invariants are validated by the algorithm, per the contract."""

    phase_rad: Tensor
    intensity: Tensor
    loss_history: list[float]
    elapsed_s: float
    metadata: dict[str, Any]


class Propagator(Protocol):
    """Differentiable complex-field mapping injected into the Adam baseline."""

    def __call__(self, field: Tensor, config: OpticsConfig) -> Tensor: ...
