"""Additive case-v1 interfaces. No propagator, solver, codec or metric implementation."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol
if TYPE_CHECKING:
    from torch import Tensor


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class FresnelGeometry:
    """Generic positive-z geometry; also supports tiny-grid independent tests."""
    wavelength_m: float
    input_pitch_m: tuple[float, float]
    distance_m: float

    def __post_init__(self) -> None:
        _positive(self.wavelength_m, "wavelength_m")
        _positive(self.distance_m, "distance_m")
        if not isinstance(self.input_pitch_m, tuple) or len(self.input_pitch_m) != 2:
            raise ValueError("input_pitch_m must be a (dy, dx) tuple")
        for value in self.input_pitch_m:
            _positive(value, "input_pitch_m")

    def output_pitch_m(self, height: int, width: int) -> tuple[float, float]:
        for size in (height, width):
            if type(size) is not int or size < 2:
                raise ValueError("height/width must be integers >= 2")
        dy, dx = self.input_pitch_m
        return (self.wavelength_m * self.distance_m / (height * dy),
                self.wavelength_m * self.distance_m / (width * dx))


@dataclass(frozen=True)
class CaseConfigV1:
    """Fixed official case, SI units. These constants cannot drift via constructor."""
    wavelength_m: float = field(default=532e-9, init=False)
    input_pitch_m: tuple[float, float] = field(default=(10e-6, 10e-6), init=False)
    distances_m: tuple[float, ...] = field(default=(24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3), init=False)
    image_shape: tuple[int, int] = field(default=(128, 128), init=False)

    def plane(self, index: int) -> FresnelGeometry:
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError("plane index must be 0..3 in the official distance order")
        return FresnelGeometry(self.wavelength_m, self.input_pitch_m, self.distances_m[index])


@dataclass
class FresnelField:
    """Complex field [B,1,H,W], output (dy,dx) in metres, distance in metres."""
    field: Tensor
    output_pitch_m: tuple[float, float]
    distance_m: float


class FresnelPropagator(Protocol):
    def __call__(self, field: Tensor, geometry: FresnelGeometry) -> FresnelField: ...


@dataclass
class MultiPlaneResult:
    """One phase [B,1,128,128]; continuous intensities [B,4,1,128,128]."""
    phase_rad: Tensor
    intensities: Tensor
    loss_history: list[float]
    elapsed_s: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairMetricsV1:
    """Undefined Pearson (constant input) is null, never a fabricated 0 or 1."""
    pearson: float | None
    ssim: float
    pearson_defined: bool


RECONSTRUCTION_POLICY_V1 = "per_plane_max_round_half_up_u8_v1"
SSIM_POLICY_V1 = "global_population_L1_K1_0.01_K2_0.03_v1"
