"""Explicit field measurements; amplitude is primary, intensity is diagnostic."""
from typing import NamedTuple
import torch


class FieldObservables(NamedTuple):
    amplitude: torch.Tensor
    intensity: torch.Tensor


def field_observables(field: torch.Tensor) -> FieldObservables:
    if not isinstance(field, torch.Tensor) or field.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("field must be a complex64/128 torch.Tensor")
    if field.numel() == 0 or not torch.isfinite(field).all():
        raise ValueError("field must be nonempty and finite")
    amplitude = field.abs()  # PyTorch defines a finite zero subgradient at U=0.
    return FieldObservables(amplitude, amplitude.square())
