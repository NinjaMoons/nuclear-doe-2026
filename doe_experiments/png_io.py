"""Amplitude display conversion and lossless reconstruction PNG output."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch


def amplitude_to_u8(amplitudes: torch.Tensor) -> np.ndarray:
    """Normalize each raw amplitude plane by its own max, then round half up."""
    if (not isinstance(amplitudes, torch.Tensor) or amplitudes.ndim != 5
            or amplitudes.shape[0] < 1 or tuple(amplitudes.shape[1:]) != (4, 1, 128, 128)
            or amplitudes.dtype not in (torch.float32, torch.float64)):
        raise ValueError("amplitudes must be float32/64 [B,4,1,128,128], B >= 1")
    values = amplitudes.detach().to(device="cpu", dtype=torch.float64).numpy()
    maxima = values.max(axis=(-2, -1), keepdims=True)
    if not np.isfinite(values).all() or (values < 0).any() or (maxima <= 1e-12).any():
        raise ValueError("amplitudes must be finite, nonnegative, with each plane max > 1e-12")
    return np.floor(255 * (values / maxima) + 0.5).astype(np.uint8)


def save_reconstructions(images_u8: np.ndarray, output_dir: Path) -> list[Path]:
    """Save one quartet without overwriting any existing reconstruction."""
    if (not isinstance(images_u8, np.ndarray) or images_u8.dtype != np.uint8
            or images_u8.shape != (1, 4, 1, 128, 128)):
        raise ValueError("images_u8 must be uint8 [1,4,1,128,128]")
    try:
        output_dir = Path(output_dir)
        paths = [output_dir / f"reconstructed_{index:02d}.png" for index in range(1, 5)]
        if any(path.exists() or path.is_symlink() for path in paths):
            raise ValueError("reconstruction path already exists")
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths):
            with path.open("xb") as stream:
                Image.fromarray(images_u8[0, index, 0]).save(stream, format="PNG")
    except (OSError, TypeError) as exc:
        raise ValueError("cannot save reconstruction PNGs without overwriting") from exc
    return paths
