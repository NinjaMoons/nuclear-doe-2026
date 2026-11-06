"""Exact 8-bit DOE codec. Export is intentionally outside the autograd path."""
from pathlib import Path
import math

import numpy as np
from PIL import Image
import torch


def encode_phase(phase_rad: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(phase_rad, torch.Tensor):
        if phase_rad.dtype not in (torch.float32, torch.float64):
            raise ValueError("phase must be float32 or float64")
        phase_rad = phase_rad.detach().cpu().numpy()
    if not isinstance(phase_rad, np.ndarray) or phase_rad.dtype not in (np.float32, np.float64):
        raise ValueError("phase must be a float32/64 array or tensor")
    if not phase_rad.size or not np.isfinite(phase_rad).all():
        raise ValueError("phase must be nonempty and finite")
    tolerance = 8 * np.finfo(phase_rad.dtype).eps
    if np.any(phase_rad < -math.pi - tolerance) or np.any(phase_rad > math.pi + tolerance):
        raise ValueError("phase must be in [-pi,+pi]")
    phase = np.clip(phase_rad.astype(np.float64), -math.pi, math.pi)
    return np.floor(255 * (phase + math.pi) / (2 * math.pi) + 0.5).astype(np.uint8)


def decode_phase(codes: np.ndarray) -> np.ndarray:
    if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or not codes.size:
        raise ValueError("DOE codes must be a nonempty uint8 array")
    return -math.pi + (2 * math.pi / 255) * codes.astype(np.float64)


def save_doe_png(phase_rad: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    codes = encode_phase(phase_rad)
    if codes.shape != (1, 1, 128, 128):
        raise ValueError("one DOE requires phase shape [1,1,128,128]")
    path = Path(path)
    with path.open("xb") as stream:
        Image.fromarray(codes[0, 0]).save(stream, format="PNG")
    return path


def load_doe_png(path: str | Path, *, dtype: torch.dtype = torch.float64,
                 device: str | torch.device = "cpu") -> torch.Tensor:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("decoded phase dtype must be float32/64")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
            raise ValueError("DOE must be PNG, grayscale L, uint8, 128x128")
        codes = np.array(image)
    if codes.dtype != np.uint8:
        raise ValueError("DOE must be 8-bit")
    return torch.from_numpy(decode_phase(codes)[None, None]).to(device=device, dtype=dtype)
