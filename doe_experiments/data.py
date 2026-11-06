"""Strict, ordered loading of the four amplitude target PNGs."""
from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_png_quartet(paths: Sequence[Path]) -> torch.Tensor:
    """Return CPU float32 [1,4,1,128,128], with PNG values divided by 255."""
    if isinstance(paths, (str, bytes, Path)) or not isinstance(paths, Sequence) or len(paths) != 4:
        raise ValueError("exactly four ordered PNG paths are required")
    planes = []
    for path in paths:
        if not isinstance(path, (str, PathLike)):
            raise ValueError("each target path must be a filesystem path")
        try:
            # Pillow expands packed 2/4-bit grayscale into mode L uint8; inspect
            # the encoded IHDR so those files cannot masquerade as 8-bit input.
            with open(path, "rb") as stream:
                header = stream.read(33)
            if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
                    or header[8:12] != b"\x00\x00\x00\x0d" or header[12:16] != b"IHDR"
                    or header[24] != 8 or header[25] != 0):
                raise ValueError("each target must encode 8-bit grayscale PNG pixels")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
                    raise ValueError("each target must be a mode L uint8 128x128 PNG")
                values = np.array(image)
                if values.dtype != np.uint8:
                    raise ValueError("each target must contain uint8 pixels")
                planes.append(torch.from_numpy(values).to(torch.float32).div_(255))
        except (OSError, TypeError) as exc:
            raise ValueError("cannot read target PNG") from exc
    return torch.stack(planes).unsqueeze(0).unsqueeze(2)
