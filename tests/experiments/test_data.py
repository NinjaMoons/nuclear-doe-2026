from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from doe_experiments.data import load_png_quartet


def quartet(tmp_path):
    paths = []
    for name, value in zip(("z", "a", "y", "b"), (17, 64, 128, 255)):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((128, 128), value, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_order_cpu_shape_and_amplitude_without_sqrt(tmp_path):
    paths = quartet(tmp_path)
    result = load_png_quartet(paths)
    assert result.shape == (1, 4, 1, 128, 128)
    assert result.device.type == "cpu" and result.dtype == torch.float32
    torch.testing.assert_close(result[0, :, 0, 0, 0], torch.tensor([17, 64, 128, 255]) / 255)
    torch.testing.assert_close(load_png_quartet(paths[::-1]), result.flip(1))


@pytest.mark.parametrize("mode,size,format", [
    ("RGB", (128, 128), "PNG"), ("P", (128, 128), "PNG"),
    ("I;16", (128, 128), "PNG"), ("1", (128, 128), "PNG"),
    ("L", (127, 128), "PNG"), ("L", (128, 128), "BMP"),
])
def test_rejects_noncontract_images(tmp_path, mode, size, format):
    paths = quartet(tmp_path)
    Image.new(mode, size).save(paths[1], format=format)
    with pytest.raises(ValueError):
        load_png_quartet(paths)


@pytest.mark.parametrize("value", [[], [Path("x")] * 3, [Path("x")] * 5, "abcd", None, [None] * 4, [12] * 4])
def test_rejects_nonquartet(value):
    with pytest.raises(ValueError):
        load_png_quartet(value)


def test_missing_and_corrupt_png_are_value_errors(tmp_path):
    paths = quartet(tmp_path)
    paths[0].write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_png_quartet(paths)
    paths[0].unlink()
    with pytest.raises(ValueError):
        load_png_quartet(paths)
