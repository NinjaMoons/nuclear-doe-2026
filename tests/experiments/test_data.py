from pathlib import Path
import struct
import zlib

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


@pytest.mark.parametrize("bit_depth", [2, 4])
def test_rejects_valid_packed_grayscale_png(tmp_path, bit_depth):
    paths = quartet(tmp_path)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    # Packed samples cycle through the full encoded range; each row uses filter0.
    per_byte = 8 // bit_depth
    mask = (1 << bit_depth) - 1
    samples = np.arange(128, dtype=np.uint8) & mask
    packed = bytes(sum(int(samples[start + offset]) << (8 - bit_depth * (offset + 1))
                       for offset in range(per_byte))
                   for start in range(0, 128, per_byte))
    header = struct.pack(">IIBBBBB", 128, 128, bit_depth, 0, 0, 0, 0)
    paths[1].write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress((b"\x00" + packed) * 128))
                         + chunk(b"IEND", b""))
    with Image.open(paths[1]) as image:
        assert image.format == "PNG" and image.mode == "L" and image.size == (128, 128)
        expected = np.tile(samples.astype(np.uint16) * (255 // mask), (128, 1)).astype(np.uint8)
        np.testing.assert_array_equal(np.array(image), expected)
    with pytest.raises(ValueError, match="8-bit grayscale"):
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
