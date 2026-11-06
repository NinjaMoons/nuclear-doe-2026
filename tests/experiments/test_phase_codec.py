import math

import numpy as np
from PIL import Image
import pytest
import torch

from doe_experiments.phase_codec import encode_phase, decode_phase, save_doe_png, load_doe_png


def test_exact_endpoints_and_all_codes_roundtrip():
    codes = np.arange(256, dtype=np.uint8)
    phases = decode_phase(codes)
    assert phases[0] == -math.pi and phases[-1] == math.pi
    np.testing.assert_array_equal(encode_phase(phases), codes)
    assert not math.isclose(phases[1], -math.pi + 2 * math.pi / 256, abs_tol=1e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_circular_error_at_most_half_step(dtype):
    phases = np.linspace(-math.pi, math.pi, 20001, dtype=dtype)
    decoded = decode_phase(encode_phase(phases))
    error = np.angle(np.exp(1j * (decoded - phases)))
    assert np.max(np.abs(error)) <= math.pi / 255 + 2e-7


def test_phase_png_really_reloaded_and_original_not_mutated(tmp_path):
    phase = torch.linspace(-math.pi, math.pi, 128 * 128, dtype=torch.float64).reshape(1, 1, 128, 128)
    phase.requires_grad_()
    before = phase.detach().clone()
    path = save_doe_png(phase, tmp_path / "DOE.png")
    with Image.open(path) as image:
        assert image.mode == "L" and image.format == "PNG" and image.size == (128, 128)
        assert np.array(image).dtype == np.uint8
    decoded = load_doe_png(path)
    assert decoded.shape == phase.shape and decoded.dtype == torch.float64
    assert torch.equal(before, phase)
    replacement = np.zeros((128, 128), dtype=np.uint8)
    Image.fromarray(replacement).save(path)
    assert torch.all(load_doe_png(path) == -math.pi)


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([4.]), np.array([0], dtype=np.int32)])
def test_bad_phase_rejected(bad):
    with pytest.raises(ValueError):
        encode_phase(bad)


@pytest.mark.parametrize("mode", ["RGB", "I;16"])
def test_wrong_doe_format_rejected(tmp_path, mode):
    dtype, shape = (np.uint8, (128, 128, 3)) if mode == "RGB" else (np.uint16, (128, 128))
    path = tmp_path / "bad.png"
    Image.fromarray(np.zeros(shape, dtype=dtype)).save(path)
    with pytest.raises(ValueError):
        load_doe_png(path)


def test_wrong_shape_and_existing_doe_not_overwritten(tmp_path):
    with pytest.raises(ValueError):
        save_doe_png(torch.zeros((128, 128)), tmp_path / "bad.png")
    path = save_doe_png(torch.zeros((1, 1, 128, 128)), tmp_path / "DOE.png")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_doe_png(torch.ones((1, 1, 128, 128)), path)
    assert path.read_bytes() == original
