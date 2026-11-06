import numpy as np
from PIL import Image
import pytest
import torch

from doe_experiments.data import load_png_quartet
from doe_experiments.png_io import amplitude_to_u8, save_reconstructions


def test_amplitude_regression_half_up_scaling_and_nonmutation():
    raw = torch.full((2, 4, 1, 128, 128), 2.0, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        raw[..., 0, 0] = 1
        raw[..., 0, 1] = 1 / 255  # 0.5 code rounds up
    before = raw.detach().clone()
    codes = amplitude_to_u8(raw)
    assert codes.dtype == np.uint8 and codes.shape == tuple(raw.shape)
    assert np.all(codes[..., 0, 0] == 128)  # squaring would yield 64
    assert np.all(codes[..., 0, 1] == 1)
    assert np.all(codes[..., 1, 0] == 255)
    np.testing.assert_array_equal(codes, amplitude_to_u8(raw * 8))
    torch.testing.assert_close(raw, before)


def test_each_batch_and_plane_has_its_own_max():
    raw = torch.zeros((2, 4, 1, 128, 128), dtype=torch.float64)
    for b in range(2):
        for p in range(4):
            raw[b, p, 0, 0, 0] = (b + 1) * (p + 2)
            raw[b, p, 0, 0, 1] = (b + 1) * (p + 2) * (p + 1) / 8
    codes = amplitude_to_u8(raw)
    assert np.all(codes[..., 0, 0] == 255)
    np.testing.assert_array_equal(codes[:, :, 0, 0, 1], [[32, 64, 96, 128]] * 2)


@pytest.mark.parametrize("kind", ["none", "dtype", "shape", "empty", "negative", "nan", "inf", "zero_plane", "tiny_plane"])
def test_invalid_amplitudes(kind):
    raw = torch.ones((1, 4, 1, 128, 128))
    if kind == "none": raw = None
    elif kind == "dtype": raw = raw.to(torch.int64)
    elif kind == "shape": raw = raw.squeeze(0)
    elif kind == "empty": raw = raw[:0]
    else:
        raw[0, 2] = {"negative": -1, "nan": float("nan"), "inf": float("inf"), "zero_plane": 0, "tiny_plane": 1e-13}[kind]
    with pytest.raises(ValueError):
        amplitude_to_u8(raw)


def test_ordered_mode_l_exact_disk_roundtrip_and_no_overwrite(tmp_path):
    codes = np.zeros((1, 4, 1, 128, 128), dtype=np.uint8)
    for p in range(4):
        codes[0, p, 0] = (np.arange(128 * 128).reshape(128, 128) + 31 * p) % 256
    paths = save_reconstructions(codes, tmp_path / "run")
    assert len(set(paths)) == 4
    for p, path in enumerate(paths):
        assert path.parent == tmp_path / "run"
        with Image.open(path) as image:
            assert image.mode == "L" and image.size == (128, 128)
            np.testing.assert_array_equal(np.array(image), codes[0, p, 0])
    np.testing.assert_array_equal(torch.round(load_png_quartet(paths) * 255).numpy().astype(np.uint8), codes)
    original = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError):
        save_reconstructions(codes, tmp_path / "run")
    assert [p.read_bytes() for p in paths] == original


def test_preflight_prevents_partial_write(tmp_path):
    existing = tmp_path / "reconstructed_04.png"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError):
        save_reconstructions(np.zeros((1, 4, 1, 128, 128), dtype=np.uint8), tmp_path)
    assert sorted(tmp_path.iterdir()) == [existing]
    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize("codes", [None, np.zeros((2, 4, 1, 128, 128), dtype=np.uint8), np.zeros((1, 4, 128, 128), dtype=np.uint8), np.zeros((1, 4, 1, 128, 128))])
def test_invalid_save_inputs(tmp_path, codes):
    with pytest.raises(ValueError):
        save_reconstructions(codes, tmp_path / "run")
    assert not (tmp_path / "run").exists()
