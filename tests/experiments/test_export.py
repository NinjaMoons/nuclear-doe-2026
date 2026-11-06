"""Orchestration tests use explicit doubles, never claim to validate Fresnel/metrics."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from doe_core.contracts_v1 import FresnelField, PairMetricsV1
from doe_experiments.export import ExportComponents, export_run, sha256
from doe_experiments.phase_codec import load_doe_png, save_doe_png


def fixture_components(calls):
    def propagate(field, geometry):
        calls.append(field.detach().clone())
        # Nonphysical test double; no FFT or alternative physics implementation.
        shift = int(round(geometry.distance_m * 1000))
        out = field + 0.3 * torch.roll(field, shifts=shift, dims=-1)
        return FresnelField(out, geometry.output_pitch_m(128, 128), geometry.distance_m)

    def load(paths):
        arrays = []
        for path in paths:
            with Image.open(path) as image:
                assert image.mode == "L" and image.size == (128, 128)
                arrays.append(np.array(image).copy())
        return torch.from_numpy(np.stack(arrays)[None, :, None].astype(np.float32) / 255)

    def convert(raw):
        array = raw.detach().cpu().numpy().astype(np.float64)
        return np.floor(255 * (array / array.max(axis=(-2, -1), keepdims=True)) + 0.5).astype(np.uint8)

    def save(codes, output_dir):
        paths = []
        for index in range(4):
            path = Path(output_dir) / f"reconstructed_{index + 1:02d}.png"
            Image.fromarray(codes[0, index, 0]).save(path)
            paths.append(path)
        return paths

    def evaluate(prediction, target):
        assert prediction.shape == target.shape == (1, 4, 1, 128, 128)
        return [[PairMetricsV1(0.1 * i, 0.5, True) for i in range(4)]]
    return ExportComponents(propagate, load, convert, save, evaluate)


def targets(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"synthetic_{i}.png"
        image = ((np.indices((128, 128))[0] * (i + 1) + np.indices((128, 128))[1]) % 256).astype(np.uint8)
        Image.fromarray(image).save(path)
        paths.append(path)
    return paths


def test_saved_doe_is_primary_source_and_hashes_reproducible(tmp_path):
    inputs = targets(tmp_path)
    phase = torch.linspace(-2.9, 3.0, 128 * 128, dtype=torch.float64).reshape(1, 1, 128, 128)
    calls = []
    components = fixture_components(calls)
    output = tmp_path / "run"
    manifest = export_run(inputs, phase, output, components=components,
                          metadata={"seed": 17}, synthetic_test=True)
    assert manifest["completed"] and manifest["synthetic_test"]
    assert manifest["observation"] == "amplitude" and manifest["case_contract"] == "1.1-amplitude"
    decoded = load_doe_png(output / "DOE.png")
    assert not torch.equal(decoded, phase)
    assert len(calls) == 8
    for field in calls[:4]:
        torch.testing.assert_close(field, torch.exp(1j * decoded), rtol=0, atol=0)
    for relative, digest in manifest["files"].items():
        assert sha256(output / relative) == digest
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics["planes"]) == 4
    assert len(metrics["decoded_phase_float_display_metrics"]) == 4
    assert len(metrics["continuous_phase_float_display_metrics"]) == 4
    assert metrics["quantization_delta"]["planes"] == [{"pearson": 0.0, "ssim": 0.0}] * 4
    assert {p["doe_sha256"] for p in metrics["planes"]} == {sha256(output / "DOE.png")}
    repeat = tmp_path / "repeat"
    export_run(inputs, decoded, repeat, components=components, synthetic_test=True)
    for i in range(4):
        assert sha256(output / f"reconstructed_{i+1:02d}.png") == sha256(repeat / f"reconstructed_{i+1:02d}.png")
    arrays = np.load(output / "raw.npz", allow_pickle=False)
    assert arrays["amplitude_quantized"].shape == (1, 4, 1, 128, 128)
    np.testing.assert_allclose(arrays["intensity_quantized"], arrays["amplitude_quantized"]**2)
    with pytest.raises(FileExistsError):
        export_run(inputs, phase, output, components=components)


def test_violating_evaluation_policy_rejected_without_success_manifest(tmp_path):
    inputs = targets(tmp_path)
    components = fixture_components([])
    components = replace(components, amplitude_to_u8=lambda raw: np.zeros(raw.shape, dtype=np.uint8))
    output = tmp_path / "bad_policy"
    with pytest.raises(ValueError, match="evaluation policy"):
        export_run(inputs, torch.zeros((1, 1, 128, 128)), output, components=components)
    assert not (output / "manifest.json").exists()


def test_no_skip_or_fake_metrics_when_component_is_broken(tmp_path):
    inputs = targets(tmp_path)
    components = replace(fixture_components([]), evaluate=lambda a, b: [[PairMetricsV1(None, 0.2, True)] * 4])
    with pytest.raises(ValueError, match="undefined Pearson"):
        export_run(inputs, torch.zeros((1, 1, 128, 128)), tmp_path / "bad_metrics", components=components)


def test_cli_reconstruct_explicit_precision_and_failure_without_modules(tmp_path, monkeypatch):
    from doe_experiments import cli
    inputs = targets(tmp_path)
    phase = torch.linspace(-3, 3, 128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    doe = save_doe_png(phase, tmp_path / "source_DOE.png")
    monkeypatch.setattr(cli, "production_components", lambda: fixture_components([]))
    output = tmp_path / "cli_run"
    assert cli.main(["reconstruct", "--targets", *map(str, inputs), "--doe", str(doe),
                     "--output-dir", str(output), "--dtype", "float32", "--synthetic-test"]) == 0
    raw = np.load(output / "raw.npz", allow_pickle=False)
    assert raw["phase_decoded"].dtype == np.float32
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metadata"]["phase_dtype"] == "float32"

    def missing():
        raise ModuleNotFoundError("real implementation missing")
    monkeypatch.setattr(cli, "production_components", missing)
    failed = tmp_path / "must_not_exist"
    assert cli.main(["adam", "--targets", *map(str, inputs), "--output-dir", str(failed)]) == 2
    assert not failed.exists()


def test_export_png_uses_amplitude_not_squared_observable(tmp_path):
    inputs = targets(tmp_path)
    gen = torch.Generator().manual_seed(81)
    phase = (torch.rand((1,1,128,128), generator=gen, dtype=torch.float64) * 2 - 1) * 3.14
    output = tmp_path / "amplitude_regression"
    export_run(inputs, phase, output, components=fixture_components([]), synthetic_test=True)
    raw = np.load(output / "raw.npz", allow_pickle=False)
    a, i = raw["amplitude_quantized"], raw["intensity_quantized"]
    expected_a = np.floor(255 * (a / a.max(axis=(-2,-1),keepdims=True)) + 0.5).astype(np.uint8)
    wrong_i = np.floor(255 * (i / i.max(axis=(-2,-1),keepdims=True)) + 0.5).astype(np.uint8)
    assert np.any(expected_a != wrong_i)
    for plane in range(4):
        with Image.open(output / f"reconstructed_{plane+1:02d}.png") as image:
            np.testing.assert_array_equal(np.array(image), expected_a[0,plane,0])


@pytest.mark.parametrize("maximum,pixel,code", [
    (7.779288333427416, 3.2184898791238914, 106),
    (1e308, 5e307, 128),
])
def test_export_conversion_uses_divide_first_float64(maximum, pixel, code):
    from doe_experiments.export import _display
    raw = torch.full((1,4,1,128,128), maximum, dtype=torch.float64)
    raw[...,0,0] = pixel
    expected = np.full(raw.shape, 255, dtype=np.uint8)
    expected[...,0,0] = code
    codes, maxima, normalized = _display(raw, lambda _: expected)
    np.testing.assert_array_equal(codes, expected)
    assert maxima == [maximum] * 4
    assert torch.isfinite(normalized).all()
