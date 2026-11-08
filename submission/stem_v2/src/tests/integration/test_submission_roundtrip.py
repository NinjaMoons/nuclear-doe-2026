"""Mandatory DATA roundtrip on real modules; this is NOT a physics oracle.

Independent scientific acceptance additionally requires test_fresnel_reference.py.
Missing implementations FAIL; there are no stubs/skips.
"""
import importlib
import json
import math
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.mark.integration
def test_submission_roundtrip(tmp_path):
    names = ("doe_core.fresnel", "doe_models.adam_multiplane", "doe_experiments.data",
             "doe_experiments.png_io", "doe_experiments.metrics")
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise  # Broken dependencies inside an existing module are real errors.
            missing.append(name)
    if missing:
        pytest.fail("v1 gate missing real modules (not skipped): " + ", ".join(missing))

    from doe_core.contracts_v1 import CaseConfigV1
    from doe_core.fresnel import propagate_fresnel
    from doe_experiments.phase_codec import load_doe_png
    from doe_experiments.export import sha256

    inputs = []
    y, x = np.indices((128, 128))
    for index in range(4):
        # Explicit test-only patterns; these are not the organizer's four objects.
        codes = ((x * (index + 1) + y * (5 - index)) % 256).astype(np.uint8)
        path = tmp_path / f"synthetic_test_target_{index+1}.png"
        Image.fromarray(codes).save(path)
        inputs.append(path)
    output = tmp_path / "adam"
    result = subprocess.run([sys.executable, "-m", "doe_experiments.cli", "adam", "--targets",
        *map(str, inputs), "--output-dir", str(output), "--steps", "16", "--seed", "17",
        "--synthetic-test"], text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert manifest["completed"] and manifest["synthetic_test"]
    assert manifest["case_contract"] == "1.1-amplitude" and manifest["observation"] == "amplitude"
    assert metrics["observation"] == "amplitude"
    history = manifest["metadata"]["loss_history"]
    assert len(history) >= 2 and all(math.isfinite(value) for value in history)
    assert min(history[1:]) < history[0], "real shared-phase Adam must improve its initial amplitude loss"
    assert manifest["config"]["distances_m"] == [0.0241, 0.0253, 0.0265, 0.0277]
    assert len(metrics["planes"]) == 4
    phase = load_doe_png(output / "DOE.png", dtype=torch.float32)
    config = CaseConfigV1()
    for index, pair in enumerate(metrics["planes"]):
        with Image.open(output / pair["reconstruction"]) as image:
            assert image.mode == "L" and image.size == (128, 128) and image.format == "PNG"
            restored = np.array(image)
        # Same production forward verifies dataflow/PNG, not correctness of physics.
        field = propagate_fresnel(torch.exp(1j * phase), config.plane(index)).field
        raw = field.abs().detach().cpu().numpy().astype(np.float64)
        expected = np.floor(255 * (raw / raw.max()) + 0.5).astype(np.uint8)[0, 0]
        np.testing.assert_array_equal(restored, expected)
        assert pair["doe_sha256"] == sha256(output / "DOE.png")
        with Image.open(inputs[index]) as image:
            a = np.array(image).astype(np.float64) / 255
        b = restored.astype(np.float64) / 255
        ac, bc = a-a.mean(), b-b.mean()
        denominator = np.sqrt(np.sum(ac*ac) * np.sum(bc*bc))
        if denominator == 0:
            assert pair["pearson"] is None and not pair["pearson_defined"]
        else:
            assert math.isclose(pair["pearson"], float(np.sum(ac*bc) / denominator), abs_tol=2e-7)
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*a.mean()*b.mean()+c1)*(2*np.mean(ac*bc)+c2) /
                ((a.mean()**2+b.mean()**2+c1)*(np.mean(ac*ac)+np.mean(bc*bc)+c2)))
        assert math.isclose(pair["ssim"], float(ssim), abs_tol=2e-7)
    repeated = tmp_path / "reconstructed_again"
    result = subprocess.run([sys.executable, "-m", "doe_experiments.cli", "reconstruct", "--doe",
        str(output / "DOE.png"), "--targets", *map(str, inputs), "--output-dir", str(repeated),
        "--synthetic-test"], text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    repeated_metrics = json.loads((repeated / "metrics.json").read_text())
    for first, second in zip(metrics["planes"], repeated_metrics["planes"]):
        assert sha256(output / first["reconstruction"]) == sha256(repeated / second["reconstruction"])
