"""Real neural export plumbing with an explicitly UNTRAINED, zero-update checkpoint.

This is not learned quality, training acceptance or an independent physics oracle.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from doe_core.contracts_v1 import CaseConfigV1
from doe_experiments.data import load_png_quartet
from doe_experiments.export import sha256
from doe_models.train_multiplane import infer_phase, load_checkpoint, save_checkpoint
from doe_models.unet_multiplane import build_model


def test_untrained_checkpoint_real_neural_cli_roundtrip(tmp_path):
    model = build_model(seed=17, base_channels=4, depth=1).to(dtype=torch.float64)
    checkpoint = tmp_path / "UNTRAINED-zero-updates.pt"
    checkpoint_hash = save_checkpoint(checkpoint, model, config=CaseConfigV1(),
        train_log={"steps": 0, "n_train": 0, "n_val": 0, "weights_delta_l2": 0.0,
                   "method": "UNTRAINED_TEST_FIXTURE", "observation": "amplitude"},
        code_commit="UNTRAINED-INTEGRATION-FIXTURE", extra={"synthetic_test": True})
    yy, xx = np.indices((128, 128))
    targets = []
    for i in range(4):
        path = tmp_path / f"synthetic_target_{i}.png"
        Image.fromarray(((xx * (i + 1) + yy * (5 - i)) % 256).astype(np.uint8)).save(path)
        targets.append(path)
    output = tmp_path / "neural-run"
    args = ["neural", "--checkpoint", str(checkpoint), "--targets", *map(str, targets),
            "--output-dir", str(output), "--synthetic-test"]
    process = subprocess.run([sys.executable, "-B", "-m", "doe_experiments.cli", *args],
                             cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["completed"] is True and manifest["synthetic_test"] is True
    metadata = manifest["metadata"]
    assert metadata["method"] == "fourinput_unet" and metadata["cli_method"] == "neural"
    assert metadata["checkpoint_sha256"] == checkpoint_hash == sha256(checkpoint)
    assert metadata["training"]["steps"] == 0 and metadata["training"]["weights_delta_l2"] == 0
    assert metadata["training_verified"] is False
    assert metadata["training_verification"] == "not_verified_by_inference_cli"
    assert metadata["code_commit"] == "UNTRAINED-INTEGRATION-FIXTURE"
    assert metadata["phase_dtype"] == "float64"
    assert metadata["checkpoint_load_elapsed_s"] >= 0 and metadata["inference_elapsed_s"] >= 0
    assert metadata["reproduce_argv"] == ["python", "-m", "doe_experiments.cli", *args]
    for relative, digest in manifest["files"].items():
        assert sha256(output / relative) == digest
    loaded_model, _ = load_checkpoint(checkpoint)
    expected_phase = infer_phase(loaded_model, load_png_quartet(targets).to(torch.float64))
    with np.load(output / "raw.npz", allow_pickle=False) as raw:
        assert raw["phase_continuous"].shape == (1, 1, 128, 128)
        assert raw["phase_continuous"].dtype == np.float64
        np.testing.assert_allclose(raw["phase_continuous"], expected_phase.numpy(), rtol=0, atol=0)
        assert raw["amplitude_quantized"].shape == (1, 4, 1, 128, 128)
    for name in ["DOE.png", *(f"reconstructed_{i:02d}.png" for i in range(1, 5))]:
        with Image.open(output / name) as image:
            assert image.mode == "L" and image.size == (128, 128)
    reconstructed = load_png_quartet([output / f"reconstructed_{i:02d}.png" for i in range(1, 5)])
    assert reconstructed.shape == (1, 4, 1, 128, 128)
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics["planes"]) == 4
    assert {plane["doe_sha256"] for plane in metrics["planes"]} == {sha256(output / "DOE.png")}
    assert all(set(("pearson", "ssim", "pearson_defined")) <= plane.keys() for plane in metrics["planes"])
