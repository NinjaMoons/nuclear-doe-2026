"""CPU acceptance of the exact, checked-in STEM V2 bundle."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from back_projection_stem.infer_neural_stem import main
from back_projection_stem.stem import BackProjectionStemUNet


PACKAGE = Path(__file__).resolve().parents[3]
DISTANCES_M = [0.0241, 0.0253, 0.0265, 0.0277]
TARGET_NAMES = ("z1_horizontal_bar.png", "z2_vertical_bar.png",
                "z3_diagonal_line.png", "z4_ring.png")


def _amplitude_png(path):
    with Image.open(path) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))
        return np.asarray(image, dtype=np.float64).ravel() / 255.0


def _reference_metrics(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    pearson = np.corrcoef(x, y)[0, 1]
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * x.mean() * y.mean() + c1) * (2 * np.mean(dx * dy) + c2)
            / ((x.mean() ** 2 + y.mean() ** 2 + c1) * (np.mean(dx**2 + dy**2) + c2)))
    return [pearson, ssim]


@pytest.mark.integration
def test_stem_v2_release(tmp_path, monkeypatch):
    checkpoint = PACKAGE / "model/stem_native_s0_step_012000.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        "19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f"
    )
    reference = json.loads((PACKAGE / "example_output/metrics.json").read_text(encoding="utf-8"))
    output = tmp_path / "stem"
    targets = [PACKAGE / "examples/synthetic_5F" / name for name in TARGET_NAMES]
    args = ["--checkpoint", str(checkpoint), "--targets", *(str(p) for p in targets),
            "--output-dir", str(output), "--synthetic-test"]
    forward = BackProjectionStemUNet.forward
    calls = []

    def counted_forward(self, x):
        assert not torch.is_grad_enabled()
        calls.append(x.shape)
        return forward(self, x)

    def forbidden_optimizer(*args, **kwargs):
        pytest.fail("The STEM inference path must not construct an optimizer")

    monkeypatch.setattr(BackProjectionStemUNet, "forward", counted_forward)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden_optimizer)
    assert main(args) == 0
    assert calls == [(1, 4, 1, 128, 128)]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] and manifest["synthetic_test"]
    assert manifest["observation"] == "amplitude"
    assert manifest["config"] == reference["config"]
    assert manifest["config"]["distances_m"] == DISTANCES_M
    assert manifest["output_pitches_m"] == reference["output_pitches_m"]
    assert manifest["metadata"]["reproduce_argv"] == [
        "python", "-m", "back_projection_stem.infer_neural_stem", *args,
    ]

    doe = output / "DOE.png"
    _amplitude_png(doe)
    doe_sha = hashlib.sha256(doe.read_bytes()).hexdigest()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["primary_source"] == "reloaded_uint8_PNG_from_reloaded_DOE"
    assert len(metrics["planes"]) == 4
    for i, (actual, expected, target) in enumerate(zip(metrics["planes"], reference["planes"], targets)):
        reconstruction = f"reconstructed_{i + 1:02d}.png"
        assert actual["index"] == i and actual["distance_m"] == DISTANCES_M[i]
        assert actual["doe_sha256"] == doe_sha and actual["reconstruction"] == reconstruction
        pair = [actual["pearson"], actual["ssim"]]
        assert pair == pytest.approx(
            _reference_metrics(_amplitude_png(output / reconstruction), _amplitude_png(target)), abs=2e-7,
        )
        assert pair == pytest.approx([expected["pearson"], expected["ssim"]], abs=1e-4)
