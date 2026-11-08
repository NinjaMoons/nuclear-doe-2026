"""Cross-module smoke, owned by the coordinator; no scientific implementation here.

Missing modules are an explicit bootstrap skip. DOE_REQUIRE_BASELINE=1 turns that
skip into a failure at the first integration checkpoint. Broken imports of an
existing module always fail; torch is deliberately imported only after readiness.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "doe_core/asm.py",
    "doe_models/adam.py",
    "doe_experiments/data.py",
    "doe_experiments/metrics.py",
)
MISSING = tuple(path for path in REQUIRED if not (ROOT / path).is_file())
REQUIRE_BASELINE = os.environ.get("DOE_REQUIRE_BASELINE") == "1"


@pytest.mark.integration
@pytest.mark.skipif(bool(MISSING) and not REQUIRE_BASELINE,
                    reason="baseline not implemented yet: " + ", ".join(MISSING))
def test_asm_adam_metrics_smoke():
    if MISSING:
        pytest.fail("integration checkpoint missing modules: " + ", ".join(MISSING))

    import torch
    from doe_core.asm import propagate, intensity
    from doe_core.contracts import OpticsConfig, SynthesisResult
    from doe_models.adam import synthesize_adam
    from doe_experiments.data import make_targets
    from doe_experiments.metrics import evaluate

    # Provisional engineering preset, not an organizer requirement. Periodic
    # pad_factor=1 is intentional; this smoke does not prove open-space sampling.
    config = OpticsConfig(wavelength_m=532e-9, pixel_pitch_m=(8e-6, 9e-6),
                          distance_m=0.01, pad_factor=1)
    target = make_targets(count=2, height=32, width=40, seed=11)
    assert target.shape == (2, 1, 32, 40)
    assert target.device.type == "cpu" and target.dtype == torch.float32
    assert torch.isfinite(target).all() and (target >= 0).all()
    torch.testing.assert_close(target.sum(dim=(-2, -1)), torch.ones((2, 1)))
    target = target.to(dtype=torch.float64)

    result = synthesize_adam(target, config, propagate_fn=propagate,
                             steps=32, lr=0.05, seed=17)
    assert isinstance(result, SynthesisResult)
    assert result.phase_rad.shape == result.intensity.shape == target.shape
    assert result.phase_rad.dtype == result.intensity.dtype == target.dtype
    assert result.phase_rad.device == result.intensity.device == target.device
    assert torch.isfinite(result.phase_rad).all()
    assert torch.isfinite(result.intensity).all() and (result.intensity >= 0).all()
    assert (result.phase_rad >= -math.pi).all() and (result.phase_rad < math.pi).all()
    assert len(result.loss_history) >= 2
    assert all(math.isfinite(value) for value in result.loss_history)
    assert math.isfinite(result.elapsed_s) and result.elapsed_s >= 0
    json.dumps(result.metadata, allow_nan=False)
    assert {"method", "seed", "config", "device", "dtype"} <= result.metadata.keys()

    field = torch.exp(1j * result.phase_rad) / math.sqrt(32 * 40)
    recomputed = intensity(propagate(field, config))
    torch.testing.assert_close(recomputed, result.intensity, rtol=1e-8, atol=1e-10)
    metrics = evaluate(result.intensity, target)
    assert set(metrics) == {"normalized_mse", "cosine_similarity", "roi_efficiency"}
    assert all(type(value) is float and math.isfinite(value) for value in metrics.values())
    assert -1e-10 <= metrics["cosine_similarity"] <= 1 + 1e-10
    assert -1e-10 <= metrics["roi_efficiency"] <= 1 + 1e-10
    assert metrics["normalized_mse"] < result.loss_history[0], "Adam must improve loss"
    assert math.isclose(metrics["normalized_mse"], min(result.loss_history),
                        rel_tol=1e-7, abs_tol=1e-12)
    print(json.dumps({"config": asdict(config), "shape": list(target.shape),
                      "initial_loss": result.loss_history[0], "metrics": metrics,
                      "elapsed_s": result.elapsed_s}, allow_nan=False))
