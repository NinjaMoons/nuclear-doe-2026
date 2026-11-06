import math

import numpy as np
import pytest
import torch

from doe_experiments.metrics import evaluate_multiplane


def scalar_reference(x, y):
    # Independent scalar population moments (no production helpers).
    x, y = x.ravel().tolist(), y.ravel().tolist()
    n = len(x)
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    vx = math.fsum((v - mx)**2 for v in x) / n
    vy = math.fsum((v - my)**2 for v in y) / n
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    pearson = cov / math.sqrt(vx * vy) if vx and vy else None
    ssim = ((2 * mx * my + 0.0001) / (mx**2 + my**2 + 0.0001)
            * (2 * cov + 0.0009) / (vx + vy + 0.0009))
    return pearson, ssim


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_planes_and_batches_against_independent_scalar_formula(dtype):
    rng = np.random.default_rng(117)
    x = torch.tensor(rng.uniform(size=(2, 4, 1, 128, 128)), dtype=dtype)
    y = x.clone()
    y[:, 1] = 1 - x[:, 1]
    y[:, 2] = x[:, 2] * 0.3 + 0.2
    y[:, 3] = torch.tensor(rng.uniform(size=(2, 1, 128, 128)), dtype=dtype)
    before_x, before_y = x.clone(), y.clone()
    actual = evaluate_multiplane(x.requires_grad_(), y)
    assert len(actual) == 2 and all(len(row) == 4 for row in actual)
    for b in range(2):
        scores = []
        for p in range(4):
            pearson, ssim = scalar_reference(x[b, p].detach().numpy(), y[b, p].numpy())
            item = actual[b][p]
            assert item.pearson_defined is True
            assert item.pearson == pytest.approx(pearson, abs=2e-14)
            assert item.ssim == pytest.approx(ssim, abs=2e-14)
            scores.append(round(item.ssim, 8))
        assert len(set(scores)) == 4
        assert actual[b][0].pearson == pytest.approx(1)
        assert actual[b][0].ssim == pytest.approx(1)
        assert actual[b][1].pearson == pytest.approx(-1)
        assert actual[b][1].ssim < 0
    torch.testing.assert_close(x, before_x)
    torch.testing.assert_close(y, before_y)


def test_constants_have_undefined_pearson_and_real_ssim():
    x = torch.zeros((1, 4, 1, 128, 128), dtype=torch.float64)
    y = x.clone()
    x[:, 1] = y[:, 1] = 0.3
    x[:, 2], y[:, 2] = 0.25, 0.75
    y[:, 3] = torch.arange(128, dtype=torch.float64) / 127
    result = evaluate_multiplane(x, y)[0]
    for p, score in enumerate(result):
        assert score.pearson is None and score.pearson_defined is False
        assert score.ssim == pytest.approx(scalar_reference(x[0, p].numpy(), y[0, p].numpy())[1])
    assert result[0].ssim == result[1].ssim == 1
    assert result[2].ssim < 1
    assert evaluate_multiplane(y, x)[0][3].pearson is None


def test_tiny_nonconstant_values_do_not_underflow_pearson():
    x = torch.zeros((1, 4, 1, 128, 128), dtype=torch.float64)
    x[..., 0, 0] = 1e-200
    result = evaluate_multiplane(x, x)[0]
    assert all(score.pearson_defined and score.pearson == pytest.approx(1) and score.ssim == 1 for score in result)


@pytest.mark.parametrize("kind", ["none", "dtype", "complex", "shape", "empty", "mismatch", "nan", "inf", "negative", "above_one"])
@pytest.mark.parametrize("side", ["prediction", "target"])
def test_invalid_metric_inputs(kind, side):
    valid = torch.zeros((1, 4, 1, 128, 128))
    bad = valid.clone()
    if kind == "none": bad = None
    elif kind == "dtype": bad = bad.to(torch.int64)
    elif kind == "complex": bad = bad.to(torch.complex64)
    elif kind == "shape": bad = bad.squeeze(0)
    elif kind == "empty": bad = bad[:0]
    elif kind == "mismatch": bad = bad.repeat(2, 1, 1, 1, 1)
    else: bad[0, 0, 0, 0, 0] = {"nan": float("nan"), "inf": float("inf"), "negative": -0.01, "above_one": 1.01}[kind]
    with pytest.raises(ValueError):
        evaluate_multiplane(bad, valid) if side == "prediction" else evaluate_multiplane(valid, bad)
