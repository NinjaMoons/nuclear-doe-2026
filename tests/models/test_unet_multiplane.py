"""Tests for the four-input U-Net skeleton (contract v1.1-amplitude, Issue #5).

Physics-informed training here uses a LOCAL differentiable stub propagator
(random phase screen + FFT, seeded by distance). It is NOT a Fresnel model: the
training smoke only shows that gradients reach the weights and the pipeline runs.
Real training needs the accepted Fresnel (#14), amplitude data (#4) and the gates.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from doe_core.contracts_v1 import CaseConfigV1, FresnelField, FresnelGeometry  # noqa: E402
from doe_models.train_multiplane import (  # noqa: E402
    check_disjoint_ids,
    checkpoint_metadata,
    evaluate_model,
    infer_phase,
    load_checkpoint,
    physics_loss,
    save_checkpoint,
    train_multiplane,
)
from doe_models.unet_multiplane import FourInputUNet, build_model  # noqa: E402

CONFIG = CaseConfigV1()
H, W = CONFIG.image_shape
SMALL = {"base_channels": 4, "depth": 2}


class ScreenStub:
    """Differentiable z-dependent stub: U -> fft2(U * exp(i*S_z)), ortho. Not Fresnel."""

    def __call__(self, field: torch.Tensor, geometry: FresnelGeometry) -> FresnelField:
        gen = torch.Generator().manual_seed(int(round(geometry.distance_m * 1e7)))
        s = (torch.rand(field.shape[-2:], generator=gen, dtype=torch.float64) * (2 * math.pi))
        s = s.to(device=field.device, dtype=field.real.dtype)
        out = torch.fft.fft2(field * torch.complex(torch.cos(s), torch.sin(s)), norm="ortho")
        h, w = field.shape[-2:]
        return FresnelField(out, geometry.output_pitch_m(h, w), geometry.distance_m)


def quartets(n: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    """n synthetic quartets [n,4,1,H,W] in [0,1]: random axis-aligned boxes per plane."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.zeros(n, 4, 1, H, W, dtype=dtype)
    for i in range(n):
        for k in range(4):
            y, x = torch.randint(0, H - 40, (2,), generator=gen).tolist()
            hh, ww = torch.randint(10, 40, (2,), generator=gen).tolist()
            t[i, k, 0, y:y + hh, x:x + ww] = float(torch.rand(1, generator=gen)) * 0.5 + 0.5
    return t


# --------------------------------------------------------------------------- model
@pytest.mark.parametrize("shape", [(1, 4, 1, 128, 128), (2, 4, 1, 128, 128), (1, 4, 1, 50, 70)])
def test_forward_shape_dtype_and_phase_range(shape):
    model = build_model(seed=0, **SMALL)
    phase = model(torch.rand(*shape))
    assert phase.shape == (shape[0], 1, shape[3], shape[4])
    assert phase.dtype == torch.float32
    assert torch.isfinite(phase).all()
    assert (phase >= -math.pi).all() and (phase < math.pi).all()


def test_float64_input_gives_float64_phase():
    model = build_model(seed=0, **SMALL).double()
    assert model(torch.rand(1, 4, 1, 32, 32, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.parametrize("training", [True, False])
def test_batch_samples_are_independent(training):
    """Batch and plane axes are not mixed: sample 0 is the same alone or in a batch,
    also in training mode (where BatchNorm would mix samples through batch statistics)."""
    model = build_model(seed=0, **SMALL).train(training)
    x = torch.rand(3, 4, 1, 64, 64, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        alone = model(x[:1])
        batched = model(x)
    assert torch.allclose(alone, batched[:1], atol=1e-5)


def test_every_input_plane_influences_the_phase():
    model = build_model(seed=0, **SMALL)
    x = torch.rand(1, 4, 1, 64, 64, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
    model(x).sum().backward()
    for k in range(4):
        assert x.grad[:, k].abs().sum() > 0, f"plane {k} does not reach the output"


def test_build_model_is_seed_local_and_reproducible():
    rng_before = torch.get_rng_state()
    a, b, c = build_model(seed=1, **SMALL), build_model(seed=1, **SMALL), build_model(seed=2, **SMALL)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.state_dict().values(), c.state_dict().values()))


def _bad_inputs():
    good = torch.rand(1, 4, 1, 32, 32)
    nan = good.clone(); nan[0, 0, 0, 0, 0] = float("nan")
    return {
        "4d": good[:, :, 0], "three_planes": good[:, :3], "two_channels": good.expand(1, 4, 2, 32, 32),
        "int_dtype": (good * 255).to(torch.uint8), "nan": nan, "tiny": good[..., :1, :1],
        "not_tensor": good.numpy(),
    }


@pytest.mark.parametrize("name", list(_bad_inputs()))
def test_forward_rejects_invalid_input(name):
    with pytest.raises(ValueError):
        build_model(seed=0, **SMALL)(_bad_inputs()[name])


@pytest.mark.parametrize("kwargs", [{"base_channels": 0}, {"depth": 0}, {"base_channels": 3}, {"depth": True}])
def test_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        FourInputUNet(**{**SMALL, **kwargs})


# --------------------------------------------------------------------------- physics loss
def test_physics_gradient_reaches_every_weight_through_four_planes():
    model = build_model(seed=0, **SMALL)
    loss, per_plane = physics_loss(model, quartets(2, seed=0), CONFIG, ScreenStub())
    assert per_plane.shape == (4,)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"no gradient reaches {name}"


def test_each_plane_loss_reaches_the_weights():
    targets = quartets(1, seed=1)
    for k in range(4):
        model = build_model(seed=0, **SMALL)
        _, per_plane = physics_loss(model, targets, CONFIG, ScreenStub())
        per_plane[k].backward()
        head = dict(model.named_parameters())["head.weight"]
        assert head.grad.abs().sum() > 0, f"plane {k} loss does not reach the weights"


# --------------------------------------------------------------------------- training smoke
@pytest.fixture(scope="module")
def smoke():
    train, val = quartets(4, seed=10), quartets(2, seed=20)
    model = build_model(seed=0, **SMALL)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    log = train_multiplane(model, train, config=CONFIG, propagate_fn=ScreenStub(), steps=20,
                           lr=3e-3, batch_size=2, seed=0, val_targets=val, eval_every=5,
                           train_ids=["t0", "t1", "t2", "t3"], val_ids=["v0", "v1"])
    return model, before, log, train, val


def test_training_changes_weights_and_lowers_train_loss(smoke):
    model, before, log, train, _ = smoke
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    initial = log["initial"]["train_loss"]
    after = evaluate_model(model, train, config=CONFIG, propagate_fn=ScreenStub())["loss"]
    assert after < initial


def test_training_log_is_complete_and_json(smoke):
    import json
    _, _, log, _, _ = smoke
    assert [h["step"] for h in log["history"]] == list(range(1, 21))
    assert all(math.isfinite(h["train_loss"]) for h in log["history"])
    assert [v["step"] for v in log["val"]] == [0, 5, 10, 15, 20]
    assert log["best_val_loss"] == min(v["val_loss"] for v in log["val"])
    assert log["observation"] == "amplitude"
    assert log["loss_policy"] == "per_plane_max_amplitude_mse_v1_1"
    json.dumps(log, allow_nan=False)


def test_model_holds_best_validation_weights(smoke):
    model, _, log, _, val = smoke
    val_now = evaluate_model(model, val, config=CONFIG, propagate_fn=ScreenStub())["loss"]
    assert val_now == pytest.approx(log["best_val_loss"], rel=1e-5)


def test_training_is_deterministic_for_a_seed():
    def run():
        model = build_model(seed=3, **SMALL)
        log = train_multiplane(model, quartets(3, seed=30), config=CONFIG, propagate_fn=ScreenStub(),
                               steps=3, lr=1e-3, batch_size=2, seed=7)
        return model, log
    (m1, l1), (m2, l2) = run(), run()
    assert [h["train_loss"] for h in l1["history"]] == [h["train_loss"] for h in l2["history"]]
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_check_disjoint_ids_rejects_leakage():
    check_disjoint_ids(train=["a", "b"], val=["c"], test=["d"])
    with pytest.raises(ValueError, match="b"):
        check_disjoint_ids(train=["a", "b"], val=["b"], test=["d"])
    with pytest.raises(ValueError):
        check_disjoint_ids(train=["a", "a"])


def test_train_rejects_overlapping_ids():
    with pytest.raises(ValueError):
        train_multiplane(build_model(seed=0, **SMALL), quartets(2, seed=0), config=CONFIG,
                         propagate_fn=ScreenStub(), steps=1, val_targets=quartets(1, seed=1),
                         train_ids=["x", "y"], val_ids=["y"])


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"lr": 0.0}, {"batch_size": 0}, {"seed": -1}, {"steps": True}])
def test_train_rejects_invalid_arguments(kwargs):
    args = {"steps": 1, "lr": 1e-3, "batch_size": 1, "seed": 0, **kwargs}
    with pytest.raises(ValueError):
        train_multiplane(build_model(seed=0, **SMALL), quartets(1, seed=0), config=CONFIG,
                         propagate_fn=ScreenStub(), **args)


# --------------------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_reproduces_forward(smoke, tmp_path):
    model, _, log, _, val = smoke
    path = tmp_path / "unet.pt"
    digest = save_checkpoint(path, model, config=CONFIG, train_log=log)
    import hashlib
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, payload = load_checkpoint(path)
    model.eval(); loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(val), loaded(val))
    assert payload["hparams"] == model.hparams
    assert payload["train_log"]["best_val_loss"] == log["best_val_loss"]
    assert payload["config"]["distances_m"] == [0.0241, 0.0253, 0.0265, 0.0277]
    assert payload["contract_version"] == "1.1-amplitude"


def test_load_rejects_non_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"not": "a checkpoint"}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


# --------------------------------------------------------------------------- #5B preparation
def test_training_log_measures_gradient_and_weight_change(smoke):
    _, before, log, _, _ = smoke
    assert all(math.isfinite(h["grad_norm"]) and h["grad_norm"] > 0 for h in log["history"])
    assert log["weights_delta_l2"] > 0


def test_infer_phase_is_detached_and_keeps_training_mode(smoke):
    model, _, _, _, val = smoke
    model.train()
    phase = infer_phase(model, val[:1])
    assert model.training
    assert phase.shape == (1, 1, H, W) and not phase.requires_grad
    assert (phase >= -math.pi).all() and (phase < math.pi).all()
    model.eval()
    with torch.no_grad():
        assert torch.equal(phase, model(val[:1]))


def test_checkpoint_records_commit_and_metadata_is_export_ready(smoke, tmp_path):
    import json
    model, _, log, _, _ = smoke
    path = tmp_path / "unet.pt"
    digest = save_checkpoint(path, model, config=CONFIG, train_log=log, code_commit="abc1234")
    _, payload = load_checkpoint(path)
    md = checkpoint_metadata(payload)
    json.dumps(md, allow_nan=False)
    assert md["method"] == "fourinput_unet"
    assert md["checkpoint_sha256"] == digest
    assert md["code_commit"] == "abc1234"
    assert md["hparams"] == model.hparams
    assert md["contract_version"] == "1.1-amplitude"
    assert md["training"]["best_val_loss"] == log["best_val_loss"]
    assert md["training"]["seed"] == 0 and md["training"]["steps"] == 20
    assert md["training"]["train_ids"] == ["t0", "t1", "t2", "t3"]


def test_fresh_process_reload_reproduces_phase(smoke, tmp_path):
    """A NEW Python process loads the checkpoint (weights_only) and gets the same phase."""
    import os
    import pathlib
    import subprocess
    import sys
    model, _, log, _, val = smoke
    ckpt, inp, out = tmp_path / "unet.pt", tmp_path / "in.pt", tmp_path / "out.pt"
    save_checkpoint(ckpt, model, config=CONFIG, train_log=log)
    torch.save(val, inp)
    script = "\n".join([
        "import sys, torch",
        "from doe_models.train_multiplane import load_checkpoint, infer_phase",
        "model, _ = load_checkpoint(sys.argv[1])",
        "x = torch.load(sys.argv[2], weights_only=True)",
        "torch.save(infer_phase(model, x), sys.argv[3])",
    ])
    root = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    subprocess.run([sys.executable, "-c", script, str(ckpt), str(inp), str(out)],
                   check=True, cwd=root, env=env, capture_output=True, timeout=300)
    assert torch.equal(torch.load(out, weights_only=True), infer_phase(model, val))
