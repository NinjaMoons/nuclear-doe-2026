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
        "minus_one": torch.full_like(good, -1.0), "two": torch.full_like(good, 2.0),
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


# --------------------------------------------------------------------------- strict checkpoint (review #19)
def _tamper(src, dst, edit):
    payload = torch.load(src, weights_only=True)
    edit(payload)
    torch.save(payload, dst)


TAMPERS = {
    "intensity_contract": lambda p: p.__setitem__("contract_version", "0-intensity"),
    "wrong_wavelength": lambda p: p["config"].__setitem__("wavelength_m", 1e-6),
    "wrong_distances": lambda p: p["config"].__setitem__("distances_m", [0.0241, 0.0253, 0.0265, 0.03]),
    "wrong_head": lambda p: p["hparams"].__setitem__("head", "atan2"),
    "wrong_in_planes": lambda p: p["hparams"].__setitem__("in_planes", 3),
    "wrong_norm": lambda p: p["hparams"].__setitem__("norm", "batchnorm"),
    "wrong_dtype": lambda p: p.__setitem__("dtype", "float16"),
    "bad_depth": lambda p: p["hparams"].__setitem__("depth", True),
    "missing_config": lambda p: p.pop("config"),
    "int64_head_weight": lambda p: p["state_dict"].__setitem__(
        "head.weight", p["state_dict"]["head.weight"].to(torch.int64)),
    "non_tensor_entry": lambda p: p["state_dict"].__setitem__("head.bias", []),
    "nan_weight": lambda p: p["state_dict"]["head.weight"].view(-1).__setitem__(0, float("nan")),
    "pos_inf_weight": lambda p: p["state_dict"]["head.bias"].view(-1).__setitem__(0, float("inf")),
    "neg_inf_weight": lambda p: next(iter(p["state_dict"].values())).view(-1).__setitem__(0, float("-inf")),
}


@pytest.mark.parametrize("name", list(TAMPERS))
def test_load_rejects_incompatible_checkpoint(name, tmp_path):
    good = tmp_path / "good.pt"
    save_checkpoint(good, build_model(seed=0, **SMALL), config=CONFIG, train_log={})
    bad = tmp_path / "bad.pt"
    _tamper(good, bad, TAMPERS[name])
    with pytest.raises(ValueError):
        load_checkpoint(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_roundtrip_keeps_dtype_and_global_rng(dtype, tmp_path):
    model = build_model(seed=0, **SMALL).to(dtype)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, config=CONFIG, train_log={})
    rng_before = torch.get_rng_state()
    loaded, payload = load_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert payload["dtype"] == str(dtype).replace("torch.", "")
    x = torch.rand(1, 4, 1, 32, 32, dtype=dtype, generator=torch.Generator().manual_seed(9))
    assert torch.equal(infer_phase(loaded, x), infer_phase(model, x))


# --------------------------------------------------------------------------- real Fresnel (integration)
# These run only where the real single-FFT Fresnel (#14) is importable (integration tree);
# in a branch without doe_core/fresnel.py they are skipped with an explicit reason.
# Synthetic box quartets, a few steps: this checks the pipeline, it is NOT a trained model.
@pytest.fixture(scope="module")
def fresnel():
    return pytest.importorskip("doe_core.fresnel",
                               reason="real propagate_fresnel (#14) is not in this tree").propagate_fresnel


@pytest.fixture(scope="module")
def real_run(fresnel):
    train, val = quartets(8, seed=300), quartets(2, seed=400)
    model = build_model(seed=0, **SMALL)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    log = train_multiplane(model, train, config=CONFIG, propagate_fn=fresnel, steps=15, lr=3e-3,
                           batch_size=4, seed=0, val_targets=val, eval_every=5,
                           train_ids=[f"t{i}" for i in range(8)], val_ids=["v0", "v1"])
    return model, before, log, train, val


@pytest.mark.integration
def test_integration_real_fresnel_gradients_reach_every_weight(fresnel):
    model = build_model(seed=0, **SMALL)
    loss, per_plane = physics_loss(model, quartets(2, seed=0), CONFIG, fresnel)
    assert torch.isfinite(per_plane).all()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"no real-Fresnel gradient reaches {name}"


@pytest.mark.integration
def test_integration_real_fresnel_training_updates_weights(real_run, fresnel):
    model, before, log, train, _ = real_run
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert log["weights_delta_l2"] > 0
    assert all(math.isfinite(h["grad_norm"]) and h["grad_norm"] > 0 for h in log["history"])
    after = evaluate_model(model, train, config=CONFIG, propagate_fn=fresnel)["loss"]
    assert after < log["initial"]["train_loss"]
    assert "fresnel" in log["propagator"]


@pytest.mark.integration
def test_integration_real_fresnel_fresh_process_reload(real_run, fresnel, tmp_path):
    import os
    import pathlib
    import subprocess
    import sys
    from doe_models.adam_multiplane import multiplane_amplitudes
    model, _, log, _, val = real_run
    ckpt, inp, out = tmp_path / "unet.pt", tmp_path / "in.pt", tmp_path / "out.pt"
    save_checkpoint(ckpt, model, config=CONFIG, train_log=log, code_commit="integration-test")
    torch.save(val, inp)
    script = "\n".join([
        "import sys, torch",
        "from doe_core.contracts_v1 import CaseConfigV1",
        "from doe_core.fresnel import propagate_fresnel",
        "from doe_models.adam_multiplane import multiplane_amplitudes",
        "from doe_models.train_multiplane import load_checkpoint, infer_phase",
        "model, _ = load_checkpoint(sys.argv[1])",
        "phase = infer_phase(model, torch.load(sys.argv[2], weights_only=True))",
        "amp = multiplane_amplitudes(phase, CaseConfigV1(), propagate_fresnel)",
        "torch.save({'phase': phase, 'amplitudes': amp}, sys.argv[3])",
    ])
    root = pathlib.Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c", script, str(ckpt), str(inp), str(out)], check=True, cwd=root,
                   env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, timeout=300)
    got = torch.load(out, weights_only=True)
    phase = infer_phase(model, val)
    assert torch.equal(got["phase"], phase)
    assert torch.equal(got["amplitudes"], multiplane_amplitudes(phase, CONFIG, fresnel))


@pytest.mark.integration
def test_integration_heldout_inference_through_export(real_run, tmp_path):
    import json
    export = pytest.importorskip("doe_experiments.export", reason="export #7 is not in this tree")
    image = pytest.importorskip("PIL.Image")
    model, _, log, _, _ = real_run
    ckpt = tmp_path / "unet.pt"
    save_checkpoint(ckpt, model, config=CONFIG, train_log=log, code_commit="integration-test")
    loaded, payload = load_checkpoint(ckpt)
    heldout = (quartets(1, seed=999) * 255).round().to(torch.uint8)       # held out: new seed
    paths = []
    for k in range(4):
        path = tmp_path / f"heldout_z{k + 1}.png"
        image.fromarray(heldout[0, k, 0].numpy()).save(path)
        paths.append(path)
    phase = infer_phase(loaded, heldout.float() / 255)
    export.export_run(paths, phase, tmp_path / "run", metadata=checkpoint_metadata(payload), synthetic_test=True)
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert metrics["observation"] == "amplitude"
    assert len(metrics["planes"]) == 4
    assert all(p["pearson_defined"] and math.isfinite(p["pearson"]) and math.isfinite(p["ssim"])
               for p in metrics["planes"])
    assert manifest["metadata"]["checkpoint_sha256"] == payload["sha256"]
    assert manifest["metadata"]["method"] == "fourinput_unet"


# --------------------------------------------------------------------------- opt-in training objective (#5, coordinator decision)
from doe_models.train_multiplane import TRAINING_OBJECTIVES, training_objective  # noqa: E402

LSQ_MIX = "lsq_gain_nmse_plus_0.1_contract_mse_v1"


def test_objective_names_and_default_is_contract():
    assert TRAINING_OBJECTIVES == ("contract_mse", LSQ_MIX)
    A = torch.rand(1, 4, 1, 3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) + 0.1
    T = torch.rand(1, 4, 1, 3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    from doe_models.adam_multiplane import multiplane_loss
    value, info = training_objective(A, T, "contract_mse")
    assert float(value) == float(multiplane_loss(A, T)[0])
    assert info["zero_target_planes"] == 0


def test_lsq_mix_matches_hand_computed_value():
    A = torch.zeros(1, 4, 1, 2, 2, dtype=torch.float64)
    T = torch.zeros_like(A)
    A[0, 0, 0] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    T[0, 0, 0] = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    for k in (1, 2, 3):
        T[0, k, 0] = torch.tensor([[0.0, 0.5], [1.0, 1.0]])
        A[0, k, 0] = 3.0 * T[0, k, 0]
    value, info = training_objective(A, T, LSQ_MIX)
    # plane 0: optimal gain s = <A,T>/<A,A> = 0 -> NMSE = 1; contract = (1+1)/4 = 0.5
    # planes 1-3: A = 3T -> NMSE 0, contract 0.  total = mean(1,0,0,0) + 0.1 * mean(.5,0,0,0)
    assert float(value) == pytest.approx(0.25 + 0.1 * 0.125, abs=1e-15)
    assert info["lsq_nmse"] == pytest.approx(0.25, abs=1e-15)
    assert info["contract_mse"] == pytest.approx(0.125, abs=1e-15)
    assert value.dtype == torch.float64


def test_lsq_mix_is_scale_invariant_and_differentiable():
    gen = torch.Generator().manual_seed(3)
    A = (torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64) + 0.1).requires_grad_(True)
    T = torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64)
    v1, _ = training_objective(A, T, LSQ_MIX)
    v5, _ = training_objective(5.0 * A, T, LSQ_MIX)
    assert float(v1.detach()) == pytest.approx(float(v5.detach()), rel=1e-12)
    assert torch.autograd.gradcheck(lambda a: training_objective(a, T, LSQ_MIX)[0], (A,))


def test_lsq_mix_float32_input_is_reduced_in_float64():
    A = torch.rand(1, 4, 1, 3, 4, generator=torch.Generator().manual_seed(4)) + 0.1
    T = torch.rand(1, 4, 1, 3, 4, generator=torch.Generator().manual_seed(5))
    v32, _ = training_objective(A, T, LSQ_MIX)
    v64, _ = training_objective(A.double(), T.double(), LSQ_MIX)
    assert v32.dtype == torch.float64 and float(v32) == float(v64)


def test_lsq_mix_zero_target_plane_falls_back_to_contract_and_is_counted():
    A = torch.rand(2, 4, 1, 3, 4, generator=torch.Generator().manual_seed(6), dtype=torch.float64) + 0.1
    T = torch.rand(2, 4, 1, 3, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    T[1, 2] = 0.0
    value, info = training_objective(A, T, LSQ_MIX)
    assert info["zero_target_planes"] == 1
    assert math.isfinite(float(value))
    D = A / A.amax(dim=(-2, -1), keepdim=True)
    per_contract = ((D - T) ** 2).mean(dim=(-2, -1))[..., 0]          # [B,4]
    s = (A * T).sum(dim=(-2, -1)) / (A * A).sum(dim=(-2, -1))
    per_lsq = ((s[..., None, None] * A - T) ** 2).sum(dim=(-2, -1)) / (T * T).sum(dim=(-2, -1))
    per_lsq = per_lsq[..., 0]
    per_lsq[1, 2] = per_contract[1, 2]
    expected = per_lsq.mean() + 0.1 * per_contract.mean()
    assert float(value) == pytest.approx(float(expected), rel=1e-12)


def test_unknown_objective_rejected():
    A = torch.ones(1, 4, 1, 2, 2)
    with pytest.raises(ValueError):
        training_objective(A, A, "pearson_only")
    with pytest.raises(ValueError):
        train_multiplane(build_model(seed=0, **SMALL), quartets(1, seed=0), config=CONFIG,
                         propagate_fn=ScreenStub(), steps=1, training_objective="pearson_only")


@pytest.fixture(scope="module")
def mix_run():
    calls = []

    def val_metrics(amplitudes, targets):
        calls.append(tuple(amplitudes.shape))
        return {"pearson_plane": [0.1, 0.2, 0.3, 0.4], "ssim_plane": [0.0, 0.1, 0.2, 0.3]}

    model = build_model(seed=0, **SMALL)
    log = train_multiplane(model, quartets(4, seed=10), config=CONFIG, propagate_fn=ScreenStub(), steps=6,
                           lr=3e-3, batch_size=2, seed=0, val_targets=quartets(2, seed=20), eval_every=3,
                           training_objective=LSQ_MIX, val_metrics_fn=val_metrics)
    return model, log, calls


def test_training_with_lsq_mix_logs_objective_and_selects_best_by_it(mix_run):
    model, log, calls = mix_run
    assert log["training_objective"] == LSQ_MIX
    assert [v["step"] for v in log["val"]] == [0, 3, 6]
    for v in log["val"]:
        assert {"objective", "contract_mse", "trivial_mean_t2", "bright_fraction",
                "zero_target_planes", "metrics"} <= set(v)
        assert 0 <= v["bright_fraction"] <= 1 and v["trivial_mean_t2"] > 0
        assert v["metrics"]["pearson_plane"] == [0.1, 0.2, 0.3, 0.4]
    assert log["best_val_loss"] == min(v["objective"] for v in log["val"])
    assert log["val"][[v["step"] for v in log["val"]].index(log["best_step"])]["objective"] == log["best_val_loss"]
    assert all({"objective", "contract_mse"} <= set(h) for h in log["history"])
    assert log["zero_target_fallbacks"] == 0
    assert calls and all(c == (2, 4, 1, H, W) for c in calls)
    import json
    json.dumps(log, allow_nan=False)


def test_model_holds_best_weights_by_objective(mix_run):
    model, log, _ = mix_run
    val = quartets(2, seed=20)
    now = evaluate_model(model, val, config=CONFIG, propagate_fn=ScreenStub(), training_objective=LSQ_MIX)
    assert now["objective"] == pytest.approx(log["best_val_loss"], rel=1e-6)


def test_default_training_still_uses_contract_objective(smoke):
    _, _, log, _, _ = smoke
    assert log["training_objective"] == "contract_mse"
    assert log["best_val_loss"] == min(v["val_loss"] for v in log["val"])


def test_best_state_is_chosen_by_objective_not_contract(monkeypatch):
    """Scripted validation where the objective and the contract MSE disagree on the best step."""
    import doe_models.train_multiplane as tm
    real = tm.evaluate_model
    scripted = iter([(0.9, 0.30), (0.2, 0.40), (0.5, 0.10)])   # (objective, contract) at val steps 0,1,2

    def fake(model, targets, **kw):
        out = real(model, targets, **kw)
        if targets.shape[0] == 2:          # validation set only
            objective, contract = next(scripted)
            out.update(objective=objective, loss=contract)
        return out

    monkeypatch.setattr(tm, "evaluate_model", fake)
    log = tm.train_multiplane(build_model(seed=0, **SMALL), quartets(3, seed=0), config=CONFIG,
                              propagate_fn=ScreenStub(), steps=2, batch_size=1, seed=0,
                              val_targets=quartets(2, seed=1), eval_every=1, training_objective=LSQ_MIX)
    assert [v["objective"] for v in log["val"]] == [0.9, 0.2, 0.5]
    assert log["best_step"] == 1 and log["best_val_loss"] == 0.2


# --------------------------------------------------------------------------- training driver / entrypoint (#5B)
import hashlib as _hashlib  # noqa: E402
import json as _json  # noqa: E402
import os as _os  # noqa: E402

from doe_models import train_multiplane as tm  # noqa: E402


def _png_loader(paths):
    """Test-only stand-in for #4A load_png_quartet: 4 mode-L PNG -> [1,4,1,H,W] float32 /255."""
    import numpy as np
    from PIL import Image
    arr = [torch.from_numpy(np.array(Image.open(p))).float() / 255 for p in paths]
    return torch.stack(arr)[None, :, None]


def _fake_export(paths, phase, run_dir, *, metadata, synthetic_test):
    """Test-only stand-in for #7 export_run: writes the files the driver reads back."""
    import numpy as np
    from PIL import Image
    run_dir.mkdir(parents=True, exist_ok=False)
    amp = tm.multiplane_amplitudes(phase.double(), CONFIG, ScreenStub())
    codes = np.floor(255 * (amp / amp.amax(dim=(-2, -1), keepdim=True)).numpy() + 0.5).astype(np.uint8)
    for k in range(4):
        Image.fromarray(codes[0, k, 0]).save(run_dir / f"reconstructed_{k + 1:02d}.png")
    np.savez(run_dir / "raw.npz", amplitude_quantized=amp.numpy())
    planes = [{"index": k, "pearson": 0.1 * (k + 1), "ssim": 0.05 * (k + 1), "pearson_defined": True}
              for k in range(4)]
    (run_dir / "metrics.json").write_text(_json.dumps({"planes": planes}))
    (run_dir / "manifest.json").write_text(_json.dumps({"metadata": metadata}))


def _components():
    return tm.DriverComponents(propagate=ScreenStub(), load_quartet=_png_loader,
                               val_metrics=lambda a, t: {"pearson_plane": [0.0] * 4, "ssim_plane": [0.0] * 4},
                               export_run=_fake_export)


def _write_manifest(root, counts=(("train", 4), ("val", 2), ("test", 2)), mutate=None):
    import numpy as np
    from PIL import Image
    samples, seed = [], 0
    for split, n in counts:
        for i in range(n):
            seed += 1
            q = (quartets(1, seed=seed)[0, :, 0].numpy() * 255).round().astype(np.uint8)
            rel = [f"c1/{split}{i}/z{k + 1}.png" for k in range(4)]
            for k, r in enumerate(rel):
                (root / r).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(q[k]).save(root / r)
            samples.append({"corpus_id": "c1", "local_sample_id": f"{split}{i}", "origin_id": f"c1:o{split}{i}",
                            "split": split, "kind": "B", "paths": rel,
                            "sha256": [_hashlib.sha256((root / r).read_bytes()).hexdigest() for r in rel]})
    manifest = {"schema": tm.MANIFEST_SCHEMA, "contract_version": "1.1-amplitude", "samples": samples}
    if mutate:
        mutate(manifest)
    path = root / "manifest.json"
    path.write_text(_json.dumps(manifest))
    return path


def test_load_manifest_returns_splits_and_hash(tmp_path):
    path = _write_manifest(tmp_path)
    data = tm.load_manifest(path, load_quartet=_png_loader)
    assert {k: data["splits"][k]["targets"].shape[0] for k in tm.SPLITS} == {"train": 4, "val": 2, "test": 2}
    assert data["splits"]["val"]["ids"] == ["c1:val0", "c1:val1"]
    assert data["splits"]["train"]["targets"].shape[1:] == (4, 1, H, W)
    assert data["manifest_sha256"] == _hashlib.sha256(path.read_bytes()).hexdigest()


MANIFEST_DEFECTS = {
    "wrong_schema": lambda m: m.__setitem__("schema", "other"),
    "intensity_contract": lambda m: m.__setitem__("contract_version", "0-intensity"),
    "bad_sha": lambda m: m["samples"][0]["sha256"].__setitem__(0, "0" * 64),
    "three_paths": lambda m: m["samples"][0]["paths"].pop(),
    "duplicate_key": lambda m: m["samples"][1].__setitem__("local_sample_id", m["samples"][0]["local_sample_id"]),
    "origin_leak": lambda m: m["samples"][-1].__setitem__("origin_id", m["samples"][0]["origin_id"]),
    "unknown_split": lambda m: m["samples"][0].__setitem__("split", "dev"),
    "escaping_path": lambda m: m["samples"][0]["paths"].__setitem__(0, "../outside.png"),
    "absolute_path": lambda m: m["samples"][0]["paths"].__setitem__(0, "C:/x.png" if _os.name == "nt" else "/x.png"),
    "colon_in_corpus": lambda m: m["samples"][0].__setitem__("corpus_id", "c:1"),
}


@pytest.mark.parametrize("name", list(MANIFEST_DEFECTS))
def test_load_manifest_rejects_defects(name, tmp_path):
    path = _write_manifest(tmp_path, mutate=MANIFEST_DEFECTS[name])
    with pytest.raises(ValueError):
        tm.load_manifest(path, load_quartet=_png_loader)


def test_load_manifest_requires_all_three_splits(tmp_path):
    path = _write_manifest(tmp_path, counts=(("train", 2), ("val", 1)))
    with pytest.raises(ValueError, match="test"):
        tm.load_manifest(path, load_quartet=_png_loader)


def test_on_validate_hook_and_continuous_single_run():
    calls = []
    model = build_model(seed=0, **SMALL)
    log = train_multiplane(model, quartets(4, seed=10), config=CONFIG, propagate_fn=ScreenStub(), steps=6,
                           batch_size=2, seed=0, val_targets=quartets(2, seed=20), eval_every=2,
                           on_validate=lambda step, m, entry, is_best, is_last: calls.append(
                               (step, is_best, is_last, entry["objective"], m is model)))
    assert [c[0] for c in calls] == [0, 2, 4, 6]
    assert [c[2] for c in calls] == [False, False, False, True]
    assert all(c[4] for c in calls)
    best_steps = [c[0] for c in calls if c[1]]
    assert best_steps[-1] == log["best_step"]
    assert [h["step"] for h in log["history"]] == [1, 2, 3, 4, 5, 6]
    assert all(h["step_s"] > 0 for h in log["history"])


def test_train_rejects_model_and_data_on_different_devices():
    model = build_model(seed=0, **SMALL).to(torch.device("meta"))
    with pytest.raises(ValueError, match="device"):
        train_multiplane(model, quartets(2, seed=0), config=CONFIG, propagate_fn=ScreenStub(), steps=1)


@pytest.fixture(scope="module")
def driver_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("driver")
    manifest = _write_manifest(root)
    out = root / "run"
    rc = tm.main(["--manifest", str(manifest), "--out-dir", str(out), "--device", "cpu", "--updates", "6",
                  "--eval-every", "2", "--batch-size", "2", "--base-channels", "4", "--depth", "2",
                  "--seed", "0"], components=_components())
    return rc, out, manifest


def test_driver_writes_checkpoints_logs_and_eval(driver_run):
    rc, out, manifest = driver_run
    assert rc == 0
    for name in ("run_config.json", "environment.json", "train_log.json", "eval_summary.json",
                 "checkpoints/best.pt", "checkpoints/last.pt", "checkpoints/step_000002.pt",
                 "checkpoints/step_000004.pt", "checkpoints/step_000006.pt"):
        assert (out / name).exists(), name
    cfg = _json.loads((out / "run_config.json").read_text())
    assert cfg["training_objective"] == "lsq_gain_nmse_plus_0.1_contract_mse_v1"
    assert cfg["updates"] == 6 and cfg["eval_every"] == 2 and cfg["amp"] is False
    assert cfg["manifest_sha256"] == _hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert set(cfg["code_sha256"]) >= {"doe_models/train_multiplane.py", "doe_models/unet_multiplane.py",
                                        "doe_models/adam_multiplane.py"}
    log = _json.loads((out / "train_log.json").read_text())
    assert [h["step"] for h in log["history"]] == list(range(1, 7))
    assert [v["step"] for v in log["val"]] == [0, 2, 4, 6]


def test_driver_checkpoints_are_marked_and_loadable(driver_run):
    _, out, _ = driver_run
    kinds = {}
    for name in ("best", "last", "step_000004"):
        _, payload = load_checkpoint(out / "checkpoints" / f"{name}.pt")
        kinds[name] = payload["extra"]
        assert payload["extra"]["resumable"] is False
    assert kinds["best"]["kind"] == "best" and kinds["last"]["kind"] == "last"
    assert kinds["step_000004"]["kind"] == "periodic" and kinds["step_000004"]["step"] == 4
    log = _json.loads((out / "train_log.json").read_text())
    assert kinds["best"]["step"] == log["best_step"]
    assert kinds["last"]["step"] == 6


def test_driver_eval_summary_reports_planes_and_collapse(driver_run):
    _, out, _ = driver_run
    s = _json.loads((out / "eval_summary.json").read_text())
    for split in ("val", "test"):
        r = s[split]
        assert r["pearson_plane"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert r["ssim_plane"] == pytest.approx([0.05, 0.1, 0.15, 0.2])
        assert r["worst_plane_pearson"] == pytest.approx(0.1)
        assert {"lit_frac", "target_lit_frac", "peak_pixel_power_frac", "collapse", "n"} <= set(r)
        assert r["n"] == 2
    assert s["checkpoint"].endswith("best.pt") and len(s["checkpoint_sha256"]) == 64


def test_driver_refuses_existing_out_dir(driver_run):
    _, out, manifest = driver_run
    with pytest.raises(FileExistsError):
        tm.main(["--manifest", str(manifest), "--out-dir", str(out), "--device", "cpu", "--updates", "1"],
                components=_components())


def test_driver_smoke_mode_measures_steps_without_checkpoints(tmp_path):
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "smoke"
    rc = tm.main(["--manifest", str(manifest), "--out-dir", str(out), "--device", "cpu", "--smoke-steps", "3",
                  "--batch-size", "2", "--base-channels", "4", "--depth", "2"], components=_components())
    assert rc == 0
    smoke = _json.loads((out / "smoke.json").read_text())
    assert smoke["steps"] == 3 and smoke["s_per_step"] > 0 and smoke["device"] == "cpu"
    assert not (out / "checkpoints").exists()


def test_driver_rejects_cuda_when_unavailable(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        tm.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "x"), "--device", "cuda"],
                components=_components())


@pytest.mark.integration
def test_integration_driver_with_production_components(tmp_path):
    """Real Fresnel + #4A loader/metrics + #7 export_run; evaluation runs in a NEW process."""
    for mod in ("doe_core.fresnel", "doe_experiments.data", "doe_experiments.export",
                "doe_experiments.metrics", "doe_experiments.png_io"):
        pytest.importorskip(mod, reason=f"{mod} is not in this tree")
    import os
    import pathlib
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "run"
    root = pathlib.Path(__file__).resolve().parents[2]
    cwd = os.getcwd()
    os.chdir(root)  # the evaluation subprocess runs `python -m doe_models.train_multiplane` from the tree root
    try:
        rc = tm.main(["--manifest", str(manifest), "--out-dir", str(out), "--device", "cpu", "--updates", "2",
                      "--eval-every", "1", "--batch-size", "2", "--base-channels", "4", "--depth", "2"])
    finally:
        os.chdir(cwd)
    assert rc == 0
    s = _json.loads((out / "eval_summary.json").read_text())
    assert s["checkpoint_kind"] == "best" and len(s["checkpoint_sha256"]) == 64
    for split in ("val", "test"):
        assert len(s[split]["pearson_plane"]) == 4 and s[split]["n"] == 2
        assert (out / "eval_best" / "export" / split / "000" / "DOE.png").exists()
    log = _json.loads((out / "train_log.json").read_text())
    assert log["propagator"] == "propagate_fresnel"
    assert log["val"][-1]["metrics"]["source"].startswith("official")
