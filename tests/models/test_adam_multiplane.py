"""Tests for multi-plane Adam synthesis (contract v1.1-amplitude, Issue #3).

Unit tests use a local differentiable stub propagator (random phase screen + FFT,
seeded by distance). The stub is NOT a Fresnel model and proves nothing about
physical correctness. Physical acceptance is the `integration` test at the end,
which runs only when the real single-FFT Fresnel from #14 is importable.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from doe_core.contracts_v1 import (  # noqa: E402
    CaseConfigV1,
    FresnelField,
    FresnelGeometry,
    MultiPlaneResult,
)
from doe_core.observables import field_observables  # noqa: E402
from doe_models.adam_multiplane import (  # noqa: E402
    multiplane_amplitudes,
    multiplane_intensities,
    multiplane_loss,
    synthesize_multiplane,
)

CONFIG = CaseConfigV1()
H, W = CONFIG.image_shape


# --------------------------------------------------------------------------- stubs
class ScreenStub:
    """Differentiable z-dependent stub: U -> fft2(U * exp(i*S_z)), ortho.

    S_z is a fixed random screen seeded by the distance, so each plane maps the
    same input phase to a different output. Records every call for inspection.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, FresnelGeometry]] = []

    def screen(self, geometry: FresnelGeometry, shape, dtype, device) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(round(geometry.distance_m * 1e7)))
        s = torch.rand(shape[-2:], generator=gen, dtype=torch.float64) * (2 * math.pi)
        return s.to(device=device, dtype=dtype)

    def __call__(self, field: torch.Tensor, geometry: FresnelGeometry) -> FresnelField:
        self.calls.append((field.detach().clone(), geometry))
        real_dtype = field.real.dtype
        s = self.screen(geometry, field.shape, real_dtype, field.device)
        mask = torch.complex(torch.cos(s), torch.sin(s))
        out = torch.fft.fft2(field * mask, norm="ortho")
        h, w = field.shape[-2:]
        return FresnelField(out, geometry.output_pitch_m(h, w), geometry.distance_m)


class WorseningStub(ScreenStub):
    """Adds a growing real offset per forward pass, so later steps are worse."""

    def __call__(self, field, geometry):
        res = super().__call__(field, geometry)
        forward_index = (len(self.calls) - 1) // 4
        return FresnelField(res.field + 5.0 * forward_index, res.output_pitch_m, res.distance_m)


class ModulusStub:
    """Phase-independent output modulus M_k: |U_out| = M_k exactly (|U_in| = 1)."""

    def __init__(self, moduli: torch.Tensor) -> None:
        self.moduli = moduli  # [4,H,W], real, >= 0

    def __call__(self, field, geometry):
        k = list(CONFIG.distances_m).index(geometry.distance_m)
        out = field * self.moduli[k].to(field.real.dtype)
        return FresnelField(out, geometry.output_pitch_m(H, W), geometry.distance_m)


class HalfDarkStub(ScreenStub):
    """ScreenStub output with exact zeros on the left half of every plane."""

    def __call__(self, field, geometry):
        res = super().__call__(field, geometry)
        out = res.field.clone()
        out[..., : W // 2] = 0
        return FresnelField(out, res.output_pitch_m, res.distance_m)


def quartet(batch: int = 1, dtype=torch.float32) -> torch.Tensor:
    """Four different, asymmetric binary targets in [0,1], shape [B,4,1,H,W]."""
    t = torch.zeros(batch, 4, 1, H, W, dtype=dtype)
    t[:, 0, 0, 20:50, 30:45] = 1.0          # off-centre bar
    t[:, 1, 0, 70:110, 15:25] = 1.0         # vertical L ...
    t[:, 1, 0, 100:110, 15:60] = 1.0        # ... and its foot
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ring = ((yy - 40) ** 2 + (xx - 90) ** 2 - 400).abs() < 60
    t[:, 2, 0][:, ring] = 1.0               # off-centre ring
    t[:, 3, 0, 90:100, 80:120] = 0.5        # grey bar, lower right
    return t


def wrap(phi: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi


# --------------------------------------------------------------------------- loss
# v1.1-amplitude policy: loss = mean((A/max(A) - T)^2), A = |U|, max per plane.
def test_loss_matches_hand_computed_value():
    amplitudes = torch.zeros(1, 4, 1, 2, 2, dtype=torch.float64)
    targets = torch.zeros_like(amplitudes)
    amplitudes[0, 0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets[0, 0, 0] = torch.tensor([[0.0, 0.5], [1.0, 1.0]])
    for k in (1, 2, 3):  # perfect match up to scale on the other planes
        targets[0, k, 0] = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
        amplitudes[0, k, 0] = 7.0 * targets[0, k, 0]

    total, per_plane = multiplane_loss(amplitudes, targets)

    # plane 0: A/max = [.25,.5,.75,1]; diff^2 = [.0625,0,.0625,0] -> mean .03125
    assert per_plane.tolist() == pytest.approx([0.03125, 0.0, 0.0, 0.0], abs=1e-15)
    assert float(total) == pytest.approx(0.0078125, abs=1e-15)


def test_loss_gradient_includes_max_denominator():
    """gradcheck over the full Jacobian fails if max(I) were detached."""
    gen = torch.Generator().manual_seed(1)
    amplitudes = (torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64) + 0.1)
    amplitudes.requires_grad_(True)
    targets = torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda i: multiplane_loss(i, targets)[0], (amplitudes,))


def test_synthesis_compares_amplitude_not_intensity():
    """|U| = [1,2] pattern: amplitude display [.5,1] matches T exactly, |U|^2 would not."""
    moduli = torch.ones(4, H, W, dtype=torch.float64)
    moduli[:, :, W // 2:] = 2.0
    targets = (moduli / 2.0).to(torch.float32).unsqueeze(0).unsqueeze(2)  # [1,4,1,H,W]
    result = synthesize_multiplane(targets, CONFIG, propagate_fn=ModulusStub(moduli), steps=1, seed=0)
    assert result.loss_history[0] == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(result.amplitudes, moduli.float().unsqueeze(0).unsqueeze(2), rtol=1e-6)
    assert torch.allclose(result.intensities, result.amplitudes.square(), rtol=1e-6)
    # the squared observable against the same targets: [.25,1] vs [.5,1] -> 0.25^2 / 2
    squared, _ = multiplane_loss(result.intensities, targets)
    assert float(squared) == pytest.approx(0.03125, rel=1e-6)


def test_amplitude_loss_gradcheck_through_complex_field():
    """Off zeros: complex field -> |U| -> loss, full Jacobian incl. the max denominator."""
    gen = torch.Generator().manual_seed(2)
    re_ = torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64) + 0.2
    im_ = torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64) + 0.2
    field = torch.complex(re_, im_).requires_grad_(True)
    targets = torch.rand(1, 4, 1, 3, 4, generator=gen, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda f: multiplane_loss(field_observables(f).amplitude, targets)[0], (field,))


def test_gradient_is_finite_where_field_is_exactly_zero():
    phase = (0.1 * torch.arange(H * W, dtype=torch.float64).reshape(1, 1, H, W)).requires_grad_(True)
    amplitudes = multiplane_amplitudes(phase, CONFIG, HalfDarkStub())
    assert (amplitudes[..., : W // 2] == 0).all()
    loss, _ = multiplane_loss(amplitudes, quartet(dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_loss_rejects_zero_amplitude_plane():
    amplitudes = torch.ones(1, 4, 1, 4, 4, dtype=torch.float64)
    amplitudes[0, 2] = 0.0
    with pytest.raises(ValueError, match="max"):
        multiplane_loss(amplitudes, torch.zeros_like(amplitudes))


# --------------------------------------------------------------------------- forward
def test_forward_uses_one_field_for_all_four_planes_in_distance_order():
    stub = ScreenStub()
    phase = torch.zeros(1, 1, H, W)
    phase[0, 0, 10:20, 10:20] = 1.3

    amplitudes = multiplane_amplitudes(phase, CONFIG, stub)

    assert amplitudes.shape == (1, 4, 1, H, W)
    assert [g.distance_m for _, g in stub.calls] == list(CONFIG.distances_m)
    first = stub.calls[0][0]
    assert all(torch.equal(f, first) for f, _ in stub.calls)
    # A=1 illumination: field is exp(i*phase)
    assert torch.allclose(first, torch.complex(torch.cos(phase), torch.sin(phase)))


def test_planes_differ_by_distance():
    amplitudes = multiplane_amplitudes(torch.zeros(1, 1, H, W) + 0.3 * quartet()[:, 0], CONFIG, ScreenStub())
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.allclose(amplitudes[:, a], amplitudes[:, b])


def test_diagnostic_intensities_are_squared_amplitudes():
    phase = 0.7 * quartet(dtype=torch.float64)[:, 2]
    amplitudes = multiplane_amplitudes(phase, CONFIG, ScreenStub())
    intensities = multiplane_intensities(phase, CONFIG, ScreenStub())
    assert torch.allclose(intensities, amplitudes.square(), rtol=1e-12, atol=0)


def test_gradient_reaches_shared_phase():
    phase = (0.1 * torch.arange(H * W, dtype=torch.float64).reshape(1, 1, H, W)).requires_grad_(True)
    loss, _ = multiplane_loss(multiplane_amplitudes(phase, CONFIG, ScreenStub()),
                              quartet(dtype=torch.float64))
    loss.backward()
    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_each_plane_loss_drives_the_same_phase():
    """Every per-plane loss has its own non-zero gradient on the ONE phase, and the
    total gradient is exactly their equal-weight (1/4) average: no plane is dropped."""
    targets = quartet(dtype=torch.float64)
    base = (0.05 * torch.arange(H * W, dtype=torch.float64).reshape(1, 1, H, W)).sin()
    grads = []
    for k in range(4):
        phase = base.clone().requires_grad_(True)
        _, per_plane = multiplane_loss(multiplane_amplitudes(phase, CONFIG, ScreenStub()), targets)
        per_plane[k].backward()
        grads.append(phase.grad)
        assert torch.isfinite(phase.grad).all()
        assert phase.grad.abs().sum() > 0, f"plane {k} does not reach the phase"
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.allclose(grads[a], grads[b])

    phase = base.clone().requires_grad_(True)
    total, _ = multiplane_loss(multiplane_amplitudes(phase, CONFIG, ScreenStub()), targets)
    total.backward()
    assert torch.allclose(phase.grad, sum(grads) / 4, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("plane", [0, 1, 2, 3])
def test_changing_one_plane_target_changes_the_optimised_phase(plane):
    targets = quartet()
    changed = targets.clone()
    changed[0, plane, 0] = changed[0, plane, 0].flip(-1)
    a = synthesize_multiplane(targets, CONFIG, propagate_fn=ScreenStub(), steps=3, seed=0)
    b = synthesize_multiplane(changed, CONFIG, propagate_fn=ScreenStub(), steps=3, seed=0)
    assert not torch.allclose(a.phase_rad, b.phase_rad)


def test_forward_rejects_bad_propagator_output():
    def bad(field, geometry):
        return FresnelField(field[..., :-1], geometry.output_pitch_m(H, W), geometry.distance_m)

    with pytest.raises(ValueError, match="propagat"):
        multiplane_amplitudes(torch.zeros(1, 1, H, W), CONFIG, bad)


# --------------------------------------------------------------------------- synthesis
@pytest.fixture(scope="module")
def run32():
    stub = ScreenStub()
    targets = quartet()
    result = synthesize_multiplane(targets, CONFIG, propagate_fn=stub, steps=300, lr=0.05, seed=3)
    return result, targets


def test_result_shapes_and_types(run32):
    result, _ = run32
    assert isinstance(result, MultiPlaneResult)
    assert result.phase_rad.shape == (1, 1, H, W)
    assert result.amplitudes.shape == (1, 4, 1, H, W)
    assert result.intensities.shape == (1, 4, 1, H, W)
    assert result.phase_rad.dtype == torch.float32
    assert not result.phase_rad.requires_grad
    assert len(result.loss_history) == 301
    assert result.elapsed_s >= 0.0


def test_result_is_finite_and_phase_in_range(run32):
    result, _ = run32
    assert torch.isfinite(result.phase_rad).all()
    assert torch.isfinite(result.amplitudes).all() and torch.isfinite(result.intensities).all()
    assert (result.amplitudes >= 0).all()
    assert all(math.isfinite(v) for v in result.loss_history)
    assert (result.phase_rad >= -math.pi).all() and (result.phase_rad < math.pi).all()


def test_total_loss_decreases(run32):
    result, _ = run32
    # measured on this stub: ~0.32 of the initial loss after 300 steps
    assert min(result.loss_history) < 0.5 * result.loss_history[0]
    assert result.metadata["best_step"] > 0


def test_returned_amplitudes_match_fresh_forward(run32):
    result, targets = run32
    again = multiplane_amplitudes(result.phase_rad, CONFIG, ScreenStub())
    assert torch.allclose(again, result.amplitudes, rtol=1e-6, atol=0)
    assert torch.allclose(result.intensities, result.amplitudes.square(), rtol=1e-6, atol=0)
    loss, per_plane = multiplane_loss(again, targets)
    assert float(loss) == pytest.approx(result.metadata["final_loss"], rel=1e-6)
    assert result.metadata["final_loss"] == pytest.approx(min(result.loss_history), rel=1e-4)
    assert per_plane.tolist() == pytest.approx(result.metadata["plane_losses"], rel=1e-6)


def test_metadata_records_planes_and_policy(run32):
    result, _ = run32
    md = result.metadata
    assert md["method"] == "adam_multiplane"
    assert md["seed"] == 3 and md["steps"] == 300 and md["lr"] == 0.05
    assert md["loss_policy"] == "per_plane_max_amplitude_mse_v1_1"
    assert md["observation"] == "amplitude"
    assert md["contract_version"] == "1.1-amplitude"
    assert md["distances_m"] == [0.0241, 0.0253, 0.0265, 0.0277]
    pitches = [p[1] for p in md["output_pitches_m"]]
    assert pitches == pytest.approx([10.0165625e-6, 10.5153125e-6, 11.0140625e-6, 11.5128125e-6], rel=1e-12)
    assert len(md["plane_losses"]) == 4
    assert md["best_loss"] == min(result.loss_history)
    assert result.loss_history[md["best_step"]] == md["best_loss"]
    assert md["config"]["wavelength_m"] == 532e-9
    assert md["device"] == "cpu" and md["dtype"] == "float32"
    import json
    json.dumps(md, allow_nan=False)


def test_same_seed_reproducible_and_global_rng_untouched():
    targets = quartet()
    rng_before = torch.get_rng_state()
    a = synthesize_multiplane(targets, CONFIG, propagate_fn=ScreenStub(), steps=5, seed=11)
    b = synthesize_multiplane(targets, CONFIG, propagate_fn=ScreenStub(), steps=5, seed=11)
    c = synthesize_multiplane(targets, CONFIG, propagate_fn=ScreenStub(), steps=5, seed=12)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(a.phase_rad, b.phase_rad)
    assert a.loss_history == b.loss_history
    assert not torch.equal(a.phase_rad, c.phase_rad)


def test_returns_best_state_not_last():
    """With a propagator that worsens every pass, step 0 must be returned."""
    seed = 5
    result = synthesize_multiplane(quartet(), CONFIG, propagate_fn=WorseningStub(), steps=4, seed=seed)
    assert result.metadata["best_step"] == 0
    assert result.loss_history[0] == min(result.loss_history)
    assert result.loss_history[-1] > result.loss_history[0]
    gen = torch.Generator().manual_seed(seed)
    initial = ((torch.rand(1, 1, H, W, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi).float()
    assert torch.allclose(result.phase_rad, wrap(initial), atol=1e-6)


def test_float64_and_batch_are_preserved():
    targets = quartet(batch=2, dtype=torch.float64)
    targets[1] = targets[1].flip(-1)
    result = synthesize_multiplane(targets, CONFIG, propagate_fn=ScreenStub(), steps=2, seed=0)
    assert result.phase_rad.shape == (2, 1, H, W)
    assert result.phase_rad.dtype == torch.float64
    assert result.amplitudes.dtype == torch.float64
    assert result.intensities.dtype == torch.float64
    assert result.metadata["dtype"] == "float64"


# --------------------------------------------------------------------------- validation
def _bad_targets():
    good = quartet()
    nan = good.clone(); nan[0, 0, 0, 0, 0] = float("nan")
    inf = good.clone(); inf[0, 1, 0, 0, 0] = float("inf")
    above = good.clone(); above[0, 2, 0, 0, 0] = 1.5
    below = good.clone(); below[0, 3, 0, 0, 0] = -0.1
    return {
        "4d": good[:, :, 0],
        "three_planes": good[:, :3],
        "two_channels": good.expand(1, 4, 2, H, W),
        "small_image": good[..., :64, :64],
        "int_dtype": (good * 255).to(torch.uint8),
        "nan": nan, "inf": inf, "above_one": above, "negative": below,
        "not_tensor": good.numpy(),
    }


@pytest.mark.parametrize("name", list(_bad_targets()))
def test_rejects_invalid_targets(name):
    with pytest.raises(ValueError):
        synthesize_multiplane(_bad_targets()[name], CONFIG, propagate_fn=ScreenStub(), steps=1)


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": True}, {"steps": 1.0},
    {"lr": 0.0}, {"lr": -1.0}, {"lr": float("nan")}, {"lr": True},
    {"seed": -1}, {"seed": True}, {"seed": 1.0},
])
def test_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        synthesize_multiplane(quartet(), CONFIG, propagate_fn=ScreenStub(), **kwargs)


def test_rejects_wrong_config_type():
    geometry = CONFIG.plane(0)
    with pytest.raises(ValueError):
        synthesize_multiplane(quartet(), geometry, propagate_fn=ScreenStub(), steps=1)


def test_rejects_zero_amplitude_from_propagator():
    def dark(field, geometry):
        return FresnelField(torch.zeros_like(field), geometry.output_pitch_m(H, W), geometry.distance_m)

    with pytest.raises(ValueError, match="max"):
        synthesize_multiplane(quartet(), CONFIG, propagate_fn=dark, steps=1)


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_integration_with_real_fresnel():
    """Acceptance with the real single-FFT Fresnel (#14); skipped until it is merged."""
    fresnel = pytest.importorskip(
        "doe_core.fresnel", reason="real propagate_fresnel (#14) is not merged yet")
    targets = quartet(dtype=torch.float64)
    result = synthesize_multiplane(targets, CONFIG, propagate_fn=fresnel.propagate_fresnel,
                                   steps=40, lr=0.1, seed=0)
    assert min(result.loss_history) < result.loss_history[0]
    assert torch.isfinite(result.amplitudes).all()
    again = multiplane_amplitudes(result.phase_rad, CONFIG, fresnel.propagate_fresnel)
    assert torch.allclose(again, result.amplitudes, rtol=1e-10, atol=0)
    pitches = [p[1] for p in result.metadata["output_pitches_m"]]
    assert pitches == pytest.approx([10.0165625e-6, 10.5153125e-6, 11.0140625e-6, 11.5128125e-6], rel=1e-12)
