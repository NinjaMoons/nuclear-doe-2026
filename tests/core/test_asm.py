"""Acceptance tests for doe_core.asm (Issue #2, contract v0 in docs/INTERFACES.md).

Geometries here are test fixtures (example values), not competition parameters.
"""
import math

import pytest
import torch

from doe_core.asm import intensity, propagate
from doe_core.contracts import OpticsConfig

LAMBDA = 532e-9  # example wavelength, m


def cfg(z, pitch=(8e-6, 8e-6), pad=1, wavelength=LAMBDA):
    return OpticsConfig(wavelength_m=wavelength, pixel_pitch_m=pitch, distance_m=z, pad_factor=pad)


def random_field(shape, dtype=torch.complex128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    re = torch.randn(shape, generator=gen, dtype=real_dtype)
    im = torch.randn(shape, generator=gen, dtype=real_dtype)
    return torch.complex(re, im)


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def dft_mode(height, width, my, mx, dtype=torch.complex128):
    """Integer DFT mode exp(i*2*pi*(my*y/H + mx*x/W)) as [1,1,H,W]."""
    y = torch.arange(height, dtype=torch.float64)[:, None]
    x = torch.arange(width, dtype=torch.float64)[None, :]
    phase = 2 * math.pi * (my * y / height + mx * x / width)
    return torch.polar(torch.ones_like(phase), phase).to(dtype)[None, None]


def kz(fy, fx, wavelength=LAMBDA):
    """Complex kz with Im >= 0 (independent of the module: uses cmath)."""
    import cmath

    return cmath.sqrt(complex(1 / wavelength**2 - fx**2 - fy**2))


# --- z = 0 identity -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("pad", [1, 2, 3])
def test_zero_distance_is_exact_identity(dtype, pad):
    field = random_field((3, 1, 7, 10), dtype)  # odd/rectangular, white noise = all frequencies
    out = propagate(field, cfg(0.0, pitch=(1e-7, 2e-7), pad=pad))  # pitch < lambda: evanescent band
    assert out.shape == field.shape and out.dtype == dtype
    assert torch.equal(out, field)
    assert out.data_ptr() != field.data_ptr()


def test_zero_distance_keeps_autograd():
    field = random_field((1, 1, 4, 4)).requires_grad_(True)
    out = propagate(field, cfg(0.0))
    intensity(out).sum().backward()
    assert torch.allclose(field.grad, 2 * field.detach())


# --- analytic modes (complex field, not only intensity) -------------------------------


def test_constant_field_gets_carrier_phase():
    z = 0.1
    field = torch.full((2, 1, 6, 9), 0.3 - 0.7j, dtype=torch.complex128)
    out = propagate(field, cfg(z))
    expected = field * complex(math.cos(2 * math.pi * z / LAMBDA), math.sin(2 * math.pi * z / LAMBDA))
    assert rel_l2(out, expected) < 1e-9


@pytest.mark.parametrize("my,mx", [(1, 2), (-3, 1), (2, -5), (0, 4)])
def test_single_dft_mode_matches_analytic(my, mx):
    height, width, dy, dx, z = 32, 24, 8e-6, 6e-6, 1e-3
    field = dft_mode(height, width, my, mx)
    out = propagate(field, cfg(z, pitch=(dy, dx)))
    fy, fx = my / (height * dy), mx / (width * dx)
    k = kz(fy, fx)
    assert k.imag == 0  # propagating
    expected = field * complex(math.cos(2 * math.pi * z * k.real), math.sin(2 * math.pi * z * k.real))
    assert rel_l2(out, expected) < 1e-9


def test_sign_convention_positive_z_is_exp_plus_ikz():
    # exp(+i kz z): phase of a DC field advances by +2*pi*z/lambda (mod 2*pi).
    z = LAMBDA / 4
    out = propagate(torch.ones(1, 1, 4, 4, dtype=torch.complex128), cfg(z))
    assert torch.allclose(out, torch.full_like(out, 1j), atol=1e-12)


def test_linearity_and_no_hidden_normalisation():
    config = cfg(2e-3, pad=2)
    f, g = random_field((2, 1, 8, 6), seed=1), random_field((2, 1, 8, 6), seed=2)
    a, b = 3.0 - 1.0j, -0.5 + 2.0j
    assert rel_l2(propagate(a * f + b * g, config), a * propagate(f, config) + b * propagate(g, config)) < 1e-12
    assert torch.allclose(intensity(3 * f), 9 * intensity(f))


# --- energy ---------------------------------------------------------------------------


def test_power_conserved_on_periodic_domain_with_propagating_spectrum():
    # dx > lambda/sqrt(2): every DFT mode propagates, |H| = 1, ortho FFT is unitary.
    field = random_field((3, 1, 16, 20), seed=3)
    out = propagate(field, cfg(5e-2, pitch=(8e-6, 7e-6)))
    p_in = intensity(field).sum(dim=(-2, -1))
    p_out = intensity(out).sum(dim=(-2, -1))
    assert torch.allclose(p_out, p_in, rtol=1e-10, atol=0)


def test_crop_never_gains_power():
    field = torch.zeros(1, 1, 16, 16, dtype=torch.complex128)
    field[..., 6:10, 6:10] = 1
    out = propagate(field, cfg(5e-3, pad=4))
    assert intensity(out).sum() <= intensity(field).sum() * (1 + 1e-12)


# --- evanescent waves -----------------------------------------------------------------


def test_evanescent_mode_decays_with_correct_branch():
    # pitch 0.2*lambda: mode (0, 2) on W=8 has fx = 2/(8*0.2*lambda) = 1.25/lambda > 1/lambda.
    height, width, pitch, z = 4, 8, 0.2 * LAMBDA, 0.1 * LAMBDA
    field = dft_mode(height, width, 0, 2)
    out = propagate(field, cfg(z, pitch=(pitch, pitch)))
    k = kz(0.0, 2 / (width * pitch))
    assert k.real == 0 and k.imag > 0
    expected = field * math.exp(-2 * math.pi * z * k.imag)  # real decay, no phase
    assert torch.isfinite(out).all()
    assert rel_l2(out, expected) < 1e-9


def test_mixed_spectrum_no_nonphysical_growth():
    pitch = 0.3 * LAMBDA  # part of the spectrum is evanescent
    field = random_field((2, 1, 16, 16), seed=4)
    p_in = intensity(field).sum(dim=(-2, -1))
    previous = p_in
    for z in (0.5 * LAMBDA, 2 * LAMBDA, 10 * LAMBDA):
        p = intensity(propagate(field, cfg(z, pitch=(pitch, pitch)))).sum(dim=(-2, -1))
        assert torch.isfinite(p).all()
        assert (p <= previous * (1 + 1e-12)).all()
        previous = p
    assert (previous < p_in).all()


# --- padding / crop semantics ---------------------------------------------------------


def test_padding_matches_manual_pad_and_central_crop():
    # H=5, W=7, factor 2 -> 10x14, start (2, 3): odd remainder goes bottom/right.
    # Note: on the periodic padded grid the operator is shift-equivariant, so the window
    # offset does not change the cropped output; this checks pad -> propagate -> crop.
    field = random_field((2, 1, 5, 7), seed=5)
    manual = torch.zeros(2, 1, 10, 14, dtype=torch.complex128)
    manual[..., 2:7, 3:10] = field
    reference = propagate(manual, cfg(1e-3, pitch=(8e-6, 8e-6), pad=1))[..., 2:7, 3:10]
    out = propagate(field, cfg(1e-3, pitch=(8e-6, 8e-6), pad=2))
    assert out.shape == field.shape
    assert rel_l2(out, reference) < 1e-12


def test_padding_converges_for_example_preset():
    # Example preset (not a competition value): 64x64, 8 um, 532 nm, sharp 16x16 px aperture,
    # z = 20 mm > N*dx^2/lambda = 7.7 mm for pad 1: the periodic domain wraps diffraction around.
    # Self-convergence against pad 16 (measured: 1.6e-1, 1.6e-3, 1.5e-5, 6.8e-7 for pad 1/2/4/8).
    n, pitch, z = 64, 8e-6, 2e-2
    field = torch.zeros(1, 1, n, n, dtype=torch.complex128)
    field[..., 24:40, 24:40] = 1
    results = {p: intensity(propagate(field, cfg(z, pad=p))) for p in (1, 2, 4, 8, 16)}
    err = [rel_l2(results[p], results[16]) for p in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(err, err[1:]))
    assert err[0] > 5e-2  # pad 1 is not usable for this geometry
    assert err[2] < 1e-4  # pad >= 4 converged in the observed window
    capture = (results[16].sum() / intensity(field).sum()).item()
    assert 0.5 < capture < 1  # power leaves the cropped window; it is not renormalised


# --- independent physical reference: paraxial Gaussian beam --------------------------


def test_gaussian_beam_matches_paraxial_analytic():
    # w0 = 20 um >> lambda: paraxial error ~ (lambda/(pi*w0))^2 ~ 7e-5.
    n, pitch, w0, pad = 128, 4e-6, 20e-6, 2
    z_r = math.pi * w0**2 / LAMBDA
    assert z_r < pad * n * pitch**2 / LAMBDA  # transfer function sampled without aliasing
    coords = (torch.arange(n, dtype=torch.float64) - n // 2) * pitch
    r2 = coords[:, None] ** 2 + coords[None, :] ** 2
    field = torch.exp(-r2 / w0**2).to(torch.complex128)[None, None]
    out = propagate(field, cfg(z_r, pitch=(pitch, pitch), pad=pad))
    i_out = intensity(out)[0, 0]
    on_axis = i_out[n // 2, n // 2].item()
    assert on_axis == pytest.approx(0.5, rel=2e-4)  # I0 / (1 + (z/zR)^2); measured 3.6e-5
    w_z = w0 * math.sqrt(2)
    expected = 0.5 * torch.exp(-2 * r2 / w_z**2)
    assert rel_l2(i_out, expected) < 2e-4  # measured 2.5e-5


# --- autograd -------------------------------------------------------------------------


@pytest.mark.parametrize("pad", [1, 2])
def test_gradcheck_wrt_real_phase(pad):
    config = cfg(1e-4, pitch=(8e-6, 8e-6), pad=pad)
    weights = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(6), dtype=torch.float64)

    def loss(phase):
        return (intensity(propagate(torch.polar(torch.ones_like(phase), phase), config)) * weights).sum()

    phase = torch.rand(1, 1, 4, 5, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    assert torch.autograd.gradcheck(loss, (phase.requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_image_loss_gradient_is_finite_and_nonzero():
    gen = torch.Generator().manual_seed(8)
    phase = (2 * math.pi * torch.rand(2, 1, 16, 16, generator=gen)).requires_grad_(True)
    target = torch.rand(2, 1, 16, 16, generator=gen)
    field = torch.polar(torch.ones_like(phase), phase) / 16.0
    i = intensity(propagate(field, cfg(1e-3, pad=2)))
    loss = ((i / i.sum(dim=(-2, -1), keepdim=True) - target / target.sum(dim=(-2, -1), keepdim=True)) ** 2).mean()
    loss.backward()
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().max() > 0


# --- dtype / device / shape -----------------------------------------------------------


@pytest.mark.parametrize("dtype,real", [(torch.complex64, torch.float32), (torch.complex128, torch.float64)])
@pytest.mark.parametrize("pad", [1, 2])
def test_preserves_shape_dtype_device(dtype, real, pad):
    field = random_field((3, 1, 9, 6), dtype)
    out = propagate(field, cfg(1e-3, pad=pad))
    assert out.shape == field.shape and out.dtype == dtype and out.device == field.device
    i = intensity(out)
    assert i.shape == field.shape and i.dtype == real and (i >= 0).all()


def test_complex64_matches_complex128_at_large_phase():
    # 2*pi*z/lambda ~ 1.2e6 rad: transfer must not lose phase accuracy in float32.
    field = random_field((1, 1, 16, 16), seed=9)
    config = cfg(0.1, pad=2)
    out64 = propagate(field.to(torch.complex64), config)
    out128 = propagate(field, config)
    assert rel_l2(out64.to(torch.complex128), out128) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preserved():
    field = random_field((1, 1, 8, 8)).cuda()
    out = propagate(field, cfg(1e-3, pad=2))
    assert out.device == field.device
    assert rel_l2(out.cpu(), propagate(field.cpu(), cfg(1e-3, pad=2))) < 1e-12


# --- invalid inputs -------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 1, 4, 4),  # real
        torch.zeros(1, 1, 4, 4, dtype=torch.float64),  # real double
        torch.zeros(1, 4, 4, dtype=torch.complex64),  # ndim 3
        torch.zeros(1, 2, 4, 4, dtype=torch.complex64),  # C != 1
        torch.zeros(0, 1, 4, 4, dtype=torch.complex64),  # B = 0
        torch.zeros(1, 1, 1, 4, dtype=torch.complex64),  # H < 2
        torch.zeros(1, 1, 4, 1, dtype=torch.complex64),  # W < 2
        [[0j]],
        None,
    ],
)
def test_invalid_field_rejected(bad):
    with pytest.raises(ValueError):
        propagate(bad, cfg(1e-3))
    with pytest.raises(ValueError):
        intensity(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), complex(0, float("nan")), complex(float("-inf"), 0)])
def test_nonfinite_field_rejected(value):
    field = torch.zeros(1, 1, 4, 4, dtype=torch.complex128)
    field[0, 0, 1, 2] = value
    with pytest.raises(ValueError):
        propagate(field, cfg(1e-3))
    with pytest.raises(ValueError):
        intensity(field)


def test_invalid_config_rejected():
    field = torch.zeros(1, 1, 4, 4, dtype=torch.complex64)
    for bad in (None, {"wavelength_m": LAMBDA}, (LAMBDA, (8e-6, 8e-6), 1e-3)):
        with pytest.raises(ValueError):
            propagate(field, bad)
