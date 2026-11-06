"""Acceptance tests for doe_core.fresnel (Issue #14, contract v1 docs/INTERFACES_V1.md).

Independent references used here (none of them calls an FFT):
- direct Fresnel sum in convolution form, NumPy float64 (and a torch matmul version for gradients);
- closed-form Fresnel integral of displaced/tilted Gaussian beams (continuous model);
- point source: one term of the Fresnel convolution;
- angular spectrum (doe_core.asm) at z = N*dx^2/lambda, where both grids coincide:
  a different algorithm without the paraxial approximation (same author, not a library).
Agreement with these does not prove the Fresnel approximation is accurate for every case
geometry; see docs/modules/fresnel.md.
"""
import cmath
import math

import numpy as np
import pytest
import torch

from doe_core.contracts import OpticsConfig
from doe_core.contracts_v1 import CaseConfigV1, FresnelField, FresnelGeometry
from doe_core.fresnel import propagate_fresnel, sampling_report

LAMBDA = 532e-9
CASE = CaseConfigV1()
CASE_PITCHES_UM = (10.0165625, 10.5153125, 11.0140625, 11.5128125)


def geom(z, pitch=(10e-6, 10e-6), wavelength=LAMBDA):
    return FresnelGeometry(wavelength_m=wavelength, input_pitch_m=pitch, distance_m=z)


def rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def random_phase_field(shape, seed=0, dtype=torch.complex128):
    gen = torch.Generator().manual_seed(seed)
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    phase = (2 * torch.rand(shape, generator=gen, dtype=real) - 1) * math.pi
    amp = 0.5 + torch.rand(shape, generator=gen, dtype=real)
    return torch.polar(amp, phase)


def axes(height, width, dy, dx):
    return (np.arange(height) - height // 2) * dy, (np.arange(width) - width // 2) * dx


# --- independent references (no FFT) --------------------------------------------------


def direct_sum_numpy(u_in, geometry):
    """Fresnel convolution sum: K(y,x) = e^{ikz}/(i lam z) sum U(v,u) e^{i pi((x-u)^2+(y-v)^2)/(lam z)} dx dy."""
    u_in = np.asarray(u_in, dtype=np.complex128)
    height, width = u_in.shape[-2:]
    lam, z = geometry.wavelength_m, geometry.distance_m
    dy, dx = geometry.input_pitch_m
    dy_o, dx_o = lam * z / (height * dy), lam * z / (width * dx)
    v, u = axes(height, width, dy, dx)
    y, x = axes(height, width, dy_o, dx_o)
    a_y = np.exp(1j * np.pi * (y[:, None] - v[None, :]) ** 2 / (lam * z))
    a_x = np.exp(1j * np.pi * (x[:, None] - u[None, :]) ** 2 / (lam * z))
    pref = cmath.exp(1j * 2 * np.pi * z / lam) / (1j * lam * z) * dy * dx
    return pref * np.einsum("pm,...mn,qn->...pq", a_y, u_in, a_x)


def direct_sum_torch(u_in, geometry):
    """Same convolution sum with torch matmuls (differentiable, still no FFT)."""
    height, width = u_in.shape[-2:]
    lam, z = geometry.wavelength_m, geometry.distance_m
    dy, dx = geometry.input_pitch_m
    v, u = (torch.from_numpy(t) for t in axes(height, width, dy, dx))
    y, x = (torch.from_numpy(t) for t in axes(height, width, lam * z / (height * dy), lam * z / (width * dx)))
    phase_y = math.pi * (y[:, None] - v[None, :]) ** 2 / (lam * z)
    phase_x = math.pi * (x[:, None] - u[None, :]) ** 2 / (lam * z)
    a_y = torch.polar(torch.ones_like(phase_y), phase_y)
    a_x = torch.polar(torch.ones_like(phase_x), phase_x)
    pref = cmath.exp(1j * 2 * math.pi * z / lam) / (1j * lam * z) * dy * dx
    return pref * (a_y @ u_in @ a_x.T)


def gaussian_fresnel(height, width, geometry, w0, center=(0.0, 0.0), tilt=(0.0, 0.0)):
    """Closed-form continuous Fresnel integral of exp(-((u-u0)^2+(v-v0)^2)/w0^2) * exp(i2pi(fx0 u + fy0 v)).

    Per axis: int exp(-a s^2 + b s + c) ds = sqrt(pi/a) exp(b^2/(4a) + c).
    Returns (input field, output field) on the contract's centred grids.
    """
    lam, z = geometry.wavelength_m, geometry.distance_m
    dy, dx = geometry.input_pitch_m
    v, u = axes(height, width, dy, dx)
    y, x = axes(height, width, lam * z / (height * dy), lam * z / (width * dx))

    def axis(s_in, s_out, s0, f0):
        a = 1 / w0**2 - 1j * np.pi / (lam * z)
        b = 2 * s0 / w0**2 + 1j * 2 * np.pi * f0 - 1j * 2 * np.pi * s_out / (lam * z)
        c = -(s0**2) / w0**2 + 1j * np.pi * s_out**2 / (lam * z)
        out = np.sqrt(np.pi / a) * np.exp(b**2 / (4 * a) + c)
        inp = np.exp(-((s_in - s0) ** 2) / w0**2 + 1j * 2 * np.pi * f0 * s_in)
        return inp, out

    in_y, out_y = axis(v, y, center[0], tilt[0])
    in_x, out_x = axis(u, x, center[1], tilt[1])
    pref = cmath.exp(1j * 2 * np.pi * z / lam) / (1j * lam * z)
    return in_y[:, None] * in_x[None, :], pref * out_y[:, None] * out_x[None, :]


# --- direct sum ------------------------------------------------------------------------


@pytest.mark.parametrize("shape,pitch,z", [((4, 6), (12e-6, 9e-6), 2e-3), ((5, 7), (8e-6, 11e-6), 3.3e-3)])
def test_matches_direct_sum_tiny_grids(shape, pitch, z):
    field = random_phase_field((3, 1) + shape, seed=1)
    g = geom(z, pitch)
    out = propagate_fresnel(field, g)
    assert isinstance(out, FresnelField)
    assert rel_l2(out.field.numpy(), direct_sum_numpy(field.numpy(), g)) < 1e-12


@pytest.mark.parametrize("plane", range(4))
def test_matches_direct_sum_case_grid(plane):
    g = CASE.plane(plane)
    field = random_phase_field((1, 1, 128, 128), seed=2)
    out = propagate_fresnel(field, g).field
    assert rel_l2(out.numpy(), direct_sum_numpy(field.numpy(), g)) < 1e-11


@pytest.mark.parametrize("shape,pitch", [((4, 6), (12e-6, 9e-6)), ((5, 7), (8e-6, 11e-6))])
def test_weighted_loss_gradient_matches_direct_sum(shape, pitch):
    g = geom(2.5e-3, pitch)
    gen = torch.Generator().manual_seed(3)
    weights = torch.rand((2, 1) + shape, generator=gen, dtype=torch.float64)
    phase0 = (2 * torch.rand((2, 1) + shape, generator=gen, dtype=torch.float64) - 1) * math.pi

    def grad(fn):
        phase = phase0.clone().requires_grad_(True)
        out = fn(torch.polar(torch.ones_like(phase), phase))
        (out.abs() ** 2 * weights).sum().backward()
        return phase.grad

    g_fft = grad(lambda f: propagate_fresnel(f, g).field)
    g_ref = grad(lambda f: direct_sum_torch(f, g))
    assert rel_l2(g_fft.numpy(), g_ref.numpy()) < 1e-10
    assert g_fft.abs().max() > 0


# --- coefficients, phase factors, coordinates -------------------------------------------


@pytest.mark.parametrize("m0,n0", [(2, 3), (0, 0), (4, 6), (1, 5)])
def test_point_source_coefficients_and_coordinates(m0, n0):
    # Delta at pixel (m0, n0): one term of the convolution sum, written out analytically.
    height, width, dy, dx, z = 5, 7, 8e-6, 11e-6, 3e-3
    g = geom(z, (dy, dx))
    field = torch.zeros(1, 1, height, width, dtype=torch.complex128)
    field[0, 0, m0, n0] = 1
    out = propagate_fresnel(field, g)
    dy_o, dx_o = LAMBDA * z / (height * dy), LAMBDA * z / (width * dx)
    assert out.output_pitch_m == pytest.approx((dy_o, dx_o), rel=1e-15)
    v0, u0 = (m0 - height // 2) * dy, (n0 - width // 2) * dx
    y = (np.arange(height) - height // 2)[:, None] * dy_o
    x = (np.arange(width) - width // 2)[None, :] * dx_o
    k = 2 * np.pi / LAMBDA
    expected = (
        np.exp(1j * k * z) / (1j * LAMBDA * z) * dy * dx
        * np.exp(1j * np.pi * ((x - u0) ** 2 + (y - v0) ** 2) / (LAMBDA * z))
    )
    assert rel_l2(out.field[0, 0].numpy(), expected) < 1e-12


def test_centered_point_source_gives_uniform_intensity_and_exact_prefactor():
    height, width, dy, dx, z = 6, 8, 10e-6, 10e-6, 24.1e-3
    field = torch.zeros(1, 1, height, width, dtype=torch.complex128)
    field[0, 0, height // 2, width // 2] = 1  # u = v = 0 by the contract's centring convention
    out = propagate_fresnel(field, geom(z, (dy, dx))).field[0, 0]
    assert torch.allclose(out.abs(), torch.full_like(out.abs(), dy * dx / (LAMBDA * z)), rtol=1e-13, atol=0)
    centre = out[height // 2, width // 2].item()  # x = y = 0: only exp(ikz)/(i lam z) * dx dy
    assert centre == pytest.approx(cmath.exp(1j * 2 * math.pi * z / LAMBDA) / (1j * LAMBDA * z) * dy * dx, rel=1e-12)


def test_case_output_pitches_and_windows():
    field = torch.ones(1, 1, 128, 128, dtype=torch.complex128)
    for plane, pitch_um in enumerate(CASE_PITCHES_UM):
        out = propagate_fresnel(field, CASE.plane(plane))
        assert out.distance_m == CASE.distances_m[plane]
        assert out.output_pitch_m[0] == pytest.approx(pitch_um * 1e-6, rel=1e-12)
        assert out.output_pitch_m[1] == pytest.approx(pitch_um * 1e-6, rel=1e-12)
        assert out.output_pitch_m == CASE.plane(plane).output_pitch_m(128, 128)
        assert 128 * out.output_pitch_m[1] == pytest.approx(CASE.wavelength_m * CASE.distances_m[plane] / 10e-6)


# --- continuous reference: Gaussian beams ------------------------------------------------


@pytest.mark.parametrize("plane", range(4))
@pytest.mark.parametrize("w0", [150e-6, 30e-6])
def test_gaussian_beam_closed_form_case_grid(plane, w0):
    g = CASE.plane(plane)
    u_in, expected = gaussian_fresnel(128, 128, g, w0)
    out = propagate_fresnel(torch.from_numpy(u_in)[None, None], g).field[0, 0].numpy()
    assert rel_l2(out, expected) < 1e-7  # measured <= 1.3e-8 (w0=150um), <= 2.8e-10 (w0=30um)


@pytest.mark.parametrize("center,tilt", [((-100e-6, 200e-6), (0.0, 0.0)), ((0.0, 0.0), (-6e3, 1e4)), ((50e-6, -150e-6), (4e3, -3e3))])
def test_orientation_shift_and_tilt_rectangular_grid(center, tilt):
    # H != W and dy != dx: axis swap, sign of the Fourier kernel or wrong centring would all fail.
    height, width, pitch, z = 96, 128, (12e-6, 10e-6), 30e-3
    g = geom(z, pitch)
    u_in, expected = gaussian_fresnel(height, width, g, 60e-6, center, tilt)
    out = propagate_fresnel(torch.from_numpy(u_in)[None, None], g)
    assert rel_l2(out.field[0, 0].numpy(), expected) < 1e-6
    # Centroid: y_c = v0 + lam*z*fy0, x_c = u0 + lam*z*fx0.
    i = out.field[0, 0].abs().numpy() ** 2
    y, x = axes(height, width, *out.output_pitch_m)
    y_c = (i.sum(axis=1) * y).sum() / i.sum()
    x_c = (i.sum(axis=0) * x).sum() / i.sum()
    assert y_c == pytest.approx(center[0] + LAMBDA * z * tilt[0], abs=1e-9)
    assert x_c == pytest.approx(center[1] + LAMBDA * z * tilt[1], abs=1e-9)


def test_beam_width_grows_with_distance_as_analytic():
    w0 = 30e-6
    z_r = math.pi * w0**2 / LAMBDA
    field = torch.from_numpy(gaussian_fresnel(128, 128, CASE.plane(0), w0)[0])[None, None]
    widths = []
    for plane in range(4):
        out = propagate_fresnel(field, CASE.plane(plane))
        i = out.field[0, 0].abs().numpy() ** 2
        _, x = axes(128, 128, *out.output_pitch_m)
        second_moment = (i.sum(axis=0) * x**2).sum() / i.sum()
        widths.append(2 * math.sqrt(second_moment))  # 1/e^2 intensity radius = 2*sigma
        z = CASE.distances_m[plane]
        assert widths[-1] == pytest.approx(w0 * math.sqrt(1 + (z / z_r) ** 2), rel=1e-6)
    assert all(a < b for a, b in zip(widths, widths[1:]))


# --- independent algorithm: angular spectrum where the grids coincide -------------------


def test_agrees_with_asm_where_output_pitch_equals_input_pitch():
    propagate_asm = pytest.importorskip("doe_core.asm", reason="ASM lives in PR #13 until merged").propagate

    n, dx = 128, 10e-6
    z = n * dx**2 / LAMBDA  # 24.06 mm: dx_out == dx, both methods on one grid
    g = geom(z)
    assert g.output_pitch_m(n, n) == pytest.approx((dx, dx), rel=1e-14)
    u_in, analytic = gaussian_fresnel(n, n, g, 150e-6)
    field = torch.from_numpy(u_in)[None, None]
    fresnel = propagate_fresnel(field, g).field[0, 0].numpy()
    asm = propagate_asm(field, OpticsConfig(LAMBDA, (dx, dx), z, pad_factor=2))[0, 0].numpy()
    # Smooth, well-contained beam: Fresnel (paraxial) model error is negligible here.
    assert rel_l2(fresnel, asm) < 1e-6  # measured 7.2e-8
    assert rel_l2(asm, analytic) < 1e-6  # measured 7.1e-8
    # NOT covered: full-aperture DOE-like masks, where the official single-FFT operator
    # differs strongly from ASM (light leaving the lambda*z/dx window); see docs/modules/fresnel.md.


# --- energy ------------------------------------------------------------------------------


@pytest.mark.parametrize("shape,pitch,z", [((5, 7), (8e-6, 11e-6), 3e-3), ((128, 128), (10e-6, 10e-6), 27.7e-3)])
def test_power_balance_with_pixel_areas(shape, pitch, z):
    field = random_phase_field((2, 1) + shape, seed=4)
    out = propagate_fresnel(field, geom(z, pitch))
    dy_o, dx_o = out.output_pitch_m
    p_in = (field.abs() ** 2).sum(dim=(-2, -1)) * pitch[0] * pitch[1]
    p_out = (out.field.abs() ** 2).sum(dim=(-2, -1)) * dy_o * dx_o
    assert torch.allclose(p_out, p_in, rtol=1e-12, atol=0)
    # Bare sums are not equal unless pitches match: no hidden normalisation.
    assert not torch.allclose((out.field.abs() ** 2).sum(), (field.abs() ** 2).sum(), rtol=1e-3)


# --- four planes from one mask -------------------------------------------------------------


def test_four_planes_from_one_shared_mask():
    gen = torch.Generator().manual_seed(5)
    phase = (2 * torch.rand(1, 1, 128, 128, generator=gen, dtype=torch.float64) - 1) * math.pi
    field = torch.polar(torch.ones_like(phase), phase)  # uniform illumination A = 1
    before = field.clone()
    outs = [propagate_fresnel(field, CASE.plane(i)) for i in range(4)]
    assert torch.equal(field, before)  # the shared mask is not modified
    intensities = [o.field.abs() ** 2 for o in outs]
    for a in range(4):
        for b in range(a + 1, 4):
            assert rel_l2(intensities[a].numpy(), intensities[b].numpy()) > 1e-2
    pitches = [o.output_pitch_m[1] for o in outs]
    assert all(p < q for p, q in zip(pitches, pitches[1:]))


def test_batch_independence():
    field = random_phase_field((3, 1, 16, 12), seed=6)
    g = geom(5e-3, (9e-6, 7e-6))
    batched = propagate_fresnel(field, g).field
    for b in range(3):
        single = propagate_fresnel(field[b : b + 1], g).field
        assert torch.allclose(batched[b : b + 1], single, rtol=1e-14, atol=0)


# --- autograd ------------------------------------------------------------------------------


def test_gradcheck_wrt_real_phase():
    g = geom(2e-3, (12e-6, 9e-6))
    weights = torch.rand(1, 1, 4, 6, generator=torch.Generator().manual_seed(7), dtype=torch.float64)

    def loss(phase):
        out = propagate_fresnel(torch.polar(torch.ones_like(phase), phase), g).field
        return (out.abs() ** 2 * weights).sum()

    phase = torch.rand(1, 1, 4, 6, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    assert torch.autograd.gradcheck(loss, (phase.requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_inference_call_does_not_break_later_autograd(dtype):
    field = random_phase_field((2, 1, 5, 7), seed=9, dtype=dtype)
    g = geom(3e-3, (8e-6, 11e-6))
    with torch.inference_mode():
        propagate_fresnel(field, g)
    leaf = field.clone().requires_grad_(True)
    (propagate_fresnel(leaf, g).field.abs() ** 2).sum().backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().max() > 0


# --- dtype / device --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_preserves_shape_dtype_device(dtype):
    field = random_phase_field((2, 1, 9, 6), dtype=dtype)
    out = propagate_fresnel(field, geom(4e-3)).field
    assert out.shape == field.shape and out.dtype == dtype and out.device == field.device


def test_complex64_accuracy_on_case_grid():
    # Carrier 2*pi*z/lambda ~ 3e5 rad and chirps are built in float64 before the cast.
    field = random_phase_field((1, 1, 128, 128), seed=10)
    for plane in range(4):
        out64 = propagate_fresnel(field.to(torch.complex64), CASE.plane(plane)).field
        out128 = propagate_fresnel(field, CASE.plane(plane)).field
        assert rel_l2(out64.to(torch.complex128).numpy(), out128.numpy()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preserved():
    field = random_phase_field((1, 1, 8, 8)).cuda()
    out = propagate_fresnel(field, geom(4e-3)).field
    assert out.device == field.device
    assert rel_l2(out.cpu().numpy(), propagate_fresnel(field.cpu(), geom(4e-3)).field.numpy()) < 1e-12


# --- sampling diagnostics ----------------------------------------------------------------------


def test_sampling_report_case_planes():
    for plane in range(4):
        report = sampling_report(128, 128, CASE.plane(plane))
        z = CASE.distances_m[plane]
        assert report["input_chirp_step_rad"] == pytest.approx(math.pi * 127 * (10e-6) ** 2 / (LAMBDA * z))
        assert report["input_chirp_step_rad"] < math.pi  # N*dx^2/lambda = 24.06 mm < 24.1 mm
        assert report["output_pitch_x_m"] == pytest.approx(CASE_PITCHES_UM[plane] * 1e-6)
    # Below z = N*dx^2/lambda the input chirp aliases.
    assert sampling_report(128, 128, geom(20e-3))["input_chirp_step_rad"] > math.pi


def test_sampling_report_matches_brute_force_chirp_steps():
    height, width, pitch, z = 9, 12, (7e-6, 5e-6), 1e-3
    g = geom(z, pitch)
    v, u = axes(height, width, *pitch)
    y, x = axes(height, width, *g.output_pitch_m(height, width))
    step = lambda s: np.abs(np.diff(np.pi * s**2 / (LAMBDA * z))).max()
    report = sampling_report(height, width, g)
    assert report["input_chirp_step_rad"] == pytest.approx(max(step(v), step(u)))
    assert report["output_chirp_step_rad"] == pytest.approx(max(step(y), step(x)))


# --- invalid inputs ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4, dtype=torch.float64),
        torch.zeros(1, 4, 4, dtype=torch.complex64),
        torch.zeros(1, 2, 4, 4, dtype=torch.complex64),
        torch.zeros(0, 1, 4, 4, dtype=torch.complex64),
        torch.zeros(1, 1, 1, 4, dtype=torch.complex64),
        torch.zeros(1, 1, 4, 1, dtype=torch.complex64),
        np.zeros((1, 1, 4, 4), dtype=np.complex128),
        None,
    ],
)
def test_invalid_field_rejected(bad):
    with pytest.raises(ValueError):
        propagate_fresnel(bad, geom(1e-3))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), complex(0, float("nan")), complex(float("-inf"), 0)])
def test_nonfinite_field_rejected(value):
    field = torch.zeros(1, 1, 4, 4, dtype=torch.complex128)
    field[0, 0, 1, 2] = value
    with pytest.raises(ValueError):
        propagate_fresnel(field, geom(1e-3))


def test_invalid_geometry_rejected():
    field = torch.zeros(1, 1, 4, 4, dtype=torch.complex64)
    for bad in (None, OpticsConfig(LAMBDA, (1e-5, 1e-5), 1e-3), (LAMBDA, (1e-5, 1e-5), 1e-3), CASE):
        with pytest.raises(ValueError):
            propagate_fresnel(field, bad)
    with pytest.raises(ValueError):
        sampling_report(4, 4, None)


def test_zero_distance_is_rejected_not_identity():
    # The single-FFT operator is singular at z = 0 (1/z prefactor, dx_out = 0); the contract rejects it.
    for z in (0.0, -1e-3):
        with pytest.raises(ValueError):
            geom(z)
