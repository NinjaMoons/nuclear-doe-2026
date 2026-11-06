"""Physical checks of the single-FFT Fresnel forward on the official case (Issue #14B).

One phase mask, four planes, amplitude observable |U| (contract v1.1-amplitude).
The expected outcome follows from optics, not from the implementation:
a thin lens exp(-i*pi*((u-x0)^2+(v-y0)^2)/(lambda*z0)) focuses a uniform wave to (y0, x0)
in the plane z0. A phase-only superposition of four such lenses must therefore put a
bright spot at its own target point in each of the four planes, at the coordinates given by
that plane's own output pitch lambda*z/(N*dx).
"""
import cmath
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel, sampling_report
from doe_core.observables import field_observables

CASE = CaseConfigV1()
N = 128
# (y0, x0) focus points in metres, one per plane, in the official distance order.
FOCI_M = ((0.2e-3, 0.2e-3), (0.2e-3, -0.2e-3), (-0.2e-3, -0.2e-3), (-0.2e-3, 0.2e-3))


def multifocal_phase(dtype=torch.float64):
    """Phase in [-pi, pi] of the superposition of four focusing lenses (uniform amplitude A=1)."""
    dy, dx = CASE.input_pitch_m
    v = (torch.arange(N, dtype=torch.float64) - N // 2)[:, None] * dy
    u = (torch.arange(N, dtype=torch.float64) - N // 2)[None, :] * dx
    total = torch.zeros(N, N, dtype=torch.complex128)
    for (y0, x0), z0 in zip(FOCI_M, CASE.distances_m):
        lens = -math.pi * ((u - x0) ** 2 + (v - y0) ** 2) / (CASE.wavelength_m * z0)
        total = total + torch.polar(torch.ones_like(lens), lens)
    return torch.angle(total).to(dtype)[None, None]


def four_planes(phase):
    field = torch.polar(torch.ones_like(phase), phase)
    outs = [propagate_fresnel(field, CASE.plane(k)) for k in range(4)]
    return field, outs


def pixel_of(point_m, pitch_m):
    return tuple(round(c / p) + N // 2 for c, p in zip(point_m, pitch_m))


def test_multifocal_doe_focuses_at_its_own_point_in_each_plane():
    phase = multifocal_phase()
    assert phase.abs().max() <= math.pi
    field, outs = four_planes(phase)
    for k, out in enumerate(outs):
        amp = field_observables(out.field).amplitude[0, 0]
        expected = pixel_of(FOCI_M[k], out.output_pitch_m)
        peak = divmod(int(amp.argmax()), N)
        assert peak == expected, (k, peak, expected)  # global maximum of the whole plane
        own = amp[expected].item()
        others = [amp[pixel_of(FOCI_M[j], out.output_pitch_m)].item() for j in range(4) if j != k]
        # Planes are only 1.2 mm apart, so the other lenses are mildly defocused:
        # measured own/others = 3.90, 3.76, 3.24, 2.45 for 24.1 ... 27.7 mm.
        assert own > 2 * max(others), (k, own, others)


def test_wrong_plane_pitch_would_misplace_the_focus():
    # Using the 24.1 mm pitch for the 27.7 mm plane moves the expected spot by ~2.6 px:
    # the per-plane output pitch is physically required, PNG pixels are not 10 um.
    _, outs = four_planes(multifocal_phase())
    amp = field_observables(outs[3].field).amplitude[0, 0]
    peak = divmod(int(amp.argmax()), N)
    right = pixel_of(FOCI_M[3], outs[3].output_pitch_m)
    wrong = pixel_of(FOCI_M[3], outs[0].output_pitch_m)
    assert right != wrong
    assert max(abs(peak[0] - right[0]), abs(peak[1] - right[1])) < max(abs(peak[0] - wrong[0]), abs(peak[1] - wrong[1]))


def test_shared_field_power_and_amplitude_observable():
    field, outs = four_planes(multifocal_phase())
    p_in = (field.abs() ** 2).sum().item() * CASE.input_pitch_m[0] * CASE.input_pitch_m[1]
    for out in outs:
        obs = field_observables(out.field)
        assert torch.equal(obs.intensity, obs.amplitude.square())
        p_out = obs.intensity.sum().item() * out.output_pitch_m[0] * out.output_pitch_m[1]
        assert p_out == pytest.approx(p_in, rel=1e-12)


def test_amplitude_loss_gradient_flows_to_the_shared_phase_from_all_planes():
    phase = multifocal_phase().requires_grad_(True)
    _, outs = four_planes(phase)
    grads = []
    for k, out in enumerate(outs):
        amp = field_observables(out.field).amplitude
        (g,) = torch.autograd.grad(amp[0, 0][pixel_of(FOCI_M[k], out.output_pitch_m)], phase, retain_graph=True)
        assert torch.isfinite(g).all() and g.abs().max() > 0
        grads.append(g)
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.allclose(grads[a], grads[b])  # each plane contributes its own gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_case_planes_sampling_and_precision(dtype):
    phase = multifocal_phase(dtype)
    _, outs = four_planes(phase)
    _, ref = four_planes(multifocal_phase(torch.float64))
    for k in range(4):
        report = sampling_report(N, N, CASE.plane(k))
        assert report["input_chirp_step_rad"] < math.pi
        amp = field_observables(outs[k].field).amplitude.double()
        amp_ref = field_observables(ref[k].field).amplitude
        assert (torch.linalg.vector_norm(amp - amp_ref) / torch.linalg.vector_norm(amp_ref)).item() < 1e-5


# --- exported DOE check (READY NEXT #14B; runs on a real artifact when one is provided) --------
#
# FRESNEL_DECODED_PHASE=<path to .npy or .npz (key "phase_rad"), decoded DOE phase 128x128, rad>
#   python -m pytest tests/core/test_fresnel_physical.py -k exported -s
# The phase must already be decoded from DOE.png by the project codec (#7); this file does not
# decode PNGs, compute PNG/metrics or copy the codec. Without the variable the artifact test skips:
# it is an optional author check, not the coordinator's mandatory gate.


def _direct_sum(field, geometry):
    """Independent separable Fresnel convolution sum (float64, no FFT, no production helpers)."""
    lam, z = geometry.wavelength_m, geometry.distance_m
    dy, dx = geometry.input_pitch_m
    h, w = field.shape[-2:]
    v = (np.arange(h) - h // 2) * dy
    u = (np.arange(w) - w // 2) * dx
    y = (np.arange(h) - h // 2) * lam * z / (h * dy)
    x = (np.arange(w) - w // 2) * lam * z / (w * dx)
    a_y = np.exp(1j * np.pi * (y[:, None] - v[None, :]) ** 2 / (lam * z))
    a_x = np.exp(1j * np.pi * (x[:, None] - u[None, :]) ** 2 / (lam * z))
    pref = cmath.exp(1j * 2 * np.pi * z / lam) / (1j * lam * z) * dy * dx
    return pref * (a_y @ np.asarray(field, dtype=np.complex128) @ a_x.T)


def check_decoded_doe(phase):
    """Validate one decoded DOE phase on the four official planes; returns per-plane numbers.

    Uniform illumination A=1 (contract v1.1). Raises AssertionError on any violation.
    """
    assert isinstance(phase, torch.Tensor) and phase.shape == (1, 1, N, N), phase.shape
    assert phase.dtype in (torch.float32, torch.float64) and torch.isfinite(phase).all()
    assert phase.min() >= -math.pi - 1e-6 and phase.max() <= math.pi + 1e-6
    phase = phase.double()
    field = torch.polar(torch.ones_like(phase), phase)
    p_in = (field.abs() ** 2).sum().item() * CASE.input_pitch_m[0] * CASE.input_pitch_m[1]
    rows = []
    for k in range(4):
        geometry = CASE.plane(k)
        out = propagate_fresnel(field, geometry)
        amp = field_observables(out.field).amplitude
        assert torch.isfinite(amp).all() and amp.max() > 0
        assert out.output_pitch_m == geometry.output_pitch_m(N, N)
        p_out = (amp**2).sum().item() * out.output_pitch_m[0] * out.output_pitch_m[1]
        ref = _direct_sum(field.numpy()[0, 0], geometry)
        err = float(np.linalg.norm(out.field.numpy()[0, 0] - ref) / np.linalg.norm(ref))
        rows.append({"z_m": geometry.distance_m, "pitch_um": out.output_pitch_m[1] * 1e6,
                     "power_rel_err": abs(p_out / p_in - 1), "direct_sum_rel_l2": err,
                     "amp_max": amp.max().item(), "amp_mean": amp.mean().item()})
        assert rows[-1]["power_rel_err"] < 1e-12 and err < 1e-11, rows[-1]
    return rows


def physical_report_for_doe(phase):
    """Protocol for a real decoded DOE: exact invariants (check_decoded_doe) + diagnostics.

    Diagnostics have no pass/fail threshold: comparison with the continuous full-fill pixelated SLM
    reference (amplitude, normalised shape, in-window power ratio), dark-pixel fraction (where the
    amplitude gradient is ill-conditioned in float32) and peak position on each plane.
    """
    rows = check_decoded_doe(phase)
    ph = phase.double().numpy()[0, 0]
    field = torch.polar(torch.ones_like(phase.double()), phase.double())
    for row, k in zip(rows, range(4)):
        g = CASE.plane(k)
        amp = np.abs(propagate_fresnel(field, g).field.numpy()[0, 0])
        ref = np.abs(continuous_pixel_reference(ph, g))
        row.update({
            "peak_yx": divmod(int(amp.argmax()), N),
            "dark_fraction": float((amp < 1e-3 * amp.max()).mean()),
            "pixel_ref_amp_rel_l2": _rel(amp, ref),
            "pixel_ref_shape_rel_l2": _rel(amp / amp.max(), ref / ref.max()),
            "pixel_ref_window_power_ratio": float((ref**2).sum() / (amp**2).sum()),
        })
    return rows


def _random_phase(seed):
    gen = torch.Generator().manual_seed(seed)
    return ((2 * torch.rand(1, 1, N, N, generator=gen, dtype=torch.float64) - 1) * math.pi)


@pytest.mark.parametrize("make_phase", [multifocal_phase, lambda: _random_phase(0)], ids=["multifocal", "random"])
def test_check_decoded_doe_on_known_phases(make_phase):
    rows = check_decoded_doe(make_phase())
    assert [r["z_m"] for r in rows] == list(CASE.distances_m)


def test_check_decoded_doe_rejects_bad_phases():
    good = multifocal_phase()
    for bad in (good[0], good.to(torch.complex128), good * 2, good.clone().fill_(float("nan"))):
        with pytest.raises(AssertionError):
            check_decoded_doe(bad)


def test_exported_doe_artifact():
    path = os.environ.get("FRESNEL_DECODED_PHASE")
    if not path:
        pytest.skip("set FRESNEL_DECODED_PHASE to a decoded DOE phase (.npy/.npz) from the export pipeline")
    data = np.load(Path(path), allow_pickle=False)
    array = data["phase_rad"] if isinstance(data, np.lib.npyio.NpzFile) else data
    phase = torch.from_numpy(np.asarray(array, dtype=np.float64)).reshape(1, 1, N, N)
    raw = Path(path).read_bytes()
    print(f"\nartifact {path} sha256={hashlib.sha256(raw).hexdigest()}")
    for row in physical_report_for_doe(phase):
        print(row)


# --- extended diagnostics: families of deterministic test DOEs (#14B, stage B) ----------------
#
# Each family is a phase mask [1,1,128,128] in [-pi, pi) with uniform illumination A = 1.
# Independent continuous reference for ANY mask: the Fresnel integral of a pixelated SLM with
# full-fill rectangular pixels (sample-and-hold). The transmission is constant inside each cell and
# the Fresnel kernel is separable, so the cell integral is exactly K_y @ exp(i*phase) @ K_x.T with
# K computed by Gauss-Legendre quadrature over each pixel, evaluated at the same output points.
# It shares the paraxial (Fresnel) approximation but not the point sampling, the FFT, the chirps
# or any production helper. Separable families also carry 1D factors a(v) + b(u) for a cross-check. No aliasing threshold or
# crop is implied by any number below; they are diagnostics.

_V = (np.arange(N) - N // 2) * CASE.input_pitch_m[0]  # metres, centred integer grid (contract)
_U = (np.arange(N) - N // 2) * CASE.input_pitch_m[1]
_LAM = CASE.wavelength_m


def _wrap(phi):
    return np.mod(phi + np.pi, 2 * np.pi) - np.pi


def _lens_1d(s, f):
    return -np.pi * s**2 / (_LAM * f)


def doe_families():
    """name -> (phase [N,N] float64 in [-pi,pi), (a(v), b(u)) or None if not separable)."""
    vv, uu = np.meshgrid(_V, _U, indexing="ij")
    r_ap = N // 2 * CASE.input_pitch_m[1]
    fam = {}

    def sep(a, b):
        return _wrap(a[:, None] + b[None, :]), (a, b)

    fam["lens f=25.3mm"] = sep(_lens_1d(_V, 25.3e-3), _lens_1d(_U, 25.3e-3))
    fam["tilt (fy,fx)=(-1,1.5)e4/m"] = sep(2 * np.pi * -1e4 * _V, 2 * np.pi * 1.5e4 * _U)
    fam["lens f=25.3mm + tilt"] = sep(_lens_1d(_V, 25.3e-3) + 2 * np.pi * -1e4 * _V,
                                      _lens_1d(_U, 25.3e-3) + 2 * np.pi * 1.5e4 * _U)
    fam["astig+cubic (separable aberration)"] = sep(
        _lens_1d(_V, 26.0e-3) + 1.5 * np.pi * (_V / r_ap) ** 3,
        _lens_1d(_U, 24.6e-3) - 1.0 * np.pi * (_U / r_ap) ** 3,
    )
    fam["checkerboard pi (Nyquist)"] = sep(np.pi * np.arange(N, dtype=float), np.pi * np.arange(N, dtype=float))
    fam["grating period 3 px"] = sep(np.zeros(N), 2 * np.pi * np.arange(N) / 3)

    bifocal = np.exp(1j * (_lens_1d(vv, 24.1e-3) + _lens_1d(uu, 24.1e-3))) + np.exp(
        1j * (_lens_1d(vv, 27.7e-3) + _lens_1d(uu, 27.7e-3)))
    fam["bifocal lens 24.1+27.7mm"] = (np.angle(bifocal), None)
    fam["multifocal 4 points"] = (multifocal_phase().numpy()[0, 0], None)
    rho2 = (vv**2 + uu**2) / r_ap**2
    fam["spherical+coma (non-separable)"] = (
        _wrap(_lens_1d(vv, 25.3e-3) + _lens_1d(uu, 25.3e-3) + 3 * np.pi * rho2**2 + 2 * np.pi * rho2 * uu / r_ap), None)
    theta = np.arctan2(vv, uu)
    fam["vortex l=3 + lens + grating"] = (
        _wrap(3 * theta + _lens_1d(vv, 26.5e-3) + _lens_1d(uu, 26.5e-3) + 2 * np.pi * uu / (8 * CASE.input_pitch_m[1])), None)
    fam["seeded random U(-pi,pi)"] = (_random_phase(0).numpy()[0, 0], None)
    return fam


def _cell_kernel(out_axis, in_axis, pitch, z, order):
    """K[p, n] = integral over pixel n of exp(i*pi*(x_p - s)^2/(lambda*z)) ds (Gauss-Legendre)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s = in_axis[None, :, None] + nodes[None, None, :] * pitch / 2
    integrand = np.exp(1j * np.pi * (out_axis[:, None, None] - s) ** 2 / (_LAM * z))
    return (integrand * weights).sum(-1) * pitch / 2


def _cell_kernels(geometry, order):
    z = geometry.distance_m
    dy_o, dx_o = geometry.output_pitch_m(N, N)
    y = (np.arange(N) - N // 2) * dy_o
    x = (np.arange(N) - N // 2) * dx_o
    pref = cmath.exp(1j * 2 * np.pi * z / _LAM) / (1j * _LAM * z)
    return pref, _cell_kernel(y, _V, CASE.input_pitch_m[0], z, order), _cell_kernel(x, _U, CASE.input_pitch_m[1], z, order)


def continuous_pixel_reference(phase, geometry, order=32):
    """Continuous Fresnel field of a full-fill pixelated phase-only SLM showing `phase` [N,N]."""
    pref, k_y, k_x = _cell_kernels(geometry, order)
    return pref * (k_y @ np.exp(1j * np.asarray(phase)) @ k_x.T)


def continuous_pixel_reference_separable(a, b, geometry, order=32):
    """Same integral for a separable mask exp(i(a(v)+b(u))), as a product of two 1D integrals."""
    pref, k_y, k_x = _cell_kernels(geometry, order)
    return pref * (k_y @ np.exp(1j * a))[:, None] * (k_x @ np.exp(1j * b))[None, :]


def quantize_uniform(phase, levels):
    """Generic uniform phase quantiser with `levels` steps on [-pi, pi) (NOT the project DOE codec)."""
    k = np.mod(np.floor((phase + np.pi) / (2 * np.pi) * levels + 0.5), levels)
    return -np.pi + 2 * np.pi * k / levels


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def family_diagnostics(name, phase, factors):
    """Per-plane numbers for one family; asserts only exact invariants, not quality thresholds."""
    phi = torch.from_numpy(phase)[None, None]
    field = torch.polar(torch.ones_like(phi), phi)
    p_in = float((field.abs() ** 2).sum()) * CASE.input_pitch_m[0] * CASE.input_pitch_m[1]
    quant = {levels: torch.from_numpy(quantize_uniform(phase, levels))[None, None] for levels in (256, 64, 16)}
    rows = []
    for k in range(4):
        g = CASE.plane(k)
        out = propagate_fresnel(field, g)
        u_out = out.field.numpy()[0, 0]
        amp = np.abs(u_out)
        assert np.isfinite(u_out).all()
        assert out.output_pitch_m == g.output_pitch_m(N, N)
        p_out = float((amp**2).sum()) * out.output_pitch_m[0] * out.output_pitch_m[1]
        row = {
            "family": name, "z_mm": round(g.distance_m * 1e3, 1), "pitch_um": out.output_pitch_m[1] * 1e6,
            "power_rel_err": abs(p_out / p_in - 1),
            "direct_sum_rel_l2": _rel(u_out, _direct_sum(field.numpy()[0, 0], g)),
            "peak_yx": divmod(int(amp.argmax()), N),
        }
        for levels, q in quant.items():
            uq = propagate_fresnel(torch.polar(torch.ones_like(q), q), g).field.numpy()[0, 0]
            row[f"quant{levels}_amp_rel_l2"] = _rel(np.abs(uq), amp)
        ref = continuous_pixel_reference(phase, g)
        row["pixel_ref_amp_rel_l2"] = _rel(amp, np.abs(ref))
        row["pixel_ref_shape_rel_l2"] = _rel(amp / amp.max(), np.abs(ref) / np.abs(ref).max())
        row["pixel_ref_window_power_ratio"] = float((np.abs(ref) ** 2).sum() / (amp**2).sum())
        row["pixel_ref_quadrature_check"] = _rel(continuous_pixel_reference(phase, g, order=16), ref)
        if factors is not None:
            row["pixel_ref_separable_check"] = _rel(continuous_pixel_reference_separable(*factors, g), ref)
        rows.append(row)
    # Gradients of neutral four-plane functionals, float64 vs float32. Intensity |U|^2 is smooth and
    # tests the propagator; amplitude |U| is the v1.1 observable but is non-smooth at U = 0, where
    # the direction U/|U| of its gradient is set by rounding noise (diagnostic only, no threshold).
    weights = torch.from_numpy(np.random.default_rng(1).random((4, N, N)))
    for kind in ("intensity", "amplitude"):
        grads = {}
        for dtype in (torch.float64, torch.float32):
            p = phi.to(dtype).clone().requires_grad_(True)
            f = torch.polar(torch.ones_like(p), p)
            total = 0
            for k in range(4):
                obs = field_observables(propagate_fresnel(f, CASE.plane(k)).field)
                total = total + (getattr(obs, kind)[0, 0] * weights[k].to(dtype)).sum()
            (grads[dtype],) = torch.autograd.grad(total, p)
        g64, g32 = grads[torch.float64].numpy(), grads[torch.float32].double().numpy()
        for row in rows:
            row[f"grad_{kind}_finite"] = bool(np.isfinite(g64).all() and np.isfinite(g32).all())
            row[f"grad_{kind}_f32_vs_f64"] = _rel(g32, g64)
    for row, k in zip(rows, range(4)):
        amp = np.abs(propagate_fresnel(field, CASE.plane(k)).field.numpy()[0, 0])
        row["dark_fraction"] = float((amp < 1e-3 * amp.max()).mean())  # pixels below 1e-3 of the peak
    return rows


FAMILIES = doe_families()


@pytest.mark.parametrize("name", list(FAMILIES))
def test_family_invariants(name):
    phase, factors = FAMILIES[name]
    assert phase.shape == (N, N) and phase.min() >= -np.pi and phase.max() <= np.pi  # contract: [-pi, pi]
    for row in family_diagnostics(name, phase, factors):
        assert row["power_rel_err"] < 1e-12, row
        assert row["direct_sum_rel_l2"] < 1e-11, row
        assert row["grad_intensity_finite"] and row["grad_amplitude_finite"], row
        assert row["grad_intensity_f32_vs_f64"] < 1e-5, row  # propagator is stable in float32
        # Coarser steps never change the field less (equal only when the phases are exactly representable).
        assert row["quant256_amp_rel_l2"] <= row["quant16_amp_rel_l2"] + 1e-12, row
        assert row["pixel_ref_quadrature_check"] < 1e-6, row  # the reference itself is converged
        if factors is not None:
            assert row["pixel_ref_separable_check"] < 1e-12, row  # 2D form == product of 1D integrals


def test_pixel_reference_reduces_to_direct_sum_for_point_pixels():
    # One quadrature node at the pixel centre turns the cell integral into the point sum dx*f(u_n):
    # the reference then equals the production operator, isolating the pixel-aperture effect.
    phase = FAMILIES["multifocal 4 points"][0]  # any mask, not only separable ones
    g = CASE.plane(1)
    field = torch.polar(torch.ones(1, 1, N, N, dtype=torch.float64), torch.from_numpy(phase)[None, None])
    point = continuous_pixel_reference(phase, g, order=1)
    assert _rel(propagate_fresnel(field, g).field.numpy()[0, 0], point) < 1e-12


def test_lens_and_lens_tilt_focus_positions():
    # A lens f focuses on axis at z = f; adding a tilt (fy, fx) moves the focus to (lam*f*fy, lam*f*fx),
    # i.e. to pixel round(lam*f*f_i / pitch_out) + N//2 on the 25.3 mm plane. A pure tilt without a lens
    # does not focus (Fresnel number of the aperture ~32): its shifted aperture shadow wraps around the
    # single-FFT window, which is reported as a diagnostic, not tested as a position.
    lens_peaks = [r["peak_yx"] for r in family_diagnostics("lens", *FAMILIES["lens f=25.3mm"])]
    assert lens_peaks[1] == (N // 2, N // 2)
    row = family_diagnostics("lens+tilt", *FAMILIES["lens f=25.3mm + tilt"])[1]
    pitch = row["pitch_um"] * 1e-6
    expected = (round(_LAM * 25.3e-3 * -1e4 / pitch) + N // 2, round(_LAM * 25.3e-3 * 1.5e4 / pitch) + N // 2)
    assert row["peak_yx"] == expected, (row, expected)


def test_family_report(capsys):
    """Run with -s to print the markdown table used in docs/modules/fresnel_validation.md."""
    if not os.environ.get("FRESNEL_REPORT"):
        pytest.skip("set FRESNEL_REPORT=1 to print the family report")
    keys = ["z_mm", "power_rel_err", "direct_sum_rel_l2", "quant256_amp_rel_l2", "quant64_amp_rel_l2",
            "quant16_amp_rel_l2", "grad_intensity_f32_vs_f64", "grad_amplitude_f32_vs_f64", "dark_fraction",
            "pixel_ref_amp_rel_l2", "pixel_ref_shape_rel_l2", "pixel_ref_window_power_ratio", "peak_yx"]
    with capsys.disabled():
        print("\n| family | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for name, (phase, factors) in FAMILIES.items():
            for row in family_diagnostics(name, phase, factors):
                cells = []
                for k in keys:
                    v = row.get(k, "—")
                    cells.append(f"{v:.2e}" if isinstance(v, float) and k != "z_mm" else str(v))
                print(f"| {name} | " + " | ".join(cells) + " |")


def test_pixel_reference_matches_sinc_power_for_nyquist_checkerboard():
    # A pi-checkerboard puts all light at spatial frequency 1/(2*pitch) on both axes; a full-fill pixel
    # transmits sinc(1/2)^2 of it per axis into that order. The point-sampled model keeps everything in
    # the window, so the in-window power ratio must be sinc(1/2)^4 = 0.164 (light in other orders is
    # outside the window). Measured 0.165-0.166: an analytic check of the reference itself.
    rows = family_diagnostics("checkerboard", *FAMILIES["checkerboard pi (Nyquist)"])
    for row in rows:
        assert row["pixel_ref_window_power_ratio"] == pytest.approx(np.sinc(0.5) ** 4, rel=2e-2), row


def test_physical_report_protocol_on_known_phase():
    rows = physical_report_for_doe(torch.from_numpy(FAMILIES["multifocal 4 points"][0])[None, None])
    assert [r["peak_yx"] for r in rows] == [(84, 84), (83, 45), (46, 46), (47, 81)]
    for row in rows:
        assert 0 < row["pixel_ref_window_power_ratio"] < 1  # pixel aperture sends light out of the window


# --- illustrations for the validation report (#14B stage D) -----------------------------------
#
# FRESNEL_FIGURES=artifacts/fresnel_validation python -m pytest tests/core/test_fresnel_physical.py -k figures
# Writes PNG montages (not committed: artifacts/ is git-ignored). Each tile is |U| scaled by its own
# maximum (display only). Row 1: point-sampled single-FFT model (the case operator); row 2: continuous
# full-fill pixelated-SLM reference at the same output points; column 0: the DOE phase.


def _tile(image, scale=2):
    from PIL import Image

    image = np.asarray(image, dtype=np.float64)
    lo, hi = image.min(), image.max()
    u8 = np.floor(255 * (image - lo) / (hi - lo if hi > lo else 1) + 0.5).astype(np.uint8)
    return Image.fromarray(u8, mode="L").resize((N * scale, N * scale), Image.NEAREST)


def render_figures(out_dir):
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("multifocal 4 points", "lens f=25.3mm + tilt", "grating period 3 px", "checkerboard pi (Nyquist)"):
        phase, _ = FAMILIES[name]
        field = torch.polar(torch.ones(1, 1, N, N, dtype=torch.float64), torch.from_numpy(phase)[None, None])
        shown = np.mod(phase, 2 * np.pi)  # display only: -pi and +pi are the same phase, show both as pi
        rows = [[_tile(shown)], [_tile(shown)]]
        for k in range(4):
            g = CASE.plane(k)
            rows[0].append(_tile(np.abs(propagate_fresnel(field, g).field.numpy()[0, 0])))
            rows[1].append(_tile(np.abs(continuous_pixel_reference(phase, g))))
        size, pad, head = 2 * N, 6, 18
        canvas = Image.new("L", (5 * size + 6 * pad, 2 * size + 3 * pad + head), 255)
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 3), f"{name}: phase | |U| at 24.1/25.3/26.5/27.7 mm; top: single-FFT model, bottom: pixelated SLM", fill=0)
        for r, row in enumerate(rows):
            for c, tile in enumerate(row):
                canvas.paste(tile, (pad + c * (size + pad), head + pad + r * (size + pad)))
        path = out_dir / ("fig_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") + ".png")
        canvas.save(path)
        written.append(path)
    return written


def test_figures():
    out = os.environ.get("FRESNEL_FIGURES")
    if not out:
        pytest.skip("set FRESNEL_FIGURES=<dir> to render the validation illustrations")
    for path in render_figures(out):
        print(path, hashlib.sha256(path.read_bytes()).hexdigest()[:16])


# --- contract amplitude loss x single-FFT operator (#14B, independent analysis) ----------------
#
# Independent of #3/#5 code: the v1.1 training loss is re-stated here from the contract text,
#   L_c = mean((A/max(A) - T)^2), max per plane,
# and a minimal own phase optimiser (torch.optim.Adam on the phase) is used only for diagnostics.
# Facts proven below are exact identities; the optional study prints numbers for the report:
#   FRESNEL_LOSS_STUDY=1 python -m pytest tests/core/test_fresnel_physical.py -k loss_operator_study -s


def _contract_loss(a, t):
    return ((a / a.amax(dim=(-2, -1), keepdim=True) - t) ** 2).mean()


def _lsq_gain_nmse(a, t):
    cos = (a * t).sum((-2, -1)) / (a.norm(dim=(-2, -1)) * t.norm(dim=(-2, -1)))
    return (1 - cos**2).mean()


def _one_minus_pearson(a, t):
    return (1 - _pearson(a, t)).mean()


def _pearson(a, t):
    ac, tc = a - a.mean((-2, -1), keepdim=True), t - t.mean((-2, -1), keepdim=True)
    return (ac * tc).sum((-2, -1)) / (ac.norm(dim=(-2, -1)) * tc.norm(dim=(-2, -1)))


def _four_amplitudes(phase):
    f = torch.polar(torch.ones_like(phase), phase)
    return torch.stack([propagate_fresnel(f, CASE.plane(k)).field.abs()[:, 0] for k in range(4)], 1)


def _sparse_quartet():
    t = np.zeros((1, 4, N, N))
    for k, (y, x) in enumerate([(30, 30), (30, 78), (78, 78), (78, 30)]):
        t[0, k, y:y + 20, x:x + 20] = 1.0  # 2.4 % bright pixels per plane
    return torch.from_numpy(t)


def _dense_quartet():
    yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    t = np.stack([0.5 + 0.5 * np.cos(3 * np.pi * xx) * np.exp(-yy**2), np.exp(-((xx - .3) ** 2 + yy**2) / .2),
                  0.5 + 0.5 * np.sin(2 * np.pi * (xx + yy)), np.clip(1 - np.sqrt(xx**2 + yy**2), 0, 1)])
    return torch.from_numpy(t / t.max(axis=(1, 2), keepdims=True))[None]


def test_parseval_makes_l2_amplitude_normalisation_a_constant():
    # Proven: sum |U_out|^2 dy_o dx_o = sum |U_in|^2 dy dx = H*W*dy*dx for A = 1, for ANY phase.
    # Hence A/||A|| = A * const per plane: an L2-normalised amplitude loss needs no data-dependent scale.
    for seed in (0, 1):
        a = _four_amplitudes(_random_phase(seed))
        for k in range(4):
            dy_o, dx_o = CASE.plane(k).output_pitch_m(N, N)
            power = float((a[0, k] ** 2).sum()) * dy_o * dx_o
            assert power == pytest.approx(N * N * CASE.input_pitch_m[0] * CASE.input_pitch_m[1], rel=1e-12)


def test_lsq_gain_nmse_closed_form():
    # Proven: min_s ||s*A - T||^2/||T||^2 = 1 - cos^2(A, T), attained at s* = <A,T>/||A||^2.
    t = _dense_quartet()
    a = _four_amplitudes(_random_phase(3))
    s_star = (a * t).sum((-2, -1), keepdim=True) / (a**2).sum((-2, -1), keepdim=True)
    direct = (((s_star * a - t) ** 2).sum((-2, -1)) / (t**2).sum((-2, -1))).mean()
    assert float(direct) == pytest.approx(float(_lsq_gain_nmse(a, t)), rel=1e-12)
    for factor in (0.9, 1.1):  # s* is the minimum
        assert float((((factor * s_star * a - t) ** 2).sum((-2, -1)) / (t**2).sum((-2, -1))).mean()) > float(direct)


def test_contract_loss_spike_limit_and_peak_gradient():
    # Proven: with a = A/max(A), L_c = mean(a^2) - 2 mean(a T) + mean(T^2). A single-pixel spike a = e_p gives
    # L_c = mean(T^2) + (1 - 2 T_p)/(H W): the "trivial" level. Raising only the peak m with all other A fixed:
    # dL_c/dm = -(2/(H W m)) * sum_{i != p} a_i (a_i - T_i) < 0 whenever the normalised prediction exceeds the
    # target in that weighted sense -> the loss rewards inflating the maximum (concentrating light).
    t = _sparse_quartet()[0, 0]
    spike = torch.zeros(N, N, dtype=torch.float64)
    spike[40, 40] = 1.0
    assert float(_contract_loss(spike, t)) == pytest.approx(float((t**2).mean() + (1 - 2 * t[40, 40]) / (N * N)), rel=1e-12)
    amp = _four_amplitudes(_random_phase(4))[0, 0].clone()
    p = divmod(int(amp.argmax()), N)
    m = amp[p].clone().requires_grad_(True)
    rest = amp.clone()
    rest[p] = 0
    full = rest + m * torch.nn.functional.one_hot(torch.tensor(p[0] * N + p[1]), N * N).reshape(N, N).double()
    (grad,) = torch.autograd.grad(_contract_loss(full, t), m)
    a = (amp / amp.max()).flatten()
    mask = torch.ones(N * N, dtype=torch.bool)
    mask[p[0] * N + p[1]] = False
    expected = -(2 / (N * N * m.item())) * float((a[mask] * (a[mask] - t.flatten()[mask])).sum())
    assert float(grad) == pytest.approx(expected, rel=1e-10)
    assert float(grad) < 0  # random-phase speckle on a sparse target: inflating the peak lowers L_c


def test_abs_gradient_is_bounded_with_zero_subgradient_at_zero():
    # Proven: |U| is 1-Lipschitz, d|U| = Re(conj(U)/|U| dU) -> |d|U|/dphi| <= |dU/dphi| (no blow-up);
    # at U = 0 the subdifferential is the unit disk and torch returns 0 (no signal to brighten a zero pixel).
    u = torch.zeros(1, dtype=torch.complex128, requires_grad=True)
    u.abs().sum().backward()
    assert float(u.grad.abs()) == 0.0
    phase = _random_phase(5).requires_grad_(True)
    f = torch.polar(torch.ones_like(phase), phase)
    out = propagate_fresnel(f, CASE.plane(0)).field[0, 0, 70, 70]
    (g_abs,) = torch.autograd.grad(out.abs(), phase, retain_graph=True)
    (g_re,) = torch.autograd.grad(out.real, phase, retain_graph=True)
    (g_im,) = torch.autograd.grad(out.imag, phase)
    assert float(g_abs.norm()) <= float(torch.sqrt(g_re.norm() ** 2 + g_im.norm() ** 2)) * (1 + 1e-12)


def _optimise_phase(targets, loss_fn, seed, steps=300, lr=0.05, planes=(0, 1, 2, 3), init=None):
    gen = torch.Generator().manual_seed(seed)
    phase = init.clone() if init is not None else (2 * torch.rand(1, 1, N, N, generator=gen, dtype=torch.float64) - 1) * math.pi
    phase.requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    idx = list(planes)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(_four_amplitudes(phase)[:, idx], targets[:, idx]).backward()
        opt.step()
    with torch.no_grad():
        return phase.detach(), _four_amplitudes(phase)


def _study_row(tag, a, t):
    bright = ((a / a.amax(dim=(-2, -1), keepdim=True)) > 0.5).double().mean((-2, -1))[0]
    top_share = ((a**2).amax((-2, -1)) / (a**2).sum((-2, -1)))[0]
    r = _pearson(a, t)[0]
    lc, triv = float(_contract_loss(a, t)), float((t**2).mean())
    return (f"| {tag} | {lc:.4f} | {lc / triv:.2f} | {' '.join(f'{v:+.2f}' for v in r.tolist())} | {float(r.mean()):+.3f} "
            f"| {' '.join(f'{100 * v:.1f}' for v in bright.tolist())} | {float(top_share.mean()) * 100:.2f} |")


def test_loss_operator_study(capsys):
    if not os.environ.get("FRESNEL_LOSS_STUDY"):
        pytest.skip("set FRESNEL_LOSS_STUDY=1 to run the ~1 min loss x operator study")
    losses = (("contract max-MSE", _contract_loss), ("LSQ-gain NMSE", _lsq_gain_nmse), ("1 - Pearson", _one_minus_pearson))
    with capsys.disabled():
        for tname, t in (("sparse boxes", _sparse_quartet()), ("dense smooth", _dense_quartet())):
            print(f"\n{tname}: mean(T^2) = {float((t**2).mean()):.4f}, target bright = "
                  f"{float((t > 0.5).double().mean()) * 100:.1f} %")
            print("| run | L_c | L_c / mean(T²) | Pearson per plane | mean Pearson | A/max>0.5, % per plane | top-pixel power share, % |")
            print("|---|---|---|---|---|---|---|")
            for lname, fn in losses:
                for seed in (0, 1, 2):
                    print(_study_row(f"{lname}, seed {seed}", _optimise_phase(t, fn, seed)[1], t))
            for k in range(4):
                print(_study_row(f"single plane {k} only, LSQ", _optimise_phase(t, _lsq_gain_nmse, 0, planes=(k,))[1], t))
            start, a = _optimise_phase(t, _lsq_gain_nmse, 0)
            print(_study_row("LSQ solution (start)", a, t))
            print(_study_row("-> +300 steps of contract loss", _optimise_phase(t, _contract_loss, 0, init=start)[1], t))


def test_lsq_zero_overlap_stationarity_vs_cosine():
    # Proven: grad(1 - cos^2) = -2 cos grad(cos) vanishes at cos = 0; grad(1 - cos) = -grad(cos) does not.
    # Zero overlap is built explicitly: amplitude only where the target is zero.
    t = torch.zeros(N, N, dtype=torch.float64)
    t[:64] = 1.0
    a = torch.zeros(N, N, dtype=torch.float64)
    a[64:] = torch.rand(64, N, generator=torch.Generator().manual_seed(0), dtype=torch.float64) + 0.1
    a.requires_grad_(True)
    cos = (a * t).sum() / (a.norm() * t.norm())
    (g_lsq,) = torch.autograd.grad(1 - cos**2, a, retain_graph=True)
    (g_cos,) = torch.autograd.grad(1 - cos, a)
    assert cos.item() == 0.0
    assert float(g_lsq.abs().max()) == 0.0
    assert float(g_cos.abs().max()) > 0  # pushes amplitude towards the target support
