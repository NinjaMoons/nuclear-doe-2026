"""Physical checks of the single-FFT Fresnel forward on the official case (Issue #14B).

One phase mask, four planes, amplitude observable |U| (contract v1.1-amplitude).
The expected outcome follows from optics, not from the implementation:
a thin lens exp(-i*pi*((u-x0)^2+(v-y0)^2)/(lambda*z0)) focuses a uniform wave to (y0, x0)
in the plane z0. A phase-only superposition of four such lenses must therefore put a
bright spot at its own target point in each of the four planes, at the coordinates given by
that plane's own output pitch lambda*z/(N*dx).
"""
import cmath
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
    for row in check_decoded_doe(phase):
        print(row)
