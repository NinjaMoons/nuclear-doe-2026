"""Mandatory independent DISCRETE Fresnel physics gate; no production oracle.

Direct quadrature uses exp(i*pi*distance_squared/(lambda*z)), not FFT/chirp
helpers from the implementation. Passing does not certify continuous optics.
"""
import importlib
import math

import numpy as np
import pytest
import torch

from doe_core.contracts_v1 import CaseConfigV1, FresnelGeometry

pytestmark = pytest.mark.integration


def _propagator():
    try:
        return importlib.import_module("doe_core.fresnel").propagate_fresnel
    except ModuleNotFoundError as error:
        if error.name != "doe_core.fresnel":
            raise
        pytest.fail("independent physics gate requires real doe_core.fresnel (no skip)")


def _quadrature_parts(shape, geometry):
    # Deliberately do NOT call geometry.output_pitch_m or production grid helpers.
    h, w = shape
    wave, z = geometry.wavelength_m, geometry.distance_m
    dy, dx = geometry.input_pitch_m
    out_dy, out_dx = wave*z/(h*dy), wave*z/(w*dx)
    yin = (np.arange(h, dtype=np.float64) - h//2) * dy
    xin = (np.arange(w, dtype=np.float64) - w//2) * dx
    yout = (np.arange(h, dtype=np.float64) - h//2) * out_dy
    xout = (np.arange(w, dtype=np.float64) - w//2) * out_dx
    ky = np.exp(1j * np.pi * (yout[:, None] - yin[None, :])**2/(wave*z))
    kx = np.exp(1j * np.pi * (xout[:, None] - xin[None, :])**2/(wave*z))
    coefficient = np.exp(1j * (2*np.pi*z/wave))/(1j*wave*z) * dy*dx
    return ky, kx, coefficient, (out_dy, out_dx)


def _reference(field, geometry):
    ky, kx, coefficient, _ = _quadrature_parts(field.shape[-2:], geometry)
    return coefficient * (ky @ field @ kx.T)


def _relative_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def _asymmetric_field(shape, batch=2):
    rng = np.random.default_rng(531)
    return rng.normal(size=(batch, 1, *shape)) + 1j*rng.normal(size=(batch, 1, *shape))


@pytest.mark.parametrize("shape,pitch", [((4, 6), (12e-6, 9e-6)),
                                             ((5, 7), (8e-6, 11e-6))])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_independent_complex_field_rectangular_and_odd(shape, pitch, dtype):
    geometry = FresnelGeometry(532e-9, pitch, 0.0253)
    field = torch.as_tensor(_asymmetric_field(shape), dtype=dtype)
    expected = _reference(field.numpy().astype(np.complex128), geometry)
    result = _propagator()(field, geometry)
    limit = 3e-6 if dtype == torch.complex64 else 1e-11
    assert _relative_error(result.field.detach().numpy(), expected) < limit
    expected_pitch = _quadrature_parts(shape, geometry)[3]
    np.testing.assert_allclose(result.output_pitch_m, expected_pitch, rtol=1e-13, atol=0)
    assert result.distance_m == geometry.distance_m
    input_power = np.sum(np.abs(field.numpy())**2, axis=(-2, -1)) * math.prod(pitch)
    output_power = np.sum(np.abs(result.field.detach().numpy())**2, axis=(-2, -1)) * math.prod(expected_pitch)
    np.testing.assert_allclose(output_power, input_power, rtol=5e-6 if dtype == torch.complex64 else 1e-12, atol=0)


@pytest.mark.parametrize("index", range(4))
def test_all_official_planes_against_direct_integral_kernel(index):
    geometry = CaseConfigV1().plane(index)
    field = _asymmetric_field((128, 128), batch=1)
    expected = _reference(field, geometry)
    result = _propagator()(torch.from_numpy(field), geometry)
    assert _relative_error(result.field.detach().numpy(), expected) < 1e-11
    np.testing.assert_allclose(result.output_pitch_m, _quadrature_parts((128,128),geometry)[3],
                               rtol=1e-13, atol=0)


@pytest.mark.parametrize("shape,pitch", [((4, 6), (12e-6, 9e-6)),
                                             ((5, 7), (8e-6, 11e-6))])
def test_amplitude_loss_phase_gradient_against_direct_sum(shape, pitch):
    geometry = FresnelGeometry(532e-9, pitch, 0.0277)
    rng = np.random.default_rng(307)
    phase = torch.tensor(rng.uniform(-2, 2, size=(1,1,*shape)), requires_grad=True)
    other_phase = phase.detach().clone().requires_grad_(True)
    weight = torch.tensor(rng.uniform(0.1, 1.5, size=(1,1,*shape)))
    actual = _propagator()(torch.exp(1j*phase), geometry).field
    ky, kx, coefficient, _ = _quadrature_parts(shape, geometry)
    reference = coefficient * (torch.from_numpy(ky) @ torch.exp(1j*other_phase) @ torch.from_numpy(kx.T.copy()))
    def loss(value):
        amplitude = value.abs()
        return ((amplitude/amplitude.amax(dim=(-2,-1),keepdim=True) - weight)**2).mean()
    grad = torch.autograd.grad(loss(actual), phase)[0]
    expected_grad = torch.autograd.grad(loss(reference), other_phase)[0]
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    torch.testing.assert_close(grad, expected_grad, rtol=1e-9, atol=1e-12)


def test_independent_oracle_rejects_shared_forward_faults():
    # A roundtrip that calls the same faulty forward twice can miss these errors.
    # Verify sensitivity of THIS acceptance comparison without editing production.
    geometry = FresnelGeometry(532e-9, (12e-6, 9e-6), 0.0253)
    field = torch.from_numpy(_asymmetric_field((4,6), batch=1))
    actual = _propagator()(field, geometry).field.detach().numpy()
    expected = _reference(field.numpy(), geometry)
    assert _relative_error(actual, expected) < 1e-11
    for faulty in (actual*2, actual.conj(), actual[..., ::-1], np.roll(actual, 1, axis=-1)):
        assert _relative_error(faulty, expected) > 1e-2
    # In particular, per-plane max would hide the *2 fault in the PNG roundtrip.
    np.testing.assert_allclose(np.abs(actual)/np.abs(actual).max(),
                               np.abs(2*actual)/np.abs(2*actual).max())
