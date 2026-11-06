"""Numerical representation regression, independent of an optical propagator."""
import numpy as np
import pytest
import torch

from doe_core.observables import field_observables


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_amplitude_and_intensity_are_numerically_distinct(dtype):
    field = torch.tensor([[1, 2], [0, 1j]], dtype=dtype)
    amplitude, intensity = field_observables(field)
    torch.testing.assert_close(amplitude, torch.tensor([[1., 2.], [0., 1.]], dtype=field.real.dtype))
    torch.testing.assert_close(intensity, amplitude.square())
    target = torch.tensor([[0.5, 1.], [0., 0.5]], dtype=amplitude.dtype)
    assert torch.mean((amplitude / amplitude.max() - target)**2) == 0
    assert float(torch.mean((intensity / intensity.max() - target)**2)) == pytest.approx(0.03125)
    a_png = torch.floor(amplitude / amplitude.max() * 255 + 0.5).to(torch.uint8)
    i_png = torch.floor(intensity / intensity.max() * 255 + 0.5).to(torch.uint8)
    assert a_png.tolist() == [[128, 255], [0, 128]]
    assert i_png.tolist() == [[64, 255], [0, 64]]


def test_amplitude_zero_field_has_finite_zero_gradient():
    field = torch.tensor([0j, 1+2j], dtype=torch.complex128, requires_grad=True)
    field_observables(field).amplitude.sum().backward()
    assert torch.isfinite(field.grad).all()
    assert field.grad[0] == 0


def test_amplitude_gradcheck_away_from_origin():
    values = torch.tensor([1+2j, -0.5+0.3j], dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: field_observables(z).amplitude, (values,))


@pytest.mark.parametrize("bad", [torch.ones(2), torch.tensor([complex('nan')]), torch.empty(0, dtype=torch.complex128)])
def test_invalid_field_rejected(bad):
    with pytest.raises(ValueError):
        field_observables(bad)
