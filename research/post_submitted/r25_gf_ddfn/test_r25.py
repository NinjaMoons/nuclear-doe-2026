"""R25 tests (CPU). run: python -m pytest -q r25_gf_ddfn/test_r25.py  (PYTHONPATH = production src ; research/post_submitted)"""
import math

import numpy as np
import torch

from back_projection_stem.stem import BackProjectionStemUNet, inverse_fresnel
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.phase_codec import decode_phase, encode_phase
from r25_gf_ddfn.gf_ddfn_fixed import GF_DDFN_Fixed, STEQuantizeL8, adjoint_fresnel, raw
from r25_gf_ddfn.model_r25 import from_v2

CFG = CaseConfigV1()


def _rc(g, *s):
    return torch.complex(torch.randn(*s, generator=g, dtype=torch.float64), torch.randn(*s, generator=g, dtype=torch.float64))


def test_fixed_adjoint_is_exact_for_all_planes():
    g = torch.Generator().manual_seed(1)
    for k in range(4):
        U, V = _rc(g, 1, 1, 128, 128), _rc(g, 1, 1, 128, 128)
        lhs = (propagate_fresnel(U, CFG.plane(k)).field.conj() * V).sum()
        rhs = (U.conj() * adjoint_fresnel(V, CFG.plane(k))).sum()
        assert abs(lhs - rhs) / abs(lhs) < 1e-12


def test_inverse_is_physical_adjoint_and_inverse():
    g = torch.Generator().manual_seed(2)
    for k in range(4):
        geo = CFG.plane(k); dyo, dxo = geo.output_pitch_m(128, 128)
        U, V = _rc(g, 1, 1, 128, 128), _rc(g, 1, 1, 128, 128)
        lhs = (propagate_fresnel(U, geo).field.conj() * V).sum() * dyo * dxo
        rhs = (U.conj() * inverse_fresnel(V, geo)).sum() * 1e-10
        assert abs(lhs - rhs) / abs(lhs) < 1e-12
        assert float((inverse_fresnel(propagate_fresnel(U, geo).field, geo) - U).abs().max()) < 1e-12


def test_gemini_raw_lift_is_not_the_adjoint():
    g = torch.Generator().manual_seed(3); gem = raw.GF_DDFN().double()
    U, V = _rc(g, 1, 1, 128, 128), _rc(g, 1, 1, 128, 128)
    V4 = torch.zeros(1, 4, 128, 128, dtype=torch.complex128); V4[:, 3] = V[:, 0]
    GV = gem.adjoint_fresnel_lift(V4.to(torch.complex64)).to(torch.complex128)[:, 3:4]
    lhs = (propagate_fresnel(U, CFG.plane(3)).field.conj() * V).sum(); rhs = (U.conj() * GV).sum()
    assert abs(abs(lhs / rhs) - 0.8686) < 1e-3  # missing |s| on the z=27.7mm plane


def test_l8_quantizer_bit_exact_with_official_codec():
    ph = torch.cat([(torch.rand(50000, dtype=torch.float64) * 2 - 1) * math.pi,
                    torch.tensor([-math.pi, math.pi, 0.0], dtype=torch.float64),
                    torch.tensor([-math.pi + 2 * math.pi * (c + .5) / 255 for c in range(255)], dtype=torch.float64)])
    off = torch.from_numpy(decode_phase(encode_phase(ph.numpy())))
    assert torch.equal(STEQuantizeL8.apply(ph), off)
    assert not torch.equal(raw.STEQuantize8Bit.apply(ph), off)
    x = ph.clone().requires_grad_(True); STEQuantizeL8.apply(x).sum().backward(); assert torch.equal(x.grad, torch.ones_like(ph))


def test_dpsi_model_equals_v2_at_init_and_trains_dpsi():
    torch.manual_seed(0); v2 = BackProjectionStemUNet().double(); m = from_v2(v2).double()
    T = torch.rand(2, 4, 1, 128, 128, dtype=torch.float64)
    assert torch.equal(m(T), v2(T))
    loss = m(T).sin().mean(); loss.backward()
    assert m.dpsi_head.weight.grad.abs().sum() > 0 and torch.isfinite(m.dpsi_head.weight.grad).all()


def test_gemini_fixed_shapes_batch1_and_4():
    m = GF_DDFN_Fixed().eval()
    for b in (1, 4):
        with torch.no_grad():
            y = m(torch.rand(b, 4, 128, 128))
        assert y.shape == (b, 1, 128, 128) and torch.isfinite(y).all()
        codes = np.floor(255 * (y.double().numpy() + math.pi) / (2 * math.pi) + 0.5)
        assert np.allclose(decode_phase(codes.astype(np.uint8)), y.double().numpy(), atol=1e-6)
