"""Address tests for the research back-projection stem. Run from the worktree root:
    python -m unittest back_projection_stem.test_stem -v
Optional env BPS_BEST_PT=<path to production best.pt> for the strict-load compatibility check."""
import math
import os
import unittest

import numpy as np
import torch

from doe_core.contracts_v1 import CaseConfigV1, FresnelGeometry
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from doe_models.unet_multiplane import FourInputUNet, build_model

from back_projection_stem.r3_loss import r3_loss
from back_projection_stem.stem import (CHANNEL_ORDER, BackProjectionStemUNet, backprojection_features,
                                       build_stem_model, inverse_fresnel)

CFG = CaseConfigV1()
SMALL_GEOMS = [FresnelGeometry(wavelength_m=5.32e-7, input_pitch_m=(10e-6, 13e-6), distance_m=z)
               for z in (0.0241, 0.0277, 0.003)]


def rel(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


def rand_complex(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    return torch.complex(torch.randn(shape, generator=g, dtype=real), torch.randn(shape, generator=g, dtype=real))


def direct_forward_matrix(h, w, geom):
    """Independent forward from the continuous formula: explicit centred-index DFT sum, numpy, complex128."""
    lam, z = geom.wavelength_m, geom.distance_m
    dy, dx = geom.input_pitch_m
    dyo, dxo = lam * z / (h * dy), lam * z / (w * dx)
    m = np.arange(h) - h // 2
    n = np.arange(w) - w // 2
    yi, xi = m[:, None] * dy, n[None, :] * dx
    yo, xo = m[:, None] * dyo, n[None, :] * dxo
    cin = np.exp(1j * np.pi * (xi ** 2 + yi ** 2) / (lam * z)).ravel()
    cout = (np.exp(1j * 2 * np.pi * z / lam) / (1j * lam * z) * np.exp(1j * np.pi * (xo ** 2 + yo ** 2) / (lam * z)) * dy * dx).ravel()
    P, Q = np.meshgrid(m, n, indexing="ij")
    phase = -2j * np.pi * (np.outer(P.ravel(), P.ravel()) / h + np.outer(Q.ravel(), Q.ravel()) / w)
    return cout[:, None] * np.exp(phase) * cin[None, :]


class TestInverseFresnel(unittest.TestCase):
    def check_roundtrip(self, shape, geoms, dtype, tol):
        for i, g in enumerate(geoms):
            u = rand_complex(shape, dtype, 10 + i)
            fwd = propagate_fresnel(u, g).field
            self.assertLess(rel(inverse_fresnel(fwd, g), u), tol, f"F^-1(F(U)) {shape} z={g.distance_m} {dtype}")
            v = rand_complex(shape, dtype, 50 + i)
            self.assertLess(rel(propagate_fresnel(inverse_fresnel(v, g), g).field, v), tol, f"F(F^-1(V)) {shape} {dtype}")

    def test_roundtrip_small_odd_even_anisotropic(self):
        for shape in ((2, 1, 4, 6), (3, 1, 5, 7)):
            self.check_roundtrip(shape, SMALL_GEOMS, torch.complex64, 1e-5)
            self.check_roundtrip(shape, SMALL_GEOMS, torch.complex128, 1e-12)

    def test_roundtrip_128_official_planes(self):
        geoms = [CFG.plane(k) for k in range(4)]
        self.check_roundtrip((2, 1, 128, 128), geoms, torch.complex64, 1e-5)
        self.check_roundtrip((2, 1, 128, 128), geoms, torch.complex128, 1e-12)

    def test_independent_direct_sum_reference(self):
        for h, w in ((4, 6), (5, 7)):
            for g in SMALL_GEOMS:
                M = direct_forward_matrix(h, w, g)
                u = rand_complex((1, 1, h, w), torch.complex128, 3)
                fwd = propagate_fresnel(u, g).field[0, 0].numpy().ravel()
                self.assertLess(np.linalg.norm(M @ u[0, 0].numpy().ravel() - fwd) / np.linalg.norm(fwd), 1e-12,
                                "official forward != direct-sum formula")
                v = rand_complex((1, 1, h, w), torch.complex128, 4)
                ref = np.linalg.solve(M, v[0, 0].numpy().ravel())
                got = inverse_fresnel(v, g)[0, 0].numpy().ravel()
                self.assertLess(np.linalg.norm(got - ref) / np.linalg.norm(ref), 1e-10, f"inverse != solve {h}x{w}")
                Minv = np.stack([inverse_fresnel(torch.from_numpy(e.reshape(1, 1, h, w)), g)[0, 0].numpy().ravel()
                                 for e in np.eye(h * w, dtype=np.complex128)], axis=1)
                self.assertLess(np.abs(Minv @ M - np.eye(h * w)).max(), 1e-10)

    def test_dtype_device_and_rejects_real(self):
        for dt in (torch.complex64, torch.complex128):
            out = inverse_fresnel(rand_complex((1, 1, 8, 8), dt, 1), SMALL_GEOMS[0])
            self.assertEqual(out.dtype, dt)
            self.assertEqual(out.device.type, "cpu")
        with self.assertRaises(ValueError):
            inverse_fresnel(torch.zeros(1, 1, 8, 8), SMALL_GEOMS[0])


class TestFeatures(unittest.TestCase):
    def targets(self, b=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.floor(torch.rand((b, 4, 1, 128, 128), generator=g) * 255 + 0.5) / 255

    def test_shape_order_and_values(self):
        t = self.targets()
        f = backprojection_features(t)
        self.assertEqual(tuple(f.shape), (3, 12, 128, 128))
        self.assertEqual(f.dtype, torch.float32)
        self.assertEqual(CHANNEL_ORDER, ["T1", "T2", "T3", "T4", "Re1", "Im1", "Re2", "Im2", "Re3", "Im3", "Re4", "Im4"])
        for k in range(4):
            self.assertTrue(torch.allclose(f[:, k], t[:, k, 0] * 2 - 1))
            v = torch.complex(t[:, k].double(), torch.zeros_like(t[:, k].double()))
            b = inverse_fresnel(v, CFG.plane(k))[:, 0]
            b = b / b.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            self.assertLess(rel(f[:, 4 + 2 * k].double(), b.real), 1e-6)
            self.assertLess(rel(f[:, 5 + 2 * k].double(), b.imag), 1e-6)
            rms = (f[:, 4 + 2 * k].double() ** 2 + f[:, 5 + 2 * k].double() ** 2).mean(dim=(-2, -1))
            self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_amplitude_not_sqrt(self):
        t = self.targets(b=2, seed=1)
        f = backprojection_features(t)
        s = torch.sqrt(t)
        v = torch.complex(s[:, 0].double(), torch.zeros_like(s[:, 0].double()))
        b = inverse_fresnel(v, CFG.plane(0))[:, 0]
        b = b / b.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        self.assertGreater(rel(f[:, 4].double(), b.real), 1e-2, "features must use T, not sqrt(T)")

    def test_zero_and_constant_planes_are_finite(self):
        t = self.targets(b=2)
        t[0, 1] = 0.0
        t[1, 2] = 0.5
        f = backprojection_features(t)
        self.assertTrue(bool(torch.isfinite(f).all()))
        self.assertTrue(bool((f[0, 6:8] == 0).all()), "zero target plane -> zero back-projection channels")


class TestModel(unittest.TestCase):
    def test_body_matches_parent(self):
        parent = build_model(seed=0)
        t = TestFeatures().targets(b=2)
        x = t[:, :, 0] * 2 - 1
        self.assertTrue(torch.equal(BackProjectionStemUNet._unet(parent, x, t.shape[-2:]), parent(t)))

    def test_shapes_range_and_hparams(self):
        m = build_stem_model(seed=0)
        self.assertEqual(m.encoders[0][0].in_channels, 12)
        self.assertEqual(build_model(seed=0).encoders[0][0].in_channels, 4)
        out = m(TestFeatures().targets(b=3))
        self.assertEqual(tuple(out.shape), (3, 1, 128, 128))
        self.assertTrue(bool((out >= -math.pi).all()) and bool((out < math.pi).all()))
        self.assertEqual(m.hparams["in_channels"], 12)
        self.assertNotEqual(m.hparams, FourInputUNet().hparams)

    def test_finite_r3_loss_and_gradients(self):
        m = build_stem_model(seed=0)
        cases = {"random": TestFeatures().targets(b=2), "zero_plane": TestFeatures().targets(b=2),
                 "constant": torch.full((2, 4, 1, 128, 128), 128 / 255)}
        cases["zero_plane"][:, 3] = 0.0
        for name, t in cases.items():
            m.zero_grad(set_to_none=True)
            loss = r3_loss(multiplane_amplitudes(m(t), CFG, propagate_fresnel), t)
            loss.backward()
            self.assertTrue(math.isfinite(float(loss.detach())), name)
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads), name)
            self.assertGreater(float(m.encoders[0][0].weight.grad.abs().sum()), 0, name)

    @unittest.skipUnless(os.environ.get("BPS_BEST_PT"), "BPS_BEST_PT not set")
    def test_production_checkpoint_still_loads(self):
        import doe_models.train_multiplane as tm
        model, payload = tm.load_checkpoint(os.environ["BPS_BEST_PT"])
        self.assertEqual(model.encoders[0][0].in_channels, 4)
        self.assertEqual(payload["sha256"] if "sha256" in payload else None, payload.get("sha256"))


if __name__ == "__main__":
    unittest.main()
