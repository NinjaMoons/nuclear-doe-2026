import math, unittest
import torch
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from back_projection_stem.stem import inverse_fresnel, backprojection_features
from d4_stem.d4 import ELEMENTS, apply, inverse
CFG = CaseConfigV1()
rel = lambda a, b: float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


def rand_phase(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.polar(torch.ones(1, 1, 128, 128, dtype=torch.float64), (torch.rand(1, 1, 128, 128, generator=g, dtype=torch.float64) * 2 - 1) * math.pi)


class TestD4(unittest.TestCase):
    def test_group(self):
        x = torch.randn(128, 128, dtype=torch.float64)
        imgs = [apply(w, x) for w in ELEMENTS]
        for a in ELEMENTS:
            self.assertTrue(torch.equal(apply(inverse(a), apply(a, x)), x))
            for b in ELEMENTS:
                self.assertTrue(any(torch.equal(apply(a, apply(b, x)), im) for im in imgs))
        self.assertTrue(all(not torch.equal(imgs[i], imgs[j]) for i in range(8) for j in range(i + 1, 8)))

    def test_c64_exact_forward_inverse_features(self):
        U = rand_phase(0)
        T = torch.floor(torch.rand((1, 4, 1, 128, 128), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 255 + 0.5) / 255
        F0 = backprojection_features(T)
        for k in range(4):
            g = CFG.plane(k); FU = propagate_fresnel(U, g).field; IU = inverse_fresnel(U, g)
            for w in ELEMENTS:
                self.assertLess(rel(propagate_fresnel(apply(w, U), g).field, apply(w, FU)), 1e-12)
                self.assertLess(rel(inverse_fresnel(apply(w, U), g), apply(w, IU)), 1e-12)
        for w in ELEMENTS:
            self.assertLess(rel(backprojection_features(apply(w, T)), apply(w, F0)), 1e-12)

    def test_pixel_centre_flip_is_not_exact(self):
        U = rand_phase(2); g = CFG.plane(0); FU = propagate_fresnel(U, g).field
        self.assertGreater(rel(propagate_fresnel(apply(("fx",), U, "c63.5"), g).field.abs(), apply(("fx",), FU, "c63.5").abs()), 1e-2)


if __name__ == "__main__":
    unittest.main()
