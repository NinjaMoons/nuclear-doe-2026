import unittest, torch
from xplane.loss import xplane_contrast, bandpass

def quartet(seed=0, correlated=False):
    g = torch.Generator().manual_seed(seed); yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    planes = []
    for k in range(4):
        if correlated:
            m = torch.exp(-(((xx - 64.0) ** 2 + (yy - 64.0) ** 2) / (2 * 5.0 ** 2)))  # centred blob: survives physical rescaling
        else:
            m = ((((xx * (k + 1)) // (5 + 2 * k)) + (yy * (4 - k)) // (7 + k)) % 2).float()
        planes.append(0.8 * m + (0.0 if correlated else 0.2) * torch.rand(128, 128, generator=g))
    return torch.stack(planes)[None, :, None]

class T(unittest.TestCase):
    def test_finite_black_constant(self):
        t = quartet(); t[0, 1] = 0; a = torch.rand_like(t, requires_grad=True)
        l = xplane_contrast(a, t); l.backward()
        self.assertTrue(torch.isfinite(l) and torch.isfinite(a.grad).all())
        c = torch.full_like(t, 0.5); self.assertTrue(torch.isfinite(xplane_contrast(torch.rand_like(t), c)))
    def test_ghost_monotonic(self):
        t = quartet(); vals = []
        for g in (0.0, 0.1, 0.3, 0.5):
            a = t.clone(); a[0, 0] = (1 - g) * t[0, 0] + g * t[0, 2]; vals.append(float(xplane_contrast(a, t)))
        self.assertTrue(all(vals[i] < vals[i + 1] for i in range(3)), vals)
    def test_dimming_not_rewarded(self):
        t = quartet(); a = t.clone() + 0.1 * quartet(5)
        self.assertAlmostEqual(float(xplane_contrast(a, t)), float(xplane_contrast(0.3 * a, t)), places=9)
    def test_correlated_pairs_excluded(self):
        t = quartet(correlated=True); a = torch.rand_like(t, requires_grad=True)
        l = xplane_contrast(a, t); self.assertEqual(float(l), 0.0)

if __name__ == "__main__":
    unittest.main()
