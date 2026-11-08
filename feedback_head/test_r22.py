import math, os, unittest, torch
from back_projection_stem.r3_loss import r3_loss
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from feedback_head.train_r22 import research_load
from feedback_head.model import from_v2
V2 = os.environ.get("R22_V2")


@unittest.skipUnless(V2, "R22_V2 not set")
class T(unittest.TestCase):
    def test_init_equals_v2_and_corr_head_gets_gradient(self):
        v2, _ = research_load(V2); m = from_v2(v2); v2.eval(); m.eval()
        g = torch.Generator().manual_seed(0); t = torch.floor(torch.rand((2, 4, 1, 128, 128), generator=g) * 255 + .5) / 255
        with torch.no_grad():
            a, b = v2(t), m(t)
        d = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
        self.assertLess(float(d.abs().max()), 1e-5)
        m.train(); loss = r3_loss(multiplane_amplitudes(m(t), CaseConfigV1(), propagate_fresnel), t); loss.backward()
        gw = m.corr.head.weight.grad
        self.assertTrue(torch.isfinite(gw).all()); self.assertGreater(float(gw.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_feedback_features_finite_and_roundtrip(self):
        import tempfile, pathlib
        import feedback_head.train_r22 as tr
        v2, _ = research_load(V2); m = from_v2(v2)
        t = torch.rand(1, 4, 1, 128, 128)
        with torch.no_grad():
            f = m.feedback_features(t, m.stage0(t))
            self.assertEqual(tuple(f.shape), (1, 22, 128, 128)); self.assertTrue(torch.isfinite(f).all())
            m.corr.head.bias.add_(0.3); m.eval()
            self.assertGreater(float((m(t) - v2.eval()(t)).abs().mean()), 1e-3)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.pt"; tr.tm.save_checkpoint(p, m, config=CaseConfigV1(), train_log={"partial": True})
            l, _ = research_load(p); l.eval()
            with torch.no_grad(): self.assertTrue(torch.equal(l(t), m(t)))

    def test_forward_has_no_optimizer_or_iteration(self):
        import inspect
        import feedback_head.model as mm
        src = inspect.getsource(mm.FeedbackStemUNet.forward) + inspect.getsource(mm.FeedbackStemUNet.feedback_features)
        for bad in ("backward", "optim", ".step(", "autograd.grad", "while "):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
