import math, os, unittest, torch
from back_projection_stem.r3_loss import r3_loss
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from bpmix_head.train_r18 import research_load
from bpmix_head.model import from_v2
V2 = os.environ.get("R18_V2")


@unittest.skipUnless(V2, "R18_V2 not set")
class T(unittest.TestCase):
    def test_init_equals_v2_and_head_gets_gradient(self):
        v2, _ = research_load(V2); m = from_v2(v2); v2.eval(); m.eval()
        g = torch.Generator().manual_seed(0); t = torch.floor(torch.rand((2, 4, 1, 128, 128), generator=g) * 255 + .5) / 255
        with torch.no_grad():
            a, b = v2(t), m(t)
        d = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
        self.assertLess(float(d.abs().max()), 1e-5)
        m.train(); loss = r3_loss(multiplane_amplitudes(m(t), CaseConfigV1(), propagate_fresnel), t); loss.backward()
        self.assertTrue(torch.isfinite(m.mix_head.weight.grad).all()); self.assertGreater(float(m.mix_head.weight.grad.abs().sum()), 0)

    def test_checkpoint_roundtrip(self):
        import tempfile, pathlib
        import bpmix_head.train_r18 as tr
        v2, _ = research_load(V2); m = from_v2(v2)
        with torch.no_grad(): m.mix_head.bias.add_(0.01)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.pt"; tr.tm.save_checkpoint(p, m, config=CaseConfigV1(), train_log={"partial": True})
            l, _ = research_load(p); t = torch.rand(1, 4, 1, 128, 128)
            l.eval(); m.eval()
            with torch.no_grad(): self.assertTrue(torch.equal(l(t), m(t)))


if __name__ == "__main__":
    unittest.main()
