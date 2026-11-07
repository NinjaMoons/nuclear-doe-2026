import tempfile, unittest
from pathlib import Path
import torch
import d4_stem.train_wrapper_d4 as w
from back_projection_stem.stem import build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from d4_stem.d4 import apply
from d4_stem.model import build_d4_model
CFG = CaseConfigV1(); tm = w.tm


def targets(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    t = torch.zeros(b, 4, 1, 128, 128)
    for i in range(b):
        for k in range(4):
            t[i, k, 0] = ((((xx + 3 * i) // (5 + 3 * k)) + yy // (9 + k)) % 2).float() * 0.8 + 0.2 * torch.rand(128, 128, generator=g)
    return torch.floor(t * 255 + 0.5) / 255


class T(unittest.TestCase):
    def test_same_init_and_eval_identity(self):
        a, b = build_stem_model(seed=0), build_d4_model(seed=0)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]))
        t = targets()
        a.eval(); b.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(a(t), b(t)))

    def test_loss_equals_training_on_transformed_quartet(self):
        m = build_d4_model(seed=0).double(); m.train()
        t = targets().double()
        phi = m(t); words = m.last_words
        self.assertGreater(len(set(words)), 1)
        loss_wrap, _ = tm._objective(multiplane_amplitudes(phi, CFG, propagate_fresnel), t, w.bw.R3_NAME)
        tg = torch.stack([apply(wd, t[i]) for i, wd in enumerate(words)])
        ref = build_stem_model(seed=0).double(); ref.load_state_dict({k: v for k, v in m.state_dict().items()}); ref.train()
        loss_aug, _ = tm._objective(multiplane_amplitudes(ref(tg), CFG, propagate_fresnel), tg, w.bw.R3_NAME)
        self.assertLess(abs(float(loss_wrap) - float(loss_aug)), 1e-12)
        loss_wrap.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_global_rng_untouched(self):
        torch.manual_seed(5); a = torch.rand(3)
        torch.manual_seed(5); m = build_d4_model(seed=0); m.train(); m(targets(b=2)); b = torch.rand(3)
        self.assertTrue(torch.equal(a, b))

    def test_checkpoint_roundtrip(self):
        m = build_d4_model(seed=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.pt"; tm.save_checkpoint(p, m, config=CFG, train_log={"partial": True})
            l, _ = w.research_load_checkpoint(p); l.eval(); m.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(l(targets(b=1)), m(targets(b=1))))
            self.assertEqual(w.EVAL_SPLITS, ("val",))


if __name__ == "__main__":
    unittest.main()
