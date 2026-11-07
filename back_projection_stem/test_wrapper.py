"""Wrapper tests without any training: objective equivalence, paired init, checkpoint save/load round trip.
python -m unittest back_projection_stem.test_wrapper -v   (optional BPS_BEST_PT for the production load path)"""
import os
import tempfile
import unittest
from pathlib import Path

import torch

import doe_models.train_multiplane as tm
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from doe_models.unet_multiplane import build_model

from back_projection_stem import train_wrapper as tw
from back_projection_stem.r3_loss import r3_loss
from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model

CFG = CaseConfigV1()


class TestWrapper(unittest.TestCase):
    def targets(self):
        g = torch.Generator().manual_seed(5)
        return torch.floor(torch.rand((2, 4, 1, 128, 128), generator=g) * 255 + 0.5) / 255

    def test_objective_equals_r3_loss(self):
        t = self.targets()
        amp = multiplane_amplitudes(build_model(seed=0)(t), CFG, propagate_fresnel).detach()
        v, info = tm._objective(amp, t, tw.R3_NAME)
        self.assertAlmostEqual(float(v), float(r3_loss(amp, t)), places=12)
        self.assertIn("contract_mse", info)
        v0, _ = tm._objective(amp, t, tm.LSQ_MIX_OBJECTIVE)
        self.assertAlmostEqual(float(v0), float(tm.training_objective(amp, t, tm.LSQ_MIX_OBJECTIVE)[0]), places=12)

    def test_paired_init_identical_after_stem(self):
        c, s = build_model(seed=0), build_stem_model(seed=0)
        cs, ss = c.state_dict(), s.state_dict()
        diff = [k for k in cs if not k.startswith("encoders.0.") and not torch.equal(cs[k], ss[k])]
        self.assertEqual(diff, [], "all layers after the stem must share the control init")
        self.assertEqual(ss["encoders.0.0.weight"].shape[1], 12)

    def test_stem_checkpoint_roundtrip_and_stock_rejects(self):
        m = build_stem_model(seed=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "stem.pt"
            tm.save_checkpoint(p, m, config=CFG, train_log={"training_objective": tw.R3_NAME}, code_commit="x" * 40)
            with self.assertRaises(ValueError):
                tm.load_checkpoint(p)                       # stock loader keeps rejecting the 12-ch model
            m2, payload = tw.research_load_checkpoint(p)
            self.assertIsInstance(m2, BackProjectionStemUNet)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values())))
            t = self.targets()
            self.assertTrue(torch.equal(tm.infer_phase(m, t), tm.infer_phase(m2, t)))
            self.assertIs(tm.FourInputUNet.__name__, "FourInputUNet")   # swap restored

    @unittest.skipUnless(os.environ.get("BPS_BEST_PT"), "BPS_BEST_PT not set")
    def test_production_best_pt_through_research_loader(self):
        m, payload = tw.research_load_checkpoint(os.environ["BPS_BEST_PT"])
        self.assertEqual(m.encoders[0][0].in_channels, 4)
        self.assertEqual(payload["sha256"], "ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2")


if __name__ == "__main__":
    unittest.main()
