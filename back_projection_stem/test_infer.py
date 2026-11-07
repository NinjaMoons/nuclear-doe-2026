"""Inference-adapter tests (no training, no TEST). Needs env BPS_STEM_PT (a 12-ch research checkpoint) and
BPS_INPUTS (dir with 4 synthetic L8 PNGs, sorted order = z1..z4); optional BPS_BEST_PT for the stock path.
python -m unittest back_projection_stem.test_infer -v"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import back_projection_stem.stem as stem_mod
import doe_core.fresnel as fresnel_mod
import doe_models.train_multiplane as tm
from back_projection_stem.infer_neural_stem import main, research_load_checkpoint

STEM = os.environ.get("BPS_STEM_PT")
INPUTS = os.environ.get("BPS_INPUTS")


@unittest.skipUnless(STEM and INPUTS, "BPS_STEM_PT / BPS_INPUTS not set")
class TestInfer(unittest.TestCase):
    def targets(self):
        return [str(p) for p in sorted(Path(INPUTS).glob("*.png"))][:4]

    def test_one_forward_no_iteration(self):
        model, _ = research_load_checkpoint(STEM)
        from doe_experiments.data import load_png_quartet
        t = load_png_quartet([Path(p) for p in self.targets()])
        t = t if t.ndim == 5 else t[None]
        inv = mock.MagicMock(side_effect=stem_mod.inverse_fresnel)
        fwd = mock.MagicMock(side_effect=fresnel_mod.propagate_fresnel)
        with mock.patch.object(stem_mod, "inverse_fresnel", inv), mock.patch.object(fresnel_mod, "propagate_fresnel", fwd), \
                mock.patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimiser used")):
            phase = tm.infer_phase(model, t)
        self.assertEqual(inv.call_count, 4, "exactly 4 inverse Fresnel FFTs per forward (one per plane)")
        self.assertEqual(fwd.call_count, 0, "no forward propagation / no iterative phase refinement inside the model")
        self.assertEqual(tuple(phase.shape), (1, 1, 128, 128))
        self.assertFalse(phase.requires_grad)
        self.assertTrue(torch.equal(phase, tm.infer_phase(model, t)), "deterministic")

    def test_cli_export_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            rc = main(["--targets", *self.targets(), "--output-dir", str(out), "--checkpoint", STEM, "--synthetic-test"])
            self.assertEqual(rc, 0)
            man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            md = man["metadata"]
            self.assertTrue(man["completed"])
            self.assertEqual(md["cli_method"], "neural")
            self.assertEqual(md["hparams"]["in_channels"], 12)
            self.assertEqual(md["hparams"]["stem"], "backprojection_v0")
            self.assertEqual(len(list(out.glob("DOE*.png"))), 1)
            self.assertEqual(sorted(p.name for p in out.glob("reconstructed_*.png")),
                             [f"reconstructed_0{k}.png" for k in range(1, 5)])
            self.assertIs(tm.load_checkpoint.__name__, "load_checkpoint", "stock loader restored")

    @unittest.skipUnless(os.environ.get("BPS_BEST_PT"), "BPS_BEST_PT not set")
    def test_production_checkpoint_unchanged_path(self):
        m, p = research_load_checkpoint(os.environ["BPS_BEST_PT"])
        self.assertEqual(m.encoders[0][0].in_channels, 4)
        self.assertEqual(p["sha256"], "ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2")


if __name__ == "__main__":
    unittest.main()
