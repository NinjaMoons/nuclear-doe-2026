"""Packaging / compliance tests for the STEM contest candidate (#40). No training, no TEST.
Env: BPS_STEM_PT (12-ch checkpoint), BPS_INPUTS (dir with 4 L8 PNG, sorted = z1..z4), optional BPS_BEST_PT (4-ch).
python -m unittest back_projection_stem.test_package -v"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import back_projection_stem.stem as stem_mod
import doe_core.fresnel as fresnel_mod
import doe_models.train_multiplane as tm
from back_projection_stem import infer_neural_stem as inf

STEM = os.environ.get("BPS_STEM_PT")
INPUTS = os.environ.get("BPS_INPUTS")
BEST = os.environ.get("BPS_BEST_PT")
PKG = Path(stem_mod.__file__).parent
STOCK_LOAD, STOCK_UNET = tm.load_checkpoint, tm.FourInputUNet


def targets():
    return [str(p) for p in sorted(Path(INPUTS).glob("*.png"))][:4]


def run_cli(out):
    return inf.main(["--targets", *targets(), "--output-dir", str(out), "--checkpoint", STEM, "--synthetic-test"])


class TestStatic(unittest.TestCase):
    def test_inference_sources_have_no_optimiser_or_solver(self):
        for name in ("stem.py", "infer_neural_stem.py"):
            src = (PKG / name).read_text(encoding="utf-8")
            for bad in ("torch.optim", "adam_multiplane", "synthesize_multiplane", ".backward(", "requires_grad_(True)",
                        "test-49b", "eval_val", "manifest_expanded", "data-4b-v1"):
                self.assertNotIn(bad, src, f"{bad!r} in {name}")


@unittest.skipUnless(STEM and INPUTS, "BPS_STEM_PT / BPS_INPUTS not set")
class TestLoader(unittest.TestCase):
    def test_restored_after_success_and_exception(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli(Path(d) / "ok"), 0)
        self.assertIs(tm.load_checkpoint, STOCK_LOAD)
        self.assertIs(tm.FourInputUNet, STOCK_UNET)
        from doe_experiments import cli
        with mock.patch.object(cli, "main", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                inf.main(["--targets", *targets(), "--output-dir", "x", "--checkpoint", STEM])
        self.assertIs(tm.load_checkpoint, STOCK_LOAD)

        def boom(path):
            raise ValueError("inner failure")
        with mock.patch.object(inf, "_stock_load", boom):
            with self.assertRaises(ValueError):
                inf.research_load_checkpoint(STEM)
        self.assertIs(tm.FourInputUNet, STOCK_UNET)

    def _tampered(self, payload_edit, src):
        payload = torch.load(src, map_location="cpu", weights_only=True)
        payload_edit(payload)
        d = tempfile.mkdtemp()
        p = Path(d) / "t.pt"
        torch.save(payload, p)
        return p

    def test_wrong_stem_metadata_is_rejected(self):
        cases = {
            "stem flag removed": lambda p: p["hparams"].pop("stem"),
            "channel order changed": lambda p: p["hparams"].__setitem__("channel_order", list(reversed(p["hparams"]["channel_order"]))),
            "in_channels lie": lambda p: p["hparams"].__setitem__("in_channels", 4),
            "first conv weights 4-ch": lambda p: p["state_dict"].__setitem__("encoders.0.0.weight", p["state_dict"]["encoders.0.0.weight"][:, :4].clone()),
        }
        for name, edit in cases.items():
            with self.assertRaises(ValueError, msg=name):
                inf.research_load_checkpoint(self._tampered(edit, STEM))
            self.assertIs(tm.FourInputUNet, STOCK_UNET, name)

    @unittest.skipUnless(BEST, "BPS_BEST_PT not set")
    def test_4ch_checkpoint_claiming_stem_is_rejected_and_4ch_path_unchanged(self):
        with self.assertRaises(ValueError):
            inf.research_load_checkpoint(self._tampered(lambda p: p["hparams"].__setitem__("stem", "backprojection_v0"), BEST))
        m, p = inf.research_load_checkpoint(BEST)
        m2, p2 = STOCK_LOAD(BEST)
        self.assertEqual(type(m).__name__, "FourInputUNet")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values())))


@unittest.skipUnless(STEM and INPUTS, "BPS_STEM_PT / BPS_INPUTS not set")
class TestCompliance(unittest.TestCase):
    def test_full_cli_one_neural_forward_no_optimiser(self):
        import dataclasses
        from doe_experiments import cli
        from doe_experiments import export as export_mod
        calls = {"forward": 0}
        orig_fwd = stem_mod.BackProjectionStemUNet.forward

        def counted(self, x):
            calls["forward"] += 1
            return orig_fwd(self, x)
        inv = mock.MagicMock(side_effect=stem_mod.inverse_fresnel)
        prop = mock.MagicMock(side_effect=fresnel_mod.propagate_fresnel)
        comps = dataclasses.replace(export_mod.production_components(), propagate=prop)
        with tempfile.TemporaryDirectory() as d,                 mock.patch.object(stem_mod.BackProjectionStemUNet, "forward", counted),                 mock.patch.object(stem_mod, "inverse_fresnel", inv),                 mock.patch.object(cli, "production_components", lambda: comps),                 mock.patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("optimiser step")),                 mock.patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward")):
            self.assertEqual(run_cli(Path(d) / "run"), 0)
        self.assertEqual(calls["forward"], 1, "exactly one neural forward")
        self.assertEqual(inv.call_count, 4, "4 analytic inverse Fresnel features inside that forward")
        self.assertEqual(prop.call_count, 8, "official evaluation only: 4 decoded-DOE + 4 continuous reconstructions")

    def test_deterministic_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            self.assertEqual(run_cli(a), 0)
            self.assertEqual(run_cli(b), 0)
            for f in ["DOE.png", "metrics.json"] + [f"reconstructed_0{k}.png" for k in range(1, 5)]:
                ha = hashlib.sha256((a / f).read_bytes()).hexdigest()
                hb = hashlib.sha256((b / f).read_bytes()).hexdigest()
                self.assertEqual(ha, hb, f)


if __name__ == "__main__":
    unittest.main()
