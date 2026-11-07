"""Math/contract tests for the R12 residual basis head. Run from the worktree root:
    python -m unittest hf_basis_head.test_hf -v
Optional env: BPS_STEM_PT=<stem_seed0_step_000900.pt>, BPS_BEST_PT=<production best.pt>."""
import inspect
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import hf_basis_head.train_wrapper_hf as hw
from back_projection_stem.r3_loss import r3_loss
from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.export import export_run
from doe_experiments.phase_codec import encode_phase, load_doe_png
from doe_models.adam_multiplane import multiplane_amplitudes
from hf_basis_head.model import COEF_GRID, HFBasisStemUNet, basis_sha256, build_hf_stem_model, make_basis

CFG = CaseConfigV1()
tm = hw.tm


def targets(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros((b, 4, 1, 128, 128))
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    for i in range(b):
        for k in range(4):   # structured, non-trivial targets (bars at different places) + noise
            m = ((xx // (8 + 4 * k) + yy // (6 + 3 * k) + i) % 2).float()
            t[i, k, 0] = torch.floor((0.7 * m + 0.3 * torch.rand((128, 128), generator=g)) * 255 + 0.5) / 255
    return t


def perturb(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.coef_head.weight.copy_(torch.randn(model.coef_head.weight.shape, generator=g) * 0.05)
        model.coef_head.bias.copy_(torch.randn(model.coef_head.bias.shape, generator=g) * 0.05)
    return model


class TestHead(unittest.TestCase):
    def test_shape_range_phase_only(self):
        for basis in ("hf", "const"):
            m = perturb(build_hf_stem_model(seed=0, basis=basis))
            phi = m(targets())
            self.assertEqual(tuple(phi.shape), (2, 1, 128, 128))
            self.assertTrue(bool((phi >= -math.pi).all()) and bool((phi < math.pi).all()))
            u = torch.polar(torch.ones_like(phi), phi)
            self.assertTrue(torch.allclose(u.abs(), torch.ones_like(phi), atol=1e-6))

    def test_step0_identical_to_stem_including_doe_png(self):
        t = targets()
        ref = build_stem_model(seed=0)
        ref_phi = ref(t)
        for basis in ("hf", "const"):
            m = build_hf_stem_model(seed=0, basis=basis)
            for k, v in ref.state_dict().items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]), f"body/stem init differs at {k} ({basis})")
            phi = m(t)
            self.assertTrue(torch.equal(phi, ref_phi), f"step-0 phase differs ({basis})")
            self.assertTrue(np.array_equal(encode_phase(phi), encode_phase(ref_phi)), "DOE.png codes differ")

    def test_global_rng_untouched_and_same_batches(self):
        torch.manual_seed(123); a = torch.rand(3)
        torch.manual_seed(123); build_hf_stem_model(seed=0, basis="hf"); b = torch.rand(3)
        self.assertTrue(torch.equal(a, b))

    def test_zero_init_but_nonzero_finite_coef_gradients(self):
        t = targets()
        for basis in ("hf", "const"):
            m = build_hf_stem_model(seed=0, basis=basis)
            loss = r3_loss(multiplane_amplitudes(m(t), CFG, propagate_fresnel), t)
            loss.backward()
            for p in (m.coef_head.weight, m.coef_head.bias):
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.abs().sum()), 0.0, f"zero coef gradient ({basis})")
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))

    def test_basis_buffer_not_parameter_and_deterministic(self):
        m = build_hf_stem_model(seed=0, basis="hf")
        names = [n for n, _ in m.named_parameters()]
        self.assertNotIn("basis_maps", names)
        self.assertIn("basis_maps", m.state_dict())
        self.assertFalse(m.basis_maps.requires_grad)
        self.assertEqual(basis_sha256(make_basis("hf")), basis_sha256(make_basis("hf")))
        self.assertEqual(basis_sha256(m.basis_maps), m.hparams["basis_sha256"])
        r = make_basis("hf")
        self.assertLess(abs(float(r.mean())), 0.02)
        self.assertTrue(torch.equal(make_basis("const"), torch.ones(16, 128, 128)))
        extra = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in build_stem_model(seed=0).parameters())
        extra_c = sum(p.numel() for p in build_hf_stem_model(seed=0, basis="const").parameters()) - \
            sum(p.numel() for p in build_stem_model(seed=0).parameters())
        self.assertEqual(extra, extra_c)
        self.assertEqual(extra, 16 * 16 + 16)

    def test_coefficients_are_32x32_and_dphi_is_their_upsample(self):
        m = perturb(build_hf_stem_model(seed=0, basis="const"))
        t = targets(b=1)
        feats = {}
        h = m.head.register_forward_hook(lambda mod, i, o: feats.setdefault("x", i[0]))
        m(t); h.remove()
        a = m.coefficients(feats["x"], (128, 128))
        self.assertEqual(tuple(a.shape), (1, 16, COEF_GRID, COEF_GRID))
        d = m.delta_phase(feats["x"], (128, 128))
        ref = torch.nn.functional.interpolate(a.sum(1, keepdim=True), size=(128, 128), mode="bilinear", align_corners=False)
        self.assertTrue(torch.allclose(d, ref, atol=1e-6))           # const basis: dphi = upsample(sum_j a_j)

    def test_no_silent_resize(self):
        m = build_hf_stem_model(seed=0)
        with self.assertRaises(ValueError):
            m(torch.rand(1, 4, 1, 64, 64))

    def test_checkpoint_roundtrip_strict_loader(self):
        for basis in ("hf", "const"):
            m = perturb(build_hf_stem_model(seed=0, basis=basis))
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "x.pt"
                tm.save_checkpoint(p, m, config=CFG, train_log={"partial": True})
                loaded, payload = hw.research_load_checkpoint(p)
                self.assertIsInstance(loaded, HFBasisStemUNet)
                self.assertTrue(torch.equal(loaded.basis_maps, m.basis_maps))
                t = targets(b=1)
                self.assertTrue(torch.equal(loaded.eval()(t), m.eval()(t)))
                with self.assertRaises(ValueError):          # stock loader must refuse the research checkpoint
                    tm.load_checkpoint(p)
                self.assertIs(tm.FourInputUNet.__name__, "FourInputUNet")

    def test_r3_loss_parity(self):
        m = perturb(build_hf_stem_model(seed=0))
        t = targets()
        amp = multiplane_amplitudes(m(t), CFG, propagate_fresnel)
        a, _ = tm._objective(amp, t, hw.R3_NAME)
        self.assertLess(abs(float(a) - float(r3_loss(amp, t))), 1e-12)

    def test_native_geometry_and_official_export_roundtrip(self):
        m = perturb(build_hf_stem_model(seed=0)).eval()
        t = targets(b=1)
        with torch.no_grad():
            phi = m(t)
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            paths = []
            for k in range(4):
                p = Path(d) / f"t{k}.png"
                Image.fromarray((t[0, k, 0].numpy() * 255).round().astype(np.uint8), mode="L").save(p)
                paths.append(p)
            man = export_run(paths, phi, Path(d) / "out", metadata={"test": True}, synthetic_test=True)
            pitches = [CFG.plane(i).output_pitch_m(128, 128) for i in range(4)]
            self.assertEqual([tuple(x) for x in man["output_pitches_m"]], [tuple(x) for x in pitches])
            self.assertAlmostEqual(pitches[0][0], 532e-9 * 0.0241 / (128 * 10e-6), places=15)
            dec = load_doe_png(Path(d) / "out" / "DOE.png", dtype=torch.float64)
            self.assertTrue(np.array_equal(encode_phase(dec), encode_phase(phi)))
            self.assertEqual(sorted(p.name for p in (Path(d) / "out").glob("reconstructed_*.png")),
                             [f"reconstructed_0{k}.png" for k in range(1, 5)])

    def test_eval_is_val_only(self):
        self.assertEqual(hw.EVAL_SPLITS, ("val",))
        self.assertNotIn('"test"', inspect.getsource(hw.main))

    @unittest.skipUnless(os.environ.get("BPS_STEM_PT"), "BPS_STEM_PT not set")
    def test_old_stem_checkpoint_loads_unchanged(self):
        p = os.environ["BPS_STEM_PT"]
        a, pa = hw.research_load_checkpoint(p)
        b, pb = hw.bw.research_load_checkpoint(p)
        self.assertIsInstance(a, BackProjectionStemUNet)
        self.assertNotIsInstance(a, HFBasisStemUNet)
        self.assertEqual(pa["sha256"], "569e170065cca160eb6ce64a0707d81e5e016857b0962498c156cba80f6d3b7d")
        t = targets(b=1)
        self.assertTrue(torch.equal(a.eval()(t), b.eval()(t)))

    @unittest.skipUnless(os.environ.get("BPS_BEST_PT"), "BPS_BEST_PT not set")
    def test_production_best_pt_unchanged(self):
        m, p = hw.research_load_checkpoint(os.environ["BPS_BEST_PT"])
        self.assertEqual(p["sha256"], "ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2")
        self.assertEqual(m.encoders[0][0].in_channels, 4)
        self.assertNotIsInstance(m, BackProjectionStemUNet)


if __name__ == "__main__":
    unittest.main()
