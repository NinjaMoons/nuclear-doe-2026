"""CPU tests for R16 losses/teacher (no student training). python -m unittest teacher_student.test_ts -v
Optional env TS_CORPUS (dir with manifest_expanded.json) for the real-TRAIN teacher timing/separation smoke."""
import json
import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from back_projection_stem.r3_loss import r3_loss
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from teacher_student.losses import circular_phase, optical_distill, optimal_piston
from teacher_student.teacher import teacher_phase

CFG = CaseConfigV1()
g = torch.Generator().manual_seed(0)


def rand_phase(seed=0):
    gg = torch.Generator().manual_seed(seed)
    return (torch.rand((1, 1, 128, 128), generator=gg, dtype=torch.float64) * 2 - 1) * math.pi


def amps(phi):
    return multiplane_amplitudes(phi, CFG, propagate_fresnel)


class TestPhysicsInvariances(unittest.TestCase):
    def test_piston_and_wrap_leave_amplitudes_unchanged(self):
        phi = rand_phase(1)
        a0 = amps(phi)
        self.assertLess(float((amps(phi + 1.234) - a0).abs().max()), 1e-9, "global piston must not change |U|")
        k = torch.randint(-3, 4, phi.shape, generator=g).double()
        self.assertLess(float((amps(phi + 2 * math.pi * k) - a0).abs().max()), 1e-9, "per-pixel 2*pi wraps must not change |U|")


class TestLosses(unittest.TestCase):
    def test_circular_phase_invariances_and_zero(self):
        phi = rand_phase(2)
        self.assertLess(float(circular_phase(phi, phi)), 1e-12)
        self.assertLess(float(circular_phase(phi + 0.7, phi)), 1e-12, "piston removed")
        k = torch.randint(-3, 4, phi.shape, generator=g).double()
        self.assertLess(float(circular_phase(phi + 2 * math.pi * k, phi)), 1e-12, "wraps")
        self.assertGreater(float(circular_phase(rand_phase(3), phi)), 0.5, "unrelated phase ~ 1")
        self.assertAlmostEqual(float(optimal_piston(phi + 0.7, phi).mean()), -0.7, places=9)

    def test_optical_distill_invariant_to_all_amplitude_preserving_ambiguities(self):
        phi = rand_phase(4)
        t = amps(phi)
        k = torch.randint(-3, 4, phi.shape, generator=g).double()
        self.assertLess(float(optical_distill(amps(phi + 0.9 + 2 * math.pi * k), t)), 1e-18)
        self.assertGreater(float(optical_distill(amps(rand_phase(5)), t)), 1e-3)

    def test_gradients_finite_through_fresnel(self):
        phi = rand_phase(6).requires_grad_(True)
        t_phi = rand_phase(7)
        for name, loss in (("optical", optical_distill(amps(phi), amps(t_phi))), ("circular", circular_phase(phi, t_phi))):
            (gr,) = torch.autograd.grad(loss, phi)
            self.assertTrue(bool(torch.isfinite(gr).all()), name)
            self.assertGreater(float(gr.abs().sum()), 0, name)

    def test_circular_phase_conflicts_with_equivalent_solution(self):
        """Documented weakness: a phase giving the SAME |U| on all planes need not be piston/wrap equivalent.
        Conjugate+flip (phi(-r)) is a twin-like candidate; measure whether |U| matches and whether circular loss is large."""
        phi = rand_phase(8)
        twin = -torch.flip(phi, dims=(-2, -1))
        same_amp = float((amps(twin) - amps(phi)).abs().max())
        circ = float(circular_phase(twin, phi))
        # recorded, not asserted as physics: report both numbers
        print(f"\n[twin check] max|U_twin - U| = {same_amp:.3e}, circular loss twin vs phi = {circ:.3f}")
        self.assertTrue(math.isfinite(same_amp) and math.isfinite(circ))


@unittest.skipUnless(os.environ.get("TS_CORPUS"), "TS_CORPUS not set")
class TestTeacherSmoke(unittest.TestCase):
    def test_teacher_runs_and_improves_on_train_quartet(self):
        corpus = Path(os.environ["TS_CORPUS"])
        man = json.loads((corpus / "manifest_expanded.json").read_text(encoding="utf-8"))
        s = {x["key"]: x for x in man["samples"]}["B208:006"]
        self.assertEqual(s["split"], "train")
        t = torch.from_numpy(np.stack([np.asarray(Image.open(corpus / p), dtype=np.float32) / 255 for p in s["paths"]]))[None, :, None]
        phi, a, info = teacher_phase(t, steps=60)
        init_obj = float(r3_loss(amps(rand_phase(0).float()), t))
        print(f"\n[teacher smoke B208:006 60 steps] best {info['best_objective']:.4f} vs init {init_obj:.4f}, {info['elapsed_s']:.1f}s, step {info['best_step']}")
        self.assertLess(info["best_objective"], init_obj)
        self.assertTrue(bool((phi >= -math.pi).all()) and bool((phi < math.pi).all()))


if __name__ == "__main__":
    unittest.main()
