import json, os, unittest
from pathlib import Path
import torch
from back_projection_stem.stem import build_stem_model, backprojection_features
from stem_sham.model import build_sham_model, load_bank, chirp_features, N_BANK
M = os.environ.get("SHAM_MANIFEST")


@unittest.skipUnless(M, "SHAM_MANIFEST not set")
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.keys = load_bank(M)

    def test_bank_is_train_and_size(self):
        man = json.loads(Path(M).read_text()); split = {s["key"]: s["split"] for s in man["samples"]}
        self.assertEqual(len(self.keys), N_BANK); self.assertTrue(all(split[k] == "train" for k in self.keys))

    def test_same_init_after_stem_and_shapes(self):
        ref = build_stem_model(seed=0).state_dict()
        for sham in ("chirp", "shuffle"):
            m = build_sham_model(seed=0, sham=sham, bank=self.bank)
            for k, v in ref.items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]), k)
            out = m(self.bank[:3]); self.assertEqual(tuple(out.shape), (3, 1, 128, 128))

    def test_shuffle_never_self_and_deterministic(self):
        m = build_sham_model(seed=0, sham="shuffle", bank=self.bank)
        idx = m.donor_index(self.bank)
        for i in range(N_BANK):
            self.assertFalse(torch.equal(self.bank[idx[i]], self.bank[i]), f"self donor {i}")
        self.assertTrue(torch.equal(idx, m.donor_index(self.bank)))

    def test_chirp_is_target_independent(self):
        a = chirp_features(2, 128, 128, torch.float32, "cpu"); self.assertEqual(tuple(a.shape), (2, 8, 128, 128))
        m = build_sham_model(seed=0, sham="chirp")
        t1, t2 = self.bank[:1], self.bank[1:2]
        self.assertFalse(torch.equal(m(t1), m(t2)))


if __name__ == "__main__":
    unittest.main()
