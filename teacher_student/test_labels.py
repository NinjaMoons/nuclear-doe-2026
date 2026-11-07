"""Label alignment / TRAIN isolation tests (#49 review items 1-2). Needs TS_CORPUS, TS_CKPT; builds labels for 4 TRAIN
quartets only (~15 s CPU). python -m unittest teacher_student.test_labels -v"""
import os
import tempfile
import unittest
from pathlib import Path

import torch

from doe_models.train_multiplane import load_manifest
from doe_experiments.data import load_png_quartet
from teacher_student.labels import LabelStore, build, tensor_key

CORPUS, CKPT = os.environ.get("TS_CORPUS"), os.environ.get("TS_CKPT")


@unittest.skipUnless(CORPUS and CKPT, "TS_CORPUS/TS_CKPT not set")
class TestLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.path = Path(cls.tmp) / "labels4.pt"
        cls.blob = build(Path(CORPUS), CKPT, cls.path, limit=4, steps=10)
        cls.store = LabelStore(cls.path)
        cls.data = load_manifest(Path(CORPUS) / "manifest_expanded.json", load_quartet=load_png_quartet)

    def test_train_only_and_metadata(self):
        self.assertEqual(self.blob["manifest_sha256"][:8], "76d2474d")
        tr = self.data["splits"]["train"]["ids"]
        self.assertTrue(all(r["key"] in tr for r in self.blob["records"].values()))
        val = set(self.data["splits"]["val"]["ids"]) | set(self.data["splits"]["test"]["ids"])
        self.assertFalse(any(r["key"] in val for r in self.blob["records"].values()))

    def test_driver_batch_alignment_by_content(self):
        train = self.data["splits"]["train"]["targets"]            # exactly what the driver trains on
        ids = self.data["splits"]["train"]["ids"]
        labelled = [i for i, k in enumerate(ids) if k in {r["key"] for r in self.blob["records"].values()}]
        idx = torch.tensor(labelled[::-1])                          # shuffled order like a random batch
        batch = train[idx]
        labs = self.store.lookup(batch)
        for b, i in enumerate(idx.tolist()):
            rec = self.store.records[tensor_key(train[i])]
            self.assertEqual(rec["key"], ids[i])
            self.assertTrue(torch.equal(labs[b], self.store.labels[rec["index"]]))

    def test_permuted_labels_are_caught(self):
        train = self.data["splits"]["train"]["targets"]
        ids = self.data["splits"]["train"]["ids"]
        i0 = ids.index(next(iter(self.blob["records"].values()))["key"])
        swapped = train[i0].clone()
        swapped[[0, 1]] = swapped[[1, 0]]                          # plane order permuted -> different content
        with self.assertRaises(KeyError):
            self.store.lookup(swapped[None])
        unlabeled = [i for i, k in enumerate(ids) if k not in {r["key"] for r in self.blob["records"].values()}][0]
        with self.assertRaises(KeyError):
            self.store.lookup(train[unlabeled][None])

    def test_label_is_display_of_quantized_teacher(self):
        labs = self.store.labels
        self.assertEqual(tuple(labs.shape[1:]), (4, 1, 128, 128))
        self.assertTrue(torch.allclose(labs.amax(dim=(-2, -1)), torch.ones(labs.shape[:3])))
        self.assertIn("best_step0_fraction", self.store.meta)


if __name__ == "__main__":
    unittest.main()
