"""Host wrapper (MVRto01 DESKTOP-T1K698V) for feedback_head.train_r22 WITHOUT decoding TEST.
The stock tm.load_manifest decodes train/val/test; this host's corpus copy has no TEST images by policy (frozen TEST is
never decoded). Only the data loading is replaced: train + val read in manifest order with per-file sha256 check
(research/wc_stem_pilot/pilot.py `native`, verified earlier). Model, sampler, objective, optimiser, checkpoint and
official VAL evaluation are the unchanged train_r19 code path (identical for both arms).
usage: python -m tphase_head.run_r19_notest <train_r19 argv...>"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\nuclear-doe-operator\b-sampler\research\wc_stem_pilot")
import pilot  # noqa: E402
import doe_models.train_multiplane as tm  # noqa: E402


def load_manifest_no_test(path, *, load_quartet):
    data, msha = pilot.native(Path(path).parent, splits=("train", "val"))
    return {"splits": data, "manifest_sha256": msha, "manifest_path": str(Path(path).resolve()),
            "n_samples": sum(len(v["ids"]) for v in data.values()), "synthetic": True, "corpus_ids": ["A64", "B64", "A208", "B208"]}


tm.load_manifest = load_manifest_no_test
from feedback_head import train_r22 as train_r19  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["train_r19"] + sys.argv[1:]
    train_r19.main()
