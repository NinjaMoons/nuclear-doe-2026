"""Host wrapper (MVRto01 DESKTOP-T1K698V) for r25_gf_ddfn.train_r25 WITHOUT decoding TEST.
The stock tm.load_manifest decodes train/val/test; this host's corpus copy has no TEST images by policy (frozen TEST is
never decoded). Only the data loading is replaced: train + val read in manifest order with per-file sha256 check
(research/wc_stem_pilot/pilot.py `native`, verified earlier). Model, sampler, objective, optimiser, checkpoint and
official VAL evaluation are the unchanged train_r25 code path (identical for both arms).
usage: python -m r25_gf_ddfn.run_r25_notest <train_r25 argv...>"""
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
from r25_gf_ddfn import train_r25  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["train_r25"] + sys.argv[1:]
    train_r25.main()
