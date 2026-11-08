"""R25 Stage 1 extension (read-only, CPU, frozen V2, TRAIN + VAL only, no TEST, no weight update).
16 TRAIN quartets per subtype (A, B/pilot_independent, B/independent, B/similar, B/partial), selected by sorted key +
torch.randperm(seed 2026) BEFORE any metric is seen; exported through the official evaluator; analysed with
compare_r25.load/summarize (native-grid own-target metrics, physically registered inter-z, low-pass control).
usage: python -m r25_gf_ddfn.train_subtypes OUT_DIR V2_CKPT"""
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, r"C:\nuclear-doe-operator\b-sampler\research\wc_stem_pilot")
import pilot  # noqa: E402
from doe_core.contracts_v1 import CaseConfigV1  # noqa: E402
import doe_models.train_multiplane as tm  # noqa: E402
from r25_gf_ddfn.train_r25 import research_load  # noqa: E402
from r25_gf_ddfn.data_truth_audit import CORPUS, group_of  # noqa: E402
from r25_gf_ddfn.compare_r25 import load, summarize  # noqa: E402


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False); v2 = Path(sys.argv[2])
    man = json.loads((CORPUS / "manifest_expanded.json").read_bytes())
    samples = {s["key"]: s for s in man["samples"] if s["split"] == "train"}
    data, msha = pilot.native(CORPUS, splits=("train",)); tr = data["train"]
    comps = tm.production_components(); tm.load_checkpoint = research_load
    g = torch.Generator().manual_seed(2026); rep = {"manifest_sha256": msha, "n_per_group": 16, "groups": {}}
    for grp in ("A", "B/pilot_independent", "B/independent", "B/similar", "B/partial"):
        pool = sorted((i for i, k in enumerate(tr["ids"]) if group_of(samples[k]) == grp), key=lambda i: tr["ids"][i])
        idx = sorted(pool[j] for j in torch.randperm(len(pool), generator=g)[:16].tolist())
        sub = {f: ([tr[f][i] for i in idx] if f != "targets" else tr["targets"][idx]) for f in ("targets", "ids", "origins", "paths", "kinds")}
        name = grp.replace("/", "_")
        tm.evaluate_checkpoint(v2, {"splits": {"val": sub}, "manifest_sha256": msha, "synthetic": True},
                               comps, out / name, torch.device("cpu"), CaseConfigV1(), ("val",))
        rows = load(out / name / "export" / v2.name / "val")
        s = summarize(rows)["all"]; s["keys"] = sub["ids"]
        rep["groups"][grp] = s
        print(grp, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("keys", "pearson_plane")}, flush=True)
    (out / "train_subtypes.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
