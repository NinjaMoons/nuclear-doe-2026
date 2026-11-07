"""R16 student fine-tune wrapper (research only). Arms: control = R3; treatment = R3 + MU * L_opt(label by content).
python -m teacher_student.train_r16 --arm {control,treatment} --init CKPT --labels LABELS.pt <driver argv>
Driver 49b96041 unchanged (module-attribute patches only); in-process stock VAL evaluation with the research loader.
If the label cache covers only a TRAIN subset, BOTH arms train on exactly that subset (same batch generator)."""
import hashlib
import json
import sys
from pathlib import Path

import torch

import doe_models.train_multiplane as tm
from back_projection_stem import train_wrapper as tw            # R3 objective + research loader
from doe_core.contracts_v1 import CaseConfigV1
from teacher_student.labels import LabelStore, tensor_key
from teacher_student.losses import optical_distill

MU = 25.0
R16_NAME = "research_r16_r3_plus_25_optical_distill_v0"
_r3 = tm._objective
STORE = None


def r16_objective(amplitudes, targets, objective=tm.CONTRACT_OBJECTIVE):
    if objective != R16_NAME:
        return _r3(amplitudes, targets, objective)
    base, info = _r3(amplitudes, targets, tw.R3_NAME)
    lab = STORE.lookup(targets)                                  # KeyError if any quartet is unlabeled/misaligned
    dist = optical_distill(amplitudes, lab)
    out = dict(info)
    out["l_opt"] = float(dist.detach())
    return base + MU * dist, out


tm._objective = r16_objective
tm.TRAINING_OBJECTIVES = tm.TRAINING_OBJECTIVES + (R16_NAME,)


def main(argv):
    global STORE
    arm = argv[argv.index("--arm") + 1]
    init = argv[argv.index("--init") + 1]
    labels = argv[argv.index("--labels") + 1]
    flags = ("--arm", "--init", "--labels")
    drv = [a for i, a in enumerate(argv) if a not in flags and argv[i - 1] not in flags]
    STORE = LabelStore(labels)
    labelled = {tensor_key_rec["key"] for tensor_key_rec in STORE.records.values()}
    obj = {"control": tw.R3_NAME, "treatment": R16_NAME}[arm]
    drv += ["--objective", obj, "--no-eval"]
    stock_load_manifest = tm.load_manifest

    def load_manifest_subset(path, **kw):                       # both arms: TRAIN restricted to the labelled subset
        data = stock_load_manifest(path, **kw)
        tr = data["splits"]["train"]
        keep = [i for i, k in enumerate(tr["ids"]) if k in labelled]
        for f in ("ids", "paths", "origins"):
            if f in tr:
                tr[f] = [tr[f][i] for i in keep]
        tr["targets"] = tr["targets"][keep]
        return data

    tm.load_manifest = load_manifest_subset
    tm.build_model = lambda **_: tw.research_load_checkpoint(init)[0]
    try:
        rc = tm.main(drv)
    finally:
        tm.load_manifest = stock_load_manifest
    if rc != 0:
        return rc
    out = Path(drv[drv.index("--out-dir") + 1])
    components = tm.production_components()
    data = stock_load_manifest(drv[drv.index("--manifest") + 1], load_quartet=components.load_quartet)
    tm.load_checkpoint = tw.research_load_checkpoint
    try:
        summary = tm.evaluate_checkpoints(sorted((out / "checkpoints").glob("*.pt")), data, components,
                                          out / "eval_val", torch.device("cpu"), ("val",), CaseConfigV1())
    finally:
        tm.load_checkpoint = tw._stock_load
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({
        "arm": arm, "objective": obj, "mu": MU if arm == "treatment" else 0.0, "labels_meta": STORE.meta,
        "n_train_labelled": len(labelled), "train_r16_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "torch": str(torch.__version__), "note": "research only; never for #8 without separate package + review"}, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
