"""R15 Stage 1 (#48): matched fine-tune from the frozen STEM checkpoint. RESEARCH ONLY.
python -m r15_axial.train_r15 --arm {base,treatment} --init CKPT <driver argv without --objective/--no-eval>
Unchanged 49b96041 driver `main` with --no-eval; module-attribute patches only: objective dispatcher (+R15 name),
build_model -> fresh copy of the frozen STEM weights (both arms identical init); in-process stock evaluation with the
research loader (12-ch). Fresh Adam in both arms (the checkpoint carries no optimizer state)."""
import hashlib
import json
import sys
from pathlib import Path

import torch

import doe_models.train_multiplane as tm
from back_projection_stem import train_wrapper as tw            # registers R3 objective + research loader
from doe_core.contracts_v1 import CaseConfigV1
from r15_axial.l_ax import LAMBDA, axial_innovation

R15_NAME = "research_r15_r3_plus_0.05_axial_innovation_v0"
_r3_dispatch = tm._objective


def r15_objective(amplitudes, targets, objective=tm.CONTRACT_OBJECTIVE):
    if objective != R15_NAME:
        return _r3_dispatch(amplitudes, targets, objective)
    base, info = _r3_dispatch(amplitudes, targets, tw.R3_NAME)
    lax = axial_innovation(amplitudes, targets)
    out = dict(info)
    out["l_ax"] = float(lax.detach())
    return base + LAMBDA * lax, out


tm._objective = r15_objective
tm.TRAINING_OBJECTIVES = tm.TRAINING_OBJECTIVES + (R15_NAME,)


def main(argv):
    arm = argv[argv.index("--arm") + 1]
    init = argv[argv.index("--init") + 1]
    drv = [a for i, a in enumerate(argv) if a not in ("--arm", "--init") and argv[i - 1] not in ("--arm", "--init")]
    obj = {"base": tw.R3_NAME, "treatment": R15_NAME}[arm]
    drv += ["--objective", obj, "--no-eval"]
    init_model, init_payload = tw.research_load_checkpoint(init)

    def build_from_init(**_):
        m, _p = tw.research_load_checkpoint(init)
        return m
    stock_build = tm.build_model
    tm.build_model = build_from_init
    try:
        rc = tm.main(drv)
    finally:
        tm.build_model = stock_build
    if rc != 0:
        return rc
    out = Path(drv[drv.index("--out-dir") + 1])
    components = tm.production_components()
    data = tm.load_manifest(drv[drv.index("--manifest") + 1], load_quartet=components.load_quartet)
    tm.load_checkpoint = tw.research_load_checkpoint
    try:
        summary = tm.evaluate_checkpoints(sorted((out / "checkpoints").glob("*.pt")), data, components,
                                          out / "eval_val", torch.device("cpu"), ("val",), CaseConfigV1())
    finally:
        tm.load_checkpoint = tw._stock_load
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({
        "arm": arm, "objective": obj, "lambda": LAMBDA if arm == "treatment" else 0.0, "init_checkpoint_sha256": init_payload["sha256"],
        "train_r15_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "l_ax_sha256": hashlib.sha256((me.parent / "l_ax.py").read_bytes()).hexdigest(),
        "torch": str(torch.__version__), "note": "research only; never for #8"}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
