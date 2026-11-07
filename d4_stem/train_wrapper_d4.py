"""RESEARCH ONLY (R14): python -m d4_stem.train_wrapper_d4 --arm {none,d4} <driver argv>; VAL-only in-process eval."""
import functools, hashlib, json, sys
from pathlib import Path
import torch
import back_projection_stem.train_wrapper as bw   # registers the identical R3 objective
from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from d4_stem.model import D4AugStemUNet, build_d4_model
tm = bw.tm
_stock_load, _stock_build = tm.load_checkpoint, tm.build_model
EVAL_SPLITS = ("val",)


def research_load_checkpoint(path):
    hp = torch.load(Path(path), map_location="cpu", weights_only=True).get("hparams", {})
    cls = D4AugStemUNet if hp.get("train_aug") == "d4_c64_sync_v0" else (BackProjectionStemUNet if hp.get("stem") == "backprojection_v0" else None)
    if cls is None:
        return _stock_load(path)
    saved = tm.FourInputUNet; tm.FourInputUNet = cls
    try:
        return _stock_load(path)
    finally:
        tm.FourInputUNet = saved


def main(argv):
    arm = argv[argv.index("--arm") + 1]
    drv = [a for i, a in enumerate(argv) if a != "--arm" and (i == 0 or argv[i - 1] != "--arm")]
    builder = {"none": build_stem_model, "d4": build_d4_model}[arm]
    drv += ["--objective", bw.R3_NAME, "--no-eval"]
    tm.build_model = builder
    try:
        rc = tm.main(drv)
    finally:
        tm.build_model = _stock_build
    if rc != 0 or "--smoke-steps" in drv:
        return rc
    out = Path(drv[drv.index("--out-dir") + 1]); comp = tm.production_components()
    data = tm.load_manifest(drv[drv.index("--manifest") + 1], load_quartet=comp.load_quartet)
    tm.load_checkpoint = research_load_checkpoint
    try:
        s = tm.evaluate_checkpoints(sorted((out / "checkpoints").glob("step_*.pt")), data, comp, out / "eval_val",
                                    torch.device("cpu"), EVAL_SPLITS, CaseConfigV1())
    finally:
        tm.load_checkpoint = _stock_load
    (out / "eval_summary.json").write_text(json.dumps(s, indent=1))
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({"arm": arm, "wrapper_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((me.parent / "model.py").read_bytes()).hexdigest(),
        "d4_sha256": hashlib.sha256((me.parent / "d4.py").read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
