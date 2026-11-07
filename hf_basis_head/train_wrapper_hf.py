"""RESEARCH ONLY (R12, #39): paired runs A=stem / B=stem_const / C=stem_hf with the R3 objective.

Same mechanics as back_projection_stem.train_wrapper (imported: it registers the identical R3 objective in-process):
UNCHANGED 49b96041 driver `tm.main` with --no-eval, model factory swapped per arm, then in-process stock
`evaluate_checkpoints` on VAL ONLY with a loader that builds the research architecture from checkpoint hparams.
usage (cwd = this worktree): python -m hf_basis_head.train_wrapper_hf --arm {stem,stem_const,stem_hf} <driver argv>
"""
import functools
import hashlib
import json
import sys
from pathlib import Path

import torch

import back_projection_stem.train_wrapper as bw   # registers R3 objective (identical to 9e8cd4a9) in tm
from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from hf_basis_head.model import PHASE_HEAD, HFBasisStemUNet, build_hf_stem_model

tm = bw.tm
R3_NAME = bw.R3_NAME
EVAL_SPLITS = ("val",)
ARMS = {"stem": build_stem_model,
        "stem_const": functools.partial(build_hf_stem_model, basis="const"),
        "stem_hf": functools.partial(build_hf_stem_model, basis="hf")}
_stock_load, _stock_build = tm.load_checkpoint, tm.build_model


def research_load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    hp = payload.get("hparams") if isinstance(payload, dict) else None
    cls = None
    if isinstance(hp, dict) and hp.get("phase_head") == PHASE_HEAD:
        cls = functools.partial(HFBasisStemUNet, basis=hp.get("basis_kind"))
    elif isinstance(hp, dict) and hp.get("stem") == "backprojection_v0":
        cls = BackProjectionStemUNet
    if cls is None:
        return _stock_load(path)
    saved = tm.FourInputUNet
    tm.FourInputUNet = cls
    try:
        return _stock_load(path)
    finally:
        tm.FourInputUNet = saved


def main(argv):
    arm = argv[argv.index("--arm") + 1]
    drv = [a for i, a in enumerate(argv) if a != "--arm" and (i == 0 or argv[i - 1] != "--arm")]
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {tuple(ARMS)}")
    if "--objective" in drv or "--no-eval" in drv or "--evaluate-checkpoint" in drv:
        raise SystemExit("wrapper fixes --objective and evaluation; do not pass them")
    drv += ["--objective", R3_NAME, "--no-eval"]
    tm.build_model = ARMS[arm]
    try:
        rc = tm.main(drv)
    finally:
        tm.build_model = _stock_build
    if rc != 0 or "--smoke-steps" in drv:
        return rc
    out = Path(drv[drv.index("--out-dir") + 1])
    device = torch.device(drv[drv.index("--device") + 1] if "--device" in drv else "cuda")
    components = tm.production_components()
    data = tm.load_manifest(drv[drv.index("--manifest") + 1], load_quartet=components.load_quartet)
    tm.load_checkpoint = research_load_checkpoint
    try:
        summary = tm.evaluate_checkpoints(sorted((out / "checkpoints").glob("*.pt")), data, components,
                                          out / "eval_val", device, EVAL_SPLITS, CaseConfigV1())
    finally:
        tm.load_checkpoint = _stock_load
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({
        "arm": arm, "objective": R3_NAME, "eval": "in-process stock evaluate_checkpoints, VAL only",
        "wrapper_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((me.parent / "model.py").read_bytes()).hexdigest(),
        "stem_sha256": hashlib.sha256((me.parent.parent / "back_projection_stem" / "stem.py").read_bytes()).hexdigest(),
        "torch": str(torch.__version__), "note": "research only; never for #8"}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
