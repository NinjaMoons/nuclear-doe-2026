"""RESEARCH ONLY (#5/6708046281): paired CPU run CONTROL (4-ch FourInputUNet) vs STEM (12-ch back-projection), both
with the R3 objective (lsq + 0.3*(1-corr) + 0.1*contract). Only the stem differs.

Runs the UNCHANGED 49b96041 driver `doe_models.train_multiplane.main` in-process with --no-eval, then evaluates every
checkpoint in-process with the stock `evaluate_checkpoints` (official DOE.png -> reload -> 4 PNG -> reload metrics).
A new-process eval would use the stock loader, which correctly rejects the 12-ch checkpoint; therefore both arms use
the same in-process eval path. Patches (module attributes only, no file changes):
  - objective dispatcher: adds the R3 research objective name (formula identical to r3_train_wrapper 9e8cd4a9)
  - --arm stem: tm.build_model -> build_stem_model (all layers after the stem get the SAME init as control because
    FourInputUNet.__init__ runs first and the stem block is created afterwards)
  - research_load_checkpoint: stock strict load_checkpoint with BackProjectionStemUNet swapped in only for checkpoints
    whose hparams say stem=backprojection_v0.
usage (cwd = this worktree):  python -m back_projection_stem.train_wrapper --arm {control,stem} <driver argv>
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

import doe_models.train_multiplane as tm
from doe_core.contracts_v1 import CaseConfigV1

from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model

R3_NAME = "research_r3_lsq_plus_0.3_centered_corr_plus_0.1_contract_v0"
BETA, EPS, CONST_STD, DIMS = 0.3, 1e-12, 0.5 / 255, (-2, -1)
_stock_objective = tm._objective
_stock_load = tm.load_checkpoint
_stock_build = tm.build_model


def research_objective(amplitudes, targets, objective=tm.CONTRACT_OBJECTIVE):
    if objective != R3_NAME:
        return _stock_objective(amplitudes, targets, objective)
    base, info = _stock_objective(amplitudes, targets, tm.LSQ_MIX_OBJECTIVE)
    a, t = amplitudes.double(), targets.double()
    x = a / a.amax(dim=DIMS, keepdim=True)
    dx, dt = x - x.mean(dim=DIMS, keepdim=True), t - t.mean(dim=DIMS, keepdim=True)
    vx, vt = (dx * dx).mean(dim=DIMS), (dt * dt).mean(dim=DIMS)
    corr = (dx * dt).mean(dim=DIMS) / torch.sqrt((vx + EPS) * (vt + EPS))
    const = vt <= CONST_STD ** 2
    term = torch.where(const, torch.zeros_like(corr), 1 - corr).mean()
    out = dict(info)
    out.update({"corr_term": float(term.detach()), "const_planes": int(const.sum())})
    return base + BETA * term, out


tm._objective = research_objective
if R3_NAME not in tm.TRAINING_OBJECTIVES:
    tm.TRAINING_OBJECTIVES = tm.TRAINING_OBJECTIVES + (R3_NAME,)


def research_load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    hp = payload.get("hparams") if isinstance(payload, dict) else None
    if isinstance(hp, dict) and hp.get("stem") == "backprojection_v0":
        saved = tm.FourInputUNet
        tm.FourInputUNet = BackProjectionStemUNet
        try:
            return _stock_load(path)
        finally:
            tm.FourInputUNet = saved
    return _stock_load(path)


def main(argv):
    arm = argv[argv.index("--arm") + 1]
    drv = [a for i, a in enumerate(argv) if a != "--arm" and (i == 0 or argv[i - 1] != "--arm")]
    if arm not in ("control", "stem"):
        raise SystemExit("--arm must be control or stem")
    if "--objective" in drv or "--no-eval" in drv or "--evaluate-checkpoint" in drv:
        raise SystemExit("wrapper fixes --objective and evaluation; do not pass them")
    drv += ["--objective", R3_NAME, "--no-eval"]
    if arm == "stem":
        tm.build_model = build_stem_model
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
                                          out / "eval_val", device, ("val",), CaseConfigV1())
    finally:
        tm.load_checkpoint = _stock_load
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({
        "arm": arm, "objective": R3_NAME, "beta": BETA, "eval": "in-process stock evaluate_checkpoints, VAL only",
        "wrapper_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "stem_sha256": hashlib.sha256((me.parent / "stem.py").read_bytes()).hexdigest(),
        "torch": str(torch.__version__), "note": "research only; never for #8"}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
