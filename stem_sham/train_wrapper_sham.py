"""RESEARCH ONLY (#44 P1): python -m stem_sham.train_wrapper_sham --arm {proper,chirp,shuffle} <driver argv> (VAL-only eval)."""
import functools, hashlib, json, sys
from pathlib import Path
import torch
import back_projection_stem.train_wrapper as bw   # registers the identical R3 objective
from back_projection_stem.stem import BackProjectionStemUNet, build_stem_model
from doe_core.contracts_v1 import CaseConfigV1
from stem_sham.model import ShamStemUNet, build_sham_model, load_bank
tm = bw.tm
_stock_load, _stock_build = tm.load_checkpoint, tm.build_model


def research_load_checkpoint(path):
    hp = torch.load(Path(path), map_location="cpu", weights_only=True).get("hparams", {})
    cls = functools.partial(ShamStemUNet, sham=hp["sham"]) if "sham" in hp else (BackProjectionStemUNet if hp.get("stem") == "backprojection_v0" else None)
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
    manifest = drv[drv.index("--manifest") + 1]
    if arm == "proper":
        builder = build_stem_model
    elif arm in ("chirp", "shuffle"):
        bank = load_bank(manifest)[0] if arm == "shuffle" else None
        builder = functools.partial(build_sham_model, sham=arm, bank=bank)
    else:
        raise SystemExit("arm")
    drv += ["--objective", bw.R3_NAME, "--no-eval"]
    tm.build_model = builder
    try:
        rc = tm.main(drv)
    finally:
        tm.build_model = _stock_build
    if rc != 0:
        return rc
    out = Path(drv[drv.index("--out-dir") + 1]); comp = tm.production_components()
    data = tm.load_manifest(manifest, load_quartet=comp.load_quartet)
    tm.load_checkpoint = research_load_checkpoint
    try:
        s = tm.evaluate_checkpoints(sorted((out / "checkpoints").glob("step_*.pt")), data, comp, out / "eval_val",
                                    torch.device("cpu"), ("val",), CaseConfigV1())
    finally:
        tm.load_checkpoint = _stock_load
    (out / "eval_summary.json").write_text(json.dumps(s, indent=1))
    me = Path(__file__)
    (out / "research_wrapper.json").write_text(json.dumps({"arm": arm, "wrapper_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((me.parent / "model.py").read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
