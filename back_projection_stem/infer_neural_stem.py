"""RESEARCH ONLY (#5): one-forward inference for the 12-ch back-projection stem through the UNCHANGED production CLI.

    python -m back_projection_stem.infer_neural_stem --targets T1 T2 T3 T4 --output-dir NEW --checkpoint STEM.pt [--synthetic-test]

It calls `doe_experiments.cli.main(["neural", ...])` and swaps ONLY the checkpoint loader for the duration of the call:
checkpoints whose hparams say stem=backprojection_v0 are built as BackProjectionStemUNet inside the stock strict
`load_checkpoint`; every other checkpoint (e.g. production best.pt) goes through the stock loader unchanged.
Everything else is the stock path: load_png_quartet -> ONE infer_phase (eval, no_grad) -> export_run
(DOE.png -> reload -> 4 Fresnel -> 4 PNG -> reload -> metrics). The stem computes its features inside the forward with
4 exact inverse Fresnel FFTs of the input PNGs; there is no optimiser, no per-input phase iteration.
"""
import sys
from pathlib import Path

import torch

import doe_models.train_multiplane as tm
from back_projection_stem.stem import BackProjectionStemUNet

_stock_load = tm.load_checkpoint


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
    from doe_experiments import cli
    tm.load_checkpoint = research_load_checkpoint
    try:
        return cli.main(["neural", *argv])
    finally:
        tm.load_checkpoint = _stock_load


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
