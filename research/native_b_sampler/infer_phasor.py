"""RESEARCH ONLY: one-forward inference for PhasorStemUNet checkpoints through the UNCHANGED production CLI
(same pattern as back_projection_stem.infer_neural_stem; only the checkpoint loader is swapped for the call).

    python research/native_b_sampler/infer_phasor.py --checkpoint CKPT --targets T1 T2 T3 T4 --output-dir NEW [--synthetic-test]
load_png_quartet -> ONE infer_phase (eval, no_grad) -> export_run (DOE.png -> reload -> 4 Fresnel -> 4 PNG -> metrics).
No optimiser, no per-input iteration.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import doe_models.train_multiplane as tm  # noqa: E402
import phasor  # noqa: E402  (defines PhasorStemUNet + load_any)

_stock = tm.load_checkpoint


def main(argv):
    from doe_experiments import cli
    tm.load_checkpoint = phasor.load_any
    try:
        return cli.main(["neural", *argv])
    finally:
        tm.load_checkpoint = _stock


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
