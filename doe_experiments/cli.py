"""Minimal production CLI: actual modules only; there is no fallback simulator."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from doe_core.contracts_v1 import CaseConfigV1, MultiPlaneResult
from doe_experiments.export import export_run, production_components
from doe_experiments.phase_codec import load_doe_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="method", required=True)
    for method in ("adam", "reconstruct"):
        command = subparsers.add_parser(method)
        command.add_argument("--targets", nargs=4, required=True, type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--synthetic-test", action="store_true")
        command.add_argument("--dtype", choices=("float32", "float64"), default="float32")
        if method == "adam":
            command.add_argument("--steps", type=int, default=100)
            command.add_argument("--lr", type=float, default=0.05)
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--doe", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        components = production_components()
        config = CaseConfigV1()
        dtype = getattr(torch, args.dtype)
        if args.method == "adam":
            from doe_models.adam_multiplane import synthesize_multiplane
            targets = components.load_quartet(args.targets).to(dtype=dtype)
            started = time.perf_counter()
            result = synthesize_multiplane(targets, config, propagate_fn=components.propagate,
                                           steps=args.steps, lr=args.lr, seed=args.seed)
            if not isinstance(result, MultiPlaneResult):
                raise ValueError("solver must return MultiPlaneResult")
            phase = result.phase_rad
            metadata = {"method": "adam", "seed": args.seed, "steps": args.steps, "lr": args.lr,
                        "solver_elapsed_s": time.perf_counter() - started,
                        "loss_history": result.loss_history, "solver_metadata": result.metadata}
        else:
            phase = load_doe_png(args.doe, dtype=dtype)
            metadata = {"method": "reconstruct", "source_doe": str(args.doe.resolve())}
        metadata["reproduce_argv"] = ["python", "-m", "doe_experiments.cli", *(argv or sys.argv[1:])]
        metadata["phase_dtype"] = args.dtype
        metadata["reproduction_note"] = "Use the same dtype and a NEW --output-dir; existing runs are not overwritten."
        manifest = export_run(args.targets, phase, args.output_dir, components=components,
                              config=config, metadata=metadata, synthetic_test=args.synthetic_test)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "completed": manifest["completed"]}))
        return 0
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        print(f"v1 pipeline failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
