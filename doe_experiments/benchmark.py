"""Collect a completed Adam export; never rerun physics or recompute metrics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys

from doe_core.contracts_v1 import (
    CaseConfigV1, CONTRACT_VERSION_V1, OBSERVATION_V1,
    RECONSTRUCTION_POLICY_V1, SSIM_POLICY_V1,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    # Reject nonstandard JSON NaN/Infinity even in supplemental metadata.
    json.dumps(value, allow_nan=False)
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive or PureWindowsPath(relative).root):
        raise ValueError("artifact path must be relative to the run directory")
    # Export manifests may have native Windows separators, including when
    # subsequently collected on a POSIX host. Resolve aliases after conversion.
    path = (root / Path(relative.replace("\\", "/"))).resolve()
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError("artifact must be a file inside the run directory")
    return path


def _number(value, name: str, *, lower: float, upper: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(f"invalid {name}")
    return float(value)


def summarize_adam_run(run_dir: Path) -> dict:
    """Validate hashes/schema and preserve all four primary metric pairs."""
    try:
        root = Path(run_dir).resolve()
        manifest_path = _inside(root, "manifest.json")
        manifest = _read(manifest_path)
        if (manifest.get("completed") is not True or manifest.get("schema_version") != 2
                or manifest.get("case_contract") != CONTRACT_VERSION_V1
                or manifest.get("observation") != OBSERVATION_V1):
            raise ValueError("a completed v1.1-amplitude export is required")
        config = CaseConfigV1()
        expected_config = json.loads(json.dumps(asdict(config)))
        if manifest.get("config") != expected_config:
            raise ValueError("official case configuration is required")
        if (manifest.get("reconstruction_policy") != RECONSTRUCTION_POLICY_V1
                or manifest.get("ssim_policy") != SSIM_POLICY_V1):
            raise ValueError("unsupported amplitude or SSIM policy")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("method") != "adam":
            raise ValueError("only Adam solver runs can be summarized")
        if metadata.get("phase_dtype") not in ("float32", "float64"):
            raise ValueError("recorded phase dtype is required")
        seed = metadata.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("recorded nonnegative integer seed is required")
        if type(manifest.get("synthetic_test")) is not bool:
            raise ValueError("explicit synthetic_test flag is required")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("manifest file hashes are required")
        resolved = {}
        for name, digest in files.items():
            path = _inside(root, name)
            if path in resolved.values() or not isinstance(digest, str) or _hash(path) != digest:
                raise ValueError("duplicate artifact path or SHA256 mismatch")
            resolved[name] = path
        required = {"DOE.png", "raw.npz", "metrics.json", *(f"targets/target_{i:02d}.png" for i in range(1, 5))}
        # Windows exporter writes native separators; compare resolved paths.
        if not all((root / name).resolve() in resolved.values() for name in required):
            raise ValueError("required export artifact is missing from manifest")
        metrics_path = _inside(root, "metrics.json")
        metrics = _read(metrics_path)
        if (metrics.get("contract") != CONTRACT_VERSION_V1 or metrics.get("observation") != OBSERVATION_V1
                or metrics.get("policy") != RECONSTRUCTION_POLICY_V1
                or metrics.get("ssim_policy") != SSIM_POLICY_V1
                or metrics.get("primary_source") != "reloaded_uint8_PNG_from_reloaded_DOE"):
            raise ValueError("metrics must use the submitted amplitude PNG policy")
        planes = metrics.get("planes")
        if not isinstance(planes, list) or len(planes) != 4:
            raise ValueError("exactly four ordered per-plane metric pairs are required")
        doe_hash = _hash(_inside(root, "DOE.png"))
        reconstructions = set()
        for index, plane in enumerate(planes):
            if (not isinstance(plane, dict) or not {"pearson", "ssim", "pearson_defined"}.issubset(plane)
                    or type(plane.get("index")) is not int or plane["index"] != index
                    or plane.get("distance_m") != config.distances_m[index] or plane.get("doe_sha256") != doe_hash):
                raise ValueError("per-plane DOE reference or distance/order mismatch")
            path = _inside(root, plane.get("reconstruction"))
            if (path not in resolved.values() or path.parent != root or path in reconstructions
                    or path.suffix.lower() != ".png" or path.name == "DOE.png"):
                raise ValueError("four distinct hashed reconstruction PNGs are required")
            reconstructions.add(path)
            defined = plane.get("pearson_defined")
            if type(defined) is not bool or defined != (plane.get("pearson") is not None):
                raise ValueError("undefined Pearson must have null and explicit false status")
            if defined:
                _number(plane["pearson"], "Pearson", lower=-1.00000001, upper=1.00000001)
            _number(plane.get("ssim"), "SSIM", lower=-1.00000001, upper=1.00000001)
        sources = manifest.get("input_sources")
        if not isinstance(sources, list) or len(sources) != 4:
            raise ValueError("four input source hashes are required")
        for index, source in enumerate(sources, 1):
            if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
                    or source.get("sha256") != _hash(_inside(root, f"targets/target_{index:02d}.png"))):
                raise ValueError("input source hash does not match the archived ordered target")
        solver_time = _number(metadata.get("solver_elapsed_s"), "solver elapsed time", lower=0)
        export_time = _number(manifest.get("elapsed_export_s"), "export elapsed time", lower=0)
        if not isinstance(manifest.get("timing_scope"), str) or not manifest["timing_scope"]:
            raise ValueError("export timing scope is required")
        return {
            "schema_version": 1, "method": "adam", "trained_neural": False,
            "learned_comparison": {"status": "not_available", "checkpoint": None},
            "case_contract": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
            "run_dir": str(root), "synthetic_test": manifest["synthetic_test"],
            "config": manifest["config"], "seed": seed, "dtype": metadata["phase_dtype"],
            "reconstruction_policy": manifest["reconstruction_policy"], "ssim_policy": manifest["ssim_policy"],
            "planes": planes, "input_sources": sources, "artifact_sha256": files,
            "manifest_sha256": _hash(manifest_path), "metrics_sha256": _hash(metrics_path),
            "git_sha": manifest.get("git_sha"), "dependencies": manifest.get("dependencies"),
            "reproduce_argv": metadata.get("reproduce_argv"),
            "timing": {"solver_elapsed_s": solver_time, "export_elapsed_s": export_time,
                       "solver_scope": "CLI solver call, including synthesis and solver final forward",
                       "export_scope": manifest["timing_scope"],
                       "training_s": None, "model_load_s": None, "warmup_s": None, "io_only_s": None,
                       "unmeasured_status": "not_measured; export time includes IO and forwards, not isolated IO"},
        }
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Adam run: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = summarize_adam_run(args.run_dir)
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return 0
    except (ValueError, OSError) as exc:
        print(f"Adam summary failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
