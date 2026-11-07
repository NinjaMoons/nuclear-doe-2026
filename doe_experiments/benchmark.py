"""Collect completed exports (Adam and trained NN) and compare two already exported methods.

Never reruns physics, inference or the solver and never recomputes Pearson/SSIM: every score is
read from the hashed ``metrics.json`` written by ``export_run``. The legacy Adam API/CLI
(``summarize_adam_run`` and ``--run-dir/--output``) keeps its report format.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np

from doe_core.contracts_v1 import (
    CaseConfigV1, CONTRACT_VERSION_V1, OBSERVATION_V1,
    RECONSTRUCTION_POLICY_V1, SSIM_POLICY_V1,
)

PRIMARY_SOURCE = "reloaded_uint8_PNG_from_reloaded_DOE"
NEURAL_METHOD = "fourinput_unet"
PAIRED_SCHEMA = "doe_paired_benchmark_v1"
# Same relative tolerance the exporter uses for its native-pitch check; not a new threshold.
PITCH_RTOL = 1e-12
NOT_MEASURED = "not_measured"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_KEY = re.compile(r"^([AB])\d+:\d+$")
METRICS = ("pearson", "ssim")


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


def _optional_time(value, name: str) -> float | None:
    return None if value is None else _number(value, name, lower=0)


def _validate_export(run_dir: Path) -> dict:
    """Method-independent checks of one completed v1.1-amplitude export directory."""
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
    height, width = config.image_shape
    pitches = manifest.get("output_pitches_m")
    if not isinstance(pitches, list) or len(pitches) != 4:
        raise ValueError("native output sampling mismatch: four recorded output pitches are required")
    for index, recorded in enumerate(pitches):
        expected = config.plane(index).output_pitch_m(height, width)
        if (not isinstance(recorded, list) or len(recorded) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in recorded)
                or not all(math.isclose(r, e, rel_tol=PITCH_RTOL, abs_tol=0.0)
                           for r, e in zip(recorded, expected))):
            raise ValueError(f"native output sampling mismatch at plane {index}")
    if (manifest.get("reconstruction_policy") != RECONSTRUCTION_POLICY_V1
            or manifest.get("ssim_policy") != SSIM_POLICY_V1):
        raise ValueError("unsupported amplitude or SSIM policy")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("export metadata is required")
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
    from doe_experiments.phase_codec import load_doe_png
    doe_path = _inside(root, "DOE.png")
    load_doe_png(doe_path)  # format only: PNG, L, uint8, 128x128; the phase is not used.
    metrics_path = _inside(root, "metrics.json")
    metrics = _read(metrics_path)
    if (metrics.get("contract") != CONTRACT_VERSION_V1 or metrics.get("observation") != OBSERVATION_V1
            or metrics.get("policy") != RECONSTRUCTION_POLICY_V1
            or metrics.get("ssim_policy") != SSIM_POLICY_V1
            or metrics.get("primary_source") != PRIMARY_SOURCE):
        raise ValueError("metrics must use the submitted amplitude PNG policy")
    planes = metrics.get("planes")
    if not isinstance(planes, list) or len(planes) != 4:
        raise ValueError("exactly four ordered per-plane metric pairs are required")
    doe_hash = _hash(doe_path)
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
        if plane["reconstruction"] != f"reconstructed_{index + 1:02d}.png":
            raise ValueError(f"reconstruction index mismatch at plane {index}")
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
    export_time = _number(manifest.get("elapsed_export_s"), "export elapsed time", lower=0)
    if not isinstance(manifest.get("timing_scope"), str) or not manifest["timing_scope"]:
        raise ValueError("export timing scope is required")
    return {"root": root, "manifest": manifest, "metadata": metadata, "metrics": metrics, "planes": planes,
            "files": files, "sources": sources, "export_time": export_time,
            "manifest_sha256": _hash(manifest_path), "metrics_sha256": _hash(metrics_path)}


def _adam_fields(metadata: dict) -> tuple[str, int, float]:
    if metadata.get("method") != "adam":
        raise ValueError("only Adam solver runs can be summarized")
    if metadata.get("phase_dtype") not in ("float32", "float64"):
        raise ValueError("recorded phase dtype is required")
    seed = metadata.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("recorded nonnegative integer seed is required")
    solver_time = _number(metadata.get("solver_elapsed_s"), "solver elapsed time", lower=0)
    return metadata["phase_dtype"], seed, solver_time


def summarize_adam_run(run_dir: Path) -> dict:
    """Validate hashes/schema and preserve all four primary metric pairs (legacy Adam report)."""
    try:
        run = _validate_export(run_dir)
        manifest, metadata = run["manifest"], run["metadata"]
        dtype, seed, solver_time = _adam_fields(metadata)
        return {
            "schema_version": 1, "method": "adam", "trained_neural": False,
            "learned_comparison": {"status": "not_available", "checkpoint": None},
            "case_contract": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
            "run_dir": str(run["root"]), "synthetic_test": manifest["synthetic_test"],
            "config": manifest["config"], "seed": seed, "dtype": dtype,
            "reconstruction_policy": manifest["reconstruction_policy"], "ssim_policy": manifest["ssim_policy"],
            "planes": run["planes"], "input_sources": run["sources"], "artifact_sha256": run["files"],
            "manifest_sha256": run["manifest_sha256"], "metrics_sha256": run["metrics_sha256"],
            "git_sha": manifest.get("git_sha"), "dependencies": manifest.get("dependencies"),
            "reproduce_argv": metadata.get("reproduce_argv"),
            "timing": {"solver_elapsed_s": solver_time, "export_elapsed_s": run["export_time"],
                       "solver_scope": "CLI solver call, including synthesis and solver final forward",
                       "export_scope": manifest["timing_scope"],
                       "training_s": None, "model_load_s": None, "warmup_s": None, "io_only_s": None,
                       "unmeasured_status": "not_measured; export time includes IO and forwards, not isolated IO"},
        }
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Adam run: {exc}") from exc


def _sha(value, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise ValueError(f"missing or malformed NN checkpoint provenance: {name}")
    return value


def _neural_provenance(metadata: dict) -> dict:
    if metadata.get("method") != NEURAL_METHOD:
        raise ValueError(f"only {NEURAL_METHOD} runs can be summarized as trained NN")
    checkpoint_sha = _sha(metadata.get("checkpoint_sha256"), "checkpoint_sha256")
    fmt, commit = metadata.get("checkpoint_format"), metadata.get("code_commit")
    hparams, training = metadata.get("hparams"), metadata.get("training")
    if not isinstance(fmt, str) or not fmt:
        raise ValueError("missing or malformed NN checkpoint provenance: checkpoint_format")
    if not isinstance(commit, str) or not re.match(r"^[0-9a-f]{40}$", commit):
        raise ValueError("missing or malformed NN checkpoint provenance: code_commit")
    if metadata.get("contract_version") != CONTRACT_VERSION_V1:
        raise ValueError("missing or malformed NN checkpoint provenance: contract_version")
    if not isinstance(hparams, dict) or not hparams:
        raise ValueError("missing or malformed NN checkpoint provenance: hparams")
    if not isinstance(training, dict):
        raise ValueError("missing or malformed NN checkpoint provenance: training")
    seed, steps = training.get("seed"), training.get("steps")
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 1:
        raise ValueError("missing or malformed NN checkpoint provenance: training seed/steps")
    dtypes = {metadata.get(k) for k in ("dtype", "phase_dtype") if k in metadata}
    if len(dtypes) != 1 or not dtypes <= {"float32", "float64"}:
        raise ValueError("missing or malformed NN checkpoint provenance: dtype")
    return {"checkpoint_sha256": checkpoint_sha, "checkpoint_format": fmt, "code_commit": commit,
            "hparams": hparams, "dtype": dtypes.pop(), "torch_version": metadata.get("torch_version"),
            "training": {k: training.get(k) for k in ("method", "propagator", "steps", "lr", "batch_size",
                                                       "seed", "n_train", "n_val", "best_step",
                                                       "observation", "loss_policy")},
            "training_manifest_sha256": metadata.get("manifest_sha256")}


def _plane_scores(planes: list[dict]) -> list[dict]:
    return [{"index": p["index"], "pearson": p["pearson"], "pearson_defined": p["pearson_defined"],
             "ssim": p["ssim"], "reconstruction": p["reconstruction"]} for p in planes]


def summarize_run(run_dir: Path) -> dict:
    """Method-aware summary (schema 2) of one completed Adam or trained-NN export."""
    try:
        run = _validate_export(run_dir)
        manifest, metadata = run["manifest"], run["metadata"]
        method = metadata.get("method")
        base = {
            "schema_version": 2, "method": method, "run_dir": str(run["root"]),
            "case_contract": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
            "synthetic_test": manifest["synthetic_test"], "config": manifest["config"],
            "output_pitches_m": manifest["output_pitches_m"],
            "reconstruction_policy": manifest["reconstruction_policy"], "ssim_policy": manifest["ssim_policy"],
            "primary_source": PRIMARY_SOURCE,
            "sample_key": metadata.get("sample_key"), "split": metadata.get("split"),
            "origin_id": metadata.get("origin_id"),
            "target_sha256": [s["sha256"] for s in run["sources"]],
            "planes": _plane_scores(run["planes"]),
            "plane_maxima": manifest.get("plane_maxima"), "plane_exposures": manifest.get("plane_exposures"),
            "artifact_sha256": run["files"], "manifest_sha256": run["manifest_sha256"],
            "metrics_sha256": run["metrics_sha256"], "git_sha": manifest.get("git_sha"),
            "dependencies": manifest.get("dependencies"), "reproduce_argv": metadata.get("reproduce_argv"),
        }
        if method == "adam":
            dtype, seed, solver_time = _adam_fields(metadata)
            solver_meta = metadata.get("solver_metadata")
            device = solver_meta.get("device") if isinstance(solver_meta, dict) else None
            base.update({
                "trained_neural": False, "dtype": dtype, "device": device or "not_recorded",
                "adam": {"seed": seed, "steps": metadata.get("steps"), "lr": metadata.get("lr")},
                "timing": {"solver_elapsed_s": solver_time,
                           "solver_scope": "CLI solver call, including synthesis and solver final forward",
                           "export_elapsed_s": run["export_time"], "export_scope": manifest["timing_scope"],
                           "checkpoint_load_s": None, "inference_s": None, "training_s": None,
                           "unmeasured_status": NOT_MEASURED},
            })
        elif method == NEURAL_METHOD:
            provenance = _neural_provenance(metadata)
            verified = metadata.get("training_verified")
            if verified is not None and type(verified) is not bool:
                raise ValueError("training_verified must be a boolean when recorded")
            scope = metadata.get("neural_timing_scope")
            scope = scope if isinstance(scope, dict) else {}
            load_s = _optional_time(metadata.get("checkpoint_load_elapsed_s"), "checkpoint load time")
            infer_s = _optional_time(metadata.get("inference_elapsed_s"), "inference time")
            base.update({
                "trained_neural": True, "dtype": provenance["dtype"], "device": "not_recorded",
                "neural": provenance,
                # Recorded fact only; an existing checkpoint never implies verified training.
                "training_verified": verified if verified is not None else "not_recorded",
                "training_verification": metadata.get("training_verification", "not_recorded_in_export"),
                "independent_training_acceptance": "not_recorded_by_collector",
                "timing": {"checkpoint_load_s": load_s,
                           "checkpoint_load_scope": scope.get("checkpoint_load") if load_s is not None else NOT_MEASURED,
                           "inference_s": infer_s,
                           "inference_scope": scope.get("inference") if infer_s is not None else NOT_MEASURED,
                           "export_elapsed_s": run["export_time"], "export_scope": manifest["timing_scope"],
                           "solver_elapsed_s": None, "training_s": None,
                           "unmeasured_status": NOT_MEASURED},
            })
        else:
            raise ValueError("only adam or fourinput_unet exports can be summarized")
        return base
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run: {exc}") from exc


# ------------------------------------------------------------------ paired NN vs Adam comparison
def _group(sample_key: str) -> str:
    match = _SAMPLE_KEY.match(sample_key)
    return match.group(1) if match else "other"


def _quartet_mean(planes: list[dict], metric: str) -> float | None:
    if metric == "pearson" and not all(p["pearson_defined"] for p in planes):
        return None  # undefined stays undefined; never replaced by 0 or by a 3-plane mean
    return float(np.mean([p[metric] for p in planes]))


def _describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "q1": None, "q3": None, "iqr": None, "wins": 0, "ties": 0, "losses": 0}
    array = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(array, [25, 75], method="linear")
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max()),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "wins": int((array > 0).sum()), "ties": int((array == 0).sum()), "losses": int((array < 0).sum())}


def _aggregate(records: list[dict], metric: str) -> dict:
    def block(rows: list[dict]) -> dict:
        used = [r for r in rows if r["delta"][metric] is not None]
        return {"n_pairs": len(rows), "n_defined": len(used),
                "undefined_sample_keys": [r["sample_key"] for r in rows if r["delta"][metric] is None],
                "nn_mean": float(np.mean([r["nn_quartet_mean"][metric] for r in used])) if used else None,
                "adam_mean": float(np.mean([r["adam_quartet_mean"][metric] for r in used])) if used else None,
                "delta": _describe([r["delta"][metric] for r in used])}

    groups = {g: block([r for r in records if r["group"] == g]) for g in ("A", "B")}
    other = [r for r in records if r["group"] == "other"]
    pooled = block(records)
    a, b = groups["A"], groups["B"]
    if not other and a["n_defined"] and a["n_defined"] == b["n_defined"]:
        half = {k: (a[k] + b[k]) / 2 for k in ("nn_mean", "adam_mean")}
        all_block = {"definition": "half-sum of A and B group means (equal defined group sizes)",
                     "n_defined": a["n_defined"] + b["n_defined"], **half,
                     "mean_delta": (a["delta"]["mean"] + b["delta"]["mean"]) / 2}
    else:
        all_block = {"definition": "pooled mean over defined quartets (A/B sizes unequal or other groups present)",
                     "n_defined": pooled["n_defined"], "nn_mean": pooled["nn_mean"],
                     "adam_mean": pooled["adam_mean"], "mean_delta": pooled["delta"]["mean"]}
    per_plane = []
    for index in range(4):
        pairs = [(r["nn_planes"][index], r["adam_planes"][index]) for r in records]
        defined = [(n[metric], a_[metric]) for n, a_ in pairs
                   if metric != "pearson" or (n["pearson_defined"] and a_["pearson_defined"])]
        per_plane.append({"index": index, "n_defined": len(defined), "n_undefined": len(pairs) - len(defined),
                          "nn_mean": float(np.mean([n for n, _ in defined])) if defined else None,
                          "adam_mean": float(np.mean([a_ for _, a_ in defined])) if defined else None,
                          "mean_delta": float(np.mean([n - a_ for n, a_ in defined])) if defined else None})
    worst = min((r for r in records if r["delta"][metric] is not None),
                key=lambda r: r["delta"][metric], default=None)
    return {"ALL": all_block, "pooled": pooled, "A": a, "B": b,
            **({"other": block(other)} if other else {}),
            "per_plane": per_plane,
            "worst_pair": None if worst is None else {"sample_key": worst["sample_key"],
                                                      "delta": worst["delta"][metric]}}


def _pair_key(explicit: str | None, nn: dict, adam: dict) -> str:
    recorded = {v for v in (nn["sample_key"], adam["sample_key"]) if v is not None}
    for value in recorded:
        if not isinstance(value, str) or not value:
            raise ValueError("recorded sample_key must be a nonempty string")
    if explicit is not None:
        recorded.add(explicit)
    if not recorded:
        raise ValueError("sample_key is required (not recorded in either export; pass it explicitly)")
    if len(recorded) != 1:
        raise ValueError(f"sample_key mismatch: {sorted(recorded)}")
    return recorded.pop()


def _check_pair(nn: dict, adam: dict) -> None:
    for index, (left, right) in enumerate(zip(nn["target_sha256"], adam["target_sha256"])):
        if left != right:
            raise ValueError(f"ordered target quartet mismatch at plane {index}")
    for field in ("case_contract", "observation", "config", "reconstruction_policy", "ssim_policy",
                  "primary_source"):
        if nn[field] != adam[field]:
            raise ValueError(f"{field} mismatch between NN and Adam exports")
    for index, (left, right) in enumerate(zip(nn["output_pitches_m"], adam["output_pitches_m"])):
        if not all(math.isclose(x, y, rel_tol=PITCH_RTOL, abs_tol=0.0) for x, y in zip(left, right)):
            raise ValueError(f"native output sampling mismatch at plane {index}")
    if nn["dtype"] != adam["dtype"]:
        raise ValueError(f"phase dtype mismatch: NN {nn['dtype']} vs Adam {adam['dtype']}")
    if nn["synthetic_test"] != adam["synthetic_test"]:
        raise ValueError("synthetic_test flag mismatch between NN and Adam exports")


def compare_runs(pairs: list[tuple[str | None, Path, Path]]) -> dict:
    """Pair already exported NN and Adam runs of the same quartets; nothing is recomputed.

    ``pairs`` holds ``(sample_key or None, nn_run_dir, adam_run_dir)``. Matching is by sample_key
    plus the ORDERED four target SHA256; plane order is never rearranged.
    """
    if not pairs:
        raise ValueError("at least one NN/Adam pair is required")
    records, keys, quartets, nn_profiles, adam_profiles = [], set(), set(), set(), set()
    for explicit, nn_dir, adam_dir in pairs:
        nn, adam = summarize_run(nn_dir), summarize_run(adam_dir)
        if nn["method"] != NEURAL_METHOD:
            raise ValueError(f"first run of a pair must be {NEURAL_METHOD}, got {nn['method']}")
        if adam["method"] != "adam":
            raise ValueError(f"second run of a pair must be adam, got {adam['method']}")
        key = _pair_key(explicit, nn, adam)
        _check_pair(nn, adam)
        quartet = tuple(nn["target_sha256"])
        if key in keys or quartet in quartets:
            raise ValueError(f"duplicate sample_key or target quartet: {key}")
        keys.add(key)
        quartets.add(quartet)
        nn_profiles.add(json.dumps({k: nn["neural"][k] for k in ("checkpoint_sha256", "code_commit")}
                                   | {"dtype": nn["dtype"]}, sort_keys=True))
        adam_profiles.add(json.dumps(adam["adam"] | {"dtype": adam["dtype"]}, sort_keys=True))
        nn_mean = {m: _quartet_mean(nn["planes"], m) for m in METRICS}
        adam_mean = {m: _quartet_mean(adam["planes"], m) for m in METRICS}
        records.append({
            "sample_key": key, "group": _group(key), "split": nn["split"] or adam["split"],
            "target_sha256": nn["target_sha256"],
            "nn_run": {"run_dir": nn["run_dir"], "manifest_sha256": nn["manifest_sha256"],
                       "metrics_sha256": nn["metrics_sha256"], "timing": nn["timing"],
                       "training_verified": nn["training_verified"]},
            "adam_run": {"run_dir": adam["run_dir"], "manifest_sha256": adam["manifest_sha256"],
                         "metrics_sha256": adam["metrics_sha256"], "timing": adam["timing"]},
            "nn_planes": nn["planes"], "adam_planes": adam["planes"],
            "nn_quartet_mean": nn_mean, "adam_quartet_mean": adam_mean,
            "delta": {m: (None if nn_mean[m] is None or adam_mean[m] is None else nn_mean[m] - adam_mean[m])
                      for m in METRICS},
            "_nn": nn, "_adam": adam,
        })
    if len(nn_profiles) != 1:
        raise ValueError("all pairs must use one NN checkpoint/code commit/dtype")
    if len(adam_profiles) != 1:
        raise ValueError("all pairs must use one fixed Adam profile (seed/steps/lr/dtype)")
    first_nn, first_adam = records[0]["_nn"], records[0]["_adam"]
    for record in records:
        del record["_nn"], record["_adam"]
    devices = {"nn": first_nn["device"], "adam": first_adam["device"]}
    reasons = ["NN and Adam timings measure different scopes (inference vs iterative solver)"]
    if "not_recorded" in devices.values():
        reasons.append("execution device not recorded in at least one export")
    elif devices["nn"] != devices["adam"]:
        reasons.append(f"different devices: NN {devices['nn']} vs Adam {devices['adam']}")
    return {
        "schema": PAIRED_SCHEMA, "case_contract": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
        "config": first_nn["config"], "output_pitches_m": first_nn["output_pitches_m"],
        "reconstruction_policy": first_nn["reconstruction_policy"], "ssim_policy": first_nn["ssim_policy"],
        "primary_source": PRIMARY_SOURCE, "synthetic_test": first_nn["synthetic_test"],
        "nn_profile": {"method": NEURAL_METHOD, **first_nn["neural"], "dtype": first_nn["dtype"],
                       "device": devices["nn"], "training_verified": first_nn["training_verified"],
                       "training_verification": first_nn["training_verification"],
                       "independent_training_acceptance": first_nn["independent_training_acceptance"]},
        "adam_profile": {"method": "adam", **first_adam["adam"], "dtype": first_adam["dtype"],
                         "device": devices["adam"]},
        "pairing": "sample_key + ordered target SHA256 (plane order never rearranged)",
        "unit_of_analysis": "quartet: per-method mean over 4 planes; delta = NN - Adam",
        "independence_note": ("the 4 planes of a quartet are not independent observations; independence "
                              "across quartets requires the corpus group/split conditions"),
        "statistics_note": "descriptive only; no CI, p-values, Wilcoxon or bootstrap",
        "iqr_convention": "numpy.percentile method='linear' (Hyndman-Fan type 7)",
        "undefined_pearson": "a quartet with any undefined plane Pearson has undefined quartet mean and delta",
        "n_pairs": len(records),
        "aggregates": {m: _aggregate(records, m) for m in METRICS},
        "timing_comparison": {"status": "not_comparable", "reasons": reasons,
                              "speedup": "not_reported"},
        "pairs": records,
    }


def _write_new(path: Path, report: dict) -> None:
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:  # never overwrite a report
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="one completed export (legacy Adam summary by default)")
    source.add_argument("--pair", nargs=3, action="append", metavar=("SAMPLE_KEY", "NN_RUN_DIR", "ADAM_RUN_DIR"),
                        help="NN/Adam exports of one quartet; repeat per quartet; SAMPLE_KEY '-' = recorded key")
    parser.add_argument("--method", choices=("adam", "any"), default="adam",
                        help="with --run-dir: 'adam' = legacy Adam report, 'any' = schema-2 Adam/NN summary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.pair:
            report = compare_runs([(None if key == "-" else key, Path(nn), Path(adam))
                                   for key, nn, adam in args.pair])
        elif args.method == "any":
            report = summarize_run(args.run_dir)
        else:
            report = summarize_adam_run(args.run_dir)
        _write_new(args.output, report)
        return 0
    except (ValueError, OSError) as exc:
        label = "Paired benchmark" if args.pair else ("Run summary" if args.method == "any" else "Adam summary")
        print(f"{label} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
