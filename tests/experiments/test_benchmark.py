"""Artifact/schema unit doubles only: no scientific or real-run acceptance."""
from dataclasses import asdict
import hashlib
import json

import pytest

from doe_core.contracts_v1 import CaseConfigV1, RECONSTRUCTION_POLICY_V1, SSIM_POLICY_V1
from doe_experiments.benchmark import main, summarize_adam_run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifact_double(tmp_path):
    root = tmp_path / "explicit-unit-double"
    root.mkdir()
    names = ["DOE.png", "raw.npz", *(f"reconstructed_{i:02d}.png" for i in range(1, 5)),
             *(f"targets/target_{i:02d}.png" for i in range(1, 5))]
    for name in names:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"EXPLICIT UNIT TEST DOUBLE {name}".encode())
    config = CaseConfigV1()
    metrics = {"contract": "1.1-amplitude", "observation": "amplitude",
               "policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
               "primary_source": "reloaded_uint8_PNG_from_reloaded_DOE",
               "planes": [{"index": i, "distance_m": config.distances_m[i],
                           "doe_sha256": digest(root / "DOE.png"), "reconstruction": f"reconstructed_{i+1:02d}.png",
                           "pearson": [1.0, -0.6, None, 0.2][i], "pearson_defined": i != 2,
                           "ssim": [1.0, -0.4, 0.1, 0.3][i]} for i in range(4)]}
    write_json(root / "metrics.json", metrics)
    manifest = {"schema_version": 2, "completed": True, "case_contract": "1.1-amplitude",
                "observation": "amplitude", "config": asdict(config), "synthetic_test": True,
                "reconstruction_policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
                "files": {name: digest(root / name) for name in [*names, "metrics.json"]},
                "input_sources": [{"path": f"original-test-double-{i}.png", "sha256": digest(root / f"targets/target_{i:02d}.png")} for i in range(1, 5)],
                "metadata": {"method": "adam", "seed": 17, "phase_dtype": "float32", "solver_elapsed_s": 0.25},
                "elapsed_export_s": 0.5, "timing_scope": "explicit unit double timing"}
    write_json(root / "manifest.json", manifest)
    return root, manifest, metrics


def update(root, manifest, metrics):
    write_json(root / "metrics.json", metrics)
    manifest["files"]["metrics.json"] = digest(root / "metrics.json")
    write_json(root / "manifest.json", manifest)


def test_preserves_all_eight_scores_flags_and_measured_scopes(artifact_double):
    root, manifest, metrics = artifact_double
    result = summarize_adam_run(root)
    assert result["planes"] == metrics["planes"]
    assert result["trained_neural"] is False
    assert result["learned_comparison"] == {"status": "not_available", "checkpoint": None}
    assert result["synthetic_test"] is True
    assert result["seed"] == 17 and result["dtype"] == "float32"
    assert result["input_sources"] == manifest["input_sources"]
    assert result["artifact_sha256"] == manifest["files"]
    assert result["manifest_sha256"] == digest(root / "manifest.json")
    assert result["timing"]["solver_elapsed_s"] == 0.25
    assert result["timing"]["export_elapsed_s"] == 0.5
    assert all(result["timing"][name] is None for name in ("training_s", "warmup_s", "model_load_s", "io_only_s"))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["not_completed", "intensity", "contract", "policy", "ssim", "method", "unknown_method", "dtype", "seed", "config", "time", "missing_hash", "bad_hash", "source_hash"])
def test_invalid_manifest_fails(artifact_double, fault):
    root, manifest, metrics = artifact_double
    if fault == "not_completed": manifest["completed"] = False
    elif fault == "intensity": manifest["observation"] = "intensity"
    elif fault == "contract": manifest["case_contract"] = "1.0-intensity"
    elif fault == "policy": manifest["reconstruction_policy"] = "intensity_max"
    elif fault == "ssim": manifest["ssim_policy"] = "windowed"
    elif fault == "method": manifest["metadata"]["method"] = "reconstruct"
    elif fault == "unknown_method": manifest["metadata"]["method"] = "unrecognized"
    elif fault == "dtype": manifest["metadata"]["phase_dtype"] = "float16"
    elif fault == "seed": manifest["metadata"]["seed"] = True
    elif fault == "config": manifest["config"]["wavelength_m"] = 633e-9
    elif fault == "time": manifest["metadata"]["solver_elapsed_s"] = -1
    elif fault == "missing_hash": del manifest["files"]["raw.npz"]
    elif fault == "bad_hash": manifest["files"]["DOE.png"] = "0" * 64
    elif fault == "source_hash": manifest["input_sources"][0]["sha256"] = "0" * 64
    update(root, manifest, metrics)
    with pytest.raises(ValueError):
        summarize_adam_run(root)


@pytest.mark.parametrize("fault", ["planes", "order", "distance", "doe_hash", "duplicate_recon", "missing_recon", "pearson", "missing_pearson", "defined", "ssim", "intensity"])
def test_invalid_plane_metrics_fail_even_with_updated_file_hash(artifact_double, fault):
    root, manifest, metrics = artifact_double
    plane = metrics["planes"][0]
    if fault == "planes": metrics["planes"].pop()
    elif fault == "order": plane["index"] = 3
    elif fault == "distance": plane["distance_m"] = 0.03
    elif fault == "doe_hash": plane["doe_sha256"] = "0" * 64
    elif fault == "duplicate_recon": plane["reconstruction"] = "reconstructed_02.png"
    elif fault == "missing_recon": plane["reconstruction"] = "absent.png"
    elif fault == "pearson": plane["pearson"] = 2
    elif fault == "missing_pearson":
        del plane["pearson"]
        plane["pearson_defined"] = False
    elif fault == "defined": plane["pearson_defined"] = False
    elif fault == "ssim": plane["ssim"] = None
    elif fault == "intensity": metrics["observation"] = "intensity"
    update(root, manifest, metrics)
    with pytest.raises(ValueError):
        summarize_adam_run(root)


@pytest.mark.parametrize("name", ["DOE.png", "raw.npz", "reconstructed_03.png", "targets/target_02.png", "metrics.json"])
@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_artifact_integrity(artifact_double, name, fault):
    root, _, _ = artifact_double
    if fault == "missing": (root / name).unlink()
    else: (root / name).write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        summarize_adam_run(root)


@pytest.mark.parametrize("escape", ["../outside.png", "C:/outside.png", "/outside.png"])
def test_manifest_path_escape(artifact_double, escape):
    root, manifest, metrics = artifact_double
    (root.parent / "outside.png").write_bytes(b"outside")
    manifest["files"][escape] = digest(root.parent / "outside.png")
    update(root, manifest, metrics)
    with pytest.raises(ValueError):
        summarize_adam_run(root)


def test_cli_writes_new_report_and_refuses_overwrite(artifact_double, tmp_path):
    root, _, metrics = artifact_double
    output = tmp_path / "summary.json"
    assert main(["--run-dir", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["planes"] == metrics["planes"]
    before = output.read_bytes()
    assert main(["--run-dir", str(root), "--output", str(output)]) == 2
    assert output.read_bytes() == before


def test_cli_failed_run_never_produces_success_report(artifact_double, tmp_path):
    root, _, _ = artifact_double
    (root / "DOE.png").unlink()
    output = tmp_path / "summary.json"
    assert main(["--run-dir", str(root), "--output", str(output)]) == 2
    assert not output.exists()
