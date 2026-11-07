"""Artifact/schema unit doubles only: no scientific or real-run acceptance."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from doe_core.contracts_v1 import CaseConfigV1, RECONSTRUCTION_POLICY_V1, SSIM_POLICY_V1
from doe_experiments.benchmark import compare_runs, main, summarize_adam_run, summarize_run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_png(path, seed):
    pixels = np.random.default_rng(seed).integers(0, 256, size=(128, 128), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path, format="PNG")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifact_double(tmp_path):
    root = tmp_path / "explicit-unit-double"
    root.mkdir()
    names = ["DOE.png", "raw.npz", *(f"reconstructed_{i:02d}.png" for i in range(1, 5)),
             *(f"targets/target_{i:02d}.png" for i in range(1, 5))]
    for index, name in enumerate(names):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        if name == "raw.npz":
            path.write_bytes(f"EXPLICIT UNIT TEST DOUBLE {name}".encode())
        else:  # explicit synthetic 128x128 L PNG doubles; not physics output
            write_png(path, index)
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
                "output_pitches_m": [list(config.plane(i).output_pitch_m(128, 128)) for i in range(4)],
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


@pytest.mark.parametrize("escape", ["../outside.png", "..\\outside.png", "C:/outside.png", "/outside.png", "\\\\server\\share\\outside.png"])
def test_manifest_path_escape(artifact_double, escape):
    root, manifest, metrics = artifact_double
    (root.parent / "outside.png").write_bytes(b"outside")
    manifest["files"][escape] = digest(root.parent / "outside.png")
    update(root, manifest, metrics)
    with pytest.raises(ValueError):
        summarize_adam_run(root)


def test_accepts_windows_manifest_separators_on_any_host(artifact_double):
    root, manifest, metrics = artifact_double
    manifest["files"] = {name.replace("/", "\\"): value for name, value in manifest["files"].items()}
    update(root, manifest, metrics)
    report = summarize_adam_run(root)
    assert report["artifact_sha256"] == manifest["files"]
    assert report["planes"] == metrics["planes"]


def test_rejects_duplicate_separator_aliases(artifact_double):
    root, manifest, metrics = artifact_double
    manifest["files"]["targets\\target_01.png"] = manifest["files"]["targets/target_01.png"]
    update(root, manifest, metrics)
    with pytest.raises(ValueError, match="duplicate artifact path"):
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


# ------------------------------------------------------------------ #6B: NN summary and paired NN/Adam
CKPT = "ab" * 32
COMMIT = "49b96041b3b4bf1b2eaa57567939bcd929defd2e"


def neural_metadata(sample_key="A64:014"):
    return {"method": "fourinput_unet", "checkpoint_format": "doe_fourinput_unet_v1",
            "checkpoint_sha256": CKPT, "code_commit": COMMIT, "contract_version": "1.1-amplitude",
            "hparams": {"base_channels": 16, "depth": 3}, "dtype": "float32", "torch_version": "unit-double",
            "training": {"method": "fourinput_unet_physics", "steps": 1000, "seed": 0, "lr": 0.001},
            "sample_key": sample_key, "split": "test", "manifest_sha256": "cd" * 32}


def adam_metadata():
    return {"method": "adam", "seed": 17, "steps": 16, "lr": 0.05, "phase_dtype": "float32",
            "solver_elapsed_s": 0.25, "solver_metadata": {"observation": "amplitude"}}


def build_run(root, metadata, *, target_seeds=(101, 102, 103, 104), pearson=(0.5, 0.5, 0.5, 0.5),
              ssim=(0.4, 0.4, 0.4, 0.4)):
    """Explicit synthetic export double with real PNG files and consistent hashes."""
    root.mkdir(parents=True)
    (root / "targets").mkdir()
    for i, seed in enumerate(target_seeds, 1):
        write_png(root / f"targets/target_{i:02d}.png", seed)
    write_png(root / "DOE.png", 7)
    for i in range(1, 5):
        write_png(root / f"reconstructed_{i:02d}.png", 200 + i)
    (root / "raw.npz").write_bytes(b"EXPLICIT UNIT TEST DOUBLE raw.npz")
    config = CaseConfigV1()
    metrics = {"contract": "1.1-amplitude", "observation": "amplitude",
               "policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
               "primary_source": "reloaded_uint8_PNG_from_reloaded_DOE",
               "planes": [{"index": i, "distance_m": config.distances_m[i], "doe_sha256": digest(root / "DOE.png"),
                           "reconstruction": f"reconstructed_{i+1:02d}.png", "pearson": pearson[i],
                           "pearson_defined": pearson[i] is not None, "ssim": ssim[i]} for i in range(4)]}
    write_json(root / "metrics.json", metrics)
    names = ["DOE.png", "raw.npz", "metrics.json", *(f"reconstructed_{i:02d}.png" for i in range(1, 5)),
             *(f"targets\\target_{i:02d}.png" for i in range(1, 5))]
    manifest = {"schema_version": 2, "completed": True, "case_contract": "1.1-amplitude",
                "observation": "amplitude", "config": asdict(config), "synthetic_test": True,
                "output_pitches_m": [list(config.plane(i).output_pitch_m(128, 128)) for i in range(4)],
                "reconstruction_policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
                "files": {n: digest(root / n.replace("\\", "/")) for n in names},
                "input_sources": [{"path": f"C:\\data\\{i}.png", "sha256": digest(root / f"targets/target_{i:02d}.png")}
                                  for i in range(1, 5)],
                "metadata": metadata, "elapsed_export_s": 0.5, "timing_scope": "explicit unit double timing"}
    write_json(root / "manifest.json", manifest)
    return root


def rewrite_manifest(root, change):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    change(manifest)
    write_json(root / "manifest.json", manifest)


def make_pair(tmp_path, key="A64:014", seeds=(101, 102, 103, 104), nn=(0.6,) * 4, adam=(0.5,) * 4,
              nn_ssim=(0.5,) * 4, adam_ssim=(0.4,) * 4):
    tag = key.replace(":", "_")
    nn_dir = build_run(tmp_path / f"nn-{tag}", neural_metadata(key), target_seeds=seeds, pearson=nn, ssim=nn_ssim)
    adam_dir = build_run(tmp_path / f"adam-{tag}", adam_metadata(), target_seeds=seeds, pearson=adam, ssim=adam_ssim)
    return nn_dir, adam_dir


def test_summary_of_both_methods(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path)
    nn, adam = summarize_run(nn_dir), summarize_run(adam_dir)
    assert nn["method"] == "fourinput_unet" and nn["trained_neural"] is True
    assert nn["neural"]["checkpoint_sha256"] == CKPT and nn["neural"]["training"]["seed"] == 0
    assert nn["neural"]["training"]["steps"] == 1000 and nn["dtype"] == "float32"
    assert nn["training_verified"] == "not_recorded"
    assert nn["independent_training_acceptance"] == "not_recorded_by_collector"
    assert nn["timing"]["inference_s"] is None and nn["timing"]["inference_scope"] == "not_measured"
    assert nn["sample_key"] == "A64:014" and nn["device"] == "not_recorded"
    assert adam["method"] == "adam" and adam["adam"] == {"seed": 17, "steps": 16, "lr": 0.05}
    assert adam["timing"]["solver_elapsed_s"] == 0.25 and adam["timing"]["inference_s"] is None
    assert nn["target_sha256"] == adam["target_sha256"]
    json.dumps(nn, allow_nan=False)


def test_cli_neural_training_verified_false_is_kept_as_fact(tmp_path):
    meta = neural_metadata(None) | {"training_verified": False,
                                    "training_verification": "not_verified_by_inference_cli",
                                    "phase_dtype": "float32", "checkpoint_load_elapsed_s": 0.1,
                                    "inference_elapsed_s": 0.02,
                                    "neural_timing_scope": {"checkpoint_load": "load scope", "inference": "infer scope"}}
    report = summarize_run(build_run(tmp_path / "nn-cli", meta))
    assert report["training_verified"] is False
    assert report["training_verification"] == "not_verified_by_inference_cli"
    assert report["timing"]["inference_s"] == 0.02 and report["timing"]["inference_scope"] == "infer scope"


@pytest.mark.parametrize("fault", ["no_checkpoint", "bad_sha", "no_commit", "no_training", "seed", "steps",
                                   "dtype_conflict", "contract", "verified_str"])
def test_missing_or_malformed_neural_provenance_fails(tmp_path, fault):
    meta = neural_metadata()
    if fault == "no_checkpoint": del meta["checkpoint_sha256"]
    elif fault == "bad_sha": meta["checkpoint_sha256"] = "not-a-sha"
    elif fault == "no_commit": meta["code_commit"] = None
    elif fault == "no_training": del meta["training"]
    elif fault == "seed": meta["training"]["seed"] = "0"
    elif fault == "steps": del meta["training"]["steps"]
    elif fault == "dtype_conflict": meta["phase_dtype"] = "float64"
    elif fault == "contract": meta["contract_version"] = "1.0-intensity"
    elif fault == "verified_str": meta["training_verified"] = "yes"
    with pytest.raises(ValueError):
        summarize_run(build_run(tmp_path / "nn", meta))


def test_neural_run_is_not_accepted_by_legacy_adam_api(tmp_path):
    with pytest.raises(ValueError, match="only Adam"):
        summarize_adam_run(build_run(tmp_path / "nn", neural_metadata()))


def test_valid_same_quartet_pair(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path, nn=(0.6, 0.7, 0.8, 0.9), adam=(0.5, 0.5, 0.5, 0.5))
    report = compare_runs([(None, nn_dir, adam_dir)])
    pair = report["pairs"][0]
    assert pair["sample_key"] == "A64:014" and pair["group"] == "A"
    assert pair["nn_quartet_mean"]["pearson"] == pytest.approx(0.75)
    assert pair["delta"]["pearson"] == pytest.approx(0.25)
    assert pair["delta"]["ssim"] == pytest.approx(0.1)
    assert report["nn_profile"]["checkpoint_sha256"] == CKPT and report["adam_profile"]["seed"] == 17
    assert report["timing_comparison"]["status"] == "not_comparable"
    assert report["timing_comparison"]["speedup"] == "not_reported"
    assert report["aggregates"]["pearson"]["per_plane"][3]["mean_delta"] == pytest.approx(0.4)
    json.dumps(report, allow_nan=False)


def test_aggregates_all_a_b_and_worst_pair(tmp_path):
    pairs = []
    for i, (key, nn, adam) in enumerate([("A64:014", 0.8, 0.5), ("A64:015", 0.6, 0.7),
                                         ("B64:003", 0.4, 0.4), ("B64:005", 0.3, 0.1)]):
        nn_dir, adam_dir = make_pair(tmp_path, key, seeds=tuple(10 * i + k for k in range(4)),
                                     nn=(nn,) * 4, adam=(adam,) * 4)
        pairs.append((None, nn_dir, adam_dir))
    agg = compare_runs(pairs)["aggregates"]["pearson"]
    assert agg["A"]["delta"]["mean"] == pytest.approx(0.1) and agg["B"]["delta"]["mean"] == pytest.approx(0.1)
    assert agg["ALL"]["definition"].startswith("half-sum")
    assert agg["ALL"]["mean_delta"] == pytest.approx(0.1)
    assert agg["ALL"]["nn_mean"] == pytest.approx((0.7 + 0.35) / 2)
    assert agg["pooled"]["delta"]["wins"] == 2 and agg["pooled"]["delta"]["ties"] == 1
    assert agg["pooled"]["delta"]["losses"] == 1
    assert agg["worst_pair"]["sample_key"] == "A64:015"
    assert agg["worst_pair"]["delta"] == pytest.approx(-0.1)


def test_undefined_pearson_is_explicit_not_zero(tmp_path):
    a_nn, a_adam = make_pair(tmp_path, "A64:014", seeds=(1, 2, 3, 4), nn=(0.6, None, 0.6, 0.6))
    b_nn, b_adam = make_pair(tmp_path, "B64:003", seeds=(5, 6, 7, 8))
    report = compare_runs([(None, a_nn, a_adam), (None, b_nn, b_adam)])
    agg = report["aggregates"]["pearson"]
    assert report["pairs"][0]["delta"]["pearson"] is None
    assert report["pairs"][0]["delta"]["ssim"] is not None
    assert agg["A"]["undefined_sample_keys"] == ["A64:014"] and agg["A"]["n_defined"] == 0
    assert agg["ALL"]["definition"].startswith("pooled")
    assert agg["per_plane"][1]["n_undefined"] == 1 and agg["per_plane"][0]["n_undefined"] == 0


def test_permuted_target_order_is_rejected_not_reordered(tmp_path):
    nn_dir = build_run(tmp_path / "nn", neural_metadata(), target_seeds=(101, 102, 103, 104))
    adam_dir = build_run(tmp_path / "adam", adam_metadata(), target_seeds=(102, 101, 103, 104))
    with pytest.raises(ValueError, match="ordered target quartet mismatch at plane 0"):
        compare_runs([(None, nn_dir, adam_dir)])


def test_swapped_reconstruction_index_is_rejected(tmp_path):
    nn_dir, _ = make_pair(tmp_path)
    metrics = json.loads((nn_dir / "metrics.json").read_text(encoding="utf-8"))
    planes = metrics["planes"]
    planes[0]["reconstruction"], planes[1]["reconstruction"] = planes[1]["reconstruction"], planes[0]["reconstruction"]
    write_json(nn_dir / "metrics.json", metrics)
    rewrite_manifest(nn_dir, lambda m: m["files"].update({"metrics.json": digest(nn_dir / "metrics.json")}))
    with pytest.raises(ValueError, match="reconstruction index mismatch at plane 0"):
        summarize_run(nn_dir)


@pytest.mark.parametrize("pitches", ["grid256", "input8um", "missing"])
def test_wrong_native_output_pitch_is_rejected(tmp_path, pitches):
    config = CaseConfigV1()
    value = None
    if pitches == "grid256":
        value = [list(config.plane(i).output_pitch_m(256, 256)) for i in range(4)]
    elif pitches == "input8um":
        value = [[config.wavelength_m * z / (128 * 8e-6)] * 2 for z in config.distances_m]
    run = build_run(tmp_path / "adam", adam_metadata())
    rewrite_manifest(run, (lambda m: m.pop("output_pitches_m")) if value is None
                     else (lambda m: m.update({"output_pitches_m": value})))
    with pytest.raises(ValueError, match="native output sampling mismatch"):
        summarize_run(run)


@pytest.mark.parametrize("fault", ["key", "dtype", "synthetic", "two_nn", "two_adam", "duplicate",
                                   "checkpoint_profile", "adam_profile", "missing_key"])
def test_incompatible_pairs_fail(tmp_path, fault):
    nn_dir, adam_dir = make_pair(tmp_path)
    pairs = [(None, nn_dir, adam_dir)]
    if fault == "key":
        pairs = [("A64:015", nn_dir, adam_dir)]
    elif fault == "dtype":
        rewrite_manifest(adam_dir, lambda m: m["metadata"].update({"phase_dtype": "float64"}))
    elif fault == "synthetic":
        rewrite_manifest(adam_dir, lambda m: m.update({"synthetic_test": False}))
    elif fault == "two_nn":
        pairs = [(None, nn_dir, nn_dir)]
    elif fault == "two_adam":
        pairs = [("A64:014", adam_dir, adam_dir)]
    elif fault == "duplicate":
        pairs = [(None, nn_dir, adam_dir), (None, nn_dir, adam_dir)]
    elif fault in ("checkpoint_profile", "adam_profile"):
        nn2, adam2 = make_pair(tmp_path, "B64:003", seeds=(5, 6, 7, 8))
        if fault == "checkpoint_profile":
            rewrite_manifest(nn2, lambda m: m["metadata"].update({"checkpoint_sha256": "ef" * 32}))
        else:
            rewrite_manifest(adam2, lambda m: m["metadata"].update({"seed": 18}))
        pairs.append((None, nn2, adam2))
    elif fault == "missing_key":
        rewrite_manifest(nn_dir, lambda m: m["metadata"].pop("sample_key"))
    with pytest.raises(ValueError):
        compare_runs(pairs)


def test_explicit_key_pairs_cli_neural_export_without_recorded_key(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path)
    rewrite_manifest(nn_dir, lambda m: m["metadata"].pop("sample_key"))
    assert compare_runs([("A64:014", nn_dir, adam_dir)])["pairs"][0]["sample_key"] == "A64:014"


def test_paired_cli_writes_new_report_and_refuses_overwrite(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path)
    output = tmp_path / "paired.json"
    assert main(["--pair", "-", str(nn_dir), str(adam_dir), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "doe_paired_benchmark_v1" and report["n_pairs"] == 1
    before = output.read_bytes()
    assert main(["--pair", "-", str(nn_dir), str(adam_dir), "--output", str(output)]) == 2
    assert output.read_bytes() == before
    failed = tmp_path / "failed.json"
    assert main(["--pair", "-", str(adam_dir), str(nn_dir), "--output", str(failed)]) == 2
    assert not failed.exists()


def test_run_summary_cli_any_method(tmp_path):
    nn_dir, _ = make_pair(tmp_path)
    output = tmp_path / "nn-summary.json"
    assert main(["--run-dir", str(nn_dir), "--method", "any", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["method"] == "fourinput_unet"


def test_legacy_adam_report_keys_unchanged(artifact_double):
    root, _, _ = artifact_double
    assert set(summarize_adam_run(root)) == {
        "schema_version", "method", "trained_neural", "learned_comparison", "case_contract", "observation",
        "run_dir", "synthetic_test", "config", "seed", "dtype", "reconstruction_policy", "ssim_policy", "planes",
        "input_sources", "artifact_sha256", "manifest_sha256", "metrics_sha256", "git_sha", "dependencies",
        "reproduce_argv", "timing"}


# ------------------------------------------------------------------ COORDINATOR_FIX bb8f660 regressions
def test_frozen_driver_nn_with_adam_cli_without_split_is_accepted(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path)  # NN: split=test + eval manifest; Adam CLI: neither
    report = compare_runs([(None, nn_dir, adam_dir)])
    pair = report["pairs"][0]
    assert pair["split"] == "test" and pair["split_recorded"] == {"nn": "test", "adam": "not_recorded"}
    assert pair["evaluation_manifest_sha256"] == "cd" * 32
    assert report["split"] == "test" and report["evaluation_manifest_sha256"] == "cd" * 32
    assert report["nn_profile"]["training_manifest_sha256"] == "not_recorded_in_export"


def test_split_and_eval_manifest_not_recorded_on_both_sides(tmp_path):
    nn_dir, adam_dir = make_pair(tmp_path)
    rewrite_manifest(nn_dir, lambda m: [m["metadata"].pop(k) for k in ("split", "manifest_sha256")])
    report = compare_runs([(None, nn_dir, adam_dir)])
    assert report["split"] == "not_recorded" and report["evaluation_manifest_sha256"] == "not_recorded"


@pytest.mark.parametrize("fault", ["pair_split", "pair_eval_sha", "mixed_split", "mixed_eval_sha",
                                   "bad_split", "bad_eval_sha"])
def test_split_and_evaluation_manifest_conflicts_fail(tmp_path, fault):
    nn_dir, adam_dir = make_pair(tmp_path)
    pairs = [(None, nn_dir, adam_dir)]
    if fault == "pair_split":
        rewrite_manifest(adam_dir, lambda m: m["metadata"].update({"split": "val"}))
    elif fault == "pair_eval_sha":
        rewrite_manifest(adam_dir, lambda m: m["metadata"].update({"manifest_sha256": "ef" * 32}))
    elif fault == "bad_split":
        rewrite_manifest(nn_dir, lambda m: m["metadata"].update({"split": ""}))
    elif fault == "bad_eval_sha":
        rewrite_manifest(nn_dir, lambda m: m["metadata"].update({"manifest_sha256": "short"}))
    else:
        nn2, adam2 = make_pair(tmp_path, "B64:003", seeds=(5, 6, 7, 8))
        change = {"split": "val"} if fault == "mixed_split" else {"manifest_sha256": "ef" * 32}
        rewrite_manifest(nn2, lambda m: m["metadata"].update(change))
        pairs.append((None, nn2, adam2))
    with pytest.raises(ValueError):
        compare_runs(pairs)


@pytest.mark.parametrize("field,value", [("steps", None), ("steps", 0), ("steps", True), ("steps", "16"),
                                         ("steps", 2.5), ("lr", None), ("lr", 0), ("lr", -0.05),
                                         ("lr", True), ("lr", "0.05")])
def test_paired_comparison_requires_recorded_adam_steps_and_lr(tmp_path, field, value):
    nn_dir, adam_dir = make_pair(tmp_path)
    rewrite_manifest(adam_dir, lambda m: m["metadata"].pop(field) if value is None
                     else m["metadata"].update({field: value}))
    with pytest.raises(ValueError, match=f"Adam {field}"):
        compare_runs([(None, nn_dir, adam_dir)])


def test_legacy_adam_api_still_accepts_missing_steps_and_lr(artifact_double):
    root, manifest, _ = artifact_double
    assert "steps" not in manifest["metadata"] and "lr" not in manifest["metadata"]
    assert summarize_adam_run(root)["seed"] == 17


@pytest.mark.parametrize("scope", [None, {}, {"checkpoint_load": "", "inference": "  "}])
def test_time_without_recorded_scope_is_not_measured(tmp_path, scope):
    meta = neural_metadata() | {"checkpoint_load_elapsed_s": 0.0001, "inference_elapsed_s": 0.0001}
    if scope is not None:
        meta["neural_timing_scope"] = scope
    timing = summarize_run(build_run(tmp_path / "nn", meta))["timing"]
    assert timing["checkpoint_load_s"] is None and timing["checkpoint_load_scope"] == "not_measured"
    assert timing["inference_s"] is None and timing["inference_scope"] == "not_measured"


def rehash(root, name):
    def change(manifest):
        key = next(k for k in manifest["files"] if k.replace("\\", "/") == name)
        manifest["files"][key] = digest(root / name)
        for i, source in enumerate(manifest["input_sources"], 1):
            if name == f"targets/target_{i:02d}.png":
                source["sha256"] = digest(root / name)
    rewrite_manifest(root, change)


@pytest.mark.parametrize("fault", ["target_256", "target_rgb", "target_16bit", "recon_rgb", "recon_256"])
def test_png_format_checked_with_strict_loader_despite_consistent_hashes(tmp_path, fault):
    run = build_run(tmp_path / "nn", neural_metadata())
    pixels = np.random.default_rng(3).integers(0, 256, size=(128, 128), dtype=np.uint8)
    name = "targets/target_01.png" if fault.startswith("target") else "reconstructed_02.png"
    if fault.endswith("256"):
        image = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")
    elif fault.endswith("rgb"):
        image = Image.fromarray(np.stack([pixels] * 3, axis=-1), mode="RGB")
    else:
        image = Image.fromarray(pixels.astype(np.uint16) * 257).convert("I;16")
    image.save(run / name, format="PNG")
    rehash(run, name)
    kind = "target" if fault.startswith("target") else "reconstruction"
    with pytest.raises(ValueError, match=f"{kind} PNG format"):
        summarize_run(run)
