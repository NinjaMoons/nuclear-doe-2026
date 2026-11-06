"""#4B pilot corpus: structure, splits/leakage, duplicates, hashes, determinism, adapter.

The A check re-runs the SAME shared forward from the saved DOE.png: it verifies data
provenance (PNG <- DOE.png), not physical correctness (tests/integration gates do that).
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from doe_experiments import corpus
from doe_experiments.phase_codec import load_doe_png


@pytest.fixture(scope="module")
def pilot(tmp_path_factory) -> Path:
    return corpus.generate_corpus(tmp_path_factory.mktemp("pilot") / "c", seed=0)


@pytest.fixture(scope="module")
def manifest(pilot) -> dict:
    return json.loads(pilot.read_text(encoding="utf-8"))


def _copy(pilot: Path, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(pilot.parent, target)
    return target / "manifest.json"


def _rewrite(manifest_path: Path, edit) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    edit(data)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_counts_splits_and_namespaced_keys(manifest):
    assert manifest["schema"] == "doe_training_manifest_v1"
    assert manifest["contract_version"] == "1.1-amplitude"
    assert manifest["status"] == "DATA_CANDIDATE"
    samples = manifest["samples"]
    assert len(samples) == 128
    for cid, kind in (("A64", "A"), ("B64", "B")):
        part = [s for s in samples if s["corpus_id"] == cid]
        assert len(part) == 64 and {s["kind"] for s in part} == {kind}
        assert [sum(s["split"] == name for s in part) for name in ("train", "val", "test")] == [48, 8, 8]
    keys = [s["key"] for s in samples]
    assert len(set(keys)) == 128
    assert all(s["key"] == f'{s["corpus_id"]}:{s["local_sample_id"]}' for s in samples)
    assert sorted(sum(manifest["splits"].values(), [])) == sorted(keys)


def test_origin_ids_never_cross_splits(manifest):
    seen = {}
    for sample in manifest["samples"]:
        assert seen.setdefault(sample["origin_id"], sample["split"]) == sample["split"]
        assert sample["origin_id"].startswith(sample["corpus_id"] + ":")


def test_files_are_strict_png_quartets_with_matching_hashes(pilot, manifest):
    root = pilot.parent
    for sample in manifest["samples"][::9]:
        assert len(sample["paths"]) == 4
        for rel, digest in zip(sample["paths"], sample["sha256"]):
            with Image.open(root / rel) as image:
                assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))
            assert corpus.file_sha256(root / rel) == digest


def test_validate_passes_and_reports_manifest_hash(pilot):
    report = corpus.validate_corpus(pilot)
    assert report["samples"] == 128
    assert report["counts"]["A64"] == {"train": 48, "val": 8, "test": 8}
    assert report["manifest_sha256"] == corpus.file_sha256(pilot)
    assert report["content_sha256"] == json.loads(pilot.read_text())["content_sha256"]


def test_a_label_is_decoded_doe_and_origin_is_provenance(pilot, manifest):
    root = pilot.parent
    for sample in [s for s in manifest["samples"] if s["kind"] == "A"][:8]:
        assert sample["phase_label"]["path"] == sample["doe_path"]
        assert sample["origin_phase"]["role"] == "provenance_only"
        phi_q = load_doe_png(root / sample["doe_path"])[0, 0].numpy()
        origin = np.load(root / sample["origin_phase"]["path"], allow_pickle=False)
        error = np.abs(np.angle(np.exp(1j * (phi_q - origin))))
        assert error.max() <= math.pi / 255 + 1e-12
        assert sample["origin_phase"]["max_abs_wrapped_decode_error_rad"] == pytest.approx(error.max())
        assert len(sample["raw_amplitude_max"]) == 4 and all(m > 1e-12 for m in sample["raw_amplitude_max"])
        assert sample["exposure"] == pytest.approx([1 / m for m in sample["raw_amplitude_max"]])


def test_b_has_no_phase_label_or_doe_and_no_constant_planes(pilot, manifest):
    root = pilot.parent
    for sample in [s for s in manifest["samples"] if s["kind"] == "B"]:
        assert sample["phase_label"] is None and sample["doe_path"] is None
        assert not (root / sample["corpus_id"] / sample["local_sample_id"] / "DOE.png").exists()
        assert set(sample["procedural"]["families"]) <= set(corpus.B_FAMILIES)
    families = {f for s in manifest["samples"] if s["kind"] == "B" for f in s["procedural"]["families"]}
    assert families == set(corpus.B_FAMILIES)


def test_no_duplicate_quartets_or_images(manifest):
    quartets = [s["quartet_sha256"] for s in manifest["samples"]]
    assert len(set(quartets)) == len(quartets)
    images = [h for s in manifest["samples"] for h in s["sha256"]]
    # A planes of one sample may coincide only if physics made them identical; across samples never.
    owners = {}
    for sample in manifest["samples"]:
        for digest in set(sample["sha256"]):
            assert owners.setdefault(digest, sample["key"]) == sample["key"]
    assert len(images) == 512


def test_same_seed_is_byte_identical_and_other_seed_differs(pilot, manifest, tmp_path):
    again = json.loads(corpus.generate_corpus(tmp_path / "again", seed=0).read_text(encoding="utf-8"))
    strip = lambda m: [(s["key"], s["split"], s["sha256"], s.get("doe_sha256")) for s in m["samples"]]
    assert strip(again) == strip(manifest)
    assert again["content_sha256"] == manifest["content_sha256"]
    other = json.loads(corpus.generate_corpus(tmp_path / "other", seed=1).read_text(encoding="utf-8"))
    assert not set(s["quartet_sha256"] for s in other["samples"]) & set(s["quartet_sha256"] for s in manifest["samples"])
    assert [s["split"] for s in other["samples"]] != [s["split"] for s in manifest["samples"]]


def test_split_is_assigned_before_rendering():
    # Split labels depend only on (seed, corpus, count, sizes), not on rendered content.
    labels = corpus._assign_splits(0, 0, 64, (48, 8, 8))
    assert labels == corpus._assign_splits(0, 0, 64, (48, 8, 8))
    assert [labels.count(n) for n in ("train", "val", "test")] == [48, 8, 8]
    with pytest.raises(ValueError):
        corpus._assign_splits(0, 0, 64, (48, 8, 9))


def test_no_overwrite_and_bad_arguments(pilot, tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        corpus.generate_corpus(pilot.parent, seed=0)
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            corpus.generate_corpus(tmp_path / f"bad{bad!r}", seed=bad)
    with pytest.raises(ValueError):
        corpus.generate_corpus(tmp_path / "counts", counts=(10, 64))


def test_tampered_png_is_rejected(pilot, tmp_path):
    copy = _copy(pilot, tmp_path)
    sample = json.loads(copy.read_text(encoding="utf-8"))["samples"][70]
    path = copy.parent / sample["paths"][2]
    values = np.array(Image.open(path))
    values[0, 0] ^= 1
    path.unlink()
    Image.fromarray(values).save(path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        corpus.validate_corpus(copy)


def test_a_png_not_from_its_doe_is_rejected(pilot, tmp_path):
    copy = _copy(pilot, tmp_path)
    data = json.loads(copy.read_text(encoding="utf-8"))
    a0, a1 = [s for s in data["samples"] if s["kind"] == "A"][:2]

    def swap(m):
        first, second = [s for s in m["samples"] if s["kind"] == "A"][:2]
        first["doe_path"], first["doe_sha256"] = second["doe_path"], second["doe_sha256"]
        first["phase_label"]["path"] = second["doe_path"]
    assert a0["doe_sha256"] != a1["doe_sha256"]
    _rewrite(copy, swap)
    with pytest.raises(ValueError, match="do not match the forward"):
        corpus.validate_corpus(copy)


def test_leakage_duplicates_labels_and_path_escape_are_rejected(pilot, tmp_path):
    def same_origin_other_split(m):
        a, b = [s for s in m["samples"] if s["corpus_id"] == "B64"][:2]
        b["origin_id"] = a["origin_id"]
        b["split"] = "test" if a["split"] != "test" else "val"
        m["splits"] = {n: [s["key"] for s in m["samples"] if s["split"] == n] for n in ("train", "val", "test")}

    def duplicate_quartet(m):
        a, b = [s for s in m["samples"] if s["corpus_id"] == "B64"][:2]
        for k in ("paths", "sha256", "quartet_sha256"):
            b[k] = list(a[k]) if isinstance(a[k], list) else a[k]

    def fake_b_label(m):
        [s for s in m["samples"] if s["kind"] == "B"][0]["phase_label"] = {"source": "invented"}

    def escape(m):
        m["samples"][0]["paths"][0] = "../outside.png"

    def stale_split_list(m):
        m["splits"]["train"] = m["splits"]["train"][1:]

    def stale_content_digest(m):
        m["content_sha256"] = "0" * 64

    cases = [(same_origin_other_split, "leakage"), (duplicate_quartet, "duplicate quartet"),
             (fake_b_label, "must not carry"), (escape, "escapes"), (stale_split_list, "split lists"),
             (stale_content_digest, "content_sha256")]
    for index, (edit, message) in enumerate(cases):
        copy = _copy(pilot, tmp_path / str(index))
        _rewrite(copy, edit)
        with pytest.raises(ValueError, match=message):
            corpus.validate_corpus(copy, check_a_regeneration=False)


def test_runtime_adapter_shapes_ids_and_values(pilot, manifest):
    splits = corpus.load_training_splits(pilot)
    assert {n: s.targets.shape[0] for n, s in splits.items()} == {"train": 96, "val": 16, "test": 16}
    for name, part in splits.items():
        assert part.targets.dtype == torch.float32 and part.targets.device.type == "cpu"
        assert tuple(part.targets.shape[1:]) == (4, 1, 128, 128)
        assert float(part.targets.min()) >= 0 and float(part.targets.max()) <= 1
        assert part.keys == manifest["splits"][name]
        assert len(part.keys) == len(part.origin_ids) == len(part.kinds) == part.targets.shape[0]
    assert not set(splits["train"].origin_ids) & set(splits["val"].origin_ids + splits["test"].origin_ids)
    only_a = corpus.load_split(pilot, "val", kinds=("A",))
    assert only_a.kinds == ["A"] * 8
    first = next(s for s in manifest["samples"] if s["key"] == only_a.keys[0])
    raw = np.stack([np.array(Image.open(pilot.parent / p)) for p in first["paths"]])
    assert torch.equal(only_a.targets[0, :, 0], torch.from_numpy(raw).float() / 255)
    with pytest.raises(ValueError):
        corpus.load_split(pilot, "holdout")
    with pytest.raises(ValueError):
        corpus.load_split(pilot, "train", kinds="A")


def test_cli_validate_and_error_exit(pilot, tmp_path, capsys):
    assert corpus.main(["validate", "--manifest", str(pilot), "--skip-a-regeneration"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 128
    assert corpus.main(["validate", "--manifest", str(tmp_path / "missing.json")]) == 2


# --------------------------------------------------------------------------- train-only expansion

@pytest.fixture(scope="module")
def expanded(pilot, tmp_path_factory) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("expanded") / "c"
    shutil.copytree(pilot.parent, root)
    pilot_copy = root / "manifest.json"
    before = corpus.file_sha256(pilot_copy)
    return corpus.expand_corpus(pilot_copy, seed=0), pilot_copy, before


def _expanded_manifest(expanded) -> dict:
    return json.loads(expanded[0].read_text(encoding="utf-8"))


def test_expansion_counts_and_train_only(expanded):
    data = _expanded_manifest(expanded)
    assert data["synthetic"] is True and data["expansion"]["train_only"] is True
    counts = {(s["kind"], s["split"]): 0 for s in data["samples"]}
    for s in data["samples"]:
        counts[(s["kind"], s["split"])] += 1
    assert counts == {("A", "train"): 256, ("A", "val"): 8, ("A", "test"): 8,
                      ("B", "train"): 256, ("B", "val"): 8, ("B", "test"): 8}
    added = [s for s in data["samples"] if s["corpus_id"] in ("A208", "B208")]
    assert len(added) == 416 and {s["split"] for s in added} == {"train"}
    assert all(len(s["pixel_sha256"]) == 4 for s in added)
    families = [s["origin_phase"]["family"] for s in added if s["kind"] == "A"]
    assert {f: families.count(f) for f in set(families)} == dict(corpus.A_EXT_PLAN)
    relations = [s["procedural"]["relation"] for s in added if s["kind"] == "B"]
    assert {r: relations.count(r) for r in set(relations)} == {"independent": 70, "similar": 69, "partial": 69}


def test_expansion_preserves_pilot_and_holdouts(expanded, manifest):
    expanded_path, pilot_copy, before = expanded
    assert corpus.file_sha256(pilot_copy) == before  # pilot manifest not rewritten
    data = _expanded_manifest(expanded)
    by_key = {s["key"]: s for s in data["samples"]}
    assert all(by_key[s["key"]] == s for s in manifest["samples"])
    assert data["splits"]["val"] == manifest["splits"]["val"]
    assert data["splits"]["test"] == manifest["splits"]["test"]
    assert data["expansion"]["pilot_content_sha256"] == manifest["content_sha256"]
    report = corpus.validate_expanded(expanded_path, pilot_copy)
    assert report["samples"] == 544 and report["added"] == 416


def test_expansion_adapter_gives_512_16_16(expanded):
    splits = corpus.load_training_splits(expanded[0])
    assert {n: s.targets.shape[0] for n, s in splits.items()} == {"train": 512, "val": 16, "test": 16}


def test_expansion_is_deterministic_and_independent_of_count(pilot, tmp_path):
    made = {}
    for name, counts in (("a", (6, 6)), ("b", (6, 6)), ("c", (8, 8))):
        root = tmp_path / name
        shutil.copytree(pilot.parent, root)
        made[name] = {s["key"]: s for s in json.loads(corpus.expand_corpus(
            root / "manifest.json", seed=0, counts=counts).read_text())["samples"]}
    strip = lambda s: (s["split"], s["sha256"], s.get("doe_sha256"), s.get("pixel_sha256"))
    assert {k: strip(v) for k, v in made["a"].items()} == {k: strip(v) for k, v in made["b"].items()}
    for index in range(6):  # B sample i does not depend on how many samples are generated
        key = f"B208:{index:03d}"
        assert strip(made["a"][key]) == strip(made["c"][key])


def test_expansion_refuses_overwrite(expanded):
    with pytest.raises(ValueError, match="overwrite"):
        corpus.expand_corpus(expanded[1], seed=0)


def test_copied_quartet_with_new_ids_in_test_is_detected(expanded, tmp_path):
    root = tmp_path / "leak"
    shutil.copytree(expanded[0].parent, root)
    path = root / corpus.EXPANDED_MANIFEST
    data = json.loads(path.read_text())
    source = next(s for s in data["samples"] if s["key"] == "B208:010")
    new_paths, new_hashes = [], []
    for index, rel in enumerate(source["paths"]):
        target = root / "smuggled" / f"plane{index}.png"
        target.parent.mkdir(exist_ok=True)
        image = Image.open(root / rel)
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "different metadata, same pixels")
        image.save(target, pnginfo=info)
        new_paths.append(f"smuggled/plane{index}.png")
        new_hashes.append(corpus.file_sha256(target))
    assert new_hashes != source["sha256"]  # file hashes differ, decoded pixels are identical
    clone = {**source, "corpus_id": "SMUG", "local_sample_id": "000", "key": "SMUG:000",
             "origin_id": "SMUG:origin-000", "split": "test", "paths": new_paths, "sha256": new_hashes,
             "quartet_sha256": __import__("hashlib").sha256("".join(new_hashes).encode()).hexdigest()}
    data["samples"].append(clone)
    data["splits"]["test"].append("SMUG:000")
    data["content_sha256"] = corpus.content_sha256(data["samples"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate decoded quartet"):
        corpus.validate_corpus(path, check_a_regeneration=False)


def test_constant_b_plane_is_rejected(pilot, tmp_path):
    copy = _copy(pilot, tmp_path)
    data = json.loads(copy.read_text())
    sample = next(s for s in data["samples"] if s["kind"] == "B")
    target = copy.parent / sample["paths"][1]
    target.unlink()
    Image.fromarray(np.full((128, 128), 77, np.uint8)).save(target)
    sample["sha256"][1] = corpus.file_sha256(target)
    sample["quartet_sha256"] = __import__("hashlib").sha256("".join(sample["sha256"]).encode()).hexdigest()
    data["content_sha256"] = corpus.content_sha256(data["samples"])
    copy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="constant"):
        corpus.validate_corpus(copy, check_a_regeneration=False)


def test_diversity_report_groups(expanded):
    report = corpus.diversity_report(expanded[0])
    assert sum(g["n"] for g in report.values()) == 544
    assert {"A/smooth_correlated", "A/low_order", "A/structured_multiscale", "A/iid_random",
            "B/independent", "B/similar", "B/partial"} <= set(report)
    assert report["B/independent"]["interplane_pearson_mean"]["median"] < \
        report["B/similar"]["interplane_pearson_mean"]["median"]
