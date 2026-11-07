"""Synthetic-fixture tests for wc_audit.py and adapter.py (no real WhiteCollars data, no TEST images).

run (cwd = repo root):  python -m pytest research/whitecollars_dataset/tests -q
"""
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root for doe_experiments
import wc_audit  # noqa: E402
from adapter import WhiteCollarsTrain  # noqa: E402
from doe_experiments.data import load_png_quartet  # noqa: E402


def png_bytes(a: np.ndarray, mode="L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def pattern(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    a = np.zeros((128, 128), np.uint8)
    for _ in range(4):
        y, x = r.integers(0, 100, 2)
        h, w = r.integers(8, 28, 2)
        a[y:y + h, x:x + w] = r.integers(60, 256)
    a[r.integers(0, 128), :] = 255
    return a


def tar_gz(path: Path, members: dict, links=()):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name); info.type = tarfile.SYMTYPE; info.linkname = target
            t.addfile(info)
    return path


@pytest.fixture()
def corpus(tmp_path):
    """Tiny 'our corpus': 2 train (A,B) + 1 val written; 1 test sample listed by sha only (no file on disk)."""
    root = tmp_path / "ours"
    samples = []
    for key, split, kind, seed in (("A64:000", "train", "A", 1000), ("B64:000", "train", "B", 2000),
                                    ("B64:009", "val", "B", 3000)):
        paths, shas = [], []
        for z in range(4):
            rel = f"{key.replace(':', '_')}/t{z + 1}.png"
            data = png_bytes(pattern(seed + z))
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
            paths.append(rel); shas.append(hashlib.sha256(data).hexdigest())
        samples.append({"key": key, "split": split, "kind": kind, "paths": paths, "sha256": shas})
    test_bytes = png_bytes(pattern(9000))
    samples.append({"key": "A64:063", "split": "test", "kind": "A", "paths": ["missing/t1.png"] * 4,
                    "sha256": [hashlib.sha256(test_bytes).hexdigest()] + ["0" * 64] * 3})
    (root / "manifest_expanded.json").write_text(json.dumps({"samples": samples, "content_sha256": "x"}))
    return root, test_bytes


def good_members(n_train=70, n_val=3):
    m = {}
    for i in range(n_train):
        for z in range(4):
            m[f"train/{i:05d}/z{z + 1}.png"] = png_bytes(pattern(10 * i + z + 7))
    for i in range(n_val):
        for z in range(4):
            m[f"val/{i:05d}/z{z + 1}.png"] = png_bytes(pattern(100000 + 10 * i + z))
    m["README.txt"] = b"WhiteCollars targets, amplitude PNG/255, wavelength 532 nm"
    return m


def run(tmp_path, corpus_dir, members, links=(), **kw):
    arc = tar_gz(tmp_path / "wc.tar.gz", members, links)
    out = tmp_path / "out"
    code = wc_audit.run(arc, corpus_dir, out, **kw)
    return code, json.loads((out / "WC_AUDIT.json").read_text(encoding="utf-8")), \
        json.loads((out / "WC_MAPPING.json").read_text(encoding="utf-8")), arc, out


def test_vendored_auditor_is_unchanged():
    assert hashlib.sha256((HERE / "external_dataset_audit.py").read_bytes()).hexdigest() == wc_audit.AUDITOR_SHA256


@pytest.mark.parametrize("name,expected", [
    ("train/00001/z3.png", ("train/00001", 3)), ("train/00001/3.png", ("train/00001", 3)),
    ("train/q_0001_z2.png", ("train/q_0001", 2)), ("train/q0001_plane4.png", ("train/q0001", 4)),
    ("a/target_01.png", ("a", 1)), ("train/00123.png", (None, None)), ("train/z5.png", (None, None)),
    ("train/sample0012.png", (None, None))])
def test_z_parsing_is_explicit(name, expected):
    assert wc_audit.z_of(name) == expected


def test_good_archive_passes_but_rights_gate_closed(tmp_path, corpus):
    code, audit, mapping, _, out = run(tmp_path, corpus[0], good_members())
    assert code == 0 and audit["technical_pass"] is True
    assert audit["usage_rights"] == "NOT_CONFIRMED" and audit["training_allowed"] is False
    assert audit["split_counts"] == {"train": 70, "val": 3}
    assert len(mapping["holdout_keys"]) == 64 and len(mapping["train_keys"]) == 6
    assert not set(mapping["holdout_keys"]) & set(mapping["train_keys"])
    assert all(mapping["excluded"][k] == "non-train split" for k in mapping["excluded"] if k.startswith("val/"))
    assert (out / "WC_MAPPING.md").exists()
    assert audit["distribution"]["wc_train"]["mean"]["n"] == 280
    assert audit["distribution"]["ours_train_A"]["mean"]["n"] == 4


def test_rights_evidence_opens_training_gate(tmp_path, corpus):
    ev = tmp_path / "rights.txt"; ev.write_text("Author X permits training use in hackathon 2026", encoding="utf-8")
    code, audit, *_ = run(tmp_path, corpus[0], good_members(), rights=ev)
    assert code == 0 and audit["training_allowed"] is True and audit["rights_evidence_sha256"]


def test_expected_sha_mismatch_blocks(tmp_path, corpus):
    code, audit, *_ = run(tmp_path, corpus[0], good_members(), expect_sha="0" * 64)
    assert code == 2 and "sha_mismatch" in audit["blocking"] and audit["training_allowed"] is False


def test_unsafe_and_forbidden_and_bad_png_block(tmp_path, corpus):
    m = good_members(5, 0)
    m["../evil.png"] = png_bytes(pattern(1))
    m["train/00000/model.pt"] = b"x"
    m["train/00001/DOE.png"] = png_bytes(pattern(2))
    m["train/00002/z1.png"] = png_bytes(np.zeros((128, 128, 3), np.uint8), mode="RGB")
    m["train/00003/z2.png"] = png_bytes(np.zeros((64, 64), np.uint8))
    code, audit, *_ = run(tmp_path, corpus[0], m, links=[("train/link.png", "/etc/passwd")])
    assert code == 2
    for k in ("unsafe", "forbidden", "bad_png", "quartet_problems"):
        assert k in audit["blocking"], k
    assert len(audit["findings"]["unsafe"]) == 2


def test_16bit_png_rejected(tmp_path, corpus):
    m = good_members(2, 0)
    buf = io.BytesIO(); Image.fromarray(np.zeros((128, 128), np.uint16) + 300).save(buf, format="PNG")
    m["train/00000/z1.png"] = buf.getvalue()
    code, audit, *_ = run(tmp_path, corpus[0], m)
    assert code == 2 and audit["findings"]["bad_png"]


def test_duplicates_permutations_and_overlap(tmp_path, corpus):
    root, _ = corpus
    m = good_members(6, 0)
    for z in range(4):  # exact copy of quartet 0, and z-permuted copy of quartet 1
        m[f"train/90000/z{z + 1}.png"] = m[f"train/00000/z{z + 1}.png"]
        m[f"train/90001/z{z + 1}.png"] = m[f"train/00001/z{4 - z}.png"]
    m["train/00002/z2.png"] = m["train/00002/z1.png"]  # identical planes inside a quartet
    ours_val = json.loads((root / "manifest_expanded.json").read_text())["samples"][2]
    for z in range(4):  # our VAL quartet with z permuted -> blocking
        m[f"train/90002/z{z + 1}.png"] = (root / ours_val["paths"][3 - z]).read_bytes()
    near = np.asarray(Image.open(root / ours_val["paths"][0])).astype(int)
    near[0, 0] = (near[0, 0] + 3) % 256
    m["train/00003/z1.png"] = png_bytes(near.astype(np.uint8))  # near-duplicate of our VAL plane
    code, audit, mapping, *_ = run(tmp_path, root, m)
    f = audit["findings"]
    assert f["internal_duplicate_quartets"] == [["train/00000", "train/90000"]]
    assert f["internal_permuted_quartets"] == [["train/00001", "train/90001"]]
    assert f["intra_quartet_identical_planes"][0]["key"] == "train/00002"
    assert f["ours_quartet_permuted"][0]["ours"] == "val:B64:009"
    assert any(n["plane"] == "train/00003#z1" and n["verdict"] == "suspect_needs_review" for n in f["trainval_near_planes"])
    assert code == 2 and "ours_quartet_permuted" in audit["blocking"]
    for k in ("train/00000", "train/90000", "train/00002", "train/00003"):
        assert k not in mapping["train_keys"] and k not in mapping["holdout_keys"]


def test_test_overlap_by_file_sha_only(tmp_path, corpus):
    root, test_bytes = corpus
    m = good_members(2, 0)
    m["train/00000/z1.png"] = test_bytes
    code, audit, *_ = run(tmp_path, root, m)
    assert code == 2 and audit["findings"]["test_overlap"][0]["ours"] == "A64:063#z1"


def test_manifest_member_defines_splits(tmp_path, corpus):
    m = {}
    rows = []
    for i in range(3):
        paths = [f"img/{i}_{z}.png" for z in range(1, 5)]
        for z, p in enumerate(paths):
            m[p] = png_bytes(pattern(500 + 10 * i + z))
        rows.append({"id": f"s{i}", "split": "train" if i < 2 else "test", "paths": paths})
    m["manifest.json"] = json.dumps({"schema": "doe-manifest/1", "samples": rows}).encode()
    code, audit, mapping, *_ = run(tmp_path, corpus[0], m)
    assert audit["manifest_member"] == "manifest.json"
    assert audit["split_counts"] == {"train": 2, "test": 1}
    assert mapping["quartets"]["s2"]["split"] == "test" and "s2" in mapping["excluded"]


def test_adapter_exact_contract_and_isolation(tmp_path, corpus):
    members = good_members()
    code, audit, mapping, arc, out = run(tmp_path, corpus[0], members)
    ds = WhiteCollarsTrain(arc, out / "WC_MAPPING.json")
    assert len(ds) == 6 and set(ds.keys) == set(mapping["train_keys"])
    t = ds.targets([0, 1])
    assert t.shape == (2, 4, 1, 128, 128) and t.dtype == torch.float32
    # identical to the production loader on the same bytes (no sqrt/gamma/max-norm, z order kept)
    k = ds.keys[1]
    files = []
    for z, p in enumerate(mapping["quartets"][k]["paths"]):
        f = tmp_path / f"chk{z}.png"; f.write_bytes(members[p]); files.append(f)
    ref = load_png_quartet(files)
    assert torch.equal(t[1:2], ref)
    raw = np.asarray(Image.open(io.BytesIO(members[mapping["quartets"][k]["paths"][2]])))
    assert torch.equal(t[1, 2, 0], torch.from_numpy(raw.astype(np.float32) / 255))
    hold = WhiteCollarsTrain(arc, out / "WC_MAPPING.json", role="holdout")
    assert len(hold) == 64 and not set(hold.keys) & set(ds.keys)
    with pytest.raises(ValueError):
        WhiteCollarsTrain(arc, out / "WC_MAPPING.json", role="val")


def test_adapter_rejects_tampering(tmp_path, corpus):
    members = good_members()
    code, audit, mapping, arc, out = run(tmp_path, corpus[0], members)
    with pytest.raises(ValueError, match="archive sha256"):
        WhiteCollarsTrain(arc, out / "WC_MAPPING.json", expect_archive_sha256="1" * 64)
    mp = json.loads((out / "WC_MAPPING.json").read_text())
    k = mp["train_keys"][0]
    mp["quartets"][k]["file_sha256"][0] = "0" * 64
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps(mp))
    with pytest.raises(ValueError, match="file sha256"):
        WhiteCollarsTrain(arc, bad)
    mp = json.loads((out / "WC_MAPPING.json").read_text())
    mp["train_keys"].append(mp["holdout_keys"][0])
    leak = tmp_path / "leak.json"; leak.write_text(json.dumps(mp))
    with pytest.raises(ValueError, match="holdout"):
        WhiteCollarsTrain(arc, leak)


def test_existing_output_dir_refused(tmp_path, corpus):
    arc = tar_gz(tmp_path / "a.tar.gz", good_members(1, 0))
    (tmp_path / "o").mkdir()
    with pytest.raises(SystemExit):
        wc_audit.run(arc, corpus[0], tmp_path / "o")
