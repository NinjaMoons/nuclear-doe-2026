"""READ-ONLY audit of an external quartet dataset archive (#14 stage B, GO 6708237953). RESEARCH ONLY.

usage:  python -I external_dataset_audit.py <archive.zip|.tar|.tar.gz> <our_corpus_dir> <report.json>

Never extracts to disk and never executes anything from the archive: members are read into memory one by one.
Checks
  - archive safety: absolute paths, '..', backslash/drive paths, symlinks/hardlinks/devices, duplicate names,
    per-member and total size limits; anything unsafe is reported and skipped.
  - PNG: decoded with Pillow, must be mode 'L', 8-bit, 128x128 (no conversion, no resize).
  - quartets: PNG grouped by parent directory; exactly 4 per group, order taken from the single integer 1..4 in the
    file name (e.g. *_01.png, z1.png); groups with missing/duplicate/ambiguous order are reported, not guessed.
  - forbidden content: model weights / pickles / arrays (.pt .pth .ckpt .pkl .pickle .npy .npz .safetensors .onnx .h5)
    and DOE / phase files (name contains 'doe' or 'phase'), executables/scripts.
  - SHA256 of file bytes and of decoded pixels; exact duplicates inside the archive.
  - overlap with OUR corpus (manifest_expanded.json in <our_corpus_dir>):
      TRAIN/VAL: exact pixel duplicates and near-duplicates (64-bit dHash, Hamming <= 6) against decoded images;
      TEST: METADATA ONLY - file-byte SHA256 from the manifest, no TEST image is opened, no inference.
  - contrast statistics per image (mean, std, p1/p99, fraction > 0.5, unique levels) with percentiles over the set.
  - optics metadata: text/JSON/YAML/CSV members are scanned for wavelength / pitch / distance values and for
    lens / objective / focal / 4f keywords (the case is free-space propagation only).
Outputs one JSON report; exit code 0 = no blocking finding, 2 = blocking findings (unsafe paths, forbidden files,
bad PNG format, incomplete quartets, TEST overlap). Nothing is merged with our frozen VAL/TEST.
"""
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

MAX_MEMBER = 20 * 2**20
MAX_TOTAL = 2 * 2**30
FORBIDDEN_EXT = {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".safetensors", ".onnx", ".h5",
                 ".exe", ".dll", ".so", ".dylib", ".sh", ".bat", ".ps1", ".py"}
TEXT_EXT = {".json", ".txt", ".md", ".yaml", ".yml", ".csv", ".ini", ".cfg"}
OPTICS_KW = re.compile(r"\b(lens|objective|focal|f-number|4f|telescope|microscope|linse|линз|объектив|фокус)\w*", re.I)
NUM_KW = {"wavelength": re.compile(r"(wavelength|lambda|λ|длина волны)[^0-9]{0,20}([0-9.,]+\s*(nm|um|µm|мкм|нм|m)?)", re.I),
          "pitch": re.compile(r"(pitch|pixel[_ ]?size|шаг)[^0-9]{0,20}([0-9.,]+\s*(um|µm|мкм|m)?)", re.I),
          "distance": re.compile(r"(distance|z[_ ]?\d?|расстояни)[^0-9]{0,12}([0-9.,]+\s*(mm|мм|m|cm)?)", re.I)}
OURS = {"wavelength": "532 nm", "pitch": "10 um", "distances": "24.1/25.3/26.5/27.7 mm"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_name(name: str):
    if "\\" in name or re.match(r"^[A-Za-z]:", name) or name.startswith("/"):
        return None
    p = PurePosixPath(name)
    if any(part in ("..", "") for part in p.parts) and str(p) not in (".",):
        return None
    return p


def members(path: Path, findings: dict):
    """Yield (name, bytes) of regular, safe members; record everything unsafe in findings."""
    total, seen = 0, set()
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for info in z.infolist():
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                findings["unsafe"].append({"name": info.filename, "why": "symlink"}); continue
            yield from _take(info.filename, info.file_size, lambda: z.read(info), findings, seen, total)
            total += info.file_size
    elif tarfile.is_tarfile(path):
        t = tarfile.open(path, "r:*")
        for m in t:
            if m.isdir():
                continue
            if not m.isfile():
                findings["unsafe"].append({"name": m.name, "why": "link/device/fifo in tar"}); continue
            yield from _take(m.name, m.size, lambda: t.extractfile(m).read(), findings, seen, total)
            total += m.size
    else:
        raise SystemExit("not a zip/tar archive")


def _take(name, size, read, findings, seen, total):
    p = safe_name(name)
    if p is None:
        findings["unsafe"].append({"name": name, "why": "absolute/traversal/backslash path"}); return
    if str(p) in seen:
        findings["unsafe"].append({"name": name, "why": "duplicate member name"}); return
    if size > MAX_MEMBER or total + size > MAX_TOTAL:
        findings["unsafe"].append({"name": name, "why": f"size limit ({size} B)"}); return
    seen.add(str(p))
    yield str(p), read()


def dhash(a: np.ndarray) -> int:
    im = Image.fromarray(a).resize((9, 8), Image.BILINEAR)
    x = np.asarray(im, dtype=np.int16)
    bits = (x[:, 1:] > x[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def order_index(name: str):
    nums = [int(n) for n in re.findall(r"\d+", PurePosixPath(name).stem) if 1 <= int(n) <= 4]
    return nums[-1] if len(set(nums)) >= 1 and len(nums) >= 1 else None


def our_index(corpus: Path):
    man = json.loads((corpus / "manifest_expanded.json").read_text())
    test_file_sha, trainval = {}, {"pixel": {}, "dhash": []}
    for s in man["samples"]:
        for i, (rel, h) in enumerate(zip(s["paths"], s["sha256"])):
            if s["split"] == "test":
                test_file_sha[h] = f"{s['key']}#z{i + 1}"            # metadata only, image never opened
                continue
            a = np.asarray(Image.open(corpus / rel))
            trainval["pixel"][sha(a.tobytes())] = f"{s['split']}:{s['key']}#z{i + 1}"
            trainval["dhash"].append((dhash(a), f"{s['split']}:{s['key']}#z{i + 1}"))
    return man, test_file_sha, trainval


def main():
    archive, corpus, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    findings = {"unsafe": [], "forbidden": [], "bad_png": [], "quartet_problems": [], "test_overlap": [],
                "trainval_exact": [], "trainval_near": [], "internal_duplicates": [], "optics_text": []}
    man, test_sha, ours = our_index(corpus)
    groups, pix_seen, stats, n_files = {}, {}, [], 0
    for name, data in members(archive, findings):
        n_files += 1
        p = PurePosixPath(name)
        ext = p.suffix.lower()
        if ext in FORBIDDEN_EXT or re.search(r"doe|phase", p.name, re.I):
            findings["forbidden"].append(name)
        fsha = sha(data)
        if fsha in test_sha:
            findings["test_overlap"].append({"name": name, "ours": test_sha[fsha], "by": "file sha256"})
        if ext in TEXT_EXT:
            txt = data[:2_000_000].decode("utf-8", "replace")
            hit = {k: [m.group(0)[:60] for m in r.finditer(txt)][:5] for k, r in NUM_KW.items()}
            kw = sorted({m.group(0).lower() for m in OPTICS_KW.finditer(txt)})
            if any(hit.values()) or kw:
                findings["optics_text"].append({"name": name, "values": hit, "lens_keywords": kw})
        if ext != ".png":
            continue
        try:
            im = Image.open(io.BytesIO(data)); im.load()
        except Exception as e:  # noqa: BLE001 - report any decoder failure
            findings["bad_png"].append({"name": name, "why": f"decode: {type(e).__name__}"}); continue
        if im.mode != "L" or im.size != (128, 128):
            findings["bad_png"].append({"name": name, "why": f"mode={im.mode} size={im.size}"}); continue
        a = np.asarray(im)
        psha = sha(a.tobytes())
        if psha in pix_seen:
            findings["internal_duplicates"].append([pix_seen[psha], name])
        pix_seen.setdefault(psha, name)
        if psha in ours["pixel"]:
            findings["trainval_exact"].append({"name": name, "ours": ours["pixel"][psha]})
        else:
            h = dhash(a)
            near = [(bin(h ^ oh).count("1"), k) for oh, k in ours["dhash"]]
            best = min(near) if near else None
            if best and best[0] <= 6:
                findings["trainval_near"].append({"name": name, "ours": best[1], "hamming": best[0]})
        x = a.astype(np.float64) / 255
        stats.append({"mean": x.mean(), "std": x.std(), "p1": np.percentile(x, 1), "p99": np.percentile(x, 99),
                      "frac_gt_0.5": (x > 0.5).mean(), "levels": len(np.unique(a))})
        groups.setdefault(str(p.parent), []).append((order_index(name), name, fsha, psha))
    quartets = {}
    for g, items in sorted(groups.items()):
        idx = sorted(i for i, *_ in items if i is not None)
        if len(items) != 4 or idx != [1, 2, 3, 4]:
            findings["quartet_problems"].append({"dir": g, "files": sorted(n for _, n, *_ in items), "order": idx}); continue
        quartets[g] = [{"z": i, "name": n, "sha256": f, "pixel_sha256": ps} for i, n, f, ps in sorted(items)]
    pct = lambda k: {q: float(np.percentile([s[k] for s in stats], q)) for q in (5, 50, 95)} if stats else None
    blocking = any(findings[k] for k in ("unsafe", "forbidden", "bad_png", "quartet_problems", "test_overlap"))
    report = {"archive": str(archive), "archive_sha256": sha(archive.read_bytes()), "members_read": n_files,
              "our_manifest_sha256": sha((corpus / "manifest_expanded.json").read_bytes()),
              "our_case": OURS, "png_images": len(stats), "complete_quartets": len(quartets),
              "contrast_percentiles": {k: pct(k) for k in ("mean", "std", "p1", "p99", "frac_gt_0.5", "levels")},
              "findings": findings, "blocking": blocking, "quartets": quartets,
              "policy": "read-only; TEST compared by manifest file sha256 only; nothing merged with frozen VAL/TEST; "
                        "no model run on external data before provenance/exchange rules are agreed"}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"blocking": blocking, "quartets": len(quartets), "png": len(stats),
                      **{k: len(v) for k, v in findings.items()}}))
    return 2 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
