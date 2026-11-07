"""READ-ONLY quartet-level audit + split mapping for the WhiteCollars archive (#41). RESEARCH ONLY.

usage:
  python -I wc_audit.py ARCHIVE OUR_CORPUS_DIR OUT_DIR [--expect-sha256 HEX] [--rights-evidence FILE]

Builds on the unchanged #14 auditor `external_dataset_audit.py` (sha256 863171ca..., @ggmaster913): archive safety
(member iteration in memory, traversal/symlink/size rejection), dHash, forbidden-file rules and optics keyword scan
are reused from it. Added here:
  * grouping that also supports flat layouts (`<prefix>_z1.png` ...); z order only from an explicit trailing 1..4
    token; ambiguous groups are reported, never guessed;
  * split assignment: (1) a JSON manifest member listing samples with 4 paths and a split, else (2) a top-level
    directory named train/val/valid/validation/test; anything else is `unassigned` (never used for training);
  * quartet checks: identical planes inside a quartet, exact duplicate quartets, z-permuted duplicate quartets,
    overlap with OUR quartets (TRAIN/VAL ordered or permuted by decoded pixels; TEST by manifest file-sha only);
  * near-duplicate confirmation: every dHash hit (Hamming <= 6) is re-checked with pixel MAE and Pearson and flagged
    `near_uniform` when either image is almost flat (dHash is unreliable there); nothing is deleted automatically;
  * descriptive distribution statistics vs our TRAIN A and B (no A/B labels are assigned to external data);
  * a deterministic 64-quartet internal holdout from external TRAIN (rule fixed before looking at quality).
Outputs OUT_DIR/WC_AUDIT.json, OUT_DIR/WC_MAPPING.json, OUT_DIR/WC_MAPPING.md. OUT_DIR must not exist.
Exit 0 = technical PASS, 2 = blocking finding. Usage rights are a separate gate: without --rights-evidence the
report says `usage_rights: NOT_CONFIRMED` and `training_allowed: false` even when the technical audit passes.
Our TEST images are never opened: the caller's corpus copy may even lack them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_dataset_audit as base  # noqa: E402  (unchanged #14 auditor)

AUDITOR_SHA256 = "863171ca879bdb7c3eb4e10bbd1a88fedbb1d0a1561948d7018d495c5fb9b64f"
HOLDOUT_SALT = "nuclear-doe-wc-holdout-v1"
HOLDOUT_SIZE = 64
NEAR_HAMMING = 6
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
Z_ONLY = re.compile(r"^(?:z|plane|target|img|image|t)?[_\-]?0*(?P<z>[1-4])$", re.I)
Z_SEP = re.compile(r"^(?P<prefix>.+?)[_\-.](?:z|plane|target|img|image|t)?[_\-]?0*(?P<z>[1-4])$", re.I)
Z_TOKEN_ONLY = re.compile(r"^(?P<prefix>.+?)(?:z|plane|target)0*(?P<z>[1-4])$", re.I)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def z_of(name: str):
    """(group_key, z) from a member name; z only from an explicit trailing 1..4 token, else (None, None).

    Accepted stems: '1'..'4' or 'z3'/'plane_3'/'target03' (zero padding allowed, group = directory); '<prefix><sep>[z|plane|target|img|image|t]<n>'
    with sep in _-. ; '<prefix>(z|plane|target)<n>' without separator. Anything else is ambiguous.
    """
    p = PurePosixPath(name)
    stem = p.stem
    m = Z_ONLY.match(stem)
    if m:
        return str(p.parent), int(m.group("z"))
    m = Z_SEP.match(stem) or Z_TOKEN_ONLY.match(stem)
    if m:
        return str(p.parent / m.group("prefix")), int(m.group("z"))
    return None, None


def split_from_path(name: str):
    first = PurePosixPath(name).parts[0].lower() if PurePosixPath(name).parts else ""
    return SPLIT_NAMES.get(first)


def find_manifest_samples(texts: dict):
    """Return (member_name, [{'key','split','paths'}]) from the first JSON member that lists 4-path samples."""
    for name, txt in sorted(texts.items()):
        if not name.lower().endswith(".json"):
            continue
        try:
            obj = json.loads(txt)
        except ValueError:
            continue
        lists = [obj] if isinstance(obj, list) else [v for v in obj.values() if isinstance(v, list)] if isinstance(obj, dict) else []
        for lst in lists:
            rows = []
            for row in lst:
                if not isinstance(row, dict):
                    break
                paths = next((row[k] for k in ("paths", "targets", "files", "images") if isinstance(row.get(k), list)), None)
                if not paths or len(paths) != 4 or not all(isinstance(x, str) for x in paths):
                    break
                split = row.get("split")
                rows.append({"key": str(row.get("key", row.get("id", ""))), "split": SPLIT_NAMES.get(str(split).lower()) if split else None,
                             "paths": [str(PurePosixPath(base_dir_join(name, x))) for x in paths]})
            else:
                if rows:
                    return name, rows
    return None, []


def base_dir_join(manifest_name: str, rel: str) -> str:
    parent = PurePosixPath(manifest_name).parent
    cand = PurePosixPath(rel)
    return str(cand) if str(parent) in (".", "") else str(parent / cand)


def image_stats(a: np.ndarray) -> dict:
    x = a.astype(np.float64) / 255.0
    f = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    yy, xx = np.mgrid[-64:64, -64:64]
    r = np.hypot(yy, xx)
    tot = f.sum()
    gy, gx = np.gradient(x)
    return {"mean": x.mean(), "std": x.std(), "frac_zero": (a == 0).mean(), "frac_gt_0.5": (x > 0.5).mean(),
            "frac_ge_0.9": (x >= 0.9).mean(), "dyn_range_p1_p99": np.percentile(x, 99) - np.percentile(x, 1),
            "levels": float(len(np.unique(a))), "binary_like": float(len(np.unique(a)) <= 2),
            "hf_power_frac_r_gt_16": float(f[r > 16].sum() / tot) if tot > 0 else 0.0,
            "edge_density_grad_gt_0.1": (np.hypot(gx, gy) > 0.1).mean()}


def pearson(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d == 0 else float((a * b).sum() / d)


def inter_z(q):
    vals = [pearson(q[i], q[j]) for i in range(4) for j in range(i + 1, 4)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    keys = [k for k in rows[0] if rows[0][k] is not None]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        out[k] = {"n": int(v.size), "mean": float(v.mean()), "p5": float(np.percentile(v, 5)),
                  "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
    return out


def our_quartets(corpus: Path):
    man_bytes = (corpus / "manifest_expanded.json").read_bytes()
    man = json.loads(man_bytes)
    ours = {"pixel": {}, "dhash": [], "arrays": {}, "quartet_ordered": {}, "quartet_set": {}, "test_file_sha": {},
            "stats": {"A": [], "B": []}, "interz": {"A": [], "B": []}}
    for s in man["samples"]:
        if s["split"] == "test":
            for i, h in enumerate(s["sha256"]):
                ours["test_file_sha"][h] = f"{s['key']}#z{i + 1}"  # metadata only; image never opened
            continue
        arrs, psh = [], []
        for i, rel in enumerate(s["paths"]):
            a = np.asarray(Image.open(corpus / rel))
            tag = f"{s['split']}:{s['key']}#z{i + 1}"
            p = sha(a.tobytes())
            ours["pixel"][p] = tag
            ours["dhash"].append((base.dhash(a), tag))
            ours["arrays"][tag] = a
            arrs.append(a); psh.append(p)
        ours["quartet_ordered"][tuple(psh)] = f"{s['split']}:{s['key']}"
        ours["quartet_set"][tuple(sorted(psh))] = f"{s['split']}:{s['key']}"
        if s["split"] == "train":
            ours["stats"][s["kind"]].extend(image_stats(a) for a in arrs)
            ours["interz"][s["kind"]].append(inter_z(arrs))
    return sha(man_bytes), man.get("content_sha256"), ours


def holdout_rule(keys: list[str], k: int = HOLDOUT_SIZE) -> list[str]:
    return sorted(keys, key=lambda q: hashlib.sha256((HOLDOUT_SALT + "|" + q).encode()).hexdigest())[:k]


def run(archive: Path, corpus: Path, out_dir: Path, expect_sha: str | None = None, rights: Path | None = None) -> int:
    if out_dir.exists():
        raise SystemExit(f"refusing to reuse existing output dir {out_dir}")
    vendored = sha((Path(__file__).resolve().parent / "external_dataset_audit.py").read_bytes())
    archive_sha = sha(archive.read_bytes())
    findings = {k: [] for k in ("unsafe", "forbidden", "bad_png", "quartet_problems", "test_overlap", "sha_mismatch",
                                 "split_conflicts", "intra_quartet_identical_planes", "internal_duplicate_quartets",
                                 "internal_permuted_quartets", "ours_quartet_exact", "ours_quartet_permuted",
                                 "trainval_exact_planes", "trainval_near_planes", "internal_duplicate_planes",
                                 "optics_text", "unassigned_quartets", "uncertain_order")}
    if expect_sha and archive_sha != expect_sha.lower():
        findings["sha_mismatch"].append({"expected": expect_sha, "actual": archive_sha})
    man_sha, man_content, ours = our_quartets(corpus)

    texts, images, n_members = {}, {}, 0
    for name, data in base.members(archive, findings):
        n_members += 1
        p = PurePosixPath(name)
        ext = p.suffix.lower()
        if ext in base.FORBIDDEN_EXT or re.search(r"doe|phase", p.name, re.I):
            findings["forbidden"].append(name)
        fsha = sha(data)
        if fsha in ours["test_file_sha"]:
            findings["test_overlap"].append({"name": name, "ours": ours["test_file_sha"][fsha], "by": "file sha256"})
        if ext in base.TEXT_EXT:
            txt = data[:2_000_000].decode("utf-8", "replace")
            texts[name] = txt
            hit = {k: [m.group(0)[:60] for m in r.finditer(txt)][:5] for k, r in base.NUM_KW.items()}
            kw = sorted({m.group(0).lower() for m in base.OPTICS_KW.finditer(txt)})
            if kw:
                findings["optics_text"].append({"name": name, "values": hit, "lens_keywords": kw})
        if ext != ".png":
            continue
        try:
            im = Image.open(io.BytesIO(data)); im.load()
        except Exception as e:  # noqa: BLE001
            findings["bad_png"].append({"name": name, "why": f"decode: {type(e).__name__}"}); continue
        bitdepth_ok = len(data) >= 26 and data[24] == 8 and data[25] == 0
        if im.mode != "L" or im.size != (128, 128) or not bitdepth_ok:
            findings["bad_png"].append({"name": name, "why": f"mode={im.mode} size={im.size} ihdr8gray={bitdepth_ok}"}); continue
        a = np.asarray(im)
        images[name] = {"file_sha256": fsha, "pixel_sha256": sha(a.tobytes()), "array": a}

    # quartets: manifest first, else filename grouping
    man_member, man_rows = find_manifest_samples(texts)
    quartets = {}
    if man_rows:
        for row in man_rows:
            missing = [x for x in row["paths"] if x not in images]
            if missing:
                findings["quartet_problems"].append({"key": row["key"], "missing_or_bad": missing}); continue
            key = row["key"] or "|".join(row["paths"])
            split = row["split"]
            psplits = {split_from_path(x) for x in row["paths"]} - {None}
            if split and psplits and psplits != {split}:
                findings["split_conflicts"].append({"key": key, "manifest": split, "path": sorted(psplits)})
                continue
            quartets[key] = {"paths": row["paths"], "split": split or (psplits.pop() if len(psplits) == 1 else None),
                             "split_source": "manifest" if split else "path"}
        listed = {x for r in man_rows for x in r["paths"]}
        stray = sorted(set(images) - listed)
        if stray:
            findings["quartet_problems"].append({"unlisted_pngs": len(stray), "examples": stray[:5]})
    else:
        groups = {}
        for name in images:
            g, z = z_of(name)
            if g is None:
                findings["uncertain_order"].append(name); continue
            groups.setdefault(g, []).append((z, name))
        for g, items in sorted(groups.items()):
            zs = sorted(z for z, _ in items)
            if zs != [1, 2, 3, 4]:
                findings["quartet_problems"].append({"group": g, "files": sorted(n for _, n in items), "z": zs}); continue
            paths = [n for _, n in sorted(items)]
            sp = {split_from_path(x) for x in paths}
            quartets[g] = {"paths": paths, "split": sp.pop() if len(sp) == 1 else None, "split_source": "path"}

    # quartet-level checks
    seen_plane, seen_q, seen_qset, records = {}, {}, {}, []
    for key, q in sorted(quartets.items()):
        info = [images[x] for x in q["paths"]]
        psh = tuple(i["pixel_sha256"] for i in info)
        q["file_sha256"] = [i["file_sha256"] for i in info]
        q["pixel_sha256"] = list(psh)
        if q["split"] is None:
            findings["unassigned_quartets"].append(key)
        if len(set(psh)) < 4:
            findings["intra_quartet_identical_planes"].append({"key": key, "pixel_sha256": list(psh)})
        if psh in seen_q:
            findings["internal_duplicate_quartets"].append([seen_q[psh], key])
        elif tuple(sorted(psh)) in seen_qset:
            findings["internal_permuted_quartets"].append([seen_qset[tuple(sorted(psh))], key])
        seen_q.setdefault(psh, key); seen_qset.setdefault(tuple(sorted(psh)), key)
        if psh in ours["quartet_ordered"]:
            findings["ours_quartet_exact"].append({"key": key, "ours": ours["quartet_ordered"][psh]})
        elif tuple(sorted(psh)) in ours["quartet_set"]:
            findings["ours_quartet_permuted"].append({"key": key, "ours": ours["quartet_set"][tuple(sorted(psh))]})
        flags = set()
        for zi, (name, i) in enumerate(zip(q["paths"], info), 1):
            a = i["array"]
            if i["pixel_sha256"] in seen_plane and seen_plane[i["pixel_sha256"]] != key:
                findings["internal_duplicate_planes"].append({"plane": f"{key}#z{zi}", "first": seen_plane[i["pixel_sha256"]]})
            seen_plane.setdefault(i["pixel_sha256"], key)
            if i["pixel_sha256"] in ours["pixel"]:
                findings["trainval_exact_planes"].append({"plane": f"{key}#z{zi}", "ours": ours["pixel"][i["pixel_sha256"]]})
                flags.add("exact_ours")
                continue
            h = base.dhash(a)
            best = min(((bin(h ^ oh).count("1"), t) for oh, t in ours["dhash"]), default=None)
            if best and best[0] <= NEAR_HAMMING:
                b = ours["arrays"][best[1]]
                flat = bool(a.std() < 5 or b.std() < 5)
                mae = float(np.abs(a.astype(float) - b.astype(float)).mean())
                findings["trainval_near_planes"].append({"plane": f"{key}#z{zi}", "ours": best[1], "hamming": best[0],
                                                         "pixel_mae_u8": mae, "pearson": pearson(a, b), "near_uniform": flat,
                                                         "verdict": "suspect_needs_review" if (mae < 8 and not flat) else "dhash_only"})
                if mae < 8 and not flat:
                    flags.add("near_ours")
        q["flags"] = sorted(flags)
        if q["split"] == "train":
            records.append((key, [i["array"] for i in info]))

    split_counts = {}
    for q in quartets.values():
        split_counts[str(q["split"])] = split_counts.get(str(q["split"]), 0) + 1
    dup_keys = {k for pair in findings["internal_duplicate_quartets"] + findings["internal_permuted_quartets"] for k in pair}
    eligible = sorted(k for k, q in quartets.items() if q["split"] == "train" and not q["flags"] and k not in dup_keys
                      and k not in {d["key"] for d in findings["intra_quartet_identical_planes"]})
    holdout = holdout_rule(eligible)
    blocking_keys = ("unsafe", "forbidden", "bad_png", "quartet_problems", "test_overlap", "sha_mismatch",
                     "split_conflicts", "ours_quartet_exact", "ours_quartet_permuted", "uncertain_order")
    blocking = {k: len(findings[k]) for k in blocking_keys if findings[k]}
    technical_pass = not blocking and split_counts.get("train", 0) > 0
    rights_text = rights.read_text(encoding="utf-8") if rights else None
    rights_ok = bool(rights_text and rights_text.strip())

    wc_stats = [image_stats(a) for _, arrs in records for a in arrs]
    dist = {"wc_train": summarize(wc_stats), "ours_train_A": summarize(ours["stats"]["A"]),
            "ours_train_B": summarize(ours["stats"]["B"]),
            "inter_z_pearson_quartet_mean": {"wc_train": summarize([{"v": inter_z(a)} for _, a in records]),
                                             "ours_train_A": summarize([{"v": v} for v in ours["interz"]["A"]]),
                                             "ours_train_B": summarize([{"v": v} for v in ours["interz"]["B"]])},
            "note": "descriptive only; external quartets get NO A/B label"}
    audit = {"tool": "wc_audit.py", "auditor_base_sha256": vendored, "auditor_base_expected": AUDITOR_SHA256,
             "archive": str(archive), "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
             "members_read": n_members, "png_valid": len(images), "manifest_member": man_member,
             "our_manifest_file_sha256": man_sha, "our_manifest_content_sha256": man_content,
             "quartets": len(quartets), "split_counts": split_counts,
             "findings": findings, "blocking": blocking, "technical_pass": technical_pass,
             "usage_rights": "CONFIRMED_BY_EVIDENCE_FILE" if rights_ok else "NOT_CONFIRMED",
             "rights_evidence_sha256": sha(rights_text.encode()) if rights_ok else None,
             "training_allowed": bool(technical_pass and rights_ok),
             "amplitude_contract": "declared by source as PNG/255 amplitude; pixels alone cannot prove amplitude vs intensity",
             "distribution": dist,
             "holdout": {"rule": f"sha256('{HOLDOUT_SALT}|'+key) ascending, first {HOLDOUT_SIZE} of eligible TRAIN",
                         "eligible": len(eligible), "keys": holdout},
             "policy": "read-only, in-memory; our TEST compared by manifest file sha only; non-train splits never used"}
    mapping = {"schema": "wc-mapping/1", "archive_sha256": archive_sha, "contract": "PNG/255 amplitude, z1..z4 = 24.1/25.3/26.5/27.7 mm",
               "train_keys": [k for k in eligible if k not in set(holdout)], "holdout_keys": holdout,
               "excluded": {k: ("non-train split" if q["split"] != "train" else "flagged") for k, q in quartets.items()
                            if q["split"] != "train" or k not in eligible},
               "quartets": {k: {"paths": q["paths"], "file_sha256": q["file_sha256"], "split": q["split"],
                                "split_source": q["split_source"]} for k, q in sorted(quartets.items())}}
    out_dir.mkdir(parents=True)
    (out_dir / "WC_AUDIT.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (out_dir / "WC_MAPPING.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# WC_MAPPING (archive sha256 `{archive_sha}`)", "",
          "Contract: each quartet = 4 L8 128x128 PNG, value/255 = target AMPLITUDE at z1..z4 = 24.1/25.3/26.5/27.7 mm.",
          "No sqrt, gamma, per-image max normalisation, resampling, z reorder or phase labels.", "",
          f"- quartets: {len(quartets)}; split counts: {split_counts}",
          f"- training keys (train, unflagged, not holdout): {len(mapping['train_keys'])}",
          f"- internal holdout ({HOLDOUT_SIZE}, excluded from training in BOTH arms): rule `{audit['holdout']['rule']}`",
          f"- excluded: {len(mapping['excluded'])} (non-train splits are never read by the adapter)",
          f"- usage rights: {audit['usage_rights']}; training_allowed: {audit['training_allowed']}",
          f"- blocking: {blocking or 'none'}"]
    (out_dir / "WC_MAPPING.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"technical_pass": technical_pass, "training_allowed": audit["training_allowed"],
                      "quartets": len(quartets), "split_counts": split_counts, "blocking": blocking}))
    return 0 if technical_pass else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path); ap.add_argument("corpus", type=Path); ap.add_argument("out_dir", type=Path)
    ap.add_argument("--expect-sha256"); ap.add_argument("--rights-evidence", type=Path)
    a = ap.parse_args(argv)
    return run(a.archive, a.corpus, a.out_dir, a.expect_sha256, a.rights_evidence)


if __name__ == "__main__":
    sys.exit(main())
