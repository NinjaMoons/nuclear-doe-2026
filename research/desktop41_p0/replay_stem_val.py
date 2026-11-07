"""Independent host replay of a STEM checkpoint on the 16 VAL quartets (#41 independent launch check). No TEST.

usage (cwd = worktree at 44f0378+):
  python research/desktop41_p0/replay_stem_val.py CKPT CORPUS_DIR REFERENCE_EXPORT_DIR NEW_OUT_DIR

For each VAL sample of manifest_expanded.json (TEST is skipped by split, never opened) it runs the one-forward
adapter `back_projection_stem.infer_neural_stem` (stock CLI, CPU) into NEW_OUT_DIR/<i>, then compares DOE.png and the
4 reconstructions byte-for-byte with REFERENCE_EXPORT_DIR/<i> (same sample_key required) and aggregates the official
per-plane Pearson/SSIM from metrics.json.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(ckpt, corpus, ref, out):
    corpus, ref, out = Path(corpus), Path(ref), Path(out)
    if out.exists():
        raise SystemExit("refusing to reuse " + str(out))
    out.mkdir(parents=True)
    man = json.loads((corpus / "manifest_expanded.json").read_text())
    val = [s for s in man["samples"] if s["split"] == "val"]
    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(val):
        d = out / f"{i:03d}"
        r = ref / f"{i:03d}"
        rmeta = json.loads((r / "manifest.json").read_text(encoding="utf-8"))["metadata"]
        if rmeta.get("sample_key") != s["key"]:
            raise SystemExit(f"reference {r} is {rmeta.get('sample_key')}, expected {s['key']}")
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "back_projection_stem.infer_neural_stem", "--checkpoint", str(ckpt),
                            "--targets", *[str(corpus / x) for x in s["paths"]], "--output-dir", str(d)],
                           capture_output=True, text=True)
        wall = time.perf_counter() - t
        if p.returncode:
            raise SystemExit(f"{s['key']}: exit {p.returncode}\n{p.stderr}")
        files = ["DOE.png"] + [f"reconstructed_{k:02d}.png" for k in range(1, 5)]
        same = {f: sha(d / f) == sha(r / f) for f in files}
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"key": s["key"], "kind": s["kind"], "wall_s": wall, "byte_identical": same,
                     "pearson": [pl["pearson"] for pl in m["planes"]], "ssim": [pl["ssim"] for pl in m["planes"]],
                     "doe_sha256": sha(d / "DOE.png")})
    P = np.array([r["pearson"] for r in rows], float)
    S = np.array([r["ssim"] for r in rows], float)
    grp = lambda k: [i for i, r in enumerate(rows) if r["kind"] == k]
    summary = {"checkpoint_sha256": sha(ckpt), "manifest_sha256": sha(corpus / "manifest_expanded.json"), "n": len(rows),
               "pearson_mean": float(P.mean()), "ssim_mean": float(S.mean()),
               "pearson_plane": P.mean(0).tolist(), "ssim_plane": S.mean(0).tolist(),
               "worst_plane_pearson": float(P.mean(0).min()), "worst_plane_ssim": float(S.mean(0).min()),
               "A": {"pearson": float(P[grp("A")].mean()), "ssim": float(S[grp("A")].mean())},
               "B": {"pearson": float(P[grp("B")].mean()), "ssim": float(S[grp("B")].mean())},
               "doe_byte_identical": sum(r["byte_identical"]["DOE.png"] for r in rows),
               "all_5_files_byte_identical": sum(all(r["byte_identical"].values()) for r in rows),
               "total_wall_s": time.perf_counter() - t0, "device": "cpu (stock CLI)", "rows": rows}
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
