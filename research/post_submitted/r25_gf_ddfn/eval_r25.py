"""R25: official VAL16 evaluation of ONE finished checkpoint (separate from training so VAL is read only after BOTH arms
have finished and their provenance is verified). TEST is never decoded (same no-TEST loader as run_r25_notest).
usage: python -m r25_gf_ddfn.eval_r25 --ckpt CKPT --expect-sha SHA256 --manifest M --out NEW_DIR [--device cuda]"""
import argparse, hashlib, json
from pathlib import Path

import torch

import r25_gf_ddfn.run_r25_notest  # noqa: F401  (installs the train+val-only loader into tm.load_manifest)
from doe_core.contracts_v1 import CaseConfigV1
from r25_gf_ddfn.train_r25 import research_load, tm, _stock_load


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ckpt", "--expect-sha", "--manifest", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ck = Path(a.ckpt); sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    if sha != a.expect_sha:
        raise SystemExit(f"checkpoint sha mismatch {sha} != {a.expect_sha}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    comps = tm.production_components(); data = tm.load_manifest(a.manifest, load_quartet=comps.load_quartet)
    assert data["manifest_sha256"].startswith("76d2474d") and "test" not in data["splits"]
    tm.load_checkpoint = research_load
    try:
        s = tm.evaluate_checkpoints([ck], data, comps, out, torch.device(a.device), ("val",), CaseConfigV1())
    finally:
        tm.load_checkpoint = _stock_load
    json.dump({"ckpt": str(ck), "ckpt_sha256": sha, "eval": s}, open(out / "eval_summary.json", "w"), indent=1)
    v = s["checkpoints"][0]["val"]
    print("VAL", ck.parent.name, round(v["pearson_mean"], 5), round(v["ssim_mean"], 5))


if __name__ == "__main__":
    main()
