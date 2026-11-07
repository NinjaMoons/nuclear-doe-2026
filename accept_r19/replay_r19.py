"""Independent Mac CPU replay of R19 s0 checkpoints on VAL16 only (no TEST read). Code = a07b6e9 (cwd).
usage: python replay_r19.py <asset_root> <out_dir> <manifest>"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
import doe_models.train_multiplane as tm
from doe_core.contracts_v1 import CaseConfigV1
from tphase_head.train_r19 import research_load, _stock_load
root, out, M = map(lambda p: Path(p).expanduser(), sys.argv[1:4]); man_raw = M.read_bytes(); man = json.loads(man_raw)
assert hashlib.sha256(man_raw).hexdigest().startswith("76d2474d")
comps = tm.production_components(); rows = [s for s in man["samples"] if s["split"] == "val"]
files = [[M.parent / p for p in s["paths"]] for s in rows]
for s, fs in zip(rows, files):
    assert all(hashlib.sha256(f.read_bytes()).hexdigest() == h for f, h in zip(fs, s["sha256"]))
val = {"targets": torch.cat([comps.load_quartet(fs) for fs in files]), "ids": [s["key"] for s in rows],
       "origins": [s["origin_id"] for s in rows], "kinds": [s["kind"] for s in rows], "paths": files}
data = {"splits": {"val": val}, "manifest_sha256": hashlib.sha256(man_raw).hexdigest(), "manifest_path": str(M), "n_samples": 16, "synthetic": True, "corpus_ids": []}
cfg = CaseConfigV1(); tm.load_checkpoint = research_load; res = {}
for arm in ("control", "tphase"):
    ck = root / arm / "step_006000.pt"; od = out / arm
    s = tm.evaluate_checkpoints([ck], data, comps, od, torch.device("cpu"), ("val",), cfg)["checkpoints"][0]["val"]
    ref = json.load(open(root / arm / "r19_summary.json"))["eval"]["checkpoints"][0]["val"]
    mine = sorted((od / "export" / "step_006000.pt" / "val").rglob("*.png")); diffs = []; same = 0
    for p in mine:
        q = root / arm / "eval_val" / "export" / "step_006000.pt" / "val" / p.relative_to(od / "export" / "step_006000.pt" / "val")
        a, b = np.asarray(Image.open(p), dtype=int), np.asarray(Image.open(q), dtype=int); same += int((a == b).all()); diffs.append(int(np.abs(a - b).max()))
    res[arm] = {"mac_P": s["pearson_mean"], "mac_S": s["ssim_mean"], "gpu_P": ref["pearson_mean"], "gpu_S": ref["ssim_mean"],
                "png_identical": f"{same}/{len(mine)}", "png_max_abs_diff": max(diffs), "ckpt_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}
    print(arm, res[arm], flush=True)
json.dump(res, open(out / "replay_r19.json", "w"), indent=1)
