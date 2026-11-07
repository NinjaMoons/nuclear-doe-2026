"""RESEARCH ONLY (R18): seed-mean paired comparison bpmix vs control from analyze_r12 JSONs (VAL16 only).
usage: python -I seedmean_r18.py <out.json> <analysis_s0.json> <analysis_s1.json> ..."""
import json, sys
import numpy as np
A = [json.load(open(p)) for p in sys.argv[2:]]
def rows(a, arm): return {r["key"]: r for r in a[arm]["checkpoints"]["final"]["rows"]}
keys = sorted(rows(A[0], "control"))
def stat(arm, f, sel=lambda r: True):  # per-quartet value averaged over seeds
    return np.array([np.mean([f(rows(a, arm)[k]) for a in A]) for k in keys if sel(rows(A[0], arm)[k])])
P = lambda r: float(np.mean(r["P"])); S = lambda r: float(np.mean(r["S"])); isB = lambda r: r["g"] == "B"
out = {"seeds": len(A), "per_seed": {}}
for i, a in enumerate(A):
    out["per_seed"][i] = {arm: {"P": a[arm]["checkpoints"]["final"]["summary"]["ALL"]} for arm in ("control", "tphase")}
rng = np.random.default_rng(0)
def boot(d):
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int((d > 0).sum()), len(d)]
for name, f, sel in [("P", P, lambda r: True), ("SSIM", S, lambda r: True), ("B_P", P, isB), ("A_P", P, lambda r: not isB(r)),
                     ("B_own", lambda r: r["own"], isB), ("B_margin", lambda r: r["margin"], isB)]:
    c, b = stat("control", f, sel), stat("tphase", f, sel)
    out[name] = {"control": float(c.mean()), "tphase": float(b.mean()), "delta_boot": boot(b - c)}
for arm in ("control", "tphase"):  # worst plane over seed-mean per-plane P
    pp = np.mean([[np.mean([rows(a, arm)[k]["P"][j] for k in keys]) for j in range(4)] for a in A], 0)
    out.setdefault("worst_plane_P", {})[arm] = float(pp.min())
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))
