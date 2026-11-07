"""RESEARCH ONLY: compare VAL exports of research arms (A/B, per-plane, worst) from existing metrics.json.

Usage: python val_compare.py <out.json> <run_dir> [<run_dir> ...]
Reads <run>/eval_val/export/<checkpoint>/val/NNN/{metrics,manifest}.json written by the production
VAL evaluator, plus <run>/train_log.json and <run>/research_lambda.json. Never recomputes metrics.
"""
import json
import statistics as st
import sys
from pathlib import Path


def group_of(manifest):
    p = manifest["input_sources"][0]["path"].replace("\\", "/").split("/")
    return p[-3][0], f"{p[-3]}:{p[-2]}"


def summarize(ck_dir):
    rows = []
    for d in sorted((ck_dir / "val").iterdir()):
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        g, key = group_of(man)
        planes = m["planes"]
        rows.append({"key": key, "group": g, "pearson": [p["pearson"] for p in planes],
                     "ssim": [p["ssim"] for p in planes],
                     "undefined": [p["index"] for p in planes if not p["pearson_defined"]]})
    out = {"n": len(rows), "undefined_pearson": [(r["key"], r["undefined"]) for r in rows if r["undefined"]]}
    for met in ("pearson", "ssim"):
        q = {r["key"]: st.mean(r[met]) for r in rows}
        res = {"ALL": st.mean(q.values())}
        for g in ("A", "B"):
            res[g] = st.mean(v for r, v in zip(rows, q.values()) if r["group"] == g)
        res["per_plane"] = [st.mean(r[met][i] for r in rows) for i in range(4)]
        res["worst_plane_mean"] = min(res["per_plane"])
        res["worst_sample_plane"] = min(((r["key"], i + 1, r[met][i]) for r in rows for i in range(4)),
                                        key=lambda x: x[2])
        res["per_quartet"] = q
        out[met] = res
    return out


def main():
    out_path, runs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    report = {}
    for run in runs:
        lam_file = run / "research_lambda.json"
        lam = json.loads(lam_file.read_text())["lambda"] if lam_file.exists() else None
        log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
        val_log = [{k: v.get(k) for k in ("step", "objective", "contract_mse", "lsq_nmse", "val_loss")}
                   for v in log["val"]]
        cks = {}
        for ck in sorted((run / "eval_val" / "export").iterdir()):
            cks[ck.name] = summarize(ck)
        report[run.name] = {"lambda": lam, "best_step": log["best_step"], "elapsed_s": log["elapsed_s"],
                            "device": log["device"], "val_log": val_log, "checkpoints": cks}
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, r in report.items():
        print(f"== {name} lambda={r['lambda']} best_step={r['best_step']} elapsed={r['elapsed_s']:.1f}s")
        for v in r["val_log"]:
            print("   val", v)
        for ck, s in r["checkpoints"].items():
            p, q = s["pearson"], s["ssim"]
            print(f"   {ck:16s} P ALL {p['ALL']:.4f} A {p['A']:.4f} B {p['B']:.4f} worst-plane {p['worst_plane_mean']:.4f}"
                  f" | S ALL {q['ALL']:.4f} A {q['A']:.4f} B {q['B']:.4f} worst-plane {q['worst_plane_mean']:.4f}"
                  f" | undef {s['undefined_pearson']}")
            print("      P planes", [round(x, 4) for x in p["per_plane"]], "S planes", [round(x, 4) for x in q["per_plane"]],
                  "worst P", p["worst_sample_plane"][:2], round(p["worst_sample_plane"][2], 4))


if __name__ == "__main__":
    main()
