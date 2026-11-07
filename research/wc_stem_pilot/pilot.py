"""#41 paired GPU pilot: STEM 12ch + R3, arm A native512 vs arm B native512 + WhiteCollars TRAIN 50/50. RESEARCH ONLY.

See PLAN_WC_PILOT.md (pre-registered). TEST is never decoded: native data are read only for split train/val from
manifest_expanded.json (file sha checked per PNG), WhiteCollars only through the audited TRAIN adapter.

  python research/wc_stem_pilot/pilot.py prep  --corpus C --wc-archive W --wc-mapping M --orgsets Z --out EVALDIR
  python research/wc_stem_pilot/pilot.py train --corpus C --arm {native,native_wc} --seed S --updates U
         --ckpt-every K --out NEWDIR [--wc-archive W --wc-mapping M]
  python research/wc_stem_pilot/pilot.py eval  --corpus C --evalsets EVALDIR --out NEWDIR CKPT [CKPT ...]
"""
import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "whitecollars_dataset"))

import doe_models.train_multiplane as tm  # noqa: E402
from doe_core.contracts_v1 import CaseConfigV1  # noqa: E402
from doe_experiments.data import load_png_quartet  # noqa: E402

from back_projection_stem import train_wrapper as tw  # noqa: E402  (registers the R3 objective)
from back_projection_stem.stem import build_stem_model  # noqa: E402

WC_SHA = "f58db14ea2fbe4235bfe4ea21855a45d52b2d506700f0c0f82ae16a3e29fc521"
NATIVE_SHA = "76d2474d"  # prefix of the expected manifest_expanded.json file sha
ORG_SHA = "ac908518047b6b73e572cfffae9aaebf5c9247aed736c37383fbeadae0c6856c"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def native(corpus: Path, splits=("train", "val")):
    raw = (corpus / "manifest_expanded.json").read_bytes()
    msha = sha(raw)
    if not msha.startswith(NATIVE_SHA):
        raise SystemExit(f"unexpected native manifest {msha}")
    man = json.loads(raw)
    out = {}
    for split in splits:
        if split == "test":
            raise SystemExit("TEST is frozen and never decoded here")
        rows = [s for s in man["samples"] if s["split"] == split]   # manifest order == stock driver order
        files = []
        for s in rows:
            fs = [corpus / p for p in s["paths"]]
            for f, h in zip(fs, s["sha256"]):
                if sha(f.read_bytes()) != h:
                    raise SystemExit(f"sha mismatch {f}")
            files.append(fs)
        out[split] = {"targets": torch.cat([load_png_quartet(f) for f in files]), "ids": [s["key"] for s in rows],
                      "origins": [s["origin_id"] for s in rows], "kinds": [s["kind"] for s in rows], "paths": files}
    return out, msha


# ------------------------------------------------------------------------------------------------ prep eval sets
def prep(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(Path(a.wc_mapping).read_text(encoding="utf-8"))
    if mapping["archive_sha256"] != WC_SHA or sha(Path(a.wc_archive).read_bytes()) != WC_SHA:
        raise SystemExit("WC archive/mapping sha mismatch")
    index = {}
    hold = mapping["holdout_keys"]
    want = {p: (k, z) for k in hold for z, p in enumerate(mapping["quartets"][k]["paths"])}
    got = {}
    with tarfile.open(a.wc_archive, "r:gz") as t:
        for m in t:
            if m.isfile() and m.name in want:
                got[m.name] = t.extractfile(m).read()
    rows = []
    for k in hold:
        d = out / "wc_holdout64" / k.replace("/", "_").replace(":", "_")
        d.mkdir(parents=True)
        q = mapping["quartets"][k]
        for z, (p, h) in enumerate(zip(q["paths"], q["file_sha256"]), 1):
            if sha(got[p]) != h:
                raise SystemExit(f"holdout sha mismatch {p}")
            (d / f"z{z}.png").write_bytes(got[p])
        rows.append({"key": "WC:" + k, "dir": str(d.relative_to(out)), "sha256": q["file_sha256"]})
    index["wc_holdout64"] = rows
    zb = Path(a.orgsets).read_bytes()
    if sha(zb) != ORG_SHA:
        raise SystemExit("orgsets zip sha mismatch")
    z = zipfile.ZipFile(io.BytesIO(zb))
    rows = []
    for name in sorted({n.split("/")[0] for n in z.namelist() if n.count("/") == 1 and n.endswith(".png")}):
        d = out / "orgsets10" / name
        d.mkdir(parents=True)
        hs = []
        for k in range(1, 5):
            b = z.read(f"{name}/z{k}.png")
            (d / f"z{k}.png").write_bytes(b)
            hs.append(sha(b))
        rows.append({"key": "ORG:" + name, "dir": str(d.relative_to(out)), "sha256": hs})
    index["orgsets10"] = rows
    (out / "evalsets.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in index.items()})


def evalset(evaldir: Path, name: str):
    idx = json.loads((evaldir / "evalsets.json").read_text(encoding="utf-8"))[name]
    files, ids = [], []
    for r in idx:
        fs = [evaldir / r["dir"] / f"z{k}.png" for k in range(1, 5)]
        for f, h in zip(fs, r["sha256"]):
            if sha(f.read_bytes()) != h:
                raise SystemExit(f"evalset sha mismatch {f}")
        files.append(fs)
        ids.append(r["key"])
    return {"targets": torch.cat([load_png_quartet(f) for f in files]), "ids": ids, "origins": ids,
            "kinds": [name] * len(ids), "paths": files}


# ------------------------------------------------------------------------------------------------ train
def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus))
    tr = data["train"]
    if len(tr["ids"]) != 512 or tr["kinds"].count("A") != 256 or len(data["val"]["ids"]) != 16:
        raise SystemExit("native split is not 256A+256B / VAL16")
    train_t = tr["targets"].to(dev)
    wc_info = None
    if a.arm == "native_wc":
        from adapter import WhiteCollarsTrain
        ds = WhiteCollarsTrain(a.wc_archive, a.wc_mapping, role="train", expect_archive_sha256=WC_SHA)
        wc_t = ds.targets().to(dev)
        if len(ds) != 4025:
            raise SystemExit(f"expected 4025 WC TRAIN quartets, got {len(ds)}")
        wc_info = {"n": len(ds), "archive_sha256": ds.archive_sha256,
                   "mapping_sha256": sha(Path(a.wc_mapping).read_bytes()), "keys_sha256": sha("\n".join(ds.keys).encode())}
    elif a.arm != "native":
        raise SystemExit("arm must be native or native_wc")
    cfg = CaseConfigV1()
    prop = tm.production_components().propagate
    model = build_stem_model(seed=a.seed, base_channels=a.base_channels, depth=a.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen, gen_w = torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(10000 + a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    run_cfg = {"arm": a.arm, "seed": a.seed, "updates": a.updates, "ckpt_every": a.ckpt_every, "lr": 1e-3, "lr_schedule": a.lr_schedule, "batch": 4,
               "objective": tw.R3_NAME, "model": "build_stem_model (12ch back-projection STEM)", "hparams": model.hparams, "precision": "float32",
               "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "native_manifest_sha256": msha,
               "native_train": 512, "val": data["val"]["ids"], "wc": wc_info, "code_commit": commit, "git_dirty": dirty,
               "pilot_sha256": sha(Path(__file__).read_bytes()),
               "sampler": "native: randperm(512,gen(seed))[:4]; native_wc: randperm(512,gen(seed))[:2] + "
                          "randperm(4025,gen(10000+seed))[:2]"}
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist, counts = [], {"A": 0, "B": 0, "WC": 0}
    kinds = tr["kinds"]
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        if a.lr_schedule == "cosine":   # 1e-3 -> 1e-5 over the run, set BEFORE the update of this step
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (step - 1) / a.updates))
        if a.arm == "native":
            idx = torch.randperm(512, generator=gen)[:4]
            batch = train_t[idx.to(dev)]
        else:
            idx = torch.randperm(512, generator=gen)[:2]
            iw = torch.randperm(wc_t.shape[0], generator=gen_w)[:2]
            batch = torch.cat([train_t[idx.to(dev)], wc_t[iw.to(dev)]])
            counts["WC"] += 2
        for i in idx.tolist():
            counts[kinds[i]] += 1
        opt.zero_grad(set_to_none=True)
        loss, info = tm._objective(tm.multiplane_amplitudes(model(batch), cfg, prop), batch, tw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        gn = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
        if not math.isfinite(gn):
            raise RuntimeError(f"non-finite grad at {step}")
        opt.step()
        hist.append({"step": step, "loss": v, "grad_norm": gn, "corr_term": info.get("corr_term")})
        if step % a.ckpt_every == 0 or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "arm": a.arm, "seed": a.seed},
                               extra={"kind": "periodic", "step": step, "resumable": False, "arm": a.arm,
                                      "research_only": True, "wc_train": a.arm == "native_wc"})
    torch.cuda.synchronize()
    log = {"elapsed_s": time.perf_counter() - t0, "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)),
           "sample_counts": counts, "history": hist,
           "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=1))


# ------------------------------------------------------------------------------------------------ eval
def evaluate(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, msha = native(Path(a.corpus), splits=("val",))
    sets = {"val16": data["val"]}
    for name in ("wc_holdout64", "orgsets10"):
        sets[name] = evalset(Path(a.evalsets), name)
    comps = tm.production_components()
    tm.load_checkpoint = tw.research_load_checkpoint
    res = []
    t0 = time.perf_counter()
    import shutil
    (out / "ckpts").mkdir()
    for src in a.ckpts:
        # export dirs are keyed by checkpoint file name -> copy under a unique '<run>__<file>' name first
        src = Path(src)
        c = out / "ckpts" / (src.parent.parent.name + "__" + src.name)
        if c.exists():
            raise SystemExit(f"duplicate checkpoint name {c.name}")
        shutil.copyfile(src, c)
        row = {"checkpoint": str(src), "name": c.name, "sha256": sha(c.read_bytes())}
        for name, part in sets.items():
            d = {"splits": {"val": part}, "manifest_sha256": msha if name == "val16" else name, "synthetic": True}
            s = tm.evaluate_checkpoint(c, d, comps, out / name, dev, CaseConfigV1(), ("val",))
            row[name] = s["val"]
            P, S = per_quartet(out / name / "export" / Path(c).name / "val")
            row[name]["per_quartet_pearson"], row[name]["per_quartet_ssim"] = P, S
            row[name]["ids"] = part["ids"]
            if name == "val16":
                for g in ("A", "B"):
                    ii = [i for i, k in enumerate(part["kinds"]) if k == g]
                    row[name][g] = {"pearson": float(np.mean([P[i] for i in ii])), "ssim": float(np.mean([S[i] for i in ii]))}
        res.append(row)
        print(Path(c).name, {n: round(row[n]["pearson_mean"], 4) for n in sets}, {n: round(row[n]["ssim_mean"], 4) for n in sets},
              flush=True)
    (out / "eval.json").write_text(json.dumps({"rows": res, "elapsed_s": time.perf_counter() - t0, "device": str(dev),
                                               "path": "official evaluate_checkpoint / export_run"}, indent=1),
                                   encoding="utf-8")


def per_quartet(exp_dir: Path):
    P, S = [], []
    for q in sorted(exp_dir.iterdir()):
        pl = json.loads((q / "metrics.json").read_text(encoding="utf-8"))["planes"]
        P.append(float(np.nanmean([p["pearson"] if p.get("pearson_defined") else np.nan for p in pl])))
        S.append(float(np.mean([p["ssim"] for p in pl])))
    return P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "train", "eval"))
    ap.add_argument("--corpus")
    ap.add_argument("--wc-archive")
    ap.add_argument("--wc-mapping")
    ap.add_argument("--orgsets")
    ap.add_argument("--evalsets")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr-schedule", choices=("const", "cosine"), default="const")
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="*")
    a = ap.parse_intermixed_args()
    {"prep": prep, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
