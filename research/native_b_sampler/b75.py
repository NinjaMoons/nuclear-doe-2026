"""#43 native-only B-focused sampler fine-tune from a frozen STEM checkpoint (PLAN_B75.md). RESEARCH ONLY.

  python research/native_b_sampler/b75.py train --corpus C --init CKPT --arm {C50,B75} --seed S --updates U --out NEW
  python research/native_b_sampler/b75.py eval  --corpus C --evalsets E --out NEW CKPT [CKPT ...]   (= pilot eval)
TEST is never decoded (native() reads train/val only).
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "wc_stem_pilot"))
import pilot  # noqa: E402  (sets sys.path for the repo, registers R3)
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402

INIT_SHA = "19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f"
CKPTS = (100, 250, 500, 1000)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if sha(Path(a.init).read_bytes()) != INIT_SHA:
        raise SystemExit("init checkpoint sha mismatch")
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus))
    tr = data["train"]
    kinds = tr["kinds"]
    ia = [i for i, k in enumerate(kinds) if k == "A"]
    ib = [i for i, k in enumerate(kinds) if k == "B"]
    if len(tr["ids"]) != 512 or len(ia) != 256 or len(ib) != 256 or len(data["val"]["ids"]) != 16:
        raise SystemExit("native split is not 256A+256B / VAL16")
    ia_t, ib_t = torch.tensor(ia), torch.tensor(ib)
    train_t = tr["targets"].to(dev)
    model, payload = tw.research_load_checkpoint(a.init)
    model = model.to(dev)
    cfg = CaseConfigV1()
    prop = tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(20000 + a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run_config.json").write_text(json.dumps({
        "arm": a.arm, "seed": a.seed, "gen_seed": 20000 + a.seed, "updates": a.updates, "checkpoints": CKPTS,
        "init_sha256": INIT_SHA, "lr": 1e-3, "batch": 4, "optimizer": "fresh Adam", "objective": tw.R3_NAME,
        "native_manifest_sha256": msha, "train_ids_sha256": sha("\n".join(tr["ids"]).encode()), "val": data["val"]["ids"],
        "sampler": {"C50": "randperm(512,gen)[:4]", "B75": "A[randperm(256,gen)[:1]] + B[randperm(256,gen)[:3]]"}[a.arm],
        "code_commit": commit, "git_dirty": dirty, "script_sha256": sha(Path(__file__).read_bytes()),
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__}, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist, counts, reuse = [], {"A": 0, "B": 0}, {}
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        if a.arm == "C50":
            idx = torch.randperm(512, generator=gen)[:4]
        elif a.arm == "B75":
            idx = torch.cat([ia_t[torch.randperm(256, generator=gen)[:1]], ib_t[torch.randperm(256, generator=gen)[:3]]])
        else:
            raise SystemExit("arm must be C50 or B75")
        for i in idx.tolist():
            counts[kinds[i]] += 1
            reuse[i] = reuse.get(i, 0) + 1
        batch = train_t[idx.to(dev)]
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
        hist.append({"step": step, "loss": v, "grad_norm": gn})
        if step in CKPTS or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "arm": a.arm, "seed": a.seed, "init": INIT_SHA},
                               extra={"kind": "periodic", "step": step, "resumable": False, "arm": a.arm,
                                      "research_only": True, "init_sha256": INIT_SHA, "native_only": True})
    torch.cuda.synchronize()
    log = {"elapsed_s": time.perf_counter() - t0, "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)),
           "sample_counts": counts, "distinct_quartets_seen": len(reuse), "max_reuse": max(reuse.values()),
           "loss_first_last_mean50": [sum(h["loss"] for h in hist[:50]) / 50, sum(h["loss"] for h in hist[-50:]) / 50],
           "history": hist, "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init")
    ap.add_argument("--evalsets")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="*")
    a = ap.parse_intermixed_args()
    train(a) if a.cmd == "train" else pilot.evaluate(a)


if __name__ == "__main__":
    main()
