"""R17-BRECOMB (IDEA_CARD @ggmaster913 f9239c5, #49/6712871198): train-only slot-preserving recombination of
B quartets from the INDEPENDENT TRAIN pool. RESEARCH ONLY, one factor, native TRAIN512 only, TEST never decoded.

  python r17_recomb.py train --corpus C --init CKPT --arm {control,recomb} --seed S --updates 6000 --out NEW
Both arms: init V2 19b05dad, fresh Adam lr 1e-3, batch 4, R3, idx = randperm(512, gen(30000+seed))[:4] (the same
sequence in both arms). recomb: every drawn B quartet is, with prob 0.5 (private gen 50000+seed), replaced by
plane k = slot k of 4 DIFFERENT quartets drawn uniformly from pool = TRAIN B64 U TRAIN B208[relation=independent].
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
import pilot  # noqa: E402
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402

INIT_SHA = "19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pool_indices(corpus: Path, train_ids):
    man = json.loads((corpus / "manifest_expanded.json").read_bytes())
    by_key = {s["key"]: s for s in man["samples"]}
    pool = []
    for i, k in enumerate(train_ids):
        s = by_key[k]
        if s["split"] != "train" or s["kind"] != "B":
            continue
        if s["corpus_id"] == "B64" or (s["corpus_id"] == "B208" and (s.get("procedural") or {}).get("relation") == "independent"):
            pool.append(i)
    held = {h for s in man["samples"] if s["split"] in ("val", "test") for h in s["sha256"]}
    leak = [k for i in pool for h in by_key[train_ids[i]]["sha256"] if h in held]
    if leak:
        raise SystemExit(f"pool plane sha found in VAL/TEST: {leak[:3]}")
    return pool


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if sha(Path(a.init).read_bytes()) != INIT_SHA:
        raise SystemExit("init sha mismatch")
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    corpus = Path(a.corpus)
    data, msha = native(corpus, splits=("train",))
    tr = data["train"]
    if len(tr["ids"]) != 512:
        raise SystemExit("TRAIN is not 512")
    kinds = tr["kinds"]
    pool = pool_indices(corpus, tr["ids"])
    if len(pool) != 118:
        raise SystemExit(f"independent pool is {len(pool)}, expected 118")
    pool_t = torch.tensor(pool)
    T = tr["targets"].to(dev)
    model, payload = tw.research_load_checkpoint(a.init)
    model = model.to(dev)
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen, gen_r = torch.Generator().manual_seed(30000 + a.seed), torch.Generator().manual_seed(50000 + a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run_config.json").write_text(json.dumps({
        "experiment_id": "R17-BRECOMB", "arm": a.arm, "seed": a.seed, "updates": a.updates, "init_sha256": INIT_SHA,
        "lr": 1e-3, "batch": 4, "objective": tw.R3_NAME, "sampler": "randperm(512,gen(30000+seed))[:4]",
        "recomb": "B -> p=.5 slot-preserving from 4 distinct pool quartets, gen(50000+seed)" if a.arm == "recomb" else None,
        "pool_n": len(pool), "pool_keys_sha256": sha("\n".join(tr["ids"][i] for i in pool).encode()),
        "native_manifest_sha256": msha, "code_commit": commit, "git_dirty": dirty,
        "script_sha256": sha(Path(__file__).read_bytes())}, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist, n_recomb, n_b = [], 0, 0
    t0 = time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        idx = torch.randperm(512, generator=gen)[:4]
        batch = T[idx.to(dev)].clone()
        if a.arm == "recomb":
            for j, i in enumerate(idx.tolist()):
                if kinds[i] != "B":
                    continue
                n_b += 1
                if float(torch.rand(1, generator=gen_r)) < 0.5:
                    donors = pool_t[torch.randperm(len(pool), generator=gen_r)[:4]].tolist()
                    for k, d in enumerate(donors):
                        batch[j, k] = T[d, k]
                    n_recomb += 1
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(model(batch), cfg, prop), batch, tw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        opt.step()
        hist.append({"step": step, "loss": v})
        if step % 2000 == 0 or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "arm": a.arm, "seed": a.seed},
                               extra={"kind": "periodic", "step": step, "resumable": False, "research_only": True,
                                      "native_only": True, "experiment_id": "R17-BRECOMB", "arm": a.arm})
    log = {"elapsed_s": time.perf_counter() - t0, "b_drawn": n_b, "b_recombined": n_recomb, "history": hist,
           "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train",))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--arm", choices=("control", "recomb"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
