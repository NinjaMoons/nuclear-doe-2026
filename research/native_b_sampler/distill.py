"""Teacher-guided physics training (RESEARCH ONLY, #43 follow-up). One forward at inference — the teacher is used
ONLY during training on TRAIN quartets; VAL/TEST are never given to the teacher.

  teacher: python distill.py teacher --corpus C --init CKPT --steps 2000 --out T.pt
     per TRAIN quartet: Adam on the R3 objective started FROM THE STUDENT'S OWN PHASE (so the teacher stays on the
     student's phase branch; no global-offset / twin-image ambiguity), lr .02, best-objective state kept.
  train:   python distill.py train --corpus C --init CKPT --teacher T.pt --lam L --seed S --updates U --out NEW
     loss = R3(student) + L * mean(1 - cos(phi_student - phi_teacher)); L = 0 is the matched control.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "wc_stem_pilot"))
import pilot  # noqa: E402
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def teacher(a):
    dev = torch.device("cuda")
    data, msha = native(Path(a.corpus), splits=("train",))
    tr = data["train"]
    T = tr["targets"].to(dev)
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    model, payload = tw.research_load_checkpoint(a.init)
    model = model.to(dev)
    with torch.no_grad():
        phi0 = torch.cat([tm.infer_phase(model, T[i:i + 64]) for i in range(0, T.shape[0], 64)])
    def per_quartet(ph):
        with torch.no_grad():
            return torch.stack([tm._objective(tm.multiplane_amplitudes(ph[i:i + 1], cfg, prop), T[i:i + 1],
                                              tw.R3_NAME)[0] for i in range(T.shape[0])]).double()

    # R3 = mean over quartets/planes of per-plane terms -> N * batch objective has exactly the per-quartet gradients
    phase = phi0.clone().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=a.lr)
    n = T.shape[0]
    t0 = time.perf_counter()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss = sum(tm._objective(tm.multiplane_amplitudes(phase[i:i + 128], cfg, prop), T[i:i + 128], tw.R3_NAME)[0]
                   * min(128, n - i) for i in range(0, n, 128))
        loss.backward()
        opt.step()
    start, end = per_quartet(phi0), per_quartet(phase.detach())
    keep = end < start                                   # never worse than the student's own phase
    best_phase = torch.where(keep[:, None, None, None], phase.detach(), phi0)
    best = torch.where(keep, end, start)
    best_phase = torch.remainder(best_phase + math.pi, 2 * math.pi) - math.pi
    torch.save({"phase": best_phase.cpu(), "ids": tr["ids"], "init_sha256": payload["sha256"], "steps": a.steps,
                "lr": a.lr, "native_manifest_sha256": msha, "objective_start": start.cpu(), "objective_best": best.cpu()},
               a.out)
    print(json.dumps({"elapsed_s": time.perf_counter() - t0, "objective_start_mean": float(start.mean()),
                      "objective_teacher_mean": float(best.mean()), "n": len(tr["ids"])}))


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus), splits=("train",))
    tr = data["train"]
    tch = torch.load(a.teacher, weights_only=False)
    if tch["ids"] != tr["ids"] or tch["native_manifest_sha256"] != msha:
        raise SystemExit("teacher does not match native TRAIN")
    T, PT = tr["targets"].to(dev), tch["phase"].to(dev)
    model, payload = tw.research_load_checkpoint(a.init)
    model = model.to(dev)
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(30000 + a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run_config.json").write_text(json.dumps({
        "lam": a.lam, "seed": a.seed, "updates": a.updates, "lr": a.lr, "batch": 4, "init_sha256": payload["sha256"],
        "teacher_sha256": sha(Path(a.teacher).read_bytes()), "teacher_steps": tch["steps"], "objective": tw.R3_NAME,
        "distill": "mean(1-cos(phi_s-phi_t))", "sampler": "randperm(512,gen(30000+seed))[:4]", "hparams": model.hparams,
        "native_manifest_sha256": msha, "code_commit": commit, "git_dirty": dirty,
        "script_sha256": sha(Path(__file__).read_bytes())}, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        idx = torch.randperm(512, generator=gen)[:4].to(dev)
        batch = T[idx]
        opt.zero_grad(set_to_none=True)
        phi = model(batch)
        r3, _ = tm._objective(tm.multiplane_amplitudes(phi, cfg, prop), batch, tw.R3_NAME)
        dist = (1 - torch.cos(phi - PT[idx])).mean()
        loss = r3 + a.lam * dist
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        opt.step()
        hist.append({"step": step, "r3": float(r3), "dist": float(dist)})
        if step % a.ckpt_every == 0 or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "lam": a.lam, "seed": a.seed},
                               extra={"kind": "periodic", "step": step, "resumable": False, "research_only": True,
                                      "native_only": True, "teacher_train_only": True, "lam": a.lam})
    log = {"elapsed_s": time.perf_counter() - t0, "history": hist,
           "r3_dist_first_last50": [np.mean([h["r3"] for h in hist[:50]]), np.mean([h["dist"] for h in hist[:50]]),
                                    np.mean([h["r3"] for h in hist[-50:]]), np.mean([h["dist"] for h in hist[-50:]])],
           "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("teacher", "train"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--teacher")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "teacher":
        a.lr = 0.02 if a.lr is None else a.lr
        teacher(a)
    else:
        a.lr = 1e-3 if a.lr is None else a.lr
        train(a)


if __name__ == "__main__":
    main()
