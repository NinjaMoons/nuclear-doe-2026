"""R8-RES phasor-residual head on top of V2 (RESEARCH ONLY; executes the unrun R8 card #35 in an init-safe residual form).

phi = wrap( head(x) + atan2(im, re) ),  (re, im) = Conv1x1([decoder features x (16ch) ; 8 back-projection Re/Im
channels at full resolution]). Phasor conv weights 0, bias (re=1, im=0) -> atan2 = 0 -> exactly V2 at step 0.
Still ONE forward, phase-only (|U| = 1 through the official exp(i phi)), no optimiser at inference. TEST never decoded.

  python phasor.py train --corpus C --init V2.pt --seed S --updates 6000 --out NEW
  python phasor.py eval  --corpus C --evalsets E --out NEW CKPT [CKPT ...]
Matched control = R17 `control_s{0,1}` (same init V2, sampler randperm(512,gen(30000+seed))[:4], R3, lr 1e-3, 6000).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "wc_stem_pilot"))
import pilot  # noqa: E402
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402
from back_projection_stem.stem import BackProjectionStemUNet, backprojection_features  # noqa: E402
from doe_models.unet_multiplane import _wrap  # noqa: E402

INIT_SHA = "19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f"


class PhasorStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels: int = 16, depth: int = 3, config=None):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self.phasor = torch.nn.Conv2d(self.head.in_channels + 8, 2, 1)
        with torch.no_grad():
            self.phasor.weight.zero_()
            self.phasor.bias.copy_(torch.tensor([1.0, 0.0]))

    @property
    def hparams(self):
        hp = dict(super().hparams)
        hp["phasor_head"] = "residual_atan2_v0"
        return hp

    def forward(self, targets):
        self._check(targets)
        feats = backprojection_features(targets, self._config)
        h, w = targets.shape[-2:]
        x = feats
        m = 2 ** self.depth
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for red, dec in zip(self.reducers, self.decoders):
            skip = skips.pop()
            x = red(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            x = dec(torch.cat([x, skip], dim=1))
        x = x[..., :h, :w]
        ri = self.phasor(torch.cat([x, feats[:, 4:]], dim=1))
        return _wrap(self.head(x) + torch.atan2(ri[:, 1:2], ri[:, 0:1]))


_stock_research_load = tw.research_load_checkpoint


def load_any(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    hp = payload.get("hparams") if isinstance(payload, dict) else None
    if isinstance(hp, dict) and hp.get("phasor_head") == "residual_atan2_v0":
        saved = tm.FourInputUNet
        tm.FourInputUNet = PhasorStemUNet
        try:
            return tm._stock_load_for_phasor(path)
        finally:
            tm.FourInputUNet = saved
    return _stock_research_load(path)


tm._stock_load_for_phasor = tw._stock_load
tw.research_load_checkpoint = load_any   # pilot.evaluate installs tw.research_load_checkpoint


def sha(b):
    return hashlib.sha256(b).hexdigest()


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if sha(Path(a.init).read_bytes()) != INIT_SHA:
        raise SystemExit("init sha mismatch")
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus), splits=("train",))
    tr = data["train"]
    if len(tr["ids"]) != 512:
        raise SystemExit("TRAIN is not 512")
    T = tr["targets"].to(dev)
    v2, _ = _stock_research_load(a.init)
    hp = v2.hparams
    model = PhasorStemUNet(base_channels=hp["base_channels"], depth=hp["depth"])
    missing, unexpected = model.load_state_dict(v2.state_dict(), strict=False)
    if unexpected or sorted(missing) != ["phasor.bias", "phasor.weight"]:
        raise SystemExit(f"unexpected init mismatch {missing} {unexpected}")
    model = model.to(dev)
    v2 = v2.to(dev)
    with torch.no_grad():
        d0 = float((model(T[:16]) - v2(T[:16])).abs().max())
    if d0 > 1e-5:
        raise SystemExit(f"step-0 phase differs from V2 by {d0}")
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(30000 + a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run_config.json").write_text(json.dumps({
        "experiment_id": "R8-RES", "seed": a.seed, "updates": a.updates, "init_sha256": INIT_SHA, "lr": 1e-3,
        "batch": 4, "objective": tw.R3_NAME, "sampler": "randperm(512,gen(30000+seed))[:4]", "hparams": model.hparams,
        "step0_max_abs_phase_diff_vs_v2": d0, "native_manifest_sha256": msha, "code_commit": commit,
        "git_dirty": dirty, "script_sha256": sha(Path(__file__).read_bytes())}, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        idx = torch.randperm(512, generator=gen)[:4].to(dev)
        batch = T[idx]
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(model(batch), cfg, prop), batch, tw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        gn = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
        if not math.isfinite(gn):
            raise RuntimeError(f"non-finite grad at {step}")
        opt.step()
        hist.append({"step": step, "loss": v, "grad_norm": gn})
        if step % 2000 == 0 or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "seed": a.seed},
                               extra={"kind": "periodic", "step": step, "resumable": False, "research_only": True,
                                      "native_only": True, "experiment_id": "R8-RES"})
    log = {"elapsed_s": time.perf_counter() - t0, "history": hist,
           "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}))


def scratch(a):
    """R8-SCRATCH: PhasorStemUNet from random init (seed), phasor weights N(0, .5), bias 0 (singularities present
    from step 0); sampler/steps identical to pilot.py native arm: randperm(512, gen(seed))[:4], R3, lr 1e-3, batch 4."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus), splits=("train",))
    T = data["train"]["targets"].to(dev)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(a.seed)
        model = PhasorStemUNet(base_channels=a.base_channels, depth=3)
        torch.nn.init.normal_(model.phasor.weight, std=0.5)
        torch.nn.init.zeros_(model.phasor.bias)
    model = model.to(dev)
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(a.seed)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run_config.json").write_text(json.dumps({
        "experiment_id": "R8-SCRATCH", "seed": a.seed, "updates": a.updates, "lr": 1e-3, "batch": 4,
        "objective": tw.R3_NAME, "sampler": "randperm(512,gen(seed))[:4] (= pilot native)", "hparams": model.hparams,
        "phasor_init": "weight N(0,.5), bias 0", "lr_schedule": a.lr_schedule, "native_manifest_sha256": msha, "code_commit": commit,
        "git_dirty": dirty, "script_sha256": sha(Path(__file__).read_bytes())}, indent=1), encoding="utf-8")
    ck = out / "checkpoints"
    ck.mkdir()
    hist, t0 = [], time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        if a.lr_schedule == "cosine":
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (step - 1) / a.updates))
        idx = torch.randperm(512, generator=gen)[:4].to(dev)
        batch = T[idx]
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(model(batch), cfg, prop), batch, tw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        gn = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
        if not math.isfinite(gn):
            raise RuntimeError(f"non-finite grad at {step}")
        opt.step()
        hist.append({"step": step, "loss": v, "grad_norm": gn})
        if step % 4000 == 0 or step == a.updates:
            tm.save_checkpoint(ck / f"step_{step:06d}.pt", model, config=cfg, code_commit=commit,
                               train_log={"partial": True, "step": step, "seed": a.seed},
                               extra={"kind": "periodic", "step": step, "resumable": False, "research_only": True,
                                      "native_only": True, "experiment_id": "R8-SCRATCH"})
    log = {"elapsed_s": time.perf_counter() - t0, "history": hist,
           "checkpoints": {p.name: sha(p.read_bytes()) for p in sorted(ck.glob("*.pt"))}}
    (out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval", "scratch"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init")
    ap.add_argument("--evalsets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--lr-schedule", choices=("const", "cosine"), default="const")
    ap.add_argument("ckpts", nargs="*")
    a = ap.parse_intermixed_args()
    {"train": train, "eval": pilot.evaluate, "scratch": scratch}[a.cmd](a)


if __name__ == "__main__":
    main()
