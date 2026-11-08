"""R23 Fourier/Gabor phasor head — PRECHECK ONLY (TRAIN-only, <=100 updates per run). RESEARCH ONLY.

One factor vs R8-SCRATCH (research/native_b_sampler/phasor.py PhasorStemUNet, from scratch):
  R8-SCRATCH : phi = wrap( head(x) + atan2(im, re) ),   (re, im) = Conv1x1([x ; bp8])
  R23        : phi = atan2(im, re)   (scalar head REMOVED),
               (re, im) = Conv1x1([x ; bp8])                       local term, same init as R8-SCRATCH
                        + SpectralConv(x, 32x32 modes)              learnable global Fourier mixing of decoder features
                        + sum_k a_k(x) * exp(i k.r)                 K learnable carriers (k up to Nyquist),
                                                                    per-pixel target-conditioned complex amplitudes
Backbone (STEM 12ch W16 depth3), seed, sampler, R3 objective, lr, batch are identical to R8-SCRATCH.
Still ONE forward, phase-only, no optimiser at inference. TEST is never decoded (pilot.native refuses it).

  python r23_fourier_phasor.py precheck --corpus C --out NEWDIR [--seed 0] [--updates 100]
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
sys.path.insert(0, str(ROOT / "research" / "native_b_sampler"))
sys.path.insert(0, str(ROOT / "research" / "wc_stem_pilot"))
import phasor as r8  # noqa: E402  (installs pilot / tm / tw)
from pilot import tm, tw, CaseConfigV1, native  # noqa: E402
from back_projection_stem.stem import backprojection_features  # noqa: E402


class SpectralConv(torch.nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        s = 1.0 / (cin * cout)
        self.w_pos = torch.nn.Parameter(s * torch.randn(cin, cout, modes, modes, 2))
        self.w_neg = torch.nn.Parameter(s * torch.randn(cin, cout, modes, modes, 2))

    def forward(self, x):
        h, w = x.shape[-2:]
        m = self.modes
        X = torch.fft.rfft2(x.float(), norm="ortho")
        out = torch.zeros(x.shape[0], self.w_pos.shape[1], h, w // 2 + 1, dtype=X.dtype, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", X[:, :, :m, :m], torch.view_as_complex(self.w_pos))
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", X[:, :, -m:, :m], torch.view_as_complex(self.w_neg))
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class CarrierBank(torch.nn.Module):
    def __init__(self, cin, k=16, seed=12345):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k = torch.nn.Parameter((torch.rand(k, 2, generator=g) * 2 - 1) * math.pi)   # rad/px, up to Nyquist
        self.coef = torch.nn.Conv2d(cin, 2 * k, 1)
        with torch.no_grad():
            self.coef.weight.normal_(0, 0.01, generator=g)
            self.coef.bias.zero_()

    def forward(self, x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
        ang = self.k[:, 0, None, None] * xx + self.k[:, 1, None, None] * yy          # [K,H,W]
        c, s = torch.cos(ang), torch.sin(ang)
        a = self.coef(x)
        ar, ai = a[:, 0::2], a[:, 1::2]                                                 # [B,K,H,W]
        re = (ar * c - ai * s).sum(1, keepdim=True)
        im = (ar * s + ai * c).sum(1, keepdim=True)
        return torch.cat([re, im], dim=1)


class FourierPhasorStemUNet(r8.PhasorStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None, modes=32, carriers=16):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        c = self.head.in_channels
        self.spectral = SpectralConv(c, 2, modes)
        self.carriers = CarrierBank(c, carriers)

    @property
    def hparams(self):
        hp = dict(super().hparams)
        hp["phasor_head"] = "r23_fourier_phasor_v0"
        return hp

    def _decoder(self, targets):
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
        return x[..., :h, :w], feats

    def forward(self, targets):
        self._check(targets)
        x, feats = self._decoder(targets)
        ri = self.phasor(torch.cat([x, feats[:, 4:]], dim=1)) + self.spectral(x) + self.carriers(x)
        return torch.atan2(ri[:, 1:2], ri[:, 0:1])


def build(arm, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cls = r8.PhasorStemUNet if arm == "r8_scratch" else FourierPhasorStemUNet
        model = cls(base_channels=16, depth=3)
        # backbone + local phasor drawn first from the same seed -> identical to R8-SCRATCH for shared params
        torch.nn.init.normal_(model.phasor.weight, std=0.5)
        torch.nn.init.zeros_(model.phasor.bias)
    return model


def plane_pearson(amp, tgt):
    a = amp.reshape(amp.shape[0], 4, -1).double()
    t = tgt.reshape(tgt.shape[0], 4, -1).double()
    a = a - a.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    return (a * t).sum(-1) / (a.norm(dim=-1) * t.norm(dim=-1) + 1e-12)          # [B,4]


@torch.no_grad()
def phase_stats(phi):
    """HF content of the DOE phasor exp(i phi) and its target specificity."""
    u = torch.exp(1j * phi[:, 0].double())
    n = u.shape[-1]
    P = torch.fft.fftshift(torch.fft.fft2(u, norm="ortho"), dim=(-2, -1)).abs() ** 2
    f = torch.fft.fftshift(torch.fft.fftfreq(n, device=phi.device)).double()
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    hf = (P * (r > 0.25)).sum((-2, -1)) / P.sum((-2, -1))                          # energy beyond 0.25 cyc/px
    dx = torch.angle(u[:, :, 1:] * u[:, :, :-1].conj()).abs().mean((-2, -1))
    dy = torch.angle(u[:, 1:, :] * u[:, :-1, :].conj()).abs().mean((-2, -1))
    U = torch.fft.fft2(u, norm="ortho") * (torch.fft.fftfreq(n, device=phi.device).abs()[:, None].max(
        torch.fft.fftfreq(n, device=phi.device).abs()[None, :]) > 0.25)
    v = U.reshape(U.shape[0], -1)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    g = (v @ v.conj().T).abs()
    off = g[~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)]
    return {"hf_energy_frac": hf.mean().item(), "grad_rad_per_px": (0.5 * (dx + dy)).mean().item(),
            "hf_cross_sample_coherence": off.mean().item() if off.numel() else float("nan")}


def run(arm, mode, T, b_idx, seed, updates, dev):
    model = build(arm, seed).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    fixed = T[b_idx]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    hist, t0 = [], time.perf_counter()
    for step in range(1, updates + 1):
        batch = fixed if mode == "overfit_trainB4" else T[torch.randperm(512, generator=gen)[:4].to(dev)]
        opt.zero_grad(set_to_none=True)
        phi = model(batch)
        amp = tm.multiplane_amplitudes(phi, cfg, prop)
        loss, _ = tm._objective(amp, batch, tw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"{arm}/{mode}: non-finite loss at {step}")
        loss.backward()
        gn = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
        if not math.isfinite(gn):
            raise RuntimeError(f"{arm}/{mode}: non-finite grad at {step}")
        opt.step()
        rec = {"step": step, "loss": v, "grad_norm": gn}
        if mode == "overfit_trainB4" and (step == 1 or step % 25 == 0):
            with torch.no_grad():
                rec["pearson_mean"] = plane_pearson(amp.detach(), batch).mean().item()
                rec.update(phase_stats(phi.detach()))
        hist.append(rec)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        phi = model(fixed)
        amp = tm.multiplane_amplitudes(phi, cfg, prop)
        final = {"pearson_per_plane_trainB4": plane_pearson(amp, fixed).mean(0).tolist(),
                 "pearson_mean_trainB4": plane_pearson(amp, fixed).mean().item(), **phase_stats(phi)}
    return {"arm": arm, "mode": mode, "params": nparams, "updates": updates, "elapsed_s": el,
            "s_per_update": el / updates, "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
            "grad_norm_max": max(h["grad_norm"] for h in hist), "final_eval_trainB4": final, "history": hist}


def precheck(a):
    if a.updates > 100:
        raise SystemExit("precheck is limited to <=100 updates until SCIENCE_GATE=GO")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus), splits=("train",))
    tr = data["train"]
    if len(tr["ids"]) != 512:
        raise SystemExit("TRAIN is not 512")
    T = tr["targets"].to(dev)
    b_all = [i for i, k in enumerate(tr["kinds"]) if str(k).upper().startswith("B")]
    b_idx = torch.tensor(b_all[:4], device=dev)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    results = []
    for mode in ("overfit_trainB4", "smoke_sampler"):
        for arm in ("r8_scratch", "r23"):
            r = run(arm, mode, T, b_idx, a.seed, a.updates, dev)
            print(json.dumps({k: v for k, v in r.items() if k != "history"}), flush=True)
            results.append(r)
    report = {"experiment_id": "R23-FOURIER-PHASOR-PRECHECK", "addendum": "NIH-DIAG-CAPACITY-015",
              "host": "DESKTOP-0GHPCI9", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "seed": a.seed, "updates_per_run": a.updates, "native_manifest_sha256": msha,
              "train_kinds": sorted(set(map(str, tr["kinds"]))), "trainB4_ids": [tr["ids"][i] for i in b_all[:4]],
              "code_commit": commit, "git_dirty": dirty, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "splits_decoded": ["train"], "results": results}
    (out / "precheck_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


_prev_load = tw.research_load_checkpoint   # phasor.load_any (handles R8 heads)


def load_r23(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    hp = payload.get("hparams") if isinstance(payload, dict) else None
    if isinstance(hp, dict) and hp.get("phasor_head") == "r23_fourier_phasor_v0":
        saved = tm.FourInputUNet
        tm.FourInputUNet = FourierPhasorStemUNet
        try:
            return tm._stock_load_for_phasor(path)
        finally:
            tm.FourInputUNet = saved
    return _prev_load(path)


tw.research_load_checkpoint = load_r23


def full(a):
    """OWNER-AUTHORISED full run (no SCIENCE_GATE=GO at launch): both arms 12k, seed, const lr, then official VAL16."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, msha = native(Path(a.corpus), splits=("train", "val"))
    T = data["train"]["targets"].to(dev)
    if len(data["train"]["ids"]) != 512 or len(data["val"]["ids"]) != 16:
        raise SystemExit("unexpected split sizes")
    cfg, prop = CaseConfigV1(), tm.production_components().propagate
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    meta = {"experiment_id": "R23-FULL-OWNER", "science_gate_at_launch": "PENDING (owner-authorised)", "seed": a.seed,
            "updates": a.updates, "lr": 1e-3, "batch": 4, "objective": tw.R3_NAME, "sampler": "randperm(512,gen(seed))[:4]",
            "native_manifest_sha256": msha, "code_commit": commit, "git_dirty": dirty,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "arms": {}}
    ckpts = []
    for arm in ("r8_scratch", "r23"):
        model = build(arm, a.seed).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(a.seed)
        ck = out / arm / "checkpoints"
        ck.mkdir(parents=True)
        hist, t0 = [], time.perf_counter()
        model.train()
        for step in range(1, a.updates + 1):
            batch = T[torch.randperm(512, generator=gen)[:4].to(dev)]
            opt.zero_grad(set_to_none=True)
            loss, _ = tm._objective(tm.multiplane_amplitudes(model(batch), cfg, prop), batch, tw.R3_NAME)
            v = float(loss.detach())
            if not math.isfinite(v):
                raise RuntimeError(f"{arm}: non-finite loss at {step}")
            loss.backward()
            opt.step()
            if step % 100 == 0:
                hist.append({"step": step, "loss": v})
            if step % 1000 == 0:
                print(json.dumps({"arm": arm, "step": step, "loss": v, "t": round(time.perf_counter() - t0, 1)}), flush=True)
            if step % 4000 == 0 or step == a.updates:
                p = ck / f"step_{step:06d}.pt"
                tm.save_checkpoint(p, model, config=cfg, code_commit=commit,
                                   train_log={"partial": step < a.updates, "step": step, "seed": a.seed},
                                   extra={"kind": "periodic", "step": step, "resumable": False, "research_only": True,
                                          "native_only": True, "experiment_id": f"R23-FULL-OWNER/{arm}"})
                ckpts.append((arm, step, p))
        meta["arms"][arm] = {"elapsed_s": time.perf_counter() - t0, "hparams": model.hparams, "history": hist}
        del model, opt
        torch.cuda.empty_cache()
    (out / "run_config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # official VAL16 path (DOE u8 save/reload -> 4 amplitude PNG save/reload -> Pearson + global SSIM)
    import numpy as np
    from pilot import per_quartet
    comps = tm.production_components()
    tm.load_checkpoint = tw.research_load_checkpoint
    part = data["val"]
    rows = []
    for arm, step, p in ckpts:
        c = out / "eval_ckpts" / f"{arm}__step_{step:06d}.pt"
        c.parent.mkdir(exist_ok=True)
        c.write_bytes(p.read_bytes())
        d = {"splits": {"val": part}, "manifest_sha256": msha, "synthetic": True}
        s = tm.evaluate_checkpoint(c, d, comps, out / "val16", dev, CaseConfigV1(), ("val",))["val"]
        P, S = per_quartet(out / "val16" / "export" / c.name / "val")
        row = {"arm": arm, "step": step, "sha256": hashlib.sha256(c.read_bytes()).hexdigest(),
               "pearson": s["pearson_mean"], "ssim": s["ssim_mean"], "per_quartet_pearson": P, "per_quartet_ssim": S,
               "ids": part["ids"]}
        for g in ("A", "B"):
            ii = [i for i, k in enumerate(part["kinds"]) if k == g]
            row[g] = {"pearson": float(np.mean([P[i] for i in ii])), "ssim": float(np.mean([S[i] for i in ii]))}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("arm", "step", "pearson", "ssim", "A", "B")}), flush=True)
    (out / "eval_val16.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("precheck", "full"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=100)
    a = ap.parse_args()
    {"precheck": precheck, "full": full}[a.cmd](a)


if __name__ == "__main__":
    main()
