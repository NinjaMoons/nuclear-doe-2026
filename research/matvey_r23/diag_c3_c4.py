"""NIH-DIAG-C3-OVERFIT (C3/C4 of ROOT_CAUSE_DECISION): expressivity vs multi-modal ambiguity. TRAIN only, diagnostic.

N1: R8-SCRATCH (fresh STEM, R3, lr 1e-3, seed 0) overfit on B64:000 alone, 3000 updates.
N64: same arm on first 64 TRAIN-B (manifest order), batch 4, 3000 updates.
ORACLE: per-quartet Adam on phase of B64:000, R3, lr .05, 1000 steps, from (i) random phase, (ii) N1 output.
Metrics on B64:000 after DOE u8 quantisation + 4 amplitude u8 quantisation: P, global SSIM (L=1), own 4x4,
mask HF energy fraction (r > .5 Nyquist of FFT(e^{i phi})), |grad phi|. C4: swap HF band between N64(B64:000)
and N64(B64:001), report dP on own quartet.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r23_fourier_phasor as r23  # noqa: E402
from r23_fourier_phasor import tm, tw, CaseConfigV1, native, build  # noqa: E402

ROOT = r23.ROOT
dev = torch.device("cuda")
cfg, prop = CaseConfigV1(), tm.production_components().propagate


def quant_doe(phi):
    v = torch.round((torch.remainder(phi + math.pi, 2 * math.pi)) / (2 * math.pi) * 255).clamp(0, 255)
    return v / 255 * 2 * math.pi - math.pi


def recon_u8(phi):
    amp = tm.multiplane_amplitudes(quant_doe(phi), cfg, prop).reshape(phi.shape[0], 4, -1)
    amp = amp / amp.amax(-1, keepdim=True).clamp_min(1e-12)
    return torch.round(amp * 255) / 255


def gssim(x, y):
    mx, my = x.mean(-1), y.mean(-1)
    vx, vy = x.var(-1, unbiased=False), y.var(-1, unbiased=False)
    cxy = ((x - mx[..., None]) * (y - my[..., None])).mean(-1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


def corr(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


def hf_mask(n):
    f = torch.fft.fftfreq(n, device=dev).abs()
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return r > 0.25          # > .5 Nyquist (Nyquist = .5 cyc/px)


@torch.no_grad()
def metrics(phi, tgt):
    phi = phi.detach().double()
    rec = recon_u8(phi.float()).double()
    t = tgt.reshape(tgt.shape[0], 4, -1).double()
    t = t / t.amax(-1, keepdim=True).clamp_min(1e-12)
    P = corr(rec, t)[0]
    S = gssim(rec, t)[0]
    M = corr(rec[0][:, None, :], t[0][None, :, :])                          # [4 rec, 4 tgt]
    own = int((M.argmax(1) == torch.arange(4, device=dev)).sum())
    u = torch.exp(1j * phi[0, 0])
    E = torch.fft.fft2(u, norm="ortho").abs() ** 2
    hf = float((E * hf_mask(u.shape[-1])).sum() / E.sum())
    g = 0.5 * (torch.angle(u[:, 1:] * u[:, :-1].conj()).abs().mean() + torch.angle(u[1:] * u[:-1].conj()).abs().mean())
    return {"pearson": float(P.mean()), "pearson_planes": P.tolist(), "ssim": float(S.mean()), "own_4x4": own,
            "own_matrix": M.tolist(), "hf_frac_gt_half_nyq": hf, "grad_rad_per_px": float(g)}


def train_net(T, pool, batch, updates, seed=0):
    model = build("r8_scratch", seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    hist = []
    for step in range(1, updates + 1):
        idx = pool if len(pool) <= batch else pool[torch.randperm(len(pool), generator=gen)[:batch]]
        b = T[idx]
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(model(b), cfg, prop), b, tw.R3_NAME)
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss at {step}")
        loss.backward()
        opt.step()
        if step % 500 == 0:
            hist.append({"step": step, "loss": float(loss)})
    model.eval()
    return model, hist


def oracle(tgt, init, steps=1000, lr=0.05):
    phi = torch.nn.Parameter(init.detach().clone().float())
    opt = torch.optim.Adam([phi], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = tm._objective(tm.multiplane_amplitudes(phi, cfg, prop), tgt, tw.R3_NAME)
        loss.backward()
        opt.step()
    return phi.detach(), float(loss)


def swap_hf(pa, pb):
    m = hf_mask(pa.shape[-1])
    A, B = torch.fft.fft2(torch.exp(1j * pa.double())), torch.fft.fft2(torch.exp(1j * pb.double()))
    A2, B2 = torch.where(m, B, A), torch.where(m, A, B)
    return torch.angle(torch.fft.ifft2(A2)).float(), torch.angle(torch.fft.ifft2(B2)).float()


def main():
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.perf_counter()
    data, msha = native(Path(sys.argv[1]), splits=("train",))
    tr = data["train"]
    T = tr["targets"].to(dev)
    ids = tr["ids"]
    i0, i1 = ids.index("B64:000"), ids.index("B64:001")
    bidx = [i for i, k in enumerate(tr["kinds"]) if k == "B"][:64]
    if i0 not in bidx or i1 not in bidx:
        raise SystemExit("B64:000/001 not in first 64 TRAIN-B")
    t0q, t1q = T[i0:i0 + 1], T[i1:i1 + 1]
    res = {"task_id": "NIH-DIAG-C3-OVERFIT", "host": "DESKTOP-0GHPCI9", "native_manifest_sha256": msha,
           "splits_decoded": ["train"], "n64_ids_first_last": [ids[bidx[0]], ids[bidx[-1]]],
           "code_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    n1, h1 = train_net(T, torch.tensor([i0], device=dev), 1, 3000)
    with torch.no_grad():
        p_n1 = n1(t0q)
    res["N1"] = {**metrics(p_n1, t0q), "loss_hist": h1}
    print("N1", {k: res["N1"][k] for k in ("pearson", "ssim", "own_4x4", "hf_frac_gt_half_nyq")}, flush=True)
    n64, h64 = train_net(T, torch.tensor(bidx, device=dev), 4, 3000)
    with torch.no_grad():
        p64_0, p64_1 = n64(t0q), n64(t1q)
    res["N64_on_B64_000"] = {**metrics(p64_0, t0q), "loss_hist": h64}
    res["N64_on_B64_001"] = metrics(p64_1, t1q)
    print("N64", {k: res["N64_on_B64_000"][k] for k in ("pearson", "ssim", "own_4x4", "hf_frac_gt_half_nyq")}, flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    rnd = ((torch.rand(1, 1, *t0q.shape[-2:], generator=g) * 2 - 1) * math.pi).to(dev)
    po_r, lr_ = oracle(t0q, rnd)
    po_n, ln_ = oracle(t0q, p_n1)
    res["ORACLE_random"] = {**metrics(po_r, t0q), "final_r3": lr_}
    res["ORACLE_from_N1"] = {**metrics(po_n, t0q), "final_r3": ln_}
    print("ORACLE", res["ORACLE_random"]["pearson"], res["ORACLE_from_N1"]["pearson"], flush=True)
    s0, s1 = swap_hf(p64_0, p64_1)
    m0, m1 = metrics(s0, t0q), metrics(s1, t1q)
    res["C4"] = {"B64_000": {"P_orig": res["N64_on_B64_000"]["pearson"], "P_swapped": m0["pearson"],
                             "dP": m0["pearson"] - res["N64_on_B64_000"]["pearson"]},
                 "B64_001": {"P_orig": res["N64_on_B64_001"]["pearson"], "P_swapped": m1["pearson"],
                             "dP": m1["pearson"] - res["N64_on_B64_001"]["pearson"]}}
    oracle_best = max(res["ORACLE_random"]["pearson"], res["ORACLE_from_N1"]["pearson"])
    pn1, pn64, hn1 = res["N1"]["pearson"], res["N64_on_B64_000"]["pearson"], res["N1"]["hf_frac_gt_half_nyq"]
    if pn1 <= oracle_best - 0.05 and hn1 <= 0.3:
        verdict = "EXPRESSIVITY"
    elif pn1 >= oracle_best - 0.02 and pn64 <= pn1 - 0.03:
        verdict = "AMBIGUITY"
    else:
        verdict = "INCONCLUSIVE"
    res["oracle_best_pearson"] = oracle_best
    res["VERDICT"] = verdict
    res["C4_coherent_HF"] = res["C4"]["B64_000"]["dP"] <= -0.02
    res["elapsed_s"] = time.perf_counter() - t0
    (out / "diag_c3_c4.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("VERDICT", verdict, "C4", res["C4"], "elapsed", round(res["elapsed_s"], 1), flush=True)


if __name__ == "__main__":
    main()
