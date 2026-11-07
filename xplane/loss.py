"""RESEARCH ONLY (R19): band-limited hard-negative cross-plane contrastive loss (TRAIN-only training term).
For plane k: c_kk = corr(BP(D_k), BP(T_k)), c_kj = corr(BP(D_k), BP(N_kj)) with D_k = A_k / max(A_k) (contract display),
N_kj = T_j resampled onto plane-k physical grid (scale z_k/z_j about the centre index, Test 2: ghosts follow physical
scaling), BP = radial FFT band 0.15 < r <= 0.5 Nyquist (mid band: teacher separates, network does not; low band left
alone because even direct optimisation cannot separate it). Pairs whose TARGETS correlate (|corr(BP T_k, BP N_kj)| >= 0.3,
e.g. group A) are excluded. InfoNCE with temperature tau: L_k = -log(e^{c_kk/tau} / (e^{c_kk/tau} + sum_j m_kj e^{c_kj/tau})).
Pearson is scale invariant -> uniform dimming is not rewarded; R3 keeps the amplitude terms."""
import math
import torch
import torch.nn.functional as F
from doe_core.contracts_v1 import CaseConfigV1

N = 128; LO, HI = 0.15, 0.5; TAU = 0.1; PAIR_MAX = 0.3; EPS = 1e-8


def _band_mask(device, dtype):
    f = torch.fft.fftfreq(N, device=device, dtype=torch.float64) / 0.5
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return ((r > LO) & (r <= HI)).to(dtype)


def bandpass(x):
    m = _band_mask(x.device, torch.float64)
    return torch.fft.ifft2(torch.fft.fft2(x.double()) * m).real


def _scale(img, s):
    """img [B,H,W] sampled at centre + s*(p - centre) (bilinear, zeros outside)."""
    n = torch.arange(N, dtype=torch.float64, device=img.device) - N // 2
    g = (n * s + N // 2) / (N - 1) * 2 - 1
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(img.shape[0], -1, -1, -1)
    return F.grid_sample(img.double()[:, None], grid, mode="bilinear", align_corners=True, padding_mode="zeros")[:, 0]


def _corr(a, b):
    a = a - a.mean(dim=(-2, -1), keepdim=True); b = b - b.mean(dim=(-2, -1), keepdim=True)
    return (a * b).sum(dim=(-2, -1)) / torch.sqrt((a * a).sum(dim=(-2, -1)) * (b * b).sum(dim=(-2, -1)) + EPS)


def xplane_contrast(amplitudes, targets, config=None, return_stats=False):
    """amplitudes, targets [B,4,1,H,W] -> scalar loss (mean over planes with >= 1 valid negative)."""
    cfg = config or CaseConfigV1(); z = [cfg.plane(k).distance_m for k in range(4)]
    a = amplitudes[:, :, 0].double(); t = targets[:, :, 0].double()
    d = a / a.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    bd = bandpass(d); bt = bandpass(t)
    losses, cpos, cneg = [], [], []
    for k in range(4):
        pos = _corr(bd[:, k], bt[:, k])
        logits, masks = [pos / TAU], [torch.ones_like(pos)]
        for j in range(4):
            if j == k:
                continue
            nk = bandpass(_scale(t[:, j], z[k] / z[j]))
            pair = _corr(bt[:, k], nk).detach()
            m = (pair.abs() < PAIR_MAX).double()
            c = _corr(bd[:, k], nk)
            logits.append(c / TAU); masks.append(m); cneg.append((c * m).detach())
        L = torch.stack(logits, 1); Mk = torch.stack(masks, 1)
        lse = torch.logsumexp(L + torch.log(Mk.clamp_min(1e-30)), dim=1)
        has_neg = (Mk[:, 1:].sum(1) > 0).double()
        losses.append(((lse - pos / TAU) * has_neg, has_neg)); cpos.append(pos.detach())
    num = sum(l for l, _ in losses).sum(); den = sum(h for _, h in losses).sum().clamp_min(1.0)
    loss = num / den
    if return_stats:
        return loss, {"c_pos": torch.stack(cpos, 1).mean().item(), "c_neg_mean": torch.stack(cneg, 1).mean().item()}
    return loss
