"""NIH-DIAG-GRADIENT-014: per-plane R3 loss decomposition and d(L_plane_i)/d(phi) diagnostics for frozen V2.

Analysis only: model in eval mode, no optimizer, weights never updated; autograd only w.r.t. the continuous
phase phi = V2(targets) detached as a leaf. TRAIN-B quartets only (pre-registered B208:006, B64:010 from
bpmix_head/PLAN_R18.md). Refuses any sample whose manifest split != 'train'. Never touches TEST.

usage: python -m qa_matvey.diag_r3_gradients --v2 CKPT --manifest manifest_expanded.json --out OUT.json
"""
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
import torch

import doe_models.train_multiplane as tm
from back_projection_stem.r3_loss import r3_loss, BETA, EPS, CONST_STD
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.data import load_png_quartet
from doe_models.adam_multiplane import multiplane_amplitudes
from feedback_head.train_r22 import research_load

KEYS = ("B208:006", "B64:010")
DIMS = (-2, -1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def per_plane_terms(a, t):
    """R3 split per plane: lsq_i, contract_i, corrterm_i (each [4]); R3 = mean(lsq)+0.1*mean(contract)+BETA*mean(corr)."""
    a, t = a.double()[0, :, 0], t.double()[0, :, 0]          # [4,H,W]
    peak = a.amax(dim=DIMS, keepdim=True)
    contract = ((a / peak - t) ** 2).mean(dim=DIMS)
    energy = (t * t).sum(dim=DIMS)
    zero = energy <= 1e-12
    gain = (a * t).sum(dim=DIMS) / (a * a).sum(dim=DIMS)
    lsq = ((gain[:, None, None] * a - t) ** 2).sum(dim=DIMS) / torch.where(zero, torch.ones_like(energy), energy)
    lsq = torch.where(zero, contract, lsq)
    x = a / peak
    dx, dt = x - x.mean(dim=DIMS, keepdim=True), t - t.mean(dim=DIMS, keepdim=True)
    vx, vt = (dx * dx).mean(dim=DIMS), (dt * dt).mean(dim=DIMS)
    corr = (dx * dt).mean(dim=DIMS) / torch.sqrt((vx + EPS) * (vt + EPS))
    cterm = torch.where(vt <= CONST_STD ** 2, torch.zeros_like(corr), 1 - corr)
    return lsq, contract, cterm, corr, gain, vt


def u8_pearson(a, t):
    """Official-style plane Pearson: per-plane max-normalised, round-half-up u8, vs target."""
    from doe_experiments.png_io import amplitude_to_u8
    from doe_experiments.metrics import evaluate_multiplane
    codes = amplitude_to_u8(a.detach().cpu())
    pairs = evaluate_multiplane(torch.from_numpy(codes.astype(np.float64) / 255), t.detach().cpu().double())
    return [m.pearson for m in pairs[0]], [m.ssim for m in pairs[0]]


def cos(u, v):
    nu, nv = u.norm(), v.norm()
    return float((u * v).sum() / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def hf_frac(g, r0=0.5):
    """Fraction of gradient energy above r0 * Nyquist (radial, centred FFT)."""
    G = torch.fft.fftshift(torch.fft.fft2(g)).abs() ** 2
    n = g.shape[-1]; f = torch.fft.fftshift(torch.fft.fftfreq(n)) / 0.5
    R = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return float(G[R > r0].sum() / G.sum())


def analyse(model, cfg, t, key):
    with torch.no_grad():
        phi0 = model(t)                                         # [1,1,H,W] continuous phase from frozen V2
    phi = phi0.detach().double().clone().requires_grad_(True)
    prop = propagate_fresnel
    # multiplane_amplitudes validates dtype; run in float64 via the shared forward if accepted, else float32
    try:
        a = multiplane_amplitudes(phi, cfg, prop); dtype = "float64"
    except Exception:
        phi = phi0.detach().float().clone().requires_grad_(True); a = multiplane_amplitudes(phi, cfg, prop); dtype = "float32"
    lsq, con, cterm, corr, gain, vt = per_plane_terms(a, t)
    L_plane = (lsq + 0.1 * con + BETA * cterm) / 4.0           # sum_i L_plane_i == R3
    R3_ref = r3_loss(a, t)
    out = {"key": key, "dtype": dtype, "R3": float(R3_ref), "R3_from_planes": float(L_plane.sum()),
           "R3_consistency_abs": abs(float(R3_ref) - float(L_plane.sum())),
           "plane": {"lsq_gain_nmse": lsq.tolist(), "contract_mse": con.tolist(), "corr_term_1mcorr": cterm.tolist(),
                     "corr_continuous": corr.tolist(), "lsq_gain": gain.tolist(), "target_var": vt.tolist(),
                     "constant_target_plane": (vt <= CONST_STD ** 2).tolist(),
                     "L_plane_total": L_plane.tolist()},
           "contrib_share": {"lsq": float(lsq.mean() / R3_ref), "contract_x0.1": float(0.1 * con.mean() / R3_ref),
                             "corr_xbeta": float(BETA * cterm.mean() / R3_ref)}}
    P, S = u8_pearson(a, t); out["official_u8"] = {"pearson": P, "ssim": S, "meanP": float(np.nanmean([p for p in P if p is not None]))}
    grads = []
    for i in range(4):
        g, = torch.autograd.grad(L_plane[i], phi, retain_graph=True); grads.append(g[0, 0].double())
    term_grads = {}
    for name, v in (("lsq", lsq.mean()), ("contract", 0.1 * con.mean()), ("corr", BETA * cterm.mean())):
        g, = torch.autograd.grad(v, phi, retain_graph=True); term_grads[name] = g[0, 0].double()
    gfull, = torch.autograd.grad(R3_ref, phi, retain_graph=True); gfull = gfull[0, 0].double()
    # differentiable 1-Pearson on raw A (scale invariant => identical in value to the corr term without the const mask/BETA)
    pear = corr.mean(); gpear, = torch.autograd.grad(1 - pear, phi, retain_graph=False); gpear = gpear[0, 0].double()
    gsum = sum(grads)
    norms = [float(g.norm()) for g in grads]
    C = [[cos(grads[i], grads[j]) for j in range(4)] for i in range(4)]
    off = [C[i][j] for i in range(4) for j in range(4) if i != j]
    out["grad"] = {"plane_L2": norms, "full_L2": float(gfull.norm()), "sum_planes_vs_full_relerr": float((gsum - gfull).norm() / gfull.norm()),
                   "cos_4x4": C, "cos_offdiag_mean": float(np.mean(off)), "cos_offdiag_min": float(np.min(off)),
                   "cancellation_ratio": float(gsum.norm() / sum(norms)),
                   "cancellation_ratio_if_orthogonal_equal": 0.5,
                   "cos_plane_vs_full": [cos(g, gfull) for g in grads],
                   "term_L2": {k: float(v.norm()) for k, v in term_grads.items()},
                   "term_cos": {f"{a}|{b}": cos(term_grads[a], term_grads[b]) for a, b in (("lsq", "contract"), ("lsq", "corr"), ("contract", "corr"))},
                   "cos_full_vs_1mPearson": cos(gfull, gpear),
                   "hf_frac_r>0.5Nyq": {"full": hf_frac(gfull), "pearson": hf_frac(gpear), "planes": [hf_frac(g) for g in grads]},
                   "random_cos_scale_1/sqrt(N)": 1 / math.sqrt(gfull.numel())}
    # first-order directional probe on phi only (analysis, NOT optimisation; weights untouched, no iteration)
    probe = []
    for name, d in (("-g_R3", -gfull), ("-g_1mPearson", -gpear)):
        for step in (0.01, 0.05):                               # rad RMS change of phi
            dphi = d / d.pow(2).mean().sqrt() * step
            with torch.no_grad():
                a2 = multiplane_amplitudes((phi + dphi[None, None].to(phi.dtype)).detach(), cfg, prop)
                l2, c2, ct2, _, _, _ = per_plane_terms(a2, t)
                P2, _ = u8_pearson(a2, t)
            probe.append({"dir": name, "rms_rad": step, "dR3": float(r3_loss(a2, t) - R3_ref),
                          "dL_plane": [float(x) for x in ((l2 + 0.1 * c2 + BETA * ct2) / 4 - L_plane.detach())],
                          "dP_official_u8": [None if (p2 is None or p is None) else p2 - p for p2, p in zip(P2, P)]})
    out["probe"] = probe
    return out


def constant_target_case(model, cfg):
    """Corner case: plane 2 constant (0.5) -> corr term must be 0 for it, grads finite."""
    t = torch.rand(1, 4, 1, 128, 128, generator=torch.Generator().manual_seed(0)); t[:, 1] = 0.5
    with torch.no_grad():
        phi = model(t)
    phi = phi.detach().clone().requires_grad_(True)
    a = multiplane_amplitudes(phi, cfg, propagate_fresnel)
    lsq, con, cterm, *_ = per_plane_terms(a, t)
    L = r3_loss(a, t); g, = torch.autograd.grad(L, phi)
    return {"corr_term": cterm.tolist(), "const_plane_corr_term_is_zero": float(cterm[1]) == 0.0,
            "grad_finite": bool(torch.isfinite(g).all()), "R3": float(L)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v2", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    v2_sha = sha(a.v2); assert v2_sha.startswith("19b05dad"), v2_sha
    man_sha = sha(a.manifest); assert man_sha.startswith("76d2474d"), man_sha
    man = json.loads(Path(a.manifest).read_text()); root = Path(a.manifest).parent
    model, payload = research_load(a.v2); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    cfg = CaseConfigV1(); model._config = cfg
    w0 = {k: v.clone() for k, v in model.state_dict().items()}
    res = []
    for key in KEYS:
        s = next(s for s in man["samples"] if s["key"] == key)
        assert s["split"] == "train" and s["kind"] == "B", s["split"]
        t = load_png_quartet([root / p for p in s["paths"]])
        r = analyse(model, cfg, t, key); r["split"] = s["split"]; r["paths"] = s["paths"]
        r["png_sha256"] = [sha(root / p) for p in s["paths"]]; res.append(r)
    weights_unchanged = all(torch.equal(w0[k], v) for k, v in model.state_dict().items())
    try: git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: git = None
    rep = {"task": "NIH-DIAG-GRADIENT-014", "v2_sha256": v2_sha, "manifest_sha256": man_sha, "code_git_head": git,
           "r3_source": "back_projection_stem/r3_loss.py (BETA=0.3) + doe_models/train_multiplane.training_objective LSQ_MIX",
           "distances_m": list(cfg.distances_m), "wavelength_m": cfg.wavelength_m, "input_pitch_m": list(cfg.input_pitch_m),
           "output_pitch_m": [cfg.wavelength_m * z / (128 * cfg.input_pitch_m[0]) for z in cfg.distances_m],
           "weights_unchanged": weights_unchanged, "optimizer_steps": 0, "test_split_touched": False,
           "quartets": res, "corner_constant_target": constant_target_case(model, cfg)}
    Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("v2_sha256", "weights_unchanged")}))


if __name__ == "__main__":
    main()
