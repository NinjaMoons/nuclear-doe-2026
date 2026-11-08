"""R25 physics gate (NIH-POST-R25-GF-DDFN-037). Read-only use of production physics; writes one JSON.
usage: python -m r25_gf_ddfn.physics_gate --out FILE.json [--v2 CKPT] [--device cuda]
Checks: (1) complex inner-product adjoint test of production propagate_fresnel vs Gemini raw lift, fixed adjoint and
production inverse_fresnel, plain and physical (pixel-area weighted) inner products, all 4 z; (2) inverse round trip;
(3) L8 codec exactness of the Gemini raw and fixed STE quantizer vs official encode_phase/decode_phase; (4) finite-
difference vs autograd directional derivative of the R3 objective through the R25 model (float64, CPU); (5) CUDA batch
1/4 forward latency p50/p90 and peak VRAM for V2 STEM, R25, Gemini raw and fixed."""
import argparse, json, math, statistics, time
from pathlib import Path

import numpy as np
import torch

import back_projection_stem.train_wrapper as bw
from back_projection_stem.stem import BackProjectionStemUNet, inverse_fresnel
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.phase_codec import decode_phase, encode_phase
from doe_models.adam_multiplane import multiplane_amplitudes
from r25_gf_ddfn.gf_ddfn_fixed import GF_DDFN_Fixed, STEQuantizeL8, adjoint_fresnel, raw
from r25_gf_ddfn.model_r25 import DPsiStemUNet

CFG = CaseConfigV1()


def ip(a, b):  # <a, b> = sum conj(a) b
    return complex((a.conj() * b).sum())


def adjoint_tests(n_trials=3):
    g = torch.Generator().manual_seed(2026)
    rnd = lambda: torch.complex(torch.randn(1, 1, 128, 128, generator=g, dtype=torch.float64),
                                torch.randn(1, 1, 128, 128, generator=g, dtype=torch.float64))
    gem = raw.GF_DDFN().double()
    out = []
    for k in range(4):
        geo = CFG.plane(k)
        dx2 = geo.input_pitch_m[0] * geo.input_pitch_m[1]
        dyo, dxo = geo.output_pitch_m(128, 128)
        row = {"plane": k + 1, "z_mm": geo.distance_m * 1e3}
        for name, op in (("gemini_raw", None), ("fixed_adjoint", adjoint_fresnel), ("inverse_fresnel", inverse_fresnel)):
            errs, ratios = [], []
            for _ in range(n_trials):
                U, V = rnd(), rnd()
                lhs = ip(propagate_fresnel(U, geo).field, V)
                if op is None:
                    V4 = torch.zeros(1, 4, 128, 128, dtype=torch.complex128); V4[:, k] = V[:, 0]
                    GV = gem.adjoint_fresnel_lift(V4.to(torch.complex64)).to(torch.complex128)[:, k:k + 1]
                else:
                    GV = op(V, geo)
                rhs = ip(U, GV)
                if name == "inverse_fresnel":  # physical inner product: <FU,V> dxo dyo  vs  <U,F^-1 V> dx dy
                    lhs, rhs = lhs * dyo * dxo, rhs * dx2
                errs.append(abs(lhs - rhs) / abs(lhs)); ratios.append(lhs / rhs)
            r = ratios[0]
            row[name] = {"rel_err_max": max(errs), "ratio_abs": abs(r), "ratio_phase_rad": math.atan2(r.imag, r.real)}
        U = rnd()
        row["inverse_roundtrip_max_abs_err"] = float((inverse_fresnel(propagate_fresnel(U, geo).field, geo) - U).abs().max())
        out.append(row)
    return out


def codec_tests():
    g = torch.Generator().manual_seed(7)
    ph = (torch.rand(200000, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    edges = torch.tensor([-math.pi, math.pi, 0.0] + [-math.pi + 2 * math.pi * (c + 0.5) / 255 for c in range(255)]
                         + [-math.pi + 2 * math.pi * c / 255 for c in range(256)], dtype=torch.float64)
    ph = torch.cat([ph, edges.clamp(-math.pi, math.pi)])
    official = torch.from_numpy(decode_phase(encode_phase(ph.numpy())))
    wrapd = lambda a, b: ((a - b + math.pi) % (2 * math.pi) - math.pi).abs()
    fixed = STEQuantizeL8.apply(ph)
    rawq = raw.STEQuantize8Bit.apply(ph)
    x = ph.clone().requires_grad_(True); STEQuantizeL8.apply(x).sum().backward()
    return {"n": int(ph.numel()),
            "fixed_exact_frac": float((fixed == official).double().mean()),
            "fixed_max_abs_diff_rad": float((fixed - official).abs().max()),
            "fixed_max_wrapped_diff_rad": float(wrapd(fixed, official).max()),
            "gemini_raw_exact_frac": float((rawq == official).double().mean()),
            "gemini_raw_max_wrapped_diff_rad": float(wrapd(rawq, official).max()),
            "gemini_raw_examples": {"q=127/255->": float(raw.STEQuantize8Bit.apply(torch.tensor([-math.pi + 2 * math.pi * 127 / 255], dtype=torch.float64))[0] + math.pi) / (2 * math.pi)},
            "ste_grad_is_identity": bool((x.grad == 1).all())}


def fd_gradient_test():
    torch.manual_seed(0)
    m = DPsiStemUNet().double()
    with torch.no_grad():
        m.dpsi_head.weight.normal_(0, 0.05)  # non-zero so the dPsi path carries gradient
    g = torch.Generator().manual_seed(3)
    T = torch.rand(2, 4, 1, 128, 128, generator=g, dtype=torch.float64)
    loss_fn = lambda: bw.research_objective(multiplane_amplitudes(m(T), CFG, propagate_fresnel), T, bw.R3_NAME)[0]
    res = {}
    for name in ("dpsi_head.weight", "dpsi_encoder.0.weight", "encoders.0.0.weight"):
        p = dict(m.named_parameters())[name]
        m.zero_grad(); loss_fn().backward()
        d = torch.randn(p.shape, generator=g, dtype=torch.float64); d /= d.norm()
        ana = float((p.grad * d).sum())
        eps = 1e-5
        with torch.no_grad():
            p += eps * d; lp = float(loss_fn()); p -= 2 * eps * d; lm = float(loss_fn()); p += eps * d
        num = (lp - lm) / (2 * eps)
        res[name] = {"autograd": ana, "finite_diff": num, "rel_err": abs(ana - num) / max(abs(num), 1e-12),
                     "grad_finite": bool(torch.isfinite(p.grad).all())}
    return res


def load_v2(path):
    import doe_models.train_multiplane as tm
    saved = tm.FourInputUNet; tm.FourInputUNet = BackProjectionStemUNet
    try:
        return tm.load_checkpoint(path)
    finally:
        tm.FourInputUNet = saved


def timing(v2path, device):
    dev = torch.device(device)
    v2, payload = load_v2(v2path)
    from r25_gf_ddfn.model_r25 import from_v2
    models = {"stem_v2": v2, "r25_dpsi": from_v2(v2), "gemini_raw": raw.GF_DDFN(), "gemini_fixed": GF_DDFN_Fixed()}
    nparams = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    res = {"v2_sha256": payload["sha256"], "params": nparams, "gpu": torch.cuda.get_device_name(dev)}
    g = torch.Generator().manual_seed(5)
    for b in (1, 4):
        T5 = torch.rand(b, 4, 1, 128, 128, generator=g).to(dev)
        for name, m in models.items():
            m = m.to(dev).eval(); x = T5 if name in ("stem_v2", "r25_dpsi") else T5[:, :, 0]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev); base = torch.cuda.memory_allocated(dev)
            ts = []
            with torch.no_grad():
                for i in range(60):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); y = m(x); torch.cuda.synchronize()
                    if i >= 10: ts.append((time.perf_counter() - t0) * 1e3)
            ts.sort()
            res[f"{name}_b{b}"] = {"p50_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * len(ts))],
                                   "peak_vram_mb_above_weights": (torch.cuda.max_memory_allocated(dev) - base) / 2**20,
                                   "out_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}
    for name in ("stem_v2", "r25_dpsi"):  # one training step (batch 4, R3), as used in the pilot
        m = models[name].to(dev).train(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        T5 = torch.rand(4, 4, 1, 128, 128, generator=g).to(dev); ts = []
        torch.cuda.reset_peak_memory_stats(dev)
        for i in range(25):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss, _ = bw.research_objective(multiplane_amplitudes(m(T5), CFG, propagate_fresnel), T5, bw.R3_NAME)
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
            if i >= 5: ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        res[f"{name}_trainstep_b4"] = {"p50_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * len(ts))],
                                       "peak_vram_mb": torch.cuda.max_memory_allocated(dev) / 2**20}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--v2"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rep = {"adjoint": adjoint_tests(), "codec": codec_tests(), "fd_gradient": fd_gradient_test()}
    if a.v2:
        rep["timing"] = timing(a.v2, a.device)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
