"""RESEARCH ONLY (R14 Stage A): exact D4 equivariance audit of the official single-FFT Fresnel, inverse_fresnel,
STEM features and the uint8 codec. usage: python -m d4_stem.d4_audit <out.json>"""
import itertools, json, math, sys
import numpy as np
import torch
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from back_projection_stem.stem import inverse_fresnel, backprojection_features
from doe_experiments.phase_codec import encode_phase, decode_phase
from doe_experiments.png_io import amplitude_to_u8
from d4_stem.d4 import ELEMENTS, apply, inverse

CFG = CaseConfigV1(); N = 128
rel = lambda a, b: float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt().clamp_min(1e-300))


def fields(seed):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(N) - N // 2, torch.arange(N) - N // 2, indexing="ij")
    rp = torch.polar(torch.ones(N, N, dtype=torch.float64), (torch.rand(N, N, generator=g, dtype=torch.float64) * 2 - 1) * math.pi)
    a = 0.002 + 0.001 * seed; b = 0.0007 * (seed + 1)
    chirp = torch.polar(torch.ones(N, N, dtype=torch.float64), a * xx.double() ** 2 + b * yy.double() ** 2 + 0.01 * xx.double() * (seed + 1))
    const = torch.polar(torch.ones(N, N, dtype=torch.float64), torch.full((N, N), 0.3 * seed, dtype=torch.float64))
    imp = torch.zeros(N, N, dtype=torch.complex128); imp[10 + seed, 77 - seed] = 1.0; imp[64, 64] = 0.5j
    cplx = torch.complex(torch.randn(N, N, generator=g, dtype=torch.float64), torch.randn(N, N, generator=g, dtype=torch.float64))
    return {"random_phase": rp, "chirp_structured": chirp, "constant_phase": const, "impulses": imp, "random_complex": cplx}


def main():
    res = {"closure": {}, "forward": {}, "inverse": {}, "features": {}, "codec": {}}
    # group closure and inverses on arrays (both conventions)
    x = torch.randn(N, N, dtype=torch.float64)
    for conv in ("c64", "c63.5"):
        imgs = [apply(w, x, conv) for w in ELEMENTS]
        closed = all(any(torch.equal(apply(a, apply(b, x, conv), conv), im) for im in imgs) for a in ELEMENTS for b in ELEMENTS)
        inv_ok = all(torch.equal(apply(inverse(w), apply(w, x, conv), conv), x) for w in ELEMENTS)
        distinct = all(not torch.equal(imgs[i], imgs[j]) for i in range(8) for j in range(i + 1, 8))
        res["closure"][conv] = {"closed": closed, "inverses": inv_ok, "8_distinct": distinct}
    for conv in ("c64", "c63.5"):
        f_c, f_a, i_c, ft = {}, {}, {}, {}
        for seed in (0, 1, 2):
            for name, U in fields(seed).items():
                U4 = U[None, None]
                for k in range(4):
                    geom = CFG.plane(k)
                    FU = propagate_fresnel(U4, geom).field
                    IV = inverse_fresnel(U4, geom)
                    for w in ELEMENTS[1:]:
                        lhs = propagate_fresnel(apply(w, U4, conv), geom).field
                        rhs = apply(w, FU, conv)
                        key = f"{name}"
                        f_c[key] = max(f_c.get(key, 0.0), rel(lhs, rhs))
                        f_a[key] = max(f_a.get(key, 0.0), rel(lhs.abs(), rhs.abs()))
                        if name != "impulses" or True:
                            i_c[key] = max(i_c.get(key, 0.0), rel(inverse_fresnel(apply(w, U4, conv), geom), apply(w, IV, conv)))
        # STEM features on random amplitude targets (B=2)
        g = torch.Generator().manual_seed(7)
        T = torch.floor(torch.rand((2, 4, 1, N, N), generator=g) * 255 + 0.5) / 255
        F0 = backprojection_features(T.double()).double()
        for w in ELEMENTS[1:]:
            lhs = backprojection_features(apply(w, T.double(), conv)).double()
            ft[str(w)] = rel(lhs, apply(w, F0, conv))
        res["forward"][conv] = {"complex_max_rel": f_c, "amplitude_max_rel": f_a}
        res["inverse"][conv] = {"complex_max_rel": i_c}
        res["features"][conv] = {"max_rel_over_elements": max(ft.values()), "per_element": ft}
    # codec + full display chain commutation (c64): encode pixelwise, PNG of |F(decoded)| via official amplitude_to_u8
    g = torch.Generator().manual_seed(3)
    phi = (torch.rand((1, 1, N, N), generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    codes = encode_phase(phi)
    worst_png = 0; enc_ok = True
    for w in ELEMENTS[1:]:
        enc_ok &= np.array_equal(encode_phase(apply(w, phi, "c64")), apply(w, torch.from_numpy(codes.astype(np.int64)), "c64").numpy().astype(np.uint8).reshape(codes.shape))
        dec_g = torch.from_numpy(decode_phase(encode_phase(apply(w, phi, "c64")))).reshape(1, 1, N, N)
        dec = torch.from_numpy(decode_phase(codes)).reshape(1, 1, N, N)
        amp_g = torch.stack([propagate_fresnel(torch.polar(torch.ones_like(dec_g), dec_g), CFG.plane(k)).field.abs() for k in range(4)], 1)
        amp = torch.stack([propagate_fresnel(torch.polar(torch.ones_like(dec), dec), CFG.plane(k)).field.abs() for k in range(4)], 1)
        png_g = amplitude_to_u8(amp_g.float()).astype(np.int16); png = amplitude_to_u8(amp.float())
        png_rot = apply(w, torch.from_numpy(png.astype(np.int64)), "c64").numpy().astype(np.int16)
        worst_png = max(worst_png, int(np.abs(png_g - png_rot).max()))
    res["codec"] = {"encode_commutes_exactly_c64": bool(enc_ok), "png_max_code_diff_c64": worst_png}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({"closure": res["closure"], "codec": res["codec"],
                      **{f"{conv}_fwd_complex_max": max(res["forward"][conv]["complex_max_rel"].values()) for conv in ("c64", "c63.5")},
                      **{f"{conv}_fwd_amp_max": max(res["forward"][conv]["amplitude_max_rel"].values()) for conv in ("c64", "c63.5")},
                      **{f"{conv}_inv_max": max(res["inverse"][conv]["complex_max_rel"].values()) for conv in ("c64", "c63.5")},
                      **{f"{conv}_feat_max": res["features"][conv]["max_rel_over_elements"] for conv in ("c64", "c63.5")}}, indent=1))


if __name__ == "__main__":
    main()
