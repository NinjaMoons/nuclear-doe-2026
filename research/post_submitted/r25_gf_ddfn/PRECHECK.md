# R25 GF-DDFN precheck — NIH-POST-R25-GF-DDFN-037 (POST-SUBMISSION research, not a contest model)

Host DESKTOP-T1K698V (RTX 4070 Ti), branch `research/post-submission-r25-gf-ddfn` (base `research/post-submission@dbcb4ad`).
The archived Gemini files in `../gemini_alexander/` are untouched. `main`, the submission snapshot and V2 weights are untouched.
No TEST was decoded: `run_r25_notest.py` loads only train+val, with a per-file SHA check.
The production physics is the code imported from `submission/stem_v2/src`. The run uses the `r22` checkout, whose `fresnel.py` and `stem.py` are byte-identical to it apart from CRLF line endings.

## 1. Physics gate (`physics_gate.py` → `evidence/physics_gate.json`)

Tests use complex128, random U and V, and all four z planes.

| check | Gemini raw | fixed adjoint (`adjoint_fresnel`) | production `inverse_fresnel` |
|---|---|---|---|
| `<FU,V>` = `<U,G V>`, relative error | 0.88 – 2.08 (**FAIL**) | ≤ 2.7e-15 | ≤ 2.6e-15 (physical, pixel-area weighted) |
| ratio `<FU,V>/<U,GV>` | abs 0.998 / 0.951 / 0.908 / 0.869, phase 3.13 / −0.89 / 1.38 / −2.63 rad | 1 | 1 |

- **Gemini's "exact adjoint" is wrong.**
  - Chirps, grids, the fftshift order and the FFT sign are correct.
  - It omits the per-plane complex scale `s = e^{i2πz/λ}/(iλz)·√(HW)·dy·dx`, so it equals `F^H/conj(s)`.
  - As a result, the four lifted planes are summed with wrong relative weights and phases.
- **Fixed:** `adjoint_fresnel` is the exact `F^H`.
- **Isometry check:** F is an isometry in physical norms. The physical adjoint therefore equals `F^{-1}`, which is the production `inverse_fresnel` that STEM V2 already uses. Round-trip error ≤ 2.8e-15.

## 2. L8 codec

- **Gemini `round(255q)/256`:** matches the official `decode_phase(encode_phase(φ))` on only 0.20% of 200,514 phases, with a maximum error of 0.049 rad. Example: q=127/255 decodes to 0.49609 instead of 0.49804.
- **Fixed `STEQuantizeL8`:** 100% bit-exact, including φ = ±π and the half-code boundaries. Its gradient is the identity (straight-through estimator).

## 3. Gradients

R3 objective through the R25 model in float64: autograd matches the finite difference along a random direction.

| parameter | relative error |
|---|---|
| `dpsi_head` | 7.7e-10 |
| `dpsi_encoder.0` | 6.9e-9 |
| `encoders.0` (U-Net) | 2.0e-9 |

All gradients are finite.

## 4. Measured cost (RTX 4070 Ti, torch 2.9 cu128)

| model | params | fwd p50 / p90, b=1 (ms) | fwd p50 / p90, b=4 (ms) | peak extra VRAM, b=4 | train step, b=4 (p50 / p90) |
|---|---|---|---|---|---|
| STEM V2 | 452,481 | 5.17 / 5.85 | 5.06 / 5.59 | 28 MB | 15.5 / 17.9 ms, 166 MB |
| R25 dPsi | 509,669 | 6.33 / 7.35 | 6.34 / 8.17 | 33 MB | 22.7 / 25.0 ms, 344 MB |
| Gemini raw | 117,470 | 3.42 / 4.10 | 3.04 / 3.42 | 52 MB | — |
| Gemini fixed | 117,470 | 5.53 / 6.22 | 5.55 / 6.05 | 41 MB | — |

The 1.8 ms figure in the Gemini report was not reproduced: the measured p50 is 3.0–3.4 ms for the raw model.

## 5. One-factor hypothesis (`model_r25.py`)

**Change vs V2.** Only the stem lift changes: `V_k = T_k + 0j` becomes `V_k = T_k·exp(i·dPsi_k(T))`, where:
- `dPsi = π·tanh(head(enc(2T−1)))`;
- `enc` is the GF-DDFN `target_encoder` (Conv 4→32, SiLU, 3 ResBlocks with GroupNorm);
- `head` is a 1×1 conv with only the 4 dPsi channels, zero-initialized, so the model is exactly V2 at init. This is tested, and smoke step 1 gives an identical objective.

**Excluded, each to be a separate later factor:**
- the phase anchor (random high-frequency plus quadratic);
- the 2-expert router (the Gumbel train vs top-1 eval mismatch is removed in the fixed copy);
- the amplitude compensation `dA`;
- the DOE-plane coherent-sum residual head;
- the straight-through quantizer in training.

Both arms train on continuous phase and are scored through the official L8 export.

**Honest overlap with R19.** R19 tphase was already target-conditioned (`theta_net` on T), not static. R19 = this dPsi lift plus a direct analytic path `arg(e^{iφ} + Σ g_k B'_k)` plus a 3-layer 5×5 GELU network. R19 had a 2-seed mean of +.0051 P and +.0087 B and was not promoted. R25-minimal therefore isolates whether the lift alone, with the GF-DDFN encoder and without the direct path, carries the R19 signal.

**Difference from R22.** R22 feeds forward-propagated residual fields, which is a different mechanism.

## 6. 100-update GPU smoke

Setup: seed 2026, 64 TRAIN quartets (32 A + 32 B drawn by seed 2026), batch 4, Adam lr 1e-3, R3 objective, initialized from V2 `19b05dad`. No VAL was run.

| arm | objective at step 1 / 10 / 50 / 100 | dPsi mean abs (rad) | time | peak VRAM | reload Δ | checkpoint SHA256 |
|---|---|---|---|---|---|---|
| control | .20657 / .20959 / .36751 / .31916 | — | 2.4 s | 150 MB | 0.0 | 575be4a8b18f9081… |
| dpsi | .20657 / .20904 / .36601 / .31841 | .033 → .084 | 3.5 s | 326 MB | 0.0 | b4755a4a48ae6831… |

Both arms show the same initial rise in the objective. This is the known lr 1e-3 fine-tune shock, which is identical in both arms.

## 7. Proposed bounded pilot — requires an independent SCIENCE_GATE=GO

**Pre-registered settings.** Seed 2026, V2 initialization, identical sampler, lr, R3 objective and budget in both arms. One pre-fixed endpoint at 6000 updates (no checkpoint picking), evaluated once on VAL16 with the official saved and reloaded DOE.png plus four L8 PNGs.

**Pairs:**
- **P1 (card spec):** `--subset64`, control vs dpsi.
- **P2 (comparable to R19/R22):** all of TRAIN512, control vs dpsi.

Estimated GPU time is about 2.5 + 3.5 minutes per pair, plus evaluation.

**Reported metrics:**
- Pearson and SSIM;
- A/B split;
- B own vs inter-z ghosting;
- worst plane;
- brightness and speckle;
- checkpoint SHA.

**Decision rule:**
- **Promising:** dpsi − control has ΔP > 0 and ΔB_own > 0 in both pairs. This then requires a second seed and an independent CPU replay before any claim.
- **NO_GO:** otherwise, reported as "lift alone does not carry the signal", not as "GF-DDFN solved".
