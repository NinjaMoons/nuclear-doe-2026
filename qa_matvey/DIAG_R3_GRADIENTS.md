# NIH-DIAG-GRADIENT-014 — R3 per-plane loss / gradient conflict on frozen V2 (CPU, analysis only)

Host WIN-J79V8KOQ600 (i5-13420H, no GPU), torch 2.10.0+cpu, 4 threads, run time ~5 s.
Script `qa_matvey/diag_r3_gradients.py`, raw numbers `qa_matvey/DIAG_R3_GRADIENTS.json`.

## Inputs (verified)
- V2 checkpoint `stem_native_s0_step_012000.pt` from release asset `stem-candidate-6785eca-native12k-s0.zip`,
  SHA256 `19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f`; loaded read-only, `eval()`,
  `requires_grad=False`, state_dict compared before/after: `weights_unchanged=true`, `optimizer_steps=0`.
- DATA: corpus `corpus-4b-win-49b9604.tar.gz` SHA256 `6c6f136b…`, `manifest_expanded.json` `76d2474d…`.
  Only `manifest_expanded.json`, `B208/006`, `B64/010` were extracted (selective tar members). Both have
  `split=train, kind=B` (asserted). These are the pre-registered TRAIN-B quartets of `bpmix_head/PLAN_R18.md`.
  TEST was not extracted, opened or decoded. No VAL used.
- Case: λ=532 nm, input pitch 10 µm, z = 24.1/25.3/26.5/27.7 mm, 128×128, native output pitch λz/(Nδx)
  = 10.017/10.515/11.014/11.513 µm (official `CaseConfigV1` + `doe_core/fresnel.propagate_fresnel`).
- R3 code: `back_projection_stem/r3_loss.py` @ `research/sanya-r22-feedback` 9ffc1c2 (BETA=0.3) on top of
  `doe_models/train_multiplane.training_objective(LSQ_MIX)`: R3 = mean_i lsq_gain_nmse_i + 0.1·mean_i contract_mse_i
  + 0.3·mean_i (1−corr_i). Per-plane split L_i = (lsq_i + 0.1·contract_i + 0.3·cterm_i)/4; Σ L_i reproduces `r3_loss`
  to ≤1.1e-16 and Σ g_i reproduces ∇R3 to ≤2.3e-16 rel. err. Autograd only w.r.t. continuous phase φ = V2(T) (float64).

## Loss values
| quartet | R3 | lsq_i | contract_i | 1−corr_i | share lsq / 0.1·con / 0.3·corr | official u8 P_i |
|---|---|---|---|---|---|---|
| B208:006 | 0.6574 | .428 .325 .715 .489 | .021 .012 .090 .133 | .494 .466 .622 .576 | 74% / 1% / 25% | .506 .534 .378 .425 |
| B64:010 | 0.5491 | .261 .410 .402 .420 | .034 .032 .056 .037 | .749 .476 .652 .415 | 68% / 1% / 31% | .252 .524 .348 .585 |

Continuous corr equals official u8 Pearson within 1e-4 on every plane.

## Gradients d(L_i)/dφ
| | B208:006 | B64:010 |
|---|---|---|
| ‖g_i‖₂ (×1e-3) | 1.9 1.8 1.4 1.9 | 1.5 1.9 1.5 2.0 |
| ‖∇R3‖₂ (×1e-3) | 2.17 | 1.75 |
| off-diag cos mean / min | −0.21 / −0.43 (z1,z2) | −0.23 / −0.47 (z2,z4) |
| cancellation ‖Σg_i‖/Σ‖g_i‖ (orthogonal equal = 0.50) | **0.31** | **0.26** |
| cos(g_i, ∇R3) | .24 .22 .21 .53 | .44 .14 .33 .17 |
| cos(∇R3, ∇(1−Pearson)) | 0.977 | 0.863 |
| term cos lsq·corr / lsq·contract / contract·corr | .96 / .05 / .05 | .76 / .01 / .02 |
| grad energy above 0.5·Nyquist (full) | 61% | 41% |
| random-vector cos scale 1/√N | 0.008 | 0.008 |

4×4 cosine matrices are in the JSON (`grad.cos_4x4`). B208:006: all six pairs negative. B64:010: five negative, (z1,z3) +0.12.

First-order probe (φ only, one step, no iteration, weights untouched): stepping φ by 0.05 rad RMS along −∇R3 lowers
**every** L_i and raises **every** official P_i (B208 ΣdP=+.055, B64 +.045); along −∇(1−Pearson) ΣdP=+.057 / +.053.

Corner case (synthetic random targets, plane 2 = constant 0.5): corr term of the constant plane = 0 exactly, ∇ finite.

## Separation of hypotheses
1. **Inconsistent gradient directions across z — present, partial.** Per-plane φ-gradients are clearly anti-aligned
   (mean cos ≈ −0.2, far outside the 0.008 random scale) and ~70% of their total magnitude cancels (0.26–0.31 vs 0.50
   for orthogonal). But every cos(g_i, ∇R3) > 0, so at the V2 point the combined direction still improves all four planes
   (probe confirms); it is a partial cancellation that shrinks the useful step, not a zero-sum trade-off.
2. **Loss vs official metric mismatch — weak.** ∇R3 is 0.86–0.98 aligned with ∇(1−Pearson); per-step official dP of the R3
   direction is 85–97% of the pure-Pearson direction. The lsq term (≈70% of the value) is itself aligned with the corr term
   (.76–.96). The contract term contributes ~1% of the value and is ~orthogonal; it neither helps nor hurts here.
3. **Vanishing / saturation — no evidence.** Plane gradient norms are balanced (max/min ≤1.36), no constant-target or
   zero-gain plane, no plane dominates. Absolute norms (~2e-3) are not interpretable without the weight-space Jacobian.

Caveats: n=2 quartets; φ-space only (cosines in φ need not equal conflicts in weight space through the V2 Jacobian);
targets in B are independent per plane, so anti-alignment may be intrinsic to the case geometry, not a loss defect.
No causal claim is made from cosines.

**LOSS_CONFLICT=SUPPORTS** (partial inter-z gradient cancellation, ~70%, non-zero-sum; loss-vs-metric mismatch weak;
no saturation). Suggested next check if wanted: same metrics in weight space (per-plane ∂L_i/∂θ) on the same 2 quartets.
