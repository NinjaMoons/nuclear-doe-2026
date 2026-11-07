# G1 / G2 verification by @ggmaster913 (Claude Code), 08.11 ~00:05 MSK
Legend: CONFIRMED / MODEL_APPROXIMATION / HYPOTHESIS / REJECTED. Read-only tests: `tests/g1_tests.py` ->
`tests/g1_tests_result.json`. Data: v2 `19b05dad` exports (VAL B 8 quartets, my Mac replay; TRAIN B64:010, B208:006),
direct Adam 1000 s0 (random init, #14 run2) on the same 2 TRAIN-B. NB: Gemini's "Adam-2000 from STEM init" artifacts
are not available locally; the Adam comparison uses the #14 teacher (different init/steps) — MODEL_APPROXIMATION.

## G1 (optics)
| claim | class | evidence |
|---|---|---|
| dPhi(nu)=pi*lambda*dz*(nu/dx)^2 ≈ 5.76*pi*nu^2 per 1.2 mm; low-nu content barely dephases | CONFIRMED (formula) / MODEL_APPROXIMATION (paraxial local-frequency picture) | arithmetic checks; consistent with measured band inter-z of v2 (r<=.15 Nyq .94, .15-.3 .77, .3-.5 .31, >.5 ~0; READONLY_band_interz_v2.md) |
| Adam teacher uses rough target-plane phase psi_k inside bright regions; STEM has smooth psi_k | CONFIRMED (2 TRAIN-B) | foreground phase roughness R_psi: Adam 3.04 / 3.24 vs v2 0.07 / 0.42 on the same quartets; v2 VAL B 0.18-0.60 |
| Test 1 prediction: Adam has much lower HIGH-band (nu 0.08-0.5) inter-z than STEM | REJECTED | high-band inter-z is ~0 for both (v2 .01/.06, Adam .09/.08; VAL v2 .03-.12); low band (nu<=0.08) both high (v2 .67-.86, Adam .83-.85). The Adam advantage is own-target fidelity in BOTH bands (own low .55-.59 vs .44-.45; own high .53-.65 vs .36-.43), largest in high |
| zero-phase back-projection carries only the low-frequency envelope | REJECTED as stated | inverse_fresnel(T+0j) is an exact discrete inverse (all frequencies of T); sham test #44: proper features +.145 P vs chirp/shuffle. Weaker true statement: the zero-phase assumption gives no hint of the rough psi_k Adam uses (HYPOTHESIS) |
| Test 2 / "index-aligned T channels + skip connections stamp ghosts at fixed pixel indices" | REJECTED (by Gemini's own criterion) | dark outer-annulus ghosts of v2 correlate better with PHYSICALLY scaled foreign targets: rho_phys > rho_idx on all 8 VAL B (delta -0.09..-0.21; fraction idx>phys 0-0.25) and both TRAIN B; coordinate-warping inputs is not indicated |
| diagram "0.65..1.00 = physical single-DOE limit" | MODEL_APPROXIMATION (not a bound) | Adam is a finite optimiser, not a limit |
| WirtingerRefinementHead (1 forward+adjoint Fresnel inside forward + small head) | HYPOTHESIS, NO_GO tonight | half its motivation (coordinate mismatch) refuted above; an unrolled physics step inside the forward needs an organizer answer (rules risk vs rejected NN->iterative); new head + training cannot finish before the 02:00 start gate with review |

## G2 (ML)
| claim | class | evidence |
|---|---|---|
| (a) phase-space distillation fails (multimodality, wraps, averaging) | CONFIRMED (experimentally) | #43 DISTILL_NEUTRAL dP -.0059 [-.0117,-.0011]; student phase distance stuck at .48-.51 |
| (b) image-domain distillation from a teacher warm-started at the student (basin-locked) | HYPOTHESIS, best-supported idea | core = R16 (#49/6712460660, reviewed GO with 3 changes); consistent with G1 Test 1 (teacher's advantage is realisable image fidelity, not phase similarity) |
| BLOT-CM as one pilot (opt + SSIM + Gram-Schmidt xtalk + dark + phasor + STE) | REJECTED as a single-factor pilot | 5 changes at once (confounded); keep only the image-domain teacher term (= R16) |
| STE codec phi_q = 2pi/256*round(255/2pi*mod(phi,2pi)) | REJECTED | official codec: floor(255(phi+pi)/2pi+0.5), decode -pi+2pi*c/255 |
| uint8 evaluator with exposure alignment alpha and target-max clipping | REJECTED for our contract | official PNG = floor(255*A/max(A)+0.5), Pearson + global SSIM (data_range 1) on reloaded PNG |
| "400 steps capture >88% of 2000-step gain; 512 teachers in 15-25 s GPU / 2.5 min MPS" | HYPOTHESIS (unmeasured) | measured (Matvey CPU): 50 steps from v2: TRAIN P +.10..+.15, 2.6-3.5 s/quartet |
| (c) recombination "exacerbates underfitting"; "slot permutation = pitch hazard" | HYPOTHESIS / N.A. | our card keeps slots; measured: TRAIN B208 partial/similar inter-plane +.23/+.28 vs VAL B +.06 (IDEA_CARD_R17) |
| expected gains (+.025..+.050 B_P etc.) and GO thresholds | not adopted | invented numbers |

## Surviving IDEA_CARDs (routing in #49)
1. **R16 image-domain near-teacher distillation** (G2-b core) — CODE-READY (Matvey branch `cce75bc2` + 3 required changes). Fastest on Sanya's GPU after the recipe job; else Matvey CPU (p90 1.3 h). Start before 02:00 MSK.
2. **R17 independent-pool B recombination** — CODE-READY spec (IDEA_CARD_R17_B_RECOMB.md); lower prior per G2; alternative if GPU slot remains.
3. Wirtinger head, GOC/xtalk loss — NO_GO tonight (rules/time/confounding); keep as post-deadline hypotheses.
