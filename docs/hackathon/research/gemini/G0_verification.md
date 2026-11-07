# G0 — verification by @ggmaster913 (Claude Code)

Classification legend: CONFIRMED / MODEL_APPROXIMATION / HYPOTHESIS / REJECTED. Posted also as #7/6711891271.

| claim (G0) | class | evidence |
|---|---|---|
| z_crit = N dx^2/lambda = 24.06 mm; output pitches 10.017..11.513 um | CONFIRMED | docs/modules/fresnel.md; doe_core/fresnel.py |
| DOF of 1.28 mm aperture exceeds 1.2 mm plane spacing -> axial selectivity of low frequencies limited | MODEL_APPROXIMATION | consistent with #39 Stage0: low-pass inter-z ~0.9 even for the Adam teacher; continuous-optics estimate, not a bound for the discrete operator |
| back-projection puts >80% energy in r<=16 px core, fields ~collinear (0.986) — 'blind core' | REJECTED | measured 07.11 on B64:010/B208:006/B64:012/B64:016: 3-22% energy in r<=16 px (4.9% of area); |cos(B_k,B_j)| full field 0.25-0.74 on B (0.85-0.95 only on correlated A); sham test #44/6710724234: proper features P .520 vs .375-.378 |
| R13 failed due to a linear tilt / aliasing | REJECTED | R13 was a quadratic carrier psi=pi(p^2+q^2)/128 that empties the aperture (#46/6711027935) |
| random phase fails because of infinite bandwidth / vortices | REJECTED (as the cause) | Adam-teacher masks are near-white (HF .77-.80, roughness ~1.5 rad) yet separate best (#39 Stage0); the diffuser fails because it is unstructured |
| L_GOC loss: +0.035..0.060 on B | HYPOTHESIS (numbers unsupported) | loss-side changes so far ~0 (R15 dP +.0002); teacher Adam with same R3 separates -> bottleneck likely learnability |
| 1-step analytic residual with forward Fresnel inside the network | HYPOTHESIS, RULES RISK | one GS-like physical iteration inside forward; organizer rejected NN->iterative refinement |
| harmonic saddle phase in back-projection | HYPOTHESIS | untested; collinearity numbers quoted (0.986->0.31) contradict measured baseline |
| SSIM with 11x11 Gaussian window and scale alignment | REJECTED for our contract | official metrics: global SSIM data_range 1, Pearson on official PNG |
| 15-point WhiteCollars audit | superseded | done in #41: WC_NEUTRAL, WC dropped |
| B-quartet recombination of independent B planes | HYPOTHESIS (in-distribution for B by construction) | B208 relation: independent; overlaps #43 sampling lane |
| submission gate thresholds (B P>=.465, matched>=27/32) | not adopted | arbitrary; current v2 B P .4516, own 25/32 |
