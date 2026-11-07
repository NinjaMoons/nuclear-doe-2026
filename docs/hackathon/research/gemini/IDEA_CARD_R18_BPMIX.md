# IDEA_CARD R18 — complex back-projection mixing head (one forward), @ggmaster913, 08.11 ~00:35 MSK
Mechanism: F1-F3 (OPUS_FINDINGS_HF_MASK.md). Output phi = angle((1+a0)*exp(i*phi_STEM) + sum_k a_k*B_k), a = 10-channel
1x1 head on last decoder features, pooled 32x32 -> bilinear 128, zero-init (exact STEM at init). B_k are the existing STEM
input features (analytic, non-trainable, computed from the input PNGs) — no iteration, no optimiser at inference.
Distinct from R12 (random fixed basis, suppressed by training) and from #43 distillation (no teacher).
Pilot (pre-registered, PLAN_R18.md @ c5da285): fine-tune v2 19b05dad 6000 updates, seed 0, matched control (R3 only),
Mac M1 CPU, ~25 min/arm; GO dP>=+.01, dP_B>=+.005, own B not lower, SSIM not lower, no speckle increase.
Rules: one forward; analytic Fresnel features already used by STEM; no new physics iteration. Owner: @ggmaster913 (host free).
If GO: seed 1 + independent cross-host review by @MVRto01/@playerr456; GPU 12k-step version only after review.
